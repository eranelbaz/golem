//! §5.2 decode hot path: a warm batch drain makes no heap allocation on the
//! calling thread beyond `std::thread::scope`'s own reader-thread spawns.
//! Shape: one expert unit = 3 planes, each larger than [`CHUNK`] so every
//! plane is two chunks, at unaligned offsets (head scratch), mirrored on 2
//! replicas.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use golem_store::{
    CHUNK, ChunkQueue, ModelManifest, ReadJob, ReplicaRegistry, ShardEntry, plan_aligned,
};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Process-wide count while armed: calling thread plus its reader threads.
static ALL_ON: AtomicBool = AtomicBool::new(false);
static ALL: AtomicUsize = AtomicUsize::new(0);

/// Counts allocations on threads that opted in, so the reader threads (and
/// the test harness) never pollute the calling thread's count.
struct CountingAlloc;
// SAFETY: forwards to System; the thread-local counter is only an observation
// (const-initialised Cells with no destructor, so no allocation on access).
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ALL_ON.load(Ordering::Relaxed) {
            ALL.fetch_add(1, Ordering::Relaxed);
        }
        let _ = COUNTING.try_with(|on| {
            if on.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        // SAFETY: delegated to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: delegated with the layout the allocation came in with.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Allocations made by `f`: (calling thread, every thread). This binary's
/// only test runs alone, so "every thread" is the drain's readers.
fn count(f: impl FnOnce()) -> (usize, usize) {
    COUNT.with(|c| c.set(0));
    ALL.store(0, Ordering::Relaxed);
    ALL_ON.store(true, Ordering::Relaxed);
    COUNTING.with(|on| on.set(true));
    f();
    COUNTING.with(|on| on.set(false));
    ALL_ON.store(false, Ordering::Relaxed);
    (COUNT.with(Cell::get), ALL.load(Ordering::Relaxed))
}

const PLANE: usize = CHUNK + 3 * 4096 + 512;
const PLANES: usize = 3;
const SHARD_BYTES: usize = PLANES * (PLANE + 4096) + 4096;

fn plane_offset(p: usize) -> u64 {
    (512 + p * (PLANE + 4096)) as u64
}

fn fixture() -> (Vec<PathBuf>, ReplicaRegistry) {
    let root = std::env::temp_dir().join("golem-store-drain-alloc");
    let bytes: Vec<u8> = (0..SHARD_BYTES).map(|i| (i % 251) as u8).collect();
    let mut replicas = Vec::new();
    for r in 0..2 {
        let dir = root.join(format!("r{r}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("s0.gguf"), &bytes).unwrap();
        replicas.push(dir);
    }
    let manifest = ModelManifest {
        name: "drain-alloc".to_owned(),
        shards: vec![ShardEntry {
            file: "s0.gguf".to_owned(),
            bytes: SHARD_BYTES as u64,
            sha256: "0".repeat(64),
            replicas: replicas.clone(),
            ..Default::default()
        }],
        ..Default::default()
    };
    (replicas, ReplicaRegistry::open(&manifest).unwrap())
}

#[test]
fn warm_batch_drain_allocates_only_reader_spawns() {
    let (_paths, reg) = fixture();
    let queue = Arc::new(ChunkQueue::new(0, None));
    let covering = plan_aligned(plane_offset(0), PLANE)
        .iter()
        .map(|c| c.1)
        .sum::<usize>();
    assert_eq!(plan_aligned(plane_offset(0), PLANE).len(), 2);
    let mut dsts: Vec<Vec<u8>> = (0..PLANES).map(|_| vec![0u8; covering]).collect();
    let drain = |dsts: &mut [Vec<u8>]| {
        let mut rb = queue.batch();
        for (p, dst) in dsts.iter_mut().enumerate() {
            rb.push(ReadJob {
                shard_index: 0,
                offset: plane_offset(p),
                len: PLANE,
                expected_crc: 0,
                verify_crc: false,
                prefetch: false,
                sources: &reg,
                dst,
            });
        }
        rb.wait().unwrap();
    };
    for _ in 0..4 {
        drain(&mut dsts);
    }
    let head = (plane_offset(1) % 4096) as usize;
    assert_eq!(dsts[1][head], (plane_offset(1) % 251) as u8);

    // Same scope-and-spawn shape as one drain: both replicas sit in one temp
    // dir (1 drive), so io_depth readers.
    let workers = golem_store::DEFAULT_IO_DEPTH;
    let (spawn_only, spawn_all) = count(|| {
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| {});
            }
        })
    });
    let want: Vec<Vec<u8>> = dsts.clone();
    for _ in 0..16 {
        dsts.iter_mut().for_each(|d| d.fill(0));
        let (n, a) = count(|| drain(&mut dsts));
        assert_eq!(
            n, spawn_only,
            "warm drain allocated {n} times on the calling thread; \
             {workers} bare reader spawns allocate {spawn_only}"
        );
        assert_eq!(
            a, spawn_all,
            "reader threads allocated: {a} across all threads vs {spawn_all} for bare spawns"
        );
        assert!(dsts == want, "warm drain landed every byte");
    }
}
