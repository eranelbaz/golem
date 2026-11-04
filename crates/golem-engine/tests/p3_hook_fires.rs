//! P3 acceptance, ABI v5: the batched `ensure_host_resident` trampoline
//! claims each routed unit once per step, hands out per-plane pool pointers
//! and holds the leases until `copy_done`; the merged routing readback books
//! recency lock-free; `SessionHooks` is fully owned per session (two
//! sequential enable/drop cycles each decode; of two concurrent enables
//! exactly one succeeds). Drives the engine-registered trampolines directly over
//! the tiny-MoE fixture on the CPU backend — no GPU split needed.
#![allow(unsafe_code)]
mod common;

use common::expert_tensor as xt;
use common::tiny_moe_path;
use golem_engine::config::EngineConfig;
use golem_engine::p3_glue;
use golem_llama_sys::raw;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, mpsc};

/// Pooled config for tiny-moe on the CPU backend: explicit K (no VRAM query
/// exists off HIP), 1 GiB pool. A failed precondition (RLIMIT_MEMLOCK below
/// the pool) FAILS — an early return reads as a pass (AGENTS.md Evidence).
fn pooled_config() -> EngineConfig {
    EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        ..EngineConfig::default()
    }
}

/// One simulated decode step through the engine-registered trampolines:
/// routing readback for blk.1 with routed ids [3, 5], then the batched
/// ensure for the three planes of expert 3, then copy_done. `payload` is
/// [`SessionHooks::payload_ptr`]; `gate` the synthetic gate tensor.
fn drive_step(payload: *mut libc::c_void, gate: *mut raw::ggml_tensor) -> [*const u8; 3] {
    let ids: [i32; 2] = [3, 5];
    p3_glue::routing(
        payload,
        xt::BLK,
        std::ptr::null(),
        ids.as_ptr(),
        2,
        xt::N_EXPERTS,
    );
    let mut out: [*const u8; 3] = [std::ptr::null(); 3];
    let kinds = [1, 2, 3]; // gate, up, down
    let miss_ids = [3, 3, 3];
    let rc = p3_glue::ensure_host_resident(
        payload,
        gate.cast(),
        kinds.as_ptr(),
        miss_ids.as_ptr(),
        3,
        out.as_mut_ptr().cast(),
        None,
        std::ptr::null_mut(),
    );
    assert_eq!(rc, 0, "ensure paged in");
    p3_glue::copy_done(payload);
    out
}

#[test]
fn ensure_pages_once_claims_once_and_counts_calls() {
    let hooks = p3_glue::enable(&pooled_config()).expect("enable");
    hooks.set_scoring(true); // Session::new arms it after the self-check
    let payload = hooks.payload_ptr();
    let (_ctx, gate) = xt::gate_tensor();

    let out = drive_step(payload, gate);
    assert!(
        out.iter().all(|p| !p.is_null()),
        "all three plane pointers must be filled: {out:?}"
    );
    assert_ne!(out[0], out[1], "distinct planes live at distinct offsets");

    // Claim-once: three plane misses of ONE unit → one demand claim, one
    // unit page-in; the routing readback and copy_done each fired once.
    let stats = hooks.state_stats();
    assert_eq!(stats.demand_claims.load(Ordering::Relaxed), 1);
    assert_eq!(stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
    assert_eq!(hooks.routing_calls(), 1);
    assert_eq!(hooks.ensure_calls(), 1);
    assert_eq!(hooks.copy_done_calls(), 1);

    // A second identical step: units resident → fast hits, no new page-ins;
    // the copy_done above released the leases so the re-claim succeeds.
    // SAFETY: as above.
    let _ = drive_step(payload, gate);
    assert_eq!(stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
    assert_eq!(hooks.ensure_calls(), 2);
}

#[test]
fn two_sequential_enable_drop_cycles_each_decode() {
    for cycle in 0..2 {
        let hooks = p3_glue::enable(&pooled_config())
            .unwrap_or_else(|e| panic!("cycle {cycle}: enable must succeed, got {e}"));
        hooks.set_scoring(true); // Session::new arms it after the self-check
        let payload = hooks.payload_ptr();
        let (_ctx, gate) = xt::gate_tensor();
        let out = drive_step(payload, gate);
        assert!(
            out.iter().all(|p| !p.is_null()),
            "cycle {cycle}: pooled decode through the fresh ctx"
        );
        drop(hooks);
        // The drop must fully unwind (unregister-join → ctx free): the next
        // cycle's enable + ctx create prove it — golem_ctx_free aborts if
        // hooks were still registered.
    }
    assert!(!p3_glue::hooks_registered(), "no hooks outlive the cycles");
}

#[test]
fn concurrent_enables_exactly_one_succeeds() {
    // One pool-enabled session at a time: two enables racing from a shared
    // start must yield exactly one Ok and one PoolInit refusal. Both results
    // are collected while the winner is still alive, so a refusal cannot be
    // explained by the winner having already dropped.
    let start = Barrier::new(2);
    let collected = Barrier::new(3);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|s| {
        for t in 0..2 {
            let (start, collected, tx) = (&start, &collected, tx.clone());
            s.spawn(move || {
                start.wait();
                let result = p3_glue::enable(&pooled_config());
                tx.send((t, result.as_ref().map(|_| ()).map_err(|e| format!("{e:?}"))))
                    .unwrap();
                collected.wait();
                if let Ok(hooks) = result {
                    hooks.set_scoring(true); // Session::new arms it after the self-check
                    let payload = hooks.payload_ptr();
                    let (_ctx, gate) = xt::gate_tensor();
                    for step in 0..8 {
                        let out = drive_step(payload, gate);
                        assert!(
                            out.iter().all(|p| !p.is_null()),
                            "thread {t} step {step}: winner's pooled decode"
                        );
                    }
                }
            });
        }
        drop(tx);
        let results: Vec<_> = (0..2).map(|_| rx.recv().unwrap()).collect();
        collected.wait();
        let oks = results.iter().filter(|(_, r)| r.is_ok()).count();
        let refused = results
            .iter()
            .filter(|(_, r)| matches!(r, Err(e) if e.starts_with("PoolInit")))
            .count();
        assert_eq!((oks, refused), (1, 1), "results: {results:?}");
    });
    assert!(!p3_glue::hooks_registered(), "no hooks outlive the winner");
}

thread_local! {
    static COUNTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Process-wide count while armed: the decode thread plus every background
/// thread (order publisher, store drain readers). Assumes one test per
/// process (nextest); under `cargo test` sibling tests pollute the count.
static ALL_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static ALL: AtomicUsize = AtomicUsize::new(0);

/// Counts allocations on threads that opted in (`COUNT`) and, while armed,
/// on every thread (`ALL`).
struct CountingAlloc;
// SAFETY: forwards to System; the counters are only an observation
// (const-initialised Cells with no destructor, so no allocation on access).
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if ALL_ON.load(Ordering::Relaxed) {
            ALL.fetch_add(1, Ordering::Relaxed);
        }
        let _ = COUNTING.try_with(|on| {
            if on.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        // SAFETY: delegated to the system allocator with the same layout.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: delegated with the layout the allocation came in with.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn trampolines_allocate_nothing_in_steady_state() {
    let hooks = p3_glue::enable(&pooled_config()).expect("enable");
    hooks.set_scoring(true); // Session::new arms it after the self-check
    let payload = hooks.payload_ptr();
    let (_ctx, gate) = xt::gate_tensor();
    // Warm: first sight registers (allocates), first page-ins fill slots.
    let _ = drive_step(payload, gate);

    COUNT.with(|c| c.set(0));
    COUNTING.with(|on| on.set(true));
    for _ in 0..200 {
        let _ = drive_step(payload, gate);
    }
    COUNTING.with(|on| on.set(false));
    let allocs = COUNT.with(std::cell::Cell::get);
    assert_eq!(
        allocs, 0,
        "batched ensure + routing + copy_done must not allocate in steady state"
    );
}

/// Page-padded tiny-moe copy plus a one-replica manifest (store-backed pool),
/// as in `knobs_tiny_moe.rs`.
fn manifest_fixture() -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("golem-p3-hook-fires-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("tiny-moe-320.gguf");
    let mut bytes = std::fs::read(tiny_moe_path()).unwrap();
    bytes.resize(bytes.len().next_multiple_of(4096), 0);
    std::fs::write(&model, &bytes).unwrap();
    let mut f = std::fs::File::open(&model).unwrap();
    let index: Vec<serde_json::Value> = golem_gguf::tensor_entries(&mut f)
        .unwrap()
        .iter()
        .filter(|e| e.expert_kind().is_some())
        .map(|e| {
            let stride = golem_gguf::ggml_row_bytes(e.ggml_type, e.dims[0]).unwrap() * e.dims[1];
            serde_json::json!({
                "tensor": e.name,
                "layer": golem_gguf::blk_of(&e.name).unwrap(),
                "experts": e.dims[2],
                "bytes_per_expert": stride,
                "offsets_bytes": (0..e.dims[2]).map(|i| i * stride).collect::<Vec<_>>(),
            })
        })
        .collect();
    let m = serde_json::json!({
        "name": "tiny-moe",
        "shards": [{
            "file": "tiny-moe-320.gguf",
            "bytes": bytes.len(),
            "sha256": "0".repeat(64),
            "replicas": [dir],
        }],
        "expert_index": index,
    });
    let p = dir.join("manifest.json");
    std::fs::write(&p, m.to_string()).unwrap();
    (model, p)
}

/// Store-backed decode: every counted step misses and pages expert 3 in
/// through the store drain, and no thread allocates. Exclusive residency
/// releases the pool slot at each `copy_done` (the unit is marked VRAM-
/// resident), so the next step's ensure misses again — the 8-expert fixture
/// fits a 1 GiB pool whole, so inclusive residency would only ever hit.
/// GLM-5.3-Flash UD-IQ3_XXS: one decode token = 42 layers × 8 routed units
/// on this path; one allocation per unit would be 336 heap ops per token.
#[test]
fn store_backed_decode_allocates_nothing_on_any_thread() {
    let (model, manifest) = manifest_fixture();
    let cfg = EngineConfig {
        model_path: model,
        manifest_path: Some(manifest),
        residency: golem_engine::config::Residency::Exclusive,
        ..pooled_config()
    };
    let hooks = p3_glue::enable(&cfg).expect("enable");
    assert!(hooks.storage_health().is_some(), "store-backed source");
    hooks.set_scoring(true); // Session::new arms it after the self-check
    let payload = hooks.payload_ptr();
    let (_ctx, gate) = xt::gate_tensor();
    let stats = hooks.state_stats();
    // Warm: first sight registers, the drain readers and scratch settle.
    for _ in 0..8 {
        let _ = drive_step(payload, gate);
    }
    // Quiesce the order publisher: wait until no layer is dirty, so it has
    // nothing left to build once it parks again. Bounded: a publisher that
    // never drains fails the test rather than hanging it.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while hooks.orders_dirty() {
        assert!(
            std::time::Instant::now() < deadline,
            "order publisher did not drain dirty layers within 5 s"
        );
        std::thread::yield_now();
    }

    const STEPS: u64 = 200;
    let pageins0 = stats.demand_slow_pageins.load(Ordering::Relaxed);
    let bytes0 = stats.bytes_paged.load(Ordering::Relaxed);
    let calls0 = hooks.ensure_calls();
    ALL.store(0, Ordering::Relaxed);
    ALL_ON.store(true, Ordering::Relaxed);
    for _ in 0..STEPS {
        let _ = drive_step(payload, gate);
    }
    ALL_ON.store(false, Ordering::Relaxed);
    let allocs = ALL.load(Ordering::Relaxed);

    assert_eq!(hooks.ensure_calls() - calls0, STEPS as usize);
    assert_eq!(
        stats.demand_slow_pageins.load(Ordering::Relaxed) - pageins0,
        STEPS,
        "every counted step must page in through the store drain"
    );
    assert!(
        stats.bytes_paged.load(Ordering::Relaxed) > bytes0,
        "disk bytes must grow"
    );
    assert_eq!(
        allocs, 0,
        "store-backed ensure + routing + copy_done must not allocate on any thread"
    );
}
