//! §5.2–5.4 pread engine: the shared chunk queue plus per-plane CRC32C
//! validation; failover happens drain-locally via exclude-mask requeue.

use std::sync::Arc;

use crate::StoreError;
use crate::queue::{ChunkQueue, ReadHandle, ReadJob};
use crate::replica::{ReplicaRegistry, ReplicaSlot};
use crate::stripe::plan_aligned;

/// Byte source for one chunk. Production impl wraps a [`ReplicaSlot`];
/// tests substitute failing sources (EIO injection without unsafe).
pub trait ChunkSource: Send + Sync {
    /// Read exactly `buf.len()` bytes at `offset` into `buf`.
    fn read_chunk(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()>;
}

impl ChunkSource for ReplicaSlot {
    fn read_chunk(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        // `read_at` already loops until the buffer is full.
        self.read_at(offset, buf)
    }
}

/// Replica handles for one shard, abstracted so tests can interpose failing
/// sources while health marking still flows through [`ShardSources::slot`].
pub trait ShardSources: Send + Sync {
    /// Chunk source per replica of `shard_index`, in manifest order.
    fn slots(&self, shard_index: usize) -> Vec<Arc<dyn ChunkSource>>;
    /// The real slot behind `replica_index`; the engine's health-marking
    /// target.
    fn slot(&self, shard_index: usize, replica_index: usize) -> &ReplicaSlot;
    /// Shard file name (e.g. `s0.gguf`) used in errors and logs.
    fn shard_name(&self, shard_index: usize) -> &str;
    /// Cached physical device identity of one replica (§5.1), resolved at
    /// registry open — the drain's drive→replica map reads this once per
    /// drain, never sysfs per read.
    fn replica_drive(&self, shard_index: usize, replica_index: usize) -> &str {
        self.slot(shard_index, replica_index).drive.as_str()
    }
}

impl ShardSources for ReplicaRegistry {
    fn slots(&self, shard_index: usize) -> Vec<Arc<dyn ChunkSource>> {
        // No wrapper needed: `Arc<ReplicaSlot>` unsized-coerces to
        // `Arc<dyn ChunkSource>`.
        self.shards()[shard_index]
            .replicas
            .iter()
            .map(|r| Arc::clone(r) as Arc<dyn ChunkSource>)
            .collect()
    }

    fn slot(&self, shard_index: usize, replica_index: usize) -> &ReplicaSlot {
        &self.shards()[shard_index].replicas[replica_index]
    }

    fn shard_name(&self, shard_index: usize) -> &str {
        self.slot(shard_index, 0)
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
    }
}

/// A logical page read: shard, byte range, and the §5.3 expected CRC32C over
/// the plane (one entry per `read_plane` call; verified once every chunk of
/// the plane has landed).
pub struct PageRead {
    /// Position of the shard in the manifest.
    pub shard_index: usize,
    /// Absolute byte offset of the logical read.
    pub offset: u64,
    /// Length of the logical read in bytes.
    pub len: usize,
    /// Expected CRC32C over the logical range.
    pub expected_crc: u32,
}

/// Default total reader threads for [`IoEngine::new`] when
/// `EngineConfig::io_threads` is unset: 1–2 in-flight reads per drive is the
/// measured optimum (t8 legs regress to ~2 GiB/s per drive vs 5.8/2.2 at t1).
pub const DEFAULT_IO_THREADS: usize = 2;

/// §5.2 pread engine: owns the shared [`ChunkQueue`] feeding demand-first
/// drains. No persistent worker threads — each `wait()` spawns its own
/// scoped, per-drive readers, so [`Drop`] only fails queued-but-undrained
/// reads.
pub struct IoEngine {
    queue: Arc<ChunkQueue>,
}

impl IoEngine {
    /// New engine; `threads` caps each drain's reader threads (0 = derive one
    /// per physical drive hosting an eligible replica).
    ///
    /// # Panics
    /// Never (the queue validates nothing here); a zero cap means derived.
    pub fn new(threads: usize) -> Self {
        Self {
            queue: Arc::new(ChunkQueue::new(threads)),
        }
    }

    /// Queues a logical read without draining it; the returned handle's
    /// `wait()` does the drain. Exposed so a batch of reads can enqueue
    /// demand-first, then drain together.
    pub fn enqueue<'a>(&self, job: ReadJob<'a>) -> ReadHandle<'a> {
        self.queue.enqueue(job)
    }

    /// Read `[offset, offset+len)` of one shard into `buf` (§5.2–5.4): plan
    /// the aligned covering range in ≤2 MiB chunks, drain them demand-first
    /// across replicas (failed chunks requeue with the failed replica
    /// excluded until a healthy one serves them), then verify the §5.3 plane
    /// CRC32C. All healthy replicas exhausted → [`StoreError::Unavailable`];
    /// checksum mismatch → [`StoreError::Crc`].
    ///
    /// `buf` must be exactly the aligned covering region
    /// (`plan_aligned(offset, len)` bytes): chunk 0 lands at the region base,
    /// so the bytes before the logical range are head scratch; expert bytes
    /// are copied disk→destination exactly once. Any other length →
    /// [`StoreError::Invalid`].
    pub fn read_page(
        &self,
        sources: &dyn ShardSources,
        req: &PageRead,
        buf: &mut [u8],
    ) -> Result<(), StoreError> {
        let covering: usize = plan_aligned(req.offset, req.len).iter().map(|c| c.1).sum();
        if buf.len() != covering {
            // Caller contract violation: `Err`, never a `split_at_mut` panic
            // inside the queue (§5.4 destination contract).
            return Err(StoreError::Invalid {
                msg: format!(
                    "destination is {} bytes but the covering region of [{}..] is {covering}",
                    buf.len(),
                    req.offset - req.offset % 4096
                ),
            });
        }
        self.enqueue(ReadJob {
            shard_index: req.shard_index,
            offset: req.offset,
            len: req.len,
            expected_crc: req.expected_crc,
            prefetch: false,
            sources,
            dst: buf,
        })
        .wait()
    }
}

impl Drop for IoEngine {
    fn drop(&mut self) {
        // Queued-but-undrained reads fail at their next wait(); in-flight
        // drains own their chunks and are unaffected (they are scope-joined
        // by their own wait()).
        self.queue.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelManifest, ShardEntry};
    use crate::replica::ReplicaHealth;
    use crate::stripe::CHUNK;
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const SHARD_BYTES: usize = 8192;
    const SHARD_FILE: &str = "s0.gguf";

    /// Deterministic fixture pattern over absolute byte index: byte i =
    /// (i*7+3) % 251.
    fn pattern(off: usize, len: usize) -> Vec<u8> {
        (off..off + len)
            .map(|i| ((i * 7 + 3) % 251) as u8)
            .collect()
    }

    /// Builds manifest + fixture files over `n` replica dirs, each with the
    /// same deterministic pattern. `tag` MUST be unique per test (nextest
    /// runs parallel processes over shared /tmp).
    fn fixture(n: usize, tag: &str) -> (PathBuf, ModelManifest) {
        let root = std::env::temp_dir().join(format!("golem-store-engine-{tag}"));
        let mut replicas = Vec::with_capacity(n);
        for r in 0..n {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, SHARD_BYTES)).unwrap();
            replicas.push(dir);
        }
        let manifest = ModelManifest {
            name: tag.to_owned(),
            shards: vec![ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: SHARD_BYTES as u64,
                sha256: "0".repeat(64),
                replicas,
            }],
            ..Default::default()
        };
        (root, manifest)
    }

    /// XOR `range` bytes of `replica_dir`'s shard file with 0xff in place
    /// (bit-rot injection on disk).
    fn flip(replica_dir: &Path, range: std::ops::Range<usize>) {
        let path = replica_dir.join(SHARD_FILE);
        let mut data = std::fs::read(&path).unwrap();
        for i in range {
            data[i] ^= 0xff;
        }
        std::fs::write(path, data).unwrap();
    }

    /// §5.3 expected plane CRC32C, computed from the pristine pattern
    /// (mirrored replicas are byte-identical).
    fn crc(off: u64, len: usize) -> u32 {
        crc32c::crc32c(&pattern(off as usize, len))
    }

    /// Delegates to the real slot until the countdown hits 0, then fails with
    /// `fail_with` (EIO flavored; `InvalidData` simulates rot) — no unsafe,
    /// no touching the real file.
    struct FailingSource {
        inner: Arc<ReplicaSlot>,
        fail_nth: AtomicUsize,
        fail_with: ErrorKind,
    }

    impl ChunkSource for FailingSource {
        fn read_chunk(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
            if self.fail_nth.load(Ordering::SeqCst) == 0 {
                return Err(if self.fail_with == ErrorKind::InvalidData {
                    std::io::Error::new(self.fail_with, "simulated bit rot")
                } else {
                    std::io::Error::from_raw_os_error(libc::EIO)
                });
            }
            self.fail_nth.fetch_sub(1, Ordering::SeqCst);
            self.inner.read_at(offset, buf)
        }
    }

    /// Test [`ShardSources`]: real registry + per-replica FailingSources via
    /// `(replica_index, fail_nth, ErrorKind)` triples.
    struct TestSources {
        reg: ReplicaRegistry,
        fail: Vec<(usize, usize, ErrorKind)>,
    }

    impl ShardSources for TestSources {
        fn slots(&self, shard_index: usize) -> Vec<Arc<dyn ChunkSource>> {
            self.reg.shards()[shard_index]
                .replicas
                .iter()
                .enumerate()
                .map(
                    |(replica, slot)| match self.fail.iter().find(|(r, _, _)| *r == replica) {
                        Some((_, nth, kind)) => Arc::new(FailingSource {
                            inner: Arc::clone(slot),
                            fail_nth: AtomicUsize::new(*nth),
                            fail_with: *kind,
                        }) as Arc<dyn ChunkSource>,
                        None => Arc::clone(slot) as Arc<dyn ChunkSource>,
                    },
                )
                .collect()
        }

        fn slot(&self, shard_index: usize, replica_index: usize) -> &ReplicaSlot {
            self.reg.slot(shard_index, replica_index)
        }

        fn shard_name(&self, shard_index: usize) -> &str {
            self.reg.shard_name(shard_index)
        }
    }

    /// Full-shard read through the engine into a fresh covering region.
    fn read_whole(engine: &IoEngine, sources: &TestSources) -> Result<Vec<u8>, StoreError> {
        let covering: usize = plan_aligned(0, SHARD_BYTES).iter().map(|c| c.1).sum();
        let mut buf = vec![0u8; covering];
        engine.read_page(
            sources,
            &PageRead {
                shard_index: 0,
                offset: 0,
                len: SHARD_BYTES,
                expected_crc: crc(0, SHARD_BYTES),
            },
            &mut buf,
        )?;
        Ok(buf)
    }

    #[test]
    fn aligned_reads_match_pattern_for_1_2_4_replicas() {
        for n in [1, 2, 4] {
            for (off, len) in [
                (0usize, 4096usize),
                (5, 1000),
                (1023, 1025),
                (4000, 96),
                (0, 1),
            ] {
                let tag = format!("aligned-{n}-{off}-{len}");
                let (_root, manifest) = fixture(n, &tag);
                let reg = ReplicaRegistry::open(&manifest).unwrap();
                let sources = TestSources {
                    reg,
                    fail: Vec::new(),
                };
                let engine = IoEngine::new(4);
                let covering: usize = plan_aligned(off as u64, len).iter().map(|c| c.1).sum();
                let mut buf = vec![0u8; covering];
                engine
                    .read_page(
                        &sources,
                        &PageRead {
                            shard_index: 0,
                            offset: off as u64,
                            len,
                            expected_crc: crc(off as u64, len),
                        },
                        &mut buf,
                    )
                    .unwrap();
                let head = off % 4096; // logical window inside the covering region
                assert_eq!(
                    &buf[head..head + len],
                    &pattern(off, len)[..],
                    "n={n} off={off} len={len}"
                );
            }
        }
    }

    #[test]
    fn eio_fails_over_mid_read_2_replicas() {
        let (_root, manifest) = fixture(2, "eio-failover-2");
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, ErrorKind::Other)],
        };
        let engine = IoEngine::new(4);
        let buf = read_whole(&engine, &sources).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
        assert_eq!(sources.slot(0, 0).health(), ReplicaHealth::Degraded);
        assert_eq!(sources.slot(0, 1).health(), ReplicaHealth::Healthy);
    }

    #[test]
    fn eio_fails_over_mid_read_4_replicas() {
        // 3 dead replicas: only replica 3 remains, the read still completes.
        let (_root, manifest) = fixture(4, "eio-failover-4");
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: vec![
                (0, 0, ErrorKind::Other),
                (1, 0, ErrorKind::Other),
                (2, 0, ErrorKind::Other),
            ],
        };
        let engine = IoEngine::new(4);
        let buf = read_whole(&engine, &sources).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
        for r in 0..3 {
            assert_eq!(sources.slot(0, r).health(), ReplicaHealth::Degraded);
        }
        assert_eq!(sources.slot(0, 3).health(), ReplicaHealth::Healthy);
    }

    #[test]
    fn degraded_replica_is_not_used_on_the_next_read() {
        // First read degrades replica 0; the second read must still be
        // correct (the degraded replica stays excluded).
        let (_root, manifest) = fixture(4, "reassign");
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, ErrorKind::Other)],
        };
        let engine = IoEngine::new(2);
        let first = read_whole(&engine, &sources).unwrap();
        assert_eq!(first, pattern(0, SHARD_BYTES));
        let second = read_whole(&engine, &sources).unwrap();
        assert_eq!(second, pattern(0, SHARD_BYTES));
    }

    #[test]
    fn single_replica_eio_is_unavailable_immediately() {
        let (_root, manifest) = fixture(1, "single-eio");
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, ErrorKind::Other)],
        };
        let engine = IoEngine::new(1);
        match read_whole(&engine, &sources) {
            Err(StoreError::Unavailable { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn all_replicas_unavailable_yields_storage_unavailable() {
        let (_root, manifest) = fixture(2, "all-unavailable");
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, ErrorKind::Other), (1, 0, ErrorKind::Other)],
        };
        let engine = IoEngine::new(2);
        match read_whole(&engine, &sources) {
            Err(StoreError::Unavailable { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn rot_is_detected_and_degrades_serving_replicas() {
        // §5.3 with the per-plane CRC: a rotted replica poisons the plane
        // digest, the read errors with Crc, and every replica that served a
        // chunk is degraded (the plane CRC cannot attribute one replica).
        let (root, manifest) = fixture(2, "rot-one");
        flip(&root.join("r0"), 100..108);
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: Vec::new(),
        };
        let engine = IoEngine::new(2);
        match read_whole(&engine, &sources) {
            Err(StoreError::Crc { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Crc, got {other:?}"),
        }
        // Only the replica that actually served chunks is degraded (all
        // workers pick the first healthy replica, so that is replica 0).
        assert_eq!(sources.slot(0, 0).health(), ReplicaHealth::Degraded);
        assert_eq!(sources.slot(0, 1).health(), ReplicaHealth::Healthy);
        // The pristine mirror still serves the read.
        let buf = read_whole(&engine, &sources).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
    }

    /// T5 acceptance: manifest JSON on disk → load_manifest →
    /// ReplicaRegistry::open → IoEngine, real-file rot.
    #[test]
    fn t5_end_to_end_manifest_to_engine_with_real_file_rot() {
        let (root, manifest) = fixture(4, "t5e2e");
        let manifest_path = root.join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();
        let loaded = crate::manifest::load_manifest(&manifest_path).unwrap();
        let reg = ReplicaRegistry::open(&loaded).unwrap();
        let engine = IoEngine::new(DEFAULT_IO_THREADS);
        let req = PageRead {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crc: crc(0, SHARD_BYTES),
        };

        // Pristine read: PASS, all replicas Healthy.
        let covering: usize = plan_aligned(0, SHARD_BYTES).iter().map(|c| c.1).sum();
        let mut buf = vec![0u8; covering];
        engine.read_page(&reg, &req, &mut buf).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
        for r in 0..4 {
            assert_eq!(reg.slot(0, r).health(), ReplicaHealth::Healthy);
        }

        // Rot on one replica: plane CRC mismatches → Err(Crc), the serving
        // replica (first healthy = r0) is degraded — silent corruption is
        // impossible.
        flip(&root.join("r0"), 100..103);
        match engine.read_page(&reg, &req, &mut buf) {
            Err(StoreError::Crc { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Crc, got {other:?}"),
        }
        assert_eq!(reg.slot(0, 0).health(), ReplicaHealth::Degraded);
        assert_eq!(reg.slot(0, 1).health(), ReplicaHealth::Healthy);

        // The pristine mirrors still serve the read.
        engine.read_page(&reg, &req, &mut buf).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
    }

    #[test]
    fn queued_but_undrained_reads_fail_on_engine_drop() {
        // Drop the engine with a second read still queued: its wait() gets
        // Err (no hang).
        let (_root, manifest) = fixture(2, "drop-queued");
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: Vec::new(),
        };
        let engine = IoEngine::new(1);
        let mut first_dst = vec![0u8; SHARD_BYTES];
        let mut second_dst = vec![0u8; SHARD_BYTES];
        let first = engine.enqueue(ReadJob {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crc: crc(0, SHARD_BYTES),
            prefetch: false,
            sources: &sources,
            dst: &mut first_dst,
        });
        let second = engine.enqueue(ReadJob {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crc: crc(0, SHARD_BYTES),
            prefetch: true,
            sources: &sources,
            dst: &mut second_dst,
        });
        first.wait().unwrap();
        drop(engine);
        match second.wait() {
            Err(StoreError::Unavailable { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Unavailable after drop, got {other:?}"),
        }
    }

    #[test]
    fn chunk_size_is_two_mib() {
        assert_eq!(CHUNK, 2 * 1024 * 1024);
    }
}
