//! §5.2–5.4 pread engine: thread-pool chunk reads, CRC32C validation, and
//! per-chunk failover across replicas.

use std::io;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::StoreError;
use crate::replica::{ReplicaHealth, ReplicaRegistry, ReplicaSlot};
use crate::stripe::{Chunk, plan_striped};

/// Byte source for one chunk. Production impl wraps a [`ReplicaSlot`];
/// tests substitute failing sources (EIO injection without unsafe).
pub trait ChunkSource: Send + Sync {
    /// Read exactly `buf.len()` bytes at `offset` into `buf`.
    fn read_chunk(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
}

impl ChunkSource for ReplicaSlot {
    fn read_chunk(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
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

/// A logical page read: shard, byte range, and the §5.3 expected CRC32C
/// per striped chunk (caller's page table supplies them).
pub struct PageRead<'a> {
    /// Position of the shard in the manifest.
    pub shard_index: usize,
    /// Absolute byte offset of the logical read.
    pub offset: u64,
    /// Length of the logical read in bytes.
    pub len: usize,
    /// Expected CRC32C per striped chunk; `len == plan_striped(..).len()`.
    pub expected_crcs: &'a [u32],
}

/// Default worker count for [`IoEngine::new`].
pub const DEFAULT_IO_THREADS: usize = 8;

/// One queued chunk read; `index` is the chunk's position in the striped plan
/// and is echoed back through `reply` so results reassemble out of order.
struct Job {
    index: usize,
    source: Arc<dyn ChunkSource>,
    offset: u64,
    len: usize,
    reply: mpsc::Sender<(usize, io::Result<Vec<u8>>)>,
}

/// Fixed pool of chunk-read workers over `std::sync::mpsc` (§5.2). Workers
/// loop until the queue closes; [`Drop`] closes it, then joins.
pub struct IoEngine {
    workers: Vec<JoinHandle<()>>,
    job_tx: Option<mpsc::Sender<Job>>,
}

impl IoEngine {
    /// Spawn `threads` workers.
    ///
    /// # Panics
    /// Asserts `threads > 0` (zero-worker pool would deadlock reply collection).
    pub fn new(threads: usize) -> Self {
        assert!(threads > 0, "IoEngine needs at least one worker");
        let (tx, rx) = mpsc::channel::<Job>();
        // `Receiver` is Send-but-not-Sync: share one queue via Mutex.
        let rx = Arc::new(Mutex::new(rx));
        let workers = (0..threads)
            .map(|_| {
                let rx = Arc::clone(&rx);
                std::thread::spawn(move || {
                    loop {
                        let job = {
                            let Ok(rx) = rx.lock() else { continue };
                            match rx.recv() {
                                Ok(job) => job,
                                Err(_) => return, // queue closed: no more senders
                            }
                        };
                        let mut buf = vec![0u8; job.len];
                        let res = job.source.read_chunk(job.offset, &mut buf).map(|()| buf);
                        let _ = job.reply.send((job.index, res));
                    }
                })
            })
            .collect();
        Self {
            workers,
            job_tx: Some(tx),
        }
    }

    /// Read `[offset, offset+len)` of one shard into `buf` (§5.2–5.4): stripe
    /// the range across replicas, verify each chunk's CRC32C against
    /// `expected_crcs`, and on I/O error or checksum mismatch mark the source
    /// replica [`ReplicaHealth::Degraded`] and retry that chunk from the next
    /// healthy replica with the same expected CRC (mirrored content is
    /// byte-identical). No healthy replica remains =>
    /// [`StoreError::Unavailable`].
    ///
    /// # Panics
    /// When `expected_crcs.len()` differs from the striped plan length.
    pub fn read_page(
        &self,
        sources: &dyn ShardSources,
        req: &PageRead<'_>,
        buf: &mut [u8],
    ) -> Result<(), StoreError> {
        let slots = sources.slots(req.shard_index);
        let plan = plan_striped(req.offset, req.len, slots.len());
        assert_eq!(
            plan.len(),
            req.expected_crcs.len(),
            "expected_crcs must have one entry per striped chunk"
        );

        // Dispatch one job per striped chunk. Chunks whose assigned replica is
        // already Degraded are re-served whole from the first healthy replica
        // (plan_single semantics), consuming their CRC by striped-plan
        // position.
        let (reply_tx, reply_rx) = mpsc::channel();
        for (index, chunk) in plan.iter().enumerate() {
            let replica =
                self.assigned_replica(sources, &slots, req.shard_index, chunk.replica_index)?;
            let job = Job {
                index,
                source: Arc::clone(&slots[replica]),
                offset: chunk.offset,
                len: chunk.len,
                reply: reply_tx.clone(),
            };
            self.job_tx
                .as_ref()
                .expect("job_tx lives until Drop")
                .send(job)
                .expect("io worker dropped the reply channel");
        }
        drop(reply_tx);

        let mut parts: Vec<Option<io::Result<Vec<u8>>>> = (0..plan.len()).map(|_| None).collect();
        while let Ok((index, res)) = reply_rx.recv() {
            parts[index] = Some(res);
        }

        for ((chunk, &expected_crc), part) in plan.iter().zip(req.expected_crcs).zip(parts) {
            let first = part.expect("every dispatched job gets a reply");
            let data = self.serve_chunk(
                sources,
                req.shard_index,
                *chunk,
                expected_crc,
                first,
                &slots,
            )?;
            let at = (chunk.offset - req.offset) as usize;
            buf[at..at + chunk.len].copy_from_slice(&data);
        }
        Ok(())
    }

    /// Replica to serve a chunk from: its assigned replica unless already
    /// [`ReplicaHealth::Degraded`], else the first healthy replica.
    fn assigned_replica(
        &self,
        sources: &dyn ShardSources,
        slots: &[Arc<dyn ChunkSource>],
        shard_index: usize,
        assigned: usize,
    ) -> Result<usize, StoreError> {
        if sources.slot(shard_index, assigned).health() == ReplicaHealth::Healthy {
            return Ok(assigned);
        }
        (0..slots.len())
            .find(|&r| sources.slot(shard_index, r).health() == ReplicaHealth::Healthy)
            .ok_or_else(|| StoreError::Unavailable {
                shard: sources.shard_name(shard_index).to_owned(),
            })
    }

    /// First-attempt result plus failover for one chunk: verify the first
    /// read (I/O or CRC32C mismatch), degrade the replica that failed, then
    /// retry the whole chunk from each remaining healthy replica until one
    /// verifies (same expected CRC: mirrors are byte-identical).
    fn serve_chunk(
        &self,
        sources: &dyn ShardSources,
        shard_index: usize,
        chunk: Chunk,
        expected_crc: u32,
        first: io::Result<Vec<u8>>,
        slots: &[Arc<dyn ChunkSource>],
    ) -> Result<Vec<u8>, StoreError> {
        let first = match first {
            Ok(data) if crc32c::crc32c(&data) == expected_crc => return Ok(data),
            Ok(_) => format!("crc32c mismatch at offset {}", chunk.offset),
            Err(e) => format!("io error: {e}"),
        };
        self.degrade(sources, shard_index, chunk.replica_index, &chunk, &first);
        for (replica, source) in slots.iter().enumerate() {
            if sources.slot(shard_index, replica).health() != ReplicaHealth::Healthy {
                continue;
            }
            let mut data = vec![0u8; chunk.len];
            let attempt: Result<Vec<u8>, String> = match source.read_chunk(chunk.offset, &mut data)
            {
                Ok(()) if crc32c::crc32c(&data) == expected_crc => Ok(data),
                Ok(()) => Err(format!("crc32c mismatch at offset {}", chunk.offset)),
                Err(e) => Err(format!("io error: {e}")),
            };
            match attempt {
                Ok(data) => return Ok(data),
                Err(why) => self.degrade(sources, shard_index, replica, &chunk, &why),
            }
        }
        Err(StoreError::Unavailable {
            shard: sources.shard_name(shard_index).to_owned(),
        })
    }

    /// One-way health flip plus a §5.3 log line naming shard, replica, and
    /// chunk range.
    fn degrade(
        &self,
        sources: &dyn ShardSources,
        shard_index: usize,
        replica_index: usize,
        chunk: &Chunk,
        why: &str,
    ) {
        sources.slot(shard_index, replica_index).mark_degraded();
        tracing::warn!(
            shard = sources.shard_name(shard_index),
            replica = replica_index,
            offset = chunk.offset,
            len = chunk.len,
            reason = %why,
            "chunk read failed; replica marked Degraded"
        );
    }
}

impl Drop for IoEngine {
    fn drop(&mut self) {
        // Close the queue so workers exit, then join them.
        self.job_tx = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelManifest, ShardEntry};
    use crate::replica::ReplicaHealth;
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const SHARD_BYTES: usize = 4096;
    const SHARD_FILE: &str = "s0.gguf";

    /// Deterministic fixture pattern over absolute byte index: byte i =
    /// (i*7+3) % 251.
    fn pattern(off: usize, len: usize) -> Vec<u8> {
        (off..off + len)
            .map(|i| ((i * 7 + 3) % 251) as u8)
            .collect()
    }

    /// Builds manifest + fixture files over `n` replica dirs, each with the
    /// same deterministic 4 KiB pattern. `tag` MUST be unique per test
    /// (nextest runs parallel processes over shared /tmp).
    /// Returns (root_dir, manifest).
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

    /// Expected §5.3 CRC32C per striped chunk, computed from the pristine
    /// pattern (mirrored replicas are byte-identical, so the CRCs are
    /// replica-independent).
    fn crcs(off: u64, len: usize, replicas: usize) -> Vec<u32> {
        plan_striped(off, len, replicas)
            .iter()
            .map(|c| crc32c::crc32c(&pattern(c.offset as usize, c.len)))
            .collect()
    }

    /// Delegates to the real slot until the countdown hits 0, then fails with
    /// `fail_with` (EIO flavored; `InvalidData` would simulate rot) — no
    /// unsafe, no touching the real file.
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
    /// `(replica_index, fail_nth, ErrorKind)` triples. Health marking flows
    /// through `slot()` — the engine's normal path.
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

    #[test]
    fn striped_read_matches_pattern_for_1_2_4_replicas() {
        for n in [1, 2, 4] {
            for (off, len) in [
                (0usize, 4096usize),
                (5, 1000),
                (1023, 1025),
                (4000, 96),
                (0, 1),
            ] {
                let tag = format!("striped-{n}-{off}-{len}");
                let (_root, manifest) = fixture(n, &tag);
                let reg = ReplicaRegistry::open(&manifest).unwrap();
                let sources = TestSources {
                    reg,
                    fail: Vec::new(),
                };
                let engine = IoEngine::new(4);
                let expected_crcs = crcs(off as u64, len, n);
                let req = PageRead {
                    shard_index: 0,
                    offset: off as u64,
                    len,
                    expected_crcs: &expected_crcs,
                };
                let mut buf = vec![0u8; len];
                engine.read_page(&sources, &req, &mut buf).unwrap();
                assert_eq!(buf, pattern(off, len), "n={n} off={off} len={len}");
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
        let expected_crcs = crcs(0, SHARD_BYTES, 2);
        let req = PageRead {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crcs: &expected_crcs,
        };
        let mut buf = vec![0u8; SHARD_BYTES];
        engine.read_page(&sources, &req, &mut buf).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
        assert_eq!(sources.slot(0, 0).health(), ReplicaHealth::Degraded);
        assert_eq!(sources.slot(0, 1).health(), ReplicaHealth::Healthy);
    }

    #[test]
    fn eio_fails_over_mid_read_4_replicas() {
        let (_root, manifest) = fixture(4, "eio-failover-4");
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, ErrorKind::Other), (2, 0, ErrorKind::Other)],
        };
        let engine = IoEngine::new(4);
        let expected_crcs = crcs(0, SHARD_BYTES, 4);
        let req = PageRead {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crcs: &expected_crcs,
        };
        let mut buf = vec![0u8; SHARD_BYTES];
        engine.read_page(&sources, &req, &mut buf).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
        assert_eq!(sources.slot(0, 0).health(), ReplicaHealth::Degraded);
        assert_eq!(sources.slot(0, 1).health(), ReplicaHealth::Healthy);
        assert_eq!(sources.slot(0, 2).health(), ReplicaHealth::Degraded);
        assert_eq!(sources.slot(0, 3).health(), ReplicaHealth::Healthy);
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
        let expected_crcs = crcs(0, SHARD_BYTES, 2);
        let req = PageRead {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crcs: &expected_crcs,
        };
        let mut buf = vec![0u8; SHARD_BYTES];
        match engine.read_page(&sources, &req, &mut buf) {
            Err(StoreError::Unavailable { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn rot_on_one_replica_is_detected_and_failed_over() {
        let (root, manifest) = fixture(2, "rot-one");
        // Expected CRCs from the pristine pattern BEFORE flipping bytes.
        let expected_crcs = crcs(0, SHARD_BYTES, 2);
        flip(&root.join("r0"), 100..108);
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: Vec::new(),
        };
        let engine = IoEngine::new(2);
        let req = PageRead {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crcs: &expected_crcs,
        };
        let mut buf = vec![0u8; SHARD_BYTES];
        engine.read_page(&sources, &req, &mut buf).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
        assert_eq!(sources.slot(0, 0).health(), ReplicaHealth::Degraded);
        assert_eq!(sources.slot(0, 1).health(), ReplicaHealth::Healthy);
    }

    /// T5 acceptance: manifest JSON on disk → load_manifest →
    /// ReplicaRegistry::open → IoEngine, real-file rot on all four replicas.
    #[test]
    fn t5_end_to_end_manifest_to_engine_with_real_file_rot() {
        let (root, manifest) = fixture(4, "t5e2e");
        let manifest_path = root.join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();
        let loaded = crate::manifest::load_manifest(&manifest_path).unwrap();
        let reg = ReplicaRegistry::open(&loaded).unwrap();
        let engine = IoEngine::new(DEFAULT_IO_THREADS);
        let expected_crcs = crcs(0, SHARD_BYTES, 4);
        let req = PageRead {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crcs: &expected_crcs,
        };

        // Pristine read: full shard, PASS, all replicas Healthy.
        let mut buf = vec![0u8; SHARD_BYTES];
        engine.read_page(&reg, &req, &mut buf).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
        for r in 0..4 {
            assert_eq!(reg.slot(0, r).health(), ReplicaHealth::Healthy);
        }

        // Rot: 3 bytes inside each damaged replica's own striped chunk
        // (4 replicas × 4096 bytes → chunk i covers [i*1024, (i+1)*1024)).
        // Expected CRCs unchanged (pristine pattern). Chunks 0/1 fail CRC on
        // their assigned replica and fail over to the pristine mirrors.
        flip(&root.join("r0"), 100..103);
        flip(&root.join("r1"), 1100..1103);
        engine.read_page(&reg, &req, &mut buf).unwrap();
        assert_eq!(buf, pattern(0, SHARD_BYTES));
        assert_eq!(reg.slot(0, 0).health(), ReplicaHealth::Degraded);
        assert_eq!(reg.slot(0, 1).health(), ReplicaHealth::Degraded);
        assert_eq!(reg.slot(0, 2).health(), ReplicaHealth::Healthy);
        assert_eq!(reg.slot(0, 3).health(), ReplicaHealth::Healthy);

        // Rot the same bytes on the remaining replicas: chunk 0 now fails on
        // every replica → Unavailable, not silent corruption.
        flip(&root.join("r2"), 100..103);
        flip(&root.join("r3"), 100..103);
        match engine.read_page(&reg, &req, &mut buf) {
            Err(StoreError::Unavailable { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn rot_on_all_replicas_is_unavailable_not_silent() {
        let (root, manifest) = fixture(2, "rot-all");
        let expected_crcs = crcs(0, SHARD_BYTES, 2);
        flip(&root.join("r0"), 100..108);
        flip(&root.join("r1"), 100..108);
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: Vec::new(),
        };
        let engine = IoEngine::new(2);
        let req = PageRead {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crcs: &expected_crcs,
        };
        let mut buf = vec![0u8; SHARD_BYTES];
        match engine.read_page(&sources, &req, &mut buf) {
            Err(StoreError::Unavailable { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }
}
