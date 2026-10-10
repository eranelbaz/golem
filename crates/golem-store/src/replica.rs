//! Replica registry: open handles per (shard, replica) pair, per-replica
//! health, and best-effort same-drive detection (§5.1).

use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use crate::StoreError;
use crate::manifest::ModelManifest;
use crate::sys::{self, DIRECT_ALIGN};

/// Read seam over one open replica handle (§5.4): full reads plus explicit
/// page-cache eviction, so the direct/buffered fallback is testable without
/// real filesystems. Implemented for [`std::fs::File`]; tests inject fakes.
pub trait FileRead: Send + Sync {
    /// Read exactly `buf.len()` bytes at `off`, retrying short reads.
    fn read_full(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()>;
    /// Best-effort page-cache eviction of `[off, off + len)`; advisory, so a
    /// failure is reported and swallowed by the caller — never a read error.
    fn drop_cache(&self, off: u64, len: usize) -> std::io::Result<()>;
}

impl FileRead for std::fs::File {
    fn read_full(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let mut filled = 0;
        while filled < buf.len() {
            match self.read_at(&mut buf[filled..], off + filled as u64) {
                // `File::read_at` does not retry EINTR; a spurious interrupt
                // must not surface as a device error.
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "short read at end of file",
                    ));
                }
                Ok(n) => filled += n,
            }
        }
        Ok(())
    }

    fn drop_cache(&self, off: u64, len: usize) -> std::io::Result<()> {
        if sys::drop_page_cache(self, off, len) {
            Ok(())
        } else {
            Err(std::io::Error::other("posix_fadvise failed"))
        }
    }
}

impl FileRead for Box<dyn FileRead> {
    fn read_full(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
        (**self).read_full(off, buf)
    }
    fn drop_cache(&self, off: u64, len: usize) -> std::io::Result<()> {
        (**self).drop_cache(off, len)
    }
}

/// Health of one replica path. One-way Healthy → Degraded this task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaHealth {
    /// Readable; eligible for striped reads.
    Healthy = 0,
    /// Failed; excluded from reads until re-opened.
    Degraded = 1,
}

/// Open handles for every (shard, replica) pair, plus health state.
pub struct ReplicaRegistry {
    shards: Vec<ShardReplicas>,
}

/// All replica slots of one shard.
pub struct ShardReplicas {
    /// Position of this shard in the manifest's `shards` vector.
    pub shard_index: usize,
    /// Arc'd so readers can hand out `Arc<dyn ChunkSource>` clones from `&self`
    /// (`File` is not `Clone`).
    pub replicas: Vec<Arc<ReplicaSlot>>,
}

/// One opened replica file plus its health state and its direct/buffered
/// read handles (§5.4, H-9).
pub struct ReplicaSlot {
    /// Full path: replica dir joined with the shard file name.
    pub path: PathBuf,
    /// `O_DIRECT` handle, `None` where the open-time probe failed.
    direct: Option<Box<dyn FileRead>>,
    /// Buffered handle, always present: the fallback needs it even when the
    /// direct handle exists.
    buffered: Box<dyn FileRead>,
    /// Cleared on the first direct-read failure (H-9 runtime fallback, not a
    /// degrade); the slot then stays buffered + `DONTNEED`.
    direct_ok: AtomicBool,
    /// Cleared after the first `DONTNEED` failure so the warn fires once per
    /// slot, not once per read (a per-read warn floods the log).
    warned_dontneed: AtomicBool,
    /// Cached physical device identity (§5.1): sysfs block name or numeric
    /// `st_dev` fallback, resolved once at open — never re-stat'ed on the
    /// read path (§5.2 hot loop is allocation- and syscall-light).
    pub drive: String,
    /// 0 = Healthy, 1 = Degraded ([`ReplicaHealth`] as u8).
    pub health: AtomicU8,
    /// Count of one-way degradations since open; feeds diagnostics and
    /// burn-down decisions upstream (health itself is `health()`, never this
    /// counter).
    errors: AtomicU64,
    /// Chunks served successfully from this replica since open (diagnostics:
    /// measured share vs. the uniform striping expectation).
    served_chunks: AtomicU64,
    /// Subset of `served_chunks` served through the `O_DIRECT` handle
    /// (diagnostics: `served_chunks - served_direct` = buffered-fallback
    /// chunks; nonzero means a bench's GB/s is not a disk number).
    served_direct: AtomicU64,
}

impl ReplicaSlot {
    /// Current health.
    pub fn health(&self) -> ReplicaHealth {
        match self.health.load(Ordering::Relaxed) {
            1 => ReplicaHealth::Degraded,
            _ => ReplicaHealth::Healthy,
        }
    }

    /// One-way transition to [`ReplicaHealth::Degraded`]; increments the
    /// degradation counter.
    pub fn mark_degraded(&self) {
        self.health
            .store(ReplicaHealth::Degraded as u8, Ordering::Relaxed);
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Degradations since open.
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Successful chunk reads served since open.
    pub fn served_chunks(&self) -> u64 {
        self.served_chunks.load(Ordering::Relaxed)
    }

    /// Whether direct (`O_DIRECT`) reads are still attempted for this slot.
    pub fn direct_ok(&self) -> bool {
        self.direct_ok.load(Ordering::Relaxed)
    }

    /// Chunks served through the direct (`O_DIRECT`) handle since open.
    pub fn served_direct(&self) -> u64 {
        self.served_direct.load(Ordering::Relaxed)
    }

    /// Whether the open-time direct probe passed (a direct handle exists);
    /// `false` means every read on this slot is buffered from the start.
    pub fn direct_available(&self) -> bool {
        self.direct.is_some()
    }

    /// Test-only view: whether a direct handle exists at all.
    #[cfg(test)]
    fn has_direct(&self) -> bool {
        self.direct.is_some()
    }

    /// Read exactly `buf.len()` bytes at `offset` (§5.4, H-9). Direct when
    /// the read is 4 KiB-aligned and the probe passed; any direct failure is
    /// a one-way fallback to buffered + `DONTNEED` — never a degrade. The
    /// buffered fallback never populates the page cache.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        let aligned = offset.is_multiple_of(DIRECT_ALIGN)
            && (buf.len() as u64).is_multiple_of(DIRECT_ALIGN)
            && (buf.as_ptr() as u64).is_multiple_of(DIRECT_ALIGN);
        if aligned
            && self.direct_ok()
            && let Some(d) = &self.direct
        {
            if let Err(e) = d.read_full(offset, buf) {
                if self.direct_ok.swap(false, Ordering::Relaxed) {
                    tracing::warn!(
                        path = %self.path.display(),
                        error = %e,
                        "O_DIRECT replica read failed; slot falls back to buffered + DONTNEED"
                    );
                }
            } else {
                self.served_chunks.fetch_add(1, Ordering::Relaxed);
                self.served_direct.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
        }
        self.buffered.read_full(offset, buf)?;
        self.served_chunks.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = self.buffered.drop_cache(offset, buf.len())
            && self.warned_dontneed.swap(false, Ordering::Relaxed)
        {
            tracing::warn!(
                path = %self.path.display(),
                error = %e,
                "POSIX_FADV_DONTNEED failed; page-cache pressure only"
            );
        }
        Ok(())
    }
}

impl ReplicaRegistry {
    /// Open every replica path in `manifest`; verifies each shard file's size
    /// against `bytes` (§5.3). First failure → [`StoreError::Io`].
    pub fn open(manifest: &ModelManifest) -> Result<Self, StoreError> {
        let mut shards = Vec::with_capacity(manifest.shards.len());
        for (shard_index, shard) in manifest.shards.iter().enumerate() {
            let mut replicas = Vec::with_capacity(shard.replicas.len());
            for replica in &shard.replicas {
                let path = replica.join(&shard.file);
                let io_err = |source: std::io::Error| StoreError::Io {
                    shard: shard.file.clone(),
                    replica: replica.clone(),
                    source,
                };
                let file = File::open(&path).map_err(io_err)?;
                let len = file.metadata().map_err(io_err)?.len();
                if len != shard.bytes {
                    return Err(io_err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "size mismatch: manifest expects {} bytes, file has {len}",
                            shard.bytes
                        ),
                    )));
                }
                let drive = physical_device_name(&path).unwrap_or_else(|| "unknown".to_owned());
                // Open-time probe (§5.4, H-9): open + one aligned 4 KiB read.
                // Failure of open or probe → buffered only; never an open error.
                let direct: Option<Box<dyn FileRead>> = match sys::open_direct(&path) {
                    Ok(f) if shard.bytes >= DIRECT_ALIGN => {
                        let f = Box::new(f);
                        // Page-aligned scratch for the probe read: allocate
                        // one extra block, then slice at the first aligned
                        // offset (allocation addresses are never aligned
                        // themselves).
                        let mut scratch = vec![0u8; DIRECT_ALIGN as usize * 2];
                        let skip = scratch.as_ptr() as usize % DIRECT_ALIGN as usize;
                        let probe = &mut scratch[skip..skip + DIRECT_ALIGN as usize];
                        match f.read_full(0, probe) {
                            Ok(()) => Some(f),
                            Err(e) => {
                                tracing::warn!(
                                    replica = %replica.display(),
                                    error = %e,
                                    "O_DIRECT probe read failed; replica reads buffered + DONTNEED"
                                );
                                None
                            }
                        }
                    }
                    Ok(_) => None, // file smaller than one aligned probe block
                    Err(e) => {
                        tracing::warn!(
                            replica = %replica.display(),
                            error = %e,
                            "O_DIRECT open failed; replica reads buffered + DONTNEED"
                        );
                        None
                    }
                };
                replicas.push(Arc::new(ReplicaSlot {
                    path,
                    drive,
                    direct,
                    buffered: Box::new(file),
                    direct_ok: AtomicBool::new(true),
                    warned_dontneed: AtomicBool::new(true),
                    health: AtomicU8::new(ReplicaHealth::Healthy as u8),
                    errors: AtomicU64::new(0),
                    served_chunks: AtomicU64::new(0),
                    served_direct: AtomicU64::new(0),
                }));
            }
            shards.push(ShardReplicas {
                shard_index,
                replicas,
            });
        }
        Ok(Self { shards })
    }

    /// Per-shard replica slots, in manifest order.
    pub fn shards(&self) -> &[ShardReplicas] {
        &self.shards
    }

    /// Replicas of `shard_index` currently in [`ReplicaHealth::Healthy`].
    pub fn healthy_replicas(&self, shard_index: usize) -> Vec<&ReplicaSlot> {
        self.shards
            .get(shard_index)
            .map(|s| {
                s.replicas
                    .iter()
                    .filter(|r| r.health() == ReplicaHealth::Healthy)
                    .map(|r| &**r)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// §5.1 warn-only diagnostics: one line per physical device hosting two or
    /// more replicas of the same shard (no bandwidth aggregation).
    pub fn shared_device_warnings(&self) -> Vec<String> {
        self.shards
            .iter()
            .flat_map(|shard| {
                let paths: Vec<PathBuf> = shard.replicas.iter().map(|r| r.path.clone()).collect();
                shared_device_pairs(&paths)
            })
            .collect()
    }
}

/// One warning per physical device hosting two or more of `roots`
/// (insertion-ordered grouping; undeterminable devices never warn). Shared
/// by [`ReplicaRegistry::shared_device_warnings`] and the §5.5 import
/// same-disk warnings — one implementation of the device-pair rule.
pub(crate) fn shared_device_pairs(roots: &[PathBuf]) -> Vec<String> {
    let mut by_device: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for root in roots {
        let Some(device) = physical_device_name(root) else {
            continue; // undeterminable → never a warning
        };
        match by_device.iter_mut().find(|(d, _)| *d == device) {
            Some((_, paths)) => paths.push(root.clone()),
            None => by_device.push((device, vec![root.clone()])),
        }
    }
    by_device
        .into_iter()
        .filter(|(_, paths)| paths.len() >= 2)
        .map(|(device, paths)| {
            format!(
                "replicas share a physical device; no bandwidth aggregation ({device}): {paths:?}"
            )
        })
        .collect()
}

/// Best-effort physical device identity for `path`.
///
/// Prefers the sysfs block-device name (e.g. `nvme0n1`); falls back to the
/// numeric `st_dev` string when sysfs has no entry (non-block filesystem,
/// restricted container) — same-`st_dev` paths still group as shared. Returns
/// `None` only when even the nearest existing ancestor of `path` cannot be
/// stat'ed. Never panics.
pub fn physical_device_name(path: &Path) -> Option<String> {
    // Resolve to nearest existing ancestor: a not-yet-created child shares its
    // parent's device.
    let mut cur = path;
    let dev = loop {
        match std::fs::metadata(cur) {
            Ok(meta) => break meta.dev(),
            Err(_) => cur = cur.parent()?,
        }
    };
    let major = libc::major(dev);
    let minor = libc::minor(dev);
    if let Ok(resolved) = std::fs::canonicalize(format!("/sys/dev/block/{major}:{minor}")) {
        for name in resolved.components().rev() {
            let Component::Normal(s) = name else { continue };
            let name = s.to_string_lossy().into_owned();
            if parent_device_name(&name) == Some(name.clone())
                && Path::new("/sys/block").join(&name).exists()
            {
                return Some(name);
            }
        }
    }
    // No sysfs entry: numeric identity still detects same-device sharing.
    Some(dev.to_string())
}

/// Pure helper: `"nvme0n1p2"` → `Some("nvme0n1")`, `"nvme0n1"` →
/// `Some("nvme0n1")`, `"sda1"` → `Some("sda")`, `"loop0"` → `Some("loop0")`.
pub fn parent_device_name(partition: &str) -> Option<String> {
    // loop/ram/zram are already top-level /sys/block devices.
    if partition.starts_with("loop")
        || partition.starts_with("ram")
        || partition.starts_with("zram")
    {
        return Some(partition.to_owned());
    }
    // nvme/mmcblk namespace partitions carry `p<N>` after a device index.
    if let Some((head, tail)) = partition.rsplit_once('p')
        && !head.is_empty()
        && !tail.is_empty()
        && tail.bytes().all(|b| b.is_ascii_digit())
        && head.bytes().any(|b| b.is_ascii_digit())
    {
        return Some(head.to_owned());
    }
    let base = partition.trim_end_matches(|b: char| b.is_ascii_digit());
    if base.is_empty() || !base.bytes().any(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    if base.len() < partition.len() {
        // nvme-style namespace device (`...n<N>`): the prefix before the final
        // `n` already holds a digit, so the name itself is the parent.
        if base.ends_with('n') && base[..base.len() - 1].bytes().any(|b| b.is_ascii_digit()) {
            return Some(partition.to_owned());
        }
        return Some(base.to_owned());
    }
    // No partition digits: input already names a parent-level device.
    Some(partition.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Test storage whose data can sit at a 4 KiB-aligned address: the
    /// returned `skip` marks the aligned window inside the over-allocation.
    fn aligned_storage(len: usize) -> (Vec<u8>, usize) {
        let v = vec![0u8; len + 4096];
        let off = v.as_ptr() as usize % 4096;
        let skip = (4096 - off) % 4096;
        (v, skip)
    }

    fn small_manifest(
        dir: &std::path::Path,
        _name: &str,
        bytes: usize,
    ) -> crate::manifest::ModelManifest {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("a.gguf"), vec![0u8; bytes]).unwrap();
        crate::manifest::ModelManifest {
            name: "t".into(),
            shards: vec![crate::manifest::ShardEntry {
                file: "a.gguf".into(),
                bytes: bytes as u64,
                sha256: "0".repeat(64),
                replicas: vec![dir.to_path_buf()],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn probe_opens_direct_or_falls_back_and_reads_correctly() {
        let dir = std::env::temp_dir().join("golem-store-replica-test");
        let manifest = small_manifest(&dir.join("probe"), "probe", 8192);
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let slot = &*reg.shards()[0].replicas[0];
        // Exactly one of {direct probe ok, direct None} holds.
        assert!(slot.direct_ok() == slot.has_direct());
        let (mut v, skip) = aligned_storage(4096);
        let buf = &mut v[skip..skip + 4096];
        slot.read_at(0, buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    /// Test interposition: direct handle always EIOs; buffered handle serves
    /// pattern bytes and records `drop_cache` calls.
    type DropCalls = std::sync::Arc<Mutex<Vec<(u64, usize)>>>;

    struct FakeReads {
        direct_err: bool,
        drop_calls: DropCalls,
    }

    impl FakeReads {
        fn buffered() -> (Box<Self>, DropCalls) {
            let calls = std::sync::Arc::new(Mutex::new(Vec::new()));
            (
                Box::new(Self {
                    direct_err: false,
                    drop_calls: calls.clone(),
                }),
                calls,
            )
        }
    }

    impl FileRead for FakeReads {
        fn read_full(&self, _off: u64, buf: &mut [u8]) -> std::io::Result<()> {
            if self.direct_err {
                return Err(std::io::Error::from_raw_os_error(libc::EIO));
            }
            for b in buf.iter_mut() {
                *b = 0xAB;
            }
            Ok(())
        }
        fn drop_cache(&self, _off: u64, _len: usize) -> std::io::Result<()> {
            self.drop_calls.lock().unwrap().push((_off, _len));
            Ok(())
        }
    }

    #[test]
    fn direct_eio_falls_back_to_buffered_without_degrading() {
        let slot = ReplicaSlot {
            path: PathBuf::from("fake"),
            drive: "fake".into(),
            direct: Some(Box::new(FakeReads {
                direct_err: true,
                drop_calls: Default::default(),
            })),
            buffered: FakeReads::buffered().0,
            direct_ok: AtomicBool::new(true),
            warned_dontneed: AtomicBool::new(true),
            health: AtomicU8::new(ReplicaHealth::Healthy as u8),
            errors: AtomicU64::new(0),
            served_chunks: AtomicU64::new(0),
            served_direct: AtomicU64::new(0),
        };
        let (mut v, skip) = aligned_storage(4096);
        let buf = &mut v[skip..skip + 4096];
        slot.read_at(0, buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAB));
        assert!(!slot.direct_ok(), "direct_ok flips false after EIO");
        assert_eq!(
            slot.health(),
            ReplicaHealth::Healthy,
            "fallback is not a degrade"
        );
        assert_eq!(slot.errors(), 0);
    }

    #[test]
    fn fallback_reads_invoke_dontneed() {
        let slot = ReplicaSlot {
            path: PathBuf::from("fake"),
            drive: "fake".into(),
            direct: None,
            buffered: FakeReads::buffered().0,
            direct_ok: AtomicBool::new(false),
            warned_dontneed: AtomicBool::new(true),
            health: AtomicU8::new(ReplicaHealth::Healthy as u8),
            errors: AtomicU64::new(0),
            served_chunks: AtomicU64::new(0),
            served_direct: AtomicU64::new(0),
        };
        let (buffered, calls) = FakeReads::buffered();
        let mut slot = slot;
        slot.buffered = buffered;
        let (mut v, skip) = aligned_storage(4096);
        let buf = &mut v[skip..skip + 4096];
        slot.read_at(0, buf).unwrap();
        assert_eq!(*calls.lock().unwrap(), vec![(0, 4096)]);
    }

    #[test]
    fn parent_device_parses_partition_suffixes() {
        assert_eq!(parent_device_name("nvme0n1p2").as_deref(), Some("nvme0n1"));
        assert_eq!(parent_device_name("nvme0n1").as_deref(), Some("nvme0n1"));
        assert_eq!(parent_device_name("sda1").as_deref(), Some("sda"));
        assert_eq!(parent_device_name("loop0").as_deref(), Some("loop0"));
    }

    #[test]
    fn same_dir_paths_share_a_device() {
        let dir = std::env::temp_dir();
        assert!(physical_device_name(&dir).is_some());
        assert_eq!(
            physical_device_name(&dir),
            physical_device_name(&dir.join("x-nonexistent"))
        );
    }

    #[test]
    fn registry_reports_shared_device_warning_for_duplicated_replica_dirs() {
        // manifest with two replicas pointing at the SAME dir → one warning naming the device
        let dir = std::env::temp_dir().join("golem-store-replica-test");
        let shard_dir = dir.join("shard");
        std::fs::create_dir_all(&shard_dir).unwrap();
        let body = vec![0u8; 64];
        std::fs::write(shard_dir.join("a.gguf"), &body).unwrap();
        let manifest = crate::manifest::ModelManifest {
            name: "t".into(),
            shards: vec![crate::manifest::ShardEntry {
                file: "a.gguf".into(),
                bytes: 64,
                sha256: "0".repeat(64),
                replicas: vec![shard_dir.clone(), shard_dir.clone()],
                ..Default::default()
            }],
            ..Default::default()
        };
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let warns = reg.shared_device_warnings();
        assert_eq!(warns.len(), 1);
        assert!(warns[0].contains("no bandwidth aggregation"));
        assert_eq!(reg.healthy_replicas(0).len(), 2);
    }

    #[test]
    fn error_counter_increments_on_degrade() {
        let dir = std::env::temp_dir().join("golem-store-replica-test");
        let shard_dir = dir.join("shard-errctr");
        std::fs::create_dir_all(&shard_dir).unwrap();
        std::fs::write(shard_dir.join("a.gguf"), [0u8; 64]).unwrap();
        let manifest = crate::manifest::ModelManifest {
            name: "t".into(),
            shards: vec![crate::manifest::ShardEntry {
                file: "a.gguf".into(),
                bytes: 64,
                sha256: "0".repeat(64),
                replicas: vec![shard_dir],
                ..Default::default()
            }],
            ..Default::default()
        };
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let slot = &*reg.shards()[0].replicas[0];
        assert_eq!(slot.errors(), 0);
        slot.mark_degraded();
        slot.mark_degraded();
        assert_eq!(slot.errors(), 2);
        assert_eq!(slot.health(), ReplicaHealth::Degraded);
    }

    #[test]
    fn size_mismatch_is_rejected() {
        let dir = std::env::temp_dir().join("golem-store-replica-test");
        let shard_dir = dir.join("shard-short");
        std::fs::create_dir_all(&shard_dir).unwrap();
        std::fs::write(shard_dir.join("a.gguf"), [0u8; 8]).unwrap();
        let manifest = crate::manifest::ModelManifest {
            name: "t".into(),
            shards: vec![crate::manifest::ShardEntry {
                file: "a.gguf".into(),
                bytes: 64,
                sha256: "0".repeat(64),
                replicas: vec![shard_dir],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(matches!(
            ReplicaRegistry::open(&manifest),
            Err(StoreError::Io { .. })
        ));
    }
}
