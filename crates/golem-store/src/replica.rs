//! Replica registry: open handles per (shard, replica) pair, per-replica
//! health, and best-effort same-drive detection (§5.1).

use std::fs::File;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::StoreError;
use crate::manifest::ModelManifest;

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

/// One opened replica file plus its health state.
pub struct ReplicaSlot {
    /// Full path: replica dir joined with the shard file name.
    pub path: PathBuf,
    /// Open handle for positioned reads.
    pub file: std::fs::File,
    /// 0 = Healthy, 1 = Degraded ([`ReplicaHealth`] as u8).
    pub health: AtomicU8,
    /// Count of one-way degradations since open; feeds diagnostics and
    /// burn-down decisions upstream (health itself is `health()`, never this
    /// counter).
    errors: AtomicU64,
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

    /// Read exactly `buf.len()` bytes at `offset`; retries short reads, errors
    /// on premature EOF.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        let mut filled = 0;
        while filled < buf.len() {
            let n = match self
                .file
                .read_at(&mut buf[filled..], offset + filled as u64)
            {
                // `File::read_at` does not retry EINTR (unlike `read`):
                // §5.2 degrades on device errors, not signals — a one-way
                // health flip must not be triggerable by a spurious
                // interrupt.
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short read at offset {offset}: {} of {} bytes",
                            filled,
                            buf.len()
                        ),
                    ));
                }
                Ok(n) => n,
            };
            filled += n;
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
                replicas.push(Arc::new(ReplicaSlot {
                    path,
                    file,
                    health: AtomicU8::new(ReplicaHealth::Healthy as u8),
                    errors: AtomicU64::new(0),
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
    if let Some((head, tail)) = partition.rsplit_once('p') {
        if !head.is_empty()
            && !tail.is_empty()
            && tail.bytes().all(|b| b.is_ascii_digit())
            && head.bytes().any(|b| b.is_ascii_digit())
        {
            return Some(head.to_owned());
        }
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
            }],
            ..Default::default()
        };
        assert!(matches!(
            ReplicaRegistry::open(&manifest),
            Err(StoreError::Io { .. })
        ));
    }
}
