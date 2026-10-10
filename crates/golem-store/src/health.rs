//! Instance-level health state machine (§5.2, §11): aggregates per-replica
//! health and the §5.3 checksum-degradation flag into one classification.
//!
//! Precedence (§11): FatalIo > DegradedChecksum > DegradedReplica > Healthy.
//! `DegradedIo` (§11) is deliberately absent: no store path in this crate
//! produces a pool-wide I/O degradation distinct from per-replica
//! degradation; the M4 server adds the variant when its trigger exists.

use crate::replica::{ReplicaHealth, ReplicaRegistry};

/// Aggregate instance health (§5.2, §11). Explicit ordinals back the
/// lock-free last-known-state tracking in the store source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum InstanceHealth {
    /// Every replica healthy, no checksum degradation.
    Healthy = 0,
    /// ≥1 replica degraded, every shard still has ≥1 healthy replica.
    DegradedReplica = 1,
    /// A §5.3 CRC mismatch was recorded; data served is suspect until the
    /// affected replicas are re-opened.
    DegradedChecksum = 2,
    /// At least one shard has zero healthy replicas: its experts are
    /// unreadable (§5.2: in-flight requests complete with error, new ones
    /// get unavailable; the process stays alive).
    FatalIo = 3,
}

impl InstanceHealth {
    /// Total inverse of the explicit ordinals. `None` only for values that
    /// no `self as u8` produces; lock-free trackers round-trip through this
    /// so the ordinal mapping exists in exactly one place.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Healthy),
            1 => Some(Self::DegradedReplica),
            2 => Some(Self::DegradedChecksum),
            3 => Some(Self::FatalIo),
            _ => None,
        }
    }
}

/// Classifies the instance from the registry's per-replica health and the
/// checksum-degradation flag (§5.2, §11). Pure: reads atomic health words,
/// performs no I/O, never panics.
///
/// Degradation is one-way (`ReplicaSlot::mark_degraded`), so transitions
/// only move toward worse states; recovery requires re-opening the store.
pub fn instance_health(registry: &ReplicaRegistry, checksum_degraded: bool) -> InstanceHealth {
    let any_shard_down = registry.shards().iter().any(|s| {
        s.replicas
            .iter()
            .all(|r| r.health() != ReplicaHealth::Healthy)
    });
    if any_shard_down {
        return InstanceHealth::FatalIo;
    }
    if checksum_degraded {
        return InstanceHealth::DegradedChecksum;
    }
    let any_degraded = registry.shards().iter().any(|s| {
        s.replicas
            .iter()
            .any(|r| r.health() == ReplicaHealth::Degraded)
    });
    if any_degraded {
        return InstanceHealth::DegradedReplica;
    }
    InstanceHealth::Healthy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelManifest, ShardEntry};

    /// Opens a registry over `shard_files` (replica dirs per shard). Real
    /// files: `ReplicaRegistry::open` verifies sizes (§5.3). Tag MUST be
    /// unique per test (nextest runs parallel processes over shared /tmp).
    fn registry(tag: &str, shard_files: &[&[&str]]) -> ReplicaRegistry {
        let root = std::env::temp_dir().join(format!("golem-store-health-{tag}"));
        let shards: Vec<ShardEntry> = shard_files
            .iter()
            .enumerate()
            .map(|(s, reps)| {
                for (r, dir) in reps.iter().enumerate() {
                    let d = root.join(format!("s{s}-r{r}-{}", dir));
                    std::fs::create_dir_all(&d).unwrap();
                    std::fs::write(d.join("a.gguf"), [0u8; 64]).unwrap();
                }
                ShardEntry {
                    file: "a.gguf".into(),
                    bytes: 64,
                    sha256: "0".repeat(64),
                    replicas: reps
                        .iter()
                        .enumerate()
                        .map(|(r, dir)| root.join(format!("s{s}-r{r}-{dir}")))
                        .collect(),
                    ..Default::default()
                }
            })
            .collect();
        ReplicaRegistry::open(&ModelManifest {
            name: tag.into(),
            shards,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn one_shard_fully_down_is_fatal_io() {
        let reg = registry("fatal", &[&["a", "b"], &["a", "b"]]);
        reg.shards()[1].replicas[0].mark_degraded();
        reg.shards()[1].replicas[1].mark_degraded();
        assert_eq!(instance_health(&reg, false), InstanceHealth::FatalIo);
    }

    #[test]
    fn degraded_replica_with_survivors_in_every_shard_is_degraded_replica() {
        let reg = registry("degraded", &[&["a", "b"], &["a", "b"]]);
        reg.shards()[0].replicas[0].mark_degraded();
        assert_eq!(
            instance_health(&reg, false),
            InstanceHealth::DegradedReplica
        );
    }

    #[test]
    fn checksum_mismatch_is_degraded_checksum() {
        // Precedence (§11): checksum beats replica degradation; a fatal
        // shard still wins over checksum.
        let reg = registry("crc", &[&["a", "b"]]);
        reg.shards()[0].replicas[0].mark_degraded();
        assert_eq!(
            instance_health(&reg, true),
            InstanceHealth::DegradedChecksum
        );
    }

    #[test]
    fn all_healthy_is_healthy() {
        let reg = registry("healthy", &[&["a", "b"]]);
        assert_eq!(instance_health(&reg, false), InstanceHealth::Healthy);
    }

    #[test]
    fn from_u8_round_trips_every_ordinal() {
        for v in 0..=3u8 {
            assert_eq!(InstanceHealth::from_u8(v).map(|h| h as u8), Some(v));
        }
        assert_eq!(InstanceHealth::from_u8(4), None);
    }
}
