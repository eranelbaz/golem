//! Golem store (tech design §5): multi-disk mirrored storage, striped reads,
//! checksummed shards, manifest-driven model layout.
//!
//! Design coverage:
//! - §5.1 replica registry: per-slot health, same-drive detection
//!   ([`replica`], [`ReplicaRegistry::shared_device_warnings`]).
//! - §5.2 shared-queue striping ([`queue`], [`stripe`]): one queue of chunk
//!   reads drained demand-first by per-drive scoped threads, with exclude-mask
//!   requeue onto the drain-local work list.
//! - §5.3 CRC32C validation against the caller-supplied per-plane digest
//!   ([`PageRead::expected_crc`]); mismatch degrades the serving replicas and
//!   surfaces [`StoreError::Crc`] (never silent corruption).
//! - §5.4 per-chunk failover to the next healthy replica with one-way
//!   [`ReplicaHealth::Degraded`] marking; all replicas down →
//!   [`StoreError::Unavailable`] (never silent corruption).
//!
//! Scope: [`StoreError`], manifest types + validating loader ([`manifest`]),
//! replica I/O, striping, and the pread engine ([`engine`]).

mod engine;
/// `golem import` (§5.5): local source discovery + expert-plane geometry.
pub mod import;
mod manifest;
/// §5.2 shared chunk queue: demand-first drain scheduling with per-drive
/// scoped reader threads.
mod queue;
mod replica;
mod stripe;

/// Raw syscall wrappers — the only `unsafe` in this crate.
#[allow(unsafe_code)]
pub mod sys;

pub use engine::{ChunkSource, DEFAULT_IO_THREADS, IoEngine, PageRead, ShardSources};
pub use manifest::{ExpertIndexEntry, ModelManifest, ShardEntry, load_manifest};
pub use queue::{ChunkQueue, ReadHandle, ReadJob};
pub use replica::{
    ReplicaHealth, ReplicaRegistry, ReplicaSlot, ShardReplicas, parent_device_name,
    physical_device_name,
};
pub use stripe::{CHUNK, plan_aligned};

use std::path::PathBuf;

/// The one reader of `GOLEM_DATA_DIR` (§4.2): root for manifests, models,
/// and placement journals. `None` when unset or non-UTF-8.
#[must_use]
pub fn data_dir() -> Option<PathBuf> {
    std::env::var_os("GOLEM_DATA_DIR").map(PathBuf::from)
}

/// Errors surfaced by the store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("manifest {path}: {msg}")]
    /// Invalid or unreadable manifest.
    Manifest {
        /// Manifest file path.
        path: PathBuf,
        /// What was wrong.
        msg: String,
    },
    #[error("io error on replica {replica} ({shard}): {source}")]
    /// I/O failure against one replica.
    Io {
        /// Shard file name.
        shard: String,
        /// Replica root the failure occurred on.
        replica: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    #[error("data corruption in replica {replica} ({shard}) at offset {offset}: {msg}")]
    /// Checksum or structure mismatch in replica data.
    Rot {
        /// Shard file name.
        shard: String,
        /// Replica root the corruption was found in.
        replica: PathBuf,
        /// Byte offset of the corruption.
        offset: u64,
        /// Details of the mismatch.
        msg: String,
    },
    #[error("all replicas of shard {shard} unreadable")]
    /// Every replica of a shard failed.
    Unavailable {
        /// Shard file name.
        shard: String,
    },
    #[error("crc mismatch in shard {shard}")]
    /// §5.3: replica data failed its checksum; the serving replicas were
    /// marked degraded.
    Crc {
        /// Shard file name.
        shard: String,
    },
}
