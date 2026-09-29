//! Golem store (tech design §5): multi-disk mirrored storage, striped reads,
//! checksummed shards, manifest-driven model layout.
//!
//! Scope: [`StoreError`], manifest types + validating loader (`manifest`).
//! Replica I/O, striping, and integrity verification land in later tasks.

mod manifest;

pub use manifest::{ExpertIndexEntry, ModelManifest, ShardEntry, load_manifest};

use std::path::PathBuf;

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
}
