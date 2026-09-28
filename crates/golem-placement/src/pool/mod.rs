//! RAM expert pool (§5.4, §6.5, §6.6): residency map keyed (layer, kind,
//! expert) with in-flight leases and claim-time atomic eviction, the mlock'd
//! slot arena, the GGUF pread expert source, and the demand pager.
//!
//! Concurrency discipline (§6.6, T6): every concurrent type comes from
//! [`primitives`] (std/loom cfg-swap) — no direct `std::sync` in pool code.
//! The residency shard lock is leaf-most: policy state reaches `claim` only
//! through lock-free [`PolicySnapshot`] reads made before the lock is taken.

pub mod primitives;

mod residency;

pub use residency::{
    Claim, Evicted, ExpertKind, P3Stats, PolicySnapshot, ResidencyMap, Victim, key_index,
};
