//! RAM expert pool (§5.4, §6.5, §6.6): residency map keyed (layer, expert)
//! — one slot per unit covering gate/up/down (D-11) — with in-flight leases
//! and claim-time atomic eviction, the mlock'd per-class slot arena, the
//! GGUF pread expert source, and the demand pager.
//!
//! Concurrency discipline (§6.6, T6): every concurrent type comes from
//! [`primitives`] (std/loom cfg-swap) — no direct `std::sync` in pool code.
//! The residency shard lock is leaf-most: policy state reaches `claim` only
//! through lock-free [`PolicySnapshot`] reads made before the lock is taken.

/// mlock'd slot arena — the only `unsafe` in the pool module.
#[allow(unsafe_code)]
pub mod arena;
pub mod primitives;

mod pager;
pub mod prefetch;
mod residency;
mod source;

pub use arena::{Arena, SlotClass};
pub use pager::{ExpertPool, Lease, PoolError, SnapshotHandle};
pub use prefetch::{AccessFeed, GATE_CAPACITY, PREFETCH_K, PrefetchHandle, spawn_prefetch};
pub use residency::{
    Claim, ClaimError, Evicted, ExpertKind, P3Stats, PolicySnapshot, ResidencyMap, Victim,
};
pub use source::{ExpertSource, GgufPreadSource, PoolGeometry, SourceError, UnitLayout};
