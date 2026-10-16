//! RAM expert pool (§5.4, §6.5, §6.6): residency map keyed (layer, expert)
//! — one slot per unit covering gate/up/down (§3.4) — with in-flight leases
//! and claim-time atomic eviction, the mlock'd per-class slot arena, the
//! GGUF pread expert source, and the demand pager.
//!
//! Concurrency discipline (§6.6, T6): every concurrent type comes from
//! [`primitives`] (std/loom cfg-swap) — no direct `std::sync` in pool code.
//! The residency shard lock is leaf-most: policy state reaches `claim` only
//! through a [`PolicySnapshot`] read (one short mutex hold per decode-path
//! compute, per prefetch/`ensure_range` call) made before any shard lock is
//! taken.

/// mlock'd slot arena — the only `unsafe` in the pool module.
#[allow(unsafe_code)]
pub mod arena;
pub mod primitives;

mod pager;
pub mod prefetch;
mod residency;
mod source;
mod store_source;

pub use arena::{Arena, SlotClass};
pub use pager::{D2hIncoming, ExpertPool, Lease, PoolError, SnapshotHandle, StepClaims};
pub use prefetch::{AccessFeed, PredictionQueue, PrefetchHandle, WarmFeed, spawn_prefetch};
pub use residency::{ExpertKind, P3Stats, PolicySnapshot, PoolConfigError};
pub use source::{
    ExpertSource, GgufPreadSource, PoolGeometry, SourceError, UnitLayout, class_slots,
};
pub use store_source::StoreExpertSource;
