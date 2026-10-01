//! Demand pager (§3.4 P3, §6.6): `ensure_range` pages missing experts in
//! (I/O outside all locks) and runs the caller's copy exactly once per key
//! while its lease is held — the P3 callback body. Eviction happens inside
//! [`ResidencyMap::claim`] (victim pick + bit clear + slot reassignment, one
//! shard-lock critical section); a claim grant on ANY path — `Filled`,
//! `Resident`, or `Err(ClaimError::Evicted)` — carries one lease, so a key being filled or
//! copied can never be picked as a victim and its slot can never be handed to
//! another claimer.

// cfg(loom) is set via RUSTFLAGS by the T6 leg; see primitives.rs.
#![allow(unexpected_cfgs)]

use std::fmt;

use super::arena::Arena;
use super::primitives::{Arc, AtomicUsize, Ordering};
use super::residency::{
    Claim, ClaimError, ExpertKind, FillClaim, P3Stats, PolicySnapshot, ResidencyMap,
};
use super::source::ExpertSource;

/// Lock-free-ish read handle over the current [`PolicySnapshot`] (rebuilt by
/// the placement thread each pass). Guarded by a primitives mutex — always
/// taken OUTSIDE the residency shard lock, preserving the shard lock's
/// leaf-most position (§6.6). (arc-swap-style interface; modeled under loom,
/// so T6 explores snapshot reads exactly.)
#[derive(Debug, Default)]
pub struct SnapshotHandle {
    current: super::primitives::Mutex<Arc<PolicySnapshot>>,
}

impl SnapshotHandle {
    /// Wraps an initial snapshot.
    pub fn new(snapshot: Arc<PolicySnapshot>) -> Self {
        Self {
            current: super::primitives::Mutex::new(snapshot),
        }
    }

    /// The current snapshot (cheap `Arc` clone; never blocks the pager's I/O).
    pub fn current(&self) -> Arc<PolicySnapshot> {
        self.current.lock().unwrap().clone()
    }

    /// Publishes a rebuilt snapshot (placement-thread pass).
    pub fn update(&self, snapshot: Arc<PolicySnapshot>) {
        *self.current.lock().unwrap() = snapshot;
    }
}

/// Demand-path failure: source read failed after bounded retries. The key's
/// claim is cancelled (no ghost residency, slot recycled); the FFI layer
/// aborts loudly on this.
#[derive(Debug)]
pub enum PoolError {
    /// Source read failed after bounded retries.
    Source {
        /// The failing source error.
        source: std::io::Error,
    },
    /// Every slot is leased or snapshot-protected after bounded CLOCK walks
    /// — the pool is over-subscribed; leases are transient, so this is a
    /// loud contract violation, not a wait.
    Busy,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolError::Source { source } => write!(f, "expert page-in failed: {source}"),
            PoolError::Busy => write!(
                f,
                "expert pool busy: no evictable slot — pool over-subscribed"
            ),
        }
    }
}

impl std::error::Error for PoolError {}

/// In-flight lease for one (layer, kind, expert); `Drop` releases it. Held
/// across the copy-under-lease; the slot cannot be refilled or evicted while
/// it exists.
#[derive(Debug)]
pub struct Lease {
    map: Arc<ResidencyMap>,
    layer: u32,
    kind: ExpertKind,
    expert: u32,
    slot: u32,
}

impl Lease {
    /// The slot this lease covers (capability check for [`Arena::slot`]).
    pub fn slot(&self) -> u32 {
        self.slot
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.map.release_lease(self.layer, self.kind, self.expert);
    }
}

/// The RAM expert pool: arena + residency map + source + snapshots + stats,
/// `Arc`-shared with the FFI glue and the prefetcher (Tasks 5–6).
pub struct ExpertPool {
    /// Locked slot arena.
    pub arena: Arc<Arena>,
    /// Residency state (§6.6).
    pub residency: Arc<ResidencyMap>,
    /// P3 counters.
    pub stats: Arc<P3Stats>,
    /// In-flight demand page-ins (§6.5 demand-priority signal): the
    /// prefetcher skips rounds while this is nonzero.
    demand_inflight: AtomicUsize,
    source: Arc<dyn ExpertSource>,
    snapshots: Arc<SnapshotHandle>,
}

impl fmt::Debug for ExpertPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertPool")
            .field("arena", &self.arena)
            .field("residency", &self.residency)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl ExpertPool {
    /// Assembles a pool over `n_layers` × `n_experts` experts and the arena's
    /// slot count.
    pub fn new(
        arena: Arena,
        source: Arc<dyn ExpertSource>,
        snapshots: Arc<SnapshotHandle>,
        stats: Arc<P3Stats>,
        n_layers: u32,
        n_experts: u32,
    ) -> Arc<Self> {
        let slots = arena.slot_count();
        Arc::new(Self {
            arena: Arc::new(arena),
            residency: Arc::new(ResidencyMap::new(n_layers, n_experts, slots)),
            stats,
            demand_inflight: super::primitives::AtomicUsize::new(0),
            source,
            snapshots,
        })
    }

    /// Number of demand page-ins currently in flight (§6.5 gate input).
    pub fn demand_inflight(&self) -> usize {
        self.demand_inflight.load(Ordering::Relaxed)
    }

    /// Page in every missing key in `[first, last]` for this (layer, kind),
    /// then run `copy` ONCE per key while its lease is held; `copy` receives
    /// the slot bytes. This is the P3 callback body.
    ///
    /// Per key: `claim` → on `Err(ClaimError::Evicted)` the returned slot is filled (I/O
    /// outside all locks) → copy under the claim's lease → lease dropped.
    /// Stats are per key (`demand`/`!demand` counted separately; Global
    /// Constraints), never per range.
    ///
    /// # Errors
    /// Source failure after bounded retries; the claim is cancelled first (no
    /// ghost residency), and the FFI layer aborts on this error.
    ///
    /// # Panics
    /// Panics (debug) on an inverted `first > last` — a contract violation the
    /// glue must reject, never silently rewrite.
    pub fn ensure_range(
        &self,
        demand: bool,
        layer: u32,
        kind: ExpertKind,
        first: u32,
        last: u32,
        copy: &mut dyn FnMut(&[u8]),
    ) -> Result<(), PoolError> {
        // The whole demand path — claims, I/O, copies — counts as in-flight
        // demand so the prefetcher's per-round check observes it (§6.5).
        if demand {
            self.demand_inflight.fetch_add(1, Ordering::Relaxed);
        }
        let result = self.ensure_range_inner(demand, layer, kind, first, last, copy);
        if demand {
            self.demand_inflight.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    fn ensure_range_inner(
        &self,
        demand: bool,
        layer: u32,
        kind: ExpertKind,
        first: u32,
        last: u32,
        copy: &mut dyn FnMut(&[u8]),
    ) -> Result<(), PoolError> {
        debug_assert!(first <= last, "inverted range ({first}, {last})");
        let snapshot = self.snapshots.current();
        for expert in first..=last {
            let (slot, lease) = self.claim_leased(demand, layer, kind, expert, &snapshot)?;
            copy(self.arena.slot(slot, &lease));
            drop(lease);
        }
        Ok(())
    }

    /// Claims one expert's slot, paging it in if needed, and returns the slot
    /// index plus the UNDROPPED lease (the caller keeps the lease
    /// until the host→device DMA is known complete, then drops it).
    fn claim_leased(
        &self,
        demand: bool,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
        snapshot: &PolicySnapshot,
    ) -> Result<(u32, Lease), PoolError> {
        let (slot, _claim) = match self.residency.claim(layer, kind, expert, snapshot) {
            Ok(Claim::Resident { slot }) => {
                self.stats.fast_hits.fetch_add(1, Ordering::Relaxed);
                (slot, Option::<FillClaim>::None)
            }
            Ok(Claim::Filled { slot, claim }) => {
                self.page_in(demand, layer, kind, expert, slot, Some(claim))?;
                (slot, None)
            }
            Err(ClaimError::Evicted(evicted)) => {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.page_in(
                    demand,
                    layer,
                    kind,
                    expert,
                    evicted.slot,
                    Some(evicted.claim),
                )?;
                (evicted.slot, None)
            }
            Err(ClaimError::Busy) => return Err(PoolError::Busy),
        };
        Ok((
            slot,
            Lease {
                map: self.residency.clone(),
                layer,
                kind,
                expert,
                slot,
            },
        ))
    }

    /// Claims `[first, last]`, pages each expert in, and returns
    /// the pool-side host pointer per expert together with the held leases.
    /// The caller MUST drop the leases only after the device has consumed the
    /// bytes (the glue releases them from the `copy_done` hook).
    pub fn ensure_range_leased(
        &self,
        demand: bool,
        layer: u32,
        kind: ExpertKind,
        first: u32,
        last: u32,
    ) -> Result<(Vec<*const u8>, Vec<Lease>), PoolError> {
        debug_assert!(first <= last, "inverted range ({first}, {last})");
        if demand {
            self.demand_inflight.fetch_add(1, Ordering::Relaxed);
        }
        let result = (|| {
            let snapshot = self.snapshots.current();
            let mut ptrs = Vec::with_capacity((last - first + 1) as usize);
            let mut leases = Vec::with_capacity((last - first + 1) as usize);
            for expert in first..=last {
                let (slot, lease) = self.claim_leased(demand, layer, kind, expert, &snapshot)?;
                ptrs.push(self.arena.slot(slot, &lease).as_ptr());
                leases.push(lease);
            }
            Ok((ptrs, leases))
        })();
        if demand {
            self.demand_inflight.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    /// Fills one slot from the source. The `Filled`/evict claim left this key
    /// mid-fill in the residency shard: the fill writes through the exclusive
    /// gate, and only `finish_fill` (below, before the lease-held copy) lets
    /// parked `Resident` claimers of this key read the slot. On failure:
    /// cancel the claim's lease (residency bit cleared, slot recycled, gate
    /// opened) and surface the error.
    fn page_in(
        &self,
        demand: bool,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
        slot: u32,
        claim: Option<FillClaim>,
    ) -> Result<(), PoolError> {
        // The claim comes from claim() (Filled/Evicted both grant one) — the
        // residency map is the only constructor (review H-1).
        let fill_claim =
            claim.expect("page_in without a FillClaim (review H-1 capability invariant)");
        match self
            .source
            .read_expert(layer, kind, expert, self.arena.slot_mut(slot, &fill_claim))
        {
            Ok(()) => {
                self.residency.finish_fill(layer, kind, expert);
                let counter = if demand {
                    &self.stats.demand_slow_pageins
                } else {
                    &self.stats.prefetch_pageins
                };
                counter.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_paged
                    .fetch_add(self.arena.slot_size() as u64, Ordering::Relaxed);
                Ok(())
            }
            Err(source) => {
                self.residency.cancel_claim(layer, kind, expert);
                self.stats.source_errors.fetch_add(1, Ordering::Relaxed);
                Err(PoolError::Source { source })
            }
        }
    }

    /// Test-visible lease surface: claims + leases one key so its slot bytes
    /// can be read via [`Self::slot_bytes_under`] outside `ensure_range`.
    /// Production reads happen only inside the copy closure.
    ///
    /// # Panics
    /// Panics when the key is not resident (the caller claimed it away).
    pub fn lease_for_test(&self, layer: u32, kind: ExpertKind, expert: u32) -> Lease {
        let snapshot = PolicySnapshot::test_empty();
        let slot = match self.residency.claim(layer, kind, expert, &snapshot) {
            Ok(Claim::Resident { slot }) | Ok(Claim::Filled { slot, .. }) => slot,
            Err(ClaimError::Evicted(evicted)) => evicted.slot,
            Err(ClaimError::Busy) => panic!("test lease on an over-subscribed pool"),
        };
        Lease {
            map: self.residency.clone(),
            layer,
            kind,
            expert,
            slot,
        }
    }

    /// Slot bytes covered by `lease` (lease guarantees no refill).
    pub fn slot_bytes_under(&self, lease: &Lease) -> &[u8] {
        self.arena.slot(lease.slot(), lease)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io;

    use super::super::primitives::{Mutex, Ordering, arc_dyn};
    use super::*;
    use crate::pool::residency::key_index;

    /// Vec-backed source writing expert-identifying bytes, with injectable
    /// (counted) failures per key.
    struct FakeSource {
        stride: usize,
        errors: Mutex<HashMap<usize, usize>>,
    }

    impl FakeSource {
        fn new(stride: usize) -> Self {
            Self {
                stride,
                errors: Mutex::new(HashMap::new()),
            }
        }

        fn fail_on(&self, layer: u32, kind: ExpertKind, expert: u32, times: usize) -> &Self {
            self.errors
                .lock()
                .unwrap()
                .insert(key_index(layer, kind, expert, 8), times);
            self
        }
    }

    impl ExpertSource for FakeSource {
        fn read_expert(
            &self,
            layer: u32,
            kind: ExpertKind,
            expert: u32,
            into: &mut [u8],
        ) -> io::Result<()> {
            let k = key_index(layer, kind, expert, 8);
            let mut errors = self.errors.lock().unwrap();
            if let Some(n) = errors.get_mut(&k) {
                if *n > 0 {
                    *n -= 1;
                    return Err(io::Error::other("injected source failure"));
                }
            }
            drop(errors);
            assert_eq!(into.len(), self.stride);
            into.fill(expert as u8);
            Ok(())
        }

        fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
            None
        }
    }

    fn test_pool(slots: u32) -> Arc<ExpertPool> {
        let slot_size = 64;
        let arena = Arena::new(slot_size * slots as usize, slot_size).unwrap();
        let source: Arc<dyn ExpertSource> =
            arc_dyn(std::sync::Arc::new(FakeSource::new(slot_size)));
        ExpertPool::new(
            arena,
            source,
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
            Arc::new(P3Stats::new()),
            1,
            8,
        )
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn miss_pages_in_then_hit_is_fast() {
        let pool = test_pool(2);
        pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |_| {})
            .unwrap();
        assert_eq!(pool.stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
        assert_eq!(pool.stats.fast_hits.load(Ordering::Relaxed), 0);
        pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |_| {})
            .unwrap();
        assert_eq!(pool.stats.fast_hits.load(Ordering::Relaxed), 1);
        assert_eq!(pool.stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
        assert_eq!(
            pool.stats.bytes_paged.load(Ordering::Relaxed),
            64,
            "one slot's worth of bytes moved"
        );
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn full_pool_evicts_atomically_and_refills_same_slot() {
        let pool = test_pool(1);
        pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut |_| {})
            .unwrap();
        pool.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut |_| {})
            .unwrap(); // must evict
        assert_eq!(pool.stats.evictions.load(Ordering::Relaxed), 1);
        assert!(!pool.residency.is_resident(0, ExpertKind::Gate, 0));
        assert!(pool.residency.is_resident(0, ExpertKind::Gate, 1));
        // The survivor's slot carries ITS bytes (provenance under lease):
        let lease = pool.lease_for_test(0, ExpertKind::Gate, 1);
        assert_eq!(pool.slot_bytes_under(&lease)[0], 1);
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn source_error_propagates_and_pool_stays_usable() {
        let slot_size = 64;
        let arena = Arena::new(slot_size * 2, slot_size).unwrap();
        let fake = FakeSource::new(slot_size);
        fake.fail_on(0, ExpertKind::Gate, 2, 1);
        let source: Arc<dyn ExpertSource> = arc_dyn(std::sync::Arc::new(fake));
        let pool = ExpertPool::new(
            arena,
            source,
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
            Arc::new(P3Stats::new()),
            1,
            8,
        );
        assert!(
            pool.ensure_range(true, 0, ExpertKind::Gate, 2, 2, &mut |_| {})
                .is_err()
        );
        assert!(
            !pool.residency.is_resident(0, ExpertKind::Gate, 2),
            "no ghost residency"
        );
        assert_eq!(pool.stats.source_errors.load(Ordering::Relaxed), 1);
        pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |_| {})
            .unwrap();
        assert!(pool.residency.is_resident(0, ExpertKind::Gate, 3));
        // The errored key's slot was recycled, not leaked:
        assert_eq!(pool.arena.slot_count(), 2);
        let lease = pool.lease_for_test(0, ExpertKind::Gate, 3);
        assert_eq!(pool.slot_bytes_under(&lease)[0], 3);
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn range_with_mixed_states_counts_per_key() {
        let pool = test_pool(4);
        pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut |_| {})
            .unwrap();
        let slow_before = pool.stats.demand_slow_pageins.load(Ordering::Relaxed);
        let hits_before = pool.stats.fast_hits.load(Ordering::Relaxed);
        pool.ensure_range(true, 0, ExpertKind::Gate, 0, 1, &mut |_| {})
            .unwrap();
        // e0 was a hit, e1 a page-in — counters per key, not per range:
        assert_eq!(
            pool.stats.demand_slow_pageins.load(Ordering::Relaxed),
            slow_before + 1
        );
        assert_eq!(
            pool.stats.fast_hits.load(Ordering::Relaxed),
            hits_before + 1
        );
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn snapshot_updates_are_visible_to_the_next_range() {
        let pool = test_pool(2);
        pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut |_| {})
            .unwrap();
        let mut snap = PolicySnapshot::test_empty();
        snap.lease(0, ExpertKind::Gate, 0);
        pool.snapshots.update(Arc::new(snap));
        pool.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut |_| {})
            .unwrap();
        // Pressure: e2 must evict — but the pinned e0 is untouchable:
        pool.ensure_range(true, 0, ExpertKind::Gate, 2, 2, &mut |_| {})
            .unwrap();
        assert!(
            pool.residency.is_resident(0, ExpertKind::Gate, 0),
            "snapshot-lease held"
        );
        assert!(!pool.residency.is_resident(0, ExpertKind::Gate, 1));
        assert!(pool.residency.is_resident(0, ExpertKind::Gate, 2));
        assert_eq!(pool.stats.evictions.load(Ordering::Relaxed), 1);
    }
}
