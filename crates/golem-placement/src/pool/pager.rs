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
use super::source::{ExpertSource, PoolGeometry};

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

/// In-flight lease for one (layer, expert) unit; `Drop` releases it. Held
/// across the copy-under-lease; the slot cannot be refilled or evicted while
/// it exists. One lease pins all three planes of the unit.
#[derive(Debug)]
pub struct Lease {
    map: Arc<ResidencyMap>,
    /// Keeps the mapping alive while raw slot pointers handed out with this
    /// lease (`ensure_range_leased`) may still be read by an in-flight H2D.
    _arena: Arc<Arena>,
    layer: u32,
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
        self.map.release_lease(self.layer, self.expert);
    }
}

/// The RAM expert pool: arena + residency map + source + snapshots + stats,
/// `Arc`-shared with the FFI glue and the prefetcher.
pub struct ExpertPool {
    /// Locked slot arena.
    pub arena: Arc<Arena>,
    /// Per-layer gate/up/down unit layout (§6.1): plane offsets/strides
    /// inside a slot.
    pub geometry: PoolGeometry,
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
    /// Assembles a pool over `geometry.layers.len()` × `geometry.n_experts`
    /// expert units; each layer claims slots only from its arena class
    /// (§5 pool amendment).
    pub fn new(
        arena: Arena,
        source: Arc<dyn ExpertSource>,
        snapshots: Arc<SnapshotHandle>,
        stats: Arc<P3Stats>,
        geometry: PoolGeometry,
        class_of_layer: Vec<u8>,
    ) -> Arc<Self> {
        let n_experts = geometry.n_experts;
        let n_classes = class_of_layer
            .iter()
            .copied()
            .max()
            .map_or(0, |c| usize::from(c) + 1);
        let class_slots: Vec<std::ops::Range<u32>> =
            (0..n_classes).map(|c| arena.class_range(c)).collect();
        Arc::new(Self {
            arena: Arc::new(arena),
            geometry,
            residency: Arc::new(ResidencyMap::new(class_of_layer, n_experts, class_slots)),
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
    /// Stats are per key (`demand`/`!demand` counted separately, §6.6),
    /// never per range.
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
            let (slot, lease) = self.claim_leased(demand, layer, expert, &snapshot)?;
            copy(self.plane_bytes(layer, slot, &lease, kind));
            drop(lease);
        }
        Ok(())
    }

    /// One unit's slot, claimed + paged in (all three planes) under its
    /// lease, released before return — the prefetch entry (§6.5).
    ///
    /// # Errors
    /// Source failure after bounded retries (claim cancelled first) or an
    /// over-subscribed pool.
    pub fn ensure_unit(&self, demand: bool, layer: u32, expert: u32) -> Result<(), PoolError> {
        if demand {
            self.demand_inflight.fetch_add(1, Ordering::Relaxed);
        }
        let result = (|| {
            let snapshot = self.snapshots.current();
            let (_, lease) = self.claim_leased(demand, layer, expert, &snapshot)?;
            drop(lease); // prefetch reads no bytes; the fill itself was the work
            Ok(())
        })();
        if demand {
            self.demand_inflight.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    /// Claims one expert's slot, paging it in if needed, and returns the slot
    /// index plus the UNDROPPED lease (the caller keeps the lease
    /// until the host→device DMA is known complete, then drops it).
    fn claim_leased(
        &self,
        demand: bool,
        layer: u32,
        expert: u32,
        snapshot: &PolicySnapshot,
    ) -> Result<(u32, Lease), PoolError> {
        let (slot, _claim) = match self.residency.claim(layer, expert, snapshot) {
            Ok(Claim::Resident { slot }) => {
                self.stats.fast_hits.fetch_add(1, Ordering::Relaxed);
                (slot, Option::<FillClaim>::None)
            }
            Ok(Claim::Filled { slot, claim }) => {
                self.page_in(demand, layer, expert, slot, Some(claim))?;
                (slot, None)
            }
            Err(ClaimError::Evicted(evicted)) => {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.page_in(demand, layer, expert, evicted.slot, Some(evicted.claim))?;
                (evicted.slot, None)
            }
            Err(ClaimError::Busy) => return Err(PoolError::Busy),
        };
        Ok((
            slot,
            Lease {
                map: self.residency.clone(),
                _arena: self.arena.clone(),
                layer,
                expert,
                slot,
            },
        ))
    }

    /// The `kind` plane's sub-slice of the unit's slot (§6.1 plane layout).
    fn plane_bytes<'a>(
        &'a self,
        layer: u32,
        slot: u32,
        lease: &'a Lease,
        kind: ExpertKind,
    ) -> &'a [u8] {
        let unit = &self.geometry.layers[layer as usize];
        let (off, stride) = unit.plane(kind);
        let full = self.arena.slot(slot, lease);
        &full[off as usize..(off + stride) as usize]
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
                let (slot, lease) = self.claim_leased(demand, layer, expert, &snapshot)?;
                ptrs.push(self.plane_bytes(layer, slot, &lease, kind).as_ptr());
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
        expert: u32,
        slot: u32,
        claim: Option<FillClaim>,
    ) -> Result<(), PoolError> {
        // The claim comes from claim() (Filled/Evicted both grant one) — the
        // residency map is the only constructor (§6.6).
        let fill_claim = claim.expect("page_in without a FillClaim (capability invariant)");
        let unit = &self.geometry.layers[layer as usize];
        // One disk→pool copy per unit: three preads straight into the
        // claimed slot's plane sub-slices (split from the single slot_mut
        // borrow — no bounce buffer, §3.4 v3).
        let slot_bytes = self.arena.slot_mut(slot, &fill_claim);
        let result: std::io::Result<()> = 'fill: {
            let (gate, rest) = slot_bytes.split_at_mut(unit.offset[1] as usize);
            let (up, down) = rest.split_at_mut((unit.offset[2] - unit.offset[1]) as usize);
            let planes = [
                (ExpertKind::Gate, &mut gate[..unit.stride[0] as usize]),
                (ExpertKind::Up, &mut up[..unit.stride[1] as usize]),
                (ExpertKind::Down, &mut down[..unit.stride[2] as usize]),
            ];
            for (kind, buf) in planes {
                if let Err(e) = self.source.read_expert(layer, kind, expert, buf) {
                    break 'fill Err(e);
                }
            }
            Ok(())
        };
        match result {
            Ok(()) => {
                self.residency.finish_fill(layer, expert);
                let counter = if demand {
                    &self.stats.demand_slow_pageins
                } else {
                    &self.stats.prefetch_pageins
                };
                counter.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_paged
                    .fetch_add(unit.bytes(), Ordering::Relaxed);
                Ok(())
            }
            Err(source) => {
                // Any plane failure aborts the whole unit: no ghost
                // residency, the slot goes back to its class's free list.
                self.residency.cancel_claim(layer, expert);
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
    /// Panics when the key is not resident — a fresh claim would leave it
    /// mid-fill and park every later claimer.
    pub fn lease_for_test(&self, layer: u32, expert: u32) -> Lease {
        let snapshot = PolicySnapshot::test_empty();
        let slot = match self.residency.claim(layer, expert, &snapshot) {
            Ok(Claim::Resident { slot }) => slot,
            Ok(Claim::Filled { .. }) | Err(ClaimError::Evicted(_)) => {
                panic!("lease_for_test on a non-resident unit")
            }
            Err(ClaimError::Busy) => panic!("test lease on an over-subscribed pool"),
        };
        Lease {
            map: self.residency.clone(),
            _arena: self.arena.clone(),
            layer,
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
    use crate::pool::SlotClass;
    use std::collections::HashMap;
    use std::io;

    use super::super::primitives::{Mutex, Ordering, arc_dyn};
    use super::*;
    use crate::key;
    use crate::pool::source::UnitLayout;

    /// Vec-backed source writing plane-identifying bytes
    /// (`0x10 × (kind + 1) + expert`), with injectable (counted) failures
    /// per unit.
    struct FakeSource {
        stride: usize,
        errors: Mutex<HashMap<usize, usize>>,
    }

    const N_EXPERTS: u32 = 8;

    impl FakeSource {
        fn new(stride: usize) -> Self {
            Self {
                stride,
                errors: Mutex::new(HashMap::new()),
            }
        }

        fn fail_on(&self, layer: u32, _kind: ExpertKind, expert: u32, times: usize) -> &Self {
            self.errors
                .lock()
                .unwrap()
                .insert(key(4, N_EXPERTS, layer, expert).unwrap(), times);
            self
        }

        fn plane_byte(&self, kind: ExpertKind, expert: u32) -> u8 {
            0x10 * (kind.ordinal() as u8 + 1) + expert as u8
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
            let k = key(4, N_EXPERTS, layer, expert).unwrap();
            let mut errors = self.errors.lock().unwrap();
            if let Some(n) = errors.get_mut(&k) {
                if *n > 0 {
                    *n -= 1;
                    return Err(io::Error::other("injected source failure"));
                }
            }
            drop(errors);
            assert_eq!(into.len(), self.stride);
            into.fill(self.plane_byte(kind, expert));
            Ok(())
        }

        fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
            None
        }
    }

    /// One layer, three uniform planes of `stride` B — the unit is 3×stride.
    fn unit_geometry(stride: u64) -> PoolGeometry {
        PoolGeometry {
            layers: vec![
                UnitLayout {
                    stride: [stride; 3],
                    offset: [0, stride, 2 * stride],
                };
                1
            ],
            n_experts: N_EXPERTS,
            expert_bytes: 0,
        }
    }

    fn test_pool(slots: u32) -> Arc<ExpertPool> {
        let stride = 64u64;
        let arena = Arena::new(&[SlotClass {
            slot_size: 3 * stride as usize,
            slots,
        }])
        .unwrap();
        let source: Arc<dyn ExpertSource> =
            arc_dyn(std::sync::Arc::new(FakeSource::new(stride as usize)));
        ExpertPool::new(
            arena,
            source,
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
            Arc::new(P3Stats::new()),
            unit_geometry(stride),
            vec![0; 1],
        )
    }

    fn test_pool_failing(slots: u32, fail: (u32, ExpertKind, u32)) -> Arc<ExpertPool> {
        let stride = 64u64;
        let arena = Arena::new(&[SlotClass {
            slot_size: 3 * stride as usize,
            slots,
        }])
        .unwrap();
        let fake = FakeSource::new(stride as usize);
        fake.fail_on(fail.0, fail.1, fail.2, 1);
        let source: Arc<dyn ExpertSource> = arc_dyn(std::sync::Arc::new(fake));
        ExpertPool::new(
            arena,
            source,
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
            Arc::new(P3Stats::new()),
            unit_geometry(stride),
            vec![0; 1],
        )
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn one_page_in_fills_all_three_planes_of_the_unit() {
        let pool = test_pool(2);
        pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |p| {
            assert_eq!(p, [0x13; 64])
        })
        .unwrap();
        assert_eq!(pool.stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
        pool.ensure_range(true, 0, ExpertKind::Down, 3, 3, &mut |p| {
            assert_eq!(p, [0x33; 64])
        })
        .unwrap();
        pool.ensure_range(true, 0, ExpertKind::Up, 3, 3, &mut |p| {
            assert_eq!(p, [0x23; 64])
        })
        .unwrap();
        assert_eq!(
            pool.stats.demand_slow_pageins.load(Ordering::Relaxed),
            1,
            "up/down rode the gate fill"
        );
        assert_eq!(pool.stats.fast_hits.load(Ordering::Relaxed), 2);
        assert_eq!(pool.stats.bytes_paged.load(Ordering::Relaxed), 192);
    }

    #[test]
    #[cfg(not(loom))]
    fn eviction_drops_the_whole_unit() {
        let pool = test_pool(1);
        pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut |_| {})
            .unwrap();
        pool.ensure_range(true, 0, ExpertKind::Up, 1, 1, &mut |_| {})
            .unwrap(); // evicts unit 0
        assert!(!pool.residency.is_resident(0, 0));
        pool.ensure_range(true, 0, ExpertKind::Down, 1, 1, &mut |p| {
            assert_eq!(p, [0x31; 64])
        })
        .unwrap();
        assert_eq!(pool.stats.demand_slow_pageins.load(Ordering::Relaxed), 2);
    }

    #[test]
    #[cfg(not(loom))]
    fn plane_failure_cancels_the_unit() {
        // Source fails the Down plane of expert 2 once: no ghost unit, slot
        // recycled.
        let pool = test_pool_failing(2, (0, ExpertKind::Down, 2));
        assert!(
            pool.ensure_range(true, 0, ExpertKind::Gate, 2, 2, &mut |_| {})
                .is_err()
        );
        assert!(!pool.residency.is_resident(0, 2));
        pool.ensure_range(true, 0, ExpertKind::Gate, 2, 2, &mut |_| {})
            .unwrap();
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
            192,
            "one unit's worth of bytes moved"
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
        assert!(!pool.residency.is_resident(0, 0));
        assert!(pool.residency.is_resident(0, 1));
        // The survivor's slot carries ITS bytes (provenance under lease):
        let lease = pool.lease_for_test(0, 1);
        assert_eq!(pool.slot_bytes_under(&lease)[0], 0x11);
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn source_error_propagates_and_pool_stays_usable() {
        let pool = test_pool_failing(2, (0, ExpertKind::Gate, 2));
        assert!(
            pool.ensure_range(true, 0, ExpertKind::Gate, 2, 2, &mut |_| {})
                .is_err()
        );
        assert!(!pool.residency.is_resident(0, 2), "no ghost residency");
        assert_eq!(pool.stats.source_errors.load(Ordering::Relaxed), 1);
        pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |_| {})
            .unwrap();
        assert!(pool.residency.is_resident(0, 3));
        // The errored unit's slot was recycled, not leaked:
        assert_eq!(pool.arena.slot_count(), 2);
        let lease = pool.lease_for_test(0, 3);
        assert_eq!(pool.slot_bytes_under(&lease)[0], 0x13);
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
        // e0 was a hit, e1 a page-in — counters per unit, not per range:
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
        snap.lease(0, 0);
        pool.snapshots.update(Arc::new(snap));
        pool.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut |_| {})
            .unwrap();
        // Pressure: e2 must evict — but the pinned e0 is untouchable:
        pool.ensure_range(true, 0, ExpertKind::Gate, 2, 2, &mut |_| {})
            .unwrap();
        assert!(pool.residency.is_resident(0, 0), "snapshot-lease held");
        assert!(!pool.residency.is_resident(0, 1));
        assert!(pool.residency.is_resident(0, 2));
        assert_eq!(pool.stats.evictions.load(Ordering::Relaxed), 1);
    }
}
