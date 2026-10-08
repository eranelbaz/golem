//! Demand pager (§3.4 P3, §6.6): `ensure_range` pages missing experts in
//! (I/O outside all locks) and runs the caller's copy exactly once per key
//! while its lease is held — the P3 callback body. Eviction happens inside
//! [`ResidencyMap::claim`] (victim pick + bit clear + slot reassignment, one
//! shard-lock critical section); a claim grant on ANY path — `Filled`,
//! `Resident`, or `Err(ClaimError::Evicted)` — carries one lease, so a key being filled or
//! copied can never be picked as a victim and its slot can never be handed to
//! another claimer.

// cfg(loom) is set via RUSTFLAGS by the T6 leg; see primitives.rs.

use super::arena::Arena;
use super::primitives::{Arc, AtomicU64, AtomicUsize, Mutex, Ordering};
use super::residency::{
    Claim, ClaimError, ExpertKind, FillClaim, P3Stats, PolicySnapshot, PoolConfigError,
    ResidencyMap,
};
use super::source::{ExpertSource, PoolGeometry};

/// Read handle over the current [`PolicySnapshot`] (rebuilt by the placement
/// thread each pass). Guarded by a primitives mutex held only for an `Arc`
/// clone/swap, taken once per scheduler compute on the decode path
/// ([`StepClaims::next_step`]) and once per `ensure_range`/`ensure_unit` call,
/// and always OUTSIDE the residency shard lock, preserving the shard lock's
/// leaf-most position (§6.6). Modeled under loom, so T6 explores snapshot reads
/// exactly.
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

    /// The current snapshot (cheap `Arc` clone; never held across I/O).
    pub fn current(&self) -> Arc<PolicySnapshot> {
        self.current.lock().unwrap().clone()
    }

    /// Publishes a rebuilt snapshot (placement-thread pass).
    pub fn update(&self, snapshot: Arc<PolicySnapshot>) {
        *self.current.lock().unwrap() = snapshot;
    }
}

/// Demand-path failure: a source read that failed after bounded retries
/// (the key's claim is cancelled — no ghost residency, slot recycled; the
/// FFI layer aborts loudly on this), an over-subscribed pool, or a caller
/// contract violation.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// Source read failed after bounded retries.
    #[error("expert page-in failed: {source}")]
    Source {
        /// The failing source error.
        source: std::io::Error,
    },
    /// Every slot is leased or snapshot-protected after bounded CLOCK walks
    /// — the pool is over-subscribed; leases are transient, so this is a
    /// loud contract violation, not a wait.
    #[error("expert pool busy: no evictable slot — pool over-subscribed")]
    Busy,
    /// Caller passed a length-mismatched `ptrs` or an out-of-range
    /// `(layer, expert)` range — a programming error, reported as a value
    /// so no `pub fn` panics (library rule: errors are `Result`).
    #[error("pool contract violated: {reason}")]
    Contract {
        /// What the caller violated.
        reason: String,
    },
}

/// In-flight lease for one (layer, expert) unit; `Drop` releases it. Held
/// across the copy-under-lease; the slot cannot be refilled or evicted while
/// it exists. One lease pins all three planes of the unit.
#[derive(Debug)]
pub struct Lease {
    map: Arc<ResidencyMap>,
    /// Keeps the mapping alive while raw slot pointers handed out with this
    /// lease ([`StepClaims`]) may still be read by an in-flight H2D.
    _arena: Arc<Arena>,
    layer: u32,
    expert: u32,
    slot: u32,
}

impl Lease {
    /// The slot this lease covers.
    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// Whether this lease pins a slot of `arena`.
    pub(crate) fn is_for(&self, arena: &Arena) -> bool {
        std::ptr::eq(&*self._arena, arena)
    }

    /// The slot's bytes (all three planes). The view borrows the lease, so it
    /// cannot outlive it:
    ///
    /// ```compile_fail,E0505
    /// # use golem_placement::pool::*;
    /// # use golem_placement::pool::primitives::arc_dyn;
    /// # struct Src;
    /// # impl ExpertSource for Src {
    /// #     fn read_expert(&self, _: u32, _: ExpertKind, _: u32, into: &mut [u8], _: usize)
    /// #         -> std::io::Result<()> { into.fill(1); Ok(()) }
    /// #     fn plane(&self, _: u32, _: ExpertKind, _: u32) -> Option<(u64, u64)> { None }
    /// # }
    /// # let arena = Arena::new(&[SlotClass { slot_size: 24, slots: 1 }]).unwrap();
    /// # let geometry = PoolGeometry {
    /// #     layers: vec![UnitLayout::new([8; 3], [None; 3]).unwrap()],
    /// #     n_experts: 1,
    /// #     expert_bytes: 0,
    /// # };
    /// # let pool = ExpertPool::try_new(
    /// #     arena,
    /// #     arc_dyn(std::sync::Arc::new(Src)),
    /// #     std::sync::Arc::new(SnapshotHandle::new(std::sync::Arc::new(PolicySnapshot::empty()))),
    /// #     std::sync::Arc::new(P3Stats::new()),
    /// #     geometry,
    /// #     vec![0],
    /// # ).unwrap();
    /// let mut step = StepClaims::new(pool);
    /// step.ensure_range_leased(0, ExpertKind::Gate, 0, 0, &mut [std::ptr::null()])
    ///     .unwrap();
    /// let bytes = step.leases()[0].bytes();
    /// drop(step);
    /// assert_eq!(bytes[0], 1);
    /// ```
    pub fn bytes(&self) -> &[u8] {
        self._arena.slot(self)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.map.release_lease(self.layer, self.expert);
    }
}

/// `lease_of` entry of a unit not claimed this step.
const UNCLAIMED: u32 = u32::MAX;

/// The decode thread's claim table for one scheduler compute (§6.6, §3.4
/// P3): the first plane of a `(layer, expert)` unit claims and leases it
/// (one shard lock); the unit's other planes reuse that lease. The policy
/// snapshot is read once per step, so a call takes no snapshot mutex. Leases
/// are held until [`Self::next_step`], which the caller runs only once the
/// device has consumed the step's bytes (the glue's `copy_done`).
pub struct StepClaims {
    pool: Arc<ExpertPool>,
    snapshot: Arc<PolicySnapshot>,
    /// `layer × n_experts + expert` → index into `leases`, or `UNCLAIMED`.
    lease_of: Vec<u32>,
    /// This step's leases; also the touched-list `next_step` resets.
    leases: Vec<Lease>,
}

impl StepClaims {
    /// An empty table over every unit of `pool` (allocated once).
    pub fn new(pool: Arc<ExpertPool>) -> Self {
        let units = pool.geometry.layers.len() * pool.geometry.n_experts as usize;
        Self {
            snapshot: pool.snapshot(),
            lease_of: vec![UNCLAIMED; units],
            leases: Vec::new(),
            pool,
        }
    }

    /// The leases held this step, in claim order.
    pub fn leases(&self) -> &[Lease] {
        &self.leases
    }

    /// Ends the step: releases every lease, resets the entries it touched
    /// and refreshes the snapshot. The caller guarantees the device has
    /// consumed every pointer handed out this step.
    pub fn next_step(&mut self) {
        let n = self.pool.geometry.n_experts as usize;
        for lease in self.leases.drain(..) {
            self.lease_of[lease.layer as usize * n + lease.expert as usize] = UNCLAIMED;
        }
        self.snapshot = self.pool.snapshot();
    }

    /// Writes the `kind` plane's pool pointer for each expert of
    /// `[first, last]` into `ptrs` (`last - first + 1` entries), claiming
    /// and paging in each unit not yet leased this step. On error the units
    /// already claimed stay leased until `next_step`; the failed one is not
    /// recorded.
    ///
    /// # Errors
    /// Source failure after bounded retries (claim cancelled first), an
    /// over-subscribed pool, or an inverted range, length-mismatched `ptrs`,
    /// or out-of-range unit (reported as [`PoolError::Contract`], not a
    /// panic).
    pub fn ensure_range_leased(
        &mut self,
        layer: u32,
        kind: ExpertKind,
        first: u32,
        last: u32,
        ptrs: &mut [*const u8],
    ) -> Result<(), PoolError> {
        let pool = &*self.pool;
        let n = pool.geometry.n_experts;
        if first > last {
            return Err(PoolError::Contract {
                reason: format!("inverted expert range [{first}, {last}]"),
            });
        }
        if (layer as usize) >= pool.geometry.layers.len() || last >= n {
            return Err(PoolError::Contract {
                reason: format!("unit out of range: layer {layer}, experts [{first}, {last}]"),
            });
        }
        if ptrs.len() != (last - first + 1) as usize {
            return Err(PoolError::Contract {
                reason: format!(
                    "ptrs length {}: expected last - first + 1 = {}",
                    ptrs.len(),
                    last - first + 1
                ),
            });
        }
        let _inflight = pool.demand_guard(true);
        let base = layer as usize * n as usize;
        for (expert, dst) in (first..=last).zip(ptrs.iter_mut()) {
            let at = &mut self.lease_of[base + expert as usize];
            if *at == UNCLAIMED {
                let lease = pool.claim_leased(true, layer, expert, &self.snapshot)?;
                // Count successes only: one increment per unit per scheduler
                // compute, never for a failed attempt that will be retried.
                pool.stats.demand_claims.fetch_add(1, Ordering::Relaxed);
                *at = self.leases.len() as u32;
                self.leases.push(lease);
            }
            *dst = pool
                .plane_bytes(layer, &self.leases[*at as usize], kind)
                .as_ptr();
        }
        Ok(())
    }
}

/// Pool-owned claim table for the batched P3 miss-list path (ABI v5): the
/// same lease-until-copy_done contract as [`StepClaims`], but reachable
/// through `&ExpertPool` so the FFI trampoline needs no thread-local state.
/// Decode-thread-only access — one [`ExpertPool::ensure_batch`] per
/// (layer, step, split-segment) and one [`ExpertPool::end_batch_step`] per
/// compute, both from the single decode thread (stated invariant; the mutex
/// exists for `&self` legality, never contended).
struct BatchClaims {
    lease_of: Vec<u32>,
    leases: Vec<Lease>,
    snapshot: Arc<PolicySnapshot>,
}

impl BatchClaims {
    fn new(n_experts: u32, n_layers: usize) -> Self {
        Self {
            lease_of: vec![UNCLAIMED; n_layers * n_experts as usize],
            leases: Vec::new(),
            snapshot: Arc::new(PolicySnapshot::empty()),
        }
    }

    fn next_step(&mut self, pool: &ExpertPool) {
        let n = pool.geometry.n_experts as usize;
        for lease in self.leases.drain(..) {
            self.lease_of[lease.layer as usize * n + lease.expert as usize] = UNCLAIMED;
        }
        self.snapshot = pool.snapshot();
    }
}

struct DemandGuard<'a>(&'a AtomicUsize);

impl Drop for DemandGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The RAM expert pool: arena + residency map + source + snapshots + stats,
/// `Arc`-shared with the FFI glue and the prefetcher.
pub struct ExpertPool {
    pub(crate) arena: Arc<Arena>,
    /// Per-layer gate/up/down unit layout (§6.1): plane offsets/strides
    /// inside a slot.
    pub geometry: PoolGeometry,
    pub(crate) residency: Arc<ResidencyMap>,
    /// P3 counters.
    pub stats: Arc<P3Stats>,
    /// In-flight demand page-ins (§6.5 demand-priority signal): the
    /// prefetcher skips rounds while this is nonzero.
    demand_inflight: AtomicUsize,
    source: Arc<dyn ExpertSource>,
    snapshots: Arc<SnapshotHandle>,
    /// Batched miss-list claim state (ABI v5); see [`BatchClaims`].
    batch_claims: super::primitives::Mutex<BatchClaims>,
    /// Pending VRAM-eviction D2H claims (O10): fill gate closed, lease held,
    /// until [`ExpertPool::commit_d2h`]. Bounded by the routed set of one
    /// compute (≤ K per layer); capacity is reserved at construction.
    d2h_pending: Mutex<Vec<PendingD2h>>,
    /// Reused lease scratch for [`ExpertPool::end_batch_step_exclusive`]
    /// (decode-thread-only; capacity stabilizes after the first step).
    exclusive_scratch: Mutex<Vec<Lease>>,
    /// Reused (layer, expert, slot) scratch beside [`ExpertPool::exclusive_scratch`].
    exclusive_triples: Mutex<Vec<(u32, u32, u32)>>,
    /// Units whose only current copy is the VRAM slab (O10 bookkeeping):
    /// steers the prefetcher off VRAM-resident experts (no disk read while a
    /// VRAM copy exists). Best-effort; a stale bit only skips a prefetch.
    vram: VramBits,
}

/// One claimed-but-not-yet-committed VRAM-eviction D2H.
struct PendingD2h {
    layer: u32,
    expert: u32,
    /// The claimed slot: keeps it resident and un-evictable until commit.
    lease: Lease,
}

/// Outcome of [`ExpertPool::claim_d2h`]. `Dest` carries the three plane
/// pointers (gate/up/down ordinal order) of the claimed slot.
pub enum D2hIncoming {
    /// The pool copy is current: the VRAM bytes are droppable.
    Resident,
    /// No acceptable pool slot (score floor kept every victim): droppable.
    Dropped,
    /// Copy the three planes into these destinations.
    Dest([*mut u8; 3]),
}

/// Flat VRAM-presence bitset over `layers × n_experts` keys.
#[derive(Debug)]
struct VramBits {
    bits: Vec<AtomicU64>,
    words_per_layer: usize,
}

impl VramBits {
    fn new(layers: usize, n_experts: usize) -> Self {
        let words_per_layer = n_experts.div_ceil(64);
        Self {
            bits: (0..layers * words_per_layer)
                .map(|_| AtomicU64::new(0))
                .collect(),
            words_per_layer,
        }
    }

    fn test(&self, layer: u32, expert: u32) -> bool {
        let w = layer as usize * self.words_per_layer + expert as usize / 64;
        self.bits
            .get(w)
            .is_some_and(|w| w.load(Ordering::Relaxed) & (1 << (expert % 64)) != 0)
    }

    fn set(&self, layer: u32, expert: u32) {
        if let Some(w) = self
            .bits
            .get(layer as usize * self.words_per_layer + expert as usize / 64)
        {
            w.fetch_or(1 << (expert % 64), Ordering::Relaxed);
        }
    }

    fn clear(&self, layer: u32, expert: u32) {
        if let Some(w) = self
            .bits
            .get(layer as usize * self.words_per_layer + expert as usize / 64)
        {
            w.fetch_and(!(1 << (expert % 64)), Ordering::Relaxed);
        }
    }

    fn clear_all(&self) {
        for w in &self.bits {
            w.store(0, Ordering::Relaxed);
        }
    }
}

impl std::fmt::Debug for ExpertPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    /// (§5.4).
    ///
    /// # Errors
    /// Inconsistent input: no layers or experts, `class_of_layer` and
    /// `geometry.layers` of different length, or a layer naming a class the
    /// arena does not have.
    pub fn try_new(
        arena: Arena,
        source: Arc<dyn ExpertSource>,
        snapshots: Arc<SnapshotHandle>,
        stats: Arc<P3Stats>,
        geometry: PoolGeometry,
        class_of_layer: Vec<u8>,
    ) -> Result<Arc<Self>, PoolConfigError> {
        if class_of_layer.len() != geometry.layers.len() {
            return Err(PoolConfigError::new(format!(
                "{} layer classes for {} geometry layers",
                class_of_layer.len(),
                geometry.layers.len()
            )));
        }
        if let Some(&c) = class_of_layer
            .iter()
            .find(|&&c| usize::from(c) >= arena.class_count())
        {
            return Err(PoolConfigError::new(format!(
                "layer class {c} but the arena has {} classes",
                arena.class_count()
            )));
        }
        let class_slots = (0..arena.class_count())
            .map(|c| arena.class_range(c))
            .collect();
        let residency = ResidencyMap::new(class_of_layer, geometry.n_experts, class_slots)?;
        let batch_claims = BatchClaims::new(geometry.n_experts, geometry.layers.len());
        let vram = VramBits::new(geometry.layers.len(), geometry.n_experts as usize);
        // One compute stages at most K units per layer into the slab; evicted
        // units per compute are bounded by the same number.
        let d2h_pending = Vec::with_capacity(geometry.layers.len() * 64);
        Ok(Arc::new(Self {
            arena: Arc::new(arena),
            geometry,
            residency: Arc::new(residency),
            stats,
            demand_inflight: super::primitives::AtomicUsize::new(0),
            source,
            snapshots,
            batch_claims: super::primitives::Mutex::new(batch_claims),
            d2h_pending: super::primitives::Mutex::new(d2h_pending),
            exclusive_scratch: super::primitives::Mutex::new(Vec::new()),
            exclusive_triples: super::primitives::Mutex::new(Vec::new()),
            vram,
        }))
    }

    /// Claims and page-ins the whole `(kind, expert)` miss list of ONE layer
    /// in a single call (ABI v5 batched P3: one shared-queue submission for
    /// all planes), writing each miss's pool-side plane pointer into `out`.
    /// Each `(layer, expert)` unit is claimed and leased exactly once per
    /// step regardless of how many of its planes miss; the leases are held
    /// until [`Self::end_batch_step`] so eviction cannot recycle a slot
    /// mid-DMA. Steady state allocates nothing (preallocated table + lease
    /// list). Caller-preallocated `out` (`misses.len()` entries).
    ///
    /// # Errors
    /// Length-mismatched `out`, a layer or expert out of range, a source
    /// failure after bounded retries, or an over-subscribed pool
    /// ([`PoolError::Contract`] for the first three shapes' contract
    /// violations). On error the units claimed so far stay leased until the
    /// next `end_batch_step`.
    pub fn ensure_batch(
        &self,
        layer: u32,
        misses: &[(ExpertKind, u32)],
        out: &mut [*const u8],
    ) -> Result<(), PoolError> {
        if out.len() != misses.len() {
            return Err(PoolError::Contract {
                reason: format!(
                    "out length {}: expected misses.len() = {}",
                    out.len(),
                    misses.len()
                ),
            });
        }
        if (layer as usize) >= self.geometry.layers.len() {
            return Err(PoolError::Contract {
                reason: format!("layer {layer} out of range"),
            });
        }
        let n = self.geometry.n_experts;
        let mut batch = self.batch_claims.lock().unwrap_or_else(|e| e.into_inner());
        let _inflight = self.demand_guard(true);
        for (&(kind, expert), dst) in misses.iter().zip(out.iter_mut()) {
            if expert >= n {
                return Err(PoolError::Contract {
                    reason: format!("expert {expert} out of range (n_experts {n})"),
                });
            }
            let base = layer as usize * n as usize + expert as usize;
            if batch.lease_of[base] == UNCLAIMED {
                let lease = self.claim_leased(true, layer, expert, &batch.snapshot)?;
                // Count successes only, as in ensure_range_leased.
                self.stats.demand_claims.fetch_add(1, Ordering::Relaxed);
                batch.lease_of[base] = batch.leases.len() as u32;
                batch.leases.push(lease);
            }
            let lease_at = batch.lease_of[base] as usize;
            *dst = self
                .plane_bytes(layer, &batch.leases[lease_at], kind)
                .as_ptr();
        }
        Ok(())
    }

    /// Ends the batched step: releases every lease taken by
    /// [`Self::ensure_batch`] since the previous call and refreshes the
    /// policy snapshot. The caller guarantees the device has consumed every
    /// pointer handed out (the glue's `copy_done`).
    pub fn end_batch_step(&self) {
        self.batch_claims
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .next_step(self);
    }

    /// [`Self::end_batch_step`] under EXCLUSIVE residency (O10): after the
    /// leases drop, every unit the step staged H2D is VRAM-resident, so its
    /// pool slot is released (each release re-checks zero leases — a
    /// concurrent prefetch claim keeps the unit inclusive) and the unit is
    /// marked VRAM-present, which steers the prefetcher off it (no disk read
    /// while the VRAM copy exists). No allocation: the release list is the
    /// drained lease vector's contents, moved into a reused scratch.
    pub fn end_batch_step_exclusive(&self) {
        let mut batch = self.batch_claims.lock().unwrap_or_else(|e| e.into_inner());
        // Swap the step's leases into a reused scratch (no per-step
        // allocation — decode hot path) so `lease_of` is writable while we
        // walk them.
        let mut scratch = self
            .exclusive_scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut triples = self
            .exclusive_triples
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::mem::swap(&mut *scratch, &mut batch.leases);
        let n = self.geometry.n_experts as usize;
        triples.extend(scratch.drain(..).map(|l| {
            batch.lease_of[l.layer as usize * n + l.expert as usize] = UNCLAIMED;
            (l.layer, l.expert, l.slot())
        }));
        // Leases dropped with the drained vector: the slots are un-leased, so
        // the release succeeds unless an in-flight prefetch claim re-leased
        // the key in between — that unit stays inclusive by design. The VRAM
        // bit is set BEFORE the release: a prefetcher that filtered on
        // `is_vram` earlier then re-checks after its claim (see
        // `release_if_vram`) and sees the bit, so it cannot re-page a key
        // whose VRAM copy exists (the exclusive invariant, O10).
        for &(layer, expert, slot) in triples.iter() {
            self.vram.set(layer, expert);
            self.residency.release_resident(layer, expert, slot);
        }
        triples.clear();
        batch.snapshot = self.snapshot();
    }

    /// Claims slots for a whole layer's incoming VRAM evictions (O10): one
    /// call per (layer, compute) — the policy snapshot is taken ONCE for the
    /// batch (cached from this compute's `batch_claims`, as the H2D path
    /// does) and the pending-D2H table is locked once, instead of per
    /// evicted expert. Per-expert `floor` is the expert's published score: a
    /// CLOCK victim scoring strictly above it is kept and the copy dropped.
    /// Outcomes align with `experts`/`floors`. The caller must have cleared
    /// each key's VRAM presence bit before this call (the C side reports
    /// the eviction once, before any later ensure).
    pub fn claim_d2h_batch(
        &self,
        layer: u32,
        experts: &[u32],
        floors: &[f32],
        out: &mut [D2hIncoming],
    ) {
        assert_eq!(
            experts.len(),
            floors.len(),
            "claim_d2h_batch: floors/experts length mismatch"
        );
        assert_eq!(
            experts.len(),
            out.len(),
            "claim_d2h_batch: out length mismatch"
        );
        if experts.is_empty() {
            return;
        }
        // One snapshot for the whole batch: the floor decisions within one
        // compute share the snapshot the H2D ensure path used (refreshed at
        // end_batch_step), so victims rank consistently across the step.
        let snapshot = self
            .batch_claims
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot
            .clone();
        // One pending-table lock per batch: every claim pushed under it, so
        // CLOCK cannot recycle a claimed slot mid-batch either.
        let mut pending = self.d2h_pending.lock().unwrap_or_else(|e| e.into_inner());
        for (i, (&expert, &floor)) in experts.iter().zip(floors).enumerate() {
            out[i] = match self.residency.claim_d2h(layer, expert, &snapshot, floor) {
                super::residency::D2hClaim::Resident => D2hIncoming::Resident,
                super::residency::D2hClaim::Dropped => D2hIncoming::Dropped,
                super::residency::D2hClaim::Claimed { slot, claim } => {
                    let unit = &self.geometry.layers[layer as usize];
                    // SAFETY: the claim holds the key's only lease and the fill
                    // gate is closed (no pool reader exists); the view ends here,
                    // the raw plane pointers stay covered by the pending lease +
                    // closed gate until commit_d2h.
                    let mut dst = [std::ptr::null_mut::<u8>(); 3];
                    for (k, d) in dst.iter_mut().enumerate() {
                        let (off, _) = unit.plane(ExpertKind::ALL[k]);
                        *d = self.arena.slot_ptr_at(claim.slot(), off);
                    }
                    claim.defer();
                    let lease = Lease {
                        map: self.residency.clone(),
                        _arena: self.arena.clone(),
                        layer,
                        expert,
                        slot,
                    };
                    pending.push(PendingD2h {
                        layer,
                        expert,
                        lease,
                    });
                    D2hIncoming::Dest(dst)
                }
            };
        }
    }

    /// Cancels one pending VRAM-eviction D2H whose copies the substrate
    /// never fully issued (ABI v9 abandon): clears the fill gate and the
    /// key's residency, returns the slot to the free list and wakes class
    /// waiters — the key stays non-resident, its bytes re-read from disk
    /// later (the VRAM bit was already cleared at eviction time). No
    /// `finish_d2h`, so no pool reader ever observes the unwritten bytes.
    /// No-op if the expert's claim was never pending (dropped/resident at
    /// claim time). The pending lease drops after the cancel (release_lease
    /// 1 → 0) — cancel_d2h deliberately does not touch the lease count.
    pub fn abandon_d2h(&self, layer: u32, expert: u32) {
        let pending = {
            let mut pending = self.d2h_pending.lock().unwrap_or_else(|e| e.into_inner());
            pending
                .iter()
                .rposition(|p| p.layer == layer && p.expert == expert)
                .map(|pos| pending.swap_remove(pos))
        };
        if let Some(p) = pending {
            self.residency.cancel_d2h(layer, expert);
            drop(p.lease);
        }
    }

    /// Single-expert form of [`Self::claim_d2h_batch`] (tests and the
    /// inclusive ABI path).
    pub fn claim_d2h(&self, layer: u32, expert: u32, floor: f32) -> D2hIncoming {
        let mut out = [D2hIncoming::Dropped];
        self.claim_d2h_batch(layer, &[expert], &[floor], &mut out);
        std::mem::replace(&mut out[0], D2hIncoming::Dropped)
    }

    /// Completes every pending VRAM-eviction D2H: opens the fill gates (the
    /// caller guarantees the copies landed — the substrate drains its
    /// backends before `copy_done`) and drops the pending leases.
    pub fn commit_d2h(&self) {
        let mut pending = self.d2h_pending.lock().unwrap_or_else(|e| e.into_inner());
        for p in pending.drain(..) {
            self.residency.finish_d2h(p.layer, p.expert);
            drop(p.lease);
        }
    }

    /// Whether `(layer, expert)`'s only current copy is known to be the VRAM
    /// slab (exclusive residency bookkeeping; best-effort — a stale bit only
    /// makes the prefetcher skip a unit that is also RAM-resident).
    pub fn is_vram(&self, layer: u32, expert: u32) -> bool {
        self.vram.test(layer, expert)
    }

    /// Clears the VRAM-presence bit of one unit (the substrate reported the
    /// slab slot evicted).
    pub fn clear_vram(&self, layer: u32, expert: u32) {
        self.vram.clear(layer, expert);
    }

    /// Clears every VRAM-presence bit (the substrate's device cache was
    /// reset between turns).
    pub fn clear_all_vram(&self) {
        self.vram.clear_all();
    }

    /// Drops every lease of the staged (not yet `copy_done`'d) exclusive
    /// batch WITHOUT releasing its pool slots: the device-cache reset between
    /// turns destroyed the staged VRAM copies, so the units must stay
    /// pool-resident (their RAM copies are current) and no later
    /// [`Self::end_batch_step_exclusive`] may release those slots or set
    /// VRAM bits for them. Decode is idle between turns (the caller's
    /// invariant), so no lease is in flight on the device.
    pub fn reset_exclusive_batch(&self) {
        let mut batch = self.batch_claims.lock().unwrap_or_else(|e| e.into_inner());
        let mut scratch = self
            .exclusive_scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::mem::swap(&mut *scratch, &mut batch.leases);
        let n = self.geometry.n_experts as usize;
        for l in scratch.drain(..) {
            batch.lease_of[l.layer as usize * n + l.expert as usize] = UNCLAIMED;
        }
        // Leases dropped with the drained vector: un-leased, slots stay
        // resident. `lease_of` fully cleared — the next step re-claims fresh.
    }

    /// Prefetcher seam (exclusive residency, O10): if the unit gained a VRAM
    /// copy while this prefetch page-in was in flight (a concurrent
    /// `end_batch_step_exclusive` promotion), release the just-refilled RAM
    /// copy — the VRAM one is authoritative, and keeping both breaks the
    /// exactly-one-tier invariant. The narrow check-vs-release race with a
    /// fresh VRAM eviction is benign: `claim_d2h` re-fills a pool slot from
    /// the slab behind a closed gate before any reader is admitted.
    pub fn release_if_vram(&self, layer: u32, expert: u32) {
        if !self.vram.test(layer, expert) {
            return;
        }
        if let Some(slot) = self.residency.resident_slot(layer, expert) {
            self.residency.release_resident(layer, expert, slot);
        }
    }

    /// The key's outstanding pool lease count (test/observability seam).
    pub fn outstanding_leases(&self, layer: u32, expert: u32) -> u16 {
        self.residency.lease_count(layer, expert)
    }

    /// Publishes a rebuilt policy snapshot (the engine's per-step scored
    /// table) for the next compute's victim picks.
    pub fn publish_snapshot(&self, snapshot: Arc<PolicySnapshot>) {
        self.snapshots.update(snapshot);
    }

    /// [`Self::try_new`] for callers that treat bad input as a bug.
    ///
    /// # Panics
    /// Panics on inconsistent input.
    pub fn new(
        arena: Arena,
        source: Arc<dyn ExpertSource>,
        snapshots: Arc<SnapshotHandle>,
        stats: Arc<P3Stats>,
        geometry: PoolGeometry,
        class_of_layer: Vec<u8>,
    ) -> Arc<Self> {
        Self::try_new(arena, source, snapshots, stats, geometry, class_of_layer)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Whether `(layer, expert)` is resident (filled or mid-fill).
    pub fn is_resident(&self, layer: u32, expert: u32) -> bool {
        self.residency.is_resident(layer, expert)
    }

    /// Counts one in-flight demand page-in until the guard drops (every
    /// exit path, including unwinding).
    fn demand_guard(&self, demand: bool) -> Option<DemandGuard<'_>> {
        demand.then(|| {
            self.demand_inflight.fetch_add(1, Ordering::Relaxed);
            DemandGuard(&self.demand_inflight)
        })
    }

    /// Number of demand page-ins currently in flight (§6.5 gate input).
    pub fn demand_inflight(&self) -> usize {
        self.demand_inflight.load(Ordering::Relaxed)
    }

    /// The current policy snapshot, counted (one `SnapshotHandle` mutex hold).
    fn snapshot(&self) -> Arc<PolicySnapshot> {
        self.stats.snapshot_reads.fetch_add(1, Ordering::Relaxed);
        self.snapshots.current()
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
        let _inflight = self.demand_guard(demand);
        self.ensure_range_inner(demand, layer, kind, first, last, copy)
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
        let snapshot = self.snapshot();
        for expert in first..=last {
            let lease = self.claim_leased(demand, layer, expert, &snapshot)?;
            copy(self.plane_bytes(layer, &lease, kind));
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
        let _inflight = self.demand_guard(demand);
        let snapshot = self.snapshot();
        // Prefetch reads no bytes; the fill itself was the work.
        drop(self.claim_leased(demand, layer, expert, &snapshot)?);
        Ok(())
    }

    /// Shields `layer`'s predicted experts from eviction (§6.5 `protect`),
    /// replacing the layer's previous prediction.
    pub fn protect(&self, layer: u32, experts: impl IntoIterator<Item = u32>) {
        self.residency.protect(layer, experts);
    }

    /// Claims one expert's slot, paging it in if needed, and returns the
    /// UNDROPPED lease (the caller keeps the lease until the host→device DMA
    /// is known complete, then drops it).
    fn claim_leased(
        &self,
        demand: bool,
        layer: u32,
        expert: u32,
        snapshot: &PolicySnapshot,
    ) -> Result<Lease, PoolError> {
        let slot =
            match self
                .residency
                .claim_for(layer, expert, snapshot, demand, f32::NEG_INFINITY)
            {
                Ok(Claim::Resident { slot, prefetch_hit }) => {
                    if demand {
                        self.stats.fast_hits.fetch_add(1, Ordering::Relaxed);
                    }
                    if prefetch_hit {
                        self.stats.prefetch_hits.fetch_add(1, Ordering::Relaxed);
                    }
                    slot
                }
                Ok(Claim::Filled { slot, claim }) => {
                    self.page_in(demand, layer, expert, claim)?;
                    slot
                }
                Err(ClaimError::Evicted(evicted)) => {
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    let slot = evicted.slot;
                    debug_assert_eq!(slot, evicted.victim.slot);
                    if evicted.victim.prefetched {
                        let unit = &self.geometry.layers[evicted.victim.layer as usize];
                        self.stats
                            .prefetch_wasted_bytes
                            .fetch_add(unit.bytes(), Ordering::Relaxed);
                    }
                    self.page_in(demand, layer, expert, evicted.claim)?;
                    slot
                }
                Err(ClaimError::Busy) => return Err(PoolError::Busy),
            };
        Ok(Lease {
            map: self.residency.clone(),
            _arena: self.arena.clone(),
            layer,
            expert,
            slot,
        })
    }

    /// The `kind` plane's sub-slice of the unit's slot (§6.1 plane layout).
    fn plane_bytes<'l>(&self, layer: u32, lease: &'l Lease, kind: ExpertKind) -> &'l [u8] {
        let unit = &self.geometry.layers[layer as usize];
        let (off, stride) = unit.plane(kind);
        &lease.bytes()[off as usize..(off + stride) as usize]
    }

    /// Leases the `kind` plane of a RESIDENT `(layer, expert)` unit and
    /// returns its pool-side pointer plus the undropped lease (prefill
    /// stream, design §7.6.2: pooled hits copy H2D from the pinned slot
    /// under a lease, but never score or page in — a streamed ubatch must
    /// not change the decode hot set). `None` when the unit is not resident;
    /// the caller stages the plane instead. The lease drops at the plane's
    /// stream release (or the step's copy_done backstop).
    pub fn lease_resident(
        &self,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
    ) -> Option<(*const u8, Lease)> {
        if (layer as usize) >= self.geometry.layers.len() || expert >= self.geometry.n_experts {
            return None;
        }
        let slot = self.residency.lease_if_resident(layer, expert)?;
        let lease = Lease {
            map: self.residency.clone(),
            _arena: self.arena.clone(),
            layer,
            expert,
            slot,
        };
        let ptr = self.plane_bytes(layer, &lease, kind).as_ptr();
        Some((ptr, lease))
    }

    /// Fills one slot from the source. The `Filled`/evict claim left this key
    /// mid-fill in the residency shard: the fill writes through the claim's
    /// exclusive view, and only `finish` (before the lease-held copy) lets
    /// parked `Resident` claimers of this key read the slot. On failure the
    /// claim is cancelled (residency bit cleared, slot recycled, gate opened)
    /// and the error surfaces; a panic in the source cancels it via
    /// `FillClaim::drop`.
    fn page_in(
        &self,
        demand: bool,
        layer: u32,
        expert: u32,
        mut claim: FillClaim<'_>,
    ) -> Result<(), PoolError> {
        let unit = &self.geometry.layers[layer as usize];
        // One disk→pool copy per unit: three reads straight into the
        // claimed slot's plane regions (split from the single slot_mut
        // borrow — no bounce buffer, §3.4; each region is sized for an
        // aligned direct read of its plane, §5.4).
        let rest = self.arena.slot_mut(&mut claim);
        // One disk→pool copy per unit: one batched read landing all three
        // planes straight in the claimed slot's regions (split from the
        // single slot_mut borrow — no bounce buffer, §3.4; each region is
        // sized for an aligned direct read of its plane, §5.4; one queued
        // drain per unit, §5.2).
        let result: std::io::Result<()> = {
            let (t0, rest) = rest.split_at_mut(unit.region[0] as usize);
            let (t1, t2) = rest.split_at_mut(unit.region[1] as usize);
            let batch = [
                (ExpertKind::Gate, expert),
                (ExpertKind::Up, expert),
                (ExpertKind::Down, expert),
            ];
            let mut targets = [
                (t0, unit.head[0] as usize),
                (t1, unit.head[1] as usize),
                (t2, unit.head[2] as usize),
            ];
            self.source.read_expert_batched(layer, &batch, &mut targets)
        };
        match result {
            Ok(()) => {
                claim.finish();
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
                claim.cancel();
                self.stats.source_errors.fetch_add(1, Ordering::Relaxed);
                Err(PoolError::Source { source })
            }
        }
    }

    /// Leases a key that is already resident so its bytes can be read
    /// outside `ensure_range`.
    ///
    /// # Panics
    /// Panics when the key is not resident (before touching any state).
    #[cfg(test)]
    fn lease_for_test(&self, layer: u32, expert: u32) -> Lease {
        assert!(
            self.residency.is_resident(layer, expert),
            "lease_for_test on a non-resident unit"
        );
        let snapshot = PolicySnapshot::empty();
        let Ok(Claim::Resident { slot, .. }) = self.residency.claim(layer, expert, &snapshot)
        else {
            panic!("lease_for_test: resident unit was not claimed as Resident");
        };
        Lease {
            map: self.residency.clone(),
            _arena: self.arena.clone(),
            layer,
            expert,
            slot,
        }
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
            head: usize,
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
            assert_eq!(into.len() - head, self.stride);
            into[head..].fill(self.plane_byte(kind, expert));
            Ok(())
        }

        fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
            None
        }
    }

    /// One layer, three uniform planes of `stride` B — the unit is 3×stride.
    fn unit_geometry(stride: u64) -> PoolGeometry {
        PoolGeometry {
            layers: vec![UnitLayout::new([stride; 3], [None; 3]).unwrap(); 1],
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
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
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
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
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

    /// Source that only serves the batched path: records every batched call
    /// (kind, expert, head, region length) and optionally fails the Down
    /// plane a bounded number of times (§5.2: one queued drain per unit).
    #[cfg(not(loom))]
    struct BatchedSource {
        stride: usize,
        fail_down_times: std::sync::atomic::AtomicUsize,
        calls: Mutex<Vec<(ExpertKind, u32, usize, usize)>>,
    }

    #[cfg(not(loom))]
    impl BatchedSource {
        fn new(stride: usize, fail_down_times: usize) -> Self {
            Self {
                stride,
                fail_down_times: std::sync::atomic::AtomicUsize::new(fail_down_times),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn plane_byte(&self, kind: ExpertKind, expert: u32) -> u8 {
            0x10 * (kind.ordinal() as u8 + 1) + expert as u8
        }
    }

    #[cfg(not(loom))]
    impl ExpertSource for BatchedSource {
        fn read_expert(
            &self,
            _: u32,
            _: ExpertKind,
            _: u32,
            _: &mut [u8],
            _: usize,
        ) -> io::Result<()> {
            panic!("serial read_expert used; page_in must batch (§5.2)");
        }

        fn read_expert_batched(
            &self,
            _layer: u32,
            batch: &[(ExpertKind, u32)],
            targets: &mut [(&mut [u8], usize)],
        ) -> io::Result<()> {
            for (&(kind, expert), (into, head)) in batch.iter().zip(targets.iter_mut()) {
                self.calls
                    .lock()
                    .unwrap()
                    .push((kind, expert, *head, into.len()));
                if kind == ExpertKind::Down
                    && self
                        .fail_down_times
                        .load(std::sync::atomic::Ordering::SeqCst)
                        > 0
                {
                    self.fail_down_times
                        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    return Err(io::Error::other("injected batched plane failure"));
                }
                into[*head..*head + self.stride].fill(self.plane_byte(kind, expert));
            }
            Ok(())
        }

        fn plane(&self, _: u32, _: ExpertKind, _: u32) -> Option<(u64, u64)> {
            None
        }
    }

    /// Pool over one layer with the given layout, backed by a shared
    /// [`BatchedSource`] failing the Down plane `fail_down_times` times.
    #[cfg(not(loom))]
    fn batched_pool(
        layout: UnitLayout,
        fail_down_times: usize,
    ) -> (Arc<ExpertPool>, Arc<BatchedSource>) {
        let arena = Arena::new(&[SlotClass {
            slot_size: layout.region.iter().sum::<u64>() as usize,
            slots: 2,
        }])
        .unwrap();
        let src = Arc::new(BatchedSource::new(
            layout.stride[0] as usize,
            fail_down_times,
        ));
        let source: Arc<dyn ExpertSource> = src.clone();
        let pool = ExpertPool::new(
            arena,
            source,
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            PoolGeometry {
                layers: vec![layout],
                n_experts: N_EXPERTS,
                expert_bytes: 0,
            },
            vec![0; 1],
        );
        (pool, src)
    }

    #[test]
    #[cfg(not(loom))]
    fn page_in_submits_one_batched_call_for_the_three_planes() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [Some(5), None, Some(13)]).unwrap();
        let (pool, src) = batched_pool(layout, 0);
        for &kind in ExpertKind::ALL.iter() {
            let byte = 0x10 * (kind.ordinal() as u8 + 1) + 3;
            pool.ensure_range(true, 0, kind, 3, 3, &mut |p| assert_eq!(p, [byte; 64]))
                .unwrap();
        }
        // Exactly one page-in: one batched call, three plane targets at the
        // layout's heads and region lengths, one unit resident.
        let calls = src.calls.lock().unwrap().clone();
        let want: Vec<(ExpertKind, u32, usize, usize)> = ExpertKind::ALL
            .iter()
            .map(|&k| {
                (
                    k,
                    3,
                    layout.head[k.ordinal()] as usize,
                    layout.region[k.ordinal()] as usize,
                )
            })
            .collect();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls, want);
        assert_eq!(pool.stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
        assert_eq!(pool.stats.bytes_paged.load(Ordering::Relaxed), 3 * stride);
        assert!(pool.residency.is_resident(0, 3));
    }

    #[test]
    #[cfg(not(loom))]
    fn batched_failure_cancels_claim_without_ghost_residency() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, _) = batched_pool(layout, 1);
        assert!(
            pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |_| {})
                .is_err()
        );
        assert!(!pool.residency.is_resident(0, 3), "no ghost residency");
        // Slot recycled: a clean retry succeeds.
        pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |p| {
            assert_eq!(p, [0x13; 64])
        })
        .unwrap();
        assert!(pool.residency.is_resident(0, 3));
    }

    /// Source that panics in `read_expert` the first time it is called.
    struct PanicOnce(std::sync::atomic::AtomicBool);

    impl ExpertSource for PanicOnce {
        fn read_expert(
            &self,
            _: u32,
            kind: ExpertKind,
            expert: u32,
            into: &mut [u8],
            head: usize,
        ) -> io::Result<()> {
            if self.0.swap(false, std::sync::atomic::Ordering::SeqCst) {
                panic!("source panicked mid-fill");
            }
            into[head..].fill(0x10 * (kind.ordinal() as u8 + 1) + expert as u8);
            Ok(())
        }

        fn plane(&self, _: u32, _: ExpertKind, _: u32) -> Option<(u64, u64)> {
            None
        }
    }

    #[test]
    #[cfg(not(loom))]
    fn panicking_source_leaves_no_filling_key_and_no_inflight_demand() {
        let stride = 64u64;
        let arena = Arena::new(&[SlotClass {
            slot_size: 3 * stride as usize,
            slots: 1,
        }])
        .unwrap();
        let source: Arc<dyn ExpertSource> = arc_dyn(std::sync::Arc::new(PanicOnce(
            std::sync::atomic::AtomicBool::new(true),
        )));
        let pool = ExpertPool::new(
            arena,
            source,
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            unit_geometry(stride),
            vec![0; 1],
        );
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |_| {});
        }));
        assert!(unwound.is_err());
        assert_eq!(pool.demand_inflight(), 0, "guard released on unwind");
        assert!(!pool.is_resident(0, 3), "no ghost residency");
        // Would park forever on the fill gate if the key stayed `filling`.
        pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |p| {
            assert_eq!(p, [0x13; 64])
        })
        .unwrap();
    }

    #[test]
    #[cfg(not(loom))]
    fn inconsistent_construction_is_an_error() {
        let arena = Arena::new(&[SlotClass {
            slot_size: 192,
            slots: 1,
        }])
        .unwrap();
        let source: Arc<dyn ExpertSource> = arc_dyn(std::sync::Arc::new(FakeSource::new(64)));
        let snaps = Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty())));
        // Two layer classes for a one-layer geometry.
        let err = ExpertPool::try_new(
            arena,
            source,
            snaps,
            Arc::new(P3Stats::new()),
            unit_geometry(64),
            vec![0, 0],
        )
        .unwrap_err();
        assert!(err.to_string().contains("geometry layers"), "{err}");
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
        assert_eq!(lease.bytes()[0], 0x11);
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
        assert_eq!(lease.bytes()[0], 0x13);
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
        let mut snap = PolicySnapshot::empty();
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

    #[cfg(not(loom))]
    fn plane_byte(kind: ExpertKind, expert: u32) -> u8 {
        0x10 * (kind.ordinal() as u8 + 1) + expert as u8
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn step_claims_each_unit_once_and_reads_the_snapshot_once_per_step() {
        let pool = test_pool(3);
        let mut step = StepClaims::new(pool.clone());
        let reads = pool.stats.snapshot_reads.load(Ordering::Relaxed);
        let mut ptrs = [std::ptr::null(); 3];
        for kind in ExpertKind::ALL {
            step.ensure_range_leased(0, kind, 0, 2, &mut ptrs).unwrap();
            for (e, &p) in (0u32..).zip(&ptrs) {
                let lease = &step.leases()[e as usize];
                let bytes = pool.plane_bytes(0, lease, kind);
                assert_eq!(
                    p,
                    bytes.as_ptr(),
                    "{kind:?} e{e} points into its unit's lease"
                );
                assert!(
                    bytes.iter().all(|&b| b == plane_byte(kind, e)),
                    "{kind:?} e{e}"
                );
            }
        }
        let s = &pool.stats;
        assert_eq!(
            s.demand_claims.load(Ordering::Relaxed),
            3,
            "one claim per unit"
        );
        assert_eq!(s.demand_slow_pageins.load(Ordering::Relaxed), 3);
        assert_eq!(
            s.fast_hits.load(Ordering::Relaxed),
            0,
            "up/down never re-claimed"
        );
        assert_eq!(step.leases().len(), 3);
        assert_eq!(
            s.snapshot_reads.load(Ordering::Relaxed),
            reads,
            "no snapshot read per call"
        );
        // All 3 slots leased across the planes: a fourth unit cannot evict.
        assert!(matches!(
            pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |_| {}),
            Err(PoolError::Busy)
        ));

        // Re-read: the Busy `ensure_range` above took its own snapshot read.
        let reads = s.snapshot_reads.load(Ordering::Relaxed);
        step.next_step();
        assert_eq!(
            s.snapshot_reads.load(Ordering::Relaxed),
            reads + 1,
            "one read per step"
        );
        assert!(step.leases().is_empty());
        // Leases released: the fourth unit now evicts one of them.
        pool.ensure_range(true, 0, ExpertKind::Gate, 3, 3, &mut |_| {})
            .unwrap();

        for kind in ExpertKind::ALL {
            step.ensure_range_leased(0, kind, 0, 1, &mut ptrs[..2])
                .unwrap();
        }
        assert_eq!(
            s.demand_claims.load(Ordering::Relaxed),
            5,
            "re-claimed once per unit"
        );
    }

    #[test]
    #[cfg(not(loom))]
    fn a_failed_step_claim_leaves_no_stale_entry() {
        let pool = test_pool_failing(2, (0, ExpertKind::Gate, 1));
        let mut step = StepClaims::new(pool.clone());
        let mut ptrs = [std::ptr::null(); 2];
        assert!(
            step.ensure_range_leased(0, ExpertKind::Gate, 0, 1, &mut ptrs)
                .is_err()
        );
        assert_eq!(step.leases().len(), 1, "e0 stays leased, e1 does not");
        // e1 was never recorded: the same step claims it afresh (fails once only).
        step.ensure_range_leased(0, ExpertKind::Up, 1, 1, &mut ptrs[..1])
            .unwrap();
        let bytes = pool.plane_bytes(0, &step.leases()[1], ExpertKind::Up);
        assert_eq!(ptrs[0], bytes.as_ptr());
        assert!(bytes.iter().all(|&b| b == plane_byte(ExpertKind::Up, 1)));
    }

    #[test]
    #[cfg(not(loom))]
    fn step_claim_reports_an_out_of_range_unit_as_contract_error() {
        let pool = test_pool(2);
        let mut step = StepClaims::new(pool);
        let mut ptrs = [std::ptr::null()];
        match step.ensure_range_leased(0, ExpertKind::Gate, N_EXPERTS, N_EXPERTS, &mut ptrs) {
            Err(PoolError::Contract { reason }) => {
                assert!(reason.contains("out of range"), "{reason}")
            }
            other => panic!("expected Contract error, got {other:?}"),
        }
    }

    #[test]
    #[cfg(not(loom))]
    fn step_claim_reports_an_inverted_range_as_contract_error() {
        let pool = test_pool(2);
        let mut step = StepClaims::new(pool);
        assert!(matches!(
            step.ensure_range_leased(0, ExpertKind::Gate, 1, 0, &mut []),
            Err(PoolError::Contract { .. })
        ));
    }

    #[test]
    #[cfg(not(loom))]
    fn abandoned_d2h_leaves_the_slot_free_gate_open_and_wakes_waiters() {
        let pool = test_pool(2);
        let _gen0 = pool.residency.class_generation(0);
        // Evict expert 1's slab slot; the pool takes the D2H (gate closed,
        // slot leased, pending).
        pool.clear_vram(0, 1);
        assert!(matches!(pool.claim_d2h(0, 1, 0.0), D2hIncoming::Dest(_)));
        assert!(pool.is_resident(0, 1));
        // The copies were never fully issued: abandon instead of commit.
        pool.abandon_d2h(0, 1);
        assert!(
            !pool.is_resident(0, 1),
            "abandoned key must not be resident"
        );
        assert!(
            !pool.is_vram(0, 1),
            "abandoned key must not be VRAM-present"
        );
        assert_eq!(
            pool.residency.lease_count(0, 1),
            0,
            "abandon must release the pending lease"
        );
        // Slot back on the free list: a fresh D2H claim for the same key
        // re-fills (Dest, not Resident) — no ghost residency.
        pool.clear_vram(0, 1);
        assert!(matches!(pool.claim_d2h(0, 1, 0.0), D2hIncoming::Dest(_)));
        // Claim/release paths also bump the generation — sample after them.
        let gen1 = pool.residency.class_generation(0);
        pool.abandon_d2h(0, 1);
        // The cancel bumped the class wake generation: no-victim parkers
        // re-check instead of sleeping through the freed slot.
        assert!(
            pool.residency.class_generation(0) > gen1,
            "abandon must bump the class generation"
        );
    }

    #[test]
    #[cfg(not(loom))]
    fn step_claim_reports_a_length_mismatch_as_contract_error() {
        let pool = test_pool(2);
        let mut step = StepClaims::new(pool);
        let mut ptrs = [std::ptr::null(); 2];
        assert!(matches!(
            step.ensure_range_leased(0, ExpertKind::Gate, 0, 0, &mut ptrs),
            Err(PoolError::Contract { .. })
        ));
    }
}
