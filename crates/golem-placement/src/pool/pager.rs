//! Demand pager (§3.4 P3, §6.6): `ensure_range` pages missing experts in
//! (I/O outside all locks) and runs the caller's copy exactly once per key
//! while its lease is held — the P3 callback body. Eviction happens inside
//! [`ResidencyMap::claim`] (victim pick + bit clear + slot reassignment, one
//! shard-lock critical section); a claim grant on ANY path — `Filled`,
//! `Resident`, or `Err(ClaimError::Evicted)` — carries one lease, so a key being filled or
//! copied can never be picked as a victim and its slot can never be handed to
//! another claimer.

// cfg(loom) is set via RUSTFLAGS by the T6 leg; see primitives.rs.

use super::arena::{Arena, DownWindow, SplitClaim};
use super::primitives::{Arc, AtomicU64, AtomicUsize, Condvar, Mutex, Ordering};
use super::residency::{
    Claim, ClaimError, ExpertKind, FillClaim, OwnedFillClaim, P3Stats, PolicySnapshot,
    PoolConfigError, ResidencyMap,
};
use super::source::{ExpertSource, PoolGeometry, UnitLayout};
use golem_store::recycle;

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

    /// Publishes `snapshot` and returns the one it replaced, so the
    /// publisher can recycle it once no reader holds it.
    pub fn swap(&self, snapshot: Arc<PolicySnapshot>) -> Arc<PolicySnapshot> {
        std::mem::replace(&mut *self.current.lock().unwrap(), snapshot)
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
    /// A role-major down read did not complete within the pool's join
    /// bound ([`DOWN_JOIN_TIMEOUT`] in production): its slots are still
    /// being written, so nothing may settle them — fatal for the caller,
    /// like a stuck device wait.
    #[error("role-major down read of layer {layer} still pending after the join bound")]
    DownStalled {
        /// The layer whose down job is stuck.
        layer: u32,
    },
}

/// Default bound on one role-major join, shared by every layer it waits on
/// (the device-wait bound).
pub const DOWN_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

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
    /// #         -> std::io::Result<()> { into[..8].fill(1); Ok(()) }
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

/// Resident-unit leases taken by [`ExpertPool::lease_resident_take`] (CPU
/// expert path), held until [`Self::release_all`] or drop. The key list is
/// preallocated at construction and never grows: a take beyond the remaining
/// capacity takes fewer units, so the decode path never allocates here.
/// Keeps the arena alive while the handed-out plane pointers may be read.
#[derive(Debug)]
pub struct LeaseBatch {
    map: Arc<ResidencyMap>,
    _arena: Arc<Arena>,
    /// `(layer, expert)` per held lease, in take order (runs of one layer).
    keys: Vec<(u32, u32)>,
}

impl LeaseBatch {
    /// An empty batch for `pool` holding at most `capacity` leases (the
    /// engine sizes it `n_layers × n_expert_used`).
    pub fn new(pool: &ExpertPool, capacity: usize) -> Self {
        Self {
            map: pool.residency.clone(),
            _arena: pool.arena.clone(),
            keys: Vec::with_capacity(capacity),
        }
    }

    /// Leases currently held.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether no lease is held.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Releases every held lease: one shard lock per layer run, one wake per
    /// affected class.
    pub fn release_all(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        self.map.release_batch(&self.keys);
        self.keys.clear();
    }
}

impl Drop for LeaseBatch {
    fn drop(&mut self) {
        self.release_all();
    }
}

/// `lease_of` entry of a unit not claimed this step.
const UNCLAIMED: u32 = u32::MAX;
/// `BatchClaims::lease_of` entry of a unit with a fill pending in the
/// current `ensure_batch` call; never survives the call.
const PENDING: u32 = u32::MAX - 1;
/// `BatchClaims::lease_of` entry of a role-major unit whose gate/up landed
/// and whose down read is queued or failed: its owned claim sits in
/// `BatchClaims::split_claims` (or `poisoned`) until a join settles it.
/// `reset_pending` leaves it untouched.
const SPLIT: u32 = u32::MAX - 2;

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
    /// Reused per-call scratch of [`ExpertPool::ensure_batch`]: the layer's
    /// pending fills `(expert, slot, claim)`, their planes and read targets.
    /// Empty between calls; [`recycle`] re-lifetimes the allocations.
    fills: Vec<(u32, u32, FillClaim<'static>)>,
    planes: Vec<(ExpertKind, u32)>,
    targets: Vec<(&'static mut [u8], usize)>,
    /// Role-major scratch beside `fills`: the call's split units, and their
    /// down windows before they move into the layer's [`DownJob`].
    splits: Vec<(u32, u32, FillClaim<'static>)>,
    windows: Vec<DownUnit>,
    /// Per layer: the split claims of units whose down read is queued
    /// (`SPLIT` in `lease_of`), in the order of the job's windows. Each
    /// holds the unit's only lease; only the decode thread settles them, at
    /// a join, after taking their windows back.
    split_claims: Vec<Vec<SplitClaim>>,
    /// Claims whose down read failed: still mid-fill (slot leased, off the
    /// free list) until `end_batch_step*` or an idle settle cancels them,
    /// after C drained every gate/up DMA from their slots.
    poisoned: Vec<OwnedFillClaim>,
}

impl BatchClaims {
    /// `split_cap` units per layer of role-major state (0 = off: nothing
    /// allocated).
    fn new(n_experts: u32, n_layers: usize, split_cap: usize) -> Self {
        let (split_claims, poisoned) = if split_cap == 0 {
            (Vec::new(), Vec::new())
        } else {
            (
                (0..n_layers)
                    .map(|_| Vec::with_capacity(split_cap))
                    .collect(),
                Vec::with_capacity(n_layers * split_cap),
            )
        };
        Self {
            lease_of: vec![UNCLAIMED; n_layers * n_experts as usize],
            leases: Vec::new(),
            snapshot: Arc::new(PolicySnapshot::empty()),
            fills: Vec::new(),
            planes: Vec::new(),
            targets: Vec::new(),
            splits: Vec::with_capacity(split_cap),
            windows: Vec::with_capacity(split_cap),
            split_claims,
            poisoned,
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

/// The error a failed role-major down read surfaces as (the store already
/// logged the underlying I/O error).
fn down_failed() -> PoolError {
    PoolError::Source {
        source: std::io::Error::other("role-major down read failed"),
    }
}

fn reset_pending(lease_of: &mut [u32], base: usize, misses: &[(ExpertKind, u32)]) {
    for &(_, expert) in misses {
        let at = &mut lease_of[base + expert as usize];
        if *at == PENDING {
            *at = UNCLAIMED;
        }
    }
}

/// One split unit's down plane, queued for the reader thread.
struct DownUnit {
    expert: u32,
    /// The plane's offset inside its region (`UnitLayout::head[2]`).
    head: usize,
    window: DownWindow,
}

/// Lifecycle of one layer's down job: the decode thread queues it, the
/// reader moves it to `Done`/`Failed`, the decode thread's join returns it
/// to `Idle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobState {
    Idle,
    Queued,
    Done,
    Failed,
}

struct DownJobInner {
    state: JobState,
    /// Preallocated to the split cap; handed to the reader by value while
    /// it reads (`mem::take`, no allocation) and returned before the state
    /// leaves `Queued`.
    units: Vec<DownUnit>,
}

/// One layer's down job: state + windows behind a per-layer mutex, and the
/// completion latch the decode thread's join waits on.
struct DownJob {
    inner: Mutex<DownJobInner>,
    done: Condvar,
}

/// Role-major state shared by the pool (decode thread) and the down reader
/// thread: per-layer jobs (preallocated at pool creation, never resized)
/// and the reader's wake queue (at most one queued job per layer, so the
/// preallocated `n_layers` capacity never grows).
pub(crate) struct DownShared {
    jobs: Vec<DownJob>,
    queue: Mutex<std::collections::VecDeque<u32>>,
    wake: Condvar,
    stop: super::primitives::AtomicBool,
    source: Arc<dyn ExpertSource>,
    /// Attached readers ([`spawn_down_reader`] threads and job runners). A
    /// join that sees 0 knows no reader will take its job's windows and
    /// fails the job at once instead of waiting out the bound.
    readers: AtomicUsize,
    /// One join's bound, shared by every layer it waits on.
    join_timeout: std::time::Duration,
    /// Test seam: the next down job fails without reading.
    #[cfg(any(test, feature = "test-support"))]
    fail_next: super::primitives::AtomicBool,
}

/// Counts one attached reader until dropped (every exit, panics included).
struct ReaderGuard(Arc<DownShared>);

impl ReaderGuard {
    fn attach(shared: Arc<DownShared>) -> Self {
        shared.readers.fetch_add(1, Ordering::AcqRel);
        Self(shared)
    }
}

impl Drop for ReaderGuard {
    fn drop(&mut self) {
        self.0.readers.fetch_sub(1, Ordering::AcqRel);
        // A join checks the count under its job lock and parks atomically
        // with releasing it: taking each lock before the wake means every
        // parked join is woken after the decrement and re-checks.
        for job in &self.0.jobs {
            let _g = job.inner.lock().unwrap_or_else(|e| e.into_inner());
            job.done.notify_all();
        }
    }
}

impl DownShared {
    fn new(
        n_layers: usize,
        cap: usize,
        source: Arc<dyn ExpertSource>,
        join_timeout: std::time::Duration,
    ) -> Self {
        Self {
            jobs: (0..n_layers)
                .map(|_| DownJob {
                    inner: Mutex::new(DownJobInner {
                        state: JobState::Idle,
                        units: Vec::with_capacity(cap),
                    }),
                    done: Condvar::new(),
                })
                .collect(),
            queue: Mutex::new(std::collections::VecDeque::with_capacity(n_layers)),
            wake: Condvar::new(),
            stop: super::primitives::AtomicBool::new(false),
            source,
            readers: AtomicUsize::new(0),
            join_timeout,
            #[cfg(any(test, feature = "test-support"))]
            fail_next: super::primitives::AtomicBool::new(false),
        }
    }

    /// Next queued layer, parking while the queue is empty; `None` once
    /// stopped with nothing queued (a queued job is always run first).
    fn next(&self) -> Option<u32> {
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(layer) = q.pop_front() {
                return Some(layer);
            }
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            q = self.wake.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Runs `layer`'s queued down job: one batched read of every unit's
    /// down plane into its window, then `Done`/`Failed` and a latch wake.
    /// The reader touches only the windows and the job state — never a
    /// claim, the batch table or residency. A panicking source fails the
    /// job instead of leaving the decode thread waiting.
    fn run_job(&self, layer: u32, scratch: &mut DownScratch) {
        let job = &self.jobs[layer as usize];
        let mut units = {
            let mut g = job.inner.lock().unwrap_or_else(|e| e.into_inner());
            // A join that found no reader already failed this job.
            if g.state != JobState::Queued {
                return;
            }
            std::mem::take(&mut g.units)
        };
        #[cfg(any(test, feature = "test-support"))]
        let failed = self.fail_next.swap(false, Ordering::Relaxed);
        #[cfg(not(any(test, feature = "test-support")))]
        let failed = false;
        let ok = !failed
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut targets: Vec<(&mut [u8], usize)> =
                    recycle(std::mem::take(&mut scratch.targets));
                scratch.planes.clear();
                for u in units.iter_mut() {
                    scratch.planes.push((ExpertKind::Down, u.expert));
                    targets.push((u.window.as_mut_slice(), u.head));
                }
                let read = self
                    .source
                    .read_expert_batched(layer, &scratch.planes, &mut targets);
                scratch.targets = recycle(targets);
                read.is_ok()
            }))
            .unwrap_or(false);
        let mut g = job.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.units = units;
        g.state = if ok { JobState::Done } else { JobState::Failed };
        drop(g);
        job.done.notify_all();
    }
}

/// The reader thread's reused read scratch (allocated once per thread).
#[derive(Default)]
struct DownScratch {
    planes: Vec<(ExpertKind, u32)>,
    targets: Vec<(&'static mut [u8], usize)>,
}

/// The role-major down reader thread (§3.4 P3): a stop flag plus the
/// thread, following [`super::prefetch::PrefetchHandle`]. Holds the shared
/// job table and the source, nothing that owns the pool. [`Self::stop`]
/// (and `Drop`) runs any queued job to completion before joining.
pub struct DownReaderHandle {
    shared: Arc<DownShared>,
    thread: Option<super::primitives::JoinHandle<()>>,
}

impl DownReaderHandle {
    /// Signals the thread and joins it; queued jobs complete first.
    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        {
            let _q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.wake.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for DownReaderHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

impl std::fmt::Debug for DownReaderHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownReaderHandle").finish_non_exhaustive()
    }
}

/// An attached reader driven by the caller, one job per
/// [`Self::run_one`] — the reader body without the loop, for models that
/// own the thread (loom). Counts as a live reader until dropped.
#[cfg(any(loom, test, feature = "test-support"))]
pub struct DownJobRunner(ReaderGuard);

#[cfg(any(loom, test, feature = "test-support"))]
impl DownJobRunner {
    /// Attaches a runner to `pool`; `None` without role-major page-in.
    pub fn attach(pool: &ExpertPool) -> Option<Self> {
        Some(Self(ReaderGuard::attach(pool.down.clone()?)))
    }

    /// Runs one queued down job on the calling thread; `false` when
    /// nothing is queued.
    pub fn run_one(&self) -> bool {
        let down = &self.0.0;
        let layer = down
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front();
        let Some(layer) = layer else {
            return false;
        };
        down.run_job(layer, &mut DownScratch::default());
        true
    }
}

/// Spawns `pool`'s down reader thread (`golem-down-reader`); `Ok(None)` when
/// the pool was built without role-major page-in (no thread exists then).
///
/// # Errors
/// The OS refused to spawn the thread.
pub fn spawn_down_reader(pool: &ExpertPool) -> std::io::Result<Option<DownReaderHandle>> {
    let Some(shared) = pool.down.clone() else {
        return Ok(None);
    };
    // Attached before the spawn: a join between spawn and first park must
    // not see the reader as absent.
    let guard = ReaderGuard::attach(shared.clone());
    let body = move || {
        let guard = guard;
        let body = &guard.0;
        let mut scratch = DownScratch::default();
        while let Some(layer) = body.next() {
            body.run_job(layer, &mut scratch);
        }
    };
    #[cfg(not(loom))]
    let thread = std::thread::Builder::new()
        .name("golem-down-reader".into())
        .spawn(body)?;
    #[cfg(loom)]
    let thread = super::primitives::spawn(body);
    Ok(Some(DownReaderHandle {
        shared,
        thread: Some(thread),
    }))
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
    /// Role-major page-in state shared with the down reader thread; `None`
    /// keeps every fill single-phase.
    down: Option<Arc<DownShared>>,
    /// Units one layer's down job holds (`min(K, n_experts)`).
    split_cap: usize,
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

impl Drop for ExpertPool {
    /// Pool shutdown join: every down read lands (the reader handle is
    /// stopped first, which runs queued jobs) before the owned claims drop.
    fn drop(&mut self) {
        if self.down.is_some() {
            let _ = self.settle_split();
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
        Self::build(
            arena,
            source,
            snapshots,
            stats,
            geometry,
            class_of_layer,
            0,
            DOWN_JOIN_TIMEOUT,
        )
    }

    /// Shared constructor; `split_cap == 0` builds no role-major state.
    #[allow(clippy::too_many_arguments)]
    fn build(
        arena: Arena,
        source: Arc<dyn ExpertSource>,
        snapshots: Arc<SnapshotHandle>,
        stats: Arc<P3Stats>,
        geometry: PoolGeometry,
        class_of_layer: Vec<u8>,
        split_cap: usize,
        join_timeout: std::time::Duration,
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
        let batch_claims = BatchClaims::new(geometry.n_experts, geometry.layers.len(), split_cap);
        let vram = VramBits::new(geometry.layers.len(), geometry.n_experts as usize);
        // One compute stages at most K units per layer into the slab; evicted
        // units per compute are bounded by the same number.
        let d2h_pending = Vec::with_capacity(geometry.layers.len() * 64);
        let n_layers = geometry.layers.len();
        let down = (split_cap > 0).then(|| {
            Arc::new(DownShared::new(
                n_layers,
                split_cap,
                source.clone(),
                join_timeout,
            ))
        });
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
            down,
            split_cap,
        }))
    }

    /// [`Self::try_new`] with role-major page-in (§3.4 P3): a disk-missed
    /// unit whose down plane is not in the current miss list gets its
    /// gate/up read and handed out first, its down plane read by the down
    /// reader thread ([`spawn_down_reader`], which the caller must run)
    /// and joined before the down plane is needed or the step ends. Per
    /// layer, `min(max_k, n_experts)` units of job state are allocated
    /// here, never on the decode path.
    ///
    /// A join waits at most [`DOWN_JOIN_TIMEOUT`] in total; with no reader
    /// running it fails at once.
    ///
    /// # Errors
    /// As [`Self::try_new`]; `max_k == 0` is rejected.
    pub fn try_new_role_major(
        arena: Arena,
        source: Arc<dyn ExpertSource>,
        snapshots: Arc<SnapshotHandle>,
        stats: Arc<P3Stats>,
        geometry: PoolGeometry,
        class_of_layer: Vec<u8>,
        max_k: u32,
    ) -> Result<Arc<Self>, PoolConfigError> {
        Self::try_new_role_major_bounded(
            arena,
            source,
            snapshots,
            stats,
            geometry,
            class_of_layer,
            max_k,
            DOWN_JOIN_TIMEOUT,
        )
    }

    /// [`Self::try_new_role_major`] with an explicit bound on one join
    /// (shared by every layer the join waits on).
    ///
    /// # Errors
    /// As [`Self::try_new_role_major`]; a zero bound is rejected.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_role_major_bounded(
        arena: Arena,
        source: Arc<dyn ExpertSource>,
        snapshots: Arc<SnapshotHandle>,
        stats: Arc<P3Stats>,
        geometry: PoolGeometry,
        class_of_layer: Vec<u8>,
        max_k: u32,
        join_timeout: std::time::Duration,
    ) -> Result<Arc<Self>, PoolConfigError> {
        if join_timeout.is_zero() {
            return Err(PoolConfigError::new(
                "role-major page-in needs a join bound > 0".to_string(),
            ));
        }
        if max_k == 0 {
            return Err(PoolConfigError::new(
                "role-major page-in needs a per-layer unit cap > 0".to_string(),
            ));
        }
        // `split_down` cuts each slot at `region[0] + region[1]`: the cut
        // must stay inside every slot of the layer's class.
        for (l, (unit, &c)) in geometry.layers.iter().zip(&class_of_layer).enumerate() {
            let c = usize::from(c);
            if c >= arena.class_count() {
                continue; // `build` rejects it.
            }
            let slot_size = arena.slot_size_of(arena.class_range(c).start) as u64;
            if unit.region[0] + unit.region[1] > slot_size {
                return Err(PoolConfigError::new(format!(
                    "layer {l}: gate/up regions {} B exceed slot size {slot_size} B",
                    unit.region[0] + unit.region[1]
                )));
            }
        }
        let cap = max_k.min(geometry.n_experts) as usize;
        Self::build(
            arena,
            source,
            snapshots,
            stats,
            geometry,
            class_of_layer,
            cap,
            join_timeout,
        )
    }

    /// Claims and page-ins the whole `(kind, expert)` miss list of ONE layer
    /// in a single call (ABI v5 batched P3), writing each miss's pool-side
    /// plane pointer into `out`. Every unit is claimed first, in miss order,
    /// so slot assignment never depends on I/O completion order; then all
    /// missing units' planes go to the source as ONE batched read (one
    /// shared-queue drain, §5.2), so the queue keeps `io_depth` reads in
    /// flight per drive instead of one unit at a time. Each `(layer,
    /// expert)` unit is claimed and leased exactly once per step regardless
    /// of how many of its planes miss; the leases are held until
    /// [`Self::end_batch_step`] so eviction cannot recycle a slot mid-DMA.
    /// Steady state allocates nothing (preallocated table, lease list and
    /// fill scratch). Caller-preallocated `out` (`misses.len()` entries).
    ///
    /// # Errors
    /// Length-mismatched `out`, a layer or expert out of range
    /// ([`PoolError::Contract`], before any claim), a source failure after
    /// bounded retries (every fill of this call is cancelled), or an
    /// over-subscribed pool (fills claimed before the `Busy` are cancelled;
    /// victims they evicted stay evicted). On error the units leased by earlier calls and
    /// this call's resident hits stay leased until the next
    /// `end_batch_step`.
    pub fn ensure_batch(
        &self,
        layer: u32,
        misses: &[(ExpertKind, u32)],
        out: &mut [*const u8],
    ) -> Result<(), PoolError> {
        self.ensure_batch_inner(layer, misses, out, None::<fn(&[*const u8])>)
    }

    /// [`Self::ensure_batch`] with a hook between claim and disk read: once
    /// every unit is claimed, `out[i]` holds the plane pointer of each miss
    /// whose unit is already resident (readable now, leased until
    /// [`Self::end_batch_step`]) and null for each unit still to be read;
    /// `hits_ready` then sees that view, before the blocking source read, so
    /// the caller can queue the hits' H2D to overlap it (§3.4 P3). It runs at
    /// most once, only when there is at least one hit, and never when the
    /// claim fails; a later source failure returns after it already ran.
    /// The closure runs under the `batch_claims` lock and the demand guard:
    /// it must not re-enter the pool.
    ///
    /// # Errors
    /// As [`Self::ensure_batch`].
    pub fn ensure_batch_with(
        &self,
        layer: u32,
        misses: &[(ExpertKind, u32)],
        out: &mut [*const u8],
        hits_ready: impl FnOnce(&[*const u8]),
    ) -> Result<(), PoolError> {
        self.ensure_batch_inner(layer, misses, out, Some(hits_ready))
    }

    /// Shared body of [`Self::ensure_batch`] and [`Self::ensure_batch_with`];
    /// with no `hits_ready` the pre-fill hit view is skipped.
    fn ensure_batch_inner(
        &self,
        layer: u32,
        misses: &[(ExpertKind, u32)],
        out: &mut [*const u8],
        hits_ready: Option<impl FnOnce(&[*const u8])>,
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
        if let Some(&(_, expert)) = misses.iter().find(|&&(_, e)| e >= n) {
            return Err(PoolError::Contract {
                reason: format!("expert {expert} out of range (n_experts {n})"),
            });
        }
        let mut guard = self.batch_claims.lock().unwrap_or_else(|e| e.into_inner());
        let batch = &mut *guard;
        let _inflight = self.demand_guard(true);
        let base = layer as usize * n as usize;
        if self.down.is_some()
            && misses
                .iter()
                .any(|&(_, e)| batch.lease_of[base + e as usize] == SPLIT)
        {
            // Join point: a miss names a unit whose down read is queued. A
            // poisoned unit (its down read failed) stays SPLIT until the
            // step ends: fail this call (its read failure was counted at
            // the join that poisoned it).
            self.join_layer(layer, batch, self.join_deadline())?;
            if misses
                .iter()
                .any(|&(_, e)| batch.lease_of[base + e as usize] == SPLIT)
            {
                return Err(down_failed());
            }
        }
        // A PENDING left by an unwind through a previous call is stale.
        reset_pending(&mut batch.lease_of, base, misses);
        let mut fills: Vec<(u32, u32, FillClaim<'_>)> = recycle(std::mem::take(&mut batch.fills));
        let mut splits: Vec<(u32, u32, FillClaim<'_>)> = recycle(std::mem::take(&mut batch.splits));
        let claimed = self.claim_misses(layer, misses, batch, &mut fills);
        let result = claimed.and_then(|()| {
            let fill = |batch: &mut BatchClaims, fills: &mut _, splits: &mut _| {
                if self.down.is_some() {
                    self.fill_batch_role_major(layer, misses, batch, fills, splits)
                } else {
                    self.fill_batch(layer, batch, fills)
                }
            };
            let Some(hits_ready) = hits_ready else {
                return fill(batch, &mut fills, &mut splits);
            };
            let mut hits = 0;
            for (&(kind, expert), dst) in misses.iter().zip(out.iter_mut()) {
                let lease_at = batch.lease_of[base + expert as usize];
                *dst = if lease_at == UNCLAIMED || lease_at == PENDING || lease_at == SPLIT {
                    std::ptr::null()
                } else {
                    hits += 1;
                    self.plane_bytes(layer, &batch.leases[lease_at as usize], kind)
                        .as_ptr()
                };
            }
            if hits > 0 {
                hits_ready(&*out);
            }
            fill(batch, &mut fills, &mut splits)
        });
        batch.fills = recycle(fills);
        batch.splits = recycle(splits);
        if result.is_err() {
            reset_pending(&mut batch.lease_of, base, misses);
        }
        result?;
        for (&(kind, expert), dst) in misses.iter().zip(out.iter_mut()) {
            let lease_at = batch.lease_of[base + expert as usize];
            *dst = if lease_at == SPLIT {
                // Gate/up of a unit whose down read is queued: the owned
                // claim holds the slot (mid-fill, leased) until the join.
                // ≤ K owned claims per layer: a short scan, no map.
                let Some(claim) = batch.split_claims[layer as usize]
                    .iter()
                    .find(|c| c.expert() == expert)
                else {
                    return Err(PoolError::Contract {
                        reason: format!("unit ({layer}, {expert}) split without its claim"),
                    });
                };
                let (off, _) = self.geometry.layers[layer as usize].plane(kind);
                self.arena.slot_ptr_at(claim.slot(), off).cast_const()
            } else {
                self.plane_bytes(layer, &batch.leases[lease_at as usize], kind)
                    .as_ptr()
            };
        }
        Ok(())
    }

    /// Phase 1 of [`Self::ensure_batch`]: claims every unit of `misses` not
    /// yet leased this step, in miss order. Resident units are leased at
    /// once; units needing a fill are pushed to `fills` (their lease is the
    /// fill claim's until [`Self::fill_batch`] settles it). On `Busy` the
    /// pending fills are left for the caller to drop (dropping cancels).
    fn claim_misses<'m>(
        &'m self,
        layer: u32,
        misses: &[(ExpertKind, u32)],
        batch: &mut BatchClaims,
        fills: &mut Vec<(u32, u32, FillClaim<'m>)>,
    ) -> Result<(), PoolError> {
        let base = layer as usize * self.geometry.n_experts as usize;
        self.residency.claim_resident_batch(
            layer,
            misses.iter().map(|&(_, e)| e),
            |expert, slot, prefetch_hit| {
                let at = base + expert as usize;
                if batch.lease_of[at] != UNCLAIMED {
                    return false;
                }
                self.stats.fast_hits.fetch_add(1, Ordering::Relaxed);
                if prefetch_hit {
                    self.stats.prefetch_hits.fetch_add(1, Ordering::Relaxed);
                }
                self.stats.demand_claims.fetch_add(1, Ordering::Relaxed);
                batch.lease_of[at] = batch.leases.len() as u32;
                batch.leases.push(self.lease(layer, expert, slot));
                true
            },
        );
        for &(_, expert) in misses {
            let at = base + expert as usize;
            if batch.lease_of[at] != UNCLAIMED {
                continue;
            }
            let snapshot = &batch.snapshot;
            match self
                .residency
                .claim_for(layer, expert, snapshot, true, f32::NEG_INFINITY)
            {
                Ok(Claim::Resident { slot, prefetch_hit }) => {
                    self.stats.fast_hits.fetch_add(1, Ordering::Relaxed);
                    if prefetch_hit {
                        self.stats.prefetch_hits.fetch_add(1, Ordering::Relaxed);
                    }
                    self.stats.demand_claims.fetch_add(1, Ordering::Relaxed);
                    batch.lease_of[at] = batch.leases.len() as u32;
                    batch.leases.push(self.lease(layer, expert, slot));
                }
                Ok(Claim::Filled { slot, claim }) => {
                    batch.lease_of[at] = PENDING;
                    fills.push((expert, slot, claim));
                }
                Err(ClaimError::Evicted(evicted)) => {
                    self.note_eviction(&evicted.victim);
                    debug_assert_eq!(evicted.slot, evicted.victim.slot);
                    batch.lease_of[at] = PENDING;
                    fills.push((expert, evicted.slot, evicted.claim));
                }
                Err(ClaimError::Busy) => return Err(PoolError::Busy),
            }
        }
        Ok(())
    }

    /// Phase 2 of [`Self::ensure_batch`] with role-major page-in: a pending
    /// unit whose down plane is not in `misses` (up to the layer's cap) is
    /// split — its gate/up regions join the call's one batched read, its
    /// down region goes to the layer's down job for the reader thread. The
    /// rest read all three planes in the same drain. Success settles the
    /// whole units as [`Self::fill_batch`] does and marks the split units
    /// `SPLIT`, their owned claims (each holding the unit's only lease)
    /// kept until a join; failure cancels every fill of the call before any
    /// pointer is handed out.
    fn fill_batch_role_major<'m>(
        &'m self,
        layer: u32,
        misses: &[(ExpertKind, u32)],
        batch: &mut BatchClaims,
        fills: &mut Vec<(u32, u32, FillClaim<'m>)>,
        splits: &mut Vec<(u32, u32, FillClaim<'m>)>,
    ) -> Result<(), PoolError> {
        if fills.is_empty() {
            return Ok(());
        }
        let mut i = 0;
        while i < fills.len() {
            let expert = fills[i].0;
            let wants_down = misses
                .iter()
                .any(|&(k, e)| e == expert && k == ExpertKind::Down);
            if !wants_down && splits.len() < self.split_cap {
                splits.push(fills.remove(i));
            } else {
                i += 1;
            }
        }
        let Some(down) = self.down.as_deref().filter(|_| !splits.is_empty()) else {
            fills.append(splits);
            return self.fill_batch(layer, batch, fills);
        };
        // A new split queues on this layer's job: join an unjoined one first.
        self.join_layer(layer, batch, self.join_deadline())?;
        let unit = &self.geometry.layers[layer as usize];
        let (r0, r1) = (unit.region[0] as usize, unit.region[1] as usize);
        let mut planes = std::mem::take(&mut batch.planes);
        let mut targets: Vec<(&mut [u8], usize)> = recycle(std::mem::take(&mut batch.targets));
        for (expert, _, claim) in fills.iter_mut() {
            planes.extend(ExpertKind::ALL.map(|k| (k, *expert)));
            let (t0, rest) = self.arena.slot_mut(claim).split_at_mut(r0);
            let (t1, t2) = rest.split_at_mut(r1);
            targets.push((t0, unit.head[0] as usize));
            targets.push((t1, unit.head[1] as usize));
            targets.push((t2, unit.head[2] as usize));
        }
        // The split claims sit in the layer's (joined, so empty) list from
        // here on, each paired by index with its window in `windows`.
        let split_claims = &mut batch.split_claims[layer as usize];
        for (expert, _, claim) in splits.drain(..) {
            let owned = claim.into_owned(self.residency.clone());
            let (split, window) = Arena::split_down(&self.arena, owned, r0 + r1);
            split_claims.push(split);
            batch.windows.push(DownUnit {
                expert,
                head: unit.head[2] as usize,
                window,
            });
        }
        for split in split_claims.iter_mut() {
            planes.push((ExpertKind::Gate, split.expert()));
            planes.push((ExpertKind::Up, split.expert()));
            let (t0, t1) = split.head_mut().split_at_mut(r0);
            targets.push((t0, unit.head[0] as usize));
            targets.push((t1, unit.head[1] as usize));
        }
        let read = self
            .source
            .read_expert_batched(layer, &planes, &mut targets);
        planes.clear();
        batch.planes = planes;
        batch.targets = recycle(targets);
        let base = layer as usize * self.geometry.n_experts as usize;
        if let Err(source) = read {
            // No pointer was handed out and no window reached the reader:
            // every window is taken back, then every claim is cancelled
            // (slots recycled, no ghost residency).
            for (split, w) in batch.split_claims[layer as usize]
                .drain(..)
                .zip(batch.windows.drain(..))
            {
                if let Some(claim) = split.rejoin(w.window) {
                    claim.cancel();
                }
            }
            for (_, _, claim) in fills.drain(..) {
                self.settle_fill(true, unit, claim, false);
            }
            self.stats.source_errors.fetch_add(1, Ordering::Relaxed);
            return Err(PoolError::Source { source });
        }
        for (expert, slot, claim) in fills.drain(..) {
            self.settle_fill(true, unit, claim, true);
            self.stats.demand_claims.fetch_add(1, Ordering::Relaxed);
            batch.lease_of[base + expert as usize] = batch.leases.len() as u32;
            batch.leases.push(self.lease(layer, expert, slot));
        }
        for split in &batch.split_claims[layer as usize] {
            batch.lease_of[base + split.expert() as usize] = SPLIT;
            self.stats.split_units.fetch_add(1, Ordering::Relaxed);
        }
        let job = &down.jobs[layer as usize];
        {
            let mut g = job.inner.lock().unwrap_or_else(|e| e.into_inner());
            debug_assert_eq!(g.state, JobState::Idle, "split queued on an unjoined job");
            g.units.append(&mut batch.windows);
            g.state = JobState::Queued;
        }
        down.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(layer);
        down.wake.notify_one();
        Ok(())
    }

    /// The deadline of a join starting now (one per join, shared by every
    /// layer it waits on).
    fn join_deadline(&self) -> std::time::Instant {
        let bound = self
            .down
            .as_deref()
            .map_or(DOWN_JOIN_TIMEOUT, |d| d.join_timeout);
        std::time::Instant::now() + bound
    }

    /// Joins `layer`'s down job (decode thread, under `batch_claims`):
    /// waits until `deadline` for the reader — failing the job at once when
    /// no reader is attached — then takes every window back and settles
    /// the layer's split units: `Done` finishes each claim and leases the
    /// unit in claim order; `Failed` moves the claims to `poisoned` (still
    /// mid-fill and leased, so no slot is recycled under an in-flight
    /// gate/up DMA) and returns [`PoolError::Source`]. Idle: no-op.
    fn join_layer(
        &self,
        layer: u32,
        batch: &mut BatchClaims,
        deadline: std::time::Instant,
    ) -> Result<(), PoolError> {
        let Some(down) = self.down.as_deref() else {
            return Ok(());
        };
        let job = &down.jobs[layer as usize];
        let mut g = job.inner.lock().unwrap_or_else(|e| e.into_inner());
        let landed = loop {
            match g.state {
                JobState::Idle => return Ok(()),
                JobState::Done => break true,
                JobState::Failed => break false,
                JobState::Queued => {}
            }
            // No reader can take the windows: they are still in `units`,
            // so failing here leaves no writer behind.
            if down.readers.load(Ordering::Acquire) == 0 {
                break false;
            }
            let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                return Err(PoolError::DownStalled { layer });
            };
            // Loom models no timed wait (primitives::wait_timeout is a
            // yield there): park on the latch; the attached reader always
            // settles or detaches, and both notify.
            #[cfg(loom)]
            let _ = left;
            #[cfg(loom)]
            {
                g = job.done.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            #[cfg(not(loom))]
            {
                g = super::primitives::wait_timeout(&job.done, g, left);
            }
        };
        g.state = JobState::Idle;
        let base = layer as usize * self.geometry.n_experts as usize;
        let unit = &self.geometry.layers[layer as usize];
        // Each split claim takes its window back (by index) before it can
        // settle; one left unpaired stays leaked rather than recycled.
        let splits = batch.split_claims[layer as usize].drain(..);
        let claims = splits
            .zip(g.units.drain(..))
            .filter_map(|(split, u)| split.rejoin(u.window));
        if landed {
            for claim in claims {
                let (expert, slot) = (claim.expert(), claim.slot());
                claim.finish();
                self.stats
                    .demand_slow_pageins
                    .fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_paged
                    .fetch_add(unit.bytes(), Ordering::Relaxed);
                self.stats.demand_claims.fetch_add(1, Ordering::Relaxed);
                batch.lease_of[base + expert as usize] = batch.leases.len() as u32;
                batch.leases.push(self.lease(layer, expert, slot));
            }
            Ok(())
        } else {
            batch.poisoned.extend(claims);
            self.stats.source_errors.fetch_add(1, Ordering::Relaxed);
            Err(down_failed())
        }
    }

    /// Joins every layer's down job under one shared deadline, stopping at
    /// the first stall (the stalled layer and those after it keep their
    /// claims and slots). A failed down read is not an error here.
    fn join_all(&self, batch: &mut BatchClaims) -> Result<(), PoolError> {
        if self.down.is_none() {
            return Ok(());
        }
        let deadline = self.join_deadline();
        for layer in 0..self.geometry.layers.len() as u32 {
            if let Err(e @ PoolError::DownStalled { .. }) = self.join_layer(layer, batch, deadline)
            {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Cancels every poisoned claim (caller: C drained every DMA from
    /// their slots) and clears their `SPLIT` entries.
    fn cancel_poisoned(&self, batch: &mut BatchClaims) {
        let n = self.geometry.n_experts as usize;
        for claim in batch.poisoned.drain(..) {
            batch.lease_of[claim.layer() as usize * n + claim.expert() as usize] = UNCLAIMED;
            claim.cancel();
        }
    }

    /// Idle settle of the role-major state (decode idle, every DMA drained:
    /// a failed decode, a device-cache reset, teardown): joins every down
    /// job, leases the units whose down landed (released at the next
    /// `end_batch_step*`) and cancels the poisoned ones — which also wakes
    /// any claimer parked on their fill gate. No-op without role-major.
    ///
    /// # Errors
    /// [`PoolError::DownStalled`] when a down read outlived the join bound.
    pub fn settle_split(&self) -> Result<(), PoolError> {
        if self.down.is_none() {
            return Ok(());
        }
        let mut batch = self.batch_claims.lock().unwrap_or_else(|e| e.into_inner());
        let joined = self.join_all(&mut batch);
        self.cancel_poisoned(&mut batch);
        joined
    }

    /// Joins every queued down read (the step-end join point: `copy_done`
    /// runs it before any lease drain or D2H commit). A failed down read
    /// is not an error here — its units stay poisoned until the step end.
    ///
    /// # Errors
    /// [`PoolError::DownStalled`] when a down read outlived the join bound.
    pub fn join_split(&self) -> Result<(), PoolError> {
        if self.down.is_none() {
            return Ok(());
        }
        let mut batch = self.batch_claims.lock().unwrap_or_else(|e| e.into_inner());
        self.join_all(&mut batch)
    }

    /// Phase 2 of [`Self::ensure_batch`]: one batched source read landing
    /// every pending unit's three planes straight in its claimed slot, then
    /// each fill settles. Success leases every unit (in claim order); any
    /// failure cancels every fill of the batch (no ghost residency).
    fn fill_batch<'m>(
        &'m self,
        layer: u32,
        batch: &mut BatchClaims,
        fills: &mut Vec<(u32, u32, FillClaim<'m>)>,
    ) -> Result<(), PoolError> {
        if fills.is_empty() {
            return Ok(());
        }
        let unit = &self.geometry.layers[layer as usize];
        let mut planes = std::mem::take(&mut batch.planes);
        let mut targets: Vec<(&mut [u8], usize)> = recycle(std::mem::take(&mut batch.targets));
        for (expert, _, claim) in fills.iter_mut() {
            planes.extend(ExpertKind::ALL.map(|k| (k, *expert)));
            let (t0, rest) = self
                .arena
                .slot_mut(claim)
                .split_at_mut(unit.region[0] as usize);
            let (t1, t2) = rest.split_at_mut(unit.region[1] as usize);
            targets.push((t0, unit.head[0] as usize));
            targets.push((t1, unit.head[1] as usize));
            targets.push((t2, unit.head[2] as usize));
        }
        let read = self
            .source
            .read_expert_batched(layer, &planes, &mut targets);
        planes.clear();
        batch.planes = planes;
        batch.targets = recycle(targets);
        let base = layer as usize * self.geometry.n_experts as usize;
        for (expert, slot, claim) in fills.drain(..) {
            if read.is_ok() {
                self.settle_fill(true, unit, claim, true);
                self.stats.demand_claims.fetch_add(1, Ordering::Relaxed);
                batch.lease_of[base + expert as usize] = batch.leases.len() as u32;
                batch.leases.push(self.lease(layer, expert, slot));
            } else {
                self.settle_fill(true, unit, claim, false);
            }
        }
        read.map_err(|source| {
            self.stats.source_errors.fetch_add(1, Ordering::Relaxed);
            PoolError::Source { source }
        })
    }

    /// Ends the batched step: releases every lease taken by
    /// [`Self::ensure_batch`] since the previous call and refreshes the
    /// policy snapshot. The caller guarantees the device has consumed every
    /// pointer handed out (the glue's `copy_done`).
    pub fn end_batch_step(&self) {
        let mut batch = self.batch_claims.lock().unwrap_or_else(|e| e.into_inner());
        // Role-major: every down read lands (or fails) before any lease
        // drops; a stalled one keeps its claims (the caller's join already
        // turned a stall fatal).
        let _ = self.join_all(&mut batch);
        self.cancel_poisoned(&mut batch);
        batch.next_step(self);
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
        let _ = self.join_all(&mut batch);
        self.cancel_poisoned(&mut batch);
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
        // Role-major: a unit with a down read queued is mid-fill, so
        // claiming it below would park the decode thread on its own fill
        // gate. Join first, and hold `batch_claims` across the batch
        // (decode-thread-only, uncontended; lock order batch_claims →
        // d2h_pending → shard) so no unit turns SPLIT mid-batch; a poisoned
        // unit (still SPLIT) has no pool copy to give and is dropped.
        let batch = self.down.is_some().then(|| {
            let mut batch = self.batch_claims.lock().unwrap_or_else(|e| e.into_inner());
            let _ = self.join_layer(layer, &mut batch, self.join_deadline());
            batch
        });
        let base = layer as usize * self.geometry.n_experts as usize;
        // One snapshot for the whole batch: the floor decisions within one
        // compute share the snapshot the H2D ensure path used (refreshed at
        // end_batch_step), so victims rank consistently across the step.
        let snapshot = match &batch {
            Some(batch) => batch.snapshot.clone(),
            None => self
                .batch_claims
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .snapshot
                .clone(),
        };
        // One pending-table lock per batch: every claim pushed under it, so
        // CLOCK cannot recycle a claimed slot mid-batch either.
        let mut pending = self.d2h_pending.lock().unwrap_or_else(|e| e.into_inner());
        for (i, (&expert, &floor)) in experts.iter().zip(floors).enumerate() {
            if let Some(batch) = &batch
                && batch.lease_of.get(base + expert as usize) == Some(&SPLIT)
            {
                out[i] = D2hIncoming::Dropped;
                continue;
            }
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
        let _ = self.join_all(&mut batch);
        self.cancel_poisoned(&mut batch);
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

    /// Leases held by the batched step claims (released at `end_batch_step*`;
    /// test/observability seam).
    pub fn batch_leases(&self) -> usize {
        self.batch_claims
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .leases
            .len()
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

    /// [`Self::publish_snapshot`] returning the replaced snapshot (see
    /// [`SnapshotHandle::swap`]).
    pub fn swap_snapshot(&self, snapshot: Arc<PolicySnapshot>) -> Arc<PolicySnapshot> {
        self.snapshots.swap(snapshot)
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

    /// Test seam: the next role-major down job fails without reading
    /// (no-op without role-major page-in).
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn fail_next_down_read_for_test(&self) {
        if let Some(down) = &self.down {
            down.fail_next.store(true, Ordering::Relaxed);
        }
    }

    /// Claimed units of `layer` (filled or mid-fill), one lock-free load:
    /// Σ[`Self::is_resident`] over the layer's experts between flips.
    pub fn resident_count(&self, layer: u32) -> u32 {
        self.residency.resident_count(layer)
    }

    /// Counts one in-flight demand page-in until the guard drops (every
    /// exit path, including unwinding).
    fn demand_guard(&self, demand: bool) -> Option<DemandGuard<'_>> {
        demand.then(|| {
            self.demand_inflight.fetch_add(1, Ordering::Relaxed);
            DemandGuard(&self.demand_inflight)
        })
    }

    /// Fraction of pool slots holding a unit (one free-list lock; called
    /// at the metrics cadence, never per expert).
    pub fn fill(&self) -> f64 {
        let total = self.arena.slot_count();
        let free = self.residency.free_slot_count();
        if total == 0 {
            0.0
        } else {
            1.0 - free as f64 / f64::from(total)
        }
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
                    self.note_eviction(&evicted.victim);
                    let slot = evicted.slot;
                    debug_assert_eq!(slot, evicted.victim.slot);
                    self.page_in(demand, layer, expert, evicted.claim)?;
                    slot
                }
                Err(ClaimError::Busy) => return Err(PoolError::Busy),
            };
        Ok(self.lease(layer, expert, slot))
    }

    /// The lease object for a claim grant on `slot` (the grant already
    /// counted the lease in the residency shard).
    fn lease(&self, layer: u32, expert: u32, slot: u32) -> Lease {
        Lease {
            map: self.residency.clone(),
            _arena: self.arena.clone(),
            layer,
            expert,
            slot,
        }
    }

    /// Eviction counters for a claim that displaced `victim`.
    fn note_eviction(&self, victim: &super::residency::Victim) {
        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        if victim.prefetched {
            let unit = &self.geometry.layers[victim.layer as usize];
            self.stats
                .prefetch_wasted_bytes
                .fetch_add(unit.bytes(), Ordering::Relaxed);
        }
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

    /// Leases the first `ceil(frac × n_resident)` RESIDENT units among `ids`
    /// (in `ids` order) under one shard lock of `layer`, one lease per unit,
    /// held by `batch` (CPU expert path: never pages in or evicts). Writes
    /// each taken id to `out_ids[i]` and its gate/up/down plane pointers to
    /// `out_ptrs3[3i..3i + 3]`; returns the count taken. Takes nothing for a
    /// layer out of range, `frac` outside `(0, 1]`, or a batch of another
    /// pool; takes fewer when `batch`, `out_ids` or `out_ptrs3` is short.
    pub fn lease_resident_take(
        &self,
        layer: u32,
        ids: &[u32],
        frac: f32,
        out_ids: &mut [i32],
        out_ptrs3: &mut [*const u8],
        batch: &mut LeaseBatch,
    ) -> usize {
        if (layer as usize) >= self.geometry.layers.len()
            || !Arc::ptr_eq(&batch.map, &self.residency)
        {
            return 0;
        }
        let unit = &self.geometry.layers[layer as usize];
        let cap = (batch.keys.capacity() - batch.keys.len())
            .min(out_ids.len())
            .min(out_ptrs3.len() / 3);
        let mut n = 0;
        self.residency
            .lease_take(layer, ids, frac, cap, |expert, slot| {
                batch.keys.push((layer, expert));
                out_ids[n] = expert as i32;
                for (k, p) in out_ptrs3[3 * n..3 * n + 3].iter_mut().enumerate() {
                    let (off, _) = unit.plane(ExpertKind::ALL[k]);
                    *p = self.arena.slot_ptr_at(slot, off).cast_const();
                }
                n += 1;
            })
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
        self.settle_fill(demand, unit, claim, result.is_ok());
        result.map_err(|source| {
            self.stats.source_errors.fetch_add(1, Ordering::Relaxed);
            PoolError::Source { source }
        })
    }

    /// Settles one fill after its read: `ok` opens the fill gate and counts
    /// the page-in; otherwise the claim is cancelled — any plane failure
    /// aborts the whole unit (no ghost residency, the slot goes back to its
    /// class's free list).
    fn settle_fill(&self, demand: bool, unit: &UnitLayout, claim: FillClaim<'_>, ok: bool) {
        if ok {
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
        } else {
            claim.cancel();
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
            if let Some(n) = errors.get_mut(&k)
                && *n > 0
            {
                *n -= 1;
                return Err(io::Error::other("injected source failure"));
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
        stride: [usize; 3],
        /// Fill the batch's targets last-to-first (I/O completion order
        /// opposite to claim order).
        reverse: bool,
        fail_down_times: std::sync::atomic::AtomicUsize,
        calls: Mutex<Vec<(ExpertKind, u32, usize, usize)>>,
        batches: std::sync::atomic::AtomicUsize,
    }

    #[cfg(not(loom))]
    impl BatchedSource {
        fn new(stride: [usize; 3], fail_down_times: usize, reverse: bool) -> Self {
            Self {
                stride,
                reverse,
                fail_down_times: std::sync::atomic::AtomicUsize::new(fail_down_times),
                calls: Mutex::new(Vec::new()),
                batches: std::sync::atomic::AtomicUsize::new(0),
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
            self.batches
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut order: Vec<_> = batch.iter().zip(targets.iter_mut()).collect();
            if self.reverse {
                order.reverse();
            }
            for (&(kind, expert), (into, head)) in order {
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
                let stride = self.stride[kind.ordinal()];
                into[*head..*head + stride].fill(self.plane_byte(kind, expert));
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
        batched_pool_slots(layout, fail_down_times, 2)
    }

    #[cfg(not(loom))]
    fn batched_pool_slots(
        layout: UnitLayout,
        fail_down_times: usize,
        slots: u32,
    ) -> (Arc<ExpertPool>, Arc<BatchedSource>) {
        batched_pool_order(layout, fail_down_times, slots, false)
    }

    #[cfg(not(loom))]
    fn batched_pool_order(
        layout: UnitLayout,
        fail_down_times: usize,
        slots: u32,
        reverse: bool,
    ) -> (Arc<ExpertPool>, Arc<BatchedSource>) {
        let arena = Arena::new(&[SlotClass {
            slot_size: layout.region.iter().sum::<u64>() as usize,
            slots,
        }])
        .unwrap();
        let src = Arc::new(BatchedSource::new(
            layout.stride.map(|s| s as usize),
            fail_down_times,
            reverse,
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
    fn ensure_batch_submits_a_layers_misses_as_one_read() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, src) = batched_pool_slots(layout, 0, 4);
        let misses = [
            (ExpertKind::Gate, 5),
            (ExpertKind::Up, 2),
            (ExpertKind::Down, 5),
            (ExpertKind::Gate, 7),
        ];
        let mut out = [std::ptr::null(); 4];
        pool.ensure_batch(0, &misses, &mut out).unwrap();
        assert_eq!(
            src.batches.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "one drain for the whole layer"
        );
        let experts: Vec<u32> = src.calls.lock().unwrap().iter().map(|c| c.1).collect();
        assert_eq!(experts, [5, 5, 5, 2, 2, 2, 7, 7, 7], "claim (miss) order");
        {
            let b = pool.batch_claims.lock().unwrap();
            for (&(kind, expert), &p) in misses.iter().zip(&out) {
                let lease = b.leases.iter().find(|l| l.expert == expert).unwrap();
                let plane = pool.plane_bytes(0, lease, kind);
                assert_eq!(plane.as_ptr(), p);
                assert!(plane.iter().all(|&x| x == plane_byte(kind, expert)));
            }
        }
        assert_eq!(pool.stats.demand_slow_pageins.load(Ordering::Relaxed), 3);
        assert_eq!(pool.stats.demand_claims.load(Ordering::Relaxed), 3);
        // A repeat within the step reads nothing.
        pool.ensure_batch(0, &misses, &mut out).unwrap();
        assert_eq!(src.batches.load(std::sync::atomic::Ordering::SeqCst), 1);
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_slots_follow_miss_order_not_io_order() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let misses = [
            (ExpertKind::Gate, 6),
            (ExpertKind::Up, 1),
            (ExpertKind::Down, 4),
            (ExpertKind::Down, 6),
        ];
        let slots_of = |reverse: bool| {
            let (pool, src) = batched_pool_order(layout, 0, 4, reverse);
            let mut out = [std::ptr::null(); 4];
            pool.ensure_batch(0, &misses, &mut out).unwrap();
            let experts: Vec<u32> = src.calls.lock().unwrap().iter().map(|c| c.1).collect();
            let want = [6, 6, 6, 1, 1, 1, 4, 4, 4];
            if reverse {
                assert!(experts.iter().eq(want.iter().rev()), "reverse I/O order");
            } else {
                assert_eq!(experts, want);
            }
            let b = pool.batch_claims.lock().unwrap();
            for (&(kind, expert), &p) in misses.iter().zip(&out) {
                let lease = b.leases.iter().find(|l| l.expert == expert).unwrap();
                let plane = pool.plane_bytes(0, lease, kind);
                assert_eq!(plane.as_ptr(), p);
                assert!(plane.iter().all(|&x| x == plane_byte(kind, expert)));
            }
            b.leases
                .iter()
                .map(|l| (l.expert, l.slot()))
                .collect::<Vec<_>>()
        };
        let in_order = slots_of(false);
        assert_eq!(in_order, slots_of(true), "slots independent of I/O order");
        assert_eq!(in_order.iter().map(|x| x.0).collect::<Vec<_>>(), [6, 1, 4]);
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_mixed_stride_one_claim_per_unit() {
        let layout = UnitLayout::new([64, 96, 160], [Some(5), None, Some(13)]).unwrap();
        let (pool, src) = batched_pool_slots(layout, 0, 4);
        let misses = [
            (ExpertKind::Down, 3),
            (ExpertKind::Gate, 2),
            (ExpertKind::Gate, 3),
            (ExpertKind::Up, 3),
            (ExpertKind::Down, 3),
        ];
        let mut out = [std::ptr::null(); 5];
        pool.ensure_batch(0, &misses, &mut out).unwrap();
        let calls = src.calls.lock().unwrap().clone();
        let want: Vec<_> = [3, 2]
            .into_iter()
            .flat_map(|e| {
                ExpertKind::ALL.map(|k| {
                    let o = k.ordinal();
                    (k, e, layout.head[o] as usize, layout.region[o] as usize)
                })
            })
            .collect();
        assert_eq!(calls, want, "one claim per unit, every plane at its head");
        assert_eq!(pool.stats.demand_claims.load(Ordering::Relaxed), 2);
        {
            let b = pool.batch_claims.lock().unwrap();
            assert_eq!(b.leases.len(), 2);
            for (&(kind, expert), &p) in misses.iter().zip(&out) {
                let lease = b.leases.iter().find(|l| l.expert == expert).unwrap();
                let plane = pool.plane_bytes(0, lease, kind);
                assert_eq!(plane.len() as u64, layout.stride[kind.ordinal()]);
                assert_eq!(plane.as_ptr(), p);
                assert!(plane.iter().all(|&x| x == plane_byte(kind, expert)));
            }
        }
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_errors_leave_no_pending_entry() {
        let pending = |pool: &ExpertPool| {
            let b = pool.batch_claims.lock().unwrap();
            b.lease_of.contains(&PENDING)
        };
        let layout = UnitLayout::new([64; 3], [None; 3]).unwrap();
        let misses = [
            (ExpertKind::Gate, 1),
            (ExpertKind::Up, 2),
            (ExpertKind::Down, 1),
        ];
        let mut out = [std::ptr::null(); 3];
        let (pool, _) = batched_pool_slots(layout, 1, 4);
        assert!(matches!(
            pool.ensure_batch(0, &misses, &mut out),
            Err(PoolError::Source { .. })
        ));
        assert!(!pending(&pool));
        pool.ensure_batch(0, &misses, &mut out).unwrap();
        assert!(pool.residency.is_resident(0, 1) && pool.residency.is_resident(0, 2));
        pool.end_batch_step();

        let (pool, _) = batched_pool_slots(layout, 0, 2);
        let mut one = [std::ptr::null()];
        pool.ensure_batch(0, &[(ExpertKind::Gate, 0)], &mut one)
            .unwrap();
        assert!(matches!(
            pool.ensure_batch(0, &misses, &mut out),
            Err(PoolError::Busy)
        ));
        assert!(!pending(&pool));
        pool.end_batch_step();
        pool.ensure_batch(0, &misses, &mut out).unwrap();
        assert!(pool.residency.is_resident(0, 1) && pool.residency.is_resident(0, 2));
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_failure_cancels_every_fill_of_the_layer() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, _) = batched_pool_slots(layout, 1, 4);
        let misses = [(ExpertKind::Gate, 1), (ExpertKind::Gate, 2)];
        let mut out = [std::ptr::null(); 2];
        assert!(matches!(
            pool.ensure_batch(0, &misses, &mut out),
            Err(PoolError::Source { .. })
        ));
        assert!(!pool.residency.is_resident(0, 1), "no ghost residency");
        assert!(!pool.residency.is_resident(0, 2), "no ghost residency");
        assert!(pool.batch_claims.lock().unwrap().leases.is_empty());
        assert_eq!(pool.stats.source_errors.load(Ordering::Relaxed), 1);
        assert_eq!(pool.fill(), 0.0, "every slot recycled");
        pool.ensure_batch(0, &misses, &mut out).unwrap();
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_busy_mid_claim_cancels_pending_fills() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, src) = batched_pool_slots(layout, 0, 2);
        let mut one = [std::ptr::null()];
        pool.ensure_batch(0, &[(ExpertKind::Gate, 0)], &mut one)
            .unwrap();
        let mut two = [std::ptr::null(); 2];
        assert!(matches!(
            pool.ensure_batch(0, &[(ExpertKind::Gate, 1), (ExpertKind::Gate, 2)], &mut two),
            Err(PoolError::Busy)
        ));
        assert!(!pool.residency.is_resident(0, 1), "pending fill cancelled");
        assert!(pool.residency.is_resident(0, 0), "earlier lease kept");
        assert_eq!(pool.fill(), 0.5, "cancelled slot back on the free list");
        assert_eq!(src.batches.load(std::sync::atomic::Ordering::SeqCst), 1);
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_with_hands_out_hits_before_the_disk_read() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, src) = batched_pool_slots(layout, 0, 4);
        let mut warm = [std::ptr::null(); 2];
        pool.ensure_batch(
            0,
            &[(ExpertKind::Gate, 1), (ExpertKind::Gate, 3)],
            &mut warm,
        )
        .unwrap();
        pool.end_batch_step();
        let batches = src.batches.load(std::sync::atomic::Ordering::SeqCst);
        let misses = [
            (ExpertKind::Gate, 1),
            (ExpertKind::Up, 2),
            (ExpertKind::Down, 3),
            (ExpertKind::Gate, 5),
        ];
        let mut out = [std::ptr::null(); 4];
        let mut calls = 0;
        pool.ensure_batch_with(0, &misses, &mut out, |seen| {
            calls += 1;
            let hits: Vec<bool> = seen.iter().map(|p| !p.is_null()).collect();
            assert_eq!(hits, [true, false, true, false]);
            assert_eq!(
                src.batches.load(std::sync::atomic::Ordering::SeqCst),
                batches,
                "no disk read before the callback"
            );
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert!(out.iter().all(|p| !p.is_null()));
        let mut again = [std::ptr::null(); 4];
        pool.ensure_batch(0, &misses, &mut again).unwrap();
        assert_eq!(out, again);
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_with_all_hits_calls_back_without_a_read() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, src) = batched_pool_slots(layout, 0, 4);
        let misses = [(ExpertKind::Gate, 1), (ExpertKind::Up, 2)];
        let mut out = [std::ptr::null(); 2];
        pool.ensure_batch(0, &misses, &mut out).unwrap();
        pool.end_batch_step();
        let batches = src.batches.load(std::sync::atomic::Ordering::SeqCst);
        let mut calls = 0;
        pool.ensure_batch_with(0, &misses, &mut out, |seen| {
            calls += 1;
            assert!(seen.iter().all(|p| !p.is_null()));
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(
            src.batches.load(std::sync::atomic::Ordering::SeqCst),
            batches
        );
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_with_all_misses_skips_the_callback() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, _) = batched_pool_slots(layout, 0, 4);
        let mut out = [std::ptr::null(); 2];
        let mut calls = 0;
        pool.ensure_batch_with(
            0,
            &[(ExpertKind::Gate, 1), (ExpertKind::Up, 2)],
            &mut out,
            |_| calls += 1,
        )
        .unwrap();
        assert_eq!(calls, 0);
        assert!(out.iter().all(|p| !p.is_null()));
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_with_busy_skips_the_callback() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, _) = batched_pool_slots(layout, 0, 2);
        let mut one = [std::ptr::null()];
        pool.ensure_batch(0, &[(ExpertKind::Gate, 0)], &mut one)
            .unwrap();
        let mut three = [std::ptr::null(); 3];
        let mut calls = 0;
        let res = pool.ensure_batch_with(
            0,
            &[
                (ExpertKind::Gate, 0),
                (ExpertKind::Gate, 1),
                (ExpertKind::Gate, 2),
            ],
            &mut three,
            |_| calls += 1,
        );
        assert!(matches!(res, Err(PoolError::Busy)));
        assert_eq!(calls, 0);
        pool.end_batch_step();
    }

    #[test]
    #[cfg(not(loom))]
    fn ensure_batch_with_source_failure_after_the_callback() {
        let stride = 64u64;
        let layout = UnitLayout::new([stride; 3], [None; 3]).unwrap();
        let (pool, src) = batched_pool_slots(layout, 0, 4);
        let mut warm = [std::ptr::null()];
        pool.ensure_batch(0, &[(ExpertKind::Gate, 0)], &mut warm)
            .unwrap();
        pool.end_batch_step();
        src.fail_down_times
            .store(1, std::sync::atomic::Ordering::SeqCst);
        let mut out = [std::ptr::null(); 2];
        let mut calls = 0;
        let res = pool.ensure_batch_with(
            0,
            &[(ExpertKind::Gate, 0), (ExpertKind::Gate, 1)],
            &mut out,
            |_| calls += 1,
        );
        assert!(matches!(res, Err(PoolError::Source { .. })));
        assert_eq!(calls, 1);
        assert!(!pool.residency.is_resident(0, 1), "no ghost residency");
        pool.end_batch_step();
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
            let n = into.len().min(head + 64);
            into[head..n].fill(0x10 * (kind.ordinal() as u8 + 1) + expert as u8);
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

    fn resident_oracle(pool: &ExpertPool) -> u32 {
        (0..N_EXPERTS).filter(|&e| pool.is_resident(0, e)).count() as u32
    }

    #[test]
    #[cfg(not(loom))]
    fn resident_count_tracks_fills_cancel_and_eviction() {
        let pool = test_pool(2);
        assert_eq!(pool.resident_count(0), 0);
        pool.ensure_range(true, 0, ExpertKind::Gate, 0, 1, &mut |_| {})
            .unwrap();
        assert_eq!(pool.resident_count(0), 2);
        assert_eq!(pool.resident_count(0), resident_oracle(&pool));
        // Full pool: the next fill evicts one and claims one.
        pool.ensure_range(true, 0, ExpertKind::Gate, 2, 2, &mut |_| {})
            .unwrap();
        assert_eq!(pool.stats.evictions.load(Ordering::Relaxed), 1);
        assert_eq!(pool.resident_count(0), 2);
        assert_eq!(pool.resident_count(0), resident_oracle(&pool));
        // A failed fill cancels its claim (evicting a victim on the way).
        let pool = test_pool_failing(2, (0, ExpertKind::Gate, 2));
        pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut |_| {})
            .unwrap();
        assert!(
            pool.ensure_range(true, 0, ExpertKind::Gate, 2, 2, &mut |_| {})
                .is_err()
        );
        assert_eq!(pool.resident_count(0), 1);
        assert_eq!(pool.resident_count(0), resident_oracle(&pool));
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

    /// Pool with experts 1, 3, 5, 6 resident and unleased.
    fn take_pool() -> Arc<ExpertPool> {
        let pool = test_pool(4);
        for e in [1, 3, 5, 6] {
            pool.ensure_unit(true, 0, e).unwrap();
        }
        pool
    }

    #[test]
    #[cfg(not(loom))]
    fn resident_take_leases_a_leading_fraction_in_id_order() {
        let pool = take_pool();
        let mut batch = LeaseBatch::new(&pool, 8);
        let (mut ids, mut ptrs) = ([-1i32; 8], [std::ptr::null(); 24]);
        // 4 of the 6 ids are resident (0 and 2 are not): ceil(0.5 × 4) = 2.
        let n =
            pool.lease_resident_take(0, &[6, 0, 3, 2, 5, 1], 0.5, &mut ids, &mut ptrs, &mut batch);
        assert_eq!(n, 2);
        assert_eq!(&ids[..2], &[6, 3]);
        for (i, &e) in ids[..2].iter().enumerate() {
            for (k, &kind) in ExpertKind::ALL.iter().enumerate() {
                let (ptr, _lease) = pool.lease_resident(0, kind, e as u32).unwrap();
                assert_eq!(ptrs[3 * i + k], ptr);
            }
        }
        assert_eq!(batch.len(), 2);
        assert_eq!(pool.outstanding_leases(0, 6), 1);
        assert_eq!(pool.outstanding_leases(0, 3), 1);
        assert_eq!(pool.outstanding_leases(0, 5), 0);
        batch.release_all();
        assert!(batch.is_empty());
        for e in [1, 3, 5, 6] {
            assert_eq!(pool.outstanding_leases(0, e), 0);
        }
    }

    #[test]
    #[cfg(not(loom))]
    fn resident_take_rejects_bad_frac_layer_and_capacity() {
        let pool = take_pool();
        let mut batch = LeaseBatch::new(&pool, 2);
        let (mut ids, mut ptrs) = ([-1i32; 8], [std::ptr::null(); 24]);
        for frac in [0.0, -1.0, 1.5, f32::NAN] {
            assert_eq!(
                pool.lease_resident_take(0, &[1, 3], frac, &mut ids, &mut ptrs, &mut batch),
                0
            );
        }
        assert_eq!(
            pool.lease_resident_take(1, &[1, 3], 1.0, &mut ids, &mut ptrs, &mut batch),
            0
        );
        // Out-of-range and repeated ids are skipped; capacity 2 caps the take.
        let n =
            pool.lease_resident_take(0, &[99, 1, 1, 3, 5], 1.0, &mut ids, &mut ptrs, &mut batch);
        assert_eq!(n, 2);
        assert_eq!(&ids[..2], &[1, 3]);
        assert_eq!(pool.outstanding_leases(0, 1), 1);
        assert_eq!(
            pool.lease_resident_take(0, &[5], 1.0, &mut ids, &mut ptrs, &mut batch),
            0
        );
    }

    #[test]
    #[cfg(not(loom))]
    fn released_batch_lets_exclusive_release_free_the_slots() {
        let pool = take_pool();
        let mut out = [std::ptr::null(); 1];
        pool.ensure_batch(0, &[(ExpertKind::Gate, 3)], &mut out)
            .unwrap();
        let mut batch = LeaseBatch::new(&pool, 8);
        let (mut ids, mut ptrs) = ([-1i32; 8], [std::ptr::null(); 24]);
        assert_eq!(
            pool.lease_resident_take(0, &[3], 1.0, &mut ids, &mut ptrs, &mut batch),
            1
        );
        assert_eq!(pool.outstanding_leases(0, 3), 2);
        batch.release_all();
        pool.end_batch_step_exclusive();
        assert!(
            !pool.is_resident(0, 3),
            "exclusive release blocked by a CPU lease"
        );
    }

    #[test]
    #[cfg(not(loom))]
    fn dropping_a_batch_releases_its_leases() {
        let pool = take_pool();
        let mut batch = LeaseBatch::new(&pool, 8);
        let (mut ids, mut ptrs) = ([-1i32; 8], [std::ptr::null(); 24]);
        assert_eq!(
            pool.lease_resident_take(0, &[1, 5], 1.0, &mut ids, &mut ptrs, &mut batch),
            2
        );
        drop(batch);
        assert_eq!(pool.outstanding_leases(0, 1), 0);
        assert_eq!(pool.outstanding_leases(0, 5), 0);
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

/// Role-major page-in (§3.4 P3) over a stub source whose down-only drains
/// block on a per-call latch. Channels with `try_recv` assert ordering;
/// every wait has a deadline — no sleeps.
#[cfg(all(test, not(loom)))]
mod role_major_tests {
    use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
    use std::time::Duration;

    use super::super::primitives::arc_dyn;
    use super::*;
    use crate::pool::SlotClass;

    const N: u32 = 8;
    const STRIDE: usize = 64;
    const DEADLINE: Duration = Duration::from_secs(10);

    fn byte(kind: ExpertKind, expert: u32) -> u8 {
        0x10 * (kind.ordinal() as u8 + 1) + expert as u8
    }

    /// What a latched down drain does once released.
    enum Release {
        Land,
        Fail,
        Panic,
    }

    /// Writes plane-identifying bytes; a drain of only down planes reports
    /// its experts on `started`, then waits for one `Release`.
    struct LatchSource {
        started: std::sync::Mutex<Sender<Vec<u32>>>,
        release: std::sync::Mutex<Receiver<Release>>,
    }

    impl ExpertSource for LatchSource {
        fn read_expert(
            &self,
            _layer: u32,
            kind: ExpertKind,
            expert: u32,
            into: &mut [u8],
            head: usize,
        ) -> std::io::Result<()> {
            into[head..head + STRIDE].fill(byte(kind, expert));
            Ok(())
        }

        fn plane(&self, _: u32, _: ExpertKind, _: u32) -> Option<(u64, u64)> {
            None
        }

        fn read_expert_batched(
            &self,
            layer: u32,
            batch: &[(ExpertKind, u32)],
            targets: &mut [(&mut [u8], usize)],
        ) -> std::io::Result<()> {
            if !batch.is_empty() && batch.iter().all(|&(k, _)| k == ExpertKind::Down) {
                let experts = batch.iter().map(|&(_, e)| e).collect();
                let _ = self.started.lock().unwrap().send(experts);
                let rel = self.release.lock().unwrap().recv_timeout(DEADLINE);
                match rel {
                    Ok(Release::Land) => {}
                    Ok(Release::Fail) => return Err(std::io::Error::other("down failed")),
                    Ok(Release::Panic) => panic!("down read panicked"),
                    Err(_) => return Err(std::io::Error::other("latch never released")),
                }
            }
            for (&(kind, expert), (into, head)) in batch.iter().zip(targets.iter_mut()) {
                self.read_expert(layer, kind, expert, into, *head)?;
            }
            Ok(())
        }
    }

    struct Rig {
        pool: Arc<ExpertPool>,
        reader: Option<DownReaderHandle>,
        started: Receiver<Vec<u32>>,
        release: Sender<Release>,
    }

    fn rig(slots: u32) -> Rig {
        rig_bounded(slots, DOWN_JOIN_TIMEOUT, true)
    }

    /// Two layers, `slots` unit slots, a join bound of `bound`; `reader`
    /// spawns the down reader.
    fn rig_bounded(slots: u32, bound: Duration, reader: bool) -> Rig {
        let (started_tx, started) = channel();
        let (release, release_rx) = channel();
        let src = LatchSource {
            started: std::sync::Mutex::new(started_tx),
            release: std::sync::Mutex::new(release_rx),
        };
        let arena = Arena::new(&[SlotClass {
            slot_size: 3 * STRIDE,
            slots,
        }])
        .unwrap();
        let unit = UnitLayout::new([STRIDE as u64; 3], [None; 3]).unwrap();
        let pool = ExpertPool::try_new_role_major_bounded(
            arena,
            arc_dyn(std::sync::Arc::new(src)),
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            PoolGeometry {
                layers: vec![unit, unit],
                n_experts: N,
                expert_bytes: 0,
            },
            vec![0, 0],
            4,
            bound,
        )
        .unwrap();
        let reader = reader.then(|| {
            spawn_down_reader(&pool)
                .expect("spawn")
                .expect("role-major reader")
        });
        Rig {
            pool,
            reader,
            started,
            release,
        }
    }

    fn ensure(pool: &ExpertPool, misses: &[(ExpertKind, u32)]) -> Result<Vec<usize>, PoolError> {
        ensure_at(pool, 0, misses)
    }

    fn ensure_at(
        pool: &ExpertPool,
        layer: u32,
        misses: &[(ExpertKind, u32)],
    ) -> Result<Vec<usize>, PoolError> {
        let mut out = vec![std::ptr::null(); misses.len()];
        pool.ensure_batch(layer, misses, &mut out)?;
        Ok(out.into_iter().map(|p| p as usize).collect())
    }

    /// Runs `f` on a thread; its result arrives on the returned channel.
    fn spawn_rx<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Receiver<T> {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx
    }

    /// The STRIDE bytes at a pool plane pointer handed out by ensure.
    fn plane_at(pool: &ExpertPool, ptr: usize) -> Vec<u8> {
        pool.arena.bytes_at_for_test(ptr as *const u8, STRIDE)
    }

    #[test]
    fn gate_up_pointers_are_returned_before_down_lands() {
        let r = rig(4);
        let out = ensure(&r.pool, &[(ExpertKind::Gate, 2), (ExpertKind::Up, 2)]).unwrap();
        assert_eq!(r.started.recv_timeout(DEADLINE).unwrap(), vec![2]);
        assert_eq!(
            plane_at(&r.pool, out[0]),
            vec![byte(ExpertKind::Gate, 2); STRIDE]
        );
        assert_eq!(
            plane_at(&r.pool, out[1]),
            vec![byte(ExpertKind::Up, 2); STRIDE]
        );
        assert_eq!(r.pool.stats.split_units.load(Ordering::Relaxed), 1);
        assert_eq!(r.pool.batch_leases(), 0, "no batch lease before the join");
        assert_eq!(r.pool.outstanding_leases(0, 2), 1, "the claim's lease");
        r.release.send(Release::Land).unwrap();
        // The down segment's ensure joins and hands out the landed plane.
        let down = ensure(&r.pool, &[(ExpertKind::Down, 2)]).unwrap();
        assert_eq!(
            plane_at(&r.pool, down[0]),
            vec![byte(ExpertKind::Down, 2); STRIDE]
        );
        assert_eq!(down[0] - out[0], 2 * STRIDE, "one slot per unit");
        r.pool.end_batch_step();
        assert!(r.pool.is_resident(0, 2));
        assert_eq!(r.pool.outstanding_leases(0, 2), 0);
    }

    #[test]
    fn down_failure_errors_at_join_and_holds_the_slot_until_step_end() {
        let r = rig(4);
        ensure(&r.pool, &[(ExpertKind::Gate, 1)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        r.release.send(Release::Fail).unwrap();
        assert!(matches!(
            ensure(&r.pool, &[(ExpertKind::Down, 1)]),
            Err(PoolError::Source { .. })
        ));
        // Poisoned: still mid-fill and leased — the slot cannot be recycled
        // while C may read its gate/up bytes.
        assert!(r.pool.is_resident(0, 1));
        assert_eq!(r.pool.outstanding_leases(0, 1), 1);
        assert_eq!(r.pool.resident_count(0), 1);
        r.pool.end_batch_step();
        assert!(!r.pool.is_resident(0, 1), "no ghost residency");
        assert_eq!(r.pool.outstanding_leases(0, 1), 0);
        assert_eq!(r.pool.resident_count(0), 0);
        assert_eq!(r.pool.fill(), 0.0, "slot back on the free list");
    }

    #[test]
    fn slots_follow_miss_order() {
        let r = rig(4);
        let out = ensure(
            &r.pool,
            &[
                (ExpertKind::Gate, 5),
                (ExpertKind::Gate, 1),
                (ExpertKind::Gate, 3),
            ],
        )
        .unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        assert!(
            out[0] < out[1] && out[1] < out[2],
            "slots in miss order: {out:?}"
        );
        r.release.send(Release::Land).unwrap();
        r.pool.end_batch_step();
    }

    #[test]
    fn end_batch_step_waits_for_the_down_read() {
        let r = rig(4);
        ensure(&r.pool, &[(ExpertKind::Gate, 4)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        let pool = r.pool.clone();
        let ended = spawn_rx(move || pool.end_batch_step());
        // The step end is blocked on the latch (the reader holds it).
        assert!(matches!(ended.try_recv(), Err(TryRecvError::Empty)));
        r.release.send(Release::Land).unwrap();
        ended.recv_timeout(DEADLINE).expect("end_batch_step joined");
        assert!(r.pool.is_resident(0, 4));
        assert_eq!(r.pool.outstanding_leases(0, 4), 0);
    }

    #[test]
    fn concurrent_claim_on_a_split_unit_parks_until_the_join() {
        let r = rig(4);
        ensure(&r.pool, &[(ExpertKind::Up, 6)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        let pool = r.pool.clone();
        let claimed = spawn_rx(move || pool.ensure_unit(false, 0, 6));
        assert!(
            matches!(claimed.try_recv(), Err(TryRecvError::Empty)),
            "a claim on a partially landed unit parks"
        );
        r.release.send(Release::Land).unwrap();
        r.pool.join_split().unwrap();
        claimed
            .recv_timeout(DEADLINE)
            .expect("parked claim woke")
            .unwrap();
        r.pool.end_batch_step();
        assert_eq!(r.pool.outstanding_leases(0, 6), 0);
    }

    #[test]
    fn second_handoff_on_a_layer_joins_the_first() {
        let r = rig(4);
        ensure(&r.pool, &[(ExpertKind::Gate, 1)]).unwrap();
        assert_eq!(r.started.recv_timeout(DEADLINE).unwrap(), vec![1]);
        let pool = r.pool.clone();
        let second = spawn_rx(move || ensure(&pool, &[(ExpertKind::Gate, 2)]).map(|_| ()));
        assert!(matches!(second.try_recv(), Err(TryRecvError::Empty)));
        r.release.send(Release::Land).unwrap();
        second.recv_timeout(DEADLINE).unwrap().unwrap();
        assert_eq!(r.started.recv_timeout(DEADLINE).unwrap(), vec![2]);
        assert_eq!(r.pool.batch_leases(), 1, "the first unit joined and leased");
        r.release.send(Release::Land).unwrap();
        r.pool.end_batch_step();
        assert_eq!(r.pool.stats.split_units.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_miss_list_with_down_never_splits() {
        let r = rig(4);
        let out = ensure(
            &r.pool,
            &[
                (ExpertKind::Gate, 3),
                (ExpertKind::Up, 3),
                (ExpertKind::Down, 3),
            ],
        )
        .unwrap();
        assert!(matches!(r.started.try_recv(), Err(TryRecvError::Empty)));
        assert_eq!(r.pool.stats.split_units.load(Ordering::Relaxed), 0);
        assert_eq!(
            plane_at(&r.pool, out[2]),
            vec![byte(ExpertKind::Down, 3); STRIDE]
        );
        r.pool.end_batch_step();
    }

    #[test]
    fn exclusive_step_end_leaves_a_split_unit_vram_only() {
        let r = rig(4);
        ensure(&r.pool, &[(ExpertKind::Gate, 7)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        r.release.send(Release::Land).unwrap();
        r.pool.end_batch_step_exclusive();
        assert!(!r.pool.is_resident(0, 7));
        assert!(r.pool.is_vram(0, 7));
    }

    #[test]
    fn panicking_down_read_fails_the_job() {
        let r = rig(4);
        ensure(&r.pool, &[(ExpertKind::Gate, 0)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        r.release.send(Release::Panic).unwrap();
        assert!(matches!(
            ensure(&r.pool, &[(ExpertKind::Down, 0)]),
            Err(PoolError::Source { .. })
        ));
        r.pool.end_batch_step();
        assert!(!r.pool.is_resident(0, 0));
        // The reader survived the panic: a later split still lands.
        ensure(&r.pool, &[(ExpertKind::Gate, 5)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        r.release.send(Release::Land).unwrap();
        r.pool.end_batch_step();
        assert!(r.pool.is_resident(0, 5));
    }

    #[test]
    fn drop_with_a_job_queued_frees_the_arena() {
        let mut r = rig(4);
        ensure(&r.pool, &[(ExpertKind::Gate, 2)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        let arena = Arc::downgrade(&r.pool.arena);
        r.release.send(Release::Land).unwrap();
        let reader = r.reader.take();
        let dropped = spawn_rx(move || {
            drop(reader);
            drop(r);
        });
        dropped.recv_timeout(DEADLINE).expect("teardown completed");
        assert!(arena.upgrade().is_none(), "arena unmapped");
    }

    #[test]
    fn settle_wakes_a_claimer_parked_on_a_poisoned_unit() {
        let r = rig(4);
        ensure(&r.pool, &[(ExpertKind::Gate, 3)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        r.release.send(Release::Fail).unwrap();
        let pool = r.pool.clone();
        let parked = spawn_rx(move || pool.ensure_unit(false, 0, 3));
        assert!(matches!(parked.try_recv(), Err(TryRecvError::Empty)));
        r.pool.settle_split().unwrap();
        parked
            .recv_timeout(DEADLINE)
            .expect("parked prefetch woke")
            .unwrap();
        assert_eq!(r.pool.outstanding_leases(0, 3), 0);
    }

    /// Double ownership: the claim's lease is handed from the call to the
    /// owned claim to the batch lease exactly once. Plain asserts (no
    /// debug_assert), so a `--release` run checks the same invariant.
    #[test]
    fn split_claims_settle_exactly_once() {
        let r = rig(4);
        for round in 0..3u32 {
            let (a, b) = (round, round + 4);
            ensure(&r.pool, &[(ExpertKind::Gate, a), (ExpertKind::Up, b)]).unwrap();
            r.started.recv_timeout(DEADLINE).unwrap();
            assert_eq!(r.pool.outstanding_leases(0, a), 1);
            assert_eq!(r.pool.outstanding_leases(0, b), 1);
            r.release
                .send(if round == 1 {
                    Release::Fail
                } else {
                    Release::Land
                })
                .unwrap();
            r.pool.end_batch_step();
            let sum: u32 = (0..N).map(|e| u32::from(r.pool.is_resident(0, e))).sum();
            assert_eq!(r.pool.resident_count(0), sum);
            for e in 0..N {
                assert_eq!(r.pool.outstanding_leases(0, e), 0, "lease leaked on {e}");
            }
        }
    }

    /// One join's bound is shared by every layer it waits on: two layers
    /// with a stalled down read yield one `DownStalled` within ~one bound,
    /// not one bound per layer.
    #[test]
    fn stalled_down_read_stalls_once_within_the_bound() {
        // Per-layer bounds would take 2 × BOUND (two layers); the 0.5 ×
        // BOUND margin absorbs scheduler jitter on a loaded box.
        const BOUND: Duration = Duration::from_millis(200);
        let r = rig_bounded(4, BOUND, true);
        ensure_at(&r.pool, 0, &[(ExpertKind::Gate, 1)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        // Layer 1's job queues behind layer 0's stalled read on the one
        // reader thread, so it stalls too.
        ensure_at(&r.pool, 1, &[(ExpertKind::Gate, 2)]).unwrap();
        let t0 = std::time::Instant::now();
        assert!(matches!(
            r.pool.join_split(),
            Err(PoolError::DownStalled { layer: 0 })
        ));
        let took = t0.elapsed();
        assert!(took >= BOUND, "returned before the bound: {took:?}");
        assert!(took < BOUND * 3 / 2, "one bound per layer: {took:?}");
        // The stalled claims keep their slots.
        assert_eq!(r.pool.outstanding_leases(0, 1), 1);
        assert_eq!(r.pool.outstanding_leases(1, 2), 1);
        r.release.send(Release::Land).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        r.release.send(Release::Land).unwrap();
        r.pool.end_batch_step();
        assert!(r.pool.is_resident(0, 1) && r.pool.is_resident(1, 2));
        assert_eq!(r.pool.outstanding_leases(0, 1), 0);
    }

    /// No reader running: the join fails the job at once (no bound wait)
    /// and the unit is cancelled at the step end.
    #[test]
    fn join_without_a_reader_fails_at_once() {
        let r = rig_bounded(4, DOWN_JOIN_TIMEOUT, false);
        ensure(&r.pool, &[(ExpertKind::Gate, 3)]).unwrap();
        let pool = r.pool.clone();
        let joined = spawn_rx(move || ensure(&pool, &[(ExpertKind::Down, 3)]).map(|_| ()));
        assert!(matches!(
            joined.recv_timeout(DEADLINE).expect("join returned"),
            Err(PoolError::Source { .. })
        ));
        r.pool.end_batch_step();
        assert!(!r.pool.is_resident(0, 3));
        assert_eq!(r.pool.outstanding_leases(0, 3), 0);
    }

    /// A poisoned unit keeps its only slot until the step ends: a
    /// concurrent claim on another key finds no slot (Busy), one on the
    /// poisoned key parks; both resolve only after `end_batch_step`.
    #[test]
    fn poisoned_slot_is_not_reusable_before_step_end() {
        let r = rig(1);
        ensure(&r.pool, &[(ExpertKind::Gate, 4)]).unwrap();
        r.started.recv_timeout(DEADLINE).unwrap();
        r.release.send(Release::Fail).unwrap();
        assert!(matches!(
            ensure(&r.pool, &[(ExpertKind::Down, 4)]),
            Err(PoolError::Source { .. })
        ));
        assert_eq!(r.pool.stats.source_errors.load(Ordering::Relaxed), 1);
        assert!(matches!(
            r.pool.ensure_unit(false, 0, 5),
            Err(PoolError::Busy)
        ));
        let pool = r.pool.clone();
        let parked = spawn_rx(move || pool.ensure_unit(false, 0, 4));
        assert!(
            matches!(parked.try_recv(), Err(TryRecvError::Empty)),
            "a claim on the poisoned unit parks"
        );
        assert_eq!(r.pool.outstanding_leases(0, 4), 1, "slot still held");
        r.pool.end_batch_step();
        // Woken by the cancel: it refills the unit itself.
        parked
            .recv_timeout(DEADLINE)
            .expect("parked claim woke")
            .unwrap();
        assert!(r.pool.is_resident(0, 4));
        assert_eq!(r.pool.outstanding_leases(0, 4), 0);
        assert_eq!(r.pool.stats.source_errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn plain_pool_has_no_reader() {
        let arena = Arena::new(&[SlotClass {
            slot_size: 3 * STRIDE,
            slots: 1,
        }])
        .unwrap();
        let (started, _) = channel();
        let (_, release) = channel();
        let pool = ExpertPool::new(
            arena,
            arc_dyn(std::sync::Arc::new(LatchSource {
                started: std::sync::Mutex::new(started),
                release: std::sync::Mutex::new(release),
            })),
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            PoolGeometry {
                layers: vec![UnitLayout::new([STRIDE as u64; 3], [None; 3]).unwrap()],
                n_experts: N,
                expert_bytes: 0,
            },
            vec![0],
        );
        assert!(spawn_down_reader(&pool).expect("spawn").is_none());
        pool.settle_split().unwrap();
    }
}
