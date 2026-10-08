//! §6.5 prefetch: background page-ins fed by an injected access source,
//! demand-priority gated (rounds skip while demand page-ins are in flight).
//! One prefetch thread, so at most one prefetch page-in is ever in flight.
//! The engine's feed is the P5 next-layer prediction queue
//! ([`PredictionQueue`], `--prefetch read`); tests also inject their own.

// cfg(loom) is set via RUSTFLAGS by the T6 leg; see primitives.rs.

use std::collections::VecDeque;
use std::time::Duration;

use super::pager::{ExpertPool, PoolError};
use super::primitives::{Arc, AtomicBool, Condvar, JoinHandle, Mutex, Ordering, spawn};

/// Units to page in: the engine backs it with P5 predictions; tests inject
/// their own.
pub trait AccessFeed: Send + Sync {
    /// Takes every pending `(layer, expert)` unit. The pool filters
    /// already-resident units itself; units may repeat across calls.
    fn take(&self) -> Vec<(u32, u32)>;

    /// Waits up to `timeout` for new units (default: a plain park, so the
    /// stop latency stays bounded).
    fn idle(&self, timeout: Duration) {
        std::thread::park_timeout(timeout);
    }
}

/// Bounded FIFO of predicted units (§6.5 `read`): the decode thread pushes
/// each prediction (one lock, no allocation once warm: capacity is
/// reserved up front), the prefetch thread drains it. Past `cap` the
/// oldest units drop — a prediction for a layer already computed is
/// worthless. A push wakes the prefetch thread at once: one layer of
/// compute is the whole lead time, a 25 ms poll would miss it.
pub struct PredictionQueue {
    keys: Mutex<VecDeque<(u32, u32)>>,
    cv: Condvar,
    cap: usize,
}

impl PredictionQueue {
    /// Empty queue holding at most `cap` units.
    pub fn new(cap: usize) -> Self {
        Self {
            keys: Mutex::new(VecDeque::with_capacity(cap.max(1))),
            cv: Condvar::new(),
            cap: cap.max(1),
        }
    }

    /// Appends one layer's predicted experts, dropping the oldest past `cap`.
    pub fn push(&self, layer: u32, experts: impl IntoIterator<Item = u32>) {
        let mut q = self.keys.lock().unwrap();
        for e in experts {
            if q.len() == self.cap {
                q.pop_front();
            }
            q.push_back((layer, e));
        }
        drop(q);
        self.cv.notify_one();
    }
}

impl AccessFeed for PredictionQueue {
    fn take(&self) -> Vec<(u32, u32)> {
        self.keys.lock().unwrap().drain(..).collect()
    }

    fn idle(&self, timeout: Duration) {
        let q = self.keys.lock().unwrap();
        if q.is_empty() {
            let _ = self.cv.wait_timeout(q, timeout).unwrap();
        }
    }
}

/// Handle to the spawned prefetch thread. `stop()` signals; `Drop` signals
/// and joins. The loop idles between empty rounds (25 ms bound) so stop
/// latency is bounded by one idle.
/// Handle to the spawned prefetch thread. `stop()` signals; `Drop` signals
/// and joins. The loop idles between empty rounds (25 ms bound) so stop
/// latency is bounded by one idle. `pause`/`resume` gate the whole loop
/// (prefill stream, design §7.6.4: stream chunks must not contend for the
/// disk queue or pool slots — decode-path demand keeps priority).
pub struct PrefetchHandle {
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl PrefetchHandle {
    /// Signals the thread to stop (join happens on `Drop`).
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Suspends the prefetch loop (`paused` rounds spin on a short backoff).
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
    }

    /// Resumes the prefetch loop.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }
}

impl Drop for PrefetchHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Spawns the §6.5 prefetch loop over `pool`: each round takes the feed's
/// pending units, filters residents, and pages the rest in with `demand =
/// false` — but only while no demand page-in is in flight
/// ([`ExpertPool::demand_inflight`]), the demand-priority rule. A prefetch
/// that finds no cold victim skips the unit (P17: it never evicts a
/// demand-hot key). Source errors warn and continue; they never kill the
/// thread.
pub fn spawn_prefetch(pool: Arc<ExpertPool>, access: Arc<dyn AccessFeed>) -> PrefetchHandle {
    const IDLE: Duration = Duration::from_millis(25);
    // One unit page-in is ≈ 15.2 MB / 6.5 GB/s ≈ 2.3 ms; re-check well inside it.
    const DEMAND_BACKOFF: Duration = Duration::from_micros(200);
    let stop = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));
    let thread = {
        let stop = stop.clone();
        let paused = paused.clone();
        spawn(move || {
            loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                if paused.load(Ordering::Relaxed) {
                    std::thread::sleep(DEMAND_BACKOFF);
                    continue;
                }
                // Demand priority: a demand page-in in flight wins the I/O;
                // skip the whole round and re-check after one idle.
                if pool.demand_inflight() > 0 {
                    // Not `idle`: it returns at once while work is queued.
                    std::thread::sleep(DEMAND_BACKOFF);
                    continue;
                }
                let batch: Vec<_> = access
                    .take()
                    .into_iter()
                    // Exclusive residency (O10): skip VRAM-resident units —
                    // never re-read from disk while a VRAM copy exists.
                    .filter(|&(layer, expert)| {
                        !pool.residency.is_resident(layer, expert) && !pool.is_vram(layer, expert)
                    })
                    .collect();
                if batch.is_empty() {
                    access.idle(IDLE);
                    continue;
                }
                for (layer, expert) in batch {
                    // Dropping the rest is deliberate: by the time demand
                    // drains, those layers have run and the units are waste.
                    if stop.load(Ordering::Relaxed) || pool.demand_inflight() > 0 {
                        break;
                    }
                    match pool.ensure_unit(false, layer, expert) {
                        Ok(()) => {
                            // The promotion bit may have landed while this
                            // page-in ran (concurrent end_batch_step_exclusive):
                            // drop the just-refilled RAM copy, the VRAM one is
                            // authoritative (exactly-one-tier, O10). No-op
                            // unless `is_vram`.
                            pool.release_if_vram(layer, expert);
                        }
                        Err(PoolError::Busy) => {}
                        // Transient source trouble must not kill prefetching;
                        // `source_errors` already counted the failure.
                        Err(e) => {
                            tracing::warn!(layer, expert, error = %e, "prefetch page-in failed; continuing")
                        }
                    }
                }
            }
        })
    };
    PrefetchHandle {
        stop,
        paused,
        thread: Some(thread),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::VecDeque;
    use std::io;
    use std::sync::{Arc as StdArc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::pool::ExpertKind;
    use crate::pool::primitives::arc_dyn;
    use crate::pool::residency::PolicySnapshot;
    use crate::pool::source::UnitLayout;
    use crate::pool::{Arena, ExpertPool, ExpertSource, P3Stats, SlotClass};
    use crate::pool::{PoolGeometry, pager::SnapshotHandle};

    const N_EXPERTS: u32 = 8;

    /// Feed backed by a queue; `take_union` drains up to `k` units.
    struct QueueFeed {
        keys: Mutex<VecDeque<(u32, u32)>>,
    }

    impl QueueFeed {
        fn new(keys: Vec<(u32, u32)>) -> Self {
            Self {
                keys: Mutex::new(keys.into()),
            }
        }

        fn push(&self, key: (u32, u32)) {
            self.keys.lock().unwrap().push_back(key);
        }
    }

    impl AccessFeed for QueueFeed {
        fn take(&self) -> Vec<(u32, u32)> {
            self.keys.lock().unwrap().drain(..).collect()
        }
    }

    /// Source that blocks one unit on a condvar until released — the
    /// demand-priority harness. Counts read_expert calls per plane.
    struct GateSource {
        attempts: Mutex<HashMap<(u32, u32), usize>>,
        blocked: (u32, u32),
        released: Mutex<bool>,
        cv: Condvar,
    }

    impl GateSource {
        fn new(blocked: (u32, u32)) -> Self {
            Self {
                attempts: Mutex::new(HashMap::new()),
                blocked,
                released: Mutex::new(false),
                cv: Condvar::new(),
            }
        }

        fn attempts(&self, key: (u32, u32)) -> usize {
            *self.attempts.lock().unwrap().get(&key).unwrap_or(&0)
        }

        fn release(&self) {
            *self.released.lock().unwrap() = true;
            self.cv.notify_all();
        }
    }

    impl ExpertSource for GateSource {
        fn read_expert(
            &self,
            layer: u32,
            _kind: ExpertKind,
            expert: u32,
            into: &mut [u8],
            head: usize,
        ) -> io::Result<()> {
            let key = (layer, expert);
            *self.attempts.lock().unwrap().entry(key).or_insert(0) += 1;
            if key == self.blocked {
                let mut released = self.released.lock().unwrap();
                while !*released {
                    released = self.cv.wait(released).unwrap();
                }
            }
            into[head..].fill(expert as u8);
            Ok(())
        }

        fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
            None
        }
    }

    fn test_pool(slots: u32, source: Arc<dyn ExpertSource>) -> Arc<ExpertPool> {
        let stride = 64;
        let arena = Arena::new(&[SlotClass {
            slot_size: 3 * stride as usize,
            slots,
        }])
        .unwrap();
        let geometry = PoolGeometry {
            layers: vec![UnitLayout::new([stride; 3], [None; 3]).unwrap(); 1],
            n_experts: N_EXPERTS,
            expert_bytes: 0,
        };
        ExpertPool::new(
            arena,
            source,
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            geometry,
            vec![0; 1],
        )
    }

    /// Polls `cond` until it holds; fails the test after a bounded deadline
    /// (no fixed sleeps where a condition can be observed).
    fn until(cond: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !cond() {
            assert!(Instant::now() < deadline, "condition not reached in time");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn prefetch_pages_missing_experts_and_stops_on_drop() {
        let source = StdArc::new(GateSource::new((u32::MAX, u32::MAX)));
        let pool = test_pool(8, arc_dyn(source.clone()));
        let feed = StdArc::new(QueueFeed::new(vec![(0, 0), (0, 1)]));
        let handle = spawn_prefetch(pool.clone(), arc_dyn(feed));
        until(|| pool.stats.prefetch_pageins.load(Ordering::Relaxed) == 2);
        assert!(pool.residency.is_resident(0, 0));
        assert!(pool.residency.is_resident(0, 1));
        drop(handle); // stop + join — returns promptly (bounded park)
    }

    #[test]
    #[cfg(not(loom))]
    fn prediction_queue_drops_the_oldest_past_its_cap_and_wakes_the_idler() {
        let q = StdArc::new(PredictionQueue::new(3));
        q.push(0, [1, 2]);
        q.push(1, [3, 4]);
        assert_eq!(q.take(), [(0, 2), (1, 3), (1, 4)]);
        assert!(q.take().is_empty());
        let q2 = q.clone();
        let t0 = Instant::now();
        let idler = std::thread::spawn(move || q2.idle(Duration::from_secs(5)));
        std::thread::sleep(Duration::from_millis(20));
        q.push(2, [5]);
        idler.join().unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "push wakes the idle wait"
        );
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn prefetch_yields_to_demand() {
        // Demand page-in of expert 7 blocks inside the source, holding
        // `demand_inflight` > 0; a queued prefetch unit must NOT be read
        // until demand completes, and prefetch concurrency never exceeds
        // (one prefetch thread, so no concurrency to bound).
        let source = StdArc::new(GateSource::new((0, 7)));
        let pool = test_pool(8, arc_dyn(source.clone()));
        let feed = StdArc::new(QueueFeed::new(vec![(0, 0)]));
        let handle = spawn_prefetch(pool.clone(), arc_dyn(feed.clone()));

        // Round 1: prefetch pages expert 0.
        until(|| pool.stats.prefetch_pageins.load(Ordering::Relaxed) == 1);

        // Demand thread: blocks inside read_expert for expert 7's gate plane.
        let p2 = pool.clone();
        let demand = std::thread::spawn(move || {
            p2.ensure_unit(true, 0, 7).unwrap();
        });
        until(|| source.attempts((0, 7)) == 1);

        // Offer another prefetch unit while demand is in flight: the
        // prefetcher must skip it (checked before every round).
        feed.push((0, 2));
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(
            source.attempts((0, 2)),
            0,
            "prefetch read while demand was in flight"
        );

        // Demand completes; the queued unit is then paged in.
        source.release();
        demand.join().unwrap();
        until(|| pool.stats.prefetch_pageins.load(Ordering::Relaxed) == 2);
        assert_eq!(pool.stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
        assert_eq!(pool.stats.prefetch_pageins.load(Ordering::Relaxed), 2);
        handle.stop();
    }
}
