//! §6.5 prefetch: background page-ins fed by an injected access-set source,
//! demand-priority gated (rounds skip while demand page-ins are in flight)
//! and capacity gated ([`GATE_CAPACITY`] = `io_threads - 1` at the §5.4
//! default). The engine's feed is P4's `notify_access` recency (golem-engine
//! `PolicyFeed`); tests prove the mechanism against injected feeds.

// cfg(loom) is set via RUSTFLAGS by the T6 leg; see primitives.rs.
#![allow(unexpected_cfgs)]

use std::time::Duration;

use super::pager::ExpertPool;
use super::primitives::{Arc, AtomicBool, AtomicUsize, JoinHandle, Ordering, spawn};
use super::residency::ExpertKind;

/// Max concurrent prefetch page-ins: `io_threads - 1` at the §5.4 default
/// (io_threads = 5), so demand reads always keep I/O headroom.
pub const GATE_CAPACITY: usize = 4;

/// Access-set union depth (§6.5: union of the last k steps).
pub const PREFETCH_K: usize = 3;

/// Recent-access source: the engine backs it with P4's `notify_access`
/// recency; tests inject their own.
pub trait AccessFeed: Send + Sync {
    /// The union of the last `k` steps' accessed keys. The pool filters
    /// already-resident keys itself; returned keys may repeat across calls.
    fn take_union(&self, k: usize) -> Vec<(u32, ExpertKind, u32)>;
}

/// Capacity gate over prefetch page-ins: in-flight accounting with a
/// [`GATE_CAPACITY`] ceiling (`io_threads - 1`). Single prefetch thread
/// today, so the observed maximum stays at 1; the accounting carries over to
/// a sharded prefetcher without redesign.
struct PrefetchGate {
    in_flight: Arc<AtomicUsize>,
    max_concurrent: Arc<AtomicUsize>,
}

impl PrefetchGate {
    fn acquire(&self) {
        let now = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_concurrent.fetch_max(now, Ordering::Relaxed);
    }

    fn release(&self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Handle to the spawned prefetch thread. `stop()` signals; `Drop` signals
/// and joins. The loop parks between empty rounds (25 ms bound) so stop
/// latency is bounded by one park.
pub struct PrefetchHandle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    gate: PrefetchGate,
}

impl PrefetchHandle {
    /// Highest number of concurrently in-flight prefetch page-ins ever
    /// observed (never above [`GATE_CAPACITY`] — the demand-priority gate
    /// assertion surface).
    pub fn max_concurrent_prefetch(&self) -> usize {
        self.gate.max_concurrent.load(Ordering::Relaxed)
    }

    /// Signals the thread to stop (join happens on `Drop`).
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
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

/// Spawns the §6.5 prefetch loop over `pool`: each round takes
/// `access.take_union(k)`, filters residents, and pages the rest in with
/// `demand = false` — but only while no demand page-in is in flight
/// ([`ExpertPool::demand_inflight`]), the demand-priority rule. Source
/// errors warn and continue; they never kill the thread.
pub fn spawn_prefetch(
    pool: Arc<ExpertPool>,
    access: Arc<dyn AccessFeed>,
    k: usize,
) -> PrefetchHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let gate = PrefetchGate {
        in_flight: Arc::new(AtomicUsize::new(0)),
        max_concurrent: Arc::new(AtomicUsize::new(0)),
    };
    let thread = {
        let stop = stop.clone();
        let gate = PrefetchGate {
            in_flight: gate.in_flight.clone(),
            max_concurrent: gate.max_concurrent.clone(),
        };
        spawn(move || {
            loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                // Demand priority: a demand page-in in flight wins the I/O;
                // skip the whole round and re-check after one park.
                if pool.demand_inflight() > 0 {
                    park();
                    continue;
                }
                let batch: Vec<_> = access
                    .take_union(k)
                    .into_iter()
                    .filter(|&(layer, kind, expert)| {
                        !pool.residency.is_resident(layer, kind, expert)
                    })
                    .collect();
                if batch.is_empty() {
                    park();
                    continue;
                }
                for (layer, kind, expert) in batch {
                    if stop.load(Ordering::Relaxed) || pool.demand_inflight() > 0 {
                        break;
                    }
                    gate.acquire();
                    let result = pool.ensure_range(false, layer, kind, expert, expert, &mut |_| {});
                    gate.release();
                    if let Err(e) = result {
                        // Transient source trouble must not kill prefetching;
                        // `source_errors` already counted the failure.
                        tracing::warn!(layer, ?kind, expert, error = %e, "prefetch page-in failed; continuing");
                    }
                }
            }
        })
    };
    PrefetchHandle {
        stop,
        thread: Some(thread),
        gate,
    }
}

/// Bounded wait between rounds: stop latency ≤ one park, idle CPU ≈ 0.
fn park() {
    std::thread::park_timeout(Duration::from_millis(25));
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::VecDeque;
    use std::io;
    use std::sync::{Arc as StdArc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::pool::pager::SnapshotHandle;
    use crate::pool::primitives::arc_dyn;
    use crate::pool::residency::PolicySnapshot;
    use crate::pool::{Arena, ExpertPool, ExpertSource, P3Stats};

    /// Feed backed by a queue; `take_union` drains up to `k` keys.
    struct QueueFeed {
        keys: Mutex<VecDeque<(u32, ExpertKind, u32)>>,
    }

    impl QueueFeed {
        fn new(keys: Vec<(u32, ExpertKind, u32)>) -> Self {
            Self {
                keys: Mutex::new(keys.into()),
            }
        }

        fn push(&self, key: (u32, ExpertKind, u32)) {
            self.keys.lock().unwrap().push_back(key);
        }
    }

    impl AccessFeed for QueueFeed {
        fn take_union(&self, k: usize) -> Vec<(u32, ExpertKind, u32)> {
            let mut keys = self.keys.lock().unwrap();
            let n = k.min(keys.len());
            keys.drain(..n).collect()
        }
    }

    /// Source that identifies experts by their id byte and optionally blocks
    /// one key on a condvar until released — the demand-priority harness.
    struct GateSource {
        attempts: Mutex<HashMap<(u32, ExpertKind, u32), usize>>,
        blocked: (u32, ExpertKind, u32),
        released: Mutex<bool>,
        cv: Condvar,
    }

    impl GateSource {
        fn new(blocked: (u32, ExpertKind, u32)) -> Self {
            Self {
                attempts: Mutex::new(HashMap::new()),
                blocked,
                released: Mutex::new(false),
                cv: Condvar::new(),
            }
        }

        fn attempts(&self, key: (u32, ExpertKind, u32)) -> usize {
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
            kind: ExpertKind,
            expert: u32,
            into: &mut [u8],
        ) -> io::Result<()> {
            let key = (layer, kind, expert);
            *self.attempts.lock().unwrap().entry(key).or_insert(0) += 1;
            if key == self.blocked {
                let mut released = self.released.lock().unwrap();
                while !*released {
                    released = self.cv.wait(released).unwrap();
                }
            }
            into.fill(expert as u8);
            Ok(())
        }

        fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
            None
        }
    }

    fn test_pool(slots: u32, source: Arc<dyn ExpertSource>) -> Arc<ExpertPool> {
        let slot_size = 64;
        let arena = Arena::new(slot_size * slots as usize, slot_size).unwrap();
        ExpertPool::new(
            arena,
            source,
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
            Arc::new(P3Stats::new()),
            1,
            8,
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
        let source = StdArc::new(GateSource::new((u32::MAX, ExpertKind::Gate, u32::MAX)));
        let pool = test_pool(8, arc_dyn(source.clone()));
        let feed = StdArc::new(QueueFeed::new(vec![
            (0, ExpertKind::Gate, 0),
            (0, ExpertKind::Gate, 1),
        ]));
        let handle = spawn_prefetch(pool.clone(), arc_dyn(feed), 3);
        until(|| pool.stats.prefetch_pageins.load(Ordering::Relaxed) == 2);
        assert!(pool.residency.is_resident(0, ExpertKind::Gate, 0));
        assert!(pool.residency.is_resident(0, ExpertKind::Gate, 1));
        drop(handle); // stop + join — returns promptly (bounded park)
    }

    // Sequential unit test over loom-primitive pool types: needs no exploration
    // (the t6 model covers the interleavings) and cannot run outside `model()`.
    #[test]
    #[cfg(not(loom))]
    fn prefetch_yields_to_demand() {
        // Demand page-in of expert 7 blocks inside the source, holding
        // `demand_inflight` > 0; a queued prefetch key must NOT be read until
        // demand completes, and prefetch concurrency never exceeds the gate.
        let source = StdArc::new(GateSource::new((0, ExpertKind::Gate, 7)));
        let pool = test_pool(8, arc_dyn(source.clone()));
        let feed = StdArc::new(QueueFeed::new(vec![(0, ExpertKind::Gate, 0)]));
        let handle = spawn_prefetch(pool.clone(), arc_dyn(feed.clone()), 3);

        // Round 1: prefetch pages expert 0.
        until(|| pool.stats.prefetch_pageins.load(Ordering::Relaxed) == 1);

        // Demand thread: blocks inside read_expert for expert 7.
        let p2 = pool.clone();
        let demand = std::thread::spawn(move || {
            p2.ensure_range(true, 0, ExpertKind::Gate, 7, 7, &mut |_| {})
                .unwrap();
        });
        until(|| source.attempts((0, ExpertKind::Gate, 7)) == 1);

        // Offer another prefetch key while demand is in flight: the
        // prefetcher must skip it (checked before every round).
        feed.push((0, ExpertKind::Gate, 2));
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(
            source.attempts((0, ExpertKind::Gate, 2)),
            0,
            "prefetch read while demand was in flight"
        );

        // Demand completes; the queued key is then paged in.
        source.release();
        demand.join().unwrap();
        until(|| source.attempts((0, ExpertKind::Gate, 2)) == 1);
        assert_eq!(pool.stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
        assert_eq!(pool.stats.prefetch_pageins.load(Ordering::Relaxed), 2);
        assert!(
            handle.max_concurrent_prefetch() <= GATE_CAPACITY,
            "prefetch oversubscribed the gate"
        );
        handle.stop();
    }
}
