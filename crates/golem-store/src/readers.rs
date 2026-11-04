//! §5.2 persistent drain readers: per-[`crate::ChunkQueue`] crews of parked
//! reader threads that run one drain's lifetime-erased task, so a warm drain
//! neither spawns a thread nor allocates.
//!
//! Invariants:
//! - [`ReaderPool::run`] returns (or unwinds) only after every worker that
//!   received the task has released it (`done == want`): nothing between
//!   publishing the task and `wait_released` can unwind — the same argument
//!   `std::thread::scope` relies on,
//!   so the erased `'env` borrow never outlives the caller's frame.
//! - A crew is owned exclusively by the drain that popped it from the idle
//!   list; only that owner grows it, and never while its task is live.
//! - Crews grow to the largest drain they served: at most `MAX_CREWS ×
//!   drives × io_depth` threads. Target (2 drives, depth 2): 3 × 2 × 2 = 12
//!   threads, 12 × 2 MiB = 24 MiB reserved virtual stack (resident is touched
//!   pages only); with `--io-threads N` it is 3 × N. On the target every
//!   crew reaches its maximum on first use.
//! - Each crew thread pins to the R-15 cpuset once, at spawn. When all crews
//!   are busy (> `MAX_CREWS` overlapping drains) the drain runs `task(0)` on
//!   the caller thread, unpinned, and bumps `serial_fallbacks`.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::PoisonError;

use sync::{Arc, AtomicU64, Condvar, JoinHandle, Mutex, MutexGuard, Ordering, spawn};

/// std/loom swap: every concurrent type here comes from this module so the
/// loom leg explores the production code paths.
// `cfg(loom)` is set via RUSTFLAGS by the loom leg and the `loom` feature
// links the crate; both are required, so `--cfg loom` alone (e.g. a
// dependent crate's loom leg) still builds on std. The workspace declares
// `loom` in `check-cfg`.
mod sync {
    #[cfg(all(loom, feature = "loom"))]
    pub use loom::sync::atomic::{AtomicU64, Ordering};
    #[cfg(all(loom, feature = "loom"))]
    pub use loom::sync::{Arc, Condvar, Mutex, MutexGuard};
    #[cfg(all(loom, feature = "loom"))]
    pub use loom::thread::{JoinHandle, spawn};

    #[cfg(not(all(loom, feature = "loom")))]
    pub use std::sync::atomic::{AtomicU64, Ordering};
    #[cfg(not(all(loom, feature = "loom")))]
    pub use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    #[cfg(not(all(loom, feature = "loom")))]
    pub use std::thread::{JoinHandle, spawn};
}

/// Overlapping drains served by parked crews; drain number 4+ runs serially
/// on its caller.
pub(crate) const MAX_CREWS: usize = 3;

/// The drain task with its `'env` lifetime erased. Only ever dereferenced by
/// a crew worker between `run` publishing it and that worker's `done += 1`.
#[derive(Clone, Copy)]
struct Erased(*const (dyn Fn(usize) + Sync + 'static));

// SAFETY: the pointee is `Sync` (shared calls from many threads are sound);
// the pointer itself only moves to crew threads while `run` keeps the
// borrowed task alive (see `ReaderPool::run`).
unsafe impl Send for Erased {}

struct CrewState {
    /// Bumped once per published task; workers run each generation once.
    generation: u64,
    task: Option<Erased>,
    /// Workers `0..want` run the current task.
    want: usize,
    /// Workers that released the current task.
    done: usize,
    /// First panic payload of the current task.
    panic: Option<Box<dyn Any + Send>>,
    shutdown: bool,
}

struct Crew {
    state: Mutex<CrewState>,
    start: Condvar,
    done: Condvar,
    /// Grown only by the crew's current owner (see [`ReaderPool::run`]).
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Crew {
    fn lock(&self) -> MutexGuard<'_, CrewState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(all(test, not(loom)))]
fn crew_threads(crew: &Crew) -> usize {
    crew.threads
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .len()
}

/// Grows `crew` to `full` threads; a new thread starts at the crew's current
/// generation, so it never runs a task published before it existed.
fn ensure_threads(crew: &Arc<Crew>, full: usize, cpuset: &Option<crate::sys::CpuSet>) -> usize {
    let mut threads = crew.threads.lock().unwrap_or_else(PoisonError::into_inner);
    while threads.len() < full {
        let idx = threads.len();
        let generation = crew.lock().generation;
        let c = Arc::clone(crew);
        let set = cpuset.clone();
        threads.push(spawn(move || worker(c, idx, generation, set)));
    }
    threads.len()
}

fn worker(crew: Arc<Crew>, idx: usize, mut last_gen: u64, cpuset: Option<crate::sys::CpuSet>) {
    // R-15: pinned once for the thread's lifetime, not per drain.
    if let Some(set) = &cpuset
        && let Err(e) = crate::sys::pin_current_thread(set)
    {
        tracing::warn!("io thread pinning: {e}");
    }
    loop {
        let task = {
            let mut st = crew.lock();
            loop {
                if st.shutdown {
                    return;
                }
                if st.generation != last_gen {
                    last_gen = st.generation;
                    if idx < st.want
                        && let Some(t) = st.task
                    {
                        break t;
                    }
                }
                st = crew.start.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
        };
        // SAFETY: `run` published `task` for this generation and does not
        // return or unwind before `wait_released` sees `done == want`; this
        // worker counts toward `want` and increments `done` only after the
        // call (and its last use of `task`) completes, so the `'env` borrow
        // is alive for the whole call.
        let r = catch_unwind(AssertUnwindSafe(|| unsafe { (*task.0)(idx) }));
        let mut st = crew.lock();
        if let Err(p) = r
            && st.panic.is_none()
        {
            st.panic = Some(p);
        }
        st.done += 1;
        crew.done.notify_all();
    }
}

/// §5.2 reader crews of one [`crate::ChunkQueue`].
pub(crate) struct ReaderPool {
    /// Idle crews; capacity `MAX_CREWS`, so pop/push never allocate.
    idle: Mutex<Vec<Arc<Crew>>>,
    /// Every crew, for shutdown.
    all: Vec<Arc<Crew>>,
    /// R-15: CPUs each crew thread pins itself to at spawn.
    cpuset: Option<crate::sys::CpuSet>,
    /// Drains run serially on their caller because every crew was busy.
    serial_fallbacks: AtomicU64,
    /// Workers handed the last task (test observation of `want`).
    #[cfg(all(test, not(loom)))]
    last_want: std::sync::atomic::AtomicUsize,
}

impl ReaderPool {
    pub fn new(cpuset: Option<crate::sys::CpuSet>) -> Self {
        let all: Vec<Arc<Crew>> = (0..MAX_CREWS)
            .map(|_| {
                Arc::new(Crew {
                    state: Mutex::new(CrewState {
                        generation: 0,
                        task: None,
                        want: 0,
                        done: 0,
                        panic: None,
                        shutdown: false,
                    }),
                    start: Condvar::new(),
                    done: Condvar::new(),
                    threads: Mutex::new(Vec::new()),
                })
            })
            .collect();
        Self {
            idle: Mutex::new(all.iter().map(Arc::clone).collect()),
            all,
            cpuset,
            serial_fallbacks: AtomicU64::new(0),
            #[cfg(all(test, not(loom)))]
            last_want: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn serial_fallbacks(&self) -> u64 {
        self.serial_fallbacks.load(Ordering::Relaxed)
    }

    #[cfg(all(test, not(loom)))]
    pub fn last_want(&self) -> usize {
        self.last_want.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Thread count of the largest crew.
    #[cfg(all(test, not(loom)))]
    pub fn max_crew_threads(&self) -> usize {
        self.all.iter().map(|c| crew_threads(c)).max().unwrap_or(0)
    }

    /// Thread count of the crew the next `run` pops.
    #[cfg(all(test, not(loom)))]
    pub fn top_idle_crew_threads(&self) -> usize {
        let idle = self.idle.lock().unwrap_or_else(PoisonError::into_inner);
        idle.last().map_or(0, |c| crew_threads(c))
    }

    /// Runs `task(0..workers)` on an idle crew grown to `full` threads and
    /// returns once every worker released it; a worker panic resumes on the
    /// caller after all workers checked out. With no idle crew, runs
    /// `task(0)` on the caller (one worker drains everything).
    pub fn run<'env>(
        &self,
        full: usize,
        workers: usize,
        task: &'env (dyn Fn(usize) + Sync + 'env),
    ) {
        let popped = self
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let Some(crew) = popped else {
            self.serial_fallbacks.fetch_add(1, Ordering::Relaxed);
            #[cfg(all(test, not(loom)))]
            self.last_want
                .store(1, std::sync::atomic::Ordering::Relaxed);
            return task(0);
        };
        let lease = Lease { pool: self, crew };
        let len = ensure_threads(&lease.crew, full.max(workers), &self.cpuset);
        debug_assert!(workers <= len);
        let want = workers.min(len).max(1);
        #[cfg(all(test, not(loom)))]
        self.last_want
            .store(want, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: only the lifetime changes (same fat-pointer layout; an `as`
        // cast cannot extend a trait-object lifetime). The pointer is used
        // solely by this crew's workers, and `wait` below does not let `run`
        // return or unwind until all `want` workers released it.
        let erased = Erased(unsafe {
            std::mem::transmute::<
                *const (dyn Fn(usize) + Sync + 'env),
                *const (dyn Fn(usize) + Sync + 'static),
            >(task)
        });
        {
            let mut st = lease.crew.lock();
            st.task = Some(erased);
            st.want = want;
            st.done = 0;
            st.panic = None;
            st.generation = st.generation.wrapping_add(1);
            lease.crew.start.notify_all();
        }
        // Nothing between publishing and this call can unwind (lock and
        // notify do not panic), so waiting directly is unwind-safe: `run`
        // cannot leave while a worker still holds `erased`.
        let panic = wait_released(&lease.crew);
        drop(lease);
        if let Some(p) = panic {
            resume_unwind(p);
        }
    }
}

/// Blocks until every worker released the published task, retracts it, and
/// takes the first worker panic.
fn wait_released(crew: &Crew) -> Option<Box<dyn Any + Send>> {
    let mut st = crew.lock();
    while st.done < st.want {
        st = crew.done.wait(st).unwrap_or_else(PoisonError::into_inner);
    }
    st.task = None;
    st.panic.take()
}

/// Exclusive ownership of a popped crew; returns it to the idle list.
struct Lease<'p> {
    pool: &'p ReaderPool,
    crew: Arc<Crew>,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.pool
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::clone(&self.crew));
    }
}

impl Drop for ReaderPool {
    fn drop(&mut self) {
        for crew in &self.all {
            crew.lock().shutdown = true;
            crew.start.notify_all();
        }
        for crew in &self.all {
            let threads =
                std::mem::take(&mut *crew.threads.lock().unwrap_or_else(PoisonError::into_inner));
            for t in threads {
                let _ = t.join();
            }
        }
    }
}

// Loom models only; the std crew paths are covered by `queue` tests.
#[cfg(all(test, loom, feature = "loom"))]
mod tests {
    use super::*;

    fn model<F: Fn() + Send + Sync + 'static>(f: F) {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound.get_or_insert(2);
        builder.check(f);
    }

    use loom::cell::UnsafeCell;
    /// Caller-owned cells written by each worker and read by the caller
    /// after `run`: any worker access after release is a data race loom
    /// reports.
    struct Cells([UnsafeCell<usize>; 2]);
    // SAFETY: worker `i` touches only cell `i`, and the caller reads only
    // between runs (the property under test).
    unsafe impl Sync for Cells {}

    #[test]
    fn loom_crew_handoff() {
        model(|| {
            let pool = ReaderPool::new(None);
            let cells = Cells([UnsafeCell::new(0), UnsafeCell::new(0)]);
            for round in 1..=2 {
                // SAFETY: worker `i` is the only writer of cell `i` within a
                // run (see `Cells`).
                let cells = &cells;
                let task = move |i: usize| cells.0[i].with_mut(|p| unsafe { *p += 1 });
                pool.run(2, 2, &task);
                for c in &cells.0 {
                    // SAFETY: `run` returned, so no worker holds the task.
                    assert_eq!(c.with(|p| unsafe { *p }), round);
                }
            }
        });
    }

    /// A worker panic resumes on the caller only after the other worker
    /// released the task; with `want < crew_len` the idle worker skips the
    /// generation and the crew serves the next run.
    #[test]
    fn loom_worker_panic_and_idle_worker() {
        model(|| {
            let pool = ReaderPool::new(None);
            let cells = Cells([UnsafeCell::new(0), UnsafeCell::new(0)]);
            let cells = &cells;
            let task = move |i: usize| {
                // SAFETY: worker `i` is the only writer of cell `i`.
                cells.0[i].with_mut(|p| unsafe { *p += 1 });
                if i == 0 {
                    panic!("injected worker panic");
                }
            };
            let r = catch_unwind(AssertUnwindSafe(|| pool.run(2, 2, &task)));
            assert!(r.is_err(), "worker panic resumes on the caller");
            for c in &cells.0 {
                // SAFETY: `run` unwound only after both workers released.
                assert_eq!(c.with(|p| unsafe { *p }), 1);
            }
            // Crew of 2, one worker wanted: worker 1 skips this generation.
            // SAFETY: only worker 0 runs, sole writer of cell 0.
            let task = move |i: usize| cells.0[i].with_mut(|p| unsafe { *p += 1 });
            pool.run(2, 1, &task);
            // SAFETY: `run` returned, so no worker holds the task.
            assert_eq!(cells.0[0].with(|p| unsafe { *p }), 2);
            assert_eq!(cells.0[1].with(|p| unsafe { *p }), 1);
        });
    }
}
