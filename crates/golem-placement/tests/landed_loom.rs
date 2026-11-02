//! §12.2 concurrency leg for the device-side landed wait: model-check the
//! request ring handoff between the decode thread (publisher) and the
//! controller (server), the lease-release ordering, and the slot-table lock
//! against every device-synchronizing slab (re)allocation. Runs in both
//! modes: std executes one pass, loom explores the interleavings.
//!
//! Protocol modeled (ggml-backend.cpp `golem_dw_setup` / `golem_device_wait_serve`
//! / `golem_dw_quiesce`, landed.rs `Poller`):
//! - publisher: write the layer setup, release-store host_seq, set active,
//!   release-store the request word (the device's publish), and count it;
//! - server: poll; for a new request, acquire host_seq == seq, read the
//!   setup, take a lease, mark landed, count the serve;
//! - decode thread: before `copy_done` releases the step's leases, wait until
//!   served == published.

use golem_placement::pool::landed::{Poller, Ring};
use golem_placement::pool::primitives::{
    self, Arc, AtomicBool, AtomicU64, AtomicUsize, Condvar, Mutex, Ordering,
};

const N_BLK: usize = 2;

/// The mapped ring + the per-blk decode-thread setup the serve reads.
struct Model {
    active: AtomicU64,
    req: [AtomicU64; N_BLK],
    host_seq: [AtomicU64; N_BLK],
    /// The layer setup (slab entries, K, predictions) written by the decode
    /// thread before the publish; the serve must observe it.
    setup: [AtomicU64; N_BLK],
    landed: [AtomicU64; N_BLK],
    /// Leases taken by serves and not yet released by copy_done.
    leases: AtomicUsize,
    served: AtomicUsize,
    /// Set if a serve saw a request whose setup was not yet visible, or
    /// served one twice (stale serve).
    violation: AtomicBool,
}

impl Model {
    fn new() -> Self {
        let z = || AtomicU64::new(0);
        Self {
            active: z(),
            req: [z(), z()],
            host_seq: [z(), z()],
            setup: [z(), z()],
            landed: [z(), z()],
            leases: AtomicUsize::new(0),
            served: AtomicUsize::new(0),
            violation: AtomicBool::new(false),
        }
    }

    /// Decode thread: one layer's publish for compute `seq`.
    fn publish(&self, blk: usize, seq: u64) {
        self.setup[blk].store(seq * 10 + blk as u64, Ordering::Relaxed);
        self.host_seq[blk].store(seq, Ordering::Release);
        self.active.store(seq, Ordering::Release);
        self.req[blk].store(seq, Ordering::Release);
    }
}

/// The controller's owned handle on the shared model (it alone serves).
struct Server(Arc<Model>);

impl Ring for Server {
    fn n_blk(&self) -> usize {
        N_BLK
    }
    fn active(&self) -> u32 {
        self.0.active()
    }
    fn request(&self, blk: usize) -> u32 {
        self.0.request(blk)
    }
    fn serve(&mut self, blk: usize, seq: u32) {
        self.0.serve(blk, seq);
    }
}

impl Model {
    fn active(&self) -> u32 {
        self.active.load(Ordering::Acquire) as u32
    }
    fn request(&self, blk: usize) -> u32 {
        self.req[blk].load(Ordering::Acquire) as u32
    }
    fn serve(&self, blk: usize, seq: u32) {
        let seq = u64::from(seq);
        // The C serve refuses a request whose host_seq is not current.
        if self.host_seq[blk].load(Ordering::Acquire) != seq {
            self.violation.store(true, Ordering::Relaxed);
            return;
        }
        if self.setup[blk].load(Ordering::Relaxed) != seq * 10 + blk as u64 {
            self.violation.store(true, Ordering::Relaxed);
        }
        if self.landed[blk].swap(seq, Ordering::AcqRel) == seq {
            self.violation.store(true, Ordering::Relaxed); // served twice
        }
        self.leases.fetch_add(1, Ordering::AcqRel);
        self.served.fetch_add(1, Ordering::Release);
    }
}

/// One compute of two layers: every published request is served exactly
/// once with its setup visible, and the decode thread's quiesce observes all
/// leases taken before it releases them.
#[test]
fn every_published_request_is_served_once_with_its_setup_visible() {
    primitives::model(|| {
        let m = Arc::new(Model::new());
        let stop = Arc::new(AtomicBool::new(false));
        let server = {
            let mut ring = Server(m.clone());
            let stop = stop.clone();
            primitives::spawn(move || {
                let mut p = Poller::new(N_BLK);
                loop {
                    p.poll(&mut ring);
                    if stop.load(Ordering::Acquire) {
                        // a final pass: the publisher stopped after its quiesce
                        p.poll(&mut ring);
                        return;
                    }
                    primitives::yield_now();
                }
            })
        };
        // decode thread
        m.publish(0, 1);
        m.publish(1, 1);
        let published = 2;
        // golem_dw_quiesce: wait for every serve before copy_done
        while m.served.load(Ordering::Acquire) < published {
            primitives::yield_now();
        }
        // copy_done: every lease of the step is visible here and released
        assert_eq!(
            m.leases.swap(0, Ordering::AcqRel),
            published,
            "copy_done must see every serve's lease before releasing"
        );
        for blk in 0..N_BLK {
            assert_eq!(m.landed[blk].load(Ordering::Acquire), 1, "blk {blk} landed");
        }
        stop.store(true, Ordering::Release);
        server.join().unwrap();
        assert!(!m.violation.load(Ordering::Relaxed), "stale or torn serve");
        assert_eq!(
            m.served.load(Ordering::Acquire),
            published,
            "no extra serve"
        );
    });
}

/// Two computes back to back on one layer: the second publish supersedes the
/// first only after the first was served (the decode thread quiesces between
/// computes), so the server never serves a request whose setup was replaced.
#[test]
fn back_to_back_computes_never_serve_a_replaced_setup() {
    primitives::model(|| {
        let m = Arc::new(Model::new());
        let stop = Arc::new(AtomicBool::new(false));
        let server = {
            let mut ring = Server(m.clone());
            let stop = stop.clone();
            primitives::spawn(move || {
                let mut p = Poller::new(N_BLK);
                loop {
                    p.poll(&mut ring);
                    if stop.load(Ordering::Acquire) {
                        p.poll(&mut ring);
                        return;
                    }
                    primitives::yield_now();
                }
            })
        };
        for seq in 1..=2u64 {
            m.publish(0, seq);
            while m.served.load(Ordering::Acquire) < seq as usize {
                primitives::yield_now();
            }
            assert_eq!(
                m.leases.swap(0, Ordering::AcqRel),
                1,
                "compute {seq}: one lease"
            );
            // compute end: the active word clears (golem_dw_finish)
            m.active.store(0, Ordering::Release);
        }
        stop.store(true, Ordering::Release);
        server.join().unwrap();
        assert!(!m.violation.load(Ordering::Relaxed), "stale or torn serve");
        assert_eq!(m.landed[0].load(Ordering::Acquire), 2);
    });
}

/// What layer 1's slab prepare does (ggml-backend.cpp `golem_slab_prepare`).
#[derive(Clone, Copy)]
enum Prepare {
    /// Fingerprint unchanged: no free, no alloc, no sync.
    Reuse,
    /// Remapped-ids tensor reshaped: `ggml_backend_synchronize`.
    IdsReshape,
    /// Slab reallocated (K lowered after ALLOC_FAILED, or a new plane
    /// stride): buffer free + clear, both device-synchronizing.
    SlabRealloc,
}

impl Prepare {
    /// `golem_slab_prepare_reallocs`: every case that can synchronize.
    fn reallocs(self) -> bool {
        !matches!(self, Prepare::Reuse)
    }
}

/// The slot-table lock across a device sync (the A/B hang): layer 0 is
/// published and its serve pending; the decode thread sets up layer 1, whose
/// prepare may synchronize the device — queued behind layer 0's landed wait.
/// The serve needs the slot-table lock to land layer 0.
struct LockModel {
    /// `dc_mutex` (the slot tables).
    dc: Mutex<()>,
    /// Layer 0's landed word, and the compute stream draining past it.
    landed: Mutex<bool>,
    landed_cv: Condvar,
    served: Mutex<usize>,
    served_cv: Condvar,
}

impl LockModel {
    fn new() -> Self {
        Self {
            dc: Mutex::new(()),
            landed: Mutex::new(false),
            landed_cv: Condvar::new(),
            served: Mutex::new(0),
            served_cv: Condvar::new(),
        }
    }

    /// The controller's serve of layer 0: slot step under the lock, then
    /// land and count it.
    fn serve(&self) {
        drop(self.dc.lock().unwrap());
        *self.landed.lock().unwrap() = true;
        self.landed_cv.notify_all();
        *self.served.lock().unwrap() += 1;
        self.served_cv.notify_all();
    }

    /// `ggml_backend_synchronize` of the compute stream: returns once layer
    /// 0's landed wait has passed.
    fn sync_compute_stream(&self) {
        let mut l = self.landed.lock().unwrap();
        while !*l {
            l = self.landed_cv.wait(l).unwrap();
        }
    }

    /// `golem_dw_quiesce`: every published request served.
    fn quiesce(&self, published: usize) {
        let mut s = self.served.lock().unwrap();
        while *s < published {
            s = self.served_cv.wait(s).unwrap();
        }
    }

    /// `golem_dw_setup` of layer 1: when the prepare reallocates (the
    /// substrate's `golem_slab_prepare_reallocs`), every published request
    /// is served (lock released) before the sync. Syncing under the lock
    /// without this quiesce is the deadlock loom reports (decode thread in
    /// the sync holding the lock, serve blocked on it).
    fn setup_next_layer(&self, prep: Prepare) {
        let g = self.dc.lock().unwrap();
        if prep.reallocs() {
            drop(g);
            self.quiesce(1);
            let _g = self.dc.lock().unwrap();
            self.sync_compute_stream();
        }
    }
}

/// The setup quiesces before any syncing prepare — an ids reshape or a slab
/// reallocation — so no interleaving deadlocks; a reusing prepare never
/// blocks on the device at all.
#[test]
fn slab_realloc_sync_never_holds_the_lock_a_pending_serve_needs() {
    for prep in [Prepare::Reuse, Prepare::IdsReshape, Prepare::SlabRealloc] {
        primitives::model(move || {
            let m = Arc::new(LockModel::new());
            let server = {
                let m = m.clone();
                primitives::spawn(move || m.serve())
            };
            m.setup_next_layer(prep);
            server.join().unwrap();
            assert!(*m.landed.lock().unwrap());
        });
    }
}

/// A split mixing a host-path plane with a device-wait plane: the copy
/// stream's compute fence (`golem_cs_begin`) quiesces first, so the served
/// layer's landed write is queued before the fence and the compute stream's
/// landed wait never sits behind it. Modeled as FIFO stream order: the
/// fence position must come after the landed write's.
#[test]
fn copy_fence_after_a_published_wait_is_queued_after_its_landed_write() {
    primitives::model(|| {
        // copy-stream queue: 1 = landed write (serve), 2 = compute fence
        let copy_q = Arc::new(Mutex::new(Vec::new()));
        let served = Arc::new(AtomicUsize::new(0));
        let server = {
            let (q, served) = (copy_q.clone(), served.clone());
            primitives::spawn(move || {
                q.lock().unwrap().push(1u8);
                served.fetch_add(1, Ordering::Release);
            })
        };
        // decode thread: published 1 request, then opens the copy stream
        while served.load(Ordering::Acquire) < 1 {
            primitives::yield_now();
        }
        copy_q.lock().unwrap().push(2u8);
        server.join().unwrap();
        assert_eq!(
            *copy_q.lock().unwrap(),
            [1, 2],
            "fence queued after the landed write"
        );
    });
}
