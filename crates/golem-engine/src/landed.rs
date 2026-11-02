//! Device-side landed wait controller thread (ABI v20).
//!
//! The decode thread no longer synchronizes per MoE layer: the GPU publishes
//! each layer's routed ids into a mapped ring and waits on the device for the
//! layer's landed words. This thread polls the ring's request words and
//! serves each new request through the substrate (`golem_device_wait_serve`):
//! the routing and predict hooks, the P2 slot step (slots assigned at claim
//! time, never in read-completion order — I1), one batched P3 ensure, and the
//! pool→slab H2D on the copy stream with each landed word queued behind its
//! copy. The ring protocol itself lives in `golem_placement::pool::landed`
//! (modeled by the §12.2 loom leg).

use golem_llama_sys::{DeviceWaitRing, Serve};
use golem_placement::pool::landed::{Poller, Ring};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The substrate ring behind the protocol trait; owned by the controller
/// thread, the ring's only server.
struct SysRing(DeviceWaitRing);

impl Ring for SysRing {
    fn n_blk(&self) -> usize {
        self.0.n_blk()
    }
    fn active(&self) -> u32 {
        self.0.active()
    }
    fn request(&self, blk: usize) -> u32 {
        self.0.request(blk).unwrap_or(0)
    }
    fn serve(&mut self, blk: usize, seq: u32) {
        match self.0.serve(blk, seq) {
            Serve::Served => {}
            Serve::Failed => tracing::error!(
                blk,
                seq,
                "device-wait serve: page-in failed; the compute fails"
            ),
            // The C side refused a superseded or not-yet-current request; the
            // poller re-serves the newest one on its next pass.
            Serve::Stale => tracing::debug!(blk, seq, "device-wait serve: stale request"),
        }
    }
}

/// The controller thread (one per session). Spins while a compute is
/// active, yields briefly when idle, then naps 50 µs at a time between
/// decode steps.
pub(crate) struct Controller {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Controller {
    /// Spawns the controller over `ring`.
    pub(crate) fn spawn(ring: DeviceWaitRing) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let stop = stop.clone();
            std::thread::Builder::new()
                .name("golem-landed".into())
                .spawn(move || {
                    // FFI-abort path: the GPU waits on landed words only this
                    // thread writes, so a dead controller would hang decode.
                    let served = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run(&mut SysRing(ring), &stop)
                    }));
                    if served.is_err() {
                        tracing::error!("device-wait controller thread panicked; aborting");
                        std::process::abort();
                    }
                })?
        };
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }

    /// Signals and joins. Runs before the hooks are unregistered: a serve
    /// invokes them. Idempotent.
    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run<R: Ring>(ring: &mut R, stop: &AtomicBool) {
    let mut poller = Poller::new(ring.n_blk());
    let mut idle = 0u32;
    while !stop.load(Ordering::Acquire) {
        if poller.poll(ring) > 0 || ring.active() != 0 {
            idle = 0;
            std::hint::spin_loop();
            continue;
        }
        idle = idle.saturating_add(1);
        if idle < 4096 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
}
