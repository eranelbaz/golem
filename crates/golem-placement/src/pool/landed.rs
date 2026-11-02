//! Device-side landed wait: the controller's request-ring protocol,
//! independent of the substrate so the §12.2 loom leg can model it.
//!
//! The GPU publishes each MoE layer's routed ids into a mapped ring and bumps
//! that layer's request word to the compute's sequence; it then waits on the
//! device for the layer's landed words. The controller polls the request
//! words and serves every new request once (routing/predict hooks, P2 slot
//! step at claim time, one batched P3 ensure, H2D with each landed word
//! queued behind its copy). The decode thread leaves a compute — and so
//! reaches the next `copy_done` that releases the step's leases — only after
//! it has seen every published request served.
//!
//! The only host poll is on the request words; it carries no DMA ordering.

/// The ring as the controller sees it. `serve` takes `&mut self`: one owner
/// serves, so the substrate's serve scratch has a single caller.
pub trait Ring {
    /// Number of blk request words.
    fn n_blk(&self) -> usize;
    /// Nonzero while a compute with published requests is in flight; the
    /// compute's sequence otherwise.
    fn active(&self) -> u32;
    /// The newest published request sequence of `blk` (0 = none).
    fn request(&self, blk: usize) -> u32;
    /// Serves `blk`'s request `seq`.
    fn serve(&mut self, blk: usize, seq: u32);
}

/// Poll state: the newest sequence served per blk. Preallocated once; a
/// pass allocates nothing (decode hot path).
pub struct Poller {
    served: Vec<u32>,
    /// Newest nonzero active word seen; a smaller one means the substrate
    /// wrapped its sequence and zeroed the ring.
    last_active: u32,
}

impl Poller {
    /// A poller for `n_blk` request words.
    pub fn new(n_blk: usize) -> Self {
        Self {
            served: vec![0; n_blk],
            last_active: 0,
        }
    }

    /// One pass over the request words: serves every blk whose newest
    /// request differs from the last one served. Returns how many it served.
    pub fn poll<R: Ring + ?Sized>(&mut self, ring: &mut R) -> usize {
        let active = ring.active();
        if active == 0 {
            return 0;
        }
        if active < self.last_active {
            // Sequence wrap: the substrate restarted every word from zero, so
            // a stale `served` entry could equal a new request and hide it.
            self.served.fill(0);
        }
        self.last_active = active;
        let mut n = 0;
        for (blk, last) in self.served.iter_mut().enumerate() {
            let seq = ring.request(blk);
            if seq != 0 && seq != *last {
                *last = seq;
                ring.serve(blk, seq);
                n += 1;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        active: u32,
        req: Vec<u32>,
        served: Vec<(usize, u32)>,
    }

    impl Ring for Fake {
        fn n_blk(&self) -> usize {
            self.req.len()
        }
        fn active(&self) -> u32 {
            self.active
        }
        fn request(&self, blk: usize) -> u32 {
            self.req[blk]
        }
        fn serve(&mut self, blk: usize, seq: u32) {
            self.served.push((blk, seq));
        }
    }

    #[test]
    fn serves_each_request_once() {
        let mut r = Fake {
            active: 3,
            req: vec![3, 0],
            served: Vec::new(),
        };
        let mut p = Poller::new(2);
        assert_eq!(p.poll(&mut r), 1);
        assert_eq!(p.poll(&mut r), 0, "already served");
        r.active = 0;
        assert_eq!(p.poll(&mut r), 0, "idle ring");
        assert_eq!(r.served, [(0, 3)]);
    }

    /// After a sequence wrap a blk's new request may equal the stale value
    /// served long ago; the wrap (active word going backwards) resets it.
    #[test]
    fn sequence_wrap_does_not_hide_a_request() {
        let mut r = Fake {
            active: 5,
            req: vec![5, 0],
            served: Vec::new(),
        };
        let mut p = Poller::new(2);
        p.poll(&mut r);
        // blk 0 idles; the sequence runs to the wrap and restarts at 1.
        r.req = vec![5, u32::MAX - 1];
        r.active = u32::MAX - 1;
        p.poll(&mut r);
        r.req = vec![0, 0];
        for seq in 1..=5 {
            r.active = seq;
            r.req[1] = seq;
            if seq == 5 {
                r.req[0] = 5;
            }
            p.poll(&mut r);
        }
        assert!(
            r.served.contains(&(0, 5)) && r.served.iter().filter(|s| **s == (0, 5)).count() == 2,
            "blk 0's post-wrap request 5 must be served: {:?}",
            r.served
        );
    }
}
