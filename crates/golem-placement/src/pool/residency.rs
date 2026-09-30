//! Residency map + P3 counters (§6.6): (layer, kind, expert) → slot with
//! in-flight leases and atomic claim-time eviction.

// `cfg(loom)` is swapped via RUSTFLAGS by the T6 loom leg (declared in
// primitives.rs); register it as known for this module's cfg uses too.
#![allow(unexpected_cfgs)]

use std::collections::HashSet;

use super::primitives::{AtomicU64, Condvar, Mutex};

// One expert-name matcher shared by C and Rust semantics (design §6.6; the C
// side mirrors the rule in golem_expert_stem_kind).
pub use golem_gguf::ExpertKind;

/// Flat pool key for `(layer, kind, expert)`.
pub fn key_index(layer: u32, kind: ExpertKind, expert: u32, n_experts: u32) -> usize {
    (layer as usize * ExpertKind::ALL.len() + kind.ordinal()) * n_experts as usize + expert as usize
}

/// Immutable, pre-materialized policy state consumed by eviction victim picks
/// (§6.6): keys protected like outstanding leases. The pool never locks
/// policy state — `claim` receives a snapshot built before the shard lock is
/// taken; the shard lock stays leaf-most.
///
/// Pool-local for M2.3 (`test_empty`/`pin`/`lease` helpers); the M2.2-backed
/// builder lands with the engine wiring.
#[derive(Debug, Clone, Default)]
pub struct PolicySnapshot {
    protected: HashSet<Key>,
}

/// Layer/kind/expert identity, layer-independent (unlike [`key_index`], whose
/// width depends on `n_experts`). Snapshot keys must match real keys at every
/// layer/kind.
type Key = (u32, u8, u32);

fn key_of(layer: u32, kind: ExpertKind, expert: u32) -> Key {
    (layer, kind.ordinal() as u8, expert)
}

impl PolicySnapshot {
    /// Empty snapshot: nothing pinned, nothing protected.
    pub fn test_empty() -> Self {
        Self::default()
    }

    /// Protect a key exactly like an outstanding map lease would.
    pub fn lease(&mut self, layer: u32, kind: ExpertKind, expert: u32) {
        self.protected.insert(key_of(layer, kind, expert));
    }

    fn excludes(&self, layer: u32, kind: ExpertKind, expert: u32) -> bool {
        self.protected.contains(&key_of(layer, kind, expert))
    }
}

/// One expert layer's residency state (§6.6): resident bits, lease counts and
/// slot assignment for this layer's keys. The slot free list and the CLOCK
/// hand are global (§6.6: one pool, no per-layer quota), so a shard's critical
/// section covers only `{bit flip, slot_of, lease grant/release}`.
#[derive(Debug)]
struct Shard {
    bits: Vec<u64>,
    /// CLOCK second-chance bits, one per key: set on every lease grant,
    /// cleared (once) by the CLOCK scan before a keyed victim is taken.
    ref_bits: Vec<u64>,
    leases: Vec<u16>,
    slot_of: Vec<i32>,
    /// Keys claimed (`Filled`/evict-refill) whose slot bytes the caller is
    /// still writing — the fill-completion gate: `Resident` claimers of these
    /// keys park until the filler reports completion.
    filling: HashSet<usize>,
}

/// A shard's lock paired with the condvar its fill gate waits on. The lock is
/// the leaf-most in the process; the condvar only ever releases/reacquires it.
#[derive(Debug)]
struct ShardLock {
    shard: Mutex<Shard>,
    fill_cv: Condvar,
}

/// Residency map over all expert layers: `(layer, kind, expert) → slot` with
/// in-flight leases and atomic claim-time eviction (§6.6).
///
/// Lock order (acyclic): `sweep` → shard; the slot free list is a leaf — a
/// shard guard is always dropped before the free list is touched, and the
/// CLOCK scan never runs under the free list. The `sweep` mutex serialises
/// evictions so two concurrent claimers can never take the same victim.
#[derive(Debug)]
pub struct ResidencyMap {
    n_experts: u32,
    total_keys: usize,
    shards: Vec<ShardLock>,
    /// Global free-slot stack (§6.6: one pool, no per-layer quota). Leaf lock.
    free_slots: Mutex<Vec<u32>>,
    /// Serialises CLOCK sweeps; holds the hand over `[0, total_keys)`.
    sweep: Mutex<ClockHand>,
}

/// Maximum keys examined per sweep call. A victimless sweep advances the hand
/// past the examined window and returns `None`; the claimer yields and
/// retries, continuing the scan. Second chances persist across sweeps, so a
/// key passed over with its reference bit set can be taken by a later sweep.
const SWEEP_BUDGET: usize = 1024;

#[derive(Debug)]
struct ClockHand {
    hand: usize,
}

/// Claim outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Claim {
    /// Newly assigned slot (bit flipped under the lock); caller fills it
    /// (I/O outside all locks) and copies under the granted lease.
    Filled {
        /// The slot the caller must fill.
        slot: u32,
        /// Exclusive-fill capability for that slot: the only
        /// way to take `&mut` on an arena slot; constructed here and in the
        /// eviction path, both inside the shard lock.
        claim: FillClaim,
    },
    /// Already resident; a lease was granted; `slot` names the filled slot.
    Resident {
        /// The filled slot the lease covers.
        slot: u32,
    },
}

/// An expert evicted by this claim; its slot is reassigned to the claimer,
/// whose I/O replaces the victim's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evicted {
    /// The evicted victim.
    pub victim: Victim,
    /// The slot the caller must refill (== `victim.slot`).
    pub slot: u32,
    /// Exclusive-fill capability for that slot.
    pub claim: FillClaim,
}

/// Exclusive-fill capability for one arena slot: a safe `&mut`
/// on a slot requires one; only the residency map constructs them, under the
/// shard lock, exactly when it has reassigned the slot to the claimer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillClaim {
    /// The slot this claim exclusively fills.
    pub(crate) slot: u32,
}

/// The key + slot an eviction displaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Victim {
    /// Victim layer.
    pub layer: u32,
    /// Victim plane kind.
    pub kind: ExpertKind,
    /// Victim expert index.
    pub expert: u32,
    /// The slot whose bytes the victim owned.
    pub slot: u32,
}

impl ResidencyMap {
    /// Builds a map over `n_layers` expert layers of `n_experts` experts with
    /// a shared pool of `pool_slots` slots (one global free list, no
    /// per-layer quota).
    pub fn new(n_layers: u32, n_experts: u32, pool_slots: u32) -> Self {
        debug_assert!(n_layers > 0 && n_experts > 0);
        debug_assert!(pool_slots > 0, "a pool without slots cannot serve");
        let keys_per_layer = Self::keys_per_layer(n_experts);
        let shards = (0..n_layers)
            .map(|_| {
                let words = keys_per_layer.div_ceil(64);
                ShardLock {
                    shard: Mutex::new(Shard {
                        bits: vec![0; words],
                        ref_bits: vec![0; words],
                        leases: vec![0; keys_per_layer],
                        slot_of: vec![-1; keys_per_layer],
                        filling: HashSet::new(),
                    }),
                    fill_cv: Condvar::new(),
                }
            })
            .collect();
        // pop() allocates slot 0 first.
        Self {
            n_experts,
            total_keys: n_layers as usize * keys_per_layer,
            shards,
            free_slots: Mutex::new((0..pool_slots).rev().collect()),
            sweep: Mutex::new(ClockHand { hand: 0 }),
        }
    }

    fn keys_per_layer(n_experts: u32) -> usize {
        ExpertKind::ALL.len() * n_experts as usize
    }

    fn local(&self, kind: ExpertKind, expert: u32) -> usize {
        kind.ordinal() * self.n_experts as usize + expert as usize
    }

    fn bit_test(bits: &[u64], i: usize) -> bool {
        bits[i / 64] & (1 << (i % 64)) != 0
    }

    fn bit_set(bits: &mut [u64], i: usize) {
        bits[i / 64] |= 1 << (i % 64);
    }

    fn bit_clear(bits: &mut [u64], i: usize) {
        bits[i / 64] &= !(1 << (i % 64));
    }

    /// Residency of one key (shard lock, flip-length hold).
    pub fn is_resident(&self, layer: u32, kind: ExpertKind, expert: u32) -> bool {
        let sh = self.shards[layer as usize].shard.lock().unwrap();
        Self::bit_test(&sh.bits, self.local(kind, expert))
    }

    /// Claims a key under ONE shard-lock critical section (§6.6):
    ///
    /// - resident → lease++, `Ok(Resident)`
    /// - free slot → assign + flip + lease, `Ok(Filled)`
    /// - full → pick victim from the lock-free [`PolicySnapshot`] (candidates
    ///   exclude leased and snapshot-excluded keys), clear the victim bit,
    ///   reassign its slot, flip the claimer's bit, grant the claimer's lease,
    ///   `Err(Evicted)` — the caller's I/O replaces the victim's bytes in that
    ///   slot.
    ///
    /// No policy locks are taken while the shard lock is held; victim pick
    /// only reads the snapshot. When every candidate is protected (all slots
    /// leased/pinned) the claim releases the lock, yields, and retries until a
    /// lease holder makes progress — a lease is always dropped by its owner in
    /// finite time, so the loop terminates.
    ///
    /// Every successful path (and `Err(Evicted)`) grants exactly one lease;
    /// the caller drops it via [`Self::release_lease`] after its copy, or
    /// [`Self::cancel_claim`] if its page-in fails.
    ///
    /// Fill-completion gating: `Filled`/evict-refill grants also mark the key
    /// mid-fill; a `Resident` claim on a mid-fill key parks on the shard
    /// condvar (lock released) until the filler's [`Self::finish_fill`] (or a
    /// `cancel_claim`) wakes it, so leased readers never observe slot bytes a
    /// concurrent filler is still writing. A claimer parked through a
    /// cancelled fill re-enters the loop and claims fresh.
    pub fn claim(
        &self,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
        snapshot: &PolicySnapshot,
    ) -> Result<Claim, Evicted> {
        let li = layer as usize;
        let i = self.local(kind, expert);
        loop {
            let mut sh = self.shards[li].shard.lock().unwrap();
            if Self::bit_test(&sh.bits, i) {
                while sh.filling.contains(&i) {
                    sh = self.shards[li].fill_cv.wait(sh).unwrap();
                }
                if Self::bit_test(&sh.bits, i) {
                    sh.leases[i] += 1;
                    Self::ref_set(&mut sh.ref_bits, i);
                    return Ok(Claim::Resident {
                        slot: sh.slot_of[i] as u32,
                    });
                }
                // Fill was cancelled while parked: fall through and claim
                // fresh (the slot is back on the free list).
            }
            // Falling through holds no state that needs the shard lock.
            // One global pool: take a free slot if any; else run the global
            // CLOCK. The shard guard stays dropped across the free list and
            // the scan — the free list is a leaf, and the sweep takes shard
            // locks one at a time under its own mutex.
            drop(sh);
            let acquired = match self.free_slots.lock().unwrap().pop() {
                Some(slot) => Some((slot, None)),
                None => self.clock_evict(snapshot).map(|(v, s)| (s, Some(v))),
            };
            let (slot, victim) = match acquired {
                Some(x) => x,
                None => {
                    super::primitives::yield_now();
                    continue;
                }
            };
            sh = self.shards[li].shard.lock().unwrap();
            if Self::bit_test(&sh.bits, i) {
                // A concurrent claimer filled this key while we hunted for a
                // slot: return the slot (its residency, if the CLOCK cleared
                // one, is already cleared) and re-read the state. The shard
                // guard is dropped before the free list (leaf) is touched.
                drop(sh);
                self.free_slots.lock().unwrap().push(slot);
                continue;
            }
            sh.filling.insert(i);
            sh.slot_of[i] = slot as i32;
            Self::bit_set(&mut sh.bits, i);
            // Reference bit stays clear on first fill: a key evicts on its
            // first sweep unless it was re-touched (Resident) in between.
            sh.leases[i] += 1;
            if let Some(victim) = victim {
                return Err(Evicted {
                    victim,
                    slot,
                    claim: FillClaim { slot },
                });
            }
            return Ok(Claim::Filled {
                slot,
                claim: FillClaim { slot },
            });
        }
    }

    /// Global CLOCK sweep: walks keys round-robin from the hand, giving each
    /// resident, unleased, unprotected key a second chance (clear its
    /// reference bit) before taking it as the victim. Reference bits are set
    /// only on Resident reuse — a key filled once and never re-touched is
    /// immediately evictable, while a reused key survives until a full pass
    /// finds nothing else. Returns the victim and its slot, with the victim's
    /// residency already cleared under its shard lock.
    fn clock_evict(&self, snapshot: &PolicySnapshot) -> Option<(Victim, u32)> {
        let kpl = Self::keys_per_layer(self.n_experts);
        let mut hand = self.sweep.lock().unwrap();
        let start = hand.hand;
        for step in 0..SWEEP_BUDGET.min(self.total_keys) {
            let idx = (start + step) % self.total_keys;
            let li = idx / kpl;
            let c = idx % kpl;
            let mut sh = self.shards[li].shard.lock().unwrap();
            if !Self::bit_test(&sh.bits, c) || sh.leases[c] != 0 {
                drop(sh);
                hand.hand = idx + 1;
                continue;
            }
            if Self::ref_test(&sh.ref_bits, c) {
                Self::ref_clear(&mut sh.ref_bits, c);
                drop(sh);
                hand.hand = idx + 1;
                continue;
            }
            let kind = ExpertKind::ALL[c / self.n_experts as usize];
            let expert = c as u32 % self.n_experts;
            if snapshot.excludes(li as u32, kind, expert) {
                drop(sh);
                continue;
            }
            let slot = sh.slot_of[c] as u32;
            Self::bit_clear(&mut sh.bits, c);
            sh.slot_of[c] = -1;
            drop(sh);
            hand.hand = idx + 1;
            return Some((
                Victim {
                    layer: li as u32,
                    kind,
                    expert,
                    slot,
                },
                slot,
            ));
        }
        // Budget exhausted without a victim: park the hand past the examined
        // window so a retry sweep continues the scan instead of re-walking it.
        hand.hand = (start + SWEEP_BUDGET.min(self.total_keys)) % self.total_keys;
        None
    }

    fn ref_test(bits: &[u64], i: usize) -> bool {
        bits[i / 64] >> (i % 64) & 1 == 1
    }

    fn ref_set(bits: &mut [u64], i: usize) {
        bits[i / 64] |= 1 << (i % 64);
    }

    fn ref_clear(bits: &mut [u64], i: usize) {
        bits[i / 64] &= !(1 << (i % 64));
    }

    /// Marks a mid-fill key's slot bytes complete: clears the fill gate and
    /// wakes parked `Resident` claimers, which then grant their leases. Pager
    /// contract: called (only) after the `Filled` claim's copy into the slot
    /// finished — the gate is what makes the slot's `&mut` phase exclusive.
    pub(crate) fn finish_fill(&self, layer: u32, kind: ExpertKind, expert: u32) {
        let mut sh = self.shards[layer as usize].shard.lock().unwrap();
        sh.filling.remove(&self.local(kind, expert));
        self.shards[layer as usize].fill_cv.notify_all();
    }

    /// Drops one lease. The slot stays resident until evicted via `claim`.
    pub fn release_lease(&self, layer: u32, kind: ExpertKind, expert: u32) {
        let mut sh = self.shards[layer as usize].shard.lock().unwrap();
        let i = self.local(kind, expert);
        debug_assert!(sh.leases[i] > 0, "release without a lease");
        sh.leases[i] = sh.leases[i].saturating_sub(1);
    }

    /// Abandon a failed page-in: drops the claim's lease and, when no other
    /// lease remains, clears the key's residency and returns the slot to the
    /// global free list — no ghost residency, no leaked slot. Also clears the
    /// fill gate and wakes claimers parked on it (they re-claim fresh).
    pub fn cancel_claim(&self, layer: u32, kind: ExpertKind, expert: u32) {
        let li = layer as usize;
        let freed = {
            let mut sh = self.shards[li].shard.lock().unwrap();
            let i = self.local(kind, expert);
            debug_assert!(sh.leases[i] > 0, "cancel without a lease");
            sh.filling.remove(&i);
            sh.leases[i] = sh.leases[i].saturating_sub(1);
            if sh.leases[i] == 0 {
                Self::bit_clear(&mut sh.bits, i);
                let s = sh.slot_of[i];
                sh.slot_of[i] = -1;
                (s >= 0).then_some(s as u32)
            } else {
                None
            }
            // Shard guard dropped before the leaf free list.
        };
        if let Some(s) = freed {
            self.free_slots.lock().unwrap().push(s);
        }
        self.shards[li].fill_cv.notify_all();
    }
}

/// P3 slow-path counters (Global Constraints: demand and prefetch counted
/// separately; P4/M2.4 absorbs this block without renames).
#[derive(Debug, Default)]
pub struct P3Stats {
    /// Page-ins on the demand (decode) path.
    pub demand_slow_pageins: AtomicU64,
    /// Page-ins issued by the §6.5 prefetcher.
    pub prefetch_pageins: AtomicU64,
    /// Claims served from residency (no I/O).
    pub fast_hits: AtomicU64,
    /// Total bytes moved from source into pool slots.
    pub bytes_paged: AtomicU64,
    /// Atomic evictions performed inside `claim`.
    pub evictions: AtomicU64,
    /// Source read failures after bounded retries.
    pub source_errors: AtomicU64,
}

impl P3Stats {
    /// All counters zero.
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::super::primitives::{self, Arc};
    use super::*;

    #[test]
    fn kind_parses_ggml_and_gguf_names() {
        assert_eq!(
            ExpertKind::from_tensor_name("blk.1.ffn_gate_exps"),
            Some(ExpertKind::Gate)
        );
        assert_eq!(
            ExpertKind::from_tensor_name("blk.1.ffn_up_exps.weight"),
            Some(ExpertKind::Up)
        );
        assert_eq!(
            ExpertKind::from_tensor_name("blk.1.ffn_down_exps.weight"),
            Some(ExpertKind::Down)
        );
        assert_eq!(
            ExpertKind::from_tensor_name("blk.1.ffn_gate_inp.weight"),
            None
        );
        assert_eq!(ExpertKind::from_tensor_name("blk.0.ffn_gate.weight"), None);
    }

    #[test]
    fn key_index_is_flat_and_injective() {
        assert_eq!(key_index(0, ExpertKind::Gate, 3, 8), 3);
        assert_eq!(key_index(0, ExpertKind::Up, 0, 8), 8);
        assert_eq!(key_index(1, ExpertKind::Down, 7, 8), 3 * 8 + 2 * 8 + 7);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn claim_flip_then_resident() {
        let m = ResidencyMap::new(2, 8, 4);
        assert!(!m.is_resident(0, ExpertKind::Gate, 3));
        let snap = PolicySnapshot::test_empty();
        match m.claim(0, ExpertKind::Gate, 3, &snap).unwrap() {
            Claim::Filled { slot, .. } => assert_eq!(slot, 0),
            Claim::Resident { .. } => panic!("first claim must fill"),
        }
        assert!(m.is_resident(0, ExpertKind::Gate, 3));
        m.release_lease(0, ExpertKind::Gate, 3);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn second_claim_while_leased_is_a_hit_with_new_lease() {
        let m = ResidencyMap::new(2, 8, 4);
        let snap = PolicySnapshot::test_empty();
        m.claim(0, ExpertKind::Gate, 3, &snap).unwrap();
        m.finish_fill(0, ExpertKind::Gate, 3); // model the pager's page-in
        assert!(matches!(
            m.claim(0, ExpertKind::Gate, 3, &snap).unwrap(),
            Claim::Resident { .. }
        ));
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn full_pool_returns_victim_excluding_leased_and_pinned() {
        let m = ResidencyMap::new(1, 8, 2);
        let mut snap = PolicySnapshot::test_empty();
        snap.lease(0, ExpertKind::Gate, 0); // protected like an outstanding lease
        m.claim(0, ExpertKind::Gate, 0, &snap).unwrap();
        m.finish_fill(0, ExpertKind::Gate, 0);
        m.claim(0, ExpertKind::Gate, 1, &snap).unwrap();
        m.finish_fill(0, ExpertKind::Gate, 1);
        m.release_lease(0, ExpertKind::Gate, 0);
        m.release_lease(0, ExpertKind::Gate, 1);
        snap.lease(0, ExpertKind::Gate, 0); // simulate an outstanding lease on expert 0
        let e = m.claim(0, ExpertKind::Gate, 2, &snap).unwrap_err();
        assert_eq!(e.victim.expert, 1, "leased expert 0 must never be picked");
        assert!(!m.is_resident(0, ExpertKind::Gate, 1), "victim bit cleared");
        assert!(m.is_resident(0, ExpertKind::Gate, 2), "claimer flipped");
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn cancel_returns_slot_and_clears_residency() {
        let m = ResidencyMap::new(1, 8, 1);
        let snap = PolicySnapshot::test_empty();
        m.claim(0, ExpertKind::Gate, 0, &snap).unwrap();
        m.cancel_claim(0, ExpertKind::Gate, 0);
        assert!(!m.is_resident(0, ExpertKind::Gate, 0));
        // The freed slot serves the next miss.
        match m.claim(0, ExpertKind::Gate, 1, &snap).unwrap() {
            Claim::Filled { slot, .. } => assert_eq!(slot, 0),
            Claim::Resident { .. } => panic!("slot must have been recycled"),
        }
    }

    #[test]
    fn concurrent_claim_two_threads_one_pagein() {
        primitives::model(|| {
            let m = Arc::new(ResidencyMap::new(1, 8, 2));
            let m2 = m.clone();
            let snap = Arc::new(PolicySnapshot::test_empty());
            let snap2 = snap.clone();
            // Models the pager per-thread: the `Filled` winner completes its
            // fill through the gate so the parked `Resident` claimer wakes.
            // Invariant: both claimers observe the SAME slot. Which thread
            // wins the fill — and which slot it took — is
            // scheduling-dependent (the global free list is shared).
            let seen = Arc::new(Mutex::new(None));
            let seen2 = seen.clone();
            let run =
                |m: Arc<ResidencyMap>, snap: Arc<PolicySnapshot>, seen: Arc<Mutex<Option<u32>>>| {
                    let slot = match m.claim(0, ExpertKind::Gate, 5, &snap).unwrap() {
                        Claim::Filled { slot, .. } => {
                            m.finish_fill(0, ExpertKind::Gate, 5);
                            slot
                        }
                        Claim::Resident { slot } => slot,
                    };
                    let mut first = seen.lock().unwrap();
                    match *first {
                        Some(prev) => assert_eq!(slot, prev, "both claimers see one slot"),
                        None => *first = Some(slot),
                    }
                };
            let t = primitives::spawn(move || run(m2, snap2, seen2));
            run(m, snap, seen);
            t.join().unwrap();
        });
    }

    #[test]
    fn concurrent_claim_and_cancel_on_full_pool() {
        // One thread claims (forcing a CLOCK eviction on the full pool) while
        // the other cancels a claim, returning its slot to the free list.
        // Either path must hand the claimer exactly one slot, and the pool
        // must end with no leaked slot: after both release/cancel, one more
        // claim must find a free slot without evicting anyone.
        primitives::model(|| {
            let m = Arc::new(ResidencyMap::new(1, 4, 2));
            let snap = Arc::new(PolicySnapshot::test_empty());
            // Fill the pool: keys 0 and 1.
            for e in [0u32, 1] {
                m.claim(0, ExpertKind::Gate, e, &snap).unwrap();
                m.finish_fill(0, ExpertKind::Gate, e);
                m.release_lease(0, ExpertKind::Gate, e);
            }
            let m2 = m.clone();
            let snap2 = snap.clone();
            let t = primitives::spawn(move || {
                // Claim key 2; the pool is full, so this either evicts or
                // consumes a slot the main thread's cancel just freed.
                match m2.claim(0, ExpertKind::Gate, 2, &snap2) {
                    Ok(Claim::Filled { .. }) | Err(_) => {
                        m2.finish_fill(0, ExpertKind::Gate, 2);
                        m2.cancel_claim(0, ExpertKind::Gate, 2);
                    }
                    Ok(Claim::Resident { .. }) => panic!("key 2 starts non-resident"),
                }
            });
            // Re-claim key 1 so both threads contend for the same slots; any
            // outcome is legal (Resident, Filled, or Evicted with key 0 or 1
            // as the victim). cancel_claim drops the lease and the residency,
            // returning the slot — so afterwards at most key 0 is resident
            // and at least one slot is free.
            match m.claim(0, ExpertKind::Gate, 1, &snap) {
                Ok(Claim::Resident { .. }) => {}
                Ok(Claim::Filled { .. }) | Err(_) => m.finish_fill(0, ExpertKind::Gate, 1),
            }
            m.cancel_claim(0, ExpertKind::Gate, 1);
            t.join().unwrap();
            // No slot leaked: a fresh claim must not evict.
            let claim = m.claim(0, ExpertKind::Gate, 3, &snap).unwrap();
            assert!(matches!(claim, Claim::Filled { .. }));
        });
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn clock_churn_does_not_single_out_one_hot_expert() {
        // Property: the victim is the CLOCK hand's coldest key, never a
        // fixed index. Trace (3 slots, hot = key 2, colds = keys 0/1/3, one
        // rotating cold claim per round after re-touching hot): the hand
        // always finds a cold key whose second chance a previous sweep
        // consumed before it wraps to the hot expert, so the hot expert is
        // never evicted — the first-index policy evicted the same
        // lowest-index key every time.
        let m = ResidencyMap::new(1, 4, 3);
        let snap = PolicySnapshot::test_empty();
        // Warm: hot, cA, cB resident (slots 0-2).
        for e in [2u32, 0, 1] {
            let _ = m.claim(0, ExpertKind::Gate, e, &snap).unwrap();
            m.finish_fill(0, ExpertKind::Gate, e);
            m.release_lease(0, ExpertKind::Gate, e);
        }
        let colds: [u32; 3] = [3, 0, 1]; // cC, then cA, then cB rotate
        for round in 0..30u32 {
            let _ = m.claim(0, ExpertKind::Gate, 2, &snap).unwrap(); // hot
            m.finish_fill(0, ExpertKind::Gate, 2);
            m.release_lease(0, ExpertKind::Gate, 2);
            let c = colds[(round % 3) as usize];
            if let Err(evicted) = m.claim(0, ExpertKind::Gate, c, &snap) {
                assert_ne!(
                    evicted.victim.expert, 2,
                    "round {round}: the hot expert was evicted"
                );
                m.finish_fill(0, ExpertKind::Gate, c);
                m.release_lease(0, ExpertKind::Gate, c);
            }
        }
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn pin_and_lease_exclude_real_keys_at_every_layer_and_kind() {
        // Regression: snapshot keys were built with key_index(.., u32::MAX),
        // which matched real keys only at (layer 0, Gate) — a pin/lease on any
        // other layer or kind was silently ignored by victim picks.
        // Snapshot-lease, layer 1, Gate:
        let m = ResidencyMap::new(2, 4, 2); // global pool: 2 slots total
        let mut snap = PolicySnapshot::test_empty();
        snap.lease(1, ExpertKind::Gate, 0);
        m.claim(1, ExpertKind::Gate, 0, &snap).unwrap();
        m.finish_fill(1, ExpertKind::Gate, 0);
        m.claim(1, ExpertKind::Down, 0, &snap).unwrap();
        m.finish_fill(1, ExpertKind::Down, 0);
        m.release_lease(1, ExpertKind::Gate, 0);
        m.release_lease(1, ExpertKind::Down, 0);
        let e = m.claim(1, ExpertKind::Gate, 1, &snap).unwrap_err();
        assert_eq!(
            (e.victim.kind, e.victim.expert),
            (ExpertKind::Down, 0),
            "snapshot-leased (1, Gate, 0) must never be picked"
        );
        assert!(m.is_resident(1, ExpertKind::Gate, 0), "lease held");

        // Snapshot lease, layer 1, non-Gate kind (victim scan hits it first):
        let m = ResidencyMap::new(2, 4, 2);
        let mut snap = PolicySnapshot::test_empty();
        snap.lease(1, ExpertKind::Gate, 1);
        m.claim(1, ExpertKind::Gate, 1, &snap).unwrap();
        m.finish_fill(1, ExpertKind::Gate, 1);
        m.claim(1, ExpertKind::Up, 0, &snap).unwrap();
        m.finish_fill(1, ExpertKind::Up, 0);
        m.release_lease(1, ExpertKind::Gate, 1);
        m.release_lease(1, ExpertKind::Up, 0);
        let e = m.claim(1, ExpertKind::Gate, 2, &snap).unwrap_err();
        assert_eq!(
            (e.victim.kind, e.victim.expert),
            (ExpertKind::Up, 0),
            "snapshot-leased (1, Gate, 1) must never be picked"
        );
        assert!(m.is_resident(1, ExpertKind::Gate, 1), "lease held");
    }

    #[test]
    #[cfg(not(loom))] // timing-based; loom explores the same gate in-model
    fn resident_claim_parks_until_fill_completes() {
        use std::sync::mpsc;
        use std::time::Duration;

        let m = Arc::new(ResidencyMap::new(1, 8, 2));
        let snap = PolicySnapshot::test_empty();
        assert!(matches!(
            m.claim(0, ExpertKind::Gate, 5, &snap).unwrap(),
            Claim::Filled { .. }
        ));
        let (tx, rx) = mpsc::channel();
        let m2 = m.clone();
        let t = std::thread::spawn(move || {
            let resident = matches!(
                m2.claim(0, ExpertKind::Gate, 5, &snap),
                Ok(Claim::Resident { slot: 0 })
            );
            tx.send(()).unwrap();
            resident
        });
        assert!(
            rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "claim must park while the fill is in flight"
        );
        m.finish_fill(0, ExpertKind::Gate, 5);
        assert!(t.join().unwrap(), "post-fill claim must be Resident");
    }

    #[test]
    fn cancel_wakes_parked_claimers_with_recycled_slot() {
        primitives::model(|| {
            let m = Arc::new(ResidencyMap::new(1, 8, 2));
            let m2 = m.clone();
            let snap = Arc::new(PolicySnapshot::test_empty());
            let snap2 = snap.clone();
            // Main holds key 5 mid-fill; the spawned claimer either parks on
            // the gate (the cancel wakes it and it claims the recycled slot
            // fresh) or runs entirely after the cancel — both end Filled on
            // the one slot.
            assert!(matches!(
                m.claim(0, ExpertKind::Gate, 5, &snap).unwrap(),
                Claim::Filled { .. }
            ));
            let t = primitives::spawn(move || {
                // Filled on whichever slot was free when it re-claimed.
                matches!(
                    m2.claim(0, ExpertKind::Gate, 5, &snap2),
                    Ok(Claim::Filled { .. })
                )
            });
            m.cancel_claim(0, ExpertKind::Gate, 5);
            assert!(t.join().unwrap());
        });
    }
}
