//! Residency map + P3 counters (§6.6): (layer, expert) unit → slot with
//! in-flight leases and atomic claim-time eviction. One unit is an expert's
//! gate, up and down planes co-allocated in one slot (D-11): one key is
//! claimed, filled, leased and evicted as a whole.

// `cfg(loom)` is swapped via RUSTFLAGS by the T6 loom leg (declared in
// primitives.rs); register it as known for this module's cfg uses too.
#![allow(unexpected_cfgs)]
// Tests build one-class slot tables as `vec![0..slots]`; the lint's
// alternatives are no clearer.
#![allow(clippy::single_range_in_vec_init)]

use std::collections::HashSet;

use super::primitives::{AtomicU64, Condvar, Mutex};

// One expert-name matcher shared by C and Rust semantics (design §6.6; the C
// side mirrors the rule in golem_expert_stem_kind).
pub use golem_gguf::ExpertKind;

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

/// Layer/expert unit identity (layer-independent).
type Key = (u32, u32);

impl PolicySnapshot {
    /// Empty snapshot: nothing pinned, nothing protected.
    pub fn test_empty() -> Self {
        Self::default()
    }

    /// Protect a unit exactly like an outstanding map lease would.
    pub fn lease(&mut self, layer: u32, expert: u32) {
        self.protected.insert((layer, expert));
    }

    fn excludes(&self, layer: u32, expert: u32) -> bool {
        self.protected.contains(&(layer, expert))
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
    /// keys park until the filler reports completion. Bitset over layer-local
    /// keys (same `bit_*` helpers as `bits`): no hash lookup on the
    /// `Resident` fast path.
    filling: Vec<u64>,
}

/// A shard's lock paired with the condvar its fill gate waits on. The lock is
/// the leaf-most in the process; the condvar only ever releases/reacquires it.
#[derive(Debug)]
struct ShardLock {
    shard: Mutex<Shard>,
    fill_cv: Condvar,
}

/// Residency map over all expert layers: `(layer, expert) → slot` with
/// in-flight leases and atomic claim-time eviction (§6.6). Slots are carved
/// into size classes (D-2); a layer's keys claim, fill and evict only inside
/// `class_of_layer[layer]`'s slot range — no class borrows another's slot.
///
/// Lock order (acyclic): `sweep` → shard; the per-class free stacks share one
/// leaf lock — a shard guard is always dropped before the free list is
/// touched, and the CLOCK scan never runs under the free list. The `sweep`
/// mutex serialises evictions so two concurrent claimers can never take the
/// same victim.
#[derive(Debug)]
pub struct ResidencyMap {
    n_experts: u32,
    /// Layer → slot class; a claim for layer `l` pops / sweeps only class
    /// `class_of_layer[l]`.
    class_of_layer: Vec<u8>,
    /// Built once: the layers served by each class (class → layers).
    class_layers: Vec<Vec<u32>>,
    /// Key count per class (`class_layers[c].len() × keys_per_layer`); the
    /// starvation bound is computed against this, not the global key count.
    class_keys: Vec<usize>,
    shards: Vec<ShardLock>,
    /// One free-slot stack per class (§6.6: no per-layer quota; per-class
    /// because slot sizes differ). Shared leaf lock.
    free_slots: Mutex<Vec<Vec<u32>>>,
    /// Serialises CLOCK sweeps; one hand per class over that class's keys.
    sweep: Mutex<Vec<ClockHand>>,
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

/// Why a claim did not return a slot. `Evicted`: the victim's slot is the
/// claimer's and its I/O replaces the victim's bytes. `Busy`: every slot is
/// leased or snapshot-protected after bounded CLOCK walks, or the key's lease
/// count is saturated — transient by contract (leases release); the pool must
/// not be over-subscribed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimError {
    /// An expert evicted by this claim; its slot is reassigned to the
    /// claimer, whose I/O replaces the victim's bytes.
    Evicted(Evicted),
    /// No evictable slot after bounded full CLOCK walks, or `u16::MAX`
    /// outstanding leases on the key.
    Busy,
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

/// The unit + slot an eviction displaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Victim {
    /// Victim layer.
    pub layer: u32,
    /// Victim expert index.
    pub expert: u32,
    /// The slot whose bytes the victim owned.
    pub slot: u32,
}

impl ResidencyMap {
    /// Builds a map over `n_experts`-expert layers mapped to slot classes by
    /// `class_of_layer`; class `c` serves exactly the slots in
    /// `class_slots[c]` (contiguous arena ranges). All layers must name a
    /// valid class; every class must be non-empty and slot-disjoint.
    pub fn new(
        class_of_layer: Vec<u8>,
        n_experts: u32,
        class_slots: Vec<std::ops::Range<u32>>,
    ) -> Self {
        let n_layers = class_of_layer.len() as u32;
        debug_assert!(n_layers > 0 && n_experts > 0);
        debug_assert!(!class_slots.is_empty(), "a pool without slots cannot serve");
        let n_classes = class_slots.len();
        let mut class_layers = vec![Vec::new(); n_classes];
        for (l, &c) in class_of_layer.iter().enumerate() {
            class_layers[c as usize].push(l as u32);
        }
        let class_keys: Vec<usize> = class_layers
            .iter()
            .map(|ls| ls.len() * Self::keys_per_layer(n_experts))
            .collect();
        let keys_per_layer = Self::keys_per_layer(n_experts);
        let filling_words = keys_per_layer.div_ceil(64);
        let shards = (0..n_layers)
            .map(|_| ShardLock {
                shard: Mutex::new(Shard {
                    bits: vec![0; keys_per_layer.div_ceil(64)],
                    ref_bits: vec![0; keys_per_layer.div_ceil(64)],
                    leases: vec![0; keys_per_layer],
                    slot_of: vec![-1; keys_per_layer],
                    filling: vec![0; filling_words],
                }),
                fill_cv: Condvar::new(),
            })
            .collect();
        // pop() allocates each class's lowest slot first.
        let free_slots: Vec<Vec<u32>> = class_slots
            .iter()
            .map(|r| r.clone().rev().collect())
            .collect();
        Self {
            n_experts,
            class_of_layer,
            class_layers,
            class_keys,
            shards,
            free_slots: Mutex::new(free_slots),
            sweep: Mutex::new((0..n_classes).map(|_| ClockHand { hand: 0 }).collect()),
        }
    }

    fn keys_per_layer(n_experts: u32) -> usize {
        n_experts as usize
    }

    fn local(&self, expert: u32) -> usize {
        expert as usize
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

    /// Residency of one unit (shard lock, flip-length hold).
    pub fn is_resident(&self, layer: u32, expert: u32) -> bool {
        let sh = self.shards[layer as usize].shard.lock().unwrap();
        Self::bit_test(&sh.bits, self.local(expert))
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
    /// leased/pinned) the claim yields and retries through a bounded number
    /// of full CLOCK walks; if no victim appears it returns
    /// [`ClaimError::Busy`] — leases are transient, so over-subscription is
    /// a loud error, not an unbounded spin.
    ///
    /// Every successful path (and `Err(ClaimError::Evicted)`) grants exactly one lease;
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
        expert: u32,
        snapshot: &PolicySnapshot,
    ) -> Result<Claim, ClaimError> {
        let li = layer as usize;
        let class = self.class_of_layer[li] as usize;
        let i = self.local(expert);
        // Bounded starvation guard: CLOCK finds a victim within two full laps
        // (the first may only clear second-chance bits), each lap
        // ⌈class_keys / SWEEP_BUDGET⌉ sweeps over the claimer's own class.
        // Beyond that every slot in the class is leased or snapshot-protected
        // — leases are transient, so the class is over-subscribed and the
        // caller gets a loud busy error instead of an unbounded spin.
        let starve_bound = 2 * self.class_keys[class].div_ceil(SWEEP_BUDGET) + 1;
        let mut starved = 0usize;
        loop {
            let mut sh = self.shards[li].shard.lock().unwrap();
            if Self::bit_test(&sh.bits, i) {
                while Self::bit_test(&sh.filling, i) {
                    sh = self.shards[li].fill_cv.wait(sh).unwrap();
                }
                if Self::bit_test(&sh.bits, i) {
                    // Saturated lease count: a wrap to 0 would make a leased
                    // key evictable under in-flight readers. Transient like
                    // every other Busy — leases release.
                    if sh.leases[i] == u16::MAX {
                        return Err(ClaimError::Busy);
                    }
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
            // Bind the pop outside the match: a match-scrutinee temporary
            // lives to the end of the match, which would hold the free-list
            // guard across the CLOCK sweep below.
            let popped = self.free_slots.lock().unwrap()[class].pop();
            let acquired = match popped {
                Some(slot) => Some((slot, None)),
                None => {
                    starved += 1;
                    if starved > starve_bound {
                        return Err(ClaimError::Busy);
                    }
                    self.clock_evict(class, snapshot).map(|(v, s)| (s, Some(v)))
                }
            };
            let (slot, victim) = match acquired {
                Some(x) => x,
                None => {
                    super::primitives::yield_now();
                    continue;
                }
            };
            starved = 0;
            sh = self.shards[li].shard.lock().unwrap();
            if Self::bit_test(&sh.bits, i) {
                // A concurrent claimer filled this key while we hunted for a
                // slot: return the slot (its residency, if the CLOCK cleared
                // one, is already cleared) and re-read the state. The shard
                // guard is dropped before the free list (leaf) is touched.
                drop(sh);
                self.free_slots.lock().unwrap()[class].push(slot);
                continue;
            }
            Self::bit_set(&mut sh.filling, i);
            sh.slot_of[i] = slot as i32;
            Self::bit_set(&mut sh.bits, i);
            // Reference bit stays clear on first fill: a key evicts on its
            // first sweep unless it was re-touched (Resident) in between.
            // A non-resident key holds no leases (eviction and cancel clear
            // the bit only at zero), so this cannot overflow.
            debug_assert_eq!(sh.leases[i], 0, "non-resident key with leases");
            sh.leases[i] += 1;
            if let Some(victim) = victim {
                return Err(ClaimError::Evicted(Evicted {
                    victim,
                    slot,
                    claim: FillClaim { slot },
                }));
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
    fn clock_evict(&self, class: usize, snapshot: &PolicySnapshot) -> Option<(Victim, u32)> {
        let kpl = Self::keys_per_layer(self.n_experts);
        let class_key_count = self.class_keys[class];
        let mut hands = self.sweep.lock().unwrap();
        let hand = &mut hands[class];
        let start = hand.hand;
        for step in 0..SWEEP_BUDGET.min(class_key_count) {
            let idx = (start + step) % class_key_count;
            // Class keys are (layer-in-class, expert): decode the flat index
            // back to the real layer and its shard-local key.
            let li = self.class_layers[class][idx / kpl] as usize;
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
            let expert = c as u32;
            if snapshot.excludes(li as u32, expert) {
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
                    expert,
                    slot,
                },
                slot,
            ));
        }
        // Budget exhausted without a victim: park the hand past the examined
        // window so a retry sweep continues the scan instead of re-walking it.
        hand.hand = (start + SWEEP_BUDGET.min(class_key_count)) % class_key_count;
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
    pub(crate) fn finish_fill(&self, layer: u32, expert: u32) {
        let mut sh = self.shards[layer as usize].shard.lock().unwrap();
        let i = self.local(expert);
        Self::bit_clear(&mut sh.filling, i);
        self.shards[layer as usize].fill_cv.notify_all();
    }

    /// Drops one lease. The slot stays resident until evicted via `claim`.
    pub fn release_lease(&self, layer: u32, expert: u32) {
        let mut sh = self.shards[layer as usize].shard.lock().unwrap();
        let i = self.local(expert);
        debug_assert!(sh.leases[i] > 0, "release without a lease");
        sh.leases[i] = sh.leases[i].saturating_sub(1);
    }

    /// Abandon a failed page-in: drops the claim's lease and, when no other
    /// lease remains, clears the key's residency and returns the slot to the
    /// global free list — no ghost residency, no leaked slot. Also clears the
    /// fill gate and wakes claimers parked on it (they re-claim fresh).
    pub fn cancel_claim(&self, layer: u32, expert: u32) {
        let li = layer as usize;
        let freed = {
            let mut sh = self.shards[li].shard.lock().unwrap();
            let i = self.local(expert);
            debug_assert!(sh.leases[i] > 0, "cancel without a lease");
            Self::bit_clear(&mut sh.filling, i);
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
            self.free_slots.lock().unwrap()[self.class_of_layer[li] as usize].push(s);
        }
        self.shards[li].fill_cv.notify_all();
    }
}

/// P3 slow-path counters (design §6.6: demand and prefetch counted
/// separately).
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
    #[cfg(not(loom))] // loom primitives require a model() context
    fn claim_flip_then_resident() {
        let m = ResidencyMap::new(vec![0; 2], 8, vec![0..4]);
        assert!(!m.is_resident(0, 3));
        let snap = PolicySnapshot::test_empty();
        match m.claim(0, 3, &snap).unwrap() {
            Claim::Filled { slot, .. } => assert_eq!(slot, 0),
            Claim::Resident { .. } => panic!("first claim must fill"),
        }
        assert!(m.is_resident(0, 3));
        m.release_lease(0, 3);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn second_claim_while_leased_is_a_hit_with_new_lease() {
        let m = ResidencyMap::new(vec![0; 2], 8, vec![0..4]);
        let snap = PolicySnapshot::test_empty();
        m.claim(0, 3, &snap).unwrap();
        m.finish_fill(0, 3); // model the pager's page-in
        assert!(matches!(
            m.claim(0, 3, &snap).unwrap(),
            Claim::Resident { .. }
        ));
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn full_pool_returns_victim_excluding_leased_and_pinned() {
        let m = ResidencyMap::new(vec![0; 1], 8, vec![0..2]);
        let mut snap = PolicySnapshot::test_empty();
        snap.lease(0, 0); // protected like an outstanding lease
        m.claim(0, 0, &snap).unwrap();
        m.finish_fill(0, 0);
        m.claim(0, 1, &snap).unwrap();
        m.finish_fill(0, 1);
        m.release_lease(0, 0);
        m.release_lease(0, 1);
        snap.lease(0, 0); // simulate an outstanding lease on expert 0
        let ClaimError::Evicted(e) = m.claim(0, 2, &snap).unwrap_err() else {
            panic!("expected eviction")
        };
        assert_eq!(e.victim.expert, 1, "leased expert 0 must never be picked");
        assert!(!m.is_resident(0, 1), "victim bit cleared");
        assert!(m.is_resident(0, 2), "claimer flipped");
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn cancel_returns_slot_and_clears_residency() {
        let m = ResidencyMap::new(vec![0; 1], 8, vec![0..1]);
        let snap = PolicySnapshot::test_empty();
        m.claim(0, 0, &snap).unwrap();
        m.cancel_claim(0, 0);
        assert!(!m.is_resident(0, 0));
        // The freed slot serves the next miss.
        match m.claim(0, 1, &snap).unwrap() {
            Claim::Filled { slot, .. } => assert_eq!(slot, 0),
            Claim::Resident { .. } => panic!("slot must have been recycled"),
        }
    }

    #[test]
    fn concurrent_claim_two_threads_one_pagein() {
        primitives::model(|| {
            let m = Arc::new(ResidencyMap::new(vec![0; 1], 8, vec![0..2]));
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
                    let slot = match m.claim(0, 5, &snap).unwrap() {
                        Claim::Filled { slot, .. } => {
                            m.finish_fill(0, 5);
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
            let m = Arc::new(ResidencyMap::new(vec![0; 1], 4, vec![0..2]));
            let snap = Arc::new(PolicySnapshot::test_empty());
            // Fill the pool: keys 0 and 1.
            for e in [0u32, 1] {
                m.claim(0, e, &snap).unwrap();
                m.finish_fill(0, e);
                m.release_lease(0, e);
            }
            let m2 = m.clone();
            let snap2 = snap.clone();
            let t = primitives::spawn(move || {
                // Claim key 2; the pool is full, so this either evicts or
                // consumes a slot the main thread's cancel just freed.
                match m2.claim(0, 2, &snap2) {
                    Ok(Claim::Filled { .. }) | Err(_) => {
                        m2.finish_fill(0, 2);
                        m2.cancel_claim(0, 2);
                    }
                    Ok(Claim::Resident { .. }) => panic!("key 2 starts non-resident"),
                }
            });
            // Re-claim key 1 so both threads contend for the same slots; any
            // outcome is legal (Resident, Filled, or Evicted with key 0 or 1
            // as the victim). cancel_claim drops the lease and the residency,
            // returning the slot — so afterwards at most key 0 is resident
            // and at least one slot is free.
            match m.claim(0, 1, &snap) {
                Ok(Claim::Resident { .. }) => {}
                Ok(Claim::Filled { .. }) | Err(_) => m.finish_fill(0, 1),
            }
            m.cancel_claim(0, 1);
            t.join().unwrap();
            // No slot leaked: a fresh claim must not evict.
            let claim = m.claim(0, 3, &snap).unwrap();
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
        let m = ResidencyMap::new(vec![0; 1], 4, vec![0..3]);
        let snap = PolicySnapshot::test_empty();
        // Warm: hot, cA, cB resident (slots 0-2).
        for e in [2u32, 0, 1] {
            let _ = m.claim(0, e, &snap).unwrap();
            m.finish_fill(0, e);
            m.release_lease(0, e);
        }
        let colds: [u32; 3] = [3, 0, 1]; // cC, then cA, then cB rotate
        for round in 0..30u32 {
            let _ = m.claim(0, 2, &snap).unwrap(); // hot
            m.finish_fill(0, 2);
            m.release_lease(0, 2);
            let c = colds[(round % 3) as usize];
            if let Err(ClaimError::Evicted(evicted)) = m.claim(0, c, &snap) {
                assert_ne!(
                    evicted.victim.expert, 2,
                    "round {round}: the hot expert was evicted"
                );
                m.finish_fill(0, c);
                m.release_lease(0, c);
            }
        }
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn pin_and_lease_exclude_real_keys_at_every_layer() {
        // Regression: snapshot keys must match real keys at every layer — a
        // pin/lease on any protected unit must be invisible to victim picks.
        // Snapshot-lease, layer 1, expert 0:
        let m = ResidencyMap::new(vec![0; 2], 4, vec![0..2]); // global pool: 2 slots total
        let mut snap = PolicySnapshot::test_empty();
        snap.lease(1, 0);
        m.claim(1, 0, &snap).unwrap();
        m.finish_fill(1, 0);
        m.claim(1, 1, &snap).unwrap();
        m.finish_fill(1, 1);
        m.release_lease(1, 0);
        m.release_lease(1, 1);
        let ClaimError::Evicted(e) = m.claim(1, 2, &snap).unwrap_err() else {
            panic!("expected eviction")
        };
        assert_eq!(
            e.victim.expert, 1,
            "snapshot-leased (1, 0) must never be picked"
        );
        assert!(m.is_resident(1, 0), "lease held");

        // Snapshot lease, layer 0 (the victim scan reaches it from layer 1's
        // claim too — the pool is global):
        let m = ResidencyMap::new(vec![0; 2], 4, vec![0..2]);
        let mut snap = PolicySnapshot::test_empty();
        snap.lease(1, 1);
        m.claim(1, 1, &snap).unwrap();
        m.finish_fill(1, 1);
        m.claim(1, 0, &snap).unwrap();
        m.finish_fill(1, 0);
        m.release_lease(1, 1);
        m.release_lease(1, 0);
        let ClaimError::Evicted(e) = m.claim(1, 2, &snap).unwrap_err() else {
            panic!("expected eviction")
        };
        assert_eq!(
            e.victim.expert, 0,
            "snapshot-leased (1, 1) must never be picked"
        );
        assert!(m.is_resident(1, 1), "lease held");
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn claim_on_fully_protected_pool_returns_busy_not_infinite_spin() {
        let m = ResidencyMap::new(vec![0; 1], 2, vec![0..2]);
        let mut snap = PolicySnapshot::test_empty();
        // Fill both slots and protect every resident.
        for e in [0u32, 1] {
            m.claim(0, e, &snap).unwrap();
            m.finish_fill(0, e);
            m.release_lease(0, e);
            snap.lease(0, e);
        }
        // No free slot, no evictable victim: bounded walks, then a typed
        // busy error instead of spinning forever.
        assert_eq!(m.claim(0, 2, &snap), Err(ClaimError::Busy));
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn victim_needing_two_clock_laps_is_found_not_busy() {
        // 4 × 3 × 344 = 4128 keys; the only resident is the last key, with
        // its second-chance bit set. Lap 1 clears it, lap 2 takes it:
        // 8256 keys = 9 sweeps of SWEEP_BUDGET, which a one-lap bound
        // (4128 / 1024 + 2 = 6) reported as Busy.
        let m = ResidencyMap::new(vec![0; 4], 344, vec![0..1]);
        let snap = PolicySnapshot::test_empty();
        let last = (3, 343);
        m.claim(last.0, last.1, &snap).unwrap();
        m.finish_fill(last.0, last.1);
        assert!(matches!(
            m.claim(last.0, last.1, &snap).unwrap(),
            Claim::Resident { .. }
        ));
        m.release_lease(last.0, last.1);
        m.release_lease(last.0, last.1);
        let ClaimError::Evicted(e) = m.claim(0, 0, &snap).unwrap_err() else {
            panic!("expected eviction")
        };
        assert_eq!((e.victim.layer, e.victim.expert), last);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn saturated_lease_count_is_busy_not_a_wrap() {
        let m = ResidencyMap::new(vec![0; 1], 2, vec![0..2]);
        let snap = PolicySnapshot::test_empty();
        m.claim(0, 0, &snap).unwrap();
        m.finish_fill(0, 0);
        m.shards[0].shard.lock().unwrap().leases[0] = u16::MAX;
        assert_eq!(m.claim(0, 0, &snap), Err(ClaimError::Busy));
        assert_eq!(m.shards[0].shard.lock().unwrap().leases[0], u16::MAX);
    }

    #[test]
    #[cfg(not(loom))] // timing-based; loom explores the same gate in-model
    fn resident_claim_parks_until_fill_completes() {
        use std::sync::mpsc;
        use std::time::Duration;

        let m = Arc::new(ResidencyMap::new(vec![0; 1], 8, vec![0..2]));
        let snap = PolicySnapshot::test_empty();
        assert!(matches!(
            m.claim(0, 5, &snap).unwrap(),
            Claim::Filled { .. }
        ));
        let (tx, rx) = mpsc::channel();
        let m2 = m.clone();
        let t = std::thread::spawn(move || {
            let resident = matches!(m2.claim(0, 5, &snap), Ok(Claim::Resident { slot: 0 }));
            tx.send(()).unwrap();
            resident
        });
        assert!(
            rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "claim must park while the fill is in flight"
        );
        m.finish_fill(0, 5);
        assert!(t.join().unwrap(), "post-fill claim must be Resident");
    }

    #[test]
    #[cfg(not(loom))]
    fn eviction_stays_inside_the_claimers_class() {
        // Layer 0 → class 0 (1 slot), layer 1 → class 1 (1 slot).
        let m = ResidencyMap::new(vec![0, 1], 4, vec![0..1, 1..2]);
        let snap = PolicySnapshot::test_empty();
        let fill = |l, e| {
            let c = m.claim(l, e, &snap);
            m.finish_fill(l, e);
            m.release_lease(l, e);
            c
        };
        assert!(matches!(fill(0, 0), Ok(Claim::Filled { slot: 0, .. })));
        assert!(matches!(fill(1, 0), Ok(Claim::Filled { slot: 1, .. })));
        // Class 0 is full: the victim must be layer 0's key, never layer 1's.
        let Err(ClaimError::Evicted(e)) = fill(0, 1) else {
            panic!("expected eviction")
        };
        assert_eq!((e.victim.layer, e.slot), (0, 0));
        assert!(m.is_resident(1, 0));
    }

    #[test]
    #[cfg(not(loom))]
    fn a_free_slot_in_another_class_is_not_taken() {
        let m = ResidencyMap::new(vec![0, 1], 4, vec![0..1, 1..3]);
        let snap = PolicySnapshot::test_empty();
        m.claim(0, 0, &snap).unwrap();
        m.finish_fill(0, 0);
        m.release_lease(0, 0);
        // Class 1 has two free slots; class 0 must still evict, not borrow.
        assert!(matches!(m.claim(0, 1, &snap), Err(ClaimError::Evicted(_))));
    }

    #[test]
    fn concurrent_claims_in_different_classes_never_share_a_slot() {
        primitives::model(|| {
            let m = Arc::new(ResidencyMap::new(vec![0, 1], 2, vec![0..1, 1..2]));
            let snap = Arc::new(PolicySnapshot::test_empty());
            let (m2, s2) = (m.clone(), snap.clone());
            let t = primitives::spawn(move || match m2.claim(1, 0, &s2).unwrap() {
                Claim::Filled { slot, .. } => slot,
                Claim::Resident { .. } => unreachable!(),
            });
            let mine = match m.claim(0, 0, &snap).unwrap() {
                Claim::Filled { slot, .. } => slot,
                Claim::Resident { .. } => unreachable!(),
            };
            assert_eq!((mine, t.join().unwrap()), (0, 1));
        });
    }

    #[test]
    fn cancel_wakes_parked_claimers_with_recycled_slot() {
        primitives::model(|| {
            let m = Arc::new(ResidencyMap::new(vec![0; 1], 8, vec![0..2]));
            let m2 = m.clone();
            let snap = Arc::new(PolicySnapshot::test_empty());
            let snap2 = snap.clone();
            // Main holds key 5 mid-fill; the spawned claimer either parks on
            // the gate (the cancel wakes it and it claims the recycled slot
            // fresh) or runs entirely after the cancel — both end Filled on
            // the one slot.
            assert!(matches!(
                m.claim(0, 5, &snap).unwrap(),
                Claim::Filled { .. }
            ));
            let t = primitives::spawn(move || {
                // Filled on whichever slot was free when it re-claimed.
                matches!(m2.claim(0, 5, &snap2), Ok(Claim::Filled { .. }))
            });
            m.cancel_claim(0, 5);
            assert!(t.join().unwrap());
        });
    }
}
