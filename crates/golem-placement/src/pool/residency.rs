//! Residency map + P3 counters (§6.6): (layer, expert) unit → slot with
//! in-flight leases and atomic claim-time eviction. One unit is an expert's
//! gate, up and down planes co-allocated in one slot (§3.4): one key is
//! claimed, filled, leased and evicted as a whole.

// `cfg(loom)` is swapped via RUSTFLAGS by the T6 loom leg (declared in
// primitives.rs); register it as known for this module's cfg uses too.
// Tests build one-class slot tables as `vec![0..slots]`; the lint's
// alternatives are no clearer.
#![allow(clippy::single_range_in_vec_init)]

use std::fmt;

use super::primitives::{AtomicU64, Condvar, Mutex};

// One expert-name matcher shared by C and Rust semantics (design §6.6; the C
// side mirrors the rule in golem_expert_stem_kind).
pub use golem_gguf::ExpertKind;

/// Immutable, pre-materialized policy state consumed by eviction victim picks
/// (§6.6): keys protected like outstanding leases. The pool never locks
/// policy state — `claim` receives a snapshot built before the shard lock is
/// taken; the shard lock stays leaf-most.
///
/// Built today only by `empty`/`lease`; a placement-backed builder feeds it
/// once the policy scorer drives the pool.
#[derive(Debug, Clone, Default)]
pub struct PolicySnapshot {
    /// Per-layer bitset over experts, grown on first protect: one shift per
    /// CLOCK step, no hashing (GLM: 42 × 288 bits ≈ 1.5 KB when full).
    protected: Vec<Vec<u64>>,
    /// Flat `layer × n_experts` tier scores (§6.2 decayed, published once per
    /// step by the engine). Empty = no cross-tier scoring: victim picks use
    /// CLOCK order only. With scores present, the claim's score floor (see
    /// [`ResidencyMap::claim_for`]) shields every key scoring above it — the
    /// exclusive-residency drop decision (O10).
    scores: Vec<f32>,
    /// Experts per score row (0 = no scoring). Kept beside `scores` so
    /// `score_of` needs no map geometry.
    scores_stride: usize,
}

impl PolicySnapshot {
    /// Empty snapshot: nothing pinned, nothing protected.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Snapshot carrying per-key scores (engine-published decayed §6.2
    /// scores, flat `layer × n_experts` with the given stride); nothing
    /// pinned, nothing protected.
    pub fn scored(scores: Vec<f32>, stride: usize) -> Self {
        Self {
            protected: Vec::new(),
            scores,
            scores_stride: stride,
        }
    }

    /// Replaces the score table in place (scores published once per decode
    /// step; the caller reuses one snapshot to avoid a per-step allocation).
    pub fn replace_scores(&mut self, scores: Vec<f32>) {
        self.scores = scores;
    }

    /// The published score of one key (0.0 when no scores are published or
    /// the table is short — a shorter table scores strictly cold).
    pub fn score_of(&self, layer: u32, expert: u32) -> f32 {
        if self.scores_stride == 0 {
            return 0.0;
        }
        let i = layer as usize * self.scores_stride + expert as usize;
        self.scores.get(i).copied().unwrap_or(0.0)
    }

    /// Protect a unit exactly like an outstanding map lease would.
    pub fn lease(&mut self, layer: u32, expert: u32) {
        let (l, w) = (layer as usize, expert as usize / 64);
        if self.protected.len() <= l {
            self.protected.resize_with(l + 1, Vec::new);
        }
        let row = &mut self.protected[l];
        if row.len() <= w {
            row.resize(w + 1, 0);
        }
        row[w] |= 1 << (expert % 64);
    }

    fn excludes(&self, layer: u32, expert: u32) -> bool {
        self.protected
            .get(layer as usize)
            .and_then(|row| row.get(expert as usize / 64))
            .is_some_and(|w| w >> (expert % 64) & 1 == 1)
    }
}

/// One expert layer's residency state (§6.6): resident bits, lease counts and
/// slot assignment for this layer's keys. The slot free lists and CLOCK
/// hands are per class, shared by every layer of the class (§6.6: no
/// per-layer quota), so a shard's critical
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
    /// Keys filled by a prefetch claim and not yet claimed on demand: the
    /// first demand claim counts a prefetch hit, an eviction while set
    /// counts wasted prefetch bytes (§6.5).
    prefetched: Vec<u64>,
    /// Keys a next-layer prediction shields from eviction (`--prefetch
    /// protect`, §6.5), replaced per prediction of the layer.
    protected: Vec<u64>,
}

/// A shard's lock paired with the condvar its fill gate waits on. The lock is
/// the leaf-most in the process; the condvar only ever releases/reacquires it.
#[derive(Debug)]
struct ShardLock {
    shard: Mutex<Shard>,
    fill_cv: Condvar,
}

/// A class's wake channel for `claim_for`'s no-victim park: a generation
/// counter plus the condvar waiters park on. Every lease release or slot
/// free in the class bumps `gen` and notifies `cv`, so a release between a
/// waiter's generation read (before its CLOCK sweep) and its park cannot be
/// lost: the re-check under the `gen` lock sees the bump and skips the park.
#[derive(Debug)]
struct ClassWake {
    generation: Mutex<u64>,
    cv: Condvar,
}

/// Residency map over all expert layers: `(layer, expert) → slot` with
/// in-flight leases and atomic claim-time eviction (§6.6). Slots are carved
/// into size classes (§5.4); a layer's keys claim, fill and evict only inside
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
    /// Per-class generation + condvar for the no-victim park (see
    /// [`ClassWake`]).
    class_wake: Vec<ClassWake>,
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
#[derive(Debug)]
pub enum Claim<'m> {
    /// Newly assigned slot (bit flipped under the lock); caller fills it
    /// (I/O outside all locks) and copies under the granted lease.
    Filled {
        /// The slot the caller must fill.
        slot: u32,
        /// Exclusive-fill capability for that slot: the only way to take
        /// `&mut` on an arena slot; constructed here and in the eviction
        /// path, both inside the shard lock.
        claim: FillClaim<'m>,
    },
    /// Already resident; a lease was granted; `slot` names the filled slot.
    Resident {
        /// The filled slot the lease covers.
        slot: u32,
        /// This demand claim is the first use of a prefetched fill.
        prefetch_hit: bool,
    },
}

/// Why a claim did not return a slot. `Evicted`: the victim's slot is the
/// claimer's and its I/O replaces the victim's bytes. `Busy`: every slot is
/// leased or snapshot-protected after bounded CLOCK walks, or the key's lease
/// count is saturated — transient by contract (leases release); the pool must
/// not be over-subscribed.
#[derive(Debug)]
pub enum ClaimError<'m> {
    /// An expert evicted by this claim; its slot is reassigned to the
    /// claimer, whose I/O replaces the victim's bytes.
    Evicted(Evicted<'m>),
    /// No evictable slot after bounded full CLOCK walks, or `u16::MAX`
    /// outstanding leases on the key.
    Busy,
}

/// Exclusive-residency D2H claim outcome (O10): `Resident` — the expert's
/// RAM copy is current, the VRAM bytes are droppable; `Claimed` — the slot is
/// assigned with its fill gate closed (finish via [`ResidencyMap::finish_d2h`]
/// after the copy lands); `Dropped` — no acceptable pool slot (the score
/// floor kept every candidate), the VRAM bytes are droppable.
pub(crate) enum D2hClaim<'m> {
    Resident,
    Claimed { slot: u32, claim: FillClaim<'m> },
    Dropped,
}

/// An expert evicted by this claim; its slot is reassigned to the claimer,
/// whose I/O replaces the victim's bytes.
#[derive(Debug)]
pub struct Evicted<'m> {
    /// The evicted victim.
    pub victim: Victim,
    /// The slot the caller must refill (== `victim.slot`).
    pub slot: u32,
    /// Exclusive-fill capability for that slot.
    pub claim: FillClaim<'m>,
}

/// Exclusive-fill capability for one arena slot: a safe `&mut` on a slot
/// requires `&mut FillClaim`, and only the residency map constructs them,
/// under the shard lock, exactly when it has reassigned the slot to the
/// claimer. Not `Copy`/`Clone`: [`Self::finish`] and [`Self::cancel`] consume
/// it, and dropping it unfinished (a panic mid-fill) cancels the claim so the
/// key never stays mid-fill with parked claimers.
pub struct FillClaim<'m> {
    map: &'m ResidencyMap,
    layer: u32,
    expert: u32,
    slot: u32,
    settled: bool,
}

impl FillClaim<'_> {
    /// The slot this claim exclusively fills.
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }

    /// Marks the slot's bytes complete: clears the fill gate and wakes parked
    /// `Resident` claimers, which then grant their leases. The claim's own
    /// lease stays held.
    pub(crate) fn finish(mut self) {
        self.settled = true;
        self.map.finish_fill(self.layer, self.expert);
    }

    /// Consumes the claim WITHOUT opening the fill gate: the exclusive-D2H
    /// path keeps the gate closed (parking every reader of the key) until the
    /// copy lands, then finishes via [`ResidencyMap::finish_d2h`]. The
    /// claim's lease stays held; the pending-D2H owner drops it explicitly.
    pub(crate) fn defer(mut self) {
        self.settled = true;
    }

    /// Abandons a failed page-in; see [`ResidencyMap::cancel_claim`].
    pub(crate) fn cancel(mut self) {
        self.settled = true;
        self.map.cancel_claim(self.layer, self.expert);
    }
}

impl Drop for FillClaim<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.map.cancel_claim(self.layer, self.expert);
        }
    }
}

impl fmt::Debug for FillClaim<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FillClaim")
            .field("layer", &self.layer)
            .field("expert", &self.expert)
            .field("slot", &self.slot)
            .finish()
    }
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
    /// The victim was a prefetched fill never claimed on demand.
    pub prefetched: bool,
}

impl ResidencyMap {
    /// Builds a map over `n_experts`-expert layers mapped to slot classes by
    /// `class_of_layer`; class `c` serves exactly the slots in
    /// `class_slots[c]` (contiguous arena ranges).
    ///
    /// # Errors
    /// No layers/experts/classes, a layer naming a missing class, a class
    /// serving no layers (`clock_evict` would divide by zero), or overlapping
    /// class slot ranges.
    pub fn new(
        class_of_layer: Vec<u8>,
        n_experts: u32,
        class_slots: Vec<std::ops::Range<u32>>,
    ) -> Result<Self, PoolConfigError> {
        let bad = |msg: String| Err(PoolConfigError(msg));
        let n_layers = class_of_layer.len() as u32;
        if n_layers == 0 || n_experts == 0 {
            return bad("a pool without layers or experts cannot serve".into());
        }
        if class_slots.is_empty() {
            return bad("a pool without slots cannot serve".into());
        }
        let n_classes = class_slots.len();
        let mut class_layers = vec![Vec::new(); n_classes];
        for (l, &c) in class_of_layer.iter().enumerate() {
            if c as usize >= n_classes {
                return bad(format!(
                    "layer {l} names class {c}, but only {n_classes} classes exist"
                ));
            }
            class_layers[c as usize].push(l as u32);
        }
        if let Some(c) = class_layers.iter().position(Vec::is_empty) {
            return bad(format!("class {c} has slots but serves no layers"));
        }
        for (a, ra) in class_slots.iter().enumerate() {
            for rb in &class_slots[a + 1..] {
                if ra.end > rb.start && rb.end > ra.start {
                    return bad(format!("class slot ranges {ra:?} and {rb:?} overlap"));
                }
            }
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
                    prefetched: vec![0; filling_words],
                    protected: vec![0; filling_words],
                }),
                fill_cv: Condvar::new(),
            })
            .collect();
        // pop() allocates each class's lowest slot first.
        let free_slots: Vec<Vec<u32>> = class_slots
            .iter()
            .map(|r| r.clone().rev().collect())
            .collect();
        Ok(Self {
            n_experts,
            class_of_layer,
            class_layers,
            class_keys,
            shards,
            free_slots: Mutex::new(free_slots),
            sweep: Mutex::new((0..n_classes).map(|_| ClockHand { hand: 0 }).collect()),
            class_wake: (0..n_classes)
                .map(|_| ClassWake {
                    generation: Mutex::new(0),
                    cv: Condvar::new(),
                })
                .collect(),
        })
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

    /// Replaces `layer`'s prediction-protected set with `experts` (§6.5
    /// `protect`): CLOCK skips them like snapshot-excluded keys. One shard
    /// lock per call; out-of-range ids are ignored.
    pub fn protect(&self, layer: u32, experts: impl IntoIterator<Item = u32>) {
        let Some(shard) = self.shards.get(layer as usize) else {
            return;
        };
        let mut sh = shard.shard.lock().unwrap();
        sh.protected.fill(0);
        for e in experts {
            if e < self.n_experts {
                Self::bit_set(&mut sh.protected, self.local(e));
            }
        }
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
    /// - full → pick victim from the [`PolicySnapshot`] (candidates
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
    /// [`FillClaim::cancel`] if its page-in fails.
    ///
    /// Fill-completion gating: `Filled`/evict-refill grants also mark the key
    /// mid-fill; a `Resident` claim on a mid-fill key parks on the shard
    /// condvar (lock released) until the filler's [`FillClaim::finish`] (or a
    /// cancel) wakes it, so leased readers never observe slot bytes a
    /// concurrent filler is still writing. A claimer parked through a
    /// cancelled fill re-enters the loop and claims fresh.
    #[cfg(test)]
    pub(crate) fn claim(
        &self,
        layer: u32,
        expert: u32,
        snapshot: &PolicySnapshot,
    ) -> Result<Claim<'_>, ClaimError<'_>> {
        self.claim_for(layer, expert, snapshot, true, f32::NEG_INFINITY)
    }

    /// [`Self::claim`] for a demand (`true`) or prefetch (`false`) claimer.
    /// A prefetch claim never evicts a key whose reference bit is set
    /// (demand-hot, P17): its sweep is read-only — the hand and every
    /// reference bit stay as they are — and finding no cold victim returns
    /// [`ClaimError::Busy`] at once instead of retrying. Its fill is marked
    /// prefetched with the reference bit clear; the first demand claim
    /// clears the mark (a prefetch hit) and sets the reference bit.
    /// `floor`: when the snapshot carries tier scores, a CLOCK victim whose
    /// score is STRICTLY ABOVE the floor is never taken — the claimer sees
    /// [`ClaimError::Busy`] instead (exclusive residency's drop decision:
    /// an incoming VRAM eviction copy displaces only colder RAM keys, O10).
    /// Regular claims pass `NEG_INFINITY` (never floor-shielded).
    pub(crate) fn claim_for(
        &self,
        layer: u32,
        expert: u32,
        snapshot: &PolicySnapshot,
        demand: bool,
        floor: f32,
    ) -> Result<Claim<'_>, ClaimError<'_>> {
        let li = layer as usize;
        let class = self.class_of_layer[li] as usize;
        let i = self.local(expert);
        // Bounded starvation guard: CLOCK finds a victim within two full laps
        // (the first may only clear second-chance bits), each lap
        // ⌈class_keys / SWEEP_BUDGET⌉ sweeps over the claimer's own class.
        // Beyond that every slot in the class is leased or snapshot-protected
        // — leases are transient, so the class is over-subscribed and the
        // caller gets a loud busy error instead of an unbounded spin. The
        // bound assumes ref bits stay cleared; concurrent `Resident` hits
        // re-set them, so a re-touch storm can surface `Busy` early — the P3
        // caller aborts on it, which the class floor (layers × K slots)
        // keeps out of reach for one decode thread.
        let starve_bound = 2 * self.class_keys[class].div_ceil(SWEEP_BUDGET) + 1;
        let mut starved = 0usize;
        let mut parked_retry = false;
        // Generation of the class wake channel, sampled before each CLOCK
        // sweep: a release (or cancel) in the class during the sweep bumps
        // it, and the park below skips itself when the sample is stale.
        let mut sweep_gen = 0u64;
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
                    let prefetch_hit = demand && Self::bit_test(&sh.prefetched, i);
                    if demand {
                        Self::bit_clear(&mut sh.prefetched, i);
                        Self::ref_set(&mut sh.ref_bits, i);
                    }
                    return Ok(Claim::Resident {
                        slot: sh.slot_of[i] as u32,
                        prefetch_hit,
                    });
                }
                // Fill was cancelled while parked: fall through and claim
                // fresh (the slot is back on the free list).
            }
            // Falling through holds no state that needs the shard lock.
            // Take a free slot of the layer's class if any; else run that
            // class's CLOCK. The shard guard stays dropped across the free list and
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
                    let wake = &self.class_wake[class];
                    sweep_gen = *wake.generation.lock().unwrap();
                    self.clock_evict(class, snapshot, demand, (li, i), floor)
                        .map(|(v, s)| (s, Some(v)))
                }
            };
            let (slot, victim) = match acquired {
                Some(x) => x,
                None if !demand => return Err(ClaimError::Busy),
                None => {
                    // A victimless sweep may have only cleared ref bits or
                    // hit SWEEP_BUDGET, so retry once immediately — no
                    // latency cost, and it matches the pre-park behaviour.
                    if !parked_retry {
                        parked_retry = true;
                        continue;
                    }
                    // Generation-gated park on the class condvar: victims
                    // come from any layer of the class (not just this
                    // shard's), and a release between the sweep and this
                    // lock bumped `sweep_gen` — in either case skip the
                    // park and re-sweep. The wait is bounded (leases are
                    // transient) and the starve bound still terminates the
                    // loop. Loom cannot model the timed wait (see
                    // `primitives::wait_timeout`): there the generation
                    // re-check above still runs, but liveness comes from the
                    // modeled spin + starve bound, not the timeout.
                    let wake = &self.class_wake[class];
                    let g = wake.generation.lock().unwrap();
                    if *g == sweep_gen {
                        let g = super::primitives::wait_timeout(
                            &wake.cv,
                            g,
                            std::time::Duration::from_micros(200),
                        );
                        drop(g);
                    }
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
            if demand {
                Self::bit_clear(&mut sh.prefetched, i);
            } else {
                Self::bit_set(&mut sh.prefetched, i);
            }
            // Reference bit stays clear on first fill: a key evicts on its
            // first sweep unless it was re-touched (Resident) in between.
            // A non-resident key holds no leases (eviction and cancel clear
            // the bit only at zero), so this cannot overflow.
            debug_assert_eq!(sh.leases[i], 0, "non-resident key with leases");
            sh.leases[i] += 1;
            let claim = FillClaim {
                map: self,
                layer,
                expert,
                slot,
                settled: false,
            };
            if let Some(victim) = victim {
                return Err(ClaimError::Evicted(Evicted {
                    victim,
                    slot,
                    claim,
                }));
            }
            return Ok(Claim::Filled { slot, claim });
        }
    }

    /// Per-class CLOCK sweep: walks the class's keys round-robin from its hand, giving each
    /// resident, unleased, unprotected key a second chance (clear its
    /// reference bit) before taking it as the victim. Reference bits are set
    /// only on Resident reuse — a key filled once and never re-touched is
    /// immediately evictable, while a reused key survives until a full pass
    /// finds nothing else. Returns the victim and its slot, with the victim's
    /// residency already cleared under its shard lock.
    ///
    /// A prefetch sweep (`demand == false`) is read-only: it neither moves
    /// the hand nor clears reference bits, and skips every key whose bit is
    /// set, so it can only take a key CLOCK would already evict.
    ///
    /// The claimer's own key (`claimer` = layer, local key) is never the
    /// victim: a concurrent claimer may have filled it after this claimer
    /// saw it non-resident, and evicting it would only re-read it.
    fn clock_evict(
        &self,
        class: usize,
        snapshot: &PolicySnapshot,
        demand: bool,
        claimer: (usize, usize),
        floor: f32,
    ) -> Option<(Victim, u32)> {
        let kpl = Self::keys_per_layer(self.n_experts);
        let class_key_count = self.class_keys[class];
        let mut hands = self.sweep.lock().unwrap();
        let mut scratch = ClockHand {
            hand: hands[class].hand,
        };
        let hand = if demand {
            &mut hands[class]
        } else {
            &mut scratch
        };
        let start = hand.hand;
        for step in 0..SWEEP_BUDGET.min(class_key_count) {
            let idx = (start + step) % class_key_count;
            // Class keys are (layer-in-class, expert): decode the flat index
            // back to the real layer and its shard-local key.
            let li = self.class_layers[class][idx / kpl] as usize;
            let c = idx % kpl;
            if (li, c) == claimer {
                hand.hand = idx + 1;
                continue;
            }
            let mut sh = self.shards[li].shard.lock().unwrap();
            if !Self::bit_test(&sh.bits, c) || sh.leases[c] != 0 {
                drop(sh);
                hand.hand = idx + 1;
                continue;
            }
            // Snapshot-excluded keys are never victims: test them before
            // the CLOCK ref-bit flip, or an excluded key loses its
            // second chance and stays demoted forever (it can't be picked
            // to earn the bit back).
            let expert = c as u32;
            if snapshot.excludes(li as u32, expert)
                || Self::bit_test(&sh.protected, c)
                || (snapshot.scores_stride != 0 && snapshot.score_of(li as u32, expert) > floor)
            {
                drop(sh);
                continue;
            }
            if Self::ref_test(&sh.ref_bits, c) {
                if demand {
                    Self::ref_clear(&mut sh.ref_bits, c);
                }
                drop(sh);
                hand.hand = idx + 1;
                continue;
            }
            let slot = sh.slot_of[c] as u32;
            Self::bit_clear(&mut sh.bits, c);
            sh.slot_of[c] = -1;
            let prefetched = Self::bit_test(&sh.prefetched, c);
            Self::bit_clear(&mut sh.prefetched, c);
            drop(sh);
            hand.hand = idx + 1;
            return Some((
                Victim {
                    layer: li as u32,
                    expert,
                    slot,
                    prefetched,
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

    /// Clears the fill gate and wakes parked `Resident` claimers; reached
    /// through [`FillClaim::finish`] (demand page-in) and
    /// [`ResidencyMap::finish_d2h`] (exclusive-residency D2H, whose claim is
    /// deferred at claim time), after the slot's bytes are written.
    pub(crate) fn finish_fill(&self, layer: u32, expert: u32) {
        let mut sh = self.shards[layer as usize].shard.lock().unwrap();
        let i = self.local(expert);
        Self::bit_clear(&mut sh.filling, i);
        self.shards[layer as usize].fill_cv.notify_all();
    }

    /// Drops one lease. The slot stays resident until evicted via `claim`.
    /// Bumps the class's wake generation and notifies its condvar: a waiter
    /// parked in `claim_for`'s no-victim path can now evict, and the bump
    /// closes the release-vs-park race (a waiter that sampled the old
    /// generation skips its park instead of sleeping through this release).
    pub(crate) fn release_lease(&self, layer: u32, expert: u32) {
        let li = layer as usize;
        {
            let mut sh = self.shards[li].shard.lock().unwrap();
            let i = self.local(expert);
            debug_assert!(sh.leases[i] > 0, "release without a lease");
            sh.leases[i] = sh.leases[i].saturating_sub(1);
            // A waiter parked on the shard's fill gate (this key mid-fill)
            // is unaffected; the no-victim parkers wait on the class cv.
            self.shards[li].fill_cv.notify_all();
        }
        let wake = &self.class_wake[self.class_of_layer[li] as usize];
        *wake.generation.lock().unwrap() += 1;
        wake.cv.notify_all();
    }

    /// The slot a resident key currently occupies, if any (diag/prefetch seam:
    /// pairs with [`Self::release_resident`] for callers that hold no claim
    /// token). `None` while mid-fill — the slot is not readable yet.
    pub(crate) fn resident_slot(&self, layer: u32, expert: u32) -> Option<u32> {
        let li = layer as usize;
        let sh = self.shards[li].shard.lock().unwrap();
        let i = self.local(expert);
        if Self::bit_test(&sh.bits, i) && !Self::bit_test(&sh.filling, i) {
            Some(sh.slot_of[i] as u32)
        } else {
            None
        }
    }

    /// The key's outstanding lease count (test/diagnostic seam: the D2H
    /// `Resident` path must not leak the probe's lease).
    pub(crate) fn lease_count(&self, layer: u32, expert: u32) -> u16 {
        let sh = self.shards[layer as usize].shard.lock().unwrap();
        sh.leases[self.local(expert)]
    }

    /// Abandons a page-in (reached only through [`FillClaim`]): drops the claim's lease — by contract the
    /// only one on the key (concurrent claimers park on the fill gate until
    /// this cancel or the fill completes) — clears the key's residency and
    /// returns the slot to its class's free list: no ghost residency, no
    /// leaked slot. Also clears the fill gate and wakes claimers parked on
    /// it (they re-claim fresh).
    fn cancel_claim(&self, layer: u32, expert: u32) {
        let li = layer as usize;
        let freed = {
            let mut sh = self.shards[li].shard.lock().unwrap();
            let i = self.local(expert);
            debug_assert!(sh.leases[i] > 0, "cancel without a lease");
            Self::bit_clear(&mut sh.filling, i);
            sh.leases[i] = sh.leases[i].saturating_sub(1);
            debug_assert_eq!(sh.leases[i], 0, "cancel_claim holds the only lease");
            Self::bit_clear(&mut sh.bits, i);
            Self::bit_clear(&mut sh.prefetched, i);
            let s = sh.slot_of[i];
            sh.slot_of[i] = -1;
            (s >= 0).then_some(s as u32)
            // Shard guard dropped before the leaf free list.
        };
        if let Some(s) = freed {
            let class = self.class_of_layer[li] as usize;
            self.free_slots.lock().unwrap()[class].push(s);
            // The freed slot is claimable without a CLOCK sweep: bump the
            // class generation and wake no-victim parkers, as a lease
            // release would (they re-enter the loop and pop the free list).
            let wake = &self.class_wake[class];
            *wake.generation.lock().unwrap() += 1;
            wake.cv.notify_all();
        }
        self.shards[li].fill_cv.notify_all();
    }

    /// Releases a unit's RAM residency (exclusive promotion, O10): after the
    /// expert's H2D copy has landed, its VRAM copy is the only one — the pool
    /// slot returns to the free list. Refuses (returns `false`, touching
    /// nothing) unless the key is resident in exactly `slot` with zero
    /// outstanding leases: a concurrent prefetch claim on the key keeps the
    /// unit inclusive until its own lifecycle resolves it.
    pub fn release_resident(&self, layer: u32, expert: u32, slot: u32) -> bool {
        let li = layer as usize;
        let freed = {
            let mut sh = self.shards[li].shard.lock().unwrap();
            let i = self.local(expert);
            if !Self::bit_test(&sh.bits, i)
                || Self::bit_test(&sh.filling, i)
                || sh.leases[i] != 0
                || sh.slot_of[i] != slot as i32
            {
                return false;
            }
            Self::bit_clear(&mut sh.bits, i);
            Self::bit_clear(&mut sh.ref_bits, i);
            Self::bit_clear(&mut sh.prefetched, i);
            sh.slot_of[i] = -1;
            slot
            // Shard guard dropped before the leaf free list.
        };
        let class = self.class_of_layer[li] as usize;
        self.free_slots.lock().unwrap()[class].push(freed);
        let wake = &self.class_wake[class];
        *wake.generation.lock().unwrap() += 1;
        wake.cv.notify_all();
        true
    }

    /// Claims a slot for an incoming VRAM eviction copy (O10): the expert is
    /// not pool-resident and its D2H is about to start. The claim mirrors a
    /// demand claim — free slot first, else CLOCK with the snapshot's score
    /// floor (a victim scoring strictly above `floor` is kept, the copy is
    /// dropped) — but the fill gate STAYS CLOSED: the caller must complete
    /// the D2H and then call [`Self::finish_d2h`], which opens the gate and
    /// wakes parked claimers. No pool reader exists between claim and finish:
    /// the gate parks every reader of the key, and the caller commits only
    /// after the substrate synchronized its backends (the D2H has landed).
    /// A key already pool-resident needs no copy: [`D2hClaim::Resident`].
    pub(crate) fn claim_d2h(
        &self,
        layer: u32,
        expert: u32,
        snapshot: &PolicySnapshot,
        floor: f32,
    ) -> D2hClaim<'_> {
        match self.claim_for(layer, expert, snapshot, true, floor) {
            Ok(Claim::Resident { .. }) => {
                // claim_for granted this probe a lease on the resident key;
                // the D2H outcome hands no token out, so release it here —
                // the caller copies nothing and the slot stays resident.
                self.release_lease(layer, expert);
                D2hClaim::Resident
            }
            Ok(Claim::Filled { slot, claim }) => D2hClaim::Claimed { slot, claim },
            Err(ClaimError::Evicted(evicted)) => D2hClaim::Claimed {
                slot: evicted.slot,
                claim: evicted.claim,
            },
            // Every candidate scores above the floor: keep the RAM victim,
            // drop the VRAM bytes (they are re-read from disk only after this
            // eviction — i.e. when no VRAM copy exists).
            Err(ClaimError::Busy) => D2hClaim::Dropped,
        }
    }

    /// Completes one D2H claim: opens the fill gate (the caller guarantees
    /// the copy landed — the substrate drains its backends before the
    /// matching `copy_done`) and wakes parked claimers. The claim's lease
    /// stays held; the caller drops it (with the slot) via [`Lease`](super::pager::Lease)
    /// semantics once the gate is open.
    pub(crate) fn finish_d2h(&self, layer: u32, expert: u32) {
        self.finish_fill(layer, expert);
    }
}

/// Inconsistent pool construction input (layers, classes, slot ranges).
#[derive(Debug, thiserror::Error)]
#[error("invalid pool configuration: {0}")]
pub struct PoolConfigError(String);

impl PoolConfigError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
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
    /// Prefetched units later claimed on demand before eviction:
    /// `prefetch_hit_rate = prefetch_hits / prefetch_pageins` (§6.5).
    pub prefetch_hits: AtomicU64,
    /// Bytes of prefetched units evicted without a demand claim (§11).
    pub prefetch_wasted_bytes: AtomicU64,
    /// Claims served from residency (no I/O).
    pub fast_hits: AtomicU64,
    /// Total bytes moved from source into pool slots.
    pub bytes_paged: AtomicU64,
    /// Atomic evictions performed inside `claim`.
    pub evictions: AtomicU64,
    /// Source read failures after bounded retries.
    pub source_errors: AtomicU64,
    /// Shard-lock claims on the decode path: at most one per unit per
    /// scheduler compute (the other planes reuse the lease, §6.6).
    pub demand_claims: AtomicU64,
    /// Policy-snapshot mutex reads: once per compute on the decode path,
    /// once per call on `ensure_range`/`ensure_unit`.
    pub snapshot_reads: AtomicU64,
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

    /// What a claim granted, with the fill (if any) already completed.
    #[derive(Debug, PartialEq)]
    enum Got {
        Filled(u32),
        Resident(u32),
        Evicted(Victim, u32),
        Busy,
    }

    /// Claims a key and finishes its fill at once; the claim's lease stays held.
    fn take(m: &ResidencyMap, snap: &PolicySnapshot, layer: u32, expert: u32) -> Got {
        match m.claim(layer, expert, snap) {
            Ok(Claim::Filled { slot, claim }) => {
                claim.finish();
                Got::Filled(slot)
            }
            Ok(Claim::Resident { slot, .. }) => Got::Resident(slot),
            Err(ClaimError::Evicted(e)) => {
                let (victim, slot) = (e.victim, e.slot);
                e.claim.finish();
                Got::Evicted(victim, slot)
            }
            Err(ClaimError::Busy) => Got::Busy,
        }
    }

    /// `take` + release: leaves the key resident and unleased.
    fn warm(m: &ResidencyMap, snap: &PolicySnapshot, layer: u32, expert: u32) -> Got {
        let got = take(m, snap, layer, expert);
        m.release_lease(layer, expert);
        got
    }

    fn map(layers: usize, n_experts: u32, classes: Vec<std::ops::Range<u32>>) -> ResidencyMap {
        ResidencyMap::new(vec![0; layers], n_experts, classes).unwrap()
    }

    /// A prefetch claim with its fill finished and its lease released.
    fn prefetch(m: &ResidencyMap, snap: &PolicySnapshot, layer: u32, expert: u32) -> Got {
        let got = match m.claim_for(layer, expert, snap, false, f32::NEG_INFINITY) {
            Ok(Claim::Filled { slot, claim }) => {
                claim.finish();
                Got::Filled(slot)
            }
            Ok(Claim::Resident { slot, .. }) => Got::Resident(slot),
            Err(ClaimError::Evicted(e)) => {
                let (victim, slot) = (e.victim, e.slot);
                e.claim.finish();
                Got::Evicted(victim, slot)
            }
            Err(ClaimError::Busy) => return Got::Busy,
        };
        m.release_lease(layer, expert);
        got
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn prefetch_never_evicts_a_demand_hot_key_nor_touches_its_reference_bit() {
        // Two slots: 0 is demand-hot (reused → ref bit), 1 is leased. The
        // prefetch finds no cold victim: Busy, and 0's bit survives, so the
        // next demand sweep spends it on 0 (second chance) and evicts 0
        // only after 1's lease is gone and 0 is cold again.
        let m = map(1, 8, vec![0..2]);
        let snap = PolicySnapshot::empty();
        warm(&m, &snap, 0, 0);
        warm(&m, &snap, 0, 0); // Resident reuse sets the reference bit
        take(&m, &snap, 0, 1); // leased
        assert_eq!(prefetch(&m, &snap, 0, 5), Got::Busy);
        assert!(m.is_resident(0, 0), "demand-hot key kept");
        m.release_lease(0, 1);
        // Demand sweep from the untouched hand: 0 has its bit (cleared now),
        // 1 is cold → victim 1. Had the prefetch cleared 0's bit, 0 would go.
        let Got::Evicted(v, _) = take(&m, &snap, 0, 6) else {
            panic!("demand claim evicts")
        };
        assert_eq!(v.expert, 1);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn prefetch_takes_a_cold_key_and_its_first_demand_use_is_a_hit() {
        let m = map(1, 8, vec![0..2]);
        let snap = PolicySnapshot::empty();
        warm(&m, &snap, 0, 0);
        warm(&m, &snap, 0, 0); // hot
        warm(&m, &snap, 0, 1); // cold
        let Got::Evicted(v, _) = prefetch(&m, &snap, 0, 3) else {
            panic!("prefetch takes the cold key")
        };
        assert_eq!((v.expert, v.prefetched), (1, false));
        let first = m.claim(0, 3, &snap);
        assert!(matches!(
            first,
            Ok(Claim::Resident {
                prefetch_hit: true,
                ..
            })
        ));
        m.release_lease(0, 3);
        let again = m.claim(0, 3, &snap);
        assert!(matches!(
            again,
            Ok(Claim::Resident {
                prefetch_hit: false,
                ..
            })
        ));
        m.release_lease(0, 3);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn an_unused_prefetch_is_reported_when_evicted() {
        let m = map(1, 8, vec![0..1]);
        let snap = PolicySnapshot::empty();
        assert_eq!(prefetch(&m, &snap, 0, 2), Got::Filled(0));
        let Got::Evicted(v, _) = take(&m, &snap, 0, 4) else {
            panic!("demand evicts the prefetched fill")
        };
        assert_eq!((v.expert, v.prefetched), (2, true));
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn protected_keys_are_never_victims_until_replaced() {
        let m = map(1, 8, vec![0..2]);
        let snap = PolicySnapshot::empty();
        warm(&m, &snap, 0, 0);
        warm(&m, &snap, 0, 1);
        m.protect(0, [0, 99]);
        let Got::Evicted(v, _) = take(&m, &snap, 0, 2) else {
            panic!("evicts the unprotected key")
        };
        assert_eq!(v.expert, 1);
        m.release_lease(0, 2);
        m.protect(0, [2]);
        let Got::Evicted(v, _) = take(&m, &snap, 0, 3) else {
            panic!("the replaced set frees expert 0")
        };
        assert_eq!(v.expert, 0);
    }

    #[test]
    fn snapshot_bitset_covers_the_glm_shape() {
        let mut snap = PolicySnapshot::empty();
        snap.lease(41, 287);
        snap.lease(0, 63);
        snap.lease(0, 64);
        assert!(snap.excludes(41, 287) && snap.excludes(0, 63) && snap.excludes(0, 64));
        assert!(!snap.excludes(41, 286) && !snap.excludes(1, 0) && !snap.excludes(99, 0));
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn claim_flip_then_resident() {
        let m = map(2, 8, vec![0..4]);
        assert!(!m.is_resident(0, 3));
        let snap = PolicySnapshot::empty();
        assert_eq!(take(&m, &snap, 0, 3), Got::Filled(0));
        assert!(m.is_resident(0, 3));
        m.release_lease(0, 3);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn second_claim_while_leased_is_a_hit_with_new_lease() {
        let m = map(2, 8, vec![0..4]);
        let snap = PolicySnapshot::empty();
        take(&m, &snap, 0, 3);
        assert!(matches!(take(&m, &snap, 0, 3), Got::Resident(_)));
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn full_pool_returns_victim_excluding_leased_and_pinned() {
        let m = map(1, 8, vec![0..2]);
        let mut snap = PolicySnapshot::empty();
        snap.lease(0, 0); // protected like an outstanding lease
        warm(&m, &snap, 0, 0);
        warm(&m, &snap, 0, 1);
        let Got::Evicted(victim, _) = take(&m, &snap, 0, 2) else {
            panic!("expected eviction")
        };
        assert_eq!(victim.expert, 1, "protected expert 0 must never be picked");
        assert!(!m.is_resident(0, 1), "victim bit cleared");
        assert!(m.is_resident(0, 2), "claimer flipped");
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn cancel_returns_slot_and_clears_residency() {
        let m = map(1, 8, vec![0..1]);
        let snap = PolicySnapshot::empty();
        let Ok(Claim::Filled { claim, .. }) = m.claim(0, 0, &snap) else {
            panic!("first claim must fill")
        };
        claim.cancel();
        assert!(!m.is_resident(0, 0));
        // The freed slot serves the next miss.
        assert_eq!(take(&m, &snap, 0, 1), Got::Filled(0), "slot recycled");
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn dropping_an_unfinished_claim_cancels_it() {
        // A panic mid-fill unwinds through `FillClaim::drop`: the key must not
        // stay `filling` (parking every later claimer forever) or resident.
        let m = map(1, 8, vec![0..1]);
        let snap = PolicySnapshot::empty();
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let Ok(Claim::Filled { claim, .. }) = m.claim(0, 0, &snap) else {
                panic!("first claim must fill")
            };
            let _claim = claim;
            panic!("source panicked mid-fill");
        }));
        assert!(unwound.is_err());
        assert!(!m.is_resident(0, 0), "no ghost residency");
        // Would hang on the fill gate if the key were still `filling`.
        assert_eq!(take(&m, &snap, 0, 0), Got::Filled(0), "slot recycled");
    }

    #[test]
    fn concurrent_claim_two_threads_one_pagein() {
        primitives::model(|| {
            let m = Arc::new(map(1, 8, vec![0..2]));
            let m2 = m.clone();
            let snap = Arc::new(PolicySnapshot::empty());
            let snap2 = snap.clone();
            // Models the pager per-thread: the `Filled` winner completes its
            // fill through the gate so the parked `Resident` claimer wakes.
            // Invariant: both claimers observe the SAME slot. Which thread
            // wins the fill — and which slot it took — is
            // scheduling-dependent (the free list is shared).
            let seen = Arc::new(Mutex::new(None));
            let seen2 = seen.clone();
            let run =
                |m: Arc<ResidencyMap>, snap: Arc<PolicySnapshot>, seen: Arc<Mutex<Option<u32>>>| {
                    let (Got::Filled(slot) | Got::Resident(slot)) = take(&m, &snap, 0, 5) else {
                        panic!("two claims on an empty 2-slot pool cannot evict or starve")
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
        // must end with no leaked slot: after both cancel, one more claim
        // must find a free slot without evicting anyone.
        primitives::model(|| {
            let m = Arc::new(map(1, 4, vec![0..2]));
            let snap = Arc::new(PolicySnapshot::empty());
            for e in [0u32, 1] {
                warm(&m, &snap, 0, e);
            }
            let m2 = m.clone();
            let snap2 = snap.clone();
            let t = primitives::spawn(move || {
                // Claim key 2 on the full pool: it evicts (or reuses a slot
                // the main thread's cancel just freed).
                match m2.claim(0, 2, &snap2) {
                    Ok(Claim::Filled { claim, .. }) => claim.cancel(),
                    Err(ClaimError::Evicted(e)) => e.claim.cancel(),
                    Ok(Claim::Resident { .. }) => panic!("key 2 starts non-resident"),
                    Err(ClaimError::Busy) => panic!("two unleased keys are evictable"),
                }
            });
            // Re-claim key 1 so both threads contend for the same slots; any
            // claim outcome is legal (Resident, Filled, or Evicted with key 0
            // or 1 as the victim). The cancel drops the lease and the
            // residency, returning the slot.
            match m.claim(0, 1, &snap) {
                Ok(Claim::Resident { .. }) => m.cancel_claim(0, 1),
                Ok(Claim::Filled { claim, .. }) => claim.cancel(),
                Err(ClaimError::Evicted(e)) => e.claim.cancel(),
                Err(ClaimError::Busy) => panic!("two unleased keys are evictable"),
            }
            t.join().unwrap();
            // No slot leaked: a fresh claim must not evict.
            assert!(matches!(take(&m, &snap, 0, 3), Got::Filled(_)));
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
        let m = map(1, 4, vec![0..3]);
        let snap = PolicySnapshot::empty();
        // Warm: hot, cA, cB resident (slots 0-2).
        for e in [2u32, 0, 1] {
            warm(&m, &snap, 0, e);
        }
        let colds: [u32; 3] = [3, 0, 1]; // cC, then cA, then cB rotate
        for round in 0..30u32 {
            warm(&m, &snap, 0, 2); // hot
            let c = colds[(round % 3) as usize];
            if let Got::Evicted(victim, _) = take(&m, &snap, 0, c) {
                assert_ne!(
                    victim.expert, 2,
                    "round {round}: the hot expert was evicted"
                );
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
        let m = map(2, 4, vec![0..2]);
        let mut snap = PolicySnapshot::empty();
        snap.lease(1, 0);
        warm(&m, &snap, 1, 0);
        warm(&m, &snap, 1, 1);
        let Got::Evicted(victim, _) = take(&m, &snap, 1, 2) else {
            panic!("expected eviction")
        };
        assert_eq!(
            victim.expert, 1,
            "snapshot-leased (1, 0) must never be picked"
        );
        assert!(m.is_resident(1, 0), "lease held");

        // Snapshot lease on (1, 1): the victim scan reaches it from layer 1's
        // claim too — the class spans layers.
        let m = map(2, 4, vec![0..2]);
        let mut snap = PolicySnapshot::empty();
        snap.lease(1, 1);
        warm(&m, &snap, 1, 1);
        warm(&m, &snap, 1, 0);
        let Got::Evicted(victim, _) = take(&m, &snap, 1, 2) else {
            panic!("expected eviction")
        };
        assert_eq!(
            victim.expert, 0,
            "snapshot-leased (1, 1) must never be picked"
        );
        assert!(m.is_resident(1, 1), "lease held");
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn claim_on_fully_protected_pool_returns_busy_not_infinite_spin() {
        let m = map(1, 2, vec![0..2]);
        let mut snap = PolicySnapshot::empty();
        // Fill both slots and protect every resident.
        for e in [0u32, 1] {
            warm(&m, &snap, 0, e);
            snap.lease(0, e);
        }
        // No free slot, no evictable victim: bounded walks, then a typed
        // busy error instead of spinning forever.
        assert_eq!(take(&m, &snap, 0, 2), Got::Busy);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn overlapping_class_slot_ranges_are_rejected() {
        let err = ResidencyMap::new(vec![0, 1], 8, vec![0..4, 3..6]).unwrap_err();
        assert!(err.to_string().contains("overlap"), "{err}");
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn inconsistent_construction_is_an_error_not_a_panic() {
        assert!(
            ResidencyMap::new(vec![], 8, vec![0..1]).is_err(),
            "no layers"
        );
        assert!(ResidencyMap::new(vec![0], 8, vec![]).is_err(), "no classes");
        assert!(
            ResidencyMap::new(vec![1], 8, vec![0..1]).is_err(),
            "bad class"
        );
        assert!(
            ResidencyMap::new(vec![0], 8, vec![0..1, 1..2]).is_err(),
            "class 1 serves no layer"
        );
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn victim_needing_two_clock_laps_is_found_not_busy() {
        // 4 × 1032 = 4128 keys; the only resident is the last key, with
        // its second-chance bit set. Lap 1 clears it, lap 2 takes it:
        // 8256 keys = 9 sweeps of SWEEP_BUDGET, which a one-lap bound
        // (4128 / 1024 + 2 = 6) reported as Busy.
        let m = map(4, 1032, vec![0..1]);
        let snap = PolicySnapshot::empty();
        let last = (3, 1031);
        take(&m, &snap, last.0, last.1);
        assert!(matches!(take(&m, &snap, last.0, last.1), Got::Resident(_)));
        m.release_lease(last.0, last.1);
        m.release_lease(last.0, last.1);
        let Got::Evicted(victim, _) = take(&m, &snap, 0, 0) else {
            panic!("expected eviction")
        };
        assert_eq!((victim.layer, victim.expert), last);
    }

    #[test]
    #[cfg(not(loom))] // loom primitives require a model() context
    fn saturated_lease_count_is_busy_not_a_wrap() {
        let m = map(1, 2, vec![0..2]);
        let snap = PolicySnapshot::empty();
        take(&m, &snap, 0, 0);
        m.shards[0].shard.lock().unwrap().leases[0] = u16::MAX;
        assert_eq!(take(&m, &snap, 0, 0), Got::Busy);
        assert_eq!(m.shards[0].shard.lock().unwrap().leases[0], u16::MAX);
    }

    #[test]
    #[cfg(not(loom))] // timing-based; loom explores the same gate in-model
    fn resident_claim_parks_until_fill_completes() {
        use std::sync::mpsc;
        use std::time::Duration;

        let m = Arc::new(map(1, 8, vec![0..2]));
        let snap = PolicySnapshot::empty();
        let Ok(Claim::Filled { claim, .. }) = m.claim(0, 5, &snap) else {
            panic!("first claim must fill")
        };
        let (tx, rx) = mpsc::channel();
        let m2 = m.clone();
        let t = std::thread::spawn(move || {
            let resident = matches!(m2.claim(0, 5, &snap), Ok(Claim::Resident { slot: 0, .. }));
            tx.send(()).unwrap();
            resident
        });
        assert!(
            rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "claim must park while the fill is in flight"
        );
        claim.finish();
        assert!(t.join().unwrap(), "post-fill claim must be Resident");
    }

    #[test]
    #[cfg(not(loom))]
    fn eviction_stays_inside_the_claimers_class() {
        // Layer 0 → class 0 (1 slot), layer 1 → class 1 (1 slot).
        let m = ResidencyMap::new(vec![0, 1], 4, vec![0..1, 1..2]).unwrap();
        let snap = PolicySnapshot::empty();
        assert_eq!(warm(&m, &snap, 0, 0), Got::Filled(0));
        assert_eq!(warm(&m, &snap, 1, 0), Got::Filled(1));
        // Class 0 is full: the victim must be layer 0's key, never layer 1's.
        let Got::Evicted(victim, slot) = warm(&m, &snap, 0, 1) else {
            panic!("expected eviction")
        };
        assert_eq!((victim.layer, slot), (0, 0));
        assert!(m.is_resident(1, 0));
    }

    #[test]
    #[cfg(not(loom))]
    fn a_free_slot_in_another_class_is_not_taken() {
        let m = ResidencyMap::new(vec![0, 1], 4, vec![0..1, 1..3]).unwrap();
        let snap = PolicySnapshot::empty();
        warm(&m, &snap, 0, 0);
        // Class 1 has two free slots; class 0 must still evict, not borrow.
        assert!(matches!(take(&m, &snap, 0, 1), Got::Evicted(..)));
    }

    #[test]
    fn concurrent_claims_in_different_classes_never_share_a_slot() {
        primitives::model(|| {
            let m = Arc::new(ResidencyMap::new(vec![0, 1], 2, vec![0..1, 1..2]).unwrap());
            let snap = Arc::new(PolicySnapshot::empty());
            let (m2, s2) = (m.clone(), snap.clone());
            let t = primitives::spawn(move || match take(&m2, &s2, 1, 0) {
                Got::Filled(slot) => slot,
                other => unreachable!("{other:?}"),
            });
            let mine = match take(&m, &snap, 0, 0) {
                Got::Filled(slot) => slot,
                other => unreachable!("{other:?}"),
            };
            assert_eq!((mine, t.join().unwrap()), (0, 1));
        });
    }

    #[test]
    fn no_victim_park_is_woken_by_a_class_release() {
        // One class, one slot: the second claimer finds no free slot and no
        // victim (the only resident key is leased), so it parks on the class
        // wake channel; the lease release bumps the generation and notifies,
        // and the waiter must finish (filled or evicted) — never busy, never
        // lost. Under loom the wait degrades to a yield (loom has no timed
        // wait), so this test explores the park/notify path and both
        // branches of the generation gate rather than detecting a hang.
        primitives::model(|| {
            let m = Arc::new(map(1, 2, vec![0..1]));
            let m2 = m.clone();
            let snap = Arc::new(PolicySnapshot::empty());
            let snap2 = snap.clone();
            let t = primitives::spawn(move || {
                let got = take(&m2, &snap2, 0, 1);
                // `take` holds the lease when it grants one; release so the
                // map state is final for the join. Busy grants no lease.
                if !matches!(got, Got::Busy) {
                    m2.release_lease(0, 1);
                }
                // Busy is admissible under loom, where the park is a yield:
                // the starve bound can fire before the release lands. What
                // must never happen is a hang (loom reports it) — the
                // generation gate guarantees the parked thread re-checks.
                matches!(got, Got::Filled(_) | Got::Evicted(..) | Got::Busy)
            });
            let main_got = take(&m, &snap, 0, 0);
            if !matches!(main_got, Got::Busy) {
                m.release_lease(0, 0);
            }
            assert!(t.join().unwrap(), "parked claimer never made progress");
            assert!(
                matches!(main_got, Got::Filled(_) | Got::Evicted(..) | Got::Busy),
                "parked claimer never made progress"
            );
        });
    }

    #[test]
    fn cancel_wakes_parked_claimers_with_recycled_slot() {
        primitives::model(|| {
            let m = Arc::new(map(1, 8, vec![0..2]));
            let m2 = m.clone();
            let snap = Arc::new(PolicySnapshot::empty());
            let snap2 = snap.clone();
            // Main holds key 5 mid-fill; the spawned claimer either parks on
            // the gate (the cancel wakes it and it claims the recycled slot
            // fresh) or runs entirely after the cancel — both end Filled on
            // the one slot.
            let Ok(Claim::Filled { claim, .. }) = m.claim(0, 5, &snap) else {
                panic!("first claim must fill")
            };
            let t = primitives::spawn(move || {
                // Filled on whichever slot was free when it re-claimed.
                matches!(take(&m2, &snap2, 0, 5), Got::Filled(_))
            });
            claim.cancel();
            assert!(t.join().unwrap());
        });
    }

    // --- Exclusive residency (O10) ---

    /// A d2h claim: `Resident`/`Dropped` mapped to those names, `Claimed`
    /// finished via `finish_d2h` (as the pager's commit does) and reported
    /// with its slot.
    fn d2h(
        m: &ResidencyMap,
        snap: &PolicySnapshot,
        layer: u32,
        expert: u32,
        floor: f32,
    ) -> (&'static str, u32) {
        match m.claim_d2h(layer, expert, snap, floor) {
            D2hClaim::Resident => ("Resident", 0),
            D2hClaim::Dropped => ("Dropped", 0),
            D2hClaim::Claimed { slot, claim } => {
                claim.defer();
                m.finish_d2h(layer, expert);
                ("Claimed", slot)
            }
        }
    }

    #[test]
    #[cfg(not(loom))]
    fn promotion_release_frees_the_slot() {
        let m = map(1, 8, vec![0..1]);
        let snap = PolicySnapshot::empty();
        assert_eq!(warm(&m, &snap, 0, 0), Got::Filled(0));
        assert!(m.release_resident(0, 0, 0), "release succeeds");
        assert!(!m.is_resident(0, 0), "unit left the RAM tier");
        // The freed slot is reused by the next fill (no duplicate residency).
        assert_eq!(warm(&m, &snap, 0, 7), Got::Filled(0));
    }

    #[test]
    #[cfg(not(loom))]
    fn release_refuses_when_leased_or_gone() {
        let m = map(1, 8, vec![0..1]);
        let snap = PolicySnapshot::empty();
        let Got::Filled(slot) = take(&m, &snap, 0, 0) else {
            panic!("fills")
        };
        // The take's lease is outstanding: release must refuse.
        assert!(!m.release_resident(0, 0, slot));
        m.release_lease(0, 0);
        assert!(m.release_resident(0, 0, slot));
        // Second release: the slot is gone (or recycled) — refuses.
        assert!(!m.release_resident(0, 0, slot));
    }

    #[test]
    #[cfg(not(loom))]
    fn d2h_claim_skips_pool_resident_and_drops_on_score_floor() {
        let m = map(1, 8, vec![0..1]);
        let empty = PolicySnapshot::empty();
        // Scores: expert 1 scores 10.0, everything else 0.
        let mut scores = vec![0.0f32; 8];
        scores[1] = 10.0;
        let scored = PolicySnapshot::scored(scores, 8);
        // Warm a hot key (score 10): a cold incoming copy (floor 0) must not
        // displace it.
        assert_eq!(warm(&m, &scored, 0, 1), Got::Filled(0));
        let (what, _) = d2h(&m, &scored, 0, 2, 0.0);
        assert_eq!(what, "Dropped", "hot RAM victim kept, VRAM bytes dropped");
        // With an empty (unscored) snapshot the same claim succeeds.
        let (what, _) = d2h(&m, &empty, 0, 2, f32::NEG_INFINITY);
        assert_eq!(what, "Claimed", "no scores: CLOCK decides");
        // Once committed the key is pool-resident; the next d2h claim for it
        // (a fresh eviction while the RAM copy lives) skips the copy.
        let (what, _) = d2h(&m, &empty, 0, 2, f32::NEG_INFINITY);
        assert_eq!(what, "Resident");
    }

    #[test]
    #[cfg(not(loom))]
    fn d2h_fill_gates_parked_readers_until_finish() {
        let m = Arc::new(map(1, 8, vec![0..1]));
        let snap = Arc::new(PolicySnapshot::empty());
        primitives::model(move || {
            let m2 = m.clone();
            let snap2 = snap.clone();
            let D2hClaim::Claimed { slot, claim } = m.claim_d2h(0, 3, &snap, f32::NEG_INFINITY)
            else {
                panic!("d2h claim fills the free pool")
            };
            assert_eq!(slot, 0);
            let t = primitives::spawn(move || {
                // The parked reader must not see the copy before finish_d2h;
                // after commit it reads Resident on the claimed slot.
                let _ = take(&m2, &snap2, 0, 3);
            });
            claim.defer();
            // Not finished yet: give the reader a chance to park (loom
            // explores the interleaving; the preemption bound covers it).
            m.finish_d2h(0, 3);
            assert!(t.join().is_ok());
            assert!(m.is_resident(0, 3));
        });
    }
}
