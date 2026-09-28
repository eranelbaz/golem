//! Residency map + P3 counters (§6.6): (layer, kind, expert) → slot with
//! in-flight leases and atomic claim-time eviction.

use std::collections::HashSet;

use super::primitives::{AtomicU64, Mutex};

/// Which of an expert layer's three planes (§3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertKind {
    /// `ffn_gate_exps`.
    Gate,
    /// `ffn_up_exps`.
    Up,
    /// `ffn_down_exps`.
    Down,
}

impl ExpertKind {
    /// All kinds, in ordinal order.
    pub const ALL: [ExpertKind; 3] = [Self::Gate, Self::Up, Self::Down];

    /// Parses from a ggml tensor name: runtime ggml names
    /// (`...ffn_gate_exps`) and GGUF-side names (`...ffn_gate_exps.weight`)
    /// both match; every other name is not an expert plane.
    pub fn from_tensor_name(name: &str) -> Option<Self> {
        if name.contains("ffn_gate_exps") {
            Some(Self::Gate)
        } else if name.contains("ffn_up_exps") {
            Some(Self::Up)
        } else if name.contains("ffn_down_exps") {
            Some(Self::Down)
        } else {
            None
        }
    }

    fn ordinal(self) -> usize {
        self as usize
    }
}

/// Flat pool key for `(layer, kind, expert)`.
pub fn key_index(layer: u32, kind: ExpertKind, expert: u32, n_experts: u32) -> usize {
    (layer as usize * ExpertKind::ALL.len() + kind.ordinal()) * n_experts as usize + expert as usize
}

/// Immutable, pre-materialized policy state consumed by eviction victim picks
/// (§6.6): keys the policy pins (tier floors) and keys protected like
/// outstanding leases. The pool never locks policy state — `claim` receives a
/// snapshot built before the shard lock is taken; the shard lock stays
/// leaf-most.
///
/// Pool-local for M2.3 (`test_empty`/`pin`/`lease` helpers); the M2.2-backed
/// builder lands with the engine wiring.
#[derive(Debug, Clone, Default)]
pub struct PolicySnapshot {
    pinned: HashSet<usize>,
    protected: HashSet<usize>,
}

impl PolicySnapshot {
    /// Empty snapshot: nothing pinned, nothing protected.
    pub fn test_empty() -> Self {
        Self::default()
    }

    /// Floor-pin a key: never picked as an eviction victim.
    pub fn pin(&mut self, layer: u32, kind: ExpertKind, expert: u32) {
        self.pinned.insert(key_index(layer, kind, expert, u32::MAX));
    }

    /// Protect a key exactly like an outstanding map lease would.
    pub fn lease(&mut self, layer: u32, kind: ExpertKind, expert: u32) {
        self.protected
            .insert(key_index(layer, kind, expert, u32::MAX));
    }

    fn excludes(&self, key: usize) -> bool {
        self.pinned.contains(&key) || self.protected.contains(&key)
    }
}

/// One expert layer's residency state (§6.6): resident bits, lease counts,
/// slot assignment, and the layer's slice of the pool slots.
///
/// Slots are partitioned across layer shards at construction (shard `l` owns
/// a contiguous range), so a shard's critical section
/// `{bit flip, victim pick, slot assignment, lease grant/release}` never
/// touches another shard's state and no global free list exists.
#[derive(Debug)]
struct Shard {
    bits: Vec<u64>,
    leases: Vec<u16>,
    slot_of: Vec<i32>,
    free_slots: Vec<u32>,
}

/// Residency map over all expert layers: `(layer, kind, expert) → slot` with
/// in-flight leases and atomic claim-time eviction (§6.6). Per-layer shard
/// locks; the shard lock is the leaf-most lock in the process.
#[derive(Debug)]
pub struct ResidencyMap {
    n_experts: u32,
    shards: Vec<Mutex<Shard>>,
}

/// Claim outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Claim {
    /// Newly assigned slot (bit flipped under the lock); caller fills it
    /// (I/O outside all locks) and copies under the granted lease.
    Filled {
        /// The slot the caller must fill.
        slot: u32,
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
    /// Builds a map over `n_layers` expert layers of `n_experts` experts,
    /// with `pool_slots` arena slots partitioned contiguously across shards.
    ///
    /// # Panics
    /// Panics (debug) when `pool_slots < n_layers` — a shard with no slots
    /// could never satisfy a claim.
    pub fn new(n_layers: u32, n_experts: u32, pool_slots: u32) -> Self {
        debug_assert!(n_layers > 0 && n_experts > 0);
        debug_assert!(pool_slots >= n_layers, "every shard needs ≥1 slot");
        let per = pool_slots / n_layers;
        let shards = (0..n_layers)
            .map(|l| {
                let start = l * per;
                let end = if l + 1 == n_layers {
                    pool_slots
                } else {
                    (l + 1) * per
                };
                let words = Self::keys_per_layer(n_experts).div_ceil(64);
                Mutex::new(Shard {
                    bits: vec![0; words],
                    leases: vec![0; Self::keys_per_layer(n_experts)],
                    slot_of: vec![-1; Self::keys_per_layer(n_experts)],
                    // pop() allocates `start` first.
                    free_slots: (start..end).rev().collect(),
                })
            })
            .collect();
        Self { n_experts, shards }
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
        let sh = self.shards[layer as usize].lock().unwrap();
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
            let mut sh = self.shards[li].lock().unwrap();
            if Self::bit_test(&sh.bits, i) {
                sh.leases[i] += 1;
                return Ok(Claim::Resident {
                    slot: sh.slot_of[i] as u32,
                });
            }
            if let Some(slot) = sh.free_slots.pop() {
                sh.slot_of[i] = slot as i32;
                Self::bit_set(&mut sh.bits, i);
                sh.leases[i] += 1;
                return Ok(Claim::Filled { slot });
            }
            // Pool (shard partition) full: atomic victim pick + clear + reassign.
            let victim = (0..Self::keys_per_layer(self.n_experts)).find(|&c| {
                Self::bit_test(&sh.bits, c)
                    && sh.leases[c] == 0
                    && !snapshot.excludes(key_index(
                        layer,
                        ExpertKind::ALL[c / self.n_experts as usize],
                        c as u32 % self.n_experts,
                        self.n_experts,
                    ))
            });
            let Some(v) = victim else {
                drop(sh);
                super::primitives::yield_now();
                continue;
            };
            Self::bit_clear(&mut sh.bits, v);
            let slot = sh.slot_of[v] as u32;
            sh.slot_of[v] = -1;
            let vkind = ExpertKind::ALL[v / self.n_experts as usize];
            let vexpert = v as u32 % self.n_experts;
            sh.slot_of[i] = slot as i32;
            Self::bit_set(&mut sh.bits, i);
            sh.leases[i] += 1;
            return Err(Evicted {
                victim: Victim {
                    layer,
                    kind: vkind,
                    expert: vexpert,
                    slot,
                },
                slot,
            });
        }
    }

    /// Drops one lease. The slot stays resident until evicted via `claim`.
    pub fn release_lease(&self, layer: u32, kind: ExpertKind, expert: u32) {
        let mut sh = self.shards[layer as usize].lock().unwrap();
        let i = self.local(kind, expert);
        debug_assert!(sh.leases[i] > 0, "release without a lease");
        sh.leases[i] = sh.leases[i].saturating_sub(1);
    }

    /// Abandon a failed page-in: drops the claim's lease and, when no other
    /// lease remains, clears the key's residency and returns the slot to the
    /// shard's free list — no ghost residency, no leaked slot.
    pub fn cancel_claim(&self, layer: u32, kind: ExpertKind, expert: u32) {
        let mut sh = self.shards[layer as usize].lock().unwrap();
        let i = self.local(kind, expert);
        debug_assert!(sh.leases[i] > 0, "cancel without a lease");
        sh.leases[i] -= 1;
        if sh.leases[i] == 0 {
            Self::bit_clear(&mut sh.bits, i);
            let s = sh.slot_of[i];
            sh.slot_of[i] = -1;
            if s >= 0 {
                sh.free_slots.push(s as u32);
            }
        }
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
    fn claim_flip_then_resident() {
        let m = ResidencyMap::new(2, 8, 4);
        assert!(!m.is_resident(0, ExpertKind::Gate, 3));
        let snap = PolicySnapshot::test_empty();
        match m.claim(0, ExpertKind::Gate, 3, &snap).unwrap() {
            Claim::Filled { slot } => assert_eq!(slot, 0),
            Claim::Resident { .. } => panic!("first claim must fill"),
        }
        assert!(m.is_resident(0, ExpertKind::Gate, 3));
        m.release_lease(0, ExpertKind::Gate, 3);
    }

    #[test]
    fn second_claim_while_leased_is_a_hit_with_new_lease() {
        let m = ResidencyMap::new(2, 8, 4);
        let snap = PolicySnapshot::test_empty();
        m.claim(0, ExpertKind::Gate, 3, &snap).unwrap();
        assert!(matches!(
            m.claim(0, ExpertKind::Gate, 3, &snap).unwrap(),
            Claim::Resident { .. }
        ));
    }

    #[test]
    fn full_pool_returns_victim_excluding_leased_and_pinned() {
        let m = ResidencyMap::new(1, 8, 2);
        let mut snap = PolicySnapshot::test_empty();
        snap.pin(0, ExpertKind::Gate, 0); // floor-pinned
        m.claim(0, ExpertKind::Gate, 0, &snap).unwrap();
        m.claim(0, ExpertKind::Gate, 1, &snap).unwrap();
        m.release_lease(0, ExpertKind::Gate, 0);
        m.release_lease(0, ExpertKind::Gate, 1);
        snap.lease(0, ExpertKind::Gate, 0); // simulate an outstanding lease on expert 0
        let e = m.claim(0, ExpertKind::Gate, 2, &snap).unwrap_err();
        assert_eq!(e.victim.expert, 1, "leased expert 0 must never be picked");
        assert!(!m.is_resident(0, ExpertKind::Gate, 1), "victim bit cleared");
        assert!(m.is_resident(0, ExpertKind::Gate, 2), "claimer flipped");
    }

    #[test]
    fn cancel_returns_slot_and_clears_residency() {
        let m = ResidencyMap::new(1, 8, 1);
        let snap = PolicySnapshot::test_empty();
        m.claim(0, ExpertKind::Gate, 0, &snap).unwrap();
        m.cancel_claim(0, ExpertKind::Gate, 0);
        assert!(!m.is_resident(0, ExpertKind::Gate, 0));
        // The freed slot serves the next miss.
        match m.claim(0, ExpertKind::Gate, 1, &snap).unwrap() {
            Claim::Filled { slot } => assert_eq!(slot, 0),
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
            let t = primitives::spawn(move || {
                matches!(
                    m2.claim(0, ExpertKind::Gate, 5, &snap2),
                    Ok(Claim::Filled { slot: 0 }) | Ok(Claim::Resident { slot: 0 })
                )
            });
            assert!(matches!(
                m.claim(0, ExpertKind::Gate, 5, &snap).unwrap(),
                Claim::Filled { slot: 0 } | Claim::Resident { slot: 0 }
            ));
            assert!(t.join().unwrap());
        });
    }
}
