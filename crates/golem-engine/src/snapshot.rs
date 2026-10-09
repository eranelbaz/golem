//! Turn-end sequence-state snapshots (§7.1, R-4): recurrent state + KV of a
//! prompt prefill, keyed by its tokens, so the next turn restores the shared
//! prefix and prefills only the delta. The only correct prefix reuse on a
//! recurrent model — no partial rewind exists (§2.5).
//!
//! Storage is two preallocated arenas (state bytes in a [`PinnedArena`],
//! token keys in a flat `Vec<i32>`), entries are byte/token ranges placed
//! first-fit; LRU entries are evicted until the new entry fits, so a
//! snapshot no larger than the arena always lands. No allocation per turn.

use std::ops::Range;

use crate::p3_glue::PinnedArena;

/// Upper bound on live entries (bounds the key arena:
/// `MAX_ENTRIES × n_ctx` tokens). Target: 10.3 GiB / 300 MiB (8k-token
/// snapshot) ≈ 35 entries, so 64 never binds before the byte budget does.
pub const MAX_ENTRIES: usize = 64;

/// Cache counters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotStats {
    /// Lookups that found a restorable prefix.
    pub hits: u64,
    /// Lookups that found none.
    pub misses: u64,
    /// Restores rejected by `llama_state_seq_set_data` (entry dropped).
    pub restore_failures: u64,
    /// Entries evicted to make room.
    pub evictions: u64,
    /// Snapshots larger than the whole arena, never stored.
    pub dropped_oversize: u64,
    /// `llama_state_seq_get_data` wrote fewer bytes than sized; not stored.
    pub short_writes: u64,
    /// Bytes held by live entries.
    pub used_bytes: u64,
}

#[derive(Clone)]
struct Entry {
    tokens: Range<usize>,
    bytes: Range<usize>,
    last_used: u64,
}

/// LRU snapshot cache over one pinned arena.
pub struct SnapshotCache {
    arena: PinnedArena,
    keys: Vec<i32>,
    entries: Vec<Entry>,
    pending: Option<Entry>,
    model_key: u64,
    tick: u64,
    stats: SnapshotStats,
}

impl SnapshotCache {
    /// Empty cache over `arena`; `model_key` names the model identity +
    /// `n_ctx` + `n_ubatch` the stored states belong to. The key arena holds
    /// `MAX_ENTRIES × max_tokens` tokens (the session passes `n_ctx`), allocated
    /// once. Target: 64 × 100k × 4 B = 25.6 MB at 100k ctx.
    pub fn new(arena: PinnedArena, model_key: u64, max_tokens: usize) -> Self {
        Self {
            arena,
            keys: vec![0; MAX_ENTRIES * max_tokens],
            entries: Vec::with_capacity(MAX_ENTRIES),
            pending: None,
            model_key,
            tick: 0,
            stats: SnapshotStats::default(),
        }
    }

    /// Identity the stored states belong to.
    pub fn model_key(&self) -> u64 {
        self.model_key
    }

    /// Longest stored key that is a strict prefix of `tokens` (≥ 1 delta
    /// token left to prefill); bumps its LRU tick.
    pub fn longest_prefix(&mut self, tokens: &[i32]) -> Option<(usize, &[u8])> {
        let keys = &self.keys;
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tokens.len() < tokens.len())
            .filter(|(_, e)| keys[e.tokens.clone()] == tokens[..e.tokens.len()])
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i);
        let Some(i) = best else {
            self.stats.misses += 1;
            return None;
        };
        self.stats.hits += 1;
        self.tick += 1;
        let e = &mut self.entries[i];
        e.last_used = self.tick;
        Some((e.tokens.len(), &self.arena.bytes()[e.bytes.clone()]))
    }

    /// Room for a `size`-byte snapshot keyed by `tokens`, invisible until
    /// [`Self::commit`]. Evicts LRU entries (and any entry with the same key)
    /// BEFORE the caller writes; `None` + `dropped_oversize` when the
    /// snapshot exceeds the arena or the key exceeds the key arena.
    pub fn reserve(&mut self, tokens: &[i32], size: usize) -> Option<&mut [u8]> {
        self.pending = None;
        if size > self.arena.len() || tokens.len() > self.keys.len() / MAX_ENTRIES {
            self.stats.dropped_oversize += 1;
            return None;
        }
        let (bytes, keys) = loop {
            let b = first_fit(&self.entries, |e| &e.bytes, self.arena.len(), size);
            let k = first_fit(&self.entries, |e| &e.tokens, self.keys.len(), tokens.len());
            match (b, k) {
                (Some(b), Some(k)) if self.entries.len() < MAX_ENTRIES => {
                    break (b..b + size, k..k + tokens.len());
                }
                _ => {
                    let lru = (0..self.entries.len())
                        .min_by_key(|&i| self.entries[i].last_used)
                        .expect("an empty cache fits any in-bounds entry");
                    self.remove(lru);
                    self.stats.evictions += 1;
                }
            }
        };
        self.keys[keys.clone()].copy_from_slice(tokens);
        self.pending = Some(Entry {
            tokens: keys,
            bytes: bytes.clone(),
            last_used: 0,
        });
        Some(&mut self.arena.bytes_mut()[bytes])
    }

    /// Publishes the last [`Self::reserve`]; call only after a clean prefill.
    pub fn commit(&mut self) {
        if let Some(mut e) = self.pending.take() {
            // A same-key entry is replaced only now, so a failed write keeps it.
            let key = &self.keys[e.tokens.clone()];
            if let Some(i) = self
                .entries
                .iter()
                .position(|o| self.keys[o.tokens.clone()] == *key)
            {
                self.remove(i);
            }
            self.tick += 1;
            e.last_used = self.tick;
            self.stats.used_bytes += e.bytes.len() as u64;
            self.entries.push(e);
        }
    }

    /// Discards the last [`Self::reserve`] after a short state write.
    pub fn abort_short_write(&mut self) {
        if self.pending.take().is_some() {
            self.stats.short_writes += 1;
        }
    }

    /// Drops the entry keyed by `tokens[..prefix_len]` after its restore was
    /// rejected.
    pub fn invalidate(&mut self, prefix_len: usize, tokens: &[i32]) {
        self.stats.restore_failures += 1;
        let key = &tokens[..prefix_len];
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| self.keys[e.tokens.clone()] == *key)
        {
            self.remove(i);
        }
    }

    /// Counter snapshot.
    pub fn stats(&self) -> SnapshotStats {
        self.stats
    }

    /// Test seam: cuts every stored state to half its length, so the next
    /// restore of it must be rejected.
    pub(crate) fn truncate_all_for_test(&mut self) {
        for e in &mut self.entries {
            let half = e.bytes.len() / 2;
            self.stats.used_bytes -= (e.bytes.len() - half) as u64;
            e.bytes.end = e.bytes.start + half;
        }
    }

    fn remove(&mut self, i: usize) {
        let e = self.entries.swap_remove(i);
        self.stats.used_bytes -= e.bytes.len() as u64;
    }
}

/// Lowest-offset gap of `len` in `[0, cap)` not covered by any entry's
/// `field` range (≤ MAX_ENTRIES entries: a rescan beats a sorted buffer).
fn first_fit(
    entries: &[Entry],
    field: fn(&Entry) -> &Range<usize>,
    cap: usize,
    len: usize,
) -> Option<usize> {
    let mut start: usize = 0;
    loop {
        let end = start.checked_add(len).filter(|&e| e <= cap)?;
        match entries
            .iter()
            .map(field)
            .filter(|r| r.start < end && start < r.end)
            .map(|r| r.end)
            .max()
        {
            None => return Some(start),
            Some(next) => start = next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(c: &mut SnapshotCache, t: &[i32], n: usize) {
        if c.reserve(t, n).is_some() {
            c.commit();
        }
    }

    #[test]
    fn longest_prefix_and_lru_eviction() {
        let mut c = SnapshotCache::new(PinnedArena::heap_for_test(10), 0, 256);
        put(&mut c, &[1, 2], 4);
        put(&mut c, &[1, 2, 3], 4);
        assert!(
            c.longest_prefix(&[1, 2, 3]).is_none_or(|(n, _)| n < 3),
            "needs a delta token"
        ); // bumps [1,2]
        assert_eq!(c.longest_prefix(&[1, 2, 3, 4]).unwrap().0, 3); // bumps [1,2,3]: [1,2] is LRU again
        put(&mut c, &[9], 4); // evicts LRU [1,2] before writing
        assert_eq!(c.longest_prefix(&[1, 2, 7]).map(|p| p.0), None);
        put(&mut c, &[5], 11); // bigger than arena: dropped, counted
        assert_eq!(c.stats().dropped_oversize, 1);
        assert!(c.stats().used_bytes <= 10);
    }

    #[test]
    fn uncommitted_reserve_is_invisible() {
        let mut c = SnapshotCache::new(PinnedArena::heap_for_test(10), 0, 256);
        let _ = c.reserve(&[1, 2], 4); // cancelled prefill: no commit
        assert!(c.longest_prefix(&[1, 2, 3]).is_none());
    }

    #[test]
    fn stored_bytes_round_trip_and_invalidate_drops_entry() {
        let mut c = SnapshotCache::new(PinnedArena::heap_for_test(16), 0, 256);
        c.reserve(&[4, 5], 3).unwrap().copy_from_slice(&[7, 8, 9]);
        c.commit();
        assert_eq!(c.longest_prefix(&[4, 5, 6]).unwrap().1, &[7, 8, 9]);
        c.invalidate(2, &[4, 5, 6]);
        assert!(c.longest_prefix(&[4, 5, 6]).is_none());
        let s = c.stats();
        assert_eq!((s.restore_failures, s.used_bytes), (1, 0));
    }

    #[test]
    fn failed_rewrite_keeps_same_key_entry() {
        let mut c = SnapshotCache::new(PinnedArena::heap_for_test(16), 0, 256);
        c.reserve(&[4, 5], 3).unwrap().copy_from_slice(&[7, 8, 9]);
        c.commit();
        let _ = c.reserve(&[4, 5], 3); // short write: never committed
        c.abort_short_write();
        assert_eq!(c.longest_prefix(&[4, 5, 6]).unwrap().1, &[7, 8, 9]);
        c.reserve(&[4, 5], 3).unwrap().copy_from_slice(&[1, 2, 3]);
        c.commit(); // replaces the old entry
        assert_eq!(c.longest_prefix(&[4, 5, 6]).unwrap().1, &[1, 2, 3]);
        let s = c.stats();
        assert_eq!((s.short_writes, s.used_bytes), (1, 3));
    }
}
