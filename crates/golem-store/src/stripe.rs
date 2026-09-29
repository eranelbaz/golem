//! §5.2 striped read planning: split logical reads across shard replicas.

/// One contiguous piece of a logical read, served from one replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    /// Index within the shard's replicas.
    pub replica_index: usize,
    /// Absolute file offset.
    pub offset: u64,
    /// Length of the chunk in bytes.
    pub len: usize,
}

/// §5.2: split `[offset, offset+len)` across `replica_count` replicas in
/// contiguous chunks; chunk i starts at `offset + i*cs`, `cs = len.div_ceil(n)`.
/// Chunks with `len == 0` (possible when `len < replica_count`) are omitted,
/// so plan length can be < `replica_count`. Deterministic: same inputs →
/// same plan, always.
pub fn plan_striped(offset: u64, len: usize, replica_count: usize) -> Vec<Chunk> {
    assert!(replica_count > 0, "replica_count must be > 0");
    if len == 0 {
        return Vec::new();
    }
    let cs = len.div_ceil(replica_count);
    (0..replica_count)
        .map(|i| Chunk {
            replica_index: i,
            offset: offset + (i as u64) * cs as u64,
            len: cs.min(len.saturating_sub(i * cs)),
        })
        .filter(|c| c.len > 0)
        .collect()
}

/// Failover plan: same logical bytes served whole from one replica
/// (always exactly one chunk).
pub fn plan_single(offset: u64, len: usize, replica_index: usize) -> Vec<Chunk> {
    if len == 0 {
        return Vec::new();
    }
    vec![Chunk {
        replica_index,
        offset,
        len,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_replica_is_single_chunk() {
        let plan = plan_striped(1000, 3000, 1);
        assert_eq!(
            plan,
            vec![Chunk {
                replica_index: 0,
                offset: 1000,
                len: 3000
            }]
        );
    }

    #[test]
    fn two_replicas_even_split() {
        let plan = plan_striped(0, 4_000_000, 2);
        assert_eq!(plan.len(), 2);
        assert_eq!(
            plan[0],
            Chunk {
                replica_index: 0,
                offset: 0,
                len: 2_000_000
            }
        );
        assert_eq!(
            plan[1],
            Chunk {
                replica_index: 1,
                offset: 2_000_000,
                len: 2_000_000
            }
        );
    }

    #[test]
    fn four_replicas_uneven_len_last_chunk_smaller() {
        let plan = plan_striped(7, 10, 4);
        let cs = 10usize.div_ceil(4); // 3
        let expected: Vec<Chunk> = (0..4)
            .map(|i| Chunk {
                replica_index: i,
                offset: 7 + (i as u64) * cs as u64,
                len: ((i + 1) * cs).min(10) - i * cs,
            })
            .collect();
        assert_eq!(plan, expected);
        let mut cursor = 7u64;
        for c in &plan {
            assert_eq!(c.offset, cursor);
            assert!(c.len > 0);
            cursor += c.len as u64;
        }
        assert_eq!(cursor, 17);
    }

    #[test]
    fn uneven_len_shorter_than_replica_count_omits_zero_chunks() {
        let plan = plan_striped(0, 1, 4);
        assert_eq!(
            plan,
            vec![Chunk {
                replica_index: 0,
                offset: 0,
                len: 1
            }]
        );
    }

    #[test]
    fn empty_read_is_empty_plan() {
        assert!(plan_striped(0, 0, 2).is_empty());
    }

    #[test]
    fn zero_replicas_panics() {
        let r = std::panic::catch_unwind(|| plan_striped(0, 10, 0));
        assert!(r.is_err());
    }

    #[test]
    fn failover_plan_matches_striped_bytes() {
        let striped = plan_striped(3, 9, 3);
        let whole = plan_single(3, 9, 1);
        assert_eq!(whole.len(), 1);
        assert_eq!(whole[0].len, striped.iter().map(|c| c.len).sum::<usize>());
        assert_eq!(whole[0].offset, 3);
    }
}
