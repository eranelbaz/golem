//! §5.4 aligned chunk planning: reads are covered by 4 KiB-aligned chunks cut
//! from a 2 MiB grid anchored at the aligned range start, so every chunk
//! satisfies the O_DIRECT offset+length alignment rule and no forced tiny head
//! chunk exists.

use crate::sys::DIRECT_ALIGN;

/// Chunk size cap (§5.4): plans never exceed 2 MiB per chunk.
pub const CHUNK: usize = 2 * 1024 * 1024;

/// Plans the aligned covering range `[align_down(base, 4096),
/// align_up(base+len, 4096))` cut at 2 MiB boundaries counted from
/// `align_down(base, 4096)`. Every chunk is 4 KiB-aligned in offset and
/// length (§5.4 O_DIRECT rule) and `len <= CHUNK`. Deterministic: same
/// inputs → same plan, always.
///
/// The destination region handed to the queue must cover exactly this range;
/// chunk 0 starts at the region base, so the bytes before the logical read
/// are head scratch (mirroring [`crate::engine`]'s region contract).
pub fn plan_aligned(base: u64, len: usize) -> Vec<(u64, usize)> {
    if len == 0 {
        return Vec::new();
    }
    let start = base - (base % DIRECT_ALIGN);
    let end = (base + len as u64).div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN;
    let mut plan = Vec::new();
    let mut off = start;
    while off < end {
        let k = (off - start) / CHUNK as u64 + 1;
        let cut = (start + k * CHUNK as u64).min(end);
        plan.push((off, (cut - off) as usize));
        off = cut;
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_aligned_range_in_2mib_pieces() {
        // std unit is 15,204,352 B but reads are per PLANE (≈4.7/5.8 MB):
        // a 5,898,496 B plane at file offset 4096 → covering [4096,
        // 5_906_432), 4096-aligned covering length 5_902_336, cut: 2 MiB +
        // 2 MiB + tail.
        let plan = plan_aligned(4096, 5_898_496);
        assert_eq!(plan.iter().map(|c| c.1).sum::<usize>(), 5_902_336);
        let mut at = 4096u64;
        for (off, len) in &plan {
            assert_eq!(off % 4096, 0); // file-offset aligned
            assert_eq!(*len % 4096, 0); // O_DIRECT length rule
            assert_eq!(*off, at); // contiguous covering range
            at += *len as u64;
            assert!(*len <= CHUNK);
        }
    }

    #[test]
    fn boundaries_are_2mib_grid_from_the_aligned_base() {
        // plane at 512, CHUNK long: covering [0, align_up(512+CHUNK)) =
        // [0, 2_101_248); cut at the 2 MiB grid from 0 → (0, 2 MiB) +
        // (2 MiB, 4096).
        let plan = plan_aligned(512, CHUNK);
        assert_eq!(plan[0], (0, 2_097_152));
        assert_eq!(plan[1], (2_097_152, 4_096));
    }

    #[test]
    fn zero_len_is_no_chunks() {
        assert!(plan_aligned(0, 0).is_empty());
    }
}
