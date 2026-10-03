#![allow(unexpected_cfgs)]

//! mlock'd RAM expert slot arena (§5.4): `mmap(MAP_ANONYMOUS|MAP_NORESERVE)`
//! locked into physical memory, carved into per-class slot regions — one run
//! of equal slots per distinct slot size (§5.4). Slots are handed out
//! exclusively by [`super::residency::ResidencyMap`] claims + leases; the
//! arena itself never coordinates access.
//!
//! All `unsafe` in the pool module lives here (mmap/mlock/munlock/munmap and
//! raw slot access), each item with its invariant named.

use std::fmt;
use std::io;
use std::num::NonZeroUsize;

/// One run of equal slots inside the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotClass {
    /// Slot size in bytes for every slot of this class.
    pub slot_size: usize,
    /// Number of slots in this class.
    pub slots: u32,
}

/// Arena construction/mapping failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArenaError {
    /// Empty class list, a zero slot size or slot count, total slot count
    /// above `u32::MAX`, or total byte size overflow.
    InvalidLayout {
        /// Rejected pool size.
        pool_bytes: usize,
        /// Rejected slot size.
        slot_size: usize,
    },
    /// `mmap` failed.
    MapFailed {
        /// `errno` from the syscall.
        errno: i32,
    },
    /// `mlock` failed (typically `ENOMEM` on low `RLIMIT_MEMLOCK` — see the
    /// CLI preflight). NEVER mapped unlocked as a fallback: mlock is a
    /// correctness feature (§5.4).
    LockFailed {
        /// `errno` from the syscall.
        errno: i32,
    },
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLayout {
                pool_bytes,
                slot_size,
            } => write!(
                f,
                "invalid arena layout: pool_bytes={pool_bytes} slot_size={slot_size}"
            ),
            Self::MapFailed { errno } => write!(f, "arena mmap failed, errno={errno}"),
            Self::LockFailed { errno } => write!(
                f,
                "arena mlock failed, errno={errno} (check RLIMIT_MEMLOCK / LimitMEMLOCK)"
            ),
        }
    }
}

impl std::error::Error for ArenaError {}

/// Page-aligned superset of `[off, off + len)`: the kernel only drops whole
/// pages inside the advised range, so the partial boundary pages must be
/// covered explicitly.
pub(crate) fn page_span(off: u64, len: usize) -> (u64, u64) {
    let page = PAGE as u64;
    let start = off - off % page;
    let end = (off + len as u64).next_multiple_of(page);
    (start, end - start)
}

/// Drops the pages covering `[off, off + len)` of `file` from the page cache
/// once its bytes sit in a pool slot: expert bytes must never stay
/// double-cached (§5.4; 172.8 GiB of experts vs 93 GiB RAM). Advisory — a
/// failure leaves only a cache-pressure cost, never wrong bytes, so it is not
/// an error.
pub(crate) fn drop_page_cache(file: &std::fs::File, off: u64, len: usize) {
    use std::os::fd::AsRawFd;
    let (start, span) = page_span(off, len);
    // SAFETY: posix_fadvise takes a live fd (borrowed from `file` for the
    // call) and plain integers; it touches no Rust-visible memory.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            start as libc::off_t,
            span as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

/// Marks `file` random-access so buffered expert reads do no readahead (it
/// would only populate the cache with neighbouring expert bytes). The flag
/// lives on the open file description, so one call at open covers every read.
pub(crate) fn advise_random(file: &std::fs::File) {
    use std::os::fd::AsRawFd;
    // SAFETY: as in `drop_page_cache`; offset 0 / len 0 means the whole file.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
    }
}

/// Class-region alignment (one page).
const PAGE: usize = 4096;

fn last_errno() -> i32 {
    io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// A locked slot arena: `len` bytes carved into per-class slot regions — one
/// run of equal slots per distinct slot size (§5.4).
///
/// # Safety contract (shared access)
/// `&Arena` hands out `&[u8]`/`&mut [u8]` views of slots, but exclusivity is
/// granted ONLY by the residency map: one lease = one reader of one slot, and
/// the filler holds the claim (granted at `claim`, exclusive against eviction)
/// until its copy completes. The fill-completion gate closes the remaining
/// window: between `claim` and the filler's `finish_fill` the key is mid-fill
/// in its shard, so NO other thread can obtain a lease on it (`Resident`
/// claims park until the gate opens) — `slot_mut`'s `&mut` is truly exclusive
/// during the fill, and `slot`'s `&` readers all start after it. The arena
/// itself never hands out overlapping views across different slots — slots
/// are disjoint across classes by construction (class regions are contiguous
/// and non-overlapping, and a slot never spans its region's end).
pub struct Arena {
    base: NonZeroUsize,
    len: usize,
    /// Per-class layout in class order; class `c` covers global slot ids
    /// `[first, first + slots)` at byte offset `offset` from `base`.
    classes: Vec<ClassRegion>,
    slots: u32,
}

/// Fixed-size layout record for one class, built once at construction.
#[derive(Debug)]
struct ClassRegion {
    first: u32,
    slots: u32,
    slot_size: usize,
    offset: usize,
}

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("len", &self.len)
            .field("slots", &self.slots)
            .field("classes", &self.classes)
            .finish_non_exhaustive()
    }
}

impl Arena {
    /// Maps `Σ slot_size × slots` anonymous + `MAP_NORESERVE` and locks it
    /// with `mlock`; class `c`'s slots are global ids
    /// `[first_c, first_c + slots_c)`, laid out contiguously in class order.
    /// Fails rather than mapping unlocked (§5.4).
    pub fn new(classes: &[SlotClass]) -> Result<Self, ArenaError> {
        if classes.is_empty()
            || classes.iter().any(|c| c.slot_size == 0 || c.slots == 0)
            || classes.iter().map(|c| c.slots as u64).sum::<u64>() > u32::MAX as u64
        {
            let (slot_size, slots) = classes.first().map_or((0, 0), |c| (c.slot_size, c.slots));
            return Err(ArenaError::InvalidLayout {
                pool_bytes: (slots as usize).saturating_mul(slot_size),
                slot_size,
            });
        }
        let mut regions = Vec::with_capacity(classes.len());
        let (mut first, mut offset) = (0u32, 0usize);
        for c in classes {
            let bytes = c
                .slot_size
                .checked_mul(c.slots as usize)
                .and_then(|b| offset.checked_add(b))
                .ok_or(ArenaError::InvalidLayout {
                    pool_bytes: usize::MAX,
                    slot_size: c.slot_size,
                })?;
            regions.push(ClassRegion {
                first,
                slots: c.slots,
                slot_size: c.slot_size,
                offset,
            });
            first += c.slots;
            // Page-aligned class regions: with page-multiple slot sizes every
            // slot (and every direct-read plane region) stays page-aligned,
            // as O_DIRECT requires (§5.4).
            offset = bytes
                .checked_next_multiple_of(PAGE)
                .ok_or(ArenaError::InvalidLayout {
                    pool_bytes: usize::MAX,
                    slot_size: c.slot_size,
                })?;
        }
        let pool_bytes = offset;
        // SAFETY: no pointer inputs; anonymous private mapping with no file
        // descriptor. MAP_NORESERVE keeps the address-space reservation from
        // committing swap up front (§5.4) — pages fault in on demand.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                pool_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(ArenaError::MapFailed {
                errno: last_errno(),
            });
        }
        let base = ptr as usize;
        // SAFETY: [base, base+pool_bytes) is a whole mapping this call owns;
        // nothing else has been handed out yet.
        let locked = unsafe { libc::mlock(base as *const libc::c_void, pool_bytes) };
        if locked != 0 {
            let errno = last_errno();
            // SAFETY: same whole mapping, still solely owned.
            unsafe { libc::munmap(base as *mut libc::c_void, pool_bytes) };
            return Err(ArenaError::LockFailed { errno });
        }
        Ok(Self {
            base: NonZeroUsize::new(base).expect("mmap never returns NULL"),
            len: pool_bytes,
            slots: first,
            classes: regions,
        })
    }

    /// Base pointer and byte length of the whole arena, for GPU pinning
    /// (§3.5) and metrics; the arena keeps ownership.
    pub fn as_ptr_range(&self) -> (*mut u8, usize) {
        (self.ptr(), self.len)
    }

    fn ptr(&self) -> *mut u8 {
        self.base.get() as *mut u8
    }

    /// Read the slot a live [`Lease`](super::pager::Lease) covers. The view
    /// borrows the lease, so it cannot outlive it: the lease is the
    /// capability that excludes writers and keeps the slot from being
    /// refilled.
    ///
    /// # Panics
    /// Panics when the lease belongs to another arena.
    pub(crate) fn slot<'l>(&self, lease: &'l crate::pool::pager::Lease) -> &'l [u8] {
        assert!(
            lease.is_for(self),
            "slot read under a lease of another arena"
        );
        let i = lease.slot();
        assert!(i < self.slots);
        let n = self.slot_size_of(i);
        // SAFETY: i < slots ⇒ in-bounds; the lease excludes writers (§6.6),
        // keeps the mapping alive (it owns an `Arc<Arena>`), and the result
        // borrows the lease, so the view ends before the lease can drop.
        unsafe { std::slice::from_raw_parts(self.ptr().add(self.offset_of(i)), n) }
    }

    /// Mutable slot view for the page-in path. The residency map's claim
    /// granted the claimer the only lease on the slot and parks every other
    /// claimer of the key until the fill is reported done; the view borrows
    /// the claim mutably, so it cannot outlive it nor coexist with a second
    /// view from the same claim.
    ///
    /// # Panics
    /// Panics when the claim names a slot outside the arena.
    #[allow(clippy::mut_from_ref)] // exclusivity is carried by `&mut FillClaim`
    pub(crate) fn slot_mut<'c>(
        &self,
        claim: &'c mut crate::pool::residency::FillClaim<'_>,
    ) -> &'c mut [u8] {
        let i = claim.slot();
        assert!(i < self.slots);
        let n = self.slot_size_of(i);
        // SAFETY: same bounds as `slot`; the claim is the sole writer and no
        // reader exists until it is finished or cancelled (consumed), which
        // ends this borrow first.
        unsafe { std::slice::from_raw_parts_mut(self.ptr().add(self.offset_of(i)), n) }
    }

    /// Number of slot classes.
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Number of slots.
    pub fn slot_count(&self) -> u32 {
        self.slots
    }

    /// Index of the class owning global slot `i` (binary search over the
    /// fixed per-class table — no allocation, no linear scan).
    ///
    /// # Panics
    /// Panics when `i >= slot_count`.
    pub fn class_of(&self, i: u32) -> usize {
        assert!(i < self.slots);
        self.classes.partition_point(|c| c.first <= i) - 1
    }

    /// Global slot ids owned by class `c`.
    ///
    /// # Panics
    /// Panics when `c >= classes.len()`.
    pub fn class_range(&self, c: usize) -> std::ops::Range<u32> {
        let r = &self.classes[c];
        r.first..r.first + r.slots
    }

    /// Slot size in bytes of global slot `i`.
    ///
    /// # Panics
    /// Panics when `i >= slot_count`.
    pub fn slot_size_of(&self, i: u32) -> usize {
        self.classes[self.class_of(i)].slot_size
    }

    /// Byte offset of slot `i` from the arena base.
    fn offset_of(&self, i: u32) -> usize {
        let c = &self.classes[self.class_of(i)];
        c.offset + (i - c.first) as usize * c.slot_size
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: base is the whole mapping from `new` (mlock'd, never
        // sub-allocated); `self` is the single owner and is dropping now, so
        // no views are live.
        unsafe {
            libc::munlock(self.ptr() as *const libc::c_void, self.len);
            libc::munmap(self.ptr() as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    fn arena() -> Arena {
        Arena::new(&[
            SlotClass {
                slot_size: 4096,
                slots: 3,
            },
            SlotClass {
                slot_size: 6144,
                slots: 2,
            },
        ])
        .expect("mlock may fail on low RLIMIT_MEMLOCK")
    }

    #[test]
    fn page_span_covers_partial_boundary_pages() {
        assert_eq!(page_span(0, 4096), (0, 4096));
        assert_eq!(page_span(100, 4096), (0, 8192), "straddles two pages");
        assert_eq!(page_span(4096 + 8, 10), (4096, 4096));
        assert_eq!(page_span(8191, 2), (4096, 8192));
    }

    #[test]
    fn class_regions_are_contiguous_and_non_overlapping() {
        let a = arena();
        assert_eq!(a.offset_of(0), 0);
        assert_eq!(a.offset_of(2), 2 * 4096);
        assert_eq!(a.offset_of(3), 3 * 4096, "class 1 starts after class 0");
        assert_eq!(a.offset_of(4), 3 * 4096 + 6144);
        assert_eq!(a.offset_of(4) + 6144, a.len, "last slot ends the arena");
    }
}
