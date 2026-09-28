//! mlock'd RAM expert slot arena (§5.4): `mmap(MAP_ANONYMOUS|MAP_NORESERVE)`
//! locked into physical memory, carved into equal slots. Slots are handed out
//! exclusively by [`super::residency::ResidencyMap`] claims + leases; the
//! arena itself never coordinates access.
//!
//! All `unsafe` in the pool module lives here (mmap/mlock/munlock/munmap and
//! raw slot access), each item with its invariant named.

use std::fmt;
use std::io;
use std::num::NonZeroUsize;

/// Arena construction/mapping failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArenaError {
    /// `pool_bytes == 0`, `slot_size == 0`, `pool_bytes % slot_size != 0`, or
    /// slot count exceeds `u32::MAX`.
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

fn last_errno() -> i32 {
    io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// A locked slot arena. `len` bytes, `slots` slots of `slot_size` bytes.
///
/// # Safety contract (Send/Sync)
/// `&Arena` hands out `&[u8]`/`&mut [u8]` views of slots, but exclusivity is
/// granted ONLY by the residency map: one lease = one reader of one slot, and
/// the filler holds the claim (granted at `claim`, exclusive against eviction)
/// until its copy completes. The arena itself never hands out overlapping
/// views across different slots — slots are disjoint by construction.
pub struct Arena {
    base: NonZeroUsize,
    len: usize,
    slot_size: usize,
    slots: u32,
}

// SAFETY: the arena is an owned raw allocation with no interior mutability;
// every byte access goes through the claim/lease exclusivity contract above
// (one writer per slot between claim and residency flip, lease-counted
// readers after it), which upholds aliasing rules for the `&[u8]`/`&mut [u8]`
// views this type vends.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("len", &self.len)
            .field("slot_size", &self.slot_size)
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}

impl Arena {
    /// Maps `pool_bytes` anonymous + `MAP_NORESERVE` and locks it with
    /// `mlock`. Fails rather than mapping unlocked (§5.4).
    pub fn new(pool_bytes: usize, slot_size: usize) -> Result<Self, ArenaError> {
        if pool_bytes == 0
            || slot_size == 0
            || pool_bytes % slot_size != 0
            || pool_bytes / slot_size > u32::MAX as usize
        {
            return Err(ArenaError::InvalidLayout {
                pool_bytes,
                slot_size,
            });
        }
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
            slot_size,
            slots: (pool_bytes / slot_size) as u32,
        })
    }

    fn ptr(&self) -> *mut u8 {
        self.base.get() as *mut u8
    }

    /// Read slot bytes. Caller must hold a lease on the key mapped to `i`.
    ///
    /// # Panics
    /// Panics when `i >= slot_count`.
    pub fn slot(&self, i: u32) -> &[u8] {
        assert!(i < self.slots);
        // SAFETY: i < slots ⇒ [base + i*slot_size, +slot_size) is inside the
        // mapping; the caller holds a lease, so no writer touches it (§6.6).
        unsafe {
            std::slice::from_raw_parts(self.ptr().add(i as usize * self.slot_size), self.slot_size)
        }
    }

    /// Mutable slot view for the page-in path. Caller must hold the `Filled`
    /// claim for `i` (exclusive): a single writer per slot between claim and
    /// the copy-under-lease, enforced by the residency map's lease granted at
    /// claim time.
    ///
    /// # Panics
    /// Panics when `i >= slot_count`.
    // &self → &mut [u8] is the point: the aliasing rule it relies on is the
    // Filled-claim exclusivity contract documented above, which the residency
    // map enforces.
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn slot_mut(&self, i: u32) -> &mut [u8] {
        assert!(i < self.slots);
        // SAFETY: same bounds as `slot`; exclusivity comes from the Filled
        // claim — the residency map guarantees no other writer or reader for
        // this slot until the claim is released.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr().add(i as usize * self.slot_size),
                self.slot_size,
            )
        }
    }

    /// Fill slot bytes. Caller must hold the `Filled` claim for `i`.
    ///
    /// # Panics
    /// Panics when `i >= slot_count` or `src` is longer than a slot.
    pub fn fill_slot(&self, i: u32, src: &[u8]) {
        assert!(src.len() <= self.slot_size, "fill larger than slot");
        self.slot_mut(i)[..src.len()].copy_from_slice(src);
    }

    /// Number of slots.
    pub fn slot_count(&self) -> u32 {
        self.slots
    }

    /// Slot size in bytes.
    pub fn slot_size(&self) -> usize {
        self.slot_size
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
