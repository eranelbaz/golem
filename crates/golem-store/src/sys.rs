//! Raw syscall wrappers (§5.4): page-cache hygiene and filesystem capacity
//! queries. The only `unsafe` in golem-store, each block with its invariant
//! named.

use std::path::Path;

/// Page size assumed for cache-hygiene spans (x86-64). Arena class-region
/// alignment uses the same constant.
const PAGE: u64 = 4096;

/// Page-aligned superset of `[off, off + len)`: the kernel only drops whole
/// pages inside the advised range, so the partial boundary pages must be
/// covered explicitly.
#[must_use]
pub fn page_span(off: u64, len: usize) -> (u64, u64) {
    let start = off - off % PAGE;
    let end = (off + len as u64).next_multiple_of(PAGE);
    (start, end - start)
}

/// Drops the pages covering `[off, off + len)` of `file` from the page cache
/// once its bytes sit in a pool slot: expert bytes must never stay
/// double-cached (§5.4; 172.8 GiB of experts vs 93 GiB RAM). Returns whether
/// the advise succeeded — advisory only, a failure leaves just a
/// cache-pressure cost, never wrong bytes.
pub fn drop_page_cache(file: &std::fs::File, off: u64, len: usize) -> bool {
    use std::os::fd::AsRawFd;
    let (start, span) = page_span(off, len);
    // SAFETY: posix_fadvise takes a live fd (borrowed from `file` for the
    // call) and plain integers; it touches no Rust-visible memory.
    let rc = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            start as libc::off_t,
            span as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
    rc == 0
}

/// Free bytes available to this user on the filesystem holding `path`.
pub fn available_bytes(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes())?;
    let mut s = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c` is a valid NUL-terminated path for the call; `s` is written
    // in full by statvfs on success and read only then.
    let rc = unsafe { libc::statvfs(c.as_ptr(), s.as_mut_ptr()) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: rc == 0 ⇒ statvfs initialized `s`.
    let s = unsafe { s.assume_init() };
    Ok(s.f_bavail * s.f_frsize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_span_covers_partial_boundary_pages() {
        assert_eq!(page_span(0, 4096), (0, 4096));
        assert_eq!(page_span(100, 4096), (0, 8192), "straddles two pages");
        assert_eq!(page_span(4096 + 8, 10), (4096, 4096));
        assert_eq!(page_span(8191, 2), (4096, 8192));
    }

    #[test]
    fn available_bytes_reports_free_space() {
        assert!(available_bytes(&std::env::temp_dir()).unwrap() > 0);
    }
}
