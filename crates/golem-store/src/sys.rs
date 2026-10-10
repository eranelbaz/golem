//! Raw syscall wrappers (§5.4): page-cache hygiene, `O_DIRECT` reads and
//! filesystem capacity queries. The only `unsafe` in golem-store, each block with its invariant
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

/// `O_DIRECT` alignment for offset, length and buffer address (§5.4): one
/// page, a multiple of every logical block size in use (512 / 4096).
pub const DIRECT_ALIGN: u64 = 4096;

/// Opens `path` with `O_DIRECT` (§5.4): expert bytes bypass the page cache
/// entirely. Fails where the filesystem (or kernel) refuses the flag; the
/// caller then keeps a buffered handle only.
pub fn open_direct(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

/// A set of logical CPU ids (R-15 pinning).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuSet(pub Vec<usize>);

impl CpuSet {
    /// Parses a sysfs cpu list (`0-7,16-23`).
    pub fn parse_list(s: &str) -> Option<Self> {
        let mut v = Vec::new();
        for part in s.trim().split(',').filter(|p| !p.is_empty()) {
            match part.split_once('-') {
                Some((a, b)) => v.extend(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?),
                None => v.push(part.parse().ok()?),
            }
        }
        (!v.is_empty()).then_some(Self(v))
    }
}

/// Binds the calling thread to `set` (`sched_setaffinity(0, …)`); threads it
/// spawns afterwards inherit the mask.
pub fn pin_current_thread(set: &CpuSet) -> std::io::Result<()> {
    // SAFETY: cpu_set_t is plain bits; zeroed is the empty set.
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in &set.0 {
        if c >= libc::CPU_SETSIZE as usize {
            return Err(std::io::Error::other(format!("cpu {c} ≥ CPU_SETSIZE")));
        }
        // SAFETY: c < CPU_SETSIZE, checked above.
        unsafe { libc::CPU_SET(c, &mut mask) };
    }
    // SAFETY: pid 0 = calling thread; `mask` is a live, initialized cpu_set_t
    // of the size passed.
    let rc = unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&mask), &mask) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// R-15 CCD split from `(l3 size, shared_cpu_list)` pairs (one per CPU's
/// `cache/index3`): `(v-cache CCD, other CCDs)`, or `None` when there is one
/// L3 domain or no size difference (nothing to prefer).
pub fn vcache_split(l3: &[(u64, String)]) -> Option<(CpuSet, CpuSet)> {
    let mut domains: Vec<(u64, &str)> = Vec::new();
    for (size, list) in l3 {
        if !domains.iter().any(|(_, l)| l == list) {
            domains.push((*size, list));
        }
    }
    let max = domains.iter().map(|d| d.0).max()?;
    if domains.len() < 2 || domains.iter().all(|d| d.0 == max) {
        return None;
    }
    let mut big = Vec::new();
    let mut rest = Vec::new();
    for (size, list) in domains {
        let cpus = CpuSet::parse_list(list)?.0;
        if size == max { &mut big } else { &mut rest }.extend(cpus);
    }
    Some((CpuSet(big), CpuSet(rest)))
}

/// Reads every CPU's L3 `(size, shared_cpu_list)` from sysfs and applies
/// [`vcache_split`].
pub fn detect_vcache_split() -> Option<(CpuSet, CpuSet)> {
    let mut l3 = Vec::new();
    for e in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name
            .strip_prefix("cpu")
            .is_some_and(|n| n.parse::<u32>().is_ok())
        {
            continue;
        }
        let d = e.path().join("cache/index3");
        let (Ok(size), Ok(list)) = (
            std::fs::read_to_string(d.join("size")),
            std::fs::read_to_string(d.join("shared_cpu_list")),
        ) else {
            continue;
        };
        let size = size.trim();
        let Ok(kib) = size.strip_suffix('K').unwrap_or(size).parse::<u64>() else {
            tracing::debug!("{name}: unparsable L3 size {size:?}; skipped");
            continue;
        };
        l3.push((kib, list.trim().to_owned()));
    }
    vcache_split(&l3)
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
    fn vcache_split_picks_the_larger_l3_and_noops_on_one_ccd() {
        let two = [
            (98304, "0-7,16-23".to_owned()),
            (98304, "0-7,16-23".to_owned()),
            (32768, "8-15,24-31".to_owned()),
        ];
        let (big, rest) = vcache_split(&two).unwrap();
        assert_eq!(
            big.0,
            [(0..8).collect::<Vec<_>>(), (16..24).collect()].concat()
        );
        assert_eq!(rest.0.len(), 16);
        assert!(vcache_split(&[(32768, "0-15".to_owned())]).is_none());
        let same = [(32768, "0-7".to_owned()), (32768, "8-15".to_owned())];
        assert!(vcache_split(&same).is_none(), "no V-cache CCD → no-op");
    }

    #[test]
    fn pin_current_thread_restricts_affinity() {
        std::thread::spawn(|| {
            pin_current_thread(&CpuSet(vec![0])).unwrap();
            // SAFETY: test-local; zeroed cpu_set_t is the empty set.
            let mut m: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            // SAFETY: valid out-pointer of the passed size.
            assert_eq!(
                unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&m), &mut m) },
                0
            );
            // SAFETY: reads bits of an initialized set.
            assert_eq!(unsafe { libc::CPU_COUNT(&m) }, 1);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn available_bytes_reports_free_space() {
        assert!(available_bytes(&std::env::temp_dir()).unwrap() > 0);
    }
}
