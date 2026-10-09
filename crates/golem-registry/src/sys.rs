//! The registry's only `unsafe`: `flock(2)` and `kill(2)` FFI.
#![allow(unsafe_code)]

use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;

/// Blocks until `f` holds an exclusive `flock`; released when `f` closes.
pub fn flock_exclusive(f: &File) -> io::Result<()> {
    loop {
        // SAFETY: `f` owns a valid open fd for the duration of the call.
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } == 0 {
            return Ok(());
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

/// `kill(pid, 0)`: the pid exists (EPERM still means it exists).
pub fn pid_exists(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // SAFETY: signal 0 performs only the existence/permission check.
    (unsafe { libc::kill(pid, 0) } == 0)
        || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Sends `sig` to `pid`.
pub fn signal(pid: u32, sig: libc::c_int) -> io::Result<()> {
    let pid = libc::pid_t::try_from(pid).map_err(io::Error::other)?;
    // SAFETY: plain kill(2); a positive pid targets exactly one process.
    if pid > 0 && unsafe { libc::kill(pid, sig) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}
