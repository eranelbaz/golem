//! The CLI's process-control FFI for `serve --detach`: pipe, fork, setsid,
//! stdio redirection.
#![allow(unsafe_code)]

use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};

/// `(read, write)` ends of a close-on-exec pipe.
pub fn pipe() -> io::Result<(File, File)> {
    let mut fds = [0; 2];
    // SAFETY: `fds` is a valid 2-int out-array; on success both fds are new
    // and owned by nothing else, so each is wrapped exactly once.
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: see above — fresh fds, sole owners.
    Ok(unsafe { (File::from_raw_fd(fds[0]), File::from_raw_fd(fds[1])) })
}

/// Which side of a `fork` this is.
pub enum Fork {
    /// The original process.
    Parent,
    /// The new process.
    Child,
}

/// `fork(2)`. Called only while the process is single-threaded (first thing
/// in `main`, before tracing, tokio or the registry heartbeat start).
pub fn fork() -> io::Result<Fork> {
    // SAFETY: single-threaded caller (see doc), so the child inherits no
    // lock held by another thread.
    match unsafe { libc::fork() } {
        -1 => Err(io::Error::last_os_error()),
        0 => Ok(Fork::Child),
        _ => Ok(Fork::Parent),
    }
}

/// `setsid(2)`: detach from the controlling terminal.
pub fn setsid() -> io::Result<()> {
    // SAFETY: no arguments; fails only if already a group leader.
    if unsafe { libc::setsid() } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Points stdin at `/dev/null` and stdout/stderr at `log`.
pub fn redirect_stdio(log: &File) -> io::Result<()> {
    let null = File::open("/dev/null")?;
    for (src, dst) in [
        (null.as_raw_fd(), 0),
        (log.as_raw_fd(), 1),
        (log.as_raw_fd(), 2),
    ] {
        // SAFETY: `src` is an open fd owned by a live `File`; `dst` is a
        // standard stream, which dup2 atomically replaces.
        if unsafe { libc::dup2(src, dst) } == -1 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// `SIGKILL`s `pid` (the detached grandchild after a load timeout).
pub fn kill(pid: u32) -> io::Result<()> {
    let pid = libc::pid_t::try_from(pid)
        .ok()
        .filter(|&p| p > 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pid must be positive"))?;
    // SAFETY: plain kill(2); a positive pid targets exactly one process.
    if unsafe { libc::kill(pid, libc::SIGKILL) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// `_exit(2)`: leave the intermediate fork child without running atexit
/// handlers or destructors that belong to the parent's state.
pub fn exit_now(code: i32) -> ! {
    // SAFETY: terminates the process; nothing runs afterwards.
    unsafe { libc::_exit(code) }
}
