//! The CLI's process-control FFI for `serve --detach`: pipe, fork, setsid,
//! stdio redirection, the optimize child's parent-death signal, and the
//! `run` shutdown signal handlers.
#![allow(unsafe_code)]

use std::fs::File;
use std::io::{self, BufRead, Read};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, OnceLock};

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

/// `prctl(PR_SET_PDEATHSIG, SIGKILL)`: the kernel kills this process when
/// its parent thread exits.
pub fn die_with_parent() -> io::Result<()> {
    // SAFETY: PR_SET_PDEATHSIG takes one integer signal argument; no memory
    // is passed.
    if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// `getppid(2)`.
pub fn parent_pid() -> u32 {
    // SAFETY: no arguments; always succeeds.
    unsafe { libc::getppid() }.unsigned_abs()
}

/// `_exit(2)`: leave the intermediate fork child without running atexit
/// handlers or destructors that belong to the parent's state.
pub fn exit_now(code: i32) -> ! {
    // SAFETY: terminates the process; nothing runs afterwards.
    unsafe { libc::_exit(code) }
}

static SHUTDOWN_FD: AtomicI32 = AtomicI32::new(-1);
static SIGNALLED: AtomicBool = AtomicBool::new(false);
static SIGNAL: AtomicI32 = AtomicI32::new(0);
static INSTALLED: AtomicBool = AtomicBool::new(false);
static CANCEL: OnceLock<Arc<AtomicBool>> = OnceLock::new();

extern "C" fn on_shutdown(sig: libc::c_int) {
    if SIGNALLED.swap(true, Ordering::SeqCst) {
        // SAFETY: signal(2) and raise(3) are async-signal-safe; SIG_DFL
        // makes the re-raised signal terminate the process.
        unsafe {
            libc::signal(sig, libc::SIG_DFL);
            libc::raise(sig);
        }
        return;
    }
    // Signal and pipe byte land before `cancel`: a reader that sees
    // `cancel` set also sees both.
    SIGNAL.store(sig, Ordering::SeqCst);
    let fd = SHUTDOWN_FD.load(Ordering::SeqCst);
    // SAFETY: write(2) is async-signal-safe; `fd` is the non-blocking write
    // end leaked by `install_shutdown`, so it stays open and never blocks.
    unsafe { libc::write(fd, [1u8].as_ptr().cast(), 1) };
    if let Some(c) = CANCEL.get() {
        c.store(true, Ordering::SeqCst);
    }
}

/// A SIGINT/SIGTERM cut the run short; `main` exits `128 + signal`.
#[derive(Debug, thiserror::Error)]
#[error("golem: interrupted")]
pub struct Interrupted(pub libc::c_int);

/// Shutdown requested by SIGINT/SIGTERM: the handler sets the turn cancel
/// flag and makes the read end of a self-pipe readable until [`rearm`]. A
/// second signal before `rearm` restores `SIG_DFL` and re-raises, killing a
/// wedged process.
///
/// [`rearm`]: Shutdown::rearm
pub struct Shutdown {
    pipe: File,
    /// Set on the first signal; pass as the turn's cancel flag.
    pub cancel: Arc<AtomicBool>,
}

/// Installs the SIGINT/SIGTERM handlers; a second call in one process is an
/// error (the handlers share process-global state).
pub fn install_shutdown() -> io::Result<Shutdown> {
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "shutdown handlers already installed",
        ));
    }
    let (r, w) = pipe()?;
    for f in [&r, &w] {
        // SAFETY: `f` is an open fd owned by a live `File`; F_SETFL takes an int.
        if unsafe { libc::fcntl(f.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK) } == -1 {
            return Err(io::Error::last_os_error());
        }
    }
    let cancel = CANCEL.get_or_init(Arc::default).clone();
    SHUTDOWN_FD.store(w.into_raw_fd(), Ordering::SeqCst);
    for sig in [libc::SIGINT, libc::SIGTERM] {
        // SAFETY: a zeroed sigaction is a valid empty mask with no flags;
        // `on_shutdown` touches only atomics and calls async-signal-safe
        // functions.
        let rc = unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_shutdown as extern "C" fn(libc::c_int) as libc::sighandler_t;
            sa.sa_flags = libc::SA_RESTART;
            libc::sigaction(sig, &sa, std::ptr::null_mut())
        };
        if rc == -1 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(Shutdown { pipe: r, cancel })
}

impl Shutdown {
    /// The first signal received since install or the last [`rearm`].
    ///
    /// [`rearm`]: Shutdown::rearm
    pub fn signal(&self) -> Option<libc::c_int> {
        self.cancel
            .load(Ordering::SeqCst)
            .then(|| SIGNAL.load(Ordering::SeqCst))
    }

    /// `Err(Interrupted)` once a signal arrived.
    pub fn check(&self) -> Result<(), Interrupted> {
        self.signal().map_or(Ok(()), |s| Err(Interrupted(s)))
    }

    /// Forgets the signal after a cancelled turn, so the next one is again
    /// a first signal.
    pub fn rearm(&self) {
        self.cancel.store(false, Ordering::SeqCst);
        SIGNAL.store(0, Ordering::SeqCst);
        let mut b = [0u8; 16];
        while (&self.pipe).read(&mut b).is_ok_and(|n| n > 0) {}
        SIGNALLED.store(false, Ordering::SeqCst);
    }

    /// Line reader over stdin that also wakes on a signal.
    pub fn stdin_lines(&self) -> StdinLines<'_> {
        StdinLines {
            buf: Vec::new(),
            pipe: &self.pipe,
            eof: false,
        }
    }
}

/// One [`StdinLines::next_line`] result.
#[derive(Debug, PartialEq, Eq)]
pub enum Input {
    /// A line without its `\n`/`\r\n`.
    Line(String),
    /// Stdin closed.
    Eof,
    /// SIGINT/SIGTERM arrived while waiting.
    Signal(libc::c_int),
}

/// [`BufRead::lines`] over stdin that also wakes on the shutdown pipe.
/// Complete lines already read are served before polling again.
pub struct StdinLines<'a> {
    buf: Vec<u8>,
    pipe: &'a File,
    eof: bool,
}

impl StdinLines<'_> {
    /// The next line, EOF, or the signal that interrupted the wait.
    pub fn next_line(&mut self) -> io::Result<Input> {
        loop {
            if let Some(n) = self.buf.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = self.buf.drain(..=n).collect();
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                return utf8(line).map(Input::Line);
            }
            if self.eof {
                if self.buf.is_empty() {
                    return Ok(Input::Eof);
                }
                return utf8(std::mem::take(&mut self.buf)).map(Input::Line);
            }
            let mut fds = [
                libc::pollfd {
                    fd: self.pipe.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: io::stdin().as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            // SAFETY: `fds` is a valid 2-element pollfd array of open fds.
            if unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) } == -1 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            if fds[0].revents != 0 {
                return Ok(Input::Signal(SIGNAL.load(Ordering::SeqCst)));
            }
            if fds[1].revents != 0 {
                // Take everything std buffered, so poll never misses data
                // already pulled off the fd.
                let mut s = io::stdin().lock();
                let got = match s.fill_buf() {
                    Ok(got) => got,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
                    Err(e) => return Err(e),
                };
                self.eof = got.is_empty();
                self.buf.extend_from_slice(got);
                let n = got.len();
                s.consume(n);
            }
        }
    }
}

fn utf8(b: Vec<u8>) -> io::Result<String> {
    String::from_utf8(b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
