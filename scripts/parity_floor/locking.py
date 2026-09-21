"""Single-leg mutual exclusion and pre-run liveness checks."""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys


def acquire_lock(outdir: str) -> int:
    """One model leg at a time (shared RAM/page cache/GPU).

    flock so the kernel releases the lock when the holder dies (SIGKILL/OOM);
    returns the fd — keep it open for the leg's lifetime.
    """
    os.makedirs(outdir, exist_ok=True)
    lock_path = os.path.join(outdir, ".leg-lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder_pid = _read_holder_pid(lock_fd)
        os.close(lock_fd)
        sys.exit(f"FATAL: another leg holds {lock_path}{holder_pid}.")
    os.truncate(lock_fd, 0)
    os.pwrite(lock_fd, str(os.getpid()).encode(), 0)
    return lock_fd


def _read_holder_pid(lock_fd: int) -> str:
    try:
        holder = os.read(lock_fd, 64).decode(errors="replace").strip()
        return f" (pid {holder})" if holder else ""
    except OSError:
        return ""


def release_lock(lock_fd: int) -> None:
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _live_model_pids(bin_path: str) -> list[str]:
    live_pids = []
    target_paths = {os.path.realpath(bin_path)}
    pgrep = subprocess.run(["pgrep", "-x", "llama-cli"], capture_output=True, text=True)
    for pid in pgrep.stdout.split():
        try:
            if _process_state(pid) == b"Z":
                continue  # zombie: parent never reaped it, must not block legs
            if os.path.realpath(f"/proc/{pid}/exe") in target_paths:
                live_pids.append(pid)
        except OSError:
            continue
    return live_pids


def _process_state(pid: str) -> bytes:
    # /proc/<pid>/stat: pid (comm) state ... — comm can contain spaces/parens,
    # so anchor on the last ')'.
    with open(f"/proc/{pid}/stat", "rb") as stat_file:
        return stat_file.read().rsplit(b")", 1)[1].split()[0]


def preflight(bin_path: str) -> None:
    if not os.path.exists(bin_path):
        sys.exit(f"FATAL: binary not found: {bin_path}")
    if not os.access(bin_path, os.X_OK):
        sys.exit(f"FATAL: binary not executable: {bin_path}")
    live_pids = _live_model_pids(bin_path)
    if live_pids:
        sys.exit(f"FATAL: llama-cli already running (pid(s): {', '.join(live_pids)}) — "
                 f"legs must run strictly sequentially.")
