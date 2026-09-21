"""Single-leg mutual exclusion and pre-run liveness checks."""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys


def acquire_lock(outdir: str) -> int:
    """One model leg at a time (shared RAM/page cache/GPU).

    flock advisory lock: the kernel releases it when the holder dies
    (SIGKILL/OOM), so stale reclaim is automatic and race-free — unlike
    O_EXCL create + liveness check, which has a reclaim TOCTOU window.
    Returns the fd (keep it open for the leg's lifetime).
    """
    os.makedirs(outdir, exist_ok=True)
    lock = os.path.join(outdir, ".leg-lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = ""
        try:
            holder = " (pid " + os.read(fd, 64).decode(errors="replace").strip() + ")"
        except OSError:
            pass
        os.close(fd)
        sys.exit(f"FATAL: another leg holds {lock}{holder}.")
    os.truncate(fd, 0)
    os.pwrite(fd, str(os.getpid()).encode(), 0)  # diagnostics only
    return fd


def release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _live_model_pids(bin_path: str) -> list[str]:
    live = []
    targets = {os.path.realpath(bin_path)}
    r = subprocess.run(["pgrep", "-x", "llama-cli"], capture_output=True, text=True)
    for pid in r.stdout.split():
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                # stat: pid (comm) state ... — comm can contain spaces/parens,
                # so anchor on the last ')'.
                state = f.read().rsplit(b")", 1)[1].split()[0]
            if state == b"Z":
                # pgrep -x also matches zombies (a defunct child systemd --user
                # never reaped), which would block legs forever. Treat as dead.
                continue
            try:
                exe = os.path.realpath(f"/proc/{pid}/exe")
                if exe not in targets:
                    continue
            except OSError:
                pass  # exe unreadable — fall back to the name match
        except OSError:
            continue
        live.append(pid)
    return live


def preflight(bin_path: str) -> None:
    if not os.path.exists(bin_path):
        sys.exit(f"FATAL: binary not found: {bin_path}")
    if not os.access(bin_path, os.X_OK):
        sys.exit(f"FATAL: binary not executable: {bin_path}")
    live = _live_model_pids(bin_path)
    if live:
        sys.exit(f"FATAL: llama-cli already running (pid(s): {', '.join(live)}) — "
                 f"legs must run strictly sequentially.")
