#!/usr/bin/env python3
"""Drop the page cache over a model's GGUF shards via POSIX_FADV_DONTNEED.

Unprivileged (works on files we can open read-only); repeatable; no root,
no drop_caches. Usage: evict-model-cache.py <shard.gguf> [<shard.gguf> ...]
"""
import os
import sys

POSIX_FADV_DONTNEED = 4


def main(paths: list[str]) -> int:
    rc = 0
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            # offset=0, len=0 → the whole file.
            os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
            print(f"fadvise DONTNEED: {p}")
        except OSError as e:
            print(f"FAIL {p}: {e}", file=sys.stderr)
            rc = 1
        finally:
            os.close(fd)
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
