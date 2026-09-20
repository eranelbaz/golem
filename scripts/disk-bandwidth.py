#!/usr/bin/env python3
"""O_DIRECT pread64 disk bandwidth benchmark (M0.4, design §14).

O_DIRECT bypasses the page cache, so no root drop_caches is needed.
Chunk = 8 MiB (expert-read scale, design §5.4). Legs: per-file and
striped-across-files x {seq, rand} x {1, 8 threads}.
"""
import argparse, hashlib, json, mmap, os, random, shutil, tempfile, time
from concurrent.futures import ThreadPoolExecutor

CHUNK = 8 << 20  # 8 MiB
ALIGN = 4096


def open_direct(path):
    return os.open(path, os.O_RDONLY | os.O_DIRECT)


def aligned_buf():
    return mmap.mmap(-1, CHUNK)  # page-aligned allocation for O_DIRECT


def read_at(fd, buf, offset):
    mv = memoryview(buf)
    done = 0
    while done < CHUNK:
        n = os.preadv(fd, [mv[done:]], offset + done)
        if n == 0:
            raise OSError(f"preadv returned 0 at offset {offset + done}")
        done += n


def seq_offsets(fsize, nbytes):
    out, off = [], 0
    while off + CHUNK <= fsize and len(out) * CHUNK < nbytes:
        out.append(off)
        off += CHUNK
    return out


def rand_offsets(fsize, nbytes, seed):
    if fsize < 2 * CHUNK:
        raise ValueError(f"file too small for rand pattern: {fsize} bytes")
    rnd = random.Random(seed)
    hi = (fsize - CHUNK) // ALIGN
    n = min(fsize, nbytes) // CHUNK
    return [rnd.randrange(0, hi) * ALIGN for _ in range(n)]


def run_jobs(jobs, threads):
    fds = sorted({fd for fd, _, _ in jobs})
    bufs = {fd: aligned_buf() for fd in fds}
    work = [(fd, bufs[fd], off) for fd, _, off in jobs]
    random.Random(1).shuffle(work)  # same interleaving discipline per leg
    try:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda j: read_at(j[0], j[1], j[2]), work))
        dt = time.perf_counter() - t0
    finally:
        for fd in fds:
            os.close(fd)
    total = len(work) * CHUNK
    return {"bytes": total, "seconds": round(dt, 3),
            "gib_per_sec": round(total / dt / 2**30, 2)}


def selftest():
    # /tmp is often tmpfs, which rejects O_DIRECT — use a scratch dir on the
    # caller's (disk-backed) filesystem instead.
    sysd = tempfile.mkdtemp(dir=os.environ.get("GOLEM_SELFTEST_DIR", "."))
    try:
        _selftest_body(sysd)
    finally:
        shutil.rmtree(sysd, ignore_errors=True)


def _selftest_body(sysd):
    path = os.path.join(sysd, "t.bin")
    payload = bytes(256) * (CHUNK // 256)
    with open(path, "wb") as f:
        for i in range(8):  # 64 MiB
            f.write(payload if i % 2 else b"\xab" * CHUNK)
    fd = open_direct(path)
    buf = aligned_buf()
    read_at(fd, buf, 0)  # first chunk = b"\xab" * CHUNK
    os.close(fd)
    assert hashlib.sha256(buf).hexdigest() == hashlib.sha256(b"\xab" * CHUNK).hexdigest()
    print("SELFTEST-OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True)
    ap.add_argument("--pattern", choices=["seq", "rand"], required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--bytes-gb", type=int, default=4)
    ap.add_argument("--striped", action="store_true",
                    help="stripe chunk offsets round-robin across all files")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    nbytes = args.bytes_gb << 30
    paths = args.file
    sizes = [os.path.getsize(p) for p in paths]
    leg_files = "+".join(os.path.basename(p) for p in paths)

    if args.striped:
        offs = [seq_offsets(s, nbytes // len(paths)) if args.pattern == "seq"
                else rand_offsets(s, nbytes // len(paths), 7) for s in sizes]
        fds = [open_direct(p) for p in paths]
        jobs = [(fds[i], None, o) for i, ol in enumerate(offs) for o in ol]
        res = run_jobs(jobs, args.threads)
    else:
        results = []
        for p, s in zip(paths, sizes):
            offs = seq_offsets(s, nbytes) if args.pattern == "seq" else rand_offsets(s, nbytes, 7)
            fd = open_direct(p)
            results.append(run_jobs([(fd, None, o) for o in offs], args.threads))
        agg_bytes = sum(r["bytes"] for r in results)
        agg_secs = sum(r["seconds"] for r in results)
        res = {"bytes": agg_bytes, "seconds": round(agg_secs, 3),
               "gib_per_sec": round(agg_bytes / agg_secs / 2**30, 2)}

    leg = {"files": leg_files, "pattern": args.pattern, "threads": args.threads,
           "striped": args.striped, **res}
    print(json.dumps(leg))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(leg) + "\n")


if __name__ == "__main__":
    main()
