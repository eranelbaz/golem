//! Store read benchmark (design §5.2, §12.3): N plane-sized reads through
//! the shared [`ChunkQueue`] against a manifest's replica set. Stands in for
//! the decode hot path's disk leg so A/B bandwidth (one drive vs both) is
//! reproducible without a full model run.
//!
//! Cold vs warm: the caller controls the page cache. For a cold run, pass a
//! fresh `--seed`: offsets are hash-scattered over each shard's usable range,
//! so different seeds touch disjoint-ish regions that a prior run's buffered
//! fallback (H-9) may have left in page cache. `drop_caches` still gives the
//! truest cold state:
//!
//! ```text
//! sync
//! echo 3 | sudo tee /proc/sys/vm/drop_caches
//! ```
//!
//! Usage:
//! ```text
//! cargo run -p golem-store --example read_bench -- \
//!     --manifest /path/to/manifest.json --n 200 --plane-bytes 5898496 --seed 7
//! ```
//!
//! Prints per-replica expected chunk share, wall-clock and disk-time GB/s,
//! and final replica health.
//!
//! `--whole` bypasses the queue: one 4 KiB-aligned `O_DIRECT` `pread` per
//! plane on replica `--replica <i>` (default 0). With `max_sectors_kb = 128`
//! a 5.9 MB read is ≈45 device requests in flight, a 2 MiB chunk 16 (§5.4). Errors return `Err` from `run`; `main` exits
//! nonzero on error (the one documented exit path in this binary).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use golem_store::{
    CHUNK, ChunkQueue, ReadJob, ReplicaHealth, ReplicaRegistry, load_manifest, plan_aligned,
};

const DEFAULT_N: usize = 200;
const DEFAULT_PLANE_BYTES: usize = 5_898_496; // GLM-5.3-Flash sparse plane ≈5.9 MB

struct Args {
    manifest: PathBuf,
    n: usize,
    plane_bytes: usize,
    seed: u64,
    io_threads: usize,
    sequential: bool,
    whole: bool,
    replica: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut manifest = None;
    let mut n = DEFAULT_N;
    let mut plane_bytes = DEFAULT_PLANE_BYTES;
    let mut seed = 0u64;
    let mut io_threads = 0usize;
    let mut sequential = false;
    let mut whole = false;
    let mut replica = 0usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sequential" => {
                sequential = true;
            }
            "--whole" => {
                whole = true;
            }
            "--replica" => {
                replica = it
                    .next()
                    .ok_or("--replica needs a value")?
                    .parse()
                    .map_err(|_| "--replica must be a usize")?;
            }
            "--io-threads" => {
                io_threads = it
                    .next()
                    .ok_or("--io-threads needs a value")?
                    .parse()
                    .map_err(|_| "--io-threads must be an integer")?;
            }
            "--manifest" => {
                manifest = Some(it.next().ok_or("--manifest needs a path")?);
            }
            "--n" => {
                n = it
                    .next()
                    .ok_or("--n needs a value")?
                    .parse()
                    .map_err(|_| "--n must be a usize")?;
            }
            "--plane-bytes" => {
                plane_bytes = it
                    .next()
                    .ok_or("--plane-bytes needs a value")?
                    .parse()
                    .map_err(|_| "--plane-bytes must be a usize")?;
            }
            "--seed" => {
                seed = it
                    .next()
                    .ok_or("--seed needs a value")?
                    .parse()
                    .map_err(|_| "--seed must be a u64")?;
            }
            _ => {
                return Err(format!(
                    "unknown arg {a}; usage: read_bench --manifest <path> [--n <reads>] [--plane-bytes <usize>] [--seed <u64>] [--sequential] [--whole [--replica <i>]] [--io-threads <usize>]"
                ));
            }
        }
    }
    Ok(Args {
        manifest: PathBuf::from(manifest.ok_or("--manifest <path> is required")?),
        n,
        plane_bytes,
        seed,
        io_threads,
        sequential,
        whole,
        replica,
    })
}

/// splitmix64 (finalizer + step): deterministic, well-mixed, no deps.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Read `i`'s offset in a shard of `bytes`, shared by both modes.
fn read_offset(args: &Args, bytes: u64, i: usize, shard_count: usize) -> u64 {
    let usable = bytes.saturating_sub(args.plane_bytes as u64).max(1);
    let steps = (usable / CHUNK as u64).max(1);
    if args.sequential {
        // Real page_in shape: a cold layer's planes are contiguous per
        // shard, so consecutive reads walk the file forward.
        ((i / shard_count) as u64 * args.plane_bytes as u64) % usable
    } else {
        (mix64(args.seed ^ mix64(i as u64)) % steps) * CHUNK as u64
    }
}

fn run() -> Result<(), golem_store::StoreError> {
    let args = parse_args().map_err(|e| golem_store::StoreError::Invalid { msg: e })?;
    let manifest = load_manifest(&args.manifest)?;
    if manifest.shards.is_empty() {
        return Err(golem_store::StoreError::Invalid {
            msg: "manifest has no shards".into(),
        });
    }
    if args.whole {
        return run_whole(&args, &manifest);
    }
    let registry = ReplicaRegistry::open(&manifest)?;
    let queue = Arc::new(ChunkQueue::new(args.io_threads, None)); // 0 = one reader thread per physical drive (§5.2); 2 = two per drive (D-12)

    // Startup hygiene (H-9): a replica whose open-time probe failed serves
    // buffered from byte 0 — its GB/s is a page-cache number, not a disk one.
    for (s, shard) in registry.shards().iter().enumerate() {
        for (r, slot) in shard.replicas.iter().enumerate() {
            if !slot.direct_available() {
                eprintln!(
                    "WARNING: shard {s} replica {r} ({}) failed the O_DIRECT probe; reads will be buffered — numbers are not disk bandwidth",
                    slot.path.display()
                );
            }
        }
    }

    // Per-read covering region ≤ logical len + head scratch + one chunk tail
    // (§5.4 plan shape); one preallocation, reused for every read. The region
    // start must be 4 KiB-aligned or every read silently takes the buffered
    // fallback (§5.4 direct reads require an aligned buffer).
    let mut raw = vec![0u8; args.plane_bytes + 2 * CHUNK + 2 * 4096];
    let skip = 4096 - raw.as_ptr() as usize % 4096;
    let buf = &mut raw[skip..];

    // Seed-scattered offsets: hash(seed, i) spread over the shard's usable
    // range in CHUNK-aligned plane steps, so different seeds touch different
    // file regions (cold reads without drop_caches). Offsets wrap so the loop
    // works on any shard size; plan_aligned keeps every read CHUNK-aligned.
    let mut total_logical = 0u64;
    let mut total_chunks = 0usize;
    let mut shard0_chunks = 0usize;
    let shard_count = manifest.shards.len();
    let t0 = Instant::now();
    for i in 0..args.n {
        let s = i % shard_count;
        let offset = read_offset(&args, manifest.shards[s].bytes, i, shard_count);
        let plan = plan_aligned(offset, args.plane_bytes);
        let covering: usize = plan.iter().map(|c| c.1).sum();
        let (dst, rest) = buf.split_at_mut(covering);
        let handle = queue.enqueue(ReadJob {
            shard_index: s,
            offset,
            len: args.plane_bytes,
            expected_crc: 0,
            verify_crc: false, // bandwidth bench, not integrity
            prefetch: false,
            sources: &registry,
            dst,
        });
        handle.wait()?;
        let _ = rest;
        total_logical += args.plane_bytes as u64;
        total_chunks += plan.len();
        if s == 0 {
            shard0_chunks += plan.len();
        }
    }
    let wall = t0.elapsed();

    let healthy: Vec<usize> = registry.shards()[0]
        .replicas
        .iter()
        .enumerate()
        .filter(|(_, r)| r.health() == ReplicaHealth::Healthy)
        .map(|(i, _)| i)
        .collect();
    let replica_count = registry.shards()[0].replicas.len();
    eprintln!("shard 0: {replica_count} replica(s), healthy: {healthy:?}");

    // Measured share: chunks each replica actually served (§5.2 striping
    // homes chunks round-robin across healthy replicas; skew shows up here).
    for (r, slot) in registry.shards()[0].replicas.iter().enumerate() {
        let served = slot.served_chunks();
        let direct = slot.served_direct();
        eprintln!(
            "replica {r}: {served}/{shard0_chunks} chunks ({:.1}%), direct: {direct}, buffered: {}",
            if shard0_chunks > 0 {
                100.0 * served as f64 / shard0_chunks as f64
            } else {
                0.0
            },
            served - direct
        );
        if served > 0 && !slot.direct_ok() {
            eprintln!(
                "WARNING: replica {r} fell back to buffered mid-run (direct_ok == false); its share of the numbers is NOT O_DIRECT"
            );
        }
    }

    let (disk_ns, disk_bytes) = queue.take_disk_stats();
    let wall_gbs = total_logical as f64 / wall.as_secs_f64() / 1e9;
    let disk_gbs = if disk_ns > 0 {
        disk_bytes as f64 / disk_ns as f64
    } else {
        0.0
    };
    println!("reads: {} × {} B (logical)", args.n, args.plane_bytes);
    println!("chunks total: {total_chunks}");
    println!("wall: {:.3?} → {:.2} GB/s", wall, wall_gbs);
    println!("disk (measured): {:.2} GB/s", disk_gbs);
    println!("degraded replicas after run: {:?}", {
        registry
            .shards()
            .iter()
            .flat_map(|s| s.replicas.iter())
            .filter(|r| r.health() == ReplicaHealth::Degraded)
            .map(|r| r.path.display().to_string())
            .collect::<Vec<_>>()
    });
    Ok(())
}

/// `--whole`: one aligned `O_DIRECT` read per plane, same offsets as the
/// queued run, rotating over shards on replica `--replica`.
fn run_whole(
    args: &Args,
    manifest: &golem_store::ModelManifest,
) -> Result<(), golem_store::StoreError> {
    use std::os::unix::fs::FileExt;
    let align = golem_store::sys::DIRECT_ALIGN;
    let files = manifest
        .shards
        .iter()
        .map(|s| {
            let root =
                s.replicas
                    .get(args.replica)
                    .ok_or_else(|| golem_store::StoreError::Invalid {
                        msg: format!("shard {} has no replica {}", s.file, args.replica),
                    })?;
            let path = root.join(&s.file);
            golem_store::sys::open_direct(&path).map_err(|e| golem_store::StoreError::Invalid {
                msg: format!("{}: O_DIRECT open: {e}", path.display()),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(s) = manifest
        .shards
        .iter()
        .find(|s| s.bytes < args.plane_bytes as u64)
    {
        return Err(golem_store::StoreError::Invalid {
            msg: format!("shard {} ({} B) is smaller than one plane", s.file, s.bytes),
        });
    }
    let span = (args.plane_bytes as u64).div_ceil(align) * align + align;
    let mut raw = vec![0u8; span as usize + align as usize];
    let skip = (align as usize - raw.as_ptr() as usize % align as usize) % align as usize;
    let buf = &mut raw[skip..skip + span as usize];
    let shard_count = manifest.shards.len();
    let mut device = 0u64;
    let t0 = Instant::now();
    for i in 0..args.n {
        let s = i % shard_count;
        let offset = read_offset(args, manifest.shards[s].bytes, i, shard_count);
        let start = offset - offset % align;
        let len = ((offset + args.plane_bytes as u64).div_ceil(align) * align - start) as usize;
        let got = files[s].read_at(&mut buf[..len], start).map_err(|e| {
            golem_store::StoreError::Invalid {
                msg: format!("shard {s} pread @{start}+{len}: {e}"),
            }
        })?;
        // The aligned tail may pass EOF; the logical plane must not.
        if (start + got as u64) < offset + args.plane_bytes as u64 {
            return Err(golem_store::StoreError::Invalid {
                msg: format!("shard {s} short read @{start}: {got}/{len}"),
            });
        }
        device += got as u64;
    }
    let wall = t0.elapsed();
    let total = args.n as u64 * args.plane_bytes as u64;
    println!(
        "reads: {} × {} B (logical), whole-plane O_DIRECT",
        args.n, args.plane_bytes
    );
    let secs = wall.as_secs_f64();
    println!(
        "wall: {:.3?} → {:.2} GB/s logical, {:.2} GB/s device",
        wall,
        total as f64 / secs / 1e9,
        device as f64 / secs / 1e9
    );
    Ok(())
}

fn main() {
    // Documented error exit: the bench has no other sensible recovery.
    if let Err(e) = run() {
        eprintln!("read_bench: {e}");
        std::process::exit(1);
    }
}
