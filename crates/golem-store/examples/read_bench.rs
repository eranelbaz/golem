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
//! and final replica health. Errors return `Err` from `run`; `main` exits
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
}

fn parse_args() -> Result<Args, String> {
    let mut manifest = None;
    let mut n = DEFAULT_N;
    let mut plane_bytes = DEFAULT_PLANE_BYTES;
    let mut seed = 0u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
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
                    "unknown arg {a}; usage: read_bench --manifest <path> [--n <reads>] [--plane-bytes <usize>] [--seed <u64>]"
                ));
            }
        }
    }
    Ok(Args {
        manifest: PathBuf::from(manifest.ok_or("--manifest <path> is required")?),
        n,
        plane_bytes,
        seed,
    })
}

/// splitmix64 (finalizer + step): deterministic, well-mixed, no deps.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn run() -> Result<(), golem_store::StoreError> {
    let args = parse_args().map_err(|e| golem_store::StoreError::Invalid { msg: e })?;
    let manifest = load_manifest(&args.manifest)?;
    if manifest.shards.is_empty() {
        return Err(golem_store::StoreError::Invalid {
            msg: "manifest has no shards".into(),
        });
    }
    let registry = ReplicaRegistry::open(&manifest)?;
    let queue = Arc::new(ChunkQueue::new(0)); // 0 = one reader thread per physical drive (§5.2)

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
    // (§5.4 plan shape); one preallocation, reused for every read.
    let mut buf = vec![0u8; args.plane_bytes + 2 * CHUNK + 4096];

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
        let shard = &manifest.shards[s];
        let usable = shard.bytes.saturating_sub(args.plane_bytes as u64).max(1);
        let steps = (usable / CHUNK as u64).max(1);
        let offset = (mix64(args.seed ^ mix64(i as u64)) % steps) * CHUNK as u64;
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

fn main() {
    // Documented error exit: the bench has no other sensible recovery.
    if let Err(e) = run() {
        eprintln!("read_bench: {e}");
        std::process::exit(1);
    }
}
