//! Streamed-prefill read replay (design §7.6.3): `--readers` threads each
//! drain whole expert planes through [`StoreExpertSource::read_expert_batched`]
//! into page-aligned per-expert regions, the layout the stream staging ring
//! uses, so the store's plane-batch rate is measured without a model run.
//!
//! Usage:
//! ```text
//! cargo run --release -p golem-placement --example stream_read_bench -- \
//!     --manifest /path/to/manifest.json [--readers 3] [--planes 60] \
//!     [--ids 384] [--io-depth 2]
//! ```
//!
//! CRC verification follows the manifest (`crc32c` per expert entry); strip
//! it from a copy of the manifest to measure without.
//!
//! Errors return `Err` from `run`; `main` exits nonzero on error (the one
//! documented exit path in this binary).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use golem_placement::pool::{DIRECT_ALIGN, ExpertKind, ExpertSource, StoreExpertSource};
use golem_store::ChunkQueue;

struct Args {
    manifest: PathBuf,
    readers: usize,
    planes: usize,
    ids: u32,
    io_depth: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        manifest: PathBuf::new(),
        readers: 3,
        planes: 60,
        ids: u32::MAX,
        io_depth: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--manifest" => a.manifest = v()?.into(),
            "--readers" => a.readers = v()?.parse().map_err(|e| format!("{e}"))?,
            "--planes" => a.planes = v()?.parse().map_err(|e| format!("{e}"))?,
            "--ids" => a.ids = v()?.parse().map_err(|e| format!("{e}"))?,
            "--io-depth" => a.io_depth = v()?.parse().map_err(|e| format!("{e}"))?,
            _ => return Err(format!("unknown arg {k}")),
        }
    }
    if a.manifest.as_os_str().is_empty() || a.readers == 0 {
        return Err(
            "usage: --manifest <path> [--readers N] [--planes N] [--ids N] [--io-depth N]".into(),
        );
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let a = parse_args()?;
    let queue = Arc::new(ChunkQueue::new(0, None).with_io_depth(a.io_depth));
    let store = StoreExpertSource::open(&a.manifest, queue).map_err(|e| e.to_string())?;
    let kinds = [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down];
    // Every live (layer, kind) plane with its expert count, in stream order.
    let mut work: Vec<(u32, ExpertKind, u32, usize)> = Vec::new();
    for layer in 0..256u32 {
        for kind in kinds {
            if store.plane(layer, kind, 0).is_none() {
                continue;
            }
            let n = (0..)
                .take_while(|&e| store.plane(layer, kind, e).is_some())
                .count() as u32;
            let stride = store.plane(layer, kind, 0).map_or(0, |p| p.1) as usize;
            work.push((layer, kind, n.min(a.ids), stride));
        }
    }
    if work.is_empty() {
        return Err("manifest has no expert planes".into());
    }
    let page = DIRECT_ALIGN as usize;
    let max_region = work
        .iter()
        .map(|w| w.2 as usize * (w.3 + 2 * page).next_multiple_of(page))
        .max()
        .unwrap_or(0);
    eprintln!(
        "{} planes in the model; replaying {} with {} readers, ≤ {} ids each, crc {}",
        work.len(),
        a.planes,
        a.readers,
        a.ids,
        store.crc_enabled()
    );
    let next = AtomicUsize::new(0);
    let bytes = AtomicUsize::new(0);
    let t0 = Instant::now();
    let busy: Vec<f64> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..a.readers)
            .map(|_| {
                s.spawn(|| -> Result<f64, String> {
                    let mut buf = vec![0u8; max_region + page];
                    let skip = (page - buf.as_ptr() as usize % page) % page;
                    let mut busy = 0.0;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= a.planes {
                            return Ok(busy);
                        }
                        let (layer, kind, n, stride) = work[i % work.len()];
                        let mut batch = Vec::with_capacity(n as usize);
                        let mut targets: Vec<(&mut [u8], usize)> = Vec::with_capacity(n as usize);
                        let mut rest: &mut [u8] = &mut buf[skip..];
                        for e in 0..n {
                            let head =
                                store.plane(layer, kind, e).map_or(0, |p| p.0) as usize % page;
                            let len = (head + stride).next_multiple_of(page);
                            let (r, tail) = std::mem::take(&mut rest).split_at_mut(len);
                            rest = tail;
                            batch.push((kind, e));
                            targets.push((r, head));
                        }
                        let t = Instant::now();
                        store
                            .read_expert_batched(layer, &batch, &mut targets)
                            .map_err(|e| e.to_string())?;
                        busy += t.elapsed().as_secs_f64();
                        bytes.fetch_add(n as usize * stride, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().map_err(|_| "reader panicked".to_string())?)
            .collect::<Result<_, _>>()
    })?;
    let wall = t0.elapsed().as_secs_f64();
    let gb = bytes.load(Ordering::Relaxed) as f64 / 1e9;
    let sum_busy: f64 = busy.iter().sum();
    println!(
        "{gb:.1} GB in {wall:.2} s → {:.2} GB/s; per drain {:.2} GB/s; serial fallbacks {}",
        gb / wall,
        gb / sum_busy,
        store.serial_fallbacks()
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("stream_read_bench: {e}");
        std::process::exit(1);
    }
}
