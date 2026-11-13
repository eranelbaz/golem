//! Decode stall watchdog: a `llama_decode` (plus its synchronize) running
//! past `GOLEM_STALL_LOG_SECS` (default 300, 0 = off) logs one WARN per
//! interval with the batch shape, the hook progress since the decode began
//! (or the last report), the last hook and layer, and a per-thread
//! `/proc/self/task` snapshot (state, CPU ticks over the interval, wchan),
//! so a hung compute leaves evidence even when ptrace is unavailable.
//!
//! State is process-wide: one decode is watched at a time (pooled sessions
//! are one per process, §3.4); a scope only clears its own id.
//!
//! Hot-path cost: a few relaxed stores per compute, one relaxed add and two
//! relaxed stores per hook call (per layer, never per expert). The watchdog
//! thread owns all timing and `/proc` reads.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The FFI hook a callback belongs to (the `last_hook` field).
#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum Hook {
    CpuLease = 1,
    EnsureHostResident,
    CopyDone,
    StreamSource,
    StreamRelease,
    EvictDestBatch,
    EvictAbandon,
    Predict,
    Routing,
}

const HOOK_NAMES: [&str; 10] = [
    "none",
    "cpu_lease",
    "ensure_host_resident",
    "copy_done",
    "stream_source",
    "stream_release",
    "evict_dest_batch",
    "evict_abandon",
    "predict",
    "routing",
];

impl Hook {
    pub(crate) fn name(self) -> &'static str {
        HOOK_NAMES[self as usize]
    }
}

/// Decodes entered so far; a scope's id is its value after the increment.
static DECODE_SEQ: AtomicU64 = AtomicU64::new(0);
/// The in-flight decode's id (0 = none).
static ACTIVE: AtomicU64 = AtomicU64::new(0);
/// Entry time of the in-flight decode, ns since [`epoch`].
static START_NS: AtomicU64 = AtomicU64::new(0);
static START_CALLS: AtomicU64 = AtomicU64::new(0);
static TOKENS: AtomicU32 = AtomicU32::new(0);
static GEN: AtomicU32 = AtomicU32::new(0);
static STREAM: AtomicU8 = AtomicU8::new(0);
static HOOK_CALLS: AtomicU64 = AtomicU64::new(0);
static LAST_HOOK: AtomicU8 = AtomicU8::new(0);
static LAST_BLK: AtomicI32 = AtomicI32::new(-1);

fn epoch() -> Instant {
    static E: OnceLock<Instant> = OnceLock::new();
    *E.get_or_init(Instant::now)
}

/// Marks one `llama_decode` + synchronize in flight until dropped.
pub(crate) struct DecodeScope(u64);

impl DecodeScope {
    pub(crate) fn enter(tokens: u32, n_gen: u32, stream: bool) -> Self {
        spawn_once();
        TOKENS.store(tokens, Ordering::Relaxed);
        GEN.store(n_gen, Ordering::Relaxed);
        STREAM.store(stream as u8, Ordering::Relaxed);
        LAST_HOOK.store(0, Ordering::Relaxed);
        LAST_BLK.store(-1, Ordering::Relaxed);
        START_CALLS.store(HOOK_CALLS.load(Ordering::Relaxed), Ordering::Relaxed);
        START_NS.store(epoch().elapsed().as_nanos() as u64, Ordering::Relaxed);
        let id = DECODE_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
        ACTIVE.store(id, Ordering::Release);
        DecodeScope(id)
    }
}

impl Drop for DecodeScope {
    fn drop(&mut self) {
        let _ = ACTIVE.compare_exchange(self.0, 0, Ordering::Release, Ordering::Relaxed);
    }
}

pub(crate) fn hook(h: Hook) {
    LAST_HOOK.store(h as u8, Ordering::Relaxed);
    HOOK_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// The layer (`blk`) the current hook call is serving.
pub(crate) fn blk(blk: i32) {
    LAST_BLK.store(blk, Ordering::Relaxed);
}

fn threshold() -> Option<Duration> {
    let secs = match std::env::var("GOLEM_STALL_LOG_SECS") {
        Err(_) => 300,
        Ok(v) => v.trim().parse::<u64>().unwrap_or_else(|_| {
            tracing::warn!(value = %v, "GOLEM_STALL_LOG_SECS: not a whole number; using 300");
            300
        }),
    };
    (secs > 0).then(|| Duration::from_secs(secs))
}

fn spawn_once() {
    static SPAWNED: OnceLock<()> = OnceLock::new();
    SPAWNED.get_or_init(|| {
        epoch();
        let Some(limit) = threshold() else {
            return;
        };
        let tick = (limit / 10).clamp(Duration::from_secs(1), Duration::from_secs(30));
        if let Err(e) = std::thread::Builder::new()
            .name("golem-stall".into())
            .spawn(move || watch(limit, tick))
        {
            tracing::warn!(error = %e, "stall watchdog not started");
        }
    });
}

/// The decode being reported on.
struct Seen {
    id: u64,
    start_ns: u64,
    last_report_ns: Option<u64>,
    calls_then: u64,
}

fn watch(limit: Duration, tick: Duration) {
    let mut seen: Option<Seen> = None;
    let mut ticks: HashMap<String, u64> = HashMap::new();
    loop {
        std::thread::sleep(tick);
        let id = ACTIVE.load(Ordering::Acquire);
        let now_ns = epoch().elapsed().as_nanos() as u64;
        if seen.as_ref().is_some_and(|s| s.id != id)
            && let Some(s) = seen.take().filter(|s| s.last_report_ns.is_some())
        {
            tracing::warn!(
                decode = s.id,
                elapsed_s = now_ns.saturating_sub(s.start_ns) / 1_000_000_000,
                "decode stall cleared"
            );
        }
        if id == 0 {
            continue;
        }
        let s = seen.get_or_insert_with(|| Seen {
            id,
            start_ns: START_NS.load(Ordering::Relaxed),
            last_report_ns: None,
            calls_then: START_CALLS.load(Ordering::Relaxed),
        });
        let limit_ns = limit.as_nanos() as u64;
        let since = |t: u64| now_ns.saturating_sub(t);
        if since(s.start_ns) < limit_ns || s.last_report_ns.is_some_and(|t| since(t) < limit_ns) {
            continue;
        }
        let calls = HOOK_CALLS.load(Ordering::Relaxed);
        tracing::warn!(
            decode = id,
            elapsed_s = since(s.start_ns) / 1_000_000_000,
            tokens = TOKENS.load(Ordering::Relaxed),
            gen = GEN.load(Ordering::Relaxed),
            stream = STREAM.load(Ordering::Relaxed) != 0,
            hook_calls_since_last_report = calls.saturating_sub(s.calls_then),
            last_hook = HOOK_NAMES[LAST_HOOK.load(Ordering::Relaxed) as usize % HOOK_NAMES.len()],
            last_blk = LAST_BLK.load(Ordering::Relaxed),
            threads = %threads(&mut ticks),
            "decode stall: llama_decode has not returned"
        );
        s.last_report_ns = Some(now_ns);
        s.calls_then = calls;
    }
}

/// `tid:comm:state:cpu_ticks_since_last:wchan` per thread, `;`-separated;
/// `prev` carries each thread's utime+stime between calls.
fn threads(prev: &mut HashMap<String, u64>) -> String {
    let Ok(dir) = std::fs::read_dir("/proc/self/task") else {
        return "unavailable".into();
    };
    let mut out = String::new();
    let mut next = HashMap::with_capacity(prev.len());
    for e in dir.flatten() {
        let p = e.path();
        let Ok(stat) = std::fs::read_to_string(p.join("stat")) else {
            continue;
        };
        // comm may contain spaces or parens: fields follow the last ')'.
        let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else {
            continue;
        };
        let comm = &stat[open + 1..close];
        let f: Vec<&str> = stat.get(close + 2..).unwrap_or("").split(' ').collect();
        let tick = |i: usize| f.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        let total = tick(11) + tick(12);
        let tid = e.file_name().to_string_lossy().into_owned();
        let delta = total.saturating_sub(prev.get(&tid).copied().unwrap_or(0));
        let wchan = std::fs::read_to_string(p.join("wchan")).unwrap_or_default();
        let wchan = match wchan.trim() {
            "" | "0" => "-",
            w => w,
        };
        if !out.is_empty() {
            out.push(';');
        }
        out.push_str(&format!(
            "{tid}:{comm}:{}:{delta}:{wchan}",
            f.first().copied().unwrap_or("?")
        ));
        next.insert(tid, total);
    }
    *prev = next;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_ids_name_their_hook() {
        assert_eq!(Hook::CpuLease.name(), "cpu_lease");
        assert_eq!(Hook::Routing.name(), "routing");
        assert_eq!(Hook::Routing as usize, HOOK_NAMES.len() - 1);
    }

    #[test]
    fn thread_snapshot_lists_the_caller() {
        let tid = std::fs::read_link("/proc/thread-self")
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .expect("thread-self");
        let mut prev = HashMap::new();
        let snap = threads(&mut prev);
        assert!(snap.split(';').any(|t| t.starts_with(&format!("{tid}:"))));
        assert!(prev.contains_key(&tid));
    }

    #[test]
    fn a_stale_scope_does_not_clear_a_newer_decode() {
        let a = DecodeScope::enter(1, 1, false);
        let b = DecodeScope::enter(1, 1, false);
        let id = b.0;
        drop(a);
        assert_eq!(ACTIVE.load(Ordering::Acquire), id);
        drop(b);
    }
}
