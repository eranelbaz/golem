//! `/metrics` (§11): Prometheus text format 0.0.4, hand-written. The worker
//! publishes a [`MetricsSnapshot`] at turn end and every 64 steps; the
//! handler only formats it — no engine access off the decode thread.

use std::fmt::Write;
use std::sync::atomic::Ordering;

use golem_api::ErrorCode;
use golem_engine::Session;

use crate::worker::HealthState;

/// Slots reported per snapshot (`--parallel` beyond this is truncated).
pub(crate) const MAX_SLOTS: usize = 32;

/// Every error code, in [`code_index`] order.
const CODES: [ErrorCode; 10] = [
    ErrorCode::BackendUnavailable,
    ErrorCode::PlanInfeasible,
    ErrorCode::ModelMismatch,
    ErrorCode::MultimodalUnsupported,
    ErrorCode::UnsupportedField,
    ErrorCode::StorageUnavailable,
    ErrorCode::InstanceNotFound,
    ErrorCode::NotFound,
    ErrorCode::Busy,
    ErrorCode::InvalidRequest,
];
pub(crate) const N_CODES: usize = CODES.len();

pub(crate) fn code_index(c: ErrorCode) -> usize {
    CODES.iter().position(|&x| x == c).unwrap_or(0)
}

/// Worker-side counters, copied out of the `Session` at the publish cadence.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetricsSnapshot {
    /// P4 device cache: bytes copied / skipped, evictions (pooled GPU only).
    pub device_cache: Option<(u64, u64, u64)>,
    /// Expert planes copied host→device: device-cache fills, streamed prefill.
    pub h2d_copies: Option<(u64, u64)>,
    /// P3: demand page-ins, fast hits, evictions, bytes paged, source errors.
    pub p3: Option<[u64; 5]>,
    /// §6.5: prefetch page-ins, hits, wasted bytes.
    pub prefetch: Option<(u64, u64, u64)>,
    /// §7.1 snapshot cache: hits, misses, restore failures, evictions,
    /// dropped oversize, used bytes.
    pub snapshots: Option<[u64; 6]>,
    /// Process-cumulative expert `MUL_MAT_ID` nodes placed (on GPU, on CPU);
    /// `None` unless the session offloads (only then must CPU stay 0, I2).
    pub placement: Option<(u64, u64)>,
    /// Expert pool + snapshot arena bytes, mlocked.
    pub locked_ram_bytes: u64,
    /// Device free bytes (`hipMemGetInfo`); `None` off HIP.
    pub vram_free: Option<u64>,
    /// Device total bytes.
    pub vram_total: Option<u64>,
    /// RAM pool slot fill (0..1).
    pub ram_fill: Option<f64>,
    /// Tokens generated per slot since load.
    pub slot_tokens: [u64; MAX_SLOTS],
    /// Slots reported.
    pub n_slots: usize,
}

impl MetricsSnapshot {
    /// Reads every counter from `s` (decode thread, between steps).
    pub(crate) fn refresh(&mut self, s: &Session, slot_tokens: &[u64]) {
        let dc = s.device_cache_stats();
        self.device_cache = dc
            .as_ref()
            .map(|d| (d.bytes_copied, d.bytes_skipped, d.evictions));
        self.h2d_copies = dc.map(|d| (d.cache_copies, d.stream_copies));
        self.p3 = s.p3_stats().map(|p| {
            [
                p.demand_slow_pageins.load(Ordering::Relaxed),
                p.fast_hits.load(Ordering::Relaxed),
                p.evictions.load(Ordering::Relaxed),
                p.bytes_paged.load(Ordering::Relaxed),
                p.source_errors.load(Ordering::Relaxed),
            ]
        });
        self.prefetch = s
            .prefetch_stats()
            .map(|p| (p.prefetch_pageins, p.prefetch_hits, p.prefetch_wasted_bytes));
        self.snapshots = s.snapshot_stats().map(|x| {
            [
                x.hits,
                x.misses,
                x.restore_failures,
                x.evictions,
                x.dropped_oversize,
                x.used_bytes,
            ]
        });
        self.placement = s
            .offloaded()
            .then(|| (s.expert_mul_mat_id_on_gpu(), s.expert_mul_mat_id_off_gpu()));
        self.locked_ram_bytes = s.locked_ram_bytes();
        let vram = device_mem();
        self.vram_free = vram.map(|v| v.0);
        self.vram_total = vram.map(|v| v.1);
        self.ram_fill = s.pool_fill();
        if !slot_tokens.is_empty() {
            self.n_slots = slot_tokens.len().min(MAX_SLOTS);
            self.slot_tokens[..self.n_slots].copy_from_slice(&slot_tokens[..self.n_slots]);
        }
    }

    /// VRAM in use / total, when the device reports it.
    pub fn vram_fill(&self) -> Option<f64> {
        let (f, t) = (self.vram_free?, self.vram_total?);
        (t > 0).then(|| 1.0 - f as f64 / t as f64)
    }
}

#[cfg(golem_backend_hip)]
fn device_mem() -> Option<(u64, u64)> {
    golem_llama_sys::pinned::free_device_mem().map(|(f, t)| (f as u64, t as u64))
}

#[cfg(not(golem_backend_hip))]
fn device_mem() -> Option<(u64, u64)> {
    None
}

/// Renders the Prometheus text exposition.
pub(crate) fn render(h: &HealthState) -> String {
    let m = *h.metrics.lock().unwrap_or_else(|p| p.into_inner());
    let mut o = String::with_capacity(4096);
    let mut labeled = String::new();
    let mut g = |name: &str, kind: &str, help: &str, v: f64| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}");
    };
    if let Some((copied, skipped, ev)) = m.device_cache {
        g(
            "golem_device_cache_bytes_copied_total",
            "counter",
            "P4 expert bytes copied into the device cache",
            copied as f64,
        );
        g(
            "golem_device_cache_bytes_skipped_total",
            "counter",
            "P4 expert bytes served from the device cache",
            skipped as f64,
        );
        g(
            "golem_device_cache_evictions_total",
            "counter",
            "P4 device cache evictions",
            ev as f64,
        );
    }
    if let Some((cache, stream)) = m.h2d_copies {
        let _ = writeln!(
            labeled,
            "# HELP golem_expert_h2d_copies_total expert planes copied host→device, by device-cache fill or streamed prefill (stream counts whole-tensor fills, which can exceed routed demand)\n# TYPE golem_expert_h2d_copies_total counter\n\
             golem_expert_h2d_copies_total{{path=\"cache\"}} {cache}\n\
             golem_expert_h2d_copies_total{{path=\"stream\"}} {stream}"
        );
    }
    if let Some([demand, fast, ev, paged, errs]) = m.p3 {
        g(
            "golem_pool_demand_pageins_total",
            "counter",
            "P3 demand page-ins",
            demand as f64,
        );
        g(
            "golem_pool_fast_hits_total",
            "counter",
            "P3 claims served from the RAM pool",
            fast as f64,
        );
        g(
            "golem_pool_evictions_total",
            "counter",
            "P3 RAM pool evictions",
            ev as f64,
        );
        g(
            "golem_pool_bytes_paged_total",
            "counter",
            "P3 bytes paged into the RAM pool",
            paged as f64,
        );
        g(
            "golem_pool_source_errors_total",
            "counter",
            "P3 source read failures",
            errs as f64,
        );
    }
    if let Some((pageins, hits, wasted)) = m.prefetch {
        let rate = if pageins == 0 {
            0.0
        } else {
            hits as f64 / pageins as f64
        };
        g(
            "golem_prefetch_hit_rate",
            "gauge",
            "Prefetched units claimed before eviction / prefetched",
            rate,
        );
        g(
            "golem_prefetch_wasted_bytes",
            "counter",
            "Prefetched bytes evicted unused",
            wasted as f64,
        );
    }
    if let Some([hits, misses, rf, ev, drop, used]) = m.snapshots {
        g(
            "golem_state_snapshot_hits_total",
            "counter",
            "State snapshot restores",
            hits as f64,
        );
        g(
            "golem_state_snapshot_misses_total",
            "counter",
            "Lookups without a restorable prefix",
            misses as f64,
        );
        g(
            "golem_state_snapshot_restore_failures_total",
            "counter",
            "Rejected restores",
            rf as f64,
        );
        g(
            "golem_state_snapshot_evictions_total",
            "counter",
            "Snapshot evictions",
            ev as f64,
        );
        g(
            "golem_state_snapshot_dropped_oversize_total",
            "counter",
            "Snapshots larger than the arena",
            drop as f64,
        );
        g(
            "golem_state_snapshot_bytes",
            "gauge",
            "Bytes held by live snapshots",
            used as f64,
        );
    }
    // I2: CPU builds (and non-offloaded sessions) place expert MUL_MAT_ID
    // on the CPU by construction.
    #[cfg(golem_backend_hip)]
    if let Some((on_gpu, on_cpu)) = m.placement {
        g(
            "golem_expert_mul_mat_id_off_gpu_total",
            "counter",
            "Expert MUL_MAT_ID nodes placed off the GPU (must be 0); deprecated alias of golem_expert_mul_mat_id_total{device=\"cpu\"}",
            on_cpu as f64,
        );
        let _ = writeln!(
            labeled,
            "# HELP golem_expert_mul_mat_id_total Expert MUL_MAT_ID graph nodes by placed device\n# TYPE golem_expert_mul_mat_id_total counter\n\
             golem_expert_mul_mat_id_total{{device=\"gpu\"}} {on_gpu}\n\
             golem_expert_mul_mat_id_total{{device=\"cpu\"}} {on_cpu}"
        );
    }
    g(
        "golem_locked_ram_bytes",
        "gauge",
        "mlocked expert pool + snapshot arena bytes",
        m.locked_ram_bytes as f64,
    );
    if let Some(f) = m.vram_free {
        g(
            "golem_vram_free_bytes",
            "gauge",
            "Device free bytes",
            f as f64,
        );
    }
    g(
        "golem_queue_depth",
        "gauge",
        "Jobs queued for a slot",
        h.queued.load(Ordering::Relaxed).max(0) as f64,
    );
    g(
        "golem_in_flight",
        "gauge",
        "Jobs on the decode worker",
        h.in_flight_count() as f64,
    );
    o.push_str(&labeled);
    let _ = writeln!(
        o,
        "# HELP golem_errors_total Error responses by code\n# TYPE golem_errors_total counter"
    );
    for (i, c) in CODES.iter().enumerate() {
        let code = serde_json::to_value(c).ok();
        let code = code.as_ref().and_then(|v| v.as_str()).unwrap_or("unknown");
        let _ = writeln!(
            o,
            "golem_errors_total{{code=\"{code}\"}} {}",
            h.errors[i].load(Ordering::Relaxed)
        );
    }
    let _ = writeln!(
        o,
        "# HELP golem_slot_tokens_total Tokens generated per slot\n# TYPE golem_slot_tokens_total counter"
    );
    for (i, t) in m.slot_tokens[..m.n_slots].iter().enumerate() {
        let _ = writeln!(o, "golem_slot_tokens_total{{slot=\"{i}\"}} {t}");
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_index_round_trips_every_code() {
        for (i, c) in CODES.iter().enumerate() {
            assert_eq!(code_index(*c), i);
        }
    }
}
