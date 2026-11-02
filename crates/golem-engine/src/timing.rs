//! §11 per-step timing CSV: one row per generation decode step, flushed at
//! turn end. Unmeasured components are held as `f64::NAN` and rendered as
//! the literal `unmeasured` — never `0`, which would silently pass as a
//! real measurement. The compute column is the remainder
//! `wall − p3_wait − h2d − policy`; the store disk time is NOT subtracted
//! because it runs inside the `p3_wait` window on a parallel worker (it can
//! even exceed the step wall time), so subtracting it would double-count.
//! Unmeasured parts contribute nothing to the remainder. The CPU expert
//! math worker's time is not subtracted either: it overlaps the GPU. Nor are
//! the MoE ids-readback columns (`ids_sync_ms`, `ids_gap_ms`): the gap from
//! a readback's sync to the next launch contains the P3 ensure and policy
//! time already subtracted, so they locate the host-sync bubbles (§3.4) and
//! are not a disjoint part of the wall.
//!
//! Hot-path cost (§11-allowed): a few `Instant::now()` per layer plus
//! per-step atomic take-snapshots. The recorder is fixed-capacity and
//! preallocated — `row` allocates nothing.

use std::io::{self, Write};

/// One generation step's timing. `f64::NAN` marks an unmeasured component
/// (`h2d_*`: measured only on the async_h2d copy stream; `policy_ms` and the
/// `ids_*` columns: measured only while the P3 hooks are registered).
pub struct StepTiming {
    /// 0-based generation step within the turn.
    pub step: u64,
    /// Tokens decoded in this step (always 1 in the greedy loop today).
    pub tokens: u32,
    /// Whole step wall time.
    pub wall_ms: f64,
    /// Time spent inside P3 `ensure_host_resident` page-ins.
    pub p3_wait_ms: f64,
    /// Ensure time blocked on disk misses after the RAM hits were handed off
    /// (a component of `p3_wait_ms`), summed over layers; NaN with no hooks.
    /// An all-miss ensure has no hand-off, so its whole ensure counts.
    pub disk_wait_ms: f64,
    /// Time spent in store shard reads on the parallel drain worker
    /// (a component of `p3_wait_ms`, not part of the compute remainder).
    pub disk_worker_ms: f64,
    /// Expert bytes read from disk this step.
    pub disk_bytes: u64,
    /// Host→device copy time (copy-stream event pairs; async_h2d only): first
    /// copy start to last copy end on the copy stream, so it includes any
    /// copy-stream wait on compute. Under the WAR rule (§11) GPU-side
    /// overlap is ~0; only the host queues ahead.
    pub h2d_ms: f64,
    /// Host→device bytes, MMQ padding tails included (unmeasured alongside
    /// `h2d_ms`).
    pub h2d_bytes: u64,
    /// Decode-thread P2 policy time: the routing readback's booking and the
    /// `copy_done` order flush (the publisher thread's builds overlap).
    pub policy_ms: f64,
    /// MoE ids-readback host-sync timing; `None` without the P3 hooks.
    pub sync_gap: Option<SyncGapStats>,
    /// Decode-only CPU expert math (design §1 O1); `None` unless
    /// `cpu_expert_math` measured it.
    pub cpu_expert: Option<CpuExpertStats>,
}

/// One step's (or a session's cumulative) CPU expert math counters.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CpuExpertStats {
    /// Routed experts computed on the CPU instead of copied to the device.
    pub experts: u64,
    /// Expert plane bytes the CPU read in place of an H2D copy.
    pub bytes: u64,
    /// Worker busy time.
    pub ms: f64,
    /// Decode-thread time blocked joining the worker.
    pub join_wait_ms: f64,
    /// Expert planes the GPU still multiplied as zero-weight padding.
    pub padded: u64,
}

impl CpuExpertStats {
    /// Adds `o` into `self`.
    pub fn add(&mut self, o: &Self) {
        self.experts += o.experts;
        self.bytes += o.bytes;
        self.ms += o.ms;
        self.join_wait_ms += o.join_wait_ms;
        self.padded += o.padded;
    }
}

/// One step's MoE ids-readback host-sync timing (design §3.4, §11): each
/// hooked MoE layer reads its routed ids back, synchronizing only the ids
/// backend; from the first readback's return until the next split launch no
/// compute is queued there (the async_h2d copy stream may still run).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SyncGapStats {
    /// Host time blocked inside the readback synchronizes, mostly waiting on
    /// the prior compute to finish (not a pure sync overhead).
    pub sync_ms: f64,
    /// Summed readback-return → next-launch spans (compute idle, host
    /// planning; copy-stream H2D may overlap).
    pub gap_ms: f64,
    /// Largest single span.
    pub gap_max_ms: f64,
    /// Readbacks counted (one per MoE layer per decode step).
    pub readbacks: u64,
}

impl SyncGapStats {
    /// Adds `o` into `self` (max for `gap_max_ms`).
    pub fn add(&mut self, o: &Self) {
        self.sync_ms += o.sync_ms;
        self.gap_ms += o.gap_ms;
        self.gap_max_ms = self.gap_max_ms.max(o.gap_max_ms);
        self.readbacks += o.readbacks;
    }
}

/// Fixed-capacity row buffer for one chat turn; no per-row allocation.
pub struct TimingRecorder {
    rows: Vec<StepTiming>,
    cap: usize,
    /// Rows rejected because the fixed capacity was full; reported by flush.
    dropped: usize,
}

/// Renders one ms column: NaN → `unmeasured`, else one decimal place.
fn ms(x: f64) -> String {
    if x.is_nan() {
        "unmeasured".to_owned()
    } else {
        format!("{x:.1}")
    }
}

impl TimingRecorder {
    /// Preallocates room for `cap` steps (turn bound: `n_predict`).
    pub fn new(cap: usize) -> Self {
        Self {
            rows: Vec::with_capacity(cap),
            cap,
            dropped: 0,
        }
    }

    /// Writes the two header lines: a `#`-comment carrying the engine
    /// version, the vendored fork commit, and the resolved config, then the
    /// CSV column names (`compute_ms` is derived, not stored).
    pub fn header(
        w: &mut impl Write,
        engine_version: &str,
        fork_commit: &str,
        resolved_config: &str,
    ) -> io::Result<()> {
        writeln!(
            w,
            "# engine_version={engine_version} fork_commit={fork_commit} {resolved_config}"
        )?;
        writeln!(
            w,
            "step,tokens,wall_ms,p3_wait_ms,disk_wait_ms,disk_worker_ms,disk_bytes,h2d_ms,h2d_bytes,policy_ms,compute_ms,\
             cpu_expert_math_experts,cpu_expert_math_bytes,cpu_expert_math_ms,cpu_expert_join_wait_ms,\
             ids_readbacks,ids_sync_ms,ids_gap_ms,ids_gap_max_ms"
        )
    }

    /// Records one step; rows beyond the fixed capacity are dropped and
    /// counted (flush reports the count — overflow means a caller
    /// mis-sized the turn bound).
    pub fn row(&mut self, t: StepTiming) {
        if self.rows.len() < self.cap {
            self.rows.push(t);
        } else {
            self.dropped += 1;
        }
    }

    /// Writes every recorded row. `compute_ms` = wall − p3_wait − h2d −
    /// policy (the disk worker runs inside the p3_wait window and is
    /// deliberately excluded); clamped at 0 defensively — the invariant is
    /// that each measured part is inside the wall window, so the remainder
    /// is non-negative, but a mis-nested timer must not emit a negative
    /// column silently.
    pub fn flush(&mut self, w: &mut impl Write) -> io::Result<()> {
        for t in &self.rows {
            let measured = [t.p3_wait_ms, t.h2d_ms, t.policy_ms]
                .iter()
                .filter(|v| !v.is_nan())
                .sum::<f64>();
            // The CPU expert worker overlaps the GPU: not subtracted.
            let compute = (t.wall_ms - measured).max(0.0);
            let cpu = |f: fn(&CpuExpertStats) -> String| {
                t.cpu_expert
                    .as_ref()
                    .map_or_else(|| "unmeasured".to_owned(), f)
            };
            let sg = |f: fn(&SyncGapStats) -> String| {
                t.sync_gap
                    .as_ref()
                    .map_or_else(|| "unmeasured".to_owned(), f)
            };
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                t.step,
                t.tokens,
                ms(t.wall_ms),
                ms(t.p3_wait_ms),
                ms(t.disk_wait_ms),
                ms(t.disk_worker_ms),
                if t.disk_worker_ms.is_nan() {
                    "unmeasured".to_owned()
                } else {
                    t.disk_bytes.to_string()
                },
                ms(t.h2d_ms),
                if t.h2d_ms.is_nan() {
                    "unmeasured".to_owned()
                } else {
                    t.h2d_bytes.to_string()
                },
                ms(t.policy_ms),
                ms(compute),
                cpu(|c| c.experts.to_string()),
                cpu(|c| c.bytes.to_string()),
                cpu(|c| ms(c.ms)),
                cpu(|c| ms(c.join_wait_ms)),
                sg(|g| g.readbacks.to_string()),
                sg(|g| ms(g.sync_ms)),
                sg(|g| ms(g.gap_ms)),
                sg(|g| ms(g.gap_max_ms)),
            )?;
        }
        if self.dropped > 0 {
            writeln!(w, "# dropped_rows={}", self.dropped)?;
        }
        self.rows.clear();
        self.dropped = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(step: u64, wall_ms: f64, p3_wait_ms: f64) -> StepTiming {
        StepTiming {
            step,
            tokens: 1,
            wall_ms,
            p3_wait_ms,
            disk_wait_ms: f64::NAN,
            disk_worker_ms: 0.5,
            disk_bytes: 64,
            h2d_ms: f64::NAN,
            h2d_bytes: 0,
            policy_ms: f64::NAN,
            sync_gap: None,
            cpu_expert: None,
        }
    }

    /// The recorder survives flush: turn 2's rows must land after turn 1's
    /// flush (regression: an Option::take-style drain destroyed the
    /// recorder, so REPL turns 2+ recorded nothing).
    #[test]
    fn flush_reuses_recorder_across_turns() {
        let mut rec = TimingRecorder::new(4);
        rec.row(row(0, 10.0, 2.0));
        let mut turn1 = Vec::new();
        rec.flush(&mut turn1).unwrap();
        rec.row(row(0, 11.0, 3.0));
        rec.row(row(1, 12.0, 3.0));
        let mut turn2 = Vec::new();
        rec.flush(&mut turn2).unwrap();
        let t2 = String::from_utf8(turn2).unwrap();
        assert_eq!(t2.lines().count(), 2, "both turn-2 rows recorded: {t2}");
        assert!(t2.starts_with("0,1,11.0"), "{t2}");
        assert!(t2.contains("1,1,12.0"), "{t2}");
    }

    /// Compute excludes the parallel disk worker; unmeasured parts
    /// contribute nothing; the remainder clamps at 0.
    #[test]
    fn compute_excludes_disk_worker_and_clamps_at_zero() {
        let mut rec = TimingRecorder::new(2);
        rec.row(row(0, 10.0, 2.0));
        // Disk worker exceeding wall must not produce a negative compute.
        rec.row(row(1, 1.0, 2.0));
        let mut out = Vec::new();
        rec.flush(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        let l0 = s.lines().next().unwrap();
        assert!(l0.contains(",8.0"), "compute = wall − p3_wait: {l0}");
        let l1 = s.lines().nth(1).unwrap();
        assert!(l1.contains(",0.0,unmeasured"), "clamped: {l1}");
    }

    /// Measured policy time is subtracted from compute; the ids-readback
    /// columns render, and are not subtracted (they overlap ensure/policy).
    #[test]
    fn policy_subtracted_and_sync_gap_columns_render_not_subtracted() {
        let mut rec = TimingRecorder::new(1);
        let mut r = row(0, 20.0, 5.0);
        r.policy_ms = 1.5;
        r.sync_gap = Some(SyncGapStats {
            sync_ms: 2.0,
            gap_ms: 9.0,
            gap_max_ms: 0.5,
            readbacks: 42,
        });
        rec.row(r);
        let mut out = Vec::new();
        rec.flush(&mut out).unwrap();
        let l = String::from_utf8(out).unwrap();
        let cols: Vec<&str> = l.trim_end().split(',').collect();
        assert_eq!(cols[9], "1.5", "policy_ms: {l}");
        assert_eq!(cols[10], "13.5", "compute = 20 − 5 − 1.5: {l}");
        assert_eq!(&cols[15..], ["42", "2.0", "9.0", "0.5"], "{l}");
    }

    /// Sync-gap merge sums and keeps the max span.
    #[test]
    fn sync_gap_add_sums_and_maxes() {
        let mut a = SyncGapStats {
            sync_ms: 1.0,
            gap_ms: 2.0,
            gap_max_ms: 0.3,
            readbacks: 1,
        };
        a.add(&SyncGapStats {
            sync_ms: 1.0,
            gap_ms: 1.0,
            gap_max_ms: 0.7,
            readbacks: 2,
        });
        assert_eq!(
            (a.sync_ms, a.gap_ms, a.gap_max_ms, a.readbacks),
            (2.0, 3.0, 0.7, 3)
        );
    }

    /// Overflow past the fixed capacity is counted and reported by flush.
    #[test]
    fn dropped_rows_are_counted_and_reported() {
        let mut rec = TimingRecorder::new(1);
        rec.row(row(0, 1.0, 0.0));
        rec.row(row(1, 1.0, 0.0));
        rec.row(row(2, 1.0, 0.0));
        let mut out = Vec::new();
        rec.flush(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("# dropped_rows=2"), "{s}");
        // The counter resets with the rows for the next turn.
        let mut out = Vec::new();
        rec.flush(&mut out).unwrap();
        assert!(String::from_utf8(out).unwrap().is_empty());
    }
}
