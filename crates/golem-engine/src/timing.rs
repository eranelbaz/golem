//! §11 per-step timing CSV: one row per generation decode step, flushed at
//! turn end. Unmeasured components are held as `f64::NAN` and rendered as
//! the literal `unmeasured` — never `0`, which would silently pass as a
//! real measurement. The compute column is the remainder
//! `wall − p3_wait − h2d − policy`; the store disk time is NOT subtracted
//! because it runs inside the `p3_wait` window on a parallel worker (it can
//! even exceed the step wall time), so subtracting it would double-count.
//! Unmeasured parts contribute nothing to the remainder. The CPU expert
//! math worker's time is not subtracted either: it overlaps the GPU.
//!
//! Hot-path cost (§11-allowed): a few `Instant::now()` per layer plus
//! per-step atomic take-snapshots. The recorder is fixed-capacity and
//! preallocated — `row` allocates nothing.

use std::io::{self, Write};

/// One generation step's timing. `f64::NAN` marks an unmeasured component
/// (`h2d_*`: measured only on the async_h2d copy stream; `policy_ms`: P2
/// order-building is not instrumented).
pub struct StepTiming {
    /// 0-based generation step within the turn.
    pub step: u64,
    /// Tokens decoded in this step (always 1 in the greedy loop today).
    pub tokens: u32,
    /// Whole step wall time.
    pub wall_ms: f64,
    /// Time spent inside P3 `ensure_host_resident` page-ins.
    pub p3_wait_ms: f64,
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
    /// P2 policy order-building time (unmeasured today).
    pub policy_ms: f64,
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
            "step,tokens,wall_ms,p3_wait_ms,disk_worker_ms,disk_bytes,h2d_ms,h2d_bytes,policy_ms,compute_ms,\
             cpu_expert_math_experts,cpu_expert_math_bytes,cpu_expert_math_ms,cpu_expert_join_wait_ms"
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
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                t.step,
                t.tokens,
                ms(t.wall_ms),
                ms(t.p3_wait_ms),
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
            disk_worker_ms: 0.5,
            disk_bytes: 64,
            h2d_ms: f64::NAN,
            h2d_bytes: 0,
            policy_ms: f64::NAN,
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
