//! §11 per-step timing CSV: one row per generation decode step, flushed at
//! turn end. Unmeasured components are held as `f64::NAN` and rendered as
//! the literal `unmeasured` — never `0`, which would silently pass as a
//! real measurement. The compute column is the remainder
//! `wall − Σ(measured parts)`; unmeasured parts contribute nothing to it.
//!
//! Hot-path cost (§11-allowed): a few `Instant::now()` per layer plus
//! per-step atomic take-snapshots. The recorder is fixed-capacity and
//! preallocated — `row` allocates nothing.

use std::io::{self, Write};

/// One generation step's timing. `f64::NAN` marks an unmeasured component
/// (`h2d_*`: the fork exposes no event pair on the copy stream until the
/// M3.8 ABI bump; `policy_ms`: P2 order-building is not instrumented).
pub struct StepTiming {
    /// 0-based generation step within the turn.
    pub step: u64,
    /// Tokens decoded in this step (always 1 in the greedy loop today).
    pub tokens: u32,
    /// Whole step wall time.
    pub wall_ms: f64,
    /// Time spent inside P3 `ensure_host_resident` page-ins.
    pub p3_wait_ms: f64,
    /// Time spent in store shard reads (drain worker view).
    pub disk_ms: f64,
    /// Expert bytes read from disk this step.
    pub disk_bytes: u64,
    /// Host→device copy time (unmeasured until the fork event pair exists).
    pub h2d_ms: f64,
    /// Host→device bytes (unmeasured alongside `h2d_ms`).
    pub h2d_bytes: u64,
    /// P2 policy order-building time (unmeasured today).
    pub policy_ms: f64,
}

/// Fixed-capacity row buffer for one chat turn; no per-row allocation.
pub struct TimingRecorder {
    rows: Vec<StepTiming>,
    cap: usize,
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
            "step,tokens,wall_ms,p3_wait_ms,disk_ms,disk_bytes,h2d_ms,h2d_bytes,policy_ms,compute_ms"
        )
    }

    /// Records one step; rows beyond the fixed capacity are dropped (the
    /// capacity bounds a turn, so overflow means a caller mis-sized it).
    pub fn row(&mut self, t: StepTiming) {
        if self.rows.len() < self.cap {
            self.rows.push(t);
        }
    }

    /// Writes every recorded row. `compute_ms` = wall − Σ(measured parts);
    /// unmeasured columns contribute nothing.
    pub fn flush(&mut self, w: &mut impl Write) -> io::Result<()> {
        for t in &self.rows {
            let measured = [t.p3_wait_ms, t.disk_ms, t.h2d_ms, t.policy_ms]
                .iter()
                .filter(|v| !v.is_nan())
                .sum::<f64>();
            let compute = t.wall_ms - measured;
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{}",
                t.step,
                t.tokens,
                ms(t.wall_ms),
                ms(t.p3_wait_ms),
                ms(t.disk_ms),
                if t.disk_ms.is_nan() {
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
            )?;
        }
        self.rows.clear();
        Ok(())
    }
}
