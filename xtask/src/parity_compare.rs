//! `xtask parity-compare` — deterministic T1/T2 parity comparator + report.
//!
//! Consumes the JSONL dump formats produced by the harness legs:
//! - T2: `golem-parity-dump` v1 (header + per-step records) written by
//!   `golem run --parity-dump` — **one dump is one prompt run**.
//! - T1: `golem-parity-op` v1 (header + per-expert-id checksum rows) written
//!   by the `parity-op` leg.
//!
//! Verdict semantics: per position, tokens are
//! compared first. On divergence, the divergence is *tolerated* iff
//! `min(gap_a, gap_b) < epsilon` where `gap = top1 - top2` at that position
//! in each leg (strict `<`: `gap == epsilon` diverges). The verdict for a
//! prompt is decided at the **first divergence**; positions after it are
//! recorded as `post_split_positions` and never scored into the match rate.
//! Per-prompt verdict: `exact` (no divergence anywhere), `tolerated` (first
//! divergence tolerated), `diverged` (first divergence real).
//!
//! Report JSON is deterministic: serde serializes struct fields in
//! declaration order, and the declaration order below is the frozen report
//! contract. Backend mismatch (cpu vs hip) is recorded in the report, never
//! failed; cross-dump guards (format/version, model sha256, n_predict,
//! prompt_tokens, step count, step-position alignment, per-dump position
//! contiguity from 0, per-step logit ordering (top1 ≥ top2), non-empty dumps,
//! header-vs-body record counts, id set, quant/dims) are usage-class errors
//! — exit 2.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Frozen format tag of the T2 end-to-end dump.
pub const T2_FORMAT: &str = "golem-parity-dump";
/// Frozen format tag of the T1 op dump.
pub const T1_FORMAT: &str = "golem-parity-op";
/// The only dump version this comparator understands.
pub const SUPPORTED_VERSION: u64 = 1;

/// Usage-class failure (exit 2): a malformed dump, a violated cross-dump
/// guard, or a bad parameter. Every variant must exit 2, never 1.
#[derive(Debug, Clone, PartialEq)]
pub enum CompareError {
    /// Line-1 header is not the expected dump format or version.
    Format {
        /// The format tag the dump was expected to carry.
        expected: &'static str,
        /// What is actually wrong with the header.
        detail: String,
    },
    /// T2 guard: the two dumps disagree on `prompt_tokens`.
    PromptTokensMismatch {
        /// `prompt_tokens` of leg A.
        a: u64,
        /// `prompt_tokens` of leg B.
        b: u64,
    },
    /// T2 guard: the two dumps were written from different model bytes
    /// (a model mix-up must never compare as a pass).
    ModelShaMismatch {
        /// `model_sha256` of leg A.
        a: String,
        /// `model_sha256` of leg B.
        b: String,
    },
    /// T2 guard: the two dumps disagree on the requested `n_predict`.
    NPredictMismatch {
        /// Header `n_predict` of leg A.
        a: u64,
        /// Header `n_predict` of leg B.
        b: u64,
    },
    /// T2 guard: the two dumps have different step counts.
    StepCountMismatch {
        /// Step count of leg A.
        a: usize,
        /// Step count of leg B.
        b: usize,
    },
    /// T2 guard: at some step index the two dumps record different
    /// `position` values. The comparison walks both step lists by index, so
    /// mis-aligned dumps would compare unrelated generation steps.
    PositionMismatch {
        /// 0-based step index where the positions disagree.
        index: usize,
        /// Position recorded by leg A at that index.
        a: u32,
        /// Position recorded by leg B at that index.
        b: u32,
    },
    /// T2 guard: a dump's step positions are not contiguous from 0 (the dump
    /// contract records every generation step, 0-based, in dump order; gaps
    /// or a non-zero start would silently mis-align the index pairing).
    NonContiguousPositions {
        /// 1-based dump line of the first offending step record.
        line: usize,
        /// The position the next step must carry (its 0-based index).
        expected: u32,
        /// The position the step actually recorded.
        actual: u32,
    },
    /// T1 guard: the two op dumps cover different expert-id sets.
    IdSetMismatch {
        /// Ids present only in leg A (ascending).
        only_a: Vec<u32>,
        /// Ids present only in leg B (ascending).
        only_b: Vec<u32>,
    },
    /// T1 guard: the two op dumps disagree on `quant` or dims.
    ShapeMismatch {
        /// What differs.
        detail: String,
    },
    /// A dump line is malformed JSON or missing required fields.
    Parse {
        /// 1-based dump line the failure is on.
        line: usize,
        /// What is wrong with the line.
        detail: String,
    },
    /// A dump parsed but recorded no records after the header (empty body).
    EmptyBody {
        /// The format tag the dump carried.
        expected: &'static str,
    },
    /// A parameter is invalid (unparseable, negative epsilon, ...).
    Usage(String),
}

impl std::fmt::Display for CompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareError::Format { expected, detail } => {
                write!(
                    f,
                    "dump header mismatch (expected {expected:?} v{SUPPORTED_VERSION}): {detail}"
                )
            }
            CompareError::PromptTokensMismatch { a, b } => {
                write!(f, "guard: prompt_tokens mismatch: {a} vs {b}")
            }
            CompareError::ModelShaMismatch { a, b } => {
                write!(f, "guard: model_sha256 mismatch: {a} vs {b}")
            }
            CompareError::NPredictMismatch { a, b } => {
                write!(f, "guard: n_predict mismatch: {a} vs {b}")
            }
            CompareError::StepCountMismatch { a, b } => {
                write!(f, "guard: step-count mismatch: {a} vs {b}")
            }
            CompareError::PositionMismatch { index, a, b } => {
                write!(
                    f,
                    "guard: step-position mismatch at index {index}: {a} vs {b}"
                )
            }
            CompareError::NonContiguousPositions {
                line,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "dump line {line}: step position {actual} is not contiguous \
                     (expected {expected})"
                )
            }
            CompareError::IdSetMismatch { only_a, only_b } => {
                write!(
                    f,
                    "guard: expert-id set mismatch: only_a={only_a:?} only_b={only_b:?}"
                )
            }
            CompareError::EmptyBody { expected } => {
                write!(
                    f,
                    "dump body is empty (expected {expected:?} records after the header)"
                )
            }
            CompareError::ShapeMismatch { detail } => {
                write!(f, "guard: op-dump shape mismatch: {detail}")
            }
            CompareError::Parse { line, detail } => {
                write!(f, "dump line {line}: {detail}")
            }
            CompareError::Usage(msg) => write!(f, "{msg}"),
        }
    }
}

/// One per-step record of a `golem-parity-dump` body (frozen field order).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DumpStep {
    /// Generation-step index (0-based).
    pub position: u32,
    /// Greedy-sampled token id.
    pub token: i32,
    /// Highest logit at this position.
    pub top1: f32,
    /// Second-highest logit at this position.
    pub top2: f32,
}

/// Parsed `golem-parity-dump` v1: the header fields the comparator guards on
/// plus every recorded step. One dump == one prompt run.
#[derive(Debug, Clone, PartialEq)]
pub struct ParityDump {
    /// Substrate backend the dump was recorded on (`cpu` / `hip`).
    pub backend: String,
    /// Header `model_sha256` — must agree across a compared pair.
    pub model_sha256: String,
    /// Header `n_predict` — must agree across a compared pair.
    pub n_predict: u64,
    /// Header `prompt_tokens` — must agree across a compared pair.
    pub prompt_tokens: u64,
    /// Recorded steps in dump order.
    pub steps: Vec<DumpStep>,
}

/// Parsed `golem-parity-op` v1 dump: header identity + checksum per id.
#[derive(Debug, Clone, PartialEq)]
pub struct OpDump {
    /// Substrate backend the dump was recorded on (`cpu` / `hip`).
    pub backend: String,
    /// Weight row spelling (`fp32` / `q8_0`).
    pub quant: String,
    /// Op dim `m` (rows per expert output).
    pub m: i64,
    /// Op dim `k` (reduction length).
    pub k: i64,
    /// Op dim `n_expert` (routed expert count).
    pub n_expert: i64,
    /// f64 row checksum per expert id (ascending by id).
    pub checksums: BTreeMap<u32, f64>,
}

/// First-divergence record for one prompt pair (frozen field order).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Divergence {
    /// Position of the first token divergence.
    pub position: u32,
    /// Leg A token at that position.
    pub a_token: i32,
    /// Leg B token at that position.
    pub b_token: i32,
    /// Leg A logit gap `top1 - top2` at that position.
    pub gap_a: f64,
    /// Leg B logit gap `top1 - top2` at that position.
    pub gap_b: f64,
    /// Whether the divergence is tolerated (`min(gap_a, gap_b) < epsilon`).
    pub tolerated: bool,
}

/// Deterministic T2 report; declaration order is the frozen report contract.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct T2Report {
    /// Always `"t2-parity"`.
    pub harness: &'static str,
    /// Report schema version (2 — v2 adds max_logit_dev/gpu_dev_bound/gpu_leg_pass).
    pub version: u32,
    /// Epsilon used for the tolerance rule.
    pub epsilon: f64,
    /// Threshold `match_rate_pct` must reach for `pass`.
    pub threshold_pct: f64,
    /// Number of dump pairs compared.
    pub prompts: u32,
    /// Prompts with no divergence anywhere.
    pub exact: u32,
    /// Prompts whose first divergence was tolerated.
    pub tolerated: u32,
    /// Prompts whose first divergence was real.
    pub diverged: u32,
    /// `100 × exact / prompts`.
    pub match_rate_pct: f64,
    /// `diverged == 0 && match_rate_pct >= threshold_pct`.
    pub pass: bool,
    /// First-divergence record per prompt pair with a divergence.
    pub divergences: Vec<Divergence>,
    /// Positions after the first divergence, summed over pairs (never scored).
    pub post_split_positions: u64,
    /// Backends of the compared legs, `[a, b]` order (the CLI compares one
    /// pair, so this is that pair's provenance). Mismatch is recorded, not failed.
    pub backends: Vec<String>,
    /// Max `|top1_a - top1_b|` over compared positions (pre-first-divergence
    /// per pair; cascade positions excluded). The GPU-kernel numerics bound
    /// evidence for the M1 escalation (plan §OPEN Step B).
    pub max_logit_dev: f64,
    /// Optional GPU-kernel deviation bound (CLI `--gpu-dev-bound`).
    pub gpu_dev_bound: Option<f64>,
    /// `Some(max_logit_dev <= gpu_dev_bound)` when the bound is set. The
    /// GPU-leg acceptance criterion; `pass` stays the token rule, untouched.
    pub gpu_leg_pass: Option<bool>,
}

/// Deterministic T1 report; declaration order is the frozen report contract.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct T1Report {
    /// Always `"t1-op-parity"`.
    pub harness: &'static str,
    /// `max over ids of |checksum_a - checksum_b|`.
    pub max_abs: f64,
    /// Bound `max_abs` must stay within for `pass`.
    pub bound: f64,
    /// `max_abs <= bound` (inclusive: a bound is a limit the measurement may touch).
    pub pass: bool,
    /// Backends of the compared legs, `[a, b]` order; mismatch recorded, not failed.
    pub backends: Vec<String>,
}

/// Splits dump text into (1-based line number, line) pairs, skipping blank
/// lines so trailing newlines are harmless.
fn dump_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .map(|(i, line)| (i + 1, line))
        .filter(|(_, line)| !line.trim().is_empty())
}

/// Parses and validates line 1: it must be a JSON object carrying the
/// expected `format` tag and `version: 1` (format/version guard).
fn parse_header(
    text: &str,
    expected_format: &'static str,
) -> Result<serde_json::Value, CompareError> {
    let (lineno, first) = dump_lines(text).next().ok_or_else(|| CompareError::Parse {
        line: 1,
        detail: "dump is empty (no header line)".to_owned(),
    })?;
    let header: serde_json::Value =
        serde_json::from_str(first).map_err(|e| CompareError::Parse {
            line: lineno,
            detail: format!("header is not valid JSON: {e}"),
        })?;
    let format = header
        .get("format")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| CompareError::Format {
            expected: expected_format,
            detail: "header has no string `format` field".to_owned(),
        })?;
    if format != expected_format {
        return Err(CompareError::Format {
            expected: expected_format,
            detail: format!("header format is {format:?}"),
        });
    }
    let version = header
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| CompareError::Format {
            expected: expected_format,
            detail: "header has no integer `version` field".to_owned(),
        })?;
    if version != SUPPORTED_VERSION {
        return Err(CompareError::Format {
            expected: expected_format,
            detail: format!("unsupported dump version {version} (expected {SUPPORTED_VERSION})"),
        });
    }
    Ok(header)
}

/// Reads a required string header field (a missing field is a header guard
/// failure: the dump does not conform to the frozen contract).
fn require_str<'a>(
    header: &'a serde_json::Value,
    key: &str,
    expected_format: &'static str,
) -> Result<&'a str, CompareError> {
    header
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| CompareError::Format {
            expected: expected_format,
            detail: format!("header has no string `{key}` field"),
        })
}

/// Reads a required unsigned integer header field.
fn require_u64(
    header: &serde_json::Value,
    key: &str,
    expected_format: &'static str,
) -> Result<u64, CompareError> {
    header
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| CompareError::Format {
            expected: expected_format,
            detail: format!("header has no integer `{key}` field"),
        })
}

/// Reads a required signed integer header field (op dims are i64).
fn require_i64(
    header: &serde_json::Value,
    key: &str,
    expected_format: &'static str,
) -> Result<i64, CompareError> {
    header
        .get(key)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| CompareError::Format {
            expected: expected_format,
            detail: format!("header has no integer `{key}` field"),
        })
}

/// Parses a `golem-parity-dump` v1 body (header + step records).
pub fn parse_t2_dump(text: &str) -> Result<ParityDump, CompareError> {
    let header = parse_header(text, T2_FORMAT)?;
    let backend = require_str(&header, "backend", T2_FORMAT)?.to_owned();
    let model_sha256 = require_str(&header, "model_sha256", T2_FORMAT)?.to_owned();
    let n_predict = require_u64(&header, "n_predict", T2_FORMAT)?;
    let prompt_tokens = require_u64(&header, "prompt_tokens", T2_FORMAT)?;
    let mut steps = Vec::new();
    for (lineno, line) in dump_lines(text).skip(1) {
        let step: DumpStep = serde_json::from_str(line).map_err(|e| CompareError::Parse {
            line: lineno,
            detail: format!("bad step record: {e}"),
        })?;
        // Logit sanity: the engine guarantees top1 >= top2. Non-finite logits
        // are guarded explicitly: `1e39` parses as f64 and narrows to f32 inf
        // (serde_json only rejects inf/NaN *literals*), which would drive
        // max_logit_dev to inf and serialize as null — breaking the frozen
        // contract silently instead of failing loudly here. A negative gap
        // would auto-tolerate any divergence under the epsilon rule, so a
        // dump violating the ordering must never reach the compare.
        if !step.top1.is_finite() || !step.top2.is_finite() || step.top1 < step.top2 {
            return Err(CompareError::Parse {
                line: lineno,
                detail: format!(
                    "step {} has non-finite logits or top1 < top2",
                    step.position
                ),
            });
        }
        // Contiguity guard: step N of a dump must record position N. A gap
        // or non-zero start means the leg lost or reordered steps — such a
        // dump must never reach the index-based pairing.
        let expected = steps.len() as u32;
        if step.position != expected {
            return Err(CompareError::NonContiguousPositions {
                line: lineno,
                expected,
                actual: step.position,
            });
        }
        steps.push(step);
    }
    // An empty dump compares nothing and would pass at 100% — a leg that
    // recorded zero steps is a failed leg, not a perfect one.
    if steps.is_empty() {
        return Err(CompareError::EmptyBody {
            expected: T2_FORMAT,
        });
    }
    Ok(ParityDump {
        backend,
        model_sha256,
        n_predict,
        prompt_tokens,
        steps,
    })
}

/// One per-expert-id checksum row of a `golem-parity-op` body.
#[derive(Deserialize)]
struct OpDumpRow {
    id: u32,
    checksum: f64,
}

/// Parses a `golem-parity-op` v1 body (header + per-id checksum rows).
pub fn parse_t1_dump(text: &str) -> Result<OpDump, CompareError> {
    let header = parse_header(text, T1_FORMAT)?;
    let backend = require_str(&header, "backend", T1_FORMAT)?.to_owned();
    let quant = require_str(&header, "quant", T1_FORMAT)?.to_owned();
    let m = require_i64(&header, "m", T1_FORMAT)?;
    let k = require_i64(&header, "k", T1_FORMAT)?;
    let n_expert = require_i64(&header, "n_expert", T1_FORMAT)?;
    let n_ids = require_u64(&header, "n_ids", T1_FORMAT)?;
    let mut checksums = BTreeMap::new();
    for (lineno, line) in dump_lines(text).skip(1) {
        let row: OpDumpRow = serde_json::from_str(line).map_err(|e| CompareError::Parse {
            line: lineno,
            detail: format!("bad checksum row: {e}"),
        })?;
        if !row.checksum.is_finite() {
            return Err(CompareError::Parse {
                line: lineno,
                detail: format!("checksum for id {} is not finite", row.id),
            });
        }
        if checksums.insert(row.id, row.checksum).is_some() {
            return Err(CompareError::Parse {
                line: lineno,
                detail: format!("duplicate expert id {}", row.id),
            });
        }
    }
    // Same empty-dump footgun as T2: zero id rows would compare as
    // max_abs = 0.0 and pass.
    if checksums.is_empty() {
        return Err(CompareError::EmptyBody {
            expected: T1_FORMAT,
        });
    }
    // Header/body count guard: a writer bug (or a truncated transfer) that
    // drops rows from *both* legs identically would slip past the id-set
    // guard — the header's `n_ids` is the tamper check.
    if checksums.len() as u64 != n_ids {
        return Err(CompareError::Format {
            expected: T1_FORMAT,
            detail: format!(
                "header n_ids {n_ids} but body records {} checksum rows",
                checksums.len()
            ),
        });
    }
    Ok(OpDump {
        backend,
        quant,
        m,
        k,
        n_expert,
        checksums,
    })
}

/// Compares one matched T2 dump pair under the first-divergence verdict rule
/// and renders the deterministic report.
///
/// Guards (exit-2-class): the pair must agree on `model_sha256`, `n_predict`
/// and `prompt_tokens`, and both step lists must record the same `position`
/// at every index (the pairing is by index). The verdict is decided at the
/// first divergence; subsequent positions only increment
/// `post_split_positions`.
pub fn compare_t2(
    a: &ParityDump,
    b: &ParityDump,
    epsilon: f64,
    threshold_pct: f64,
    gpu_dev_bound: Option<f64>,
) -> Result<T2Report, CompareError> {
    if let Some(b) = gpu_dev_bound {
        if !b.is_finite() || b < 0.0 {
            return Err(CompareError::Usage(format!(
                "gpu-dev-bound must be finite and >= 0, got {b}"
            )));
        }
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(CompareError::Usage(format!(
            "epsilon must be finite and >= 0, got {epsilon}"
        )));
    }
    if !threshold_pct.is_finite() || !(0.0..=100.0).contains(&threshold_pct) {
        return Err(CompareError::Usage(format!(
            "threshold must be finite and within 0..=100, got {threshold_pct}"
        )));
    }
    let mut exact = 0u32;
    let mut tolerated = 0u32;
    let mut diverged = 0u32;
    let mut divergences = Vec::new();
    let mut post_split_positions = 0u64;
    let mut max_logit_dev = 0.0f64;
    // Cross-dump guards: the pair must describe the same model run. A model
    // mix-up or a differing generation-length knob must fail outright —
    // standalone invocations have no driver above them to catch it.
    if a.model_sha256 != b.model_sha256 {
        return Err(CompareError::ModelShaMismatch {
            a: a.model_sha256.clone(),
            b: b.model_sha256.clone(),
        });
    }
    if a.n_predict != b.n_predict {
        return Err(CompareError::NPredictMismatch {
            a: a.n_predict,
            b: b.n_predict,
        });
    }
    if a.prompt_tokens != b.prompt_tokens {
        return Err(CompareError::PromptTokensMismatch {
            a: a.prompt_tokens,
            b: b.prompt_tokens,
        });
    }
    if a.steps.len() != b.steps.len() {
        return Err(CompareError::StepCountMismatch {
            a: a.steps.len(),
            b: b.steps.len(),
        });
    }
    // First-divergence verdict: decide at the first token divergence;
    // every position after it only counts toward post_split, never the
    // match rate (cascade rule). The position-alignment guard runs at
    // every index — including post-split ones: alignment is a usage
    // property of the pair, not part of any verdict.
    let mut first: Option<Divergence> = None;
    for (index, (sa, sb)) in a.steps.iter().zip(b.steps.iter()).enumerate() {
        if sa.position != sb.position {
            return Err(CompareError::PositionMismatch {
                index,
                a: sa.position,
                b: sb.position,
            });
        }
        if first.is_some() {
            post_split_positions += 1;
            continue;
        }
        max_logit_dev = max_logit_dev.max((f64::from(sa.top1) - f64::from(sb.top1)).abs());
        if sa.token == sb.token {
            continue;
        }
        // Gaps are computed in f64 from the exact f32 logits: when
        // top1 ≈ top2 (the regime the epsilon rule decides), f32
        // subtraction would lose the precision the comparison needs.
        let gap_a = f64::from(sa.top1) - f64::from(sa.top2);
        let gap_b = f64::from(sb.top1) - f64::from(sb.top2);
        first = Some(Divergence {
            position: sa.position,
            a_token: sa.token,
            b_token: sb.token,
            gap_a,
            gap_b,
            tolerated: gap_a.min(gap_b) < epsilon,
        });
    }
    match first {
        None => exact += 1,
        Some(d) => {
            if d.tolerated {
                tolerated += 1;
            } else {
                diverged += 1;
            }
            divergences.push(d);
        }
    }
    let match_rate_pct = f64::from(exact) * 100.0;
    Ok(T2Report {
        harness: "t2-parity",
        version: 2,
        epsilon,
        threshold_pct,
        prompts: 1,
        exact,
        tolerated,
        diverged,
        match_rate_pct,
        pass: diverged == 0 && match_rate_pct >= threshold_pct,
        divergences,
        post_split_positions,
        backends: vec![a.backend.clone(), b.backend.clone()],
        max_logit_dev,
        gpu_dev_bound,
        gpu_leg_pass: gpu_dev_bound.map(|b| max_logit_dev <= b),
    })
}

/// Compares two T1 op dumps: `max_abs = max over ids of |checksum_a -
/// checksum_b|` (cross-leg difference, F5 — never either leg's own reference
/// error). Guards (exit-2-class): same id sets; same `quant` and dims.
pub fn compare_t1(a: &OpDump, b: &OpDump, bound: f64) -> Result<T1Report, CompareError> {
    if !bound.is_finite() || bound < 0.0 {
        return Err(CompareError::Usage(format!(
            "bound must be finite and >= 0, got {bound}"
        )));
    }
    // Shape guards: comparing across quant or dims would measure nothing.
    if a.quant != b.quant {
        return Err(CompareError::ShapeMismatch {
            detail: format!("quant differs: {:?} vs {:?}", a.quant, b.quant),
        });
    }
    if a.m != b.m || a.k != b.k || a.n_expert != b.n_expert {
        return Err(CompareError::ShapeMismatch {
            detail: format!(
                "dims differ: m={} k={} n_expert={} vs m={} k={} n_expert={}",
                a.m, a.k, a.n_expert, b.m, b.k, b.n_expert
            ),
        });
    }
    let only_a: Vec<u32> = a
        .checksums
        .keys()
        .filter(|id| !b.checksums.contains_key(id))
        .copied()
        .collect();
    let only_b: Vec<u32> = b
        .checksums
        .keys()
        .filter(|id| !a.checksums.contains_key(id))
        .copied()
        .collect();
    if !only_a.is_empty() || !only_b.is_empty() {
        return Err(CompareError::IdSetMismatch { only_a, only_b });
    }
    let mut max_abs = 0.0f64;
    for (id, ca) in &a.checksums {
        let cb = b
            .checksums
            .get(id)
            .copied()
            .expect("id sets verified equal above");
        max_abs = max_abs.max((ca - cb).abs());
    }
    Ok(T1Report {
        harness: "t1-op-parity",
        max_abs,
        bound,
        pass: max_abs <= bound,
        backends: vec![a.backend.clone(), b.backend.clone()],
    })
}

/// Parsed `parity-compare` CLI invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum Cli {
    /// `parity-compare t2 --a <DUMP> --b <DUMP> --epsilon E --threshold PCT`.
    T2 {
        /// Leg A dump path.
        a: PathBuf,
        /// Leg B dump path.
        b: PathBuf,
        /// Logit-gap epsilon.
        epsilon: f64,
        /// Required match-rate percentage.
        threshold_pct: f64,
        /// Optional GPU-kernel deviation bound.
        gpu_dev_bound: Option<f64>,
    },
    /// `parity-compare t1 --a <OPDUMP> --b <OPDUMP> --bound B`.
    T1 {
        /// Leg A op-dump path.
        a: PathBuf,
        /// Leg B op-dump path.
        b: PathBuf,
        /// Max allowed `max_abs`.
        bound: f64,
    },
}

/// Parses `parity-compare` arguments (pure; no I/O, no exiting).
pub fn parse_args(args: &[String]) -> Result<Cli, String> {
    const USAGE: &str = "usage: xtask parity-compare t2 --a <DUMP> --b <DUMP> --epsilon E \
                         --threshold PCT | t1 --a <OPDUMP> --b <OPDUMP> --bound B";
    let mut it = args.iter();
    let leg = it.next().ok_or(USAGE)?;
    let mut a: Option<PathBuf> = None;
    let mut b: Option<PathBuf> = None;
    let mut epsilon: Option<f64> = None;
    let mut threshold_pct: Option<f64> = None;
    let mut gpu_dev_bound: Option<f64> = None;
    let mut bound: Option<f64> = None;
    match leg.as_str() {
        "t2" => {
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--a" => a = Some(PathBuf::from(value(it.next(), "--a")?)),
                    "--b" => b = Some(PathBuf::from(value(it.next(), "--b")?)),
                    "--epsilon" => {
                        epsilon = Some(parse_f64(value(it.next(), "--epsilon")?, "--epsilon")?)
                    }
                    "--threshold" => {
                        threshold_pct =
                            Some(parse_f64(value(it.next(), "--threshold")?, "--threshold")?);
                    }
                    "--gpu-dev-bound" => {
                        gpu_dev_bound = Some(parse_f64(
                            value(it.next(), "--gpu-dev-bound")?,
                            "--gpu-dev-bound",
                        )?);
                    }
                    other => {
                        return Err(format!("unknown parity-compare t2 flag: {other}; {USAGE}"));
                    }
                }
            }
            Ok(Cli::T2 {
                a: a.ok_or("--a <DUMP> is required")?,
                b: b.ok_or("--b <DUMP> is required")?,
                epsilon: epsilon.ok_or("--epsilon E is required")?,
                threshold_pct: threshold_pct.ok_or("--threshold PCT is required")?,
                gpu_dev_bound,
            })
        }
        "t1" => {
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--a" => a = Some(PathBuf::from(value(it.next(), "--a")?)),
                    "--b" => b = Some(PathBuf::from(value(it.next(), "--b")?)),
                    "--bound" => bound = Some(parse_f64(value(it.next(), "--bound")?, "--bound")?),
                    other => {
                        return Err(format!("unknown parity-compare t1 flag: {other}; {USAGE}"));
                    }
                }
            }
            Ok(Cli::T1 {
                a: a.ok_or("--a <OPDUMP> is required")?,
                b: b.ok_or("--b <OPDUMP> is required")?,
                bound: bound.ok_or("--bound B is required")?,
            })
        }
        other => Err(format!(
            "unknown parity-compare leg {other:?} (expected t1 or t2); {USAGE}"
        )),
    }
}

/// Reads the value following a flag.
fn value<'a>(arg: Option<&'a String>, flag: &str) -> Result<&'a str, String> {
    arg.map(String::as_str)
        .ok_or(format!("{flag} requires a value"))
}

/// Parses a finite f64 parameter (NaN/inf are rejected up front).
fn parse_f64(s: &str, flag: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("{flag} must be a number, got {s:?}"))?;
    if !v.is_finite() {
        return Err(format!("{flag} must be finite, got {s:?}"));
    }
    Ok(v)
}

/// Executes a parsed invocation: reads both dumps, compares, renders the
/// single-line report. Every `Err` is usage-class (the caller exits 2).
pub fn execute(cli: &Cli) -> Result<String, String> {
    match cli {
        Cli::T2 {
            a,
            b,
            epsilon,
            threshold_pct,
            gpu_dev_bound,
        } => {
            let da = read_t2(a)?;
            let db = read_t2(b)?;
            let report = compare_t2(&da, &db, *epsilon, *threshold_pct, *gpu_dev_bound)
                .map_err(|e| e.to_string())?;
            serde_json::to_string(&report).map_err(|e| format!("report serialization failed: {e}"))
        }
        Cli::T1 { a, b, bound } => {
            let da = read_t1(a)?;
            let db = read_t1(b)?;
            let report = compare_t1(&da, &db, *bound).map_err(|e| e.to_string())?;
            serde_json::to_string(&report).map_err(|e| format!("report serialization failed: {e}"))
        }
    }
}

fn read_t2(path: &std::path::Path) -> Result<ParityDump, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_t2_dump(&text).map_err(|e| e.to_string())
}

fn read_t1(path: &std::path::Path) -> Result<OpDump, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_t1_dump(&text).map_err(|e| e.to_string())
}

/// Runs `xtask parity-compare ...`; exits 2 on bad usage or any input error,
/// matching the xtask exit-code contract. On success prints exactly one
/// report JSON line to stdout.
pub fn run(args: Vec<String>) {
    let cli = match parse_args(&args) {
        Ok(cli) => cli,
        Err(msg) => die(&msg),
    };
    match execute(&cli) {
        Ok(line) => {
            // Broken pipe (`| head`) must not panic: the frozen exit contract
            // is 0 or 2, never a 101 from println!.
            let _ = writeln!(std::io::stdout().lock(), "{line}");
        }
        Err(msg) => die(&msg),
    }
}

fn die(msg: &str) -> ! {
    eprintln!("parity-compare: {msg}");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- fixed points -----------------------------------------------------
    // Every gap value in these tests is a difference of exact binary f32s
    // (0.125 / 0.25 / 0.5), so `gap == epsilon` comparisons are exact and
    // the strict-`<` boundary is exercised without float fuzz.

    const EPS: f64 = 0.25;

    /// Builds a T2 dump body: frozen header + one record per step.
    fn t2_dump(backend: &str, prompt_tokens: u64, steps: &[(u32, i32, f32, f32)]) -> String {
        let mut s = format!(
            "{{\"format\":\"golem-parity-dump\",\"version\":1,\"model_sha256\":\"abc\",\"\
             model_bytes\":7,\"backend\":\"{backend}\",\"offload\":false,\
             \"prompt_tokens\":{prompt_tokens},\"n_predict\":{}}}\n",
            steps.len()
        );
        for (p, t, top1, top2) in steps {
            s.push_str(&format!(
                "{{\"position\":{p},\"token\":{t},\"top1\":{top1},\"top2\":{top2}}}\n"
            ));
        }
        s
    }

    /// In-memory step constructor (no text round trip).
    fn steps(spec: &[(u32, i32, f32, f32)]) -> Vec<DumpStep> {
        spec.iter()
            .map(|&(position, token, top1, top2)| DumpStep {
                position,
                token,
                top1,
                top2,
            })
            .collect()
    }

    fn t2(backend: &str, prompt_tokens: u64, spec: &[(u32, i32, f32, f32)]) -> ParityDump {
        ParityDump {
            backend: backend.to_owned(),
            model_sha256: "abc".to_owned(),
            n_predict: spec.len() as u64,
            prompt_tokens,
            steps: steps(spec),
        }
    }

    /// Builds a T1 op dump body: frozen header + one checksum row per id.
    fn t1_dump(backend: &str, quant: &str, checksums: &[(u32, f64)]) -> String {
        let mut s = format!(
            "{{\"format\":\"golem-parity-op\",\"version\":1,\"backend\":\"{backend}\",\
             \"m\":256,\"k\":128,\"n_expert\":8,\"n_ids\":{},\"seed\":1234,\
             \"quant\":\"{quant}\"}}\n",
            checksums.len()
        );
        for (id, checksum) in checksums {
            s.push_str(&format!("{{\"checksum\":{checksum},\"id\":{id}}}\n"));
        }
        s
    }

    fn op(backend: &str, quant: &str, checksums: &[(u32, f64)]) -> OpDump {
        OpDump {
            backend: backend.to_owned(),
            quant: quant.to_owned(),
            m: 256,
            k: 128,
            n_expert: 8,
            checksums: checksums.iter().copied().collect(),
        }
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "golem-xtask-parity-compare-{}-{tag}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_dump(dir: &std::path::Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    // ---- T2 parsing -------------------------------------------------------

    #[test]
    fn parse_t2_round_trips_header_and_steps() {
        let text = t2_dump("hip", 33, &[(0, 10, 1.0, 0.5), (1, -3, 2.0, 0.25)]);
        let dump = parse_t2_dump(&text).expect("valid dump must parse");
        assert_eq!(dump.backend, "hip");
        assert_eq!(dump.model_sha256, "abc");
        assert_eq!(dump.n_predict, 2);
        assert_eq!(dump.prompt_tokens, 33);
        assert_eq!(dump.steps, steps(&[(0, 10, 1.0, 0.5), (1, -3, 2.0, 0.25)]));
    }

    #[test]
    fn parse_t2_rejects_wrong_format() {
        let text = t2_dump("cpu", 1, &[(0, 1, 1.0, 0.5)]).replace(
            "\"format\":\"golem-parity-dump\"",
            "\"format\":\"golem-parity-op\"",
        );
        assert_eq!(
            parse_t2_dump(&text),
            Err(CompareError::Format {
                expected: T2_FORMAT,
                detail: "header format is \"golem-parity-op\"".into(),
            })
        );
    }

    #[test]
    fn parse_t2_rejects_wrong_version() {
        let text = t2_dump("cpu", 1, &[(0, 1, 1.0, 0.5)]).replace("\"version\":1", "\"version\":2");
        assert!(matches!(
            parse_t2_dump(&text),
            Err(CompareError::Format { .. })
        ));
    }

    #[test]
    fn parse_t2_rejects_malformed_step() {
        let text = t2_dump("cpu", 1, &[(0, 1, 1.0, 0.5)]).replace("\"token\":1", "\"token\":\"x\"");
        assert!(matches!(
            parse_t2_dump(&text),
            Err(CompareError::Parse { line: 2, .. })
        ));
    }

    #[test]
    fn parse_t2_rejects_empty_body() {
        assert!(matches!(
            parse_t2_dump(""),
            Err(CompareError::Parse { line: 1, .. })
        ));
    }

    #[test]
    fn parse_t2_rejects_header_only_dump_with_no_steps() {
        let text = t2_dump("cpu", 1, &[]);
        assert_eq!(
            parse_t2_dump(&text),
            Err(CompareError::EmptyBody {
                expected: T2_FORMAT,
            })
        );
    }

    #[test]
    fn parse_t2_rejects_top1_below_top2() {
        // A negative gap would auto-tolerate any divergence under the
        // epsilon rule, so a corrupted leg must never reach the compare.
        let text = t2_dump("cpu", 1, &[(0, 1, 0.5, 1.0)]);
        assert!(matches!(
            parse_t2_dump(&text),
            Err(CompareError::Parse { line: 2, .. })
        ));
    }

    #[test]
    fn parse_t1_rejects_header_only_dump_with_no_checksum_rows() {
        let text = t1_dump("cpu", "fp32", &[]);
        assert_eq!(
            parse_t1_dump(&text),
            Err(CompareError::EmptyBody {
                expected: T1_FORMAT,
            })
        );
    }

    #[test]
    fn compare_t2_rejects_threshold_outside_zero_to_hundred() {
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]);
        let b = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]);
        for bad in [-1.0, 100.5, f64::NAN] {
            assert!(
                matches!(
                    compare_t2(&a, &b, EPS, bad, None),
                    Err(CompareError::Usage(_))
                ),
                "threshold {bad} must be rejected"
            );
        }
    }

    #[test]
    fn parse_t2_rejects_missing_prompt_tokens() {
        let text = t2_dump("cpu", 1, &[(0, 1, 1.0, 0.5)]).replace("\"prompt_tokens\":1,", "");
        assert!(matches!(
            parse_t2_dump(&text),
            Err(CompareError::Format { .. })
        ));
    }

    // ---- T1 parsing -------------------------------------------------------

    #[test]
    fn parse_t1_round_trips_header_and_rows() {
        let text = t1_dump("cpu", "q8_0", &[(0, 1.5), (1, -2.25)]);
        let dump = parse_t1_dump(&text).expect("valid op dump must parse");
        assert_eq!(dump.backend, "cpu");
        assert_eq!(dump.quant, "q8_0");
        assert_eq!(dump.m, 256);
        assert_eq!(dump.k, 128);
        assert_eq!(dump.n_expert, 8);
        assert_eq!(dump.checksums, BTreeMap::from([(0, 1.5), (1, -2.25)]));
    }

    #[test]
    fn parse_t1_rejects_duplicate_id() {
        let text = t1_dump("cpu", "fp32", &[(0, 1.0), (0, 2.0)]);
        assert!(matches!(
            parse_t1_dump(&text),
            Err(CompareError::Parse { line: 3, .. })
        ));
    }

    #[test]
    fn parse_t1_rejects_non_finite_checksum() {
        // NOTE: format! renders f64 1.0 as "1", so anchor on the full row.
        let text = t1_dump("cpu", "fp32", &[(0, 1.0)])
            .replace("\"checksum\":1,\"id\":0", "\"checksum\":null,\"id\":0");
        assert!(matches!(
            parse_t1_dump(&text),
            Err(CompareError::Parse { .. })
        ));
    }

    // ---- T2 comparison: verdicts ------------------------------------------

    #[test]
    fn identical_dumps_are_exact_and_pass() {
        let spec = [(0u32, 5i32, 1.0f32, 0.5f32), (1, 6, 2.0, 0.25)];
        let a = t2("cpu", 9, &spec);
        let b = t2("hip", 9, &spec);
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!(r.harness, "t2-parity");
        assert_eq!(r.version, 2);
        assert_eq!((r.prompts, r.exact, r.tolerated, r.diverged), (1, 1, 0, 0));
        assert_eq!(r.match_rate_pct, 100.0);
        assert!(r.pass);
        assert!(r.divergences.is_empty());
        assert_eq!(r.post_split_positions, 0);
        assert_eq!(r.backends, vec!["cpu", "hip"]);
    }

    #[test]
    fn first_divergence_decides_later_real_divergence_never_flips() {
        // pos1: tolerated first divergence (gap_a 0.125 < eps); pos3: a real
        // divergence (gaps 0.5) — the verdict must stay `tolerated`.
        let a = t2(
            "cpu",
            1,
            &[
                (0, 10, 1.0, 0.5),
                (1, 11, 1.0, 0.875), // gap_a 0.125 < eps
                (2, 12, 1.0, 0.5),
                (3, 99, 1.0, 0.5), // real divergence, gaps 0.5 >= eps
            ],
        );
        let b = t2(
            "hip",
            1,
            &[
                (0, 10, 1.0, 0.5),
                (1, 77, 1.0, 0.5), // gap_b 0.5
                (2, 12, 1.0, 0.5),
                (3, 42, 1.0, 0.0), // gap_b 1.0
            ],
        );
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!((r.exact, r.tolerated, r.diverged), (0, 1, 0));
        assert_eq!(r.divergences.len(), 1);
        let d = &r.divergences[0];
        assert_eq!(d.position, 1);
        assert_eq!((d.a_token, d.b_token), (11, 77));
        assert!((d.gap_a - 0.125).abs() < 1e-12);
        assert!((d.gap_b - 0.5).abs() < 1e-12);
        assert!(d.tolerated);
        // Everything after position 1 (positions 2 and 3) is post-split.
        assert_eq!(r.post_split_positions, 2);
    }

    #[test]
    fn max_logit_dev_tracks_compared_positions_only() {
        // Pre-split top1 diffs: 0.5 at position 0, 0.25 at position 1 (the
        // divergence); the huge 21.0 post-split diff must be excluded.
        let a = t2(
            "cpu",
            1,
            &[(0, 1, 1.0, 0.5), (1, 2, 1.0, 0.875), (2, 9, 30.0, 0.0)],
        );
        let b = t2(
            "hip",
            1,
            &[(0, 1, 0.5, 0.25), (1, 7, 0.75, 0.5), (2, 8, 9.0, 0.0)],
        );
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert!((r.max_logit_dev - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gpu_dev_bound_sets_gpu_leg_pass_without_touching_token_pass() {
        // Diverged token stream (beyond eps), but bounded logit deviation 0.5.
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]);
        let b = t2("hip", 1, &[(0, 2, 0.5, 0.0)]);
        let r = compare_t2(&a, &b, EPS, 100.0, Some(0.5)).expect("compare");
        assert!(!r.pass, "token pass must stay false");
        assert_eq!(r.gpu_dev_bound, Some(0.5));
        assert_eq!(r.gpu_leg_pass, Some(true));

        let r2 = compare_t2(&a, &b, EPS, 100.0, Some(0.25)).expect("compare");
        assert_eq!(r2.gpu_leg_pass, Some(false));
        let r3 = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!(r3.gpu_leg_pass, None);
        assert!((r3.max_logit_dev - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tolerated_asymmetry_gap_a_below_gap_b_above() {
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.875)]); // gap_a 0.125 < eps
        let b = t2("hip", 1, &[(0, 2, 1.0, 0.5)]); // gap_b 0.5 >= eps
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!((r.tolerated, r.diverged), (1, 0));
        assert!(r.divergences[0].tolerated);
    }

    #[test]
    fn tolerated_asymmetry_mirrored() {
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]); // gap_a 0.5 >= eps
        let b = t2("hip", 1, &[(0, 2, 1.0, 0.875)]); // gap_b 0.125 < eps
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!((r.tolerated, r.diverged), (1, 0));
        assert!(r.divergences[0].tolerated);
    }

    #[test]
    fn gap_equal_epsilon_diverges_strict_less_than() {
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.75)]); // gap_a 0.25 == eps
        let b = t2("hip", 1, &[(0, 2, 1.0, 0.75)]); // gap_b 0.25 == eps
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!((r.exact, r.tolerated, r.diverged), (0, 0, 1));
        assert!(!r.divergences[0].tolerated);
        assert!(!r.pass);
    }

    #[test]
    fn real_first_divergence_is_not_rescued_by_later_tolerated_one() {
        let a = t2(
            "cpu",
            1,
            &[
                (0, 1, 1.0, 0.5),   // real divergence, gaps 0.5 >= eps
                (1, 3, 1.0, 0.875), // tolerated-gap divergence later
            ],
        );
        let b = t2("hip", 1, &[(0, 2, 1.0, 0.0), (1, 4, 1.0, 0.5)]);
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!(r.diverged, 1);
        assert_eq!(r.divergences[0].position, 0);
        assert_eq!(r.post_split_positions, 1);
    }

    #[test]
    fn post_split_positions_excluded_from_scoring() {
        // Divergence at position 0; the 7 following positions (even matching
        // ones) are post-split and never scored.
        let a = t2(
            "cpu",
            1,
            &[
                (0, 1, 1.0, 0.875),
                (1, 3, 1.0, 0.5),
                (2, 4, 1.0, 0.5),
                (3, 5, 1.0, 0.5),
                (4, 6, 1.0, 0.5),
                (5, 7, 1.0, 0.5),
                (6, 8, 1.0, 0.5),
                (7, 9, 1.0, 0.5),
            ],
        );
        let b = t2(
            "hip",
            1,
            &[
                (0, 2, 1.0, 0.5),
                (1, 3, 1.0, 0.5),
                (2, 4, 1.0, 0.5),
                (3, 5, 1.0, 0.5),
                (4, 6, 1.0, 0.5),
                (5, 7, 1.0, 0.5),
                (6, 8, 1.0, 0.5),
                (7, 9, 1.0, 0.5),
            ],
        );
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!(r.post_split_positions, 7);
        assert_eq!(r.divergences.len(), 1);
        assert_eq!((r.exact, r.tolerated, r.diverged), (0, 1, 0));
    }

    // ---- T2 comparison: pass rule -----------------------------------------

    #[test]
    fn pass_rule_diverged_fails_even_at_zero_threshold() {
        // Even a 0% threshold cannot rescue a diverged prompt...
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]);
        let b = t2("hip", 1, &[(0, 2, 1.0, 0.5)]);
        let r = compare_t2(&a, &b, EPS, 0.0, None).expect("compare");
        assert_eq!((r.prompts, r.exact, r.diverged), (1, 0, 1));
        assert!(!r.pass, "diverged > 0 must fail regardless of match rate");
        // ...and an all-exact pair passes at the 100% threshold.
        let b = t2("hip", 1, &[(0, 1, 1.0, 0.5)]);
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert!(r.pass);
    }

    #[test]
    fn pass_rule_match_rate_below_threshold_fails() {
        // Tolerated only: diverged == 0, but match_rate 0% < 100%.
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.875)]);
        let b = t2("hip", 1, &[(0, 2, 1.0, 0.5)]);
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("compare");
        assert_eq!(r.diverged, 0);
        assert_eq!(r.match_rate_pct, 0.0);
        assert!(!r.pass, "match_rate below threshold must fail");
    }

    // ---- T2 guards + parameter validation ---------------------------------

    #[test]
    fn guard_prompt_tokens_mismatch_is_usage_error() {
        let a = t2("cpu", 11, &[(0, 1, 1.0, 0.5)]);
        let b = t2("hip", 12, &[(0, 1, 1.0, 0.5)]);
        assert_eq!(
            compare_t2(&a, &b, EPS, 100.0, None),
            Err(CompareError::PromptTokensMismatch { a: 11, b: 12 })
        );
    }

    #[test]
    fn guard_model_sha_mismatch_is_usage_error() {
        // Standalone comparator runs have no driver above them: a model
        // mix-up must fail here, not compare as a pass.
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]);
        let mut b = t2("hip", 1, &[(0, 1, 1.0, 0.5)]);
        b.model_sha256 = "different".to_owned();
        assert_eq!(
            compare_t2(&a, &b, EPS, 100.0, None),
            Err(CompareError::ModelShaMismatch {
                a: "abc".to_owned(),
                b: "different".to_owned(),
            })
        );
    }

    #[test]
    fn guard_n_predict_mismatch_is_usage_error() {
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]);
        let mut b = t2("hip", 1, &[(0, 1, 1.0, 0.5)]);
        b.n_predict = 2;
        assert_eq!(
            compare_t2(&a, &b, EPS, 100.0, None),
            Err(CompareError::NPredictMismatch { a: 1, b: 2 })
        );
    }

    #[test]
    fn parse_t2_rejects_missing_model_sha256() {
        let text = t2_dump("cpu", 1, &[(0, 1, 1.0, 0.5)]).replace("\"model_sha256\":\"abc\",", "");
        assert!(matches!(
            parse_t2_dump(&text),
            Err(CompareError::Format { .. })
        ));
    }

    #[test]
    fn parse_t1_rejects_body_shorter_than_n_ids() {
        // Truncation that hits both legs identically slips past the id-set
        // guard — the header's `n_ids` count is the tamper check.
        let text = t1_dump("cpu", "fp32", &[(0, 1.0)]).replace("\"n_ids\":1", "\"n_ids\":4");
        assert!(matches!(
            parse_t1_dump(&text),
            Err(CompareError::Format {
                detail,
                ..
            })
            if detail.contains("n_ids 4 but body records 1")
        ));
    }

    #[test]
    fn guard_step_count_mismatch_is_usage_error() {
        // Headers agree (same n_predict — e.g. an EOG-truncated leg); the
        // body step counts differ.
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5), (1, 2, 1.0, 0.5)]);
        let mut b = t2("hip", 1, &[(0, 1, 1.0, 0.5)]);
        b.n_predict = 2;
        assert_eq!(
            compare_t2(&a, &b, EPS, 100.0, None),
            Err(CompareError::StepCountMismatch { a: 2, b: 1 })
        );
    }

    #[test]
    fn guard_position_mismatch_is_usage_error() {
        // Both legs record two steps, but at index 1 the recorded positions
        // disagree — the index pairing would compare unrelated steps.
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5), (1, 2, 1.0, 0.5)]);
        let b = t2("hip", 1, &[(0, 1, 1.0, 0.5), (2, 2, 1.0, 0.5)]);
        assert_eq!(
            compare_t2(&a, &b, EPS, 100.0, None),
            Err(CompareError::PositionMismatch {
                index: 1,
                a: 1,
                b: 2
            })
        );
    }

    #[test]
    fn guard_position_mismatch_fires_even_after_first_divergence() {
        // Position 0 diverges (tolerated), position 1 is mis-aligned: the
        // guard must still fire — alignment is a usage error, not a verdict
        // the first divergence could absorb.
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.875), (1, 3, 1.0, 0.5)]);
        let b = t2("hip", 1, &[(0, 2, 1.0, 0.5), (5, 3, 1.0, 0.5)]);
        assert_eq!(
            compare_t2(&a, &b, EPS, 100.0, None),
            Err(CompareError::PositionMismatch {
                index: 1,
                a: 1,
                b: 5
            })
        );
    }

    #[test]
    fn parse_t2_rejects_non_finite_logits() {
        // `1e39` narrows to f32 inf: must die at parse, not serialize a null
        // max_logit_dev into the frozen report contract. (Raw JSON literal —
        // a `1e39` f32 literal in Rust is a compile error.)
        let mut text = t2_dump("cpu", 1, &[(0, 1, 5.5, 0.5)]);
        text = text.replace("\"top1\":5.5", "\"top1\":1e39");
        assert_eq!(
            parse_t2_dump(&text),
            Err(CompareError::Parse {
                line: 2,
                detail: "step 0 has non-finite logits or top1 < top2".into(),
            })
        );
    }

    #[test]
    fn guard_non_contiguous_positions_surfaces_as_exit2_through_files() {
        // Through execute(): a dump whose steps are not contiguous from 0 is
        // rejected at parse time — exit-2-class, before any pairing happens.
        let dir = scratch_dir("t2-pos");
        let cli = Cli::T2 {
            a: write_dump(&dir, "a.jsonl", &t2_dump("cpu", 4, &[(0, 5, 1.0, 0.5)])),
            b: write_dump(&dir, "b.jsonl", &t2_dump("hip", 4, &[(7, 5, 1.0, 0.5)])),
            epsilon: 0.01,
            threshold_pct: 100.0,
            gpu_dev_bound: None,
        };
        let err = execute(&cli).expect_err("contiguity guard must fail");
        assert!(err.contains("not contiguous"), "got: {err}");
    }

    #[test]
    fn parse_t2_rejects_positions_not_starting_at_zero() {
        let text = t2_dump("cpu", 1, &[(1, 1, 1.0, 0.5), (2, 2, 1.0, 0.5)]);
        assert_eq!(
            parse_t2_dump(&text),
            Err(CompareError::NonContiguousPositions {
                line: 2,
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn parse_t2_rejects_gap_in_step_positions() {
        let text = t2_dump("cpu", 1, &[(0, 1, 1.0, 0.5), (2, 2, 1.0, 0.5)]);
        assert_eq!(
            parse_t2_dump(&text),
            Err(CompareError::NonContiguousPositions {
                line: 3,
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn parse_t2_accepts_contiguous_positions_from_zero() {
        let text = t2_dump(
            "cpu",
            1,
            &[(0, 1, 1.0, 0.5), (1, 2, 1.0, 0.5), (2, 3, 1.0, 0.5)],
        );
        assert_eq!(
            parse_t2_dump(&text)
                .expect("contiguous dump must parse")
                .steps
                .len(),
            3
        );
    }

    #[test]
    fn bad_epsilon_and_threshold_are_usage_errors() {
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]);
        let b = t2("hip", 1, &[(0, 1, 1.0, 0.5)]);
        assert!(matches!(
            compare_t2(&a, &b, -1.0, 100.0, None),
            Err(CompareError::Usage(_))
        ));
        assert!(matches!(
            compare_t2(&a, &b, f64::NAN, 100.0, None),
            Err(CompareError::Usage(_))
        ));
        assert!(matches!(
            compare_t2(&a, &b, EPS, f64::INFINITY, None),
            Err(CompareError::Usage(_))
        ));
    }

    // ---- T1 comparison -----------------------------------------------------

    #[test]
    fn t1_identical_dumps_pass_with_zero_max_abs() {
        let a = op("cpu", "fp32", &[(0, 1.5), (1, -2.25), (2, 7.0)]);
        let b = op("hip", "fp32", &[(0, 1.5), (1, -2.25), (2, 7.0)]);
        let r = compare_t1(&a, &b, 0.01).expect("compare");
        assert_eq!(r.harness, "t1-op-parity");
        assert_eq!(r.max_abs, 0.0);
        assert_eq!(r.bound, 0.01);
        assert!(r.pass);
        assert_eq!(r.backends, vec!["cpu", "hip"]);
    }

    #[test]
    fn t1_max_abs_is_max_over_ids_and_bound_is_inclusive() {
        let a = op("cpu", "fp32", &[(0, 1.0), (1, 2.0), (2, 3.0)]);
        let b = op("hip", "fp32", &[(0, 1.5), (1, 2.1), (2, 3.25)]);
        // |diffs| = 0.5, 0.1, 0.25 → max 0.5.
        let r = compare_t1(&a, &b, 0.5).expect("compare");
        assert_eq!(r.max_abs, 0.5);
        assert!(r.pass, "max_abs == bound must pass (inclusive bound)");
        let r = compare_t1(&a, &b, 0.25).expect("compare");
        assert!(!r.pass);
    }

    #[test]
    fn t1_guard_id_set_mismatch_is_usage_error() {
        let a = op("cpu", "fp32", &[(0, 1.0), (2, 2.0)]);
        let b = op("hip", "fp32", &[(0, 1.0), (3, 2.0)]);
        assert_eq!(
            compare_t1(&a, &b, 0.01),
            Err(CompareError::IdSetMismatch {
                only_a: vec![2],
                only_b: vec![3],
            })
        );
    }

    #[test]
    fn t1_guard_quant_mismatch_is_usage_error() {
        let a = op("cpu", "fp32", &[(0, 1.0)]);
        let b = op("hip", "q8_0", &[(0, 1.0)]);
        assert!(matches!(
            compare_t1(&a, &b, 0.01),
            Err(CompareError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn t1_guard_dims_mismatch_is_usage_error() {
        let mut a = op("cpu", "fp32", &[(0, 1.0)]);
        let b = op("hip", "fp32", &[(0, 1.0)]);
        a.k = 64;
        assert!(matches!(
            compare_t1(&a, &b, 0.01),
            Err(CompareError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            compare_t1(&a, &a.clone(), -1.0),
            Err(CompareError::Usage(_))
        ));
    }

    // ---- backend mismatch is recorded, not failed --------------------------

    #[test]
    fn backend_mismatch_is_recorded_never_failed() {
        let a = t2("cpu", 1, &[(0, 1, 1.0, 0.5)]);
        let b = t2("hip", 1, &[(0, 1, 1.0, 0.5)]);
        let r = compare_t2(&a, &b, EPS, 100.0, None).expect("backend mismatch must not fail");
        assert_eq!(r.backends, vec!["cpu", "hip"]);
        assert!(r.pass);
        let a = op("cpu", "fp32", &[(0, 1.0)]);
        let b = op("hip", "fp32", &[(0, 1.0)]);
        let r = compare_t1(&a, &b, 0.01).expect("backend mismatch must not fail");
        assert_eq!(r.backends, vec!["cpu", "hip"]);
    }

    // ---- CLI parsing --------------------------------------------------------

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn parse_args_t2_full() {
        let cli = parse_args(&args(&[
            "t2",
            "--a",
            "a.jsonl",
            "--b",
            "b.jsonl",
            "--epsilon",
            "1e-2",
            "--threshold",
            "100.0",
        ]))
        .expect("valid t2 args");
        assert_eq!(
            cli,
            Cli::T2 {
                a: PathBuf::from("a.jsonl"),
                b: PathBuf::from("b.jsonl"),
                epsilon: 1e-2,
                threshold_pct: 100.0,
                gpu_dev_bound: None,
            }
        );
    }

    #[test]
    fn parse_args_t1_full() {
        let cli = parse_args(&args(&[
            "t1", "--a", "a.jsonl", "--b", "b.jsonl", "--bound", "1e-2",
        ]))
        .expect("valid t1 args");
        assert_eq!(
            cli,
            Cli::T1 {
                a: PathBuf::from("a.jsonl"),
                b: PathBuf::from("b.jsonl"),
                bound: 1e-2,
            }
        );
    }

    #[test]
    fn parse_args_rejects_missing_unknown_and_bad_values() {
        // no leg
        assert!(parse_args(&[]).is_err());
        // unknown leg
        assert!(parse_args(&args(&["t3", "--a", "x"])).is_err());
        // missing --b
        assert!(
            parse_args(&args(&[
                "t2",
                "--a",
                "x",
                "--epsilon",
                "0.01",
                "--threshold",
                "1"
            ]))
            .is_err()
        );
        // missing --threshold
        assert!(parse_args(&args(&["t2", "--a", "x", "--b", "y", "--epsilon", "0.01"])).is_err());
        // unknown flag
        assert!(
            parse_args(&args(&[
                "t2",
                "--a",
                "x",
                "--b",
                "y",
                "--epsilon",
                "0.01",
                "--threshold",
                "1",
                "--wat"
            ]))
            .is_err()
        );
        // unparseable epsilon
        assert!(
            parse_args(&args(&[
                "t2",
                "--a",
                "x",
                "--b",
                "y",
                "--epsilon",
                "abc",
                "--threshold",
                "1"
            ]))
            .is_err()
        );
        // flag without value
        assert!(parse_args(&args(&["t2", "--a"])).is_err());
        // t1 missing --bound
        assert!(parse_args(&args(&["t1", "--a", "x", "--b", "y"])).is_err());
    }

    // ---- end-to-end execution through files ---------------------------------

    #[test]
    fn execute_t2_exact_emits_deterministic_report_line() {
        let dir = scratch_dir("t2-exact");
        let text = t2_dump("cpu", 4, &[(0, 5, 1.0, 0.5), (1, 6, 2.0, 0.25)]);
        let cli = Cli::T2 {
            a: write_dump(&dir, "a.jsonl", &text),
            b: write_dump(&dir, "b.jsonl", &text),
            epsilon: 0.01,
            threshold_pct: 100.0,
            gpu_dev_bound: None,
        };
        let line = execute(&cli).expect("execute must succeed");
        // The full frozen line, byte-exact — declaration order + ryu floats.
        assert_eq!(
            line,
            "{\"harness\":\"t2-parity\",\"version\":2,\"epsilon\":0.01,\
             \"threshold_pct\":100.0,\"prompts\":1,\"exact\":1,\"tolerated\":0,\
             \"diverged\":0,\"match_rate_pct\":100.0,\"pass\":true,\
             \"divergences\":[],\"post_split_positions\":0,\
             \"backends\":[\"cpu\",\"cpu\"],\"max_logit_dev\":0.0,\
             \"gpu_dev_bound\":null,\"gpu_leg_pass\":null}"
        );
    }

    #[test]
    fn execute_t2_divergence_emits_single_first_divergence_record() {
        let dir = scratch_dir("t2-div");
        // pos0 diverges tolerantly (gap_a 0.125 < eps 0.25); pos1 matches and
        // lands in post_split.
        let a = t2_dump("cpu", 4, &[(0, 5, 1.0, 0.875), (1, 6, 2.0, 0.5)]);
        let b = t2_dump("hip", 4, &[(0, 7, 1.0, 0.5), (1, 6, 2.0, 0.5)]);
        let cli = Cli::T2 {
            a: write_dump(&dir, "a.jsonl", &a),
            b: write_dump(&dir, "b.jsonl", &b),
            epsilon: 0.25,
            threshold_pct: 100.0,
            gpu_dev_bound: None,
        };
        let line = execute(&cli).expect("execute must succeed");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["tolerated"], 1);
        assert_eq!(v["divergences"][0]["position"], 0);
        assert_eq!(v["divergences"][0]["a_token"], 5);
        assert_eq!(v["divergences"][0]["b_token"], 7);
        assert_eq!(v["divergences"][0]["tolerated"], true);
        assert_eq!(v["post_split_positions"], 1);
        assert_eq!(v["backends"][0], "cpu");
        assert_eq!(v["backends"][1], "hip");
    }

    #[test]
    fn execute_t1_emits_report_and_guards_are_usage_errors() {
        let dir = scratch_dir("t1");
        let cli = Cli::T1 {
            a: write_dump(
                &dir,
                "a.jsonl",
                &t1_dump("cpu", "fp32", &[(0, 1.0), (1, 2.0)]),
            ),
            b: write_dump(
                &dir,
                "b.jsonl",
                &t1_dump("hip", "fp32", &[(0, 1.0), (1, 2.25)]),
            ),
            bound: 0.5,
        };
        let line = execute(&cli).expect("execute must succeed");
        assert_eq!(
            line,
            "{\"harness\":\"t1-op-parity\",\"max_abs\":0.25,\"bound\":0.5,\
             \"pass\":true,\"backends\":[\"cpu\",\"hip\"]}"
        );
        // Guard through files: id-set mismatch is exit-2-class.
        let cli = Cli::T1 {
            a: dir.join("a.jsonl"),
            b: write_dump(
                &dir,
                "c.jsonl",
                &t1_dump("hip", "fp32", &[(0, 1.0), (7, 2.25)]),
            ),
            bound: 0.5,
        };
        let err = execute(&cli).expect_err("guard must fail");
        assert!(err.contains("id set mismatch"), "got: {err}");
        // Missing file is exit-2-class too.
        let cli = Cli::T1 {
            a: dir.join("missing.jsonl"),
            b: dir.join("b.jsonl"),
            bound: 0.5,
        };
        assert!(execute(&cli).is_err());
    }

    #[test]
    fn execute_t2_guard_through_files_is_usage_error() {
        let dir = scratch_dir("t2-guard");
        let cli = Cli::T2 {
            a: write_dump(&dir, "a.jsonl", &t2_dump("cpu", 4, &[(0, 5, 1.0, 0.5)])),
            b: write_dump(&dir, "b.jsonl", &t2_dump("hip", 5, &[(0, 5, 1.0, 0.5)])),
            epsilon: 0.01,
            threshold_pct: 100.0,
            gpu_dev_bound: None,
        };
        let err = execute(&cli).expect_err("prompt_tokens guard must fail");
        assert!(err.contains("prompt_tokens mismatch"), "got: {err}");
    }
}
