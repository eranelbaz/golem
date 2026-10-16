//! Warm-start score file (§6.3 warm-up, §11): the engine's raw §6.2 policy
//! words (per-key last-access word and packed score word) plus the decode
//! step, persisted verbatim so a restart resumes exactly where the previous
//! run left off — no gap decay (§6.2 decays per decode step, not wall time)
//! and no reconstruction. Pure format/parse; the engine owns the atomics.
//!
//! Format `golem-scores v1`, text, sparse (touched keys only):
//! `golem-scores v1 layers=L experts=E half_life=H step=S keys=N model=<fp>`
//! then `N` lines `<key> <last_access> <score_word>`. GLM: ≤ 42 × 288 =
//! 12,096 lines × ~30 B ≈ 360 KB.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const MAGIC: &str = "golem-scores v1";

/// Highest restorable step: the packed score word keeps `step + 1` in its
/// low 32 bits, so a step within 2^24 of `u32::MAX` would wrap mid-session.
pub const MAX_STEP: u64 = u32::MAX as u64 - (1 << 24);

/// The policy shape a score file must match to be restored.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmShape {
    /// Expert layers.
    pub layers: u32,
    /// Experts per layer.
    pub experts: u32,
    /// §6.2 score half-life in steps.
    pub half_life: f64,
    /// Model fingerprint ([`model_fingerprint`]).
    pub model: String,
}

/// A parsed, validated score file.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmState {
    /// Decode step at save time.
    pub step: u64,
    /// `(flat key, last_access word, score word)`, ascending key.
    pub words: Vec<(usize, u64, u64)>,
}

/// Why a score file was refused (the caller starts cold).
#[derive(Debug, thiserror::Error)]
pub enum WarmError {
    /// Unreadable or structurally invalid file (truncation included).
    #[error("malformed score file: {0}")]
    Malformed(String),
    /// Recorded under a different policy shape or model.
    #[error("score file recorded for {file:?}, this session is {want:?}")]
    Mismatch {
        /// Shape in the file header.
        file: WarmShape,
        /// This session's shape.
        want: WarmShape,
    },
    /// Step too close to the packed word's wrap ([`MAX_STEP`]).
    #[error("score file step {0} exceeds the restorable maximum {MAX_STEP}")]
    StepWrap(u64),
}

/// Renders a score file: header, then one line per `(key, last_access,
/// score_word)` with `last_access != 0`.
pub fn format_scores(shape: &WarmShape, step: u64, words: &[(usize, u64, u64)]) -> String {
    let mut s = String::with_capacity(96 + words.len() * 32);
    let _ = writeln!(
        s,
        "{MAGIC} layers={} experts={} half_life={} step={step} keys={} model={}",
        shape.layers,
        shape.experts,
        shape.half_life,
        words.len(),
        shape.model
    );
    for &(k, a, w) in words {
        let _ = writeln!(s, "{k} {a} {w}");
    }
    s
}

/// Parses and validates `text` against `want`. Refuses a truncated file
/// (key count short or an unterminated last line), a header for another
/// shape or model, an out-of-range or repeated key, a word for a step past
/// the header's, and a step past [`MAX_STEP`].
pub fn parse_scores(text: &str, want: &WarmShape) -> Result<WarmState, WarmError> {
    let bad = |m: String| WarmError::Malformed(m);
    let body = text
        .strip_suffix('\n')
        .ok_or_else(|| bad("unterminated last line".into()))?;
    let mut lines = body.split('\n');
    let header = lines.next().unwrap_or_default();
    let rest = header
        .strip_prefix(MAGIC)
        .and_then(|r| r.strip_prefix(' '))
        .ok_or_else(|| bad(format!("expected `{MAGIC} ...` header: {header:?}")))?;
    let mut f = rest.split(' ');
    let mut field = |name: &str| {
        f.next()
            .and_then(|kv| kv.strip_prefix(name)?.strip_prefix('='))
            .ok_or_else(|| bad(format!("header field {name} missing: {header:?}")))
    };
    let num = |v: &str| {
        v.parse::<u64>()
            .map_err(|_| bad(format!("header value {v:?}")))
    };
    let layers = u32::try_from(num(field("layers")?)?).map_err(|_| bad("layers".into()))?;
    let experts = u32::try_from(num(field("experts")?)?).map_err(|_| bad("experts".into()))?;
    let half_life: f64 = field("half_life")?
        .parse()
        .map_err(|_| bad("half_life".into()))?;
    let step = num(field("step")?)?;
    let n_keys = num(field("keys")?)?;
    let model = field("model")?.to_owned();
    if f.next().is_some() {
        return Err(bad(format!("extra header fields: {header:?}")));
    }
    let file = WarmShape {
        layers,
        experts,
        half_life,
        model,
    };
    if file != *want {
        return Err(WarmError::Mismatch {
            file,
            want: want.clone(),
        });
    }
    if step > MAX_STEP {
        return Err(WarmError::StepWrap(step));
    }
    let n = layers as usize * experts as usize;
    let mut words = Vec::with_capacity(n.min(n_keys as usize));
    for line in lines {
        let mut f = line.split(' ');
        let mut next = || {
            f.next()
                .and_then(|v| v.parse::<u64>().ok())
                .ok_or_else(|| bad(format!("record {line:?}")))
        };
        let (k, a, w) = (next()?, next()?, next()?);
        if next().is_ok() {
            return Err(bad(format!("record {line:?}")));
        }
        let k = usize::try_from(k).map_err(|_| bad(format!("key {k}")))?;
        if k >= n {
            return Err(bad(format!("key {k} outside {layers} × {experts}")));
        }
        if words.last().is_some_and(|&(p, _, _)| p >= k) {
            return Err(bad(format!("key {k} out of order")));
        }
        // Both words carry `booked step + 1`; a booking is never later than
        // the step counter it was made under.
        if a == 0 || a > step || w & u64::from(u32::MAX) != a {
            return Err(bad(format!("words of key {k}: {a} {w}")));
        }
        let s = f32::from_bits((w >> 32) as u32);
        if !s.is_finite() || s < 0.0 {
            return Err(bad(format!("score of key {k}: {s}")));
        }
        words.push((k, a, w));
    }
    if words.len() as u64 != n_keys {
        return Err(bad(format!(
            "{} records, header says {n_keys} (truncated?)",
            words.len()
        )));
    }
    Ok(WarmState { step, words })
}

/// Stat-only model identity: first 16 hex of sha256 over every shard's
/// `(name, size, mtime)` ([`golem_gguf::shard_stamps`]). A requantized or
/// replaced model gets a different file, so a score file never cross-loads.
///
/// # Errors
/// The model header or a shard `stat` fails.
pub fn model_fingerprint(model: &Path) -> Result<String, golem_gguf::Error> {
    let mut h = Sha256::new();
    for (name, size, mtime) in golem_gguf::shard_stamps(model)? {
        h.update(format!("{name}\0{size}\0{mtime}\n"));
    }
    Ok(h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Score file name for a model fingerprint.
pub fn scores_file_name(fp: &str) -> String {
    format!("scores-{fp}.txt")
}

/// Per-instance warm-start directory under an explicit data dir; `None`
/// when no data dir is configured.
pub fn journal_dir_from(data_dir: Option<&str>, instance: Option<&str>) -> Option<PathBuf> {
    Some(
        Path::new(data_dir?)
            .join("journals")
            .join(instance.unwrap_or("default")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> WarmShape {
        WarmShape {
            layers: 2,
            experts: 4,
            half_life: 512.0,
            model: "0123456789abcdef".into(),
        }
    }

    fn word(score: f32, a: u64) -> u64 {
        u64::from(score.to_bits()) << 32 | a
    }

    fn sample() -> (u64, Vec<(usize, u64, u64)>) {
        (10, vec![(1, 3, word(1.5, 3)), (7, 10, word(2.0, 10))])
    }

    #[test]
    fn round_trip_preserves_step_and_words() {
        let (step, words) = sample();
        let text = format_scores(&shape(), step, &words);
        assert!(text.starts_with(
            "golem-scores v1 layers=2 experts=4 half_life=512 step=10 keys=2 model=0123456789abcdef\n"
        ));
        let back = parse_scores(&text, &shape()).unwrap();
        assert_eq!(back, WarmState { step, words });
        let empty = format_scores(&shape(), 0, &[]);
        assert!(parse_scores(&empty, &shape()).unwrap().words.is_empty());
    }

    #[test]
    fn truncated_tail_is_refused() {
        let (step, words) = sample();
        let text = format_scores(&shape(), step, &words);
        let cut = &text[..text.len() - 1];
        assert!(matches!(
            parse_scores(cut, &shape()),
            Err(WarmError::Malformed(_))
        ));
        let line_cut = &text[..text.trim_end().rfind('\n').unwrap() + 1];
        assert!(matches!(
            parse_scores(line_cut, &shape()),
            Err(WarmError::Malformed(_))
        ));
    }

    #[test]
    fn mismatched_shape_or_model_is_refused() {
        let (step, words) = sample();
        for other in [
            WarmShape {
                layers: 1,
                experts: 8,
                ..shape()
            },
            WarmShape {
                half_life: 100.0,
                ..shape()
            },
            WarmShape {
                model: "fedcba9876543210".into(),
                ..shape()
            },
        ] {
            let text = format_scores(&other, step, &words);
            assert!(
                matches!(
                    parse_scores(&text, &shape()),
                    Err(WarmError::Mismatch { .. })
                ),
                "{other:?}"
            );
        }
    }

    #[test]
    fn out_of_range_key_and_bad_words_are_refused() {
        for words in [
            vec![(8, 3, word(1.0, 3))],                       // key ≥ 2 × 4
            vec![(1, 11, word(1.0, 11))],                     // booked after step
            vec![(1, 3, word(1.0, 4))],                       // words disagree
            vec![(1, 3, word(f32::NAN, 3))],                  // non-finite score
            vec![(2, 3, word(1.0, 3)), (2, 3, word(1.0, 3))], // repeated key
        ] {
            let text = format_scores(&shape(), 10, &words);
            assert!(
                matches!(parse_scores(&text, &shape()), Err(WarmError::Malformed(_))),
                "{words:?}"
            );
        }
    }

    #[test]
    fn step_near_the_packed_wrap_is_refused() {
        let text = format_scores(&shape(), MAX_STEP + 1, &[]);
        assert!(matches!(
            parse_scores(&text, &shape()),
            Err(WarmError::StepWrap(_))
        ));
    }

    #[test]
    fn journal_dir_from_table() {
        let from = |d, i| journal_dir_from(d, i).map(|p| p.to_string_lossy().into_owned());
        assert_eq!(
            from(Some("/d"), None).as_deref(),
            Some("/d/journals/default")
        );
        assert_eq!(
            from(Some("/d"), Some("i")).as_deref(),
            Some("/d/journals/i")
        );
        assert_eq!(from(None, Some("i")), None);
        assert_eq!(from(None, None), None);
    }
}
