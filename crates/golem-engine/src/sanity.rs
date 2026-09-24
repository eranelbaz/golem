//! Load-time GGUF sanity checks (tech design §4.3, M1.3 subset).

use std::collections::BTreeMap;

use crate::error::EngineError;
use crate::gguf::{GgufMetadata, GgufValue, TensorMeta};

/// FFN expert-tensor suffixes, in canonical order.
const EXPERT_SUFFIXES: [&str; 3] = ["gate", "up", "down"];

/// Expert-tensor table: layer number -> suffix -> tensor info.
type Experts<'a> = BTreeMap<u32, BTreeMap<&'a str, &'a TensorMeta>>;

/// Parses `blk.<N>.ffn_<gate|up|down>_exps.*` into `(layer, suffix)`.
fn parse_expert_name(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let layer = rest[..dot].parse().ok()?;
    let rest = &rest[dot + 1..];
    EXPERT_SUFFIXES
        .into_iter()
        .find(|s| {
            rest.strip_prefix("ffn_")
                .and_then(|r| r.strip_prefix(s))
                .and_then(|r| r.strip_prefix("_exps."))
                .is_some()
        })
        .map(|s| (layer, s))
}

/// Static name of a non-integral `GgufValue`, for loud error messages.
fn value_type_name(value: &GgufValue) -> &'static str {
    match value {
        GgufValue::F32(_) => "f32",
        GgufValue::F64(_) => "f64",
        GgufValue::Bool(_) => "bool",
        GgufValue::Str(_) => "string",
        GgufValue::Array(_) => "array",
        GgufValue::U8(_) | GgufValue::U16(_) | GgufValue::U32(_) | GgufValue::U64(_) => "integer",
        GgufValue::I8(_) | GgufValue::I16(_) | GgufValue::I32(_) | GgufValue::I64(_) => "integer",
    }
}

/// Reads `<arch>.block_count`, accepting any integral `GgufValue` variant.
/// `None` = KV (or `general.architecture`) absent; present-but-non-integral
/// or negative is a loud error, never a silent fallback.
fn block_count(metadata: &GgufMetadata) -> Result<Option<u64>, EngineError> {
    let Some(arch) = metadata.kv_str("general.architecture") else {
        return Ok(None);
    };
    let key = format!("{arch}.block_count");
    let Some((_, value)) = metadata.kvs.iter().find(|(k, _)| *k == key) else {
        return Ok(None);
    };
    let signed = |v: i64| -> Result<Option<u64>, EngineError> {
        u64::try_from(v)
            .map(Some)
            .map_err(|_| EngineError::UnsupportedModel(format!("{key}: negative block_count {v}")))
    };
    match value {
        GgufValue::U8(v) => Ok(Some(u64::from(*v))),
        GgufValue::U16(v) => Ok(Some(u64::from(*v))),
        GgufValue::U32(v) => Ok(Some(u64::from(*v))),
        GgufValue::U64(v) => Ok(Some(*v)),
        GgufValue::I8(v) => signed(i64::from(*v)),
        GgufValue::I16(v) => signed(i64::from(*v)),
        GgufValue::I32(v) => signed(i64::from(*v)),
        GgufValue::I64(v) => signed(*v),
        other => Err(EngineError::UnsupportedModel(format!(
            "{key}: non-integral value of type {}; block_count must be an integer",
            value_type_name(other)
        ))),
    }
}

/// True if `layers` (ascending) is a run with step 1.
fn is_contiguous(layers: &[u32]) -> bool {
    layers.windows(2).all(|w| w[1] == w[0] + 1)
}

fn layer_list(layers: &[u32]) -> String {
    layers
        .iter()
        .map(|l| format!("blk.{l}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Expert-bearing layers must form a contiguous run `L..=block_count-1`
/// (a dense lead is valid); without the KV, observed layers must at least be
/// contiguous. Every covered layer must carry all three suffixes.
fn check_coverage(metadata: &GgufMetadata, experts: &Experts) -> Result<(), EngineError> {
    let observed: Vec<u32> = experts.keys().copied().collect();
    match block_count(metadata)? {
        Some(count) => {
            let ends_at_last = observed
                .last()
                .is_some_and(|&l| l as i64 == count as i64 - 1);
            if !(ends_at_last && is_contiguous(&observed)) {
                let last = count.saturating_sub(1);
                return Err(EngineError::UnsupportedModel(format!(
                    "expert tensors cover [{}] but must form a contiguous run of layers ending \
                     at the last block (block_count = {count}); expected blk.{last} to bear \
                     ffn_gate_exps/ffn_up_exps/ffn_down_exps",
                    layer_list(&observed)
                )));
            }
        }
        None => {
            if !is_contiguous(&observed) {
                return Err(EngineError::UnsupportedModel(format!(
                    "expert tensors cover non-contiguous layers [{}] and no \
                     <arch>.block_count KV is present to define the expected run",
                    layer_list(&observed)
                )));
            }
        }
    }
    for (layer, suffixes) in experts {
        for suffix in EXPERT_SUFFIXES {
            if !suffixes.contains_key(suffix) {
                return Err(EngineError::UnsupportedModel(format!(
                    "expert layer blk.{layer} is missing ffn_{suffix}_exps; every expert layer \
                     must bear ffn_gate_exps, ffn_up_exps, and ffn_down_exps"
                )));
            }
        }
    }
    Ok(())
}

/// `ne[2]` (stacked-expert rows) must be equal across gate/up/down within
/// each layer and > 0. Runs before cross-layer uniformity: a mutated
/// `ne[2]` breaks both, and this check owns the message.
fn check_expert_counts(experts: &Experts) -> Result<(), EngineError> {
    for (layer, suffixes) in experts {
        let gate = suffixes["gate"].ne[2];
        let up = suffixes["up"].ne[2];
        let down = suffixes["down"].ne[2];
        for (suffix, count) in EXPERT_SUFFIXES.iter().zip([gate, up, down]) {
            if count == 0 {
                return Err(EngineError::UnsupportedModel(format!(
                    "blk.{layer}: expert count is zero: ffn_{suffix}_exps has ne[2] = 0"
                )));
            }
        }
        if gate != up || gate != down {
            return Err(EngineError::UnsupportedModel(format!(
                "blk.{layer}: expert count mismatch across ffn_gate_exps/ffn_up_exps/ffn_down_exps: \
                 ne[2] = {gate}, {up}, {down}"
            )));
        }
    }
    Ok(())
}

/// For each suffix, `(ne, ggml_type)` must be identical across all layers.
fn check_cross_layer_uniformity(experts: &Experts) -> Result<(), EngineError> {
    for suffix in EXPERT_SUFFIXES {
        let mut layers = experts.values().filter_map(|s| s.get(suffix)).copied();
        let Some(first) = layers.next() else {
            continue;
        };
        for t in layers {
            if t.ne != first.ne || t.ggml_type != first.ggml_type {
                return Err(EngineError::UnsupportedModel(format!(
                    "ffn_{suffix}_exps tensors differ across layers: {} has ne = {:?}, \
                     ggml_type = {}, but {} has ne = {:?}, ggml_type = {}",
                    first.name, first.ne, first.ggml_type, t.name, t.ne, t.ggml_type
                )));
            }
        }
    }
    Ok(())
}

/// Runs the M1.3 subset of the tech-design §4.3 sanity checks against parsed
/// GGUF metadata: expert-tensor coverage, per-layer expert count, and
/// cross-layer shape/dtype uniformity. Violations are
/// [`EngineError::UnsupportedModel`] with the offending names and shapes.
pub fn check(metadata: &GgufMetadata) -> Result<(), EngineError> {
    let mut experts: Experts = BTreeMap::new();
    for tensor in &metadata.tensors {
        if let Some((layer, suffix)) = parse_expert_name(&tensor.name) {
            experts.entry(layer).or_default().insert(suffix, tensor);
        }
    }
    if experts.is_empty() {
        return Ok(()); // dense model
    }
    check_coverage(metadata, &experts)?;
    check_expert_counts(&experts)?;
    check_cross_layer_uniformity(&experts)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufValue;
    use crate::gguf::test_gguf;

    fn expert_tensor(
        layer: u32,
        suffix: &str,
        n_expert: u64,
        n_ff: u64,
        n_embd: u64,
    ) -> test_gguf::Tensor {
        // real MoE layout (tiny_moe.rs): gate/up [n_embd, n_ff, n_exp], down [n_ff, n_embd, n_exp]
        let ne = if suffix == "down" {
            [n_ff, n_embd, n_expert, 0]
        } else {
            [n_embd, n_ff, n_expert, 0]
        };
        test_gguf::Tensor {
            name: format!("blk.{layer}.ffn_{suffix}_exps.weight"),
            n_dims: 3,
            ne,
            ggml_type: 0, // F32
        }
    }

    fn valid_meta() -> crate::gguf::GgufMetadata {
        let kvs = vec![
            (
                "general.architecture".to_string(),
                GgufValue::Str("glm4moe".to_string()),
            ),
            ("glm4moe.block_count".to_string(), GgufValue::U32(2)),
        ];
        let tensors = vec![
            expert_tensor(0, "gate", 8, 16, 32),
            expert_tensor(0, "up", 8, 16, 32),
            expert_tensor(0, "down", 8, 32, 16),
            expert_tensor(1, "gate", 8, 16, 32),
            expert_tensor(1, "up", 8, 16, 32),
            expert_tensor(1, "down", 8, 32, 16),
        ];
        test_gguf::metadata_with(&kvs, &tensors)
    }

    #[test]
    fn uniform_experts_pass() {
        assert!(check(&valid_meta()).is_ok());
    }

    #[test]
    fn shape_mismatch_across_layers_rejected() {
        let mut m = valid_meta();
        m.tensors[3].ne[0] += 8;
        let err = check(&m).unwrap_err();
        assert!(matches!(err, EngineError::UnsupportedModel(_)));
        assert!(err.to_string().contains("ffn_gate_exps"));
        assert!(err.to_string().contains("blk.0") && err.to_string().contains("blk.1"));
    }

    #[test]
    fn expert_count_mismatch_within_layer_rejected() {
        let mut m = valid_meta();
        m.tensors[2].ne[2] = 4;
        let err = check(&m).unwrap_err();
        assert!(err.to_string().contains("expert count"));
    }

    #[test]
    fn dense_lead_then_experts_is_valid() {
        // tiny-moe shape: experts ONLY on the last block (blk.1), blk.0 dense
        let mut m = valid_meta();
        m.tensors.retain(|t| !t.name.starts_with("blk.0."));
        assert!(check(&m).is_ok());
    }

    #[test]
    fn expert_run_not_ending_at_last_block_rejected() {
        let mut m = valid_meta();
        m.tensors.retain(|t| !t.name.starts_with("blk.1."));
        let err = check(&m).unwrap_err();
        assert!(err.to_string().contains("blk.1"));
    }

    #[test]
    fn no_expert_tensors_is_fine() {
        let m = test_gguf::metadata_with(&[], &[]);
        assert!(check(&m).is_ok());
    }
}
