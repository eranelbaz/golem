//! HTTP API contract: OpenAI-compatible types, the Golem error envelope, and
//! the request field policy (tech design §8).

pub mod chat;
pub mod policy;

pub use chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    MessageContent, ModelList, Usage,
};

use serde::{Deserialize, Serialize};

/// Stable error codes shared by HTTP responses, CLI output, and logs
/// (tech design §8.4). Wire format is the design's literal `golem_*` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    /// HIP backend missing, wrong GPU target, or kernel self-test failed.
    #[serde(rename = "golem_backend_unavailable")]
    BackendUnavailable,
    /// `golem plan` arithmetic says the configuration cannot fit.
    #[serde(rename = "golem_plan_infeasible")]
    PlanInfeasible,
    /// Requested model does not match the instance's loaded model.
    #[serde(rename = "golem_model_mismatch")]
    ModelMismatch,
    /// Multimodal input on the text-only v1 path.
    #[serde(rename = "golem_multimodal_unsupported")]
    MultimodalUnsupported,
    /// Request field present but not supported; always names the field.
    #[serde(rename = "golem_unsupported_field")]
    UnsupportedField,
    /// Storage tier cannot serve reads (all replicas of a shard unreadable).
    #[serde(rename = "golem_storage_unavailable")]
    StorageUnavailable,
    /// Registry lookup failed for the requested instance.
    #[serde(rename = "golem_instance_not_found")]
    InstanceNotFound,
    /// Unknown endpoint or route (tech design §8.1).
    #[serde(rename = "golem_not_found")]
    NotFound,
    /// Instance is draining or overloaded; retry semantics attached.
    #[serde(rename = "golem_busy")]
    Busy,
    /// Malformed request that does not fit a more specific code.
    #[serde(rename = "golem_invalid_request")]
    InvalidRequest,
}

/// Machine-readable error payload; `message` is human-readable (tech design §8.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    /// Stable, matchable code.
    pub code: ErrorCode,
    /// Human-readable explanation; never the only error signal.
    pub message: String,
    /// Structured specifics (e.g. per-tier byte overruns for
    /// `golem_plan_infeasible`); omitted when not applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

/// Single envelope used everywhere an error is surfaced (tech design §8.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    /// Wrapped body so JSON mirrors `{"error": {...}}`.
    pub error: ErrorBody,
}

impl ErrorEnvelope {
    /// Constructor without structured detail.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self::with_detail(code, message, None)
    }

    /// Constructor with structured detail attached.
    pub fn with_detail(
        code: ErrorCode,
        message: impl Into<String>,
        detail: Option<serde_json::Value>,
    ) -> Self {
        Self {
            error: ErrorBody {
                code,
                message: message.into(),
                detail,
            },
        }
    }
}

/// JSON form, for CLI output and logs (one envelope everywhere, §8.4).
impl std::fmt::Display for ErrorEnvelope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string(self).map_err(|_| std::fmt::Error)?;
        f.write_str(&json)
    }
}

impl std::error::Error for ErrorEnvelope {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_codes_serialize_to_design_8_4_strings() {
        let cases = [
            (ErrorCode::BackendUnavailable, "golem_backend_unavailable"),
            (ErrorCode::PlanInfeasible, "golem_plan_infeasible"),
            (ErrorCode::ModelMismatch, "golem_model_mismatch"),
            (
                ErrorCode::MultimodalUnsupported,
                "golem_multimodal_unsupported",
            ),
            (ErrorCode::UnsupportedField, "golem_unsupported_field"),
            (ErrorCode::StorageUnavailable, "golem_storage_unavailable"),
            (ErrorCode::InstanceNotFound, "golem_instance_not_found"),
            (ErrorCode::NotFound, "golem_not_found"),
            (ErrorCode::Busy, "golem_busy"),
            (ErrorCode::InvalidRequest, "golem_invalid_request"),
        ];
        for (code, wire) in cases {
            assert_eq!(
                serde_json::to_value(code).unwrap(),
                serde_json::Value::String(wire.into()),
                "wire string mismatch for {code:?}"
            );
            assert_eq!(
                serde_json::from_value::<ErrorCode>(serde_json::to_value(code).unwrap()).unwrap(),
                code
            );
        }
    }

    #[test]
    fn envelope_roundtrips_and_omits_absent_detail() {
        let env = ErrorEnvelope::new(ErrorCode::BackendUnavailable, "no HIP device gfx1100");
        let json = serde_json::to_string(&env).unwrap();
        assert!(
            json.contains(r#""code":"golem_backend_unavailable""#),
            "{json}"
        );
        assert!(!json.contains("detail"), "{json}");
        let back: ErrorEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(back.error.code, ErrorCode::BackendUnavailable);
        assert!(back.error.detail.is_none());
    }

    #[test]
    fn envelope_carries_structured_detail_when_present() {
        let env = ErrorEnvelope::with_detail(
            ErrorCode::PlanInfeasible,
            "VRAM over budget",
            Some(serde_json::json!({ "vram_over_bytes": 1_572_864 })),
        );
        let json = serde_json::to_string(&env).unwrap();
        assert!(
            json.contains(r#""detail":{"vram_over_bytes":1572864}"#),
            "{json}"
        );
        let back: ErrorEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.error.detail,
            Some(serde_json::json!({ "vram_over_bytes": 1_572_864 }))
        );
    }
}
