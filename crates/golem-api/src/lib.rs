//! HTTP API contract: OpenAI-compatible types, the Golem error envelope, and
//! the request field policy (tech design §8).
//!
//! Scaffold (task B1): only the error envelope exists; the full type set and
//! the field-policy table land in task M4.1.

use serde::{Deserialize, Serialize};

/// Stable error codes shared by HTTP responses, CLI output, and logs
/// (tech design §8.4). Serialized as `snake_case`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// HIP backend missing, wrong GPU target, or kernel self-test failed.
    BackendUnavailable,
    /// `golem plan` arithmetic says the configuration cannot fit.
    PlanInfeasible,
    /// Requested model does not match the instance's loaded model.
    ModelMismatch,
    /// Multimodal input on the text-only v1 path.
    MultimodalUnsupported,
    /// Request field present but not supported; always names the field.
    UnsupportedField,
    /// Storage tier cannot serve reads (all replicas of a shard unreadable).
    StorageUnavailable,
    /// Registry lookup failed for the requested instance.
    InstanceNotFound,
    /// Instance is draining or overloaded; retry semantics attached.
    Busy,
    /// Malformed request that does not fit a more specific code.
    InvalidRequest,
}

/// Machine-readable error payload; `message` is human-readable (tech design §8.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    /// Stable, matchable code.
    pub code: ErrorCode,
    /// Human-readable explanation; never the only error signal.
    pub message: String,
}

/// Single envelope used everywhere an error is surfaced (tech design §8.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    /// Wrapped body so JSON mirrors `{"error": {...}}`.
    pub error: ErrorBody,
}

impl ErrorEnvelope {
    /// Convenience constructor for tests and early call sites.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            error: ErrorBody {
                code,
                message: message.into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_serializes_snake_case_code() {
        let env = ErrorEnvelope::new(ErrorCode::BackendUnavailable, "no HIP device gfx1100");
        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains(r#""code":"backend_unavailable""#), "{json}");
        let back: ErrorEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(back.error.code, ErrorCode::BackendUnavailable);
    }
}
