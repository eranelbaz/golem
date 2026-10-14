//! Request field policy as data (tech design §8.2); T7 tests iterate this table.
use crate::{ChatCompletionRequest, ErrorCode, ErrorEnvelope, chat::MessageContent};

/// How a top-level request field is treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldRule {
    /// Functional and honoured.
    Supported,
    /// Accepted, no functional effect (§8.2: metadata, user, request_id, `store: false`).
    Tolerated,
    /// Rejected with `golem_unsupported_field`; payload is the reason.
    Unsupported(&'static str),
    /// Golem extension (`top_k`).
    Extension,
}

/// §8.2 table. Adding a field here is the only way to accept it.
pub const FIELD_POLICY: &[(&str, FieldRule)] = &[
    ("model", FieldRule::Supported),
    ("messages", FieldRule::Supported),
    ("temperature", FieldRule::Supported),
    ("top_p", FieldRule::Supported),
    ("max_tokens", FieldRule::Supported),
    ("max_completion_tokens", FieldRule::Supported),
    ("stop", FieldRule::Supported),
    ("seed", FieldRule::Supported),
    ("frequency_penalty", FieldRule::Supported),
    ("presence_penalty", FieldRule::Supported),
    ("n", FieldRule::Supported),
    ("stream", FieldRule::Supported),
    ("stream_options", FieldRule::Supported),
    ("tools", FieldRule::Supported),
    ("tool_choice", FieldRule::Supported),
    ("reasoning_effort", FieldRule::Supported),
    ("metadata", FieldRule::Tolerated),
    ("user", FieldRule::Tolerated),
    ("request_id", FieldRule::Tolerated),
    ("store", FieldRule::Tolerated),
    ("top_k", FieldRule::Extension),
    (
        "response_format",
        FieldRule::Unsupported("structured output is not implemented in v1"),
    ),
    (
        "logit_bias",
        FieldRule::Unsupported("logit bias is not implemented in v1"),
    ),
    (
        "logprobs",
        FieldRule::Unsupported("logprobs are not implemented in v1"),
    ),
    (
        "parallel_tool_calls",
        FieldRule::Unsupported("at most one tool call is generated per turn"),
    ),
];

fn err(code: ErrorCode, msg: String) -> ErrorEnvelope {
    ErrorEnvelope::new(code, msg)
}

/// Validates a raw JSON body against §8.2 and returns the typed request.
pub fn validate(
    raw: &serde_json::Value,
    instance_model: &str,
) -> Result<ChatCompletionRequest, ErrorEnvelope> {
    let obj = raw.as_object().ok_or_else(|| {
        err(
            ErrorCode::InvalidRequest,
            "body must be a JSON object".into(),
        )
    })?;
    for key in obj.keys() {
        match FIELD_POLICY.iter().find(|(f, _)| f == key) {
            None => {
                return Err(err(
                    ErrorCode::UnsupportedField,
                    format!("field `{key}` is not supported"),
                ));
            }
            Some((_, FieldRule::Unsupported(why))) => {
                return Err(err(
                    ErrorCode::UnsupportedField,
                    format!("field `{key}` is not supported: {why}"),
                ));
            }
            Some(_) => {}
        }
    }
    let mut req: ChatCompletionRequest = serde_json::from_value(raw.clone())
        .map_err(|e| err(ErrorCode::InvalidRequest, e.to_string()))?;
    if req.store == Some(true) {
        return Err(err(
            ErrorCode::UnsupportedField,
            "field `store` is not supported: completions are not stored".into(),
        ));
    }
    if req.model != instance_model {
        return Err(err(
            ErrorCode::ModelMismatch,
            format!(
                "model `{}` is not loaded; this instance serves `{instance_model}`",
                req.model
            ),
        ));
    }
    if req.max_tokens.or(req.max_completion_tokens) == Some(0) {
        return Err(err(
            ErrorCode::InvalidRequest,
            "`max_tokens` must be ≥ 1".into(),
        ));
    }
    if req.n.unwrap_or(1) != 1 {
        return Err(err(ErrorCode::InvalidRequest, "`n` must be 1".into()));
    }
    if req.seed.is_some_and(|s| s >= u64::from(u32::MAX)) {
        return Err(err(
            ErrorCode::InvalidRequest,
            "`seed` must fit in u32 and must not equal u32::MAX (llama's random-seed sentinel)"
                .into(),
        ));
    }
    for (name, v, lo, hi) in [
        ("temperature", req.temperature, 0.0, 2.0),
        ("top_p", req.top_p, 0.0, 1.0),
        ("frequency_penalty", req.frequency_penalty, -2.0, 2.0),
        ("presence_penalty", req.presence_penalty, -2.0, 2.0),
    ] {
        if v.is_some_and(|v| !v.is_finite() || !(lo..=hi).contains(&v)) {
            return Err(err(
                ErrorCode::InvalidRequest,
                format!("`{name}` must be a finite number in [{lo}, {hi}]"),
            ));
        }
    }
    let fn_name = |v: &serde_json::Value| {
        v.pointer("/function/name")
            .and_then(|n| n.as_str())
            .map(str::to_owned)
    };
    if let Some(name) = req.tool_choice.as_ref().and_then(fn_name)
        && !req
            .tools
            .iter()
            .flatten()
            .any(|t| fn_name(t).as_deref() == Some(name.as_str()))
    {
        return Err(err(
            ErrorCode::InvalidRequest,
            format!("tool_choice names `{name}`, which is not in `tools`"),
        ));
    }
    if req.messages.iter().any(|m| matches!(m.content, Some(MessageContent::Parts(ref p)) if p.iter().any(|p| p.kind != "text"))) {
        return Err(err(ErrorCode::MultimodalUnsupported, "only text content parts are supported".into()));
    }
    if req.max_tokens.is_none() {
        req.max_tokens = req.max_completion_tokens.take();
    } else {
        req.max_completion_tokens = None;
    }
    Ok(req)
}
