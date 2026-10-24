//! Anthropic Messages API (`POST /v1/messages`, design §8.6): request
//! translation into the OpenAI chat body that `policy::validate` checks, and
//! the Anthropic error envelope. One decode path: nothing here runs a model.

use serde_json::{Map, Value, json};

use crate::policy::{FieldRule, check_fields};
use crate::{ErrorCode, ErrorEnvelope};

/// Top-level `/v1/messages` fields. `context_management`, `safeguards` and
/// `metadata` are sent by Claude Code and have no effect here.
pub const MESSAGES_FIELD_POLICY: &[(&str, FieldRule)] = &[
    ("model", FieldRule::Supported),
    ("messages", FieldRule::Supported),
    ("system", FieldRule::Supported),
    ("max_tokens", FieldRule::Supported),
    ("stop_sequences", FieldRule::Supported),
    ("temperature", FieldRule::Supported),
    ("top_p", FieldRule::Supported),
    ("top_k", FieldRule::Supported),
    ("stream", FieldRule::Supported),
    ("tools", FieldRule::Supported),
    ("tool_choice", FieldRule::Supported),
    ("thinking", FieldRule::Supported),
    ("output_config", FieldRule::Supported),
    ("metadata", FieldRule::Tolerated),
    ("context_management", FieldRule::Tolerated),
    ("safeguards", FieldRule::Tolerated),
];

/// Block keys accepted on every content block besides its own fields.
const BLOCK_TOLERATED: &[&str] = &["cache_control", "citations"];

fn invalid(msg: impl Into<String>) -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCode::InvalidRequest, msg)
}

fn unsupported(msg: impl Into<String>) -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCode::UnsupportedField, msg)
}

fn multimodal(kind: &str) -> ErrorEnvelope {
    ErrorEnvelope::new(
        ErrorCode::MultimodalUnsupported,
        format!("`{kind}` content blocks are not supported; text only"),
    )
}

fn str_field<'a>(v: &'a Value, key: &str, at: &str) -> Result<&'a str, ErrorEnvelope> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{at}: `{key}` must be a string")))
}

/// Rejects block keys outside `own` and [`BLOCK_TOLERATED`].
fn check_block(b: &Map<String, Value>, kind: &str, own: &[&str]) -> Result<(), ErrorEnvelope> {
    match b.keys().find(|k| {
        *k != "type" && !own.contains(&k.as_str()) && !BLOCK_TOLERATED.contains(&k.as_str())
    }) {
        Some(k) => Err(unsupported(format!(
            "field `{k}` of a `{kind}` block is not supported"
        ))),
        None => Ok(()),
    }
}

/// Content as a list of blocks: a string is one `text` block.
fn blocks(content: &Value, at: &str) -> Result<Vec<Map<String, Value>>, ErrorEnvelope> {
    match content {
        Value::String(s) => Ok(vec![
            json!({"type": "text", "text": s})
                .as_object()
                .cloned()
                .unwrap_or_default(),
        ]),
        Value::Array(a) => a
            .iter()
            .map(|b| {
                b.as_object()
                    .cloned()
                    .ok_or_else(|| invalid(format!("{at}: content blocks must be objects")))
            })
            .collect(),
        _ => Err(invalid(format!(
            "{at}: `content` must be a string or an array of blocks"
        ))),
    }
}

/// Joined text of `text` blocks; any other block kind is rejected (`image`
/// and `document` as multimodal).
fn text_only(content: &Value, at: &str, sep: &str) -> Result<String, ErrorEnvelope> {
    let mut parts = Vec::new();
    for b in blocks(content, at)? {
        let kind = b.get("type").and_then(Value::as_str).unwrap_or_default();
        match kind {
            "text" => {
                check_block(&b, kind, &["text"])?;
                parts.push(str_field(&Value::Object(b.clone()), "text", at)?.to_owned());
            }
            "image" | "document" => return Err(multimodal(kind)),
            _ => {
                return Err(unsupported(format!(
                    "{at}: `{kind}` content blocks are not supported here"
                )));
            }
        }
    }
    Ok(parts.join(sep))
}

/// `tool_result` blocks become `tool` messages, then the text blocks one
/// `user` message after them (Anthropic puts results first in the turn).
fn user_messages(content: &Value, at: &str, out: &mut Vec<Value>) -> Result<(), ErrorEnvelope> {
    let mut text = Vec::new();
    for b in blocks(content, at)? {
        let kind = b.get("type").and_then(Value::as_str).unwrap_or_default();
        let bv = Value::Object(b.clone());
        match kind {
            "text" => {
                check_block(&b, kind, &["text"])?;
                text.push(str_field(&bv, "text", at)?.to_owned());
            }
            "tool_result" => {
                check_block(&b, kind, &["tool_use_id", "content", "is_error"])?;
                let id = str_field(&bv, "tool_use_id", at)?;
                let mut content = match b.get("content") {
                    None | Some(Value::Null) => String::new(),
                    Some(c) => text_only(c, &format!("{at} tool_result"), "\n\n")?,
                };
                // The chat `tool` role has no error flag; keep it visible.
                if b.get("is_error") == Some(&Value::Bool(true)) {
                    content.insert_str(0, "Error: ");
                }
                out.push(json!({"role": "tool", "tool_call_id": id, "content": content}));
            }
            "image" | "document" => return Err(multimodal(kind)),
            _ => {
                return Err(unsupported(format!(
                    "{at}: `{kind}` content blocks are not supported in user messages"
                )));
            }
        }
    }
    if !text.is_empty() {
        out.push(json!({"role": "user", "content": text.join("\n\n")}));
    }
    Ok(())
}

fn assistant_message(content: &Value, at: &str) -> Result<Value, ErrorEnvelope> {
    let (mut text, mut thinking, mut calls) = (String::new(), String::new(), Vec::new());
    for b in blocks(content, at)? {
        let kind = b.get("type").and_then(Value::as_str).unwrap_or_default();
        let bv = Value::Object(b.clone());
        match kind {
            "text" => {
                check_block(&b, kind, &["text"])?;
                text.push_str(str_field(&bv, "text", at)?);
            }
            "thinking" => {
                check_block(&b, kind, &["thinking", "signature"])?;
                thinking.push_str(str_field(&bv, "thinking", at)?);
            }
            "tool_use" => {
                check_block(&b, kind, &["id", "name", "input"])?;
                let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
                calls.push(json!({
                    "id": str_field(&bv, "id", at)?,
                    "type": "function",
                    "function": {
                        "name": str_field(&bv, "name", at)?,
                        "arguments": input.to_string(),
                    },
                }));
            }
            "image" | "document" => return Err(multimodal(kind)),
            _ => {
                return Err(unsupported(format!(
                    "{at}: `{kind}` content blocks are not supported in assistant messages"
                )));
            }
        }
    }
    let mut m = Map::new();
    m.insert("role".into(), "assistant".into());
    if !text.is_empty() {
        m.insert("content".into(), text.into());
    }
    if !thinking.is_empty() {
        m.insert("reasoning_content".into(), thinking.into());
    }
    if !calls.is_empty() {
        m.insert("tool_calls".into(), calls.into());
    }
    Ok(Value::Object(m))
}

fn tool(t: &Value, i: usize) -> Result<Value, ErrorEnvelope> {
    let at = format!("tools[{i}]");
    let o = t
        .as_object()
        .ok_or_else(|| invalid(format!("{at} must be an object")))?;
    match o.get("type").and_then(Value::as_str) {
        None | Some("custom") => {}
        Some(k) => {
            return Err(unsupported(format!(
                "{at}: server tool `{k}` is not supported; only client tools with `input_schema`"
            )));
        }
    }
    check_block(o, "tool", &["name", "description", "input_schema"])?;
    let mut f = Map::new();
    f.insert("name".into(), str_field(t, "name", &at)?.into());
    if let Some(d) = o.get("description") {
        f.insert("description".into(), d.clone());
    }
    f.insert(
        "parameters".into(),
        o.get("input_schema")
            .cloned()
            .ok_or_else(|| invalid(format!("{at}: `input_schema` is required")))?,
    );
    Ok(json!({"type": "function", "function": f}))
}

fn tool_choice(c: &Value) -> Result<Value, ErrorEnvelope> {
    let o = c
        .as_object()
        .ok_or_else(|| invalid("`tool_choice` must be an object"))?;
    let kind = str_field(c, "type", "tool_choice")?;
    let own: &[&str] = match kind {
        "tool" => &["name", "disable_parallel_tool_use"],
        _ => &["disable_parallel_tool_use"],
    };
    check_block(o, "tool_choice", own)?;
    Ok(match kind {
        "auto" => "auto".into(),
        "any" => "required".into(),
        "none" => "none".into(),
        "tool" => {
            json!({"type": "function", "function": {"name": str_field(c, "name", "tool_choice")?}})
        }
        k => return Err(invalid(format!("unknown `tool_choice.type` `{k}`"))),
    })
}

/// `thinking`: `enabled`/`adaptive` keep the template's reasoning (the
/// default); `disabled` selects the template's no-think form (design §8.6).
/// `disabled` together with `output_config.effort` is a conflict — the
/// effort kwarg would silently re-enable thinking — and is a 400, never a
/// silent re-enable.
fn check_thinking(t: &Value) -> Result<(), ErrorEnvelope> {
    match t.get("type").and_then(Value::as_str) {
        Some("enabled" | "adaptive" | "disabled") => Ok(()),
        _ => Err(invalid(
            "`thinking.type` must be `enabled`, `adaptive` or `disabled`",
        )),
    }
}

/// `output_config.effort` → `reasoning_effort`; other keys are refused
/// (design §8.6). The value passes through verbatim under the chat-path
/// `reasoning_effort` rule: `max` reaches the template, which
/// renders 100 on DeepSeek-V4.1 and its default on GLM; anything else is a
/// 400, never a silent drop.
fn effort(c: &Value) -> Result<Option<Value>, ErrorEnvelope> {
    let o = c
        .as_object()
        .ok_or_else(|| invalid("`output_config` must be an object"))?;
    if let Some(k) = o.keys().find(|k| *k != "effort") {
        return Err(unsupported(format!(
            "field `output_config.{k}` is not supported"
        )));
    }
    match o.get("effort") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => crate::chat::parse_effort(v)
            .map(|_| Some(v.clone()))
            .map_err(|e| invalid(format!("`output_config.effort` {e}"))),
    }
}

/// Translates a raw `/v1/messages` body into the OpenAI chat body fed to
/// `policy::validate` (which still checks model, ranges and tool names).
pub fn to_chat(raw: &Value) -> Result<Value, ErrorEnvelope> {
    check_fields(raw, MESSAGES_FIELD_POLICY)?;
    if raw.get("max_tokens").is_none() {
        return Err(invalid("`max_tokens` is required"));
    }
    let mut msgs = Vec::new();
    if let Some(s) = raw.get("system") {
        let text = text_only(s, "system", "\n\n")?;
        if !text.is_empty() {
            msgs.push(json!({"role": "system", "content": text}));
        }
    }
    let list = raw
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("`messages` must be an array"))?;
    for (i, m) in list.iter().enumerate() {
        let at = format!("messages[{i}]");
        let content = m
            .get("content")
            .ok_or_else(|| invalid(format!("{at}: `content` is required")))?;
        match m.get("role").and_then(Value::as_str) {
            Some("user") => user_messages(content, &at, &mut msgs)?,
            Some("assistant") => msgs.push(assistant_message(content, &at)?),
            _ => {
                return Err(invalid(format!(
                    "{at}: `role` must be `user` or `assistant`"
                )));
            }
        }
    }
    let mut out = Map::new();
    out.insert("messages".into(), msgs.into());
    for (from, to) in [
        ("model", "model"),
        ("max_tokens", "max_tokens"),
        ("stop_sequences", "stop"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stream", "stream"),
    ] {
        if let Some(v) = raw.get(from) {
            out.insert(to.into(), v.clone());
        }
    }
    if let Some(t) = raw.get("tools") {
        let list = t
            .as_array()
            .ok_or_else(|| invalid("`tools` must be an array"))?;
        if !list.is_empty() {
            let tools = list
                .iter()
                .enumerate()
                .map(|(i, t)| tool(t, i))
                .collect::<Result<Vec<_>, _>>()?;
            out.insert("tools".into(), tools.into());
        }
    }
    if let Some(c) = raw.get("tool_choice") {
        out.insert("tool_choice".into(), tool_choice(c)?);
    }
    let thinking_disabled = raw
        .get("thinking")
        .map(|t| {
            check_thinking(t)?;
            Ok(t.get("type").and_then(Value::as_str) == Some("disabled"))
        })
        .transpose()?;
    let effort = raw.get("output_config").map(effort).transpose()?.flatten();
    if thinking_disabled == Some(true) && effort.is_some() {
        return Err(invalid(
            "`thinking: disabled` conflicts with `output_config.effort`",
        ));
    }
    if thinking_disabled == Some(true) {
        out.insert("reasoning_effort".into(), "none".into());
    } else if let Some(e) = effort {
        out.insert("reasoning_effort".into(), e);
    }
    Ok(Value::Object(out))
}

/// Anthropic `error.type` for a Golem code (the HTTP status is unchanged).
pub fn error_type(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::UnsupportedField
        | ErrorCode::InvalidRequest
        | ErrorCode::ModelMismatch
        | ErrorCode::MultimodalUnsupported => "invalid_request_error",
        ErrorCode::NotFound => "not_found_error",
        ErrorCode::Busy | ErrorCode::StorageUnavailable => "overloaded_error",
        ErrorCode::BackendUnavailable
        | ErrorCode::PlanInfeasible
        | ErrorCode::InstanceNotFound
        | ErrorCode::StopTimeout => "api_error",
    }
}

/// `{"type":"error","error":{"type":T,"message":"<golem code>: …"}}`.
pub fn error_body(e: &ErrorEnvelope, kind: &str) -> Value {
    let code = serde_json::to_value(e.error.code).unwrap_or_default();
    json!({
        "type": "error",
        "error": {
            "type": kind,
            "message": format!("{}: {}", code.as_str().unwrap_or_default(), e.error.message),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(body: Value) -> Value {
        to_chat(&body).unwrap()
    }

    fn code(body: Value) -> ErrorCode {
        to_chat(&body).unwrap_err().error.code
    }

    fn base(messages: Value) -> Value {
        json!({"model": "m", "max_tokens": 8, "messages": messages})
    }

    #[test]
    fn system_string_and_text_blocks_become_one_system_message() {
        let mut b = base(json!([{"role": "user", "content": "hi"}]));
        b["system"] = "s".into();
        assert_eq!(
            ok(b.clone())["messages"][0],
            json!({"role": "system", "content": "s"})
        );
        b["system"] = json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}]);
        assert_eq!(
            ok(b)["messages"][0],
            json!({"role": "system", "content": "a\n\nb"})
        );
    }

    #[test]
    fn user_text_and_tool_result_blocks_map_to_tool_then_user() {
        let b = base(json!([{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "42"}], "is_error": false},
            {"type": "text", "text": "go on"},
        ]}]));
        assert_eq!(
            ok(b)["messages"],
            json!([
                {"role": "tool", "tool_call_id": "t1", "content": "42"},
                {"role": "user", "content": "go on"},
            ])
        );
        let b = base(json!([{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "boom", "is_error": true},
        ]}]));
        assert_eq!(ok(b)["messages"][0]["content"], "Error: boom");
    }

    #[test]
    fn assistant_blocks_map_to_content_reasoning_and_tool_calls() {
        let b = base(json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": ""},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}},
            ]},
        ]));
        assert_eq!(
            ok(b)["messages"][1],
            json!({"role": "assistant", "reasoning_content": "plan", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":1}"}}
            ]})
        );
    }

    #[test]
    fn image_and_document_blocks_are_multimodal_400() {
        for kind in ["image", "document"] {
            let b = base(json!([{"role": "user", "content": [{"type": kind, "source": {}}]}]));
            assert_eq!(code(b), ErrorCode::MultimodalUnsupported);
        }
        let b = base(
            json!([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image", "source": {}}]}]}]),
        );
        assert_eq!(code(b), ErrorCode::MultimodalUnsupported);
    }

    #[test]
    fn tools_and_every_tool_choice_map_to_openai() {
        let mut b = base(json!([{"role": "user", "content": "q"}]));
        b["tools"] = json!([{"name": "f", "description": "d", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}]);
        assert_eq!(
            ok(b.clone())["tools"],
            json!([{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}])
        );
        for (from, to) in [
            (json!({"type": "auto"}), json!("auto")),
            (
                json!({"type": "any", "disable_parallel_tool_use": true}),
                json!("required"),
            ),
            (json!({"type": "none"}), json!("none")),
            (
                json!({"type": "tool", "name": "f"}),
                json!({"type": "function", "function": {"name": "f"}}),
            ),
        ] {
            b["tool_choice"] = from;
            assert_eq!(ok(b.clone())["tool_choice"], to);
        }
        b["tools"] = json!([{"type": "web_search_20250305", "name": "web_search"}]);
        b.as_object_mut().unwrap().remove("tool_choice");
        assert_eq!(code(b), ErrorCode::UnsupportedField);
    }

    #[test]
    fn sampling_fields_copy_and_stop_sequences_become_stop() {
        let mut b = base(json!([{"role": "user", "content": "q"}]));
        for (k, v) in [
            ("stop_sequences", json!(["X"])),
            ("temperature", json!(0.5)),
            ("top_p", json!(0.9)),
            ("top_k", json!(5)),
            ("stream", json!(true)),
        ] {
            b[k] = v;
        }
        b["metadata"] = json!({"user_id": "u"});
        let c = ok(b);
        assert_eq!(
            (
                &c["stop"],
                &c["temperature"],
                &c["top_p"],
                &c["top_k"],
                &c["stream"],
                c.get("metadata")
            ),
            (
                &json!(["X"]),
                &json!(0.5),
                &json!(0.9),
                &json!(5),
                &json!(true),
                None
            )
        );
    }

    #[test]
    fn thinking_disabled_maps_to_no_think() {
        let mut b = base(json!([{"role": "user", "content": "q"}]));
        for t in [
            json!({"type": "enabled", "budget_tokens": 1024}),
            json!({"type": "adaptive", "display": "omitted"}),
        ] {
            b["thinking"] = t;
            let c = ok(b.clone());
            assert!(c.get("thinking").is_none() && c.get("reasoning_effort").is_none());
        }
        b["thinking"] = json!({"type": "disabled"});
        let c = ok(b);
        assert!(c.get("thinking").is_none());
        assert_eq!(c["reasoning_effort"], "none");
    }

    #[test]
    fn thinking_disabled_with_output_config_effort_is_400() {
        let mut b = base(json!([{"role": "user", "content": "q"}]));
        b["thinking"] = json!({"type": "disabled"});
        b["output_config"] = json!({"effort": "high"});
        assert_eq!(code(b), ErrorCode::InvalidRequest);
    }

    #[test]
    fn output_config_effort_max_and_integer_pass_through() {
        let mut b = base(json!([{"role": "user", "content": "q"}]));
        b["output_config"] = json!({"effort": "max"});
        assert_eq!(ok(b.clone())["reasoning_effort"], "max");
        b["output_config"] = json!({"effort": 10});
        assert_eq!(ok(b.clone())["reasoning_effort"], 10);
        for bad in [json!("medium"), json!(0), json!(101), json!(10.5)] {
            b["output_config"] = json!({"effort": bad});
            assert_eq!(code(b.clone()), ErrorCode::InvalidRequest, "{bad}");
        }
    }

    #[test]
    fn output_config_effort_maps_to_reasoning_effort() {
        let mut b = base(json!([{"role": "user", "content": "q"}]));
        b["output_config"] = json!({"effort": "high"});
        assert_eq!(ok(b.clone())["reasoning_effort"], "high");
        b["output_config"] = json!({"format": {}});
        assert_eq!(code(b), ErrorCode::UnsupportedField);
    }

    #[test]
    fn claude_code_top_level_fields_are_tolerated_and_unknown_ones_rejected() {
        let mut b = base(json!([{"role": "user", "content": "q"}]));
        b["context_management"] = json!({"edits": []});
        b["safeguards"] = json!([]);
        let c = ok(b.clone());
        assert!(c.get("context_management").is_none() && c.get("safeguards").is_none());
        b["service_tier"] = "auto".into();
        let e = to_chat(&b).unwrap_err();
        assert_eq!(e.error.code, ErrorCode::UnsupportedField);
        assert!(e.error.message.contains("service_tier"));
    }

    #[test]
    fn missing_max_tokens_and_bad_roles_are_invalid() {
        let mut b = base(json!([{"role": "user", "content": "q"}]));
        b.as_object_mut().unwrap().remove("max_tokens");
        assert_eq!(code(b), ErrorCode::InvalidRequest);
        assert_eq!(
            code(base(json!([{"role": "system", "content": "q"}]))),
            ErrorCode::InvalidRequest
        );
        let b = base(
            json!([{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]),
        );
        assert_eq!(code(b), ErrorCode::UnsupportedField);
    }

    #[test]
    fn translated_body_passes_policy_validate() {
        let b = base(json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
        ]));
        let mut b = b;
        b["tools"] = json!([{"name": "f", "input_schema": {"type": "object"}}]);
        b["tool_choice"] = json!({"type": "tool", "name": "f"});
        crate::policy::validate(&ok(b), "m").unwrap();
    }

    #[test]
    fn error_body_carries_type_and_golem_code() {
        let e = ErrorEnvelope::new(ErrorCode::Busy, "all slots busy");
        assert_eq!(
            error_body(&e, error_type(e.error.code)),
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "golem_busy: all slots busy"}})
        );
    }
}
