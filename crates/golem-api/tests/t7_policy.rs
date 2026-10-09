//! T7: §8.2 field-policy table behaviour and request round trip.
use golem_api::{
    ErrorCode,
    policy::{FIELD_POLICY, FieldRule, validate},
};
use serde_json::json;

fn base() -> serde_json::Value {
    json!({"model":"tiny-moe","messages":[{"role":"user","content":"hi"}]})
}

fn sample(field: &str) -> serde_json::Value {
    match field {
        "model" => json!("tiny-moe"),
        "messages" => json!([{"role":"user","content":"hi"}]),
        "temperature" | "top_p" | "frequency_penalty" | "presence_penalty" => json!(0.5),
        "max_tokens" | "max_completion_tokens" | "seed" | "top_k" => json!(8),
        "n" => json!(1),
        "stop" => json!(["\n"]),
        "stream" => json!(true),
        "stream_options" => json!({"include_usage": true}),
        "tools" => {
            json!([{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}])
        }
        "tool_choice" => json!("auto"),
        "metadata" => json!({"k":"v"}),
        "user" | "request_id" => json!("x"),
        "response_format" => json!({"type":"json_schema"}),
        "logit_bias" => json!({"1": 1}),
        "logprobs" => json!(true),
        "reasoning_effort" => json!("high"),
        other => panic!("table field {other} has no sample; add one"),
    }
}

#[test]
fn every_table_row_behaves_as_declared_and_round_trips() {
    for &(field, rule) in FIELD_POLICY {
        let mut req = base();
        req[field] = sample(field);
        let res = validate(&req, "tiny-moe");
        match rule {
            FieldRule::Unsupported(_) => {
                let e = res.expect_err(field);
                assert_eq!(e.error.code, ErrorCode::UnsupportedField, "{field}");
                assert!(
                    e.error.message.contains(field),
                    "{field}: {}",
                    e.error.message
                );
            }
            _ => {
                let typed = res.unwrap_or_else(|e| panic!("{field}: {e:?}"));
                let back = serde_json::to_value(&typed).unwrap();
                // A field dropped by serde matches on both sides of a to_value round trip;
                // compare against the sample so a dropped field fails.
                let key = if field == "max_completion_tokens" {
                    "max_tokens"
                } else {
                    field
                };
                assert_eq!(
                    back.get(key),
                    Some(&sample(field)),
                    "{field} dropped by serde or aliased wrong"
                );
            }
        }
    }
}

#[test]
fn unknown_field_is_typed_400_naming_field() {
    let mut req = base();
    req["frobnicate"] = json!(1);
    let e = validate(&req, "tiny-moe").unwrap_err();
    assert_eq!(e.error.code, ErrorCode::UnsupportedField);
    assert!(e.error.message.contains("frobnicate"));
}

#[test]
fn model_mismatch_n_and_multimodal() {
    let mut r = base();
    r["model"] = json!("other");
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::ModelMismatch
    );
    let mut r = base();
    r["n"] = json!(2);
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::InvalidRequest
    );
    let mut r = base();
    r["messages"] =
        json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]);
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::MultimodalUnsupported
    );
    let mut r = base();
    r["seed"] = json!(u64::from(u32::MAX) + 1);
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::InvalidRequest
    );
    let mut r = base();
    r["seed"] = json!(-1);
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::InvalidRequest
    );
    let mut r = base();
    r["seed"] = json!(u32::MAX);
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::InvalidRequest,
        "u32::MAX is LLAMA_DEFAULT_SEED (random)"
    );
    let mut r = base();
    r["tools"] = json!([{"type":"function","function":{"name":"f"}}]);
    r["tool_choice"] = json!({"type":"function","function":{"name":"g"}});
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::InvalidRequest
    );
}

#[test]
fn max_completion_tokens_aliases_max_tokens() {
    let mut r = base();
    r["max_completion_tokens"] = json!(7);
    assert_eq!(validate(&r, "tiny-moe").unwrap().max_tokens, Some(7));
}
