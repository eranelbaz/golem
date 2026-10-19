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
        "store" => json!(false),
        "response_format" => json!({"type":"json_schema"}),
        "logit_bias" => json!({"1": 1}),
        "logprobs" => json!(true),
        "reasoning_effort" => json!("high"),
        "parallel_tool_calls" => json!(true),
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
    r["seed"] = json!(u32::MAX - 1);
    assert_eq!(
        validate(&r, "tiny-moe").unwrap().seed,
        Some(u64::from(u32::MAX - 1))
    );
    let mut r = base();
    r["seed"] = json!(u32::MAX);
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::InvalidRequest,
        "u32::MAX is LLAMA_DEFAULT_SEED (random)"
    );
    for (field, bad) in [
        ("temperature", json!(2.5)),
        ("temperature", json!(-0.1)),
        ("temperature", json!(1e39)),
        ("top_p", json!(1.5)),
        ("top_p", json!(-0.1)),
        ("top_p", json!(1e39)),
        ("frequency_penalty", json!(-2.5)),
        ("frequency_penalty", json!(-1e39)),
        ("presence_penalty", json!(3)),
        ("presence_penalty", json!(1e39)),
    ] {
        let mut r = base();
        r[field] = bad.clone();
        let e = validate(&r, "tiny-moe").unwrap_err().error;
        assert_eq!(e.code, ErrorCode::InvalidRequest, "{field}={bad}");
        assert!(e.message.contains(field), "{}", e.message);
    }
    let mut r = base();
    r["temperature"] = json!(2);
    r["top_p"] = json!(0);
    r["frequency_penalty"] = json!(-2);
    r["presence_penalty"] = json!(2);
    validate(&r, "tiny-moe").unwrap();
    let mut r = base();
    r["tools"] = json!([{"type":"function","function":{"name":"f"}}]);
    r["tool_choice"] = json!({"type":"function","function":{"name":"g"}});
    assert_eq!(
        validate(&r, "tiny-moe").unwrap_err().error.code,
        ErrorCode::InvalidRequest
    );
}

#[test]
fn store_true_is_rejected_false_and_null_tolerated() {
    let mut r = base();
    r["store"] = json!(true);
    let e = validate(&r, "tiny-moe").unwrap_err().error;
    assert_eq!(e.code, ErrorCode::UnsupportedField);
    assert!(e.message.contains("store"), "{}", e.message);
    for ok in [json!(false), json!(null)] {
        let mut r = base();
        r["store"] = ok;
        validate(&r, "tiny-moe").unwrap();
    }
}

#[test]
fn max_completion_tokens_aliases_max_tokens() {
    let mut r = base();
    r["max_completion_tokens"] = json!(7);
    assert_eq!(validate(&r, "tiny-moe").unwrap().max_tokens, Some(7));
}

#[test]
fn reasoning_effort_accepts_low_high_none_else_400() {
    let with = |e: serde_json::Value| {
        let mut r = base();
        r["reasoning_effort"] = e;
        validate(&r, "tiny-moe")
    };
    for ok in [json!("low"), json!("high"), json!("none")] {
        assert_eq!(
            with(ok.clone()).unwrap().reasoning_effort,
            Some(ok.as_str().unwrap().to_string()),
            "{ok}"
        );
    }
    assert_eq!(with(json!(null)).unwrap().reasoning_effort, None);
    for bad in [
        json!("max"),
        json!("medium"),
        json!("minimal"),
        json!(""),
        json!("LOW"),
        json!(" low"),
        json!("low\0"),
        json!(5),
        json!(true),
        json!({"level": "low"}),
    ] {
        let e = with(bad.clone()).unwrap_err().error;
        assert_eq!(e.code, ErrorCode::InvalidRequest, "{bad}: {}", e.message);
        assert!(
            e.message.contains("reasoning_effort"),
            "{bad}: {}",
            e.message
        );
    }
    // Absent field → max (template default), nothing sent.
    assert_eq!(
        validate(&base(), "tiny-moe").unwrap().reasoning_effort,
        None
    );
}
