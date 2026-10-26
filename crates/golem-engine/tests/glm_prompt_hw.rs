//! Real-target leg for the GLM jinja render: the GGUF's own template adds the
//! `Reasoning Effort` header and the `<think>` generation prompt. Checks the
//! rendered bytes, then that each effort yields the intended reasoning split
//! (reasoning for low/high, none for no-think) and a correct answer.
#![cfg(feature = "hip-tests")]
use golem_engine::config::EngineConfig;
use golem_engine::session::expert_placement;
use golem_engine::{ChatTurn, FinishReason, Session, TurnRequest};

#[test]
#[ignore = "real model; GOLEM_HW=1 GOLEM_HW_MODEL=<shard 1> --run-ignored"]
fn hw_glm_jinja_prompt_reasoning() {
    assert_eq!(
        std::env::var("GOLEM_HW").as_deref(),
        Ok("1"),
        "set GOLEM_HW=1"
    );
    let model: std::path::PathBuf = std::env::var("GOLEM_HW_MODEL")
        .expect("GOLEM_HW_MODEL must name shard 1")
        .into();
    let cfg = EngineConfig {
        model_path: model.clone(),
        n_ctx: 4096,
        n_predict: 1024,
        n_gpu_layers: 99,
        flash_attn_on: true,
        ram_experts_gib: 57,
        manifest_path: Some(model.with_file_name("manifest.json")),
        ..EngineConfig::default()
    };
    let msgs = [ChatTurn {
        role: "user",
        content: "What is 17 * 23? Reply with just the number.",
    }];
    let mut s = Session::new(&cfg).expect("real-model session");
    let before = expert_placement();
    // §8.6: no-think cannot drop the header, so the shim pins it to Low.
    for (effort, header, think) in [
        (Some("none"), "Reasoning Effort: Low", false),
        (Some("low"), "Reasoning Effort: Low", true),
        (Some("high"), "Reasoning Effort: High", true),
    ] {
        let req = TurnRequest::new(&msgs).reasoning_effort(effort);
        let r = s.render(&req).expect("render");
        let p = &r.prompt;
        assert!(p.starts_with("[gMASK]<sop>"), "{p:?}");
        assert_eq!(p.matches("[gMASK]<sop>").count(), 1, "{p:?}");
        assert!(p.contains(header), "{effort:?}: {p:?}");
        let tail = if think {
            "<|assistant|><think>"
        } else {
            "<|assistant|><think></think>"
        };
        assert!(p.ends_with(tail), "{effort:?}: {p:?}");
        let mut text = String::new();
        let out = s
            .turn(&req.rendered(&r), &mut |t| text.push_str(t))
            .expect("turn");
        let msg = r.parser.parse(&text, false).expect("parse");
        let reasoning = msg
            .get("reasoning_content")
            .map(|v| v.as_str().expect("reasoning_content"))
            .unwrap_or("");
        let content = msg["content"].as_str().expect("content");
        println!(
            "HW effort={effort:?} prompt_tokens={} completion_tokens={} finish={:?} \
             reasoning_chars={} content={content:?}",
            out.prompt_tokens,
            out.completion_tokens,
            out.finish,
            reasoning.len()
        );
        assert_eq!(out.finish, FinishReason::Stop, "{effort:?}: {text:?}");
        assert_eq!(
            content.trim().trim_end_matches('.'),
            "391",
            "{effort:?}: {text:?}"
        );
        assert!(!content.contains("think>"), "{effort:?}: {text:?}");
        assert!(!reasoning.contains("think>"), "{effort:?}: {text:?}");
        assert_eq!(!reasoning.is_empty(), think, "{effort:?}: {text:?}");
    }
    let after = expert_placement();
    assert_eq!(after.on_cpu, before.on_cpu, "CPU expert math");
    assert!(after.on_gpu > before.on_gpu, "experts never ran on the GPU");
}
