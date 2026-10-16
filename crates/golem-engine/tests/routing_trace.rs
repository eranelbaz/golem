//! Routing traces (issue #50, §2.3 step 4): with `routing_trace` set, the
//! routing readback appends one record per (step, layer) to a preallocated
//! ring flushed by `SessionHooks::write_routing_trace` under a
//! `# golem-routing-trace v1` header; with it unset, nothing is written.
//! Drives the engine-registered `routing` trampoline directly over the
//! tiny-MoE fixture on the CPU backend (no GPU split needed — the trace
//! records what the readback delivers, wherever the scheduler runs).
mod common;

use common::{expert_tensor as xt, tiny_moe_path};
use golem_engine::config::EngineConfig;
use golem_engine::p3_glue;

/// The fixture's expert layers (tiny-moe: one expert layer, dense lead 1 → blk 1).
const ALL_BLKS: [i32; 1] = [xt::BLK];

fn cfg(trace: Option<std::path::PathBuf>) -> EngineConfig {
    EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        routing_trace: trace,
        ..EngineConfig::default()
    }
}

/// Drives `n_steps` decode-phase routing readbacks for every expert layer of
/// the fixture (blk.0 and blk.1, ids = the step's routed pair).
fn drive_steps(hooks: &p3_glue::SessionHooks, n_steps: u32, layers: &[i32]) {
    for step in 0..n_steps {
        let ids: [i32; 2] = [
            step as i32 % xt::N_EXPERTS,
            (step + 1) as i32 % xt::N_EXPERTS,
        ];
        for &blk in layers {
            p3_glue::routing(
                hooks.payload_ptr(),
                blk,
                std::ptr::null(),
                ids.as_ptr(),
                2,
                xt::N_EXPERTS,
            );
        }
        p3_glue::copy_done(hooks.payload_ptr());
        // One trunk decode = one step (§6.2): the trace's step column follows
        // the same counter the Session bumps per generation decode.
        hooks.advance_decode_step();
    }
}

#[test]
fn trace_records_one_row_per_layer_per_step_and_round_trips() {
    let dir = std::env::temp_dir().join(format!("golem-trace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("trace.tsv");

    let hooks = p3_glue::enable(&cfg(Some(path.clone()))).expect("pooled enable with trace");
    hooks.set_scoring(true); // Session::new arms it after the self-check
    let layers: &[i32] = &ALL_BLKS;
    drive_steps(&hooks, 3, layers);

    let mut out = Vec::new();
    {
        let hooks = &hooks;
        hooks
            .write_routing_trace(&mut out)
            .expect("trace flush writes");
    } // guard drop: unregister-join (no in-flight callbacks here)

    let text = String::from_utf8(out).unwrap();
    let mut lines = text.lines();
    let header = lines.next().expect("header line");
    assert!(
        header.starts_with("# golem-routing-trace v1 model=") && header.contains("top_k=2"),
        "header must carry version/model/layers/top_k: {header}"
    );
    let rows: Vec<&str> = lines.collect();
    assert_eq!(
        rows.len(),
        3 * layers.len(),
        "one row per (step, layer): {rows:?}"
    );
    for (i, row) in rows.iter().enumerate() {
        let cols: Vec<&str> = row.split('\t').collect();
        assert_eq!(cols.len(), 4, "step/phase/layer/ids: {row}");
        assert_eq!(cols[1], "dc", "decode-phase records");
        assert_eq!(
            cols[3].split(',').count(),
            2,
            "ids = one per routed expert: {row}"
        );
        assert_eq!(
            cols[0].parse::<u32>().unwrap(),
            (i / layers.len()) as u32,
            "step-local monotonic: {row}"
        );
        // Ids cycle with the step: the trace carries the real routed pair.
        let want_step = (i / layers.len()) as i32;
        let want = format!(
            "{},{}",
            want_step % xt::N_EXPERTS,
            (want_step + 1) % xt::N_EXPERTS
        );
        assert_eq!(cols[3], want, "routed ids per step: {row}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_disabled_writes_nothing() {
    let hooks = p3_glue::enable(&cfg(None)).expect("pooled enable without trace");
    hooks.set_scoring(true); // Session::new arms it after the self-check
    drive_steps(&hooks, 2, &ALL_BLKS);
    let mut out = Vec::new();
    {
        let hooks = &hooks;
        hooks
            .write_routing_trace(&mut out)
            .expect("flush is a no-op, not an error");
    }
    assert!(
        out.is_empty(),
        "no trace configured → no bytes: {:?}",
        String::from_utf8_lossy(&out)
    );
}

/// The CLI surface maps `--routing-trace <FILE>` onto
/// `EngineConfig::routing_trace` and flushes per turn end (mirrors the CSV).
#[test]
fn engine_config_carries_the_trace_path() {
    let p = std::env::temp_dir().join(format!("golem-trace-cfg-{}.tsv", std::process::id()));
    let c = cfg(Some(p.clone()));
    assert_eq!(c.routing_trace.as_deref(), Some(p.as_path()));
    // Sanity: the default is off.
    assert!(EngineConfig::default().routing_trace.is_none());
}
