//! T8 (§12, M4.6): `golem_plan::plan` predicted VRAM vs `hipMemGetInfo` used
//! after load + one streamed prefill, on the real target, over a ctx ×
//! ubatch sweep at parallel 1. Pass: |predicted − actual| ≤ 256 MiB at every
//! point. Opt-in hardware leg (GOLEM_HW=1, GOLEM_HW_MODEL = shard 1 of
//! GLM-5.3-Flash UD-Q4_K_XL): each point is a full real-model load. Budget
//! ≤ 60 loads; `GOLEM_T8_MAX_POINTS` caps the sweep (record the cap in
//! docs/baselines/m4.6-plan-t8.md). Fit `golem_plan::WsModel` from this
//! table's `ws` column before tightening the bound.
#![cfg(feature = "hip-tests")]
use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session, device_vram};
use golem_engine::slot::TurnRequest;

const TOL: i64 = 256 << 20;
const UBATCHES: [u32; 5] = [512, 1024, 2048, 4096, 8192];

fn cfg(model: &std::path::Path, ctx: u32, ub: u32) -> EngineConfig {
    EngineConfig {
        model_path: model.to_path_buf(),
        n_ctx: ctx,
        n_predict: 1,
        n_gpu_layers: 99,
        flash_attn_on: true,
        ram_experts_gib: 57,
        manifest_path: Some(model.with_file_name("manifest.json")),
        prefill_stream: true,
        prefill_ubatch: ub,
        ..EngineConfig::default()
    }
}

/// Predicted device bytes the process adds: every VRAM term except ROCm's
/// fixed overhead, which `hipMemGetInfo` before vs after does not see.
fn predicted(m: &golem_plan::ModelFacts, c: &EngineConfig, free: u64) -> (i64, bool) {
    let mach = golem_plan::Machine {
        vram_total: free,
        ram_total: u64::MAX,
        ram_free: u64::MAX,
        rocm_overhead: 0,
    };
    let i = golem_plan::PlanInput {
        ctx: c.n_ctx,
        parallel: 1,
        prefill_ubatch: c.prefill_ubatch.min(c.n_ctx),
        prefill_stream: true,
        vram_experts: 0,
        ram_experts: c.ram_experts_gib << 30,
        snapshot_ram: 0,
        ws: golem_plan::WS_UNMEASURED,
    };
    let r = golem_plan::plan(m, &mach, &i);
    (golem_plan::total(&r.vram) as i64, r.feasible)
}

#[test]
#[ignore = "real model; GOLEM_HW=1 GOLEM_HW_MODEL=<shard 1> --run-ignored"]
fn t8_plan_matches_hip_used_vram() {
    assert_eq!(
        std::env::var("GOLEM_HW").as_deref(),
        Ok("1"),
        "set GOLEM_HW=1"
    );
    let model: std::path::PathBuf = std::env::var("GOLEM_HW_MODEL")
        .expect("GOLEM_HW_MODEL must name shard 1")
        .into();
    let cap: usize = std::env::var("GOLEM_T8_MAX_POINTS")
        .ok()
        .map_or(60, |v| v.parse().expect("GOLEM_T8_MAX_POINTS: integer"));
    let m = golem_plan::ModelFacts::from_gguf(&model).expect("header");
    let (free0, _) = device_vram().expect("HIP VRAM query");
    // Largest ctx the plan admits at ubatch 512 (power of two), then halves.
    let top = (10..=20)
        .rev()
        .map(|p| 1u32 << p)
        .find(|&c| predicted(&m, &cfg(&model, c, 512), free0).1)
        .expect("no ctx fits");
    let points: Vec<(u32, u32)> = (0..4)
        .map(|h| top >> h)
        .flat_map(|c| {
            UBATCHES
                .into_iter()
                .filter(move |&u| u <= c)
                .map(move |u| (c, u))
        })
        .filter(|&(c, u)| predicted(&m, &cfg(&model, c, u), free0).1)
        .take(cap.min(60))
        .collect();
    println!(
        "T8 sweep: {} points (cap {cap}), top ctx {top}",
        points.len()
    );
    let prompt = "the quick brown fox jumps over the lazy dog ".repeat(240);
    let mut worst = 0i64;
    println!("ctx\tubatch\tpredicted\tactual\tdelta");
    for (ctx, ub) in points {
        let c = cfg(&model, ctx, ub);
        let (free_before, _) = device_vram().unwrap();
        let (pred, _) = predicted(&m, &c, free_before);
        let mut s = Session::new(&c).expect("load");
        let msgs = [ChatTurn {
            role: "user",
            content: &prompt,
        }];
        s.turn(&TurnRequest::new(&msgs), &mut |_| {})
            .expect("one prefill");
        let (free_after, _) = device_vram().unwrap();
        drop(s);
        let actual = free_before as i64 - free_after as i64;
        let d = pred - actual;
        worst = worst.max(d.abs());
        println!("{ctx}\t{ub}\t{pred}\t{actual}\t{d}");
    }
    assert!(
        worst <= TOL,
        "max |predicted − actual| = {worst} B > 256 MiB"
    );
}
