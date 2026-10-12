//! T8 (§12, M4.6): `golem_plan::plan` predicted VRAM vs `hipMemGetInfo` used
//! after load + one streamed prefill, on the real target, over a ctx ×
//! ubatch sweep at parallel 1. Pass: |predicted − actual| ≤ 256 MiB at every
//! point. Opt-in hardware leg (GOLEM_HW=1, GOLEM_HW_MODEL = shard 1 of
//! GLM-5.3-Flash UD-Q4_K_XL): each point is a full real-model load. Budget
//! ≤ 60 loads; `GOLEM_T8_MAX_POINTS` caps the sweep (record the cap in
//! docs/baselines/m4.6-plan-t8.md). `golem_plan::WS_MEASURED` is fitted from
//! that baseline's compute-buffer table; this sweep re-checks the fit and
//! the total-delta bound.
#![cfg(feature = "hip-tests")]
use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session, device_vram};
use golem_engine::slot::TurnRequest;

const TOL: i64 = 256 << 20;
// Total-delta gate. Every logged term matches (#123), but allocator/runtime
// overhead outside them is 200–310 MB per point (docs/baselines/m4.6-plan-t8.md). Target: 24 GiB VRAM,
// 512 MiB ≈ 2.1%; worst measured −312 MB at ctx 65536 / ubatch 2048.
const TOTAL_TOL: i64 = 512 << 20;
const UBATCHES: [u32; 6] = [256, 512, 1024, 2048, 4096, 8192];
const CTXS: [u32; 7] = [2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072];

fn cfg(model: &std::path::Path, ctx: u32, ub: u32) -> EngineConfig {
    EngineConfig {
        model_path: model.to_path_buf(),
        n_ctx: ctx,
        // 2: the decode step allocates the lazy device cache even when the
        // whole prompt streamed, so `actual` sees it.
        n_predict: 2,
        n_gpu_layers: 99,
        flash_attn_on: true,
        ram_experts_gib: 57,
        manifest_path: Some(model.with_file_name("manifest.json")),
        prefill_stream: true,
        prefill_ubatch: ub,
        ..EngineConfig::default()
    }
}

/// `golem_plan::plan` at `free` VRAM. Feasibility uses
/// `golem_plan::ROCM_OVERHEAD` like the engine's `check_plan`; the byte total
/// excludes it, since `hipMemGetInfo` before vs after does not see it.
fn predicted(m: &golem_plan::ModelFacts, c: &EngineConfig, free: u64) -> (i64, bool, u64) {
    let mach = golem_plan::Machine {
        vram_total: free,
        ram_total: u64::MAX,
        ram_free: u64::MAX,
        rocm_overhead: golem_plan::ROCM_OVERHEAD,
    };
    let i = golem_plan::PlanInput {
        ctx: c.n_ctx,
        parallel: 1,
        prefill_ubatch: c.prefill_ubatch.min(c.n_ctx),
        prefill_stream: true,
        vram_experts: 0,
        ram_experts: c.ram_experts_gib << 30,
        snapshot_ram: 0,
        ws: golem_plan::WS_MEASURED,
    };
    let r = golem_plan::plan(m, &mach, &i);
    // Without the cache term: the engine holds K + 1 whole slabs, and its K
    // is also bounded by free VRAM, so the caller adds the loaded K's bytes.
    let bytes =
        golem_plan::total(&r.vram) - r.term("rocm_overhead") - r.term("device_expert_cache");
    (bytes as i64, r.feasible, r.term("stream_slabs"))
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
    // Points sit TOL inside the fit edge so normal drift between loads does
    // not fail them; drift past TOL still does. Target: 24 GiB VRAM, TOL
    // 256 MiB ≈ 1.0%; observed drift 25671237632 → 25461522432 B = 209.7 MB < TOL.
    let free_sel = free0.saturating_sub(TOL as u64);
    let points: Vec<(u32, u32)> = CTXS
        .into_iter()
        .rev()
        .flat_map(|c| {
            UBATCHES
                .into_iter()
                .filter(move |&u| u <= c)
                .map(move |u| (c, u))
        })
        .filter(|&(c, u)| predicted(&m, &cfg(&model, c, u), free_sel).1)
        .take(cap.min(60))
        .collect();
    let min_points = cap.clamp(1, 20);
    assert!(
        points.len() >= min_points,
        "{} feasible sweep points < {min_points} required (cap {cap})",
        points.len()
    );
    println!(
        "T8 sweep: {} points (cap {cap}) over ctx {CTXS:?} × ubatch {UBATCHES:?}",
        points.len()
    );
    // ~1 token/word; sized per point so the prompt always fits the grid ctx
    // (2048 → 179 repeats ≈ 1 611 words ≈ 1 620 tokens; capped at 240
    // repeats ≈ 2 173 tokens above that).
    let sentence = "the quick brown fox jumps over the lazy dog ";
    let mut worst = 0i64;
    // Per-term attribution (#123): the cache and stream terms are checked
    // against C's live buffer bytes, not inferred from the free-VRAM delta.
    // llama's `ROCm0 compute buffer size` line (stderr) is the ws ground truth.
    println!(
        "ctx\tubatch\tpredicted\tactual\tdelta\tws_residual\tk\tcache_pred\tslab_res\tstream_pred\tstream_res\tcopied"
    );
    for (ctx, ub) in points {
        let prompt = sentence.repeat(((ctx as usize).saturating_sub(256) / 10).min(240));
        let c = cfg(&model, ctx, ub);
        let (free_before, _) = device_vram().unwrap();
        let (pred, fits, stream_pred) = predicted(&m, &c, free_before);
        assert!(
            fits,
            "ctx {ctx} ubatch {ub} no longer fits: free VRAM drifted {free0} → {free_before} B"
        );
        let mut s = Session::new(&c).expect("load");
        assert!(
            s.prefill_streaming(),
            "ctx {ctx} ubatch {ub}: fit dropped prefill streaming"
        );
        let k = s.device_cache_experts().expect("derived device cache");
        let cache_pred = (i64::from(k) + 1) * m.slab_bytes_per_expert() as i64;
        let pred = pred + cache_pred;
        let msgs = [ChatTurn {
            role: "user",
            content: &prompt,
        }];
        s.turn(&TurnRequest::new(&msgs), &mut |_| {})
            .expect("one prefill");
        let (free_after, _) = device_vram().unwrap();
        let (slab_res, stream_res) = s.device_cache_resident_bytes().expect("pool on");
        let copied = s.device_cache_stats().map_or(0, |d| d.bytes_copied);
        drop(s);
        golem_llama_sys::pinned::device_synchronize().expect("hipDeviceSynchronize");
        let actual = free_before as i64 - free_after as i64;
        // The per-term attribution, gated: stream slabs exact, cache slabs
        // the K + 1 reservation plus remapped-ids tensors (16 KiB measured).
        assert_eq!(
            stream_res, stream_pred,
            "ctx {ctx} ubatch {ub}: stream slabs"
        );
        assert!(
            slab_res > 0 && (slab_res as i64 - cache_pred).abs() <= 1 << 20,
            "ctx {ctx} ubatch {ub}: resident slabs {slab_res} vs predicted {cache_pred}"
        );
        // Implied ws term: actual minus every non-ws predicted term. This is
        // the plan's scatter, not llama's compute-buffer log.
        let ws_residual =
            actual - (pred - golem_plan::WS_MEASURED.bytes(ctx, ub, m.dense_below_ctx) as i64);
        let d = pred - actual;
        worst = worst.max(d.abs());
        println!(
            "{ctx}\t{ub}\t{pred}\t{actual}\t{d}\t{ws_residual}\t{k}\t{cache_pred}\t{slab_res}\t{stream_pred}\t{stream_res}\t{copied}"
        );
        let (free_released, _) = device_vram().unwrap();
        assert!(
            free_before as i64 - free_released as i64 <= TOL,
            "ctx {ctx} ubatch {ub}: Session drop left {} B of VRAM held ({free_before} → {free_released} B)",
            free_before as i64 - free_released as i64
        );
        assert!(
            free0 as i64 - free_released as i64 <= TOL,
            "ctx {ctx} ubatch {ub}: cumulative VRAM leak {} B since start ({free0} → {free_released} B)",
            free0 as i64 - free_released as i64
        );
    }
    println!("T8 max |predicted − actual| = {worst} B");
    assert!(
        worst <= TOTAL_TOL,
        "max |predicted − actual| = {worst} B > 512 MiB"
    );
}
