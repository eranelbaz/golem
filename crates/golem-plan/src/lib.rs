//! Offline VRAM/RAM/disk estimation shared by CLI and runtime (tech design §10).
//!
//! Every figure derives from the GGUF header; no model tables.

use std::fs::File;
use std::path::Path;

use golem_gguf::{
    KvVal, arch_kv_int, arch_kv_per_layer, blk_of, ggml_row_bytes, kv_int, live_layers, read_kvs,
    shard_path, tensor_entries,
};
use serde::Serialize;

/// Bytes per element of the default KV cache type (f16) and of the
/// recurrent-state types `type_r`/`type_s` (f32).
const KV_ELEM: u64 = 2;
const RS_ELEM: u64 = 4;

/// Loader + activations host overhead (§10, starts at 2 GiB until measured).
pub const HOST_OVERHEAD: u64 = 2 << 30;

/// A `ModelFacts` / plan failure.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// GGUF parse failure.
    #[error("{path}: {source}")]
    Gguf {
        /// Offending shard.
        path: std::path::PathBuf,
        /// Parse error.
        source: golem_gguf::Error,
    },
    /// Header is parseable but lacks or contradicts a needed fact.
    #[error("{0}")]
    Invalid(String),
}

/// Header-derived model facts the memory model needs (§10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ModelFacts {
    /// Non-expert weight bytes, shared experts included.
    pub t_ne: u64,
    /// Shared-expert bytes (a sub-line of `t_ne`, never added twice).
    pub shexp: u64,
    /// Routed experts per layer.
    pub n_expert: u32,
    /// Live layers carrying routed-expert planes.
    pub expert_layers: u32,
    /// Largest per-expert plane stride (bytes of one expert in one plane).
    pub max_plane_stride: u64,
    /// Σ routed-expert plane bytes over live layers.
    pub expert_bytes_total: u64,
    /// KV bytes per token per sequence, attention layers only.
    pub kv_per_token: u64,
    /// Recurrent-state bytes per sequence (constant in ctx).
    pub rs_bytes: u64,
    /// Live attention (non-recurrent) layers.
    pub sparse_attn_layers: u32,
}

impl ModelFacts {
    /// Reads every shard's tensor table through the one GGUF reader; experts
    /// are classified by the shared matcher ([`golem_gguf::TensorEntry::expert_kind`]).
    /// NextN/MTP blocks are skipped at load, so their tensors are not counted.
    ///
    /// # Errors
    /// Unreadable/corrupt shard, or missing hparams.
    pub fn from_gguf(path: &Path) -> Result<Self, PlanError> {
        let gguf = |p: &Path, e| PlanError::Gguf {
            path: p.to_path_buf(),
            source: e,
        };
        let open = |p: &Path| File::open(p).map_err(|e| gguf(p, e.into()));
        let mut f = open(path)?;
        let kvs = read_kvs(&mut f).map_err(|e| gguf(path, e))?;
        let live = live_layers(&kvs).map_err(|e| gguf(path, e))?;
        let count = kv_int(&kvs, "split.count").unwrap_or(1).max(1);

        let mut m = Self {
            t_ne: 0,
            shexp: 0,
            n_expert: hp_u32(&kvs, "expert_count")?,
            expert_layers: 0,
            max_plane_stride: 0,
            expert_bytes_total: 0,
            kv_per_token: 0,
            rs_bytes: 0,
            sparse_attn_layers: 0,
        };
        let mut expert_blk = vec![false; live.end as usize];
        for i in 0..count {
            let p = shard_path(path, i, count).map_err(|e| gguf(path, e.into()))?;
            for t in tensor_entries(&mut open(&p)?).map_err(|e| gguf(&p, e))? {
                let blk = blk_of(&t.name);
                if blk.is_some_and(|b| b >= live.end) {
                    continue;
                }
                let bytes = ggml_row_bytes(t.ggml_type, t.dims[0])
                    .and_then(|row| {
                        t.dims[1..]
                            .iter()
                            .try_fold(row, |a, &d| a.checked_mul(d.max(1)))
                    })
                    .ok_or_else(|| PlanError::Invalid(format!("{}: bad shape/type", t.name)))?;
                match (t.expert_kind(), blk) {
                    (Some(_), Some(b)) => {
                        expert_blk[b as usize] = true;
                        m.expert_bytes_total += bytes;
                        m.max_plane_stride =
                            m.max_plane_stride.max(bytes / u64::from(m.n_expert.max(1)));
                    }
                    _ => {
                        m.t_ne += bytes;
                        if t.name.contains("_shexp") {
                            m.shexp += bytes;
                        }
                    }
                }
            }
        }
        m.expert_layers = expert_blk.iter().filter(|&&b| b).count() as u32;
        m.state_terms(&kvs, live.end as usize)?;
        Ok(m)
    }

    /// KV per token and recurrent-state bytes, mirroring the vendored
    /// llama.cpp allocation: `head_count_kv == 0` marks a recurrent (KDA)
    /// layer (glm5next); MLA caches K only; a pooling indexer caches 3 heads.
    fn state_terms(&mut self, kvs: &[(String, KvVal)], n_layer: usize) -> Result<(), PlanError> {
        let heads_kv = arch_kv_per_layer(kvs, "attention.head_count_kv", n_layer)
            .ok_or_else(|| PlanError::Invalid("missing attention.head_count_kv".into()))?;
        let opt = |k: &str| arch_kv_int(kvs, k).unwrap_or(0);
        let k_len = u64::from(hp_u32(kvs, "attention.key_length")?);
        let v_len = if opt("attention.key_length_mla") > 0 {
            0
        } else {
            arch_kv_int(kvs, "attention.value_length").unwrap_or(k_len)
        };
        let idx_heads = if opt("attention.indexer.kpool") > 0 {
            3
        } else {
            1
        };
        let idx = idx_heads * opt("attention.indexer.key_length");
        let d_inner = opt("attention.head_count") * opt("kda.head_dim");
        let conv = 3 * opt("ssm.conv_kernel").saturating_sub(1) * d_inner;
        let ssm = opt("kda.head_dim") * d_inner;
        if d_inner == 0 && heads_kv.contains(&0) {
            return Err(PlanError::Invalid(
                "recurrent layers without kda.head_dim".into(),
            ));
        }
        for &h in &heads_kv {
            if h == 0 {
                self.rs_bytes += (conv + ssm) * RS_ELEM;
            } else {
                self.sparse_attn_layers += 1;
                self.kv_per_token += (h * (k_len + v_len) + idx) * KV_ELEM;
            }
        }
        Ok(())
    }
}

fn hp_u32(kvs: &[(String, KvVal)], suffix: &str) -> Result<u32, PlanError> {
    arch_kv_int(kvs, suffix)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| PlanError::Invalid(format!("missing or invalid {suffix}")))
}

/// The machine a plan is checked against.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Machine {
    /// Device VRAM.
    pub vram_total: u64,
    /// Physical RAM.
    pub ram_total: u64,
    /// Currently available RAM.
    pub ram_free: u64,
    /// Combined ROCm overhead (2.64 GiB measured, m0-notes).
    pub rocm_overhead: u64,
}

/// Prefill workspace model, fitted from measured points.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct WsModel {
    /// Fixed part (512 MiB until measured).
    pub base: u64,
    /// Bytes per ubatch token.
    pub per_ubatch_token: u64,
}

/// `ws` until the T8 hardware sweep fits it (§12, M4.6): the §10 512 MiB
/// starting point, slope **unmeasured** (0). Placeholder, not a measurement.
pub const WS_UNMEASURED: WsModel = WsModel {
    base: 512 << 20,
    per_ubatch_token: 0,
};

/// Combined ROCm overhead on gfx1100, measured in M0 (2.64 GiB, m0-notes).
pub const ROCM_OVERHEAD: u64 = 2_834_678_415;

/// Candidate `--prefill-ubatch` values, largest first.
const UBATCHES: [u32; 5] = [8192, 4096, 2048, 1024, 512];

/// Plan knobs; zero `ram_experts` means the §10 default pool.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct PlanInput {
    /// Context length per sequence.
    pub ctx: u32,
    /// Concurrent sequences.
    pub parallel: u32,
    /// Context ubatch tokens (the streamed prefill ubatch; sizes `ws`).
    pub prefill_ubatch: u32,
    /// Prefill layer streaming enabled (§7.6).
    pub prefill_stream: bool,
    /// Device expert-cache budget; zero = whatever VRAM the other terms
    /// leave (the runtime's derived K).
    pub vram_experts: u64,
    /// Host expert-pool budget.
    pub ram_experts: u64,
    /// State-snapshot cache budget (§7.1).
    pub snapshot_ram: u64,
    /// Workspace model.
    pub ws: WsModel,
}

/// One named line of a tier's budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Term {
    /// Stable term name (§10).
    pub name: &'static str,
    /// Bytes.
    pub bytes: u64,
}

/// A tier's sum; sub-lines (`parent.child`, e.g. `t_ne.shexp`) are already
/// inside their parent and never counted twice.
#[must_use]
pub fn total(terms: &[Term]) -> u64 {
    terms
        .iter()
        .filter(|t| !t.name.contains('.'))
        .map(|t| t.bytes)
        .sum()
}

/// The §10 verdict per tier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlanReport {
    /// VRAM terms (§10 order).
    pub vram: Vec<Term>,
    /// RAM terms (§10 order).
    pub ram: Vec<Term>,
    /// `vram_total − Σ vram`.
    pub vram_headroom: i64,
    /// `ram_free − Σ ram`.
    pub ram_headroom: i64,
    /// Both headrooms ≥ 0.
    pub feasible: bool,
    /// When VRAM is over: the largest candidate ubatch that fits, if any.
    pub largest_fitting_ubatch: Option<u32>,
}

impl PlanReport {
    /// The failing inequalities with actual numbers, e.g. `VRAM 25.1 GiB >
    /// 24.0 GiB; largest --prefill-ubatch that fits: 2048`; `None` when
    /// feasible.
    #[must_use]
    pub fn violation(&self) -> Option<String> {
        let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
        let cap = |t: &[Term], h: i64| total(t).saturating_add_signed(h);
        let (vram_cap, ram_cap) = (
            cap(&self.vram, self.vram_headroom),
            cap(&self.ram, self.ram_headroom),
        );
        let mut out = Vec::new();
        if self.vram_headroom < 0 {
            out.push(format!(
                "VRAM {:.1} GiB > {:.1} GiB",
                gib(total(&self.vram)),
                gib(vram_cap)
            ));
            if let Some(u) = self.largest_fitting_ubatch {
                out.push(format!("largest --prefill-ubatch that fits: {u}"));
            }
        }
        if self.ram_headroom < 0 {
            out.push(format!(
                "RAM {:.1} GiB > {:.1} GiB",
                gib(total(&self.ram)),
                gib(ram_cap)
            ));
        }
        (!out.is_empty()).then(|| out.join("; "))
    }

    /// The named term's bytes (0 when absent).
    #[must_use]
    pub fn term(&self, name: &str) -> u64 {
        self.vram
            .iter()
            .chain(&self.ram)
            .find(|t| t.name == name)
            .map_or(0, |t| t.bytes)
    }
}

fn headroom(cap: u64, used: u64) -> i64 {
    i64::try_from(cap)
        .unwrap_or(i64::MAX)
        .saturating_sub(i64::try_from(used).unwrap_or(i64::MAX))
}

/// VRAM tier (§10). Target at ctx 100k, parallel 1, ubatch 2048, unset
/// cache: 8.98 GiB t_ne + 1.84 GiB kv + 0.14 GiB rs + 0.5 GiB ws + 3.69 GiB
/// slabs + 2.64 GiB ROCm = 17.8 GiB → 6.2 GiB device expert cache.
#[must_use]
pub fn vram_terms(m: &ModelFacts, mach: &Machine, i: &PlanInput) -> Vec<Term> {
    let par = u64::from(i.parallel);
    let mut t = vec![
        Term {
            name: "t_ne",
            bytes: m.t_ne,
        },
        Term {
            name: "t_ne.shexp",
            bytes: m.shexp,
        },
        Term {
            name: "kv",
            bytes: m
                .kv_per_token
                .saturating_mul(u64::from(i.ctx))
                .saturating_mul(par),
        },
        Term {
            name: "recurrent_state",
            bytes: m.rs_bytes.saturating_mul(par),
        },
        Term {
            name: "ws",
            bytes: i.ws.base.saturating_add(
                i.ws.per_ubatch_token
                    .saturating_mul(u64::from(i.prefill_ubatch)),
            ),
        },
        Term {
            name: "stream_slabs",
            bytes: stream_bytes(m, i),
        },
        Term {
            name: "rocm_overhead",
            bytes: mach.rocm_overhead,
        },
    ];
    let cache = if i.vram_experts > 0 {
        i.vram_experts
    } else {
        mach.vram_total.saturating_sub(total(&t))
    };
    t.insert(
        5,
        Term {
            name: "device_expert_cache",
            bytes: cache,
        },
    );
    t
}

/// The §10 memory model: one implementation for `golem plan` and the
/// engine's load-time check (§6.4).
#[must_use]
pub fn plan(m: &ModelFacts, mach: &Machine, i: &PlanInput) -> PlanReport {
    let vram = vram_terms(m, mach, i);
    let ram = ram_terms(m, mach, i);
    let vram_headroom = headroom(mach.vram_total, total(&vram));
    let ram_headroom = headroom(mach.ram_free, total(&ram));
    let largest_fitting_ubatch = (vram_headroom < 0)
        .then(|| {
            UBATCHES.into_iter().find(|&u| {
                u < i.prefill_ubatch && {
                    let j = PlanInput {
                        prefill_ubatch: u,
                        ..*i
                    };
                    total(&vram_terms(m, mach, &j)) <= mach.vram_total
                }
            })
        })
        .flatten();
    PlanReport {
        feasible: vram_headroom >= 0 && ram_headroom >= 0,
        vram,
        ram,
        vram_headroom,
        ram_headroom,
        largest_fitting_ubatch,
    }
}

/// One state snapshot: recurrent state plus KV for `n_tokens` (§7.1).
#[must_use]
pub fn snapshot_bytes(rs_bytes: u64, kv_per_token: u64, n_tokens: u64) -> u64 {
    rs_bytes.saturating_add(kv_per_token.saturating_mul(n_tokens))
}

/// Snapshot arena `(floor, default)` (§7.1): floor = `parallel ×
/// snapshot_bytes(ctx)`, one full-context snapshot per slot; default =
/// `max(2 GiB, 2 × floor)`. Target: 145.56 MiB + 19,712 B × 100k ≈ 2.0 GiB
/// floor → 4.0 GiB default at parallel 1.
#[must_use]
pub fn snapshot_arena(m: &ModelFacts, parallel: u32, ctx: u32) -> (u64, u64) {
    let floor = u64::from(parallel).saturating_mul(snapshot_bytes(
        m.rs_bytes,
        m.kv_per_token,
        u64::from(ctx),
    ));
    (floor, (2u64 << 30).max(floor.saturating_mul(2)))
}

/// Double-buffered per-layer expert slabs for prefill streaming (§7.6).
/// Target: 2 × 288 × 6,881,280 = 3,963,617,280 B (3.69 GiB).
fn stream_bytes(m: &ModelFacts, i: &PlanInput) -> u64 {
    if i.prefill_stream {
        2 * u64::from(m.n_expert) * m.max_plane_stride
    } else {
        0
    }
}

/// RAM tier (§10). Default pool = min(0.75 · free, Σ expert bytes); target:
/// 0.75 × 88 GiB = 66 GiB < Σ experts, + 2 GiB host + 3.69 GiB staging ring.
#[must_use]
pub fn ram_terms(m: &ModelFacts, mach: &Machine, i: &PlanInput) -> Vec<Term> {
    let pool = if i.ram_experts > 0 {
        i.ram_experts
    } else {
        (mach.ram_free / 4 * 3).min(m.expert_bytes_total)
    };
    vec![
        Term {
            name: "ram_pool",
            bytes: pool,
        },
        Term {
            name: "host_overhead",
            bytes: HOST_OVERHEAD,
        },
        Term {
            name: "state_snapshots",
            bytes: i.snapshot_ram,
        },
        Term {
            name: "staging_ring",
            bytes: stream_bytes(m, i),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glm() -> ModelFacts {
        ModelFacts {
            t_ne: 8 << 30,
            shexp: 1 << 30,
            n_expert: 288,
            expert_layers: 42,
            max_plane_stride: 6_881_280,
            expert_bytes_total: 42 * 288 * 3 * 4_718_592,
            kv_per_token: 19_712,
            rs_bytes: 152_630_000,
            sparse_attn_layers: 11,
        }
    }
    fn box24() -> Machine {
        Machine {
            vram_total: 24 << 30,
            ram_total: 93 << 30,
            ram_free: 88 << 30,
            rocm_overhead: 2_834_678_415,
        }
    }
    fn get(t: &[Term], n: &str) -> u64 {
        t.iter().find(|t| t.name == n).unwrap().bytes
    }

    /// Real target: stream_slabs = 2 × 288 × 6,881,280 = 3,963,617,280 B;
    /// staging_ring the same; rs = 145.56 MiB.
    #[test]
    fn stream_slabs_match_design_7_6() {
        let r = plan(
            &glm(),
            &box24(),
            &PlanInput {
                prefill_stream: true,
                ..PlanInput::default()
            },
        );
        assert_eq!(get(&r.vram, "stream_slabs"), 3_963_617_280);
        assert_eq!(get(&r.ram, "staging_ring"), 3_963_617_280);
    }

    #[test]
    fn rs_and_kv_scale_with_parallel_and_ctx() {
        let a = plan(
            &glm(),
            &box24(),
            &PlanInput {
                ctx: 4096,
                parallel: 1,
                ..Default::default()
            },
        );
        let b = plan(
            &glm(),
            &box24(),
            &PlanInput {
                ctx: 4096,
                parallel: 2,
                ..Default::default()
            },
        );
        assert_eq!(
            get(&b.vram, "recurrent_state"),
            2 * get(&a.vram, "recurrent_state")
        );
        assert_eq!(get(&a.vram, "kv"), 19_712 * 4096);
    }

    /// Target at ubatch 8192 with 1 MiB/token ws: 8 + 8.5 + 6 + 3.69 + 2.64
    /// ≈ 28.8 GiB > 24 GiB; at 2048: ≈ 22.8 GiB fits, at 4096 ≈ 24.8 does not.
    #[test]
    fn oversized_ubatch_refused_with_largest_fitting_suggestion() {
        let ws = WsModel {
            base: 512 << 20,
            per_ubatch_token: 1 << 20,
        };
        let r = plan(
            &glm(),
            &box24(),
            &PlanInput {
                prefill_stream: true,
                prefill_ubatch: 8192,
                vram_experts: 6 << 30,
                ws,
                ..Default::default()
            },
        );
        assert!(!r.feasible);
        let v = r.violation().unwrap();
        assert!(v.starts_with("VRAM ") && v.contains("> 24.0 GiB"), "{v}");
        assert!(v.contains("largest --prefill-ubatch that fits: "), "{v}");
        let s = r.largest_fitting_ubatch.unwrap();
        assert!([512, 1024, 2048, 4096].contains(&s));
        assert!(
            plan(
                &glm(),
                &box24(),
                &PlanInput {
                    prefill_ubatch: s,
                    prefill_stream: true,
                    vram_experts: 6 << 30,
                    ws,
                    ..Default::default()
                }
            )
            .feasible
        );
    }

    #[test]
    fn unset_device_cache_takes_the_remaining_vram() {
        let i = PlanInput {
            ctx: 4096,
            parallel: 1,
            ws: WS_UNMEASURED,
            ..Default::default()
        };
        let r = plan(&glm(), &box24(), &i);
        assert!(r.feasible);
        assert_eq!(r.vram_headroom, 0);
        let set = plan(
            &glm(),
            &box24(),
            &PlanInput {
                vram_experts: 30 << 30,
                ..i
            },
        );
        assert!(!set.feasible);
        assert!(set.vram_headroom < 0);
    }

    #[test]
    fn staging_ring_matches_design_7_6() {
        let i = PlanInput {
            prefill_stream: true,
            ..PlanInput::default()
        };
        assert_eq!(
            get(&ram_terms(&glm(), &box24(), &i), "staging_ring"),
            3_963_617_280
        );
        let off = ram_terms(&glm(), &box24(), &PlanInput::default());
        assert_eq!(get(&off, "staging_ring"), 0);
    }

    #[test]
    fn ram_pool_defaults_to_three_quarters_free_capped_by_experts() {
        let t = ram_terms(&glm(), &box24(), &PlanInput::default());
        // 0.75 × 88 GiB = 66 GiB < 42 × 288 × 3 × 4.5 MiB ≈ 159.5 GiB.
        assert_eq!(get(&t, "ram_pool"), 66 << 30);
        assert_eq!(get(&t, "host_overhead"), 2 << 30);
        let small = ModelFacts {
            expert_bytes_total: 1 << 30,
            ..glm()
        };
        assert_eq!(
            get(
                &ram_terms(&small, &box24(), &PlanInput::default()),
                "ram_pool"
            ),
            1 << 30
        );
        let set = PlanInput {
            ram_experts: 40 << 30,
            snapshot_ram: 3 << 30,
            ..Default::default()
        };
        let t = ram_terms(&glm(), &box24(), &set);
        assert_eq!(get(&t, "ram_pool"), 40 << 30);
        assert_eq!(get(&t, "state_snapshots"), 3 << 30);
    }

    fn kvs(extra: &[(&str, KvVal)]) -> Vec<(String, KvVal)> {
        let mut v = vec![("general.architecture".to_string(), KvVal::Str("a".into()))];
        v.extend(extra.iter().map(|(k, x)| (format!("a.{k}"), x.clone())));
        v
    }
    fn facts(kvs: &[(String, KvVal)], n: usize) -> Result<ModelFacts, PlanError> {
        let mut m = ModelFacts {
            kv_per_token: 0,
            rs_bytes: 0,
            sparse_attn_layers: 0,
            ..glm()
        };
        m.state_terms(kvs, n).map(|()| m)
    }

    #[test]
    fn recurrent_layers_need_kda_dims() {
        let k = kvs(&[
            ("attention.head_count_kv", KvVal::IntArr(vec![0, 4])),
            ("attention.key_length", KvVal::Int(128)),
        ]);
        assert!(matches!(facts(&k, 2), Err(PlanError::Invalid(_))));
    }

    #[test]
    fn value_length_used_as_is_when_present() {
        let k = kvs(&[
            ("attention.head_count_kv", KvVal::Int(1)),
            ("attention.key_length", KvVal::Int(192)),
            ("attention.value_length", KvVal::Int(128)),
        ]);
        // 1 head × (192 + 128) × 2 B = 640 B/token.
        assert_eq!(facts(&k, 1).unwrap().kv_per_token, 640);
        let k = kvs(&[
            ("attention.head_count_kv", KvVal::Int(1)),
            ("attention.key_length", KvVal::Int(192)),
        ]);
        assert_eq!(facts(&k, 1).unwrap().kv_per_token, 768);
    }

    #[test]
    fn snapshot_is_state_plus_kv_prefix() {
        // Target 4096-token snapshot: 152,630,000 + 19,712 × 4096 = 233,370,352 B.
        assert_eq!(snapshot_bytes(152_630_000, 19_712, 4096), 233_370_352);
        assert_eq!(snapshot_bytes(0, 0, u64::MAX), 0);
        assert_eq!(snapshot_bytes(1, u64::MAX, 2), u64::MAX);
    }
}
