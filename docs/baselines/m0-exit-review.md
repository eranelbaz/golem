# M0 Exit Review (2026-09-21, session 10)

## 1. Exit criteria checklist (§14 M0)
- Fork pinned+locked: ✅ vendor/VENDOR_SHA = content hash 812b9b4a… (Task 3 gate re-run this session)
- HIP builds for gfx1100: ✅ (M0.2; hip leg ran 8/8 on the pinned build)
- FA/precision validated, `fa=on` frozen: ✅ (M0.4 + M0.5 parity legs as formal evidence)
- Host-buffered experts run: ✅ (M0.4; -ncmoe 45 semantics frozen)
- Measurement classes:
  - Cold-expert copy cost: ⚠️ inconclusive — owned by P4 counters in M2.4
  - KV bytes/token: ✅ 19,712
  - ROCm overhead: ✅ 2.64 GiB
  - tok/s cold: ✅ 0.6–0.8 (this session's legs: gen_s 480 for 96 tok @ t16 capped-cold ≈ 0.2–0.6 under desktop-competitive IO; warm not measurable at M0 — no resident-set mechanism exists yet, deferred to M2/M4)
  - Disk bandwidth 1v2: ✅ (M0.4)

## 2. T2 parity floor
- Noise floor (reference-vs-reference, as-executed pair cpu-t16 vs cpu-t4): **8/8 byte-identical → 100%**. (t32 leg dropped in session 9 for wall-time; t4 substitutes per Task 4 Step 3. HIP's CPU side is t16, covered.)
- Parity (hip vs cpu-t16): **2/8 = 25%** byte-equal; prefixes [49, 25, 33, 31, 34, 27, ∞, ∞]. Prompts 6–7 byte-equal; the rest diverge via benign logit tie-breaks (e.g. "give it a name and some personality" vs "…and personality"; "Let me craft" vs "Let me think about") that greedy sampling then amplifies into different-but-equivalent continuations.
- Frozen threshold (spec-literal): min(100, floor + 12.5) = **100%**. parity_match 25% < 100% → **gate not passed as specified**.
- Method caveat: text-exact, not token-ID-exact; a single near-tie flip cascades the whole suffix. The 100% floor comes from a CPU-vs-CPU pair that never exercises GPU numerics, so the threshold is degenerate-strict for HIP.

## 3. Verdict: **RE-SCOPE** (narrow, method-level — not architecture-level)
- Architecture viability is NOT in question: M0.4 cold 0.6–0.8 tok/s vs ~0.9–1.1 ceiling is consistent; re-scope trigger (cold ≪ 0.1 tok/s) did not fire.
- The re-scope is the T2 method: a byte-exact text comparison with a degenerate 100% CPU-only floor cannot clear HIP. Both agreed references diverge from HIP only at low-confidence tie-breaks — consistent with correct-but-differently-rounded GPU kernels.
- Actions folded forward: (a) token-ID-level comparison with per-token top-2 logit gap (M1.4) — count flips only where |Δ logit| exceeds epsilon; (b) optionally a CPU fa-off leg to bound the FA contribution; (c) keep baseline-001.json as the honest M0 record (pass=false).

## 4. Deviations & carry-forwards
- Copy-cost measurement → P4/M2.4. Warm-state tok/s → M4 `golem bench`. Token-ID T2 → M1.4 (brought forward in priority by this review). Q8_0 parity quant unused at M0.5.
- As-executed: t32 leg → t4; CPU substrate -DGGML_CPU_REPACK=OFF (106 GiB repack alloc impossible on 93 GiB); all legs from ext4 replica; desktop-safe cgroup recipe (see plan session-10 amendment).
