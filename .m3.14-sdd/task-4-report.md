# Task 4 report — baseline doc

Status: DONE_WITH_CONCERNS
Commit: 23e9cd449a91f54976a9e5291aec55d15498873e
`docs/baselines/m3.14-pool-size-baseline.md` (189 lines, only file changed).

## What was written
- Config block: machine (24 GiB VRAM / 93 GiB RAM / omarchy / two NVMe replicas), model + manifest paths, commit 612d86bc, exact flags line, prompt text, expert arithmetic (12,096 × 15.34 MB = 172.8 GiB, 15.2 MB rounding noted), cold/warm definitions, pool-warm non-measurement, asc/desc single-run disclosure.
- Per-leg table (12 legs + probe + smoke) from scripts/pool-size-report.py output, compute column labeled "compute + H2D (H2D unmeasured)", >100% disk-ms footnote.
- Per-pool means + monotone scaling result (cold tok/s 0.633→1.430→2.580; disk share 75.1%→33.6%→16.6%).
- §2.4 comparison: full 14-row table with per-row arithmetic, all legs flagged (+27% to +103%), byte-ratio expectations 5.8/23.1/36.5%, miss analysis without fabricated split (H2D inside compute column; disk ms share >100% shows serialized wait, not raw bandwidth, dominates).
- Preflight arithmetic: 63 GiB needs ≥84 GiB available, guard fired twice at 83 GiB, drop_caches legs listed (all 63-GiB legs + probe); 70 GiB needs 93.4 GiB > 93 GiB physical ⇒ not startable; PSI/vmstat evidence (avg10 0.00, si 0, MemFree 90.3→8.3 GB), ~17 GiB headroom arithmetic, ceiling-raise recommendation only.
- H2D unmeasured stated with fork-event-pair reason (timing.rs:20-24).
- Incident disclosure (first asc-10-cold GPU hang, wedged-state artifact).
- Precedent mismatches (m3.5 turn-2 band; warm-cold inversions within variance).

## Gates
- py_compile pool-size-report.py: pass; bash -n pool-size-bench.sh: pass.
- cargo fmt --check: pass; cargo clippy --workspace --all-targets -D warnings: clean.
- Every number carries one line of arithmetic; no unmeasured splits claimed.

## Concerns
1. PSI probe covers pool fill only — decode-phase samples missing (sampling loop ended early); disclosed in §5 of the doc.
2. Single-run legs; desc warm-vs-cold inversions unresolved (disclosed as variance, no repeats run).
3. compute_ms contains H2D time until the fork event pair lands; all split analysis is bounded by that.

## Fix round 1 (reviewer findings applied)

1. (Medium) §1 disk-share corrected to per-token expert traffic basis:
   `disk GB/token ÷ (336 × 15.34 MB ≈ 5.154 GB) × 100`; example line now
   shows 4.044/5.154 ≈ 78.5%, matching the table's 79.2% to rounding of the
   GB/tok cell. Old formula (÷172.8 GiB total) removed.
2. (Low) Chose exclusion: 10 GiB warm mean is now 2-leg (0.715, disk share
   76.6%), smoke-10-warm excluded; asymmetry and the would-be 3-leg values
   (0.755 / 75.1) disclosed in §2. Scaling sentence updated to
   76.6% → 33.6% → 16.6%.
3. (Low) §5 preflight-guard firings and §6 GPU-hang rewrote to past tense
   with evidence caveats: failed attempt's stdout.log overwritten by the
   clean rerun (hang text from coordinator session record); guard firings
   not persisted in any artifact.

Commits: 8b56b3d (doc fixes), ff963f1 (compgen glob quoting fix in
pool-size-bench.sh line 31, the fix all 12 legs actually ran with).
Gates: `python3 -m py_compile scripts/pool-size-report.py` pass;
`bash -n scripts/pool-size-bench.sh` pass.
