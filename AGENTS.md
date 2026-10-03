# AGENTS.md — rules for agents working on Golem

Process (plan-first, one task = one PR, worklog upkeep) lives in `docs/worklog.md` §0 — follow it; this file adds engineering rules. Open review findings: `docs/opus-review.md`.

## Evidence

- A test that cannot run its precondition **fails**; it never `return`s early as a pass. Opt-in skips only via `#[ignore]` + an explicit env flag, and they are reported as skipped, not ok.
- GPU-path (P1–P4) changes need pasted `GOLEM_LLAMA_SYS_BACKEND=hip cargo nextest run -p golem-engine --features hip-tests` output with zero `SKIPPED` lines, through the **engine-registered** hooks (not test-installed ones).
- Every design deviation or memory/VRAM decision gets one line of arithmetic against the real target (GLM-5.3-Flash UD-Q4_K_XL: 42 live MoE layers (+1 MTP layer skipped at load) × 288 experts × 3 planes, hybrid linear/sparse attention with recurrent state, ~4.7/5.8 MB per plane, 6 split shards with a metadata-only shard 1, 24 GiB VRAM, 93 GiB RAM). tiny-moe passing proves nothing about scale.
- Fixtures must match real shape where it matters: `.weight` tensor names, split GGUF, quantized mixed-stride planes, device budget < one expert tensor.
- Keep `docs/worklog.md` board status in sync with merged PRs in the same PR that merges.

## Hot path (decode step)

- No heap allocation, `HashMap` lookup, sort, or mutex per routed expert in P1–P4 callbacks; preallocate per-tensor scratch, index flat `Vec`s by `(layer, kind)`, batch notifications per tensor.
- Expert bytes are copied at most once disk→pool (read directly into the slot, no bounce `Vec`) and once pool→device (from pinned memory). Any extra copy needs a benchmark justifying it.
- Pool memory is mlocked **and** GPU-registered (`hipHostRegister`); pageable H2D is a bug.
- Nothing may populate the page cache with expert bytes: no loader `MAP_POPULATE` of expert tensors, `O_DIRECT` where the filesystem supports it, else `POSIX_FADV_DONTNEED` after reads.
- Every expert `MUL_MAT_ID` runs on the GPU backend, verified at runtime; the engine (not tests) sets whatever makes that true. Silent CPU expert math is a correctness bug (spec, design I2).
- Performance claims cite a measured number (tok/s, GB/s, hit rate) from this machine, cold and warm separately.

## Rust

- `unsafe` stays confined to the FFI/arena modules, each block with a `// SAFETY:` naming the invariant. A safe `pub fn` must not be able to cause UB: aliasing contracts are enforced by types (lease/claim tokens), not doc comments. No `&self → &mut` accessors in public API.
- `unsafe impl Send/Sync` only on the narrowest wrapper around the raw pointer, never on a whole aggregate.
- One implementation per concern: one GGUF reader (`golem-gguf`), one expert-tensor name matcher shared by C and Rust semantics (exact stem, `.weight` suffix), one expert index. Use workspace deps (`sha2`, `thiserror`) instead of hand-rolling.
- Errors: libraries return `Result`; `panic!`/`process::exit` only in binaries' `main` or FFI-abort paths explicitly documented as such.
- Comments state invariants and design §refs, not plan-task jargon ("Task 4a", "deviation 2", "Global Constraints") — plans are local-only and comments outlive them.
- Gates: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo nextest run --workspace`, loom leg for `golem-placement::pool` concurrency changes.

## Vendored C++ (`vendor/llama.cpp`)

- Patches stay minimal and null-guarded; any change rehashes `vendor/VENDOR_SHA` + `engine/llama-cpp.lock` in the same commit.
- Device allocations are bounded by the configured VRAM budget — never proportional to total expert count.
- The P1 ABI is versioned, not frozen: change it when the design needs it, update bindings and both sides in one PR.
