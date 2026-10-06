# AGENTS.md — engineering rules for Golem

## Evidence

- A test that cannot meet its precondition **fails**. Opt-in skips use `#[ignore]` plus an explicit env flag and are reported as skipped.
- GPU-path changes (substrate hooks, pool, device cache) need HIP test output with zero `SKIPPED` lines, run through the engine-registered hooks, not test-installed ones.
- Every memory, VRAM or throughput decision carries one line of arithmetic against the real target, not just the tiny fixture:
  - Model: GLM-5.3-Flash UD-Q4_K_XL, 42 MoE layers × 288 experts × 3 planes (~4.7/5.8 MB per plane), hybrid linear/sparse attention with recurrent state, 6 split shards.
  - Machine: 24 GiB VRAM, 93 GiB RAM.
- Fixtures match the real shape where it matters: `.weight` tensor names, split GGUF, quantized mixed-stride planes, a device budget smaller than one expert tensor.
- Performance claims cite a number measured on this machine, cold and warm separately, A/B in both orders.

## Decode hot path

- No heap allocation, `HashMap` lookup, sort or mutex per routed expert. Preallocate scratch, index flat `Vec`s by `(layer, kind)`, batch per tensor.
- Expert bytes are copied at most once disk → pool (straight into the slot) and once pool → device (from pinned memory). Any extra copy needs a benchmark.
- Pool memory is mlocked **and** GPU-registered; pageable H2D is a bug.
- Expert bytes never enter the page cache: no `MAP_POPULATE` of expert tensors. Use `O_DIRECT`, else `POSIX_FADV_DONTNEED` after the read.
- Every expert `MUL_MAT_ID` runs on the GPU and is verified at runtime by the engine. Silent CPU expert math is a correctness bug.

## Rust

- `unsafe` stays in the FFI and arena modules, with a `// SAFETY:` per block naming the invariant.
- A safe `pub fn` cannot cause UB: aliasing is enforced by types (lease/claim tokens), never by doc comments. No `&self → &mut` accessors in public API.
- `unsafe impl Send/Sync` only on the narrowest raw-pointer wrapper.
- One implementation per concern: one GGUF reader, one expert-tensor name matcher shared by C and Rust, one expert index. Use workspace deps (`sha2`, `thiserror`) instead of hand-rolling.
- Libraries return `Result`. `panic!` and `process::exit` only in a binary's `main` or a documented FFI-abort path.
- Comments state invariants and design § refs, never task or plan jargon.
- Gates: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo nextest run --workspace`, plus the loom leg for pool concurrency changes.

## Vendored C++

- Patches stay minimal and null-guarded. Any change rehashes the vendor pin in the same commit.
- Device allocations are bounded by the configured VRAM budget, never proportional to the total expert count.
- The hook ABI is versioned, not frozen: change it when the design needs it, and update the bindings and both sides together.
