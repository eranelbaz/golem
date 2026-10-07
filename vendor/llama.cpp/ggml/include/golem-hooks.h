// Golem policy-hook ABI + expert-metadata queries (Golem patch P1, ABI v5:
// per-model hook context). Part of the vendored ggml include tree; the ABI is
// versioned — a change updates the Rust bindings and both sides together.
// Hook STATE lives in a per-model `golem_ctx` (created by the engine, attached
// to llama_model / scheds); only process-global capability flags remain here.
#ifndef GOLEM_HOOKS_H
#define GOLEM_HOOKS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend_sched;
struct golem_ctx;
/* Opaque per-model hook context (state defined cpp-side in
 * ggml/src/golem-hook-context.hpp; shared by ggml-backend.cpp and the vendor
 * ctest TUs via include, never copied). */
typedef struct golem_ctx *golem_ctx_t;
/* Handle returned by golem_register_hooks; consumed by golem_unregister_hooks. */
typedef struct golem_hook_handle *golem_hook_handle_t;

/* Hook ABI revision: bump on any change to the hook signatures or to the
 * golem_* queries; the Rust bindings assert their header constant equals the
 * compiled library's golem_hooks_abi_version(). */
#define GOLEM_HOOKS_ABI_VERSION 6
int golem_hooks_abi_version(void);

/* Hook: ensure the listed expert-plane misses are resident in the host expert
 * pool before the substrate reads them (blocks; called on the decode thread).
 * ONE call per (layer, step, split-segment): miss_kinds[i] (1=gate 2=up
 * 3=down) / miss_ids[i] describe the i-th miss across ALL planes of the
 * layer; fills out_ptrs[i] with the pool-side host pointer of that plane's
 * expert — the RAM pool is the source of the host→device copy, because the
 * loader's expert bytes may live in a read-only file mmap. The pool holds a
 * lease on each returned slot until the matching copy_done(payload) call, so
 * eviction cannot recycle a slot mid-DMA. `tensor` is the first expert input
 * of the layer that step (identity only for logging). */
typedef void (*golem_ensure_host_resident_fn)(void *payload, const void *tensor,
                                              const int32_t *miss_kinds,
                                              const int32_t *miss_ids,
                                              int32_t n_miss,
                                              const void **out_ptrs);

/* Hook: releases the host-pool leases taken by ensure_host_resident since
 * the previous copy_done. Called at the start of each scheduler compute,
 * after the scheduler has synchronized its backends, so the previous
 * compute's DMAs are complete and the slots are safe to recycle. */
typedef void (*golem_copy_done_fn)(void *payload);

/* Hook: merged per-step routing readback. Fired ONCE per changed ids tensor
 * per step with the DENSELY gathered ids (n_ids = ne0 × ne1; strides are not
 * passed). `blk` is the layer number (golem_tensor_blk of the expert tensor:
 * fused gate_up layouts have no separate gate tensor, and gate/up/down share
 * ids). Also replaces the per-expert notify_access fan-out: score decay,
 * access atomics and dirty-order marking happen inside this single call.
 * Layers whose experts are fully device-resident take no ids readback and
 * produce no routing call — documented coverage gap (those layers evict
 * nothing). */
typedef void (*golem_routing_fn)(void *payload, int32_t blk, const void *tensor,
                                 const int32_t *ids_dense, int64_t n_ids,
                                 int32_t n_experts);

/* Hook (P5, design §6.5): next-layer routing prediction. When
 * golem_set_predict enabled it, the model graph builds, from layer `src_blk`'s
 * FFN input, the top (n_expert_used + margin) experts layer `target_blk`'s
 * router would pick (target = src + 1 .. src + lookahead). The scheduler
 * reads them right after src_blk's routing readback (the device is already
 * synchronized there) and passes them here, decode ubatches (1 token) only.
 * Real routing is untouched (I1): predictions only steer staging. */
typedef void (*golem_predict_fn)(void *payload, int32_t src_blk, int32_t target_blk,
                                 const int32_t *ids, int32_t n_ids);

/* --- Context lifecycle. --- */

/* Create a per-model hook context. `n_layers` sizes the flat eviction-order
 * table (indexed blk × kind; aborts later if a blk >= n_layers reaches the
 * order path); `n_experts` sizes the per-(blk,kind) triple buffers and the
 * slab scratch. Order buffers and scratch are preallocated here: a decode
 * step performs no heap allocation on the ctx path. */
golem_ctx_t golem_ctx_create(int32_t n_layers, int32_t n_experts);

/* Free a context. Aborts while a sched is attached (attach refcount != 0 —
 * the sched destructor or llama_golem_context_detach must run first) and
 * while hooks are still registered. */
void golem_ctx_free(golem_ctx_t ctx);

/* Register the hooks on `ctx`. `payload` is passed back as the first
 * argument of every hook invocation. Not synchronized with a running step: call
 * before any scheduler compute starts and not concurrently with one (the
 * engine registers at pool enable, before the first decode). Passing NULL
 * for a hook restores upstream behavior for that concern (unset hooks ==
 * byte-identical to upstream). Returns a handle that must be unregistered
 * with golem_unregister_hooks before golem_ctx_free. */
golem_hook_handle_t golem_register_hooks(golem_ctx_t ctx,
                                         golem_ensure_host_resident_fn ensure,
                                         golem_routing_fn routing,
                                         golem_copy_done_fn copy_done,
                                         void *payload);

/* Marks the ctx unhooked, then JOINS: waits until the ctx's in-flight step
 * count reaches 0, so no callback can touch the engine payload after this
 * returns. Aborts if called from inside a hook callback (deadlock by
 * construction — the join would wait for the calling step). */
void golem_unregister_hooks(golem_hook_handle_t handle);

/* Monotonic per-ctx counter, bumped on every register: the engine tags
 * scratch/pool state with it to reject addresses recycled across sessions. */
uint64_t golem_ctx_generation(golem_ctx_t ctx);

/* Process-global capability flag: number of hooked ctxs (> 0 = some model
 * has hooks). Consumed ONLY by golem_graphs_disabled(); per-sched decisions
 * (e.g. the cpy shrink) read the sched's OWN ctx, so a hookless sched under a
 * hooked process takes the upstream path verbatim. Documented deviation:
 * while ANY ctx is hooked, HIP graph capture is disabled process-wide. */
int golem_any_ctx_hooked(void);

/* --- Eviction order (design §3.4). --- */

/* Read the current eviction order for the (blk, kind) of `tensor`: a
 * permutation of [0, n_experts) in evict-first order, or NULL before the
 * first push (callers fall back to identity order + a one-time warning).
 * Triple-buffered: the pointer stays valid for the duration of the step that
 * acquired it (the publisher skips a buffer instead of overwriting one a
 * reader of the current epoch may hold). Flat per-ctx table indexed
 * blk × kind — no hash lookup. C never frees it and never blocks. */
const int32_t *golem_policy_order(golem_ctx_t ctx, const void *tensor,
                                  int32_t n_experts);

/* Publish a fresh eviction order for (blk, kind): copies `n` entries into a
 * free triple-buffer slot and swaps it in (release). Skips (returns 0)
 * without blocking when no buffer is free under the reader-epoch gate; the
 * publisher retries next step. `kind` is 1-based (1=gate 2=up 3=down).
 * Called off the decode thread — the only synchronized ctx entry point. */
int golem_policy_order_push(golem_ctx_t ctx, int32_t blk, int32_t kind,
                            const int32_t *order, int32_t n);

/* Advance the order epoch (one per decode step): releases the triple-buffer
 * slot the previous step's readers hold. The scheduler calls this at graph
 * compute entry; exposed so tests (and embedders driving steps by hand) can
 * match publisher progress without running a step. */
void golem_policy_order_epoch_advance(golem_ctx_t ctx);

/* --- Device expert cache budget + stats (per ctx, design §3.4). --- */

/* Expert-slot budget for the device cache, in EXPERTS: a count per layer
 * (one slot table shared by the layer's three expert tensors), never a byte
 * cap derived from one tensor's stride. Values above a tensor's expert count
 * clamp to it; <= 0 is "unset" and aborts when the slab path first engages
 * (device allocations are budget-bounded, never proportional to the expert
 * count). Set once by the engine at pool enable. */
void golem_set_device_cache_experts(golem_ctx_t ctx, int32_t k);

/* P5: enable next-layer prediction nodes (lookahead layers ahead, top
 * n_expert_used + margin each) delivered to `predict` with the registered
 * payload. lookahead <= 0 or predict == NULL disables (no graph nodes). Set
 * before the first graph build. */
void golem_set_predict(golem_ctx_t ctx, int32_t lookahead, int32_t margin, golem_predict_fn predict);
/* Graph-build query: prediction lookahead (0 = off, also for a NULL ctx). */
int32_t golem_predict_lookahead(golem_ctx_t ctx);
/* Graph-build query: prediction margin beyond n_expert_used. */
int32_t golem_predict_margin(golem_ctx_t ctx);
/* Current budget (<= 0 = unset). */
int32_t golem_device_cache_experts(golem_ctx_t ctx);

/* Clear the ctx's device cache maps — the invalidation seam (also the
 * cold-start lever). */
void golem_reset_device_cache(golem_ctx_t ctx);

/* P4 counters. Opaque here (C++ side keeps the members atomic); each member
 * is a lock-free 8-byte counter read by Rust as u64. The pointer is stable
 * for the ctx's lifetime. */
struct golem_device_cache_stats;
const struct golem_device_cache_stats *golem_get_device_cache_stats(golem_ctx_t ctx);

/* Placement stats stay PROCESS-GLOBAL by design: they count graph-build
 * placements per device across the process, are a windowed tripwire, and
 * outlive any ctx. */
struct golem_expert_placement_stats;
const struct golem_expert_placement_stats *golem_get_expert_placement_stats(void);

/* I2 runtime proof (design §2.3): expert MUL_MAT_ID nodes counted at
 * scheduler graph split by the device type of their assigned backend
 * (on_gpu = any non-CPU device). Cumulative since process start, bumped once
 * per graph build (never per routed expert); a reused graph is not recounted,
 * so treat them as a tripwire over a window, not an inventory. Covers the
 * three split stems only (fused ffn_gate_up_exps is excluded, as in P2).
 * Opaque here; members are lock-free 8-byte counters read by Rust as u64, in
 * order on_gpu, on_cpu. */

/* Force GPU offload of ops on host-resident weights below the upstream
 * batch-32 threshold (design I2): MUL_MAT_ID's batch is ne[2] = tokens, 1
 * during decode, so without this experts compute on the CPU. Values ≤ 0
 * restore the env/default behavior. Read once at device init — call before
 * backend init. Deliberate process-global setter. */
void golem_set_expert_offload_min_batch(int n);

/* --- Internal seams (not part of the stable engine-facing ABI). --- */

/* Attach/detach a ctx to a scheduler. set_ctx increments the ctx's attach
 * refcount; the sched destructor and detach decrement it. NULL ctx detaches.
 * libllama re-applies it after EVERY ggml_backend_sched_new. */
void golem_sched_set_ctx(struct ggml_backend_sched *sched, golem_ctx_t ctx);
void golem_sched_detach_ctx(struct ggml_backend_sched *sched);

/* libllama seam: store `ctx` on the model before llama_init_from_model;
 * llama_context stores it and re-applies golem_sched_set_ctx to every sched
 * it creates. detach clears a live context's sched attachment. Both
 * null-guarded. */
struct llama_model;
struct llama_context;
void llama_golem_model_attach(struct llama_model *model, golem_ctx_t ctx);
void llama_golem_context_detach(struct llama_context *ctx);

/* 0 when `t` is not an expert tensor; else expert count. */
int32_t golem_tensor_expert_count(const void *tensor);
/* 0 when count is 0; else bytes per expert (one ne[0]×ne[1] plane stride, nb[2]). */
int32_t golem_tensor_expert_stride_bytes(const void *tensor);

#ifdef __cplusplus
}
#endif
#endif /* GOLEM_HOOKS_H */
