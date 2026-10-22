// Golem policy-hook ABI + expert-metadata queries (Golem patch P1).
// ABI history:
//   v7  exclusive-residency eviction hook.
//   v8  eviction hook batched per (layer, compute).
//   v9  abandoned-D2H hook: a parked eviction whose plane copies were not all
//       issued by compute end is reported so the pool cancels the claim.
//   v10 per-model hook context; prefill-stream hooks (design section 7.6).
//   v11 the stream source may abort the compute (returns 1).
//   v12 a failed device-slab allocation fails the compute instead of aborting.
//   v13 ensure may fail a compute when the store cannot page the misses in.
//   v14 resident device bytes query.
//   v15 GOLEM_STREAM_SRC_STORAGE_LOST; device-cache stats gain H2D expert
//       copies; dedicated H2D copy stream (golem_set_async_h2d,
//       golem_take_h2d_timing).
//   v16 opt-in decode-only CPU expert math (design §1 O1):
//       golem_set_cpu_expert_math, golem_cpu_lease_fn,
//       golem_take_cpu_expert_stats, golem_sched_get_ctx.
// Part of the vendored ggml include tree; the ABI is versioned -- a change
// updates the Rust bindings and both sides together.
// Hook STATE lives in a per-model `golem_ctx` (created by the engine, attached
// to llama_model / scheds); only process-global capability flags remain here.
#ifndef GOLEM_HOOKS_H
#define GOLEM_HOOKS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ggml-base is built with hidden visibility; libllama and the ctest TUs link
 * these across the shared-object boundary. */
#if defined(_WIN32) && !defined(__MINGW32__)
#    define GOLEM_API
#else
#    define GOLEM_API __attribute__((visibility("default")))
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
#define GOLEM_HOOKS_ABI_VERSION 16
GOLEM_API int golem_hooks_abi_version(void);

/* Hook: ensure the listed expert-plane misses are resident in the host expert
 * pool before the substrate reads them (blocks; called on the decode thread).
 * ONE call per (layer, step, split-segment): miss_kinds[i] (1=gate 2=up
 * 3=down) / miss_ids[i] describe the i-th miss across ALL planes of the
 * layer; fills out_ptrs[i] with the pool-side host pointer of that plane's
 * expert — the RAM pool is the source of the host→device copy, because the
 * loader's expert bytes may live in a read-only file mmap. The pool holds a
 * lease on each returned slot until the matching copy_done(payload) call, so
 * eviction cannot recycle a slot mid-DMA. `tensor` is the first expert input
 * of the layer that step (identity only for logging).
 * Returns 0 = out_ptrs filled; nonzero (ABI v13) = the misses could not be
 * paged in (every replica unreadable): C drains the split's issued copies
 * and the compute returns GGML_STATUS_FAILED -- llama_decode returns -3. */
typedef int32_t (*golem_ensure_host_resident_fn)(void *payload, const void *tensor,
                                                 const int32_t *miss_kinds,
                                                 const int32_t *miss_ids,
                                                 int32_t n_miss,
                                                 const void **out_ptrs);

/* Hook (exclusive residency, design §6.3 O10): device-slab slots were
 * evicted; decide where their bytes go. ONE call per (layer, compute) with
 * the layer's whole eviction list. For expert i, fills out_ptrs[3i..3i+3]
 * (indexed kind-1: gate/up/down) with the pool-side host destinations of
 * that expert's three planes and sets keep[i] != 0 → the substrate issues
 * ONE D2H copy per plane for that expert; keep[i] == 0 drops the bytes
 * (pool copy already current, or the pool's eviction victim scores
 * higher); the substrate then simply invalidates the slot. The pool keeps
 * the returned slots gated until the matching copy_done, which runs only
 * after the substrate synchronized its backends, so the D2H has landed
 * before any pool reader is admitted. Called on the decode thread, once
 * per layer per compute.
 *
 * Stream/cost note: the D2H copies run on the SPLIT BACKEND's stream — the
 * same stream the H2D refills of the stolen slots use — and are issued
 * BEFORE those refills, so stream order lands the eviction D2H first (it
 * is NOT on a separate promotion stream). Cost ≈ 15.2 MB / 25 GB/s ≈
 * 0.6 ms per evicted expert; measured decode impact pending A/B. */
typedef void (*golem_evict_batch_fn)(void *payload, const void *tensor,
                                     const int32_t *experts, int32_t n_evicts,
                                     void *const *out_ptrs, int32_t *keep);

/* Hook (exclusive residency, ABI v9): a parked eviction D2H whose plane
 * copies were NOT all issued by compute end (the owning plane's drain was
 * skipped — node2 NULL, non-host/non-weights buffer, mode-0/1 plan — or the
 * eviction was parked from a segment later than an earlier plane's copy
 * phase). ONE call per (layer, compute) listing the affected experts; runs
 * after the scheduler synchronized its backends, so any issued copy has
 * landed and the destinations are safe to recycle. The pool must cancel the
 * claim: slot back to its free list, fill gate re-opened, key NOT marked
 * pool-resident (its VRAM bit was already cleared at eviction time, so it
 * is re-read from disk later). Never delivered for experts whose three
 * copies were all issued — those commit via the matching copy_done. */
typedef void (*golem_evict_abandon_fn)(void *payload, int32_t blk,
                                       const int32_t *experts, int32_t n_experts);

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

/* Hook (prefill stream, design section 7.6): host byte source for ONE streamed
 * plane (all n_ids experts of layer `blk`, plane `kind` 1=gate 2=up 3=down).
 * `ids` holds the experts whose ranges C will copy (routed-only set, or the
 * whole identity range when the routed share exceeds the configured
 * threshold); the hook fills out_ptrs[i] with a stable host pointer for
 * expert ids[i]: the RAM-pool plane when the unit is resident (a lease keyed
 * by `seq` is taken -- no score, no residency change), else a pinned staging
 * slot the reader thread has already filled (the call blocks on that fill;
 * the overlap comes from the routing readback enqueueing reads ahead of the
 * copy). Never pages misses into the pool: a streamed ubatch must not move
 * the decode hot set (section 7.6.4). `seq` is C's monotonic plane counter -- the
 * matching golem_stream_release(seq) releases the leases and staging slot.
 * NULL hook -> golem_set_prefill_stream aborts (streaming unavailable).
 * Returns GOLEM_STREAM_SRC_OK = out_ptrs filled; otherwise C waits for the
 * split's issued H2D and calls golem_stream_release(seq) for every seq
 * issued so far (this one included): GOLEM_STREAM_SRC_CANCEL (ABI v11,
 * request cancelled) -> GGML_STATUS_ABORTED, llama_decode returns 2;
 * GOLEM_STREAM_SRC_STORAGE_LOST (v15) or any other value ->
 * GGML_STATUS_FAILED, llama_decode returns -3. */
#define GOLEM_STREAM_SRC_OK 0
#define GOLEM_STREAM_SRC_CANCEL 1
#define GOLEM_STREAM_SRC_STORAGE_LOST 2
typedef int32_t (*golem_stream_source_fn)(void *payload, int32_t blk, int32_t kind,
                                       uint64_t seq, const int32_t *ids,
                                       int32_t n_ids, const void **out_ptrs);

/* Hook (prefill stream, design section 7.6): everything keyed by `seq` and every
 * smaller seq is released -- pool leases drop, staging slots recycle. C calls
 * it when a slab is about to be refilled (seq >= 2: sync of the slab's
 * recorded event first, so the H2D + compute reading those bytes completed)
 * and, for anything left, the engine releases the remainder at copy_done. */
typedef void (*golem_stream_release_fn)(void *payload, uint64_t seq);

/* Hook (P5, design 6.5): next-layer routing prediction. When
 * golem_set_predict enabled it, the model graph builds, from layer `src_blk`'s
 * FFN input, the top (n_expert_used + margin) experts layer `target_blk`'s
 * router would pick (target = src + 1 .. src + lookahead). The scheduler
 * reads them right after src_blk's routing readback (the device is already
 * synchronized there) and passes them here, decode ubatches (1 token) only.
 * Delivered only from source layers whose expert weights are host-resident
 * (the pooled path that runs the routing readback); the prediction nodes
 * must share the source layer's split or an earlier one (all-GPU layers).
 * Real routing is untouched (I1): predictions only steer staging. */
typedef void (*golem_predict_fn)(void *payload, int32_t src_blk, int32_t target_blk,
                                 const int32_t *ids, int32_t n_ids);

/* --- Context lifecycle. --- */

/* Create a per-model hook context. `n_layers` sizes the flat eviction-order
 * table (indexed blk × kind; aborts later if a blk >= n_layers reaches the
 * order path); `n_experts` sizes the per-(blk,kind) triple buffers and the
 * slab scratch. Order buffers and scratch are preallocated here: a decode
 * step performs no heap allocation on the ctx path. */
GOLEM_API golem_ctx_t golem_ctx_create(int32_t n_layers, int32_t n_experts);

/* Free a context. Aborts while a sched is attached (attach refcount != 0 —
 * the sched destructor or llama_golem_context_detach must run first) and
 * while hooks are still registered. */
GOLEM_API void golem_ctx_free(golem_ctx_t ctx);

/* Register the hooks on `ctx`. `payload` is passed back as the first
 * argument of every hook invocation. Not synchronized with a running step: call
 * before any scheduler compute starts and not concurrently with one (the
 * engine registers at pool enable, before the first decode). Passing NULL
 * for a hook restores upstream behavior for that concern (unset hooks ==
 * byte-identical to upstream). Returns a handle that must be unregistered
 * with golem_unregister_hooks before golem_ctx_free. */
GOLEM_API golem_hook_handle_t golem_register_hooks(golem_ctx_t ctx,
                                         golem_ensure_host_resident_fn ensure,
                                         golem_routing_fn routing,
                                         golem_copy_done_fn copy_done,
                                         golem_evict_batch_fn evict_batch,
                                         golem_evict_abandon_fn evict_abandon,
                                         golem_stream_source_fn stream_source,
                                         golem_stream_release_fn stream_release,
                                         void *payload);

/* Marks the ctx unhooked, then JOINS: waits until the ctx's in-flight step
 * count reaches 0, so no callback can touch the engine payload after this
 * returns. Aborts if called from inside a hook callback (deadlock by
 * construction — the join would wait for the calling step). */
GOLEM_API void golem_unregister_hooks(golem_hook_handle_t handle);

/* Monotonic per-ctx counter, bumped on every register: the engine tags
 * scratch/pool state with it to reject addresses recycled across sessions. */
GOLEM_API uint64_t golem_ctx_generation(golem_ctx_t ctx);

/* Process-global capability flag: number of hooked ctxs (> 0 = some model
 * has hooks). Consumed ONLY by golem_graphs_disabled(); per-sched decisions
 * (e.g. the cpy shrink) read the sched's OWN ctx, so a hookless sched under a
 * hooked process takes the upstream path verbatim. Documented deviation:
 * while ANY ctx is hooked, HIP graph capture is disabled process-wide. */
GOLEM_API int golem_any_ctx_hooked(void);

/* --- Eviction order (design §3.4). --- */

/* Read the current eviction order for the (blk, kind) of `tensor`: a
 * permutation of [0, n_experts) in evict-first order, or NULL before the
 * first push (callers fall back to identity order + a one-time warning).
 * Triple-buffered: the pointer stays valid for the duration of the step that
 * acquired it (the publisher skips a buffer instead of overwriting one a
 * reader of the current epoch may hold). Flat per-ctx table indexed
 * blk × kind — no hash lookup. C never frees it and never blocks. */
GOLEM_API const int32_t *golem_policy_order(golem_ctx_t ctx, const void *tensor,
                                  int32_t n_experts);

/* Publish a fresh eviction order for (blk, kind): copies `n` entries into a
 * free triple-buffer slot and swaps it in (release). Skips (returns 0)
 * without blocking when no buffer is free under the reader-epoch gate; the
 * publisher retries next step. `kind` is 1-based (1=gate 2=up 3=down).
 * Called off the decode thread — the only synchronized ctx entry point. */
GOLEM_API int golem_policy_order_push(golem_ctx_t ctx, int32_t blk, int32_t kind,
                            const int32_t *order, int32_t n);

/* Advance the order epoch (one per decode step): releases the triple-buffer
 * slot the previous step's readers hold. The scheduler calls this at graph
 * compute entry; exposed so tests (and embedders driving steps by hand) can
 * match publisher progress without running a step. */
GOLEM_API void golem_policy_order_epoch_advance(golem_ctx_t ctx);

/* --- Device expert cache budget + stats (per ctx, design §3.4). --- */

/* Expert-slot budget for the device cache, in EXPERTS: a count per layer
 * (one slot table shared by the layer's three expert tensors), never a byte
 * cap derived from one tensor's stride. Values above a tensor's expert count
 * clamp to it; <= 0 is "unset" and aborts when the slab path first engages
 * (device allocations are budget-bounded, never proportional to the expert
 * count). Set once by the engine at pool enable. */
GOLEM_API void golem_set_device_cache_experts(golem_ctx_t ctx, int32_t k);

/* P5: enable next-layer prediction nodes (lookahead layers ahead, top
 * n_expert_used + margin each) delivered to `predict` with the registered
 * payload. lookahead <= 0 or predict == NULL disables (no graph nodes).
 * Init-only: set before the first graph build, never changed afterwards
 * (built graphs keep the node set they were built with). */
GOLEM_API void golem_set_predict(golem_ctx_t ctx, int32_t lookahead, int32_t margin, golem_predict_fn predict);
/* Graph-build query: prediction lookahead (0 = off, also for a NULL ctx). */
GOLEM_API int32_t golem_predict_lookahead(golem_ctx_t ctx);
/* Graph-build query: prediction margin beyond n_expert_used. */
GOLEM_API int32_t golem_predict_margin(golem_ctx_t ctx);
/* Current budget (<= 0 = unset). */
GOLEM_API int32_t golem_device_cache_experts(golem_ctx_t ctx);
/* ABI v12: a device-slab allocation that fails (VRAM taken after load)
 * fails the compute with GGML_STATUS_ALLOC_FAILED (llama_decode -2) instead
 * of aborting; the engine resets the cache and lowers K. Count of such
 * failures since ctx creation (0 for a NULL ctx). */
GOLEM_API uint64_t golem_slab_alloc_failures(golem_ctx_t ctx);
#ifdef GOLEM_TEST_SUPPORT
/* Test seam (ABI v15: declared only under GOLEM_TEST_SUPPORT): the next n
 * slab (re)allocations fail as if out of VRAM. */
GOLEM_API void golem_inject_slab_alloc_failure(golem_ctx_t ctx, int32_t n);
#endif
/* ABI v14: device bytes currently allocated by the ctx — expert slabs plus
 * remapped-ids tensors (*slab_bytes) and prefill-stream slabs
 * (*stream_bytes). Slabs allocate lazily, so this is what is resident, not
 * the K + 1 reservation. Call between computes (stream slabs are mutated
 * outside dc_mutex). Zeros for a NULL ctx; either out pointer may be NULL. */
GOLEM_API void golem_device_cache_resident_bytes(golem_ctx_t ctx, uint64_t *slab_bytes, uint64_t *stream_bytes);

/* Clear the ctx's device cache maps — the invalidation seam (also the
 * cold-start lever). Also frees the prefill-stream slabs. */
GOLEM_API void golem_reset_device_cache(golem_ctx_t ctx);

/* Prefill-stream configuration (design section 7.6, ABI v10). `on != 0` arms
 * streaming for subsequent computes; the device slabs (two, each
 * `n_experts x max_plane_stride` bytes) are allocated lazily on the first
 * streamed split's backend and freed by golem_reset_device_cache. When a
 * plane's routed share exceeds `routed_max` (0 < r <= 1) C hands the stream
 * source the whole identity id range instead of the routed set. Requires the
 * stream hooks to be registered (else aborts). Init/config-time call --
 * not concurrent with a compute. */
GOLEM_API void golem_set_prefill_stream(golem_ctx_t ctx, int32_t on, int32_t n_experts,
                              int32_t max_plane_stride, float routed_max);

/* Dedicated H2D copy stream (ABI v15). `on != 0` issues the pooled and
 * device-cache H2D on the split backend's copy stream, event-joined with the
 * compute stream (WAR/evict-before-refill wait before, RAW wait after) and
 * drained before every lease release. Returns 0, or -1 when refused
 * (GGML_CUDA_GRAPH_OPT=1 concurrent-stream mode owns the stream indices).
 * The backend must export "golem_copy_stream_iface" via
 * ggml_backend_reg_get_proc_address; a compute on a backend without it is a
 * documented FFI abort (the engine refuses async_h2d on a non-HIP substrate).
 * Config-time call. One device backend per ctx: a compute on a second device
 * backend is a documented FFI abort (the engine refuses multi-GPU async_h2d
 * up front). */
GOLEM_API int32_t golem_set_async_h2d(golem_ctx_t ctx, int32_t on);

/* Copy-stream H2D time (sum of per-split event pairs) and bytes since the
 * last call; resets both. Returns 1 when measured, 0 when async_h2d is off
 * or the per-step ring overflowed. Call between computes. */
GOLEM_API int32_t golem_take_h2d_timing(golem_ctx_t ctx, double *ms, uint64_t *bytes);

/* Hook (ABI v16, design §1 O1): among ids[0..n) of layer `blk`, count the
 * units RESIDENT in the host pool (no paging in, no scoring), and lease the
 * first ceil(frac * n_resident) of them in id order -- one lease per resident
 * unit taken, never one for a unit not taken. Writes the taken ids to
 * out_ids[i] and their three plane pointers to out_ptrs3[3i + k]
 * (k = kind - 1: gate, up, down); returns the count taken (0..n). The leases
 * are released at the next copy_done(payload), before its exclusive-residency
 * release, so the CPU reads the slots in place until the step's join. Decode
 * thread, once per (layer, step). */
typedef int32_t (*golem_cpu_lease_fn)(void *payload, int32_t blk, const int32_t *ids, int32_t n,
                                      float frac, int32_t *out_ids, const void **out_ptrs3);

/* Opt-in CPU expert math (ABI v16, design §1 O1). `on != 0` arms it: in
 * decode, a share `frac` in (0, 1] of a layer's resident slab misses is
 * computed on the CPU by an `n_threads` (>= 1) worker reading the leased
 * pool slots, and joined into ffn_moe_out through the graph's
 * golem_cpu_acc-<blk> input. Requires lease != NULL and the routing hook
 * registered. Init-only (before the first graph build). The CPU worker is
 * created lazily on the first eligible compute. Returns 0, or -1 when refused
 * (bad arguments, no routing hook, already armed); `on == 0` arms nothing
 * but still returns -1 for a NULL ctx, no routing hook or an armed ctx. */
GOLEM_API int32_t golem_set_cpu_expert_math(golem_ctx_t ctx, int32_t on, float frac, int32_t n_threads,
                                  golem_cpu_lease_fn lease);
/* Graph-build query: 1 when armed, 0 otherwise (0 for a NULL ctx). */
GOLEM_API int32_t golem_cpu_expert_math_on(golem_ctx_t ctx);
/* Since the last call: experts computed on the CPU, their plane bytes, worker
 * busy ms, decode-thread join wait ms, and ids remapped to the padding slot K
 * (summed over planes); resets all five. Returns 1 when armed, 0 otherwise
 * (outputs untouched). Call between computes. */
GOLEM_API int32_t golem_take_cpu_expert_stats(golem_ctx_t ctx, uint64_t *experts, uint64_t *bytes, double *ms,
                                    double *join_wait_ms, uint64_t *padded);

/* P4 counters. Opaque here (C++ side keeps the members atomic); each member
 * is a lock-free 8-byte counter read by Rust as u64. The pointer is stable
 * for the ctx's lifetime. */
struct golem_device_cache_stats;
GOLEM_API const struct golem_device_cache_stats *golem_get_device_cache_stats(golem_ctx_t ctx);

/* Placement stats stay PROCESS-GLOBAL by design: they count graph-build
 * placements per device across the process, are a windowed tripwire, and
 * outlive any ctx. */
struct golem_expert_placement_stats;
GOLEM_API const struct golem_expert_placement_stats *golem_get_expert_placement_stats(void);

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
GOLEM_API void golem_set_expert_offload_min_batch(int n);

/* --- Internal seams (not part of the stable engine-facing ABI). --- */

/* Attach/detach a ctx to a scheduler. set_ctx increments the ctx's attach
 * refcount; the sched destructor and detach decrement it. NULL ctx detaches.
 * libllama re-applies it after EVERY ggml_backend_sched_new. */
GOLEM_API void golem_sched_set_ctx(struct ggml_backend_sched *sched, golem_ctx_t ctx);
GOLEM_API void golem_sched_detach_ctx(struct ggml_backend_sched *sched);
/* The ctx attached to `sched` (ABI v16); NULL for a NULL or unattached sched. */
GOLEM_API golem_ctx_t golem_sched_get_ctx(struct ggml_backend_sched *sched);

/* libllama seam: store `ctx` on the model before llama_init_from_model;
 * llama_context stores it and re-applies golem_sched_set_ctx to every sched
 * it creates. detach clears a live context's sched attachment. Both
 * null-guarded. */
struct llama_model;
struct llama_context;
GOLEM_API void llama_golem_model_attach(struct llama_model *model, golem_ctx_t ctx);
GOLEM_API void llama_golem_context_detach(struct llama_context *ctx);

/* 0 when `t` is not an expert tensor; else expert count. */
GOLEM_API int32_t golem_tensor_expert_count(const void *tensor);
/* 0 when count is 0; else bytes per expert (one ne[0]×ne[1] plane stride, nb[2]). */
GOLEM_API int32_t golem_tensor_expert_stride_bytes(const void *tensor);

#ifdef __cplusplus
}
#endif
#endif /* GOLEM_HOOKS_H */
