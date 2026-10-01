// Golem policy-hook ABI + expert-metadata queries (Golem patch P1).
// Part of the vendored ggml include tree; the ABI is versioned — a change
// updates the Rust bindings and both sides together. Wired into
// ggml-backend.cpp by P1.
#ifndef GOLEM_HOOKS_H
#define GOLEM_HOOKS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque policy-state handle computed and owned by Rust. */
typedef struct golem_policy_state golem_policy_state;

/* Hook: ensure [first_id, last_id] of `tensor` are resident in the host expert
 * pool before the substrate reads them (blocks; called on the decode thread).
 * ABI v2 (design §3.4): fills out_ptrs[e - first_id] with the pool-side host
 * pointer of expert e's plane — the RAM pool is the source of the host→device
 * copy, because the loader's expert bytes may live in a read-only file mmap.
 * The pool holds a lease on each returned slot until the matching
 * copy_done(ctx) call, so eviction cannot recycle a slot mid-DMA. */
typedef void (*golem_ensure_host_resident_fn)(void *ctx, const void *tensor,
                                              int32_t first_id, int32_t last_id,
                                              const void **out_ptrs);

/* Hook: releases the host-pool leases taken by ensure_host_resident since
 * the previous copy_done. Called at the start of each scheduler compute —
 * the engine synchronizes after every llama_decode (session.rs), so the
 * previous decode's DMAs are complete and the slots are safe to recycle. */
typedef void (*golem_copy_done_fn)(void *ctx);

/* Hook: supply eviction/keep ordering for the device expert-copy cache.
 * policy_query returns a permutation of [0, n_experts) in evict-first order;
 * the pointer stays valid until the next policy_query call for the same
 * tensor; C never frees it. */
typedef const int32_t *(*golem_policy_query_fn)(void *ctx, const void *tensor,
                                                int32_t n_experts);

/* Hook: per-step routing notification for score decay/accumulation. */
typedef void (*golem_notify_access_fn)(void *ctx, int64_t expert_id);

/* Register hooks. Passing NULL for any hook restores upstream behavior for
 * that hook (design invariant: unset hooks == byte-identical to upstream). */
void golem_register_hooks(golem_ensure_host_resident_fn ensure_host_resident,
                          golem_policy_query_fn policy_query,
                          golem_notify_access_fn notify_access,
                          golem_copy_done_fn copy_done,
                          void *ctx);

/* 0 when `t` is not an expert tensor; else expert count. */
int32_t golem_tensor_expert_count(const void *tensor);
/* 0 when count is 0; else bytes per expert (one ne[0]×ne[1] plane stride, nb[2]). */
int32_t golem_tensor_expert_stride_bytes(const void *tensor);

/* Force GPU offload of ops on host-resident weights below the upstream
 * batch-32 threshold (design I2): MUL_MAT_ID's batch is ne[2] = tokens, 1
 * during decode, so without this experts compute on the CPU. Values ≤ 0
 * restore the env/default behavior. Read once at device init — call before
 * backend init. Implemented in ggml-backend.cpp (backend-agnostic); the
 * CUDA/HIP device init consults it. */
void golem_set_expert_offload_min_batch(int n);

/* --- P2/P4: slot-indirected device expert cache (design sec 3.4 v2). --- */

/* Per-tensor expert-slot budget for the device cache, in EXPERTS: a
 * count per (layer, kind), never a byte cap derived from one tensor's
 * stride. 0 = unlimited (a full-size slab). Mutated over FFI by
 * tests and set once by the engine at pool enable. */
void golem_set_device_cache_experts(int32_t k);
/* Current budget (0 = unlimited). */
int32_t golem_device_cache_experts(void);

/* Clear the device cache map - the invalidation/multi-model seam (also the cold-start lever). */
void golem_reset_device_cache(void);

/* P4 counters. Opaque here (C++ side keeps the members atomic); each member is a lock-free 8-byte counter read by Rust as u64. */
struct golem_device_cache_stats;
const struct golem_device_cache_stats *golem_get_device_cache_stats(void);

#ifdef __cplusplus
}
#endif
#endif /* GOLEM_HOOKS_H */
