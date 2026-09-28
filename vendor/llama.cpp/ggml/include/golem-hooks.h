// Golem policy-hook ABI + expert-metadata queries (Golem patch P1).
// Part of the vendored ggml include tree; the ABI is frozen so Rust-side
// code does not churn. Wired into ggml-backend.cpp by P1.
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
 * pool before the substrate reads them (blocks; called on the decode thread). */
typedef void (*golem_ensure_host_resident_fn)(void *ctx, const void *tensor,
                                              int32_t first_id, int32_t last_id);

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
                          void *ctx);

/* 0 when `t` is not an expert tensor; else expert count. */
int32_t golem_tensor_expert_count(const void *tensor);
/* 0 when count is 0; else bytes per expert (one ne[0]×ne[1] plane stride, nb[2]). */
int32_t golem_tensor_expert_stride_bytes(const void *tensor);

/* --- P2/P4: persistent device expert-copy cache (design §3.4). --- */

/* Soft cap (bytes) for the device expert-copy cache; 0 = full capacity
 * (never evict). Mutated over FFI by tests. */
void golem_set_device_cache_budget(size_t bytes);

/* Clear the device cache map — the invalidation/multi-model seam (also
 * Task 4's cold-start lever). */
void golem_reset_device_cache(void);

/* P4 counters. Opaque here (C++ side keeps the members atomic); each member
 * is a lock-free 8-byte counter read by Rust as u64. */
struct golem_device_cache_stats;
const struct golem_device_cache_stats *golem_get_device_cache_stats(void);

#ifdef __cplusplus
}
#endif
#endif /* GOLEM_HOOKS_H */
