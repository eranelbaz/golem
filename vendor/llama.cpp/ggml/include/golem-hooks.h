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

/* Hook: supply eviction/keep ordering for the device expert-copy cache. */
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
/* 0 when count is 0; else bytes per expert (one row-block stride). */
int32_t golem_tensor_expert_stride_bytes(const void *tensor);

#ifdef __cplusplus
}
#endif
#endif /* GOLEM_HOOKS_H */
