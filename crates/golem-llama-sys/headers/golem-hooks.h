// golem-hooks.h — Golem policy-hook ABI stub (design tech-design.md §3.4 P1).
// Crate-local stub while the vendor tree is pristine (`patches = none-yet` in
// engine/llama-cpp.lock). This exact header moves into ggml/include/ as patch
// P1 in M2.1; the ABI declared here is frozen now so Rust-side code does not churn.
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

#ifdef __cplusplus
}
#endif
#endif /* GOLEM_HOOKS_H */
