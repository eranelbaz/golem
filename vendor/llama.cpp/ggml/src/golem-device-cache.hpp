// Golem patch P2: device expert-copy cache state machine (design §3.4).
// Included by BOTH ggml-backend.cpp (split-scheduler copy path) and
// tests/golem-p2-cache.cpp (fuzz harness): a single shared definition —
// hand-copied declarations would be an ODR drift (silent wrong-layout crash).
#ifndef GOLEM_DEVICE_CACHE_HPP
#define GOLEM_DEVICE_CACHE_HPP

#include "ggml-impl.h" // ggml_bitset_* (non-public; resolved via ggml/src include dir)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;

namespace golem {

// Validity state for one device expert-copy cache entry. Reset on ANY
// fingerprint mismatch (recycled host addresses across same-n_expert tensors
// must not alias stale bits).
struct GolemDeviceCache {
    std::vector<ggml_bitset_t> valid; // bitset over n_expert
    size_t valid_bytes = 0;
    const void *owner = nullptr; // host tensor fingerprint (alias guard)
    size_t expert_size = 0;
    // Persistent OWNED device region (design §3.4): a full-size twin of the
    // host expert tensor allocated once on the split backend; compute reads
    // it (node->src[0] is redirected to it). The scheduler's transient cpy
    // slots alias one address across the expert planes, so caching validity
    // there is unsound — each plane's copy overwrites the others'. The owned
    // region is exclusive per host tensor. Held (and freed by
    // golem_reset_device_cache) by the map entry.
    struct ggml_tensor *         owned     = nullptr;
    struct ggml_backend_buffer * owned_buf = nullptr;
    struct ggml_context *        owned_ctx = nullptr;
};

// Per-step accounting; the caller folds these into the P4 atomic counters.
struct GolemDcCounters {
    uint64_t copied = 0;  // host→device expert bytes written (incl. padding tails)
    uint64_t skipped = 0; // bytes skipped because bit-valid
    uint64_t evicted = 0; // experts invalidated to stay under budget
};

inline size_t golem_dc_popcount_bytes(const std::vector<ggml_bitset_t> &bs, size_t expert_size) {
    size_t n = 0;
    for (ggml_bitset_t w : bs) {
#if defined(__GNUC__) || defined(__clang__)
        n += (size_t) __builtin_popcount(w);
#else
        for (; w; w >>= 1) {
            n += (size_t) (w & 1);
        }
#endif
    }
    return n * expert_size;
}

// One scheduler step against `st` — pure state machine; the physical
// host→device write is delegated to on_copy(first_id, last_id, pad_ok).
//
// - `order` must be a permutation of [0, n_experts) in evict-first order;
//   NULL / duplicate / out-of-range ids abort (C-side choke point).
// - `budget_bytes` is a soft cap: walk `order` clearing non-routed valid bits
//   while valid_bytes > budget; routed experts are never evicted and overflow
//   is tolerated (walk stops when only routed experts remain valid).
//   0 ⇒ full capacity (never evict).
// - Copy phase: each maximal run [first_id, last_id] of routed ids is skipped
//   (accounted in `skipped`) when all its bits are valid; otherwise it is
//   copied via on_copy with bits set and `copied` incremented, including the
//   padding_end tail. pad_ok = last_id+1 < n_experts && that expert's bit is
//   invalid (correctness core §3: keeps "bit ⇒ byte-identity" provable).
// - valid_bytes is recomputed from popcount after the eviction walk and again
//   after the copy phase (no drift).
// - Any fingerprint mismatch (owner / expert_size / n_experts) resets `st` to
//   all-invalid first.
template <typename CopyFn>
GolemDcCounters golem_dc_step_impl(GolemDeviceCache &st, const std::vector<int32_t> &routed,
                                   const int32_t *order, size_t budget_bytes, size_t expert_size,
                                   int32_t n_experts, const void *host_data, CopyFn &&on_copy) {
    GolemDcCounters c;
    GGML_ASSERT(n_experts > 0);
    GGML_ASSERT(expert_size > 0);

    // Validate order FIRST: all ids distinct, all in [0, n_experts).
    {
        std::vector<ggml_bitset_t> seen(ggml_bitset_size(n_experts), 0);
        GGML_ASSERT(order != NULL);
        for (int32_t i = 0; i < n_experts; ++i) {
            GGML_ASSERT(order[i] >= 0 && order[i] < n_experts);
            GGML_ASSERT(!ggml_bitset_get(seen.data(), order[i]));
            ggml_bitset_set(seen.data(), order[i]);
        }
    }

    // Fresh or aliased/stale entry: reset.
    if (st.valid.size() != (size_t) ggml_bitset_size(n_experts) || st.owner != host_data ||
        st.expert_size != expert_size) {
        // Preserve the owned persistent region across state resets: the
        // region is keyed by the entry (host tensor), not by the resettable
        // validity state.
        struct ggml_tensor *         owned     = st.owned;
        struct ggml_backend_buffer * owned_buf = st.owned_buf;
        struct ggml_context *        owned_ctx = st.owned_ctx;
        st                = GolemDeviceCache{};
        st.owned          = owned;
        st.owned_buf      = owned_buf;
        st.owned_ctx      = owned_ctx;
        st.valid.assign(ggml_bitset_size(n_experts), 0);
        st.owner          = host_data;
        st.expert_size    = expert_size;
    }

    // (a) budget eviction (soft cap): evict-first walk of `order`.
    const size_t budget = budget_bytes == 0 ? (size_t) n_experts * expert_size : budget_bytes;
    if (st.valid_bytes > budget) {
        std::vector<ggml_bitset_t> is_routed(ggml_bitset_size(n_experts), 0);
        for (int32_t id : routed) {
            GGML_ASSERT(id >= 0 && id < n_experts);
            ggml_bitset_set(is_routed.data(), id);
        }
        size_t vb = st.valid_bytes;
        for (int32_t i = 0; i < n_experts && vb > budget; ++i) {
            const int32_t id = order[i];
            if (!ggml_bitset_get(is_routed.data(), id) && ggml_bitset_get(st.valid.data(), id)) {
                ggml_bitset_clear(st.valid.data(), id);
                vb -= expert_size;
                c.evicted++;
            }
        }
    }
    st.valid_bytes = golem_dc_popcount_bytes(st.valid, expert_size);

    // (b) copy routed ranges with invalid experts; skip fully-valid runs.
    bool    in_run   = false;
    int32_t first_id = -1, last_id = -1;
    const auto flush = [&](int32_t f, int32_t l) {
        const size_t run_bytes = (size_t) (l - f + 1) * expert_size;
        for (int32_t id = f; id <= l; ++id) {
            if (!ggml_bitset_get(st.valid.data(), id)) {
                const bool pad_ok = l + 1 < n_experts && !ggml_bitset_get(st.valid.data(), l + 1);
                for (int32_t id2 = f; id2 <= l; ++id2) {
                    ggml_bitset_set(st.valid.data(), id2);
                }
                c.copied += run_bytes + (pad_ok ? std::min<size_t>(expert_size, 512) : 0);
                on_copy(f, l, pad_ok);
                return;
            }
        }
        c.skipped += run_bytes; // all bits in the run valid
    };
    for (int32_t id : routed) {
        if (!in_run || id != last_id + 1) {
            if (in_run) {
                flush(first_id, last_id);
            }
            first_id = id;
            in_run   = true;
        }
        last_id = id;
    }
    if (in_run) {
        flush(first_id, last_id);
    }

    // Recompute from popcount after the copy phase (no drift).
    st.valid_bytes = golem_dc_popcount_bytes(st.valid, expert_size);
    return c;
}

// Non-template, non-static entry point linked from the ctest fuzz TU
// (tests/golem-p2-cache.cpp): counts physical writes without performing them —
// the state invariants do not depend on byte contents.
inline GolemDcCounters golem_dc_step(GolemDeviceCache &st, const std::vector<int32_t> &routed,
                                     const int32_t *order, size_t budget_bytes, size_t expert_size,
                                     int32_t n_experts, const void *host_data) {
    return golem_dc_step_impl(st, routed, order, budget_bytes, expert_size, n_experts, host_data,
                              [](int32_t, int32_t, bool) {});
}

} // namespace golem
#endif /* GOLEM_DEVICE_CACHE_HPP */
