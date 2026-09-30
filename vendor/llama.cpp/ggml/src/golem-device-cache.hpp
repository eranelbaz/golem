// Golem patch P2 v2 (design §3.4 v2): SLOT-INDIRECTED device expert cache.
// Included by BOTH ggml-backend.cpp (split-scheduler copy path) and
// tests/golem-p2-cache.cpp (fuzz harness): a single shared definition —
// hand-copied declarations would be an ODR drift (silent wrong-layout crash).
//
// The device cache is a slab of K+1 expert-plane slots (the shrank scheduler
// input cpy; +1 non-NaN padding slot), and MUL_MAT_ID reads it through a
// REMAPPED ids tensor (values in [0, K)): routing is untouched — only the
// addressing moves. A full-size device twin (184 GB on the target model) and
// the full-size compute-buffer cpy are both gone.
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

// One (layer, kind) slab. Keyed by the host tensor fingerprint; any sched
// slab pointer change resets the state, so
// validity bits never describe stale device bytes.
struct GolemSlabEntry {
    // fingerprint
    const void *owner = nullptr;   // host tensor data ptr
    int32_t K = 0;                 // addressable slots (padding slot excluded)
    int32_t n_experts = 0;
    size_t expert_size = 0;
    // persistent device slab (K+1 planes; owns its buffer, independent of
    // galloc plans — cached slot bytes survive graph rebuilds)
    struct ggml_tensor *slab = nullptr;
    struct ggml_backend_buffer *buf = nullptr;
    struct ggml_context *ctx = nullptr;
    int64_t ids_ne0 = 0;           // remapped-ids tensor shape fingerprint
    int64_t ids_ne1 = 0;
    // state
    std::vector<int32_t> slot_of;  // expert id → slot, -1 = unassigned
    std::vector<ggml_bitset_t> valid; // bitset over slots: slot bytes current
    // remapped ids device tensor (int32, same 2-D shape as the ids node)
    struct ggml_tensor *ids_dev = nullptr;
    struct ggml_backend_buffer *ids_buf = nullptr;
    struct ggml_context *ids_ctx = nullptr;
    std::vector<int32_t> remap;    // host staging for the remapped ids
};

// Per-step accounting; the caller folds these into the P4 atomic counters.
struct GolemSlabCounters {
    uint64_t copied = 0;   // expert planes written this step
    uint64_t skipped = 0;  // routed experts whose slot bytes were current
    uint64_t evicted = 0;  // slots stolen from non-routed experts
};

// One scheduler step against `st`.
//
// - `routed` = sorted used-expert ids (from the ids bitset); `ids`/`ids_count`
//   = the raw ids values; `order` = Rust's evict-first permutation of
//   [0, n_experts) (NULL / duplicate / out-of-range aborts — C-side choke
//   point); `K` = addressable slots of the slab.
// - Every routed expert gets a valid slot: free first, else evict the first
//   non-routed assigned expert in `order` (policy-driven victim). Routed set
//   larger than K cannot be staged → GGML_ABORT (fail-loud; prefill slab
//   sizing is design D-7, deferred to M3.5).
// - on_copy(slot, id): physical host→device write of expert `id`'s plane into
//   `slot`. Called once per non-current routed expert.
// - st.remap = the remapped ids (same element order/count as `ids`), ready
//   for the caller's H2D of the ids_dev tensor.
template <typename OnCopy>
GolemSlabCounters golem_slab_step(GolemSlabEntry &st, const std::vector<int32_t> &routed,
                                  const int32_t *ids, int64_t ids_count, const int32_t *order,
                                  int32_t K, int32_t n_experts, size_t expert_size,
                                  const void *host_data, OnCopy &&on_copy) {
    GolemSlabCounters c;
    GGML_ASSERT(n_experts > 0);
    GGML_ASSERT(expert_size > 0);
    GGML_ASSERT(K > 0 && K <= n_experts);

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

    // Fresh or stale (address reuse across model/graph changes): reset.
    if ((int32_t) st.slot_of.size() != n_experts || st.owner != host_data ||
        st.K != K || st.expert_size != expert_size) {
        st.slot_of.assign(n_experts, -1);
        st.valid.assign(ggml_bitset_size(K + 1), 0);
        st.owner = host_data;
        st.K = K;
        st.n_experts = n_experts;
        st.expert_size = expert_size;
    }

    std::vector<ggml_bitset_t> is_routed(ggml_bitset_size(n_experts), 0);
    for (int32_t id : routed) {
        GGML_ASSERT(id >= 0 && id < n_experts);
        ggml_bitset_set(is_routed.data(), id);
    }

    const auto find_free_slot = [&]() -> int32_t {
        for (int32_t s = 0; s < K; ++s) {
            if (!ggml_bitset_get(st.valid.data(), s)) {
                // A free slot must not be owned by a live assignment.
                return s;
            }
        }
        return -1;
    };
    const auto evict_one = [&]() -> int32_t {
        for (int32_t i = 0; i < n_experts; ++i) {
            const int32_t cand = order[i];
            const int32_t s = st.slot_of[cand];
            if (s != -1 && !ggml_bitset_get(is_routed.data(), cand)) {
                ggml_bitset_clear(st.valid.data(), s);
                st.slot_of[cand] = -1;
                c.evicted++;
                return s;
            }
        }
        return -1;
    };

    for (int32_t id : routed) {
        const int32_t cur = st.slot_of[id];
        if (cur != -1 && ggml_bitset_get(st.valid.data(), cur)) {
            c.skipped++;
            continue;
        }
        int32_t s = find_free_slot();
        if (s == -1) {
            s = evict_one();
        }
        if (s == -1) {
            // K smaller than the routed set: the ids cannot be staged. Fail
            // loud — silently dropping an expert would corrupt decode.
            GGML_ABORT("golem P2 slab: routed set (%d) exceeds slots (K=%d); "
                       "raise --vram-experts or lower the ubatch (design D-7)",
                       (int) routed.size(), (int) K);
        }
        // Stealing a slot may orphan its previous owner's assignment; the
        // evict path already cleared it. Free-slot reuse below can also hit a
        // slot still referenced by a dead slot_of entry — clear that too.
        for (int32_t id2 = 0; id2 < n_experts; ++id2) {
            if (st.slot_of[id2] == s && id2 != id) {
                st.slot_of[id2] = -1;
            }
        }
        st.slot_of[id] = s;
        ggml_bitset_set(st.valid.data(), s);
        on_copy(s, id);
        c.copied++;
    }

    // Build the remapped ids (values in [0, K]; ids_dev adds the padding slot
    // as index K — never referenced by valid remaps).
    st.remap.resize(ids_count);
    for (int64_t i = 0; i < ids_count; ++i) {
        const int32_t v = ids[i];
        GGML_ASSERT(v >= 0 && v < n_experts);
        const int32_t s = st.slot_of[v];
        GGML_ASSERT(s != -1 && "routed bitset must cover every ids value");
        st.remap[i] = s;
    }
    return c;
}

// Non-template, non-static entry point linked from the ctest fuzz TU
// (tests/golem-p2-cache.cpp): counts physical writes without performing them
// (on_copy is a no-op) — the state invariants do not depend on byte contents.
inline GolemSlabCounters golem_slab_step_nop(GolemSlabEntry &st, const std::vector<int32_t> &routed,
                                             const int32_t *ids, int64_t ids_count,
                                             const int32_t *order, int32_t K, int32_t n_experts,
                                             size_t expert_size, const void *host_data) {
    return golem_slab_step(st, routed, ids, ids_count, order, K, n_experts, expert_size,
                           host_data, [](int32_t, int32_t) {});
}

} // namespace golem
#endif /* GOLEM_DEVICE_CACHE_HPP */
