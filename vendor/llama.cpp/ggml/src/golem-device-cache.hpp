// Golem patch P2 (design §3.4): SLOT-INDIRECTED device expert cache, one slot
// table per expert layer shared by its gate/up/down tensors.
// Included by BOTH ggml-backend.cpp (split-scheduler copy path) and
// tests/golem-p2-cache.cpp (fuzz harness): a single shared definition —
// hand-copied declarations would be an ODR drift (silent wrong-layout crash).
//
// The device cache is a slab of K+1 expert-plane slots per tensor (+1 non-NaN
// padding slot), and MUL_MAT_ID reads it
// through a REMAPPED ids tensor (values in [0, K)): routing is untouched —
// only the addressing moves. The three expert tensors of a layer consume the
// same routed ids (llama-graph build_moe_ffn), so slot assignment and
// eviction are decided once per layer; each tensor keeps only its own
// per-slot validity bits.
#ifndef GOLEM_DEVICE_CACHE_HPP
#define GOLEM_DEVICE_CACHE_HPP

#include "ggml-impl.h" // ggml_bitset_* (non-public; resolved via ggml/src include dir)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;

namespace golem {

// One expert layer's device slot table, shared by its gate/up/down slabs
// (they consume the same routed ids). valid[k]: bitset over slots — slab k's
// slot bytes are current for slot_of's owner.
struct GolemLayerSlots {
    int32_t K = 0, n_experts = 0;
    std::vector<int32_t> slot_of;   // expert → slot, -1 = unassigned
    std::vector<int32_t> owner_of;  // slot → expert, -1 = free
    // Slots [0, next_free) have all been assigned: an evicted slot is handed
    // straight to the evictor, so free slots only exist at/after the cursor.
    int32_t next_free = 0;
    std::vector<ggml_bitset_t> valid[3];
    // Per-step scratch (n_experts bits each), reused so a decode step does
    // no heap allocation after the first.
    std::vector<ggml_bitset_t> seen, is_routed;
};

// One (layer, kind) slab, keyed by the host tensor data pointer. A change of
// K or plane stride reallocates the slab and invalidates its validity bits, so
// they never describe stale device bytes. Slot assignment lives in
// GolemLayerSlots (per layer, shared across kinds).
struct GolemSlabEntry {
    int32_t K = 0;                 // addressable slots (padding slot excluded)
    size_t expert_size = 0;
    // persistent device slab (K+1 planes) in its own buffer, independent of
    // galloc plans — cached slot bytes survive graph rebuilds
    struct ggml_tensor *slab = nullptr;
    struct ggml_backend_buffer *buf = nullptr;
    struct ggml_context *ctx = nullptr;
    int64_t ids_ne0 = 0;           // remapped-ids tensor shape fingerprint
    int64_t ids_ne1 = 0;
    // remapped ids device tensor (int32, same 2-D shape as the ids node)
    struct ggml_tensor *ids_dev = nullptr;
    struct ggml_backend_buffer *ids_buf = nullptr;
    struct ggml_context *ids_ctx = nullptr;
    // Pinned host staging for the remapped-ids H2D (AGENTS.md: pageable H2D
    // is a bug; size is not gated). Bounded by ids_ne0 × ids_ne1 × 4 B,
    // allocated once per ids-shape (re)allocation alongside ids_dev/ids_buf.
    void *ids_pin = nullptr;                           // host-pinned base (int32 remaps)
    size_t ids_pin_bytes = 0;
    struct ggml_backend_buffer *ids_pin_buf = nullptr; // owns ids_pin, freed via buffer
    struct ggml_backend_buffer_type *ids_pin_buft = nullptr; // buffer type captured at alloc (pinnedness assert)
    uint64_t ids_pin_step = 0;                         // compute_splits sequence of the last ids_pin write (same-step reuse detection)
    std::vector<int32_t> remap;    // host staging for the remapped ids
};

// Per-step accounting; the caller folds these into the P4 atomic counters.
struct GolemSlabCounters {
    uint64_t copied = 0;   // expert planes written this step
    uint64_t skipped = 0;  // routed experts whose slot bytes were current
    uint64_t evicted = 0;  // slots stolen from non-routed experts
};

// Gathers a strided int32 ids tensor (ggml_argsort_top_k is a view: nb[1] =
// n_expert*4) into `out` as ne0*ne1 densely packed values, row-major — the
// layout of the remapped-ids device tensor. `out` keeps its capacity across
// calls, so steady-state decode does no heap allocation.
inline void golem_gather_ids_dense(const void *raw, int64_t ne0, int64_t ne1,
                                   size_t nb0, size_t nb1, std::vector<int32_t> &out) {
    out.resize((size_t) (ne0 * ne1));
    const char *base = (const char *) raw;
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            memcpy(&out[(size_t) (i1 * ne0 + i0)], base + i1 * nb1 + i0 * nb0, sizeof(int32_t));
        }
    }
}

// A slab (re)allocation: slab k's device bytes are gone — clear its whole
// validity word so every routed expert is re-copied on the next step.
inline void golem_layer_invalidate(GolemLayerSlots &L, int kind) {
    std::fill(L.valid[kind].begin(), L.valid[kind].end(), 0);
}

// One scheduler step for one (layer, kind) slab against the layer's shared
// slot table `L`.
//
// - `routed` = sorted used-expert ids (from the ids bitset); `ids`/`ids_count`
//   = the raw ids values; `order` = Rust's evict-first permutation of
//   [0, n_experts) (NULL / duplicate / out-of-range aborts — C-side choke
//   point); `K` = addressable slots of the slab; `kind` in [0, 3).
// - Slot assignment runs ONCE per layer (first plane of the step): every
//   routed expert without a slot takes a free slot (owner_of[s] == -1), else
//   the first assigned non-routed expert in `order` is evicted — its
//   assignment dropped and its bit cleared in ALL THREE valid[k] words, so
//   the later planes re-copy instead of reading freed slots. Routed set
//   larger than K cannot be staged → GGML_ABORT (fail-loud; prefill slab
//   sizing is design §2.4).
// - Validity is per kind: copy iff !valid[kind][slot].
// - on_copy(slot, id): physical host→device write of expert `id`'s plane into
//   `slot`. Called once per non-current routed expert.
// - remap = the remapped ids (same element order/count as `ids`), ready for
//   the caller's H2D of the ids_dev tensor.
template <typename OnCopy>
GolemSlabCounters golem_slab_step(GolemLayerSlots &L, int kind, const std::vector<int32_t> &routed,
                                  const int32_t *ids, int64_t ids_count, const int32_t *order,
                                  int32_t K, int32_t n_experts, std::vector<int32_t> &remap,
                                  OnCopy &&on_copy) {
    GolemSlabCounters c;
    GGML_ASSERT(K > 0 && K <= n_experts);
    GGML_ASSERT(kind >= 0 && kind < 3);

    // Validate order FIRST: all ids distinct, all in [0, n_experts).
    {
        std::vector<ggml_bitset_t> &seen = L.seen;
        seen.assign(ggml_bitset_size(n_experts), 0);
        GGML_ASSERT(order != NULL);
        for (int32_t i = 0; i < n_experts; ++i) {
            GGML_ASSERT(order[i] >= 0 && order[i] < n_experts);
            GGML_ASSERT(!ggml_bitset_get(seen.data(), order[i]));
            ggml_bitset_set(seen.data(), order[i]);
        }
    }

    // Fresh or re-shaped layer: reset the shared table. All three valid words
    // are cleared together — a re-mapped slot must never keep a stale bit
    // from a previous geometry.
    if ((int32_t) L.slot_of.size() != n_experts || L.K != K) {
        L.slot_of.assign(n_experts, -1);
        L.owner_of.assign(K, -1);
        L.next_free = 0;
        for (int k = 0; k < 3; ++k) {
            L.valid[k].assign(ggml_bitset_size(K + 1), 0);
        }
        L.K = K;
        L.n_experts = n_experts;
    }

    std::vector<ggml_bitset_t> &is_routed = L.is_routed;
    is_routed.assign(ggml_bitset_size(n_experts), 0);
    for (int32_t id : routed) {
        GGML_ASSERT(id >= 0 && id < n_experts);
        ggml_bitset_set(is_routed.data(), id);
    }

    // Assign slots once per layer: routed experts first take free slots, then
    // evict by `order`. owner_of is slot_of's inverse over live assignments —
    // no per-copy orphan scan.
    for (int32_t id : routed) {
        if (L.slot_of[id] != -1) {
            continue;
        }
        int32_t s = L.next_free < K ? L.next_free++ : -1;
        if (s == -1) {
            for (int32_t i = 0; i < n_experts && s == -1; ++i) {
                const int32_t cand = order[i];
                const int32_t cs = L.slot_of[cand];
                if (cs != -1 && !ggml_bitset_get(is_routed.data(), cand)) {
                    L.slot_of[cand] = -1;
                    L.owner_of[cs] = -1;
                    for (int k = 0; k < 3; ++k) {
                        ggml_bitset_clear(L.valid[k].data(), cs);
                    }
                    c.evicted++;
                    s = cs;
                }
            }
        }
        if (s == -1) {
            // K smaller than the routed set: the ids cannot be staged. Fail
            // loud — silently dropping an expert would corrupt decode. A
            // chunk larger than K/top-k routes more than K distinct experts
            // per layer; the engine caps the prefill chunk at K/top-k so
            // this only fires on a mispriced K.
            GGML_ABORT("golem P2 slab: routed set (%d) exceeds slots (K=%d) — "
                       "a decode batch routed more distinct experts than the "
                       "slab holds; the engine caps prefill chunks at K/top-k, "
                       "so a caller bypassed that cap or K was mispriced "
                       "(design sec 2.4 prefill)",
                       (int) routed.size(), (int) K);
        }
        L.slot_of[id] = s;
        L.owner_of[s] = id;
    }

    // Per-kind validity: copy only what this slab lost (or never had).
    for (int32_t id : routed) {
        const int32_t s = L.slot_of[id];
        if (!ggml_bitset_get(L.valid[kind].data(), s)) {
            ggml_bitset_set(L.valid[kind].data(), s);
            on_copy(s, id);
            c.copied++;
        } else {
            c.skipped++;
        }
    }

    // Build the remapped ids (values in [0, K); ids_dev's padding slot K is
    // never referenced by remaps).
    remap.resize(ids_count);
    for (int64_t i = 0; i < ids_count; ++i) {
        const int32_t v = ids[i];
        GGML_ASSERT(v >= 0 && v < n_experts);
        const int32_t s = L.slot_of[v];
        GGML_ASSERT(s != -1 && "routed bitset must cover every ids value");
        remap[i] = s;
    }
    return c;
}

// Non-template, non-static entry point linked from the ctest fuzz TU
// (tests/golem-p2-cache.cpp): counts physical writes without performing them
// (on_copy is a no-op) — the state invariants do not depend on byte contents.
inline GolemSlabCounters golem_slab_step_nop(GolemLayerSlots &L, int kind,
                                             const std::vector<int32_t> &routed,
                                             const int32_t *ids, int64_t ids_count,
                                             const int32_t *order, int32_t K, int32_t n_experts,
                                             std::vector<int32_t> &remap) {
    return golem_slab_step(L, kind, routed, ids, ids_count, order, K, n_experts, remap,
                           [](int32_t, int32_t) {});
}

} // namespace golem
#endif /* GOLEM_DEVICE_CACHE_HPP */
