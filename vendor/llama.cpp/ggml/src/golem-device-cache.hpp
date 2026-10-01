// Golem patch P2 v3 (design §3.4 v3): SLOT-INDIRECTED device expert cache,
// one slot table per expert layer shared by its gate/up/down tensors (D-11).
// Included by BOTH ggml-backend.cpp (split-scheduler copy path) and
// tests/golem-p2-cache.cpp (fuzz harness): a single shared definition —
// hand-copied declarations would be an ODR drift (silent wrong-layout crash).
//
// The device cache is a slab of K+1 expert-plane slots per tensor (the shrank
// scheduler input cpy; +1 non-NaN padding slot), and MUL_MAT_ID reads it
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
    std::vector<ggml_bitset_t> valid[3];
};

// One (layer, kind) slab. Keyed by the host tensor fingerprint; any sched
// slab pointer change resets the state, so
// validity bits never describe stale device bytes. Slot assignment lives in
// GolemLayerSlots (per layer, shared across kinds).
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
//   sizing is design §2.4, deferred to M3.5).
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
        std::vector<ggml_bitset_t> seen(ggml_bitset_size(n_experts), 0);
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
        for (int k = 0; k < 3; ++k) {
            L.valid[k].assign(ggml_bitset_size(K + 1), 0);
        }
        L.K = K;
        L.n_experts = n_experts;
    }

    std::vector<ggml_bitset_t> is_routed(ggml_bitset_size(n_experts), 0);
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
        int32_t s = -1;
        for (int32_t t = 0; t < K; ++t) {
            if (L.owner_of[t] == -1) {
                s = t;
                break;
            }
        }
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
            // loud — silently dropping an expert would corrupt decode.
            GGML_ABORT("golem P2 slab: routed set (%d) exceeds slots (K=%d); "
                       "raise --vram-experts or lower the ubatch (design sec 2.4 prefill)",
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

    // Build the remapped ids (values in [0, K]; ids_dev adds the padding slot
    // as index K — never referenced by valid remaps).
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
