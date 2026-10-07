// Golem per-model hook context (ABI v5, design §3.4 amendment): ALL hook
// state — function pointers, device-cache slab maps, budget, P4 stats,
// eviction-order buffers — lives in one `golem_ctx` owned by the engine and
// attached to a llama_model / its scheds. Included (never copied) by
// ggml-backend.cpp and the vendor ctest TUs. Decode hot path: no heap
// allocation, no hash lookup, no mutex per callback — the only synchronized
// entry point is the off-decode-thread order publisher.
//
// In-flight tracking is one atomic add/sub around the WHOLE sched
// graph-compute/alloc entry (covers graph-build reads of ctx fields, not just
// compute_splits), so golem_unregister_hooks can only complete between
// engine steps.
#ifndef GOLEM_HOOK_CONTEXT_HPP
#define GOLEM_HOOK_CONTEXT_HPP

#include "golem-hooks.h"
#include "golem-device-cache.hpp" // golem::GolemSlabEntry + GolemLayerSlots (ODR: shared with the ctest fuzz TU)

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

// P4 counters; layout mirrors the Rust [AtomicU64; 3] FFI cast (the
// placement stats are process-global and stay in ggml-backend.cpp).
struct golem_device_cache_stats {
    std::atomic<uint64_t> bytes_copied; // host->device expert bytes written
    std::atomic<uint64_t> bytes_skipped; // bytes skipped because bit-valid
    std::atomic<uint64_t> evictions; // experts invalidated to stay under budget
};
static_assert(sizeof(golem_device_cache_stats) == 3 * sizeof(uint64_t), "golem_device_cache_stats layout must match the Rust [AtomicU64; 3] FFI cast");
static_assert(std::atomic<uint64_t>::is_always_lock_free && alignof(std::atomic<uint64_t>) == alignof(uint64_t),
              "counters are read from Rust as plain AtomicU64: lock-free and u64-aligned");

// Detects unregister-from-inside-a-callback (deadlock by construction):
// incremented around every hook invocation in ggml-backend.cpp.
inline thread_local int golem_cb_depth = 0;
struct golem_cb_guard {
    golem_cb_guard() { ++golem_cb_depth; }
    ~golem_cb_guard() { --golem_cb_depth; }
};

struct golem_ctx {
    // --- hooks (written only by register/unregister; read on the decode
    // thread — register/unregister must not run concurrently with a step) ---
    golem_ensure_host_resident_fn ensure = nullptr;
    golem_routing_fn routing = nullptr;
    golem_copy_done_fn copy_done = nullptr;
    void *payload = nullptr;
    // P5 next-layer prediction (golem_set_predict; 0 = off)
    golem_predict_fn predict = nullptr;
    std::atomic<int32_t> pred_lookahead{0};
    std::atomic<int32_t> pred_margin{0};
    std::atomic<bool> hooked{false};

    // in-flight whole-step depth (join point for unregister) + the reader
    // epoch for the order buffers (bumped once per step entry)
    std::atomic<int> in_flight{0};
    std::atomic<uint64_t> epoch{0};

    // scheds currently attached via golem_sched_set_ctx; golem_ctx_free
    // aborts while this is nonzero
    std::atomic<int> attach_refs{0};

    // bumped on every register; the engine tags recycled scratch/pool
    // addresses with it to reject cross-session reuse
    std::atomic<uint64_t> generation{1};

    // unregister-join
    std::mutex join_mutex;
    std::condition_variable join_cv;

    // P2 budget (<= 0 = unset; the slab path aborts) + P4 counters
    std::atomic<int32_t> k_cfg{0};
    golem_device_cache_stats dc_stats{};

    // persistent device slab maps (moved here from ggml-backend.cpp file
    // statics; key of `dc` = HOST weights tensor data pointer)
    std::mutex dc_mutex;
    std::unordered_map<const void *, golem::GolemSlabEntry> dc;
    std::unordered_map<int, golem::GolemLayerSlots> layers;

    // --- per-(blk, kind) triple-buffered eviction orders ---
    // Publisher (off the decode thread) fills a FREE slot and swaps it in
    // with release. A slot is free iff it is not current AND no reader of the
    // current epoch has acquired it (readers publish the epoch on acquire);
    // when no slot is free the publisher skips and stays dirty — never
    // blocks, never overwrites a buffer a reader may still hold.
    struct order_slot {
        std::vector<int32_t> buf;
        std::atomic<uint64_t> reader_epoch{~uint64_t(0)};
        // Seqlock over buf: odd = a writer is mid-copy. Closes the window
        // where a reader publishes its epoch and loads cur while a publisher
        // that snapshotted an older reader_epoch is already copying into
        // that slot (epoch gate alone cannot see a reader that has not
        // published yet).
        std::atomic<uint64_t> gen{0};
    };
    struct order_entry {
        order_slot slots[3];
        std::atomic<int32_t> cur{-1}; // -1 = never published (order == NULL)
    };
    std::vector<order_entry> orders; // flat: [blk * 3 + (kind - 1)]
    const int32_t n_experts;
    std::atomic<bool> order_warned{false}; // one-time NULL-order warning

    golem_ctx(int32_t n_layers, int32_t n_experts_)
        : orders((size_t) (n_layers > 0 ? n_layers : 1) * 3), n_experts(n_experts_) {
        for (auto &e : orders) {
            for (auto &s : e.slots) {
                s.buf.resize(n_experts);
            }
        }
    }

    order_entry *order_at(int32_t blk, int32_t kind) {
        const size_t i = (size_t) blk * 3 + (size_t) (kind - 1);
        if (blk < 0 || kind < 1 || kind > 3 || i >= orders.size()) {
            return nullptr;
        }
        return &orders[i];
    }

    // true = published; false = skipped (no free buffer under the epoch gate)
    bool push_order(int32_t blk, int32_t kind, const int32_t *order, int32_t n) {
        order_entry *e = order_at(blk, kind);
        if (e == nullptr || n > n_experts) {
            return false;
        }
        const int32_t cur = e->cur.load(std::memory_order_relaxed);
        const uint64_t now = epoch.load(std::memory_order_relaxed);
        for (int t = 0; t < 3; ++t) {
            const int32_t cand = (cur + 1 + t) % 3;
            order_slot &s = e->slots[cand];
            if (cand == cur ||
                s.reader_epoch.load(std::memory_order_acquire) == now) {
                continue; // current, or a reader of this epoch may hold it
            }
            // Seqlock: mark the copy, fill, mark quiescent — a racing
            // reader retrying on gen catches the torn window.
            s.gen.fetch_add(1, std::memory_order_acq_rel);
            std::copy(order, order + n, s.buf.begin());
            if (n < n_experts) {
                std::fill(s.buf.begin() + n, s.buf.end(), 0);
            }
            s.gen.fetch_add(1, std::memory_order_acq_rel);
            e->cur.store(cand, std::memory_order_release);
            return true;
        }
        return false; // stays dirty; retried on the next publish
    }

    // current order or NULL before the first push; publishes the reader epoch
    // BEFORE validating, with a seqlock retry so a publisher that started a
    // copy before the epoch publish cannot hand back torn contents.
    const int32_t *acquire_order(int32_t blk, int32_t kind) {
        order_entry *e = order_at(blk, kind);
        if (e == nullptr) {
            return nullptr;
        }
        const uint64_t now = epoch.load(std::memory_order_relaxed);
        for (int tries = 0; tries < 8; ++tries) {
            const int32_t cur = e->cur.load(std::memory_order_acquire);
            if (cur < 0) {
                return nullptr;
            }
            order_slot &s = e->slots[cur];
            const uint64_t g1 = s.gen.load(std::memory_order_acquire);
            if (g1 & 1) {
                continue; // writer mid-copy on the current slot
            }
            const int32_t *p = s.buf.data();
            s.reader_epoch.store(now, std::memory_order_release);
            if (e->cur.load(std::memory_order_acquire) != cur ||
                s.gen.load(std::memory_order_acquire) != g1) {
                continue; // published away (or rewritten) under us — retry
            }
            return p;
        }
        return nullptr; // falls back to the identity order at the call site
    }
};

#endif /* GOLEM_HOOK_CONTEXT_HPP */
