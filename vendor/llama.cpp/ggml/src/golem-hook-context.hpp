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
#include "golem-copy-stream.h"
#include "golem-device-wait.h"
#include "golem-device-cache.hpp" // golem::GolemSlabEntry + GolemLayerSlots (ODR: shared with the ctest fuzz TU)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_event;
struct golem_cpu_expert_iface; // golem-cpu-expert.h

// P4 counters; layout mirrors the Rust [AtomicU64; 6] FFI cast (the
// placement stats are process-global and stay in ggml-backend.cpp).
struct golem_device_cache_stats {
    std::atomic<uint64_t> bytes_copied; // host->device expert bytes written
    std::atomic<uint64_t> bytes_skipped; // bytes skipped because bit-valid
    std::atomic<uint64_t> evictions; // experts invalidated to stay under budget
    std::atomic<uint64_t> cache_experts_copied; // expert planes copied into the device cache
    std::atomic<uint64_t> stream_experts_copied; // expert planes copied by streamed prefill
    std::atomic<uint64_t> early_h2d_units; // expert planes H2D-queued from ensure's hits_ready (ABI v17)
};
static_assert(sizeof(golem_device_cache_stats) == 6 * sizeof(uint64_t), "golem_device_cache_stats layout must match the Rust [AtomicU64; 6] FFI cast");
static_assert(offsetof(golem_device_cache_stats, cache_experts_copied) == 3 * sizeof(uint64_t) &&
              offsetof(golem_device_cache_stats, stream_experts_copied) == 4 * sizeof(uint64_t) &&
              offsetof(golem_device_cache_stats, early_h2d_units) == 5 * sizeof(uint64_t),
              "golem_device_cache_stats member order must match the Rust FFI cast");
static_assert(std::atomic<uint64_t>::is_always_lock_free && alignof(std::atomic<uint64_t>) == alignof(uint64_t),
              "counters are read from Rust as plain AtomicU64: lock-free and u64-aligned");

// Detects unregister-from-inside-a-callback (deadlock by construction):
// incremented around every hook invocation in ggml-backend.cpp.
GOLEM_API inline thread_local int golem_cb_depth = 0;
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
    golem_evict_batch_fn evict_batch = nullptr; // exclusive residency (ABI v8)
    golem_evict_abandon_fn evict_abandon = nullptr; // abandoned D2H (ABI v9)
    golem_stream_source_fn stream_source = nullptr; // prefill stream (ABI v11: returns 1 = abort)
    golem_stream_release_fn stream_release = nullptr; // prefill stream (ABI v10)
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
    // ABI v12: device-slab allocations that failed (compute returned
    // ALLOC_FAILED), and the test seam forcing the next n to fail.
    std::atomic<uint64_t> slab_alloc_failures{0};
    std::atomic<int32_t> slab_fail_inject{0};
    golem_device_cache_stats dc_stats{};

    // --- prefill stream (design section 7.6, ABI v10) ---
    // Double-buffered device slabs, one plane each: plane p computes from
    // slab p%2 while p+1 fills. Allocated lazily on the first streamed
    // split's backend (the allocation needs it), freed by
    // golem_reset_device_cache. `stream_seq` is the global plane counter --
    // it keys the stream_source/stream_release pair and picks the slab.
    struct golem_stream_slab {
        struct ggml_context * ctx = nullptr;
        struct ggml_backend_buffer * buf = nullptr;
        struct ggml_tensor * tensor = nullptr; // typed view, rebuilt on shape change
        uint64_t fp = 0;                       // (type, ne0, ne1) fingerprint
        struct ggml_backend_event * event = nullptr; // recorded after the compute reading this slab
        bool used = false;
    };
    std::atomic<int32_t> stream_on{0};
    std::atomic<uint64_t> stream_seq{0};
    float stream_routed_max = 0.85f;
    int32_t stream_n_experts = 0;
    int32_t stream_max_stride = 0;
    golem_stream_slab stream_slabs[2];

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

    // decode-thread-only; kept here so the copy loop can flag which slabs a
    // split streamed (their events are recorded after that split's compute).
    bool stream_used_in_split[2] = {false, false};

    // --- dedicated H2D copy stream (async_h2d; golem_set_async_h2d) ---
    // Decode-thread-only. `cs`/`cs_backend`/events are bound lazily by the
    // first split that copies on the copy stream and released by
    // golem_reset_device_cache (after draining the stream). Ring of timed
    // event pairs, one per copying split; read by golem_take_h2d_timing.
    static constexpr int GOLEM_CS_RING = 256;
    int32_t async_h2d = 0;
    const struct golem_copy_stream_iface * cs = nullptr;
    ggml_backend_t cs_backend = nullptr;
    void * cs_ev_ready = nullptr; // compute stream -> copy stream (WAR, evict-before-refill)
    void * cs_ev_done = nullptr;  // copy stream -> compute stream (RAW)
    void * cs_t0[GOLEM_CS_RING] = {};
    void * cs_t1[GOLEM_CS_RING] = {};
    int32_t cs_t_n = 0;
    bool cs_t_overflow = false;
    uint64_t cs_bytes = 0;

    // --- VRAM victim ring (golem_set_victim_ring, ABI v18) ---
    // Decode-thread-only. One device buffer of vr_slots x 3 x vr_stride
    // bytes on cs_backend: a parked exclusive eviction reserves a slot, each
    // plane is copied slab -> ring on the compute stream (the stolen slab
    // slot is refillable right after) and ring -> host on the D2H stream.
    // Per-slot vectors are sized by the setter; buffer + events are created
    // at the first park and released by golem_reset_device_cache. A slot is
    // FREE (unused this compute), RESERVED (owned by a parked entry) or
    // RELEASING (its D2H is issued; vr_rel_ev recorded after it on the D2H
    // stream — reuse makes the compute stream wait on it). All slots return
    // to FREE at compute entry after the backend + copy/D2H stream drain.
    enum : uint8_t { GOLEM_VR_FREE = 0, GOLEM_VR_RESERVED = 1, GOLEM_VR_RELEASING = 2 };
    int32_t vr_slots = 0;  // <= 0 = off (inclusive-cost same-stream D2H)
    int32_t vr_stride = 0; // bytes per plane slot
    bool vr_off = false;   // bind/tensor setup failed: ring disabled until reset
    bool vr_alloc_failed = false; // device allocation failed: sticky across reset
    int32_t vr_next = 0;   // round-robin start for RELEASING-slot reuse
    struct ggml_context * vr_ctx = nullptr;
    struct ggml_backend_buffer * vr_buf = nullptr;
    struct ggml_tensor * vr_tensor = nullptr; // I8 view over the whole buffer
    std::vector<uint8_t> vr_state;            // [vr_slots]
    std::vector<void *> vr_rel_ev;            // [vr_slots]
    void * vr_ev_fence = nullptr;             // compute stream -> D2H stream (ring D2D landed)
    void * vr_t0[GOLEM_CS_RING] = {};         // timed pairs, one per issuing drain
    void * vr_t1[GOLEM_CS_RING] = {};
    int32_t vr_t_n = 0;
    bool vr_t_overflow = false; // a ring drain went untimed (timing ring full)
    uint64_t vr_units = 0; // planes copied through the ring
    uint64_t vr_full = 0;  // parked entries that found every slot RESERVED
    uint64_t vr_bytes = 0;

    // --- O1 CPU expert math (golem_set_cpu_expert_math, design §1 O1) ---
    // Config is init-only; everything else is decode-thread-only. The worker
    // and iface are bound lazily on the first eligible compute. Per-blk
    // pinned accumulators (n_blk x n_embd f32, one host buffer of the split
    // device) are never reused within a compute; freed by
    // golem_reset_device_cache. Invariant between computes:
    // cem_pending_blk == -1 (every submitted job is joined).
    std::atomic<int32_t> cem_on{0}; // read at graph build
    float cem_frac = 0.0f;
    int32_t cem_threads = 0;
    golem_cpu_lease_fn cem_lease = nullptr;
    const struct golem_cpu_expert_iface * cem_if = nullptr;
    void * cem_worker = nullptr;
    struct ggml_backend_buffer * cem_acc_buf = nullptr;
    float * cem_acc = nullptr; // [n_blk * cem_acc_n_embd]
    int32_t cem_acc_n_embd = 0;
    int32_t cem_pending_blk = -1; // blk whose job is submitted and not yet joined
    uint64_t cem_experts = 0;
    uint64_t cem_bytes = 0;
    double cem_ms = 0.0;
    double cem_join_wait_ms = 0.0;
    uint64_t cem_padded = 0;

    // --- §11 host-sync timing (golem_take_sync_gap_stats), decode-thread-only ---
    // Each MoE layer's ids readback synchronizes the ids backend; the gap runs
    // from the first such return to the next split launch (no compute queued
    // there; the async_h2d copy stream may still run).
    bool sg_pending = false;
    std::chrono::steady_clock::time_point sg_t_ids{};
    double sg_sync_ms = 0.0;
    double sg_gap_ms = 0.0;
    double sg_gap_max_ms = 0.0;
    uint64_t sg_n = 0;

    // --- device-side landed wait (golem_set_device_wait, ABI v20) ---
    // One mapped allocation, sized at arm time by n_blk x (n_used + n_pred):
    //   req[n_blk + 1]             newest published compute seq per blk; active word last
    //   ids[n_blk][stride]         routed ids then P5 prediction ids
    //   remap[n_blk][3][n_used]    slab slot per routed position (reply)
    //   landed[n_blk][3][n_used]   seq once the unit's plane is in the slab
    // GLM-5.3-Flash (n_used 8, n_pred 12, 46 blk): (47 + 46 x 68) x 4 B ≈ 12.7 KB.
    // Decode-thread fields are written before the blk's publish and read by
    // the serving thread after blk_seq (release/acquire) names that publish.
    struct golem_dw_blk {
        std::atomic<uint32_t> host_seq{0}; // compute seq whose setup below is current
        uint32_t served_seq = 0;           // serving-thread-only
        struct ggml_tensor * inp[3] = {};  // host weights tensors (gate, up, down)
        golem::GolemSlabEntry * st[3] = {};
        int32_t K = 0;
        int32_t n_expert = 0;
        int32_t n_pred_t = 0;
        int32_t pred_target[GOLEM_DW_MAX_PRED] = {};
        int32_t pred_n[GOLEM_DW_MAX_PRED] = {};
    };
    int32_t dw_mode = 0;      // resolved: 0 off, 1 stream wait, 2 wait kernel
    int32_t dw_n_used = 0;
    int32_t dw_n_pred = 0;
    int32_t dw_stride = 0;    // ids words per blk
    int32_t dw_n_blk = 0;
    const struct golem_device_wait_iface * dw = nullptr;
    uint32_t * dw_mem = nullptr;
    uint32_t dw_seq = 0;      // decode-thread compute sequence (published ones only)
    std::vector<golem_dw_blk> dw_blk;
    std::atomic<uint32_t> dw_served{0};   // serves completed in the current compute
    std::atomic<int32_t> dw_failed{0};    // a serve's page-in failed this compute
    std::atomic<uint64_t> dw_served_total{0};
    std::atomic<uint64_t> dw_host_computes{0};
    // Serving-thread scratch, preallocated at arm time (n_used / n_experts).
    std::vector<int32_t> dw_dense, dw_routed, dw_identity, dw_miss_kinds, dw_miss_ids;
    std::vector<int32_t> dw_miss_k, dw_miss_j, dw_miss_slot;
    std::vector<const void *> dw_miss_out;
    std::vector<std::pair<int32_t, int32_t>> dw_pending;

    std::vector<uint8_t> dw_miss_issued;
    uint32_t dw_published = 0;            // decode-thread: requests published this compute
    // Serving-thread copy accounting: written by serves, read and reset by
    // golem_take_h2d_timing on the decode thread after golem_dw_quiesce (the
    // acquire on dw_served orders every serve's writes before the read).
    uint64_t dw_cs_bytes = 0;
    void * dw_t0[GOLEM_CS_RING] = {};
    void * dw_t1[GOLEM_CS_RING] = {};
    int32_t dw_t_n = 0;
    bool dw_t_overflow = false;

    // Word offsets: req[n_blk + 1] (the last is the active word), ids, remap, landed.
    uint32_t * dw_req() const { return dw_mem; }
    uint32_t * dw_ids(int32_t blk) const { return dw_mem + (dw_n_blk + 1) + (size_t) blk * dw_stride; }
    uint32_t * dw_remap(int32_t blk, int32_t k) const {
        return dw_mem + (dw_n_blk + 1) + (size_t) dw_n_blk * dw_stride + ((size_t) blk * 3 + k) * dw_n_used;
    }
    uint32_t * dw_landed(int32_t blk, int32_t k) const {
        return dw_remap(blk, k) + (size_t) dw_n_blk * 3 * dw_n_used;
    }
    size_t dw_words() const {
        return (size_t) (dw_n_blk + 1) + (size_t) dw_n_blk * dw_stride + (size_t) dw_n_blk * 6 * dw_n_used;
    }

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
