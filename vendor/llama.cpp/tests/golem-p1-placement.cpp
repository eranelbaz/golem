// Golem ABI v5 vendor tests (P1 placement + ctx lifecycle):
//   1. placement counters: ggml_backend_sched_split_graph counts expert
//      MUL_MAT_ID nodes by the device type of their assigned backend.
//   2. two golem_ctxs with distinct hook closures interleave; each observes
//      only its own state.
//   3. unregister-join: a sleeping callback blocks unregister until done.
//   4. unregister from inside a callback aborts (SIGABRT).
//   5. golem_ctx_free with a sched still attached aborts (SIGABRT).
//   6. allocation-free steady state: repeated ggml_backend_sched_graph_compute
//      over a synthetic MUL_MAT_ID slab graph with an overriding operator new
//      allocates nothing after the first step.
//   7. ABI v17 hits_ready, pooled (mode 1): hits are H2D-queued inside the
//      callback, before the stub fills the misses; a run split by a miss is
//      not issued early (MMQ padding); output equals the hookless reference.
//   8. hits_ready, slab (mode 2) with a parked eviction D2H: the eviction
//      destination holds the old bytes, the slab the new ones.
//   9. ensure failing after hits_ready ran: compute FAILED, then recovers.
//  10. hits_ready under async_h2d (copy stream): needs a GPU device; a
//      CPU-only build skips it only with GOLEM_CTEST_CPU_ONLY=1 (reported).
#include <atomic>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "golem-hook-context.hpp"
#include "golem-test-fakegpu.hpp"

static uint64_t load(int i) {
    const auto *c = reinterpret_cast<const std::atomic<uint64_t> *>(golem_get_expert_placement_stats());
    return c[i].load(std::memory_order_relaxed);
}

static int g_failures = 0;
#define REQUIRE(cond, ...)                                    \
    do {                                                      \
        if (!(cond)) {                                        \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);       \
            printf(__VA_ARGS__);                              \
            printf("\n");                                     \
            g_failures++;                                     \
        }                                                     \
    } while (0)

// --- allocation counter (global operator new override) ---
static std::atomic<long> g_allocs{0};
static std::atomic<bool> g_count_allocs{false};
void * operator new(size_t size) {
    if (g_count_allocs.load(std::memory_order_relaxed)) {
        g_allocs.fetch_add(1, std::memory_order_relaxed);
        void * bt[12];
        int n = backtrace(bt, 12);
        backtrace_symbols_fd(bt, n, 2);
        for (int k = 0; k < n; ++k) { Dl_info info; if (dladdr(bt[k], &info)) fprintf(stderr, "BT%d %s %p\n", k, info.dli_fname ? info.dli_fname : "?", (void*)((char*)bt[k] - (char*)info.dli_fbase)); }
    }
    void * p = malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}
void * operator new[](size_t size) { return operator new(size); }
void operator delete(void * p) noexcept { free(p); }
void operator delete[](void * p) noexcept { free(p); }
void operator delete(void * p, size_t) noexcept { free(p); }
void operator delete[](void * p, size_t) noexcept { free(p); }

// --- synthetic expert graph: blk.0.ffn_up_exps.weight [8,4,4], 2 routed ids ---
static const int64_t NE0 = 8, NE1 = 4;
static const int32_t N_EXP = 4, N_LAYERS = 4;

static void fill_w(float * w) {
    for (int i = 0; i < (int) (NE0 * NE1 * N_EXP); ++i) w[i] = (float) (i % 7) * 0.25f;
}

// Weights tensor in a WEIGHTS-usage host buffer on cpu_a; the MUL_MAT_ID node
// is forced onto cpu_b so the split's copy path (and its golem branches)
// engages.
struct ExpertSched {
    ggml_backend_t cpu_a = nullptr; // + the fake "GPU" backend (static)
    ggml_backend_sched_t sched = nullptr;
    ggml_context * ctx = nullptr;      // no_alloc metadata + graph
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_tensor * w = nullptr, * b = nullptr, * ids = nullptr, * out = nullptr;
    ggml_cgraph * gf = nullptr;

    void build() {
        ggml_init_params ip = {8 * 1024 * 1024, nullptr, true};
        ctx = ggml_init(ip);
        w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NE0, NE1, N_EXP);
        ggml_set_name(w, "blk.0.ffn_up_exps.weight");
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, 2);
        ggml_set_name(b, "blk.0.cur_ffn_inp");
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
        ggml_set_name(ids, "blk.0.ffn_router");
        out = ggml_mul_mat_id(ctx, w, b, ids);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        cpu_a = ggml_backend_cpu_init();
        ggml_backend_t backs[2] = {golem_test::fake_backend(), cpu_a};
        sched = ggml_backend_sched_new(backs, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        ggml_backend_sched_set_tensor_backend(sched, out, golem_test::fake_backend());

        wbuf = ggml_backend_alloc_buffer(cpu_a, ggml_nbytes(w) * 2);
        ggml_backend_tensor_alloc(wbuf, w, ggml_backend_buffer_get_base(wbuf));
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        float tmp[NE0 * NE1 * N_EXP];
        fill_w(tmp);
        ggml_backend_tensor_set(w, tmp, 0, ggml_nbytes(w));

        REQUIRE(ggml_backend_sched_alloc_graph(sched, gf), "alloc_graph");
        const int32_t hids[2] = {0, 1};
        ggml_backend_tensor_set(ids, hids, 0, ggml_nbytes(ids));
        float hb[NE0 * 2];
        for (int i = 0; i < (int) (NE0 * 2); ++i) hb[i] = 0.5f;
        ggml_backend_tensor_set(b, hb, 0, ggml_nbytes(b));
    }
    ~ExpertSched() {
        golem_sched_detach_ctx(sched);
        golem_sched_detach_ctx(sched);
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(wbuf);
        ggml_backend_free(cpu_a);
        ggml_free(ctx);
    }
};

// --- hook state (the Rust-payload stand-in) ---
struct HState {
    std::atomic<int> ensure_calls{0}, routing_calls{0}, copy_done_calls{0};
    std::atomic<bool> cb_returned{false};
    std::vector<int32_t> miss_kinds, miss_ids;
    int32_t last_blk = -1;
    int64_t last_n_ids = -1;
    std::vector<int32_t> last_ids;
};

static std::vector<char> g_pool; // host expert pool stand-in
static std::atomic<int> g_join_sleep_ms{0};
static std::atomic<bool> g_cb_entered{false}; // leg 3: observed in-flight entry
golem_hook_handle_t g_h_abort = nullptr; // leg 5: unregistered from inside the callback
static void pool_init() {
    const size_t plane = (size_t) NE0 * NE1 * sizeof(float);
    g_pool.assign(plane * 3 * N_EXP, (char) 0x5a);
}

static int32_t h_ensure(void * p, const void * tensor, const int32_t * mk, const int32_t * mi,
                     int32_t n, const void ** out, golem_hits_ready_fn hits_ready, void * hits_ud) {
    auto * s = (HState *) p;
    s->ensure_calls.fetch_add(1);
    s->miss_kinds.assign(mk, mk + n);
    s->miss_ids.assign(mi, mi + n);
    const size_t plane = (size_t) NE0 * NE1 * sizeof(float);
    for (int32_t i = 0; i < n; ++i) {
        const int kind = mk[i]; // 1=gate 2=up 3=down
        REQUIRE(mi[i] >= 0 && mi[i] < N_EXP, "miss id out of range");
        out[i] = g_pool.data() + plane * ((size_t) (kind - 1) * N_EXP + mi[i]);
    }
    (void) tensor;
    if (hits_ready != nullptr) {
        hits_ready(hits_ud, out);
    }
    g_cb_entered.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(g_join_sleep_ms.load()));
    s->cb_returned.store(true);
    return 0;
}

static int32_t h_abort_ensure(void *, const void *, const int32_t *, const int32_t *,
                           int32_t, const void **, golem_hits_ready_fn, void *) {
    golem_unregister_hooks(g_h_abort); // must GGML_ABORT (join waits for the calling step)
    return 0;
}

static void h_routing(void * p, int32_t blk, const void * tensor, const int32_t * dense,
                      int64_t n_ids, int32_t n_experts) {
    auto * s = (HState *) p;
    s->routing_calls.fetch_add(1);
    s->last_blk = blk;
    s->last_n_ids = n_ids;
    s->last_ids.assign(dense, dense + n_ids);
    REQUIRE(n_experts == N_EXP, "routing n_experts");
    (void) tensor;
}

static void h_copy_done(void * p) { ((HState *) p)->copy_done_calls.fetch_add(1); }

// --- ABI v17 hits_ready legs (7-10) ---
static const int32_t HX = 8; // experts per plane
static const size_t HX_PLANE = (size_t) NE0 * NE1 * sizeof(float);

static float hx_weight(size_t i) { return (float) (i % 7) * 0.25f + 0.125f * (float) (i / (NE0 * NE1)); }

// Pool stand-in: plane (kind, e) holds the bytes of the weights' expert e, so
// the hookless upstream copy is the reference.
static std::vector<float> & hx_pool() {
    static std::vector<float> p;
    if (p.empty()) {
        p.resize((size_t) 3 * HX * NE0 * NE1);
        for (int k = 0; k < 3; ++k) {
            for (size_t i = 0; i < (size_t) HX * NE0 * NE1; ++i) {
                p[(size_t) k * HX * NE0 * NE1 + i] = hx_weight(i);
            }
        }
    }
    return p;
}
static const void * hx_plane(int kind, int32_t e) {
    return (const char *) hx_pool().data() + HX_PLANE * ((size_t) (kind - 1) * HX + e);
}

struct HitState {
    golem_ctx_t c = nullptr;
    std::vector<int32_t> miss;  // ids pending (disk read) at hits_ready time
    int32_t rc = 0;             // ensure return code
    int ensure_calls = 0;
    std::vector<char> order;    // 'c' = hits_ready, 'f' = miss fill
    uint64_t early_at_cb = 0;   // early_h2d_units right after hits_ready returned
    std::vector<int32_t> nulls_at_cb;
};
static HitState g_hit;

static uint64_t hx_early(golem_ctx_t c) {
    const auto * st = reinterpret_cast<const std::atomic<uint64_t> *>(golem_get_device_cache_stats(c));
    return st[5].load(std::memory_order_relaxed);
}

static int32_t hx_ensure(void * p, const void *, const int32_t * mk, const int32_t * mi, int32_t n,
                         const void ** out, golem_hits_ready_fn hits_ready, void * hits_ud) {
    auto * s = (HitState *) p;
    s->ensure_calls++;
    auto is_miss = [&](int32_t id) {
        return std::find(s->miss.begin(), s->miss.end(), id) != s->miss.end();
    };
    for (int32_t i = 0; i < n; ++i) {
        out[i] = is_miss(mi[i]) ? nullptr : hx_plane(mk[i], mi[i]);
    }
    if (hits_ready != nullptr) {
        s->order.push_back('c');
        for (int32_t i = 0; i < n; ++i) {
            if (out[i] == nullptr) {
                s->nulls_at_cb.push_back(mi[i]);
            }
        }
        hits_ready(hits_ud, out);
        s->early_at_cb = hx_early(s->c);
    }
    s->order.push_back('f');
    for (int32_t i = 0; i < n; ++i) {
        out[i] = hx_plane(mk[i], mi[i]);
    }
    return s->rc;
}

static void hx_routing(void *, int32_t, const void *, const int32_t *, int64_t, int32_t) {}
static void hx_copy_done(void *) {}

struct HitEvict {
    std::vector<int32_t> evicted;
    std::vector<std::vector<char>> dst = std::vector<std::vector<char>>(3 * HX);
};
static HitEvict g_hev;
static void hx_evict(void *, const void *, const int32_t * experts, int32_t n, void * const * out, int32_t * keep) {
    for (int32_t i = 0; i < n; ++i) {
        g_hev.evicted.push_back(experts[i]);
        keep[i] = 1;
        for (int k = 0; k < 3; ++k) {
            auto & d = g_hev.dst[(size_t) k * HX + experts[i]];
            d.assign(HX_PLANE, 0);
            const_cast<void **>(out)[3 * i + k] = d.data();
        }
    }
}

// One layer, `planes` expert tensors (gate/up/down), top-`k` routing of one
// token, each MUL_MAT_ID forced onto `gpu`; weights in a host WEIGHTS buffer.
struct HitSched {
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::vector<ggml_tensor *> outs;
    ggml_tensor * ids = nullptr;
    ggml_cgraph * gf = nullptr;
    int k = 0;

    void build(ggml_backend_t gpu, int planes, int k_) {
        static const char * names[3] = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                        "blk.0.ffn_down_exps.weight"};
        k = k_;
        ggml_init_params ip = {16 * 1024 * 1024, nullptr, true};
        ctx = ggml_init(ip);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, k);
        ggml_set_name(b, "blk.0.cur_ffn_inp");
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
        ggml_set_name(ids, "blk.0.ffn_router");
        std::vector<ggml_tensor *> w;
        for (int p = 0; p < planes; ++p) {
            ggml_tensor * wk = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NE0, NE1, HX);
            ggml_set_name(wk, names[p]);
            w.push_back(wk);
            outs.push_back(ggml_mul_mat_id(ctx, wk, b, ids));
        }
        gf = ggml_new_graph(ctx);
        for (auto * o : outs) {
            ggml_build_forward_expand(gf, o);
        }
        cpu = ggml_backend_cpu_init();
        ggml_backend_t backs[2] = {gpu, cpu};
        sched = ggml_backend_sched_new(backs, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        for (auto * o : outs) {
            ggml_backend_sched_set_tensor_backend(sched, o, gpu);
        }
        wbuf = ggml_backend_alloc_buffer(cpu, (size_t) planes * HX * HX_PLANE + 4096);
        char * base = (char *) ggml_backend_buffer_get_base(wbuf);
        for (int p = 0; p < planes; ++p) {
            ggml_backend_tensor_alloc(wbuf, w[p], base + (size_t) p * HX * HX_PLANE);
            ggml_backend_tensor_set(w[p], hx_pool().data(), 0, ggml_nbytes(w[p]));
        }
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        REQUIRE(ggml_backend_sched_alloc_graph(sched, gf), "hit sched alloc_graph");
        std::vector<float> hb((size_t) NE0 * k);
        for (size_t i = 0; i < hb.size(); ++i) hb[i] = 0.5f + 0.25f * (float) (i % 5);
        ggml_backend_tensor_set(b, hb.data(), 0, ggml_nbytes(b));
    }
    enum ggml_status run(const std::vector<int32_t> & route, std::vector<float> & res) {
        ggml_backend_tensor_set(ids, route.data(), 0, ggml_nbytes(ids));
        const enum ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
        res.clear();
        if (st == GGML_STATUS_SUCCESS) {
            for (auto * o : outs) {
                std::vector<float> v(ggml_nelements(o));
                ggml_backend_tensor_get(o, v.data(), 0, ggml_nbytes(o));
                res.insert(res.end(), v.begin(), v.end());
            }
        }
        return st;
    }
    ~HitSched() {
        golem_sched_detach_ctx(sched);
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(wbuf);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }
};

// Hookless upstream reference on the same backend + route.
static std::vector<float> hx_reference(ggml_backend_t gpu, int planes, const std::vector<int32_t> & route) {
    HitSched hs;
    hs.build(gpu, planes, (int) route.size());
    std::vector<float> res;
    REQUIRE(hs.run(route, res) == GGML_STATUS_SUCCESS, "reference compute");
    return res;
}

static bool hx_same(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-4f * (1.0f + std::fabs(b[i]))) {
            return false;
        }
    }
    return true;
}

static golem_hook_handle_t g_hx_handle = nullptr;
static golem_ctx_t hx_ctx(bool slab, bool evict) {
    golem_ctx_t c = golem_ctx_create(N_LAYERS, HX);
    golem_set_device_cache_experts(c, 2);
    g_hx_handle = golem_register_hooks(c, hx_ensure, slab ? hx_routing : nullptr, hx_copy_done, evict ? hx_evict : nullptr,
                         nullptr, nullptr, nullptr, &g_hit);
    g_hit = HitState{};
    g_hit.c = c;
    return c;
}
static void hx_free(golem_ctx_t c) {
    golem_unregister_hooks(g_hx_handle);
    golem_reset_device_cache(c);
    golem_ctx_free(c);
}

// Pooled (mode 1) hits_ready checks on `gpu`; returns nothing, REQUIREs.
static void hx_pooled_legs(ggml_backend_t gpu, const char * tag) {
    const std::vector<int32_t> route = {0, 1, 3, 4, 5}; // runs {0,1} and {3,4,5}
    const std::vector<float> ref = hx_reference(gpu, 1, route);
    struct Case { std::vector<int32_t> miss; uint64_t early; };
    // {4} splits run {3,4,5}: only {0,1} goes early. No miss: all 5. All
    // misses: no hits, no callback work.
    const Case cases[3] = {{{4}, 2}, {{}, 5}, {{0, 1, 3, 4, 5}, 0}};
    for (const Case & cs : cases) {
        golem_ctx_t c = hx_ctx(false, false);
        g_hit.miss = cs.miss;
        {
            HitSched hs;
            hs.build(gpu, 1, (int) route.size());
            golem_sched_set_ctx(hs.sched, c);
            const uint64_t e0 = hx_early(c);
            std::vector<float> res;
            REQUIRE(hs.run(route, res) == GGML_STATUS_SUCCESS, "%s: pooled compute", tag);
            REQUIRE(g_hit.ensure_calls == 1, "%s: one ensure, got %d", tag, g_hit.ensure_calls);
            REQUIRE((g_hit.order == std::vector<char>{'c', 'f'}), "%s: hits_ready before the miss fill", tag);
            REQUIRE(g_hit.nulls_at_cb == cs.miss, "%s: pending ids at hits_ready", tag);
            REQUIRE(g_hit.early_at_cb - e0 == cs.early, "%s: %llu units queued inside hits_ready (want %llu)",
                    tag, (unsigned long long) (g_hit.early_at_cb - e0), (unsigned long long) cs.early);
            REQUIRE(hx_early(c) - e0 == cs.early, "%s: early units total", tag);
            REQUIRE(hx_same(res, ref), "%s: pooled output == hookless reference (miss set size %zu)", tag,
                    cs.miss.size());
        }
        hx_free(c);
    }
}

int main() {
    if (golem_hooks_abi_version() != GOLEM_HOOKS_ABI_VERSION) {
        printf("FAIL: abi %d != header %d\n", golem_hooks_abi_version(), GOLEM_HOOKS_ABI_VERSION);
        return 1;
    }
    pool_init();

    {   // --- leg 1: placement counters (unchanged semantics, global stats) ---
        ggml_init_params p = {16 * 1024 * 1024, nullptr, true};
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * as = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 4);
        ggml_set_name(as, "blk.1.ffn_up_exps.weight");
        ggml_tensor * bb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
        ggml_tensor * expert = ggml_mul_mat_id(ctx, as, bb, ids);
        ggml_tensor * as2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 4);
        ggml_set_name(as2, "blk.1.ffn_up_shexp.weight");
        ggml_tensor * bb2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 1);
        ggml_tensor * ids2 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
        ggml_tensor * other = ggml_mul_mat_id(ctx, as2, bb2, ids2);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, expert);
        ggml_build_forward_expand(gf, other);

        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

        const uint64_t gpu0 = load(0), cpu0 = load(1);
        REQUIRE(ggml_backend_sched_alloc_graph(sched, gf), "alloc_graph");
        const uint64_t dgpu = load(0) - gpu0, dcpu = load(1) - cpu0;
        REQUIRE(dcpu == 1 && dgpu == 0, "on_cpu +%llu (want 1), on_gpu +%llu (want 0)",
                (unsigned long long) dcpu, (unsigned long long) dgpu);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }

    {   // --- leg 2: two ctxs, distinct closures, interleaved; each observes
        //     only its own state; a hookless third sched stays upstream ---
        HState sa, sb;
        golem_ctx_t ca = golem_ctx_create(N_LAYERS, N_EXP);
        golem_ctx_t cb = golem_ctx_create(N_LAYERS, N_EXP);
        golem_set_device_cache_experts(ca, 2);
        golem_set_device_cache_experts(cb, 2);
        auto * ha = golem_register_hooks(ca, h_ensure, h_routing, h_copy_done, nullptr, nullptr, nullptr, nullptr, &sa);
        auto * hb = golem_register_hooks(cb, h_ensure, h_routing, h_copy_done, nullptr, nullptr, nullptr, nullptr, &sb);
        ExpertSched ea, eb;
        ea.build();
        eb.build();
        golem_sched_set_ctx(ea.sched, ca);
        golem_sched_set_ctx(eb.sched, cb);
        ggml_backend_sched_graph_compute(ea.sched, ea.gf);
        ggml_backend_sched_graph_compute(eb.sched, eb.gf);
        ggml_backend_sched_graph_compute(ea.sched, ea.gf);
        // slab path: the second step has no misses (slots bit-valid), so ctx
        // A ensures exactly once over two computes; routing fires per step.
        REQUIRE(sa.ensure_calls.load() == 1, "ctx A ensure calls %d (want 1)", sa.ensure_calls.load());
        REQUIRE(sb.ensure_calls.load() == 1, "ctx B ensure calls %d (want 1)", sb.ensure_calls.load());
        REQUIRE(sa.routing_calls.load() == 2, "ctx A routing calls %d", sa.routing_calls.load());
        REQUIRE(sb.routing_calls.load() == 1, "ctx B routing calls %d", sb.routing_calls.load());
        REQUIRE(sa.last_blk == 0 && sb.last_blk == 0, "routing keyed by blk");
        REQUIRE(sa.last_n_ids == 2, "n_ids = tokens x top-k (2), got %lld", (long long) sa.last_n_ids);
        REQUIRE((sa.last_ids == std::vector<int32_t>{0, 1}), "dense ids");
        // hookless sched under a hooked process: upstream verbatim
        {
            ExpertSched ec;
            ec.build();
            const int ea_before = sa.ensure_calls.load() + sb.ensure_calls.load();
            ggml_backend_sched_graph_compute(ec.sched, ec.gf);
            REQUIRE(sa.ensure_calls.load() + sb.ensure_calls.load() == ea_before,
                    "hookless sched must not fire any ctx's hooks");
        }
        golem_unregister_hooks(ha);
        golem_unregister_hooks(hb);
        golem_sched_detach_ctx(ea.sched);
        golem_sched_detach_ctx(eb.sched);
        golem_reset_device_cache(ca);
        golem_ctx_free(ca);
        golem_reset_device_cache(cb);
        golem_ctx_free(cb);
    }

    {   // --- leg 3: unregister-join ---
        HState s;
        g_cb_entered.store(false); // leg 2's h_ensure runs left it true
        g_join_sleep_ms.store(200);
        golem_ctx_t c = golem_ctx_create(N_LAYERS, N_EXP);
        golem_set_device_cache_experts(c, 2);
        auto * h = golem_register_hooks(c, h_ensure, h_routing, h_copy_done, nullptr, nullptr, nullptr, nullptr, &s);
        ExpertSched es;
        es.build();
        golem_sched_set_ctx(es.sched, c);
        std::thread worker([&] { ggml_backend_sched_graph_compute(es.sched, es.gf); });
        // deterministic in-flight detection: wait for the callback's own entry signal
        for (int i = 0; i < 5000 && !g_cb_entered.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(g_cb_entered.load(), "callback must start before unregister");
        const auto t0 = std::chrono::steady_clock::now();
        golem_unregister_hooks(h);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        worker.join(); // join before REQUIRE so a failure cannot terminate on a live thread
        REQUIRE(ms >= 100, "unregister must join the in-flight callback (%lld ms)", (long long) ms);
        REQUIRE(s.ensure_calls.load() == 1, "callback ran once");
        REQUIRE(s.cb_returned.load(), "callback completed before unregister returned");
        g_cb_entered.store(false);
        g_join_sleep_ms.store(0);
        golem_sched_detach_ctx(es.sched);
        golem_reset_device_cache(c);
        golem_ctx_free(c);
    }

    {   // --- leg 4: allocation-free steady state over the slab path ---
        HState s;
        golem_ctx_t c = golem_ctx_create(N_LAYERS, N_EXP);
        golem_set_device_cache_experts(c, 2);
        auto * h = golem_register_hooks(c, h_ensure, h_routing, h_copy_done, nullptr, nullptr, nullptr, nullptr, &s);
        ExpertSched es;
        es.build();
        golem_sched_set_ctx(es.sched, c);
        // warm-up step (allocates thread_local scratch, slab, map entries)
        ggml_backend_sched_graph_compute(es.sched, es.gf);
        REQUIRE(s.routing_calls.load() == 1 && s.ensure_calls.load() >= 1, "hooks fire");
        g_allocs.store(0);
        g_count_allocs.store(true);
        for (int i = 0; i < 8; ++i) {
            ggml_backend_sched_graph_compute(es.sched, es.gf);
        }
        g_count_allocs.store(false);
        REQUIRE(g_allocs.load() == 0, "steady state allocated %ld times", g_allocs.load());
        golem_unregister_hooks(h);
        golem_sched_detach_ctx(es.sched);
        golem_reset_device_cache(c);
        golem_ctx_free(c);
    }

    {   // --- leg 7: hits_ready, pooled (mode 1), fake GPU ---
        hx_pooled_legs(golem_test::fake_backend(), "leg 7");
    }

    {   // --- leg 8: hits_ready, slab (mode 2) + parked eviction D2H ---
        golem_ctx_t c = hx_ctx(true, true);
        g_hev = HitEvict{};
        {
            HitSched hs;
            hs.build(golem_test::fake_backend(), 3, 2);
            golem_sched_set_ctx(hs.sched, c);
            std::vector<float> res;
            REQUIRE(hs.run({0, 1}, res) == GGML_STATUS_SUCCESS, "leg 8: step 1");
            REQUIRE(hx_same(res, hx_reference(golem_test::fake_backend(), 3, {0, 1})), "leg 8: step 1 output");
            REQUIRE(g_hev.evicted.empty(), "leg 8: no eviction before the slots fill");
            // Step 2: slots of 0,1 are stolen by 2,3; 3 is a disk miss.
            g_hit.miss = {3};
            g_hit.order.clear();
            g_hit.nulls_at_cb.clear();
            const uint64_t e0 = hx_early(c);
            REQUIRE(hs.run({2, 3}, res) == GGML_STATUS_SUCCESS, "leg 8: step 2");
            REQUIRE(hx_early(c) - e0 == 3, "leg 8: expert 2 early in all 3 planes, got %llu",
                    (unsigned long long) (hx_early(c) - e0));
            REQUIRE(g_hev.evicted.size() == 2, "leg 8: both owners evicted, got %zu", g_hev.evicted.size());
            for (int32_t e : g_hev.evicted) {
                for (int k = 0; k < 3; ++k) {
                    const auto & d = g_hev.dst[(size_t) k * HX + e];
                    REQUIRE(d.size() == HX_PLANE && memcmp(d.data(), hx_plane(k + 1, e), HX_PLANE) == 0,
                            "leg 8: evicted expert %d plane %d holds the OLD bytes (D2H before refill)", e, k);
                }
            }
            REQUIRE(hx_same(res, hx_reference(golem_test::fake_backend(), 3, {2, 3})), "leg 8: slab holds the NEW bytes");
        }
        hx_free(c);
    }

    {   // --- leg 9: ensure fails after hits_ready ran ---
        golem_ctx_t c = hx_ctx(false, false);
        g_hit.miss = {4};
        g_hit.rc = 1;
        {
            HitSched hs;
            hs.build(golem_test::fake_backend(), 1, 5);
            golem_sched_set_ctx(hs.sched, c);
            std::vector<float> res;
            REQUIRE(hs.run({0, 1, 3, 4, 5}, res) == GGML_STATUS_FAILED, "leg 9: compute FAILED");
            REQUIRE((g_hit.order == std::vector<char>{'c', 'f'}), "leg 9: hits_ready ran before the failure");
            g_hit.rc = 0;
            REQUIRE(hs.run({0, 1, 3, 4, 5}, res) == GGML_STATUS_SUCCESS, "leg 9: recovery compute");
            REQUIRE(hx_same(res, hx_reference(golem_test::fake_backend(), 1, {0, 1, 3, 4, 5})), "leg 9: recovery output");
        }
        hx_free(c);
    }

    {   // --- leg 10: hits_ready under async_h2d (copy stream, GPU only) ---
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev == nullptr) {
            const char * cpu_only = getenv("GOLEM_CTEST_CPU_ONLY");
            if (cpu_only != nullptr && strcmp(cpu_only, "1") == 0) {
                printf("SKIPPED: leg 10 async_h2d (no GPU device, GOLEM_CTEST_CPU_ONLY=1)\n");
                fflush(stdout);
            } else {
                REQUIRE(false, "leg 10: no GPU device for async_h2d (set GOLEM_CTEST_CPU_ONLY=1 on a CPU-only build)");
            }
        } else {
            ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(gpu != nullptr, "leg 10: GPU backend init");
            const std::vector<int32_t> route = {0, 1, 3, 4, 5};
            for (int slab = 0; slab < 2; ++slab) {
                const std::vector<float> ref = hx_reference(gpu, 1, route);
                golem_ctx_t c = hx_ctx(slab != 0, false);
                if (slab) {
                    golem_set_device_cache_experts(c, 5);
                }
                REQUIRE(golem_set_async_h2d(c, 1) == 0, "leg 10: async_h2d on");
                g_hit.miss = {4};
                {
                    HitSched hs;
                    hs.build(gpu, 1, (int) route.size());
                    golem_sched_set_ctx(hs.sched, c);
                    const uint64_t e0 = hx_early(c);
                    std::vector<float> res;
                    REQUIRE(hs.run(route, res) == GGML_STATUS_SUCCESS, "leg 10 (%s): compute", slab ? "slab" : "pooled");
                    // pooled: run {0,1} (2); slab: every pending hit (4).
                    const uint64_t want = slab ? 4 : 2;
                    REQUIRE(hx_early(c) - e0 == want, "leg 10 (%s): early units %llu (want %llu)",
                            slab ? "slab" : "pooled", (unsigned long long) (hx_early(c) - e0),
                            (unsigned long long) want);
                    REQUIRE(hx_same(res, ref), "leg 10 (%s): output == hookless reference", slab ? "slab" : "pooled");
                    double ms = -1.0;
                    uint64_t bytes = 0;
                    REQUIRE(golem_take_h2d_timing(c, &ms, &bytes) == 1, "leg 10: timing pairs closed, no overflow");
                    REQUIRE(ms >= 0.0 && bytes >= 5 * HX_PLANE, "leg 10: timed bytes %llu",
                            (unsigned long long) bytes);
                }
                hx_free(c);
            }
            ggml_backend_free(gpu);
        }
    }

    fflush(stdout); // forked legs below must not re-emit buffered output
    {   // --- leg 5: unregister inside a callback aborts ---
        if (fork() == 0) {
            HState s;
            golem_ctx_t c = golem_ctx_create(N_LAYERS, N_EXP);
            g_h_abort = golem_register_hooks(c, h_abort_ensure, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &s);
            // Invoke the hook exactly as the scheduler would (inside the
            // C-side golem_cb_guard); computing in a forked child of a
            // multithreaded parent deadlocks in the CPU threadpool.
            static int32_t mk[1] = {1}, mi[1] = {0};
            const void * optr[1] = {nullptr};
            golem_cb_guard cb;
            h_abort_ensure(&s, nullptr, mk, mi, 1, optr, nullptr, nullptr);
            _exit(0); // must not survive
        }
        int st = 0; wait(&st);
        REQUIRE(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT, "unregister in callback must abort");
    }

    {   // --- leg 6: ctx_free with attached sched aborts ---
        if (fork() == 0) {
            golem_ctx_t c = golem_ctx_create(N_LAYERS, N_EXP);
            ExpertSched es;
            es.build();
            golem_sched_set_ctx(es.sched, c);
        golem_reset_device_cache(c);
            golem_ctx_free(c);
            _exit(0);
        }
        int st = 0; wait(&st);
        REQUIRE(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT, "ctx_free with attach_refs != 0 must abort");
    }

    if (g_failures != 0) {
        printf("golem-p1-placement: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("golem-p1-placement: all legs passed\n");
    return 0;
}
