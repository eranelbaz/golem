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
#include <atomic>
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
                     int32_t n, const void ** out) {
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
    g_cb_entered.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(g_join_sleep_ms.load()));
    s->cb_returned.store(true);
    return 0;
}

static int32_t h_abort_ensure(void *, const void *, const int32_t *, const int32_t *,
                           int32_t, const void **) {
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

int main() {
    if (golem_hooks_abi_version() != GOLEM_HOOKS_ABI_VERSION) {
        printf("FAIL: abi %d != header %d\n", golem_hooks_abi_version(), GOLEM_HOOKS_ABI_VERSION);
        return 1;
    }
    REQUIRE(GOLEM_HOOKS_ABI_VERSION == 13, "abi v13 expected");
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
            h_abort_ensure(&s, nullptr, mk, mi, 1, optr);
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
