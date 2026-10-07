// Golem vendor-test shared harness: synthetic MoE graphs (expert weights in
// WEIGHTS-usage host buffers, MUL_MAT_ID forced onto the fake GPU backend)
// that drive the scheduler's golem copy paths, plus recording hook closures
// standing in for the engine payload.
#ifndef GOLEM_TEST_SCHED_HPP
#define GOLEM_TEST_SCHED_HPP

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "golem-hook-context.hpp"
#include "golem-test-fakegpu.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace golem_test {

static const int64_t NE0 = 8, NE1 = 4;
static const int32_t N_EXP = 4, N_LAYERS = 4;

inline int & failures() { static int f = 0; return f; }
#define T_REQUIRE(cond, ...)                             \
    do {                                                 \
        if (!(cond)) {                                   \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);  \
            printf(__VA_ARGS__);                         \
            printf("\n");                                \
            golem_test::failures()++;                    \
        }                                                \
    } while (0)

// --- hook state (the Rust-payload stand-in) ---
struct HState {
    std::atomic<int> ensure_calls{0}, routing_calls{0}, copy_done_calls{0};
    std::atomic<bool> cb_returned{false};
    std::vector<int32_t> miss_kinds, miss_ids;      // last call
    std::vector<int32_t> all_kinds, all_ids;        // accumulated across calls
    int32_t last_blk = -1;
    int64_t last_n_ids = -1;
    int32_t last_n_experts = -1;
    std::vector<int32_t> last_ids;
    const void * last_tensor = nullptr;
};

inline std::vector<char> & pool() {
    static std::vector<char> p;
    return p;
}
inline void pool_init() {
    const size_t plane = (size_t) NE0 * NE1 * sizeof(float);
    pool().assign(plane * 3 * N_EXP, (char) 0x5a);
}
inline std::atomic<int> & join_sleep_ms() { static std::atomic<int> ms{0}; return ms; }
inline golem_hook_handle_t & abort_handle() { static golem_hook_handle_t h = nullptr; return h; }

inline void h_ensure(void * p, const void * tensor, const int32_t * mk, const int32_t * mi,
                     int32_t n, const void ** out) {
    auto * s = (HState *) p;
    s->ensure_calls.fetch_add(1);
    s->miss_kinds.assign(mk, mk + n);
    s->all_kinds.insert(s->all_kinds.end(), mk, mk + n);
    s->all_ids.insert(s->all_ids.end(), mi, mi + n);
    s->miss_ids.assign(mi, mi + n);
    const size_t plane = (size_t) NE0 * NE1 * sizeof(float);
    for (int32_t i = 0; i < n; ++i) {
        T_REQUIRE(mi[i] >= 0 && mi[i] < N_EXP, "miss id out of range");
        // kind is 1-based per golem_tensor_kind: gate=1 up=2 down=3
        const int kind = s->miss_kinds[i];
        out[i] = pool().data() + plane * ((size_t) (kind - 1) * N_EXP + mi[i]);
    }
    (void) tensor;
    std::this_thread::sleep_for(std::chrono::milliseconds(join_sleep_ms().load()));
    s->cb_returned.store(true);
}

inline void h_abort_ensure(void *, const void *, const int32_t *, const int32_t *,
                           int32_t, const void **) {
    golem_unregister_hooks(abort_handle()); // must GGML_ABORT (join waits for the calling step)
}

inline void h_routing(void * p, int32_t blk, const void * tensor, const int32_t * dense,
                      int64_t n_ids, int32_t n_experts) {
    auto * s = (HState *) p;
    s->routing_calls.fetch_add(1);
    s->last_blk = blk;
    s->last_n_ids = n_ids;
    s->last_n_experts = n_experts;
    s->last_ids.assign(dense, dense + n_ids);
    s->last_tensor = tensor;
    (void) n_experts;
}

inline void h_copy_done(void * p) { ((HState *) p)->copy_done_calls.fetch_add(1); }

// One-layer MoE graph: `n_planes` expert tensors of the same layer (gate/up/
// down names so golem_tensor_kind resolves), one shared ids tensor, each
// MUL_MAT_ID forced onto the fake GPU backend so the split's cross-backend
// copy path (and its golem branches) engages.
struct ExpertSched {
    int n_planes = 1;
    ggml_backend_t cpu_a = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::vector<ggml_tensor *> w, outs;
    ggml_tensor * b = nullptr, * ids = nullptr;
    ggml_cgraph * gf = nullptr;

    void build(int planes) {
        n_planes = planes;
        static const char * names[3] = {
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        };
        ggml_init_params ip = {16 * 1024 * 1024, nullptr, true};
        ctx = ggml_init(ip);
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, 2);
        ggml_set_name(b, "blk.0.cur_ffn_inp");
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
        ggml_set_name(ids, "blk.0.ffn_router");
        for (int k = 0; k < planes; ++k) {
            ggml_tensor * wk = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NE0, NE1, N_EXP);
            ggml_set_name(wk, names[k]);
            w.push_back(wk);
            outs.push_back(ggml_mul_mat_id(ctx, wk, b, ids));
        }
        gf = ggml_new_graph(ctx);
        for (auto * o : outs) {
            ggml_build_forward_expand(gf, o);
        }

        cpu_a = ggml_backend_cpu_init();
        ggml_backend_t backs[2] = {fake_backend(), cpu_a};
        sched = ggml_backend_sched_new(backs, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        for (auto * o : outs) {
            ggml_backend_sched_set_tensor_backend(sched, o, fake_backend());
        }

        size_t wbytes = 0;
        for (auto * wk : w) {
            wbytes += ggml_nbytes(wk);
        }
        wbuf = ggml_backend_alloc_buffer(cpu_a, wbytes * 2);
        char * base = (char *) ggml_backend_buffer_get_base(wbuf);
        size_t off = 0;
        for (auto * wk : w) {
            ggml_backend_tensor_alloc(wbuf, wk, base + off);
            off += ggml_nbytes(wk);
        }
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        float tmp[NE0 * NE1 * N_EXP];
        for (int i = 0; i < (int) (NE0 * NE1 * N_EXP); ++i) tmp[i] = (float) (i % 7) * 0.25f;
        for (auto * wk : w) {
            ggml_backend_tensor_set(wk, tmp, 0, ggml_nbytes(wk));
        }

        T_REQUIRE(ggml_backend_sched_alloc_graph(sched, gf), "alloc_graph");
        const int32_t hids[2] = {0, 1};
        ggml_backend_tensor_set(ids, hids, 0, ggml_nbytes(ids));
        float hb[NE0 * 2];
        for (int i = 0; i < (int) (NE0 * 2); ++i) hb[i] = 0.5f;
        ggml_backend_tensor_set(b, hb, 0, ggml_nbytes(b));
    }
    ~ExpertSched() {
        golem_sched_detach_ctx(sched);
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(wbuf);
        ggml_backend_free(cpu_a);
        ggml_free(ctx);
    }
};

} // namespace golem_test

#endif /* GOLEM_TEST_SCHED_HPP */
