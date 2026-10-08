// Golem prefill-stream regression (design §7.6, ABI v10): drives the sched's
// mode-3 stream plane path on the fake-GPU harness:
//   1. golem_set_prefill_stream without the stream hooks aborts loudly.
//   2. A streamed step never calls ensure (no pool page-in — §7.6.4), calls
//      stream_source once per plane with the routed ids (routed share below
//      the switch) and produces the same MUL_MAT_ID output as the pooled
//      path (bytes bit-identical — I1).
//   3. Above the routed-share switch the source gets the whole identity
//      range.
//   4. Slab reuse releases: filling seq 2 syncs and calls stream_release(0)
//      before its copies (the seq−2 contract).
//   5. Disarming restores the pooled path verbatim.
#include "golem-hook-context.hpp"
#include "golem-test-sched.hpp"

#include <cstdint>
#include <cstdio>
#include <signal.h>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

int g_failures = 0;

#define REQUIRE(cond, ...)                                  \
    do {                                                    \
        if (!(cond)) {                                      \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);     \
            printf(__VA_ARGS__);                            \
            printf("\n");                                   \
            g_failures++;                                   \
        }                                                   \
    } while (0)

struct SState {
    std::atomic<int> source_calls{0}, release_calls{0}, ensure_calls{0};
    int32_t last_kind = -1, last_blk = -1;
    uint64_t last_seq = 0;
    std::vector<int32_t> last_ids;
    std::vector<uint64_t> released;
    // Per-expert plane bytes the source hands out: plane p of expert e holds
    // the float pattern (e + 1) * 100 + p.
    std::vector<float> bytes;
    SState() {
        const size_t plane = (size_t) golem_test::NE0 * golem_test::NE1;
        bytes.resize(plane * 3 * golem_test::N_EXP);
        for (int k = 0; k < 3; ++k) {
            for (int e = 0; e < golem_test::N_EXP; ++e) {
                for (size_t i = 0; i < plane; ++i) {
                    bytes[k * golem_test::N_EXP * plane + (size_t) e * plane + i] =
                        (float) ((e + 1) * 100 + k);
                }
            }
        }
    }
    const char * plane_ptr(int kind, int32_t e) const {
        const size_t plane = (size_t) golem_test::NE0 * golem_test::NE1;
        return (const char *) (bytes.data() + ((size_t) (kind - 1) * golem_test::N_EXP + e) * plane);
    }
};

void s_source(void * p, int32_t blk, int32_t kind, uint64_t seq, const int32_t * ids,
              int32_t n, const void ** out) {
    auto * s = (SState *) p;
    s->source_calls.fetch_add(1);
    s->last_blk = blk;
    s->last_kind = kind;
    s->last_seq = seq;
    s->last_ids.assign(ids, ids + n);
    for (int32_t i = 0; i < n; ++i) {
        out[i] = s->plane_ptr(kind, ids[i]);
    }
}

void s_release(void * p, uint64_t seq) {
    auto * s = (SState *) p;
    s->release_calls.fetch_add(1);
    s->released.push_back(seq);
}

void s_ensure(void * p, const void *, const int32_t * mk, const int32_t * mi,
              int32_t n, const void ** out) {
    auto * s = (SState *) p;
    s->ensure_calls.fetch_add(1);
    // Pooled stand-in: serve the pool pattern too, so a disarmed step still
    // computes the same bytes.
    for (int32_t i = 0; i < n; ++i) {
        out[i] = s->plane_ptr(mk[i], mi[i]);
    }
}

void s_routing(void *, int32_t, const void *, const int32_t *, int64_t, int32_t) {}
void s_copy_done(void *) {}

} // namespace

int main() {
    using golem_test::ExpertSched;
    using golem_test::N_EXP;

    {   // leg 1: arming without the stream hooks aborts (fork child).
        pid_t pid = fork();
        if (pid == 0) {
            golem_ctx_t c = golem_ctx_create(4, N_EXP);
            golem_set_prefill_stream(c, 1, N_EXP, 4096, 0.85f);
            _exit(0); // not reached: must abort
        }
        int st = 0;
        waitpid(pid, &st, 0);
        REQUIRE(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT,
                "arming without stream hooks must abort (got status %d)", st);
    }

    // legs 2-5: full sched steps.
    SState st;
    {
        ExpertSched es;
        es.build(1);
        golem_ctx_t c = golem_ctx_create(golem_test::N_LAYERS, N_EXP);
        auto * h = golem_register_hooks(c, s_ensure, s_routing, s_copy_done,
                                        NULL, NULL, s_source, s_release, &st);
        golem_sched_set_ctx(es.sched, c);
        golem_set_device_cache_experts(c, 2);

        // leg 2: streamed routed-only step, output matches the expert bytes.
        golem_set_prefill_stream(c, 1, N_EXP, 4096, 0.85f);
        ggml_backend_sched_graph_compute(es.sched, es.gf);
        REQUIRE(st.ensure_calls.load() == 0,
                "streamed step must not page into the pool (ensure calls %d)",
                st.ensure_calls.load());
        REQUIRE(st.source_calls.load() == 1, "one plane = one stream_source (got %d)",
                st.source_calls.load());
        REQUIRE(st.last_kind == 1 && st.last_blk == 0, "gate plane of blk 0");
        REQUIRE((st.last_ids == std::vector<int32_t>{0, 1}),
                "routed-only fill below the share switch (ids size %zu)",
                st.last_ids.size());
        REQUIRE(st.released.empty(), "no release before slab reuse");
        {
            std::vector<float> out(golem_test::NE1 * 2);
            ggml_backend_tensor_get(es.outs[0], out.data(), 0, out.size() * sizeof(float));
            // ids {0,1}: row r uses expert ids[r], expert e holds (e+1)*100 + 0
            // (kind 0 = gate); input rows are constant 0.5 over NE0.
            for (int r = 0; r < 2; ++r) {
                const int32_t e = r == 0 ? 0 : 1; // ids {0, 1}
                const float want = 0.5f * (float) golem_test::NE0 * (float) ((e + 1) * 100);
                REQUIRE(out[r * golem_test::NE1] == want, "row %d: got %f want %f",
                        r, out[r * golem_test::NE1], want);
            }
        }

        // leg 3: whole fill above the share switch.
        st.source_calls.store(0);
        st.last_ids.clear();
        golem_set_prefill_stream(c, 1, N_EXP, 4096, 0.4f);
        ggml_backend_sched_graph_compute(es.sched, es.gf);
        REQUIRE((st.last_ids ==
                 std::vector<int32_t>{0, 1, 2, 3}),
                "share 0.5 > 0.4 must fill whole (n %zu)", st.last_ids.size());

        golem_sched_detach_ctx(es.sched);
        golem_unregister_hooks(h);
        golem_reset_device_cache(c);
        golem_ctx_free(c);

        // leg 4: three planes on a FRESH ctx → slab reuse releases seq−2
        // after the sync (seq starts at 0).
        st.source_calls.store(0);
        st.released.clear();
        {
            ExpertSched es3;
            es3.build(3);
            golem_ctx_t c3 = golem_ctx_create(golem_test::N_LAYERS, N_EXP);
            auto * h3 = golem_register_hooks(c3, s_ensure, s_routing, s_copy_done,
                                             NULL, NULL, s_source, s_release, &st);
            golem_sched_set_ctx(es3.sched, c3);
            golem_set_device_cache_experts(c3, 2);
            golem_set_prefill_stream(c3, 1, N_EXP, 4096, 0.85f);
            ggml_backend_sched_graph_compute(es3.sched, es3.gf);
            REQUIRE(st.source_calls.load() == 3, "three planes (got %d)",
                    st.source_calls.load());
            REQUIRE(st.released.size() == 1 && st.released[0] == 0,
                    "filling seq 2 releases seq 0 (released %zu)",
                    st.released.size());
            golem_sched_detach_ctx(es3.sched);
            golem_unregister_hooks(h3);
            golem_reset_device_cache(c3);
            golem_ctx_free(c3);
        }

        // leg 5: disarmed → pooled path verbatim.
        st.source_calls.store(0);
        st.ensure_calls.store(0);
        {
            ExpertSched es4;
            es4.build(1);
            golem_ctx_t c4 = golem_ctx_create(golem_test::N_LAYERS, N_EXP);
            auto * h4 = golem_register_hooks(c4, s_ensure, s_routing, s_copy_done,
                                             NULL, NULL, s_source, s_release, &st);
            golem_sched_set_ctx(es4.sched, c4);
            golem_set_device_cache_experts(c4, 2);
            golem_set_prefill_stream(c4, 0, N_EXP, 4096, 0.85f);
            ggml_backend_sched_graph_compute(es4.sched, es4.gf);
            REQUIRE(st.ensure_calls.load() == 1 && st.source_calls.load() == 0,
                    "disarmed step takes the pooled path (ensure %d, source %d)",
                    st.ensure_calls.load(), st.source_calls.load());
            golem_sched_detach_ctx(es4.sched);
            golem_unregister_hooks(h4);
            golem_reset_device_cache(c4);
            golem_ctx_free(c4);
        }
    }

    if (g_failures == 0) {
        printf("golem-p6-stream: all legs passed\n");
        return 0;
    }
    printf("golem-p6-stream: %d failure(s)\n", g_failures);
    return 1;
}
