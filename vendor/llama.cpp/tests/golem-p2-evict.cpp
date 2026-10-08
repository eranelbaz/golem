// Golem exclusive-residency regression: a layer's planes span several split
// segments, but evict_batch delivers all three plane destinations at once in
// the segment that triggered the eviction. The D2H for a plane can only be
// issued by the segment that owns that plane's slab entry, so the substrate
// must defer each plane's copy to its own segment (before that segment's
// staged H2D refills). This test forces a fully-valid eviction with the
// planes in separate splits and requires every plane's D2H destination to
// receive the evicted bytes — not just the plane(s) of the reporting segment.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "golem-hook-context.hpp"
#include "golem-test-sched.hpp"

namespace {

int g_failures = 0;

#define REQUIRE(cond, ...)                     \
    do {                                       \
        if (!(cond)) {                         \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);               \
            printf("\n");                      \
            g_failures++;                      \
        }                                      \
    } while (0)

constexpr int32_t N_EXPERTS = 8;

// Per-plane distinct fill: plane k of expert e holds bytes equal to
// (10 * (k + 1) + e) so a wrong plane, wrong slot, or an unwritten
// destination is detected byte-exactly.
constexpr size_t PLANE_BYTES = (size_t) golem_test::NE0 * golem_test::NE1 * sizeof(float);

static std::vector<char> & pool_plane(int kind0, int expert) {
    static std::vector<std::vector<char>> p(3 * N_EXPERTS);
    auto & v = p[(size_t) kind0 * N_EXPERTS + expert];
    if (v.empty()) {
        v.assign(PLANE_BYTES, (char) (10 * (kind0 + 1) + expert));
    }
    return v;
}

struct EvictState {
    std::vector<char> dst[3 * N_EXPERTS];   // D2H destinations, zero-filled
    int evict_calls = 0;
    std::vector<int32_t> evicted;           // experts reported evicted
};

static EvictState g_ev;

// ABI v9: substrate-reported abandoned parked D2Hs (plane copies not all
// issued by compute end). The pool stand-in records the call; the commit
// stand-in (ev_copy_done) never sees these experts again.
struct AbandonState {
    int calls = 0;
    int32_t last_blk = -1;
    std::vector<int32_t> experts;
};

static AbandonState g_ab;

static void ev_abandon(void *, int32_t blk, const int32_t * experts, int32_t n) {
    g_ab.calls++;
    g_ab.last_blk = blk;
    g_ab.experts.assign(experts, experts + n);
}

static void ev_ensure(void *, const void *, const int32_t * mk, const int32_t * mi,
                      int32_t n, const void ** out) {
    for (int32_t i = 0; i < n; ++i) {
        T_REQUIRE(mi[i] >= 0 && mi[i] < N_EXPERTS, "miss id out of range");
        out[i] = pool_plane(mk[i] - 1, mi[i]).data();
    }
}

static void ev_routing(void *, int32_t, const void *, const int32_t *,
                       int64_t, int32_t) {}

static void ev_copy_done(void *) {}

static void ev_evict(void *, const void *, const int32_t * experts, int32_t n,
                     void *const * out, int32_t * keep) {
    auto & out_m = const_cast<void **&>(out);
    g_ev.evict_calls++;
    for (int32_t i = 0; i < n; ++i) {
        REQUIRE(experts[i] >= 0 && experts[i] < N_EXPERTS, "evict id out of range");
        g_ev.evicted.push_back(experts[i]);
        keep[i] = 1;
        for (int k = 0; k < 3; ++k) {
            auto & d = g_ev.dst[(size_t) k * N_EXPERTS + experts[i]];
            d.assign(PLANE_BYTES, 0);
            const_cast<void **>(out)[3 * i + k] = d.data();
        }
    }
}

} // namespace

int main() {
    using namespace golem_test;
    pool_init();
    HState s;
    golem_ctx_t c = golem_ctx_create(N_LAYERS, N_EXPERTS);
    golem_set_device_cache_experts(c, 2); // K = 2 < 8: forced eviction churn
    auto * h = golem_register_hooks(c, ev_ensure, ev_routing, ev_copy_done,
                                    ev_evict, ev_abandon, nullptr, nullptr, &s);
    (void) h;
    {
        ExpertSched es;
        es.build(3); // gate + up + down, separate segments
        golem_sched_set_ctx(es.sched, c);

        // Step 1: route experts 0,1 — fills slab slots 0,1 in all planes.
        {
            const int32_t hids[2] = {0, 1};
            ggml_backend_tensor_set(es.ids, hids, 0, ggml_nbytes(es.ids));
        }
        ggml_backend_sched_graph_compute(es.sched, es.gf);
        REQUIRE(g_ev.evict_calls == 0, "no evictions before slots are full");

        // Step 2: route experts 2,3 — slots 0,1 (fully valid: all three
        // planes copied in step 1) are stolen; their bytes must land in the
        // D2H destinations, one copy per plane, across the three segments.
        {
            const int32_t hids[2] = {2, 3};
            ggml_backend_tensor_set(es.ids, hids, 0, ggml_nbytes(es.ids));
        }
        ggml_backend_sched_graph_compute(es.sched, es.gf);

        REQUIRE(g_ev.evict_calls == 1, "one batched evict call, got %d",
                g_ev.evict_calls);
        REQUIRE(g_ev.evicted.size() == 2, "both slot owners evicted, got %zu",
                g_ev.evicted.size());
        for (int32_t e : g_ev.evicted) {
            for (int k = 0; k < 3; ++k) {
                const auto & d = g_ev.dst[(size_t) k * N_EXPERTS + e];
                const auto & want = pool_plane(k, e);
                REQUIRE(d.size() == PLANE_BYTES && d == want,
                        "evicted expert %d plane %d: D2H destination must hold "
                        "the evicted bytes (multi-segment deferral)", e, k);
            }
        }
    }
    // A plane whose segment is absent from a compute (fused/upstream path
    // variations) can never receive its D2H, yet its slab bytes may still be
    // fully valid and get evicted. The substrate must report the parked
    // entry abandoned at compute end — committing it would admit unwritten
    // plane bytes — and the skipped plane's destination must stay untouched.
    // Scenario A omits the down plane; scenario B omits the gate plane (the
    // eviction is then parked from the first PRESENT segment, later than the
    // missing plane's would-be copy phase). Step 3 proves no re-abandon.
    // --- Mid-compute failure after a parked eviction: the gate segment's
    // evict_batch + D2H share land (done = gate bit only), then the next
    // segment's graph compute fails. The error return must report the parked
    // entries abandoned — committing them at the next copy_done would open
    // the fill gate over unwritten up/down plane bytes — and never re-fire
    // on the recovery step. (This is the substrate-side contract; the pool
    // side — slot back to the free list, gate open, key non-resident,
    // waiters woken — is covered by the Rust abandon_d2h unit test.)
    {
        g_ev = EvictState{};
        g_ab = AbandonState{};
        ExpertSched es2;
        es2.build(3);
        golem_sched_set_ctx(es2.sched, c);

        {
            const int32_t hids[2] = {0, 1};
            ggml_backend_tensor_set(es2.ids, hids, 0, ggml_nbytes(es2.ids));
        }
        ggml_backend_sched_graph_compute(es2.sched, es2.gf);
        REQUIRE(g_ev.evict_calls == 0, "C: no evictions before slots are full");

        {
            const int32_t hids[2] = {2, 3};
            ggml_backend_tensor_set(es2.ids, hids, 0, ggml_nbytes(es2.ids));
        }
        fg_fail_next_graph().store(true);
        ggml_backend_sched_graph_compute(es2.sched, es2.gf);
        REQUIRE(g_ev.evict_calls == 1, "C: one batched evict call, got %d",
                g_ev.evict_calls);
        REQUIRE(g_ab.calls == 1, "C: one abandon call at the error return, got %d",
                g_ab.calls);
        REQUIRE(g_ab.last_blk == 0, "C: abandon blk 0, got %d", (int) g_ab.last_blk);
        REQUIRE(g_ab.experts.size() == 2, "C: both evicted experts abandoned, got %zu",
                g_ab.experts.size());
        for (int32_t e : {0, 1}) {
            REQUIRE(g_ab.experts.size() == 2 &&
                        (g_ab.experts[0] == e || g_ab.experts[1] == e),
                    "C: expert %d in the abandon list", e);
            const auto & dg = g_ev.dst[0 * N_EXPERTS + e];
            REQUIRE(dg.size() == PLANE_BYTES && dg == pool_plane(0, e),
                    "C: expert %d gate plane drained before the failure", e);
            for (int k : {1, 2}) {
                const auto & d = g_ev.dst[(size_t) k * N_EXPERTS + e];
                bool zero = d.size() == PLANE_BYTES;
                for (char ch : d) {
                    zero = zero && ch == 0;
                }
                REQUIRE(zero, "C: abandoned expert %d plane %d never written", e, k);
            }
        }

        {
            const int32_t hids[2] = {0, 1};
            ggml_backend_tensor_set(es2.ids, hids, 0, ggml_nbytes(es2.ids));
        }
        ggml_backend_sched_graph_compute(es2.sched, es2.gf);
        REQUIRE(g_ab.calls == 1, "C: no second abandon, got %d", g_ab.calls);
    }

    golem_unregister_hooks(h);
    golem_reset_device_cache(c);
    golem_ctx_free(c);

    if (g_failures != 0) {
        printf("golem-p2-evict: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("golem-p2-evict: all legs passed\n");
    return 0;
}
