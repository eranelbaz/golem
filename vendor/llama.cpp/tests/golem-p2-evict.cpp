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

// Layer l's expert e is keyed e + l * N_EXP (fixture has N_EXP = 4 experts,
// N_EXPERTS = 8 keys): blk.1's bytes and destinations differ from blk.0's,
// so a ring slot reused across layers before its D2H ran is detected.
static int32_t ev_key(const void * tensor, int32_t e) {
    return e + golem_test::N_EXP * (strncmp(ggml_get_name((const ggml_tensor *) tensor), "blk.1.", 6) == 0);
}

static int32_t ev_ensure(void *, const void * tensor, const int32_t * mk, const int32_t * mi,
                      int32_t n, const void ** out, golem_hits_ready_fn hits_ready, void * hits_ud) {
    for (int32_t i = 0; i < n; ++i) {
        T_REQUIRE(mi[i] >= 0 && mi[i] < golem_test::N_EXP, "miss id out of range");
        out[i] = pool_plane(mk[i] - 1, ev_key(tensor, mi[i])).data();
    }
    if (hits_ready != nullptr) {
        hits_ready(hits_ud, out);
    }
    return 0;
}

static void ev_routing(void *, int32_t, const void *, const int32_t *,
                       int64_t, int32_t) {}

static void ev_copy_done(void *) {}

static void ev_evict(void *, const void * tensor, const int32_t * experts, int32_t n,
                     void *const * out, int32_t * keep) {
    g_ev.evict_calls++;
    for (int32_t i = 0; i < n; ++i) {
        REQUIRE(experts[i] >= 0 && experts[i] < golem_test::N_EXP, "evict id out of range");
        const int32_t key = ev_key(tensor, experts[i]);
        g_ev.evicted.push_back(key);
        keep[i] = 1;
        for (int k = 0; k < 3; ++k) {
            auto & d = g_ev.dst[(size_t) k * N_EXPERTS + key];
            d.assign(PLANE_BYTES, 0);
            const_cast<void **>(out)[3 * i + k] = d.data();
        }
    }
}

// Every evicted expert's three plane destinations hold its bytes exactly.
static void require_dst_exact(const char * leg) {
    for (int32_t e : g_ev.evicted) {
        for (int k = 0; k < 3; ++k) {
            const auto & d = g_ev.dst[(size_t) k * N_EXPERTS + e];
            REQUIRE(d.size() == PLANE_BYTES && d == pool_plane(k, e),
                    "%s: evicted expert %d plane %d destination not exact", leg, e, k);
        }
    }
}

static void route(golem_test::ExpertSched & es, int32_t a, int32_t b) {
    const int32_t hids[2] = {a, b};
    ggml_backend_tensor_set(es.ids, hids, 0, ggml_nbytes(es.ids));
}

// Victim ring (ABI v18) on a fresh ctx: `slots` ring slots, three steps of
// 2 evictions per layer each after the warm-up (routes {2,3} / {0,1} / {2,3}
// evict 0,1 / 2,3 / 0,1; the fixture graph has 4 experts). The fake D2H stream defers its copies until waited on,
// so a slot reused before its D2H ran would corrupt a destination.
static void ring_leg(int32_t slots, const char * leg) {
    using namespace golem_test;
    golem_test::HState s;
    golem_ctx_t c = golem_ctx_create(N_LAYERS, N_EXPERTS);
    golem_set_device_cache_experts(c, 2);
    REQUIRE(golem_set_victim_ring(c, slots, (int32_t) PLANE_BYTES) == -1,
            "%s: ring refused before copy_done/evict_batch are registered", leg);
    auto * h = golem_register_hooks(c, ev_ensure, ev_routing, ev_copy_done,
                                    ev_evict, ev_abandon, nullptr, nullptr, &s);
    REQUIRE(golem_set_victim_ring(c, slots, 0) == -1, "%s: zero stride refused", leg);
    REQUIRE(golem_set_victim_ring(c, slots, (int32_t) PLANE_BYTES) == 0, "%s: ring on", leg);
    {
        ExpertSched es;
        es.build(3);
        golem_sched_set_ctx(es.sched, c);
        g_ev = EvictState{};
        g_ab = AbandonState{};
        route(es, 0, 1);
        ggml_backend_sched_graph_compute(es.sched, es.gf);
        REQUIRE(golem_set_victim_ring(c, slots, (int32_t) PLANE_BYTES) == 0,
                "%s: setter accepted before the ring is allocated", leg);
        uint64_t units_all = 0, full_all = 0, bytes_all = 0;
        for (int32_t step = 0; step < 3; ++step) {
            g_ev = EvictState{};
            route(es, (step & 1) ? 0 : 2, (step & 1) ? 1 : 3);
            ggml_backend_sched_graph_compute(es.sched, es.gf);
            // Destinations are read after the next compute's entry drain in
            // the engine; here after an explicit sync (drains the D2H stream).
            ggml_backend_sched_synchronize(es.sched);
            golem_test::fg_d2h_run_until(nullptr);
            REQUIRE(g_ev.evicted.size() == 2, "%s step %d: two evictions, got %zu", leg, step,
                    g_ev.evicted.size());
            require_dst_exact(leg);
            uint64_t units = 0, full = 0, bytes = 0;
            double ms = -1.0;
            REQUIRE(golem_take_victim_ring_stats(c, &units, &full, &ms, &bytes) == 1, "%s: ring armed", leg);
            units_all += units;
            full_all += full;
            bytes_all += bytes;
        }
        REQUIRE(golem_set_victim_ring(c, slots, (int32_t) PLANE_BYTES) == -1,
                "%s: setter refused once the ring is allocated", leg);
        // slots >= 2: every plane of both evictions stages through the ring;
        // 1 slot: one entry per step finds the ring full (same-stream D2H).
        const uint64_t ring_entries = slots >= 2 ? 2 : 1;
        REQUIRE(full_all == 3 * (2 - ring_entries), "%s: ring_full %llu", leg, (unsigned long long) full_all);
        REQUIRE(units_all == 3 * 3 * ring_entries, "%s: ring units %llu", leg, (unsigned long long) units_all);
        REQUIRE(bytes_all == units_all * PLANE_BYTES, "%s: ring bytes %llu", leg, (unsigned long long) bytes_all);
        REQUIRE(g_ab.calls == 0, "%s: nothing abandoned", leg);
    }
    golem_unregister_hooks(h);
    golem_reset_device_cache(c);
    golem_ctx_free(c);
}

// Victim ring of 1 slot over blk.0 + blk.1 in ONE compute, one eviction per
// layer: blk.1's park finds blk.0's slot RELEASING (its D2H still deferred on
// the fake D2H stream) and reuses it, so blk.1's D2D into the slot must wait
// on the release event or blk.0's destinations receive blk.1's bytes.
static void ring_reuse_leg() {
    using namespace golem_test;
    golem_test::HState s;
    golem_ctx_t c = golem_ctx_create(N_LAYERS, N_EXPERTS);
    golem_set_device_cache_experts(c, 2);
    auto * h = golem_register_hooks(c, ev_ensure, ev_routing, ev_copy_done,
                                    ev_evict, ev_abandon, nullptr, nullptr, &s);
    REQUIRE(golem_set_victim_ring(c, 1, (int32_t) PLANE_BYTES) == 0, "R4: ring on");
    {
        ExpertSched es;
        es.build(3, 2);
        golem_sched_set_ctx(es.sched, c);
        g_ev = EvictState{};
        g_ab = AbandonState{};
        route(es, 0, 1);
        ggml_backend_sched_graph_compute(es.sched, es.gf);
        g_ev = EvictState{};
        route(es, 0, 2); // one miss per layer: one eviction per layer
        ggml_backend_sched_graph_compute(es.sched, es.gf);
        ggml_backend_sched_synchronize(es.sched);
        golem_test::fg_d2h_run_until(nullptr);
        REQUIRE(g_ev.evicted.size() == 2, "R4: one eviction per layer, got %zu", g_ev.evicted.size());
        int32_t n_blk1 = 0;
        for (int32_t key : g_ev.evicted) {
            n_blk1 += key >= N_EXP;
        }
        REQUIRE(n_blk1 == 1, "R4: one blk.1 eviction, got %d", n_blk1);
        require_dst_exact("R4");
        uint64_t units = 0, full = 0, bytes = 0;
        double ms = -1.0;
        REQUIRE(golem_take_victim_ring_stats(c, &units, &full, &ms, &bytes) == 1, "R4: ring armed");
        REQUIRE(full == 0, "R4: ring_full %llu", (unsigned long long) full);
        REQUIRE(units == 6, "R4: both evictions through the one slot, units %llu", (unsigned long long) units);
        REQUIRE(g_ab.calls == 0, "R4: nothing abandoned");
    }
    golem_unregister_hooks(h);
    golem_reset_device_cache(c);
    golem_ctx_free(c);
}

} // namespace

int main() {
    using namespace golem_test;
    pool_init();
    ggml_backend_device_register(fake_dev());
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

    // --- Victim ring (ABI v18): R1 ring of 4 slots, R2 ring of 1 slot with
    // two evictions per layer (ring_full counted, fallback still exact).
    ring_leg(4, "R1");
    ring_leg(1, "R2");
    ring_reuse_leg();

    // --- R3: mid-compute failure with the ring on: the gate plane's ring
    // D2H lands, the abandon is reported once, and the recovery step does
    // not re-fire it.
    {
        HState s3;
        golem_ctx_t c3 = golem_ctx_create(N_LAYERS, N_EXPERTS);
        golem_set_device_cache_experts(c3, 2);
        auto * h3 = golem_register_hooks(c3, ev_ensure, ev_routing, ev_copy_done,
                                         ev_evict, ev_abandon, nullptr, nullptr, &s3);
        REQUIRE(golem_set_victim_ring(c3, 4, (int32_t) PLANE_BYTES) == 0, "R3: ring on");
        {
            g_ev = EvictState{};
            g_ab = AbandonState{};
            ExpertSched es3;
            es3.build(3);
            golem_sched_set_ctx(es3.sched, c3);
            route(es3, 0, 1);
            ggml_backend_sched_graph_compute(es3.sched, es3.gf);
            route(es3, 2, 3);
            fg_fail_next_graph().store(true);
            ggml_backend_sched_graph_compute(es3.sched, es3.gf);
            REQUIRE(g_ev.evict_calls == 1, "R3: one batched evict call, got %d", g_ev.evict_calls);
            REQUIRE(g_ab.calls == 1, "R3: one abandon call, got %d", g_ab.calls);
            REQUIRE(g_ab.experts.size() == 2, "R3: both experts abandoned, got %zu", g_ab.experts.size());
            for (int32_t e : {0, 1}) {
                const auto & dg = g_ev.dst[0 * N_EXPERTS + e];
                REQUIRE(dg.size() == PLANE_BYTES && dg == pool_plane(0, e),
                        "R3: expert %d gate plane drained through the ring before the abandon", e);
            }
            uint64_t units = 0, full = 0, bytes = 0;
            double ms = 0.0;
            REQUIRE(golem_take_victim_ring_stats(c3, &units, &full, &ms, &bytes) == 1 && units == 2 && full == 0,
                    "R3: two gate units through the ring, got %llu (full %llu)",
                    (unsigned long long) units, (unsigned long long) full);
            route(es3, 0, 1);
            ggml_backend_sched_graph_compute(es3.sched, es3.gf);
            REQUIRE(g_ab.calls == 1, "R3: no second abandon, got %d", g_ab.calls);
        }
        golem_unregister_hooks(h3);
        golem_reset_device_cache(c3);
        golem_ctx_free(c3);
    }

    if (g_failures != 0) {
        printf("golem-p2-evict: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("golem-p2-evict: all legs passed\n");
    return 0;
}
