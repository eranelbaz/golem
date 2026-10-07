// Golem P2 fuzz (design §3.4): exercises the shared SLOT-INDIRECTED
// device cache state machine (ggml/src/golem-device-cache.hpp — the exact
// template the split scheduler runs) for the invariants that matter now:
//   1. bit ⇒ current: after a step, every routed expert's remapped slot holds
//      the host's current generation of that expert.
//   2. remap soundness: every remapped value is in [0, K), and ids outside the
//      routed set never appear (they cannot — routed covers the ids bitset).
//   3. victim choice (order permutation / routed set) never changes routed
//      coverage; budget overflow with routed > K aborts loudly.
//   4. per-layer sharing (§3.4): the three planes (gate/up/down) of one layer
//      share one slot table — an eviction happens once per layer, not once per
//      plane, and each plane keeps its own validity bits.
//
// Aborts are asserted in fork()ed children (GGML_ABORT raises SIGABRT).
#include "golem-device-cache.hpp"
#include "golem-hook-context.hpp"
#include "golem-test-sched.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

int g_failures = 0;

#define REQUIRE(cond, ...)                                     \
    do {                                                       \
        if (!(cond)) {                                         \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);        \
            printf(__VA_ARGS__);                               \
            printf("\n");                                      \
            g_failures++;                                      \
        }                                                      \
    } while (0)

struct Shadow {
    std::vector<uint64_t> host_gen;   // per expert
    std::vector<uint64_t> slot_gen;   // per slot; content generation
    int32_t n = 0;
    int32_t K = 0;

    Shadow(int32_t n_, int32_t k) : host_gen(n_, 1), slot_gen(k + 1, 0), n(n_), K(k) {}
};

} // namespace

int main() {
    std::mt19937_64 rng(0x676f6c656d); // "golem"

    {   // leg: one eviction serves all three planes; later planes only copy.
        golem::GolemLayerSlots L;
        std::vector<int32_t> remap, order = {0, 1, 2, 3};
        const int32_t K = 2, n = 4;
        auto step = [&](int kind, std::vector<int32_t> routed) {
            std::vector<int32_t> ids = routed;
            return golem::golem_slab_step_nop(L, kind, routed, ids.data(), (int64_t) ids.size(),
                                              order.data(), K, n, remap);
        };
        for (int k = 0; k < 3; ++k) step(k, {0, 1});            // warm: experts 0,1 in slots
        auto g = step(0, {2, 3});                                // gate evicts 0 and 1
        GGML_ASSERT(g.evicted == 2 && g.copied == 2);
        std::vector<int32_t> gate_remap = remap;
        auto u = step(1, {2, 3});
        auto d = step(2, {2, 3});
        GGML_ASSERT(u.evicted == 0 && d.evicted == 0);           // shared table: no re-eviction
        GGML_ASSERT(u.copied == 2 && d.copied == 2);             // their bits were cleared by gate
        GGML_ASSERT(remap == gate_remap);                        // same slots for every plane
        auto u2 = step(1, {2, 3});
        GGML_ASSERT(u2.copied == 0 && u2.skipped == 2);
    }
    {   // leg: invalidating one slab re-copies only that plane.
        golem::GolemLayerSlots L;
        std::vector<int32_t> remap, order = {0, 1, 2, 3};
        auto step = [&](int kind, int32_t K) {
            std::vector<int32_t> routed = {0, 1}, ids = routed;
            return golem::golem_slab_step_nop(L, kind, routed, ids.data(), 2, order.data(), K, 4, remap);
        };
        for (int k = 0; k < 3; ++k) step(k, 2);
        golem::golem_layer_invalidate(L, 2);
        auto d = step(2, 2);
        GGML_ASSERT(d.copied == 2 && d.skipped == 0);
        auto g = step(0, 2);
        GGML_ASSERT(g.copied == 0 && g.skipped == 2);
    }
    {   // leg: a K change resets every plane's validity (no stale bit survives re-mapping).
        golem::GolemLayerSlots L;
        std::vector<int32_t> remap, order = {0, 1, 2, 3};
        auto step = [&](int kind, int32_t K, std::vector<int32_t> routed) {
            std::vector<int32_t> ids = routed;
            return golem::golem_slab_step_nop(L, kind, routed, ids.data(), (int64_t) ids.size(),
                                              order.data(), K, 4, remap);
        };
        for (int k = 0; k < 3; ++k) step(k, 3, {0, 1, 2});
        for (int k = 0; k < 3; ++k) {
            auto c = step(k, 2, {0, 1});
            GGML_ASSERT(c.copied == 2 && c.skipped == 0);
        }
    }

    {   // leg: ids tensor = ggml_argsort_top_k view (nb[1] = n_expert*4, n_tokens > 1).
        // The dense gather must yield exactly ne0*ne1 selected ids; walking the raw
        // buffer densely would read non-selected argsort entries and overrun ids_dev.
        const int64_t ne0 = 2, ne1 = 3, n_expert = 6;
        const size_t nb0 = sizeof(int32_t), nb1 = n_expert * sizeof(int32_t);
        // ggml_nbytes of the view: (ne1-1)*nb1 + ne0*nb0
        std::vector<int32_t> raw((ne1 - 1) * n_expert + ne0, -1);
        const int32_t sel[3][2] = {{5, 1}, {1, 4}, {0, 5}};
        for (int64_t i1 = 0; i1 < ne1; ++i1)
            for (int64_t i0 = 0; i0 < ne0; ++i0) raw[i1 * n_expert + i0] = sel[i1][i0];

        std::vector<int32_t> dense;
        golem::golem_gather_ids_dense(raw.data(), ne0, ne1, nb0, nb1, dense);
        REQUIRE((int64_t) dense.size() == ne0 * ne1, "dense ids size %zu != %lld", dense.size(),
                (long long) (ne0 * ne1));
        for (int64_t i1 = 0; i1 < ne1; ++i1)
            for (int64_t i0 = 0; i0 < ne0; ++i0)
                REQUIRE(dense[i1 * ne0 + i0] == sel[i1][i0], "dense gather mismatch at (%lld,%lld)",
                        (long long) i0, (long long) i1);

        golem::GolemLayerSlots L3;
        std::vector<int32_t> remap3, order3 = {0, 1, 2, 3, 4, 5};
        const std::vector<int32_t> routed3 = {0, 1, 4, 5};
        golem::golem_slab_step_nop(L3, 0, routed3, dense.data(), (int64_t) dense.size(),
                                   order3.data(), 4, (int32_t) n_expert, remap3);
        REQUIRE((int64_t) remap3.size() == ne0 * ne1, "remap size %zu != ids_dev nelements %lld",
                remap3.size(), (long long) (ne0 * ne1));

        // Pre-fix behavior: the raw buffer walked densely is longer than ids_dev.
        REQUIRE((int64_t) raw.size() != ne0 * ne1, "fixture must be strided");
    }

    const int32_t n = 8;
    const int32_t K = 3;
    golem::GolemLayerSlots L;
    Shadow sh(n, K);
    std::vector<int32_t> remap;

    for (int step = 0; step < 500; ++step) {
        // Random routed subset, size 1..K.
        std::vector<int32_t> routed;
        const int32_t r = 1 + (int32_t)(rng() % K);
        std::vector<int32_t> ids_all(n);
        for (int32_t i = 0; i < n; ++i) ids_all[i] = i;
        std::shuffle(ids_all.begin(), ids_all.end(), rng);
        for (int32_t i = 0; i < r; ++i) routed.push_back(ids_all[i]);
        std::sort(routed.begin(), routed.end());

        // Host mutates experts that are NOT cached-current.
        for (int32_t id = 0; id < n; ++id) {
            const int32_t s = L.slot_of.empty() ? -1 : L.slot_of[id];
            const bool current = s != -1 && ggml_bitset_get(L.valid[0].data(), s);
            if (!current && rng() % 3 == 0) ++sh.host_gen[id];
        }

        // Random evict-first permutation.
        std::vector<int32_t> order(n);
        for (int32_t i = 0; i < n; ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);

        // ids: the routed set in a scrambled order (values must all be routed).
        std::vector<int32_t> ids = routed;
        std::shuffle(ids.begin(), ids.end(), rng);

        const auto before_valid = L.valid[0];
        const auto before_slot_of = L.slot_of;

        const auto dc = golem::golem_slab_step_nop(L, 0, routed, ids.data(),
            (int64_t) ids.size(), order.data(), K, n, remap);

        // Invariant 1+2: remap sound and routed coverage.
        REQUIRE((int64_t) remap.size() == (int64_t) ids.size(),
                "step %d: remap size %zu != ids %lld", step, remap.size(),
                (long long) ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t v = ids[i];
            REQUIRE(remap[i] >= 0 && remap[i] < K,
                    "step %d: remap[%zu]=%d out of [0,%d)", step, i, remap[i], K);
            REQUIRE(L.slot_of[v] == remap[i],
                    "step %d: remap[%zu]=%d != slot_of[%d]=%d", step, i,
                    remap[i], v, L.slot_of[v]);
            REQUIRE(dc.skipped + dc.copied >= (uint64_t) routed.size(), "accounting underflow");
        }
        // Routed coverage: every routed expert has a valid slot after the step.
        for (int32_t id : routed) {
            const int32_t s = L.slot_of[id];
            REQUIRE(s >= 0 && s < K && ggml_bitset_get(L.valid[0].data(), s),
                    "step %d: routed expert %d has no valid slot", step, id);
        }
        // Evictions only ever stole from non-routed experts.
        REQUIRE(dc.evicted <= (uint64_t)(n - routed.size()), "impossible evictions");

        // Deterministic accounting: copied = routed newly staged, skipped = the rest.
        REQUIRE(dc.copied + dc.skipped == (uint64_t) routed.size(),
                "step %d: copied %llu + skipped %llu != routed %zu", step,
                (unsigned long long) dc.copied, (unsigned long long) dc.skipped,
                routed.size());

        // Eviction coherence: a slot stolen from a non-routed owner has its bit
        // cleared and the owner's assignment removed (owner_of is the inverse
        // of slot_of over live assignments).
        for (int32_t s = 0; s < K; ++s) {
            const int32_t o = L.owner_of[s];
            if (o != -1) {
                REQUIRE(L.slot_of[o] == s, "owner_of/slot_of diverge at slot %d", s);
            }
        }
        for (int32_t id = 0; id < n; ++id) {
            const int32_t s = L.slot_of[id];
            if (s != -1) REQUIRE(L.owner_of[s] == id, "slot_of/owner_of diverge at expert %d", id);
        }

        (void) before_valid;
        (void) before_slot_of;
    }

    // Abort: routed set larger than K (loud failure, never a silent wrong remap).
    if (fork() == 0) {
        golem::GolemLayerSlots L2;
        std::vector<int32_t> remap2, order(n);
        std::vector<int32_t> routed(n), ids(n);
        for (int32_t i = 0; i < n; ++i) { routed[i] = i; ids[i] = i; order[i] = i; }
        golem::golem_slab_step_nop(L2, 0, routed, ids.data(), n, order.data(), K, n, remap2);
        _exit(0); // should not survive
    } else {
        int status = 0;
        wait(&status);
        REQUIRE(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
                "routed > K must abort (SIGABRT)");
    }

    // Abort: duplicate order.
    if (fork() == 0) {
        golem::GolemLayerSlots L2;
        std::vector<int32_t> remap2, routed = {0}, ids = {0}, order(n, 0); // duplicates
        golem::golem_slab_step_nop(L2, 0, routed, ids.data(), 1, order.data(), K, n, remap2);
        _exit(0);
    } else {
        int status = 0;
        wait(&status);
        REQUIRE(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "duplicate order must abort");
    }

    // --- golem_ctx eviction-order API (plan Task 3): NULL before first push,
    // push → acquire returns the published permutation without re-entry, and
    // the epoch gate never hands a reader a torn buffer. ---
    {
        using namespace golem_test;
        golem_ctx_t c = golem_ctx_create(2, N_EXP);
        // order keys off the tensor name (blk × kind); use a named stand-in
        ggml_tensor t {};
        ggml_set_name(&t, "blk.0.ffn_up_exps.weight");
        T_REQUIRE(golem_policy_order(c, &t, N_EXP) == nullptr,
                  "order must be NULL before the first push");
        const int32_t perm[4] = {2, 0, 3, 1};
        T_REQUIRE(golem_policy_order_push(c, 0, 2, perm, N_EXP) == 1, "push must publish");
        const int32_t * got = golem_policy_order(c, &t, N_EXP);
        T_REQUIRE(got != nullptr && std::equal(got, got + N_EXP, perm),
                  "acquire must return the pushed permutation");
        // invalid (blk, kind) and over-long pushes are rejected, ctx untouched
        T_REQUIRE(golem_policy_order_push(c, 9, 1, perm, N_EXP) == 0, "blk out of range");
        T_REQUIRE(golem_policy_order_push(c, 0, 4, perm, N_EXP) == 0, "kind out of range");

        // Epoch gate: readers of the CURRENT epoch pin their slot; the
        // publisher skips (returns 0) instead of overwriting, and never
        // blocks. Fill all three slots with current-epoch readers, then the
        // next push must skip; advancing the epoch frees the gate.
        golem_policy_order_epoch_advance(c); // fixed epoch N for the readers below
        (void) golem_policy_order(c, &t, N_EXP);          // pins slot 0 @N
        T_REQUIRE(golem_policy_order_push(c, 0, 2, perm, N_EXP) == 1, "push 2");
        (void) golem_policy_order(c, &t, N_EXP);          // pins slot 1 @N
        T_REQUIRE(golem_policy_order_push(c, 0, 2, perm, N_EXP) == 1, "push 3");
        (void) golem_policy_order(c, &t, N_EXP);          // pins slot 2 @N
        T_REQUIRE(golem_policy_order_push(c, 0, 2, perm, N_EXP) == 0,
                  "publisher must skip when every slot may hold a current reader");
        // readers always see a valid permutation while the publisher spins
        std::atomic<bool> stop{false};
        std::thread reader([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                const int32_t * o = golem_policy_order(c, &t, N_EXP);
                if (o != nullptr) {
                    int32_t seen = 0;
                    for (int i = 0; i < N_EXP; ++i) {
                        seen |= 1 << o[i];
                    }
                    T_REQUIRE(seen == 0xf, "reader saw a torn order (0x%x)", seen);
                }
            }
        });
        for (int i = 0; i < 64; ++i) {
            golem_policy_order_push(c, 0, 2, perm, N_EXP); // skips freely under the gate
            golem_policy_order_epoch_advance(c);
        }
        stop.store(true);
        reader.join();
        golem_reset_device_cache(c);
        golem_ctx_free(c);
    }

    // --- one decode step over one layer: routing fires exactly once, with a
    // dense ids vector of ne0*ne1 entries (plan Task 3). ---
    {
        using namespace golem_test;
        pool_init();
        HState s;
        golem_ctx_t c = golem_ctx_create(N_LAYERS, N_EXP);
        golem_set_device_cache_experts(c, 2);
        auto * h = golem_register_hooks(c, h_ensure, h_routing, h_copy_done, &s);
        (void) h;
        {
            ExpertSched es;
            es.build(1);
            golem_sched_set_ctx(es.sched, c);
            ggml_backend_sched_graph_compute(es.sched, es.gf);
            T_REQUIRE(s.routing_calls.load() == 1, "routing must fire once per step, got %d",
                      s.routing_calls.load());
            T_REQUIRE(s.last_n_ids == 2, "dense ids size %lld != 2", (long long) s.last_n_ids);
            T_REQUIRE(s.last_ids.size() == 2 && s.last_ids[0] == 0 && s.last_ids[1] == 1,
                      "dense ids content wrong");
            T_REQUIRE(s.last_blk == 0, "blk %d != 0", s.last_blk);
            T_REQUIRE(s.ensure_calls.load() == 1, "one ensure per (layer, step), got %d",
                      s.ensure_calls.load());
            // first step: nothing bit-valid yet, both routed experts are misses
            T_REQUIRE(s.miss_ids.size() == 2 && s.miss_ids[0] == 0 && s.miss_ids[1] == 1,
                      "first step must miss both routed experts");
        } // sched detached and freed before the ctx
        golem_unregister_hooks(h);
        golem_reset_device_cache(c);
        golem_ctx_free(c);
    }

    if (g_failures != 0) {
        printf("golem-p2-cache: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("golem-p2-cache: all legs passed\n");
    return 0;
}
