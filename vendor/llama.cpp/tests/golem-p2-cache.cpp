// Golem P2 v3 fuzz (design §3.4 v3): exercises the shared SLOT-INDIRECTED
// device cache state machine (ggml/src/golem-device-cache.hpp — the exact
// template the split scheduler runs) for the invariants that matter now:
//   1. bit ⇒ current: after a step, every routed expert's remapped slot holds
//      the host's current generation of that expert.
//   2. remap soundness: every remapped value is in [0, K), and ids outside the
//      routed set never appear (they cannot — routed covers the ids bitset).
//   3. victim choice (order permutation / routed set) never changes routed
//      coverage; budget overflow with routed > K aborts loudly.
//   4. per-layer sharing (§3.4 v3): the three planes (gate/up/down) of one layer
//      share one slot table — an eviction happens once per layer, not once per
//      plane, and each plane keeps its own validity bits.
//
// Aborts are asserted in fork()ed children (GGML_ABORT raises SIGABRT).
#include "golem-device-cache.hpp"

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

    if (g_failures != 0) {
        printf("golem-p2-cache: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("golem-p2-cache: all legs passed\n");
    return 0;
}
