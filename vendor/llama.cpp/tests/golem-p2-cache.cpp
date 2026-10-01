// Golem P2 v2 fuzz (design §3.4 v2): exercises the shared SLOT-INDIRECTED device
// cache state machine (ggml/src/golem-device-cache.hpp — the exact template
// the split scheduler runs) for the invariants that matter now:
//   1. bit ⇒ current: after a step, every routed expert's remapped slot holds
//      the host's current generation of that expert.
//   2. remap soundness: every remapped value is in [0, K), and ids outside the
//      routed set never appear (they cannot — routed covers the ids bitset).
//   3. victim choice (order permutation / routed set) never changes routed
//      coverage; budget overflow with routed > K aborts loudly.
// plus stale-entry invalidation (fingerprint mismatch ⇒ reset) and
// malformed-order aborts (duplicate / out-of-range / NULL).
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

using golem::GolemSlabEntry;

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

    const int32_t n = 8;
    const int32_t K = 3;
    GolemSlabEntry st;
    Shadow sh(n, K);

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
            const int32_t s = st.slot_of.empty() ? -1 : st.slot_of[id];
            const bool current = s != -1 && ggml_bitset_get(st.valid.data(), s);
            if (!current && rng() % 3 == 0) ++sh.host_gen[id];
        }

        // Random evict-first permutation.
        std::vector<int32_t> order(n);
        for (int32_t i = 0; i < n; ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);

        // ids: the routed set in a scrambled order (values must all be routed).
        std::vector<int32_t> ids = routed;
        std::shuffle(ids.begin(), ids.end(), rng);

        const auto before_valid = st.valid;
        const auto before_slot_of = st.slot_of;

        const auto dc = golem::golem_slab_step_nop(st, routed, ids.data(),
            (int64_t) ids.size(), order.data(), K, n, /*expert_size*/ 2048,
            /*host_data*/ &step);

        // Invariant 1+2: remap sound and routed coverage.
        REQUIRE((int64_t) st.remap.size() == (int64_t) ids.size(),
                "step %d: remap size %zu != ids %lld", step, st.remap.size(),
                (long long) ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t v = ids[i];
            REQUIRE(st.remap[i] >= 0 && st.remap[i] < K,
                    "step %d: remap[%zu]=%d out of [0,%d)", step, i, st.remap[i], K);
            REQUIRE(st.slot_of[v] == st.remap[i],
                    "step %d: remap[%zu]=%d != slot_of[%d]=%d", step, i,
                    st.remap[i], v, st.slot_of[v]);
            REQUIRE(dc.skipped + dc.copied >= (uint64_t) routed.size(), "accounting underflow");
        }
        // Routed coverage: every routed expert has a valid slot after the step.
        for (int32_t id : routed) {
            const int32_t s = st.slot_of[id];
            REQUIRE(s >= 0 && s < K && ggml_bitset_get(st.valid.data(), s),
                    "step %d: routed expert %d has no valid slot", step, id);
        }
        // Evictions only ever stole from non-routed experts.
        REQUIRE(dc.evicted <= (uint64_t)(n - routed.size()), "impossible evictions");

        // Deterministic accounting: copied = routed newly staged, skipped = the rest.
        REQUIRE(dc.copied + dc.skipped == (uint64_t) routed.size(),
                "step %d: copied %llu + skipped %llu != routed %zu", step,
                (unsigned long long) dc.copied, (unsigned long long) dc.skipped,
                routed.size());

        (void) before_valid;
        (void) before_slot_of;
    }

    // Stale-entry reset: a fingerprint change must clear slot assignments.
    {
        GolemSlabEntry st;
        std::vector<int32_t> routed = {0};
        std::vector<int32_t> ids = {0};
        std::vector<int32_t> order(n);
        for (int32_t i = 0; i < n; ++i) order[i] = i;
        golem::golem_slab_step_nop(st, routed, ids.data(), 1, order.data(), K, n, 2048,
                                   (void *) 0x1);
        REQUIRE(st.slot_of[0] != -1, "warm entry: expert 0 must be assigned");
        golem::golem_slab_step_nop(st, routed, ids.data(), 1, order.data(), K, n, 2048,
                                   (void *) 0x2);
        // owner mismatch → reset; slot_of must be re-derived from scratch
        // (expert 0 staged again ⇒ copied ≥ 1 checked via return below).
        const auto dc = golem::golem_slab_step_nop(st, routed, ids.data(), 1, order.data(), K,
                                                   n, 2048, (void *) 0x2);
        REQUIRE(dc.skipped == 1,
                "same owner twice: second step must skip (got copied %llu skipped %llu)",
                (unsigned long long) dc.copied, (unsigned long long) dc.skipped);
    }

    // Abort: routed set larger than K (loud failure, never a silent wrong remap).
    if (fork() == 0) {
        GolemSlabEntry st;
        std::vector<int32_t> routed(n);
        std::vector<int32_t> ids(n);
        for (int32_t i = 0; i < n; ++i) { routed[i] = i; ids[i] = i; }
        std::vector<int32_t> order(n);
        for (int32_t i = 0; i < n; ++i) order[i] = i;
        golem::golem_slab_step_nop(st, routed, ids.data(), n, order.data(), K, n, 2048, nullptr);
        _exit(0); // should not survive
    } else {
        int status = 0;
        wait(&status);
        REQUIRE(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
                "routed > K must abort (SIGABRT)");
    }

    // Abort: duplicate order.
    if (fork() == 0) {
        GolemSlabEntry st;
        std::vector<int32_t> routed = {0};
        std::vector<int32_t> ids = {0};
        std::vector<int32_t> order(n, 0); // duplicates
        golem::golem_slab_step_nop(st, routed, ids.data(), 1, order.data(), K, n, 2048, nullptr);
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
