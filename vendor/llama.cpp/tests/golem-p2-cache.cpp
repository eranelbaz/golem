// Golem P2 fuzz: exercises the shared device expert-copy cache state machine
// (ggml/src/golem-device-cache.hpp — the exact template the split scheduler
// runs) for the T4-fuzzed invariants:
//   1. bit ⇒ current: a routed expert's device generation always equals the
//      host generation after a step (copied this step, or copied when its bit
//      was set and not overwritten since).
//   4. victim choice (order permutation / routed set / budget) never changes
//      the routed-coverage result — budget overflow included.
// plus: stale-entry invalidation (fingerprint mismatch ⇒ reset) and
// malformed-order aborts (duplicate / out-of-range / NULL).
//
// Aborts are asserted in fork()ed children (GGML_ABORT raises SIGABRT).
#include "golem-device-cache.hpp"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using golem::GolemDeviceCache;

namespace {

int g_failures = 0;

#define REQUIRE(cond, ...)                                                        \
    do {                                                                          \
        if (!(cond)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                           \
            printf(__VA_ARGS__);                                                  \
            printf("\n");                                                         \
            g_failures++;                                                         \
        }                                                                         \
    } while (0)

// Shadow model: per-expert host generation vs device-observed generation.
struct Shadow {
    std::vector<uint64_t> host_gen;
    std::vector<uint64_t> dev_gen; // (uint64_t)-1 = never written
    size_t expert_size;
    int32_t n;

    explicit Shadow(int32_t n_, size_t es) :
            host_gen(n_, 1), dev_gen(n_, ~(ggml_bitset_t) 0), expert_size(es), n(n_) {}

    // host mutates a random subset of experts between steps. Only experts that
    // are NOT bit-valid may mutate: MoE weights are immutable while a bit is
    // valid (that IS the bit⇒current premise); mutating under a valid bit would
    // be a host-side write the cache cannot observe.
    void mutate(std::mt19937_64 &rng, const std::vector<ggml_bitset_t> &valid) {
        for (int32_t id = 0; id < n; ++id) {
            if (!ggml_bitset_get(valid.data(), id) && rng() % 3 == 0) {
                ++host_gen[id];
            }
        }
    }

    // physical write of one run (+ padding tail when pad_ok)
    void apply_copy(int32_t f, int32_t l, bool pad_ok) {
        REQUIRE(f >= 0 && l < n && f <= l, "on_copy: run [%d, %d] out of range (n=%d)", f, l, n);
        for (int32_t id = f; id <= l; ++id) {
            dev_gen[id] = host_gen[id];
        }
        if (pad_ok) {
            REQUIRE(l + 1 < n, "pad_ok with last_id+1 == n_experts");
            dev_gen[l + 1] = host_gen[l + 1]; // padding writes that expert's own host bytes
        }
    }
};

void check_state(const GolemDeviceCache &st, const Shadow &sh, const std::vector<int32_t> &routed) {
    // invariant 1 (bit ⇒ current) + routed ⇒ current
    for (int32_t id = 0; id < sh.n; ++id) {
        if (ggml_bitset_get(st.valid.data(), id)) {
            REQUIRE(sh.dev_gen[id] == sh.host_gen[id],
                    "bit-valid expert %d is stale (dev=%" PRIu64 " host=%" PRIu64 ")", id,
                    sh.dev_gen[id], sh.host_gen[id]);
        }
    }
    for (int32_t id : routed) {
        REQUIRE(sh.dev_gen[id] == sh.host_gen[id],
                "routed expert %d not read-current (dev=%" PRIu64 " host=%" PRIu64 ")", id,
                sh.dev_gen[id], sh.host_gen[id]);
    }
    // no accounting drift: valid_bytes == popcount(valid) * expert_size
    size_t pc = golem::golem_dc_popcount_bytes(st.valid, 1);
    REQUIRE(st.valid_bytes == pc * sh.expert_size, "valid_bytes drift: %zu != popcount %zu * %zu",
            st.valid_bytes, pc, sh.expert_size);
}

size_t routed_bytes(const std::vector<int32_t> &routed, size_t expert_size) {
    return routed.size() * expert_size;
}

std::vector<int32_t> rand_perm(std::mt19937_64 rng, int32_t n) {
    std::vector<int32_t> v(n);
    for (int32_t i = 0; i < n; ++i) {
        v[i] = i;
    }
    for (int32_t i = n - 1; i > 0; --i) {
        std::swap(v[i], v[rng() % (i + 1)]);
    }
    return v;
}

// Deterministic leg: fresh cache, budget 0 (never evict) — exact accounting
// (lives below the overflow test; both defined once).

// Budget-overflow leg: routed bytes > budget must terminate (soft-cap rule,
// routed experts never evicted) and still leave the routed set read-current.
void test_budget_overflow_terminates() {
    const int32_t n = 32;
    const size_t es = 64;
    GolemDeviceCache st;
    Shadow sh(n, es);
    std::mt19937_64 rng(7);
    const void *host = (const void *) 0x2;
    auto step = [&](const std::vector<int32_t> &routed, size_t budget) {
        return golem::golem_dc_step_impl(st, routed, rand_perm(rng, n).data(), budget, es, n, host,
                                         [&](int32_t f, int32_t l, bool p) {
                                             sh.apply_copy(f, l, p);
                                         });
    };

    // fill all experts as valid
    std::vector<int32_t> all;
    for (int32_t i = 0; i < n; ++i) {
        all.push_back(i);
    }
    step(all, 0);
    REQUIRE(st.valid_bytes == n * es, "warmup valid_bytes %zu", st.valid_bytes);
    check_state(st, sh, all);

    // tiny routed set, budget below current footprint ⇒ evictions, no overflow
    std::vector<int32_t> routed = {3, 4};
    golem::GolemDcCounters c = step(routed, 4 * es);
    REQUIRE(c.evicted > 0, "expected evictions under soft cap");
    REQUIRE(st.valid_bytes <= 4 * es, "soft cap violated: valid_bytes=%zu budget=%zu",
            st.valid_bytes, (size_t) (4 * es));
    check_state(st, sh, routed);

    // overflow: routed bytes alone exceed the budget — must still terminate
    // and keep every routed expert read-current
    std::vector<int32_t> big = {10, 11, 12, 13, 14};
    step(big, 2 * es);
    check_state(st, sh, big);
}

// Deterministic leg: fresh cache, budget 0 (never evict) — exact accounting.
void test_exact_accounting() {
    const int32_t n = 17;
    const size_t es = 100;
    GolemDeviceCache st;
    Shadow sh(n, es);
    std::vector<int32_t> routed = {0, 1, 2, 5, 6, 16}; // runs: [0,2], [5,6], [16,16]
    auto order = rand_perm(std::mt19937_64(1), n);

    golem::GolemDcCounters c = golem::golem_dc_step_impl(st, routed, order.data(), 0, es, n,
                                                         (const void *) 0x1,
                                                         [&](int32_t f, int32_t l, bool p) {
                                                             sh.apply_copy(f, l, p);
                                                         });
    // fresh: all runs copied; pad_ok for runs followed by an invalid expert
    // ([0,2]→+es pad, [5,6]→+es pad, [16,16]→ no expert 17 ⇒ no pad)
    REQUIRE(c.copied == routed_bytes(routed, es) + 2 * es, "step1 copied %zu", c.copied);
    REQUIRE(c.skipped == 0, "step1 skipped %zu", c.skipped);
    REQUIRE(c.evicted == 0, "step1 evicted %zu", c.evicted);
    check_state(st, sh, routed);

    // second identical step, budget 0: every run fully valid ⇒ all skipped, no
    // copy, no padding tail (skipped runs never reach the pad decision)
    golem::GolemDcCounters c2 = golem::golem_dc_step_impl(st, routed, order.data(), 0, es, n,
                                                          (const void *) 0x1,
                                                          [&](int32_t f, int32_t l, bool p) {
                                                              sh.apply_copy(f, l, p);
                                                          });
    REQUIRE(c2.copied == 0, "step2 copied %zu", c2.copied);
    REQUIRE(c2.skipped == routed_bytes(routed, es), "step2 skipped %zu", c2.skipped);
    REQUIRE(c2.evicted == 0, "step2 evicted %zu", c2.evicted);
    check_state(st, sh, routed);
}

// Stale-entry invalidation: fingerprint mismatch (valid.size() vs n_experts,
// owner, expert_size) resets to all-invalid.
void test_stale_reset() {
    const int32_t n = 8;
    const size_t es = 32;
    std::mt19937_64 rng(11);
    auto order = rand_perm(rng, n);
    std::vector<int32_t> routed = {0, 1};

    { // valid.size() mismatch
        GolemDeviceCache st;
        st.valid.assign(3, ~(ggml_bitset_t) 0); // wrong size, "all valid"
        st.valid_bytes = 3 * 64;
        st.owner       = (const void *) 0x3;
        st.expert_size = es;
        golem::golem_dc_step(st, routed, order.data(), 0, es, n, (const void *) 0x3);
        REQUIRE(st.valid.size() == (size_t) ggml_bitset_size(n), "size-mismatch not reset");
        REQUIRE(!ggml_bitset_get(st.valid.data(), 7), "size-mismatch state not invalidated");
    }
    { // owner mismatch (recycled device address aliased to a different tensor)
        GolemDeviceCache st;
        st.valid.assign(ggml_bitset_size(n), ~(ggml_bitset_t) 0);
        st.valid_bytes = n * es;
        st.owner       = (const void *) 0xdead;
        st.expert_size = es;
        golem::golem_dc_step(st, routed, order.data(), 0, es, n, (const void *) 0x3);
        REQUIRE(st.owner == (const void *) 0x3, "owner not re-fingerprinted");
        REQUIRE(!ggml_bitset_get(st.valid.data(), 5), "owner-mismatch state not invalidated");
        REQUIRE(st.valid_bytes == routed_bytes(routed, es), "owner-mismatch valid_bytes=%zu",
                st.valid_bytes);
    }
    { // expert_size mismatch
        GolemDeviceCache st;
        st.valid.assign(ggml_bitset_size(n), ~(ggml_bitset_t) 0);
        st.valid_bytes = n * 128;
        st.owner       = (const void *) 0x3;
        st.expert_size = 128;
        golem::golem_dc_step(st, routed, order.data(), 0, es, n, (const void *) 0x3);
        REQUIRE(st.expert_size == es, "expert_size not re-fingerprinted");
        REQUIRE(!ggml_bitset_get(st.valid.data(), 5), "size-mismatch state not invalidated");
    }
}

// Invariant 1 + 4 fuzz loop: random routed sets, random order permutations,
// random budgets (0 / small / exact / huge / random), host mutations between
// steps. Victim selection may differ; routed coverage must not.
void test_fuzz_random_steps() {
    std::mt19937_64 rng(0x6f6c656d); // "olem"
    for (int iter = 0; iter < 300; ++iter) {
        const int32_t n = 1 + (int32_t) (rng() % 64);
        const size_t es = 1 + rng() % 512;
        GolemDeviceCache st;
        Shadow sh(n, es);
        const void *host = (const void *) (uintptr_t) (0x10 + (iter % 3)); // varying fingerprint

        for (int step = 0; step < 12; ++step) {
            // random routed set (nonempty)
            std::vector<int32_t> routed;
            for (int32_t id = 0; id < n; ++id) {
                if (rng() % 3 == 0) {
                    routed.push_back(id);
                }
            }
            if (routed.empty()) {
                routed.push_back((int32_t) (rng() % n));
            }
            if (!st.valid.empty()) {
                sh.mutate(rng, st.valid);
            }

            size_t budget = 0;
            switch (rng() % 5) {
                case 0:
                    budget = 0; // full capacity
                    break;
                case 1:
                    budget = routed_bytes(routed, es) / 2; // below routed ⇒ overflow tolerated
                    break;
                case 2:
                    budget = routed_bytes(routed, es);
                    break;
                case 3:
                    budget = ~(size_t) 0;
                    break;
                default:
                    budget = rng() % (n * es + 1);
                    break;
            }

            golem::GolemDcCounters c = golem::golem_dc_step_impl(
                st, routed, rand_perm(rng, n).data(), budget, es, n, host,
                [&](int32_t f, int32_t l, bool p) { sh.apply_copy(f, l, p); });
            (void) c;
            check_state(st, sh, routed);
        }
    }
}

// Malformed-order legs run in a fork()ed child; the parent asserts SIGABRT.
void run_child(int leg) {
    const int32_t n = 8;
    const size_t es = 32;
    GolemDeviceCache st;
    st.valid.assign(ggml_bitset_size(n), 0);
    st.owner = (const void *) 0x5;
    st.expert_size = es;
    std::vector<int32_t> routed = {0, 1};
    switch (leg) {
        case 0:
        { // duplicate id
            std::vector<int32_t> order = {0, 1, 2, 3, 4, 5, 6, 6};
            golem::golem_dc_step(st, routed, order.data(), 0, es, n, (const void *) 0x5);
            break;
        }
        case 1:
        { // out-of-range id
            std::vector<int32_t> order = {0, 1, 2, 3, 4, 5, 6, 8};
            golem::golem_dc_step(st, routed, order.data(), 0, es, n, (const void *) 0x5);
            break;
        }
        case 2:
        { // negative id
            std::vector<int32_t> order = {0, 1, 2, 3, 4, 5, 6, -1};
            golem::golem_dc_step(st, routed, order.data(), 0, es, n, (const void *) 0x5);
            break;
        }
        case 3:
        { // short coverage: a repeat stands in for the missing id
            std::vector<int32_t> order = {0, 1, 2, 3, 4, 5, 6, 4};
            golem::golem_dc_step(st, routed, order.data(), 0, es, n, (const void *) 0x5);
            break;
        }
        case 4: // NULL order
            golem::golem_dc_step(st, routed, nullptr, 0, es, n, (const void *) 0x5);
            break;
    }
    printf("FAIL: malformed-order leg %d did not abort\n", leg);
    _exit(1);
}

void test_malformed_order_aborts() {
    for (int leg = 0; leg < 5; ++leg) {
        pid_t pid = fork();
        REQUIRE(pid >= 0, "fork failed");
        if (pid == 0) {
            run_child(leg); // never returns
        }
        int status = 0;
        waitpid(pid, &status, 0);
        REQUIRE(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
                "leg %d: expected SIGABRT (signaled=%d term=%d)", leg, WIFSIGNALED(status),
                WTERMSIG(status));
    }
}

} // namespace

int main() {
    test_exact_accounting();
    test_budget_overflow_terminates();
    test_stale_reset();
    test_fuzz_random_steps();
    test_malformed_order_aborts();
    if (g_failures != 0) {
        printf("golem-p2-cache: FAIL (%d assertion%s)\n", g_failures, g_failures == 1 ? "" : "s");
        return 1;
    }
    printf("golem-p2-cache: PASS (accounting, soft-cap, stale-reset, 300-step fuzz, abort legs)\n");
    return 0;
}
