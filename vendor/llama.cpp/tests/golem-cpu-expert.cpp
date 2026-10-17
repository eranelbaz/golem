// Golem O1 CPU expert kernel (design §1 O1): the worker exported by ggml-cpu
// as "golem_cpu_expert_iface" must match a ggml CPU graph of the same expert
// FFN (mul_mat gate/up → swiglu[_clamp] → mul_mat down → Σ w·y) for every
// plane type of the real target, at both plane widths (ne0 256 and 4096).
// Tolerance: per element |cpu - ref| <= 1e-3 · ||ref||∞. The clamped case
// asserts that > 10% of the activations actually hit the limit.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "golem-cpu-expert.h"

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

constexpr int N_EXP = 3;

std::vector<char> quantize(ggml_type type, const std::vector<float> &src, int64_t ne0, int64_t nrows) {
    std::vector<char> dst(ggml_row_size(type, ne0) * nrows);
    ggml_quantize_chunk(type, src.data(), dst.data(), 0, nrows, ne0, nullptr);
    return dst;
}

void run_case(const golem_cpu_expert_iface *ifc, void *worker, ggml_type type, int64_t n_embd,
              int64_t n_ff, float limit, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto rnd = [&](size_t n) {
        std::vector<float> v(n);
        for (auto &f : v) f = dist(rng);
        return v;
    };

    std::vector<float> x = rnd(n_embd);
    const std::vector<float> w = {0.5f, 0.3f, 0.2f};
    std::vector<std::vector<char>> qg(N_EXP), qu(N_EXP), qd(N_EXP);
    for (int e = 0; e < N_EXP; ++e) {
        qg[e] = quantize(type, rnd(n_embd * n_ff), n_embd, n_ff);
        qu[e] = quantize(type, rnd(n_embd * n_ff), n_embd, n_ff);
        qd[e] = quantize(type, rnd(n_ff * n_embd), n_ff, n_embd);
    }

    // Reference: one ggml CPU graph over the same quantized planes.
    const size_t mem = 3 * N_EXP * qg[0].size() + (size_t) 64 * 1024 * 1024;
    ggml_init_params ip = {mem, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *tx = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    memcpy(tx->data, x.data(), n_embd * sizeof(float));
    ggml_tensor *ref = nullptr;
    std::vector<ggml_tensor *> gs, us;
    for (int e = 0; e < N_EXP; ++e) {
        ggml_tensor *wg = ggml_new_tensor_2d(ctx, type, n_embd, n_ff);
        ggml_tensor *wu = ggml_new_tensor_2d(ctx, type, n_embd, n_ff);
        ggml_tensor *wd = ggml_new_tensor_2d(ctx, type, n_ff, n_embd);
        memcpy(wg->data, qg[e].data(), qg[e].size());
        memcpy(wu->data, qu[e].data(), qu[e].size());
        memcpy(wd->data, qd[e].data(), qd[e].size());
        ggml_tensor *g = ggml_mul_mat(ctx, wg, tx);
        ggml_tensor *u = ggml_mul_mat(ctx, wu, tx);
        gs.push_back(g);
        us.push_back(u);
        ggml_tensor *h = limit > 1e-6f ? ggml_swiglu_clamp(ctx, g, u, limit) : ggml_swiglu_split(ctx, g, u);
        ggml_tensor *y = ggml_scale(ctx, ggml_mul_mat(ctx, wd, h), w[e]);
        ref = ref ? ggml_add(ctx, ref, y) : y;
    }
    ggml_cgraph *gf = ggml_new_graph(ctx);
    for (int e = 0; e < N_EXP; ++e) {
        ggml_build_forward_expand(gf, gs[e]);
        ggml_build_forward_expand(gf, us[e]);
    }
    ggml_build_forward_expand(gf, ref);
    REQUIRE(ggml_graph_compute_with_ctx(ctx, gf, 4) == GGML_STATUS_SUCCESS, "reference compute failed");

    if (limit > 1e-6f) {
        int64_t clamped = 0;
        for (int e = 0; e < N_EXP; ++e) {
            const float *g = (const float *) gs[e]->data, *u = (const float *) us[e]->data;
            for (int64_t k = 0; k < n_ff; ++k) clamped += (g[k] > limit || std::fabs(u[k]) > limit);
        }
        REQUIRE(clamped * 10 > N_EXP * n_ff, "%s: only %lld/%lld activations clamped", ggml_type_name(type),
                (long long) clamped, (long long) (N_EXP * n_ff));
    }

    golem_cpu_expert_job job = {};
    job.x = x.data();
    job.n_embd = n_embd;
    job.n_ff = n_ff;
    job.n = N_EXP;
    job.gate.type = job.up.type = job.down.type = type;
    job.gate.row_bytes = job.up.row_bytes = ggml_row_size(type, n_embd);
    job.down.row_bytes = ggml_row_size(type, n_ff);
    for (int e = 0; e < N_EXP; ++e) {
        job.gate.data[e] = qg[e].data();
        job.up.data[e] = qu[e].data();
        job.down.data[e] = qd[e].data();
    }
    job.w = w.data();
    job.limit = limit;
    std::vector<float> acc(n_embd, NAN);
    job.acc = acc.data();
    ifc->submit(worker, &job);
    const double ms = ifc->wait(worker);
    REQUIRE(ms >= 0.0, "wait returned %f ms", ms);
    REQUIRE(ifc->wait(worker) == 0.0, "second wait must report no job");

    const float *r = (const float *) ref->data;
    float rmax = 0.0f, emax = 0.0f;
    for (int64_t i = 0; i < n_embd; ++i) rmax = std::max(rmax, std::fabs(r[i]));
    for (int64_t i = 0; i < n_embd; ++i) emax = std::max(emax, std::fabs(acc[i] - r[i]));
    REQUIRE(rmax > 0.0f, "degenerate reference");
    REQUIRE(emax <= 1e-3f * rmax, "%s n_embd=%lld n_ff=%lld limit=%g: max|cpu-ref|=%g > 1e-3*%g",
            ggml_type_name(type), (long long) n_embd, (long long) n_ff, limit, emax, rmax);
    printf("%-7s n_embd=%-5lld n_ff=%-5lld limit=%-4g max|cpu-ref|/||ref||=%.2e (%.3f ms)\n",
           ggml_type_name(type), (long long) n_embd, (long long) n_ff, limit, emax / rmax, ms);
    ggml_free(ctx);
}

} // namespace

int main() {
    auto *ifc = (const golem_cpu_expert_iface *) ggml_backend_reg_get_proc_address(
        ggml_backend_cpu_reg(), "golem_cpu_expert_iface");
    REQUIRE(ifc != nullptr, "ggml-cpu must export golem_cpu_expert_iface");
    if (ifc == nullptr) {
        return 1;
    }
    void *worker = ifc->create(4);
    REQUIRE(worker != nullptr, "create(4) failed");
    REQUIRE(ifc->wait(worker) == 0.0, "wait with nothing submitted must return 0");

    std::mt19937 rng(0x676f6c656d);
    const ggml_type types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS};
    // (n_embd, n_ff): gate/up planes have ne0 = n_embd, down has ne0 = n_ff.
    const int64_t shapes[][2] = {{256, 256}, {4096, 256}, {256, 4096}};
    for (ggml_type t : types) {
        for (const auto &s : shapes) {
            run_case(ifc, worker, t, s[0], s[1], 0.25f, rng);
            run_case(ifc, worker, t, s[0], s[1], 0.0f, rng);
        }
    }
    ifc->destroy(worker);

    if (g_failures != 0) {
        printf("golem-cpu-expert: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("golem-cpu-expert: all legs passed\n");
    return 0;
}
