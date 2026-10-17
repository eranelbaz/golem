// Golem CPU expert math worker (design §1 O1): a persistent thread pool that
// computes one layer's CPU-assigned experts for one decode token from pool
// memory in place. Scratch grows on the first job of a shape and is reused,
// so steady-state jobs do no heap allocation.
#include "golem-cpu-expert.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "vec.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct golem_cpu_expert_worker {
    std::vector<std::thread> threads;
    int nth = 0;

    std::mutex mtx;
    std::condition_variable cv_start, cv_done;
    uint64_t gen = 0;      // bumped per submitted job (under mtx)
    int n_done = 0;        // threads finished with the current gen (under mtx)
    bool pending = false;  // a job is submitted and not yet waited (under mtx)
    bool quit = false;

    // Phase barrier (sense-reversing); per job, never per expert.
    std::atomic<int> bar_count{0};
    std::atomic<int> bar_sense{0};

    golem_cpu_expert_job job{};
    std::chrono::steady_clock::time_point t_submit, t_end;

    // Scratch: g/u = [n][n_ff] f32 (h overwrites g), xq_g/xq_u = x in the
    // gate/up vec_dot types, hq = [n] rows of h in down's vec_dot type.
    std::vector<float> g, u;
    std::vector<char> xq_g, xq_u, hq;
    size_t hq_row = 0;
};

template <typename T>
void golem_grow(std::vector<T> &v, size_t n) {
    if (v.size() < n) {
        v.resize(n);
    }
}

void golem_quantize_row(enum ggml_type vdt, const float *src, void *dst, int64_t n) {
    if (vdt == GGML_TYPE_F32) {
        memcpy(dst, src, n * sizeof(float));
        return;
    }
    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(vdt)->from_float;
    GGML_ASSERT(from_float != nullptr);
    from_float(src, dst, n);
}

void golem_barrier(golem_cpu_expert_worker *W, int &sense) {
    sense ^= 1;
    if (W->bar_count.fetch_add(1, std::memory_order_acq_rel) == W->nth - 1) {
        W->bar_count.store(0, std::memory_order_relaxed);
        W->bar_sense.store(sense, std::memory_order_release);
        return;
    }
    while (W->bar_sense.load(std::memory_order_acquire) != sense) {
        std::this_thread::yield();
    }
}

void golem_range(int64_t n, int ith, int nth, int64_t &b, int64_t &e) {
    const int64_t d = (n + nth - 1) / nth;
    b = std::min(n, d * ith);
    e = std::min(n, b + d);
}

void golem_run(golem_cpu_expert_worker *W, int ith, int &sense) {
    const golem_cpu_expert_job &J = W->job;
    const int nth = W->nth;
    const int64_t n_embd = J.n_embd, n_ff = J.n_ff;
    const auto *tg = ggml_get_type_traits_cpu(J.gate.type);
    const auto *tu = ggml_get_type_traits_cpu(J.up.type);
    const auto *td = ggml_get_type_traits_cpu(J.down.type);

    if (ith == 0) {
        golem_quantize_row(tg->vec_dot_type, J.x, W->xq_g.data(), n_embd);
        if (tu->vec_dot_type != tg->vec_dot_type) {
            golem_quantize_row(tu->vec_dot_type, J.x, W->xq_u.data(), n_embd);
        }
    }
    golem_barrier(W, sense);

    // Phase A: (plane, expert, row) of gate and up.
    const void *xq_u = tu->vec_dot_type != tg->vec_dot_type ? W->xq_u.data() : W->xq_g.data();
    {
        int64_t b, e;
        golem_range(2 * J.n * n_ff, ith, nth, b, e);
        for (int64_t i = b; i < e; ++i) {
            const bool is_up = i >= J.n * n_ff;
            const int64_t j = is_up ? i - J.n * n_ff : i;
            const int64_t ex = j / n_ff, r = j % n_ff;
            const golem_cpu_expert_plane &P = is_up ? J.up : J.gate;
            const char *row = (const char *) P.data[ex] + r * P.row_bytes;
            float *dst = (is_up ? W->u.data() : W->g.data()) + j;
            (is_up ? tu : tg)->vec_dot((int) n_embd, dst, 0, row, 0, is_up ? xq_u : W->xq_g.data(), 0, 1);
        }
    }
    golem_barrier(W, sense);

    // Phase B: activation (same op as the graph's ffn_moe_swiglu[_limited])
    // and quantize h into down's vec_dot type, per expert.
    {
        int64_t b, e;
        golem_range(J.n, ith, nth, b, e);
        for (int64_t ex = b; ex < e; ++ex) {
            float *g = W->g.data() + ex * n_ff;
            const float *u = W->u.data() + ex * n_ff;
            if (J.limit > 1e-6f) {
                const float limit = J.limit;
                for (int64_t k = 0; k < n_ff; ++k) {
                    const float gate = std::min(g[k], limit);
                    const float up = std::clamp(u[k], -limit, limit);
                    g[k] = gate / (1.f + expf(-gate)) * up;
                }
            } else {
                ggml_vec_swiglu_f32((int) n_ff, g, g, u);
            }
            golem_quantize_row(td->vec_dot_type, g, W->hq.data() + ex * W->hq_row, n_ff);
        }
    }
    golem_barrier(W, sense);

    // Phase C: each thread owns a disjoint range of output rows.
    {
        int64_t b, e;
        golem_range(n_embd, ith, nth, b, e);
        for (int64_t r = b; r < e; ++r) {
            float sum = 0.0f;
            for (int32_t ex = 0; ex < J.n; ++ex) {
                const char *row = (const char *) J.down.data[ex] + r * J.down.row_bytes;
                float y = 0.0f;
                td->vec_dot((int) n_ff, &y, 0, row, 0, W->hq.data() + ex * W->hq_row, 0, 1);
                sum += J.w[ex] * y;
            }
            J.acc[r] = sum;
        }
    }
}

void golem_thread_main(golem_cpu_expert_worker *W, int ith) {
    uint64_t seen = 0;
    int sense = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(W->mtx);
            W->cv_start.wait(lk, [&] { return W->quit || W->gen != seen; });
            if (W->quit) {
                return;
            }
            seen = W->gen;
        }
        golem_run(W, ith, sense);
        std::lock_guard<std::mutex> lk(W->mtx);
        if (++W->n_done == W->nth) {
            W->t_end = std::chrono::steady_clock::now();
            W->cv_done.notify_all();
        }
    }
}

void * golem_cpu_expert_create(int n_threads) {
    if (n_threads < 1) {
        return nullptr;
    }
    auto *W = new golem_cpu_expert_worker;
    W->nth = n_threads;
    W->threads.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        W->threads.emplace_back(golem_thread_main, W, i);
    }
    return W;
}

void golem_cpu_expert_destroy(void *w) {
    auto *W = (golem_cpu_expert_worker *) w;
    if (W == nullptr) {
        return;
    }
    {
        std::unique_lock<std::mutex> lk(W->mtx);
        W->cv_done.wait(lk, [&] { return !W->pending || W->n_done == W->nth; });
        W->quit = true;
    }
    W->cv_start.notify_all();
    for (auto &t : W->threads) {
        t.join();
    }
    delete W;
}

void golem_cpu_expert_submit(void *w, const golem_cpu_expert_job *job) {
    auto *W = (golem_cpu_expert_worker *) w;
    GGML_ASSERT(W != nullptr && job != nullptr);
    GGML_ASSERT(job->n >= 0 && job->n <= GOLEM_CPU_EXPERT_MAX);
    GGML_ASSERT(job->x != nullptr && job->acc != nullptr && (job->n == 0 || job->w != nullptr));
    const enum ggml_type vg = ggml_get_type_traits_cpu(job->gate.type)->vec_dot_type;
    const enum ggml_type vu = ggml_get_type_traits_cpu(job->up.type)->vec_dot_type;
    const enum ggml_type vd = ggml_get_type_traits_cpu(job->down.type)->vec_dot_type;
    GGML_ASSERT(job->n_embd % ggml_blck_size(vg) == 0 && job->n_embd % ggml_blck_size(vu) == 0);
    GGML_ASSERT(job->n_ff % ggml_blck_size(vd) == 0);
    GGML_ASSERT(job->n_embd % ggml_blck_size(job->gate.type) == 0 && job->n_embd % ggml_blck_size(job->up.type) == 0);
    GGML_ASSERT(job->n_ff % ggml_blck_size(job->down.type) == 0);

    std::unique_lock<std::mutex> lk(W->mtx);
    GGML_ASSERT(!W->pending && "golem_cpu_expert: submit while a job is pending");
    W->job = *job;
    golem_grow(W->g, (size_t) (job->n * job->n_ff));
    golem_grow(W->u, (size_t) (job->n * job->n_ff));
    golem_grow(W->xq_g, ggml_row_size(vg, job->n_embd));
    golem_grow(W->xq_u, ggml_row_size(vu, job->n_embd));
    W->hq_row = ggml_row_size(vd, job->n_ff);
    golem_grow(W->hq, (size_t) job->n * W->hq_row);
    W->t_submit = std::chrono::steady_clock::now();
    W->n_done = 0;
    W->pending = true;
    ++W->gen;
    lk.unlock();
    W->cv_start.notify_all();
}

double golem_cpu_expert_wait(void *w) {
    auto *W = (golem_cpu_expert_worker *) w;
    GGML_ASSERT(W != nullptr);
    std::unique_lock<std::mutex> lk(W->mtx);
    if (!W->pending) {
        return 0.0;
    }
    W->cv_done.wait(lk, [&] { return W->n_done == W->nth; });
    W->pending = false;
    return std::chrono::duration<double, std::milli>(W->t_end - W->t_submit).count();
}

const golem_cpu_expert_iface g_golem_cpu_expert_iface = {
    /* .create  = */ golem_cpu_expert_create,
    /* .destroy = */ golem_cpu_expert_destroy,
    /* .submit  = */ golem_cpu_expert_submit,
    /* .wait    = */ golem_cpu_expert_wait,
};

} // namespace

const struct golem_cpu_expert_iface * golem_cpu_expert_get_iface(void) {
    return &g_golem_cpu_expert_iface;
}
