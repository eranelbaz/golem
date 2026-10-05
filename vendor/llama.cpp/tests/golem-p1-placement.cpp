// Golem P1 placement counters: ggml_backend_sched_split_graph counts expert
// MUL_MAT_ID nodes (weights named per golem_expert_stem_kind) by the device
// type of their assigned backend. A CPU-only scheduler must count the expert
// node as on_cpu and ignore a non-expert MUL_MAT_ID.

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "golem-hooks.h"

static uint64_t load(int i) {
    const auto *c = reinterpret_cast<const std::atomic<uint64_t> *>(golem_get_expert_placement_stats());
    return c[i].load(std::memory_order_relaxed);
}

static ggml_tensor *mmid(ggml_context *ctx, const char *weight_name) {
    ggml_tensor *as = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 4);
    ggml_set_name(as, weight_name);
    ggml_tensor *b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 1);
    ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    return ggml_mul_mat_id(ctx, as, b, ids);
}

int main() {
    if (golem_hooks_abi_version() != GOLEM_HOOKS_ABI_VERSION) {
        printf("FAIL: abi %d != header %d\n", golem_hooks_abi_version(), GOLEM_HOOKS_ABI_VERSION);
        return 1;
    }
    ggml_init_params p = {16 * 1024 * 1024, nullptr, true};
    ggml_context *ctx = ggml_init(p);
    ggml_tensor *expert = mmid(ctx, "blk.1.ffn_up_exps.weight");
    ggml_tensor *other = mmid(ctx, "blk.1.ffn_up_shexp.weight");
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, expert);
    ggml_build_forward_expand(gf, other);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    const uint64_t gpu0 = load(0), cpu0 = load(1);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        printf("FAIL: alloc_graph\n");
        return 1;
    }
    const uint64_t dgpu = load(0) - gpu0, dcpu = load(1) - cpu0;
    int rc = 0;
    if (dcpu != 1 || dgpu != 0) {
        printf("FAIL: on_cpu +%llu (want 1), on_gpu +%llu (want 0)\n",
               (unsigned long long) dcpu, (unsigned long long) dgpu);
        rc = 1;
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    if (rc == 0) printf("OK\n");
    return rc;
}
