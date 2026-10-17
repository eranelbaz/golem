// Golem CPU expert math (design §1 O1). ggml-base cannot call into the CPU
// backend, so ggml-cpu exports this table through
// ggml_backend_reg_get_proc_address(reg, "golem_cpu_expert_iface").
// One job = one layer's CPU-assigned experts for one decode token:
//   acc = Σ_e w[e] · down_e(swiglu_clamp(gate_e·x, up_e·x, limit)).
#ifndef GOLEM_CPU_EXPERT_H
#define GOLEM_CPU_EXPERT_H

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#define GOLEM_CPU_EXPERT_MAX 64

#ifdef __cplusplus
extern "C" {
#endif

struct golem_cpu_expert_plane {
    const void *data[GOLEM_CPU_EXPERT_MAX];
    enum ggml_type type;
    size_t row_bytes;
};

struct golem_cpu_expert_job {
    const float *x;          // [n_embd] FFN input (one token)
    int64_t n_embd, n_ff;
    int32_t n;               // experts in this job (<= GOLEM_CPU_EXPERT_MAX)
    struct golem_cpu_expert_plane gate, up, down; // gate/up: n_ff rows of n_embd; down: n_embd rows of n_ff
    const float *w;          // [n] routing weights
    float limit;             // swiglu_clamp limit; <= 1e-6 = plain swiglu
    float *acc;              // [n_embd] output, overwritten
};

struct golem_cpu_expert_iface {
    void * (*create)(int n_threads);
    void (*destroy)(void *w);
    // Copies the job and returns immediately; x, w, plane data and acc must
    // stay valid until wait returns.
    void (*submit)(void *w, const struct golem_cpu_expert_job *job);
    // Blocks until the submitted job is done; returns its wall ms (0 if none).
    double (*wait)(void *w);
};

const struct golem_cpu_expert_iface * golem_cpu_expert_get_iface(void);

#ifdef __cplusplus
}
#endif

#endif
