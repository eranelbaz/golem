// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "golem-p3-copy.h"
#include "golem-cpu-expert.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

// Hook STATE lives in a per-model golem_ctx (golem-hook-context.hpp); the
// sched-attached ctx pointer is captured once per compute entry. The
// golem_register_hooks / ctx-lifecycle plumbing stays in the P1 section near
// the end.
#include "golem-hook-context.hpp"

static int golem_expert_stem_kind(const char *name); // defined in the P1 section
static int golem_tensor_is_expert(const struct ggml_tensor *t); // defined in the P1 section
static int golem_tensor_blk(const char *name); // defined in the P1 section (layer key of the shared slot table)

// Copy a pooled expert run slot by slot: the plan and its slot-overrun
// invariant are documented in golem-p3-copy.h (shared with the regression test).
void golem_copy_pooled_expert_run(void *dst, size_t expert_offset, size_t expert_size,
                                  size_t padding_len, const void *const *ptrs, size_t n,
                                  golem_set_copy_fn set_fn, void *ud) {
    if (n == 0) {
        return;
    }
    size_t off = expert_offset;
    for (size_t i = 0; i < n; ++i) {
        set_fn(dst, ptrs[i], off, expert_size, ud);
        off += expert_size;
    }
    if (padding_len != 0) {
        set_fn(dst, ptrs[n - 1], off, padding_len, ud);
    }
}

// Process-global placement stats: graph-build placements per device across
// the process — a windowed tripwire that outlives any ctx (documented in the
// header).
struct golem_expert_placement_stats {
    std::atomic<uint64_t> on_gpu; // expert MUL_MAT_ID nodes assigned to a non-CPU backend
    std::atomic<uint64_t> on_cpu; // expert MUL_MAT_ID nodes assigned to the CPU backend (I2 violation unless opted in)
};
static_assert(sizeof(golem_expert_placement_stats) == 2 * sizeof(uint64_t), "golem_expert_placement_stats layout must match the Rust [AtomicU64; 2] FFI cast");
static golem_expert_placement_stats golem_h_placement_stats = {};
const golem_expert_placement_stats *golem_get_expert_placement_stats(void) { return &golem_h_placement_stats; }


// Allocate the pinned staging under the SAME condition as the ids_dev realloc
// (ids shape change): the caller invokes this only from inside that realloc
// block, so the pinned registration tracks the same device as ids_dev -- never
// a size-only reuse check, which could keep a buffer registered against a
// different device after a backend switch.
static void golem_ids_pin_alloc(golem::GolemSlabEntry &st, ggml_backend_t split_backend, int64_t ne0, int64_t ne1) {
    ggml_backend_dev_t dev = ggml_backend_get_device(split_backend);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    if (buft == nullptr) {
        GGML_ABORT("golem P2: device exposes no host buffer type; cannot pin remapped-ids staging");
    }
    st.ids_pin = nullptr;
    st.ids_pin_bytes = 0;
    if (st.ids_pin_buf) {
        ggml_backend_buffer_free(st.ids_pin_buf);
        st.ids_pin_buf = nullptr;
    }
    st.ids_pin_buf = ggml_backend_buft_alloc_buffer(buft, (size_t) ne0 * (size_t) ne1 * sizeof(int32_t));
    GGML_ASSERT(st.ids_pin_buf != nullptr);
    st.ids_pin_buft = buft;
    st.ids_pin = ggml_backend_buffer_get_base(st.ids_pin_buf);
    st.ids_pin_bytes = (size_t) ne0 * (size_t) ne1 * sizeof(int32_t);
    st.ids_pin_step = 0;
}

namespace golem { // same namespace as the shared header's inline types
} // namespace golem // (the dc/layers maps + their mutex moved into golem_ctx)

void golem_set_prefill_stream(golem_ctx_t ctx, int32_t on, int32_t n_experts,
                              int32_t max_plane_stride, float routed_max) {
    if (ctx == NULL) {
        return;
    }
    if (on && (ctx->stream_source == NULL || ctx->stream_release == NULL)) {
        GGML_ABORT("golem_set_prefill_stream without the stream hooks registered");
    }
    if (on && (n_experts <= 0 || max_plane_stride <= 0 || !(routed_max > 0.0f && routed_max <= 1.0f))) {
        GGML_ABORT("golem_set_prefill_stream: bad config");
    }
    ctx->stream_n_experts = n_experts;
    ctx->stream_max_stride = max_plane_stride;
    ctx->stream_routed_max = routed_max;
    ctx->stream_on.store(on, std::memory_order_release);
}

int32_t golem_set_async_h2d(golem_ctx_t ctx, int32_t on) {
    if (ctx == NULL) {
        return -1;
    }
    // Upstream concurrent-stream forking (GGML_CUDA_GRAPH_OPT=1) uses the
    // stream indices the copy stream lives on.
    const char * env = getenv("GGML_CUDA_GRAPH_OPT");
    if (on && env != nullptr && atoi(env) == 1) {
        return -1;
    }
    ctx->async_h2d = on ? 1 : 0;
    return 0;
}

// Blocks until the copy stream is idle. Every release of H2D sources (pool
// leases, stream staging, evict destinations) and every slab free runs after
// this: ggml_backend_synchronize covers the compute stream only.
static void golem_drain_copy_stream(golem_ctx_t ctx) {
    if (ctx != NULL && ctx->cs_backend != nullptr) {
        ctx->cs->sync(ctx->cs_backend);
    }
}

static void golem_cs_free(golem_ctx_t ctx) {
    if (ctx->cs == nullptr) {
        return;
    }
    if (ctx->cs_backend != nullptr) {
        ctx->cs->sync(ctx->cs_backend);
    }
    ctx->cs->event_free(ctx->cs_ev_ready);
    ctx->cs->event_free(ctx->cs_ev_done);
    for (int i = 0; i < golem_ctx::GOLEM_CS_RING; ++i) {
        if (ctx->cs_t0[i] != nullptr) {
            ctx->cs->event_free(ctx->cs_t0[i]);
            ctx->cs->event_free(ctx->cs_t1[i]);
            ctx->cs_t0[i] = ctx->cs_t1[i] = nullptr;
        }
    }
    ctx->cs_ev_ready = ctx->cs_ev_done = nullptr;
    ctx->cs_backend = nullptr;
    ctx->cs = nullptr;
    ctx->cs_t_n = 0;
    ctx->cs_t_overflow = false;
    ctx->cs_bytes = 0;
}

// Orders the copy stream after everything already queued on the split
// backend's compute stream (prior computes reading the destination, O10
// eviction D2H of the slots about to be refilled). Called before each
// plane's copy-stream H2D batch; `first` opens the split's timing pair;
// `wait` is false when nothing was queued on compute since the last wait.
static void golem_cs_begin(golem_ctx_t ctx, ggml_backend_t be, bool first, bool wait) {
    if (ctx->cs_backend == nullptr) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
        ctx->cs = (const golem_copy_stream_iface *) ggml_backend_reg_get_proc_address(reg, "golem_copy_stream_iface");
        if (ctx->cs == nullptr) {
            GGML_ABORT("golem async_h2d on a backend without a copy stream");
        }
        ctx->cs_backend = be;
        ctx->cs_ev_ready = ctx->cs->event_new(be, 0);
        ctx->cs_ev_done = ctx->cs->event_new(be, 0);
        for (int i = 0; i < golem_ctx::GOLEM_CS_RING; ++i) {
            ctx->cs_t0[i] = ctx->cs->event_new(be, 1);
            ctx->cs_t1[i] = ctx->cs->event_new(be, 1);
        }
    }
    GGML_ASSERT(ctx->cs_backend == be && "golem async_h2d: one device backend per ctx");
    if (wait) {
        ctx->cs->record_compute(be, ctx->cs_ev_ready);
        ctx->cs->copy_wait(be, ctx->cs_ev_ready);
    }
    if (first) {
        if (ctx->cs_t_n < golem_ctx::GOLEM_CS_RING) {
            ctx->cs->record_copy(be, ctx->cs_t0[ctx->cs_t_n]);
        } else {
            ctx->cs_t_overflow = true;
        }
    }
}

static void golem_cs_set(golem_ctx_t ctx, ggml_tensor * t, const void * src, size_t off, size_t len) {
    GGML_ASSERT(t->data != NULL && off + len <= ggml_nbytes(t) && "tensor write out of bounds");
    ctx->cs->set_async(ctx->cs_backend, t, src, off, len);
    ctx->cs_bytes += len;
}

// Closes a copying split: the compute stream waits for every H2D the split
// queued on the copy stream before the split's graph reads it.
static void golem_cs_join(golem_ctx_t ctx) {
    ggml_backend_t be = ctx->cs_backend;
    if (ctx->cs_t_n < golem_ctx::GOLEM_CS_RING) {
        ctx->cs->record_copy(be, ctx->cs_t1[ctx->cs_t_n]);
        ctx->cs_t_n++;
    }
    ctx->cs->record_copy(be, ctx->cs_ev_done);
    ctx->cs->compute_wait(be, ctx->cs_ev_done);
}

int32_t golem_take_h2d_timing(golem_ctx_t ctx, double * ms, uint64_t * bytes) {
    if (ctx == NULL || !ctx->async_h2d) {
        return 0;
    }
    golem_drain_copy_stream(ctx);
    double total = 0.0;
    for (int32_t i = 0; i < ctx->cs_t_n; ++i) {
        total += ctx->cs->elapsed_ms(ctx->cs_t0[i], ctx->cs_t1[i]);
    }
    const bool ok = !ctx->cs_t_overflow;
    *ms = total;
    *bytes = ctx->cs_bytes;
    ctx->cs_t_n = 0;
    ctx->cs_t_overflow = false;
    ctx->cs_bytes = 0;
    return ok ? 1 : 0;
}

int32_t golem_set_cpu_expert_math(golem_ctx_t ctx, int32_t on, float frac, int32_t n_threads,
                                  golem_cpu_lease_fn lease) {
    if (ctx == NULL || ctx->routing == nullptr || ctx->cem_on.load(std::memory_order_acquire)) {
        return -1;
    }
    if (on == 0) {
        return 0;
    }
    if (!(frac > 0.0f && frac <= 1.0f) || n_threads < 1 || lease == nullptr) {
        return -1;
    }
    ctx->cem_frac = frac;
    ctx->cem_threads = n_threads;
    ctx->cem_lease = lease;
    ctx->cem_on.store(1, std::memory_order_release);
    return 0;
}

int32_t golem_cpu_expert_math_on(golem_ctx_t ctx) {
    return ctx != NULL && ctx->cem_on.load(std::memory_order_acquire) ? 1 : 0;
}

int32_t golem_take_cpu_expert_stats(golem_ctx_t ctx, uint64_t * experts, uint64_t * bytes, double * ms,
                                    double * join_wait_ms, uint64_t * padded) {
    if (!golem_cpu_expert_math_on(ctx)) {
        return 0;
    }
    *experts = ctx->cem_experts;
    *bytes = ctx->cem_bytes;
    *ms = ctx->cem_ms;
    *join_wait_ms = ctx->cem_join_wait_ms;
    *padded = ctx->cem_padded;
    ctx->cem_experts = 0;
    ctx->cem_bytes = 0;
    ctx->cem_ms = 0.0;
    ctx->cem_join_wait_ms = 0.0;
    ctx->cem_padded = 0;
    return 1;
}

// Joins the submitted CPU expert job, if any: worker busy ms and the
// decode thread's blocked time accrue separately (design §1 O1).
static void golem_cem_join(golem_ctx_t ctx) {
    if (ctx->cem_pending_blk < 0) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ctx->cem_ms += ctx->cem_if->wait(ctx->cem_worker);
    ctx->cem_join_wait_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ctx->cem_pending_blk = -1;
}

// Frees the prefill-stream slabs + events (device buffers: the owner must run
// this while the backend is alive -- golem_reset_device_cache).
static void golem_stream_slabs_free(golem_ctx_t ctx) {
    for (auto & sl : ctx->stream_slabs) {
        if (sl.event != nullptr) {
            if (sl.used) {
                ggml_backend_event_synchronize(sl.event);
            }
            ggml_backend_event_free(sl.event);
            sl.event = nullptr;
        }
        if (sl.buf != nullptr) {
            ggml_backend_buffer_free(sl.buf);
            sl.buf = nullptr;
        }
        if (sl.ctx != nullptr) {
            ggml_free(sl.ctx);
            sl.ctx = nullptr;
        }
        sl.tensor = nullptr;
        sl.fp = 0;
        sl.used = false;
    }
    ctx->stream_seq.store(0, std::memory_order_release);
}

void golem_reset_device_cache(golem_ctx_t ctx) {
    if (ctx == NULL) {
        return;
    }
    GGML_ASSERT(ctx->cem_pending_blk == -1);
    if (ctx->cem_acc_buf != nullptr) {
        ggml_backend_buffer_free(ctx->cem_acc_buf);
        ctx->cem_acc_buf = nullptr;
        ctx->cem_acc = nullptr;
        ctx->cem_acc_n_embd = 0;
    }
    golem_cs_free(ctx);
    golem_stream_slabs_free(ctx);
    std::lock_guard<std::mutex> lk(ctx->dc_mutex);
    // The invalidation seam is also the reclamation point (model reload):
    // free each entry's slab and remapped-ids device regions.
    for (auto & kv : ctx->dc) {
        if (kv.second.buf != nullptr) {
            ggml_backend_buffer_free(kv.second.buf);
        }
        if (kv.second.ctx != nullptr) {
            ggml_free(kv.second.ctx);
        }
        if (kv.second.ids_buf != nullptr) {
            ggml_backend_buffer_free(kv.second.ids_buf);
        }
        if (kv.second.ids_pin_buf != nullptr) {
            ggml_backend_buffer_free(kv.second.ids_pin_buf);
            kv.second.ids_pin_buf = nullptr;
            kv.second.ids_pin = nullptr;
            kv.second.ids_pin_buft = nullptr;
            kv.second.ids_pin_step = 0;
        }
        if (kv.second.ids_ctx != nullptr) {
            ggml_free(kv.second.ids_ctx);
        }
    }
    ctx->dc.clear();
    ctx->layers.clear();
}

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    void * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    void * callback_realloc_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int graph_size;
    int prev_graph_size;

    int debug_realloc;
    // Golem per-model hook context (ABI v5); NULL = hookless sched takes the
    // upstream paths verbatim. Attached via golem_sched_set_ctx (libllama
    // re-applies after every ggml_backend_sched_new).
    golem_ctx_t golem_ctx;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // Golem I2 proof (design §2.3): node backends are final here and src[0]
    // still names the weight (pass 5 swaps it for the per-backend copy).
    {
        uint64_t on_gpu = 0, on_cpu = 0;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT_ID || !golem_tensor_is_expert(node->src[0])) {
                continue;
            }
            const int b = tensor_backend_id(node);
            GGML_ASSERT(b != -1);
            // Zero-token ubatch (prefill/generation chunk with n_outputs == 0):
            // the MoE trunk is gated through get_rows(out_ids), so expert MMID
            // nodes collapse to ne[2] == 0 — no routed experts, no expert math
            // runs regardless of backend (upstream offload_op rejects batch 0
            // by design). Real-token builds always differ in n_outputs, so the
            // graph is rebuilt and placed on the GPU. Exempt from the I2 proof.
            if (node->ne[2] == 0) {
                continue;
            }
            if (ggml_backend_dev_type(ggml_backend_get_device(sched->backends[b])) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                on_cpu++;
            } else {
                on_gpu++;
            }
        }
        if (on_gpu) golem_h_placement_stats.on_gpu.fetch_add(on_gpu, std::memory_order_relaxed);
        if (on_cpu) golem_h_placement_stats.on_cpu.fetch_add(on_cpu, std::memory_order_relaxed);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            // Golem device slab: when the sched's ctx has a
                            // routing hook, a routed-expert weights tensor
                            // feeding MUL_MAT_ID never uses the
                            // full-size compute-buffer cpy —
                            // the data lives in a persistent per-entry device
                            // slab (K+1 planes, budgeted by
                            // golem_set_device_cache_experts; the +1 is a
                            // non-NaN MMQ padding slot). The cpy is a tiny
                            // placeholder so the compute buffer does not carry
                            // a full-size copy of every expert tensor, and the
                            // node srcs are redirected to the slab per compute
                            // (and restored after — kernels capture pointers
                            // at launch; graphs are off while hooks are set).
                            // Unset hooks keep the full upstream dup.
                            struct ggml_tensor * tensor_copy;
                            const bool golem_expert_input =
                                sched->golem_ctx != NULL &&
                                sched->golem_ctx->routing != NULL &&
                                sched->golem_ctx->hooked.load(std::memory_order_relaxed) &&
                                node->op == GGML_OP_MUL_MAT_ID && j == 0 &&
                                golem_expert_stem_kind(src->name) != 0;
                            if (golem_expert_input) {
                                tensor_copy = ggml_new_tensor_1d(sched->ctx, src->type, ggml_blck_size(src->type));
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            }
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        const bool graph_size_changed = sched->prev_graph_size != sched->graph_size;

        if (sched->callback_realloc != NULL) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && !graph_size_changed;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {
    return t->buffer != NULL &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

namespace golem {
// monotonically increasing decode-step id (whole sched step, all split
// segments); keys the thread-local ids readback cache in compute_splits
static thread_local std::atomic<uint64_t> step_id{0};
} // namespace golem

// Golem: one in-flight release — the decrement and the join wakeup share the
// join mutex, so an unregister waiter that checked the predicate before this
// decrement cannot miss the wakeup (lost-wakeup).
static void golem_in_flight_done(golem_ctx_t gctx) {
    {
        std::lock_guard<std::mutex> lk(gctx->join_mutex);
        gctx->in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }
    gctx->join_cv.notify_all();
}

// O1 graph names are matched exactly: "<prefix><blk>" and nothing after
// (the scheduler's "<name> (view)" deps and "<dev>#<name>#<c>" copies
// never match). Returns the blk, or -1.
static int golem_cem_name_blk(const char * name, const char * prefix) {
    const size_t pl = strlen(prefix);
    if (strncmp(name, prefix, pl) != 0 || name[pl] < '0' || name[pl] > '9') {
        return -1;
    }
    char * end = nullptr;
    const long v = strtol(name + pl, &end, 10);
    return (*end == '\0' && v >= 0 && v < INT_MAX) ? (int) v : -1;
}

// Per-blk pinned accumulators (design §1 O1): n_blk x n_embd f32 in one host
// buffer of the split device, (re)allocated once per n_embd. The old buffer
// is freed only after every backend drained its H2Ds. false = no pinned
// host buffer type or allocation failed.
static bool golem_cem_acc_ensure(golem_ctx_t gctx, ggml_backend_sched_t sched, ggml_backend_t split_backend,
                                 int32_t n_blk, int64_t n_embd) {
    if (gctx->cem_acc_buf != nullptr && gctx->cem_acc_n_embd == n_embd) {
        return true;
    }
    if (gctx->cem_acc_buf != nullptr) {
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }
        ggml_backend_buffer_free(gctx->cem_acc_buf);
        gctx->cem_acc_buf = nullptr;
        gctx->cem_acc = nullptr;
        gctx->cem_acc_n_embd = 0;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(split_backend));
    if (buft == nullptr) {
        return false;
    }
    gctx->cem_acc_buf = ggml_backend_buft_alloc_buffer(buft, (size_t) n_blk * (size_t) n_embd * sizeof(float));
    if (gctx->cem_acc_buf == nullptr) {
        return false;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(gctx->cem_acc_buf));
    gctx->cem_acc = (float *) ggml_backend_buffer_get_base(gctx->cem_acc_buf);
    gctx->cem_acc_n_embd = (int32_t) n_embd;
    return true;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // Golem ABI v5: capture the sched-attached ctx once per entry.
    golem_ctx_t gctx = sched->golem_ctx;
    // In-flight pin incremented BEFORE the hooked re-check: an unregister that
    // races this entry either sees the pin (and joins) or this entry sees
    // hooked == false and runs unhooked — never a half-registered step.
    bool golem_hooked = false;
    if (gctx != NULL) {
        gctx->epoch.fetch_add(1, std::memory_order_relaxed);
        gctx->in_flight.fetch_add(1, std::memory_order_acq_rel);
        golem_hooked = gctx->hooked.load(std::memory_order_acquire);
        if (!golem_hooked) {
            golem_in_flight_done(gctx);
        }
    }
    // The thread-local ids readback cache is valid WITHIN one step only: a
    // reused graph hands back the same ids-tensor pointer on the next decode,
    // and stale (previous-step) ids would copy the wrong experts. Reset per
    // entry (declaration hoisted above) — this also keeps hookless runs
    // byte-identical to upstream.
    static thread_local ggml_tensor * prev_ids_tensor = nullptr;
    prev_ids_tensor = nullptr;
    // O1 (design §1 O1): the per-blk accumulators and leases assume one
    // attached sched and the copy_done entry drain.
    const bool golem_cem_on = golem_hooked && gctx->cem_on.load(std::memory_order_relaxed) != 0;
    if (golem_cem_on) {
        GGML_ASSERT(gctx->cem_pending_blk == -1);
        if (gctx->attach_refs.load(std::memory_order_acquire) != 1 || gctx->copy_done == NULL) {
            golem_in_flight_done(gctx);
            return GGML_STATUS_FAILED;
        }
    }
    // Golem: release the host-pool leases taken during the PREVIOUS compute.
    // Every llama_decode holds exactly one ubatch (the engine caps its prefill
    // chunk at n_ubatch); the leases must survive until here so eviction cannot
    // recycle a slot mid-DMA, so drain the backends first (a no-op after the
    // caller's own synchronize).
    if (golem_hooked && gctx->copy_done != NULL) {
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }
        golem_drain_copy_stream(gctx);
        golem_cb_guard cb;
        gctx->copy_done(gctx->payload);
    }
    struct ggml_backend_sched_split * splits = sched->splits;

    // Monotonic per-call sequence for golem_split_seq-stamped state
    // (GolemSlabEntry::ids_pin_step same-step reuse detection).
    static std::atomic<uint64_t> golem_h_split_seq{0};
    const uint64_t golem_split_seq = golem_h_split_seq.fetch_add(1, std::memory_order_relaxed) + 1;

    // thread-local + capacity-stable: a steady-state decode step allocates
    // nothing on the compute path (AGENTS.md decode hot path)
    static thread_local std::vector<int32_t> ids;
    static thread_local std::vector<ggml_bitset_t> used_ids;
    // one merged routing readback per changed ids TENSOR per step: a layer's
    // planes may span several split segments, but the ids values are fixed
    // for the whole step, so only the first segment reads + routes.
    // (prev_ids_tensor is declared at the top of this function — reset there.)
    // Golem ABI v5 scratch: per-(layer, step) miss hoisting + slab plans.
    // Thread-local, capacity-stable — a decode step performs no heap
    // allocation on the ctx path after the first.
    struct golem_plane_plan {
        int mode = 0; // 0 = upstream, 1 = pooled (batched ensure), 2 = slab
        std::vector<int32_t> routed;                      // sorted routed expert ids
        std::vector<std::pair<int32_t, int32_t>> pending; // slab: (slot, expert id)
        std::vector<const void *> ptrs;                   // per routed id / per pending entry
        struct ggml_tensor * slab = nullptr;              // slab tensor for the async copies
        golem::GolemSlabEntry * st = nullptr;             // slab entry (map stable within the step)
        golem::GolemSlabCounters dc = {};
        size_t out_off = 0;                               // slice of the batched out_ptrs
    };
    static thread_local std::vector<golem_plane_plan> golem_planes;
    static thread_local std::vector<int32_t> golem_dense;    // dense ids (routing readback + remap)
    static thread_local std::vector<int32_t> golem_identity; // NULL-order fallback (identity permutation)
    // Exclusive residency (ABI v8, design §6.3 O10): this layer's fully-valid
    // slab evictions of the current pre-pass, as (slot, owner id); drained per
    // layer into one evict_batch call + per-plane D2H copies.
    static thread_local std::vector<std::pair<int32_t, int32_t>> golem_evicts;
    // Batched evict_batch scratch: owner ids, per-expert destinations (3 per
    // expert, kind-1 order) and per-expert keep flags.
    static thread_local std::vector<int32_t> golem_evict_ids;
    static thread_local std::vector<void *> golem_evict_out;
    static thread_local std::vector<int32_t> golem_evict_keep;
    // Exclusive D2H deferral: a layer's planes span several split segments
    // (fused MoE graphs), but evict_batch delivers all three plane
    // destinations at once, in the segment that triggered the eviction. A
    // plane's D2H can only be issued from the segment that owns that plane's
    // slab entry, so park (blk, slot → dst[3]) here until each plane's copy
    // has been issued (done bit per kind); drained in the slab copy phase
    // before that segment's staged H2D refills (stream order: the D2H must
    // read a stolen slot's bytes before the refill overwrites them). All
    // segments of one step run inside one compute_splits call, ahead of the
    // next copy_done drain. Entries whose planes never all drain are
    // reported abandoned at compute end (ABI v9), so nothing incomplete
    // survives to a copy_done commit.
    struct golem_evict_d2h_entry { int32_t blk; int32_t id; int32_t slot; void * dst[3]; uint32_t done; };
    static thread_local std::vector<golem_evict_d2h_entry> golem_evict_d2h;
    golem_evict_d2h.clear();
    // ABI v9 abandon scratch: expert ids of one blk's unfinished parked D2Hs.
    static thread_local std::vector<int32_t> golem_evict_abandon_ids;
    // ABI v9 abandon reporter: groups the still-parked (done != 7)
    // D2H entries by blk and hands their expert ids to the pool's
    // evict_abandon hook. Complete entries (done == 7) were removed
    // at their last drain, so whatever remains here is incomplete
    // by construction; the caller synchronizes the backends first.
    auto golem_evict_abandon_unfinished = [&]() {
        if (gctx->evict_abandon == NULL) {
            golem_evict_d2h.clear();
            return;
        }
        for (size_t r = 0; r < golem_evict_d2h.size();) {
            const int32_t blk_a = golem_evict_d2h[r].blk;
            golem_evict_abandon_ids.clear();
            size_t w = r;
            for (size_t q = r; q < golem_evict_d2h.size(); ++q) {
                if (golem_evict_d2h[q].blk == blk_a) {
                    golem_evict_abandon_ids.push_back(golem_evict_d2h[q].id);
                } else {
                    golem_evict_d2h[w++] = golem_evict_d2h[q];
                }
            }
            golem_evict_d2h.resize(w);
            {
                golem_cb_guard cb;
                gctx->evict_abandon(gctx->payload, blk_a,
                                    golem_evict_abandon_ids.data(),
                                    (int32_t) golem_evict_abandon_ids.size());
            }
        }
    };
    // P5: prediction nodes named golem_pred-<target blk>-<source blk>, indexed
    // once per compute (one strncmp per node) and delivered in the source
    // layer's pre-pass, right after its ids readback synchronized the device.
    struct golem_pred_ref {
        int src;
        int target;
        struct ggml_tensor * t;
    };
    static thread_local std::vector<golem_pred_ref> golem_preds;
    static thread_local std::vector<int32_t> golem_pred_ids;
    golem_preds.clear();
    if (golem_hooked && gctx->predict != NULL) {
        for (int i = 0; i < sched->graph.n_nodes; ++i) {
            struct ggml_tensor * t = sched->graph.nodes[i];
            int target = -1;
            int src = -1;
            if (strncmp(t->name, "golem_pred-", 11) == 0 && sscanf(t->name + 11, "%d-%d", &target, &src) == 2 &&
                t->type == GGML_TYPE_I32 && t->ne[1] == 1 && ggml_is_contiguous(t)) {
                golem_preds.push_back({src, target, t});
            }
        }
    }
    // O1 per-compute scan: per blk, [x, w, acc, gate, up, down] in
    // golem_cem_t[blk * 6 + k], the swiglu_clamp limit (0 when absent) and a
    // bad flag. Every acc must be a device-split input (the join writes it);
    // an acc read on the CPU or never copied fails the compute. Planes are the ORIGINAL host weights from split->inputs
    // (split_graph rewrote MUL_MAT_ID src[0] to a placeholder copy).
    // golem_cem_mask holds one ggml_bitset row per blk, valid only while
    // golem_cem_sel_step[blk] == golem_split_seq and golem_cem_n_cpu[blk] > 0.
    static thread_local std::vector<struct ggml_tensor *> golem_cem_t;
    static thread_local std::vector<float> golem_cem_limit;
    static thread_local std::vector<uint8_t> golem_cem_bad;
    static thread_local std::vector<uint8_t> golem_cem_has_acc;
    bool golem_cem_acc_misplaced = false;
    static thread_local std::vector<ggml_bitset_t> golem_cem_mask;
    static thread_local std::vector<uint64_t> golem_cem_sel_step;
    static thread_local std::vector<int32_t> golem_cem_n_cpu;
    static thread_local std::vector<float> golem_cem_x;
    static thread_local std::vector<float> golem_cem_wraw;
    static thread_local std::vector<float> golem_cem_w;
    const int32_t golem_cem_n_blk = golem_cem_on ? (int32_t) (gctx->orders.size() / 3) : 0;
    const size_t golem_cem_row = golem_cem_on ? ggml_bitset_size((size_t) std::max<int32_t>(gctx->n_experts, 1)) : 0;
    if (golem_cem_on) {
        golem_cem_t.assign((size_t) golem_cem_n_blk * 6, nullptr);
        golem_cem_limit.assign(golem_cem_n_blk, 0.0f);
        golem_cem_bad.assign(golem_cem_n_blk, 0);
        golem_cem_has_acc.assign(golem_cem_n_blk, 0);
        golem_cem_mask.resize((size_t) golem_cem_n_blk * golem_cem_row);
        golem_cem_sel_step.resize(golem_cem_n_blk, 0);
        golem_cem_n_cpu.resize(golem_cem_n_blk, 0);
        auto golem_cem_note = [&](struct ggml_tensor * t) {
            int b;
            if ((b = golem_cem_name_blk(t->name, "golem_cpu_x-")) >= 0 && b < golem_cem_n_blk) {
                golem_cem_t[(size_t) b * 6 + 0] = t;
            } else if ((b = golem_cem_name_blk(t->name, "golem_cpu_w-")) >= 0 && b < golem_cem_n_blk) {
                golem_cem_t[(size_t) b * 6 + 1] = t;
            } else if ((b = golem_cem_name_blk(t->name, "golem_cpu_acc-")) >= 0 && b < golem_cem_n_blk) {
                if (t->op != GGML_OP_NONE || !(t->flags & GGML_TENSOR_FLAG_INPUT)) {
                    golem_cem_bad[b] = 1;
                } else {
                    golem_cem_has_acc[b] = 1;
                }
            } else if ((b = golem_cem_name_blk(t->name, "ffn_moe_swiglu_limited-")) >= 0 && b < golem_cem_n_blk) {
                if (t->op == GGML_OP_GLU && ggml_get_glu_op(t) == GGML_GLU_OP_SWIGLU_CLAMP) {
                    golem_cem_limit[b] = ggml_get_op_params_f32(t, 3);
                } else {
                    golem_cem_bad[b] = 1;
                }
            }
        };
        for (int i = 0; i < sched->graph.n_nodes; ++i) {
            golem_cem_note(sched->graph.nodes[i]);
        }
        for (int i = 0; i < sched->graph.n_leafs; ++i) {
            golem_cem_note(sched->graph.leafs[i]);
        }
        for (int s = 0; s < sched->n_splits; ++s) {
            const struct ggml_backend_sched_split * sp = &sched->splits[s];
            const bool on_cpu = ggml_backend_dev_type(ggml_backend_get_device(sched->backends[sp->backend_id])) ==
                                GGML_BACKEND_DEVICE_TYPE_CPU;
            for (int j = 0; j < sp->n_inputs; ++j) {
                struct ggml_tensor * in = sp->inputs[j];
                const int kind = golem_expert_stem_kind(in->name);
                if (kind != 0) {
                    if (in->buffer != NULL && ggml_backend_buffer_get_usage(in->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                        ggml_backend_buffer_is_host(in->buffer)) {
                        const int b = golem_tensor_blk(in->name);
                        if (b >= 0 && b < golem_cem_n_blk) {
                            golem_cem_t[(size_t) b * 6 + 2 + kind] = in;
                        }
                    }
                    continue;
                }
                const int b = golem_cem_name_blk(in->name, "golem_cpu_acc-");
                if (b >= 0 && b < golem_cem_n_blk && in->op == GGML_OP_NONE && (in->flags & GGML_TENSOR_FLAG_INPUT)) {
                    if (on_cpu) {
                        golem_cem_acc_misplaced = true;
                    } else {
                        golem_cem_t[(size_t) b * 6 + 2] = in;
                    }
                }
            }
        }
        for (int32_t b = 0; b < golem_cem_n_blk; ++b) {
            if (golem_cem_has_acc[b] && golem_cem_t[(size_t) b * 6 + 2] == nullptr) {
                golem_cem_acc_misplaced = true;
            }
        }
    }
    static thread_local std::vector<int32_t> golem_miss_kinds, golem_miss_ids;
    static thread_local std::vector<const void *> golem_miss_out;
    // Prefill-stream scratch (ABI v10): the id list handed to stream_source
    // (routed set or the whole identity range) and its per-expert pointers.
    static thread_local std::vector<int32_t> golem_stream_ids;
    static thread_local std::vector<const void *> golem_stream_ptrs;
    // (src slot, original) of every MUL_MAT_ID src[0]/src[2] redirected to the
    // device slab / remapped-ids tensor. Restored after each split: llama
    // reuses the graph across decode steps, and the next step must read the
    // router's ids, not the previous step's slot numbers. Thread-local so
    // decode steps do not allocate.
    static thread_local std::vector<std::pair<ggml_tensor **, ggml_tensor *>> golem_ids_redirects;
    const auto golem_restore_ids = [&]() {
        for (auto & r : golem_ids_redirects) {
            *r.first = r.second;
        }
        golem_ids_redirects.clear();
    };
    golem_restore_ids();
    // O1 failure: join first, then the shared drain; the leases of
    // a never-run job drop at the next copy_done.
    auto golem_cem_fail = [&]() -> enum ggml_status {
        golem_cem_join(gctx);
        golem_restore_ids();
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }
        golem_drain_copy_stream(gctx);
        if (!golem_evict_d2h.empty()) {
            golem_evict_abandon_unfinished();
        }
        golem_in_flight_done(gctx);
        return GGML_STATUS_FAILED;
    };
    if (golem_cem_acc_misplaced) {
        return golem_cem_fail();
    }
    static thread_local std::vector<int32_t> golem_cem_cand;
    static thread_local std::vector<int32_t> golem_cem_took;
    static thread_local std::vector<const void *> golem_cem_ptrs;

    int prev_backend_id = -1;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        // async_h2d: did this split queue pooled H2D on the copy stream
        // (joined into the compute stream before the split's graph).
        bool golem_cs_used = false;
        // Copy stream already ordered after every compute-stream op queued
        // so far in this split (cleared by eviction D2H).
        bool golem_cs_fenced = false;

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // Golem: lazy ids read shared by the layer-level pre-pass and the
        // copy loop below; returns the resolved ids tensor (NULL = none).
        // With a routing hook, gathers the dense ids and fires the merged
        // routing readback ONCE per changed ids tensor.
        auto golem_ensure_ids = [&](int from_input_id, struct ggml_tensor * node,
                                    int64_t n_expert) -> struct ggml_tensor * {
            ggml_tensor * ids_tensor = node->src[2];
            ggml_backend_t ids_backend = split_backend;
            if (ggml_nelements(ids_tensor) == 0) {
                return nullptr;
            }
            // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
            // in that case, we use the original ids tensor
            for (int i = from_input_id + 1; i < split->n_inputs; i++) {
                if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                    ids_tensor = split->inputs[i];
                    ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                    break;
                }
            }
            if (ids_tensor != prev_ids_tensor) {
                // ids may be a strided view (argsort top-k: nb[1] = n_expert*4); read
                // its full extent, then index by nb[]
                ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                ggml_backend_synchronize(ids_backend);
                // find the used experts
                used_ids.clear();
                used_ids.resize(ggml_bitset_size(n_expert));
                for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                    for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                        int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                        GGML_ASSERT(id >= 0 && id < n_expert);
                        ggml_bitset_set(used_ids.data(), id);
                    }
                }
                prev_ids_tensor = ids_tensor;
                // Dense gather hoisted above the slab/upstream branch; the
                // merged routing readback (replaces policy_query +
                // notify_access) fires once per changed ids tensor.
                if (gctx != NULL && gctx->routing != NULL) {
                    golem::golem_gather_ids_dense(ids.data(), ids_tensor->ne[0], ids_tensor->ne[1],
                                                  ids_tensor->nb[0], ids_tensor->nb[1], golem_dense);
                    golem_cb_guard cb;
                    gctx->routing(gctx->payload, golem_tensor_blk(node->src[0]->name), node->src[0],
                                  golem_dense.data(), (int64_t) golem_dense.size(), (int32_t) n_expert);
                }
            }
            return ids_tensor;
        };

        // Golem: layer-level miss hoisting pre-pass over split->inputs,
        // grouped by golem_tensor_blk — ONE ensure_host_resident call per
        // (layer, step, split-segment) with all planes' misses (a layer
        // whose planes span several splits fires once per segment: counted,
        // not an error). The copy loop below only performs the physical
        // copies from the returned pointers. All plans live in the
        // thread-local golem_planes scratch.
        if (golem_hooked && gctx->ensure != NULL) {
            golem_planes.resize(split->n_inputs);
            for (auto & pl : golem_planes) {
                pl.mode = 0;
            }
            int done_blk = -1;
            for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
                struct ggml_tensor * input = split->inputs[input_id];
                if (golem_expert_stem_kind(input->name) == 0) {
                    continue;
                }
                struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
                // the layer's expert input feeds a MUL_MAT_ID node somewhere in
                // the split (a multi-plane layer spans several such nodes)
                struct ggml_tensor * node = nullptr;
                for (int ni = 0; ni < split->graph.n_nodes; ++ni) {
                    struct ggml_tensor * n2 = split->graph.nodes[ni];
                    if (n2->op == GGML_OP_MUL_MAT_ID && n2->src[0] == input_cpy) {
                        node = n2;
                        break;
                    }
                }
                if (!(node != nullptr &&
                      ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                      ggml_backend_buffer_is_host(input->buffer))) {
                    continue;
                }
                const int64_t n_expert = input->ne[2];
                // The pre-pass reads the ORIGINAL ids tensor: the split's ids
                // copies are only populated by the copy loop below, and the
                // original holds this step's values (the engine writes them
                // before the compute entry). Upstream's not-yet-copied
                // substitution stays in golem_ensure_ids for the verbatim
                // paths.
                // split_graph rewires node->src[2] to the split's ids COPY;
                // recover the original input (its data is written by the
                // engine before the compute entry and is not yet copied).
                struct ggml_tensor * ids_orig = node->src[2];
                for (int ii = 0; ii < split->n_inputs; ++ii) {
                    if (tensor_copy(split->inputs[ii], split_backend_id, sched->cur_copy) == ids_orig) {
                        ids_orig = split->inputs[ii];
                        break;
                    }
                }
                if (ggml_nelements(ids_orig) == 0) {
                    continue;
                }
                if (ids_orig != prev_ids_tensor) {
                    ggml_backend_t ids_backend = ggml_backend_sched_get_tensor_backend(sched, ids_orig);
                    ids.resize(ggml_nbytes(ids_orig) / sizeof(int32_t));
                    ggml_backend_tensor_get_async(ids_backend, ids_orig, ids.data(), 0, ggml_nbytes(ids_orig));
                    ggml_backend_synchronize(ids_backend);
                    used_ids.clear();
                    used_ids.resize(ggml_bitset_size(n_expert));
                    for (int64_t i1 = 0; i1 < ids_orig->ne[1]; i1++) {
                        for (int64_t i0 = 0; i0 < ids_orig->ne[0]; i0++) {
                            int32_t id = ids[i1 * ids_orig->nb[1]/sizeof(int32_t) + i0 * ids_orig->nb[0]/sizeof(int32_t)];
                            GGML_ASSERT(id >= 0 && id < n_expert);
                            ggml_bitset_set(used_ids.data(), id);
                        }
                    }
                    prev_ids_tensor = ids_orig;
                    // Dense gather hoisted above the slab/upstream branch; the
                    // merged routing readback (replaces policy_query +
                    // notify_access) fires once per changed ids tensor.
                    if (gctx->routing != NULL) {
                        golem::golem_gather_ids_dense(ids.data(), ids_orig->ne[0], ids_orig->ne[1],
                                                      ids_orig->nb[0], ids_orig->nb[1], golem_dense);
                        golem_cb_guard cb;
                        gctx->routing(gctx->payload, golem_tensor_blk(input->name), input,
                                      golem_dense.data(), (int64_t) golem_dense.size(), (int32_t) n_expert);
                    }
                    const int src_blk = golem_tensor_blk(input->name);
                    for (const auto & p : golem_preds) {
                        if (p.src != src_blk) {
                            continue;
                        }
                        GGML_ASSERT(ggml_nbytes(p.t) == (size_t) p.t->ne[0] * sizeof(int32_t));
                        golem_pred_ids.resize(p.t->ne[0]);
                        // Usually computed on the ids backend (already
                        // synchronized); otherwise drain its own backend.
                        ggml_backend_t pb = ggml_backend_sched_get_tensor_backend(sched, p.t);
                        if (pb != NULL && pb != ids_backend) {
                            ggml_backend_synchronize(pb);
                        }
                        ggml_backend_tensor_get(p.t, golem_pred_ids.data(), 0, ggml_nbytes(p.t));
                        golem_cb_guard cb;
                        gctx->predict(gctx->payload, src_blk, p.target, golem_pred_ids.data(),
                                      (int32_t) golem_pred_ids.size());
                    }
                }
                const int blk = golem_tensor_blk(input->name);
                if (blk == done_blk) {
                    continue; // already batched this layer in this segment
                }
                done_blk = blk;
                const bool slab_mode = gctx->routing != NULL;
                // Prefill stream (ABI v10, design section 7.6): armed per chunk by
                // the engine; takes the same single-GPU shape as the slab
                // path (one split per layer segment, n_copies == 1).
                const bool stream_mode = gctx->stream_on.load(std::memory_order_acquire) != 0;
                if ((slab_mode || stream_mode) && sched->n_copies != 1) {
                    GGML_ABORT("golem P2: requires n_copies == 1 (single GPU)");
                }
                // O1 selection + submit, once per (blk, compute), after the
                // ids readback and before plan.routed is built; one job in flight
                // at a time, so a second blk in the same pre-pass stays all-GPU.
                if (golem_cem_on && slab_mode && !stream_mode && ids_orig->ne[1] == 1 &&
                    blk >= 0 && blk < golem_cem_n_blk && golem_cem_sel_step[blk] != golem_split_seq &&
                    gctx->cem_pending_blk == -1) {
                    golem_cem_sel_step[blk] = golem_split_seq;
                    golem_cem_n_cpu[blk] = 0;
                    struct ggml_tensor * const * ct = &golem_cem_t[(size_t) blk * 6];
                    struct ggml_tensor * cx = ct[0];
                    struct ggml_tensor * cw = ct[1];
                    struct ggml_tensor * cg = ct[3];
                    struct ggml_tensor * cu = ct[4];
                    struct ggml_tensor * cd = ct[5];
                    const int64_t n_used = ids_orig->ne[0];
                    bool ok = !golem_cem_bad[blk] && cx != nullptr && cw != nullptr && ct[2] != nullptr &&
                              cg != nullptr && cu != nullptr && cd != nullptr &&
                              n_used <= GOLEM_CPU_EXPERT_MAX && cg->ne[2] == n_expert &&
                              n_expert <= gctx->n_experts;
                    int64_t n_embd = 0;
                    int64_t n_ff = 0;
                    if (ok) {
                        n_embd = cg->ne[0];
                        n_ff = cg->ne[1];
                        for (int k = 0; k < 3; ++k) {
                            struct ggml_tensor * p = ct[3 + k];
                            GGML_ASSERT(p->nb[2] == (size_t) golem_tensor_expert_stride_bytes(p));
                            GGML_ASSERT(p->ne[0] == (k < 2 ? n_embd : n_ff) && p->ne[1] == (k < 2 ? n_ff : n_embd));
                        }
                        // Fail-safe block-size gate: the kernel's vec_dot
                        // types (ggml-cpu traits) are not visible here.
                        ok = n_embd % 256 == 0 && n_ff % 256 == 0 &&
                             cx->type == GGML_TYPE_F32 && ggml_nelements(cx) == n_embd && ggml_is_contiguous(cx) &&
                             cw->type == GGML_TYPE_F32 && cw->ne[0] == 1 && cw->ne[1] == n_used &&
                             ct[2]->type == GGML_TYPE_F32 && ggml_nelements(ct[2]) == n_embd;
                    }
                    if (ok) {
                        // Candidates: routed ids without a device slot.
                        const int32_t K_c = std::min<int32_t>(gctx->k_cfg.load(std::memory_order_relaxed),
                                                              (int32_t) n_expert);
                        golem_cem_cand.clear();
                        {
                            std::lock_guard<std::mutex> lk(gctx->dc_mutex);
                            auto it = gctx->layers.find(blk);
                            const golem::GolemLayerSlots * L = it != gctx->layers.end() &&
                                (int64_t) it->second.slot_of.size() == n_expert && it->second.K == K_c
                                ? &it->second : nullptr;
                            for (int64_t id3 = 0; id3 < n_expert; ++id3) {
                                if (ggml_bitset_get(used_ids.data(), id3) && (L == nullptr || L->slot_of[id3] == -1)) {
                                    golem_cem_cand.push_back((int32_t) id3);
                                }
                            }
                        }
                        if (!golem_cem_cand.empty()) {
                            if (gctx->cem_if == nullptr) {
                                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(
                                    ggml_backend_get_device(sched->backends[sched->n_backends - 1]));
                                gctx->cem_if = (const golem_cpu_expert_iface *)
                                    ggml_backend_reg_get_proc_address(reg, "golem_cpu_expert_iface");
                                if (gctx->cem_if == nullptr) {
                                    return golem_cem_fail();
                                }
                            }
                            if (gctx->cem_worker == nullptr) {
                                gctx->cem_worker = gctx->cem_if->create(gctx->cem_threads);
                                if (gctx->cem_worker == nullptr) {
                                    return golem_cem_fail();
                                }
                            }
                            if (!golem_cem_acc_ensure(gctx, sched, split_backend, golem_cem_n_blk, n_embd)) {
                                return golem_cem_fail();
                            }
                            const int32_t n_cand = (int32_t) golem_cem_cand.size();
                            golem_cem_took.resize(n_cand);
                            golem_cem_ptrs.resize((size_t) n_cand * 3);
                            int32_t took = 0;
                            {
                                golem_cb_guard cb;
                                took = gctx->cem_lease(gctx->payload, blk, golem_cem_cand.data(), n_cand,
                                                       gctx->cem_frac, golem_cem_took.data(), golem_cem_ptrs.data());
                            }
                            GGML_ASSERT(took >= 0 && took <= n_cand);
                            if (took > 0) {
                                ggml_backend_t ids_be = ggml_backend_sched_get_tensor_backend(sched, ids_orig);
                                ggml_backend_t xb = ggml_backend_sched_get_tensor_backend(sched, cx);
                                ggml_backend_t wb = ggml_backend_sched_get_tensor_backend(sched, cw);
                                if (xb != NULL && xb != ids_be) {
                                    ggml_backend_synchronize(xb);
                                }
                                if (wb != NULL && wb != ids_be && wb != xb) {
                                    ggml_backend_synchronize(wb);
                                }
                                golem_cem_x.resize(n_embd);
                                ggml_backend_tensor_get(cx, golem_cem_x.data(), 0, ggml_nbytes(cx));
                                golem_cem_wraw.resize(ggml_nbytes(cw) / sizeof(float));
                                ggml_backend_tensor_get(cw, golem_cem_wraw.data(), 0, ggml_nbytes(cw));
                                golem_cem_w.resize(took);
                                ggml_bitset_t * row = &golem_cem_mask[(size_t) blk * golem_cem_row];
                                std::fill(row, row + golem_cem_row, 0);
                                golem_cpu_expert_job job = {};
                                job.x = golem_cem_x.data();
                                job.n_embd = n_embd;
                                job.n_ff = n_ff;
                                job.n = took;
                                golem_cpu_expert_plane * planes[3] = {&job.gate, &job.up, &job.down};
                                for (int k = 0; k < 3; ++k) {
                                    planes[k]->type = ct[3 + k]->type;
                                    planes[k]->row_bytes = ct[3 + k]->nb[1];
                                }
                                for (int32_t e = 0; e < took; ++e) {
                                    const int32_t id = golem_cem_took[e];
                                    GGML_ASSERT(id >= 0 && id < n_expert && ggml_bitset_get(used_ids.data(), id));
                                    int64_t pos = -1;
                                    for (int64_t i0 = 0; i0 < n_used; ++i0) {
                                        if (ids[i0 * ids_orig->nb[0] / sizeof(int32_t)] == id) {
                                            pos = i0;
                                            break;
                                        }
                                    }
                                    GGML_ASSERT(pos >= 0);
                                    golem_cem_w[e] = golem_cem_wraw[pos * cw->nb[1] / sizeof(float)];
                                    for (int k = 0; k < 3; ++k) {
                                        GGML_ASSERT(golem_cem_ptrs[(size_t) e * 3 + k] != nullptr);
                                        planes[k]->data[e] = golem_cem_ptrs[(size_t) e * 3 + k];
                                    }
                                    ggml_bitset_set(row, id);
                                }
                                job.w = golem_cem_w.data();
                                job.limit = golem_cem_limit[blk];
                                job.acc = gctx->cem_acc + (size_t) blk * (size_t) n_embd;
                                GGML_ASSERT(gctx->cem_pending_blk == -1);
                                gctx->cem_if->submit(gctx->cem_worker, &job);
                                gctx->cem_pending_blk = blk;
                                golem_cem_n_cpu[blk] = took;
                                gctx->cem_experts += (uint64_t) took;
                                gctx->cem_bytes += (uint64_t) took * (cg->nb[2] + cu->nb[2] + cd->nb[2]);
                            }
                        }
                    }
                }
                const ggml_bitset_t * golem_cem_blk_mask =
                    golem_cem_on && blk >= 0 && blk < golem_cem_n_blk && golem_cem_sel_step[blk] == golem_split_seq &&
                    golem_cem_n_cpu[blk] > 0 ? &golem_cem_mask[(size_t) blk * golem_cem_row] : nullptr;
                golem_miss_kinds.clear();
                golem_miss_ids.clear();
                golem_evicts.clear();
                bool golem_slab_failed = false;
                for (int j = input_id; j < split->n_inputs; ++j) {
                    struct ggml_tensor * inp = split->inputs[j];
                    const int kind = golem_expert_stem_kind(inp->name);
                    if (kind == 0 || golem_tensor_blk(inp->name) != blk) {
                        continue;
                    }
                    struct ggml_tensor * cpy2 = tensor_copy(inp, split_backend_id, sched->cur_copy);
                    struct ggml_tensor * node2 = nullptr;
                    for (int ni = 0; ni < split->graph.n_nodes; ++ni) {
                        struct ggml_tensor * n3 = split->graph.nodes[ni];
                        if (n3->op == GGML_OP_MUL_MAT_ID && n3->src[0] == cpy2) {
                            node2 = n3;
                            break;
                        }
                    }
                    if (!(node2 != nullptr &&
                          ggml_backend_buffer_get_usage(inp->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                          ggml_backend_buffer_is_host(inp->buffer))) {
                        continue;
                    }
                    const int64_t n_expert_j = inp->ne[2];
                    if (ggml_nelements(node2->src[2]) == 0) {
                        continue;
                    }
                    // planes share the layer's ids tensor: already read above
                    // (the first plane of the blk performed the orig read)
                    golem_plane_plan & plan = golem_planes[j];
                    if (stream_mode) {
                        // Prefill stream: no ensure misses (never pages into
                        // the pool -- the decode hot set must survive a long
                        // prompt, section 7.6.4), no slab step (P2 slot-cache slabs
                        // untouched); the copy loop calls stream_source.
                        plan.mode = 3;
                        plan.slab = nullptr;
                        plan.st = nullptr;
                        plan.dc = golem::GolemSlabCounters{};
                        plan.routed.clear();
                        plan.pending.clear();
                        plan.ptrs.clear();
                        plan.out_off = 0;
                        for (int64_t id3 = 0; id3 < n_expert_j; ++id3) {
                            if (ggml_bitset_get(used_ids.data(), id3)) {
                                plan.routed.push_back((int32_t) id3);
                            }
                        }
                        continue;
                    }
                    plan.mode = slab_mode ? 2 : 1;
                    plan.slab = nullptr;
                    plan.st = nullptr;
                    plan.dc = golem::GolemSlabCounters{};
                    plan.routed.clear();
                    plan.pending.clear();
                    plan.ptrs.clear();
                    plan.out_off = golem_miss_ids.size();
                    for (int64_t id3 = 0; id3 < n_expert_j; ++id3) {
                        if (ggml_bitset_get(used_ids.data(), id3) &&
                            (golem_cem_blk_mask == nullptr || !ggml_bitset_get(golem_cem_blk_mask, id3))) {
                            plan.routed.push_back((int32_t) id3);
                        }
                    }
                    if (slab_mode) {
                        // Slot-indirected device cache (design §3.4): the
                        // persistent slab holds K+1 expert planes; MUL_MAT_ID
                        // reads it through a remapped ids tensor (values in
                        // [0, K)) — routing untouched (I1), addressing moved.
                        // State machine shared verbatim with
                        // tests/golem-p2-cache.cpp (ODR).
                        const int32_t K_cfg = gctx->k_cfg.load(std::memory_order_relaxed);
                        if (K_cfg <= 0) {
                            GGML_ABORT("golem P2: device cache budget unset (K=%d): the slab must be "
                                       "bounded by golem_set_device_cache_experts, never by the expert count",
                                       (int) K_cfg);
                        }
                        const int32_t K = std::min<int32_t>(K_cfg, (int32_t) n_expert_j);
                        const int32_t *order = golem_policy_order(gctx, inp, (int32_t) n_expert_j);
                        if (order == NULL) {
                            // NULL before the first push: identity order + a
                            // one-time warning (order may lag one step; C
                            // never blocks on the publisher).
                            if (!gctx->order_warned.exchange(true)) {
                                GGML_LOG_WARN("%s: golem eviction order not yet published (blk %d kind %d); "
                                              "identity order in effect\n", __func__, blk, kind);
                            }
                            golem_identity.resize(n_expert_j);
                            for (int32_t i4 = 0; i4 < (int32_t) n_expert_j; ++i4) {
                                golem_identity[i4] = i4;
                            }
                            order = golem_identity.data();
                        }
                        struct ggml_tensor * slab = nullptr;
                        golem::GolemSlabEntry * pst = nullptr;
                        {
                            std::lock_guard<std::mutex> lk(gctx->dc_mutex);
                            golem::GolemSlabEntry & st = gctx->dc[inp->data];
                            // §3.4: the three planes of a layer consume the same routed
                            // ids, so slot assignment + eviction run once per layer; the
                            // table lives past any single slab's (re)allocation.
                            golem::GolemLayerSlots & L = gctx->layers[blk];
                            // (Re)allocate the slab + remapped-ids tensor on
                            // fingerprint change. The slab is zeroed once (the
                            // padding slot must stay non-NaN); its plane bytes
                            // are written only by the copies below.
                            const size_t expert_size_j = inp->nb[2];
                            if (st.slab == nullptr || st.K != K || st.expert_size != expert_size_j) {
                                if (st.buf != nullptr) {
                                    ggml_backend_buffer_free(st.buf);
                                }
                                if (st.ctx != nullptr) {
                                    ggml_free(st.ctx);
                                }
                                struct ggml_init_params ip = { 512, nullptr, /* no_alloc */ true };
                                st.ctx = ggml_init(ip);
                                GGML_ASSERT(st.ctx != nullptr);
                                st.slab = ggml_new_tensor_3d(st.ctx, inp->type,
                                    inp->ne[0], inp->ne[1], (int64_t) K + 1);
                                GGML_ASSERT(st.slab != nullptr);
                                // ABI v12: a slab that no longer fits (VRAM taken
                                // after load) fails the compute, never the process;
                                // the engine resets the cache and lowers K.
#ifdef GOLEM_TEST_SUPPORT
                                const bool inject = gctx->slab_fail_inject.load(std::memory_order_relaxed) > 0 &&
                                                    gctx->slab_fail_inject.fetch_sub(1, std::memory_order_relaxed) > 0;
                                st.buf = inject ? nullptr : ggml_backend_alloc_ctx_tensors(st.ctx, split_backend);
#else
                                st.buf = ggml_backend_alloc_ctx_tensors(st.ctx, split_backend);
#endif
                                if (st.buf == nullptr) {
                                    ggml_free(st.ctx);
                                    st.ctx = nullptr;
                                    st.slab = nullptr;
                                    st.K = 0;
                                    golem::golem_layer_invalidate(L, kind - 1);
                                    gctx->slab_alloc_failures.fetch_add(1, std::memory_order_relaxed);
                                    golem_slab_failed = true;
                                } else {
                                    GGML_ASSERT(st.slab->nb[2] == expert_size_j);
                                    ggml_backend_buffer_clear(st.buf, 0);
                                    st.K = K;
                                    st.expert_size = expert_size_j;
                                    // The slab's device bytes are gone: this plane must
                                    // re-copy everything on its next step.
                                    golem::golem_layer_invalidate(L, kind - 1);
                                }
                            }
                            if (golem_slab_failed) {
                                break;
                            }
                            // Remapped-ids device tensor: resized on ids-shape
                            // change (slot validity is independent of it).
                            if (st.ids_dev == nullptr || st.ids_ne0 != (int64_t) node2->src[2]->ne[0] ||
                                st.ids_ne1 != (int64_t) node2->src[2]->ne[1]) {
                                // A prior step's H2D may still be reading the old
                                // ids_pin (and the old ids_dev may be queued behind
                                // graph work); drain before freeing either buffer.
                                // One sync per ids-shape change, not per step.
                                ggml_backend_synchronize(split_backend);
                                if (st.ids_buf != nullptr) {
                                    ggml_backend_buffer_free(st.ids_buf);
                                }
                                if (st.ids_ctx != nullptr) {
                                    ggml_free(st.ids_ctx);
                                }
                                struct ggml_init_params ip = { 512, nullptr, /* no_alloc */ true };
                                st.ids_ctx = ggml_init(ip);
                                GGML_ASSERT(st.ids_ctx != nullptr);
                                st.ids_dev = ggml_new_tensor_2d(st.ids_ctx, GGML_TYPE_I32,
                                    node2->src[2]->ne[0], node2->src[2]->ne[1]);
                                GGML_ASSERT(st.ids_dev != nullptr);
                                st.ids_buf = ggml_backend_alloc_ctx_tensors(st.ids_ctx, split_backend);
                                if (st.ids_buf == nullptr) {
                                    // Same ABI v12 path as the slab: fail the compute.
                                    ggml_free(st.ids_ctx);
                                    st.ids_ctx = nullptr;
                                    st.ids_dev = nullptr;
                                    st.ids_ne0 = st.ids_ne1 = 0;
                                    gctx->slab_alloc_failures.fetch_add(1, std::memory_order_relaxed);
                                    golem_slab_failed = true;
                                    break;
                                }
                                st.ids_ne0 = node2->src[2]->ne[0];
                                st.ids_ne1 = node2->src[2]->ne[1];
                                // Pinned staging for the H2D below: (re)allocated
                                // under this SAME ids-shape-change condition as
                                // ids_dev, so its registration never outlives the
                                // device it was pinned against. Pinnedness is
                                // load-bearing: assert it so a regression to
                                // pageable memory cannot pass silently.
                                golem_ids_pin_alloc(st, split_backend, node2->src[2]->ne[0], node2->src[2]->ne[1]);
                            }
                            // Slab step: decide skips/copies/evictions + remap.
                            plan.dc = golem::golem_slab_step_ev(L, kind - 1, plan.routed, golem_dense.data(),
                                (int64_t) golem_dense.size(), order, K, (int32_t) n_expert_j, st.remap,
                                gctx->evict_batch != NULL ? &golem_evicts : nullptr,
                                [&](int32_t slot, int32_t id5) { plan.pending.emplace_back(slot, id5); },
                                golem_cem_blk_mask);
                            gctx->cem_padded += plan.dc.padded;
                            GGML_ASSERT((int64_t) st.remap.size() == ggml_nelements(st.ids_dev));
                            GGML_ASSERT(ggml_backend_buffer_is_host(st.ids_pin_buf));
                            GGML_ASSERT(ggml_backend_buffer_get_type(st.ids_pin_buf) == st.ids_pin_buft);
                            // Same-step reuse: if this entry already fed a split in
                            // THIS compute_splits call, its ids_pin may still have an
                            // H2D DMA in flight; the entry-time drain (copy_done) only
                            // covers cross-call reuse. Not reachable with llama.cpp MoE
                            // graphs (each expert weights tensor feeds exactly one
                            // MUL_MAT_ID, and only that split's nodes[0] takes this
                            // branch), but the guard costs nothing off that path.
                            if (st.ids_pin_step == golem_split_seq) {
                                ggml_backend_synchronize(split_backend);
                            }
                            st.ids_pin_step = golem_split_seq;
                            slab = st.slab;
                            pst = &st;
                        }
                        plan.slab = slab;
                        plan.st = pst;
                        for (const auto & pr : plan.pending) {
                            golem_miss_kinds.push_back(kind);
                            golem_miss_ids.push_back(pr.second);
                        }
                    } else {
                        for (int32_t id3 : plan.routed) {
                            golem_miss_kinds.push_back(kind);
                            golem_miss_ids.push_back(id3);
                        }
                    }
                }
                if (golem_slab_failed) {
                    // Nothing of this layer was issued yet; drain what
                    // earlier layers queued, then fail the compute
                    // (llama_decode -2). The slot tables already stepped
                    // are stale until the engine resets the cache.
                    golem_cem_join(gctx);
                    golem_restore_ids();
                    for (int i = 0; i < sched->n_backends; i++) {
                        ggml_backend_synchronize(sched->backends[i]);
                    }
                    golem_drain_copy_stream(gctx);
                    if (!golem_evict_d2h.empty()) {
                        golem_evict_abandon_unfinished();
                    }
                    golem_in_flight_done(gctx);
                    return GGML_STATUS_ALLOC_FAILED;
                }
                // Exclusive residency (O10): hand the layer's fully-valid
                // evictions to the pool in ONE batched call — it returns D2H
                // destinations per kept expert (copied here, before any H2D
                // re-fill of the stolen slots; both run on the split
                // backend's stream — the SAME stream as the H2D refills, not
                // a separate promotion stream — and the D2H is issued first,
                // so stream order lands it before the refill; ≈ 15.2 MB /
                // 25 GB/s ≈ 0.6 ms per evicted expert, measured decode
                // impact pending A/B) or scores the bytes droppable. Runs
                // only when the evict_batch hook is registered; NULL-hooked
                // runs keep inclusive residency verbatim. Post-drain, so the
                // previous compute has finished reading these slots.
                if (gctx->evict_batch != NULL && !golem_evicts.empty()) {
                    const size_t n_ev = golem_evicts.size();
                    golem_evict_ids.resize(n_ev);
                    golem_evict_out.assign(3 * n_ev, NULL);
                    golem_evict_keep.assign(n_ev, 0);
                    for (size_t i_ev = 0; i_ev < n_ev; ++i_ev) {
                        golem_evict_ids[i_ev] = golem_evicts[i_ev].second;
                    }
                    {
                        golem_cb_guard cb;
                        gctx->evict_batch(gctx->payload, input, golem_evict_ids.data(),
                                          (int32_t) n_ev, golem_evict_out.data(),
                                          golem_evict_keep.data());
                    }
                    for (size_t i_ev = 0; i_ev < n_ev; ++i_ev) {
                        if (golem_evict_keep[i_ev] == 0) {
                            continue;
                        }
                        // Park the D2H: each plane's copy is issued by the
                        // segment that owns that plane's slab entry (see
                        // golem_evict_d2h) — issuing here would only cover
                        // the planes that happen to live in THIS split.
                        golem_evict_d2h_entry e;
                        e.blk = blk;
                        e.id = golem_evicts[i_ev].second;
                        e.slot = golem_evicts[i_ev].first;
                        for (int k = 0; k < 3; ++k) {
                            GGML_ASSERT(golem_evict_out[3 * i_ev + k] != NULL);
                            e.dst[k] = golem_evict_out[3 * i_ev + k];
                        }
                        e.done = 0;
                        golem_evict_d2h.push_back(e);
                    }
                }
                // ONE ensure call per (layer, step, segment): all planes'
                // (kind, id) misses; out_ptrs distributed back to the planes.
                if (!golem_miss_ids.empty()) {
                    golem_miss_out.resize(golem_miss_ids.size());
                    std::fill(golem_miss_out.begin(), golem_miss_out.end(), nullptr);
                    int32_t ensure_rc = 0;
                    {
                        golem_cb_guard cb;
                        ensure_rc = gctx->ensure(gctx->payload, input, golem_miss_kinds.data(), golem_miss_ids.data(),
                                                 (int32_t) golem_miss_ids.size(), golem_miss_out.data());
                    }
                    if (ensure_rc != 0) {
                        // ABI v13: the misses could not be paged in; drain
                        // what earlier layers queued and fail the compute.
                        golem_cem_join(gctx);
                        golem_restore_ids();
                        for (int i = 0; i < sched->n_backends; i++) {
                            ggml_backend_synchronize(sched->backends[i]);
                        }
                        golem_drain_copy_stream(gctx);
                        if (!golem_evict_d2h.empty()) {
                            golem_evict_abandon_unfinished();
                        }
                        golem_in_flight_done(gctx);
                        return GGML_STATUS_FAILED;
                    }
                    for (size_t i5 = 0; i5 < golem_miss_out.size(); ++i5) {
                        GGML_ASSERT(golem_miss_out[i5] != NULL);
                    }
                    for (int j = input_id; j < split->n_inputs; ++j) {
                        golem_plane_plan & plan = golem_planes[j];
                        if (plan.mode == 0) {
                            continue;
                        }
                        const size_t n_plane = plan.mode == 1 ? plan.routed.size() : plan.pending.size();
                        for (size_t k3 = 0; k3 < n_plane; ++k3) {
                            plan.ptrs.push_back(golem_miss_out[plan.out_off + k3]);
                        }
                    }
                }
            }
        }


        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                // O1 join: this blk's own job, or zeros when none ran;
                // a different blk pending means its padded experts would be
                // lost, so the compute fails.
                const int ablk = golem_cem_on ? golem_cem_name_blk(input->name, "golem_cpu_acc-") : -1;
                if (ablk >= 0 && ablk < golem_cem_n_blk && input->op == GGML_OP_NONE) {
                    if (gctx->cem_pending_blk == ablk) {
                        golem_cem_join(gctx);
                    } else if (gctx->cem_pending_blk >= 0) {
                        return golem_cem_fail();
                    } else {
                        if (!golem_cem_acc_ensure(gctx, sched, split_backend, golem_cem_n_blk, ggml_nelements(input))) {
                            return golem_cem_fail();
                        }
                        memset(gctx->cem_acc + (size_t) ablk * (size_t) gctx->cem_acc_n_embd, 0, ggml_nbytes(input));
                    }
                    GGML_ASSERT(gctx->cem_acc != nullptr && gctx->cem_acc_n_embd == ggml_nelements(input));
                    ggml_backend_tensor_set_async(split_backend, input_cpy,
                                                  gctx->cem_acc + (size_t) ablk * (size_t) gctx->cem_acc_n_embd,
                                                  0, ggml_nbytes(input));
                    continue;
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                // Golem: upstream only inspects nodes[0], which stops being the
                // MUL_MAT_ID once other ops share the split (prefill ubatches
                // > 32); hooked scheds find the plane's MUL_MAT_ID like the
                // pre-pass does, hookless ones keep upstream verbatim.
                ggml_tensor * node = split->graph.n_nodes > 0 ? split->graph.nodes[0] : nullptr;
                if (golem_hooked && gctx->ensure != NULL && golem_expert_stem_kind(input->name) != 0) {
                    node = nullptr;
                    for (int ni = 0; ni < split->graph.n_nodes; ++ni) {
                        ggml_tensor * n2 = split->graph.nodes[ni];
                        if (n2->op == GGML_OP_MUL_MAT_ID && n2->src[0] == input_cpy) {
                            node = n2;
                            break;
                        }
                    }
                }
                if (node != nullptr &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // Same predicate as the split-time cpy shrink: only golem
                    // expert planes take the pooled/slab paths.
                    const bool golem_expert = golem_expert_stem_kind(input->name) != 0;
                    const golem_plane_plan & plan = golem_planes[input_id];
                    // plan.mode is only meaningful when this sched's ctx is
                    // hooked AND ran the pre-pass for this split (the scratch
                    // is thread-local and shared across scheds)
                    const bool golem_planned = golem_expert && golem_hooked && gctx->ensure != NULL && plan.mode != 0;

                    // group consecutive experts and copy them together;
                    // `ptrs` != NULL selects the golem pooled path (the
                    // pointers came from the batched ensure call)
                    auto copy_experts = [&](int32_t first_id, int32_t last_id, const void * const * ptrs) {
                        struct ggml_tensor * dst = input_cpy;
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_len = last_id < n_expert - 1 ? padding : 0;

                        if (ptrs != NULL) {
                            // Golem: the RAM pool is the H2D source — the
                            // loader's expert bytes may be a read-only file
                            // mmap. Expert slots are independent allocations,
                            // so copy per expert; the 512 B non-NaN MMQ tail
                            // after the last expert is a separate copy of that
                            // expert's own first bytes (content never read by
                            // the kernels; upstream used the next expert's
                            // bytes). A slot is exactly expert_size whenever
                            // expert_size is the pool's max stride, so reading
                            // past it could leave the arena mapping entirely.
                            if (gctx->async_h2d) {
                                // golem_cs_begin ran once for the plane (mode 1).
                                golem_copy_pooled_expert_run(dst, expert_offset, expert_size, padding_len,
                                    ptrs, last_id - first_id + 1,
                                    [](void *dst, const void *src, size_t dst_off, size_t len, void *ud) {
                                        golem_cs_set((golem_ctx_t) ud, (ggml_tensor *) dst, src, dst_off, len);
                                    }, gctx);
                                return;
                            }
                            golem_copy_pooled_expert_run(dst, expert_offset, expert_size, padding_len,
                                ptrs, last_id - first_id + 1,
                                [](void *dst, const void *src, size_t dst_off, size_t len, void *ud) {
                                    ggml_backend_tensor_set_async((ggml_backend_t) ud, (ggml_tensor *) dst, src, dst_off, len);
                                }, split_backend);
                            return;
                        }

                        ggml_backend_tensor_set_async(split_backend,
                            dst,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_len);
                    };

                    if (!golem_planned && golem_expert && golem_hooked && gctx->ensure != NULL) {
                        // hooked expert plane the pre-pass left unplanned
                        // (no routed ids): nothing to copy
                    } else if (golem_planned && plan.mode == 3) {
                        // Prefill stream (design section 7.6): fill this plane's
                        // range of the double-buffered stream slab from the
                        // stream_source pointers (pooled slots under lease or
                        // pinned staging slots the reader thread filled),
                        // then redirect the MUL_MAT_ID's weights to the slab.
                        // Identity ids: src[2] stays untouched -- no remap.
                        const int32_t kind_s = golem_expert_stem_kind(input->name);
                        const int32_t blk_s = golem_tensor_blk(input->name);
                        const uint64_t seq = gctx->stream_seq.fetch_add(1, std::memory_order_relaxed);
                        golem_ctx::golem_stream_slab & sl = gctx->stream_slabs[seq & 1];
                        // Slab reuse: plane seq-2's compute read the other
                        // half -- its event (or a full drain) proves the H2D +
                        // graph reads finished before the new H2D overwrites
                        // the bytes. Release order: sync first, then release
                        // the engine's leases/staging slot for seq-2.
                        if (sl.used) {
                            if (sl.event != nullptr) {
                                ggml_backend_event_synchronize(sl.event);
                            } else {
                                ggml_backend_synchronize(split_backend);
                            }
                            if (seq >= 2 && gctx->stream_release != NULL) {
                                golem_cb_guard cb;
                                gctx->stream_release(gctx->payload, seq - 2);
                            }
                        }
                        // Typed slab view, (re)built on shape/type change;
                        // bytes zeroed once at allocation -- unfilled ranges
                        // stay finite (MMQ padding guard, section 7.6.6).
                        const uint64_t fp = ((uint64_t) input->type << 48) ^
                            ((uint64_t) (uint32_t) input->ne[0] << 24) ^
                            (uint64_t) (uint32_t) input->ne[1];
                        GGML_ASSERT((int64_t) input->nb[2] <= gctx->stream_max_stride);
                        GGML_ASSERT(input->ne[2] <= gctx->stream_n_experts);
                        if (sl.buf == nullptr) {
                            // One fixed n_experts x max-stride buffer per half
                            // (design 7.6.1); only the typed view changes. One
                            // stride of slack covers the backend's quantized
                            // row padding (< one row), e.g. ne0 = 2304 on HIP.
                            sl.buf = ggml_backend_buft_alloc_buffer(
                                ggml_backend_get_default_buffer_type(split_backend),
                                ((size_t) gctx->stream_n_experts + 1) * (size_t) gctx->stream_max_stride);
                            GGML_ASSERT(sl.buf != nullptr);
                            ggml_backend_buffer_clear(sl.buf, 0);
                            sl.fp = 0;
                            sl.used = false;
                        }
                        if (sl.fp != fp) {
                            if (sl.ctx != nullptr) {
                                ggml_free(sl.ctx);
                            }
                            struct ggml_init_params ip = { 512, nullptr, /* no_alloc */ true };
                            sl.ctx = ggml_init(ip);
                            GGML_ASSERT(sl.ctx != nullptr);
                            sl.tensor = ggml_new_tensor_3d(sl.ctx, input->type,
                                input->ne[0], input->ne[1], gctx->stream_n_experts);
                            GGML_ASSERT(sl.tensor != nullptr);
                            GGML_ASSERT(ggml_backend_tensor_alloc(sl.buf, sl.tensor,
                                ggml_backend_buffer_get_base(sl.buf)) == GGML_STATUS_SUCCESS);
                            sl.fp = fp;
                        }
                        if (sl.event == nullptr) {
                            sl.event = ggml_backend_event_new(split_backend->device);
                        }
                        // Routed-only below the share switch; whole above.
                        const float share = (float) plan.routed.size() / (float) n_expert;
                        const bool whole = share > gctx->stream_routed_max;
                        golem_stream_ids.clear();
                        if (whole) {
                            golem_stream_ids.resize(n_expert);
                            for (int64_t id6 = 0; id6 < n_expert; ++id6) {
                                golem_stream_ids[id6] = (int32_t) id6;
                            }
                        } else {
                            golem_stream_ids = plan.routed; // sorted copy
                        }
                        golem_stream_ptrs.assign(golem_stream_ids.size(), nullptr);
                        int32_t src_rc;
                        {
                            golem_cb_guard cb;
                            src_rc = gctx->stream_source(gctx->payload, blk_s, kind_s, seq,
                                                golem_stream_ids.data(),
                                                (int32_t) golem_stream_ids.size(),
                                                golem_stream_ptrs.data());
                        }
                        if (src_rc != GOLEM_STREAM_SRC_OK) {
                            // ABI v11 abort: drain this split's issued H2D,
                            // hand back every issued seq (leases + staging),
                            // and fail the compute; the graph never runs.
                            // Only a cancel aborts; storage loss fails.
                            golem_cem_join(gctx);
                            golem_restore_ids();
                            // Leases are released below: drain all backends + copy stream (any may hold an earlier split's async copy).
                            for (int i = 0; i < sched->n_backends; i++) {
                                ggml_backend_synchronize(sched->backends[i]);
                            }
                            golem_drain_copy_stream(gctx);
                            if (gctx->stream_release != NULL) {
                                golem_cb_guard cb;
                                gctx->stream_release(gctx->payload, seq);
                            }
                            gctx->stream_used_in_split[0] = false;
                            gctx->stream_used_in_split[1] = false;
                            if (!golem_evict_d2h.empty()) {
                                golem_evict_abandon_unfinished();
                            }
                            golem_in_flight_done(gctx);
                            return src_rc == GOLEM_STREAM_SRC_CANCEL ? GGML_STATUS_ABORTED
                                                                     : GGML_STATUS_FAILED;
                        }
                        for (size_t i6 = 0; i6 < golem_stream_ptrs.size(); ++i6) {
                            GGML_ASSERT(golem_stream_ptrs[i6] != NULL);
                        }
                        for (int ni = 0; ni < split->graph.n_nodes; ++ni) {
                            struct ggml_tensor * n2 = split->graph.nodes[ni];
                            if (n2->op == GGML_OP_MUL_MAT_ID && n2->src[0] == input_cpy) {
                                golem_ids_redirects.emplace_back(&n2->src[0], n2->src[0]);
                                n2->src[0] = sl.tensor;
                            }
                        }
                        // Coalesced H2D: one set_async per run of consecutive
                        // ids whose host pointers are contiguous (staging
                        // runs); pooled slots are independent allocations and
                        // copy per expert -- no extra byte is copied.
                        {
                            size_t i7 = 0;
                            while (i7 < golem_stream_ids.size()) {
                                size_t j7 = i7;
                                while (j7 + 1 < golem_stream_ids.size() &&
                                       golem_stream_ids[j7 + 1] == golem_stream_ids[j7] + 1 &&
                                       (const uint8_t *) golem_stream_ptrs[j7 + 1] ==
                                           (const uint8_t *) golem_stream_ptrs[j7] + expert_size) {
                                    ++j7;
                                }
                                ggml_backend_tensor_set_async(split_backend, sl.tensor,
                                    golem_stream_ptrs[i7],
                                    (size_t) golem_stream_ids[i7] * expert_size,
                                    (j7 - i7 + 1) * expert_size);
                                i7 = j7 + 1;
                            }
                        }
                        gctx->dc_stats.stream_experts_copied.fetch_add(golem_stream_ids.size(), std::memory_order_relaxed);
                        sl.used = true;
                        gctx->stream_used_in_split[seq & 1] = true;
                    } else if (golem_planned && plan.mode == 1) {
                        // Pooled path (ensure only, no routing): the misses
                        // were ensured in the layer-level pre-pass; runs of
                        // consecutive routed experts map onto contiguous
                        // slices of the returned pointer array (plan.routed
                        // is sorted — no dependency on the mutable used_ids).
                        if (gctx->async_h2d && !plan.routed.empty()) {
                            golem_cs_begin(gctx, split_backend, !golem_cs_used, !golem_cs_fenced);
                            golem_cs_used = true;
                            golem_cs_fenced = true;
                        }
                        size_t i = 0;
                        while (i < plan.routed.size()) {
                            const size_t run0 = i;
                            int32_t first_id = plan.routed[i], last_id = first_id;
                            while (i + 1 < plan.routed.size() && plan.routed[i + 1] == last_id + 1) {
                                ++i;
                                last_id = plan.routed[i];
                            }
                            ++i;
                            copy_experts(first_id, last_id, &plan.ptrs[run0]);
                        }
                    } else if (golem_planned && plan.mode == 2) {
                        // Slab path: the decision pass ran in the pre-pass;
                        // here only the physical writes: remapped-ids H2D
                        // from PINNED host memory, consumer redirects, and
                        // one async copy per staged slot from the batched
                        // ensure pointers (no extra copy).
                        golem::GolemSlabEntry & st = *plan.st;
                        // Exclusive D2H drain (see golem_evict_d2h): issue
                        // this plane's share of the layer's pending evictions
                        // BEFORE the staged H2D refills below — stream order
                        // must land the D2H read of a stolen slot ahead of
                        // the refill that overwrites it.
                        {
                            const int blk_d = golem_tensor_blk(input->name);
                            const int kind_d = golem_expert_stem_kind(input->name) - 1;
                            if (blk_d >= 0 && kind_d >= 0 && kind_d < 3) {
                                const uint32_t bit = 1u << kind_d;
                                size_t w = 0;
                                for (size_t r = 0; r < golem_evict_d2h.size(); ++r) {
                                    golem_evict_d2h_entry & e = golem_evict_d2h[r];
                                    if (e.blk != blk_d) {
                                        golem_evict_d2h[w++] = e;
                                        continue;
                                    }
                                    if (!(e.done & bit)) {
                                        ggml_backend_tensor_get_async(split_backend, st.slab,
                                            e.dst[kind_d], (size_t) e.slot * st.expert_size,
                                            st.expert_size);
                                        e.done |= bit;
                                        golem_cs_fenced = false;
                                    }
                                    if (e.done != 7u) {
                                        golem_evict_d2h[w++] = e;
                                    }
                                }
                                golem_evict_d2h.resize(w);
                            }
                        }
                        // SAFETY (ordering invariant): the step-(N+1) memcpy
                        // below into ids_pin must never overlap the step-N DMA
                        // reading it. This holds because compute_splits drains
                        // every backend at entry when copy_done is set (the
                        // engine registers it alongside routing — see
                        // p3_glue.rs), so step N's H2D is complete before this
                        // code runs again; a sync-granularity change must not
                        // silently tear the remap. Defensive drain for any
                        // registration path that sets routing without
                        // copy_done (no-op in the engine configuration).
                        if (gctx->copy_done == NULL) {
                            ggml_backend_synchronize(split_backend);
                        }
                        memcpy(st.ids_pin, st.remap.data(), st.remap.size() * sizeof(int32_t));
                        ggml_backend_tensor_set_async(split_backend, st.ids_dev, st.ids_pin,
                            0, st.remap.size() * sizeof(int32_t));
                        for (int ni = 0; ni < split->graph.n_nodes; ++ni) {
                            struct ggml_tensor * n2 = split->graph.nodes[ni];
                            if (n2->op == GGML_OP_MUL_MAT_ID && n2->src[0] == input_cpy) {
                                golem_ids_redirects.emplace_back(&n2->src[0], n2->src[0]);
                                n2->src[0] = plan.slab;
                                golem_ids_redirects.emplace_back(&n2->src[2], n2->src[2]);
                                n2->src[2] = st.ids_dev;
                            }
                        }
                        if (gctx->async_h2d && !plan.pending.empty()) {
                            // After this plane's D2H issue above: the copy
                            // stream waits on the compute stream, so the
                            // eviction read of a stolen slot lands first.
                            golem_cs_begin(gctx, split_backend, !golem_cs_used, !golem_cs_fenced);
                            golem_cs_used = true;
                            golem_cs_fenced = true;
                            for (size_t k4 = 0; k4 < plan.pending.size(); ++k4) {
                                golem_cs_set(gctx, plan.slab, plan.ptrs[k4],
                                    (size_t) plan.pending[k4].first * expert_size, expert_size);
                            }
                        } else {
                            for (size_t k4 = 0; k4 < plan.pending.size(); ++k4) {
                                ggml_backend_tensor_set_async(split_backend, plan.slab, plan.ptrs[k4],
                                    (size_t) plan.pending[k4].first * expert_size, expert_size);
                            }
                        }
                        gctx->dc_stats.bytes_copied.fetch_add(plan.dc.copied * expert_size, std::memory_order_relaxed);
                        gctx->dc_stats.cache_experts_copied.fetch_add(plan.dc.copied, std::memory_order_relaxed);
                        gctx->dc_stats.bytes_skipped.fetch_add(plan.dc.skipped * expert_size, std::memory_order_relaxed);
                        gctx->dc_stats.evictions.fetch_add(plan.dc.evicted, std::memory_order_relaxed);
                    } else {
                        // upstream loop, verbatim (unset hooks == byte-identical
                        // behavior); the hooks pre-pass already read the ids
                        // for planned expert layers, so only the lazy upstream
                        // paths get here — prev_ids_tensor caches.
                        if (golem_ensure_ids(input_id, node, n_expert) == nullptr) {
                            continue;
                        }
                        int id = 0;
                        while (!ggml_bitset_get(used_ids.data(), id)) {
                            id++;
                        }
                        int32_t first_id = id;
                        int32_t last_id = first_id;

                        for (++id; id < n_expert; ++id) {
                            if (!ggml_bitset_get(used_ids.data(), id)) {
                                continue;
                            }

                            if (id == last_id + 1) {
                                last_id = id;
                                continue;
                            }

                            copy_experts(first_id, last_id, NULL);

                            first_id = id;
                            last_id = id;
                        }
                        copy_experts(first_id, last_id, NULL);
                    }
                    // done with the expert weights input
                } else {
                    if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input) &&
                        sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
                        continue;
                    }
                    // never reached with a shrunk golem placeholder (the cpy is not full-size)
                    GGML_ASSERT(ggml_are_same_shape(input, input_cpy));
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (golem_cs_used) {
            golem_cs_join(gctx);
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            // Prefill stream: the compute above consumed this split's slabs --
            // record their events so the next fill of the same half waits
            // exactly for this work (not the whole backend).
            if (golem_hooked) {
                for (int b7 = 0; b7 < 2; ++b7) {
                    if (gctx->stream_used_in_split[b7]) {
                        gctx->stream_used_in_split[b7] = false;
                        if (gctx->stream_slabs[b7].event != nullptr) {
                            ggml_backend_event_record(gctx->stream_slabs[b7].event, split_backend);
                        }
                    }
                }
            }
            golem_restore_ids();
            if (ec != GGML_STATUS_SUCCESS) {
                if (golem_cem_on) {
                    golem_cem_join(gctx);
                }
                for (int i = 0; i < sched->n_backends; i++) {
                    ggml_backend_synchronize(sched->backends[i]);
                }
                golem_drain_copy_stream(gctx);
                if (!golem_evict_d2h.empty()) {
                    golem_evict_abandon_unfinished();
                }
                if (golem_hooked) {
                    golem_in_flight_done(gctx);
                }
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    if (golem_cem_on) {
                        golem_cem_join(gctx);
                    }
                    golem_restore_ids();
                    for (int i = 0; i < sched->n_backends; i++) {
                        ggml_backend_synchronize(sched->backends[i]);
                    }
                    golem_drain_copy_stream(gctx);
                    if (!golem_evict_d2h.empty()) {
                        golem_evict_abandon_unfinished();
                    }
                    if (golem_hooked) {
                        golem_in_flight_done(gctx);
                    }
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
            golem_restore_ids();
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    // ABI v9: report parked D2H entries whose plane copies were not all
    // issued (see golem_evict_d2h_entry) so the pool cancels their claims
    // instead of committing unwritten plane bytes. All segments have run,
    // so anything still parked here is undrainable; sync the backends first
    // so any issued copy has landed before the destinations are recycled.
    if (golem_hooked && !golem_evict_d2h.empty()) {
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }
        golem_drain_copy_stream(gctx);
        golem_evict_abandon_unfinished();
    }
    // Release this entry's in-flight pin (the graph_compute_async caller
    // holds its own; the depth is per nested entry). Copy-stream work never
    // outlives the entry that queued it.
    if (golem_hooked) {
        golem_cem_join(gctx);
        golem_drain_copy_stream(gctx);
        golem_in_flight_done(gctx);
    }
    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    golem_sched_detach_ctx(sched);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    // Golem: in-flight depth covers the whole entry (graph-build reads of ctx
    // fields included), so unregister can only join between engine steps.
    golem_ctx_t gctx = sched->golem_ctx;
    // In-flight pin incremented BEFORE the hooked re-check: an unregister that
    // races this entry either sees the pin (and joins) or this entry runs
    // unhooked — never a half-registered step.
    const bool golem_on = [&] {
        if (gctx == NULL) {
            return false;
        }
        gctx->epoch.fetch_add(1, std::memory_order_relaxed);
        gctx->in_flight.fetch_add(1, std::memory_order_acq_rel);
        if (!gctx->hooked.load(std::memory_order_acquire)) {
            golem_in_flight_done(gctx);
            return false;
        }
        return true;
    }();

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        if (golem_on) {
            golem_in_flight_done(gctx);
        }
        return false;
    }

    ggml_backend_sched_reset(sched);

    if (golem_on) {
        golem_in_flight_done(gctx);
    }
    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    golem_ctx_t gctx = sched->golem_ctx;
    // In-flight pin incremented BEFORE the hooked re-check: an unregister that
    // races this entry either sees the pin (and joins) or this entry runs
    // unhooked — never a half-registered step.
    const bool golem_on = [&] {
        if (gctx == NULL) {
            return false;
        }
        gctx->epoch.fetch_add(1, std::memory_order_relaxed);
        gctx->in_flight.fetch_add(1, std::memory_order_acq_rel);
        if (!gctx->hooked.load(std::memory_order_acquire)) {
            golem_in_flight_done(gctx);
            return false;
        }
        return true;
    }();

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        if (golem_on) {
            golem_in_flight_done(gctx);
        }
        return false;
    }

    sched->is_alloc = true;

    if (golem_on) {
        golem_in_flight_done(gctx);
    }
    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    golem::step_id.fetch_add(1, std::memory_order_relaxed);
    GGML_ASSERT(sched);
    golem_ctx_t gctx = sched->golem_ctx;
    // In-flight pin incremented BEFORE the hooked re-check: an unregister that
    // races this entry either sees the pin (and joins) or this entry runs
    // unhooked — never a half-registered step.
    const bool golem_on = [&] {
        if (gctx == NULL) {
            return false;
        }
        gctx->epoch.fetch_add(1, std::memory_order_relaxed);
        gctx->in_flight.fetch_add(1, std::memory_order_acq_rel);
        if (!gctx->hooked.load(std::memory_order_acquire)) {
            golem_in_flight_done(gctx);
            return false;
        }
        return true;
    }();
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            if (golem_on) {
                golem_in_flight_done(gctx);
            }
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    enum ggml_status err = ggml_backend_sched_compute_splits(sched);
    if (golem_on) {
        golem_in_flight_done(gctx);
    }
    return err;
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

void ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}

// --- golem P1: policy-hook plumbing (design §3.4, ABI v5: per-model
// golem_ctx). All behavior is behind null guards; with hooks unset every code
// path here is a pure query and the sched paths are byte-identical to
// upstream. ---

int golem_hooks_abi_version(void) { return GOLEM_HOOKS_ABI_VERSION; }

// Process-global capability flag: refcount over hooked ctxs. Consumed ONLY by
// golem_graphs_disabled(); per-sched decisions read the sched's own ctx.
static std::atomic<int> golem_hooked_ctxs{0};
int golem_any_ctx_hooked(void) { return golem_hooked_ctxs.load(std::memory_order_acquire) > 0; }

golem_ctx_t golem_ctx_create(int32_t n_layers, int32_t n_experts) {
    return new golem_ctx(n_layers, n_experts);
}

void golem_ctx_free(golem_ctx_t ctx) {
    if (ctx == NULL) {
        return;
    }
    if (ctx->attach_refs.load(std::memory_order_acquire) != 0) {
        GGML_ABORT("golem ctx_free with %d sched(s) still attached: detach (or "
                   "free the scheds) first — drop order is unregister-join, "
                   "llama_free, then ctx_free", ctx->attach_refs.load(std::memory_order_relaxed));
    }
    if (ctx->hooked.load(std::memory_order_relaxed)) {
        GGML_ABORT("golem ctx_free while hooks are registered: golem_unregister_hooks first");
    }
    if (!ctx->dc.empty()) {
        GGML_ABORT("golem ctx_free with %zu live slab entr(ies): the owner must "
                   "call golem_reset_device_cache while the backend is still "
                   "alive (before llama_free) — freeing device buffers after "
                   "the backend is gone is a use-after-free",
                   ctx->dc.size());
    }
    if (ctx->cs_backend != nullptr) {
        GGML_ABORT("golem ctx_free with a bound copy stream: the owner must "
                   "call golem_reset_device_cache while the backend is still alive");
    }
    if (ctx->stream_slabs[0].buf != nullptr || ctx->stream_slabs[1].buf != nullptr) {
        GGML_ABORT("golem ctx_free with live prefill-stream slabs: the owner must "
                   "call golem_reset_device_cache while the backend is still alive");
    }
    if (ctx->cem_acc_buf != nullptr) {
        GGML_ABORT("golem ctx_free with a live CPU expert accumulator: the owner must "
                   "call golem_reset_device_cache while the backend is still alive");
    }
    GGML_ASSERT(ctx->cem_pending_blk == -1);
    if (ctx->cem_worker != nullptr) {
        ctx->cem_if->destroy(ctx->cem_worker);
    }
    delete ctx;
}

struct golem_hook_handle { golem_ctx_t ctx; };


golem_hook_handle_t golem_register_hooks(golem_ctx_t ctx,
                                         golem_ensure_host_resident_fn ensure,
                                         golem_routing_fn routing,
                                         golem_copy_done_fn copy_done,
                                         golem_evict_batch_fn evict_batch,
                                         golem_evict_abandon_fn evict_abandon,
                                         golem_stream_source_fn stream_source,
                                         golem_stream_release_fn stream_release,
                                         void *payload) {
    if (ctx == NULL) {
        return NULL;
    }
    const bool was = ctx->hooked.exchange(true, std::memory_order_acq_rel);
    if (was) {
        GGML_ABORT("golem_register_hooks on an already-hooked ctx: unregister "
                   "the previous registration first (double registration would "
                   "leak the handle and silently swap live hooks)");
    }
    golem_hooked_ctxs.fetch_add(1, std::memory_order_acq_rel);
    ctx->stream_source = stream_source;
    ctx->stream_release = stream_release;
    ctx->ensure = ensure;
    ctx->routing = routing;
    ctx->copy_done = copy_done;
    ctx->evict_batch = evict_batch;
    ctx->evict_abandon = evict_abandon;
    ctx->payload = payload;
    ctx->generation.fetch_add(1, std::memory_order_relaxed);
    return new golem_hook_handle{ctx};
}

void golem_unregister_hooks(golem_hook_handle_t handle) {
    if (handle == NULL) {
        return;
    }
    golem_ctx_t ctx = handle->ctx;
    delete handle;
    if (golem_cb_depth > 0) {
        GGML_ABORT("golem_unregister_hooks from inside a hook callback: the "
                   "join would wait for the calling step (deadlock by construction)");
    }
    if (ctx == NULL || !ctx->hooked.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    golem_hooked_ctxs.fetch_sub(1, std::memory_order_acq_rel);
    // Join: no callback can touch the engine payload after this returns.
    std::unique_lock<std::mutex> lk(ctx->join_mutex);
    ctx->join_cv.wait(lk, [&] { return ctx->in_flight.load(std::memory_order_acquire) == 0; });
    ctx->ensure = NULL;
    ctx->routing = NULL;
    ctx->copy_done = NULL;
    ctx->evict_batch = NULL;
    ctx->evict_abandon = NULL;
    ctx->stream_source = NULL;
    ctx->stream_release = NULL;
    ctx->stream_on.store(0, std::memory_order_release);
    ctx->payload = NULL;
}

uint64_t golem_ctx_generation(golem_ctx_t ctx) {
    return ctx == NULL ? 0 : ctx->generation.load(std::memory_order_relaxed);
}

void golem_sched_set_ctx(struct ggml_backend_sched *sched, golem_ctx_t ctx) {
    if (sched == NULL) {
        return;
    }
    golem_ctx_t old = sched->golem_ctx;
    if (old == ctx) {
        return;
    }
    if (old != NULL) {
        old->attach_refs.fetch_sub(1, std::memory_order_acq_rel);
    }
    sched->golem_ctx = ctx;
    if (ctx != NULL) {
        ctx->attach_refs.fetch_add(1, std::memory_order_acq_rel);
    }
}

void golem_sched_detach_ctx(struct ggml_backend_sched *sched) {
    golem_sched_set_ctx(sched, NULL);
}

golem_ctx_t golem_sched_get_ctx(struct ggml_backend_sched *sched) {
    return sched == NULL ? NULL : sched->golem_ctx;
}

const int32_t *golem_policy_order(golem_ctx_t ctx, const void *tensor, int32_t n_experts) {
    if (ctx == NULL || tensor == NULL) {
        return NULL;
    }
    const struct ggml_tensor *t = (const struct ggml_tensor *) tensor;
    const int kind = golem_expert_stem_kind(t->name);
    if (kind == 0) {
        return NULL;
    }
    return ctx->acquire_order(golem_tensor_blk(t->name), kind);
}

int golem_policy_order_push(golem_ctx_t ctx, int32_t blk, int32_t kind,
                            const int32_t *order, int32_t n) {
    if (ctx == NULL || order == NULL) {
        return 0;
    }
    return ctx->push_order(blk, kind, order, n) ? 1 : 0;
}

void golem_policy_order_epoch_advance(golem_ctx_t ctx) {
    if (ctx != NULL) {
        ctx->epoch.fetch_add(1, std::memory_order_relaxed);
    }
}

uint64_t golem_slab_alloc_failures(golem_ctx_t ctx) {
    return ctx == NULL ? 0 : ctx->slab_alloc_failures.load(std::memory_order_relaxed);
}

#ifdef GOLEM_TEST_SUPPORT
void golem_inject_slab_alloc_failure(golem_ctx_t ctx, int32_t n) {
    if (ctx != NULL) {
        ctx->slab_fail_inject.store(n, std::memory_order_relaxed);
    }
}
#endif

void golem_device_cache_resident_bytes(golem_ctx_t ctx, uint64_t * slab_bytes, uint64_t * stream_bytes) {
    uint64_t slab = 0;
    uint64_t stream = 0;
    if (ctx != NULL) {
        std::lock_guard<std::mutex> lk(ctx->dc_mutex);
        for (const auto & kv : ctx->dc) {
            if (kv.second.buf != nullptr) {
                slab += ggml_backend_buffer_get_size(kv.second.buf);
            }
            if (kv.second.ids_buf != nullptr) {
                slab += ggml_backend_buffer_get_size(kv.second.ids_buf);
            }
        }
        for (const auto & sl : ctx->stream_slabs) {
            if (sl.buf != nullptr) {
                stream += ggml_backend_buffer_get_size(sl.buf);
            }
        }
    }
    if (slab_bytes != NULL) {
        *slab_bytes = slab;
    }
    if (stream_bytes != NULL) {
        *stream_bytes = stream;
    }
}

void golem_set_device_cache_experts(golem_ctx_t ctx, int32_t k) {
    if (ctx != NULL) {
        ctx->k_cfg.store(k, std::memory_order_relaxed);
    }
}

int32_t golem_device_cache_experts(golem_ctx_t ctx) {
    return ctx == NULL ? 0 : ctx->k_cfg.load(std::memory_order_relaxed);
}

void golem_set_predict(golem_ctx_t ctx, int32_t lookahead, int32_t margin, golem_predict_fn predict) {
    if (ctx == NULL) {
        return;
    }
    const bool on = lookahead > 0 && predict != NULL;
    ctx->predict = on ? predict : NULL;
    ctx->pred_margin.store(on ? std::max<int32_t>(margin, 0) : 0, std::memory_order_relaxed);
    ctx->pred_lookahead.store(on ? lookahead : 0, std::memory_order_relaxed);
}

int32_t golem_predict_lookahead(golem_ctx_t ctx) {
    return ctx == NULL ? 0 : ctx->pred_lookahead.load(std::memory_order_relaxed);
}

int32_t golem_predict_margin(golem_ctx_t ctx) {
    return ctx == NULL ? 0 : ctx->pred_margin.load(std::memory_order_relaxed);
}

const struct golem_device_cache_stats *golem_get_device_cache_stats(golem_ctx_t ctx) {
    return ctx == NULL ? NULL : &ctx->dc_stats;
}

/* Golem (design I2): process-lifetime override for the device backends' op-offload
 * threshold; -1 = unset (env / 32 default). Device init reads it once. */
static int golem_op_offload_min_batch_override = -1;

void golem_set_expert_offload_min_batch(int n) {
    golem_op_offload_min_batch_override = n > 0 ? n : -1;
}

// Declared `extern "C"` in the CUDA/HIP TUs (ggml-cuda.cu); the header does
// not declare these internal helpers, so pin the linkage here.
extern "C" int golem_expert_offload_min_batch(int default_value);
extern "C" int golem_graphs_disabled(void);

int golem_graphs_disabled(void) {
    return golem_any_ctx_hooked();
}

int golem_expert_offload_min_batch(int default_value) {
    return golem_op_offload_min_batch_override > 0
        ? golem_op_offload_min_batch_override
        : default_value;
}

/* Expert weights in glm-arch MoE models are 3-D tensors whose names contain
 * the stem ffn_{gate,up,down}_exps as a whole name segment, optionally
 * followed by ".weight" (runtime ggml names: blk.%d.ffn_*_exps.weight;
 * gguf-side names drop the suffix). The same rule is implemented in Rust
 * (golem-gguf::ExpertKind::from_tensor_name) — one shared matcher semantics
 * (the shared expert-matcher rule): the loader can create `.scale` variants and dense
 * ffn_gate tensors whose names contain the stems too. One expert is one
 * ne[0]×ne[1] plane; nb[2] is the authoritative plane stride (also correct
 * for padded/non-contiguous tensors, where recomputing from ne is not).
 * Fused ffn_gate_up_exps is deliberately excluded — P2
 * splits routed gate/up/down ranges independently. */
/* Layer number of a runtime tensor name ("blk.%d...."); the layer key of the
 * shared per-layer device slot table. Aborts on a name without the blk.N.
 * segment — expert tensors always carry it (GGML_TOKEN/MTP layers never reach
 * this path). */
static int golem_tensor_blk(const char *name) {
    const char *p = strstr(name, "blk.");
    if (p == NULL) {
        GGML_ABORT("golem P2: expert tensor name without blk.N. segment: %s", name);
    }
    return atoi(p + 4);
}

static int golem_name_ends_with(const char *name, const char *suffix) {
    size_t nl = strlen(name), sl = strlen(suffix);
    return nl >= sl && strcmp(name + nl - sl, suffix) == 0;
}

static int golem_expert_stem_kind(const char *name) {
    const char *suffixes[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    char base[GGML_MAX_NAME];
    size_t nl = strnlen(name, GGML_MAX_NAME);
    if (nl >= sizeof(base)) return 0;
    memcpy(base, name, nl + 1);
    size_t bl = strlen(base);
    if (golem_name_ends_with(base, ".weight")) base[bl - 7] = '\0';
    for (int k = 0; k < 3; k++) {
        size_t sl = strlen(suffixes[k]);
        size_t rest = strlen(base);
        if (rest >= sl && strcmp(base + rest - sl, suffixes[k]) == 0 &&
            (rest == sl || base[rest - sl - 1] == '.')) return k + 1;
    }
    return 0;
}

static int golem_tensor_is_expert(const struct ggml_tensor *t) {
    if (t == NULL || ggml_n_dims(t) != 3) return 0;
    return golem_expert_stem_kind(t->name) != 0;
}

int32_t golem_tensor_expert_count(const void *tensor) {
    const struct ggml_tensor *t = (const struct ggml_tensor *) tensor;
    return golem_tensor_is_expert(t) ? (int32_t) t->ne[2] : 0;
}

int32_t golem_tensor_expert_stride_bytes(const void *tensor) {
    const struct ggml_tensor *t = (const struct ggml_tensor *) tensor;
    if (!golem_tensor_is_expert(t)) return 0;
    GGML_ASSERT(t->nb[2] <= (size_t) INT32_MAX);
    return (int32_t) t->nb[2];
}
