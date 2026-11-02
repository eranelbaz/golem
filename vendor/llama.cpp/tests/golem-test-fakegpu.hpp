// Golem vendor-test helper: a minimal fake "GPU" backend whose buffers are
// plain host memory but whose buffer type is NOT supported by the CPU backend
// (is_host == false). The scheduler therefore creates cross-backend copies of
// expert weights for splits assigned to it, which is the only way to drive the
// golem copy path (cpy shrink / pooled / slab) in a CPU-only test build.
// Compute is delegated to the ggml CPU kernels over the host-memory tensors.
#ifndef GOLEM_TEST_FAKEGPU_HPP
#define GOLEM_TEST_FAKEGPU_HPP

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h" // tests link ggml and include ggml/src
#include "golem-copy-stream.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>

namespace golem_test {

inline ggml_backend_dev_t fake_dev();
inline ggml_backend_t fake_backend();

inline void * fg_buf_base(ggml_backend_buffer_t buffer) { return ((void **) buffer->context)[0]; }
inline void fg_buf_free_buffer(ggml_backend_buffer_t buffer) {
    free(((void **) buffer->context)[0]);
    free(buffer->context);
}
inline void fg_buf_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor,
                              const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
}
inline void fg_buf_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor,
                              void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}
inline bool fg_buf_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor * src, ggml_tensor * dst) {
    memcpy(dst->data, src->data, ggml_nbytes(dst));
    return true;
}
inline void fg_buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(fg_buf_base(buffer), value, buffer->size);
}

inline ggml_backend_buffer_type_t fake_buft() {
    static ggml_backend_buffer_type buft = {};
    if (buft.iface.get_name == nullptr) {
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "FAKEGPU"; };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft_, size_t size) {
            void ** ctx = (void **) malloc(2 * sizeof(void *));
            void * mem = nullptr;
            if (posix_memalign(&mem, 64, size) != 0) mem = nullptr;
            ctx[0] = mem;
            ctx[1] = nullptr;
            ggml_backend_buffer_i buf_i = {};
            buf_i.free_buffer = fg_buf_free_buffer;
            buf_i.get_base = [](ggml_backend_buffer_t buffer) { return fg_buf_base(buffer); };
            buf_i.set_tensor = fg_buf_set_tensor;
            buf_i.get_tensor = fg_buf_get_tensor;
            buf_i.cpy_tensor = fg_buf_cpy_tensor;
            buf_i.clear = fg_buf_clear;
            return ggml_backend_buffer_init(buft_, buf_i, ctx, size);
        };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) { return (size_t) 64; };
        buft.iface.get_max_size = [](ggml_backend_buffer_type_t) { return (size_t) SIZE_MAX; };
        buft.iface.get_alloc_size = [](ggml_backend_buffer_type_t, const ggml_tensor * t) { return ggml_nbytes(t); };
        // NOT host: the CPU backend must not claim this buft, so splits on the
        // fake backend take the cross-backend copy path.
        buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        buft.device = fake_dev();
    }
    return &buft;
}

// Fake copy-stream iface (golem-copy-stream.h). The compute and copy streams
// are synchronous (every op lands at issue); the D2H stream is DEFERRED: its
// copies and event records queue up and run only when something waits on
// them (sync, a compute-stream wait on a D2H-recorded event), and they read
// their source at RUN time — so a ring slot reused before its D2H ran, or a
// missing release wait, shows up as wrong destination bytes.
struct fg_event { bool done = true; };
struct fg_d2h_op { void * dst; const char * src; size_t size; fg_event * ev; };
inline std::deque<fg_d2h_op> & fg_d2h_queue() {
    static std::deque<fg_d2h_op> q;
    return q;
}
inline void fg_d2h_run_until(const fg_event * ev) {
    auto & q = fg_d2h_queue();
    while (!q.empty() && (ev == nullptr || !ev->done)) {
        const fg_d2h_op op = q.front();
        q.pop_front();
        if (op.ev != nullptr) {
            op.ev->done = true;
        } else {
            memcpy(op.dst, op.src, op.size);
        }
    }
}
inline const golem_copy_stream_iface * fake_cs_iface() {
    static golem_copy_stream_iface cs = {};
    if (cs.set_async == nullptr) {
        cs.set_async = [](ggml_backend_t, ggml_tensor * t, const void * data, size_t off, size_t size) {
            memcpy((char *) t->data + off, data, size);
        };
        cs.event_new = [](ggml_backend_t, int) { return (void *) new fg_event(); };
        cs.event_free = [](void * ev) { delete (fg_event *) ev; };
        cs.record_compute = [](ggml_backend_t, void * ev) { ((fg_event *) ev)->done = true; };
        cs.record_copy = [](ggml_backend_t, void * ev) { ((fg_event *) ev)->done = true; };
        cs.copy_wait = [](ggml_backend_t, void *) {};
        cs.compute_wait = [](ggml_backend_t, void * ev) { fg_d2h_run_until((fg_event *) ev); };
        cs.sync = [](ggml_backend_t) { fg_d2h_run_until(nullptr); };
        cs.elapsed_ms = [](void *, void *) { return 0.0f; };
        cs.d2d_compute = [](ggml_backend_t, ggml_tensor * dst, size_t dst_off, const ggml_tensor * src,
                            size_t src_off, size_t size) {
            memcpy((char *) dst->data + dst_off, (const char *) src->data + src_off, size);
        };
        cs.d2h_get_async = [](ggml_backend_t, void * dst, const ggml_tensor * src, size_t off, size_t size) {
            fg_d2h_queue().push_back({dst, (const char *) src->data + off, size, nullptr});
        };
        cs.record_d2h = [](ggml_backend_t, void * ev) {
            ((fg_event *) ev)->done = false;
            fg_d2h_queue().push_back({nullptr, nullptr, 0, (fg_event *) ev});
        };
        cs.d2h_wait = [](ggml_backend_t, void *) {}; // compute-recorded events are complete
        cs.event_query = [](void * ev) { return ((fg_event *) ev)->done ? 1 : 0; };
    }
    return &cs;
}

inline ggml_backend_reg_t fake_reg() {
    static ggml_backend_reg reg = {};
    if (reg.iface.get_name == nullptr) {
        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.iface.get_name = [](ggml_backend_reg_t) { return "FAKEGPU"; };
        reg.iface.get_device_count = [](ggml_backend_reg_t) { return (size_t) 1; };
        reg.iface.get_device = [](ggml_backend_reg_t, size_t) { return fake_dev(); };
        reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) {
            return strcmp(name, "golem_copy_stream_iface") == 0 ? const_cast<void *>((const void *) fake_cs_iface()) : nullptr;
        };
    }
    return &reg;
}

inline ggml_backend_dev_t fake_dev() {
    static ggml_backend_device dev = {};
    if (dev.iface.get_name == nullptr) {
        dev.iface.get_name = [](ggml_backend_dev_t) { return "FAKEGPU0"; };
        dev.iface.get_description = [](ggml_backend_dev_t) { return "golem test fake GPU"; };
        dev.iface.get_memory = [](ggml_backend_dev_t, size_t * fr, size_t * to) { *fr = *to = 0; };
        dev.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        dev.iface.get_props = [](ggml_backend_dev_t, ggml_backend_dev_props * p) {
            memset(p, 0, sizeof(*p));
            p->name = p->description = "FAKEGPU0";
        };
        dev.iface.init_backend = [](ggml_backend_dev_t, const char *) { return fake_backend(); };
        dev.iface.get_buffer_type = [](ggml_backend_dev_t) { return fake_buft(); };
        // ids-pin staging and the like use the CPU host buffer type
        dev.iface.get_host_buffer_type = [](ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); };
        dev.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
        dev.iface.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t buft_) {
            return buft_ == fake_buft();
        };
        dev.reg = fake_reg();
    }
    return &dev;
}

// One-shot failure injection: when armed, the next fake graph_compute
// returns GGML_STATUS_FAILED without running (mid-compute error tests).
inline std::atomic<bool> & fg_fail_next_graph() {
    static std::atomic<bool> f{false};
    return f;
}

inline ggml_backend_t fake_backend() {
    static ggml_backend bk = {};
    if (bk.iface.get_name == nullptr) {
        bk.iface.get_name = [](ggml_backend_t) { return "FAKEGPU"; };
        bk.iface.synchronize = [](ggml_backend_t) {};
        bk.iface.graph_compute = [](ggml_backend_t, ggml_cgraph * gf) {
            if (fg_fail_next_graph().exchange(false)) {
                return GGML_STATUS_FAILED;
            }
            ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
            if (plan.work_size > 0) {
                plan.work_data = (uint8_t *) malloc(plan.work_size);
            }
            enum ggml_status st = ggml_graph_compute(gf, &plan);
            free(plan.work_data);
            return st;
        };
        bk.device = fake_dev();
    }
    return &bk;
}

} // namespace golem_test

#endif /* GOLEM_TEST_FAKEGPU_HPP */
