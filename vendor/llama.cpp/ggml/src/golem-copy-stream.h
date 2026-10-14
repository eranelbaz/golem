// Golem dedicated H2D copy stream (async_h2d, design option A). ggml-base
// cannot call into a device backend, so the device backend exports this table
// through ggml_backend_reg_get_proc_address(reg, "golem_copy_stream_iface");
// NULL there means the backend has no copy stream: an async_h2d compute on it
// is a documented FFI abort (the engine refuses async_h2d on a non-HIP
// substrate). Events are opaque device events created by
// event_new; `timing != 0` creates one usable with elapsed_ms.
#ifndef GOLEM_COPY_STREAM_H
#define GOLEM_COPY_STREAM_H

#include "ggml-backend.h"

struct golem_copy_stream_iface {
    // H2D on the backend's copy stream (same contract as set_tensor_async).
    void (*set_async)(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void * (*event_new)(ggml_backend_t backend, int timing);
    void (*event_free)(void * event);
    // record on the compute stream / on the copy stream
    void (*record_compute)(ggml_backend_t backend, void * event);
    void (*record_copy)(ggml_backend_t backend, void * event);
    // the copy stream waits for `event` / the compute stream waits for `event`
    void (*copy_wait)(ggml_backend_t backend, void * event);
    void (*compute_wait)(ggml_backend_t backend, void * event);
    // blocks until the copy stream is idle
    void (*sync)(ggml_backend_t backend);
    float (*elapsed_ms)(void * start, void * end);
};

#endif
