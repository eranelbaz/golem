// Golem device-side landed wait (ABI v19). ggml-base cannot call into a
// device backend, so the device backend exports this table through
// ggml_backend_reg_get_proc_address(reg, "golem_device_wait_iface"); NULL
// there means device wait is unavailable on that backend (golem_set_device_wait
// refuses). Words are uint32 in host-mapped memory from host_alloc; every
// device-side call takes the HOST pointer and maps it itself. Sequence words
// grow monotonically, so waits compare `>=`.
#ifndef GOLEM_DEVICE_WAIT_H
#define GOLEM_DEVICE_WAIT_H

#include "ggml-backend.h"

#include <stdint.h>

#define GOLEM_DW_MAX_PRED 4

// One request publish: the layer's routed ids (a device tensor, element
// stride ids_nb0 bytes) and up to GOLEM_DW_MAX_PRED contiguous prediction
// tensors are copied into `dst` (ids first, predictions after), then `req` is
// set to `seq` -- after the copies are visible to the host.
struct golem_dw_publish {
    const struct ggml_tensor * ids;
    int32_t n_ids;
    const struct ggml_tensor * pred[GOLEM_DW_MAX_PRED];
    int32_t n_pred[GOLEM_DW_MAX_PRED];
    int32_t n_pred_tensors;
    uint32_t * dst;
    uint32_t * req;
    uint32_t seq;
};

struct golem_device_wait_iface {
    // 1 when the device supports stream wait-value (hipDeviceAttributeCanUseStreamWaitValue).
    int (*can_stream_wait)(ggml_backend_dev_t dev);
    // zeroed host-mapped memory visible to every device; NULL on failure
    void * (*host_alloc)(size_t size);
    void (*host_free)(void * ptr);
    // compute stream: the publish above
    void (*publish)(ggml_backend_t backend, const struct golem_dw_publish * p);
    // compute stream waits until every words[i] >= value. mode 1: one stream
    // wait-value per word; mode 2: one wait kernel spinning over the words.
    void (*wait)(ggml_backend_t backend, int mode, const uint32_t * words, int n, uint32_t value);
    // compute stream: dst (device I32 tensor, n contiguous elements) = src[0..n) (host-mapped)
    void (*gather)(ggml_backend_t backend, struct ggml_tensor * dst, const uint32_t * src, int n);
    // copy stream, after every copy queued before it: words[i] = value for
    // each set bit i of mask (one landed word per unit)
    void (*land_copy)(ggml_backend_t backend, uint32_t * words, uint32_t mask, uint32_t value);
    // 1 when the compute stream has no pending work (test probe)
    int (*compute_idle)(ggml_backend_t backend);
};

#endif
