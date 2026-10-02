// Golem pooled-run H2D copy plan (P3copy). The pooled-run copy never reads
// past the last expert's slot. Slots are exactly expert_size whenever
// expert_size is the pool's max stride, so a single copy of expert_size +
// padding from the last slot would read 512 B past it — normally the
// neighbour slot, and past the arena mapping (SIGSEGV) when that slot ends
// the arena. This plan issues the padding as a separate copy of the last
// expert's own first bytes.
//
// set_fn issues one copy (the scheduler passes ggml_backend_tensor_set_async
// bound to the split backend).
#ifndef GOLEM_P3_COPY_H
#define GOLEM_P3_COPY_H

#include <cstddef>

typedef void (*golem_set_copy_fn)(void *dst, const void *src, size_t dst_off, size_t len, void *ud);

void golem_copy_pooled_expert_run(void *dst, size_t expert_offset, size_t expert_size,
                                  size_t padding_len, const void *const *ptrs, size_t n,
                                  golem_set_copy_fn set_fn, void *ud);

#endif
