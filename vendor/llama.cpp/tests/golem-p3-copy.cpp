// Golem pooled-copy regression: the pooled-run copy never reads past the last
// expert's slot. Slots are exactly expert_size whenever expert_size is the
// pool's max stride, so the old single copy of expert_size + padding from the
// last slot read 512 B past it — normally the neighbour slot, and past the
// arena mapping (SIGSEGV) when that slot ends the arena. The fixed copy plan
// issues the padding as a separate copy of the last expert's own first bytes.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

void golem_copy_pooled_expert_run(void *dst, size_t expert_offset, size_t expert_size,
                                  size_t padding_len, const void *const *ptrs, size_t n,
                                  void (*set_fn)(void *dst, const void *src, size_t dst_off,
                                                 size_t len, void *ud),
                                  void *ud);

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

struct Copy {
    const void *src;
    size_t dst_off;
    size_t len;
};

static void recording_set(void *dst, const void *src, size_t dst_off, size_t len, void *ud) {
    ((std::vector<Copy> *) ud)->push_back({src, dst_off, len});
    (void) dst;
}

// Real memcpy: an out-of-bounds source read hits the PROT_NONE guard page.
static void memcpy_set(void *dst, const void *src, size_t dst_off, size_t len, void *ud) {
    memcpy((char *) dst + dst_off, src, len);
    (void) ud;
}

} // namespace

int main() {
    const size_t expert_size = 4096;
    const size_t padding = 512;

    // Plan leg: every expert copies exactly expert_size; the padding is one
    // extra copy of the LAST routed expert's first bytes at the run's end.
    {
        std::vector<char> slots(2 * expert_size, 'x');
        std::vector<const void *> ptrs = {slots.data(), slots.data() + expert_size};
        std::vector<char> dst(2 * expert_size + padding, 0);
        std::vector<Copy> copies;
        golem_copy_pooled_expert_run(dst.data(), 3 * expert_size, expert_size, padding,
                                     ptrs.data(), 2, recording_set, &copies);
        REQUIRE(copies.size() == 3, "expected 3 copies, got %zu", copies.size());
        if (copies.size() == 3) {
            REQUIRE(copies[0].src == ptrs[0] && copies[0].dst_off == 3 * expert_size &&
                        copies[0].len == expert_size,
                    "copy 0 off/len wrong");
            REQUIRE(copies[1].src == ptrs[1] && copies[1].dst_off == 4 * expert_size &&
                        copies[1].len == expert_size,
                    "copy 1 off/len wrong");
            REQUIRE(copies[2].src == ptrs[1] && copies[2].dst_off == 5 * expert_size &&
                        copies[2].len == padding,
                    "padding copy must be the last expert's own first bytes");
        }
        // No padding when the run is not padded (pad_ok false path).
        copies.clear();
        golem_copy_pooled_expert_run(dst.data(), 0, expert_size, 0, ptrs.data(), 2,
                                     recording_set, &copies);
        REQUIRE(copies.size() == 2, "unpadded run must not copy a tail");
    }

    // Guard-page leg: the slot ends flush against a PROT_NONE guard page,
    // so a padded run reading one byte past the slot faults (SIGSEGV
    // without the fix). Sized from the real page size, not expert_size.
    {
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        const size_t body = (expert_size + page - 1) / page * page;
        char *map = (char *) mmap(nullptr, body + page, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        REQUIRE(map != MAP_FAILED, "mmap failed");
        char *guard = map + body;
        char *slot = guard - expert_size;
        memset(slot, 0x5a, expert_size);
        REQUIRE(mprotect(guard, page, PROT_NONE) == 0, "mprotect failed");
        std::vector<char> dst(expert_size + padding, 0);
        const void *ptrs[1] = {slot};
        golem_copy_pooled_expert_run(dst.data(), 0, expert_size, padding, ptrs, 1,
                                     memcpy_set, nullptr);
        REQUIRE((unsigned char) dst[expert_size] == 0x5a,
                "padding tail must hold the last expert's own bytes");
        munmap(map, body + page);
    }

    if (g_failures != 0) {
        printf("golem-p3-copy: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("golem-p3-copy: all legs passed\n");
    return 0;
}
