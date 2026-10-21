#include "ops/linear/q4/q4_shapes_vision.h"
#include <stdexcept>

namespace ninfer::ops::detail {

// A vision projection runs at one image's patch count: a few hundred columns, ~1,024 at most
// (`--vision-max-merged`). The thresholds follow the 1152-wide tower's Q4 vision shapes, which are
// the same three kernels over a narrower input.
Q4Launch select_q4_n4096_k1024(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0) {
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    }
    if (tokens <= 36) return launch_q4_simt_r8_c4;
    if (tokens <= 320) return launch_q4_mma_r64_c64;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
