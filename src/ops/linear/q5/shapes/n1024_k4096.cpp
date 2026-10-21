#include "ops/linear/q5/q5_shapes_vision.h"
#include <stdexcept>

namespace ninfer::ops::detail {

Q5Launch select_q5_n1024_k4096(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0) {
        throw std::invalid_argument("q5 linear: T must be a multiple of 4 in [4,131072]");
    }
    if (tokens <= 76) return launch_q5_simt_r8_c4;
    if (tokens <= 636) return launch_q5_mma_r64_c64;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
