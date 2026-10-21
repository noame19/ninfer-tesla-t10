#include "ops/linear/q8/q8_shapes_vision.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// A vision merger projection runs at the token count of one image's merged patches: a few hundred
// to ~1,024 at most (`--vision-max-merged`), and its weight matrix is 4096 x 4608. The 5120 x 4608
// merger of the 27B uses the SIMT kernel below 6 columns and the 64x128 MMA above; the row extent
// does not change which of those is cheaper, so this table follows it.
Q8Launch select_q8_n4096_k4608(std::int32_t tokens) {
    if (tokens > 32768) throw std::invalid_argument("q8 linear: column extent exceeds 32768");
    if (tokens <= 4) return launch_q8_simt_r8_c4;
    if (tokens <= 5) return launch_q8_simt_r8_c8;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
