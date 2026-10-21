#pragma once

// Selectors for shapes that are not in the upstream tables.
//
// These live in their own header rather than in `q8_shapes.h` because that header is included by
// 36 translation units: adding one declaration there rebuilds all of them for no reason, while
// only `q8_dispatch.cpp` needs to know these shapes exist.

#include "ops/linear/q8/q8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

// The vision merger's second projection on a tower whose text hidden size is 4096 -- the Qwen3.5
// and Qwen3.6 9B towers. The 27B's is 5120 x 4608 and is in the upstream table; a 9B's merges
// 2x2 patches of the 1152-wide tower (4608 wide, the same as the 27B's) onto a 4096-wide text
// embedding, so only the output row count differs.
[[nodiscard]] Q8Launch select_q8_n4096_k4608(std::int32_t tokens);

} // namespace ninfer::ops::detail
