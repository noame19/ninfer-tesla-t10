#pragma once

// Selectors for vision-tower shapes that are not in the upstream tables.
//
// These live in their own header rather than in `q4_shapes.h` because that header is included by
// 16 translation units and only `q4_dispatch.cpp` needs to know these shapes exist.
//
// The 4B tower is hidden 1024 with 16 heads and ffn 4096; the 0.8B is hidden 768 with 12 heads and
// ffn 3072. Both are narrower than the 1152-wide tower the upstream Q4 vision shapes were measured
// on, and neither shares a (rows, width) pair with it.

#include "ops/linear/q4/q4_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

// 4B: fused q/k/v (3 x 1024) and the MLP's first projection (4096), both over a 1024-wide input.
[[nodiscard]] Q4Launch select_q4_n3072_k1024(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4096_k1024(std::int32_t tokens);
// 0.8B: fused q/k/v (3 x 768) and the MLP's first projection (3072), over a 768-wide input.
[[nodiscard]] Q4Launch select_q4_n2304_k768(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n3072_k768(std::int32_t tokens);

} // namespace ninfer::ops::detail
