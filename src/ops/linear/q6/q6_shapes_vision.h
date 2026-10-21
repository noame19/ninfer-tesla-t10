#pragma once

// Selectors for vision-tower shapes that are not in the upstream tables. See q4_shapes_vision.h
// for why they are not in q6_shapes.h.

#include "ops/linear/q6/q6_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

// The patch embedding: 3 * temporal(2) * patch(16) * patch(16) = 1536 wide, one row per patch
// position, projected onto the tower's hidden width. The 1152-wide tower's is in the upstream table.
[[nodiscard]] Q6Launch select_q6_n1024_k1536(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n768_k1536(std::int32_t tokens);

} // namespace ninfer::ops::detail
