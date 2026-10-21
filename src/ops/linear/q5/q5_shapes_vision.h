#pragma once

// Selectors for vision-tower shapes that are not in the upstream tables. See q4_shapes_vision.h
// for why they are not in q5_shapes.h.

#include "ops/linear/q5/q5_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

// 4B: the attention output projection (1024 x 1024) and the MLP's second (1024 x 4096).
[[nodiscard]] Q5Launch select_q5_n1024_k1024(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n1024_k4096(std::int32_t tokens);
// 0.8B: 768 x 768 and 768 x 3072.
[[nodiscard]] Q5Launch select_q5_n768_k768(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n768_k3072(std::int32_t tokens);

} // namespace ninfer::ops::detail
