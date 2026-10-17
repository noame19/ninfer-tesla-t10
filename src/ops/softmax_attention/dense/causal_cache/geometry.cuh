#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = 85 * SmallTSplitScale;
};

// The split scale follows the KV-head count. A split is a CTA, the grid is
// (KVHeads, splits, batch), and the ceiling was fitted so the product stays at the same ~340 CTAs
// whatever the head count, so two KV heads take twice the splits of four. Both values the table
// used before are reproduced exactly: 4 KV heads give 1 (the 27B's tuned value, which the 4B and 9B
// share) and 2 KV heads give 2.
template <int KVHeadsValue>
inline constexpr int kCausalSmallTSplitScale = 4 / KVHeadsValue;

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, kCausalSmallTSplitScale<4>>;
using CausalD256H16Kv4 = CausalAttentionGeometry<16, 4, kCausalSmallTSplitScale<4>>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, kCausalSmallTSplitScale<2>>;
using CausalD256H8Kv2  = CausalAttentionGeometry<8, 2, kCausalSmallTSplitScale<2>>;

// Runs `body` with the geometry the head pair names bound as `Geometry`. The GQA group
// (QHeads / KVHeads) sizes every small-T tile at compile time, so the pair is the identity rather
// than the query-head count alone: a 16-head tower over 4 KV heads is a different instantiation
// from a 16-head tower over 2, and an 8-head tower over 2 shares the second's group but not its
// row tiles. A pair outside the four is rejected where the shape is read instead of being launched
// as the nearest one.
template <typename Body>
void with_causal_geometry(std::int32_t q_heads, std::int32_t kv_heads, Body&& body) {
    if (q_heads == 24 && kv_heads == 4) {
        body.template operator()<CausalD256H24Kv4>();
    } else if (q_heads == 16 && kv_heads == 4) {
        body.template operator()<CausalD256H16Kv4>();
    } else if (q_heads == 16 && kv_heads == 2) {
        body.template operator()<CausalD256H16Kv2>();
    } else if (q_heads == 8 && kv_heads == 2) {
        body.template operator()<CausalD256H8Kv2>();
    } else {
        throw std::invalid_argument("causal attention: unsupported head geometry");
    }
}

[[nodiscard]] constexpr bool valid_causal_geometry(std::int32_t q_heads,
                                                   std::int32_t kv_heads) noexcept {
    return (q_heads == 24 && kv_heads == 4) || (q_heads == 16 && kv_heads == 4) ||
           (q_heads == 16 && kv_heads == 2) || (q_heads == 8 && kv_heads == 2);
}

} // namespace ninfer::ops
