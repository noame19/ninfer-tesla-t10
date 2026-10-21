#include "ops/softmax_attention/dense/packed/launch.h"

#include "ops/softmax_attention/dense/packed/kernel.cuh"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

std::int64_t stride_elements(const Tensor& tensor, int dim) {
    return tensor.nb[dim] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

template <int Br, int Bc, int HeadDim = kPackedAttentionHeadDim,
          int Heads = kPackedAttentionHeads>
void launch_flash(const Tensor& q, const Tensor& k, const Tensor& v,
                  const PackedAttentionTile* tiles, std::int32_t uniform_segment_length,
                  std::int32_t query_tiles, Tensor& out, cudaStream_t stream) {
    constexpr int kThreads = Br * 2;
    constexpr int kSmemBytes =
        (Br + 2 * Bc) * kPackedAttentionPaddedD * static_cast<int>(sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(query_tiles), static_cast<unsigned>(Heads), 1u);
    packed_attention_flash_kernel<Br, Bc, HeadDim, Heads><<<grid, kThreads, kSmemBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), tiles, q.ne[2], uniform_segment_length,
        static_cast<__nv_bfloat16*>(out.data), stride_elements(q, 0), stride_elements(q, 1),
        stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1), stride_elements(k, 2),
        stride_elements(v, 0), stride_elements(v, 1), stride_elements(v, 2));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void packed_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream) {
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t heads    = q.ne[1];
    const bool packed_segments = tiles != nullptr;
    const int max_tiles =
        packed_segments ? tiles->ne[1] : (q.ne[2] + kPackedAttentionBr - 1) / kPackedAttentionBr;
    if (packed_segments) {
        packed_attention_prepare_tiles_kernel<<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<PackedAttentionTile*>(tiles->data), max_tiles, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    const auto* tile_data =
        packed_segments ? static_cast<const PackedAttentionTile*>(tiles->data) : nullptr;
    if (head_dim == kPackedAttentionVisionHeadDim) {
        // The 4B (16 heads) and 0.8B (12 heads) vision towers.
        if (heads == 16) {
            launch_flash<kPackedAttentionBr, kPackedAttentionBc, 64, 16>(q, k, v, tile_data, 0,
                                                                        max_tiles, out, stream);
            return;
        }
        if (heads == 12) {
            launch_flash<kPackedAttentionBr, kPackedAttentionBc, 64, 12>(q, k, v, tile_data, 0,
                                                                        max_tiles, out, stream);
            return;
        }
    }
    launch_flash<kPackedAttentionBr, kPackedAttentionBc>(q, k, v, tile_data, 0, max_tiles, out,
                                                        stream);
}

std::int32_t packed_attention_uniform_tile(std::int32_t segment_length) {
    // Measured issued-MMA rates are approximately 3:6:8 for the 16/32/64
    // tiles. Minimize padded tile area divided by that rate.
    constexpr std::int32_t tiles[] = {16, 32, 64};
    constexpr std::int32_t rates[] = {3, 6, 8};
    std::int32_t best_tile         = tiles[0];
    std::int64_t best_padded       = ((segment_length + best_tile - 1) / best_tile) * best_tile;
    std::int64_t best_square       = best_padded * best_padded;
    std::int32_t best_rate         = rates[0];
    for (int i = 1; i < 3; ++i) {
        const std::int64_t padded = ((segment_length + tiles[i] - 1) / tiles[i]) * tiles[i];
        const std::int64_t square = padded * padded;
        if (square * best_rate < best_square * rates[i]) {
            best_tile   = tiles[i];
            best_square = square;
            best_rate   = rates[i];
        }
    }
    return best_tile;
}

void packed_attention_uniform_launch_with_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                                               std::int32_t segment_length, std::int32_t tile_size,
                                               Tensor& out, cudaStream_t stream) {
    const std::int32_t head_dim    = q.ne[0];
    const std::int32_t heads       = q.ne[1];
    const std::int32_t segments    = q.ne[2] / segment_length;
    const std::int32_t query_tiles = segments * ((segment_length + tile_size - 1) / tile_size);
    const bool narrow_head         = head_dim == kPackedAttentionVisionHeadDim;
    const auto dispatch = [&]<int Tile, int HeadDim, int Heads>() {
        launch_flash<Tile, Tile, HeadDim, Heads>(q, k, v, nullptr, segment_length, query_tiles, out,
                                                stream);
    };
    const auto per_tile = [&]<int Tile>() {
        if (narrow_head && heads == 16) {
            dispatch.template operator()<Tile, 64, 16>();
        } else if (narrow_head && heads == 12) {
            dispatch.template operator()<Tile, 64, 12>();
        } else {
            dispatch.template operator()<Tile, kPackedAttentionHeadDim, kPackedAttentionHeads>();
        }
    };
    switch (tile_size) {
    case 16:
        per_tile.template operator()<16>();
        return;
    case 32:
        per_tile.template operator()<32>();
        return;
    case 64:
        per_tile.template operator()<64>();
        return;
    default:
        throw std::invalid_argument("packed_softmax_attention: invalid uniform tile size");
    }
}

void packed_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out,
                                     cudaStream_t stream) {
    packed_attention_uniform_launch_with_tile(
        q, k, v, segment_length, packed_attention_uniform_tile(segment_length), out, stream);
}

} // namespace ninfer::ops::detail
