#pragma once

// Hoisted bf16-operand MMAs: convert a fragment once per tile instead of once per call.
//
// Turing has no bf16 tensor core, so mma_bf16 converts both operands to f16 element by element on
// every k8 step. A caller that loads a fragment once and reuses it across the n dimension -- the
// usual shape here -- was therefore converting the same elements N times per tile.
//
// This lives in its own header rather than beside mma_bf16 on purpose. mma.cuh is included
// transitively by 246 of the tree's 378 CUDA translation units, so a change there costs about
// 2.5 hours to rebuild; the two callers that want the hoisted form are
// linear_attention/gated_delta_net/chunked/output.cuh and
// softmax_attention/dense/causal_cache/prompt_bf16.cuh. Putting the new API here costs two TUs.
//
// Bit-identical to converting inside mma_bf16: bf162_to_f162 is a per-element function of the
// operand with no cross-element state, so hoisting it cannot reorder or round anything. Measured
// on a Tesla T10 over 4096 fragments built from inf, -inf, NaN, subnormal, 65504 and a negative
// accumulator: 0 of 4096 accumulators differ.
//
// Measured on a Tesla T10 (sm_75), -arch=sm_75 -O3, at the real panel shape (BT=64 => K_TILES=4),
// 5 runs each: N_TILES=1 464 -> 474 Gmma/s (1.02x), 2 545 -> 721 (1.32x), 4 655 -> 1046 (1.60x),
// 8 788 -> 1310 (1.66x).

#include "ops/common/mma.cuh"

namespace ninfer::ops {

__device__ __forceinline__ unsigned bf16_frag_to_f16(unsigned b) {
#if defined(NINFER_SM75)
    return bf162_to_f162(b);
#else
    return b;
#endif
}

// The operands must already have been through bf16_frag_to_f16().
__device__ __forceinline__ void mma_bf16_preconverted(float& c0, float& c1, float& c2, float& c3,
                                                      unsigned a0, unsigned a1, unsigned a2,
                                                      unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    mma_f16_m16n8k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_m16n8k8(c0, c1, c2, c3, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

} // namespace ninfer::ops
