#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

#if defined(NINFER_SM75)
// --- Turing (sm_75) tensor-core stand-ins --------------------------------------------------------
//
// Turing has f16 and int8 tensor cores but only the m16n8k8 (f16) and m8n8k16 (s8) shapes; the
// m16n8k16/m16n8k32 shapes every kernel in this tree uses arrived with Ampere (sm_80). Each
// emulation below splits its operands along the shape the hardware does have, using the register
// fragment layouts PTX specifies, which compose exactly:
//
//   * m16n8k16 (f16/bf16) = two m16n8k8 over the k halves. The A fragment is
//     {a0=(r0..7,k0..7), a1=(r8..15,k0..7), a2=(r0..7,k8..15), a3=(r8..15,k8..15)}, and an
//     m16n8k8 takes (rows0..7, rows8..15) as its A, so (a0,a1) and (a2,a3) are the two steps with
//     B halves b0 and b1. bf16 has no Turing MMA at all, so it is converted to f16 per element
//     first. f16 holds fewer exponent bits than bf16, so a magnitude above 65504 rounds to
//     infinity; one infinity in an operand reaches the accumulator as inf or, through inf*0, as
//     NaN. The conversion saturates instead, as cvt.rn.satfinite.f16x2.f32 does on Ampere and
//     later, so an out-of-range value costs its distance to 65504 rather than the whole tile.
//
//   * m16n8k32 (s8) = four m8n8k16. The A fragment holds {a0=(r0..7,k0..15), a1=(r8..15,k0..15),
//     a2=(r0..7,k16..31), a3=(r8..15,k16..31)} and B {b0=(k0..15), b1=(k16..31)}, and the 8-row
//     halves of the accumulator are (c0,c1) for rows0..7 and (c2,c3) for rows8..15: each m8n8k16
//     takes one row half, one k half and its C pair.
//
// ldmatrix (m8n8.x1/x2/x4) exists on Turing, so fragment assembly is unchanged.
//
// NaN (which rounds to NaN, not to infinity) is preserved.
__device__ __forceinline__ __half bf16_to_f16_saturated(float value) {
    const __half rounded = __float2half_rn(value);
    if (__hisinf(rounded)) { return __float2half_rn(value > 0.0f ? 65504.0f : -65504.0f); }
    return rounded;
}

__device__ __forceinline__ unsigned bf162_to_f162(unsigned b) {
    const auto* bf  = reinterpret_cast<const __nv_bfloat162*>(&b);
    const float2 f  = __bfloat1622float2(*bf);
    const __half2 h = __halves2half2(bf16_to_f16_saturated(f.x), bf16_to_f16_saturated(f.y));
    return *reinterpret_cast<const unsigned*>(&h);
}

// Turing-native f16 MMA, 16x8x8 (two k-steps of the m16n8k16 the callers ask for).
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Turing-native f16 MMA with f16 accumulation, 16x8x8.
__device__ __forceinline__ void mma_f16_f16acc_m16n8k8(unsigned& d0, unsigned& d1, unsigned a0,
                                                       unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Turing-native s8 MMA, 8x8x16.
__device__ __forceinline__ void mma_s8_m8n8k16(int& c0, int& c1, unsigned a0, unsigned b0) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
}
#endif // NINFER_SM75

// bf16x2 -> f16x2 for one MMA operand pair, so a caller holding a fragment across several MMAs
// converts it once instead of once per call. Turing has no bf16 tensor core, so every bf16 MMA
// pays a per-element conversion; on the three k8 steps of an m16n8k16 that is 8 conversions of the
// same A values when the inner loop walks n (the usual shape here -- see prompt_bf16.cuh's score
// loop and chunked/output.cuh's panel). Measured on a T10 at N_TILES 8: 788 -> 1310 Gmma/s.
//
// Bit-identical to converting inside mma_bf16: bf162_to_f162 is a per-element function of the
// operand with no cross-element state, so hoisting it cannot reorder or round anything.
__device__ __forceinline__ unsigned bf16_frag_to_f16(unsigned b) {
#if defined(NINFER_SM75)
    return bf162_to_f162(b);
#else
    return b;
#endif
}

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

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(NINFER_SM75)
    mma_f16_m16n8k8(c0, c1, c2, c3, bf162_to_f162(a0), bf162_to_f162(a1), bf162_to_f162(b0));
    mma_f16_m16n8k8(c0, c1, c2, c3, bf162_to_f162(a2), bf162_to_f162(a3), bf162_to_f162(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(NINFER_SM75)
    mma_f16_m16n8k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_m16n8k8(c0, c1, c2, c3, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// FP16 accumulation: twice the FP32-accumulator rate on GeForce parts. d0 holds the row-gid pair
// and d1 the row-gid+8 pair of the fragment, in the same order as mma_f16's c0..c3.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                               unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
#if defined(NINFER_SM75)
    mma_f16_f16acc_m16n8k8(d0, d1, a0, a1, b0);
    mma_f16_f16acc_m16n8k8(d0, d1, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // a0=(r0..7,k0..15), a1=(r8..15,k0..15), a2=(r0..7,k16..31), a3=(r8..15,k16..31);
    // b0 covers k0..15 and b1 k16..31; the accumulator halves are (c0,c1) rows0..7 and (c2,c3)
    // rows8..15. Each m8n8k16 therefore takes one row half, one k half and its C pair.
    mma_s8_m8n8k16(c0, c1, a0, b0);
    mma_s8_m8n8k16(c2, c3, a1, b0);
    mma_s8_m8n8k16(c0, c1, a2, b1);
    mma_s8_m8n8k16(c2, c3, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// On RTX 5090, block-scaled FP8 avoids the plain form's half-rate FP32 accumulation.
// UE8M0 0x7f is exactly 1: preserve the raw E4M3 dot product and keep the callers'
// existing row scales in FP32 after the MMA. No operand repacking is needed.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // Turing has no tf32 tensor core, so run the m16n8k8 product on the f16 unit. The accumulator
    // fragment is identical between the two, but the operand fragments are not: tf32 spreads a
    // thread's A elements as a0=(g,t) a1=(g+8,t) a2=(g,t+4) a3=(g+8,t+4) (scalar registers), while
    // f16 m16n8k8 packs A as {(g,2t),(g,2t+1)} and {(g+8,2t),(g+8,2t+1)}. The two k-rows a thread
    // needs therefore live in other lanes, so gather them with shuffles; B is transposed the same
    // way (tf32 b0=(t,g) b1=(t+4,g) against f16 {B[2t][g], B[2t+1][g]}). A tf32 column c lives in
    // lane (g, c & 3), in the low register if c < 4 and the high one otherwise.
    const unsigned lane = threadIdx.x & 31u;
    const unsigned g    = lane >> 2;
    const unsigned t    = lane & 3u;
    const unsigned c_lo = 2u * t;
    const unsigned c_hi = c_lo + 1u;
    const unsigned l_lo = (g << 2) | (c_lo & 3u);
    const unsigned l_hi = (g << 2) | (c_hi & 3u);
    const bool hi_lo    = c_lo >= 4u;
    const bool hi_hi    = c_hi >= 4u;
    constexpr unsigned kMask = 0xffffffffu;

    const unsigned a0_lo = __shfl_sync(kMask, a0, l_lo), a2_lo = __shfl_sync(kMask, a2, l_lo);
    const unsigned a0_hi = __shfl_sync(kMask, a0, l_hi), a2_hi = __shfl_sync(kMask, a2, l_hi);
    const unsigned a1_lo = __shfl_sync(kMask, a1, l_lo), a3_lo = __shfl_sync(kMask, a3, l_lo);
    const unsigned a1_hi = __shfl_sync(kMask, a1, l_hi), a3_hi = __shfl_sync(kMask, a3, l_hi);
    const unsigned b0_lo = __shfl_sync(kMask, b0, l_lo), b1_lo = __shfl_sync(kMask, b1, l_lo);
    const unsigned b0_hi = __shfl_sync(kMask, b0, l_hi), b1_hi = __shfl_sync(kMask, b1, l_hi);

    const float a_g_lo  = __uint_as_float(hi_lo ? a2_lo : a0_lo);
    const float a_g_hi  = __uint_as_float(hi_hi ? a2_hi : a0_hi);
    const float a_g8_lo = __uint_as_float(hi_lo ? a3_lo : a1_lo);
    const float a_g8_hi = __uint_as_float(hi_hi ? a3_hi : a1_hi);
    const float b_lo    = __uint_as_float(hi_lo ? b1_lo : b0_lo);
    const float b_hi    = __uint_as_float(hi_hi ? b1_hi : b0_hi);

    const half2 fa0 = __floats2half2_rn(a_g_lo, a_g_hi);
    const half2 fa1 = __floats2half2_rn(a_g8_lo, a_g8_hi);
    const half2 fb0 = __floats2half2_rn(b_lo, b_hi);
    mma_f16_m16n8k8(c0, c1, c2, c3, *reinterpret_cast<const unsigned*>(&fa0),
                    *reinterpret_cast<const unsigned*>(&fa1),
                    *reinterpret_cast<const unsigned*>(&fb0));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops
