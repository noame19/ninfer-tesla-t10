// Width-2 H16_h16kv4 instantiation of the INT8-family small-T partial launch, one geometry per
// unit: `small_t_i8.cuh` is a heavy template, and instantiating several geometries in one
// translation unit makes ptxas hold all their register pressure at once (measured: 6 GiB for
// three geometries on one TU, enough to exhaust this host's 23 GiB when six TUs run in
// parallel).
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh"

namespace ninfer::ops::detail {

template void launch_small_t_i8<CausalD256H16Kv4, 2, CausalAppendInput>(bool, bool, const Tensor&, CausalAppendInput, const Tensor&, float,
    PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t,
    std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
