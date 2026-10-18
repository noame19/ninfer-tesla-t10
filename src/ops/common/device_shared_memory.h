#pragma once

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>

namespace ninfer::ops::detail {

// Largest dynamic shared-memory request the calling thread's current device will grant one block
// through cudaFuncSetAttribute(MaxDynamicSharedMemorySize). Cached per device index, because a
// model split over several GPUs launches each stage on its own device and the parts need not match.
//
// This is a scheduling input, not a validation: schedule tables are written and measured on sm_86
// and sm_120, whose opt-in ceiling is 99 KiB, while Turing grants 65,536 bytes. A selector that
// hands Turing a larger schedule does not get a slow kernel, it gets
// `cudaFuncSetAttribute ... cudaErrorInvalidValue` at first launch, so selectors whose candidates
// straddle the ceiling have to ask here rather than assume a device class.
//
// Returns `fallback` when the device cannot be queried.
inline int current_device_shared_memory_limit(int fallback = 48 * 1024) noexcept {
    constexpr int kCachedDevices = 64;
    static std::array<std::atomic<int>, kCachedDevices> cache{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kCachedDevices) {
        return fallback;
    }
    auto& slot = cache[static_cast<std::size_t>(device)];
    if (const int known = slot.load(std::memory_order_relaxed); known > 0) { return known; }
    int bytes = 0;
    if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
            cudaSuccess ||
        bytes <= 0) {
        return fallback;
    }
    slot.store(bytes, std::memory_order_relaxed);
    return bytes;
}

} // namespace ninfer::ops::detail
