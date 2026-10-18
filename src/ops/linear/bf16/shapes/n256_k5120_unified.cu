#include "ops/common/device_shared_memory.h"
#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes_unified.h"
#include "ops/linear/bf16/bf16_launch_unified.cuh"

namespace ninfer::ops::detail::unified {
Bf16Launch select_bf16_n256_k5120(std::int32_t tokens) {
    // Same ceiling as the legacy selector: <16,8,16> stages 98,304 bytes and Turing grants
    // 65,536, so the wide split is only launchable on a card that will admit it. Reachable on
    // Turing through NINFER_LINEAR_ROUTES=unified, which bypasses the measured width bands.
    if (tokens <= 76 && current_device_shared_memory_limit() >=
                            Bf16A16SlicedKMmaSchedule<16, 8, 16>::kSharedBytes)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 16>, 5120>,
            pdl::Dependency::Programmatic>;
    if (tokens <= 160)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8>, 5120>,
            pdl::Dependency::Programmatic>;
    if (tokens <= 640) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 5120>>;
    if (tokens <= 1024) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
    if (tokens <= 1280)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T32K64S3, 5120>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 5120>>;
}
} // namespace ninfer::ops::detail::unified
