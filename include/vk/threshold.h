#ifndef VK_THRESHOLD_H
#define VK_THRESHOLD_H

#include "support/context.h"
#include "support/pipeline.h"
#include "support/timer.h"

#include "shared/threshold.h"

namespace avk {

struct Threshold {
    explicit Threshold(const support::Context &ctx);

    static vk::DeviceSize image_bytes(uint32_t width, uint32_t height) {
        return vk::DeviceSize(width) * height;
    }
    static vk::DeviceSize tile_bytes(uint32_t width, uint32_t height) {
        return vk::DeviceSize(width / 4) * (height / 4) * sizeof(uint32_t);
    }

    void record(vk::CommandBuffer cmd, const ThresholdPush &push,
                support::GpuTimer *timer = nullptr) const;

    support::Pipeline minmax;
    support::Pipeline filter;
    support::Pipeline apply;
};

} // namespace avk

#endif
