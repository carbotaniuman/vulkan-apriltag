#ifndef VK_SEGMENTATION_H
#define VK_SEGMENTATION_H

#include "support/context.h"
#include "support/pipeline.h"
#include "support/timer.h"

#include "shared/segmentation.h"

namespace avk {

struct Segmentation {
    explicit Segmentation(const support::Context &ctx);

    // Size of `scratch`, `labels` and `sizes`.
    static vk::DeviceSize word_bytes(uint32_t width, uint32_t height) {
        return vk::DeviceSize(width) * height * sizeof(uint32_t);
    }

    void record(vk::CommandBuffer cmd, const SegmentationPush &push,
                support::GpuTimer *timer = nullptr) const;

    support::Pipeline init;
    support::Pipeline compress;
    support::Pipeline merge;
    support::Pipeline final_labelling;
};

} // namespace avk

#endif
