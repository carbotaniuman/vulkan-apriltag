#ifndef VK_BOUNDARIES_H
#define VK_BOUNDARIES_H

#include "support/context.h"
#include "support/pipeline.h"
#include "support/timer.h"

#include "shared/boundaries.h"

namespace avk {

struct Boundaries {
    explicit Boundaries(const support::Context &ctx);

    static vk::DeviceSize point_bytes(uint32_t width, uint32_t height) {
        return MAX_POINTS_PER_PIXEL * vk::DeviceSize(width) * height *
               sizeof(uint32_t);
    }
    static vk::DeviceSize cluster_index_bytes(uint32_t width, uint32_t height) {
        return MAX_POINTS_PER_PIXEL * vk::DeviceSize(width) * height *
               sizeof(uint16_t);
    }
    static constexpr vk::DeviceSize KEY_BYTES = MAX_CLUSTERS * sizeof(uint64_t);

    // How many cluster slots for an image of a given size.
    static uint32_t cluster_slots(uint32_t width, uint32_t height) {
        return uint64_t(width) * height <= 1024 * 1024 ? 1u << 14
                                                       : MAX_CLUSTERS;
    }

    // Read a key from the map, as `(min label << 32) | max label`,
    // or 0 for an empty slot (the map stores the min label + 1).
    static uint64_t label_key(uint64_t stored) {
        return stored == 0 ? 0 : stored - (uint64_t(1) << 32);
    }
    static constexpr vk::DeviceSize CLUSTER_BYTES =
        MAX_CLUSTERS * sizeof(ClusterAccumulator);

    void record(vk::CommandBuffer cmd, const BoundariesPush &push,
                support::GpuTimer *timer = nullptr) const;

    // boundaries_find_points' work-group size.
    uint32_t find_points_wg_size;
    support::Pipeline reset;
    support::Pipeline find;
    support::Pipeline accumulate;
};

} // namespace avk

#endif
