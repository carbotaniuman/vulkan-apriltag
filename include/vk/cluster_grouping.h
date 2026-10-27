#ifndef VK_CLUSTER_GROUPING_H
#define VK_CLUSTER_GROUPING_H

#include "support/context.h"
#include "support/pipeline.h"
#include "support/timer.h"

#include "shared/cluster_grouping.h"

namespace avk {

struct ClusterGrouping {
    explicit ClusterGrouping(const support::Context &ctx);

    static vk::DeviceSize point_bytes(uint32_t width, uint32_t height) {
        return MAX_POINTS_PER_PIXEL * vk::DeviceSize(width) * height *
               sizeof(ClusterPoint);
    }
    static vk::DeviceSize cluster_index_bytes(uint32_t width, uint32_t height) {
        return MAX_POINTS_PER_PIXEL * vk::DeviceSize(width) * height *
               sizeof(uint16_t);
    }
    static constexpr vk::DeviceSize REMAP_BYTES =
        MAX_CLUSTERS * sizeof(uint16_t);
    static constexpr vk::DeviceSize BOUNDS_BYTES =
        MAX_CLUSTERS * sizeof(ClusterBounds);
    static constexpr vk::DeviceSize EXTENTS_BYTES =
        MAX_CLUSTERS * sizeof(ClusterExtents);

    static constexpr vk::DeviceSize WORD_BYTES =
        MAX_CLUSTERS * sizeof(uint32_t);
    static constexpr vk::DeviceSize SCRATCH_BYTES =
        3 * (MAX_CLUSTERS / 128) * sizeof(uint32_t);

    void record(vk::CommandBuffer cmd, const ClusterGroupingPush &push,
                support::GpuTimer *timer = nullptr) const;

    uint32_t tile;
    support::Pipeline reduce;
    support::Pipeline scan;
    support::Pipeline scatter;
};

} // namespace avk

#endif
