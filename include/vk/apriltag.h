#ifndef VK_APRILTAG_H
#define VK_APRILTAG_H

#include "support/buffer.h"
#include "support/context.h"
#include "support/timer.h"
#include "vk/boundaries.h"
#include "vk/cluster_grouping.h"
#include "vk/fit_quad.h"
#include "vk/segmentation.h"
#include "vk/threshold.h"

#include <optional>
#include <span>

namespace avk {

struct AprilTagPipeline {
    AprilTagPipeline(const support::Context &ctx, uint32_t width,
                     uint32_t height);

    void upload(const uint8_t *grayscale);

    void record(vk::CommandBuffer cmd,
                support::GpuTimer *timer = nullptr) const;

    std::span<const QuadCorners> corners() const;

    uint32_t dropped_points() const;

    uint32_t width;
    uint32_t height;

    Threshold threshold;
    Segmentation segmentation;
    Boundaries boundaries;
    ClusterGrouping grouping;
    FitQuad fit_quad;

    // Device memory.
    support::Buffer arena;
    support::Buffer key_buffer;
    support::Buffer line_fit_buffer;

    // Discrete devices only.
    std::optional<support::Buffer> upload_buffer;
    std::optional<support::Buffer> readback_buffer;

    support::Range grayscale;
    support::Range corner_count;
    support::Range corners_range;
    support::Range dropped_points_range;

    ThresholdPush threshold_push;
    SegmentationPush segmentation_push;
    BoundariesPush boundaries_push;
    ClusterGroupingPush grouping_push;
    FitQuadPush fit_quad_push;
};

} // namespace avk

#endif
