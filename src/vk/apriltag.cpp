#include "vk/apriltag.h"

#include "support/pipeline.h"

#include <cstring>

namespace avk {

using Stage = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;
using Memory = vk::MemoryPropertyFlagBits;

static constexpr vk::DeviceSize READBACK_DROPPED_OFFSET = 4;
static constexpr vk::DeviceSize READBACK_CORNERS_OFFSET = 256;

AprilTagPipeline::AprilTagPipeline(const support::Context &ctx, uint32_t width,
                                   uint32_t height)
    : width(width), height(height), threshold(ctx), segmentation(ctx),
      boundaries(ctx), grouping(ctx), fit_quad(ctx) {
    const vk::DeviceSize pixels = vk::DeviceSize(width) * height;
    const vk::DeviceSize word_bytes = Segmentation::word_bytes(width, height);
    const vk::DeviceSize tile_bytes = Threshold::tile_bytes(width, height);

    vk::DeviceSize size = 0;
    grayscale = support::align_range(size, pixels);
    // Threshold:
    const support::Range extrema = support::align_range(size, tile_bytes);
    const support::Range filtered = support::align_range(size, tile_bytes);
    const support::Range thresholded = support::align_range(size, pixels);
    // Segmentation:
    const support::Range seg_scratch = support::align_range(size, word_bytes);
    const support::Range labels = support::align_range(size, word_bytes);
    const support::Range sizes = support::align_range(size, word_bytes);
    // Boundaries:
    const support::Range points =
        support::align_range(size, Boundaries::point_bytes(width, height));
    const support::Range cluster_indexes = support::align_range(
        size, Boundaries::cluster_index_bytes(width, height));
    const support::Range point_count = support::align_range(size, 4);
    const support::Range keys =
        support::align_range(size, Boundaries::KEY_BYTES);
    const support::Range clusters =
        support::align_range(size, Boundaries::CLUSTER_BYTES);
    dropped_points_range = support::align_range(size, 4);
    // Cluster grouping:
    const support::Range remap =
        support::align_range(size, ClusterGrouping::REMAP_BYTES);
    const support::Range bounds =
        support::align_range(size, ClusterGrouping::BOUNDS_BYTES);
    const support::Range extents =
        support::align_range(size, ClusterGrouping::EXTENTS_BYTES);
    const support::Range sort_unit_starts =
        support::align_range(size, ClusterGrouping::WORD_BYTES);
    const support::Range cursors =
        support::align_range(size, ClusterGrouping::WORD_BYTES);
    const support::Range grouping_scratch =
        support::align_range(size, ClusterGrouping::SCRATCH_BYTES);
    const support::Range totals =
        support::align_range(size, sizeof(ClusterTotals));
    const support::Range out_points =
        support::align_range(size, ClusterGrouping::point_bytes(width, height));
    const support::Range out_cluster_indexes = support::align_range(
        size, ClusterGrouping::cluster_index_bytes(width, height));
    // Quad fitting:
    const support::Range fit_scratch =
        support::align_range(size, FitQuad::scratch_bytes(width, height));
    const support::Range quads =
        support::align_range(size, FitQuad::QUAD_BYTES);
    corners_range = support::align_range(size, FitQuad::CORNER_BYTES);
    corner_count = support::align_range(size, 4);

    const vk::MemoryPropertyFlags device_memory =
        ctx.caps.uma ? Memory::eDeviceLocal | Memory::eHostVisible
                     : Memory::eDeviceLocal;
    arena = support::make_buffer(ctx, size, device_memory);
    key_buffer = support::make_buffer(ctx, FitQuad::key_bytes(width, height),
                                      Memory::eDeviceLocal);
    line_fit_buffer = support::make_buffer(
        ctx, FitQuad::line_fit_bytes(width, height), Memory::eDeviceLocal);
    if (!arena.mapped) {
        upload_buffer = support::make_buffer(ctx, pixels, Memory::eHostVisible);
        readback_buffer = support::make_buffer(
            ctx, READBACK_CORNERS_OFFSET + corners_range.size,
            Memory::eHostVisible, Memory::eHostCached);
    }

    const vk::DeviceAddress a = arena.address;
    threshold_push = {.width = width,
                      .height = height,
                      .grayscale = a + grayscale.offset,
                      .extrema = a + extrema.offset,
                      .filtered = a + filtered.offset,
                      .thresholded = a + thresholded.offset};
    segmentation_push = {.width = width,
                         .height = height,
                         .thresholded = a + thresholded.offset,
                         .scratch = a + seg_scratch.offset,
                         .labels = a + labels.offset,
                         .sizes = a + sizes.offset};
    boundaries_push = {.width = width,
                       .height = height,
                       .labels = a + labels.offset,
                       .sizes = a + sizes.offset,
                       .points = a + points.offset,
                       .cluster_indexes = a + cluster_indexes.offset,
                       .point_count = a + point_count.offset,
                       .keys = a + keys.offset,
                       .clusters = a + clusters.offset,
                       .dropped_points = a + dropped_points_range.offset,
                       .cluster_slots =
                           Boundaries::cluster_slots(width, height)};
    grouping_push = {.points = a + points.offset,
                     .cluster_indexes = a + cluster_indexes.offset,
                     .point_count = a + point_count.offset,
                     .clusters = a + clusters.offset,
                     .remap = a + remap.offset,
                     .bounds = a + bounds.offset,
                     .extents = a + extents.offset,
                     .sort_unit_starts = a + sort_unit_starts.offset,
                     .cursors = a + cursors.offset,
                     .scratch = a + grouping_scratch.offset,
                     .totals = a + totals.offset,
                     .out_points = a + out_points.offset,
                     .out_cluster_indexes = a + out_cluster_indexes.offset,
                     .width = width,
                     .height = height,
                     .cluster_slots = Boundaries::cluster_slots(width, height)};
    fit_quad_push = {.points = a + out_points.offset,
                     .cluster_indexes = a + out_cluster_indexes.offset,
                     .extents = a + extents.offset,
                     .totals = a + totals.offset,
                     .sort_unit_starts = a + sort_unit_starts.offset,
                     .grayscale = a + grayscale.offset,
                     .keys = key_buffer.address,
                     .line_fits = line_fit_buffer.address,
                     .scratch = a + fit_scratch.offset,
                     .quads = a + quads.offset,
                     .corners = a + corners_range.offset,
                     .corner_count = a + corner_count.offset,
                     .width = width,
                     .height = height,
                     .round = 0};
}

void AprilTagPipeline::upload(const uint8_t *image) {
    std::byte *dst =
        upload_buffer ? upload_buffer->mapped : arena.mapped + grayscale.offset;
    std::memcpy(dst, image, vk::DeviceSize(width) * height);
}

void AprilTagPipeline::record(vk::CommandBuffer cmd,
                              support::GpuTimer *timer) const {
    if (timer) {
        timer->reset(cmd);
    }
    if (upload_buffer) {
        cmd.copyBuffer(*upload_buffer->buffer, *arena.buffer,
                       vk::BufferCopy{0, grayscale.offset,
                                      vk::DeviceSize(width) * height});
        support::barrier(cmd, Stage::eAllTransfer, Access::eTransferWrite,
                         Stage::eComputeShader, Access::eShaderStorageRead);
    }
    if (timer) {
        timer->mark(cmd, "start");
    }

    threshold.record(cmd, threshold_push, timer);
    support::compute_to_compute_barrier(cmd);
    segmentation.record(cmd, segmentation_push, timer);
    support::compute_to_compute_barrier(cmd);
    boundaries.record(cmd, boundaries_push, timer);
    support::compute_to_compute_barrier(cmd);
    grouping.record(cmd, grouping_push, timer);
    support::compute_to_compute_barrier(cmd);
    fit_quad.record(cmd, fit_quad_push, timer);

    if (readback_buffer) {
        support::barrier(cmd, Stage::eComputeShader,
                         Access::eShaderStorageWrite, Stage::eAllTransfer,
                         Access::eTransferRead);
        cmd.copyBuffer(*arena.buffer, *readback_buffer->buffer,
                       vk::BufferCopy{corner_count.offset, 0, 4});
        cmd.copyBuffer(*arena.buffer, *readback_buffer->buffer,
                       vk::BufferCopy{dropped_points_range.offset,
                                      READBACK_DROPPED_OFFSET, 4});
        cmd.copyBuffer(*arena.buffer, *readback_buffer->buffer,
                       vk::BufferCopy{corners_range.offset,
                                      READBACK_CORNERS_OFFSET,
                                      corners_range.size});
    }
    support::barrier(cmd, Stage::eAllTransfer | Stage::eComputeShader,
                     Access::eTransferWrite | Access::eShaderStorageWrite,
                     Stage::eHost, Access::eHostRead);
}

std::span<const QuadCorners> AprilTagPipeline::corners() const {
    const std::byte *count_ptr = readback_buffer
                                     ? readback_buffer->mapped
                                     : arena.mapped + corner_count.offset;
    const std::byte *corners_ptr =
        readback_buffer ? readback_buffer->mapped + READBACK_CORNERS_OFFSET
                        : arena.mapped + corners_range.offset;
    uint32_t count;
    std::memcpy(&count, count_ptr, sizeof(count));
    count = std::min<uint32_t>(count, MAX_CLUSTERS);
    return {reinterpret_cast<const QuadCorners *>(corners_ptr), count};
}

uint32_t AprilTagPipeline::dropped_points() const {
    const std::byte *ptr =
        readback_buffer ? readback_buffer->mapped + READBACK_DROPPED_OFFSET
                        : arena.mapped + dropped_points_range.offset;
    uint32_t count;
    std::memcpy(&count, ptr, sizeof(count));
    return count;
}

} // namespace avk
