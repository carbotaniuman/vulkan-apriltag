#include "vk/cluster_grouping.h"

#include "shaders/cluster_grouping/scan.slang.h"
#include "shaders/cluster_grouping/scatter.slang.h"

#include <stdexcept>
#include <string>

namespace avk {

static constexpr uint32_t SCATTER_WG_SIZE[] = {256};
static constexpr uint32_t SCATTER_GROUPS = 256;

static uint32_t pick_tile(const support::Context &ctx) {
    if (!ctx.caps.int16_storage) {
        throw std::runtime_error("cluster grouping needs "
                                 "storageBuffer16BitAccess and shaderInt16 "
                                 "for its 16-bit cluster indexes, which " +
                                 ctx.report.name + " lacks");
    }
    if (!ctx.caps.subgroup_ops) {
        throw std::runtime_error(
            "cluster grouping needs subgroup basic, ballot and arithmetic "
            "operations in compute shaders, which " +
            ctx.report.name + " lacks");
    }
    const uint32_t tile = ctx.caps.max_workgroup_invocations >= 256 ? 256 : 128;
    if (tile % ctx.caps.max_subgroup_size != 0) {
        throw std::runtime_error(
            "cluster grouping needs a work-group size of " +
            std::to_string(tile) + " to be a multiple of the maximum " +
            "subgroup size " + std::to_string(ctx.caps.max_subgroup_size) +
            " on " + ctx.report.name);
    }
    return tile;
}

ClusterGrouping::ClusterGrouping(const support::Context &ctx)
    : tile(pick_tile(ctx)),
      scatter(support::make_pipeline(ctx, cluster_grouping_scatter_slang,
                                     sizeof(ClusterGroupingPush),
                                     SCATTER_WG_SIZE)) {
    const uint32_t tile_size[] = {tile};
    const auto full_subgroups =
        vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroups;
    reduce = support::make_pipeline(ctx, cluster_grouping_scan_slang,
                                    sizeof(ClusterGroupingPush), tile_size,
                                    "reduce", full_subgroups);
    scan = support::make_pipeline(ctx, cluster_grouping_scan_slang,
                                  sizeof(ClusterGroupingPush), tile_size,
                                  "scan", full_subgroups);
}

void ClusterGrouping::record(vk::CommandBuffer cmd,
                             const ClusterGroupingPush &push,
                             support::GpuTimer *timer) const {
    const auto compute = vk::PipelineBindPoint::eCompute;
    const auto stage = vk::ShaderStageFlagBits::eCompute;

    cmd.bindPipeline(compute, *reduce.pipeline);
    cmd.pushConstants<ClusterGroupingPush>(*reduce.layout, stage, 0, push);
    cmd.dispatch(push.cluster_slots / tile, 1, 1);
    if (timer) {
        timer->mark(cmd, "cluster_grouping_reduce");
    }
    support::compute_to_compute_barrier(cmd);

    cmd.bindPipeline(compute, *scan.pipeline);
    cmd.pushConstants<ClusterGroupingPush>(*scan.layout, stage, 0, push);
    cmd.dispatch(push.cluster_slots / tile, 1, 1);
    if (timer) {
        timer->mark(cmd, "cluster_grouping_scan");
    }
    support::compute_to_compute_barrier(cmd);

    cmd.bindPipeline(compute, *scatter.pipeline);
    cmd.pushConstants<ClusterGroupingPush>(*scatter.layout, stage, 0, push);
    cmd.dispatch(SCATTER_GROUPS, 1, 1);
    if (timer) {
        timer->mark(cmd, "cluster_grouping_scatter");
    }
}

} // namespace avk
