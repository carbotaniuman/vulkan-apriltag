#include "vk/boundaries.h"

#include <algorithm>

#include "shaders/boundaries/accumulate.slang.h"
#include "shaders/boundaries/find_points.slang.h"
#include "shaders/boundaries/reset.slang.h"

namespace avk {

static constexpr uint32_t RESET_WG_SIZE[] = {128};

static constexpr uint32_t ACCUMULATE_WG_SIZE[] = {256};
static constexpr uint32_t ACCUMULATE_GROUPS = 256;

static constexpr uint32_t FIND_POINTS_WG_SIZE = 256;
static constexpr uint32_t FIND_POINTS_ROWS = 8;

static uint32_t groups(uint32_t n, uint32_t size) {
    return (n + size - 1) / size;
}

static uint32_t pick_find_points_wg_size(const support::Context &ctx) {
    if (!ctx.caps.int16_storage) {
        throw std::runtime_error("boundaries needs storageBuffer16BitAccess "
                                 "and shaderInt16 for its 16-bit cluster "
                                 "indexes, which " +
                                 ctx.report.name + " lacks");
    }
    if (!ctx.caps.subgroup_ops) {
        throw std::runtime_error("boundaries needs subgroup ballot and "
                                 "arithmetic, which " +
                                 ctx.report.name + " lacks");
    }
    return std::min(FIND_POINTS_WG_SIZE, ctx.caps.max_workgroup_invocations);
}

Boundaries::Boundaries(const support::Context &ctx)
    : find_points_wg_size(pick_find_points_wg_size(ctx)),
      reset(support::make_pipeline(ctx, boundaries_reset_slang,
                                   sizeof(BoundariesPush), RESET_WG_SIZE)) {
    const uint32_t find_spec[] = {find_points_wg_size};
    find = support::make_pipeline(ctx, boundaries_find_points_slang,
                                  sizeof(BoundariesPush), find_spec);
    accumulate =
        support::make_pipeline(ctx, boundaries_accumulate_slang,
                               sizeof(BoundariesPush), ACCUMULATE_WG_SIZE);
}

void Boundaries::record(vk::CommandBuffer cmd, const BoundariesPush &push,
                        support::GpuTimer *timer) const {
    const auto compute = vk::PipelineBindPoint::eCompute;
    const auto stage = vk::ShaderStageFlagBits::eCompute;

    cmd.bindPipeline(compute, *reset.pipeline);
    cmd.pushConstants<BoundariesPush>(*reset.layout, stage, 0, push);
    cmd.dispatch(groups(push.cluster_slots, RESET_WG_SIZE[0]), 1, 1);
    if (timer) {
        timer->mark(cmd, "boundaries_reset");
    }
    support::compute_to_compute_barrier(cmd);

    cmd.bindPipeline(compute, *find.pipeline);
    cmd.pushConstants<BoundariesPush>(*find.layout, stage, 0, push);
    cmd.dispatch(groups(push.width, find_points_wg_size),
                 groups(push.height, FIND_POINTS_ROWS), 1);
    if (timer) {
        timer->mark(cmd, "boundaries_find");
    }
    support::compute_to_compute_barrier(cmd);

    cmd.bindPipeline(compute, *accumulate.pipeline);
    cmd.pushConstants<BoundariesPush>(*accumulate.layout, stage, 0, push);
    cmd.dispatch(ACCUMULATE_GROUPS, 1, 1);
    if (timer) {
        timer->mark(cmd, "boundaries_accumulate");
    }
}

} // namespace avk
