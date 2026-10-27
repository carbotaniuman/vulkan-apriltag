#include "vk/fit_quad.h"

#include "shaders/fit_quad/maxima.slang.h"
#include "shaders/fit_quad/scan.slang.h"
#include "shaders/fit_quad/search.slang.h"
#include "shaders/fit_quad/sort.slang.h"

#include <algorithm>

namespace avk {
constexpr uint32_t GROUPS = 256;
constexpr uint32_t SORT_GROUPS = 512;

constexpr uint32_t SEARCH_WG_SIZE[] = {128};

static uint32_t pick_wg_size(const support::Context &ctx) {
    if (!ctx.caps.int16_storage) {
        throw std::runtime_error("quad fitting needs storageBuffer16BitAccess "
                                 "and shaderInt16 for its 16-bit cluster "
                                 "indexes, which " +
                                 ctx.report.name + " lacks");
    }
    return ctx.caps.max_workgroup_invocations >= 256 ? 256 : 128;
}

static void mark(vk::CommandBuffer cmd, support::GpuTimer *timer,
                 const char *label) {
    if (timer) {
        timer->mark(cmd, label);
    }
}

static void dispatch(vk::CommandBuffer cmd, const support::Pipeline &pipeline,
                     const FitQuadPush &push, uint32_t groups,
                     support::GpuTimer *timer, const char *label = nullptr,
                     bool last = false) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline.pipeline);
    cmd.pushConstants<FitQuadPush>(*pipeline.layout,
                                   vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(groups, 1, 1);
    if (label) {
        mark(cmd, timer, label);
    }
    if (!last) {
        support::compute_to_compute_barrier(cmd);
    }
}

FitQuad::FitQuad(const support::Context &ctx) : wg_size(pick_wg_size(ctx)) {
    const uint32_t spec[] = {wg_size};
    const auto make = [&](std::span<const uint32_t> spirv, const char *entry,
                          std::span<const uint32_t> spec_constants) {
        return support::make_pipeline(ctx, spirv, sizeof(FitQuadPush),
                                      spec_constants, entry);
    };
    units = make(fit_quad_sort_slang, "units", spec);
    sort = make(fit_quad_sort_slang, "sort", spec);
    merge = make(fit_quad_sort_slang, "merge", spec);
    gather = make(fit_quad_sort_slang, "gather", spec);
    scan_reduce = make(fit_quad_scan_slang, "scan_reduce", spec);
    scan_tiles = make(fit_quad_scan_slang, "scan_tiles", spec);
    scan = make(fit_quad_scan_slang, "scan", spec);
    window_errors = make(fit_quad_maxima_slang, "window_errors", spec);
    filter = make(fit_quad_maxima_slang, "filter", spec);
    maxima = make(fit_quad_maxima_slang, "maxima", spec);
    search = make(fit_quad_search_slang, "search", SEARCH_WG_SIZE);
}

uint32_t FitQuad::merge_rounds(uint32_t width, uint32_t height) {
    const uint32_t largest =
        std::min(MAX_POINTS_PER_PIXEL * width * height, 4 * (width + height));
    uint32_t rounds = 0;
    while ((SORT_CHUNK << rounds) < largest) {
        rounds++;
    }
    return rounds;
}

void FitQuad::record(vk::CommandBuffer cmd, FitQuadPush push,
                     support::GpuTimer *timer) const {
    push.round = 0;
    record_sort(cmd, push, timer);
    record_scan(cmd, push, timer);
    record_maxima(cmd, push, timer);
    record_search(cmd, push, timer);
}

void FitQuad::record_sort(vk::CommandBuffer cmd, FitQuadPush push,
                          support::GpuTimer *timer) const {
    dispatch(cmd, units, push, GROUPS, timer, "fit_quad_units");
    dispatch(cmd, sort, push, SORT_GROUPS, timer, "fit_quad_sort");
    const uint32_t rounds = merge_rounds(push.width, push.height);
    for (uint32_t round = 0; round < rounds; round++) {
        push.round = round;
        dispatch(cmd, merge, push, SORT_GROUPS, timer);
    }
    push.round = 0;
    mark(cmd, timer, "fit_quad_merge");
    dispatch(cmd, gather, push, SORT_GROUPS, timer, "fit_quad_gather");
}

void FitQuad::record_scan(vk::CommandBuffer cmd, const FitQuadPush &push,
                          support::GpuTimer *timer) const {
    dispatch(cmd, scan_reduce, push, GROUPS, timer, "fit_quad_scan_reduce");
    dispatch(cmd, scan_tiles, push, 1, timer, "fit_quad_scan_tiles");
    dispatch(cmd, scan, push, GROUPS, timer, "fit_quad_scan");
}

void FitQuad::record_maxima(vk::CommandBuffer cmd, const FitQuadPush &push,
                            support::GpuTimer *timer) const {
    dispatch(cmd, window_errors, push, GROUPS, timer);
    dispatch(cmd, filter, push, GROUPS, timer);
    dispatch(cmd, maxima, push, GROUPS, timer);
    mark(cmd, timer, "fit_quad_maxima");
}

void FitQuad::record_search(vk::CommandBuffer cmd, const FitQuadPush &push,
                            support::GpuTimer *timer) const {
    dispatch(cmd, search, push, GROUPS, timer, "fit_quad_search", true);
}

} // namespace avk
