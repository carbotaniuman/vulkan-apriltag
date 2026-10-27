#ifndef VK_FIT_QUAD_H
#define VK_FIT_QUAD_H

#include "support/context.h"
#include "support/pipeline.h"
#include "support/timer.h"

#include "shared/fit_quad.h"

namespace avk {

struct FitQuad {
    explicit FitQuad(const support::Context &ctx);

    static vk::DeviceSize key_bytes(uint32_t width, uint32_t height) {
        return FIT_QUAD_KEYS_BYTES(MAX_POINTS_PER_PIXEL * width * height);
    }
    static vk::DeviceSize line_fit_bytes(uint32_t width, uint32_t height) {
        return FIT_QUAD_LINE_FIT_BYTES(MAX_POINTS_PER_PIXEL * width * height);
    }
    static vk::DeviceSize scratch_bytes(uint32_t width, uint32_t height) {
        return FIT_QUAD_SCRATCH_BYTES(MAX_POINTS_PER_PIXEL * width * height);
    }
    static constexpr vk::DeviceSize QUAD_BYTES =
        MAX_CLUSTERS * sizeof(FittedQuad);
    static constexpr vk::DeviceSize CORNER_BYTES =
        MAX_CLUSTERS * sizeof(QuadCorners);

    // Merge rounds recorded for an image.
    static uint32_t merge_rounds(uint32_t width, uint32_t height);

    void record(vk::CommandBuffer cmd, FitQuadPush push,
                support::GpuTimer *timer = nullptr) const;

    uint32_t wg_size;
    // Step 1: sort.slang.
    support::Pipeline units;
    support::Pipeline sort;
    support::Pipeline merge;
    support::Pipeline gather;
    // Step 2: scan.slang.
    support::Pipeline scan_reduce;
    support::Pipeline scan_tiles;
    support::Pipeline scan;
    // Step 3: maxima.slang.
    support::Pipeline window_errors;
    support::Pipeline filter;
    support::Pipeline maxima;
    // Step 4: search.slang.
    support::Pipeline search;

private:
    void record_sort(vk::CommandBuffer cmd, FitQuadPush push,
                     support::GpuTimer *timer) const;
    void record_scan(vk::CommandBuffer cmd, const FitQuadPush &push,
                     support::GpuTimer *timer) const;
    void record_maxima(vk::CommandBuffer cmd, const FitQuadPush &push,
                       support::GpuTimer *timer) const;
    void record_search(vk::CommandBuffer cmd, const FitQuadPush &push,
                       support::GpuTimer *timer) const;
};

} // namespace avk

#endif
