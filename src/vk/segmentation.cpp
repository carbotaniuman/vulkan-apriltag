#include "vk/segmentation.h"

#include "shaders/segmentation/final.slang.h"
#include "shaders/segmentation/label.slang.h"

namespace avk {

static constexpr uint32_t WG_X = 16;
static constexpr uint32_t WG_Y = 8;
static constexpr uint32_t WG_SIZE[] = {WG_X, WG_Y};

// Final labelling only.
static constexpr uint32_t RUN = 64;
static constexpr uint32_t RUN_SIZE[] = {RUN};

static uint32_t groups(uint32_t n, uint32_t size) {
    return (n + size - 1) / size;
}

Segmentation::Segmentation(const support::Context &ctx)
    : init(support::make_pipeline(ctx, segmentation_label_slang,
                                  sizeof(SegmentationPush), WG_SIZE, "init")),
      compress(support::make_pipeline(ctx, segmentation_label_slang,
                                      sizeof(SegmentationPush), WG_SIZE,
                                      "compress")),
      merge(support::make_pipeline(ctx, segmentation_label_slang,
                                   sizeof(SegmentationPush), WG_SIZE, "merge")),
      final_labelling(support::make_pipeline(
          ctx, segmentation_final_slang, sizeof(SegmentationPush), RUN_SIZE)) {}

void Segmentation::record(vk::CommandBuffer cmd, const SegmentationPush &push,
                          support::GpuTimer *timer) const {
    if (push.width % 2 != 0 || push.height % 2 != 0) {
        throw std::runtime_error("segmentation: image size " +
                                 std::to_string(push.width) + "x" +
                                 std::to_string(push.height) + " is not even");
    }
    const uint32_t blocks_w = push.width / 2;
    const uint32_t blocks_h = push.height / 2;
    const auto compute = vk::PipelineBindPoint::eCompute;
    const auto stage = vk::ShaderStageFlagBits::eCompute;

    cmd.bindPipeline(compute, *init.pipeline);
    cmd.pushConstants<SegmentationPush>(*init.layout, stage, 0, push);
    cmd.dispatch(groups(blocks_w, WG_X), groups(blocks_h, WG_Y), 1);
    if (timer) {
        timer->mark(cmd, "segmentation_init");
    }
    support::compute_to_compute_barrier(cmd);

    cmd.bindPipeline(compute, *compress.pipeline);
    cmd.pushConstants<SegmentationPush>(*compress.layout, stage, 0, push);
    cmd.dispatch(groups(blocks_w, WG_X), groups(blocks_h, WG_Y), 1);
    if (timer) {
        timer->mark(cmd, "segmentation_compress");
    }
    support::compute_to_compute_barrier(cmd);

    cmd.bindPipeline(compute, *merge.pipeline);
    cmd.pushConstants<SegmentationPush>(*merge.layout, stage, 0, push);
    cmd.dispatch(groups(blocks_w, WG_X), groups(blocks_h, WG_Y), 1);
    if (timer) {
        timer->mark(cmd, "segmentation_merge");
    }
    support::compute_to_compute_barrier(cmd);

    cmd.bindPipeline(compute, *compress.pipeline);
    cmd.pushConstants<SegmentationPush>(*compress.layout, stage, 0, push);
    cmd.dispatch(groups(blocks_w, WG_X), groups(blocks_h, WG_Y), 1);
    if (timer) {
        timer->mark(cmd, "segmentation_compress2");
    }
    support::compute_to_compute_barrier(cmd);

    cmd.bindPipeline(compute, *final_labelling.pipeline);
    cmd.pushConstants<SegmentationPush>(*final_labelling.layout, stage, 0,
                                        push);
    cmd.dispatch(groups(blocks_w, RUN), blocks_h, 1);
    if (timer) {
        timer->mark(cmd, "segmentation_final");
    }
}

} // namespace avk
