#include "support/pipeline.h"

#include <vector>

namespace support {

Pipeline make_pipeline(const Context &ctx, std::span<const uint32_t> spirv,
                       uint32_t push_constant_size,
                       std::span<const uint32_t> spec_constants,
                       const char *entry,
                       vk::PipelineShaderStageCreateFlags stage_flags) {
    const vk::raii::ShaderModule module(
        ctx.device, vk::ShaderModuleCreateInfo{.codeSize = spirv.size_bytes(),
                                               .pCode = spirv.data()});

    const vk::PushConstantRange push_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = push_constant_size};
    Pipeline p;
    p.layout = vk::raii::PipelineLayout(
        ctx.device,
        vk::PipelineLayoutCreateInfo{.pushConstantRangeCount =
                                         push_constant_size > 0 ? 1u : 0u,
                                     .pPushConstantRanges = &push_range});

    std::vector<vk::SpecializationMapEntry> entries;
    for (uint32_t i = 0; i < spec_constants.size(); i++) {
        entries.push_back({.constantID = i,
                           .offset = i * uint32_t(sizeof(uint32_t)),
                           .size = sizeof(uint32_t)});
    }
    const vk::SpecializationInfo spec_info{
        .mapEntryCount = uint32_t(entries.size()),
        .pMapEntries = entries.data(),
        .dataSize = spec_constants.size_bytes(),
        .pData = spec_constants.data()};

    p.pipeline = vk::raii::Pipeline(
        ctx.device, nullptr,
        vk::ComputePipelineCreateInfo{
            .stage = {.flags = stage_flags,
                      .stage = vk::ShaderStageFlagBits::eCompute,
                      .module = *module,
                      .pName = entry,
                      .pSpecializationInfo = &spec_info},
            .layout = *p.layout});
    return p;
}

void barrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 src_stage,
             vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
             vk::AccessFlags2 dst_access) {
    const vk::MemoryBarrier2 b{.srcStageMask = src_stage,
                               .srcAccessMask = src_access,
                               .dstStageMask = dst_stage,
                               .dstAccessMask = dst_access};
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &b});
}

void compute_to_compute_barrier(vk::CommandBuffer cmd) {
    barrier(cmd, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderStorageRead |
                vk::AccessFlagBits2::eShaderStorageWrite);
}

} // namespace support
