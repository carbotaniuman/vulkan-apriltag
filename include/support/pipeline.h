#ifndef SUPPORT_PIPELINE_H
#define SUPPORT_PIPELINE_H

#include "support/context.h"

#include <vulkan/vulkan_raii.hpp>

#include <span>

namespace support {

struct Pipeline {
    vk::raii::PipelineLayout layout{nullptr};
    vk::raii::Pipeline pipeline{nullptr};
};

Pipeline make_pipeline(const Context &ctx, std::span<const uint32_t> spirv,
                       uint32_t push_constant_size,
                       std::span<const uint32_t> spec_constants = {},
                       const char *entry = "main",
                       vk::PipelineShaderStageCreateFlags stage_flags = {});

void barrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 src_stage,
             vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
             vk::AccessFlags2 dst_access);

void compute_to_compute_barrier(vk::CommandBuffer cmd);

} // namespace support

#endif
