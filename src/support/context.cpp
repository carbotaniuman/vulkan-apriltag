#include "support/context.h"

#include "device_select.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace support {

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debug_callback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT *data, void *user_data) {
    const bool error =
        severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    if (error) {
        ++*static_cast<std::atomic<uint32_t> *>(user_data);
    }
    std::cerr << (error ? "[vulkan error] " : "[vulkan warning] ")
              << data->pMessage << "\n";
    return VK_FALSE;
}

Context::Context(const ContextOptions &options) {
    const char *env = std::getenv("APRILTAG_VK_VALIDATION");
    const bool validation =
        options.validation || (env && *env && std::strcmp(env, "0") != 0);

    const vk::DebugUtilsMessengerCreateInfoEXT debug_info{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debug_callback,
        .pUserData = &validation_errors};
    instance = create_instance(loader, validation, debug_info);
    if (validation) {
        messenger = vk::raii::DebugUtilsMessengerEXT(instance, debug_info);
    }

    std::vector<Candidate> candidates = evaluate_all(instance);
    Candidate &selected = select_device(candidates, options.device);
    physical = selected.physical;
    report = selected.report;
    caps = selected.caps;
    caps.uma = caps.uma && !options.force_staging;
    queue_family = selected.queue_family;

    const float priority = 1.0f;
    const vk::DeviceQueueCreateInfo queue_info{.queueFamilyIndex = queue_family,
                                               .queueCount = 1,
                                               .pQueuePriorities = &priority};
    const vk::DeviceCreateInfo device_info{
        .pNext = &selected.enable.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info};
    device = vk::raii::Device(physical, device_info);
    queue = vk::raii::Queue(device, queue_family, 0);
    command_pool = vk::raii::CommandPool(
        device, vk::CommandPoolCreateInfo{
                    .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                    .queueFamilyIndex = queue_family});
}

void Context::submit_and_wait(
    const std::function<void(vk::CommandBuffer)> &record) {
    vk::raii::CommandBuffers cmds(
        device,
        vk::CommandBufferAllocateInfo{.commandPool = *command_pool,
                                      .level = vk::CommandBufferLevel::ePrimary,
                                      .commandBufferCount = 1});
    const vk::raii::CommandBuffer &cmd = cmds[0];
    cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    record(*cmd);
    cmd.end();

    vk::raii::Fence fence(device, vk::FenceCreateInfo{});
    const vk::CommandBufferSubmitInfo cmd_info{.commandBuffer = *cmd};
    queue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1,
                                  .pCommandBufferInfos = &cmd_info},
                  *fence);
    if (device.waitForFences(*fence, true, UINT64_MAX) !=
        vk::Result::eSuccess) {
        throw std::runtime_error("waiting for the command buffer failed");
    }
}

} // namespace support
