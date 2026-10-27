#ifndef SUPPORT_CONTEXT_H
#define SUPPORT_CONTEXT_H

#include "support/device.h"

#include <vulkan/vulkan_raii.hpp>

#include <atomic>
#include <functional>
#include <string>

namespace support {

struct ContextOptions {
    std::string device;
    // Enable Khornos validation.
    bool validation = false;
    bool force_staging = false;
};

struct Context {
    explicit Context(const ContextOptions &options = {});

    void submit_and_wait(const std::function<void(vk::CommandBuffer)> &record);

    vk::raii::Context loader;
    vk::raii::Instance instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT messenger{nullptr};
    vk::raii::PhysicalDevice physical{nullptr};
    vk::raii::Device device{nullptr};
    uint32_t queue_family = 0;
    vk::raii::Queue queue{nullptr};
    vk::raii::CommandPool command_pool{nullptr};

    DeviceReport report;
    DeviceCaps caps;
    std::atomic<uint32_t> validation_errors = 0;
};

} // namespace support

#endif
