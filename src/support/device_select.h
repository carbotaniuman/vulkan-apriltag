#ifndef SUPPORT_DEVICE_SELECT_H
#define SUPPORT_DEVICE_SELECT_H

#include "support/device.h"

#include <vulkan/vulkan_raii.hpp>

#include <string>
#include <vector>

namespace support {

using FeatureChain = vk::StructureChain<
    vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
    vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features>;

struct Candidate {
    vk::raii::PhysicalDevice physical;
    DeviceReport report;
    DeviceCaps caps;
    uint32_t queue_family = 0;
    FeatureChain enable;
};

vk::raii::Instance
create_instance(const vk::raii::Context &loader, bool validation,
                vk::DebugUtilsMessengerCreateInfoEXT debug_info);

std::vector<Candidate> evaluate_all(const vk::raii::Instance &instance);

Candidate &select_device(std::vector<Candidate> &candidates,
                         std::string selector);

} // namespace support

#endif
