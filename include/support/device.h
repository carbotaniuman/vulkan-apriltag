#ifndef SUPPORT_DEVICE_H
#define SUPPORT_DEVICE_H

#include <vulkan/vulkan_raii.hpp>

#include <iosfwd>
#include <string>
#include <vector>

namespace support {

struct FeatureCheck {
    std::string name;
    bool required;
    bool supported;
};

struct DeviceReport {
    uint32_t index = 0;
    std::string name;
    vk::PhysicalDeviceType type{};
    uint32_t api_version = 0;
    std::vector<FeatureCheck> checks;

    bool suitable() const;
};

std::ostream &operator<<(std::ostream &os, const DeviceReport &report);

std::vector<DeviceReport> probe_devices();

struct DeviceCaps {
    bool int16_storage = false;
    bool timestamps = false;
    bool subgroup_ops = false;
    bool uma = false;

    uint32_t max_workgroup_invocations = 0;
    uint32_t max_subgroup_size = 0;
    uint32_t timestamp_valid_bits = 0;
    float timestamp_period_ns = 0;
};

} // namespace support

#endif
