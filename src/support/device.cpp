#include "device_select.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace support {

static const char *const VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation";

static std::string version_string(uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

static bool
has_extension(const std::vector<vk::ExtensionProperties> &extensions,
              const char *name) {
    return std::any_of(extensions.begin(), extensions.end(),
                       [&](const vk::ExtensionProperties &e) {
                           return std::strcmp(e.extensionName, name) == 0;
                       });
}

static Candidate evaluate(const vk::raii::PhysicalDevice &physical,
                          uint32_t index) {
    Candidate c{physical};
    const vk::PhysicalDeviceProperties props = physical.getProperties();
    c.report.index = index;
    c.report.name = props.deviceName.data();
    c.report.type = props.deviceType;
    c.report.api_version = props.apiVersion;

    auto check = [&](std::string name, bool required, bool supported) {
        c.report.checks.push_back({std::move(name), required, supported});
    };

    check("Vulkan 1.3", true, props.apiVersion >= VK_API_VERSION_1_3);
    if (props.apiVersion < VK_API_VERSION_1_3) {
        return c;
    }

    const auto have =
        physical.getFeatures2<vk::PhysicalDeviceFeatures2,
                              vk::PhysicalDeviceVulkan11Features,
                              vk::PhysicalDeviceVulkan12Features,
                              vk::PhysicalDeviceVulkan13Features>();
    const auto &have10 = have.get<vk::PhysicalDeviceFeatures2>().features;
    const auto &have11 = have.get<vk::PhysicalDeviceVulkan11Features>();
    const auto &have12 = have.get<vk::PhysicalDeviceVulkan12Features>();
    const auto &have13 = have.get<vk::PhysicalDeviceVulkan13Features>();
    auto &want10 = c.enable.get<vk::PhysicalDeviceFeatures2>().features;
    auto &want11 = c.enable.get<vk::PhysicalDeviceVulkan11Features>();
    auto &want12 = c.enable.get<vk::PhysicalDeviceVulkan12Features>();
    auto &want13 = c.enable.get<vk::PhysicalDeviceVulkan13Features>();

    auto feature = [&](const char *name, bool required, vk::Bool32 has,
                       vk::Bool32 &enable) {
        enable = has;
        check(name, required, has);
    };
    feature("shaderInt64", true, have10.shaderInt64, want10.shaderInt64);
    feature("storageBuffer16BitAccess", false, have11.storageBuffer16BitAccess,
            want11.storageBuffer16BitAccess);
    feature("shaderInt16", false, have10.shaderInt16, want10.shaderInt16);
    feature("bufferDeviceAddress", true, have12.bufferDeviceAddress,
            want12.bufferDeviceAddress);
    feature("scalarBlockLayout", true, have12.scalarBlockLayout,
            want12.scalarBlockLayout);
    feature("vulkanMemoryModel", true, have12.vulkanMemoryModel,
            want12.vulkanMemoryModel);
    feature("vulkanMemoryModelDeviceScope", true,
            have12.vulkanMemoryModelDeviceScope,
            want12.vulkanMemoryModelDeviceScope);
    feature("synchronization2", true, have13.synchronization2,
            want13.synchronization2);
    feature("maintenance4", true, have13.maintenance4, want13.maintenance4);
    feature("computeFullSubgroups", true, have13.computeFullSubgroups,
            want13.computeFullSubgroups);

    const vk::PhysicalDeviceLimits &limits = props.limits;
    check("maxComputeSharedMemorySize >= 16 KB", true,
          limits.maxComputeSharedMemorySize >= 16 * 1024);
    check("maxComputeWorkGroupInvocations >= 256", false,
          limits.maxComputeWorkGroupInvocations >= 256);

    const auto families = physical.getQueueFamilyProperties();
    std::optional<uint32_t> family;
    for (uint32_t i = 0; i < families.size(); i++) {
        if ((families[i].queueFlags & vk::QueueFlagBits::eCompute) &&
            (!family || (families[*family].timestampValidBits == 0 &&
                         families[i].timestampValidBits != 0))) {
            family = i;
        }
    }
    const uint32_t timestamp_bits =
        family ? families[*family].timestampValidBits : 0;
    check("compute queue", true, family.has_value());
    check("timestamps", false, timestamp_bits > 0);

    const auto props2 =
        physical.getProperties2<vk::PhysicalDeviceProperties2,
                                vk::PhysicalDeviceVulkan11Properties,
                                vk::PhysicalDeviceVulkan13Properties>();
    const auto &props11 = props2.get<vk::PhysicalDeviceVulkan11Properties>();
    const auto &props13 = props2.get<vk::PhysicalDeviceVulkan13Properties>();
    const vk::SubgroupFeatureFlags subgroup_needed =
        vk::SubgroupFeatureFlagBits::eBasic |
        vk::SubgroupFeatureFlagBits::eBallot |
        vk::SubgroupFeatureFlagBits::eArithmetic;
    const bool subgroup_ops =
        (props11.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
        (props11.subgroupSupportedOperations & subgroup_needed) ==
            subgroup_needed;
    check("subgroup ballot + arithmetic", false, subgroup_ops);

    c.queue_family = family.value_or(0);
    c.caps.int16_storage =
        have11.storageBuffer16BitAccess && have10.shaderInt16;
    c.caps.timestamps = timestamp_bits > 0;
    c.caps.subgroup_ops = subgroup_ops;
    c.caps.uma = props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu ||
                 props.deviceType == vk::PhysicalDeviceType::eCpu;
    c.caps.max_workgroup_invocations = limits.maxComputeWorkGroupInvocations;
    c.caps.max_subgroup_size = props13.maxSubgroupSize;
    c.caps.timestamp_valid_bits = timestamp_bits;
    c.caps.timestamp_period_ns = limits.timestampPeriod;
    return c;
}

static std::string describe_missing(const DeviceReport &r) {
    std::string s =
        "  [" + std::to_string(r.index) + "] " + r.name + ": missing";
    for (const FeatureCheck &c : r.checks) {
        if (c.required && !c.supported) {
            s += " " + c.name;
        }
    }
    return s;
}

static int type_rank(vk::PhysicalDeviceType type) {
    switch (type) {
    case vk::PhysicalDeviceType::eDiscreteGpu:
        return 0;
    case vk::PhysicalDeviceType::eIntegratedGpu:
        return 1;
    case vk::PhysicalDeviceType::eVirtualGpu:
        return 2;
    case vk::PhysicalDeviceType::eCpu:
        return 3;
    default:
        return 4;
    }
}

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

// debug_info is used when validation is onm
vk::raii::Instance
create_instance(const vk::raii::Context &loader, bool validation,
                vk::DebugUtilsMessengerCreateInfoEXT debug_info) {
    if (loader.enumerateInstanceVersion() < VK_API_VERSION_1_3) {
        throw std::runtime_error(
            "Vulkan loader is " +
            version_string(loader.enumerateInstanceVersion()) + ", need 1.3");
    }

    vk::ApplicationInfo app{.pApplicationName = "apriltag_vk",
                            .apiVersion = VK_API_VERSION_1_3};
    vk::InstanceCreateInfo info{.pApplicationInfo = &app};
    std::vector<const char *> layers;
    std::vector<const char *> extensions;

    if (has_extension(loader.enumerateInstanceExtensionProperties(),
                      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }

    const vk::ValidationFeatureEnableEXT sync_validation =
        vk::ValidationFeatureEnableEXT::eSynchronizationValidation;
    vk::ValidationFeaturesEXT validation_features{
        .enabledValidationFeatureCount = 1,
        .pEnabledValidationFeatures = &sync_validation};

    if (validation) {
        const auto available = loader.enumerateInstanceLayerProperties();
        if (std::none_of(available.begin(), available.end(),
                         [](const vk::LayerProperties &l) {
                             return std::strcmp(l.layerName,
                                                VALIDATION_LAYER) == 0;
                         })) {
            throw std::runtime_error("validation requested but " +
                                     std::string(VALIDATION_LAYER) +
                                     " is not installed");
        }
        layers.push_back(VALIDATION_LAYER);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        info.pNext = &debug_info;
        if (has_extension(loader.enumerateInstanceExtensionProperties(
                              std::string(VALIDATION_LAYER)),
                          VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
            extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
            debug_info.pNext = &validation_features;
        }
    }

    info.setPEnabledLayerNames(layers);
    info.setPEnabledExtensionNames(extensions);
    return vk::raii::Instance(loader, info);
}

std::vector<Candidate> evaluate_all(const vk::raii::Instance &instance) {
    std::vector<Candidate> candidates;
    uint32_t index = 0;
    for (const vk::raii::PhysicalDevice &physical :
         instance.enumeratePhysicalDevices()) {
        candidates.push_back(evaluate(physical, index++));
    }
    return candidates;
}

Candidate &select_device(std::vector<Candidate> &candidates,
                         std::string selector) {
    if (candidates.empty()) {
        throw std::runtime_error("no Vulkan devices found");
    }
    if (selector.empty()) {
        if (const char *env = std::getenv("APRILTAG_VK_DEVICE")) {
            selector = env;
        }
    }

    if (!selector.empty()) {
        const bool is_index =
            std::all_of(selector.begin(), selector.end(),
                        [](unsigned char c) { return std::isdigit(c); });
        for (Candidate &c : candidates) {
            if (is_index ? std::to_string(c.report.index) == selector
                         : to_lower(c.report.name).find(to_lower(selector)) !=
                               std::string::npos) {
                if (!c.report.suitable()) {
                    throw std::runtime_error(
                        "selected Vulkan device is unsuitable:\n" +
                        describe_missing(c.report));
                }
                return c;
            }
        }
        throw std::runtime_error("no Vulkan device matches '" + selector + "'");
    }

    Candidate *best = nullptr;
    for (Candidate &c : candidates) {
        if (c.report.suitable() &&
            (!best ||
             type_rank(c.report.type) < type_rank(best->report.type))) {
            best = &c;
        }
    }
    if (!best) {
        std::string message = "no Vulkan device meets the feature floor:";
        for (const Candidate &c : candidates) {
            message += "\n" + describe_missing(c.report);
        }
        throw std::runtime_error(message);
    }
    return *best;
}

bool DeviceReport::suitable() const {
    return std::all_of(checks.begin(), checks.end(), [](const FeatureCheck &c) {
        return c.supported || !c.required;
    });
}

std::ostream &operator<<(std::ostream &os, const DeviceReport &r) {
    os << "[" << r.index << "] " << r.name << " (" << vk::to_string(r.type)
       << ", Vulkan " << version_string(r.api_version) << ")\n";
    const std::ios::fmtflags flags = os.flags();
    for (const FeatureCheck &c : r.checks) {
        os << "    " << std::left << std::setw(40) << c.name
           << (c.supported  ? "yes"
               : c.required ? "MISSING"
                            : "no")
           << "\n";
    }
    os.flags(flags);
    return os << "    => " << (r.suitable() ? "suitable" : "unsuitable")
              << "\n";
}

std::vector<DeviceReport> probe_devices() {
    vk::raii::Context loader;
    vk::raii::Instance instance = create_instance(loader, false, {});
    std::vector<DeviceReport> reports;
    for (Candidate &c : evaluate_all(instance)) {
        reports.push_back(std::move(c.report));
    }
    return reports;
}

} // namespace support
