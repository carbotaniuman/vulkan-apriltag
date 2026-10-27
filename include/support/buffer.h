#ifndef SUPPORT_BUFFER_H
#define SUPPORT_BUFFER_H

#include "support/context.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstddef>

namespace support {

struct Buffer {
    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::Buffer buffer{nullptr};
    vk::DeviceAddress address = 0;
    std::byte *mapped = nullptr;
};

Buffer make_buffer(const Context &ctx, vk::DeviceSize size,
                   vk::MemoryPropertyFlags required,
                   vk::MemoryPropertyFlags preferred = {});

struct Range {
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
};

// Places a 256-byte-aligned range of `size` bytes at `end`
// and adjusts `end`.
Range align_range(vk::DeviceSize &end, vk::DeviceSize size);

} // namespace support

#endif
