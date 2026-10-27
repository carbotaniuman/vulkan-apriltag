#include "support/buffer.h"

#include <optional>
#include <stdexcept>

namespace support {

Buffer make_buffer(const Context &ctx, vk::DeviceSize size,
                   vk::MemoryPropertyFlags required,
                   vk::MemoryPropertyFlags preferred) {
    if (required & vk::MemoryPropertyFlagBits::eHostVisible) {
        required |= vk::MemoryPropertyFlagBits::eHostCoherent;
    }

    Buffer b;
    b.buffer = vk::raii::Buffer(
        ctx.device,
        vk::BufferCreateInfo{
            .size = size,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eTransferSrc |
                     vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eIndirectBuffer |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress});
    const vk::MemoryRequirements reqs = b.buffer.getMemoryRequirements();

    const vk::PhysicalDeviceMemoryProperties memory =
        ctx.physical.getMemoryProperties();
    std::optional<uint32_t> type;
    for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
        const vk::MemoryPropertyFlags flags =
            memory.memoryTypes[i].propertyFlags;
        if (!(reqs.memoryTypeBits & (1u << i)) ||
            (flags & required) != required) {
            continue;
        }
        if (!type || ((flags & preferred) == preferred &&
                      (memory.memoryTypes[*type].propertyFlags & preferred) !=
                          preferred)) {
            type = i;
        }
    }
    if (!type) {
        throw std::runtime_error("no memory type with properties " +
                                 vk::to_string(required));
    }

    const vk::MemoryAllocateFlagsInfo flags_info{
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
    b.memory = vk::raii::DeviceMemory(
        ctx.device, vk::MemoryAllocateInfo{.pNext = &flags_info,
                                           .allocationSize = reqs.size,
                                           .memoryTypeIndex = *type});
    b.buffer.bindMemory(*b.memory, 0);
    b.address = ctx.device.getBufferAddress({.buffer = *b.buffer});
    if (required & vk::MemoryPropertyFlagBits::eHostVisible) {
        b.mapped = static_cast<std::byte *>(b.memory.mapMemory(0, size));
    }
    return b;
}

Range align_range(vk::DeviceSize &end, vk::DeviceSize size) {
    Range r{.offset = (end + 255) / 256 * 256, .size = (size + 3) / 4 * 4};
    end = r.offset + r.size;
    return r;
}

} // namespace support
