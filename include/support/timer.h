#ifndef SUPPORT_TIMER_H
#define SUPPORT_TIMER_H

#include "support/context.h"

#include <vulkan/vulkan_raii.hpp>

#include <string>
#include <utility>
#include <vector>

namespace support {

// `reset` to start
// `mark` after each stage
// `read` at end
struct GpuTimer {
    explicit GpuTimer(const Context &ctx, uint32_t capacity = 64);

    void reset(vk::CommandBuffer cmd);
    void mark(vk::CommandBuffer cmd, std::string label);
    // Unit is milis
    std::vector<std::pair<std::string, double>> read() const;

    vk::raii::QueryPool pool{nullptr};
    uint32_t capacity;
    double period_ns;
    uint64_t valid_mask;
    std::vector<std::string> labels;
};

} // namespace support

#endif
