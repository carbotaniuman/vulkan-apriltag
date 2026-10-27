#include "support/timer.h"

#include <stdexcept>

namespace support {

GpuTimer::GpuTimer(const Context &ctx, uint32_t capacity)
    : capacity(capacity), period_ns(ctx.caps.timestamp_period_ns),
      valid_mask(ctx.caps.timestamp_valid_bits >= 64
                     ? ~uint64_t(0)
                     : (uint64_t(1) << ctx.caps.timestamp_valid_bits) - 1) {
    if (ctx.caps.timestamps) {
        pool = vk::raii::QueryPool(
            ctx.device,
            vk::QueryPoolCreateInfo{.queryType = vk::QueryType::eTimestamp,
                                    .queryCount = capacity});
    }
}

void GpuTimer::reset(vk::CommandBuffer cmd) {
    labels.clear();
    if (*pool) {
        cmd.resetQueryPool(*pool, 0, capacity);
    }
}

void GpuTimer::mark(vk::CommandBuffer cmd, std::string label) {
    if (!*pool) {
        return;
    }
    if (labels.size() >= capacity) {
        throw std::runtime_error("GpuTimer: more than " +
                                 std::to_string(capacity) + " marks");
    }
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *pool,
                        uint32_t(labels.size()));
    labels.push_back(std::move(label));
}

std::vector<std::pair<std::string, double>> GpuTimer::read() const {
    std::vector<std::pair<std::string, double>> intervals;
    if (!*pool || labels.size() < 2) {
        return intervals;
    }
    const uint32_t n = uint32_t(labels.size());
    const auto [result, ticks] = pool.getResults<uint64_t>(
        0, n, n * sizeof(uint64_t), sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    for (uint32_t i = 1; i < n; i++) {
        const uint64_t delta = (ticks[i] - ticks[i - 1]) & valid_mask;
        intervals.emplace_back(labels[i], delta * period_ns * 1e-6);
    }
    return intervals;
}

} // namespace support
