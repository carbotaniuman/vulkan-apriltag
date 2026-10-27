#ifndef QUAD_DECODING_H
#define QUAD_DECODING_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "shared/fit_quad.h"

struct apriltag_detector;
struct apriltag_family;

struct Detection {
    int id;
    int hamming;
    float decision_margin;
    // Cennter and corners in pixels.
    double center[2];
    double corners[4][2];
    uint32_t blob_index;
};

class QuadDecoder {
public:
    explicit QuadDecoder(unsigned threads = 1);
    ~QuadDecoder();

    QuadDecoder(const QuadDecoder &) = delete;
    QuadDecoder &operator=(const QuadDecoder &) = delete;

    std::vector<Detection> decode(const QuadCorners *corners, size_t count,
                                  uint32_t width, uint32_t height,
                                  const uint8_t *grayscale);

private:
    void decode_some();
    void worker();

    apriltag_detector *m_detector;
    apriltag_family *m_family;

    // The current decode call, for the workers.
    const QuadCorners *m_corners = nullptr;
    size_t m_count = 0;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    const uint8_t *m_grayscale = nullptr;
    std::vector<std::vector<Detection>> m_per_quad;
    std::atomic<size_t> m_next{0};

    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_start;
    std::condition_variable m_finished;

    uint64_t m_generation = 0;
    unsigned m_busy = 0;
    bool m_stop = false;
};

#endif
