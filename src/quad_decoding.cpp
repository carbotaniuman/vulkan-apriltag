#include "quad_decoding.h"

#include "apriltag.h"
#include "common/matd.h"
#include "tag36h11.h"

#include <cstring>

struct quad_decode_task_data {
    int i0, i1;
    zarray_t *quads;
    apriltag_detector_t *td;

    image_u8_t *im;
    zarray_t *detections;

    image_u8_t *im_samples;
};

extern "C" {
void quad_decode_task(void *_u);
}

static Detection to_detection(const apriltag_detection_t *det,
                              uint32_t blob_index) {
    Detection d{.id = det->id,
                .hamming = det->hamming,
                .decision_margin = det->decision_margin,
                .center = {det->c[0], det->c[1]},
                .corners = {},
                .blob_index = blob_index};
    for (int k = 0; k < 4; k++) {
        d.corners[k][0] = det->p[k][0];
        d.corners[k][1] = det->p[k][1];
    }
    return d;
}

static std::vector<Detection> decode_quad(apriltag_detector_t *detector,
                                          const QuadCorners &corners,
                                          image_u8_t *im) {
    std::vector<Detection> found;
    zarray_t *detections = zarray_create(sizeof(apriltag_detection_t *));
    zarray_t *quads = zarray_create(sizeof(struct quad));
    struct quad q{};
    std::memcpy(q.p, corners.corners, sizeof(q.p));
    q.reversed_border = corners.reversed_border != 0;
    zarray_add(quads, &q);

    quad_decode_task_data task{};
    task.i0 = 0;
    task.i1 = 1;
    task.quads = quads;
    task.td = detector;
    task.im = im;
    task.detections = detections;
    quad_decode_task(&task);

    for (int j = 0; j < zarray_size(detections); j++) {
        apriltag_detection_t *det;
        zarray_get(detections, j, &det);
        found.push_back(to_detection(det, corners.blob_index));
    }

    zarray_get(quads, 0, &q);
    if (q.H) {
        matd_destroy(q.H);
    }
    if (q.Hinv) {
        matd_destroy(q.Hinv);
    }
    zarray_destroy(quads);
    apriltag_detections_destroy(detections);
    return found;
}

QuadDecoder::QuadDecoder(unsigned threads)
    : m_detector(apriltag_detector_create()), m_family(tag36h11_create()) {
    apriltag_detector_add_family(m_detector, m_family);
    m_detector->quad_decimate = 1;
    m_detector->debug = false;
    m_detector->qtp.min_white_black_diff = 5;
    for (unsigned t = 1; t < threads; t++) {
        m_workers.emplace_back([this] { worker(); });
    }
}

QuadDecoder::~QuadDecoder() {
    {
        std::lock_guard lock(m_mutex);
        m_stop = true;
    }
    m_start.notify_all();
    for (std::thread &t : m_workers) {
        t.join();
    }
    apriltag_detector_destroy(m_detector);
    tag36h11_destroy(m_family);
}

std::vector<Detection> QuadDecoder::decode(const QuadCorners *corners,
                                           size_t count, uint32_t width,
                                           uint32_t height,
                                           const uint8_t *grayscale) {
    m_corners = corners;
    m_count = count;
    m_width = width;
    m_height = height;
    m_grayscale = grayscale;
    m_per_quad.assign(count, {});
    m_next.store(0, std::memory_order_relaxed);

    const bool wake = !m_workers.empty() && count > 1;
    if (wake) {
        std::lock_guard lock(m_mutex);
        m_generation++;
        m_busy = unsigned(m_workers.size());
    }
    if (wake) {
        m_start.notify_all();
    }
    decode_some();
    if (wake) {
        std::unique_lock lock(m_mutex);
        m_finished.wait(lock, [&] { return m_busy == 0; });
    }

    std::vector<Detection> found;
    for (const auto &detections : m_per_quad) {
        found.insert(found.end(), detections.begin(), detections.end());
    }
    return found;
}

void QuadDecoder::decode_some() {
    image_u8_t im{
        .width = static_cast<int32_t>(m_width),
        .height = static_cast<int32_t>(m_height),
        .stride = static_cast<int32_t>(m_width),
        .buf = const_cast<uint8_t *>(m_grayscale),
    };
    for (size_t i = m_next.fetch_add(1, std::memory_order_relaxed); i < m_count;
         i = m_next.fetch_add(1, std::memory_order_relaxed)) {
        m_per_quad[i] = decode_quad(m_detector, m_corners[i], &im);
    }
}

void QuadDecoder::worker() {
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(m_mutex);
            m_start.wait(lock, [&] { return m_stop || m_generation != seen; });
            if (m_stop) {
                return;
            }
            seen = m_generation;
        }
        decode_some();
        bool last;
        {
            std::lock_guard lock(m_mutex);
            last = --m_busy == 0;
        }
        if (last) {
            m_finished.notify_one();
        }
    }
}
