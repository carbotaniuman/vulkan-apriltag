#ifndef SHARED_BOUNDARIES_H
#define SHARED_BOUNDARIES_H

#ifndef __SLANG__
#include <cstdint>
#endif

// Points are as follows:
// bits 0-15 packed_x [half pixel : 2, x : 14],
// bits 16-31 packed_y [black to white : 1, unused : 1, y : 14].

// Maximum number of cluster slots, including slot 0, which means
// no cluster
#define MAX_CLUSTERS (1u << 16)

// Point capacity per pixel. Because of how the thresholding works each
// pixel can contirbute at most 2 points.
#define MAX_POINTS_PER_PIXEL 2u

struct ClusterAccumulator {
    uint32_t x_min;
    uint32_t x_max;
    uint32_t y_min;
    uint32_t y_max;
    int32_t gx_sum;
    int32_t gy_sum;
    uint32_t count;
    int32_t pxgx_pygy_sum;
};

struct BoundariesPush {
    uint32_t width;
    uint32_t height;

    // From segmentation.
    uint64_t labels;
    uint64_t sizes;

    // MAX_POINTS_PER_PIXEL * width * height
    uint64_t points;
    uint64_t cluster_indexes;
    uint64_t point_count;
    // MAX_CLUSTERS
    // 64-bit keys; 0 is empty
    uint64_t keys;
    // MAX_CLUSTERS
    // ClusterAccumulators
    uint64_t clusters;
    // 1
    // Number of points that were dropped
    uint64_t dropped_points;

    uint32_t cluster_slots;
    uint32_t pad;
};

#endif
