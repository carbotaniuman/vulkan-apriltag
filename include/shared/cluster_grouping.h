#ifndef SHARED_CLUSTER_GROUPING_H
#define SHARED_CLUSTER_GROUPING_H

#include "shared/boundaries.h"

// Marks a rejected cluster in the remap table.
#define REJECTED_CLUSTER 0xffffu

// The sort chunk for the points in cluster, really shouild
// go in fit_quad but the structure is a bit weird.
#define SORT_CHUNK 1024u

struct ClusterBounds {
    uint16_t x_min;
    uint16_t x_max;
    uint16_t y_min;
    uint16_t y_max;
    int32_t gx_sum;
    int32_t gy_sum;
    int32_t pxgx_pygy_sum;
    uint32_t start;
    uint32_t count;
};

struct ClusterExtents {
    uint32_t start;
    uint32_t count;
};

struct ClusterTotals {
    // Total points in boundaries.
    uint32_t point_count;
    // Number of valid clusters.
    uint32_t cluster_count;
    // Total points belonging to valid clusters.
    uint32_t clustered_point_count;
    // Number of sort units corresponding to valid clusters.
    uint32_t sort_unit_count;
};

struct ClusterPoint {
    uint16_t packed_x;
    uint16_t packed_y;
    float slope;
};

struct ClusterGroupingPush {
    // From the boundaries stage.
    uint64_t points;
    uint64_t cluster_indexes;
    uint64_t point_count;
    uint64_t clusters;

    // Indexed by cluster slot.
    uint64_t remap;

    // MAX_CLUSTERS for all of these
    // ClusterBounds
    uint64_t bounds;
    // ClusterExtents
    uint64_t extents;
    // first sort unit
    uint64_t sort_unit_starts;
    // for the scatter
    uint64_t cursors;

    uint64_t scratch;
    uint64_t totals;

    uint64_t out_points;
    uint64_t out_cluster_indexes;

    uint32_t width;
    uint32_t height;

    uint32_t cluster_slots;
    uint32_t pad;
};

#endif
