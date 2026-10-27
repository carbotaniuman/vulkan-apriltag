#ifndef SHARED_FIT_QUAD_H
#define SHARED_FIT_QUAD_H

#include "shared/cluster_grouping.h"

// From AprilTag.
#define MAX_MAXIMA 10u
#define MAX_LINE_FIT_MSE 10.0f

struct LineFitPoint {
    int64_t Mx;
    int64_t My;
    int64_t Mxx;
    int64_t Myy;
    int64_t Mxy;
    int32_t W;
    uint32_t pad;
};

// A cluster's fitted quad. Zeroed if no quad was found.
struct FittedQuad {
    LineFitPoint moments[4];
    uint16_t num_in_moments[4];
    // Corner positions within the cluster's angle-sorted points.
    uint16_t indices[4];
};

struct QuadCorners {
    float corners[4][2];
    uint32_t reversed_border;
    uint32_t blob_index;
};

struct FitQuadScanTile {
    int64_t m[5];
    int32_t W;
    uint32_t head;
};

#define FIT_QUAD_MIN_SCAN_TILE 1024u

// TODO document this better
#define FIT_QUAD_KEYS_BYTES(p) (uint64_t(p) * 16u)
#define FIT_QUAD_LINE_FIT_BYTES(p) (uint64_t(p) * 48u)
#define FIT_QUAD_ERRORS_OFFSET(p) (uint64_t(p) * 4u)
#define FIT_QUAD_FILTERED_OFFSET(p) (uint64_t(p) * 8u)

// These all share on scratch buffer.
#define FIT_QUAD_SCAN_TILES(p) ((p) / FIT_QUAD_MIN_SCAN_TILE + 2u)
#define FIT_QUAD_MAXIMA_COUNTS_OFFSET(p)                                       \
    (uint64_t(FIT_QUAD_SCAN_TILES(p)) * 48u)
#define FIT_QUAD_MAX_UNITS(p) (MAX_CLUSTERS + (p) / SORT_CHUNK)
#define FIT_QUAD_UNIT_MAP_OFFSET(p)                                            \
    (FIT_QUAD_MAXIMA_COUNTS_OFFSET(p) + uint64_t(MAX_CLUSTERS) * 4u)

#define FIT_QUAD_SCRATCH_BYTES(p)                                              \
    (FIT_QUAD_UNIT_MAP_OFFSET(p) + uint64_t(FIT_QUAD_MAX_UNITS(p)) * 4u)

struct FitQuadPush {
    // From cluster grouping. `points` is sorted in place.
    uint64_t points;
    uint64_t cluster_indexes;
    uint64_t extents;
    uint64_t totals;
    uint64_t sort_unit_starts;

    uint64_t grayscale;

    // Scratch
    // Holds 3 arrays, tile_sums, maxima_counts, unit_map
    uint64_t keys;
    uint64_t line_fits;
    uint64_t scratch;

    // Outputs.
    uint64_t quads;
    uint64_t corners;
    uint64_t corner_count;
    uint32_t width;
    uint32_t height;

    // The merge round.
    uint32_t round;
};

#endif
