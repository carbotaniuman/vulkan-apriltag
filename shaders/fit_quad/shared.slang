#ifndef FIT_QUAD_SHARED_SLANG
#define FIT_QUAD_SHARED_SLANG

#include "shared/fit_quad.h"

[ForceInline]
uint32_t capacity(FitQuadPush pc) {
    return MAX_POINTS_PER_PIXEL * pc.width * pc.height;
}
[ForceInline]
uint64_t *keys_a(FitQuadPush pc) { return (uint64_t *)pc.keys; }
[ForceInline]
uint64_t *keys_b(FitQuadPush pc) {
    return (uint64_t *)(pc.keys + uint64_t(capacity(pc)) * 8u);
}
[ForceInline]
LineFitPoint *line_fits(FitQuadPush pc) { return (LineFitPoint *)pc.line_fits; }
[ForceInline]
ClusterPoint *sorted_points(FitQuadPush pc) {
    return (ClusterPoint *)pc.points;
}
[ForceInline]
uint16_t *point_clusters(FitQuadPush pc) {
    return (uint16_t *)pc.cluster_indexes;
}
[ForceInline]
uint16_t *point_weights(FitQuadPush pc) { return (uint16_t *)pc.keys; }
[ForceInline]
uint32_t *maxima_list(FitQuadPush pc) { return (uint32_t *)pc.keys; }
[ForceInline]
float *errors(FitQuadPush pc) {
    return (float *)(pc.keys + FIT_QUAD_ERRORS_OFFSET(capacity(pc)));
}
[ForceInline]
float *filtered_errors(FitQuadPush pc) {
    return (float *)(pc.keys + FIT_QUAD_FILTERED_OFFSET(capacity(pc)));
}
[ForceInline]
FitQuadScanTile *tile_sums(FitQuadPush pc) {
    return (FitQuadScanTile *)pc.scratch;
}
[ForceInline]
uint32_t *maxima_counts(FitQuadPush pc) {
    return (uint32_t *)(pc.scratch +
                        FIT_QUAD_MAXIMA_COUNTS_OFFSET(capacity(pc)));
}
[ForceInline]
uint32_t *unit_map(FitQuadPush pc) {
    return (uint32_t *)(pc.scratch + FIT_QUAD_UNIT_MAP_OFFSET(capacity(pc)));
}

// The float bits are flipped so the floats sort as expected as part of the
// uint64_t.
uint64_t angle_key(float value, uint32_t index) {
    uint32_t bits = asuint(value);
    bits ^= (bits >> 31) != 0 ? 0xffffffffu : 0x80000000u;
    return (uint64_t(bits) << 32) | uint64_t(index);
}

[ForceInline]
Tuple<float, uint32_t> from_angle_key(uint64_t key) {
    uint32_t packed_coords = uint32_t(key);
    uint32_t bits = uint32_t(key >> 32);
    bits ^= (bits >> 31) != 0 ? 0x80000000u : 0xffffffffu;
    return makeTuple(asfloat(bits), packed_coords);
}

uint32_t point_word(ClusterPoint *points, uint32_t i) {
    return uint32_t(points[i].packed_x) | (uint32_t(points[i].packed_y) << 16);
}

// The window half-width AprilTag uses for a cluster of n points.
uint32_t window_half_width(uint32_t n) { return min(20u, n / 12); }

#endif
