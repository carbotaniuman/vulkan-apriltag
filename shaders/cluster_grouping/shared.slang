#ifndef CLUSTER_GROUPING_SHARED_SLANG
#define CLUSTER_GROUPING_SHARED_SLANG

#include "shared/cluster_grouping.h"

static const uint32_t TAG_WIDTH = 4;
static const uint32_t MIN_CLUSTER_PIXELS = 5;
static const uint32_t MIN_CLUSTER_POINTS = 24;

float centre_x(uint32_t x_min, uint32_t x_max) {
    precise float c = float(x_min + x_max) * 0.5 + 0.05118;
    return c;
}

float centre_y(uint32_t y_min, uint32_t y_max) {
    precise float c = float(y_min + y_max) * 0.5 - 0.028581;
    return c;
}

// sum((x - cx) * gx + (y - cy) * gy)
float cluster_dot(ClusterAccumulator acc) {
    precise float cx_cy_contrib =
        float(int32_t(acc.x_min + acc.x_max) * acc.gx_sum +
              int32_t(acc.y_min + acc.y_max) * acc.gy_sum) *
        0.5;
    precise float gx_contrib = 0.05118 * float(acc.gx_sum);
    precise float gy_contrib = -0.028581 * float(acc.gy_sum);
    precise float dot =
        float(acc.pxgx_pygy_sum) - cx_cy_contrib - gx_contrib - gy_contrib;
    return dot;
}

// Order points by angle around (cx, cy).
float calc_theta(uint32_t px, uint32_t py, float cx, float cy) {
    precise float dx = float(px) - cx;
    precise float dy = float(py) - cy;
    const float quadrant =
        dx > 0 ? (dy > 0 ? 65536.0 : 0.0) : (dy > 0 ? 131072.0 : -65536.0);
    precise float theta =
        (dx > 0) == (dy > 0) ? quadrant + dy / dx : quadrant - dx / dy;
    return theta;
}

bool is_valid_blob(uint32_t slot, ClusterAccumulator acc) {
    if (slot == 0 || acc.count == 0) {
        return false;
    }
    const uint32_t max_cluster_pixels = 2 * (2 * pc.width + 2 * pc.height);
    if (acc.count < MIN_CLUSTER_PIXELS || acc.count < MIN_CLUSTER_POINTS ||
        acc.count > max_cluster_pixels) {
        return false;
    }
    if ((acc.x_max - acc.x_min) * (acc.y_max - acc.y_min) < TAG_WIDTH) {
        return false;
    }
    // Reversed borders are rejected.
    return !(cluster_dot(acc) < 0.0);
}

// Maybe could be replaced? TODO: carbo
uint32_t subgroup_id() {
    // clang-format off
    return spirv_asm {
        result:$$uint32_t = OpLoad builtin(SubgroupId:uint32_t);
    };
    // clang-format on
}

// Only safe with full subgroup extension, we otherwise can't get a subgroup_id
// to the LocalInvocationIndex.
uint32_t tile_index() {
    return subgroup_id() * WaveGetLaneCount() + WaveGetLaneIndex();
}

uint32_t sort_units_for(uint32_t points) {
    return points <= SORT_CHUNK ? 1 : (points + SORT_CHUNK - 1) / SORT_CHUNK;
}

#endif
