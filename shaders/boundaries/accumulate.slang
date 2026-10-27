// Accumulates the clusters in a separate pass to reduce atomic usage.
// Each invocation processes RUN points.
#include "shared/boundaries.h"
#include "shared/segmentation.h"

[vk::push_constant]
ConstantBuffer<BoundariesPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 256;
static const uint32_t RUN = 8;

static const int32_t HALF_DX[4] = { 1, -1, 0, 1 };
static const int32_t HALF_DY[4] = { 0, 1, 1, 1 };

void add_to_global(ClusterAccumulator *clusters, uint32_t cluster,
                   ClusterAccumulator acc) {
    InterlockedMin(clusters[cluster].x_min, acc.x_min);
    InterlockedMax(clusters[cluster].x_max, acc.x_max);
    InterlockedMin(clusters[cluster].y_min, acc.y_min);
    InterlockedMax(clusters[cluster].y_max, acc.y_max);
    InterlockedAdd(clusters[cluster].gx_sum, acc.gx_sum);
    InterlockedAdd(clusters[cluster].gy_sum, acc.gy_sum);
    InterlockedAdd(clusters[cluster].count, acc.count);
    InterlockedAdd(clusters[cluster].pxgx_pygy_sum, acc.pxgx_pygy_sum);
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID) {
    ClusterAccumulator *clusters = (ClusterAccumulator *)pc.clusters;
    uint32_t *points = (uint32_t *)pc.points;
    uint16_t *cluster_indexes = (uint16_t *)pc.cluster_indexes;

    const uint32_t total = ((uint32_t *)pc.point_count)[0];
    const uint32_t stride = WorkgroupCount().x * WG_SIZE * RUN;
    for (uint32_t first = global_id.x * RUN; first < total; first += stride) {
        uint32_t pending_cluster = 0;
        ClusterAccumulator pending;
        for (uint32_t i = first; i < min(first + RUN, total); i++) {
            const uint32_t word = points[i];
            const uint32_t x = word & 0x3fff;
            const uint32_t half_pixel = (word >> 14) & 3;
            const uint32_t y = (word >> 16) & 0x3fff;
            const bool black_to_white = (word >> 31) != 0;
            const int32_t dx = HALF_DX[half_pixel];
            const int32_t dy = HALF_DY[half_pixel];

            // Ignore a cluster of 0.
            const uint32_t cluster = cluster_indexes[i];
            if (cluster == 0) {
                continue;
            }

            const uint32_t px = uint32_t(int32_t(2 * x) + dx);
            const uint32_t py = uint32_t(int32_t(2 * y) + dy);
            const int32_t gx = black_to_white ? dx : -dx;
            const int32_t gy = black_to_white ? dy : -dy;
            const int32_t dot = int32_t(px) * gx + int32_t(py) * gy;
            if (cluster != pending_cluster) {
                if (pending_cluster != 0) {
                    add_to_global(clusters, pending_cluster, pending);
                }
                pending_cluster = cluster;
                pending = { px, px, py, py, gx, gy, 1, dot };
            } else {
                pending.x_min = min(pending.x_min, px);
                pending.x_max = max(pending.x_max, px);
                pending.y_min = min(pending.y_min, py);
                pending.y_max = max(pending.y_max, py);
                pending.gx_sum += gx;
                pending.gy_sum += gy;
                pending.count += 1;
                pending.pxgx_pygy_sum += dot;
            }
        }
        if (pending_cluster != 0) {
            add_to_global(clusters, pending_cluster, pending);
        }
    }
}
