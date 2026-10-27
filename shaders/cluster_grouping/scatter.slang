// Writes the points of valid clusters out.
#include "shared.slang"

[vk::push_constant]
ConstantBuffer<ClusterGroupingPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 256;

static const int32_t HALF_DX[4] = { 1, -1, 0, 1 };
static const int32_t HALF_DY[4] = { 0, 1, 1, 1 };

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID) {
    uint32_t *points = (uint32_t *)pc.points;
    uint16_t *cluster_indexes = (uint16_t *)pc.cluster_indexes;
    uint16_t *remap = (uint16_t *)pc.remap;
    ClusterBounds *bounds = (ClusterBounds *)pc.bounds;
    Atomic<uint32_t> *cursors = (Atomic<uint32_t> *)pc.cursors;
    ClusterPoint *out_points = (ClusterPoint *)pc.out_points;
    uint16_t *out_cluster_indexes = (uint16_t *)pc.out_cluster_indexes;

    const uint32_t point_count = ((ClusterTotals *)pc.totals)->point_count;
    const uint32_t stride = WorkgroupCount().x * WG_SIZE;
    for (uint32_t i = global_id.x; i < point_count; i += stride) {
        const uint32_t index = uint32_t(remap[uint32_t(cluster_indexes[i])]);
        if (index == REJECTED_CLUSTER) {
            continue;
        }

        const float cx = centre_x(uint32_t(bounds[index].x_min),
                                  uint32_t(bounds[index].x_max));
        const float cy = centre_y(uint32_t(bounds[index].y_min),
                                  uint32_t(bounds[index].y_max));
        const uint32_t position = bounds[index].start + cursors[index].add(1);

        const uint32_t word = points[i];
        const uint32_t half_pixel = (word >> 14) & 3;
        const uint32_t px = 2 * (word & 0x3fff) + HALF_DX[half_pixel];
        const uint32_t py = 2 * ((word >> 16) & 0x3fff) + HALF_DY[half_pixel];

        out_points[position].packed_x = uint16_t(word);
        out_points[position].packed_y = uint16_t(word >> 16);
        out_points[position].slope = calc_theta(px, py, cx, cy);
        out_cluster_indexes[position] = uint16_t(index);
    }
}
