#include "shared/boundaries.h"
#include "shared/segmentation.h"

[vk::push_constant]
ConstantBuffer<BoundariesPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 1;

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID) {
    uint64_t *keys = (uint64_t *)pc.keys;
    ClusterAccumulator *clusters = (ClusterAccumulator *)pc.clusters;

    const uint32_t i = global_id.x;
    if (i >= pc.cluster_slots) {
        return;
    }
    keys[i] = 0;
    const ClusterAccumulator empty = { 0xffffffffu, 0, 0xffffffffu, 0,
                                       0,           0, 0,           0 };
    clusters[i] = empty;
    if (i == 0) {
        ((uint32_t *)pc.point_count)[0] = 0;
        ((uint32_t *)pc.dropped_points)[0] = 0;
    }
}
