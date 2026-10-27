// Basically a hand written find_if
#include "shared.slang"

[vk::push_constant]
ConstantBuffer<ClusterGroupingPush> pc;

[vk::constant_id(0)]
const uint32_t TILE = 256;

groupshared uint3 subgroup_earlier[TILE];
groupshared uint3 subgroup_sums[TILE];

uint3 slot_values(uint32_t slot, ClusterAccumulator acc) {
    return is_valid_blob(slot, acc)
               ? uint3(1, acc.count, sort_units_for(acc.count))
               : uint3(0);
}

// First pass, a reduction to get the counts filtering out the valid subgroups.
[require(spvGroupNonUniformArithmetic)]
[require(spvGroupNonUniformBallot)]
[shader("compute")]
[numthreads(TILE, 1, 1)]
void reduce(uint3 group_id: SV_GroupID) {
    ClusterAccumulator *clusters = (ClusterAccumulator *)pc.clusters;
    uint32_t *scratch = (uint32_t *)pc.scratch;

    const uint32_t slot = group_id.x * TILE + tile_index();
    const uint32_t subgroup = subgroup_id();
    const uint32_t lanes = WaveGetLaneCount();
    const uint32_t tile_count = pc.cluster_slots / TILE;

    const uint3 sums = WaveActiveSum(slot_values(slot, clusters[slot]));
    if (WaveIsFirstLane()) {
        subgroup_sums[subgroup] = sums;
    }
    GroupMemoryBarrierWithGroupSync();

    // Add all the sums up, one subgroup per workgroup.
    if (subgroup == 0) {
        uint3 total = uint3(0);
        for (uint32_t s = WaveGetLaneIndex(); s < TILE / lanes; s += lanes) {
            total += subgroup_sums[s];
        }
        total = WaveActiveSum(total);

        if (WaveIsFirstLane()) {
            const uint32_t tile = group_id.x;
            scratch[tile] = total.x;
            scratch[tile_count + tile] = total.y;
            scratch[2 * tile_count + tile] = total.z;
        }
    }
}

// Second pass, do a scan to figure out where to put the tiles.
[require(spvGroupNonUniformArithmetic)]
[require(spvGroupNonUniformBallot)]
[shader("compute")]
[numthreads(TILE, 1, 1)]
void scan(uint3 group_id: SV_GroupID) {
    ClusterAccumulator *clusters = (ClusterAccumulator *)pc.clusters;
    uint32_t *scratch = (uint32_t *)pc.scratch;
    uint16_t *remap = (uint16_t *)pc.remap;
    ClusterBounds *bounds = (ClusterBounds *)pc.bounds;
    ClusterExtents *extents = (ClusterExtents *)pc.extents;
    uint32_t *sort_unit_starts = (uint32_t *)pc.sort_unit_starts;
    uint32_t *cursors = (uint32_t *)pc.cursors;

    const uint32_t tile = group_id.x;
    const uint32_t position = tile_index();
    const uint32_t slot = tile * TILE + position;
    const uint32_t tile_count = pc.cluster_slots / TILE;
    const uint32_t lanes = WaveGetLaneCount();
    const uint32_t lane = WaveGetLaneIndex();
    const uint32_t subgroup = subgroup_id();
    const uint32_t subgroup_count = TILE / lanes;

    // Calculate the total of every previous tile.
    uint3 earlier = uint3(0);
    for (uint32_t t = position; t < tile; t += TILE) {
        earlier += uint3(scratch[t], scratch[tile_count + t],
                         scratch[2 * tile_count + t]);
    }
    earlier = WaveActiveSum(earlier);

    const ClusterAccumulator acc = clusters[slot];
    const uint3 mine = slot_values(slot, acc);
    const bool keep = mine.x != 0;
    const uint32_t units = mine.z;

    const uint3 inclusive = WavePrefixSum(mine) + mine;
    const uint3 sums = WaveActiveSum(mine);
    if (WaveIsFirstLane()) {
        subgroup_earlier[subgroup] = earlier;
        subgroup_sums[subgroup] = sums;
    }
    GroupMemoryBarrierWithGroupSync();

    // The first subgroup turns the subgroups' sums into their offsets: the
    // earlier tiles' totals plus the sums of the subgroups before.
    if (subgroup == 0) {
        uint3 carry = uint3(0);
        for (uint32_t s = lane; s < subgroup_count; s += lanes) {
            carry += subgroup_earlier[s];
        }
        carry = WaveActiveSum(carry);
        for (uint32_t first = 0; first < subgroup_count; first += lanes) {
            const uint32_t s = first + lane;
            const uint3 value =
                s < subgroup_count ? subgroup_sums[s] : uint3(0);
            const uint3 before = WavePrefixSum(value);
            if (s < subgroup_count) {
                subgroup_sums[s] = carry + before;
            }
            carry += WaveActiveSum(value);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    const uint3 end = subgroup_sums[subgroup] + inclusive;
    const uint32_t cluster_end = end.x;
    const uint32_t point_end = end.y;
    const uint32_t unit_end = end.z;

    const uint32_t index = keep ? cluster_end - 1 : REJECTED_CLUSTER;
    remap[slot] = uint16_t(index);
    if (keep) {
        const uint32_t start = point_end - acc.count;
        bounds[index].x_min = uint16_t(acc.x_min);
        bounds[index].x_max = uint16_t(acc.x_max);
        bounds[index].y_min = uint16_t(acc.y_min);
        bounds[index].y_max = uint16_t(acc.y_max);
        bounds[index].gx_sum = acc.gx_sum;
        bounds[index].gy_sum = acc.gy_sum;
        bounds[index].pxgx_pygy_sum = acc.pxgx_pygy_sum;
        bounds[index].start = start;
        bounds[index].count = acc.count;
        const ClusterExtents extent = { start, acc.count };
        extents[index] = extent;
        sort_unit_starts[index] = unit_end - units;
        cursors[index] = 0;
    }

    if (slot == pc.cluster_slots - 1) {
        const ClusterTotals totals = { ((uint32_t *)pc.point_count)[0],
                                       cluster_end, point_end, unit_end };
        *(ClusterTotals *)pc.totals = totals;
    }
}
