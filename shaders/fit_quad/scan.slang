// Step 2 of quad fitting
// Prefix sums of the line-fit momments within each cluster, using
// a segmented reduce-then-scan.
#include "line_fit.slang"
#include "shared.slang"

static const uint32_t POINTS_PER_INVOCATION = 8;
static const uint32_t SCAN_TILE = WG_SIZE * POINTS_PER_INVOCATION;

FitQuadScanTile segmented_identity() {
    const FitQuadScanTile identity = { { 0, 0, 0, 0, 0 }, 0, 0 };
    return identity;
}

FitQuadScanTile to_segmented(LineFitPoint p, bool head) {
    const FitQuadScanTile tile = { { p.Mx, p.My, p.Mxx, p.Myy, p.Mxy },
                                   p.W,
                                   head ? 1u : 0u };
    return tile;
}

FitQuadScanTile combine(FitQuadScanTile a, FitQuadScanTile b) {
    if (b.head != 0) {
        return b;
    }
    for (uint32_t c = 0; c < 5; c++) {
        b.m[c] += a.m[c];
    }
    b.W += a.W;
    b.head = a.head;
    return b;
}

LineFitPoint moments_of(FitQuadScanTile s) {
    return LineFitPoint(s.m[0], s.m[1], s.m[2], s.m[3], s.m[4], s.W, 0);
}

groupshared FitQuadScanTile scan_values[WG_SIZE];

// Combines `scan_values` into one, the result is in `scan_values[0]`.
void group_combine(uint32_t lid) {
    for (uint32_t stride = 1; stride < WG_SIZE; stride *= 2) {
        GroupMemoryBarrierWithGroupSync();
        if (lid % (2 * stride) == 0) {
            scan_values[lid] =
                combine(scan_values[lid], scan_values[lid + stride]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// Hillis-Steele inclusive scan on `scan_values`.
void group_scan(uint32_t lid) {
    for (uint32_t stride = 1; stride < WG_SIZE; stride *= 2) {
        GroupMemoryBarrierWithGroupSync();
        const FitQuadScanTile v =
            lid >= stride ? combine(scan_values[lid - stride], scan_values[lid])
                          : scan_values[lid];
        GroupMemoryBarrierWithGroupSync();
        scan_values[lid] = v;
    }
    GroupMemoryBarrierWithGroupSync();
}

FitQuadScanTile load_point(uint32_t i, uint32_t total, bool first_pass) {
    FitQuadScanTile v = segmented_identity();
    if (i < total) {
        const uint32_t cluster = uint32_t(point_clusters(pc)[i]);
        const bool head = ((ClusterExtents *)pc.extents)[cluster].start == i;
        const uint32_t word = point_word(sorted_points(pc), i);
        int32_t w;
        if (first_pass) {
            w = point_weight(word, (uint32_t *)pc.grayscale);
            point_weights(pc) [i] = uint16_t(w);
        } else {
            w = int32_t(point_weights(pc)[i]);
        }
        v = to_segmented(initial_line_fit(word, w), head);
    }
    return v;
}

// Each invocation combines POINTS_PER_INVOCATION consecutive points.
FitQuadScanTile reduce_points(uint32_t first, uint32_t total, bool first_pass) {
    FitQuadScanTile acc = segmented_identity();
    for (uint32_t k = 0; k < POINTS_PER_INVOCATION; k++) {
        acc = combine(acc, load_point(first + k, total, first_pass));
    }
    return acc;
}

[vk::push_constant]
ConstantBuffer<FitQuadPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 256;
// Reduce each tile of SCAN_TILE points to a combained value.
[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void scan_reduce(uint3 local_id: SV_GroupThreadID, uint3 group_id: SV_GroupID) {
    FitQuadScanTile *tiles = tile_sums(pc);
    const uint32_t lid = local_id.x;
    const uint32_t total = ((ClusterTotals *)pc.totals)->clustered_point_count;
    const uint32_t tile_count = (total + SCAN_TILE - 1) / SCAN_TILE;
    for (uint32_t tile = group_id.x; tile < tile_count;
         tile += WorkgroupCount().x) {
        scan_values[lid] = reduce_points(
            tile * SCAN_TILE + lid * POINTS_PER_INVOCATION, total, true);
        group_combine(lid);
        if (lid == 0) {
            tiles[tile] = scan_values[0];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

groupshared FitQuadScanTile carry;

// Do a prefix sum on every tile's combined value in-place.
[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void scan_tiles(uint3 local_id: SV_GroupThreadID) {
    FitQuadScanTile *tiles = tile_sums(pc);
    const uint32_t lid = local_id.x;
    const uint32_t total = ((ClusterTotals *)pc.totals)->clustered_point_count;
    const uint32_t tile_count = (total + SCAN_TILE - 1) / SCAN_TILE;
    if (lid == 0) {
        carry = segmented_identity();
    }
    for (uint32_t base = 0; base < tile_count; base += WG_SIZE) {
        const uint32_t tile = base + lid;
        scan_values[lid] =
            tile < tile_count ? tiles[tile] : segmented_identity();
        group_scan(lid);
        if (tile < tile_count) {
            tiles[tile] = combine(carry, lid == 0 ? segmented_identity()
                                                  : scan_values[lid - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        if (lid == 0) {
            carry = combine(carry, scan_values[WG_SIZE - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// Combine the combined value with the one points and write the output.
[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void scan(uint3 local_id: SV_GroupThreadID, uint3 group_id: SV_GroupID) {
    FitQuadScanTile *tiles = tile_sums(pc);
    LineFitPoint *fits = line_fits(pc);
    const uint32_t lid = local_id.x;
    const uint32_t total = ((ClusterTotals *)pc.totals)->clustered_point_count;
    const uint32_t tile_count = (total + SCAN_TILE - 1) / SCAN_TILE;
    for (uint32_t tile = group_id.x; tile < tile_count;
         tile += WorkgroupCount().x) {
        const FitQuadScanTile carry = tiles[tile];

        const uint32_t first = tile * SCAN_TILE + lid * POINTS_PER_INVOCATION;
        scan_values[lid] = reduce_points(first, total, false);
        group_scan(lid);

        FitQuadScanTile acc = combine(carry, lid == 0 ? segmented_identity()
                                                      : scan_values[lid - 1]);
        for (uint32_t k = 0; k < POINTS_PER_INVOCATION; k++) {
            if (first + k < total) {
                acc = combine(acc, load_point(first + k, total, false));
                fits[first + k] = moments_of(acc);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
