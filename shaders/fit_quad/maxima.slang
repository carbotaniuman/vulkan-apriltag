// Step 3 of quad fitting
// This finds the corner candidates using a similar algorithm to AprilTag's.
#include "line_fit.slang"
#include "shared.slang"

// AprilTag's 7-elem filter
static const uint32_t FILTER_RADIUS = 3;
static const uint32_t FILTER_BITS[7] = { 0x3c360282u, 0x3e0a9555u, 0x3f1b4598u,
                                         0x3f800000u, 0x3f1b4598u, 0x3e0a9555u,
                                         0x3c360282u };

struct PointInCluster {
    uint32_t cluster;
    uint32_t start;
    uint32_t n;
    uint32_t ksz;
    uint32_t j;
};

PointInCluster point_in_cluster(uint32_t i) {
    const uint32_t cluster = uint32_t(point_clusters(pc)[i]);
    const ClusterExtents extent = ((ClusterExtents *)pc.extents)[cluster];
    const PointInCluster p = { cluster, extent.start, extent.count,
                               window_half_width(extent.count),
                               i - extent.start };
    return p;
}

uint32_t clustered_point_count() {
    return ((ClusterTotals *)pc.totals)->clustered_point_count;
}

[vk::push_constant]
ConstantBuffer<FitQuadPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 256;

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void window_errors(uint3 global_id: SV_DispatchThreadID) {
    float *errs = errors(pc);
    LineFitPoint *fits = line_fits(pc);

    const float nan = asfloat(0x7fc00000u);
    const uint32_t total = clustered_point_count();
    for (uint32_t i = global_id.x; i < total;
         i += WorkgroupCount().x * WG_SIZE) {
        const PointInCluster p = point_in_cluster(i);
        float error = nan;
        if (p.ksz >= 2) {
            uint32_t count;
            const WindowMoments moment = window_of(
                get_moment(fits, p.start, p.n, (p.j + p.n - p.ksz) % p.n,
                           (p.j + p.ksz) % p.n, count));
            error = window_line_error(moment, count);
        }
        errs[i] = error;
    }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void filter(uint3 global_id: SV_DispatchThreadID) {
    float *errs = errors(pc);
    float *filtered = filtered_errors(pc);

    const float nan = asfloat(0x7fc00000u);
    const uint32_t total = clustered_point_count();
    for (uint32_t i = global_id.x; i < total;
         i += WorkgroupCount().x * WG_SIZE) {
        const PointInCluster p = point_in_cluster(i);
        precise float acc = 0;
        if (p.ksz >= 2) {
            for (uint32_t t = 0; t < 7; t++) {
                acc += errs[p.start + (p.j + p.n + t - FILTER_RADIUS) % p.n] *
                       asfloat(FILTER_BITS[t]);
            }
        }
        filtered[i] = p.ksz >= 2 ? acc : nan;
    }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void maxima(uint3 global_id: SV_DispatchThreadID) {
    float *filtered = filtered_errors(pc);
    uint32_t *counts = maxima_counts(pc);
    uint32_t *found = maxima_list(pc);

    const uint32_t total = clustered_point_count();
    for (uint32_t i = global_id.x; i < total;
         i += WorkgroupCount().x * WG_SIZE) {
        const PointInCluster p = point_in_cluster(i);
        if (p.ksz < 2) {
            continue;
        }
        const float here = filtered[i];
        const float before = filtered[p.start + (p.j + p.n - 1) % p.n];
        const float after = filtered[p.start + (p.j + 1) % p.n];
        if (here > after && here > before) {
            uint32_t slot;
            InterlockedAdd(counts[p.cluster], 1u, slot);
            found[p.start + slot] = p.j;
        }
    }
}
