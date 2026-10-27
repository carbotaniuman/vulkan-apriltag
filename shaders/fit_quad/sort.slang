// Step 1 of quad fitting
// We sort each cluster's points by angle around the center.
#include "shared.slang"

// Unit map helpers.
uint32_t unit_cluster(uint32_t entry) { return entry >> 16; }
uint32_t unit_chunk(uint32_t entry) { return entry & 0xffffu; }

// Dual buffers for the sort to ping-pong between.
groupshared uint64_t local_a[SORT_CHUNK];
groupshared uint64_t local_b[SORT_CHUNK];

uint64_t local_read(bool in_b, uint32_t i) {
    return in_b ? local_b[i] : local_a[i];
}

void local_write(bool in_b, uint32_t i, uint64_t value) {
    if (in_b) {
        local_b[i] = value;
    } else {
        local_a[i] = value;
    }
}

static const uint32_t RANK_RUN = 16;

// Sorts `count` keys in `local_a`. Keys cannot be equal.
// Returns true if the results are in `local_b`.
bool local_rank_merge_sort(uint32_t lid, uint32_t count) {
    // Sort runs of 16 with a brute force rank sort.
    for (uint32_t i = lid; i < count; i += WG_SIZE) {
        const uint32_t start = i & ~(RANK_RUN - 1);
        const uint32_t end = min(start + RANK_RUN, count);
        const uint64_t key = local_a[i];
        uint32_t rank = 0;
        for (uint32_t j = start; j < end; j++) {
            rank += local_a[j] < key ? 1 : 0;
        }
        local_b[start + rank] = key;
    }

    // Merge parts of `run` length runs into `2 * run` length runs,
    // using the other local buffer as output space.
    bool in_b = true;
    for (uint32_t run = RANK_RUN; run < count; run <<= 1) {
        GroupMemoryBarrierWithGroupSync();
        for (uint32_t i = lid; i < count; i += WG_SIZE) {
            const uint64_t key = local_read(in_b, i);
            const uint32_t pair_start = i & ~(2 * run - 1);
            const bool left = i - pair_start < run;
            const uint32_t own_start = left ? pair_start : pair_start + run;
            const uint32_t other_start = left ? pair_start + run : pair_start;

            uint32_t lo = 0;
            uint32_t hi =
                other_start < count ? min(run, count - other_start) : 0;

            // Basic binary search, optimizing this didn't seem to do much.
            while (lo < hi) {
                // We will never get high enough to overflow this.
                const uint32_t mid = (lo + hi) / 2;
                if (local_read(in_b, other_start + mid) < key) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            local_write(!in_b, pair_start + (i - own_start) + lo, key);
        }
        in_b = !in_b;
    }
    GroupMemoryBarrierWithGroupSync();

    return in_b;
}

[vk::push_constant]
ConstantBuffer<FitQuadPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 256;

// Write the unit map, for every sort unit, write the cluster
// and its chunk within that cluster. This reduces chains
// of dependent loads.
[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void units(uint3 global_id: SV_DispatchThreadID) {
    uint32_t *unit_starts = (uint32_t *)pc.sort_unit_starts;
    uint32_t *map = unit_map(pc);

    const uint32_t unit_count = ((ClusterTotals *)pc.totals)->sort_unit_count;
    const uint32_t cluster_count = ((ClusterTotals *)pc.totals)->cluster_count;
    for (uint32_t cluster = global_id.x; cluster < cluster_count;
         cluster += WorkgroupCount().x * WG_SIZE) {
        const uint32_t first = unit_starts[cluster];
        const uint32_t end =
            cluster + 1 < cluster_count ? unit_starts[cluster + 1] : unit_count;
        for (uint32_t unit = first; unit < end; unit++) {
            map[unit] = (cluster << 16) | (unit - first);
        }
    }
}

// Sorts every sort unit by angle in shared memory, one unit per work-group.
// Gather whole clusters but leave chunks of larger clusters for later merging.
[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void sort(uint3 global_id: SV_DispatchThreadID,
          uint3 local_id: SV_GroupThreadID, uint3 group_id: SV_GroupID) {
    ClusterPoint *points = (ClusterPoint *)pc.points;
    ClusterExtents *extents = (ClusterExtents *)pc.extents;
    uint32_t *map = unit_map(pc);
    uint64_t *keys = keys_a(pc);
    ClusterPoint *sorted = sorted_points(pc);
    uint32_t *counts = maxima_counts(pc);

    const uint32_t lid = local_id.x;
    const uint32_t unit_count = ((ClusterTotals *)pc.totals)->sort_unit_count;
    if (global_id.x == 0) {
        ((uint32_t *)pc.corner_count)[0] = 0;
    }

    for (uint32_t unit = group_id.x; unit < unit_count;
         unit += WorkgroupCount().x) {
        const uint32_t entry = map[unit];
        const uint32_t cluster = unit_cluster(entry);
        const uint32_t chunk = unit_chunk(entry);
        const ClusterExtents extent = extents[cluster];
        if (lid == 0 && chunk == 0) {
            counts[cluster] = 0;
        }

        const uint32_t first = chunk * SORT_CHUNK;
        const uint32_t count = min(SORT_CHUNK, extent.count - first);
        for (uint32_t i = lid; i < count; i += WG_SIZE) {
            const uint32_t at = extent.start + first + i;
            local_a[i] = angle_key(points[at].slope, point_word(points, at));
        }
        GroupMemoryBarrierWithGroupSync();

        const bool in_b = local_rank_merge_sort(lid, count);

        if (extent.count <= SORT_CHUNK) {
            // We reuse the buffer so we need a device barrier.
            DeviceMemoryBarrierWithGroupSync();

            for (uint32_t i = lid; i < count; i += WG_SIZE) {
                const uint64_t key = local_read(in_b, i);
                const Tuple<float, uint32_t> point = from_angle_key(key);
                const float slope = point._0;
                const uint32_t packed_coords = point._1;
                const uint32_t at = extent.start + i;
                sorted[at].packed_x = uint16_t(packed_coords);
                sorted[at].packed_y = uint16_t(packed_coords >> 16);
                sorted[at].slope = slope;
            }
        } else {
            for (uint32_t i = lid; i < count; i += WG_SIZE) {
                keys[extent.start + first + i] = local_read(in_b, i);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

groupshared uint64_t local_keys[SORT_CHUNK];
groupshared uint32_t splits[2];

// Number of keys of run a (at a_start, a_len keys) among the first k outputs
// of merging it with run b. Keys are unique.
uint32_t merge_split_global(uint64_t *keys, uint32_t a_start, uint32_t a_len,
                            uint32_t b_start, uint32_t b_len, uint32_t k) {
    uint32_t lo = k > b_len ? k - b_len : 0;
    uint32_t hi = min(k, a_len);
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (keys[a_start + mid] < keys[b_start + k - mid - 1]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// The same for two runs side by side in local_keys: a at 0, b at a_len.
uint32_t merge_split_local(uint32_t a_len, uint32_t b_len, uint32_t k) {
    uint32_t lo = k > b_len ? k - b_len : 0;
    uint32_t hi = min(k, a_len);
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (local_keys[mid] < local_keys[a_len + k - mid - 1]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void merge(uint3 local_id: SV_GroupThreadID, uint3 group_id: SV_GroupID) {
    ClusterExtents *extents = (ClusterExtents *)pc.extents;
    uint32_t *map = unit_map(pc);
    uint64_t *src = pc.round % 2 == 0 ? keys_a(pc) : keys_b(pc);
    uint64_t *dst = pc.round % 2 == 0 ? keys_b(pc) : keys_a(pc);

    const uint32_t lid = local_id.x;
    const uint32_t unit_count = ((ClusterTotals *)pc.totals)->sort_unit_count;
    const uint32_t run = SORT_CHUNK << pc.round;

    // Loop over sort units
    for (uint32_t unit = group_id.x; unit < unit_count;
         unit += WorkgroupCount().x) {
        const uint32_t entry = map[unit];
        const uint32_t cluster = unit_cluster(entry);
        const uint32_t chunk = unit_chunk(entry);
        const ClusterExtents extent = extents[cluster];
        const uint32_t n = extent.count;
        if (n <= run) {
            // This is a small cluster or already sorted run.
            continue;
        }

        // Find the pair of runs this slice belongs to
        const uint32_t pair_start =
            (chunk * SORT_CHUNK) / (2 * run) * (2 * run);
        const uint32_t left_len = min(run, n - pair_start);
        const uint32_t right_len = min(run, n - pair_start - left_len);
        const uint32_t left = extent.start + pair_start;
        const uint32_t right = left + left_len;
        const uint32_t out_begin = chunk * SORT_CHUNK - pair_start;
        const uint32_t out_end =
            min(out_begin + SORT_CHUNK, left_len + right_len);

        if (lid < 2) {
            splits[lid] =
                merge_split_global(src, left, left_len, right, right_len,
                                   lid == 0 ? out_begin : out_end);
        }
        GroupMemoryBarrierWithGroupSync();

        const uint32_t left_begin = splits[0];
        const uint32_t left_end = splits[1];
        const uint32_t right_begin = out_begin - left_begin;
        const uint32_t right_end = out_end - left_end;
        const uint32_t left_count = left_end - left_begin;
        const uint32_t right_count = right_end - right_begin;

        for (uint32_t i = lid; i < left_count; i += WG_SIZE) {
            local_keys[i] = src[left + left_begin + i];
        }
        for (uint32_t i = lid; i < right_count; i += WG_SIZE) {
            local_keys[left_count + i] = src[right + right_begin + i];
        }
        GroupMemoryBarrierWithGroupSync();

        // The actual merge.
        for (uint32_t k = lid; k < left_count + right_count; k += WG_SIZE) {
            const uint32_t from_left =
                merge_split_local(left_count, right_count, k);
            const uint32_t from_right = k - from_left;
            const bool take_left =
                from_right >= right_count ||
                (from_left < left_count &&
                 local_keys[from_left] < local_keys[left_count + from_right]);
            dst[left + out_begin + k] =
                take_left ? local_keys[from_left]
                          : local_keys[left_count + from_right];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// How many merge rounds the merge pass had to do.
uint32_t merge_rounds(uint32_t n) {
    return n <= SORT_CHUNK ? 0
                           : firstbithigh(n - 1) + 1 - firstbithigh(SORT_CHUNK);
}

// Gathers the sorted points of clusters larger than SORT_CHUNK.
[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void gather(uint3 global_id: SV_DispatchThreadID) {
    ClusterExtents *extents = (ClusterExtents *)pc.extents;
    ClusterPoint *sorted = sorted_points(pc);
    uint16_t *clusters = point_clusters(pc);

    const uint32_t total = ((ClusterTotals *)pc.totals)->clustered_point_count;
    const uint32_t stride = WorkgroupCount().x * WG_SIZE;
    for (uint32_t i = global_id.x; i < total; i += stride) {
        const ClusterExtents extent = extents[uint32_t(clusters[i])];
        if (extent.count <= SORT_CHUNK) {
            continue;
        }
        uint64_t *keys =
            merge_rounds(extent.count) % 2 == 0 ? keys_a(pc) : keys_b(pc);
        const uint64_t key = keys[i];
        const Tuple<float, uint32_t> point = from_angle_key(key);
        const float slope = point._0;
        const uint32_t packed_coords = point._1;
        sorted[i].packed_x = uint16_t(packed_coords);
        sorted[i].packed_y = uint16_t(packed_coords >> 16);
        sorted[i].slope = slope;
    }
}
