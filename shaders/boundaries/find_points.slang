// Finds the boundary points and assigns each its cluster,
// Each invocation processes `ROWS` rows of one column.
#include "lookup.slang"
#include "shared/boundaries.h"
#include "shared/segmentation.h"

[vk::push_constant]
ConstantBuffer<BoundariesPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 256;
static const uint32_t ROWS = 8;

// Components smaller than this are not considered for boundaries.
static const uint32_t MIN_COMPONENT_SIZE = 25;

static const uint32_t HALF_RIGHT = 0;
static const uint32_t HALF_BOTTOM_LEFT = 1;
static const uint32_t HALF_BOTTOM = 2;
static const uint32_t HALF_BOTTOM_RIGHT = 3;
static const uint32_t COLOR_DIRECTION_BIT = 1u << 15;

groupshared uint32_t group_point_count;
groupshared uint32_t group_output_base;

static uint32_t *labels;
static uint32_t *sizes;

// Load the label with bounds checking and size checks.
uint32_t load_label(uint32_t x, uint32_t y) {
    if (x >= pc.width || y >= pc.height) {
        return 0;
    }
    const uint32_t label = labels[y * pc.width + x];
    if (label != 0 && sizes[label & LABEL_VALUE_MASK] < MIN_COMPONENT_SIZE) {
        return 0;
    }
    return label;
}

// Each invocations tests 8 points for connectivity, doing 4 tests per.
// Each of these takes a bit, so 4 * 8 = 32.
static uint32_t found_mask = 0;
// Set for points that go black to white.
static uint32_t direction_mask = 0;
// This is not stored `first_key_mask` and `second_key_mask` as an optimization,
// we assume cache hits. If a point is in neither mask, it's key is the first
// below.
static uint32_t second_key_mask = 0;
static uint32_t other_key_mask = 0;
// The cached keys.
static uint32_t key_count = 0;
static uint32_t key_lo[2];
static uint32_t key_hi[2];

void test(uint32_t bit, uint32_t local_label, uint32_t test_label) {
    const uint32_t local_color = local_label & LABEL_PIXEL_MASK;
    const uint32_t test_color = test_label & LABEL_PIXEL_MASK;

    // Ignored pixels and small components have label 0 and never form
    // boundaries.
    if (test_label == 0 || test_color == local_color) {
        return;
    }

    found_mask |= 1u << bit;
    if (test_color > local_color) {
        direction_mask |= 1u << bit;
    }
    const uint32_t a = local_label & LABEL_VALUE_MASK;
    const uint32_t b = test_label & LABEL_VALUE_MASK;
    const uint32_t lo = min(a, b);
    const uint32_t hi = max(a, b);
    if (key_count > 0 && lo == key_lo[0] && hi == key_hi[0]) {
        return;
    }
    if (key_count > 1 && lo == key_lo[1] && hi == key_hi[1]) {
        second_key_mask |= 1u << bit;
    } else if (key_count == 0) {
        // The compiler will spill this with variable indices.
        key_lo[0] = lo;
        key_hi[0] = hi;
        key_count = 1;
    } else if (key_count == 1) {
        key_lo[1] = lo;
        key_hi[1] = hi;
        key_count = 2;
        second_key_mask |= 1u << bit;
    } else {
        other_key_mask |= 1u << bit;
    }
}

[require(spvGroupNonUniformArithmetic)]
[require(spvGroupNonUniformBallot)]
[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(uint3 group_id: SV_GroupID, uint32_t local_index: SV_GroupIndex) {
    labels = (uint32_t *)pc.labels;
    sizes = (uint32_t *)pc.sizes;
    uint32_t *points = (uint32_t *)pc.points;
    uint16_t *cluster_indexes = (uint16_t *)pc.cluster_indexes;
    Atomic<uint32_t> *point_count = (Atomic<uint32_t> *)pc.point_count;
    Atomic<uint32_t> *key_words = (Atomic<uint32_t> *)pc.keys;

    const uint32_t x = group_id.x * WG_SIZE + local_index;
    const uint32_t y0 = group_id.y * ROWS;

    if (local_index == 0) {
        group_point_count = 0;
    }

    uint32_t here = load_label(x, y0);
    uint32_t left = load_label(x - 1, y0);
    uint32_t right = load_label(x + 1, y0);
    [ForceUnroll]
    for (uint32_t r = 0; r < ROWS; r++) {
        const uint32_t below = load_label(x, y0 + r + 1);
        const uint32_t below_left = load_label(x - 1, y0 + r + 1);
        const uint32_t below_right = load_label(x + 1, y0 + r + 1);

        if (here != 0) {
            test(4 * r + HALF_RIGHT, here, right);
            // `BOTTOM_RIGHT` on the left pixel and `BOTTOM_LEFT`
            // on our pixel will result in duplicates - let's check
            // for this case and ignore them. For some reason I can't
            // understand, this is really important and results in 2-wide
            // borders otherwise, which messes with the quad detection.
            const bool would_duplicate =
                (left & LABEL_PIXEL_MASK) != (below & LABEL_PIXEL_MASK) &&
                left != 0 && below != 0;
            if (!would_duplicate) {
                test(4 * r + HALF_BOTTOM_LEFT, here, below_left);
            }
            test(4 * r + HALF_BOTTOM, here, below);
            test(4 * r + HALF_BOTTOM_RIGHT, here, below_right);
        }
        here = below;
        left = below_left;
        right = below_right;
    }

    // Reserve the subgroup's range within the group, then the group's
    // range in the output.
    const uint32_t found_count = countbits(found_mask);
    const uint32_t lane_offset = WavePrefixSum(found_count);
    const uint32_t subgroup_count = WaveActiveSum(found_count);
    GroupMemoryBarrierWithGroupSync();
    uint32_t subgroup_offset = 0;
    if (WaveIsFirstLane() && subgroup_count > 0) {
        InterlockedAdd(group_point_count, subgroup_count, subgroup_offset);
    }
    subgroup_offset = WaveReadLaneFirst(subgroup_offset);
    GroupMemoryBarrierWithGroupSync();
    if (local_index == 0) {
        group_output_base =
            group_point_count > 0 ? point_count[0].add(group_point_count) : 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint32_t clusters[2] = { 0, 0 };
    [loop]
    for (uint32_t i = 0; i < key_count; i++) {
        const uint32_t lo = i == 0 ? key_lo[0] : key_lo[1];
        const uint32_t hi = i == 0 ? key_hi[0] : key_hi[1];
        const uint32_t cluster = lookup(key_words, pc.cluster_slots,
                                        (uint64_t(lo) << 32) | uint64_t(hi));
        clusters[0] = i == 0 ? cluster : clusters[0];
        clusters[1] = i == 1 ? cluster : clusters[1];
    }

    // Write the points in bit order (by row, then test).
    uint32_t output_index = group_output_base + subgroup_offset + lane_offset;
    uint32_t remaining = found_mask;
    [loop]
    while (remaining != 0) {
        const uint32_t bit = firstbitlow(remaining);
        remaining &= remaining - 1;
        const uint32_t y = y0 + bit / 4;
        const uint32_t half_pixel = bit % 4;
        const uint32_t packed_x = x | (half_pixel << 14);
        const uint32_t packed_y =
            (direction_mask & (1u << bit)) != 0 ? (COLOR_DIRECTION_BIT | y) : y;
        uint32_t cluster =
            (second_key_mask & (1u << bit)) != 0 ? clusters[1] : clusters[0];
        if ((other_key_mask & (1u << bit)) != 0) {
            // Another label pair, fall back to loading this from memory.
            const int32_t dx = half_pixel == HALF_RIGHT
                                   ? 1
                                   : int32_t(half_pixel) - int32_t(HALF_BOTTOM);
            const uint32_t dy = half_pixel == HALF_RIGHT ? 0 : 1;
            const uint32_t a = labels[y * pc.width + x] & LABEL_VALUE_MASK;
            const uint32_t b =
                labels[(y + dy) * pc.width + uint32_t(int32_t(x) + dx)] &
                LABEL_VALUE_MASK;
            cluster = lookup(key_words, pc.cluster_slots,
                             (uint64_t(min(a, b)) << 32) | uint64_t(max(a, b)));
        }

        // Our table is full, drop the point. If this happens, we will probably
        // see massive slowdowns, so this is good to keep track.
        if (cluster == 0) {
            ((Atomic<uint32_t> *)pc.dropped_points) [0].add(1);
        }
        points[output_index] = packed_x | (packed_y << 16);
        cluster_indexes[output_index] = uint16_t(cluster);
        output_index++;
    }
}
