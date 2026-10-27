// Step 4 of quad fitting
// It's time to do the actual quad search.
#include "line_fit.slang"
#include "shared.slang"

static const float COS_CRITICAL_RAD = 0.98480775301220802; // cos(10 degrees)
static const float MIN_TAG_WIDTH = 8;

// The filtered errors of a cluster's maxima, for lists of up to SORT_CHUNK.
groupshared float maxima_errors[SORT_CHUNK];
groupshared uint32_t kept[MAX_MAXIMA];
groupshared uint32_t kept_count;
groupshared float pair_error[MAX_MAXIMA * MAX_MAXIMA];
groupshared float pair_mse[MAX_MAXIMA * MAX_MAXIMA];

// Fitted line of each ordered pair (point x, point y, normal x, normal y).
groupshared float4 pair_line[MAX_MAXIMA * MAX_MAXIMA];

// binomial(MAX_MAXIMA, 4) table computed once per workgroup.
static const uint32_t COMBINATIONS = 210;
groupshared uint32_t combination_table[COMBINATIONS];

groupshared uint32_t best_error_bits;
groupshared uint32_t best_rank_shared;

void keep(uint32_t position) {
    uint32_t slot;
    InterlockedAdd(kept_count, 1u, slot);
    kept[slot] = position;
}

// n choose k for k <= 4, in the combinatorial number system.
// Hardcoding the divisor is used to optimize the division out.
uint32_t binomial(uint32_t n, uint32_t k) {
    if (n < k) {
        return 0;
    }
    switch (k) {
    case 0:
        return 1;
    case 1:
        return n;
    case 2:
        return n * (n - 1) / 2;
    case 3:
        return n * (n - 1) * (n - 2) / 6;
    default:
        return n * (n - 1) * (n - 2) * (n - 3) / 24;
    }
}

void decode_combination(uint32_t number, out uint32_t position[4]) {
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t element = 0;
        for (uint32_t e = 0; e < 16; e++) {
            if (binomial(e, 4 - i) <= number) {
                element = e;
            }
        }
        number -= binomial(element, 4 - i);
        position[3 - i] = element;
    }
}

bool corners_from_lines(float4 lines[4], out float2 corners[4]) {
    for (uint32_t i = 0; i < 4; i++) {
        // Intersect lines i and (i + 1) & 3: p0 + lambda0 * u0 = p1 +
        // lambda1 * u1, where u are the line directions (perpendicular to
        // the stored normals).
        const uint32_t j = (i + 1) & 3;
        precise float a00 = lines[i].w;
        precise float a01 = -lines[j].w;
        precise float a10 = -lines[i].z;
        precise float a11 = lines[j].z;
        precise float b0 = -lines[i].x + lines[j].x;
        precise float b1 = -lines[i].y + lines[j].y;

        // Written so that NaN (from an empty or degenerate quad) fails.
        precise float det = a00 * a11 - a10 * a01;
        if (!(abs(det) >= 0.001)) {
            return false;
        }
        precise float w00 = a11 / det;
        precise float w01 = -a01 / det;
        precise float l0 = w00 * b0 + w01 * b1;
        precise float cx = lines[i].x + l0 * a00;
        precise float cy = lines[i].y + l0 * a10;
        corners[i] = float2(cx, cy);
    }

    // Area from the two triangles (0, 1, 2) and (2, 3, 0), by Heron's
    // formula.
    precise float area = 0;
    const uint32_t triangles[6] = { 0, 1, 2, 2, 3, 0 };
    for (uint32_t t = 0; t < 2; t++) {
        float length[3];
        for (uint32_t k = 0; k < 3; k++) {
            const float2 a = corners[triangles[t * 3 + k]];
            const float2 c = corners[triangles[t * 3 + (k + 1) % 3]];
            precise float dx = c.x - a.x;
            precise float dy = c.y - a.y;
            precise float l = sqrt(dx * dx + dy * dy);
            length[k] = l;
        }
        precise float p = (length[0] + length[1] + length[2]) / 2;
        precise float s =
            sqrt(p * (p - length[0]) * (p - length[1]) * (p - length[2]));
        area += s;
    }
    if (!(area >= 0.95 * MIN_TAG_WIDTH * MIN_TAG_WIDTH)) {
        return false;
    }

    // Every corner angle must be away from 0 and 180 degrees, and turn in
    // the same direction.
    for (uint32_t i = 0; i < 4; i++) {
        const float2 c0 = corners[i];
        const float2 c1 = corners[(i + 1) & 3];
        const float2 c2 = corners[(i + 2) & 3];
        precise float dx1 = c1.x - c0.x;
        precise float dy1 = c1.y - c0.y;
        precise float dx2 = c2.x - c1.x;
        precise float dy2 = c2.y - c1.y;
        precise float cos_dtheta =
            (dx1 * dx2 + dy1 * dy2) /
            sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
        precise float turn_a = dx1 * dy2;
        precise float turn_b = dy1 * dx2;
        if (!(abs(cos_dtheta) <= COS_CRITICAL_RAD) || !(turn_a >= turn_b)) {
            return false;
        }
    }
    return true;
}

void clear_quad(FittedQuad *quads, uint32_t cluster) {
    for (uint32_t side = 0; side < 4; side++) {
        quads[cluster].moments[side] = LineFitPoint(0, 0, 0, 0, 0, 0, 0);
        quads[cluster].num_in_moments[side] = 0;
        quads[cluster].indices[side] = 0;
    }
}

[vk::push_constant]
ConstantBuffer<FitQuadPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 256;

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void search(uint3 local_id: SV_GroupThreadID, uint3 group_id: SV_GroupID) {
    ClusterExtents *extents = (ClusterExtents *)pc.extents;
    LineFitPoint *fits = line_fits(pc);
    uint32_t *found = maxima_list(pc);
    float *filtered = filtered_errors(pc);
    uint32_t *counts = maxima_counts(pc);
    FittedQuad *quads = (FittedQuad *)pc.quads;
    QuadCorners *corners_out = (QuadCorners *)pc.corners;

    const uint32_t lid = local_id.x;
    const uint32_t cluster_count = ((ClusterTotals *)pc.totals)->cluster_count;

    for (uint32_t number = lid; number < COMBINATIONS; number += WG_SIZE) {
        uint32_t position[4];
        decode_combination(number, position);
        combination_table[number] = position[0] | (position[1] << 4) |
                                    (position[2] << 8) | (position[3] << 12);
    }

    // The cluster loop's barriers order these writes before the reads in the
    // combination loop.
    for (uint32_t cluster = group_id.x; cluster < cluster_count;
         cluster += WorkgroupCount().x) {
        const ClusterExtents extent = extents[cluster];
        const uint32_t n = extent.count;
        const uint32_t start = extent.start;
        const uint32_t maxima_count = counts[cluster];

        if (window_half_width(n) < 2 || maxima_count < 4) {
            if (lid == 0) {
                clear_quad(quads, cluster);
            }
            continue;
        }

        // Like AprilTag, keep a maximum when it is strictly above the
        // (MAX_MAXIMA + 1)th largest error counting ties.
        if (lid == 0) {
            kept_count = 0;
        }
        if (maxima_count <= MAX_MAXIMA) {
            GroupMemoryBarrierWithGroupSync();
            if (lid < maxima_count) {
                keep(found[start + lid]);
            }
        } else {
            const bool local = maxima_count <= SORT_CHUNK;
            if (local) {
                for (uint32_t j = lid; j < maxima_count; j += WG_SIZE) {
                    maxima_errors[j] = filtered[start + found[start + j]];
                }
            }
            GroupMemoryBarrierWithGroupSync();

            for (uint32_t j = lid; j < maxima_count; j += WG_SIZE) {
                const float error = local ? maxima_errors[j]
                                          : filtered[start + found[start + j]];
                uint32_t at_least = 0;
                for (uint32_t k = 0; k < maxima_count && at_least <= MAX_MAXIMA;
                     k++) {
                    const float other =
                        local ? maxima_errors[k]
                              : filtered[start + found[start + k]];
                    at_least += other >= error ? 1 : 0;
                }
                if (at_least <= MAX_MAXIMA) {
                    keep(found[start + j]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        const uint32_t kept_maxima = kept_count;
        if (lid == 0) {
            // Back into point order.
            for (uint32_t a = 1; a < kept_maxima; a++) {
                const uint32_t value = kept[a];
                uint32_t c = a;
                while (c > 0 && kept[c - 1] > value) {
                    kept[c] = kept[c - 1];
                    c--;
                }
                kept[c] = value;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint32_t pair = lid; pair < MAX_MAXIMA * MAX_MAXIMA;
             pair += WG_SIZE) {
            const uint32_t a = pair / MAX_MAXIMA;
            const uint32_t c = pair % MAX_MAXIMA;
            if (a != c && a < kept_maxima && c < kept_maxima) {
                uint32_t count;
                const LineFitPoint moment =
                    get_moment(fits, start, n, kept[a], kept[c], count);
                float4 line;
                float err;
                float mse;
                fit_line(moment, count, line, err, mse);
                pair_error[pair] = err;
                pair_mse[pair] = mse;
                pair_line[pair] = line;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Combinations of 4 kept maxima.
        const uint32_t combinations =
            kept_maxima >= 4 ? binomial(kept_maxima, 4) : 0;
        float my_error = asfloat(0x7f800000u); // infinity
        uint32_t my_rank = 0xffffffffu;
        uint32_t my_position[4] = { 0, 0, 0, 0 };
        for (uint32_t combination = lid; combination < combinations;
             combination += WG_SIZE) {
            const uint32_t packed = combination_table[combination];
            const uint32_t position[4] = { packed & 0xf, (packed >> 4) & 0xf,
                                           (packed >> 8) & 0xf, packed >> 12 };
            const uint32_t side01 = position[0] * MAX_MAXIMA + position[1];
            const uint32_t side12 = position[1] * MAX_MAXIMA + position[2];
            const uint32_t side23 = position[2] * MAX_MAXIMA + position[3];
            const uint32_t side30 = position[3] * MAX_MAXIMA + position[0];

            precise float total_error = 0;
            total_error += pair_error[side01];
            total_error += pair_error[side12];
            total_error += pair_error[side23];
            total_error += pair_error[side30];

            precise float dot = pair_line[side01].z * pair_line[side12].z +
                                pair_line[side01].w * pair_line[side12].w;
            const bool valid = pair_mse[side01] <= MAX_LINE_FIT_MSE &&
                               pair_mse[side12] <= MAX_LINE_FIT_MSE &&
                               abs(dot) <= COS_CRITICAL_RAD &&
                               pair_mse[side23] <= MAX_LINE_FIT_MSE &&
                               pair_mse[side30] <= MAX_LINE_FIT_MSE;

            // AprilTag keeps the first best combination in its loop order,
            // which is lexicographic.
            const uint32_t rank =
                ((position[0] * 16 + position[1]) * 16 + position[2]) * 16 +
                position[3];
            if (valid && (total_error < my_error ||
                          (total_error == my_error && rank < my_rank))) {
                my_error = total_error;
                my_rank = rank;
                my_position = position;
            }
        }

        // Lowest error wins, ties go to the lowest rank.
        const uint32_t my_error_bits = uint32_t(angle_key(my_error, 0) >> 32);
        if (lid == 0) {
            best_error_bits = 0xffffffffu;
            best_rank_shared = 0xffffffffu;
        }
        GroupMemoryBarrierWithGroupSync();
        InterlockedMin(best_error_bits, my_error_bits);
        GroupMemoryBarrierWithGroupSync();
        const uint32_t best_bits = best_error_bits;
        if (my_error_bits == best_bits) {
            InterlockedMin(best_rank_shared, my_rank);
        }
        GroupMemoryBarrierWithGroupSync();
        const uint32_t best_rank = best_rank_shared;
        // Before the next cluster resets them.
        GroupMemoryBarrierWithGroupSync();
        const float best_error = from_angle_key(uint64_t(best_bits) << 32)._0;

        if (best_rank == 0xffffffffu ||
            !(best_error / float(n) < MAX_LINE_FIT_MSE)) {
            if (lid == 0) {
                clear_quad(quads, cluster);
            }
            continue;
        }

        if (my_rank == best_rank) {
            float4 lines[4];
            for (uint32_t side = 0; side < 4; side++) {
                const uint32_t from = my_position[side];
                const uint32_t to = my_position[(side + 1) % 4];
                uint32_t count;
                quads[cluster].moments[side] =
                    get_moment(fits, start, n, kept[from], kept[to], count);
                quads[cluster].num_in_moments[side] = uint16_t(count);
                quads[cluster].indices[side] = uint16_t(kept[from]);
                lines[side] = pair_line[from * MAX_MAXIMA + to];
            }

            float2 c[4];
            if (corners_from_lines(lines, c)) {
                uint32_t slot;
                InterlockedAdd(((uint32_t *)pc.corner_count)[0], 1u, slot);
                const QuadCorners quad = { { { c[0].x, c[0].y },
                                             { c[1].x, c[1].y },
                                             { c[2].x, c[2].y },
                                             { c[3].x, c[3].y } },
                                           0,
                                           cluster };
                corners_out[slot] = quad;
            }
        }
    }
}
