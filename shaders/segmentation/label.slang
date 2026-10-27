// Checks neighboring blocks, with C being this block and X being ignored:
//
//  P | Q | R
// ---+---+---
//  S | C | X
//
// The 255s check the pixels around the block as bits of a 4 by 4 grid, where
// 5 is this block's top-left pixel:
//
//  0 | 1 | 2 | 3
// ---+---+---+---
//  4 | 5 | 6 | 7
// ---+---+---+---
//  8 | 9 | 10| 11
// ---+---+---+---
//  12| 13| 14| 15
//
// Because AprilTag requires 2 different union-finds, one for 255s
// and one for 0s, we have had to duplicate this enum twice. We treat
// 255s as white and 0 as black. This involves changing the label memory
// layout a bit. Because we operate on blocks of 2x2 with our image
// dimensions required to be even, we can fit arrange the data as follows.
//
//  0 | 1
// ---+---
//  2 | 3
//
// Where `uint32_t` 0 contains the labels for the 255s, `uint32_t` 1 contains
// the labels for the 0s, and `uint32_t` 2 contains the information byte,
// which is really a `uint16_t`. `uint32_t` 3 contains (if the block is split)
// the label for the 0s.
//
// The labelling passes over these blocks, each its own entry point: init,
// then compress, merge and compress again (src/vk/segmentation.cpp). The
// final labelling is in final.slang.
import shared;
import union_find;

[vk::push_constant]
ConstantBuffer<SegmentationPush> pc;

[vk::constant_id(0)]
const uint32_t WG_X = 1;
[vk::constant_id(1)]
const uint32_t WG_Y = 1;

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void init(uint3 global_id: SV_DispatchThreadID) {
    uint32_t *thresholded = (uint32_t *)pc.thresholded;
    uint32_t *scratch = (uint32_t *)pc.scratch;
    uint32_t *sizes = (uint32_t *)pc.sizes;

    const uint32_t width = pc.width;
    const uint32_t height = pc.height;
    const uint32_t x = global_id.x * 2;
    const uint32_t y = global_id.y * 2;
    if (x >= width || y >= height) {
        return;
    }
    const uint32_t image_linear_id = y * width + x;

    let value = (uint32_t index) => pixel(thresholded, index);
    let black = (uint32_t index) => value(index) == 0;

    // Whether a block is "split", ie it has 0s as a diagonal pair,
    // which create 2 regions under 4-connectivity.
    let is_split_0 = (bool tl, bool tr, bool bl, bool br) =>
        (tl && br && !tr && !bl) || (tr && bl && !tl && !br);
    let block_is_split_0 = (uint32_t index) =>
        is_split_0(black(index), black(index + 1), black(index + width),
                   black(index + width + 1));

    const uint32_t tl_value = value(image_linear_id);
    const uint32_t tr_value = value(image_linear_id + 1);
    const uint32_t bl_value = value(image_linear_id + width);
    const uint32_t br_value = value(image_linear_id + width + 1);

    // The bits of `information_byte` are the BKE information bits.
    uint32_t information_byte = 0;
    uint32_t pixels_to_check_255 = 0;
    if (tl_value == 255) {
        pixels_to_check_255 |= 0x777u;
        information_byte |= TOP_LEFT_255;
    } else if (tl_value == 0) {
        information_byte |= TOP_LEFT_0;
    }
    if (tr_value == 255) {
        pixels_to_check_255 |= 0x777u << 1;
        information_byte |= TOP_RIGHT_255;
    } else if (tr_value == 0) {
        information_byte |= TOP_RIGHT_0;
    }
    if (bl_value == 255) {
        pixels_to_check_255 |= 0x777u << 4;
        information_byte |= BOTTOM_LEFT_255;
    } else if (bl_value == 0) {
        information_byte |= BOTTOM_LEFT_0;
    }
    if (br_value == 255) {
        information_byte |= BOTTOM_RIGHT_255;
    } else if (br_value == 0) {
        information_byte |= BOTTOM_RIGHT_0;
    }

    // Mask out the parts of the grid outside the image.
    if (x == 0) {
        pixels_to_check_255 &= 0xeeeeu;
    } else if (x + 2 >= width) {
        pixels_to_check_255 &= 0x7777u;
    }
    if (y == 0) {
        pixels_to_check_255 &= 0xfff0u;
    } else if (y + 2 >= height) {
        pixels_to_check_255 &= 0x0fffu;
    }

    // Whether the 255s connect to the outer pixels.
    let handle_pixel_test_255 =
        (int32_t pixel_dy, int32_t pixel_dx, uint32_t bit_index) =>
            (pixels_to_check_255 & (1u << bit_index)) != 0 &&
            value(uint32_t(int32_t(image_linear_id) +
                           pixel_dy * int32_t(width) + pixel_dx)) == 255;
    const bool hit_0 = handle_pixel_test_255(-1, -1, 0);
    const bool hit_1 = handle_pixel_test_255(-1, 0, 1);
    const bool hit_2 = handle_pixel_test_255(-1, 1, 2);
    const bool hit_3 = handle_pixel_test_255(-1, 2, 3);
    const bool hit_4 = handle_pixel_test_255(0, -1, 4);
    const bool hit_8 = handle_pixel_test_255(1, -1, 8);

    // The neighbouring blocks the 255s connect to, dropping redundant merges.
    // - P is redundant if pixels 0 and 1 both match (Q unions P);
    // - R is redundant if pixels 2 and 3 both match (R unions Q);
    // - S is redundant if pixels 0 and 4 both match (S unions P).
    const bool connects_p = hit_0 && !hit_1;
    const bool connects_q = hit_1 || hit_2;
    const bool connects_r = hit_3 && !hit_2;
    const bool connects_s = (hit_4 || hit_8) && !(hit_0 && hit_4);

    // Tthe first block joined (in P, Q, R, S order) becomes slot 0's parent,
    // and the others are merged later.
    const int32_t w = int32_t(width);
    int32_t label_offset_255 = 0;
    if (connects_p) {
        label_offset_255 = -2 * w - 2;
    } else if (connects_q) {
        label_offset_255 = -2 * w;
    } else if (connects_r) {
        label_offset_255 = -2 * w + 2;
    } else if (connects_s) {
        label_offset_255 = -2;
    }
    if (connects_q && connects_p) {
        information_byte |= MUST_UNION_Q_255;
    }
    if (connects_r && (connects_p || connects_q)) {
        information_byte |= MUST_UNION_R_255;
    }
    if (connects_s && (connects_p || connects_q || connects_r)) {
        information_byte |= MUST_UNION_S_255;
    }

    // 0s only use 4 connectivity.
    const bool tl = tl_value == 0;
    const bool tr = tr_value == 0;
    const bool bl = bl_value == 0;
    const bool br = br_value == 0;
    const bool split = is_split_0(tl, tr, bl, br);
    if (split) {
        information_byte |= SPLIT_0;
    }

    const uint32_t first_node = image_linear_id + 1;
    const uint32_t second_node = image_linear_id + width + 1;

    // The top pixels touch the bottom row of block Q, which is Q's
    // slot 3 iff Q is split.
    bool union_q = false;
    bool q_second = false;
    if (y > 0 && ((tl && black(image_linear_id - width)) ||
                  (tr && black(image_linear_id - width + 1)))) {
        union_q = true;
        q_second = block_is_split_0(image_linear_id - 2 * width);
    }

    bool s_first = false;
    bool s_second_from_first = false;
    bool s_from_second = false;
    bool s_from_second_is_second = false;
    if (x > 0) {
        const uint32_t s = image_linear_id - 2;
        s_first = tl && black(s + 1);
        if (bl && black(s + width + 1)) {
            const bool s_split = block_is_split_0(s);
            if (split) {
                s_from_second = true;
                s_from_second_is_second = s_split;
            } else if (s_split) {
                s_second_from_first = true;
            } else {
                s_first = true;
            }
        }
    }

    const uint32_t q_node =
        image_linear_id - 2 * width + (q_second ? width + 1 : 1);
    uint32_t first_parent = first_node;
    if (union_q) {
        first_parent = q_node;
        if (s_first) {
            information_byte |= MUST_UNION_S_FIRST_0;
        }
    } else if (s_first) {
        first_parent = image_linear_id - 1;
    }
    if (s_second_from_first) {
        information_byte |= MUST_UNION_S_SECOND_0;
    }

    // Only write slot 3 if this is a split block.
    if (split) {
        scratch[second_node] =
            s_from_second
                ? (s_from_second_is_second ? image_linear_id + width - 1
                                           : image_linear_id - 1)
                : second_node;
    }

    scratch[image_linear_id] =
        uint32_t(int32_t(image_linear_id) + label_offset_255);
    scratch[first_node] = first_parent;
    scratch[image_linear_id + width] = information_byte;

    sizes[image_linear_id] = 0;
    sizes[first_node] = 0;
    sizes[second_node] = 0;
}

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void compress(uint3 global_id: SV_DispatchThreadID) {
    Atomic<uint32_t> *scratch = (Atomic<uint32_t> *)pc.scratch;

    const uint32_t x = global_id.x * 2;
    const uint32_t y = global_id.y * 2;
    if (x >= pc.width || y >= pc.height) {
        return;
    }
    const uint32_t id = y * pc.width + x;

    uf_find_compress(scratch, id);
    uf_find_compress(scratch, id + 1);
}

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void merge(uint3 global_id: SV_DispatchThreadID) {
    uint32_t *scratch = (uint32_t *)pc.scratch;
    Atomic<uint32_t> *uf = (Atomic<uint32_t> *)pc.scratch;

    const uint32_t width = pc.width;
    const uint32_t x = global_id.x * 2;
    const uint32_t y = global_id.y * 2;
    if (x >= width || y >= pc.height) {
        return;
    }
    const uint32_t id = y * width + x;

    // Slot 2 is not UF so can use a plain load.
    const uint32_t info = scratch[id + width];

    if ((info & MUST_UNION_Q_255) != 0) {
        uf_merge(uf, id, id - 2 * width);
    }
    if ((info & MUST_UNION_R_255) != 0) {
        uf_merge(uf, id, id - 2 * width + 2);
    }
    if ((info & MUST_UNION_S_255) != 0) {
        uf_merge(uf, id, id - 2);
    }
    if ((info & MUST_UNION_S_FIRST_0) != 0) {
        uf_merge(uf, id + 1, id - 1);
    }
    if ((info & MUST_UNION_S_SECOND_0) != 0) {
        uf_merge(uf, id + 1, id + width - 1);
    }
}
