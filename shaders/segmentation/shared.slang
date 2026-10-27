#include "shared/segmentation.h"

// BKE information bits, stored in slot 2 of each block.
static const uint32_t TOP_LEFT_255 = 1u << 0;
static const uint32_t TOP_RIGHT_255 = 1u << 1;
static const uint32_t BOTTOM_LEFT_255 = 1u << 2;
static const uint32_t BOTTOM_RIGHT_255 = 1u << 3;
static const uint32_t BITMASK_POS_255 = 0xfu;

// This is unused in the original paper, but has been placed
// here to allow some nicer code structure.
static const uint32_t MUST_UNION_P_255 = 1u << 4;
static const uint32_t MUST_UNION_Q_255 = 1u << 5;
static const uint32_t MUST_UNION_R_255 = 1u << 6;
static const uint32_t MUST_UNION_S_255 = 1u << 7;

// Our own unique bits because we have to duplicate them.
static const uint32_t TOP_LEFT_0 = 1u << 8;
static const uint32_t TOP_RIGHT_0 = 1u << 9;
static const uint32_t BOTTOM_LEFT_0 = 1u << 10;
static const uint32_t BOTTOM_RIGHT_0 = 1u << 11;
static const uint32_t BITMASK_POS_0 = 0xfu << 8;

static const uint32_t SPLIT_0 = 1u << 12;
static const uint32_t MUST_UNION_S_FIRST_0 = 1u << 13;
static const uint32_t MUST_UNION_S_SECOND_0 = 1u << 14;

uint32_t pixel(uint32_t *image, uint32_t index) {
    return bitfieldExtract(image[index >> 2], int32_t((index & 3) * 8), 8);
}
