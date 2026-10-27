#ifndef SHARED_SEGMENTATION_H
#define SHARED_SEGMENTATION_H

#ifndef __SLANG__
#include <cstdint>
#endif

struct SegmentationPush {
    uint32_t width;
    uint32_t height;
    // width * height bytes
    // Read as uint32_t for bandwidth reasons
    uint64_t thresholded;
    // width * height
    // explained in the shader
    uint64_t scratch;
    // width * height
    // label per pixel, 0 if pixel is 127
    uint64_t labels;
    // valid only at label indices
    uint64_t sizes;
};

static const uint32_t LABEL_PIXEL_MASK = 0x80000000u;
static const uint32_t LABEL_VALUE_MASK = 0x7FFFFFFFu;

#endif
