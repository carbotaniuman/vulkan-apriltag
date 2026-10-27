#ifndef SHARED_THRESHOLD_H
#define SHARED_THRESHOLD_H

#ifndef __SLANG__
#include <cstdint>
#endif

struct ThresholdPush {
    uint32_t width;
    uint32_t height;
    // width * height
    // bytes of a gray scale image
    uint64_t grayscale;
    // (width / 4) * (height / 4)
    // stores lo and hi value in the block
    uint64_t extrema;
    // same as above
    // min lo and max hi in 3 x 3 window
    uint64_t filtered;
    // width * height thresholded output
    uint64_t thresholded;
};

#endif
