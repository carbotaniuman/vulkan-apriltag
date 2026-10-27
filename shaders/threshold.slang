// Apriltag style adaptive thresholding.
#include "shared/threshold.h"

[vk::push_constant]
ConstantBuffer<ThresholdPush> pc;

[vk::constant_id(0)]
const uint32_t WG_X = 1;
[vk::constant_id(1)]
const uint32_t WG_Y = 1;

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void minmax(uint3 global_id: SV_DispatchThreadID) {
    uint32_t *grayscale = (uint32_t *)pc.grayscale;
    uint32_t *extrema = (uint32_t *)pc.extrema;

    const uint32_t tiles_w = pc.width / 4;
    const uint32_t tx = global_id.x;
    const uint32_t ty = global_id.y;
    if (tx >= tiles_w || ty >= pc.height / 4) {
        return;
    }

    uint32_t lo = 255;
    uint32_t hi = 0;
    for (uint32_t row = 0; row < 4; row++) {
        const uint32_t word = grayscale[(ty * 4 + row) * tiles_w + tx];
        for (uint32_t b = 0; b < 4; b++) {
            const uint32_t value = bitfieldExtract(word, int32_t(b * 8), 8);
            lo = min(lo, value);
            hi = max(hi, value);
        }
    }
    extrema[ty * tiles_w + tx] = lo | (hi << 8);
}

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void filter(uint3 global_id: SV_DispatchThreadID) {
    uint32_t *extrema = (uint32_t *)pc.extrema;
    uint32_t *filtered = (uint32_t *)pc.filtered;

    const int32_t tiles_w = int32_t(pc.width / 4);
    const int32_t tiles_h = int32_t(pc.height / 4);
    const int32_t tx = int32_t(global_id.x);
    const int32_t ty = int32_t(global_id.y);
    if (tx >= tiles_w || ty >= tiles_h) {
        return;
    }

    uint32_t lo = 255;
    uint32_t hi = 0;
    for (int32_t y = max(ty - 1, 0); y <= min(ty + 1, tiles_h - 1); y++) {
        for (int32_t x = max(tx - 1, 0); x <= min(tx + 1, tiles_w - 1); x++) {
            const uint32_t e = extrema[y * tiles_w + x];
            lo = min(lo, e & 0xff);
            hi = max(hi, (e >> 8) & 0xff);
        }
    }
    filtered[ty * tiles_w + tx] = lo | (hi << 8);
}

static const uint32_t MIN_CONTRAST = 5;

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void apply(uint3 global_id: SV_DispatchThreadID) {
    uint32_t *grayscale = (uint32_t *)pc.grayscale;
    uint32_t *filtered = (uint32_t *)pc.filtered;
    uint32_t *thresholded = (uint32_t *)pc.thresholded;

    const uint32_t words_w = pc.width / 4;
    const uint32_t wx = global_id.x;
    const uint32_t y = global_id.y;
    if (wx >= words_w || y >= pc.height) {
        return;
    }

    const uint32_t f = filtered[(y / 4) * words_w + wx];
    const uint32_t lo = f & 0xff;
    const uint32_t hi = (f >> 8) & 0xff;

    uint32_t result = 0x7f7f7f7f;
    if (hi - lo >= MIN_CONTRAST) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const uint32_t word = grayscale[y * words_w + wx];
        result = 0;
        for (uint32_t b = 0; b < 4; b++) {
            if (bitfieldExtract(word, int32_t(b * 8), 8) > mid) {
                result |= 0xffu << (b * 8);
            }
        }
    }
    thresholded[y * words_w + wx] = result;
}
