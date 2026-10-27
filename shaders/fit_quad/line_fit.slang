#ifndef FIT_QUAD_LINE_FIT_SLANG
#define FIT_QUAD_LINE_FIT_SLANG

#include "shared.slang"

static const int32_t HALF_DX[4] = { 1, -1, 0, 1 };
static const int32_t HALF_DY[4] = { 0, 1, 1, 1 };

uint32_t isqrt(uint32_t n) {
    uint32_t s = uint32_t(sqrt(float(n)));
    while (s * s > n) {
        s--;
    }
    while ((s + 1) * (s + 1) <= n) {
        s++;
    }
    return s;
}

int2 offset_coordinates(uint32_t point_coord) {
    const uint32_t packed_x = point_coord & 0xffff;
    const uint32_t packed_y = point_coord >> 16;
    const uint32_t half_pixel = packed_x >> 14;
    return int2(int32_t(2 * (packed_x & 0x3fff)) + HALF_DX[half_pixel] + 1,
                int32_t(2 * (packed_y & 0x3fff)) + HALF_DY[half_pixel] + 1);
}

uint32_t read_pixel(uint32_t *image, uint32_t index) {
    return bitfieldExtract(image[index >> 2], int32_t((index & 3) * 8), 8);
}

// A point's line-fit weight
int32_t point_weight(uint32_t word, uint32_t *image) {
    const int2 r = offset_coordinates(word);
    const uint32_t x = uint32_t(r.x) / 2;
    const uint32_t y = uint32_t(r.y) / 2;
    const uint32_t width = pc.width;

    int32_t w = 1;
    if (x > 0 && x + 1 < width && y > 0 && y + 1 < pc.height) {
        const int32_t grad_x = int32_t(read_pixel(image, y * width + x + 1)) -
                               int32_t(read_pixel(image, y * width + x - 1));
        const int32_t grad_y = int32_t(read_pixel(image, (y + 1) * width + x)) -
                               int32_t(read_pixel(image, (y - 1) * width + x));
        w += int32_t(isqrt(uint32_t(grad_x * grad_x + grad_y * grad_y)));
    }
    return w;
}

LineFitPoint initial_line_fit(uint32_t word, int32_t w) {
    const int2 r = offset_coordinates(word);
    const int32_t rx = r.x;
    const int32_t ry = r.y;
    const int32_t wx = w * rx;
    const int32_t wy = w * ry;
    return LineFitPoint(int64_t(wx), int64_t(wy), int64_t(wx) * int64_t(rx),
                        int64_t(wy) * int64_t(ry), int64_t(wx) * int64_t(ry), w,
                        0);
}

LineFitPoint moment_sub(LineFitPoint a, LineFitPoint b) {
    return LineFitPoint(a.Mx - b.Mx, a.My - b.My, a.Mxx - b.Mxx, a.Myy - b.Myy,
                        a.Mxy - b.Mxy, a.W - b.W, 0);
}

LineFitPoint moment_add(LineFitPoint a, LineFitPoint b) {
    return LineFitPoint(a.Mx + b.Mx, a.My + b.My, a.Mxx + b.Mxx, a.Myy + b.Myy,
                        a.Mxy + b.Mxy, a.W + b.W, 0);
}

LineFitPoint get_moment(LineFitPoint *sums, uint32_t start, uint32_t n,
                        uint32_t i0, uint32_t i1, out uint32_t count) {
    LineFitPoint result = LineFitPoint(0, 0, 0, 0, 0, 0, 0);
    count = 0;
    if (i0 < i1) {
        count = i1 - i0 + 1;
        result = sums[start + i1];
        if (i0 > 0) {
            result = moment_sub(result, sums[start + i0 - 1]);
        }
    } else if (i1 < i0) {
        count = n - i0 + i1 + 1;
        result =
            moment_add(moment_sub(sums[start + n - 1], sums[start + i0 - 1]),
                       sums[start + i1]);
    }
    return result;
}

struct WindowMoments {
    int32_t Mx;
    int32_t My;
    int64_t Mxx;
    int64_t Myy;
    int64_t Mxy;
    int32_t W;
};

WindowMoments window_of(LineFitPoint m) {
    return WindowMoments(int32_t(m.Mx), int32_t(m.My), m.Mxx, m.Myy, m.Mxy,
                         m.W);
}

float window_line_error(WindowMoments m, uint32_t count) {
    precise float cxx =
        float(m.Mxx * int64_t(m.W) - int64_t(m.Mx) * int64_t(m.Mx));
    precise float cxy =
        float(m.Mxy * int64_t(m.W) - int64_t(m.Mx) * int64_t(m.My));
    precise float cyy =
        float(m.Myy * int64_t(m.W) - int64_t(m.My) * int64_t(m.My));
    precise float w = float(m.W);
    precise float root = sqrt((cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy);
    precise float eig_small = 0.5 * (cxx + cyy - root) / (4 * w * w);
    precise float error = float(count) * eig_small;
    return error;
}

void fit_line(LineFitPoint m, uint32_t count, out float4 line, out float err,
              out float mse) {
    precise float w = float(m.W);
    const int32_t qx = m.W > 0 ? int32_t(float(m.Mx) / w) : 0;
    const int32_t qy = m.W > 0 ? int32_t(float(m.My) / w) : 0;
    const int64_t sx = m.Mx - int64_t(qx) * int64_t(m.W);
    const int64_t sy = m.My - int64_t(qy) * int64_t(m.W);
    const int64_t sxx = m.Mxx - (m.Mx + sx) * int64_t(qx);
    const int64_t syy = m.Myy - (m.My + sy) * int64_t(qy);
    const int64_t sxy = m.Mxy - m.My * int64_t(qx) - sx * int64_t(qy);

    precise float ex = float(sx) / w;
    precise float ey = float(sy) / w;
    precise float cxx = (float(sxx) / w - ex * ex) / 4;
    precise float cxy = (float(sxy) / w - ex * ey) / 4;
    precise float cyy = (float(syy) / w - ey * ey) / 4;

    precise float root = sqrt((cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy);
    precise float eig_small = 0.5 * (cxx + cyy - root);

    precise float px = (float(qx) + ex) / 2;
    precise float py = (float(qy) + ey) / 2;
    precise float eig = 0.5 * (cxx + cyy + root);
    precise float nx1 = cxx - eig;
    precise float ny1 = cxy;
    precise float m1 = nx1 * nx1 + ny1 * ny1;
    precise float nx2 = cxy;
    precise float ny2 = cyy - eig;
    precise float m2 = nx2 * nx2 + ny2 * ny2;
    precise float nx = m1 > m2 ? nx1 : nx2;
    precise float ny = m1 > m2 ? ny1 : ny2;
    precise float length = sqrt(m1 > m2 ? m1 : m2);
    line = abs(length) < 1e-12 ? float4(px, py, 0, 0)
                               : float4(px, py, nx / length, ny / length);

    precise float e = float(count) * eig_small;
    err = e;
    mse = eig_small;
}

#endif
