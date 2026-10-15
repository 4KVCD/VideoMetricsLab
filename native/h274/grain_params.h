// The grain of a frame as d3d11_film_grain.dll takes it: worked out from a
// film grain characteristics SEI message (H.274) by the Python app
// (vmaf_app/core/d3d11_film_grain.py, native_params) and by the C++ app
// (cpp/core/include/vml/film_grain.h), both laid out as here.
#pragma once

#include <stdint.h>

// The grain of one plane. The model values are already adapted to the
// plane's subsampling, and the cutoff frequencies are already indices into
// the pattern database.
struct GrainPlane {
    int width, height;  // the plane's picture size, in samples
    int intervals;      // how many intensity intervals; 0 means no grain on this plane
    uint8_t lower[256];  // each interval's bounds, as 8-bit codes
    uint8_t upper[256];
    int16_t scale[256];  // each interval's grain strength
    uint8_t h[256];      // each interval's horizontal cutoff index, 0-12 (H.274's cutoffs 2-14)
    uint8_t v[256];      // and vertical cutoff index
};

// The grain of one frame.
struct GrainParams {
    int depth;  // the video's bit depth: 8 to 12
    int log2_scale_factor;
    int poc;  // the frame's picture order count, which seeds the random pattern
    GrainPlane plane[3];
};
