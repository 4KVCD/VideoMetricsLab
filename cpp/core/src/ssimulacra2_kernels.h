// The vector kernels of VideoMetricsLab's SSIMULACRA2 (ssimulacra2.cpp), in
// ssimulacra2_kernels.cpp: compiled for every instruction set Highway knows
// and chosen for the CPU at run time, as libjxl chooses its own.
//
// Each kernel performs libjxl 0.12's operations in libjxl's order, with the
// same fused multiply-adds, so its results are bit-identical to libjxl's on
// the same CPU. Rows are padded: `count` is a multiple of 16 floats and every
// row starts on a 64-byte boundary.
#pragma once

#include <cstddef>

namespace vml::ssimulacra2_kernels {

// Whether the code chosen for this CPU fuses multiply-adds (libjxl's AVX2,
// AVX-512 and NEON code does, its SSE code does not). The colour step must
// round as libjxl's does on the same CPU (JxlLinearRgb::make).
bool fused_multiply_add();

// libjxl's XYB of linear sRGB (enc_xyb.cc's LinearRGBRowToXYB), then
// SSIMULACRA2's shift into positive values (MakePositiveXYB). `opsin` holds
// the absorbance matrix scaled by the intensity target (9 floats, row-major)
// followed by the bias and minus its cube root.
void positive_xyb_row(const float* red, const float* green, const float* blue, const float* opsin, float* x,
                      float* y, float* b, std::size_t count);

// out = a * b, element by element.
void multiply_row(const float* a, const float* b, float* out, std::size_t count);

// libjxl's recursive Gaussian (sigma 1.5) along one row of `width` values
// (gauss_blur.cc's FastGaussian1D).
void blur_row(const float* in, float* out, std::size_t width);

// One output row of libjxl's vertical recursive Gaussian
// (gauss_blur.cc's VerticalBlock) over `count` columns. Its input is the
// horizontally blurred row 4 below the output row (`bottom`, a row of zeros
// past the picture's end) plus, from the output row 6 on, the row 6 above
// it (`top`, null before). `state` is 6 rows: the three filters' previous
// two outputs, updated in place. `out` is null while the filter warms up.
void blur_column_step(const float* top, const float* bottom, float* state, float* out, std::size_t count);

}  // namespace vml::ssimulacra2_kernels
