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

// blur_row of two rows of the same width, at once where vectors are wide
// enough for both.
void blur_rows2(const float* in_a, float* out_a, const float* in_b, float* out_b, std::size_t width);

// `steps` consecutive output rows of libjxl's vertical recursive Gaussian
// (gauss_blur.cc's VerticalBlock) over `count` columns. Step k's input is
// the horizontally blurred row 4 below its output row (`bottoms[k]`, a row
// of zeros past the picture's end) plus, from the output row 6 on, the row
// 6 above it (`tops[k]`, null before). `state` is 6 rows: the three
// filters' previous two outputs, updated in place. `outs[k]` is null while
// the filter warms up.
void blur_column_steps(const float* const* tops, const float* const* bottoms, float* state, float* const* outs,
                       std::size_t steps, std::size_t count);

// libjxl's SSIMMap for one row of `width` pixels of one channel (the blurred
// means m1, m2 and the blurred products s11, s22, s12; `c2` its constant),
// adding to its two sums (the terms, their fourth powers) in pixel order.
// It and add_edge_row read whole vectors, up to 7 floats past `width`: the
// rows must be padded that far.
void add_ssim_row(const float* m1, const float* m2, const float* s11, const float* s22, const float* s12,
                  std::size_t width, float c2, double* sums);

// libjxl's EdgeDiffMap for one row of one channel (the pictures and their
// blurred means), adding to its four sums (artifacts, their fourth powers,
// detail lost, its fourth powers) in pixel order.
void add_edge_row(const float* img1, const float* mu1, const float* img2, const float* mu2, std::size_t width,
                  double* sums);

}  // namespace vml::ssimulacra2_kernels
