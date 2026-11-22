// SSIMULACRA2's vector kernels (ssimulacra2_kernels.h), ported from libjxl
// 0.12 (lib/jxl/enc_xyb.cc, lib/jxl/base/fast_math-inl.h, tools/gauss_blur.cc,
// tools/ssimulacra2.cc; BSD licence, docs/THIRD_PARTY.md) with the same
// Highway operations in the same order, so that every value rounds as in
// libjxl. Highway compiles this file once per instruction set
// (foreach_target.h includes it again for each) and HWY_DYNAMIC_DISPATCH
// picks the best one for the CPU, as in libjxl.
#include "ssimulacra2_kernels.h"

#include <algorithm>
#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ssimulacra2_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace vml::ssimulacra2_kernels {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

bool FusedMultiplyAdd() { return HWY_NATIVE_FMA != 0; }

// libjxl's cube root plus `add` for non-negative x (CubeRootAndAdd): a first
// guess from the exponent, three Newton steps for 1/cbrt(x), a last step,
// then x * r^2 = cbrt(x).
template <class D, class V>
HWY_INLINE V CubeRootAndAdd(D df, V x, V add) {
    const hn::RebindToSigned<D> di;
    const auto exponent_bias = hn::Set(di, 0x54800000);  // bits of 1.0 plus a third of them
    const auto exponent_third = hn::Set(di, 0x002AAAAA);  // 1/3 shifted to the exponent
    const V third = hn::Set(df, 1.0f / 3);
    const V four_thirds = hn::Set(df, 4.0f / 3);
    const V x_third = hn::Mul(third, x);
    // Multiply the exponent by -1/3; 0 (exponent 0) would give a wrong guess
    // and NaNs below, so its guess is 0.
    const auto bits = hn::BitCast(di, x);
    const auto guess = hn::IfThenZeroElse(
        hn::Eq(bits, hn::Zero(di)), hn::Sub(exponent_bias, hn::Mul(hn::ShiftRight<23>(bits), exponent_third)));
    V r = hn::BitCast(df, guess);
    for (int i = 0; i < 3; ++i) {
        const V r2 = hn::Mul(r, r);
        r = hn::NegMulAdd(x_third, hn::Mul(r2, r2), hn::Mul(four_thirds, r));
    }
    V r2 = hn::Mul(r, r);
    r = hn::MulAdd(third, hn::NegMulAdd(x, hn::Mul(r2, r2), r), r);
    r2 = hn::Mul(r, r);
    return hn::MulAdd(r2, x, add);
}

void PositiveXybRow(const float* HWY_RESTRICT red, const float* HWY_RESTRICT green,
                    const float* HWY_RESTRICT blue, const float* HWY_RESTRICT opsin, float* HWY_RESTRICT x_out,
                    float* HWY_RESTRICT y_out, float* HWY_RESTRICT b_out, size_t count) {
    const hn::ScalableTag<float> d;
    using V = hn::Vec<decltype(d)>;
    // One variable per matrix entry, as libjxl writes them: an SVE vector has
    // no compile-time size, so Highway's SVE targets cannot make an array of them.
    const V m0 = hn::Set(d, opsin[0]);
    const V m1 = hn::Set(d, opsin[1]);
    const V m2 = hn::Set(d, opsin[2]);
    const V m3 = hn::Set(d, opsin[3]);
    const V m4 = hn::Set(d, opsin[4]);
    const V m5 = hn::Set(d, opsin[5]);
    const V m6 = hn::Set(d, opsin[6]);
    const V m7 = hn::Set(d, opsin[7]);
    const V m8 = hn::Set(d, opsin[8]);
    const V bias = hn::Set(d, opsin[9]);
    const V neg_cbrt_bias = hn::Set(d, opsin[10]);
    const V half = hn::Set(d, 0.5f);
    for (size_t i = 0; i < count; i += hn::Lanes(d)) {
        const V r = hn::Load(d, red + i);
        const V g = hn::Load(d, green + i);
        const V b = hn::Load(d, blue + i);
        // OpsinAbsorbance, then ZeroIfNegative and CubeRootAndAdd per channel.
        V mixed0 = hn::MulAdd(m0, r, hn::MulAdd(m1, g, hn::MulAdd(m2, b, bias)));
        V mixed1 = hn::MulAdd(m3, r, hn::MulAdd(m4, g, hn::MulAdd(m5, b, bias)));
        V mixed2 = hn::MulAdd(m6, r, hn::MulAdd(m7, g, hn::MulAdd(m8, b, bias)));
        mixed0 = CubeRootAndAdd(d, hn::ZeroIfNegative(mixed0), neg_cbrt_bias);
        mixed1 = CubeRootAndAdd(d, hn::ZeroIfNegative(mixed1), neg_cbrt_bias);
        mixed2 = CubeRootAndAdd(d, hn::ZeroIfNegative(mixed2), neg_cbrt_bias);
        // StoreXYB.
        const V x = hn::Mul(half, hn::Sub(mixed0, mixed1));
        const V y = hn::Mul(half, hn::Add(mixed0, mixed1));
        // MakePositiveXYB, which libjxl compiles as plain scalar code: no
        // fused multiply-add there.
        hn::Store(hn::Add(hn::Sub(mixed2, y), hn::Set(d, 0.55f)), d, b_out + i);
        hn::Store(hn::Add(hn::Mul(x, hn::Set(d, 14.f)), hn::Set(d, 0.42f)), d, x_out + i);
        hn::Store(hn::Add(y, hn::Set(d, 0.01f)), d, y_out + i);
    }
}

void MultiplyRow(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, float* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<float> d;
    for (size_t i = 0; i < count; i += hn::Lanes(d)) {
        hn::Store(hn::Mul(hn::Load(d, a + i), hn::Load(d, b + i)), d, out + i);
    }
}

// libjxl's CreateRecursiveGaussian(1.5), as its official tools compute it:
// the three filters (k = 1, 3, 5) of radius 5. mul_in, mul_prev and
// mul_prev2 give four consecutive outputs of the recursion at once, from
// four inputs and the two outputs before them; n2 and d1 are the
// recursion's own two weights.
constexpr int kRadius = 5;
alignas(16) constexpr float kMulIn[3][4] = {
    {0x1.c4fa84p-5f, 0x1.aecee8p-4f, 0x1.287a74p-3f, 0x1.5c87fap-3f},
    {-0x1.e1fd7ap-5f, -0x1.1b4e84p-4f, -0x1.70352cp-6f, 0x1.5e2fb4p-5f},
    {0x1.a8894ap-7f, 0x1.d44a64p-60f, -0x1.a8894ap-7f, -0x1.d44a64p-59f}};
alignas(16) constexpr float kMulPrev[3][4] = {
    {0x1.e6f0e2p+0f, 0x1.4f1bbcp+1f, 0x1.89f188p+1f, 0x1.9e377ap+1f},
    {0x1.2cf23p+0f, 0x1.87221ap-2f, -0x1.73fd62p-1f, -0x1.3c6ef4p+0f},
    {0x1.1a6264p-53f, -0x1p+0f, -0x1.1a6264p-52f, 0x1p+0f}};
alignas(16) constexpr float kMulPrev2[3][4] = {
    {-0x1p+0f, -0x1.e6f0e2p+0f, -0x1.4f1bbcp+1f, -0x1.89f188p+1f},
    {-0x1p+0f, -0x1.2cf23p+0f, -0x1.87221ap-2f, 0x1.73fd62p-1f},
    {-0x1p+0f, -0x1.1a6264p-53f, 0x1p+0f, 0x1.1a6264p-52f}};
constexpr float kN2[3] = {0x1.c4fa84p-5f, -0x1.e1fd7ap-5f, 0x1.a8894ap-7f};
constexpr float kD1[3] = {-0x1.e6f0e2p+0f, -0x1.2cf23p+0f, -0x1.1a6264p-53f};

// FastGaussian1D, step for step. Its vectors are capped at 4 lanes: each
// holds four consecutive outputs.
void BlurRow(const float* HWY_RESTRICT in, float* HWY_RESTRICT out, size_t width) {
    const hn::CappedTag<float, 4> d;
    using V = hn::Vec<decltype(d)>;
    const V mul_in_1 = hn::Load(d, kMulIn[0]);
    const V mul_in_3 = hn::Load(d, kMulIn[1]);
    const V mul_in_5 = hn::Load(d, kMulIn[2]);
    const V mul_prev_1 = hn::Load(d, kMulPrev[0]);
    const V mul_prev_3 = hn::Load(d, kMulPrev[1]);
    const V mul_prev_5 = hn::Load(d, kMulPrev[2]);
    const V mul_prev2_1 = hn::Load(d, kMulPrev2[0]);
    const V mul_prev2_3 = hn::Load(d, kMulPrev2[1]);
    const V mul_prev2_5 = hn::Load(d, kMulPrev2[2]);
    V prev_1 = hn::Zero(d), prev_3 = hn::Zero(d), prev_5 = hn::Zero(d);
    V prev2_1 = hn::Zero(d), prev2_3 = hn::Zero(d), prev2_5 = hn::Zero(d);
    const ptrdiff_t xsize = static_cast<ptrdiff_t>(width);
    constexpr ptrdiff_t N = kRadius;

    // One output at a time (only lane 0 matters), reading zeros outside the
    // row; outputs before x = 0 only warm the filter up.
    const auto single_step = [&](ptrdiff_t n) {
        const ptrdiff_t left = n - N - 1;
        const ptrdiff_t right = n + N - 1;
        const float left_val = left >= 0 ? in[left] : 0.0f;
        const float right_val = right < xsize ? in[right] : 0.0f;
        const V sum = hn::Set(d, left_val + right_val);
        V out_1 = hn::Mul(sum, mul_in_1);
        V out_3 = hn::Mul(sum, mul_in_3);
        V out_5 = hn::Mul(sum, mul_in_5);
        out_1 = hn::MulAdd(mul_prev2_1, prev2_1, out_1);
        out_3 = hn::MulAdd(mul_prev2_3, prev2_3, out_3);
        out_5 = hn::MulAdd(mul_prev2_5, prev2_5, out_5);
        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;
        out_1 = hn::MulAdd(mul_prev_1, prev_1, out_1);
        out_3 = hn::MulAdd(mul_prev_3, prev_3, out_3);
        out_5 = hn::MulAdd(mul_prev_5, prev_5, out_5);
        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;
        if (n >= 0) out[n] = hn::GetLane(hn::Add(out_1, hn::Add(out_3, out_5)));
    };

    ptrdiff_t n = -N + 1;
    const ptrdiff_t first_aligned = static_cast<ptrdiff_t>((N + 1 + 3) / 4 * 4);  // RoundUpTo(N + 1, 4)
    for (; n < std::min(first_aligned, xsize); ++n) single_step(n);

    // Four outputs per step: lane k adds input k's share to each later
    // output (the weights shifted by k lanes) and the two previous outputs'
    // shares; the last two outputs become the next step's previous ones.
    prev2_1 = hn::Broadcast<0>(prev2_1);
    prev2_3 = hn::Broadcast<0>(prev2_3);
    prev2_5 = hn::Broadcast<0>(prev2_5);
    prev_1 = hn::Broadcast<0>(prev_1);
    prev_3 = hn::Broadcast<0>(prev_3);
    prev_5 = hn::Broadcast<0>(prev_5);
    for (; n < xsize - N + 1 - 3; n += 4) {
        const V sum = hn::Add(hn::LoadU(d, in + n - N - 1), hn::LoadU(d, in + n + N - 1));
        const V in0 = hn::Broadcast<0>(sum);
        V out_1 = hn::Mul(in0, mul_in_1);
        V out_3 = hn::Mul(in0, mul_in_3);
        V out_5 = hn::Mul(in0, mul_in_5);
        const V in1 = hn::Broadcast<1>(sum);
        out_1 = hn::MulAdd(hn::ShiftLeftLanes<1>(mul_in_1), in1, out_1);
        out_3 = hn::MulAdd(hn::ShiftLeftLanes<1>(mul_in_3), in1, out_3);
        out_5 = hn::MulAdd(hn::ShiftLeftLanes<1>(mul_in_5), in1, out_5);
        const V in2 = hn::Broadcast<2>(sum);
        out_1 = hn::MulAdd(hn::ShiftLeftLanes<2>(mul_in_1), in2, out_1);
        out_3 = hn::MulAdd(hn::ShiftLeftLanes<2>(mul_in_3), in2, out_3);
        out_5 = hn::MulAdd(hn::ShiftLeftLanes<2>(mul_in_5), in2, out_5);
        const V in3 = hn::Broadcast<3>(sum);
        out_1 = hn::MulAdd(hn::ShiftLeftLanes<3>(mul_in_1), in3, out_1);
        out_3 = hn::MulAdd(hn::ShiftLeftLanes<3>(mul_in_3), in3, out_3);
        out_5 = hn::MulAdd(hn::ShiftLeftLanes<3>(mul_in_5), in3, out_5);
        out_1 = hn::MulAdd(mul_prev2_1, prev2_1, out_1);
        out_3 = hn::MulAdd(mul_prev2_3, prev2_3, out_3);
        out_5 = hn::MulAdd(mul_prev2_5, prev2_5, out_5);
        out_1 = hn::MulAdd(mul_prev_1, prev_1, out_1);
        out_3 = hn::MulAdd(mul_prev_3, prev_3, out_3);
        out_5 = hn::MulAdd(mul_prev_5, prev_5, out_5);
        prev2_1 = hn::Broadcast<2>(out_1);
        prev2_3 = hn::Broadcast<2>(out_3);
        prev2_5 = hn::Broadcast<2>(out_5);
        prev_1 = hn::Broadcast<3>(out_1);
        prev_3 = hn::Broadcast<3>(out_3);
        prev_5 = hn::Broadcast<3>(out_5);
        hn::StoreU(hn::Add(out_1, hn::Add(out_3, out_5)), d, out + n);
    }

    for (; n < xsize; ++n) single_step(n);
}

// VerticalBlock for one row: each lane is one column. libjxl adds the
// zero row too (TwoInputs) or loads it alone (SingleInput); so does this.
void BlurColumnStep(const float* HWY_RESTRICT top, const float* HWY_RESTRICT bottom, float* HWY_RESTRICT state,
                    float* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<float> d;
    using V = hn::Vec<decltype(d)>;
    const V d1_1 = hn::Set(d, kD1[0]), d1_3 = hn::Set(d, kD1[1]), d1_5 = hn::Set(d, kD1[2]);
    const V n2_1 = hn::Set(d, kN2[0]), n2_3 = hn::Set(d, kN2[1]), n2_5 = hn::Set(d, kN2[2]);
    float* HWY_RESTRICT prev_1 = state;
    float* HWY_RESTRICT prev2_1 = state + count;
    float* HWY_RESTRICT prev_3 = state + 2 * count;
    float* HWY_RESTRICT prev2_3 = state + 3 * count;
    float* HWY_RESTRICT prev_5 = state + 4 * count;
    float* HWY_RESTRICT prev2_5 = state + 5 * count;
    for (size_t i = 0; i < count; i += hn::Lanes(d)) {
        const V sum = top ? hn::Add(hn::Load(d, top + i), hn::Load(d, bottom + i)) : hn::Load(d, bottom + i);
        const V y_n1_1 = hn::Load(d, prev_1 + i), y_n2_1 = hn::Load(d, prev2_1 + i);
        const V y_n1_3 = hn::Load(d, prev_3 + i), y_n2_3 = hn::Load(d, prev2_3 + i);
        const V y_n1_5 = hn::Load(d, prev_5 + i), y_n2_5 = hn::Load(d, prev2_5 + i);
        const V y1 = hn::MulAdd(n2_1, sum, hn::NegMulSub(d1_1, y_n1_1, y_n2_1));
        const V y3 = hn::MulAdd(n2_3, sum, hn::NegMulSub(d1_3, y_n1_3, y_n2_3));
        const V y5 = hn::MulAdd(n2_5, sum, hn::NegMulSub(d1_5, y_n1_5, y_n2_5));
        hn::Store(y_n1_1, d, prev2_1 + i);
        hn::Store(y_n1_3, d, prev2_3 + i);
        hn::Store(y_n1_5, d, prev2_5 + i);
        hn::Store(y1, d, prev_1 + i);
        hn::Store(y3, d, prev_3 + i);
        hn::Store(y5, d, prev_5 + i);
        if (out) hn::Store(hn::Add(y1, hn::Add(y3, y5)), d, out + i);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace vml::ssimulacra2_kernels
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace vml::ssimulacra2_kernels {

HWY_EXPORT(FusedMultiplyAdd);
HWY_EXPORT(PositiveXybRow);
HWY_EXPORT(MultiplyRow);
HWY_EXPORT(BlurRow);
HWY_EXPORT(BlurColumnStep);

bool fused_multiply_add() { return HWY_DYNAMIC_DISPATCH(FusedMultiplyAdd)(); }

void positive_xyb_row(const float* red, const float* green, const float* blue, const float* opsin, float* x,
                      float* y, float* b, std::size_t count) {
    HWY_DYNAMIC_DISPATCH(PositiveXybRow)(red, green, blue, opsin, x, y, b, count);
}

void multiply_row(const float* a, const float* b, float* out, std::size_t count) {
    HWY_DYNAMIC_DISPATCH(MultiplyRow)(a, b, out, count);
}

void blur_row(const float* in, float* out, std::size_t width) { HWY_DYNAMIC_DISPATCH(BlurRow)(in, out, width); }

void blur_column_step(const float* top, const float* bottom, float* state, float* out, std::size_t count) {
    HWY_DYNAMIC_DISPATCH(BlurColumnStep)(top, bottom, state, out, count);
}

}  // namespace vml::ssimulacra2_kernels
#endif  // HWY_ONCE
