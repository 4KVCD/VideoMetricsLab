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
// holds four consecutive outputs of a row.
using D4 = hn::CappedTag<float, 4>;
using V4 = hn::Vec<D4>;
constexpr ptrdiff_t N = kRadius;

// The three filters' last two outputs (prev_1, prev_3, prev_5, then
// prev2_1, prev2_3, prev2_5).
template <class V>
using Filters = V[6];

// The outputs before the first whole step of four, and those after the last.
ptrdiff_t first_four(ptrdiff_t xsize) {
    const ptrdiff_t first_aligned = (N + 1 + 3) / 4 * 4;  // RoundUpTo(N + 1, 4)
    return std::min(first_aligned, xsize);
}
ptrdiff_t past_fours(ptrdiff_t xsize) { return xsize - N + 1 - 3; }

// One output n alone (only lane 0 matters), reading zeros outside the row;
// outputs before x = 0 only warm the filter up.
HWY_INLINE void SingleStep(const float* HWY_RESTRICT in, float* HWY_RESTRICT out, ptrdiff_t xsize, ptrdiff_t n,
                           Filters<V4>& f) {
    const D4 d;
    const ptrdiff_t left = n - N - 1;
    const ptrdiff_t right = n + N - 1;
    const float left_val = left >= 0 ? in[left] : 0.0f;
    const float right_val = right < xsize ? in[right] : 0.0f;
    const V4 sum = hn::Set(d, left_val + right_val);
    V4 out_1 = hn::Mul(sum, hn::Load(d, kMulIn[0]));
    V4 out_3 = hn::Mul(sum, hn::Load(d, kMulIn[1]));
    V4 out_5 = hn::Mul(sum, hn::Load(d, kMulIn[2]));
    out_1 = hn::MulAdd(hn::Load(d, kMulPrev2[0]), f[3], out_1);
    out_3 = hn::MulAdd(hn::Load(d, kMulPrev2[1]), f[4], out_3);
    out_5 = hn::MulAdd(hn::Load(d, kMulPrev2[2]), f[5], out_5);
    f[3] = f[0];
    f[4] = f[1];
    f[5] = f[2];
    out_1 = hn::MulAdd(hn::Load(d, kMulPrev[0]), f[0], out_1);
    out_3 = hn::MulAdd(hn::Load(d, kMulPrev[1]), f[1], out_3);
    out_5 = hn::MulAdd(hn::Load(d, kMulPrev[2]), f[2], out_5);
    f[0] = out_1;
    f[1] = out_3;
    f[2] = out_5;
    if (n >= 0) out[n] = hn::GetLane(hn::Add(out_1, hn::Add(out_3, out_5)));
}

// The single steps before the whole steps of four; then every lane of the
// filters holds lane 0's output, as the steps of four want.
void StartRow(const float* HWY_RESTRICT in, float* HWY_RESTRICT out, ptrdiff_t xsize, ptrdiff_t* n, Filters<V4>& f) {
    for (V4& v : f) v = hn::Zero(D4());
    *n = -N + 1;
    for (; *n < first_four(xsize); ++*n) SingleStep(in, out, xsize, *n, f);
    for (V4& v : f) v = hn::Broadcast<0>(v);
}

// Steps of four outputs from n while n < end. Lane k adds input k's share
// to each later output (the weights shifted by k lanes) and the two
// previous outputs' shares; the last two outputs become the next step's
// previous ones. In vectors of 8 lanes, each 128-bit block holds its own
// row: the weights are repeated in both blocks, and Broadcast and
// ShiftLeftLanes act within a block, so each block does the 4-lane
// operations of its row. `sum_at(n)` gives the step's inputs (left + right),
// `store_at(n, v)` takes its outputs.
template <class D, class SumAt, class StoreAt>
HWY_INLINE void StepsOfFour(D d, ptrdiff_t* n, ptrdiff_t end, Filters<hn::Vec<D>>& f, const SumAt& sum_at,
                            const StoreAt& store_at) {
    using V = hn::Vec<D>;
    const V mul_in_1 = hn::LoadDup128(d, kMulIn[0]);
    const V mul_in_3 = hn::LoadDup128(d, kMulIn[1]);
    const V mul_in_5 = hn::LoadDup128(d, kMulIn[2]);
    const V mul_prev_1 = hn::LoadDup128(d, kMulPrev[0]);
    const V mul_prev_3 = hn::LoadDup128(d, kMulPrev[1]);
    const V mul_prev_5 = hn::LoadDup128(d, kMulPrev[2]);
    const V mul_prev2_1 = hn::LoadDup128(d, kMulPrev2[0]);
    const V mul_prev2_3 = hn::LoadDup128(d, kMulPrev2[1]);
    const V mul_prev2_5 = hn::LoadDup128(d, kMulPrev2[2]);
    V prev_1 = f[0], prev_3 = f[1], prev_5 = f[2];
    V prev2_1 = f[3], prev2_3 = f[4], prev2_5 = f[5];
    for (; *n < end; *n += 4) {
        const V sum = sum_at(*n);
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
        store_at(*n, hn::Add(out_1, hn::Add(out_3, out_5)));
    }
    f[0] = prev_1, f[1] = prev_3, f[2] = prev_5;
    f[3] = prev2_1, f[4] = prev2_3, f[5] = prev2_5;
}

void BlurRow(const float* HWY_RESTRICT in, float* HWY_RESTRICT out, size_t width) {
    const D4 d;
    const auto xsize = static_cast<ptrdiff_t>(width);
    Filters<V4> f;
    ptrdiff_t n = 0;
    StartRow(in, out, xsize, &n, f);
    StepsOfFour(
        d, &n, past_fours(xsize), f,
        [&](ptrdiff_t at) { return hn::Add(hn::LoadU(d, in + at - N - 1), hn::LoadU(d, in + at + N - 1)); },
        [&](ptrdiff_t at, V4 v) { hn::StoreU(v, d, out + at); });
    for (; n < xsize; ++n) SingleStep(in, out, xsize, n, f);
}

// BlurRow of two rows of the same width at once: where vectors have 8
// lanes, row a in the lower 128-bit block and row b in the upper one (the
// steps of four are bound by the arithmetic, so this nearly halves the
// time); elsewhere one row after the other.
void BlurRows2(const float* HWY_RESTRICT in_a, float* HWY_RESTRICT out_a, const float* HWY_RESTRICT in_b,
               float* HWY_RESTRICT out_b, size_t width) {
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
    const D4 d4;
    const hn::CappedTag<float, 8> d8;
    using V8 = hn::Vec<decltype(d8)>;
    const auto xsize = static_cast<ptrdiff_t>(width);
    Filters<V4> fa, fb;
    ptrdiff_t n = 0;
    StartRow(in_a, out_a, xsize, &n, fa);
    StartRow(in_b, out_b, xsize, &n, fb);
    Filters<V8> f;
    for (size_t i = 0; i < 6; ++i) f[i] = hn::Combine(d8, fb[i], fa[i]);
    const auto row_sum = [&](const float* in, ptrdiff_t at) {
        return hn::Add(hn::LoadU(d4, in + at - N - 1), hn::LoadU(d4, in + at + N - 1));
    };
    StepsOfFour(
        d8, &n, past_fours(xsize), f, [&](ptrdiff_t at) { return hn::Combine(d8, row_sum(in_b, at), row_sum(in_a, at)); },
        [&](ptrdiff_t at, V8 v) {
            hn::StoreU(hn::LowerHalf(d4, v), d4, out_a + at);
            hn::StoreU(hn::UpperHalf(d4, v), d4, out_b + at);
        });
    for (size_t i = 0; i < 6; ++i) {
        fa[i] = hn::LowerHalf(d4, f[i]);
        fb[i] = hn::UpperHalf(d4, f[i]);
    }
    for (ptrdiff_t m = n; m < xsize; ++m) SingleStep(in_a, out_a, xsize, m, fa);
    for (; n < xsize; ++n) SingleStep(in_b, out_b, xsize, n, fb);
#else
    BlurRow(in_a, out_a, width);
    BlurRow(in_b, out_b, width);
#endif
}

// VerticalBlock for `steps` consecutive rows: each lane is one column, and
// a vector's filter state stays in registers from one step to the next.
// libjxl adds the zero row too (TwoInputs) or loads it alone (SingleInput);
// so does this.
using DF = hn::ScalableTag<float>;
using VF = hn::Vec<DF>;

// One vector of columns' filter state: the three filters' previous two
// outputs, kept in `state`'s 6 rows (prev_1, prev2_1, prev_3, prev2_3,
// prev_5, prev2_5) between calls.
struct ColumnFilters {
    VF y_n1_1, y_n2_1, y_n1_3, y_n2_3, y_n1_5, y_n2_5;

    static ColumnFilters load(const float* HWY_RESTRICT state, size_t count, size_t i) {
        const DF d;
        return {hn::Load(d, state + i),             hn::Load(d, state + count + i),
                hn::Load(d, state + 2 * count + i), hn::Load(d, state + 3 * count + i),
                hn::Load(d, state + 4 * count + i), hn::Load(d, state + 5 * count + i)};
    }
    void store(float* HWY_RESTRICT state, size_t count, size_t i) const {
        const DF d;
        hn::Store(y_n1_1, d, state + i);
        hn::Store(y_n2_1, d, state + count + i);
        hn::Store(y_n1_3, d, state + 2 * count + i);
        hn::Store(y_n2_3, d, state + 3 * count + i);
        hn::Store(y_n1_5, d, state + 4 * count + i);
        hn::Store(y_n2_5, d, state + 5 * count + i);
    }
    // One step: the filters' new outputs from the step's input; returns
    // their sum, the output row's value.
    HWY_INLINE VF step(VF sum) {
        const DF d;
        const VF y1 = hn::MulAdd(hn::Set(d, kN2[0]), sum, hn::NegMulSub(hn::Set(d, kD1[0]), y_n1_1, y_n2_1));
        const VF y3 = hn::MulAdd(hn::Set(d, kN2[1]), sum, hn::NegMulSub(hn::Set(d, kD1[1]), y_n1_3, y_n2_3));
        const VF y5 = hn::MulAdd(hn::Set(d, kN2[2]), sum, hn::NegMulSub(hn::Set(d, kD1[2]), y_n1_5, y_n2_5));
        y_n2_1 = y_n1_1, y_n2_3 = y_n1_3, y_n2_5 = y_n1_5;
        y_n1_1 = y1, y_n1_3 = y3, y_n1_5 = y5;
        return hn::Add(y1, hn::Add(y3, y5));
    }
};

// A step's input: libjxl adds the zero row too (TwoInputs) or loads the
// bottom row alone (SingleInput); so does this.
HWY_INLINE VF ColumnInput(const float* top, const float* bottom, size_t i) {
    const DF d;
    const VF below = hn::Load(d, bottom + i);
    return top ? hn::Add(hn::Load(d, top + i), below) : below;
}

// VerticalBlock for `steps` consecutive rows: each lane is one column, and
// a vector's filter state stays in registers from one step to the next.
void BlurColumnSteps(const float* const* tops, const float* const* bottoms, float* HWY_RESTRICT state,
                     float* const* outs, size_t steps, size_t count) {
    const size_t lanes = hn::Lanes(DF());
    for (size_t i = 0; i < count; i += lanes) {
        ColumnFilters f = ColumnFilters::load(state, count, i);
        for (size_t k = 0; k < steps; ++k) {
            const VF value = f.step(ColumnInput(tops[k], bottoms[k], i));
            if (outs[k]) hn::Store(value, DF(), outs[k] + i);
        }
        f.store(state, count, i);
    }
}

// ---- The maps' sums ----
//
// libjxl's SSIMMap and EdgeDiffMap are scalar code. Here each pixel's terms
// are computed in vectors for a chunk of the row, then added to the sums in
// scalar code in the pixels' order. Each term rounds as libjxl's: in float
// where libjxl computes in float, in double from its first double operation
// on, and max as std::max (v unless v < 0, so NaN and -0 stay). Each sum
// adds the same values in the same order.
constexpr size_t kChunk = 256;  // pixels per chunk; its terms stay in the L1 cache

// std::max(v, 0.0).
template <class D>
HWY_INLINE hn::Vec<D> MaxWithZero(D d, hn::Vec<D> v) {
    const hn::Vec<D> zero = hn::Zero(d);
    return hn::IfThenElse(hn::Lt(v, zero), zero, v);
}

template <class V>
HWY_INLINE V Quartic(V x) {
    x = hn::Mul(x, x);
    return hn::Mul(x, x);
}

void AddSsimRow(const float* HWY_RESTRICT m1, const float* HWY_RESTRICT m2, const float* HWY_RESTRICT s11,
                const float* HWY_RESTRICT s22, const float* HWY_RESTRICT s12, size_t width, float c2,
                double* HWY_RESTRICT sums) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<float, decltype(dd)> df;  // as many floats as dd has doubles
    const size_t lanes = hn::Lanes(dd);
    const auto one = hn::Set(dd, 1.0);
    const auto two = hn::Set(df, 2.0f);
    const auto c2_vector = hn::Set(df, c2);
    HWY_ALIGN double terms[kChunk];
    HWY_ALIGN double quartics[kChunk];
    double sum_mean = sums[0];
    double sum_quartic = sums[1];
    for (size_t x0 = 0; x0 < width; x0 += kChunk) {
        const size_t count = std::min(kChunk, width - x0);
        for (size_t i = 0; i < count; i += lanes) {
            const size_t x = x0 + i;
            const auto mu1 = hn::LoadU(df, m1 + x);
            const auto mu2 = hn::LoadU(df, m2 + x);
            const auto mu11 = hn::Mul(mu1, mu1);
            const auto mu22 = hn::Mul(mu2, mu2);
            const auto mu12 = hn::Mul(mu1, mu2);
            // As in libjxl: the square in float, the subtraction in double.
            const auto diff = hn::Sub(mu1, mu2);
            const auto num_m = hn::DemoteTo(df, hn::Sub(one, hn::PromoteTo(dd, hn::Mul(diff, diff))));
            const auto num_s = hn::Add(hn::Mul(two, hn::Sub(hn::LoadU(df, s12 + x), mu12)), c2_vector);
            const auto denom_s = hn::Add(
                hn::Add(hn::Sub(hn::LoadU(df, s11 + x), mu11), hn::Sub(hn::LoadU(df, s22 + x), mu22)), c2_vector);
            const auto d =
                MaxWithZero(dd, hn::Sub(one, hn::PromoteTo(dd, hn::Div(hn::Mul(num_m, num_s), denom_s))));
            hn::Store(d, dd, terms + i);
            hn::Store(Quartic(d), dd, quartics + i);
        }
        for (size_t i = 0; i < count; ++i) {
            sum_mean += terms[i];
            sum_quartic += quartics[i];
        }
    }
    sums[0] = sum_mean;
    sums[1] = sum_quartic;
}

void AddEdgeRow(const float* HWY_RESTRICT img1, const float* HWY_RESTRICT mu1, const float* HWY_RESTRICT img2,
                const float* HWY_RESTRICT mu2, size_t width, double* HWY_RESTRICT sums) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<float, decltype(dd)> df;
    const size_t lanes = hn::Lanes(dd);
    const auto one = hn::Set(dd, 1.0);
    HWY_ALIGN double artifacts[kChunk];
    HWY_ALIGN double artifact_quartics[kChunk];
    HWY_ALIGN double details_lost[kChunk];
    HWY_ALIGN double detail_lost_quartics[kChunk];
    double artifact_sum = sums[0];
    double artifact_quartic = sums[1];
    double detail_lost_sum = sums[2];
    double detail_lost_quartic = sums[3];
    for (size_t x0 = 0; x0 < width; x0 += kChunk) {
        const size_t count = std::min(kChunk, width - x0);
        for (size_t i = 0; i < count; i += lanes) {
            const size_t x = x0 + i;
            // The differences in float, the rest in double.
            const auto test = hn::PromoteTo(dd, hn::Abs(hn::Sub(hn::LoadU(df, img2 + x), hn::LoadU(df, mu2 + x))));
            const auto source = hn::PromoteTo(dd, hn::Abs(hn::Sub(hn::LoadU(df, img1 + x), hn::LoadU(df, mu1 + x))));
            const auto d1 = hn::Sub(hn::Div(hn::Add(one, test), hn::Add(one, source)), one);
            const auto artifact = MaxWithZero(dd, d1);
            const auto detail_lost = MaxWithZero(dd, hn::Neg(d1));
            hn::Store(artifact, dd, artifacts + i);
            hn::Store(Quartic(artifact), dd, artifact_quartics + i);
            hn::Store(detail_lost, dd, details_lost + i);
            hn::Store(Quartic(detail_lost), dd, detail_lost_quartics + i);
        }
        for (size_t i = 0; i < count; ++i) {
            artifact_sum += artifacts[i];
            artifact_quartic += artifact_quartics[i];
            detail_lost_sum += details_lost[i];
            detail_lost_quartic += detail_lost_quartics[i];
        }
    }
    sums[0] = artifact_sum;
    sums[1] = artifact_quartic;
    sums[2] = detail_lost_sum;
    sums[3] = detail_lost_quartic;
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
HWY_EXPORT(BlurRows2);
HWY_EXPORT(BlurColumnSteps);
HWY_EXPORT(AddSsimRow);
HWY_EXPORT(AddEdgeRow);

bool fused_multiply_add() { return HWY_DYNAMIC_DISPATCH(FusedMultiplyAdd)(); }

void positive_xyb_row(const float* red, const float* green, const float* blue, const float* opsin, float* x,
                      float* y, float* b, std::size_t count) {
    HWY_DYNAMIC_DISPATCH(PositiveXybRow)(red, green, blue, opsin, x, y, b, count);
}

void multiply_row(const float* a, const float* b, float* out, std::size_t count) {
    HWY_DYNAMIC_DISPATCH(MultiplyRow)(a, b, out, count);
}

void blur_row(const float* in, float* out, std::size_t width) { HWY_DYNAMIC_DISPATCH(BlurRow)(in, out, width); }

void blur_rows2(const float* in_a, float* out_a, const float* in_b, float* out_b, std::size_t width) {
    HWY_DYNAMIC_DISPATCH(BlurRows2)(in_a, out_a, in_b, out_b, width);
}

void blur_column_steps(const float* const* tops, const float* const* bottoms, float* state, float* const* outs,
                       std::size_t steps, std::size_t count) {
    HWY_DYNAMIC_DISPATCH(BlurColumnSteps)(tops, bottoms, state, outs, steps, count);
}

void add_ssim_row(const float* m1, const float* m2, const float* s11, const float* s22, const float* s12,
                  std::size_t width, float c2, double* sums) {
    HWY_DYNAMIC_DISPATCH(AddSsimRow)(m1, m2, s11, s22, s12, width, c2, sums);
}

void add_edge_row(const float* img1, const float* mu1, const float* img2, const float* mu2, std::size_t width,
                  double* sums) {
    HWY_DYNAMIC_DISPATCH(AddEdgeRow)(img1, mu1, img2, mu2, width, sums);
}

}  // namespace vml::ssimulacra2_kernels
#endif  // HWY_ONCE
