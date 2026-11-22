// Butteraugli's vector kernels, ported from libjxl 0.12
// (lib/jxl/butteraugli/butteraugli.cc, lib/jxl/enc_convolve_separable5.cc,
// lib/jxl/convolve-inl.h, lib/jxl/base/fast_math-inl.h and
// rational_polynomial-inl.h; BSD licence, docs/THIRD_PARTY.md) with the same
// Highway operations in the same order, so that every value rounds as in
// libjxl's butteraugli_main. Highway compiles this file once per instruction
// set (foreach_target.h includes it again for each) and HWY_DYNAMIC_DISPATCH
// picks the best one for the CPU, as in libjxl.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

#include "butteraugli_image.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "butteraugli_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

#ifndef VML_BUTTERAUGLI_KERNELS_ONCE
#define VML_BUTTERAUGLI_KERNELS_ONCE
#define BUTTERAUGLI_RESTRICT HWY_RESTRICT
#define BUTTERAUGLI_INLINE HWY_INLINE
namespace vml::butteraugli {
// Lets PaddedMaltaUnit call either kernel by overloading.
struct MaltaTagLF {};
struct MaltaTag {};
// libjxl's kInvLog2e (lib/jxl/base/common.h), named after its use: ln(2).
constexpr float kInvLog2e = 0.6931471805599453;
}  // namespace vml::butteraugli
#endif

#ifdef _MSC_VER
// libjxl's code as written: double constants passed to Set() for float
// vectors, sizes converted to int, parameters it does not use.
#pragma warning(push)
#pragma warning(disable : 4100 4244 4267 4305)
#endif

HWY_BEFORE_NAMESPACE();
namespace vml::butteraugli {
namespace HWY_NAMESPACE {
using namespace hwy::HWY_NAMESPACE;

// libjxl's scalar code in this file is compiled by clang once per target
// like the vector code. In the official Windows tools, whose scores the app
// matches there, clang fuses a * b + c within one expression wherever the
// target has FMA (its default -ffp-contract=on; the left product first:
// a * b + c * d is fma(a, b, c * d)). MSVC never fuses, so each place clang
// fuses is written with this: one fused rounding on AVX2, two roundings on
// SSE2 and SSE4, as there. On the Mac, the app's libjxl tools are built with
// -ffp-contract=off (scripts/build_libjxl_macos.sh), as the app is, so
// nothing is fused there.
template <typename T>
HWY_INLINE T ScalarMulAdd(T a, T b, T c) {
#if HWY_NATIVE_FMA && defined(_WIN32)
    const CappedTag<T, 1> d;
    return GetLane(MulAdd(Set(d, a), Set(d, b), Set(d, c)));
#else
    return a * b + c;
#endif
}

// ScalarMulAdd for a vector of pixels: each lane rounds as ScalarMulAdd's
// value, for the loops VideoMetricsLab vectorised.
template <class V>
HWY_INLINE V ContractedMulAdd(V a, V b, V c) {
#if HWY_NATIVE_FMA && defined(_WIN32)
    return MulAdd(a, b, c);
#else
    return Add(Mul(a, b), c);
#endif
}

// libjxl's EvalRationalPolynomial for the (2, 2) degrees FastLog2f uses:
// Horner steps with MulAdd, then an exact division.
template <class D, class V>
HWY_INLINE V EvalRationalPolynomial22(const D d, const V x, const float (&p)[12], const float (&q)[12]) {
    auto yp = LoadDup128(d, &p[8]);
    auto yq = LoadDup128(d, &q[8]);
    yp = MulAdd(yp, x, LoadDup128(d, p + 4));
    yq = MulAdd(yq, x, LoadDup128(d, q + 4));
    yp = MulAdd(yp, x, LoadDup128(d, p));
    yq = MulAdd(yq, x, LoadDup128(d, q));
    return Div(yp, yq);
}

// libjxl's FastLog2f: log2(x) from the exponent and a rational polynomial of
// the mantissa reduced to [-1/3, 1/3].
template <class DF, class V>
HWY_INLINE V FastLog2f(const DF df, V x) {
    HWY_ALIGN const float p[4 * 3] = {HWY_REP4(-1.8503833400518310E-06f), HWY_REP4(1.4287160470083755E+00f),
                                      HWY_REP4(7.4245873327820566E-01f)};
    HWY_ALIGN const float q[4 * 3] = {HWY_REP4(9.9032814277590719E-01f), HWY_REP4(1.0096718572241148E+00f),
                                      HWY_REP4(1.7409343003366853E-01f)};
    const Rebind<int32_t, DF> di;
    const auto x_bits = BitCast(di, x);
    const auto exp_bits = Sub(x_bits, Set(di, 0x3f2aaaab));  // 2/3
    const auto exp_shifted = ShiftRight<23>(exp_bits);
    const auto mantissa = BitCast(df, Sub(x_bits, ShiftLeft<23>(exp_shifted)));
    const auto exp_val = ConvertTo(df, exp_shifted);
    return Add(EvalRationalPolynomial22(df, Sub(mantissa, Set(df, 1.0f)), p, q), exp_val);
}

// libjxl's Separable5 (enc_convolve_separable5.cc) for one row: a 5x5
// separable blur with mirrored borders. VideoMetricsLab: the five input rows
// (y - 2 to y + 2, mirrored at the picture's top and bottom: the rows
// libjxl's border code, and its table for pictures of 4 rows or fewer, pick)
// are passed in. Rows are convolved horizontally with vectors; the last
// columns of rows whose width is not a whole number of vectors (from x = the
// end of the vector loop) are computed by a scalar 2D sum, as libjxl does,
// so the vector width decides which columns those are.
class Separable5Row {
public:
    using D = HWY_CAPPED(float, 16);
    using DI32 = HWY_CAPPED(int32_t, 16);
    using V = Vec<D>;
    using VI32 = Vec<DI32>;
    using I = decltype(SetTableIndices(D(), static_cast<int32_t*>(nullptr)));
    static constexpr int64_t kRadius = 2;

    static void Run(const float* const rows[5], const size_t xsize, const WeightsSeparable5& weights,
                    float* const row_out) {
        const size_t min_width = Lanes(D()) + kRadius;
        if (xsize < min_width) {
            slow_separable5_row(rows, xsize, weights, row_out);
            return;
        }
        switch (xsize % Lanes(D())) {
        case 0:
            ConvolveRow<0>(rows, xsize, weights, row_out);
            break;
        case 1:
            ConvolveRow<1>(rows, xsize, weights, row_out);
            break;
        case 2:
            ConvolveRow<2>(rows, xsize, weights, row_out);
            break;
        default:
            ConvolveRow<3>(rows, xsize, weights, row_out);
            break;
        }
    }

private:
    template <size_t kSizeModN>
    static void ConvolveRow(const float* const rows[5], const size_t xsize, const WeightsSeparable5& weights_,
                            float* const row_out) {
        const D d;
        const float* const row_t2 = rows[0];
        const float* const row_t1 = rows[1];
        const float* const row_m = rows[2];
        const float* const row_b1 = rows[3];
        const float* const row_b2 = rows[4];
        const V wh0 = LoadDup128(d, weights_.horz + 0 * 4);
        const V wh1 = LoadDup128(d, weights_.horz + 1 * 4);
        const V wh2 = LoadDup128(d, weights_.horz + 2 * 4);
        const V wv0 = LoadDup128(d, weights_.vert + 0 * 4);
        const V wv1 = LoadDup128(d, weights_.vert + 1 * 4);
        const V wv2 = LoadDup128(d, weights_.vert + 2 * 4);
        const I ml1 = MirrorLanes<1>();
        const I ml2 = MirrorLanes<2>();

        // Rows combined vertically as libjxl does: middle * w0, then
        // (top1 + bottom1) * w1 added, then (top2 + bottom2) * w2.
        const auto vertical = [&](V m, V t1, V b1, V t2, V b2) HWY_ATTR {
            const V conv0 = Mul(m, wv0);
            const V conv1 = MulAdd(Add(t1, b1), wv1, conv0);
            return MulAdd(Add(t2, b2), wv2, conv1);
        };
        size_t x = 0;
        for (; x < kRadius; x += Lanes(d)) {
            const auto h = [&](const float* row) HWY_ATTR { return HorzConvolveFirst(row, x, wh0, wh1, wh2); };
            Store(vertical(h(row_m), h(row_t1), h(row_b1), h(row_t2), h(row_b2)), d, row_out + x);
        }
        for (; x + Lanes(d) + kRadius <= xsize; x += Lanes(d)) {
            const auto h = [&](const float* row) HWY_ATTR { return HorzConvolve(row + x, wh0, wh1, wh2); };
            Store(vertical(h(row_m), h(row_t1), h(row_b1), h(row_t2), h(row_b2)), d, row_out + x);
        }
        if constexpr (kSizeModN < kRadius) {
            const auto h = [&](const float* row) HWY_ATTR {
                return HorzConvolveLast<kSizeModN>(row, x, xsize, wh0, wh1, wh2, ml1, ml2);
            };
            Store(vertical(h(row_m), h(row_t1), h(row_b1), h(row_t2), h(row_b2)), d, row_out + x);
            x += Lanes(d);
        }
        if constexpr (kSizeModN != 0) {
            for (; x < xsize; ++x) {
                float mul = 0.0f;
                for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
                    const float wy = weights_.vert[std::abs(dy) * 4];
                    const float* clamped_row = rows[dy + 2];
                    for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
                        const float wx = weights_.horz[std::abs(dx) * 4];
                        const int64_t clamped_x = Mirror(static_cast<int64_t>(x) + dx, static_cast<int64_t>(xsize));
                        mul = ScalarMulAdd(clamped_row[clamped_x] * wx, wy, mul);  // as clang fuses `mul += a * wx * wy`
                    }
                }
                row_out[x] = mul;
            }
        }
    }

    // Indices that make TableLookupLanes on the rightmost vector (its last
    // lane the row's last sample) return the samples mirrored past it.
    template <int M>
    static I MirrorLanes() {
        const D d;
        const DI32 di32;
        const VI32 up = Min(Iota(di32, M), Set(di32, static_cast<int32_t>(Lanes(d)) - 1));
        const VI32 down = Max(Iota(di32, M - static_cast<int>(Lanes(d))), Zero(di32));
        return IndicesFromVec(d, Sub(up, down));
    }

    // The first vector's left neighbours, mirrored: l1[i] = c[Mirror(i - 1)],
    // l2[i] = c[Mirror(i - 2)] (libjxl's Neighbors::FirstL1 / FirstL2).
    static V FirstL1(const V c) {
        const D d;
        HWY_ALIGN static constexpr int32_t lanes[16] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
        return TableLookupLanes(c, SetTableIndices(d, lanes));
    }
    static V FirstL2(const V c) {
        const D d;
        HWY_ALIGN static constexpr int32_t lanes[16] = {1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
        return TableLookupLanes(c, SetTableIndices(d, lanes));
    }

    // Horizontal 5-tap sums, as libjxl's HorzConvolve*: center * w0, then
    // (left1 + right1) * w1 added, then (left2 + right2) * w2.
    static V HorzConvolveFirst(const float* row, const size_t x, const V wh0, const V wh1, const V wh2) {
        const D d;
        const V c = LoadU(d, row + x);
        const V mul0 = Mul(c, wh0);
        const V l1 = FirstL1(c);
        const V l2 = FirstL2(c);
        const V r1 = LoadU(d, row + x + 1);
        const V r2 = LoadU(d, row + x + 2);
        const V mul1 = MulAdd(Add(l1, r1), wh1, mul0);
        return MulAdd(Add(l2, r2), wh2, mul1);
    }

    template <size_t kSizeModN>
    static V HorzConvolveLast(const float* row, const size_t x, const size_t xsize, const V wh0, const V wh1,
                              const V wh2, const I ml1, const I ml2) {
        const D d;
        const V c = LoadU(d, row + x);
        const V mul0 = Mul(c, wh0);
        const V l1 = LoadU(d, row + x - 1);
        const V l2 = LoadU(d, row + x - 2);
        V r1;
        V r2;
        if constexpr (kSizeModN == 0) {
            r2 = TableLookupLanes(c, ml2);
            r1 = TableLookupLanes(c, ml1);
        } else {  // == 1
            const auto last = LoadU(d, row + xsize - Lanes(d));
            r2 = TableLookupLanes(last, ml1);
            r1 = last;
        }
        const V mul1 = MulAdd(Add(l1, r1), wh1, mul0);
        return MulAdd(Add(l2, r2), wh2, mul1);
    }

    static V HorzConvolve(const float* pos, const V wh0, const V wh1, const V wh2) {
        const D d;
        const V c = LoadU(d, pos);
        const V mul0 = Mul(c, wh0);
        const V l1 = LoadU(d, pos - 1);
        const V r1 = LoadU(d, pos + 1);
        const V l2 = LoadU(d, pos - 2);
        const V r2 = LoadU(d, pos + 2);
        const V mul1 = MulAdd(Add(l1, r1), wh1, mul0);
        return MulAdd(Add(l2, r2), wh2, mul1);
    }
};

// ---- libjxl's butteraugli.cc, its Highway section ----

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Div;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::MulSub;
using hwy::HWY_NAMESPACE::Neg;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Vec;
using hwy::HWY_NAMESPACE::ZeroIfNegative;

template <class D, class V>
HWY_INLINE V MaximumClamp(D d, V v, double kMaxVal) {
  static const double kMul = 0.724216145665;
  const V mul = Set(d, kMul);
  const V maxval = Set(d, kMaxVal);
  // If greater than maxval or less than -maxval, replace with if_*.
  const V if_pos = MulAdd(Sub(v, maxval), mul, maxval);
  const V if_neg = MulSub(Add(v, maxval), mul, maxval);
  const V pos_or_v = IfThenElse(Ge(v, maxval), if_pos, v);
  return IfThenElse(Lt(v, Neg(maxval)), if_neg, pos_or_v);
}

// Make area around zero less important (remove it).
template <class D, class V>
HWY_INLINE V RemoveRangeAroundZero(const D d, const double kw, const V x) {
  const auto w = Set(d, kw);
  return IfThenElse(Gt(x, w), Sub(x, w),
                    IfThenElseZero(Lt(x, Neg(w)), Add(x, w)));
}

// Make area around zero more important (2x it until the limit).
template <class D, class V>
HWY_INLINE V AmplifyRangeAroundZero(const D d, const double kw, const V x) {
  const auto w = Set(d, kw);
  return IfThenElse(Gt(x, w), Add(x, w),
                    IfThenElse(Lt(x, Neg(w)), Sub(x, w), Add(x, x)));
}

// XybLowFreqToVals converts from low-frequency XYB space to the 'vals' space.
// Vals space can be converted to L2-norm space (Euclidean and normalized)
// through visual masking.
template <class D, class V>
HWY_INLINE void XybLowFreqToVals(const D d, const V& x, const V& y,
                                 const V& b_arg, V* HWY_RESTRICT valx,
                                 V* HWY_RESTRICT valy, V* HWY_RESTRICT valb) {
  static const double xmul_scalar = 33.832837186260;
  static const double ymul_scalar = 14.458268100570;
  static const double bmul_scalar = 49.87984651440;
  static const double y_to_b_mul_scalar = -0.362267051518;
  const V xmul = Set(d, xmul_scalar);
  const V ymul = Set(d, ymul_scalar);
  const V bmul = Set(d, bmul_scalar);
  const V y_to_b_mul = Set(d, y_to_b_mul_scalar);
  const V b = MulAdd(y_to_b_mul, y, b_arg);
  *valb = Mul(b, bmul);
  *valx = Mul(x, xmul);
  *valy = Mul(y, ymul);
}

// libjxl's SuppressXByY for one vector: hf's X scaled down where hf's Y is
// large ("suppress red-green by intensity change in the high freq channels").
template <class D, class V>
HWY_INLINE V SuppressXByY(const D d, const V vx, const V vy) {
  static const double suppress = 46.0;
  static const double s = 0.653020556257;
  const auto sv = Set(d, s);
  const auto one_minus_s = Set(d, 1.0 - s);
  const auto ywv = Set(d, suppress);
  const auto scaler = MulAdd(Div(ywv, MulAdd(vy, vy, ywv)), one_minus_s, sv);
  return Mul(scaler, vx);
}

namespace {
template <typename V>
BUTTERAUGLI_INLINE V Sum(V a, V b, V c, V d) {
  return Add(Add(a, b), Add(c, d));
}
template <typename V>
BUTTERAUGLI_INLINE V Sum(V a, V b, V c, V d, V e) {
  return Sum(a, b, c, Add(d, e));
}
template <typename V>
BUTTERAUGLI_INLINE V Sum(V a, V b, V c, V d, V e, V f, V g) {
  return Sum(a, b, c, Sum(d, e, f, g));
}
template <typename V>
BUTTERAUGLI_INLINE V Sum(V a, V b, V c, V d, V e, V f, V g, V h, V i) {
  return Add(Add(Sum(a, b, c, d), Sum(e, f, g, h)), i);
}
}  // namespace

template <class D>
Vec<D> MaltaUnit(MaltaTagLF /*tag*/, const D df,
                 const float* BUTTERAUGLI_RESTRICT d, const ptrdiff_t xs) {
  const ptrdiff_t xs3 = 3 * xs;

  const auto center = LoadU(df, d);

  // x grows, y constant
  const auto sum_yconst = Sum(LoadU(df, d - 4), LoadU(df, d - 2), center,
                              LoadU(df, d + 2), LoadU(df, d + 4));
  // Will return this, sum of all line kernels
  auto retval = Mul(sum_yconst, sum_yconst);
  {
    // y grows, x constant
    auto sum = Sum(LoadU(df, d - xs3 - xs), LoadU(df, d - xs - xs), center,
                   LoadU(df, d + xs + xs), LoadU(df, d + xs3 + xs));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // both grow
    auto sum = Sum(LoadU(df, d - xs3 - 3), LoadU(df, d - xs - xs - 2), center,
                   LoadU(df, d + xs + xs + 2), LoadU(df, d + xs3 + 3));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // y grows, x shrinks
    auto sum = Sum(LoadU(df, d - xs3 + 3), LoadU(df, d - xs - xs + 2), center,
                   LoadU(df, d + xs + xs - 2), LoadU(df, d + xs3 - 3));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    auto sum =
        Sum(LoadU(df, d - xs3 - xs + 1), LoadU(df, d - xs - xs + 1), center,
            LoadU(df, d + xs + xs - 1), LoadU(df, d + xs3 + xs - 1));
    retval = MulAdd(sum, sum, retval);
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    auto sum =
        Sum(LoadU(df, d - xs3 - xs - 1), LoadU(df, d - xs - xs - 1), center,
            LoadU(df, d + xs + xs + 1), LoadU(df, d + xs3 + xs + 1));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    auto sum = Sum(LoadU(df, d - 4 - xs), LoadU(df, d - 2 - xs), center,
                   LoadU(df, d + 2 + xs), LoadU(df, d + 4 + xs));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    auto sum = Sum(LoadU(df, d - 4 + xs), LoadU(df, d - 2 + xs), center,
                   LoadU(df, d + 2 - xs), LoadU(df, d + 4 - xs));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7______*__
       8_________ */
    auto sum = Sum(LoadU(df, d - xs3 - 2), LoadU(df, d - xs - xs - 1), center,
                   LoadU(df, d + xs + xs + 1), LoadU(df, d + xs3 + 2));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7__*______
       8_________ */
    auto sum = Sum(LoadU(df, d - xs3 + 2), LoadU(df, d - xs - xs + 1), center,
                   LoadU(df, d + xs + xs - 1), LoadU(df, d + xs3 - 2));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__*______
       4____0____
       5______*__
       6_______*_
       7_________
       8_________ */
    auto sum = Sum(LoadU(df, d - xs - xs - 3), LoadU(df, d - xs - 2), center,
                   LoadU(df, d + xs + 2), LoadU(df, d + xs + xs + 3));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_______*_
       3______*__
       4____0____
       5__*______
       6_*_______
       7_________
       8_________ */
    auto sum = Sum(LoadU(df, d - xs - xs + 3), LoadU(df, d - xs + 2), center,
                   LoadU(df, d + xs - 2), LoadU(df, d + xs + xs - 3));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2________*
       3______*__
       4____0____
       5__*______
       6*________
       7_________
       8_________ */

    auto sum = Sum(LoadU(df, d + xs + xs - 4), LoadU(df, d + xs - 2), center,
                   LoadU(df, d - xs + 2), LoadU(df, d - xs - xs + 4));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2*________
       3__*______
       4____0____
       5______*__
       6________*
       7_________
       8_________ */
    auto sum = Sum(LoadU(df, d - xs - xs - 4), LoadU(df, d - xs - 2), center,
                   LoadU(df, d + xs + 2), LoadU(df, d + xs + xs + 4));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0__*______
       1_________
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7_________
       8______*__ */
    auto sum =
        Sum(LoadU(df, d - xs3 - xs - 2), LoadU(df, d - xs - xs - 1), center,
            LoadU(df, d + xs + xs + 1), LoadU(df, d + xs3 + xs + 2));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0______*__
       1_________
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7_________
       8__*______ */
    auto sum =
        Sum(LoadU(df, d - xs3 - xs + 2), LoadU(df, d - xs - xs + 1), center,
            LoadU(df, d + xs + xs - 1), LoadU(df, d + xs3 + xs - 2));
    retval = MulAdd(sum, sum, retval);
  }
  return retval;
}

template <class D>
Vec<D> MaltaUnit(MaltaTag /*tag*/, const D df,
                 const float* BUTTERAUGLI_RESTRICT d, const ptrdiff_t xs) {
  const ptrdiff_t xs3 = 3 * xs;

  const auto center = LoadU(df, d);

  // x grows, y constant
  const auto sum_yconst =
      Sum(LoadU(df, d - 4), LoadU(df, d - 3), LoadU(df, d - 2),
          LoadU(df, d - 1), center, LoadU(df, d + 1), LoadU(df, d + 2),
          LoadU(df, d + 3), LoadU(df, d + 4));
  // Will return this, sum of all line kernels
  auto retval = Mul(sum_yconst, sum_yconst);

  {
    // y grows, x constant
    auto sum = Sum(LoadU(df, d - xs3 - xs), LoadU(df, d - xs3),
                   LoadU(df, d - xs - xs), LoadU(df, d - xs), center,
                   LoadU(df, d + xs), LoadU(df, d + xs + xs),
                   LoadU(df, d + xs3), LoadU(df, d + xs3 + xs));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // both grow
    auto sum = Sum(LoadU(df, d - xs3 - 3), LoadU(df, d - xs - xs - 2),
                   LoadU(df, d - xs - 1), center, LoadU(df, d + xs + 1),
                   LoadU(df, d + xs + xs + 2), LoadU(df, d + xs3 + 3));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // y grows, x shrinks
    auto sum = Sum(LoadU(df, d - xs3 + 3), LoadU(df, d - xs - xs + 2),
                   LoadU(df, d - xs + 1), center, LoadU(df, d + xs - 1),
                   LoadU(df, d + xs + xs - 2), LoadU(df, d + xs3 - 3));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    auto sum = Sum(LoadU(df, d - xs3 - xs + 1), LoadU(df, d - xs3 + 1),
                   LoadU(df, d - xs - xs + 1), LoadU(df, d - xs), center,
                   LoadU(df, d + xs), LoadU(df, d + xs + xs - 1),
                   LoadU(df, d + xs3 - 1), LoadU(df, d + xs3 + xs - 1));
    retval = MulAdd(sum, sum, retval);
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    auto sum = Sum(LoadU(df, d - xs3 - xs - 1), LoadU(df, d - xs3 - 1),
                   LoadU(df, d - xs - xs - 1), LoadU(df, d - xs), center,
                   LoadU(df, d + xs), LoadU(df, d + xs + xs + 1),
                   LoadU(df, d + xs3 + 1), LoadU(df, d + xs3 + xs + 1));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    auto sum =
        Sum(LoadU(df, d - 4 - xs), LoadU(df, d - 3 - xs), LoadU(df, d - 2 - xs),
            LoadU(df, d - 1), center, LoadU(df, d + 1), LoadU(df, d + 2 + xs),
            LoadU(df, d + 3 + xs), LoadU(df, d + 4 + xs));
    retval = MulAdd(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    auto sum =
        Sum(LoadU(df, d - 4 + xs), LoadU(df, d - 3 + xs), LoadU(df, d - 2 + xs),
            LoadU(df, d - 1), center, LoadU(df, d + 1), LoadU(df, d + 2 - xs),
            LoadU(df, d + 3 - xs), LoadU(df, d + 4 - xs));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3___*_____
       4____0____
       5_____*___
       6_____*___
       7______*__
       8_________ */
    auto sum = Sum(LoadU(df, d - xs3 - 2), LoadU(df, d - xs - xs - 1),
                   LoadU(df, d - xs - 1), center, LoadU(df, d + xs + 1),
                   LoadU(df, d + xs + xs + 1), LoadU(df, d + xs3 + 2));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_____*___
       4____0____
       5___*_____
       6___*_____
       7__*______
       8_________ */
    auto sum = Sum(LoadU(df, d - xs3 + 2), LoadU(df, d - xs - xs + 1),
                   LoadU(df, d - xs + 1), center, LoadU(df, d + xs - 1),
                   LoadU(df, d + xs + xs - 1), LoadU(df, d + xs3 - 2));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__**_____
       4____0____
       5_____**__
       6_______*_
       7_________
       8_________ */
    auto sum = Sum(LoadU(df, d - xs - xs - 3), LoadU(df, d - xs - 2),
                   LoadU(df, d - xs - 1), center, LoadU(df, d + xs + 1),
                   LoadU(df, d + xs + 2), LoadU(df, d + xs + xs + 3));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_______*_
       3_____**__
       4____0____
       5__**_____
       6_*_______
       7_________
       8_________ */
    auto sum = Sum(LoadU(df, d - xs - xs + 3), LoadU(df, d - xs + 2),
                   LoadU(df, d - xs + 1), center, LoadU(df, d + xs - 1),
                   LoadU(df, d + xs - 2), LoadU(df, d + xs + xs - 3));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_________
       3______***
       4___*0*___
       5***______
       6_________
       7_________
       8_________ */

    auto sum =
        Sum(LoadU(df, d + xs - 4), LoadU(df, d + xs - 3), LoadU(df, d + xs - 2),
            LoadU(df, d - 1), center, LoadU(df, d + 1), LoadU(df, d - xs + 2),
            LoadU(df, d - xs + 3), LoadU(df, d - xs + 4));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_________
       3***______
       4___*0*___
       5______***
       6_________
       7_________
       8_________ */
    auto sum =
        Sum(LoadU(df, d - xs - 4), LoadU(df, d - xs - 3), LoadU(df, d - xs - 2),
            LoadU(df, d - 1), center, LoadU(df, d + 1), LoadU(df, d + xs + 2),
            LoadU(df, d + xs + 3), LoadU(df, d + xs + 4));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0___*_____
       1___*_____
       2___*_____
       3____*____
       4____0____
       5____*____
       6_____*___
       7_____*___
       8_____*___ */
    auto sum = Sum(LoadU(df, d - xs3 - xs - 1), LoadU(df, d - xs3 - 1),
                   LoadU(df, d - xs - xs - 1), LoadU(df, d - xs), center,
                   LoadU(df, d + xs), LoadU(df, d + xs + xs + 1),
                   LoadU(df, d + xs3 + 1), LoadU(df, d + xs3 + xs + 1));
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_____*___
       1_____*___
       2____ *___
       3____*____
       4____0____
       5____*____
       6___*_____
       7___*_____
       8___*_____ */
    auto sum = Sum(LoadU(df, d - xs3 - xs + 1), LoadU(df, d - xs3 + 1),
                   LoadU(df, d - xs - xs + 1), LoadU(df, d - xs), center,
                   LoadU(df, d + xs), LoadU(df, d + xs + xs - 1),
                   LoadU(df, d + xs3 - 1), LoadU(df, d + xs3 + xs - 1));
    retval = MulAdd(sum, sum, retval);
  }
  return retval;
}

// Returns MaltaUnit. Avoids bounds-checks when x0 and y0 are known
// to be far enough from the image borders. "diffs" is a packed image.
// libjxl's MaltaDiffMapT difference of one pixel: row0's value r0 against
// row1's r1.
HWY_INLINE float MaltaDiffPixel(const float r0, const float r1, const double norm1, const float norm2_0gt1,
                                const float norm2_0lt1) {
  const float absval = 0.5f * (std::abs(r0) + std::abs(r1));
  const float diff = r0 - r1;
  const float scaler = norm2_0gt1 / (static_cast<float>(norm1) + absval);

  // Primary symmetric quadratic objective.
  float out = scaler * diff;

  const float scaler2 = norm2_0lt1 / (static_cast<float>(norm1) + absval);
  const double fabs0 = std::fabs(r0);

  // Secondary half-open quadratic objectives.
  const double too_small = 0.55 * fabs0;
  const double too_big = 1.05 * fabs0;

  if (r0 < 0) {
    if (r1 > -too_small) {
      double impact = scaler2 * (r1 + too_small);
      out -= impact;
    } else if (r1 < -too_big) {
      double impact = scaler2 * (-r1 - too_big);
      out += impact;
    }
  } else {
    if (r1 < too_small) {
      double impact = scaler2 * (too_small - r1);
      out += impact;
    } else if (r1 > too_big) {
      double impact = scaler2 * (r1 - too_big);
      out -= impact;
    }
  }
  return out;
}

// VideoMetricsLab: MaltaDiffPixel's half-open objectives for a vector of
// pixels, in doubles as there (primary is the symmetric objective, s2 the
// second scaler, both widened from float exactly). Each lane takes the
// branch MaltaDiffPixel takes; a lane no objective applies to keeps
// primary, which then narrows back to the same float.
template <class DD>
HWY_INLINE Vec<DD> MaltaHalfOpen(DD dd, Vec<DD> primary, Vec<DD> s2, Vec<DD> r0, Vec<DD> r1) {
  const auto too_small = Mul(Set(dd, 0.55), Abs(r0));
  const auto too_big = Mul(Set(dd, 1.05), Abs(r0));
  // r0 < 0: r1 above -too_small, else below -too_big.
  const auto neg_small = Sub(primary, Mul(s2, Add(r1, too_small)));
  const auto neg_big = Add(primary, Mul(s2, Sub(Neg(r1), too_big)));
  const auto negative = IfThenElse(Gt(r1, Neg(too_small)), neg_small,
                                   IfThenElse(Lt(r1, Neg(too_big)), neg_big, primary));
  // r0 >= 0: r1 below too_small, else above too_big.
  const auto pos_small = Add(primary, Mul(s2, Sub(too_small, r1)));
  const auto pos_big = Sub(primary, Mul(s2, Sub(r1, too_big)));
  const auto positive = IfThenElse(Lt(r1, too_small), pos_small,
                                   IfThenElse(Gt(r1, too_big), pos_big, primary));
  return IfThenElse(Lt(r0, Zero(dd)), negative, positive);
}

// VideoMetricsLab: MaltaDiffPixel for a vector of pixels, each lane with its
// operations: the float parts in floats, the half-open objectives in doubles
// (the lower and the upper half of the lanes), narrowed back to floats.
template <class DF>
HWY_INLINE Vec<DF> MaltaDiffs(DF df, Vec<DF> r0, Vec<DF> r1, float norm1, float norm2_0gt1, float norm2_0lt1) {
  const auto absval = Mul(Set(df, 0.5f), Add(Abs(r0), Abs(r1)));
  const auto denominator = Add(Set(df, norm1), absval);
  const auto primary = Mul(Div(Set(df, norm2_0gt1), denominator), Sub(r0, r1));
  const auto scaler2 = Div(Set(df, norm2_0lt1), denominator);
  const Repartition<double, DF> dd;
  const Rebind<float, decltype(dd)> dh;  // half of df's lanes
  const auto lower = MaltaHalfOpen(dd, PromoteLowerTo(dd, primary), PromoteLowerTo(dd, scaler2),
                                   PromoteLowerTo(dd, r0), PromoteLowerTo(dd, r1));
  const auto upper = MaltaHalfOpen(dd, PromoteUpperTo(dd, primary), PromoteUpperTo(dd, scaler2),
                                   PromoteUpperTo(dd, r0), PromoteUpperTo(dd, r1));
  return Combine(df, DemoteTo(dh, upper), DemoteTo(dh, lower));
}

// std::log(80.0) / std::log(255.0);
constexpr float kIntensityTargetNormalizationHack = 0.79079917404f;
static const float kInternalGoodQualityThreshold =
    17.83f * kIntensityTargetNormalizationHack;
static const float kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

void StoreMin3(const float v, float& min0, float& min1, float& min2) {
  if (v < min2) {
    if (v < min0) {
      min2 = min1;
      min1 = min0;
      min0 = v;
    } else if (v < min1) {
      min2 = min1;
      min1 = v;
    } else {
      min2 = v;
    }
  }
}

// Look for smooth areas near the area of degradation.
// If the areas area generally smooth, don't do masking.
// VideoMetricsLab: the three smallest of the pixel's value v, 2v twice and
// its eight neighbours kStep away, kept with minimums and maximums. That is
// the values libjxl's insertions (StoreMin3) keep, as long as v >= 0 (so
// that v, 2v, 2v starts sorted) and nothing is NaN: the pixels are blurred
// square roots less the root of their bias (DiffPrecompute), at least +0.
template <class V>
HWY_INLINE void KeepMin3(const V v, V& min0, V& min1, V& min2) {
  const V larger0 = Max(min0, v);
  min0 = Min(min0, v);
  const V larger1 = Max(min1, larger0);
  min1 = Min(min1, larger0);
  min2 = Min(min2, larger1);
}

void FuzzyErosionPixel(const float* above, const float* row, const float* below, size_t xsize, size_t x,
                       float* out);

// libjxl's FuzzyErosion for row `row`: `above` and `below` are the rows kStep
// above and below it, null outside the picture.
void FuzzyErosionRow(const float* above, const float* row, const float* below, size_t xsize, float* out) {
  static const int kStep = 3;
  const HWY_FULL(float) d;
  const size_t lanes = Lanes(d);
  // Columns [kStep, middle_end) of rows with all eight neighbours.
  size_t x = 0;
  if (above != nullptr && below != nullptr && xsize >= 2 * kStep) {
    for (; x < kStep; ++x) FuzzyErosionPixel(above, row, below, xsize, x, out);
    for (; x + lanes + kStep <= xsize; x += lanes) {
      auto min0 = LoadU(d, row + x);
      auto min1 = Add(min0, min0);  // 2 * v, exactly
      auto min2 = min1;
      KeepMin3(LoadU(d, row + x - kStep), min0, min1, min2);
      KeepMin3(LoadU(d, above + x - kStep), min0, min1, min2);
      KeepMin3(LoadU(d, below + x - kStep), min0, min1, min2);
      KeepMin3(LoadU(d, row + x + kStep), min0, min1, min2);
      KeepMin3(LoadU(d, above + x + kStep), min0, min1, min2);
      KeepMin3(LoadU(d, below + x + kStep), min0, min1, min2);
      KeepMin3(LoadU(d, above + x), min0, min1, min2);
      KeepMin3(LoadU(d, below + x), min0, min1, min2);
      StoreU(ContractedMulAdd(Set(d, 0.25f), min2, ContractedMulAdd(Set(d, 0.45f), min0, Mul(Set(d, 0.3f), min1))),
             d, out + x);
    }
  }
  for (; x < xsize; ++x) FuzzyErosionPixel(above, row, below, xsize, x, out);
}

// libjxl's FuzzyErosion for one pixel.
void FuzzyErosionPixel(const float* above, const float* row, const float* below, size_t xsize, size_t x,
                       float* out) {
  static const int kStep = 3;
  float min0 = row[x];
  float min1 = 2 * min0;
  float min2 = min1;
  if (x >= kStep) {
    StoreMin3(row[x - kStep], min0, min1, min2);
    if (above != nullptr) {
      StoreMin3(above[x - kStep], min0, min1, min2);
    }
    if (below != nullptr) {
      StoreMin3(below[x - kStep], min0, min1, min2);
    }
  }
  if (x < xsize - kStep) {
    StoreMin3(row[x + kStep], min0, min1, min2);
    if (above != nullptr) {
      StoreMin3(above[x + kStep], min0, min1, min2);
    }
    if (below != nullptr) {
      StoreMin3(below[x + kStep], min0, min1, min2);
    }
  }
  if (above != nullptr) {
    StoreMin3(above[x], min0, min1, min2);
  }
  if (below != nullptr) {
    StoreMin3(below[x], min0, min1, min2);
  }
  out[x] = ScalarMulAdd(0.25f, min2, ScalarMulAdd(0.45f, min0, 0.3f * min1));
}

double MaskY(double delta) {
  static const double offset = 0.829591754942;
  static const double scaler = 0.451936922203;
  static const double mul = 2.5485944793;
  const double c = mul / ScalarMulAdd(scaler, delta, offset);
  const double retval = kGlobalScale * (1.0 + c);
  return retval * retval;
}

double MaskDcY(double delta) {
  static const double offset = 0.20025578522;
  static const double scaler = 3.87449418804;
  static const double mul = 0.505054525019;
  const double c = mul / ScalarMulAdd(scaler, delta, offset);
  const double retval = kGlobalScale * (1.0 + c);
  return retval * retval;
}

inline float MaskColor(const float color[3], const float mask) {
  return ScalarMulAdd(color[2], mask, ScalarMulAdd(color[0], mask, color[1] * mask));
}

// Diffmap := sqrt of sum{diff images by multiplied by X and Y/B masks}
// (libjxl's CombineChannelsToDiffmap, for one row).
void CombineChannelsToDiffmapRow(const float* mask, const float* const block_diff_dc[3],
                                 const float* const block_diff_ac[3], float xmul, size_t xsize, float* row_out) {
  for (size_t x = 0; x < xsize; ++x) {
    float val = mask[x];
    float maskval = MaskY(val);
    float dc_maskval = MaskDcY(val);
    float diff_dc[3];
    float diff_ac[3];
    for (int i = 0; i < 3; ++i) {
      diff_dc[i] = block_diff_dc[i][x];
      diff_ac[i] = block_diff_ac[i][x];
    }
    diff_ac[0] *= xmul;
    diff_dc[0] *= xmul;
    row_out[x] = std::sqrt(MaskColor(diff_dc, dc_maskval) +
                           MaskColor(diff_ac, maskval));
  }
}

// Adds weighted L2 difference between row0 and row1 to row_diff.
static void L2DiffRow(const float* BUTTERAUGLI_RESTRICT row0, const float* BUTTERAUGLI_RESTRICT row1, size_t xsize,
                      const float w, float* BUTTERAUGLI_RESTRICT row_diff) {
  if (w == 0) return;

  const HWY_FULL(float) d;
  const auto weight = Set(d, w);

  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    const auto diff = Sub(Load(d, row0 + x), Load(d, row1 + x));
    const auto diff2 = Mul(diff, diff);
    const auto prev = Load(d, row_diff + x);
    Store(MulAdd(diff2, weight, prev), d, row_diff + x);
  }
}

// Sets row_diff to the weighted L2 difference between row0 and row1.
static void SetL2DiffRow(const float* BUTTERAUGLI_RESTRICT row0, const float* BUTTERAUGLI_RESTRICT row1, size_t xsize,
                         const float w, float* BUTTERAUGLI_RESTRICT row_diff) {
  if (w == 0) return;

  const HWY_FULL(float) d;
  const auto weight = Set(d, w);

  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    const auto diff = Sub(Load(d, row0 + x), Load(d, row1 + x));
    const auto diff2 = Mul(diff, diff);
    Store(Mul(diff2, weight), d, row_diff + x);
  }
}

// row0 is the original image's.
// row1 is the deformed copy's.
static void L2DiffAsymmetricRow(const float* BUTTERAUGLI_RESTRICT row0, const float* BUTTERAUGLI_RESTRICT row1,
                                size_t xsize, float w_0gt1, float w_0lt1, float* BUTTERAUGLI_RESTRICT row_diff) {
  if (w_0gt1 == 0 && w_0lt1 == 0) {
    return;
  }

  const HWY_FULL(float) d;
  const auto vw_0gt1 = Set(d, w_0gt1 * 0.8);
  const auto vw_0lt1 = Set(d, w_0lt1 * 0.8);

  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    const auto val0 = Load(d, row0 + x);
    const auto val1 = Load(d, row1 + x);

    // Primary symmetric quadratic objective.
    const auto diff = Sub(val0, val1);
    auto total = MulAdd(Mul(diff, diff), vw_0gt1, Load(d, row_diff + x));

    // Secondary half-open quadratic objectives.
    const auto fabs0 = Abs(val0);
    const auto too_small = Mul(Set(d, 0.4), fabs0);
    const auto too_big = fabs0;

    const auto if_neg = IfThenElse(
        Gt(val1, Neg(too_small)), Add(val1, too_small),
        IfThenElseZero(Lt(val1, Neg(too_big)), Sub(Neg(val1), too_big)));
    const auto if_pos =
        IfThenElse(Lt(val1, too_small), Sub(too_small, val1),
                   IfThenElseZero(Gt(val1, too_big), Sub(val1, too_big)));
    const auto v = IfThenElse(Lt(val0, Zero(d)), if_neg, if_pos);
    total = MulAdd(vw_0lt1, Mul(v, v), total);
    Store(total, d, row_diff + x);
  }
}

// A simple HDR compatible gamma function.
template <class DF, class V>
V Gamma(const DF df, V v) {
  // ln(2) constant folded in because we want std::log but have FastLog2f.
  const auto kRetMul = Set(df, 19.245013259874995f * kInvLog2e);
  const auto kRetAdd = Set(df, -23.16046239805755);
  // This should happen rarely, but may lead to a NaN in log, which is
  // undesirable. Since negative photons don't exist we solve the NaNs by
  // clamping here.
  v = ZeroIfNegative(v);

  const auto biased = Add(v, Set(df, 9.9710635769299145));
  const auto log = FastLog2f(df, biased);
  // We could fold this into a custom Log2 polynomial, but there would be
  // relatively little gain.
  return MulAdd(kRetMul, log, kRetAdd);
}

template <bool Clamp, class DF, class V>
BUTTERAUGLI_INLINE void OpsinAbsorbance(const DF df, const V& in0, const V& in1,
                                        const V& in2, V* HWY_RESTRICT out0,
                                        V* HWY_RESTRICT out1,
                                        V* HWY_RESTRICT out2) {
  // https://en.wikipedia.org/wiki/Photopsin absorbance modeling.
  static const double mixi0 = 0.29956550340058319;
  static const double mixi1 = 0.63373087833825936;
  static const double mixi2 = 0.077705617820981968;
  static const double mixi3 = 1.7557483643287353;
  static const double mixi4 = 0.22158691104574774;
  static const double mixi5 = 0.69391388044116142;
  static const double mixi6 = 0.0987313588422;
  static const double mixi7 = 1.7557483643287353;
  static const double mixi8 = 0.02;
  static const double mixi9 = 0.02;
  static const double mixi10 = 0.20480129041026129;
  static const double mixi11 = 12.226454707163354;

  const V mix0 = Set(df, mixi0);
  const V mix1 = Set(df, mixi1);
  const V mix2 = Set(df, mixi2);
  const V mix3 = Set(df, mixi3);
  const V mix4 = Set(df, mixi4);
  const V mix5 = Set(df, mixi5);
  const V mix6 = Set(df, mixi6);
  const V mix7 = Set(df, mixi7);
  const V mix8 = Set(df, mixi8);
  const V mix9 = Set(df, mixi9);
  const V mix10 = Set(df, mixi10);
  const V mix11 = Set(df, mixi11);

  *out0 = MulAdd(mix0, in0, MulAdd(mix1, in1, MulAdd(mix2, in2, mix3)));
  *out1 = MulAdd(mix4, in0, MulAdd(mix5, in1, MulAdd(mix6, in2, mix7)));
  *out2 = MulAdd(mix8, in0, MulAdd(mix9, in1, MulAdd(mix10, in2, mix11)));

  if (Clamp) {
    *out0 = Max(*out0, mix3);
    *out1 = Max(*out1, mix7);
    *out2 = Max(*out2, mix11);
  }
}

// libjxl's OpsinDynamicsImage for one row: `blurred` holds the row of each
// rgb plane's 5x5 blur.
void OpsinDynamicsRow(const float* const rgb[3], const float* const blurred[3], size_t xsize, float intensity_target,
                      float* const xyb[3]) {
  const HWY_FULL(float) df;
  const auto intensity_target_multiplier = Set(df, intensity_target);
  const float* row_r = rgb[0];
  const float* row_g = rgb[1];
  const float* row_b = rgb[2];
  const float* row_blurred_r = blurred[0];
  const float* row_blurred_g = blurred[1];
  const float* row_blurred_b = blurred[2];
  float* row_out_x = xyb[0];
  float* row_out_y = xyb[1];
  float* row_out_b = xyb[2];
  const auto min = Set(df, 1e-4f);
  for (size_t x = 0; x < xsize; x += Lanes(df)) {
    auto sensitivity0 = Undefined(df);
    auto sensitivity1 = Undefined(df);
    auto sensitivity2 = Undefined(df);
    {
      // Calculate sensitivity based on the smoothed image gamma derivative.
      auto pre_mixed0 = Undefined(df);
      auto pre_mixed1 = Undefined(df);
      auto pre_mixed2 = Undefined(df);
      OpsinAbsorbance<true>(
          df, Mul(Load(df, row_blurred_r + x), intensity_target_multiplier),
          Mul(Load(df, row_blurred_g + x), intensity_target_multiplier),
          Mul(Load(df, row_blurred_b + x), intensity_target_multiplier),
          &pre_mixed0, &pre_mixed1, &pre_mixed2);
      pre_mixed0 = Max(pre_mixed0, min);
      pre_mixed1 = Max(pre_mixed1, min);
      pre_mixed2 = Max(pre_mixed2, min);
      sensitivity0 = Div(Gamma(df, pre_mixed0), pre_mixed0);
      sensitivity1 = Div(Gamma(df, pre_mixed1), pre_mixed1);
      sensitivity2 = Div(Gamma(df, pre_mixed2), pre_mixed2);
      sensitivity0 = Max(sensitivity0, min);
      sensitivity1 = Max(sensitivity1, min);
      sensitivity2 = Max(sensitivity2, min);
    }
    auto cur_mixed0 = Undefined(df);
    auto cur_mixed1 = Undefined(df);
    auto cur_mixed2 = Undefined(df);
    OpsinAbsorbance<false>(
        df, Mul(Load(df, row_r + x), intensity_target_multiplier),
        Mul(Load(df, row_g + x), intensity_target_multiplier),
        Mul(Load(df, row_b + x), intensity_target_multiplier), &cur_mixed0,
        &cur_mixed1, &cur_mixed2);
    cur_mixed0 = Mul(cur_mixed0, sensitivity0);
    cur_mixed1 = Mul(cur_mixed1, sensitivity1);
    cur_mixed2 = Mul(cur_mixed2, sensitivity2);
    // This is a kludge. The negative values should be zeroed away before
    // blurring. Ideally there would be no negative values in the first place.
    const auto min01 = Set(df, 1.7557483643287353f);
    const auto min2 = Set(df, 12.226454707163354f);
    cur_mixed0 = Max(cur_mixed0, min01);
    cur_mixed1 = Max(cur_mixed1, min01);
    cur_mixed2 = Max(cur_mixed2, min2);

    Store(Sub(cur_mixed0, cur_mixed1), df, row_out_x + x);
    Store(Add(cur_mixed0, cur_mixed1), df, row_out_y + x);
    Store(cur_mixed2, df, row_out_b + x);
  }
}

// VideoMetricsLab's form of libjxl's Blur for its 7, 13, 15 and 33 tap
// kernels. libjxl convolves the rows and writes the result transposed, then
// does the same again, so that its second pass convolves the columns; here
// a horizontal pass (ConvolveRowSegment) of each row goes into a ring of
// rows as the row is made, and the vertical pass (VerticalRow) of a row
// reads the ring once the rows kHalf below it are in: both run in vectors
// across a row with plain loads and stores, without the transposes, and the
// ring stays in the cache where a whole plane went out to memory and back.
// Every output pixel gets libjxl's operations in libjxl's order
// (its ConvolutionWithTranspose, scalar code): away from the borders the
// kHalf mirrored pairs of the pre-scaled kernel, (a + b) * weight, go
// round-robin into four sums, then centre * weight + sum0 + sum1 + sum2
// (+ sum3), left to right; near a border the kernel's taps inside the
// picture, summed in order and scaled by 1 / their weight. Mul and Add only:
// libjxl compiles that code outside its per-target code, for SSE2, and
// neither MSVC nor this build's -ffp-contract=off fuses them.

// One mirrored pair of taps, kPair from the first one, added to its sum.
// `step` is the distance between taps: 1 along a row, the row stride down a
// column.
template <size_t kPair, size_t kHalf, class D, class V>
HWY_INLINE void AddMirroredPair(D d, const float* first, ptrdiff_t step, const float* sk, V& sum0, V& sum1, V& sum2,
                                V& sum3) {
  const V pair = Mul(Add(LoadU(d, first + kPair * step), LoadU(d, first + (2 * kHalf - kPair) * step)),
                     Set(d, sk[kPair]));
  V& sum = kPair % 4 == 0 ? sum0 : kPair % 4 == 1 ? sum1 : kPair % 4 == 2 ? sum2 : sum3;
  sum = kPair < 4 ? pair : Add(sum, pair);
}

// The convolution of the vector of pixels whose first taps start at
// `first`, away from the borders.
template <size_t kHalf, class D, size_t... kPairs>
HWY_INLINE Vec<D> ConvolveMiddle(D d, const float* first, ptrdiff_t step, const float* sk,
                                 std::index_sequence<kPairs...>) {
  using V = Vec<D>;
  V sum0 = Zero(d), sum1 = Zero(d), sum2 = Zero(d), sum3 = Zero(d);
  (AddMirroredPair<kPairs, kHalf>(d, first, step, sk, sum0, sum1, sum2, sum3), ...);
  V sum = Mul(LoadU(d, first + kHalf * step), Set(d, sk[kHalf]));
  sum = Add(Add(Add(sum, sum0), sum1), sum2);
  if constexpr (kHalf >= 4) sum = Add(sum, sum3);
  return sum;
}

// The same for one pixel of a row, for the last ones that fill no vector.
template <size_t kHalf>
float ConvolveMiddlePixel(const float* first, const float* sk) {
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (size_t pair = 0; pair < kHalf; ++pair) {
    const float product = (first[pair] + first[2 * kHalf - pair]) * sk[pair];
    sums[pair % 4] = pair < 4 ? product : sums[pair % 4] + product;
  }
  float sum = first[kHalf] * sk[kHalf];
  sum = sum + sums[0];
  sum = sum + sums[1];
  sum = sum + sums[2];
  if constexpr (kHalf >= 4) sum = sum + sums[3];
  return sum;
}

// libjxl's weight scale of a pixel near a border (ConvolveBorderColumn):
// 1 / the sum, in order, of the kernel's taps inside [0, size).
inline float BorderScale(const float* kernel, size_t half, size_t pos, size_t size, size_t* first, size_t* last) {
  *first = pos < half ? 0 : pos - half;
  *last = std::min(size - 1, pos + half);
  float weight = 0.0f;
  for (size_t j = *first; j <= *last; ++j) weight += kernel[j + half - pos];
  return 1.0f / weight;
}

// The horizontal pass of columns [x0, x1) of one row into dst (dst[0] is
// column x0). Columns below kHalf and from xsize - kHalf on are near a
// border (both, in a row narrower than the kernel); the decision is the
// picture's, not the strip's.
template <size_t kHalf>
void ConvolveRowSegment(const float* row, size_t xsize, const float* kernel, const float* sk, size_t x0, size_t x1,
                        float* dst) {
  const HWY_FULL(float) d;
  const size_t lanes = Lanes(d);
  const size_t border2 = xsize > kHalf ? xsize - kHalf : 0;
  const size_t middle_begin = std::min(std::max(x0, kHalf), x1);
  const size_t middle_end = std::max(middle_begin, std::min(x1, border2));
  const auto border_pixel = [&](size_t x) {
    size_t first, last;
    const float scale = BorderScale(kernel, kHalf, x, xsize, &first, &last);
    float sum = 0.0f;
    for (size_t j = first; j <= last; ++j) sum += row[j] * kernel[j + kHalf - x];
    dst[x - x0] = sum * scale;
  };
  const auto pairs = std::make_index_sequence<kHalf>();
  size_t x = x0;
  for (; x < middle_begin; ++x) border_pixel(x);
  for (; x + lanes <= middle_end; x += lanes) {
    StoreU(ConvolveMiddle<kHalf>(d, row + x - kHalf, 1, sk, pairs), d, dst + (x - x0));
  }
  for (; x < middle_end; ++x) dst[x - x0] = ConvolveMiddlePixel<kHalf>(row + x - kHalf, sk);
  for (; x < x1; ++x) border_pixel(x);
}

// `xsize` rounded up to whole vectors: the columns the vector loops cover.
inline size_t VectorColumns(size_t xsize) {
  const HWY_FULL(float) d;
  return (xsize + Lanes(d) - 1) / Lanes(d) * Lanes(d);
}

// The vertical pass of row y, `width` columns (whole vectors), from `ring`'s
// rows y - kHalf to y + kHalf of the horizontal pass. Away from the top and
// bottom those lie one stride apart from row y - kHalf's slot on (Ring);
// near them, the rows inside the picture are read one by one.
template <size_t kHalf>
void VerticalRow(const Ring& ring, size_t y, size_t ysize, const BlurKernel& kernel, size_t width, float* row_out) {
  const HWY_FULL(float) d;
  const size_t lanes = Lanes(d);
  if (y >= kHalf && y + kHalf < ysize) {
    const float* first = ring.row(y - kHalf);
    const auto pairs = std::make_index_sequence<kHalf>();
    for (size_t x = 0; x < width; x += lanes) {
      Store(ConvolveMiddle<kHalf>(d, first + x, static_cast<ptrdiff_t>(ring.stride), kernel.sk, pairs), d,
            row_out + x);
    }
    return;
  }
  size_t first, last;
  const auto scale = Set(d, BorderScale(kernel.taps, kHalf, y, ysize, &first, &last));
  for (size_t x = 0; x < width; x += lanes) {
    auto sum = Zero(d);
    for (size_t j = first; j <= last; ++j) {
      sum = Add(sum, Mul(Load(d, ring.row(j) + x), Set(d, kernel.taps[j + kHalf - y])));
    }
    Store(Mul(sum, scale), d, row_out + x);
  }
}

// The horizontal pass of row y (`row`) into its slot of `ring`. The slot's
// columns from xsize to the last whole vector are zeros, so that the
// vertical pass's padding gets zeros.
template <size_t kHalf>
void HorizontalRow(const float* row, size_t xsize, const BlurKernel& kernel, const Ring& ring, size_t y) {
  float* const slot = ring.row(y);
  ConvolveRowSegment<kHalf>(row, xsize, kernel.taps, kernel.sk, 0, xsize, slot);
  std::fill(slot + xsize, slot + VectorColumns(xsize), 0.0f);
  ring.copy_second(y);
}

// ---- VideoMetricsLab: the stages, one row at a time ----

// The weights of the per-band differences (libjxl's, as it names them).
constexpr double wMfMalta = 37.0819870399;
constexpr double norm1Mf = 130262059.556;
constexpr double wMfMaltaX = 8246.75321353;
constexpr double norm1MfX = 1009002.70582;
constexpr double wHfMalta = 18.7237414387;
constexpr double norm1Hf = 4498534.45232;
constexpr double wHfMaltaX = 6923.99476109;
constexpr double norm1HfX = 8051.15833247;
constexpr double wUhfMalta = 1.10039032555;
constexpr double norm1Uhf = 71.7800275169;
constexpr double wUhfMaltaX = 173.5;
constexpr double norm1UhfX = 5.0;
constexpr double wmul[9] = {
    400.0, 1.50815703118, 0, 2150.0, 10.6195433239, 16.2176043152, 29.2353797994, 0.844626970982, 0.703646627719,
};

// A thread's own rows for a stage's intermediate results, which no other
// row needs: `count` rows of `stride` floats, 64-byte aligned.
float* ScratchRows(size_t count, size_t stride) {
  struct Scratch {
    float* data = nullptr;
    size_t floats = 0;
    ~Scratch() { ::operator delete[](data, std::align_val_t{64}); }
  };
  thread_local Scratch scratch;
  const size_t floats = count * stride;
  if (scratch.floats < floats) {
    ::operator delete[](scratch.data, std::align_val_t{64});
    scratch.data = static_cast<float*>(::operator new[](floats * sizeof(float), std::align_val_t{64}));
    std::fill_n(scratch.data, floats, 0.0f);  // the padding never holds NaNs
    scratch.floats = floats;
  }
  return scratch.data;
}

// The planes one picture's stages read and write at one resolution.
struct PictureRows {
  const LevelRings& level;
  size_t picture;
  const Ring& at(Group group, size_t plane) const { return level.at(group, picture, plane); }
};

// kStageOpsin: libjxl's OpsinDynamicsImage of row y (its 5x5 blur of each
// rgb plane, then the conversion), then the horizontal pass of the low
// frequencies' blur of each XYB plane.
void OpsinRow(const StreamFrame& frame, const PictureRows& p, size_t y) {
  const size_t xsize = p.level.xsize;
  const int64_t ysize = static_cast<int64_t>(p.level.ysize);
  const size_t stride = p.at(kRgb, 0).stride;
  float* const scratch = ScratchRows(3, stride);
  float* const blurred[3] = {scratch, scratch + stride, scratch + 2 * stride};
  for (size_t c = 0; c < 3; ++c) {
    const float* rows[5];
    for (int64_t dy = -2; dy <= 2; ++dy) {
      rows[dy + 2] = p.at(kRgb, c).row(static_cast<size_t>(Mirror(static_cast<int64_t>(y) + dy, ysize)));
    }
    Separable5Row::Run(rows, xsize, frame.opsin_blur, blurred[c]);
  }
  const float* const rgb[3] = {p.at(kRgb, 0).row(y), p.at(kRgb, 1).row(y), p.at(kRgb, 2).row(y)};
  float* const xyb[3] = {p.at(kXyb, 0).row(y), p.at(kXyb, 1).row(y), p.at(kXyb, 2).row(y)};
  OpsinDynamicsRow(rgb, blurred, xsize, frame.params.intensity_target, xyb);
  for (size_t c = 0; c < 3; ++c) HorizontalRow<16>(xyb[c], xsize, frame.lf_blur, p.at(kLfh, c), y);
}

// kStageLf: libjxl's SeparateLFAndMF for row y (lf = xyb blurred, mf = xyb -
// lf, lf converted to vals), then the horizontal pass of mf's blur. The
// blue mf is needed only by that pass.
void LfRow(const StreamFrame& frame, const PictureRows& p, size_t y) {
  const HWY_FULL(float) d;
  const size_t xsize = p.level.xsize;
  const size_t width = VectorColumns(xsize);
  float* const mf_raw[3] = {p.at(kMfRaw, 0).row(y), p.at(kMfRaw, 1).row(y), ScratchRows(1, p.at(kLf, 0).stride)};
  for (size_t i = 0; i < 3; ++i) {
    float* const row_lf = p.at(kLf, i).row(y);
    VerticalRow<16>(p.at(kLfh, i), y, p.level.ysize, frame.lf_blur, width, row_lf);
    const float* const row_xyb = p.at(kXyb, i).row(y);
    for (size_t x = 0; x < xsize; x += Lanes(d)) {
      Store(Sub(Load(d, row_xyb + x), Load(d, row_lf + x)), d, mf_raw[i] + x);
    }
  }
  float* BUTTERAUGLI_RESTRICT row_x = p.at(kLf, 0).row(y);
  float* BUTTERAUGLI_RESTRICT row_y = p.at(kLf, 1).row(y);
  float* BUTTERAUGLI_RESTRICT row_b = p.at(kLf, 2).row(y);
  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    auto valx = Undefined(d);
    auto valy = Undefined(d);
    auto valb = Undefined(d);
    XybLowFreqToVals(d, Load(d, row_x + x), Load(d, row_y + x),
                     Load(d, row_b + x), &valx, &valy, &valb);
    Store(valx, d, row_x + x);
    Store(valy, d, row_y + x);
    Store(valb, d, row_b + x);
  }
  for (size_t i = 0; i < 3; ++i) HorizontalRow<7>(mf_raw[i], xsize, frame.mf_blur, p.at(kMfh, i), y);
}

// kStageMf: libjxl's SeparateMFAndHF for row y (mf blurred; for X and Y,
// hf = mf - the blurred mf and mf's ranges reshaped, then SuppressXByY on hf;
// blue mf = the blurred mf), then the horizontal pass of hf's blur.
void MfRow(const StreamFrame& frame, const PictureRows& p, size_t y) {
  const HWY_FULL(float) d;
  static const double kRemoveMfRange = 0.29;
  static const double kAddMfRange = 0.1;
  const size_t xsize = p.level.xsize;
  const size_t width = VectorColumns(xsize);
  const size_t stride = p.at(kMf, 0).stride;
  float* const scratch = ScratchRows(2, stride);
  float* BUTTERAUGLI_RESTRICT blurred_x = scratch;
  float* BUTTERAUGLI_RESTRICT blurred_y = scratch + stride;
  VerticalRow<7>(p.at(kMfh, 0), y, p.level.ysize, frame.mf_blur, width, blurred_x);
  VerticalRow<7>(p.at(kMfh, 1), y, p.level.ysize, frame.mf_blur, width, blurred_y);
  const float* BUTTERAUGLI_RESTRICT row_raw_x = p.at(kMfRaw, 0).row(y);
  const float* BUTTERAUGLI_RESTRICT row_raw_y = p.at(kMfRaw, 1).row(y);
  float* BUTTERAUGLI_RESTRICT row_mf_x = p.at(kMf, 0).row(y);
  float* BUTTERAUGLI_RESTRICT row_mf_y = p.at(kMf, 1).row(y);
  float* BUTTERAUGLI_RESTRICT row_hf_x = p.at(kHfRaw, 0).row(y);
  float* BUTTERAUGLI_RESTRICT row_hf_y = p.at(kHfRaw, 1).row(y);
  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    auto mfx = Load(d, blurred_x + x);
    const auto hfx = Sub(Load(d, row_raw_x + x), mfx);
    mfx = RemoveRangeAroundZero(d, kRemoveMfRange, mfx);
    Store(mfx, d, row_mf_x + x);

    auto mfy = Load(d, blurred_y + x);
    const auto hfy = Sub(Load(d, row_raw_y + x), mfy);
    mfy = AmplifyRangeAroundZero(d, kAddMfRange, mfy);
    Store(mfy, d, row_mf_y + x);
    Store(hfy, d, row_hf_y + x);

    Store(SuppressXByY(d, hfx, hfy), d, row_hf_x + x);
  }
  VerticalRow<7>(p.at(kMfh, 2), y, p.level.ysize, frame.mf_blur, width, p.at(kMf, 2).row(y));
  HorizontalRow<3>(row_hf_x, xsize, frame.hf_blur, p.at(kHfh, 0), y);
  HorizontalRow<3>(row_hf_y, xsize, frame.hf_blur, p.at(kHfh, 1), y);
}

// kStageHf: libjxl's SeparateHFAndUHF for row y (hf blurred; uhf = hf - the
// blurred hf; both reshaped), then the masking's input (libjxl's
// CombineChannelsForMasking and DiffPrecompute) and the horizontal pass of
// its blur.
void HfRow(const StreamFrame& frame, const PictureRows& p, size_t y) {
  const HWY_FULL(float) d;
  static const double kRemoveHfRange = 1.5;
  static const double kAddHfRange = 0.132;
  static const double kRemoveUhfRange = 0.04;
  static const double kMaxclampHf = 28.4691806922;
  static const double kMaxclampUhf = 5.19175294647;
  static const double kMulYHf = 2.155;
  static const double kMulYUhf = 2.69313763794;
  const size_t xsize = p.level.xsize;
  const size_t width = VectorColumns(xsize);
  const size_t stride = p.at(kHf, 0).stride;
  float* const scratch = ScratchRows(2, stride);
  float* BUTTERAUGLI_RESTRICT blurred_x = scratch;
  float* BUTTERAUGLI_RESTRICT blurred_y = scratch + stride;
  VerticalRow<3>(p.at(kHfh, 0), y, p.level.ysize, frame.hf_blur, width, blurred_x);
  VerticalRow<3>(p.at(kHfh, 1), y, p.level.ysize, frame.hf_blur, width, blurred_y);
  const float* BUTTERAUGLI_RESTRICT row_raw_x = p.at(kHfRaw, 0).row(y);
  const float* BUTTERAUGLI_RESTRICT row_raw_y = p.at(kHfRaw, 1).row(y);
  float* BUTTERAUGLI_RESTRICT row_hf_x = p.at(kHf, 0).row(y);
  float* BUTTERAUGLI_RESTRICT row_hf_y = p.at(kHf, 1).row(y);
  float* BUTTERAUGLI_RESTRICT row_uhf_x = p.at(kUhf, 0).row(y);
  float* BUTTERAUGLI_RESTRICT row_uhf_y = p.at(kUhf, 1).row(y);
  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    auto hfx = Load(d, blurred_x + x);
    auto uhfx = Sub(Load(d, row_raw_x + x), hfx);
    hfx = RemoveRangeAroundZero(d, kRemoveHfRange, hfx);
    uhfx = RemoveRangeAroundZero(d, kRemoveUhfRange, uhfx);
    Store(hfx, d, row_hf_x + x);
    Store(uhfx, d, row_uhf_x + x);

    auto hfy = Load(d, blurred_y + x);
    hfy = MaximumClamp(d, hfy, kMaxclampHf);
    auto uhfy = Sub(Load(d, row_raw_y + x), hfy);
    uhfy = MaximumClamp(d, uhfy, kMaxclampUhf);
    uhfy = Mul(uhfy, Set(d, kMulYUhf));
    Store(uhfy, d, row_uhf_y + x);
    hfy = Mul(hfy, Set(d, kMulYHf));
    hfy = AmplifyRangeAroundZero(d, kAddHfRange, hfy);
    Store(hfy, d, row_hf_y + x);
  }

  // Only X and Y components are involved in masking. B's influence
  // is considered less important in the high frequency area, and we
  // don't model masking from lower frequency signals.
  static const float muls[3] = {
      2.5f,
      0.4f,
      0.4f,
  };
  static const float kMul = 6.19424080439;
  static const float kBias = 12.61050594197;
  const auto mul0 = Set(d, muls[0]);
  const auto mul1 = Set(d, muls[1]);
  const auto mul2 = Set(d, muls[2]);
  const float bias = kMul * kBias;
  const float sqrt_bias = std::sqrt(bias);
  const auto mulv = Set(d, kMul);
  const auto biasv = Set(d, bias);
  const auto sqrt_biasv = Set(d, sqrt_bias);
  float* const masking = blurred_x;  // read above, free now
  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    const auto xdiff = Mul(Add(Load(d, row_uhf_x + x), Load(d, row_hf_x + x)), mul0);
    const auto ydiff = ContractedMulAdd(Load(d, row_uhf_y + x), mul1, Mul(Load(d, row_hf_y + x), mul2));
    const auto in = Sqrt(ContractedMulAdd(xdiff, xdiff, Mul(ydiff, ydiff)));
    // kBias makes sqrt behave more linearly.
    Store(Sub(Sqrt(ContractedMulAdd(mulv, Abs(in), biasv)), sqrt_biasv), d, masking + x);
  }
  HorizontalRow<6>(masking, xsize, frame.mask_blur, p.at(kMh, 0), y);
}

// kStageMaskBlur: the vertical pass of the masking's blur, row y.
void MaskBlurRow(const StreamFrame& frame, const PictureRows& p, size_t y) {
  VerticalRow<6>(p.at(kMh, 0), y, p.level.ysize, frame.mask_blur, VectorColumns(p.level.xsize), p.at(kBm, 0).row(y));
}

// The six Malta differences, in the order their sums are added to the AC
// differences: the band and channel compared, the kernel, the weights, and
// the AC plane they go to.
struct MaltaBand {
  Group group;
  size_t channel;
  bool lf;  // MaltaTagLF's kernel, else MaltaTag's
  size_t ac;
};
constexpr MaltaBand kMaltaBands[6] = {
    {kUhf, 1, false, 1}, {kUhf, 0, false, 0}, {kHf, 1, true, 1}, {kHf, 0, true, 0}, {kMf, 1, true, 1}, {kMf, 0, true, 0},
};

// libjxl's weights of the Malta difference `band` (DiffmapPsychoImage).
void MaltaWeights(size_t band, float hf_asymmetry, double* w_0gt1, double* w_0lt1, double* norm1) {
  switch (band) {
    case 0: *w_0gt1 = wUhfMalta * hf_asymmetry; *w_0lt1 = wUhfMalta / hf_asymmetry; *norm1 = norm1Uhf; break;
    case 1: *w_0gt1 = wUhfMaltaX * hf_asymmetry; *w_0lt1 = wUhfMaltaX / hf_asymmetry; *norm1 = norm1UhfX; break;
    case 2:
      *w_0gt1 = wHfMalta * std::sqrt(hf_asymmetry); *w_0lt1 = wHfMalta / std::sqrt(hf_asymmetry); *norm1 = norm1Hf;
      break;
    case 3:
      *w_0gt1 = wHfMaltaX * std::sqrt(hf_asymmetry); *w_0lt1 = wHfMaltaX / std::sqrt(hf_asymmetry);
      *norm1 = norm1HfX;
      break;
    case 4: *w_0gt1 = wMfMalta; *w_0lt1 = wMfMalta; *norm1 = norm1Mf; break;
    default: *w_0gt1 = wMfMaltaX; *w_0lt1 = wMfMaltaX; *norm1 = norm1MfX; break;
  }
}

// kStageMaltaDiffs: row p of the six Malta difference rings (libjxl's
// MaltaDiffMapT: the pixels in vectors (MaltaDiffs), the last ones of a row
// that do not fill one in its scalar code (MaltaDiffPixel)). The rows of the
// margin above and below the picture are zeros.
void MaltaDiffsRow(const StreamFrame& frame, const LevelRings& level, size_t p) {
  const size_t xsize = level.xsize;
  if (p < kMaltaMargin || p >= level.ysize + kMaltaMargin) {
    for (size_t band = 0; band < 6; ++band) {
      const Ring& ring = level.at(kMalta, 0, band);
      std::fill_n(ring.row(p), ring.stride, 0.0f);
      ring.copy_second(p);
    }
    return;
  }
  const size_t y = p - kMaltaMargin;
  const float kWeight0 = 0.5;
  const float kWeight1 = 0.33;
  const double len = 3.75;
  const HWY_FULL(float) d;
  const size_t lanes = Lanes(d);
  for (size_t band = 0; band < 6; ++band) {
    const MaltaBand& b = kMaltaBands[band];
    double w_0gt1, w_0lt1, norm1;
    MaltaWeights(band, frame.params.hf_asymmetry, &w_0gt1, &w_0lt1, &norm1);
    const double mulli = b.lf ? 0.611612573796 : 0.39905817637;
    const double w_pre0gt1 = mulli * std::sqrt(kWeight0 * w_0gt1) / (len * 2 + 1);
    const double w_pre0lt1 = mulli * std::sqrt(kWeight1 * w_0lt1) / (len * 2 + 1);
    const float norm2_0gt1 = w_pre0gt1 * norm1;
    const float norm2_0lt1 = w_pre0lt1 * norm1;
    const float* HWY_RESTRICT row0 = level.at(b.group, 0, b.channel).row(y);
    const float* HWY_RESTRICT row1 = level.at(b.group, 1, b.channel).row(y);
    const Ring& ring = level.at(kMalta, 0, band);
    float* HWY_RESTRICT row_diffs = ring.row(p) + kMaltaLeft;
    std::fill(row_diffs - kMaltaMargin, row_diffs, 0.0f);
    size_t x = 0;
    for (; x + lanes <= xsize; x += lanes) {
      Store(MaltaDiffs(d, Load(d, row0 + x), Load(d, row1 + x), static_cast<float>(norm1), norm2_0gt1, norm2_0lt1),
            d, row_diffs + x);
    }
    for (; x < xsize; ++x) {
      row_diffs[x] = MaltaDiffPixel(row0[x], row1[x], norm1, norm2_0gt1, norm2_0lt1);
    }
    std::fill(row_diffs + xsize, row_diffs + xsize + kMaltaMargin, 0.0f);
    ring.copy_second(p);
  }
}

// libjxl's sums of Malta's line kernels (MaltaUnit) round every pixel of
// row y, added to row_ac. `center` is the row's first pixel in the ring of
// differences. The last vector of a row may run past its last pixel: those
// lanes read the padding past the margin and land in row_ac's padding, and
// no lane of a pixel reads them.
template <class Tag>
void AddMaltaRow(Tag tag, const float* center, size_t stride, size_t xsize, float* row_ac) {
  const HWY_FULL(float) d;
  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    const auto unit = MaltaUnit(tag, d, center + x, static_cast<ptrdiff_t>(stride));
    Store(Add(Load(d, row_ac + x), unit), d, row_ac + x);
  }
}

// kStageCompare: libjxl's DiffmapPsychoImage for row y. Every difference
// lands in each pixel in libjxl's order: the DC differences (set), then per
// AC plane the Malta sums, the L2 differences and, for Y, the mask's; then
// the mask (libjxl's Mask: FuzzyErosion of the reference's blurred masking
// differences) and the combination into the map's row.
void CompareRow(const StreamFrame& frame, size_t level_index, size_t y) {
  const HWY_FULL(float) d;
  const LevelRings& level = frame.level[level_index];
  const size_t xsize = level.xsize;
  const size_t ysize = level.ysize;
  const size_t stride = level.at(kLf, 0, 0).stride;
  const float hf_asymmetry = frame.params.hf_asymmetry;
  float* const scratch = ScratchRows(7, stride);
  float* const dc[3] = {scratch, scratch + stride, scratch + 2 * stride};
  float* const ac[3] = {scratch + 3 * stride, scratch + 4 * stride, scratch + 5 * stride};
  float* const mask = scratch + 6 * stride;
  const auto row = [&](Group group, size_t picture, size_t plane, size_t at) {
    return level.at(group, picture, plane).row(at);
  };

  for (size_t c = 0; c < 3; ++c) {
    SetL2DiffRow(row(kLf, 0, c, y), row(kLf, 1, c, y), xsize, static_cast<float>(wmul[6 + c]), dc[c]);
    std::fill_n(ac[c], stride, 0.0f);
  }
  for (size_t band = 0; band < 6; ++band) {
    const Ring& ring = level.at(kMalta, 0, band);
    // Rows y to y + 2 * kMaltaMargin of the margined differences, around the
    // picture's row y.
    const float* center = ring.row(y) + kMaltaMargin * ring.stride + kMaltaLeft;
    if (kMaltaBands[band].lf) {
      AddMaltaRow(MaltaTagLF(), center, ring.stride, xsize, ac[kMaltaBands[band].ac]);
    } else {
      AddMaltaRow(MaltaTag(), center, ring.stride, xsize, ac[kMaltaBands[band].ac]);
    }
  }
  for (size_t c = 0; c < 3; ++c) {
    if (c < 2) {  // no blue-channel error at high frequencies
      L2DiffAsymmetricRow(row(kHf, 0, c, y), row(kHf, 1, c, y), xsize, static_cast<float>(wmul[c] * hf_asymmetry),
                          static_cast<float>(wmul[c] / hf_asymmetry), ac[c]);
    }
    L2DiffRow(row(kMf, 0, c, y), row(kMf, 1, c, y), xsize, static_cast<float>(wmul[3 + c]), ac[c]);
  }

  static const int kStep = 3;
  const float* const blurred0 = row(kBm, 0, 0, y);
  const float* const blurred1 = row(kBm, 1, 0, y);
  FuzzyErosionRow(y >= kStep ? row(kBm, 0, 0, y - kStep) : nullptr, blurred0,
                  y + kStep < ysize ? row(kBm, 0, 0, y + kStep) : nullptr, xsize, mask);
  static const float kMaskToErrorMul = 10.0;
  const auto error_mul = Set(d, kMaskToErrorMul);
  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    const auto diff = Sub(Load(d, blurred0 + x), Load(d, blurred1 + x));
    Store(ContractedMulAdd(Mul(error_mul, diff), diff, Load(d, ac[1] + x)), d, ac[1] + x);
  }

  float* const out = level_index == 0 ? frame.map + y * xsize : row(kMap, 0, 0, y);
  CombineChannelsToDiffmapRow(mask, dc, ac, frame.params.xmul, xsize, out);
}

void RunStageRow(const StreamFrame& frame, StageKind kind, size_t level, size_t picture, size_t row) {
  const PictureRows p{frame.level[level], picture};
  switch (kind) {
    case kStageOpsin: OpsinRow(frame, p, row); break;
    case kStageLf: LfRow(frame, p, row); break;
    case kStageMf: MfRow(frame, p, row); break;
    case kStageHf: HfRow(frame, p, row); break;
    case kStageMaskBlur: MaskBlurRow(frame, p, row); break;
    case kStageMaltaDiffs: MaltaDiffsRow(frame, frame.level[level], row); break;
    case kStageCompare: CompareRow(frame, level, row); break;
    default: break;  // the rgb rows are the engine's
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace vml::butteraugli
HWY_AFTER_NAMESPACE();

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#if HWY_ONCE
namespace vml::butteraugli {

HWY_EXPORT(RunStageRow);

void run_stage_row(const StreamFrame& frame, StageKind kind, std::size_t level, std::size_t picture,
                   std::size_t row) {
    HWY_DYNAMIC_DISPATCH(RunStageRow)(frame, kind, level, picture, row);
}

}  // namespace vml::butteraugli
#endif  // HWY_ONCE
