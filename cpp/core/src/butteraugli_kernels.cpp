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

// libjxl's Separable5 (enc_convolve_separable5.cc) over a whole plane: a 5x5
// separable blur with mirrored borders. Rows are convolved horizontally with
// vectors; the last columns of rows whose width is not a whole number of
// vectors (from x = the end of the vector loop) are computed by a scalar 2D
// sum, as libjxl does, so the vector width decides which columns those are.
class Separable5Impl {
public:
    using D = HWY_CAPPED(float, 16);
    using DI32 = HWY_CAPPED(int32_t, 16);
    using V = Vec<D>;
    using VI32 = Vec<DI32>;
    using I = decltype(SetTableIndices(D(), static_cast<int32_t*>(nullptr)));
    static constexpr int64_t kRadius = 2;

    Separable5Impl(RowPool& pool, const ImageF& in, const WeightsSeparable5& weights, ImageF* out)
        : pool_(pool), in_(in), weights_(weights), out_(out) {}

    void Run() {
        const size_t min_width = Lanes(D()) + kRadius;
        if (in_.xsize() < min_width) {
            slow_separable5(in_, weights_, out_);
            return;
        }
        switch (in_.xsize() % Lanes(D())) {
        case 0:
            RunRows<0>();
            break;
        case 1:
            RunRows<1>();
            break;
        case 2:
            RunRows<2>();
            break;
        default:
            RunRows<3>();
            break;
        }
    }

private:
    // VideoMetricsLab: the rows in bands; a row within kRadius of the top or
    // the bottom takes the border code, as in libjxl's three loops.
    template <size_t kSizeModN>
    void RunRows() {
        const size_t ysize = in_.ysize();
        pool_.run(ysize, [&](size_t row_begin, size_t row_end) {
            for (size_t y = row_begin; y < row_end; ++y) {
                if (y < kRadius || y + kRadius >= ysize) {
                    ConvolveRow<kSizeModN, true>(y);
                } else {
                    ConvolveRow<kSizeModN, false>(y);
                }
            }
        });
    }

    template <size_t kSizeModN, bool kBorder>
    void ConvolveRow(const size_t y) {
        const D d;
        const int64_t stride = static_cast<int64_t>(in_.PixelsPerRow());
        const size_t xsize = in_.xsize();
        const float* const row_m = in_.ConstRow(y);
        float* const row_out = out_->Row(y);
        const float* row_t2 = row_m - 2 * stride;
        const float* row_t1 = row_m - stride;
        const float* row_b1 = row_m + stride;
        const float* row_b2 = row_m + 2 * stride;
        if constexpr (kBorder) {
            const size_t ysize = in_.ysize();
            if (ysize <= 2 * kRadius) {  // double reflections
                static constexpr size_t kBorderLut[4 * 8] = {
                    0, 0, 0, 0, 0, 0xBAD, 0xBAD, 0xBAD,  // 1 row
                    1, 0, 0, 1, 1, 0,     0xBAD, 0xBAD,  // 2 rows
                    1, 0, 0, 1, 2, 2,     1,     0xBAD,  // 3 rows
                    1, 0, 0, 1, 2, 3,     3,     2,      // 4 rows
                };
                const size_t o = ysize * 8 - 6 + y;
                row_t2 = in_.ConstRow(kBorderLut[o - 2]);
                row_t1 = in_.ConstRow(kBorderLut[o - 1]);
                row_b1 = in_.ConstRow(kBorderLut[o + 1]);
                row_b2 = in_.ConstRow(kBorderLut[o + 2]);
            } else if (y < kRadius) {
                if (y == 0) {
                    row_t1 = row_m;
                    row_t2 = row_b1;
                } else {
                    row_t2 = row_t1;
                }
            } else {
                if (y + 1 == ysize) {
                    row_b1 = row_m;
                    row_b2 = row_t1;
                } else {
                    row_b2 = row_b1;
                }
            }
        }
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
            const float* rows[5] = {row_t2, row_t1, row_m, row_b1, row_b2};
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

    RowPool& pool_;
    const ImageF& in_;
    const WeightsSeparable5& weights_;
    ImageF* out_;
};

void Separable5(RowPool& pool, const ImageF& in, const WeightsSeparable5& weights, ImageF* out) {
    Separable5Impl(pool, in, weights, out).Run();
}

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

// VideoMetricsLab: libjxl blurs each plane and subtracts it (Subtract) one
// plane at a time, then converts lf to vals (XybLowFreqToVals) in a pass of
// its own. Here the three blurs come first (a blur reads only its own plane
// of xyb, which no subtraction of another plane has touched), then a single
// pass does each row's three subtractions and then its conversion.
void SeparateLFAndMF(RowPool& pool, const Image3F& xyb, Image3F* lf, Image3F* mf, BlurTemp* blur_temp) {
  static const double kSigmaLf = 7.15593339443;
  for (int i = 0; i < 3; ++i) {
    blur(pool, xyb.Plane(i), kSigmaLf, blur_temp, &lf->Plane(i));
  }
  const HWY_FULL(float) d;
  const size_t xsize = xyb.xsize();
  pool.run(xyb.ysize(), [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      // mf (which may be xyb itself) keeps what lf does not have.
      for (int i = 0; i < 3; ++i) {
        const float* row_xyb = xyb.ConstPlaneRow(i, y);
        const float* row_lf = lf->ConstPlaneRow(i, y);
        float* row_mf = mf->PlaneRow(i, y);
        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          Store(Sub(Load(d, row_xyb + x), Load(d, row_lf + x)), d, row_mf + x);
        }
      }
      float* BUTTERAUGLI_RESTRICT row_x = lf->PlaneRow(0, y);
      float* BUTTERAUGLI_RESTRICT row_y = lf->PlaneRow(1, y);
      float* BUTTERAUGLI_RESTRICT row_b = lf->PlaneRow(2, y);
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
    }
  });
  return;
}

// VideoMetricsLab: libjxl copies mf's X and Y planes into hf, blurs mf in
// place and subtracts the blurred mf from the copy. Here the blur goes into
// hf and the loop after it subtracts it from mf, which still holds the
// original. One pass does the X and Y planes, then SuppressXByY on them.
void SeparateMFAndHF(RowPool& pool, Image3F* mf, ImageF* hf, BlurTemp* blur_temp) {
  const HWY_FULL(float) d;
  static const double kSigmaHf = 3.22489901262;
  static const double kRemoveMfRange = 0.29;
  static const double kAddMfRange = 0.1;
  const size_t xsize = mf->xsize();
  const size_t ysize = mf->ysize();
  hf[0].ensure(xsize, ysize);
  hf[1].ensure(xsize, ysize);

  blur(pool, mf->Plane(0), kSigmaHf, blur_temp, &hf[0]);
  blur(pool, mf->Plane(1), kSigmaHf, blur_temp, &hf[1]);
  pool.run(ysize, [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      float* BUTTERAUGLI_RESTRICT row_mf_x = mf->PlaneRow(0, y);
      float* BUTTERAUGLI_RESTRICT row_mf_y = mf->PlaneRow(1, y);
      float* BUTTERAUGLI_RESTRICT row_hf_x = hf[0].Row(y);
      float* BUTTERAUGLI_RESTRICT row_hf_y = hf[1].Row(y);
      for (size_t x = 0; x < xsize; x += Lanes(d)) {
        auto mfx = Load(d, row_hf_x + x);  // the blurred mf
        const auto hfx = Sub(Load(d, row_mf_x + x), mfx);
        mfx = RemoveRangeAroundZero(d, kRemoveMfRange, mfx);
        Store(mfx, d, row_mf_x + x);

        auto mfy = Load(d, row_hf_y + x);  // the blurred mf
        const auto hfy = Sub(Load(d, row_mf_y + x), mfy);
        mfy = AmplifyRangeAroundZero(d, kAddMfRange, mfy);
        Store(mfy, d, row_mf_y + x);
        Store(hfy, d, row_hf_y + x);

        Store(SuppressXByY(d, hfx, hfy), d, row_hf_x + x);
      }
    }
  });

  blur(pool, mf->Plane(2), kSigmaHf, blur_temp, &mf->Plane(2));
  return;
}

// VideoMetricsLab: as in SeparateMFAndHF, the blur goes into uhf instead of
// blurring a copy, and one pass after both blurs reads the originals from hf.
void SeparateHFAndUHF(RowPool& pool, ImageF* hf, ImageF* uhf, BlurTemp* blur_temp) {
  const HWY_FULL(float) d;
  const size_t xsize = hf[0].xsize();
  const size_t ysize = hf[0].ysize();
  static const double kSigmaUhf = 1.56416327805;
  static const double kRemoveHfRange = 1.5;
  static const double kAddHfRange = 0.132;
  static const double kRemoveUhfRange = 0.04;
  static const double kMaxclampHf = 28.4691806922;
  static const double kMaxclampUhf = 5.19175294647;
  static const double kMulYHf = 2.155;
  static const double kMulYUhf = 2.69313763794;
  uhf[0].ensure(xsize, ysize);
  uhf[1].ensure(xsize, ysize);

  blur(pool, hf[0], kSigmaUhf, blur_temp, &uhf[0]);
  blur(pool, hf[1], kSigmaUhf, blur_temp, &uhf[1]);
  pool.run(ysize, [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      float* BUTTERAUGLI_RESTRICT row_uhf_x = uhf[0].Row(y);
      float* BUTTERAUGLI_RESTRICT row_hf_x = hf[0].Row(y);
      float* BUTTERAUGLI_RESTRICT row_uhf_y = uhf[1].Row(y);
      float* BUTTERAUGLI_RESTRICT row_hf_y = hf[1].Row(y);
      for (size_t x = 0; x < xsize; x += Lanes(d)) {
        auto hfx = Load(d, row_uhf_x + x);  // the blurred hf
        auto uhfx = Sub(Load(d, row_hf_x + x), hfx);
        hfx = RemoveRangeAroundZero(d, kRemoveHfRange, hfx);
        uhfx = RemoveRangeAroundZero(d, kRemoveUhfRange, uhfx);
        Store(hfx, d, row_hf_x + x);
        Store(uhfx, d, row_uhf_x + x);

        auto hfy = Load(d, row_uhf_y + x);  // the blurred hf
        hfy = MaximumClamp(d, hfy, kMaxclampHf);
        auto uhfy = Sub(Load(d, row_hf_y + x), hfy);
        uhfy = MaximumClamp(d, uhfy, kMaxclampUhf);
        uhfy = Mul(uhfy, Set(d, kMulYUhf));
        Store(uhfy, d, row_uhf_y + x);
        hfy = Mul(hfy, Set(d, kMulYHf));
        hfy = AmplifyRangeAroundZero(d, kAddHfRange, hfy);
        Store(hfy, d, row_hf_y + x);
      }
    }
  });
  return;
}

void SeparateFrequencies(RowPool& pool, BlurTemp* blur_temp, Image3F& xyb, PsychoImage& ps) {
  // VideoMetricsLab: the planes are kept from frame to frame, and mf is
  // computed in xyb's planes (Subtract is element-wise), which then trade
  // places with ps.mf's: a picture's bands take no planes of their own for mf.
  ps.lf.ensure(xyb.xsize(), xyb.ysize());
  SeparateLFAndMF(pool, xyb, &ps.lf, &xyb, blur_temp);
  std::swap(ps.mf, xyb);
  SeparateMFAndHF(pool, &ps.mf, &ps.hf[0], blur_temp);
  SeparateHFAndUHF(pool, &ps.hf[0], &ps.uhf[0], blur_temp);
  return;
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

// VideoMetricsLab: Malta's differences go into a plane with a margin of
// zeros round them, kMaltaLeft pixels at the left (so that the pixels' rows
// stay 64-byte aligned), kMaltaMargin at the right and kMaltaMargin rows
// above and below: what libjxl's PaddedMaltaUnit gives MaltaUnit for a
// pixel near a border, one pixel at a time. MaltaUnit reads kMaltaMargin
// pixels round its own, so it then runs in vectors on every pixel, each
// lane doing what PaddedMaltaUnit's one lane did.
constexpr size_t kMaltaLeft = 16;
constexpr size_t kMaltaMargin = 4;

template <class Tag>
static void MaltaDiffMapT(RowPool& pool, const Tag /*tag*/, const ImageF& lum0,
                            const ImageF& lum1, const double w_0gt1,
                            const double w_0lt1, const double norm1,
                            const double len, const double mulli,
                            ImageF* HWY_RESTRICT padded_diffs,
                            ImageF* HWY_RESTRICT block_diff_ac) {
  const size_t xsize_ = lum0.xsize();
  const size_t ysize_ = lum0.ysize();

  const float kWeight0 = 0.5;
  const float kWeight1 = 0.33;

  const double w_pre0gt1 = mulli * std::sqrt(kWeight0 * w_0gt1) / (len * 2 + 1);
  const double w_pre0lt1 = mulli * std::sqrt(kWeight1 * w_0lt1) / (len * 2 + 1);
  const float norm2_0gt1 = w_pre0gt1 * norm1;
  const float norm2_0lt1 = w_pre0lt1 * norm1;

  padded_diffs->ensure(kMaltaLeft + xsize_ + kMaltaMargin, ysize_ + 2 * kMaltaMargin);
  const size_t stride = padded_diffs->PixelsPerRow();
  for (size_t y = 0; y < kMaltaMargin; ++y) {
    std::fill(padded_diffs->Row(y), padded_diffs->Row(y) + stride, 0.0f);
    std::fill(padded_diffs->Row(ysize_ + kMaltaMargin + y), padded_diffs->Row(ysize_ + kMaltaMargin + y) + stride,
              0.0f);
  }
  // The pixels in vectors (MaltaDiffs), the last ones of a row that do not
  // fill one in libjxl's scalar code (MaltaDiffPixel).
  const HWY_FULL(float) d;
  const size_t lanes = Lanes(d);
  pool.run(ysize_, [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      const float* HWY_RESTRICT row0 = lum0.ConstRow(y);
      const float* HWY_RESTRICT row1 = lum1.ConstRow(y);
      float* HWY_RESTRICT row_diffs = padded_diffs->Row(y + kMaltaMargin) + kMaltaLeft;
      std::fill(row_diffs - kMaltaMargin, row_diffs, 0.0f);
      size_t x = 0;
      for (; x + lanes <= xsize_; x += lanes) {
        Store(MaltaDiffs(d, Load(d, row0 + x), Load(d, row1 + x), static_cast<float>(norm1), norm2_0gt1, norm2_0lt1),
              d, row_diffs + x);
      }
      for (; x < xsize_; ++x) {
        row_diffs[x] = MaltaDiffPixel(row0[x], row1[x], norm1, norm2_0gt1, norm2_0lt1);
      }
      std::fill(row_diffs + xsize_, row_diffs + xsize_ + kMaltaMargin, 0.0f);
    }
  });

  // libjxl's sums of the line kernels (MaltaUnit) round every pixel, added
  // to block_diff_ac. The last vector of a row may run past its last pixel:
  // those lanes read the padding past the margin and land in block_diff_ac's
  // padding, and no lane of a pixel reads them.
  pool.run(ysize_, [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      const float* BUTTERAUGLI_RESTRICT row_in = padded_diffs->ConstRow(y + kMaltaMargin) + kMaltaLeft;
      float* BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->Row(y);
      for (size_t x = 0; x < xsize_; x += lanes) {
        const auto unit = MaltaUnit(Tag(), d, row_in + x, static_cast<ptrdiff_t>(stride));
        Store(Add(Load(d, row_diff + x), unit), d, row_diff + x);
      }
    }
  });
  return;
}

// Need non-template wrapper functions for HWY_EXPORT.
void MaltaDiffMap(RowPool& pool, const ImageF& lum0, const ImageF& lum1, const double w_0gt1,
                    const double w_0lt1, const double norm1,
                    ImageF* HWY_RESTRICT diffs,
                    ImageF* HWY_RESTRICT block_diff_ac) {
  const double len = 3.75;
  static const double mulli = 0.39905817637;
  MaltaDiffMapT(pool, MaltaTag(), lum0, lum1, w_0gt1, w_0lt1,
                                    norm1, len, mulli, diffs, block_diff_ac);
  return;
}

void MaltaDiffMapLF(RowPool& pool, const ImageF& lum0, const ImageF& lum1,
                      const double w_0gt1, const double w_0lt1,
                      const double norm1, ImageF* HWY_RESTRICT diffs,
                      ImageF* HWY_RESTRICT block_diff_ac) {
  const double len = 3.75;
  static const double mulli = 0.611612573796;
  MaltaDiffMapT(pool, MaltaTagLF(), lum0, lum1, w_0gt1, w_0lt1,
                                    norm1, len, mulli, diffs, block_diff_ac);
  return;
}

void CombineChannelsForMasking(RowPool& pool, const ImageF* hf, const ImageF* uhf,
                               ImageF* out) {
  // Only X and Y components are involved in masking. B's influence
  // is considered less important in the high frequency area, and we
  // don't model masking from lower frequency signals.
  static const float muls[3] = {
      2.5f,
      0.4f,
      0.4f,
  };
  // VideoMetricsLab: libjxl's scalar loop in vectors, each lane with its
  // operations (the rows are padded to whole vectors).
  const HWY_FULL(float) d;
  const auto mul0 = Set(d, muls[0]);
  const auto mul1 = Set(d, muls[1]);
  const auto mul2 = Set(d, muls[2]);
  pool.run(hf[0].ysize(), [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      const float* BUTTERAUGLI_RESTRICT row_y_hf = hf[1].Row(y);
      const float* BUTTERAUGLI_RESTRICT row_y_uhf = uhf[1].Row(y);
      const float* BUTTERAUGLI_RESTRICT row_x_hf = hf[0].Row(y);
      const float* BUTTERAUGLI_RESTRICT row_x_uhf = uhf[0].Row(y);
      float* BUTTERAUGLI_RESTRICT row = out->Row(y);
      for (size_t x = 0; x < hf[0].xsize(); x += Lanes(d)) {
        const auto xdiff = Mul(Add(Load(d, row_x_uhf + x), Load(d, row_x_hf + x)), mul0);
        const auto ydiff = ContractedMulAdd(Load(d, row_y_uhf + x), mul1, Mul(Load(d, row_y_hf + x), mul2));
        Store(Sqrt(ContractedMulAdd(xdiff, xdiff, Mul(ydiff, ydiff))), d, row + x);
      }
      }
  });
}

void DiffPrecompute(RowPool& pool, const ImageF& xyb, float mul, float bias_arg, ImageF* out) {
  const size_t xsize = xyb.xsize();
  const size_t ysize = xyb.ysize();
  const float bias = mul * bias_arg;
  const float sqrt_bias = std::sqrt(bias);
  // VideoMetricsLab: libjxl's scalar loop in vectors, each lane with its
  // operations.
  const HWY_FULL(float) d;
  const auto mulv = Set(d, mul);
  const auto biasv = Set(d, bias);
  const auto sqrt_biasv = Set(d, sqrt_bias);
  pool.run(ysize, [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      const float* BUTTERAUGLI_RESTRICT row_in = xyb.Row(y);
      float* BUTTERAUGLI_RESTRICT row_out = out->Row(y);
      for (size_t x = 0; x < xsize; x += Lanes(d)) {
        // kBias makes sqrt behave more linearly.
        const auto in = Load(d, row_in + x);
        Store(Sub(Sqrt(ContractedMulAdd(mulv, Abs(in), biasv)), sqrt_biasv), d, row_out + x);
      }
      }
  });
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

void FuzzyErosionPixel(const ImageF& from, size_t x, size_t y, ImageF* to);

void FuzzyErosion(RowPool& pool, const ImageF& from, ImageF* to) {
  const size_t xsize = from.xsize();
  const size_t ysize = from.ysize();
  static const int kStep = 3;
  const HWY_FULL(float) d;
  const size_t lanes = Lanes(d);
  pool.run(ysize, [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      // Columns [kStep, middle_end) of rows with all eight neighbours.
      size_t x = 0;
      if (y >= kStep && y + kStep < ysize && xsize >= 2 * kStep) {
        for (; x < kStep; ++x) FuzzyErosionPixel(from, x, y, to);
        const float* above = from.Row(y - kStep);
        const float* row = from.Row(y);
        const float* below = from.Row(y + kStep);
        float* out = to->Row(y);
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
      for (; x < xsize; ++x) FuzzyErosionPixel(from, x, y, to);
      }
  });
}

// libjxl's FuzzyErosion for one pixel.
void FuzzyErosionPixel(const ImageF& from, size_t x, size_t y, ImageF* to) {
  const size_t xsize = from.xsize();
  const size_t ysize = from.ysize();
  static const int kStep = 3;
  {
    {
      float min0 = from.Row(y)[x];
      float min1 = 2 * min0;
      float min2 = min1;
      if (x >= kStep) {
        StoreMin3(from.Row(y)[x - kStep], min0, min1, min2);
        if (y >= kStep) {
          StoreMin3(from.Row(y - kStep)[x - kStep], min0, min1, min2);
        }
        if (y < ysize - kStep) {
          StoreMin3(from.Row(y + kStep)[x - kStep], min0, min1, min2);
        }
      }
      if (x < xsize - kStep) {
        StoreMin3(from.Row(y)[x + kStep], min0, min1, min2);
        if (y >= kStep) {
          StoreMin3(from.Row(y - kStep)[x + kStep], min0, min1, min2);
        }
        if (y < ysize - kStep) {
          StoreMin3(from.Row(y + kStep)[x + kStep], min0, min1, min2);
        }
      }
      if (y >= kStep) {
        StoreMin3(from.Row(y - kStep)[x], min0, min1, min2);
      }
      if (y < ysize - kStep) {
        StoreMin3(from.Row(y + kStep)[x], min0, min1, min2);
      }
      to->Row(y)[x] = ScalarMulAdd(0.25f, min2, ScalarMulAdd(0.45f, min0, 0.3f * min1));
    }
  }
}

// Compute values of local frequency and dc masking based on the activity
// in the two images. img_diff_ac may be null.
void Mask(RowPool& pool, const ImageF& mask0, const ImageF& mask1, BlurTemp* blur_temp,
            ImageF* BUTTERAUGLI_RESTRICT mask,
            ImageF* BUTTERAUGLI_RESTRICT diff_ac, const MaskScratch& scratch) {
  const size_t xsize = mask0.xsize();
  const size_t ysize = mask0.ysize();
  mask->ensure(xsize, ysize);
  static const float kMul = 6.19424080439;
  static const float kBias = 12.61050594197;
  static const float kRadius = 2.7;
  // VideoMetricsLab: lent planes instead of new ones.
  ImageF& diff0 = *scratch.diff0;
  ImageF& diff1 = *scratch.diff1;
  ImageF& blurred0 = *scratch.blurred0;
  ImageF& blurred1 = *scratch.blurred1;
  DiffPrecompute(pool, mask0, kMul, kBias, &diff0);
  DiffPrecompute(pool, mask1, kMul, kBias, &diff1);
  blur(pool, diff0, kRadius, blur_temp, &blurred0);
  FuzzyErosion(pool, blurred0, mask);  // VideoMetricsLab: libjxl erodes into diff0, then copies it into the mask
  blur(pool, diff1, kRadius, blur_temp, &blurred1);
  if (diff_ac != nullptr) {
    // VideoMetricsLab: libjxl's scalar loop in vectors.
    static const float kMaskToErrorMul = 10.0;
    const HWY_FULL(float) d;
    const auto error_mul = Set(d, kMaskToErrorMul);
    pool.run(ysize, [&](size_t row_begin, size_t row_end) {
      for (size_t y = row_begin; y < row_end; ++y) {
        const float* row0 = blurred0.Row(y);
        const float* row1 = blurred1.Row(y);
        float* row_ac = diff_ac->Row(y);
        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          const auto diff = Sub(Load(d, row0 + x), Load(d, row1 + x));
          Store(ContractedMulAdd(Mul(error_mul, diff), diff, Load(d, row_ac + x)), d, row_ac + x);
        }
          }
    });
  }
  return;
}

// `diff_ac` may be null.
void MaskPsychoImage(RowPool& pool, const PsychoImage& pi0, const PsychoImage& pi1, BlurTemp* blur_temp,
                       const MaskScratch& scratch,
                       ImageF* BUTTERAUGLI_RESTRICT mask,
                       ImageF* BUTTERAUGLI_RESTRICT diff_ac) {
  ImageF& mask0 = *scratch.mask0;
  ImageF& mask1 = *scratch.mask1;
  CombineChannelsForMasking(pool, &pi0.hf[0], &pi0.uhf[0], &mask0);
  CombineChannelsForMasking(pool, &pi1.hf[0], &pi1.uhf[0], &mask1);
  Mask(pool, mask0, mask1, blur_temp, mask, diff_ac, scratch);
  return;
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
void CombineChannelsToDiffmap(RowPool& pool, const ImageF& mask,
                                const Image3F& block_diff_dc,
                                const Image3F& block_diff_ac, float xmul,
                                ImageF* result) {
  size_t xsize = mask.xsize();
  size_t ysize = mask.ysize();
  pool.run(ysize, [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      float* BUTTERAUGLI_RESTRICT row_out = result->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        float val = mask.Row(y)[x];
        float maskval = MaskY(val);
        float dc_maskval = MaskDcY(val);
        float diff_dc[3];
        float diff_ac[3];
        for (int i = 0; i < 3; ++i) {
          diff_dc[i] = block_diff_dc.PlaneRow(i, y)[x];
          diff_ac[i] = block_diff_ac.PlaneRow(i, y)[x];
        }
        diff_ac[0] *= xmul;
        diff_dc[0] *= xmul;
        row_out[x] = std::sqrt(MaskColor(diff_dc, dc_maskval) +
                               MaskColor(diff_ac, maskval));
      }
      }
  });
  return;
}

// Adds weighted L2 difference between i0 and i1 to diffmap.
static void L2Diff(RowPool& pool, const ImageF& i0, const ImageF& i1, const float w,
                   ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  if (w == 0) return;

  const HWY_FULL(float) d;
  const auto weight = Set(d, w);

  pool.run(i0.ysize(), [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      const float* BUTTERAUGLI_RESTRICT row0 = i0.ConstRow(y);
      const float* BUTTERAUGLI_RESTRICT row1 = i1.ConstRow(y);
      float* BUTTERAUGLI_RESTRICT row_diff = diffmap->Row(y);

      for (size_t x = 0; x < i0.xsize(); x += Lanes(d)) {
        const auto diff = Sub(Load(d, row0 + x), Load(d, row1 + x));
        const auto diff2 = Mul(diff, diff);
        const auto prev = Load(d, row_diff + x);
        Store(MulAdd(diff2, weight, prev), d, row_diff + x);
      }
      }
  });
}

// Initializes diffmap to the weighted L2 difference between i0 and i1.
static void SetL2Diff(RowPool& pool, const ImageF& i0, const ImageF& i1, const float w,
                      ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  if (w == 0) return;

  const HWY_FULL(float) d;
  const auto weight = Set(d, w);

  pool.run(i0.ysize(), [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      const float* BUTTERAUGLI_RESTRICT row0 = i0.ConstRow(y);
      const float* BUTTERAUGLI_RESTRICT row1 = i1.ConstRow(y);
      float* BUTTERAUGLI_RESTRICT row_diff = diffmap->Row(y);

      for (size_t x = 0; x < i0.xsize(); x += Lanes(d)) {
        const auto diff = Sub(Load(d, row0 + x), Load(d, row1 + x));
        const auto diff2 = Mul(diff, diff);
        Store(Mul(diff2, weight), d, row_diff + x);
      }
      }
  });
}

// i0 is the original image.
// i1 is the deformed copy.
static void L2DiffAsymmetric(RowPool& pool, const ImageF& i0, const ImageF& i1, float w_0gt1,
                             float w_0lt1,
                             ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  if (w_0gt1 == 0 && w_0lt1 == 0) {
    return;
  }

  const HWY_FULL(float) d;
  const auto vw_0gt1 = Set(d, w_0gt1 * 0.8);
  const auto vw_0lt1 = Set(d, w_0lt1 * 0.8);

  pool.run(i0.ysize(), [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      const float* BUTTERAUGLI_RESTRICT row0 = i0.Row(y);
      const float* BUTTERAUGLI_RESTRICT row1 = i1.Row(y);
      float* BUTTERAUGLI_RESTRICT row_diff = diffmap->Row(y);

      for (size_t x = 0; x < i0.xsize(); x += Lanes(d)) {
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
  });
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

// `blurred` is a temporary image used inside this function and not returned.
void OpsinDynamicsImage(RowPool& pool, const Image3F& rgb, const ButteraugliParams& params,
                          Image3F* blurred, BlurTemp* blur_temp, Image3F* xyb) {
  const double kSigma = 1.2;
  blur(pool, rgb.Plane(0), kSigma, blur_temp, &blurred->Plane(0));
  blur(pool, rgb.Plane(1), kSigma, blur_temp, &blurred->Plane(1));
  blur(pool, rgb.Plane(2), kSigma, blur_temp, &blurred->Plane(2));
  const HWY_FULL(float) df;
  const auto intensity_target_multiplier = Set(df, params.intensity_target);
  pool.run(rgb.ysize(), [&](size_t row_begin, size_t row_end) {
    for (size_t y = row_begin; y < row_end; ++y) {
      const float* row_r = rgb.ConstPlaneRow(0, y);
      const float* row_g = rgb.ConstPlaneRow(1, y);
      const float* row_b = rgb.ConstPlaneRow(2, y);
      const float* row_blurred_r = blurred->ConstPlaneRow(0, y);
      const float* row_blurred_g = blurred->ConstPlaneRow(1, y);
      const float* row_blurred_b = blurred->ConstPlaneRow(2, y);
      float* row_out_x = xyb->PlaneRow(0, y);
      float* row_out_y = xyb->PlaneRow(1, y);
      float* row_out_b = xyb->PlaneRow(2, y);
      const auto min = Set(df, 1e-4f);
      for (size_t x = 0; x < rgb.xsize(); x += Lanes(df)) {
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
  });
  return;
}

// VideoMetricsLab's form of libjxl's Blur for its 7, 13, 15 and 33 tap
// kernels. libjxl convolves the rows and writes the result transposed, then
// does the same again, so that its second pass convolves the columns; here
// a horizontal pass (ConvolveRowsHorizontal) and a vertical pass
// (ConvolveRowsVertical) work on planes the right way round, so that both
// run in vectors across a row with plain loads and stores, without the
// transposes. Every output pixel gets libjxl's operations in libjxl's order
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

// Rows [y0, y1) of the horizontal pass. The row padding of `out` is set to
// zero, so that the vertical pass's lanes past the last pixel compute zeros.
template <size_t kHalf>
void ConvolveRowsHorizontal(const ImageF& in, const float* kernel, const float* sk, size_t y0, size_t y1,
                            ImageF* out) {
  const HWY_FULL(float) d;
  const size_t lanes = Lanes(d);
  const size_t xsize = in.xsize();
  // Columns [0, border1) and [border2, xsize) are near a border (the two
  // overlap in a picture narrower than the kernel).
  const size_t border1 = std::min(xsize, kHalf);
  const size_t border2 = xsize > kHalf ? xsize - kHalf : 0;
  const auto pairs = std::make_index_sequence<kHalf>();
  for (size_t y = y0; y < y1; ++y) {
    const float* row = in.ConstRow(y);
    float* row_out = out->Row(y);
    const auto border_pixel = [&](size_t x) {
      size_t first, last;
      const float scale = BorderScale(kernel, kHalf, x, xsize, &first, &last);
      float sum = 0.0f;
      for (size_t j = first; j <= last; ++j) sum += row[j] * kernel[j + kHalf - x];
      row_out[x] = sum * scale;
    };
    for (size_t x = 0; x < border1; ++x) border_pixel(x);
    size_t x = kHalf;
    for (; x + lanes <= border2; x += lanes) {
      StoreU(ConvolveMiddle<kHalf>(d, row + x - kHalf, 1, sk, pairs), d, row_out + x);
    }
    for (; x < border2; ++x) row_out[x] = ConvolveMiddlePixel<kHalf>(row + x - kHalf, sk);
    for (x = std::max(border1, border2); x < xsize; ++x) border_pixel(x);
    std::fill(row_out + xsize, row_out + out->PixelsPerRow(), 0.0f);
  }
}

// Rows [y0, y1) of the vertical pass: the second transposed pass's work on
// column y of the transposed plane, done on row y, a vector of columns at a
// time (lanes past the last pixel land in the row padding).
template <size_t kHalf>
void ConvolveRowsVertical(const ImageF& in, const float* kernel, const float* sk, size_t y0, size_t y1,
                          ImageF* out) {
  const HWY_FULL(float) d;
  const size_t lanes = Lanes(d);
  const size_t xsize = in.xsize();
  const size_t ysize = in.ysize();
  const ptrdiff_t stride = static_cast<ptrdiff_t>(in.PixelsPerRow());
  const auto pairs = std::make_index_sequence<kHalf>();
  for (size_t y = y0; y < y1; ++y) {
    float* row_out = out->Row(y);
    if (y >= kHalf && y + kHalf < ysize) {
      const float* first = in.ConstRow(y - kHalf);
      for (size_t x = 0; x < xsize; x += lanes) {
        Store(ConvolveMiddle<kHalf>(d, first + x, stride, sk, pairs), d, row_out + x);
      }
      continue;
    }
    size_t first, last;
    const auto scale = Set(d, BorderScale(kernel, kHalf, y, ysize, &first, &last));
    for (size_t x = 0; x < xsize; x += lanes) {
      auto sum = Zero(d);
      for (size_t j = first; j <= last; ++j) {
        sum = Add(sum, Mul(Load(d, in.ConstRow(j) + x), Set(d, kernel[j + kHalf - y])));
      }
      Store(Mul(sum, scale), d, row_out + x);
    }
  }
}

template <size_t kHalf>
void SeparableBlurT(RowPool& pool, const ImageF& in, const float* kernel, const float* sk, ImageF* temp,
                    ImageF* out) {
  pool.run(in.ysize(), [&](size_t y0, size_t y1) { ConvolveRowsHorizontal<kHalf>(in, kernel, sk, y0, y1, temp); });
  pool.run(in.ysize(), [&](size_t y0, size_t y1) { ConvolveRowsVertical<kHalf>(*temp, kernel, sk, y0, y1, out); });
}

// `kernel` has 2 * half + 1 taps; `temp` has in's size, and out may be in.
void SeparableBlur(RowPool& pool, const ImageF& in, const float* kernel, size_t half, ImageF* temp, ImageF* out) {
  // libjxl's scale of the taps away from the borders: 1 / their sum.
  float weight_no_border = 0.0f;
  for (size_t j = 0; j < 2 * half + 1; ++j) weight_no_border += kernel[j];
  const float scale_no_border = 1.0f / weight_no_border;
  float sk[17];  // the scaled kernel's centre and left half (half <= 16)
  for (size_t i = 0; i <= half; ++i) sk[i] = kernel[i] * scale_no_border;
  switch (half) {
    case 3: SeparableBlurT<3>(pool, in, kernel, sk, temp, out); break;
    case 6: SeparableBlurT<6>(pool, in, kernel, sk, temp, out); break;
    case 7: SeparableBlurT<7>(pool, in, kernel, sk, temp, out); break;
    case 16: SeparableBlurT<16>(pool, in, kernel, sk, temp, out); break;
    default: break;  // blur takes no other length
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

HWY_EXPORT(Separable5);
HWY_EXPORT(OpsinDynamicsImage);
HWY_EXPORT(SeparateFrequencies);
HWY_EXPORT(MaltaDiffMap);
HWY_EXPORT(MaltaDiffMapLF);
HWY_EXPORT(L2DiffAsymmetric);
HWY_EXPORT(L2Diff);
HWY_EXPORT(SetL2Diff);
HWY_EXPORT(MaskPsychoImage);
HWY_EXPORT(CombineChannelsToDiffmap);
HWY_EXPORT(SeparableBlur);

void separable5(RowPool& pool, const ImageF& in, const WeightsSeparable5& weights, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(Separable5)(pool, in, weights, out);
}

void opsin_dynamics_image(RowPool& pool, const Image3F& rgb, const ButteraugliParams& params, Image3F* blurred,
                          BlurTemp* blur_temp, Image3F* xyb) {
    HWY_DYNAMIC_DISPATCH(OpsinDynamicsImage)(pool, rgb, params, blurred, blur_temp, xyb);
}

void separate_frequencies(RowPool& pool, BlurTemp* blur_temp, Image3F& xyb, PsychoImage& ps) {
    HWY_DYNAMIC_DISPATCH(SeparateFrequencies)(pool, blur_temp, xyb, ps);
}

void malta_diff_map(RowPool& pool, const ImageF& lum0, const ImageF& lum1, double w_0gt1, double w_0lt1, double norm1,
                    ImageF* diffs, ImageF* block_diff_ac) {
    HWY_DYNAMIC_DISPATCH(MaltaDiffMap)(pool, lum0, lum1, w_0gt1, w_0lt1, norm1, diffs, block_diff_ac);
}

void malta_diff_map_lf(RowPool& pool, const ImageF& lum0, const ImageF& lum1, double w_0gt1, double w_0lt1, double norm1,
                       ImageF* diffs, ImageF* block_diff_ac) {
    HWY_DYNAMIC_DISPATCH(MaltaDiffMapLF)(pool, lum0, lum1, w_0gt1, w_0lt1, norm1, diffs, block_diff_ac);
}

void l2_diff_asymmetric(RowPool& pool, const ImageF& i0, const ImageF& i1, float w_0gt1, float w_0lt1, ImageF* diffmap) {
    HWY_DYNAMIC_DISPATCH(L2DiffAsymmetric)(pool, i0, i1, w_0gt1, w_0lt1, diffmap);
}

void l2_diff(RowPool& pool, const ImageF& i0, const ImageF& i1, float w, ImageF* diffmap) {
    HWY_DYNAMIC_DISPATCH(L2Diff)(pool, i0, i1, w, diffmap);
}

void set_l2_diff(RowPool& pool, const ImageF& i0, const ImageF& i1, float w, ImageF* diffmap) {
    HWY_DYNAMIC_DISPATCH(SetL2Diff)(pool, i0, i1, w, diffmap);
}

void mask_psycho_image(RowPool& pool, const PsychoImage& pi0, const PsychoImage& pi1, BlurTemp* blur_temp,
                       const MaskScratch& scratch, ImageF* mask, ImageF* diff_ac) {
    HWY_DYNAMIC_DISPATCH(MaskPsychoImage)(pool, pi0, pi1, blur_temp, scratch, mask, diff_ac);
}

void combine_channels_to_diffmap(RowPool& pool, const ImageF& mask, const Image3F& block_diff_dc, const Image3F& block_diff_ac,
                                 float xmul, ImageF* result) {
    HWY_DYNAMIC_DISPATCH(CombineChannelsToDiffmap)(pool, mask, block_diff_dc, block_diff_ac, xmul, result);
}

void separable_blur(RowPool& pool, const ImageF& in, const float* kernel, std::size_t half, ImageF* temp,
                    ImageF* out) {
    HWY_DYNAMIC_DISPATCH(SeparableBlur)(pool, in, kernel, half, temp, out);
}

}  // namespace vml::butteraugli
#endif  // HWY_ONCE
