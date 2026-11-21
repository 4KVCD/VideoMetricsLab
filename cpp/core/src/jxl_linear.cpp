// libjxl 0.12's linear sRGB reading of a 16-bit PNG (see jxl_linear.h).
//
// Which steps JxlCms runs (lib/jxl/cms/jxl_cms.cc, JxlCmsInit) for a picture
// converted to linear sRGB:
//   - PQ and HLG, and sRGB with sRGB primaries: libjxl's own curve, then the
//     picture is treated as linear in its own primaries (HLG adds its OOTF).
//   - Linear sRGB in, or a curve undone above with sRGB primaries: nothing more.
//   - Otherwise skcms (third_party/skcms) with the PNG's ICC profile: a curve
//     per channel, then one matrix when the primaries differ from sRGB's.
// The curve and matrix constants below are what skcms reads from the ICC
// profiles libjxl makes for each description (dumped with the development
// harness's skcms command, exact as hex floats).

#include "vml/jxl_linear.h"

#include <bit>
#include <cmath>
#include <cstring>

namespace vml {

namespace {

// ---------------------------------------------------------------- skcms ---
// skcms's fast log2, exp2 and pow (src/Transform_inl.h), one lane at a time.
// The baseline x86 build that libjxl's tools use has no FMA, and every step
// below is written in skcms's order so the rounding is the same.

float skcms_log2(float x) {
    const std::int32_t bits = std::bit_cast<std::int32_t>(x);
    const float e = static_cast<float>(bits) * (1.0f / (1 << 23));
    const float m = std::bit_cast<float>((bits & 0x007fffff) | 0x3f000000);
    return e - 124.225514990f - 1.498030302f * m - 1.725879990f / (0.3520887068f + m);
}

float skcms_exp2(float x) {
    const float fract = x - std::floor(x);
    float fbits = (1.0f * (1 << 23)) * (x + 121.274057500f - 1.490129070f * fract + 27.728023300f / (4.84252568f - fract));
    // max_ and min_ as skcms writes them: if_then_else(x < y, y, x) and
    // if_then_else(x > y, y, x); 2139095040 is the bit pattern of +infinity.
    fbits = fbits < 0.0f ? 0.0f : fbits;
    fbits = fbits > 2139095040.0f ? 2139095040.0f : fbits;
    return std::bit_cast<float>(static_cast<std::int32_t>(fbits));
}

float skcms_pow(float x, float y) {
    return (x == 0.0f || x == 1.0f) ? x : skcms_exp2(skcms_log2(x) * y);
}

// skcms's parametric curve (apply_tf) for a non-negative x:
// x < d ? c*x + f : (a*x + b)^g + e.
struct SkcmsCurve {
    float g, a, b, c, d, e, f;
};

float skcms_apply_tf(const SkcmsCurve& tf, float x) {
    return x < tf.d ? tf.c * x + tf.f : skcms_pow(tf.a * x + tf.b, tf.g) + tf.e;
}

// The linear curve in libjxl's linear ICC profiles (g = a = c = 1, d = 0).
// skcms does not see it as the identity, so values go through skcms_pow(x, 1).
float skcms_linear(float x) {
    return skcms_pow(x, 1.0f);
}

// ---------------------------------------------------------------- libjxl ---
// libjxl's own curves (lib/jxl/cms/transfer_functions-inl.h). Their Highway
// code computes a 4/4 rational polynomial with MulAdd (fused on AVX2,
// AVX-512 and NEON, a multiply then an add on SSE4) and an exact division.

float mul_add(float a, float b, float c, bool fused) {
    return fused ? std::fma(a, b, c) : a * b + c;
}

float rational_polynomial(float x, const float (&p)[5], const float (&q)[5], bool fused) {
    float yp = p[4];
    float yq = q[4];
    for (int i = 3; i >= 0; --i) {
        yp = mul_add(yp, x, p[i], fused);
        yq = mul_add(yq, x, q[i], fused);
    }
    return yp / yq;
}

// TF_SRGB::DisplayFromEncoded for a non-negative x.
float jxl_srgb_to_linear(float x, bool fused) {
    static constexpr float p[5] = {2.200248328e-04f, 1.043637593e-02f, 1.624820318e-01f, 7.961564959e-01f,
                                   8.210152774e-01f};
    static constexpr float q[5] = {2.631846970e-01f, 1.076976492e+00f, 4.987528350e-01f, -5.512498495e-02f,
                                   6.521209011e-03f};
    constexpr float low_divisor_inverse = 1.0f / 12.92f;
    const float magnitude = x > 0.04045f ? rational_polynomial(x, p, q, fused) : x * low_divisor_inverse;
    return std::fabs(magnitude);  // the AndNot of the sign bit
}

// TF_PQ::DisplayFromEncoded for a non-negative x; 1 is 10000 nits, which is
// also the picture's intensity target, so the final scaling is by 1.
float jxl_pq_to_linear(float x, bool fused) {
    static constexpr float p[5] = {2.62975656e-04f, -6.23553089e-03f, 7.38602301e-01f, 2.64553172e+00f,
                                   5.50034862e-01f};
    static constexpr float q[5] = {4.21350107e+02f, -4.28736818e+02f, 1.74364667e+02f, -3.39078883e+01f,
                                   2.67718770e+00f};
    const float x_plus_x_squared = mul_add(x, x, x, fused);
    return std::fabs(rational_polynomial(x_plus_x_squared, p, q, fused) * (10000.0f / 10000.0f));
}

// TF_HLG_Base::DisplayFromEncoded (double maths; copysignf rounds the result
// to float), for a non-negative e.
float jxl_hlg_to_scene(float e_float) {
    constexpr double a = 0.17883277;
    constexpr double ra = 1.0 / a;
    constexpr double b = 1 - 4 * a;
    constexpr double c = 0.5599107295;
    constexpr double inverse_12 = 1.0 / 12.0;
    const double e = e_float;
    if (e == 0.0) return 0.0f;
    if (e <= 0.5) return static_cast<float>(e * e * (1.0 / 3));
    return static_cast<float>((std::exp((e - c) * ra) + b) * inverse_12);
}

// ------------------------------------------------------- descriptions ---

enum class Curve { identity, skcms_gamma, skcms_tf, jxl_srgb, jxl_pq, jxl_hlg };

struct Reading {
    Curve curve;
    SkcmsCurve skcms;  // for skcms_gamma (g only) and skcms_tf
    float intensity_target;
};

std::optional<Reading> reading_of(int transfer, int primaries) {
    constexpr float none = 0.0f;
    switch (transfer) {
        case 1:  // BT.709
        case 7:  // SMPTE 240M; libjxl reads both as gamma 1/0.41667 (gAMA)
            return Reading{Curve::skcms_gamma, {0x1.33328p+1f, 1, none, none, none, none, none}, 255.f};
        case 4:  // gamma 2.2
            return Reading{Curve::skcms_gamma, {0x1.1999p+1f, 1, none, none, none, none, none}, 255.f};
        case 5:  // gamma 2.8
        case 6:  // BT.601
            return Reading{Curve::skcms_gamma, {0x1.6667p+1f, 1, none, none, none, none, none}, 255.f};
        case 8:  // linear
            return Reading{primaries == 1 ? Curve::identity : Curve::skcms_tf, {1, 1, none, 1, none, none, none},
                           255.f};
        case 13:  // sRGB
            if (primaries == 1) return Reading{Curve::jxl_srgb, {}, 255.f};
            return Reading{Curve::skcms_tf,
                           {0x1.3333p+1f, 0x1.e54ep-1f, 0x1.ab2p-5f, 0x1.3dp-4f, 0x1.4b6p-5f, none, none},
                           255.f};
        case 16:  // PQ
            return Reading{Curve::jxl_pq, {}, 10000.f};
        case 17:  // SMPTE 428; skcms's apply_tf with c = 1, d = 0
            return Reading{Curve::skcms_tf, {0x1.4ccdp+1f, 1, none, 1, none, none, none}, 255.f};
        case 18:  // HLG
            return Reading{Curve::jxl_hlg, {}, 1000.f};
        default:
            return std::nullopt;
    }
}

// skcms's matrix from the source primaries to sRGB's (D65 white) for each
// primaries code; sRGB's own (1) has none.
//
// For PQ and HLG, skcms gets the linear profile JxlCmsInit builds from the
// description it reads back out of the PNG's ICC profile. A profile without a
// CICP tag (primaries other than sRGB's, BT.2020's and P3's) stores the
// primaries as XYZ values, and reading them back rounds them (0.31 comes back
// as 0.310037), so that matrix differs slightly from the one skcms takes from
// the original profile.
std::optional<std::optional<std::array<float, 9>>> matrix_of(int primaries, bool linear_profile) {
    using Matrix = std::array<float, 9>;
    if (linear_profile) {
        switch (primaries) {
            case 4:
                return Matrix{0x1.7c7c3ep+0f,  -0x1.9d55acp-2f, -0x1.52216p-4f,  -0x1.9c4514p-6f, 0x1.e88094p-1f,
                              0x1.22de44p-4f,  -0x1.bdbafp-6f,  -0x1.693738p-5f, 0x1.1240f2p+0f};
            case 5:
                return Matrix{0x1.0b47c8p+0f,  -0x1.684fdap-5f, -0x1.12p-17f,  -0x1.99398p-17f, 0x1.fffba4p-1f,
                              0x1.4ep-20f,     0x1.3f7p-16f,    0x1.82acp-7f,  0x1.f9f36cp-1f};
            case 6:
            case 7:
                return Matrix{0x1.e10f9cp-1f, 0x1.9ba1cap-5f,  0x1.4e614p-7f,  0x1.22a3dcp-6f, 0x1.ee7912p-1f,
                              0x1.0dbf5ap-6f, -0x1.a2bb2p-10f, -0x1.1da3cp-8f, 0x1.01868cp+0f};
            default:
                break;  // CICP profiles: the same matrix
        }
    }
    switch (primaries) {
        case 1:  // BT.709 / sRGB
            return std::optional<Matrix>{};
        case 4:  // BT.470M
            return Matrix{0x1.7c791ep+0f,  -0x1.9d404cp-2f, -0x1.52552p-4f, -0x1.9c0694p-6f, 0x1.e87504p-1f,
                          0x1.231b94p-4f,  -0x1.bdbf8cp-6f, -0x1.694e3p-5f, 0x1.12407cp+0f};
        case 5:  // BT.470BG
            return Matrix{0x1.0b47c8p+0f,  -0x1.68b41ap-5f, -0x1.12p-17f,  -0x1.99398p-17f, 0x1.fffd9ap-1f,
                          0x1.4ep-20f,     0x1.3f7p-16f,    0x1.82a2cp-7f, 0x1.f9f36cp-1f};
        case 6:  // BT.601 525
        case 7:  // SMPTE 240M
            return Matrix{0x1.e10f9cp-1f, 0x1.9b3d8ap-5f,  0x1.4fb38p-7f, 0x1.22a3dcp-6f, 0x1.ee7b06p-1f,
                          0x1.0d82dep-6f, -0x1.a2bb2p-10f, -0x1.1db6p-8f, 0x1.018806p+0f};
        case 9:  // BT.2020
            return Matrix{0x1.a91a3ap+0f,  -0x1.2cdf7cp-1f, -0x1.2a5bcp-4f,  -0x1.fe5072p-4f, 0x1.2205b2p+0f,
                          -0x1.11c924p-7f, -0x1.2911cap-6f, -0x1.9bf8e8p-4f, 0x1.1e6424p+0f};
        case 12:  // Display P3
            return Matrix{0x1.3999b2p+0f,  -0x1.ccad38p-3f, -0x1.368p-15f,   -0x1.58d9a6p-5f, 0x1.0ac34p+0f,
                          0x1.a408p-16f,   -0x1.417538p-6f, -0x1.4211c2p-4f, 0x1.19272cp+0f};
        default:
            return std::nullopt;
    }
}

// The luminance of each primary that libjxl weighs HLG's OOTF with
// (GetPrimariesLuminances in jxl_cms.cc), from the primaries as read back out
// of the ICC profile (see matrix_of).
std::array<float, 3> hlg_luminances(int primaries) {
    switch (primaries) {
        case 4:
            return {0x1.322892p-2f, 0x1.2c448ap-1f, 0x1.d5396p-4f};
        case 5:
            return {0x1.c6ac7ap-3f, 0x1.69ced8p-1f, 0x1.24304cp-4f};
        case 6:
        case 7:
            return {0x1.b2ee9p-3f, 0x1.66f288p-1f, 0x1.628ea4p-4f};
        case 9:
            return {0x1.0d0148p-2f, 0x1.5b229p-1f, 0x1.e5ccb4p-5f};
        case 12:
            return {0x1.d4f09cp-3f, 0x1.622b8ep-1f, 0x1.44c25ap-4f};
        default:  // 1
            return {0x1.b37dc6p-3f, 0x1.6e2a76p-1f, 0x1.27b0bep-4f};
    }
}

float curve_value(const Reading& reading, bool has_matrix, float x, bool fused) {
    switch (reading.curve) {
        case Curve::identity:
            return x;
        case Curve::skcms_gamma:
            return skcms_pow(x, reading.skcms.g);
        case Curve::skcms_tf:
            return skcms_apply_tf(reading.skcms, x);
        case Curve::jxl_srgb:
            return jxl_srgb_to_linear(x, fused);
        case Curve::jxl_pq: {
            // Linear in the picture's primaries; skcms then reads it through
            // the linear profile's curve when it has a matrix to apply.
            const float linear = jxl_pq_to_linear(x, fused);
            return has_matrix ? skcms_linear(linear) : linear;
        }
        case Curve::jxl_hlg:
            return jxl_hlg_to_scene(x);  // the OOTF and skcms follow per pixel
    }
    return x;
}

// skcms's matrix stage: each output is m0*r + m1*g + m2*b, left to right.
void apply_matrix(const std::array<float, 9>& m, float& r, float& g, float& b) {
    const float red = m[0] * r + m[1] * g + m[2] * b;
    const float green = m[3] * r + m[4] * g + m[5] * b;
    const float blue = m[6] * r + m[7] * g + m[8] * b;
    r = red;
    g = green;
    b = blue;
}

}  // namespace

std::optional<JxlLinearRgb> JxlLinearRgb::make(int transfer, int primaries, bool fused_multiply_add) {
    const std::optional<Reading> reading = reading_of(transfer, primaries);
    const bool linear_profile = transfer == 16 || transfer == 18;
    const auto matrix = matrix_of(primaries, linear_profile);
    if (!reading || !matrix) return std::nullopt;

    JxlLinearRgb result;
    result.matrix_ = *matrix;
    result.intensity_target_ = reading->intensity_target;
    auto curve = std::make_shared<std::vector<float>>(65536);
    // libjxl's sample scaling: the 16-bit value times 1/65535 in float.
    const float scale = 1.0f / 65535.0f;
    for (std::uint32_t sample = 0; sample < 65536; ++sample) {
        (*curve)[sample] =
            curve_value(*reading, result.matrix_.has_value(), static_cast<float>(sample) * scale, fused_multiply_add);
    }
    result.curve_ = std::move(curve);

    if (reading->curve == Curve::jxl_hlg) {
        // ApplyHlgOotf's gamma for the picture's intensity target (1000 nits:
        // outside 295..305, so the OOTF runs).
        const float target = reading->intensity_target;
        const float gamma = 1.2f * std::pow(1.111f, std::log2(target * 1e-3f));
        result.hlg_ = HlgOotf{hlg_luminances(primaries), gamma - 1};
        result.linear_profile_after_hlg_ = result.matrix_.has_value();
    }
    return result;
}

void JxlLinearRgb::convert_row(const std::uint16_t* rgb48, std::size_t width, float* red, float* green,
                               float* blue) const {
    const float* curve = curve_->data();
    if (!hlg_) {
        for (std::size_t x = 0; x < width; ++x) {
            float r = curve[rgb48[3 * x]];
            float g = curve[rgb48[3 * x + 1]];
            float b = curve[rgb48[3 * x + 2]];
            if (matrix_) apply_matrix(*matrix_, r, g, b);
            red[x] = r;
            green[x] = g;
            blue[x] = b;
        }
        return;
    }
    // HLG: ApplyHlgOotf (forward, gamma > 1 so no highlight clamp) on the
    // scene light, then skcms with the linear profile.
    const std::array<float, 3>& l = hlg_->luminances;
    for (std::size_t x = 0; x < width; ++x) {
        float r = curve[rgb48[3 * x]];
        float g = curve[rgb48[3 * x + 1]];
        float b = curve[rgb48[3 * x + 2]];
        const float luminance = r * l[0] + g * l[1] + b * l[2];
        const float ratio = std::pow(luminance, hlg_->exponent);
        if (std::isfinite(ratio)) {
            r *= ratio;
            g *= ratio;
            b *= ratio;
        }
        if (linear_profile_after_hlg_) {
            r = skcms_linear(r);
            g = skcms_linear(g);
            b = skcms_linear(b);
            apply_matrix(*matrix_, r, g, b);
        }
        red[x] = r;
        green[x] = g;
        blue[x] = b;
    }
}

}  // namespace vml
