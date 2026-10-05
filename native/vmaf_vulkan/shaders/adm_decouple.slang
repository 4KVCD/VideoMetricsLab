// ADM, decoupling and contrast sensitivity in one pass. SCALE0:
// adm_decouple_kernel (adm_decouple.cu) and adm_csf_kernel (adm_csf.cu) on
// 16-bit bands; otherwise adm_decouple_s123_kernel and i4_adm_csf_kernel.
// The CUDA code keeps the additive image (decouple_a) only to filter it in
// its csf kernel; here it is filtered where it is calculated.
//
// The enhancement gain limit is an integer here (VMAF's 100, VMAF NEG's 1);
// the CUDA code multiplies by it as a double and converts back, saturating.
#include "common.slang"

struct Params {
    int top, bottom, left, right;
    int inStride, outStride;
    int gainLimit;
    uint rfactor0, rfactor1, rfactor2;  // scalars: Intel's driver reads an array here as zeros
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> refBands;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disBands;
[[vk::binding(2, 0)]] RWStructuredBuffer<int> restored;  // decouple_r
[[vk::binding(3, 0)]] RWStructuredBuffer<int> csfA;
[[vk::binding(4, 0)]] RWStructuredBuffer<int> csfF;
// (int32_t)(2^30 / (float)(i - 32768)), from the host: the CUDA kernel's
// float division, which is correctly rounded there and on the host alike.
[[vk::binding(5, 0)]] StructuredBuffer<uint> divTable;

// (float)v of a non-negative integer: m * 2^e, m below 2^24.
void to_float(uint64_t v, out uint m, out int e)
{
    if (v < (1ull << 24)) {
        m = uint(v);
        e = 0;
        return;
    }
    int sh = msb64(v) - 23;
    uint64_t q = v >> sh, rem = v & ((1ull << sh) - 1), half = 1ull << (sh - 1);
    if (rem > half || (rem == half && (q & 1) != 0)) {
        q++;
        if (q == (1ull << 24)) {
            q >>= 1;
            sh++;
        }
    }
    m = uint(q);
    e = sh;
}

// A non-negative 64-bit integer as a float, to within a few units in the
// last place whatever the driver's own conversion does.
float approximately(int64_t v)
{
    return float(uint(uint64_t(v) >> 32)) * 4294967296.0 + float(uint(v));
}

// Whether the two vectors are within one degree, as the CUDA kernels decide
// it: the three integers rounded to float, then in double
//     ot >= 0 && ot * ot >= (cos_1deg_sq * o_mag_sq) * t_mag_sq
// (the common division by 4096 * 4096 changes nothing). cos_1deg_sq is the
// float 0xFFEC0A * 2^-24.
bool within_one_degree(int64_t otDp, int64_t oMagSq, int64_t tMagSq)
{
    if (otDp < 0)
        return false;
    // Nearly every vector is far from one degree: decided in float when the
    // two sides differ by more than float's error can explain, exactly
    // otherwise.
    const float ot = approximately(otDp);
    const float left_ = ot * ot, right_ = 0.99969541 * approximately(oMagSq) * approximately(tMagSq);
    if (left_ > right_ * 1.0001)
        return true;
    if (left_ < right_ * 0.9999)
        return false;
    uint mot, mo, mt;
    int eot, eo, et;
    to_float(uint64_t(otDp), mot, eot);
    to_float(uint64_t(oMagSq), mo, eo);
    to_float(uint64_t(tMagSq), mt, et);
    if (mo == 0 || mt == 0)
        return true;
    if (mot == 0)
        return false;
    uint64_t left = uint64_t(mot) * mot;           // exact in a double
    uint64_t right;
    int re;
    round53(mul64(uint64_t(0xFFEC0Au) * mo, uint64_t(mt)), right, re);
    re += eo + et - 24;
    int leftTop = msb64(left), topLeft = leftTop + 2 * eot, topRight = 52 + re;
    if (topLeft != topRight)
        return topLeft > topRight;
    return (left << (52 - leftTop)) >= right;
}

static const int kAdd[3] = { 16384, 16384, 65535 };
static const int kShift[3] = { 15, 15, 17 };

int saturate32(int64_t v)
{
    return int(clamp(v, int64_t(-2147483647 - 1), int64_t(2147483647)));
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int i = pc.top + int(id.y), j = pc.left + int(id.x);
    if (i >= pc.bottom || j >= pc.right)
        return;
    int from = (i * pc.inStride + j) * 4, to = (i * pc.outStride + j) * 4;
    int o[3], t[3];
    for (int b = 0; b < 3; ++b) {
        o[b] = int(refBands[from + 1 + b]);
        t[b] = int(disBands[from + 1 + b]);
    }
    int64_t otDp = int64_t(o[0]) * t[0] + int64_t(o[1]) * t[1];
    int64_t oMagSq = int64_t(o[0]) * o[0] + int64_t(o[1]) * o[1];
    int64_t tMagSq = int64_t(t[0]) * t[0] + int64_t(t[1]) * t[1];
    bool aligned = within_one_degree(otDp, oMagSq, tMagSq);
    int64_t limit = aligned ? int64_t(pc.gainLimit) : int64_t(1);
    const uint rfactor[3] = { pc.rfactor0, pc.rfactor1, pc.rfactor2 };

    for (int b = 0; b < 3; ++b) {
        int additive;
#if SCALE0
        int k = 32768;
#if VARIANT == 1
        // As the CUDA kernel writes it, in 64 bits: what VARIANT 0 was until
        // 2026-10. Kept for scripts/diagnose_vmaf_vulkan.py.
        if (o[b] != 0)
            k = clamp(int((int64_t(int(divTable[o[b] + 32768])) * t[b] + 16384) >> 15), 0, 32768);
        int product = k * o[b];
        int rest = saturate32(int64_t((product + 16384) >> 15) * limit);
#else
        int quotient = 0;  // divTable's entry: 2^30 / o
        if (o[b] != 0)
            quotient = int(divTable[o[b] + 32768]);
#if VARIANT == 0
        // k = clamp((quotient * t + 16384) >> 15, 0, 32768) and
        // rest = ((k * o + 16384) >> 15) * limit, in 32-bit steps. The CUDA
        // kernel's 64-bit expressions (VARIANT 1) gave rest = 0 on a Radeon
        // 780M whenever o and t were both negative: its driver's fault, and
        // its sums failed the probe. With quotient = high * 2^15 + low, low
        // in 0..32767:
        //   (quotient * t + 16384) >> 15 = high * t + ((low * t + 16384) >> 15)
        // exactly, and neither term passes 2^30: the bands are 16-bit here.
        // rest is at most 32768 * 100, so nothing saturates. The same sums
        // as the 64-bit ones on every GPU that gets those right.
        if (o[b] != 0)
            k = clamp((quotient >> 15) * t[b] + (((quotient & 32767) * t[b] + 16384) >> 15), 0, 32768);
        int product = k * o[b];
        int rest = ((product + 16384) >> 15) * int(limit);
#else
        if (o[b] != 0)
            k = clamp(int((int64_t(quotient) * t[b] + 16384) >> 15), 0, 32768);
        int product = k * o[b];
        int rest = saturate32(int64_t((product + 16384) >> 15) * limit);
#endif
#endif
        if (aligned) {
            if (product > 0) rest = min(rest, t[b]);
            if (product < 0) rest = max(rest, t[b]);
        }
#if VARIANT >= 2
        // For scripts/diagnose_vmaf_vulkan.py: one step of the above in
        // place of the result, to see which a driver gets wrong.
        int64_t wide = int64_t(quotient) * t[b] + 16384;
        int shown[6] = { quotient, k, int(wide >> 15), int(uint(wide)), int(uint(wide >> 32)), product };
        restored[to + b] = shown[VARIANT - 2];
#else
        restored[to + b] = sext16(rest);
        additive = sext16(t[b] - rest);

        int filtered = sext16((int(rfactor[b] * uint(additive)) + kAdd[b]) >> kShift[b]);
        csfA[to + b] = filtered;
        csfF[to + b] = sext16((4369 * abs(filtered) + 2048) >> 12);
#endif
#else
        uint magnitude = uint(abs(o[b]));
        int shift = 0;
        uint top = magnitude;
        if (magnitude >= 32768u) {  // get_best15_from32
            shift = msb32(magnitude) - 14;
            top = ((magnitude + (1u << (shift - 1))) >> shift) & 0xFFFFu;
        }
        int64_t k = 32768;
        if (o[b] != 0) {
            int64_t sign = o[b] < 0 ? -1 : 1;
            k = (int64_t(1073741824 / int(top)) * t[b] * sign + int64_t(1 << (14 + shift))) >> (15 + shift);
            k = clamp(k, int64_t(0), int64_t(32768));
        }
        int rest = saturate32(int64_t(int((k * o[b] + 16384) >> 15)) * limit);
        if (aligned && k > 0) {
            if (o[b] > 0) rest = min(rest, t[b]);
            if (o[b] < 0) rest = max(rest, t[b]);
        }
        restored[to + b] = rest;
        additive = t[b] - rest;

        int filtered = int((int64_t(rfactor[b]) * int64_t(additive) + 134217728) >> 28);
        csfA[to + b] = filtered;
        // The CUDA kernel's rounding term is (int32_t)(1u << 31): negative.
        csfF[to + b] = int((int64_t(143165577) * abs(filtered) - 2147483648ll) >> 32);
#endif
    }
}
