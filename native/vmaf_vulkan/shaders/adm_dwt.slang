// ADM, the wavelet transform of both pictures. SCALE0: adm_dwt2_8_vert_hori_kernel
// of adm_dwt2.cu (the pictures to 16-bit bands); otherwise
// dwt_s123_combined_vert_kernel and _hori_kernel (the previous scale's
// approximation band to 32-bit bands). Each output takes its four columns'
// vertical sums itself instead of from a temporary image: the same integers.
//
// A band image holds four values per position: a, h, v, d. Scale 0's a is
// kept in 32 bits (the CUDA code's i4 copy; nothing reads the 16-bit one).
#include "common.slang"

struct Params {
    int w, h;          // of the input
    int inStride;      // words per picture row, or positions per band row
    int outStride;
    int shiftV, addV;
    int shiftH, addH;
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> refIn;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<int> refOut;
[[vk::binding(3, 0)]] RWStructuredBuffer<int> disOut;

static const int kLo[4] = { 15826, 27411, 7345, -4240 };
static const int kHi[4] = { -4240, -7345, 27411, -15826 };
static const int kLoSum = 46342;

// calculate_indices
int4 indices(int n, int limit)
{
    int4 at = int4(2 * n - 1, 2 * n, 2 * n + 1, 2 * n + 2);
    if (n == 0)
        at = int4(1, 0, 1, 2);
    if (at.x >= limit) at.x = 2 * limit - at.x - 1;
    if (at.y >= limit) at.y = 2 * limit - at.y - 1;
    if (at.z >= limit) at.z = 2 * limit - at.z - 1;
    if (at.w >= limit) at.w = 2 * limit - at.w - 1;
    return at;
}

void transform(StructuredBuffer<uint> source, RWStructuredBuffer<int> bands, int4 columns, int4 rows, int at)
{
#if SCALE0
    int lo[4], hi[4];
    for (int c = 0; c < 4; ++c) {
        int sumLo = 0, sumHi = 0;
        for (int i = 0; i < 4; ++i) {
            int sample = int(pixel(source, columns[c], rows[i], pc.inStride));
            sumLo += kLo[i] * sample;
            sumHi += kHi[i] * sample;
        }
        sumLo -= kLoSum * pc.addV;
        lo[c] = sext16((sumLo + pc.addV) >> pc.shiftV);
        hi[c] = sext16((sumHi + pc.addV) >> pc.shiftV);
    }
    int a = 0, v = 0, h = 0, d = 0;
    for (int c = 0; c < 4; ++c) {
        a += kLo[c] * lo[c];
        v += kHi[c] * lo[c];
        h += kLo[c] * hi[c];
        d += kHi[c] * hi[c];
    }
    bands[at + 0] = (a + pc.addH) >> pc.shiftH;
    bands[at + 1] = sext16((h + pc.addH) >> pc.shiftH);
    bands[at + 2] = sext16((v + pc.addH) >> pc.shiftH);
    bands[at + 3] = sext16((d + pc.addH) >> pc.shiftH);
#else
    int lo[4], hi[4];
    for (int c = 0; c < 4; ++c) {
        int64_t sumLo = 0, sumHi = 0;
        for (int i = 0; i < 4; ++i) {
            int64_t sample = int64_t(int(source[(rows[i] * pc.inStride + columns[c]) * 4]));
            sumLo += int64_t(kLo[i]) * sample;
            sumHi += int64_t(kHi[i]) * sample;
        }
        lo[c] = int((sumLo + pc.addV) >> pc.shiftV);
        hi[c] = int((sumHi + pc.addV) >> pc.shiftV);
    }
    int64_t a = 0, v = 0, h = 0, d = 0;
    for (int c = 0; c < 4; ++c) {
        a += int64_t(kLo[c]) * lo[c];
        v += int64_t(kHi[c]) * lo[c];
        h += int64_t(kLo[c]) * hi[c];
        d += int64_t(kHi[c]) * hi[c];
    }
    bands[at + 0] = int((a + pc.addH) >> pc.shiftH);
    bands[at + 1] = int((h + pc.addH) >> pc.shiftH);
    bands[at + 2] = int((v + pc.addH) >> pc.shiftH);
    bands[at + 3] = int((d + pc.addH) >> pc.shiftH);
#endif
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int x = int(id.x), y = int(id.y);
    if (x >= (pc.w + 1) / 2 || y >= (pc.h + 1) / 2)
        return;
    int4 columns = indices(x, pc.w), rows = indices(y, pc.h);
    int at = (y * pc.outStride + x) * 4;
    transform(refIn, refOut, columns, rows, at);
    transform(disIn, disOut, columns, rows, at);
}
