// VIF, vertical pass: filter1d_8_vertical_kernel / filter1d_16_vertical_kernel
// of filter1d.cu. One shader per scale (SCALE, its filter width FW and the
// next scale's FW_RD); scale 0 reads the pictures (SRC_PICTURE, with BPC16
// for more than 8 bits), the others the previous scale's downsampled images.
//
// The 8-bit CUDA kernel sums its squares in 32 bits and the 16-bit one in 64;
// both are summed in 32-bit steps here (SMALL, Sum48).
#include "common.slang"
#include "vif_filter.slang"

struct Params {
    int w, h;
    int stride;                 // words per picture row, or samples per row
    uint shiftVP, addVP;
    uint shiftSq, addSq;
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> refIn;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> tmp;   // 8 words per pixel

#define OFF ((FW - FW_RD) / 2)
// 8-bit pictures: the sums of squares fit in 32 bits (65536 * 255 * 255), and
// are stored as they are, as by the CUDA code's 8-bit kernel.
#define SMALL (SRC_PICTURE && !BPC16)

// The sum of up to 17 products of a 32-bit and a 16-bit number, in 32-bit
// steps: a GPU without 64-bit integers of its own (Intel's) takes several
// times longer over each 64-bit multiply-add than over these. The products'
// high parts cannot overflow 32 bits: sum((c * r) >> 16) * r' is below
// sum(c) * 65536 = 2^32.
struct Sum48 { uint high, low, carries; };

void add48(inout Sum48 sum, uint a, uint b)
{
    sum.high += (a >> 16) * b;
    uint low = (a & 0xFFFFu) * b;
    sum.low += low;
    sum.carries += sum.low < low ? 1u : 0u;
}

uint64_t total48(Sum48 sum)
{
    return (uint64_t(sum.high) << 16) + (uint64_t(sum.carries) << 32) + uint64_t(sum.low);
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int x = int(id.x), y = int(id.y);
    if (x >= pc.w || y >= pc.h)
        return;
    uint mu1 = 0, mu2 = 0, refRd = 0, disRd = 0;
#if SMALL
    uint refSq = 0, disSq = 0, refDis = 0;
#else
    Sum48 refSq = { 0, 0, 0 }, disSq = { 0, 0, 0 }, refDis = { 0, 0, 0 };
#endif
    for (int fi = 0; fi < FW; ++fi) {
        int row = abs(y - FW / 2 + fi);
        if (row >= pc.h)
            row = 2 * pc.h - row - 2;
#if SRC_PICTURE
        uint r = pixel(refIn, x, row, pc.stride), d = pixel(disIn, x, row, pc.stride);
#else
        uint r = refIn[row * pc.stride + x], d = disIn[row * pc.stride + x];
#endif
        uint coefficient = kVifFilter[SCALE][fi];
        uint cr = coefficient * r, cd = coefficient * d;
        mu1 += cr;
        mu2 += cd;
#if SMALL
        refSq += cr * r;
        disSq += cd * d;
        refDis += cr * d;
#else
        add48(refSq, cr, r);
        add48(disSq, cd, d);
        add48(refDis, cr, d);
#endif
#if FW_RD > 0
        if (fi >= OFF && fi < FW - OFF) {
            uint next = kVifFilter[SCALE + 1][fi - OFF];
            refRd += next * r;
            disRd += next * d;
        }
#endif
    }
    uint at = uint(y * pc.w + x) * 8;
    tmp[at + 0] = ((mu1 + pc.addVP) >> pc.shiftVP) & 0xFFFFu;
    tmp[at + 1] = ((mu2 + pc.addVP) >> pc.shiftVP) & 0xFFFFu;
#if SMALL
    tmp[at + 2] = refSq;
    tmp[at + 3] = disSq;
    tmp[at + 4] = refDis;
#else
    tmp[at + 2] = uint((total48(refSq) + pc.addSq) >> pc.shiftSq);
    tmp[at + 3] = uint((total48(disSq) + pc.addSq) >> pc.shiftSq);
    tmp[at + 4] = uint((total48(refDis) + pc.addSq) >> pc.shiftSq);
#endif
#if FW_RD > 0
    tmp[at + 5] = ((refRd + pc.addVP) >> pc.shiftVP) & 0xFFFFu;
    tmp[at + 6] = ((disRd + pc.addVP) >> pc.shiftVP) & 0xFFFFu;
#endif
}
