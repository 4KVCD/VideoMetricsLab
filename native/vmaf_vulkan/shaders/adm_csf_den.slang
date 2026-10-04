// ADM, the denominator: adm_csf_den_scale_line_kernel (SCALE0) and
// adm_csf_den_s123_line_kernel of adm_csf_den.cu, the sum of the reference
// bands' cubes.
//
// The CUDA kernels round each warp's partial sum before adding it to the
// total, so the total depends on which columns share a warp. The same
// columns share one here: a block of 128 threads takes 1024 columns of a
// row, thread t the columns t, t + 128, ..., and threads 32k..32k+31 are a
// warp.
#include "common.slang"

struct Params {
    int top, left, right;
    int stride;
    uint shiftSq, addSq;
    uint shiftCub, addCub;
    uint shiftAccum, addAccum;
    uint slot;
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> refBands;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> acc;

groupshared uint64_t gSums[128 * 3];

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    int row = (pc.top + int(group.y)) * pc.stride;
    int j = pc.left + 1024 * int(group.x) + int(local);
    uint64_t sums[3] = { 0, 0, 0 };
    for (int i = 0; i < 8; ++i) {
        if (j < pc.right) {
            for (int b = 0; b < 3; ++b) {
                int value = int(refBands[(row + j) * 4 + 1 + b]);
#if SCALE0
                uint64_t t = uint64_t(uint(abs(value)) & 0xFFFFu);
                sums[b] += t * t * t;
#else
                uint64_t t = uint64_t(uint(abs(value)));
                sums[b] += ((((t * t + pc.addSq) >> pc.shiftSq) * t) + pc.addCub) >> pc.shiftCub;
#endif
            }
        }
        j += 128;
    }
    for (int b = 0; b < 3; ++b)
        gSums[local * 3 + b] = sums[b];
    GroupMemoryBarrierWithGroupSync();
    if ((local & 31) == 0) {
        for (int b = 0; b < 3; ++b) {
            uint64_t sum = 0;
            for (uint t = 0; t < 32; ++t)
                sum += gSums[(local + t) * 3 + b];
            sum = (sum + pc.addAccum) >> pc.shiftAccum;
            if (sum != 0)
                acc_add64(acc, pc.slot + uint(b), sum);
        }
    }
}
