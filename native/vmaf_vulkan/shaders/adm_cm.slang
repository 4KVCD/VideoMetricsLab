// ADM, contrast masking and the numerator's sum. SCALE0: adm_cm_line_kernel;
// otherwise i4_adm_cm_line_kernel; both with adm_cm_reduce_line_kernel
// (adm_cm.cu).
//
// The CUDA code masks every value into a temporary image and then sums each
// row of it, rounding once per row. Here one block of threads takes a row,
// masks and sums it and rounds: the same sum, without the image. The masking
// threshold is the same for the three bands, so it is calculated once for
// them instead of once per band.
#include "common.slang"

struct Params {
    int w, h;
    int startRow, startCol, endCol;
    int stride;
    // Per band, as scalars: Intel's driver reads an array here as zeros.
    uint rfactor0, rfactor1, rfactor2;
    int shiftSub0, shiftSub1, shiftSub2;
    int shiftSq0, shiftSq1, shiftSq2, addSq0, addSq1, addSq2;
    int shiftCub0, shiftCub1, shiftCub2, addCub0, addCub1, addCub2;
    int shiftInner, addInner;
    uint slot;
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> restored;
[[vk::binding(1, 0)]] StructuredBuffer<uint> csfA;
[[vk::binding(2, 0)]] StructuredBuffer<uint> csfF;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;

#define GROUP 128
groupshared int64_t gSums[GROUP * 3];

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    int i = pc.startRow + int(group.y);
    int rows[3] = { min(abs(i - 1), pc.h - 1), i, min(i + 1, pc.h - 1) };
    int64_t sums[3] = { 0, 0, 0 };
    const uint rfactor[3] = { pc.rfactor0, pc.rfactor1, pc.rfactor2 };
    const int shiftSub[3] = { pc.shiftSub0, pc.shiftSub1, pc.shiftSub2 };
    const int shiftSq[3] = { pc.shiftSq0, pc.shiftSq1, pc.shiftSq2 };
    const int addSq[3] = { pc.addSq0, pc.addSq1, pc.addSq2 };
    const int shiftCub[3] = { pc.shiftCub0, pc.shiftCub1, pc.shiftCub2 };
    const int addCub[3] = { pc.addCub0, pc.addCub1, pc.addCub2 };
    for (int j = pc.startCol + int(local); j < pc.endCol; j += GROUP) {
        int columns[3] = { min(abs(j - 1), pc.w - 1), j, min(j + 1, pc.w - 1) };
        int threshold = 0;
        for (int theta = 0; theta < 3; ++theta) {
            for (int di = 0; di < 3; ++di) {
                int row = rows[di] * pc.stride;
                threshold += int(csfF[(row + columns[0]) * 4 + theta]);
                if (di == 1) {
                    int center = abs(int(csfA[(row + j) * 4 + theta]));
#if SCALE0
                    threshold += sext16((8738 * center + 2048) >> 12);
#else
                    // (int32_t)(1u << 31) again: the rounding term is negative.
                    threshold += int((int64_t(286331153) * center - 2147483648ll) >> 32);
#endif
                } else {
                    threshold += int(csfF[(row + columns[1]) * 4 + theta]);
                }
                threshold += int(csfF[(row + columns[2]) * 4 + theta]);
            }
        }
        int at = (i * pc.stride + j) * 4;
        for (int b = 0; b < 3; ++b) {
            int source = int(restored[at + b]);
#if SCALE0
            int value = abs(int(rfactor[b] * uint(source))) - (threshold << shiftSub[b]);
#else
            int value = abs(int((int64_t(source) * int64_t(rfactor[b]) + 134217728) >> 28)) - threshold;
#endif
            value = max(value, 0);
            int square = int((int64_t(value) * value + addSq[b]) >> shiftSq[b]);
            sums[b] += (int64_t(square) * value + addCub[b]) >> shiftCub[b];
        }
    }
    for (int b = 0; b < 3; ++b)
        gSums[local * 3 + b] = sums[b];
    GroupMemoryBarrierWithGroupSync();
    if ((local & 15) == 0) {
        for (int b = 0; b < 3; ++b) {
            int64_t sum = 0;
            for (uint t = 0; t < 16; ++t)
                sum += gSums[(local + t) * 3 + b];
            gSums[local * 3 + b] = sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        for (int b = 0; b < 3; ++b) {
            int64_t sum = 0;
            for (uint t = 0; t < GROUP; t += 16)
                sum += gSums[t * 3 + b];
            sum = (sum + pc.addInner) >> pc.shiftInner;
            if (sum != 0)
                acc_add64(acc, pc.slot + uint(b), uint64_t(sum));
        }
    }
}
