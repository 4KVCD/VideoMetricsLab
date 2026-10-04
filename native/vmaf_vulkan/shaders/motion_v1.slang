// Motion for VMAF v1: motion_score_pipeline_8 / _16 of libvmaf's
// integer_motion.c (the CPU code: VMAF v1's motion has no CUDA version). The
// difference of two reference frames is blurred with the 5-tap filter, each
// column rounded before the horizontal pass, and its absolute values summed.
#include "common.slang"

struct Params {
    int w, h, strideWords;
    int shiftY, addY;   // the picture's bit depth, and half of 1 << depth
    uint slot;
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> current;
[[vk::binding(1, 0)]] StructuredBuffer<uint> previous;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> acc;

static const int kFilter[5] = { 3571, 16004, 26386, 16004, 3571 };

groupshared uint gSum[256];

int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 2;
    return idx;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint local : SV_GroupIndex)
{
    int x = int(id.x), y = int(id.y);
    uint distance = 0;
    if (x < pc.w && y < pc.h) {
        int rows[5];
        for (int k = 0; k < 5; ++k)
            rows[k] = mirror(y - 2 + k, pc.h);
        int64_t blurred = 0;
        for (int xf = 0; xf < 5; ++xf) {
            int column = mirror(x - 2 + xf, pc.w);
            int64_t vertical = 0;
            for (int k = 0; k < 5; ++k) {
                int difference = int(pixel(previous, column, rows[k], pc.strideWords))
                               - int(pixel(current, column, rows[k], pc.strideWords));
                vertical += int64_t(kFilter[k] * difference);
            }
            int rounded = int((vertical + pc.addY) >> pc.shiftY);
            blurred += int64_t(kFilter[xf]) * rounded;
        }
        distance = uint(abs(int((blurred + 32768) >> 16)));
    }
    gSum[local] = distance;
    GroupMemoryBarrierWithGroupSync();
    if (local < 16) {
        uint sum = 0;
        for (uint i = 0; i < 16; ++i)
            sum += gSum[local * 16 + i];
        gSum[local * 16] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        uint sum = 0;
        for (uint i = 0; i < 16; ++i)
            sum += gSum[i * 16];
        if (sum != 0)
            acc_add64(acc, pc.slot, uint64_t(sum));
    }
}
