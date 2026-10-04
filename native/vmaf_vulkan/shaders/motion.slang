// Motion: calculate_motion_score_kernel_8bpc / _16bpc of motion_score.cu.
// Blurs the reference with the 5-tap filter (each column rounded before the
// horizontal pass, as the CUDA kernel does), keeps the blur for the next
// frame and sums its absolute difference from the previous frame's blur.
#include "common.slang"

struct Params {
    int w, h, strideWords;
    int shiftY, addY;   // the picture's bit depth, and half of 1 << depth
    uint slot;
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> blur;
[[vk::binding(2, 0)]] StructuredBuffer<uint> previous;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;

static const uint kFilter[5] = { 3571, 16004, 26386, 16004, 3571 };

groupshared uint gSum[256];

int mirror(int idx, int sup)
{
    int o = abs(idx);
    return (o < sup) ? o : (sup - (o - sup + 2));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint local : SV_GroupIndex)
{
    int x = int(id.x), y = int(id.y);
    uint distance = 0;
    if (x < pc.w && y < pc.h) {
        int rows[5];
        for (int yf = 0; yf < 5; ++yf)
            rows[yf] = mirror(y - 2 + yf, pc.h);
        uint blurred = 0;
        for (int xf = 0; xf < 5; ++xf) {
            int column = mirror(x - 2 + xf, pc.w);
            uint vertical = 0;
            for (int yf = 0; yf < 5; ++yf)
                vertical += kFilter[yf] * pixel(src, column, rows[yf], pc.strideWords);
            blurred += kFilter[xf] * ((vertical + uint(pc.addY)) >> uint(pc.shiftY));
        }
        blurred = (blurred + 32768u) >> 16;
        int at = y * pc.w + x;
        blur[at] = blurred & 0xFFFFu;
        distance = uint(abs(int(blurred) - int(previous[at])));
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
