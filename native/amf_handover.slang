// amf_frames.cpp's picture hand-over (Params.handover, amf_handover.h): one
// decoded picture, already copied out of its shared texture by the GPU -- the
// crop's luma into the slot buffer's start, its interleaved U and V into
// `chroma` -- laid out as nvdec_frames' pool holds a picture: Y, then U, then
// V, planar, samples moved, never computed. 16-bit samples (BITS 16, P010's)
// are shifted right by `shift` (6: to the low bits, as yuv420p10le has them;
// 0: kept at the top). Compiled to SPIR-V by scripts/build_amf_shader.py into
// amf_handover_spv.h.

struct Params {
    uint first;        // the first sample to write: 0, or the luma's count when the luma needs no shift
    uint lumaCount;    // the crop's luma samples, at the slot's start
    uint chromaCount;  // each chroma plane's samples
    uint lumaOnly;     // 1: no chroma
    uint shift;        // 0 or 6
    uint stride;       // invocations in the dispatch: each takes every stride-th sample
};
[[vk::push_constant]] Params pc;

#if BITS == 16
typealias Sample = uint16_t;
#else
typealias Sample = uint8_t;
#endif
[[vk::binding(0, 0)]] RWStructuredBuffer<Sample> frame;  // the slot: Y (copied in), then U, then V
[[vk::binding(1, 0)]] StructuredBuffer<Sample> chroma;   // U and V interleaved, as the picture has them

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    const uint end = pc.lumaCount + (pc.lumaOnly != 0 ? 0 : pc.chromaCount);
    for (uint i = pc.first + id.x; i < end; i += pc.stride) {
        if (i < pc.lumaCount) {
            frame[i] = Sample(uint(frame[i]) >> pc.shift);
        } else {
            const uint c = i - pc.lumaCount;
            frame[pc.lumaCount + c] = Sample(uint(chroma[2 * c]) >> pc.shift);
            frame[pc.lumaCount + pc.chromaCount + c] = Sample(uint(chroma[2 * c + 1]) >> pc.shift);
        }
    }
}
