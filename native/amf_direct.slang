// amf_frames.cpp's picture hand-over (Params.handover), straight from AMF's
// own picture (amf_handover.h's Slots::direct_plane): one plane of the crop
// -- Y, U or V -- read from AMF's texture, which Vulkan imported (a view of
// its luma plane, and of its chroma plane with U and V interleaved), into a
// buffer as nvdec_frames' pool holds a picture: planar, rows `pitch` samples
// apart. 16-bit samples (BITS 16, P010's) are shifted right by `shift` (6: to
// the low bits, as yuv420p10le has them; 0: kept at the top). The samples
// are moved, never computed: the views are UINT, so a sample is its bits.
//
// Where the plane's rows start on 16 bytes (`wide`), a thread takes 16 bytes
// of a row and writes them at once; otherwise a sample. Compiled to SPIR-V by
// scripts/build_amf_shader.py into amf_handover_spv.h.

struct Params {
    int x, y;         // the crop's first sample in the plane
    uint width;       // the crop's samples across in the plane
    uint height;      // and down
    uint offset;      // where its first sample goes in `frame`, in samples
    uint pitch;       // samples from one of its rows to the next in `frame`
    uint shift;       // 0 or 6
    uint component;   // 0: Y (the luma view), 1: U, 2: V (the chroma view's first and second)
    uint wide;        // 1: offset and pitch are multiples of PER (a thread per PER samples)
};
[[vk::push_constant]] Params pc;

#if BITS == 16
typealias Sample = uint16_t;
#define PER 8   // samples in 16 bytes
#else
typealias Sample = uint8_t;
#define PER 16
#endif
[[vk::binding(0, 0)]] Texture2D<uint> luma;      // the picture's luma plane (R16_UINT or R8_UINT)
[[vk::binding(1, 0)]] Texture2D<uint2> chroma;   // its chroma plane (R16G16_UINT or R8G8_UINT)
[[vk::binding(0, 1)]] RWStructuredBuffer<Sample> frame;
[[vk::binding(1, 1)]] RWStructuredBuffer<uint4> frame16;  // the same buffer, 16 bytes at a time

uint sample_at(int x, int y)
{
    const int3 at = int3(pc.x + x, pc.y + y, 0);
    if (pc.component == 0)
        return luma.Load(at) >> pc.shift;
    const uint2 both = chroma.Load(at);
    return (pc.component == 1 ? both.x : both.y) >> pc.shift;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.y >= pc.height)
        return;
    const uint row = pc.offset + id.y * pc.pitch;
    if (pc.wide == 0) {
        if (id.x < pc.width)
            frame[row + id.x] = Sample(sample_at(int(id.x), int(id.y)));
        return;
    }
    const uint first = id.x * PER;
    if (first >= pc.width)
        return;
    if (first + PER <= pc.width) {
        uint words[4] = { 0, 0, 0, 0 };
        [unroll] for (uint k = 0; k < PER; ++k) {
#if BITS == 16
            words[k / 2] |= sample_at(int(first + k), int(id.y)) << ((k % 2) * 16);
#else
            words[k / 4] |= sample_at(int(first + k), int(id.y)) << ((k % 4) * 8);
#endif
        }
        frame16[(row + first) / PER] = uint4(words[0], words[1], words[2], words[3]);
        return;
    }
    for (uint k = first; k < pc.width; ++k)  // the row's last samples
        frame[row + k] = Sample(sample_at(int(k), int(id.y)));
}
