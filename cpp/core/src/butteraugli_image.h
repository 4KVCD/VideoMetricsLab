// The shared parts of VideoMetricsLab's Butteraugli engine
// (butteraugli.cpp drives it, butteraugli_kernels.cpp holds its per-row
// kernels). The engine computes a picture pair's map row by row: each stage
// of libjxl's comparator makes one row of its output at a time from rows of
// its inputs, and keeps its output in a ring of just the rows its consumers
// still need (every Butteraugli filter reaches a few rows only), where
// libjxl keeps whole planes. Rows are 64-byte aligned and padded to a whole
// number of vectors, as libjxl's ImageF rows are, so that its vector loops
// can run past the last pixel of a row without leaving it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vml::butteraugli {

// Floats per row are rounded up to this (64 bytes, one AVX-512 vector), and
// one more vector is added so that unaligned loads just past the last pixel
// stay inside the row.
inline constexpr std::size_t kRowAlign = 16;

inline std::size_t row_floats(std::size_t xsize) { return ((xsize + kRowAlign - 1) / kRowAlign + 1) * kRowAlign; }

struct ButteraugliParams {
    float hf_asymmetry = 1.0f;     // multiplier for new HF artifacts against blurred-away features
    float xmul = 1.0f;             // multiplier for the X channel's difference
    float intensity_target = 80.0f;  // nits of an input value of 1.0
};

// The weights of a 5x5 separable blur, as libjxl's WeightsSeparable5: per
// direction the weights at distances 0, 1 and 2, each repeated 4 times.
struct WeightsSeparable5 {
    float horz[3 * 4];
    float vert[3 * 4];
};

// A blur's kernel (libjxl's ComputeKernel, at most 33 taps) and its centre
// and left half scaled by 1 / the sum of its taps, as libjxl's convolution
// weighs the pixels away from the borders.
struct BlurKernel {
    float taps[33];
    float sk[17];
};

// libjxl's Mirror: out-of-range coordinates reflected, the border pixel
// repeated (-1 -> 0, xsize -> xsize - 1).
inline std::int64_t Mirror(std::int64_t x, const std::int64_t xsize) {
    while (x < 0 || x >= xsize) {
        x = x < 0 ? -x - 1 : 2 * xsize - 1 - x;
    }
    return x;
}

// Malta's differences have a margin of zeros round them: kMaltaLeft pixels
// at the left of each row (so that the pixels stay 64-byte aligned),
// kMaltaMargin at the right, and kMaltaMargin rows of zeros above and below
// the picture. That is what libjxl's PaddedMaltaUnit gives a pixel near a
// border, one pixel at a time; with the margins, MaltaUnit (which reads
// kMaltaMargin pixels round its own) runs in vectors on every pixel.
inline constexpr std::size_t kMaltaLeft = 16;
inline constexpr std::size_t kMaltaMargin = 4;

// The rows of one plane that a stage keeps: row y in slot y % rows. A ring
// read through a window of consecutive rows (a vertical blur, Malta) keeps a
// second copy of each slot `rows` slots on, so that any window of up to
// `rows` rows lies one stride apart from its first row's slot on; a ring of
// a whole plane (rows = the plane's height) needs no copy.
struct Ring {
    float* data = nullptr;
    std::size_t stride = 0;  // floats from one slot to the next
    std::size_t rows = 0;
    bool doubled = false;

    float* row(std::size_t y) const { return data + y % rows * stride; }
    // Called once row y is written: its second copy.
    void copy_second(std::size_t y) const {
        if (doubled) {
            float* const slot = row(y);
            std::memcpy(slot + rows * stride, slot, stride * sizeof(float));
        }
    }
};

// The stages of one resolution, in the order a row goes through them (the
// names are what each makes).
enum StageKind : std::uint8_t {
    kStageRgb,         // linear rgb (at half resolution: 2x2 means of the full one's)
    kStageOpsin,       // libjxl's OpsinDynamicsImage, and the low frequencies' horizontal blur
    kStageLf,          // the low frequencies (SeparateLFAndMF), and the medium's horizontal blur
    kStageMf,          // the medium frequencies (SeparateMFAndHF), and the high's horizontal blur
    kStageHf,          // the high and ultra-high frequencies (SeparateHFAndUHF), and the masking's horizontal blur
    kStageMaskBlur,    // the masking's blurred differences (Mask)
    kStageMaltaDiffs,  // Malta's differences of the two pictures' bands
    kStageCompare,     // the map (DiffmapPsychoImage)
    kStageKinds,
};

// What the stages keep, each made by one stage (`producer`): `planes` rings
// per picture, or per pair of pictures.
enum Group : std::uint8_t {
    kRgb, kXyb, kLfh, kLf, kMfRaw, kMfh, kMf, kHfRaw, kHfh, kHf, kUhf, kMh, kBm, kMalta, kMap, kGroups,
};
struct GroupInfo {
    std::uint8_t planes;
    bool per_picture;
    bool windowed;  // read through windows of consecutive rows (Ring::doubled)
    StageKind producer;
};
inline constexpr GroupInfo kGroupInfo[kGroups] = {
    {3, true, false, kStageRgb},         // kRgb: linear rgb
    {3, true, false, kStageOpsin},       // kXyb: XYB
    {3, true, true, kStageOpsin},        // kLfh: XYB blurred horizontally for the low frequencies
    {3, true, false, kStageLf},          // kLf: the low frequencies, as vals
    {2, true, false, kStageLf},          // kMfRaw: X and Y less the low frequencies
    {3, true, true, kStageLf},           // kMfh: that blurred horizontally
    {3, true, false, kStageMf},          // kMf: the medium frequencies
    {2, true, false, kStageMf},          // kHfRaw: X and Y less the medium frequencies
    {2, true, true, kStageMf},           // kHfh: that blurred horizontally
    {2, true, false, kStageHf},          // kHf: the high frequencies
    {2, true, false, kStageHf},          // kUhf: the ultra-high frequencies
    {1, true, true, kStageHf},           // kMh: the masking's differences blurred horizontally
    {1, true, false, kStageMaskBlur},    // kBm: the masking's blurred differences
    {6, false, true, kStageMaltaDiffs},  // kMalta: the six Malta differences, with their margins
    {1, false, false, kStageCompare},    // kMap: the half resolution's map
};

// One resolution's rings: [group][picture][plane]; groups made per pair use
// picture 0. Malta's rows are counted from the top of its margin: its row p
// is the picture's row p - kMaltaMargin.
struct LevelRings {
    std::size_t xsize = 0;
    std::size_t ysize = 0;
    Ring ring[kGroups][2][6];

    const Ring& at(Group group, std::size_t picture, std::size_t plane) const { return ring[group][picture][plane]; }
};

// Everything the stages read, for one engine.
struct StreamFrame {
    LevelRings level[2];  // full, half resolution
    WeightsSeparable5 opsin_blur;
    BlurKernel lf_blur, mf_blur, hf_blur, mask_blur;  // 33, 15, 7 and 13 taps
    ButteraugliParams params;
    float* map = nullptr;  // the full resolution's map, `xsize` floats per row
};

// The most rows of a stage with a vertical blur pass (kStageLf to
// kStageMaskBlur) that one call makes: the pass runs in column strips over
// all of them, so that its input rows are read from the cache once, not once
// per output row.
inline constexpr std::size_t kStageChunkRows = 8;

// Whether stage `kind` makes its rows in chunks of up to kStageChunkRows.
constexpr bool stage_in_chunks(StageKind kind) { return kind >= kStageLf && kind <= kStageMaskBlur; }

// butteraugli_kernels.cpp, dispatched to the CPU's best instruction set:
// rows [first, end) of stage `kind` (kStageOpsin to kStageCompare) at
// resolution `level` for picture `picture` (0 for the pair stages); at most
// kStageChunkRows rows for a stage in chunks. Reads only rows its inputs
// have made, writes only its own rows.
void run_stage_rows(const StreamFrame& frame, StageKind kind, std::size_t level, std::size_t picture,
                    std::size_t first, std::size_t end);

// butteraugli.cpp: convolve_slow.cc's SlowSeparable5 for one row of a plane
// too narrow for the vector code, rows[0..4] the input rows y - 2 to y + 2.
void slow_separable5_row(const float* const rows[5], std::size_t xsize, const WeightsSeparable5& weights,
                         float* row_out);

}  // namespace vml::butteraugli
