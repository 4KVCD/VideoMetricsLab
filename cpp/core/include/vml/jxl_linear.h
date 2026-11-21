// libjxl 0.12's reading of a 16-bit PNG as linear sRGB, reproduced bit for bit
// (the colour step of VideoMetricsLab's own SSIMULACRA2 and Butteraugli).
//
// The official tools decode the PNG, scale each sample by 1/65535 and convert
// it with JxlCms: libjxl's own sRGB, PQ or HLG curve where JxlCms undoes the
// curve itself, then skcms's curve and matrix to linear sRGB. Every curve here
// acts on one sample at a time, so it becomes one table of 65536 floats; only
// HLG's OOTF and the primaries matrix work per pixel. The tables are built in
// scalar code that rounds exactly as the vector code it copies (std::fma where
// libjxl's Highway code fuses a multiply-add, separate operations where skcms
// does not).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace vml {

class JxlLinearRgb {
public:
    // The reading of a PNG tagged with this H.273 transfer and primaries code
    // as the app tagged them for libjxl's tools (Vship's reading: BT.709's
    // and the other SDR camera curves as plain power curves in gAMA and
    // cHRM, every other transfer in cICP), or nothing for a description this
    // engine does not reproduce. fused_multiply_add says whether libjxl's
    // Highway code fuses MulAdd on this CPU (its AVX2, AVX-512 and NEON
    // targets do, SSE4 does not).
    static std::optional<JxlLinearRgb> make(int transfer, int primaries, bool fused_multiply_add);

    // The picture's intensity target in nits, as libjxl gives it: 10000 for
    // PQ, 1000 for HLG, 255 otherwise.
    float intensity_target() const { return intensity_target_; }

    // One row of rgb48 samples (red, green, blue interleaved) to three planes.
    void convert_row(const std::uint16_t* rgb48, std::size_t width, float* red, float* green, float* blue) const;

private:
    struct HlgOotf {
        std::array<float, 3> luminances;  // GetPrimariesLuminances for the primaries
        float exponent;                   // gamma - 1
    };

    JxlLinearRgb() = default;

    std::shared_ptr<const std::vector<float>> curve_;  // 65536 entries, one per sample value
    std::optional<HlgOotf> hlg_;
    bool linear_profile_after_hlg_ = false;  // skcms's linear source curve, after the OOTF
    std::optional<std::array<float, 9>> matrix_;  // source to linear sRGB, row-major
    float intensity_target_ = 255.f;
};

}  // namespace vml
