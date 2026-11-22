// SSIMULACRA2 and Butteraugli of frame pairs on the CPU, by VideoMetricsLab's
// own engines (vml/ssimulacra2.h, vml/butteraugli.h), with the scores of
// libjxl 0.12's command-line tools to the last bit, without writing and
// reading picture files or starting a program for every frame.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "vml/colour.h"

namespace vml {

// The implementation version of these scores' provenance: libjxl's tools'
// scores, made by the app's engines.
inline constexpr std::string_view in_app_scoring_version = "libjxl 0.12.0 scores, VideoMetricsLab's engines";

// The display brightness Butteraugli views a pair at, in nits, as libjxl's
// butteraugli_main picks it when the app ran it: butteraugli_intensity_nits
// when neither picture is HDR; else the source's own (10000 for PQ, 1000
// for HLG), or the tool's 80 when the source is neither PQ nor HLG.
float butteraugli_intensity_of(const VideoColour& source, const VideoColour& test);

// One worker's scorer of frame pairs of `width` x `height`, the source's
// pictures read with `source_colour` and the test's with `test_colour`: each
// pair by `keys`' metrics ("ssimulacra2", "butteraugli"). Butteraugli splits
// each pair's work between `butteraugli_threads` threads and SSIMULACRA2
// between `ssimulacra2_threads`; with both metrics they run at the same time.
// Throws PerceptualRunError when there is no memory for its engines.
class CpuPairScorer {
public:
    CpuPairScorer(const std::vector<std::string>& keys, const VideoColour& source_colour,
                  const VideoColour& test_colour, int width, int height, std::size_t butteraugli_threads,
                  std::size_t ssimulacra2_threads = 1);
    ~CpuPairScorer();
    CpuPairScorer(const CpuPairScorer&) = delete;
    CpuPairScorer& operator=(const CpuPairScorer&) = delete;

    // Where the pair's pictures are written before score(): 16-bit RGB
    // samples, rows of width * 3 one after the other, top row first
    // (FFmpeg's rgb48le).
    std::uint16_t* source_samples();
    std::uint16_t* test_samples();

    // The pair's scores, in the keys' order: SSIMULACRA2's as the tool
    // prints it, Butteraugli's as the 3-norm of the tool's distortion map
    // (butteraugli_norm3_top_row_first).
    std::vector<double> score();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace vml
