#include "vml/cpu_pair_scorer.h"

#include <array>
#include <cstdio>
#include <future>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>

#include "vml/butteraugli.h"
#include "vml/jxl_linear.h"
#include "vml/perceptual.h"
#include "vml/perceptual_cpu.h"
#include "vml/ssimulacra2.h"

namespace vml {
namespace {

// H.273's transfer codes that libjxl's tools treat as absolute light.
constexpr int pq_transfer = 16;
constexpr int hlg_transfer = 18;

// The engines' reading of a picture tagged with `colour`. Every colour that
// video_colour gives has one (jxl_linear.cpp covers all its transfers and
// primaries).
JxlLinearRgb reading_of(const VideoColour& colour) {
    std::optional<JxlLinearRgb> reading =
        JxlLinearRgb::make(colour.transfer, colour.primaries, Ssimulacra2::fused_multiply_add());
    if (!reading) {
        throw std::logic_error("no reading for transfer " + std::to_string(colour.transfer) + " with primaries " +
                               std::to_string(colour.primaries));
    }
    return *reading;
}

// What the run says when the engines cannot get the memory a picture size
// needs, when they are built or as a thread's rows grow while scoring.
PerceptualRunError out_of_memory(std::size_t width, std::size_t height) {
    return PerceptualRunError("There is not enough memory to score " + std::to_string(width) + "x" +
                              std::to_string(height) + " pictures with SSIMULACRA2 or Butteraugli.");
}

// What the run says when an engine's thread cannot start.
PerceptualRunError no_thread(const std::system_error& error) {
    return PerceptualRunError(std::string("SSIMULACRA2 or Butteraugli could not start its threads: ") + error.what());
}

}  // namespace

float butteraugli_intensity_of(const VideoColour& source, const VideoColour& test) {
    if (!source.hdr() && !test.hdr()) {
        // The app ran the tool with --intensity_target for SDR pairs.
        return static_cast<float>(butteraugli_intensity_nits);
    }
    // Without it, the tool takes the first picture's (the source's) for PQ
    // and HLG, and 80 otherwise (butteraugli_main.cc).
    if (source.transfer == pq_transfer || source.transfer == hlg_transfer) {
        return reading_of(source).intensity_target();
    }
    return 80.0f;
}

struct CpuPairScorer::State {
    std::vector<std::string> keys;
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<std::uint16_t> source;
    std::vector<std::uint16_t> test;
    std::optional<Ssimulacra2> ssimulacra2;
    std::optional<Butteraugli> butteraugli;

    // SSIMULACRA2's score as the tool prints it ("%.8f") and the tool route
    // reads it back: its scores are those decimals.
    double ssimulacra2_score() {
        const std::size_t stride = width * 3;
        std::array<char, 64> printed{};
        std::snprintf(printed.data(), printed.size(), "%.8f", ssimulacra2->score(source.data(), test.data(), stride));
        return parse_score("ssimulacra2", printed.data());
    }

    // Butteraugli's score as the tool route takes it from the tool's map
    // file, which holds the bottom row first: the order sets the 3-norm's
    // last bits.
    double butteraugli_score() {
        const std::vector<float>& map = butteraugli->distortion_map(source.data(), test.data(), width * 3);
        return butteraugli_norm3_top_row_first(std::span<const float>(map), width);
    }
};

CpuPairScorer::CpuPairScorer(const std::vector<std::string>& keys, const VideoColour& source_colour,
                             const VideoColour& test_colour, int width, int height, std::size_t butteraugli_threads,
                             std::size_t ssimulacra2_threads)
    : state_(std::make_unique<State>()) {
    State& s = *state_;
    s.keys = keys;
    s.width = static_cast<std::size_t>(width);
    s.height = static_cast<std::size_t>(height);
    const std::size_t rows = s.height;
    const JxlLinearRgb source_reading = reading_of(source_colour);
    const JxlLinearRgb test_reading = reading_of(test_colour);
    try {
        s.source.resize(s.width * rows * 3);
        s.test.resize(s.width * rows * 3);
        for (const std::string& key : keys) {
            if (key == "ssimulacra2") {
                s.ssimulacra2.emplace(source_reading, test_reading, s.width, rows, Ssimulacra2::Terms::weighted,
                                      ssimulacra2_threads);
            } else if (key == "butteraugli") {
                s.butteraugli.emplace(source_reading, test_reading, s.width, rows,
                                      butteraugli_intensity_of(source_colour, test_colour), butteraugli_threads);
            } else {
                throw std::invalid_argument("the in-app engines do not score " + key);
            }
        }
    } catch (const std::bad_alloc&) {
        throw out_of_memory(s.width, s.height);
    } catch (const std::system_error& error) {
        throw no_thread(error);
    }
}

CpuPairScorer::~CpuPairScorer() = default;

std::uint16_t* CpuPairScorer::source_samples() { return state_->source.data(); }

std::uint16_t* CpuPairScorer::test_samples() { return state_->test.data(); }

std::vector<double> CpuPairScorer::score() {
    State& s = *state_;
    double ssimulacra2 = 0.0;
    double butteraugli = 0.0;
    try {
        if (s.ssimulacra2 && s.butteraugli) {
            // SSIMULACRA2 (its pool's threads) beside Butteraugli (its own):
            // both only read the pictures. The future's destructor waits for
            // it, also when Butteraugli throws.
            std::future<double> ssimulacra2_done =
                std::async(std::launch::async, [&s] { return s.ssimulacra2_score(); });
            butteraugli = s.butteraugli_score();
            ssimulacra2 = ssimulacra2_done.get();
        } else if (s.ssimulacra2) {
            ssimulacra2 = s.ssimulacra2_score();
        } else if (s.butteraugli) {
            butteraugli = s.butteraugli_score();
        }
    } catch (const std::bad_alloc&) {
        throw out_of_memory(s.width, s.height);  // a thread's scratch rows could not grow
    } catch (const std::system_error& error) {
        throw no_thread(error);  // SSIMULACRA2's thread beside Butteraugli
    }
    std::vector<double> values;
    for (const std::string& key : s.keys) {
        values.push_back(key == "ssimulacra2" ? ssimulacra2 : butteraugli);
    }
    return values;
}

}  // namespace vml
