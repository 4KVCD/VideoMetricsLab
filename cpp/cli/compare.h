// The compare command: the test videos against the reference, as the
// Python app's command line compares them (vmaf_app/cli.py).
//
// It calculates FFmpeg's metrics: VMAF v0.6.1, NEG and
// v1 with FFmpeg's libvmaf, VMAF, NEG and v1 on the GPU with Vulkan or
// libvmaf's CUDA code, and PSNR, SSIM and XPSNR in the app or with FFmpeg's
// filters, a video's VMAF on the GPU beside its other metrics on the CPU;
// SSIMULACRA2, Butteraugli and CVVDP (on each --cvvdp-display) on the GPU
// with Vship; and SSIMULACRA2 and Butteraugli on the CPU with libjxl's
// tools. What it cannot do yet (FFmpeg pairing the frames, with
// VML_VMAF_PIPES=canvas) it says, and exits with code 2, rather than giving
// other answers than the Python app would.
// Scores are read from and saved to the app's saved scores (metric_cache.h),
// which both apps share: a score saved before is not calculated again
// (unless --recalculate), and every result is saved as it comes in.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "arguments.h"
#include "vml/cvvdp.h"
#include "vml/options.h"
#include "vml/settings.h"
#include "vml/video.h"

namespace vml::cli {

// Runs the compare command; returns the exit code.
int compare(const CompareArguments& arguments);

// The options the window's row would hold for these choices, for the test
// video `test`: the command's, else the app's (Settings).
VmafOptions row_options(const CompareArguments& arguments, const Settings& settings,
                        const std::vector<std::string>& metric_keys, const VideoInfo& test);

// Whether videos are scored with their film grain: --film-grain's choice,
// else the app's default (Settings), or nothing where the app asks (they
// are then scored with it, as the window's dialog has it chosen).
std::optional<bool> film_grain_choice(const CompareArguments& arguments, const Settings& settings);

// Why --gpu-decode (`gpu_decode`) cannot be used: it names a maker none of
// whose GPUs is among those `found` where the system lists its GPUs
// (`listed`: Windows; a Mac or Linux PC decodes with what it has). Nothing
// when it can.
std::optional<std::string> missing_gpu_maker(const std::optional<std::string>& gpu_decode,
                                             const std::vector<GpuVendor>& found, bool listed);

// A display CVVDP models, by its preset's name.
using CvvdpDisplayChoice = std::pair<std::string, CvvdpSettings>;

// The CVVDP displays to score on: those --cvvdp-display names, each once in
// the order given, else the app's default; none for Automatic, the default
// unless Settings name a preset, which gives each video its reference's
// (display_for). Each with its frames scaled to fill it by --cvvdp-fill.
// Throws std::invalid_argument for a name that is no preset.
std::vector<CvvdpDisplayChoice> cvvdp_displays(const CompareArguments& arguments, const Settings& settings);

// The display a video's CVVDP models first: the first of `displays`, or with
// none (Automatic) its reference's, SDR or HDR (automatic_cvvdp_preset),
// filled with `fill`.
CvvdpDisplayChoice display_for(const std::vector<CvvdpDisplayChoice>& displays, const VideoInfo& reference,
                               bool fill);

// A path as Python's Path(text) gives it: the system's separators, repeated
// separators and "." parts dropped, no separator at the end.
std::filesystem::path python_path(const std::string& text);

// The test videos named: each file as given, and what each pattern finds,
// in name order; the reference and files already listed are left out of
// what patterns find. A pattern that finds nothing is kept, to be reported
// as a video that could not be read.
std::vector<std::filesystem::path> expand_tests(const std::vector<std::string>& patterns,
                                                const std::filesystem::path& reference);

}  // namespace vml::cli
