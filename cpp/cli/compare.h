// The compare command: the test videos against the reference, as the
// Python app's command line compares them (vmaf_app/cli.py).
//
// It calculates FFmpeg's metrics: VMAF v0.6.1, NEG and
// v1 with FFmpeg's libvmaf, VMAF, NEG and v1 on the GPU with Vulkan or
// libvmaf's CUDA code, and PSNR, SSIM and XPSNR in the app or with FFmpeg's
// filters, a video's VMAF on the GPU beside its other metrics on the CPU;
// SSIMULACRA2, Butteraugli and CVVDP (on each --cvvdp-display) on the GPU
// with Vship; and SSIMULACRA2 and Butteraugli on the CPU with the app's own
// engines. What it cannot do yet (FFmpeg pairing the frames, with
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
#include "common.h"
#include "vml/cvvdp.h"
#include "vml/options.h"
#include "vml/platform/system.h"
#include "vml/settings.h"
#include "vml/video.h"

namespace vml::cli {

// Runs the compare command with `chosen` settings; returns the exit code.
// `gpu_from_settings`: the GPU scored on is the settings' (no --gpu, and
// none named in the environment).
int compare(const CompareArguments& arguments, const ChosenSettings& chosen, bool gpu_from_settings);

// The options left out of the command that the settings fill in, as they are
// written (--metrics, --vship...): whether or not the run uses them.
std::vector<std::string> options_from_settings(const CompareArguments& arguments, bool gpu_from_settings);

// --json: the error document (error_document) of a run that compared
// nothing, as its results would be written (to standard output for "-"):
// the --batch list or the --settings file not read. Nothing where the path
// has a field that needs the videos, which name_paths refuses.
void write_error_json(const CompareArguments& arguments, const std::string& error, const std::string& code);

// The options the window's row would hold for these choices: the command's,
// else the app's (Settings).
VmafOptions row_options(const CompareArguments& arguments, const Settings& settings,
                        const std::vector<std::string>& metric_keys);

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

#ifdef __APPLE__
// Why --gpu-decode (`gpu_decode`) cannot name a maker on a Mac, which decodes
// with its own media engine; nothing for auto.
std::optional<std::string> mac_gpu_decode_refusal(const std::optional<std::string>& gpu_decode);
#endif

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

// `path` made absolute, from the working folder, as the window gives its
// videos: a video's path is recorded and read again from other folders (the
// saved scores' context.json, which the window's Saved Results reads). Links
// are not followed and ".." is kept, as Python's path.absolute() keeps them.
// `path` as it is where it cannot be made absolute.
std::filesystem::path absolute_path(const std::filesystem::path& path);

// A path resolved (links followed, made absolute) as text that is equal for
// two paths of the same file, as Python compares path.resolve() with another:
// on Windows ignoring case.
std::string resolved(const std::filesystem::path& path);

// The test videos named, made absolute (absolute_path): each file as given,
// and what each pattern finds, in name order; the reference and files
// already listed are left out of what patterns find. A pattern that finds
// nothing is kept, to be reported as a video not compared (found_nothing).
std::vector<std::filesystem::path> expand_tests(const std::vector<std::string>& patterns,
                                                const std::filesystem::path& reference);
// expand_tests for any command's videos: `excluded` (if any) left out of what
// patterns find, as the reference is.
std::vector<std::filesystem::path> expand_videos(const std::vector<std::string>& patterns,
                                                 const std::optional<std::filesystem::path>& excluded);
// Why a path expand_videos gave is not read, if it is a pattern it kept
// because the pattern found no file: ffprobe would only call it an invalid
// argument.
std::optional<std::string> found_nothing(const std::filesystem::path& path);

// Why --gpu cannot be used: it names none of `adapters`, the GPUs DirectX
// lists (gpu_choice.h's chosen_among: a whole name or a part of one), or a
// part of a name that more than one has (which would leave each API to take
// a GPU of its own). Nothing
// when it can, for auto, and where the system lists no GPUs (Vulkan's names
// are then matched as the metrics start).
std::optional<std::string> gpu_refusal(const std::optional<std::string>& gpu,
                                       const std::vector<platform::GpuAdapter>& adapters);

// The same for a GPU chosen without --gpu: by the settings
// (`from_settings`), else by VML_GPU. Not refused (a choice made in the
// window, of an eGPU now unplugged, should not stop a script), but said as
// a problem: the GPU metrics then use no GPU. Nothing where gpu_refusal
// finds nothing.
std::optional<std::string> chosen_gpu_problem(const std::optional<std::string>& gpu, bool from_settings,
                                              const std::vector<platform::GpuAdapter>& adapters);

// Why --difference cannot be used with `test_count` test videos (as
// expand_tests finds them): it needs exactly two. Nothing when it can.
std::optional<std::string> difference_refusal(std::size_t test_count);

}  // namespace vml::cli
