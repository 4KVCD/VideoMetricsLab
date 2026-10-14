// The compare command: the test videos against the reference, as the
// Python app's command line compares them (vmaf_app/cli.py).
//
// So far the C++ version calculates FFmpeg's metrics: VMAF v0.6.1, NEG and
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
#include <string>
#include <vector>

#include "arguments.h"

namespace vml::cli {

// Runs the compare command; returns the exit code.
int compare(const CompareArguments& arguments);

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
