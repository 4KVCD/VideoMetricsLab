// The compare command: the test videos against the reference, as the
// Python app's command line compares them (vmaf_app/cli.py).
//
// This first C++ version compares on the CPU path only: VMAF v0.6.1, NEG
// or v1 calculated by FFmpeg's libvmaf, with PSNR, SSIM and XPSNR beside
// them. What it cannot do yet (VMAF on the GPU, SSIMULACRA2, Butteraugli,
// CVVDP, PSNR/SSIM/XPSNR alone, --cvvdp-display) it says, and exits with
// code 2, rather than giving other answers than the Python app would. It
// neither reads nor saves the app's saved scores yet: it calculates every
// score, as the Python app does with --recalculate, and keeps none.
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
