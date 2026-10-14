// Saved runs: a comparison's result in a file of its own, to look at again
// later or on another PC, and read back (the Python app's run_io.save_run
// and load_run). Either app opens the other's files. They are the Python
// app's byte for byte, but for the order of the keys in a score's
// provenance parameters: this app keeps them sorted (MetricProvenance),
// the Python app in the order they were made. The order means nothing.
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "vml/results.h"

namespace vml {

// The extension of a saved run. A double extension rather than a bare .json
// keeps the app's file dialogs away from unrelated JSON files.
inline constexpr std::string_view result_suffix = ".metrics.json";

// A saved run that cannot be read, saying why as the Python app does.
class RunFileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// What save_run writes: the result under `label` (its test video's name
// when empty). Throws std::invalid_argument for a NaN or an infinity where
// a score's cannot be written (the frame rate, say), as Python refuses
// them.
std::string run_file_text(const ComparisonResult& result, const std::string& label);

// Writes the result to `path`. Throws std::runtime_error when the file
// cannot be written.
void save_run(const ComparisonResult& result, const std::filesystem::path& path, const std::string& label);

// A saved run's result and label, from the file's text. A run saved while
// VMAF v1 was a model of the one VMAF column comes back with those scores
// as VMAF v1's. Throws RunFileError.
std::pair<ComparisonResult, std::string> run_from_text(const std::string& text);

// run_from_text of the file at `path`. Throws RunFileError, also when the
// file cannot be read.
std::pair<ComparisonResult, std::string> load_run(const std::filesystem::path& path);

}  // namespace vml
