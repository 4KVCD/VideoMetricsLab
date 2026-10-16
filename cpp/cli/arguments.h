// The command line's arguments, read as the Python app's argparse parser
// reads them: the same commands and options, unambiguous abbreviations of
// long options, --option=value, and exit code 2 for a wrong command.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vml::cli {

inline constexpr const char* program = "VideoMetricsLab-cli";
inline constexpr int exit_ok = 0;
inline constexpr int exit_failed = 1;
inline constexpr int exit_usage = 2;
inline constexpr int exit_cancelled = 130;

// The compare command's arguments.
struct CompareArguments {
    std::string reference;
    std::vector<std::string> tests;  // files, or patterns such as encodes\*.mkv
    std::optional<std::vector<std::string>> metrics;  // keys, in the registry's order
    std::string model = "auto";
    std::string black_bars = "auto";
    std::string scale_to = "test";
    std::string scaler = "bicubic";
    std::optional<double> duration;
    std::optional<long long> frames;
    std::optional<std::string> colour;
    std::optional<double> fps;  // a raw stream's frame rate
    std::vector<std::string> cvvdp_displays;
    int subsample = 1;
    std::optional<int> threads;
    std::optional<std::string> vmaf_on;
    std::optional<std::string> ssimulacra2_on;
    std::optional<std::string> butteraugli_on;
    bool cpu = false;
    bool no_gpu_decode = false;
    std::optional<int> parallel;
    bool recalculate = false;
    std::optional<std::string> csv;
    std::optional<std::string> json_path;  // "-" prints the JSON instead of the tables
    bool quiet = false;
    bool verbose = false;
};

enum class Command { none, compare, devices };

struct Arguments {
    Command command = Command::none;
    // Help or the version was asked for: print this and exit 0.
    std::optional<std::string> print_and_exit;
    CompareArguments compare;
};

// A wrong command; the message is argparse's (usage, then the error).
class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The arguments, without the program's name. Throws UsageError.
Arguments parse_arguments(const std::vector<std::string>& arguments);

// The top-level help, printed when no command is given.
std::string top_help();

// The app's name and version, as --version prints it.
const std::string& version_line();

}  // namespace vml::cli
