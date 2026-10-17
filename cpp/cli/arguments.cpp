#include "arguments.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <set>

#include "vml/metrics.h"
#include "vml/python_text.h"

namespace vml::cli {

namespace {

// ------------------------------------------------------------------ the texts
//
// As Python's argparse prints them where its output is not a terminal (80
// columns wide). The reference test holds them to the Python app's.

const char* const top_usage = "usage: VideoMetricsLab-cli [-h] [--version] COMMAND ...\n";

const char* const compare_usage = R"(usage: VideoMetricsLab-cli compare [-h] [-m LIST] [--model MODEL]
                                   [--black-bars {auto,none}]
                                   [--scale-to {test,reference}]
                                   [--scaler {bicubic,bilinear,lanczos,spline}]
                                   [--duration SECONDS | --frames N]
                                   [--colour {pq-bt2020,hlg-bt2020,bt709}]
                                   [--fps RATE] [--cvvdp-display PRESET]
                                   [--subsample N] [--threads N]
                                   [--vmaf-on {gpu,cpu}]
                                   [--ssimulacra2-on {gpu,cpu}]
                                   [--butteraugli-on {gpu,cpu}] [--cpu]
                                   [--no-gpu-decode] [--parallel N]
                                   [--recalculate] [--csv FOLDER]
                                   [--json FILE] [-q] [-v]
                                   REFERENCE TEST [TEST ...]
)";

const char* const devices_usage = "usage: VideoMetricsLab-cli devices [-h]\n";

const char* const examples = R"(examples:
  VideoMetricsLab-cli compare reference.mkv encode.mkv
      the app's default metrics for one encode
  VideoMetricsLab-cli compare reference.mkv encodes\*.mkv -m vmaf,ssimulacra2
      two metrics for every .mkv in a folder, with a summary side by side
  VideoMetricsLab-cli compare reference.mkv a.mkv b.mkv -m all --csv results
      every metric, and each video's per-frame scores in results\a.csv, b.csv
  VideoMetricsLab-cli compare reference.mkv encode.mkv --duration 30 --cpu
      the first 30 seconds only, everything calculated on the CPU
  VideoMetricsLab-cli compare reference.mkv encode.mkv -m vmaf --json - -q
      only JSON on the output, for a script to read
  VideoMetricsLab-cli compare reference.y4m encode.y4m -m cvvdp --colour pq-bt2020 --frames 33
      HDR10 video in files with no colour tags, the first 33 frames
  VideoMetricsLab-cli compare reference.y4m encode.266 --fps 24000/1001
      a raw VVC stream from an encoder, at 23.976 frames a second
  VideoMetricsLab-cli compare reference.mkv encode.mkv -m cvvdp --cvvdp-display "30-inch 4K HDR monitor, dim room"
      CVVDP on a display other than the app's default (repeat it for several)
  VideoMetricsLab-cli devices
      what would calculate each metric on this PC
)";

const char* const top_sections = R"(
positional arguments:
  COMMAND
    compare   calculate metrics for test videos against a reference
    devices   show FFmpeg and which GPU each metric would use

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit

)";

const char* const compare_sections = R"(
Calculates metrics for each TEST video against REFERENCE. Options left out take the app's saved settings; scores already saved by the app or an earlier run are not calculated again.

positional arguments:
  REFERENCE
  TEST                  a test video, or a pattern such as encodes\*.mkv (the
                        reference is left out of what a pattern finds)

options:
  -h, --help            show this help message and exit
  -m, --metrics LIST    comma-separated: vmaf, vmaf_neg, vmaf_v1, psnr, ssim,
                        xpsnr, ssimulacra2, butteraugli, cvvdp; or all
                        (default: the app's)
  --model MODEL         VMAF v0.6.1's model: auto (4K model for a 4K
                        comparison), standard, 4k, or a model .json file
                        (default: auto)
  --black-bars {auto,none}
                        auto: detect and cut black bars; none: compare the
                        whole pictures (default: auto)
  --scale-to {test,reference}
                        which video's size the other is scaled to when they
                        differ (default: test)
  --scaler {bicubic,bilinear,lanczos,spline}
                        scaling algorithm (default: bicubic)
  --duration SECONDS    compare only the first SECONDS of each video
  --frames N            compare only the first N frames of each video (a short
                        encode of a longer reference, say)
  --colour, --color {pq-bt2020,hlg-bt2020,bt709}
                        the colours of a video with no colour tags (y4m, say):
                        pq-bt2020 (HDR10), hlg-bt2020 or bt709 (SDR). A tagged
                        video keeps its own. Without it such a video takes the
                        other video's tags, if that one has them, and is
                        otherwise taken as SDR, which changes CVVDP,
                        SSIMULACRA2 and Butteraugli. Scores with colours given
                        or taken are neither read from nor saved to the saved
                        scores
  --fps RATE            the frame rate of a raw video stream (a .264, .265,
                        .266 or .obu file, with no container), which records
                        none, such as 24000/1001, 23.976 or 60: required for
                        one. These scores are neither read from nor saved to
                        the saved scores
  --cvvdp-display PRESET
                        the display CVVDP models, by preset name (default: the
                        app's). Repeat it for a CVVDP score on each display;
                        each extra one decodes the videos again. Presets:
                        '30-inch 4K monitor, office', '24-inch 1080p monitor,
                        office', '30-inch 4K HDR monitor, dim room', '30-inch
                        4K HDR monitor, dark room', '65-inch 4K HDR TV, 1000
                        nits, living room', '65-inch 4K HDR TV, 2000 nits,
                        living room', '65-inch 4K HDR TV, 4000 nits, living
                        room', '6-inch phone', or one saved in the app
  --subsample N         score every N-th frame (default: 1, every frame; CVVDP
                        needs every frame)
  --threads N           libvmaf threads (default: automatic)
  --vmaf-on {gpu,cpu}   where VMAF v0.6.1, VMAF NEG and VMAF v1 is calculated
                        (default: the app's)
  --ssimulacra2-on {gpu,cpu}
                        where SSIMULACRA2 is calculated (default: the app's)
  --butteraugli-on {gpu,cpu}
                        where Butteraugli is calculated (default: the app's)
  --cpu                 calculate on the CPU every metric that can be (CVVDP
                        is GPU only); the same as --vmaf-on cpu
                        --ssimulacra2-on cpu --butteraugli-on cpu
  --no-gpu-decode       decode the videos on the CPU
  --parallel N          videos calculated at once on the CPU, 1-2 (default:
                        the app's)
  --recalculate         calculate every metric again instead of using saved
                        scores
  --csv FOLDER          write each video's per-frame scores to FOLDER as
                        <video>.csv
  --json FILE           also write the results as JSON to FILE; - prints the
                        JSON instead of the tables
  -q, --quiet           no progress, only the results
  -v, --verbose         the app's log as well

)";

const char* const devices_sections = R"(
Shows FFmpeg's version and what would calculate each metric on this PC.

options:
  -h, --help  show this help message and exit
)";

const std::string version_text = std::string("VideoMetricsLab ") + VML_VERSION;

// ------------------------------------------------------------ argparse's way
//
// A port of what Python's argparse does with the parsers in
// vmaf_app/cli.py: the same reading of option strings (abbreviations,
// --option=value, -mVALUE, -qv), of arguments ("--", negative numbers),
// the same order of reading and of errors, and the same messages.

// A wrong command line: argparse's ArgumentError. The message is what
// follows "error: ".
struct ArgumentError {
    std::string message;
};

// Help or the version, to be printed (the program then exits with 0).
struct Printed {
    std::string text;
};

// How many strings an action takes (argparse's nargs).
enum class Nargs {
    none,         // a flag
    one,          // a value
    one_or_more,  // a positional's list
    command,      // a command and the strings after it, for its own parser
};

struct Action {
    std::vector<std::string> option_strings;  // empty for a positional
    std::string metavar;                      // a positional's name
    Nargs nargs;
    bool required = false;
    // Converts the action's strings, checks them, and keeps the value.
    // Throws ArgumentError.
    std::function<void(const std::vector<std::string>&)> take;

    // Its name in messages: "-m/--metrics", or "REFERENCE".
    std::string name() const {
        if (option_strings.empty()) {
            return metavar;
        }
        std::string joined;
        for (const std::string& option : option_strings) {
            joined += (joined.empty() ? "" : "/") + option;
        }
        return joined;
    }
};

struct Parser {
    std::vector<Action> actions;  // in the order they were added
    // Pairs of options that cannot be given together, by their first
    // option strings.
    std::vector<std::pair<std::string, std::string>> exclusive;

    const Action* option(const std::string& name) const {
        for (const Action& action : actions) {
            if (!action.option_strings.empty() && action.option_strings.front() == name) {
                return &action;
            }
        }
        return nullptr;
    }
};

// One reading of an option string (argparse's option tuple): its action
// (nullptr: no such option), the option string it was read as, and an
// argument given in the same string.
struct OptionTuple {
    const Action* action;
    std::string option_string;
    std::optional<std::string> separator;  // "=", "" or none
    std::optional<std::string> explicit_argument;
};

using OptionStrings = std::vector<std::pair<std::string, const Action*>>;  // in the order they were added

const Action* exact_option(const OptionStrings& options, const std::string& text) {
    for (const auto& [option, action] : options) {
        if (option == text) {
            return action;
        }
    }
    return nullptr;
}

// argparse's _get_option_tuples: the options a string can be an
// abbreviation of, or a short option with its argument attached.
std::vector<OptionTuple> option_tuples(const OptionStrings& options, const std::string& text) {
    std::vector<OptionTuple> found;
    const std::size_t equals = text.find('=');
    const std::string prefix = text.substr(0, equals);
    const std::optional<std::string> separator =
        equals == std::string::npos ? std::nullopt : std::optional<std::string>("=");
    const std::optional<std::string> explicit_argument =
        equals == std::string::npos ? std::nullopt : std::optional(text.substr(equals + 1));
    if (text[1] == '-') {
        // A long option is only split at the "=".
        for (const auto& [option, action] : options) {
            if (option.starts_with(prefix)) {
                found.push_back({action, option, separator, explicit_argument});
            }
        }
        return found;
    }
    // A short option can have its argument attached: -mvmaf.
    const std::string short_prefix = text.substr(0, 2);
    for (const auto& [option, action] : options) {
        if (option == short_prefix) {
            found.push_back({action, option, "", text.substr(2)});
        } else if (option.starts_with(prefix)) {
            found.push_back({action, option, separator, explicit_argument});
        }
    }
    return found;
}

// argparse's _parse_optional: how a string is read, as one or more options,
// or nothing for an argument.
std::optional<std::vector<OptionTuple>> parse_optional(const OptionStrings& options, const std::string& text) {
    if (text.empty() || text[0] != '-') {
        return std::nullopt;
    }
    if (const Action* action = exact_option(options, text)) {
        return std::vector<OptionTuple>{{action, text, std::nullopt, std::nullopt}};
    }
    if (text.size() == 1) {
        return std::nullopt;  // "-" is an argument
    }
    if (const std::size_t equals = text.find('='); equals != std::string::npos) {
        if (const Action* action = exact_option(options, text.substr(0, equals))) {
            return std::vector<OptionTuple>{{action, text.substr(0, equals), "=", text.substr(equals + 1)}};
        }
    }
    if (std::vector<OptionTuple> found = option_tuples(options, text); !found.empty()) {
        return found;
    }
    // A negative number is an argument (no option here looks like one).
    if (text.size() >= 2 && (std::isdigit(static_cast<unsigned char>(text[1])) ||
                             (text[1] == '.' && text.size() >= 3 && std::isdigit(static_cast<unsigned char>(text[2]))))) {
        return std::nullopt;
    }
    if (text.find(' ') != std::string::npos) {
        return std::nullopt;
    }
    return std::vector<OptionTuple>{{nullptr, text, std::nullopt, std::nullopt}};
}

// How many of the pattern's strings, from `at`, each positional takes, as
// argparse's regular expressions match them: -*A-* for one, -*A[A-]* for one
// or more, -*A[-AO]* for a command. Nothing if they cannot all match.
std::optional<std::vector<std::size_t>> match_positionals(const std::vector<const Action*>& positionals,
                                                          std::size_t first, const std::string& pattern,
                                                          std::size_t at) {
    if (first == positionals.size()) {
        return std::vector<std::size_t>{};
    }
    std::size_t end = at;
    while (end < pattern.size() && pattern[end] == '-') {
        ++end;
    }
    if (end >= pattern.size() || pattern[end] != 'A') {
        return std::nullopt;
    }
    ++end;
    const Nargs nargs = positionals[first]->nargs;
    const auto in_tail = [nargs](char c) {
        return c == '-' || (nargs != Nargs::one && c == 'A') || (nargs == Nargs::command && c == 'O');
    };
    std::size_t longest = end;
    while (longest < pattern.size() && in_tail(pattern[longest])) {
        ++longest;
    }
    // Greedy: the longest tail the rest can follow.
    for (std::size_t stop = longest + 1; stop-- > end;) {
        if (auto rest = match_positionals(positionals, first + 1, pattern, stop)) {
            rest->insert(rest->begin(), stop - at);
            return rest;
        }
    }
    return std::nullopt;
}

// Reads command-line strings with a Parser, as argparse's parse_known_args
// reads them: its _parse_known_args, whose consume_optional and
// consume_positionals are the methods of the same names.
class ArgumentReader {
public:
    ArgumentReader(const Parser& parser, const std::vector<std::string>& strings);

    // The strings that are none of the parser's. Throws ArgumentError, and
    // Printed (from a help or version action).
    std::vector<std::string> read();

private:
    void take_action(const Action& action, const std::vector<std::string>& values);
    std::size_t consume_optional(std::size_t start);
    std::size_t consume_positionals(std::size_t start);
    void check_required() const;

    const Parser& parser_;
    const std::vector<std::string>& strings_;
    OptionStrings options_;
    std::vector<const Action*> positionals_;  // those not taken yet
    // Each string as an option (O), an argument (A), or the "--" (-) after
    // which every string is an argument; and how each option is read.
    std::string pattern_;
    std::map<std::size_t, std::vector<OptionTuple>> option_at_;
    std::set<const Action*> seen_;
    std::vector<std::string> extras_;
};

ArgumentReader::ArgumentReader(const Parser& parser, const std::vector<std::string>& strings)
    : parser_(parser), strings_(strings) {
    for (const Action& action : parser.actions) {
        for (const std::string& option : action.option_strings) {
            options_.emplace_back(option, &action);
        }
        if (action.option_strings.empty()) {
            positionals_.push_back(&action);
        }
    }
    for (std::size_t i = 0; i < strings.size(); ++i) {
        if (strings[i] == "--") {
            pattern_ += '-';
            pattern_.append(strings.size() - i - 1, 'A');
            break;
        }
        if (std::optional<std::vector<OptionTuple>> tuples = parse_optional(options_, strings[i])) {
            option_at_[i] = std::move(*tuples);
            pattern_ += 'O';
        } else {
            pattern_ += 'A';
        }
    }
}

std::vector<std::string> ArgumentReader::read() {
    // Positionals and options in turn, to the last option.
    std::size_t start = 0;
    if (!option_at_.empty()) {
        const std::size_t last_option = option_at_.rbegin()->first;
        while (start <= last_option) {
            std::size_t next_option = start;
            while (next_option <= last_option && !option_at_.contains(next_option)) {
                ++next_option;
            }
            if (start != next_option) {
                const std::size_t end = consume_positionals(start);
                const bool moved = end > start;
                start = end;
                if (moved) {
                    continue;
                }
            }
            if (!option_at_.contains(start)) {
                extras_.insert(extras_.end(), strings_.begin() + start, strings_.begin() + next_option);
                start = next_option;
            }
            start = consume_optional(start);
        }
    }
    const std::size_t stop = consume_positionals(start);
    extras_.insert(extras_.end(), strings_.begin() + stop, strings_.end());
    check_required();
    return std::move(extras_);
}

// Takes `values` for `action`; not with an option already given that
// excludes it.
void ArgumentReader::take_action(const Action& action, const std::vector<std::string>& values) {
    seen_.insert(&action);
    action.take(values);
    for (const auto& [a, b] : parser_.exclusive) {
        const Action* other = &action == parser_.option(a)   ? parser_.option(b)
                              : &action == parser_.option(b) ? parser_.option(a)
                                                             : nullptr;
        if (other != nullptr && seen_.contains(other)) {
            throw ArgumentError{"argument " + action.name() + ": not allowed with argument " + other->name()};
        }
    }
}

// Takes the option at `start`, its value, and the flags run together with
// it (-qv); returns where the next string to read is.
std::size_t ArgumentReader::consume_optional(std::size_t start) {
    const std::vector<OptionTuple>& tuples = option_at_.at(start);
    if (tuples.size() > 1) {
        std::string matches;
        for (const OptionTuple& tuple : tuples) {
            matches += (matches.empty() ? "" : ", ") + tuple.option_string;
        }
        throw ArgumentError{"ambiguous option: " + strings_[start] + " could match " + matches};
    }
    OptionTuple tuple = tuples.front();
    std::vector<std::pair<const Action*, std::vector<std::string>>> taken;
    std::size_t stop = start + 1;
    while (true) {
        if (tuple.action == nullptr) {
            extras_.push_back(strings_[start]);
            return start + 1;
        }
        const Action& action = *tuple.action;
        if (!tuple.explicit_argument) {
            // The option's value is the next string, if it is an argument.
            std::vector<std::string> values;
            if (action.nargs == Nargs::one) {
                if (start + 1 >= pattern_.size() || pattern_[start + 1] != 'A') {
                    throw ArgumentError{"argument " + action.name() + ": expected one argument"};
                }
                values.push_back(strings_[start + 1]);
            }
            stop = start + 1 + values.size();
            taken.emplace_back(&action, std::move(values));
            break;
        }
        const std::string argument = *tuple.explicit_argument;
        if (action.nargs == Nargs::one) {
            taken.emplace_back(&action, std::vector<std::string>{argument});
            break;
        }
        // A flag with more after it: -qv is -q -v.
        if (tuple.option_string[1] == '-' || argument.empty() || (tuple.separator && !tuple.separator->empty()) ||
            argument[0] == '-') {
            throw ArgumentError{"argument " + action.name() + ": ignored explicit argument " + python_string_repr(argument)};
        }
        taken.emplace_back(&action, std::vector<std::string>{});
        const std::string next = std::string("-") + argument[0];
        if (const Action* other = exact_option(options_, next)) {
            const std::string rest = argument.substr(1);
            tuple = {other, next, std::nullopt, std::nullopt};
            if (!rest.empty()) {
                tuple.separator = rest[0] == '=' ? "=" : "";
                tuple.explicit_argument = rest[0] == '=' ? rest.substr(1) : rest;
            }
        } else {
            extras_.push_back("-" + argument);
            break;
        }
    }
    for (const auto& [action, values] : taken) {
        take_action(*action, values);
    }
    return stop;
}

// Takes as many of the positionals left as can match the strings from
// `start`, from the first; returns where the next string to read is.
std::size_t ArgumentReader::consume_positionals(std::size_t start) {
    std::vector<std::size_t> counts;
    for (std::size_t matched = positionals_.size(); matched > 0; --matched) {
        const std::vector<const Action*> some(positionals_.begin(), positionals_.begin() + matched);
        if (std::optional<std::vector<std::size_t>> found = match_positionals(some, 0, pattern_, start)) {
            counts = std::move(*found);
            break;
        }
    }
    for (std::size_t i = 0; i < counts.size(); ++i) {
        std::vector<std::string> values(strings_.begin() + start, strings_.begin() + start + counts[i]);
        // The "--" is not a value.
        const std::size_t dash = pattern_.find('-', start);
        if (positionals_[i]->nargs == Nargs::command ? pattern_[start] == '-' : dash < start + counts[i]) {
            values.erase(std::find(values.begin(), values.end(), "--"));
        }
        start += counts[i];
        take_action(*positionals_[i], values);
    }
    positionals_.erase(positionals_.begin(), positionals_.begin() + counts.size());
    return start;
}

// Throws for the required arguments not given, naming them all.
void ArgumentReader::check_required() const {
    std::string missing;
    for (const Action& action : parser_.actions) {
        if (action.required && !seen_.contains(&action)) {
            missing += (missing.empty() ? "" : ", ") + action.name();
        }
    }
    if (!missing.empty()) {
        throw ArgumentError{"the following arguments are required: " + missing};
    }
}

// Reads `strings` with `parser`, as argparse's parse_known_args does, and
// returns the strings that are none of its. Throws ArgumentError, and
// Printed (from a help or version action).
std::vector<std::string> parse_known(const Parser& parser, const std::vector<std::string>& strings) {
    return ArgumentReader(parser, strings).read();
}

// ---------------------------------------------------------------- the values

// The metric keys named in a --metrics value ("all" is every metric), in
// the registry's order.
std::vector<std::string> metric_list(const std::string& text) {
    std::vector<std::string> keys;
    if (ascii_lower(python_strip(text)) == "all") {
        for (const MetricDefinition& metric : metrics()) {
            keys.emplace_back(metric.key);
        }
        return keys;
    }
    std::vector<std::string> named;
    std::string unknown;
    for (std::size_t start = 0; start <= text.size();) {
        std::size_t end = text.find(',', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string part = ascii_lower(python_strip(text.substr(start, end - start)));
        if (!part.empty() && std::find(named.begin(), named.end(), part) == named.end()) {
            named.push_back(part);
            if (find_metric(part) == nullptr) {
                unknown += (unknown.empty() ? "" : ", ") + part;
            }
        }
        start = end + 1;
    }
    if (!unknown.empty() || named.empty()) {
        std::string choices = "all";
        for (const MetricDefinition& metric : metrics()) {
            choices += ", " + std::string(metric.key);
        }
        throw ArgumentError{"unknown metric " + (unknown.empty() ? text : unknown) + "; choose from: " + choices};
    }
    for (const MetricDefinition& metric : metrics()) {
        if (std::find(named.begin(), named.end(), metric.key) != named.end()) {
            keys.emplace_back(metric.key);
        }
    }
    return keys;
}

// A number above 0, read as Python's float() or int() reads it (the app's
// _positive types).
double positive_float(const std::string& text) {
    const std::optional<double> value = python_float(text);
    if (!value) {
        throw ArgumentError{"invalid parse value: " + python_string_repr(text)};
    }
    if (!(*value > 0) || !std::isfinite(*value)) {
        throw ArgumentError{"must be above 0"};
    }
    return *value;
}

long long positive_int(const std::string& text) {
    const std::optional<long long> value = python_int(text);
    if (!value) {
        throw ArgumentError{"invalid parse value: " + python_string_repr(text)};
    }
    if (*value <= 0) {
        throw ArgumentError{"must be above 0"};
    }
    return *value;
}

// A frame rate (--fps): a fraction of whole numbers below 2^53, as many as a
// double holds exactly ("24000/1001"), or a number ("23.976"), above 0. The
// reference's make_reference.py reads it by the same rule.
double frame_rate(const std::string& text) {
    static const std::regex fraction("([0-9]+)/([0-9]+)");
    const std::string stripped(python_strip(text));
    std::optional<double> value;
    std::smatch parts;
    if (std::regex_match(stripped, parts, fraction)) {
        constexpr long long exact = 1LL << 53;
        const std::optional<long long> numerator = python_int(parts[1].str());
        const std::optional<long long> denominator = python_int(parts[2].str());
        if (numerator && denominator && *denominator != 0 && *numerator < exact && *denominator < exact) {
            value = static_cast<double>(*numerator) / static_cast<double>(*denominator);
        }
    } else {
        value = python_float(text);
    }
    if (!value) {
        throw ArgumentError{"invalid parse value: " + python_string_repr(text)};
    }
    if (!(*value > 0) || !std::isfinite(*value)) {
        throw ArgumentError{"must be above 0"};
    }
    return *value;
}

std::string choice(const std::string& value, const std::vector<std::string>& choices) {
    if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
        std::string listed;
        for (const std::string& c : choices) {
            listed += (listed.empty() ? "" : ", ") + c;
        }
        throw ArgumentError{"invalid choice: " + python_string_repr(value) + " (choose from " + listed + ")"};
    }
    return value;
}

// An action that converts its one value with `read`, its message prefixed
// with the action's name, as argparse reports a type's error.
Action option(std::vector<std::string> names, std::function<void(const std::string&)> read) {
    Action action{std::move(names), "", Nargs::one, false, nullptr};
    const std::string name = action.name();
    action.take = [name, read = std::move(read)](const std::vector<std::string>& values) {
        try {
            read(values.front());
        } catch (const ArgumentError& error) {
            throw ArgumentError{"argument " + name + ": " + error.message};
        }
    };
    return action;
}

Action flag(std::vector<std::string> names, bool& target) {
    return {std::move(names), "", Nargs::none, false, [&target](const std::vector<std::string>&) { target = true; }};
}

Action help(const std::string& text) {
    return {{"-h", "--help"}, "", Nargs::none, false, [text](const std::vector<std::string>&) { throw Printed{text}; }};
}

// The compare command's parser, keeping what it reads in `parsed`.
Parser compare_parser(CompareArguments& parsed) {
    const std::string help_text = std::string(compare_usage) + compare_sections + examples;
    const auto where = [](std::optional<std::string>& target) {
        return [&target](const std::string& value) { target = choice(value, {"gpu", "cpu"}); };
    };
    Parser parser;
    parser.actions = {
        help(help_text),
        {{}, "REFERENCE", Nargs::one, true,
         [&parsed](const std::vector<std::string>& values) { parsed.reference = values.front(); }},
        {{}, "TEST", Nargs::one_or_more, true,
         [&parsed](const std::vector<std::string>& values) { parsed.tests = values; }},
        option({"-m", "--metrics"}, [&parsed](const std::string& value) { parsed.metrics = metric_list(value); }),
        option({"--model"}, [&parsed](const std::string& value) { parsed.model = value; }),
        option({"--black-bars"},
               [&parsed](const std::string& value) { parsed.black_bars = choice(value, {"auto", "none"}); }),
        option({"--scale-to"},
               [&parsed](const std::string& value) { parsed.scale_to = choice(value, {"test", "reference"}); }),
        option({"--scaler"},
               [&parsed](const std::string& value) {
                   parsed.scaler = choice(value, {"bicubic", "bilinear", "lanczos", "spline"});
               }),
        option({"--duration"}, [&parsed](const std::string& value) { parsed.duration = positive_float(value); }),
        option({"--frames"}, [&parsed](const std::string& value) { parsed.frames = positive_int(value); }),
        option({"--colour", "--color"},
               [&parsed](const std::string& value) {
                   parsed.colour = choice(value, {"pq-bt2020", "hlg-bt2020", "bt709"});
               }),
        option({"--fps"}, [&parsed](const std::string& value) { parsed.fps = frame_rate(value); }),
        option({"--cvvdp-display"}, [&parsed](const std::string& value) { parsed.cvvdp_displays.push_back(value); }),
        option({"--subsample"},
               [&parsed](const std::string& value) { parsed.subsample = static_cast<int>(positive_int(value)); }),
        option({"--threads"},
               [&parsed](const std::string& value) { parsed.threads = static_cast<int>(positive_int(value)); }),
        option({"--vmaf-on"}, where(parsed.vmaf_on)),
        option({"--ssimulacra2-on"}, where(parsed.ssimulacra2_on)),
        option({"--butteraugli-on"}, where(parsed.butteraugli_on)),
        flag({"--cpu"}, parsed.cpu),
        flag({"--no-gpu-decode"}, parsed.no_gpu_decode),
        option({"--parallel"},
               [&parsed](const std::string& value) {
                   const std::optional<long long> number = python_int(value);
                   if (!number) {
                       throw ArgumentError{"invalid int value: " + python_string_repr(value)};
                   }
                   parsed.parallel = static_cast<int>(std::stoi(choice(std::to_string(*number), {"1", "2"})));
               }),
        flag({"--recalculate"}, parsed.recalculate),
        option({"--csv"}, [&parsed](const std::string& value) { parsed.csv = value; }),
        option({"--json"}, [&parsed](const std::string& value) { parsed.json_path = value; }),
        flag({"-q", "--quiet"}, parsed.quiet),
        flag({"-v", "--verbose"}, parsed.verbose),
    };
    parser.exclusive = {{"--duration", "--frames"}};
    return parser;
}

// Reads a command's strings with its parser; its errors are its own
// (argparse's error() of the command's parser).
std::vector<std::string> parse_command(const Parser& parser, const std::string& usage, const std::string& prog,
                                       const std::vector<std::string>& strings) {
    try {
        return parse_known(parser, strings);
    } catch (const ArgumentError& error) {
        throw UsageError(usage + prog + ": error: " + error.message);
    }
}

}  // namespace

std::string top_help() {
    return std::string(top_usage) + "\n" + version_text + ": video quality metrics from the command line.\n" +
           top_sections + examples + "\n" + program + " compare --help lists every option.\n";
}

const std::string& version_line() { return version_text; }

Arguments parse_arguments(const std::vector<std::string>& arguments) {
    Arguments parsed;
    std::vector<std::string> command_extras;
    Parser top;
    top.actions = {
        help(top_help()),
        {{"--version"}, "", Nargs::none, false,
         [](const std::vector<std::string>&) { throw Printed{version_text + "\n"}; }},
        {{}, "COMMAND", Nargs::command, false,
         [&](const std::vector<std::string>& values) {
             std::string command;
             try {
                 command = choice(values.front(), {"compare", "devices"});
             } catch (const ArgumentError& error) {
                 throw ArgumentError{"argument COMMAND: " + error.message};
             }
             const std::vector<std::string> rest(values.begin() + 1, values.end());
             if (command == "compare") {
                 parsed.command = Command::compare;
                 const Parser parser = compare_parser(parsed.compare);
                 command_extras = parse_command(parser, compare_usage, std::string(program) + " compare", rest);
             } else {
                 parsed.command = Command::devices;
                 const Parser parser{{help(std::string(devices_usage) + devices_sections)}, {}};
                 command_extras = parse_command(parser, devices_usage, std::string(program) + " devices", rest);
             }
         }},
    };
    try {
        std::vector<std::string> extras = parse_known(top, arguments);
        extras.insert(extras.end(), command_extras.begin(), command_extras.end());
        if (!extras.empty()) {
            std::string listed;
            for (const std::string& extra : extras) {
                listed += (listed.empty() ? "" : " ") + extra;
            }
            throw ArgumentError{"unrecognized arguments: " + listed};
        }
    } catch (const ArgumentError& error) {
        throw UsageError(std::string(top_usage) + program + ": error: " + error.message);
    } catch (const Printed& printed) {
        parsed.print_and_exit = printed.text;
    }
    return parsed;
}

}  // namespace vml::cli
