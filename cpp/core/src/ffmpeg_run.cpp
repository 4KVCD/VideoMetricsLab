#include "vml/ffmpeg_run.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <tuple>

#include "vml/crop_detect.h"
#include "vml/ffmpeg_tools.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/time_format.h"

namespace vml {

namespace {

// Separates the filter graphs of a run that has more than one; each becomes
// a -filter_complex of its own, ending in [graph0], [graph1].
constexpr char graph_separator = '\n';

std::string joined(const std::vector<std::string>& parts, const std::string& separator) {
    std::string text;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        text += (i == 0 ? "" : separator) + parts[i];
    }
    return text;
}

bool uses_vmaf_model(const VmafOptions& options) {
    return options.compute_vmaf || options.compute_vmaf_neg || options.compute_vmaf_v1;
}

// The VMAF v1 model file a run uses, once the run has resolved it.
std::optional<std::filesystem::path> v1_model_file(const VmafOptions& options) {
    if (options.compute_vmaf_v1 && options.model_v1.starts_with("path=")) {
        return platform::path_from_text(options.model_v1.substr(5));
    }
    return std::nullopt;
}

// libvmaf's options: its log, the models, threads, subsampling, extra
// features (PSNR, SSIM) and frame sync. FFmpeg's filter-graph parser cannot
// take a Windows path with its drive letter's colon, so FFmpeg runs in the
// log's folder and every file is named by its bare name.
std::vector<std::string> libvmaf_options(const VmafOptions& options, const std::string& log_name,
                                         const std::string& model, unsigned cores) {
    std::string model_value = model;
    const std::optional<std::filesystem::path> v1_file = v1_model_file(options);
    if (options.compute_vmaf_neg || v1_file) {
        // Each model gets a name, so each has its own scores in the log
        // ("\\:" is a colon escaped for the filter graph).
        std::vector<std::string> models;
        if (options.compute_vmaf) {
            models.push_back(model_value + "\\\\:name=vmaf");
        }
        if (v1_file) {
            models.push_back("path=" + platform::path_text(v1_file->filename()) + "\\\\:name=vmaf_v1");
        }
        if (options.compute_vmaf_neg) {
            models.emplace_back("version=vmaf_v0.6.1neg\\\\:name=vmaf_neg");
        }
        model_value = joined(models, "|");
    }
    std::vector<std::string> made{"log_path=" + log_name, "log_fmt=json",
                                  uses_vmaf_model(options) ? "model=" + model_value : "model=''"};
    // libvmaf runs on one thread unless told otherwise, so "Auto" is every
    // core (or this job's share of them, set before the run).
    made.push_back("n_threads=" + std::to_string(options.n_threads > 0 ? options.n_threads : static_cast<int>(cores)));
    if (options.n_subsample > 1) {
        made.push_back("n_subsample=" + std::to_string(options.n_subsample));
    }
    if (!options.extra_features.empty()) {
        made.push_back("feature=" + joined(options.extra_features, "|"));
    }
    for (const std::string& option : framesync_opts()) {
        made.push_back(option);
    }
    return made;
}

// FFmpeg's xpsnr filter on the two videos, the source first: it weights
// each block's error by its first input's activity, which must be the
// original's.
std::string xpsnr_filter(const std::string& source_label, const std::string& test_label,
                         const std::string& xpsnr_log_name) {
    return "[" + source_label + "][" + test_label + "]xpsnr=stats_file=" + xpsnr_log_name + ":" +
           joined(framesync_opts(), ":");
}

// The end of the filter graph: XPSNR and libvmaf.
std::string libvmaf_stage(const VmafOptions& options, const std::string& log_name, const std::string& model,
                          const std::optional<std::string>& xpsnr_log_name, unsigned cores) {
    if (options.requested_metrics().empty()) {
        throw VmafRunError("Select at least one metric to calculate.");
    }
    std::string main_label = "main";
    std::string ref_label = "ref";
    if (!uses_vmaf_model(options) && options.extra_features.empty()) {
        return xpsnr_filter(ref_label, main_label, *xpsnr_log_name);  // XPSNR alone needs no libvmaf
    }
    std::vector<std::string> chains;
    if (options.compute_xpsnr && xpsnr_log_name) {
        // xpsnr takes [main] and libvmaf needs it too, but a label can be
        // taken once: without this split, FFmpeg silently gives libvmaf the
        // wrong stream (one video against itself: VMAF 100 for every frame).
        chains.emplace_back("[" + main_label + "]split=2[main_xpsnr][main_vmaf]");
        chains.push_back(xpsnr_filter(ref_label, "main_xpsnr", *xpsnr_log_name) + "[xref]");
        main_label = "main_vmaf";
        ref_label = "xref";
    }
    chains.push_back("[" + main_label + "][" + ref_label + "]libvmaf=" +
                     joined(libvmaf_options(options, log_name, model, cores), ":"));
    return joined(chains, ";");
}

// The hardware decoders that hand frames over in system memory themselves
// when no output format is given (QSV's still hands over its surfaces).
bool decoder_downloads(const std::optional<std::string>& hwaccel) {
    return hwaccel == "cuda" || hwaccel == "d3d11va";
}

// The -hwaccel options for one input of a run with several graphs: where the
// decoder can, it downloads each picture once and every graph gets that copy.
std::vector<std::string> split_hwaccel_args(const std::optional<std::string>& hwaccel) {
    if (decoder_downloads(hwaccel)) {
        return {"-hwaccel", *hwaccel};
    }
    return hwaccel_args(hwaccel);
}

// The options after the inputs: the graphs and the outputs, each with the
// duration limit (an output's -t limits how many frames reach libvmaf).
std::vector<std::string> output_args(const std::string& filtergraph, double duration_limit) {
    const std::vector<std::string> cut =
        duration_args(duration_limit > 0 ? std::optional<std::string>(format_fixed(duration_limit, 3)) : std::nullopt);
    std::vector<std::string> graphs;
    std::stringstream lines(filtergraph);
    for (std::string graph; std::getline(lines, graph, graph_separator);) {
        graphs.push_back(graph);
    }
    std::vector<std::string> args;
    if (graphs.size() > 1) {
        for (const std::string& graph : graphs) {
            args.insert(args.end(), {"-filter_complex", graph});
        }
        args.insert(args.end(), {"-progress", "pipe:1", "-nostats"});
        for (std::size_t index = 0; index < graphs.size(); ++index) {
            args.push_back("-map");
            args.push_back("[graph" + std::to_string(index) + "]");
            args.insert(args.end(), cut.begin(), cut.end());
            args.insert(args.end(), {"-f", "null", "-"});
        }
        return args;
    }
    args = {"-lavfi", filtergraph, "-progress", "pipe:1", "-nostats"};
    args.insert(args.end(), cut.begin(), cut.end());
    args.insert(args.end(), {"-f", "null", "-"});
    return args;
}

// A folder of its own for one run's logs, deleted with it.
class RunFolder {
public:
    RunFolder() {
        std::random_device random;
        for (int attempt = 0; attempt < 100; ++attempt) {
            std::filesystem::path candidate =
                std::filesystem::temp_directory_path() / ("vmaf_run_" + std::to_string(random()));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                path_ = candidate;
                return;
            }
        }
        throw VmafRunError("A folder for the run's logs could not be made.");
    }
    ~RunFolder() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    RunFolder(const RunFolder&) = delete;
    RunFolder& operator=(const RunFolder&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// When a model= value names a custom file, copies it next to the log and
// names it by its bare file name (see libvmaf_options).
std::string model_for_run_folder(const std::string& model, const std::filesystem::path& folder) {
    if (!model.starts_with("path=")) {
        return model;
    }
    const std::filesystem::path file = platform::path_from_text(model.substr(5));
    std::filesystem::copy_file(file, folder / file.filename(), std::filesystem::copy_options::overwrite_existing);
    return "path=" + platform::path_text(file.filename());
}

// The decode plans to try in turn until one runs: the test video, the
// unknown one, loses GPU decoding first.
std::vector<HardwareDecoding> fallback_ladder(const HardwareDecoding& plan) {
    std::vector<HardwareDecoding> ladder{plan};
    if (plan.source && plan.distorted) {
        ladder.push_back(HardwareDecoding{plan.source, std::nullopt});
        ladder.push_back(HardwareDecoding{std::nullopt, plan.distorted});
    }
    if (plan.uses_gpu()) {
        ladder.push_back(HardwareDecoding{});
    }
    return ladder;
}

struct FfmpegOutcome {
    int exit_code = 0;
    std::string errors;
};

// Runs one FFmpeg command, reporting its progress. Throws Cancelled.
FfmpegOutcome run_ffmpeg(const std::vector<std::string>& command, long long total_frames,
                         const ProgressCallback& on_progress, const CancelToken& cancel,
                         const std::filesystem::path& folder) {
    cancel.throw_if_cancelled();  // not one more FFmpeg after a Cancel
    platform::ChildProcess ffmpeg = platform::ChildProcess::start(command, platform::ProcessOptions{folder});
    // -progress writes each update as key=value lines; the latest fps is
    // reported with every frame= line.
    static const std::regex frame_line(R"(^frame=(\d+))");
    static const std::regex fps_line(R"(^fps=\s*([\d.]+))");
    double last_fps = 0.0;
    while (const std::optional<std::string> line = ffmpeg.read_line()) {
        if (cancel.cancelled()) {
            ffmpeg.terminate();
            break;
        }
        std::smatch match;
        if (std::regex_search(*line, match, fps_line)) {
            last_fps = python_float(match[1].str()).value_or(last_fps);
        } else if (std::regex_search(*line, match, frame_line) && on_progress) {
            on_progress(std::stoll(match[1].str()), total_frames, last_fps);
        }
    }
    const int code = ffmpeg.wait();
    cancel.throw_if_cancelled();
    return FfmpegOutcome{code, ffmpeg.standard_error()};
}

// The last `count` lines of `text`.
std::string last_lines(const std::string& text, std::size_t count) {
    std::vector<std::string> lines;
    std::stringstream stream(text);
    for (std::string line; std::getline(stream, line);) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    const std::size_t first = lines.size() > count ? lines.size() - count : 0;
    return joined(std::vector<std::string>(lines.begin() + static_cast<std::ptrdiff_t>(first), lines.end()), "\n");
}

// XPSNR's luma score per frame from the xpsnr filter's stats file, by frame
// number counted from 0 (the file counts from 1).
std::map<int, double> parse_xpsnr_log(const std::filesystem::path& path) {
    std::map<int, double> scores;
    std::ifstream file(path, std::ios::binary);
    static const std::regex line_pattern(
        R"(^n:\s*(\d+)\s+XPSNR y:\s*([+-]?(?:inf|nan|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)))", std::regex::icase);
    for (std::string line; std::getline(file, line);) {
        std::smatch match;
        if (std::regex_search(line, match, line_pattern)) {
            scores[std::stoi(match[1].str()) - 1] = python_float(match[2].str()).value_or(std::nan(""));
        }
    }
    return scores;
}

// One run's per-frame scores, as the Python app's FrameScores keeps them
// (32-bit floats; NaN where a frame has none), by metric key.
struct ParsedScores {
    std::vector<int> frame;
    std::vector<double> time;
    std::map<std::string, std::vector<float>> metrics;
    std::vector<std::string> order;  // the keys in the Python app's order

    bool has(const std::string& key) const { return metrics.contains(key); }
};

std::vector<double> times_of(const std::vector<int>& frames, double fps) {
    std::vector<double> times;
    for (const int frame : frames) {
        times.push_back(fps > 0 ? frame / fps : 0.0);
    }
    return times;
}

// The columns of the scores read, in the Python app's order.
enum Column { vmaf_column, psnr_column, ssim_column, xpsnr_column, neg_column, v1_column, column_count };
constexpr std::array<const char*, column_count> column_keys{"vmaf", "psnr", "ssim", "xpsnr", "vmaf_neg", "vmaf_v1"};

// Reads libvmaf's JSON log as it streams past, keeping only the per-frame
// scores the app uses. A feature-length log runs to hundreds of megabytes,
// and a parsed copy of the whole of it to gigabytes; this keeps four bytes
// per score. It reads what the Python app's _parse_log reads: the root's
// "frames", each frame's "frameNum" and its "metrics", the last of a key
// given twice.
class LogReader final : public nlohmann::json::json_sax_t {
public:
    explicit LogReader(const std::map<int, double>& xpsnr) : xpsnr_(xpsnr) {}

    std::vector<int> frames;
    std::array<std::vector<float>, column_count> columns;
    std::array<bool, column_count> scored{};  // whether any frame has a score in the column

    bool null() override { return value(std::nullopt); }
    bool boolean(bool) override { return value(std::nullopt); }
    bool number_integer(number_integer_t number) override { return value(static_cast<double>(number)); }
    bool number_unsigned(number_unsigned_t number) override { return value(static_cast<double>(number)); }
    bool number_float(number_float_t number, const string_t&) override { return value(number); }
    bool string(string_t&) override { return value(std::nullopt); }
    bool binary(binary_t&) override { return value(std::nullopt); }

    bool key(string_t& name) override {
        next_ = Next::other;
        switch (open_.empty() ? Place::other : open_.back()) {
            case Place::root:
                next_ = name == "frames" ? Next::frames : Next::other;
                break;
            case Place::frame:
                if (name == "frameNum") {
                    next_ = Next::frame_number;
                } else if (name == "metrics") {
                    next_ = Next::metrics;
                    frame_.metrics_is_object = false;  // until it is seen to be one
                    frame_.present.fill(false);
                    frame_.number_of.fill(std::nullopt);
                }
                break;
            case Place::metrics:
                metric_ = index_of(name);
                if (metric_) {
                    frame_.present[*metric_] = true;
                    frame_.number_of[*metric_].reset();
                }
                break;
            default:
                break;
        }
        return true;
    }

    bool start_object(std::size_t) override {
        Place place = Place::other;
        if (open_.empty()) {
            place = Place::root;
        } else if (open_.back() == Place::frames) {
            place = Place::frame;
            frame_ = Frame{};
        } else if (open_.back() == Place::frame && next_ == Next::metrics) {
            place = Place::metrics;
            frame_.metrics_is_object = true;
        }
        open_.push_back(place);
        next_ = Next::other;
        return true;
    }

    bool start_array(std::size_t) override {
        const bool frames_array = !open_.empty() && open_.back() == Place::root && next_ == Next::frames;
        if (frames_array) {
            // The last "frames" is the one read, as by a parser that keeps the
            // last of a key given twice.
            frames.clear();
            for (std::vector<float>& column : columns) {
                column.clear();
            }
            scored.fill(false);
        }
        open_.push_back(frames_array ? Place::frames : Place::other);
        next_ = Next::other;
        return true;
    }

    bool end_object() override { return close(); }
    bool end_array() override { return close(); }

    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& error) override {
        throw std::runtime_error(error.what());
    }

private:
    // What an open object or array is.
    enum class Place { root, frames, frame, metrics, other };
    // What the value after a key is.
    enum class Next { other, frames, frame_number, metrics };

    // The metric keys whose presence and numbers a frame is read for.
    static constexpr std::array<const char*, 7> metric_keys{"vmaf",   "vmaf_neg",   "vmaf_v1", "psnr_y",
                                                            "psnr",   "float_ssim", "ssim"};
    enum Metric { vmaf, vmaf_neg, vmaf_v1, psnr_y, psnr, float_ssim, ssim };

    static std::optional<std::size_t> index_of(const std::string& name) {
        for (std::size_t i = 0; i < metric_keys.size(); ++i) {
            if (name == metric_keys[i]) {
                return i;
            }
        }
        return std::nullopt;
    }

    struct Frame {
        std::optional<int> number;
        bool metrics_is_object = false;
        std::array<bool, metric_keys.size()> present{};
        std::array<std::optional<double>, metric_keys.size()> number_of{};
    };

    bool value(std::optional<double> number) {
        constexpr double int_min = std::numeric_limits<int>::min();
        constexpr double int_max = std::numeric_limits<int>::max();
        if (!open_.empty() && open_.back() == Place::frame && next_ == Next::frame_number && number &&
            *number >= int_min && *number <= int_max) {
            frame_.number = static_cast<int>(*number);
        } else if (!open_.empty() && open_.back() == Place::metrics && metric_ && number) {
            frame_.number_of[*metric_] = number;
        }
        next_ = Next::other;
        return true;
    }

    bool close() {
        const Place closed = open_.back();
        open_.pop_back();
        if (closed == Place::frame) {
            finish_frame();
        }
        return true;
    }

    // Keeps a frame's scores, if it has any of the metrics the app reads.
    void finish_frame() {
        const bool has_any =
            frame_.metrics_is_object && std::any_of(frame_.present.begin(), frame_.present.end(), [](bool p) { return p; });
        if (!has_any) {
            return;
        }
        const int frame_number = frame_.number.value_or(static_cast<int>(frames.size()));
        // A real 0.0 is a score: "psnr" is read only where "psnr_y" has no
        // number, not where it is zero.
        const auto first = [this](Metric a, Metric b) {
            return frame_.number_of[a] ? frame_.number_of[a] : frame_.number_of[b];
        };
        const auto at = xpsnr_.find(frame_number);
        const std::array<std::optional<double>, column_count> values{
            frame_.number_of[vmaf], first(psnr_y, psnr), first(float_ssim, ssim),
            at == xpsnr_.end() ? std::nullopt : std::optional<double>(at->second), frame_.number_of[vmaf_neg],
            frame_.number_of[vmaf_v1]};
        frames.push_back(frame_number);
        for (std::size_t column = 0; column < column_count; ++column) {
            columns[column].push_back(values[column] ? static_cast<float>(*values[column]) : std::nanf(""));
            scored[column] = scored[column] || values[column].has_value();
        }
    }

    const std::map<int, double>& xpsnr_;
    std::vector<Place> open_;  // the objects and arrays open, outermost first
    Next next_ = Next::other;
    std::optional<std::size_t> metric_;  // the metric key just read, if it is one
    Frame frame_;
};

// The scores in libvmaf's JSON log, with XPSNR's from its stats file.
ParsedScores parse_log(const std::filesystem::path& log_path, double fps,
                       const std::optional<std::filesystem::path>& xpsnr_log_path) {
    const std::map<int, double> xpsnr = xpsnr_log_path ? parse_xpsnr_log(*xpsnr_log_path) : std::map<int, double>{};
    LogReader reader(xpsnr);
    std::ifstream log(log_path, std::ios::binary);
    nlohmann::json::sax_parse(log, &reader);
    ParsedScores parsed;
    if (reader.frames.empty()) {
        return parsed;
    }
    parsed.frame = std::move(reader.frames);
    parsed.time = times_of(parsed.frame, fps);
    for (std::size_t column = 0; column < column_count; ++column) {
        if (!reader.scored[column]) {
            continue;  // the run did not ask for this metric
        }
        parsed.metrics[column_keys[column]] = std::move(reader.columns[column]);
        parsed.order.emplace_back(column_keys[column]);
    }
    return parsed;
}

// Runs the FFmpeg command to its end and parses its logs, falling back to
// other decode plans when hardware decoding fails.
ParsedScores execute_run(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                         const VmafOptions& options, const std::optional<CropBox>& source_crop,
                         const std::optional<CropBox>& distorted_crop, const std::string& model, double fps,
                         long long total_frames, const HardwareDecoding& decoding, const ProgressCallback& on_progress,
                         const CancelToken& cancel) {
    if (options.requested_metrics().empty()) {
        throw VmafRunError("Select at least one metric to calculate.");
    }
    const RunFolder folder;
    const std::filesystem::path log_path = folder.path() / "vmaf_log.json";
    const std::optional<std::filesystem::path> xpsnr_log_path =
        options.compute_xpsnr ? std::optional(folder.path() / "xpsnr_log.txt") : std::nullopt;
    const std::string run_model = model_for_run_folder(options.compute_vmaf ? model : "", folder.path());
    if (const std::optional<std::filesystem::path> v1_file = v1_model_file(options)) {
        std::filesystem::copy_file(*v1_file, folder.path() / v1_file->filename(),
                                   std::filesystem::copy_options::overwrite_existing);
    }
    const std::optional<std::string> xpsnr_log_name =
        xpsnr_log_path ? std::optional(platform::path_text(xpsnr_log_path->filename())) : std::nullopt;

    FfmpegOutcome outcome;
    for (const HardwareDecoding& plan : fallback_ladder(decoding)) {
        const std::string graph = build_filtergraph(source, distorted, options, source_crop, distorted_crop, plan,
                                                    "vmaf_log.json", run_model, xpsnr_log_name, environment.cores);
        const std::vector<std::string> command = build_ffmpeg_command(environment.ffmpeg, distorted.path, source.path,
                                                                      graph, plan, options.duration_limit);
        try {
            outcome = run_ffmpeg(command, total_frames, on_progress, cancel, folder.path());
        } catch (const platform::ProcessError& error) {
            throw VmafRunError(std::string("FFmpeg could not be started: ") + error.what());
        }
        if (outcome.exit_code == 0) {
            break;
        }
        // A failed attempt can leave a partial log, which would be read as
        // the retry's.
        std::error_code ignored;
        std::filesystem::remove(log_path, ignored);
        if (xpsnr_log_path) {
            std::filesystem::remove(*xpsnr_log_path, ignored);
        }
    }
    if (outcome.exit_code != 0) {
        throw VmafRunError("ffmpeg exited with code " + std::to_string(outcome.exit_code),
                           last_lines(outcome.errors, 25));
    }

    ParsedScores scores;
    if (!uses_vmaf_model(options) && options.extra_features.empty()) {
        // XPSNR alone, from its stats file.
        for (const auto& [frame, value] : parse_xpsnr_log(*xpsnr_log_path)) {
            scores.frame.push_back(frame);
            scores.metrics["xpsnr"].push_back(static_cast<float>(value));
        }
        scores.time = times_of(scores.frame, fps);
        if (!scores.frame.empty()) {
            scores.order.emplace_back("xpsnr");
        } else {
            scores.metrics.clear();
        }
    } else {
        std::error_code error;
        if (!std::filesystem::exists(log_path, error)) {
            std::string tail = outcome.errors.size() > 2000 ? outcome.errors.substr(outcome.errors.size() - 2000)
                                                            : outcome.errors;
            throw VmafRunError("ffmpeg finished but no metric log was produced.", tail);
        }
        scores = parse_log(log_path, fps, xpsnr_log_path);
    }
    std::vector<std::string> missing;
    for (const std::string& key : options.requested_metrics()) {
        if (!scores.has(key)) {
            missing.push_back(key);
        }
    }
    if (scores.frame.empty() || !missing.empty()) {
        throw VmafRunError("No results for requested metrics: " +
                           joined(missing.empty() ? options.requested_metrics() : missing, ", "));
    }
    return scores;
}

}  // namespace

FrameScores parse_libvmaf_log(const std::filesystem::path& log, double fps,
                              const std::optional<std::filesystem::path>& xpsnr_log) {
    ParsedScores parsed = parse_log(log, fps, xpsnr_log);
    return FrameScores{std::move(parsed.frame), std::move(parsed.time), std::move(parsed.metrics)};
}

void validate_video_pair(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options) {
    if (const std::optional<std::string> problem = pair_problem(source, distorted, options.duration_limit)) {
        throw VmafRunError(*problem);
    }
}

void validate_display_geometry(const VideoInfo& source, const VideoInfo& distorted,
                               const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop) {
    const double source_shape = display_aspect_ratio(source, source_crop);
    const double distorted_shape = display_aspect_ratio(distorted, distorted_crop);
    if (source_shape <= 0 || distorted_shape <= 0) {
        return;  // nothing meaningful to compare
    }
    // Wide enough for the rounding even sizes force (1918x1080 against
    // 1920x1080), far tighter than a real mismatch.
    if (std::abs(source_shape - distorted_shape) <= 0.01 * std::max(source_shape, distorted_shape)) {
        return;
    }
    const auto describe = [](const VideoInfo& info, const std::optional<CropBox>& crop, double shape) {
        const std::string size = crop ? std::to_string(crop->w) + "x" + std::to_string(crop->h) + " (cropped)"
                                      : std::to_string(info.width) + "x" + std::to_string(info.height);
        return size + ", " + format_fixed(shape, 3) + ":1";
    };
    throw VmafRunError("The two videos are different shapes after cropping: source " +
                       describe(source, source_crop, source_shape) + " versus distorted " +
                       describe(distorted, distorted_crop, distorted_shape) +
                       ". Scoring them would stretch one to fit the other and the result would be meaningless. If "
                       "one is letterboxed, set black-bar handling to 'Auto-detect' so the bars are removed before "
                       "comparison.");
}

long long estimate_total_frames(const VideoInfo& reference, const VmafOptions& options, const VideoInfo* other) {
    long long frames = reference.estimated_frame_count();
    if (other != nullptr) {
        frames = std::min(frames, other->estimated_frame_count());
    }
    if (options.duration_limit > 0) {
        // The frames the -t limit keeps, counted as the run counts them.
        const std::optional<long long> kept =
            frames_within(format_fixed(options.duration_limit, 3), frame_rate(reference));
        frames = std::min(frames, kept.value_or(0));
    }
    return frames;
}

std::optional<std::string> short_comparison(long long expected, long long compared, double fps, int step) {
    // Short by at most half a second (at least three frames), plus one step
    // of subsampling, still covers the video.
    const long long slack = std::max(3LL, static_cast<long long>(std::ceil(0.5 * fps))) + std::max(1, step);
    if (expected <= 0 || compared >= expected - slack) {
        return std::nullopt;
    }
    return "Only " + std::to_string(compared) + " of the " + std::to_string(expected) +
           " frames expected could be compared: one of the videos ends after " +
           format_hms(static_cast<double>(compared) / std::max(fps, 1.0), 1) +
           ", though its file says it is longer. It may have been cut short: an encode that stopped early, or a "
           "copy that did not finish.";
}

std::string build_filtergraph(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                              const HardwareDecoding& decoding, const std::string& log_name, const std::string& model,
                              const std::optional<std::string>& xpsnr_log_name, unsigned cores) {
    const auto [distorted_w, distorted_h] = content_size(distorted, distorted_crop);
    const auto [source_w, source_h] = content_size(source, source_crop);
    const bool sizes_differ = source_w != distorted_w || source_h != distorted_h;
    const bool upscale_distorted = sizes_differ && options.scale_direction == ScaleDirection::distorted_to_source;
    // Both videos reach libvmaf in one format, the deeper video's.
    const std::string analysis_format = analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt});
    // XPSNR beside libvmaf's metrics: each in a graph of its own, which
    // FFmpeg runs on a thread of its own (44 fps at 4K, against 31 chained).
    const bool split = options.compute_xpsnr && xpsnr_log_name &&
                       (uses_vmaf_model(options) || !options.extra_features.empty());
    const std::vector<std::string> numbering = pairing_filters(frame_rate(distorted));
    const std::string stream = std::string(video_stream);

    // The test video's chain (input 0).
    std::vector<std::string> main_ops;
    if (decoding.distorted && !(split && decoder_downloads(decoding.distorted))) {
        // Frames come as GPU surfaces and must come back to system memory
        // before any other filter can use them.
        main_ops.emplace_back("hwdownload");
        main_ops.push_back("format=" + hw_native_format(distorted.pix_fmt));
    }
    if (distorted_crop && !distorted_crop->is_noop(distorted.width, distorted.height)) {
        main_ops.push_back(distorted_crop->as_filter());
    }
    main_ops.push_back("format=" + analysis_format);
    if (upscale_distorted) {
        main_ops.push_back("scale=" + std::to_string(source_w) + ":" + std::to_string(source_h) +
                           ":flags=" + options.scale_algorithm);
    }
    main_ops.insert(main_ops.end(), numbering.begin(), numbering.end());
    const std::string main_chain = "[0:" + stream + "]" + joined(main_ops, ",") + "[main]";

    // The source's chain (input 1): cropped before the format conversion, as
    // the test video's, so chroma at a crop's edge is not filtered with the
    // bars' samples.
    std::vector<std::string> ref_ops;
    if (decoding.source && !(split && decoder_downloads(decoding.source))) {
        ref_ops.emplace_back("hwdownload");
        ref_ops.push_back("format=" + hw_native_format(source.pix_fmt));
    }
    if (source_crop && !source_crop->is_noop(source.width, source.height)) {
        ref_ops.push_back(source_crop->as_filter());
    }
    ref_ops.push_back("format=" + analysis_format);
    if (sizes_differ && !upscale_distorted) {
        ref_ops.push_back("scale=" + std::to_string(distorted_w) + ":" + std::to_string(distorted_h) +
                          ":flags=" + options.scale_algorithm);
    }
    ref_ops.insert(ref_ops.end(), numbering.begin(), numbering.end());
    const std::string ref_chain = "[1:" + stream + "]" + joined(ref_ops, ",") + "[ref]";

    if (split) {
        // The graphs decode nothing themselves: FFmpeg gives every graph
        // that reads an input the same decoded frames.
        const auto chains = [&](const std::string& suffix) {
            return "[0:" + stream + "]" + joined(main_ops, ",") + "[main_" + suffix + "];[1:" + stream + "]" +
                   joined(ref_ops, ",") + "[ref_" + suffix + "]";
        };
        return chains("v") + ";[main_v][ref_v]libvmaf=" + joined(libvmaf_options(options, log_name, model, cores), ":") +
               "[graph0]" + graph_separator + chains("x") + ";" + xpsnr_filter("ref_x", "main_x", *xpsnr_log_name) +
               "[graph1]";
    }
    return main_chain + ";" + ref_chain + ";" + libvmaf_stage(options, log_name, model, xpsnr_log_name, cores);
}

std::vector<std::string> build_ffmpeg_command(const std::string& ffmpeg, const std::filesystem::path& distorted,
                                              const std::filesystem::path& source, const std::string& filtergraph,
                                              const HardwareDecoding& decoding, double duration_limit) {
    // The inputs are plain arguments, so absolute Windows paths work here;
    // absolute because FFmpeg runs in the run's own folder.
    const bool several_graphs = filtergraph.find(graph_separator) != std::string::npos;
    const auto decode = [several_graphs](const std::optional<std::string>& hwaccel) {
        return several_graphs ? split_hwaccel_args(hwaccel) : hwaccel_args(hwaccel);
    };
    const auto absolute = [](const std::filesystem::path& path) {
        std::error_code error;
        const std::filesystem::path resolved = std::filesystem::weakly_canonical(std::filesystem::absolute(path), error);
        return platform::path_text(error ? std::filesystem::absolute(path) : resolved);
    };
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-y"};
    for (const auto& [hwaccel, path] :
         {std::pair{decoding.distorted, distorted}, std::pair{decoding.source, source}}) {
        const std::vector<std::string> options = decode(hwaccel);
        command.insert(command.end(), options.begin(), options.end());
        command.insert(command.end(), {"-i", absolute(path)});
    }
    const std::vector<std::string> outputs = output_args(filtergraph, duration_limit);
    command.insert(command.end(), outputs.begin(), outputs.end());
    return command;
}

ComparisonResult run_vmaf(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                          VmafOptions options, const ProgressCallback& on_progress, const CancelToken& cancel,
                          const std::optional<std::filesystem::path>& result_distorted_path) {
    validate_video_pair(source, distorted, options);
    // The decode plan comes before black-bar detection, so the detection
    // decodes the way the run will.
    HardwareDecoding decoding;
    if (options.gpu_decode) {
        decoding = plan_hardware_decoding(environment.decoding, options.gpu_vendor, source, distorted);
    }
    std::optional<CropBox> source_crop;
    std::optional<CropBox> distorted_crop;
    if (options.crop_mode == CropMode::automatic) {
        const auto [source_box, distorted_box] =
            detect_pair(environment.ffmpeg, source, distorted, cancel, decoding.source, decoding.distorted);
        std::tie(source_crop, distorted_crop) = common_picture(source, distorted, source_box, distorted_box);
    }
    // After cropping: removing a letterbox is what makes a letterboxed source
    // and a cropped encode the same shape.
    validate_display_geometry(source, distorted, source_crop, distorted_crop);

    // Auto's model comes from the size frames are compared at, known only
    // now; an explicit or custom choice is the user's.
    const auto [width, height] = analysis_dimensions(source, distorted, options, source_crop, distorted_crop);
    const std::string model = !options.compute_vmaf                       ? std::string()
                              : options.model_choice != auto_model_choice ? options.model
                                                                          : model_for_resolution(width, height);
    if (options.compute_vmaf_v1) {
        options.model_v1 = resolve_v1_model(options, width, height);
    }
    const long long total_frames = estimate_total_frames(distorted, options, &source);
    const ParsedScores scores = execute_run(environment, source, distorted, options, source_crop, distorted_crop,
                                            model, distorted.fps, total_frames, decoding, on_progress, cancel);
    const long long compared = scores.frame.empty() ? 0 : static_cast<long long>(scores.frame.back()) + 1;
    if (const std::optional<std::string> short_run =
            short_comparison(total_frames, compared, distorted.fps, options.n_subsample)) {
        throw VmafRunError(*short_run);
    }

    ComparisonResult result;
    result.source = source.path;
    result.distorted = result_distorted_path.value_or(distorted.path);
    result.fps = distorted.fps;
    result.model = model;
    result.source_crop = source_crop;
    result.distorted_crop = distorted_crop;
    result.source_info = source;
    result.distorted_info = distorted;
    result.scale_direction = options.scale_direction;
    result.scale_algorithm = options.scale_algorithm;
    result.compared_frame_count = total_frames;
    result.model_choice = options.model_choice;
    result.model_v1 = options.model_v1;
    if (options.compute_vmaf_v1) {
        result.model_choice_v1 = options.model_choice_v1;
    }
    for (const std::string& key : scores.order) {
        std::map<std::string, std::string> parameters;
        if (key == "vmaf_neg") {
            parameters["model"] = "version=vmaf_v0.6.1neg";
        } else if (key == "vmaf") {
            parameters["model"] = model;
        } else if (key == "vmaf_v1") {
            parameters["model"] = options.model_v1;
        }
        result.metric_results.add(FrameMetricResult{key, scores.frame, scores.time, scores.metrics.at(key),
                                                    current_ffmpeg_provenance(key, environment.ffmpeg_version,
                                                                              std::move(parameters))});
    }
    return result;
}

}  // namespace vml
