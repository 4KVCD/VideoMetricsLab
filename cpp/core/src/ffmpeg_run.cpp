#include "vml/ffmpeg_run.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>

#include "vml/app_log.h"
#include "vml/crop_detect.h"
#include "vml/ffmpeg_tools.h"
#include "vml/film_grain_scoring.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/grain_feed.h"
#include "vml/isolated.h"
#include "vml/libvmaf_cpu.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/run_json.h"
#include "vml/size_changes.h"
#include "vml/temp_files.h"
#include "vml/time_format.h"
#include "vml/vmaf_cuda.h"
#include "vml/vmaf_v1_gpu.h"

namespace vml {

namespace {

// Separates the filter graphs of a run that has more than one; each becomes
// a -filter_complex of its own, ending in [graph0], [graph1].
constexpr char graph_separator = '\n';

bool uses_vmaf_model(const VmafOptions& options) {
    return options.compute_vmaf || options.compute_vmaf_neg || options.compute_vmaf_v1;
}

// The VMAF v1 model file a run uses, once the run has resolved it.
std::optional<std::filesystem::path> v1_model_file(const VmafOptions& options) {
    if (const std::optional<std::string_view> file = model_file_of(options.model_v1); options.compute_vmaf_v1 && file) {
        return platform::path_from_text(*file);
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
            models.push_back(model_option_for_file(platform::path_text(v1_file->filename())) + "\\\\:name=vmaf_v1");
        }
        if (options.compute_vmaf_neg) {
            models.emplace_back("version=vmaf_v0.6.1neg\\\\:name=vmaf_neg");
        }
        model_value = join(models, "|");
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
        made.push_back("feature=" + join(options.extra_features, "|"));
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
           join(framesync_opts(), ":");
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
                     join(libvmaf_options(options, log_name, model, cores), ":"));
    return join(chains, ";");
}

// The hardware decoders that hand frames over in system memory themselves
// when no output format is given (QSV's still hands over its surfaces).
bool decoder_downloads(const std::optional<std::string>& hwaccel) {
    return hwaccel == "cuda" || hwaccel == "d3d11va" || hwaccel == "videotoolbox";
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

// Each video's filters before the two are paired: downloaded from its GPU
// decoder if need be (in a run of several graphs, a decoder that downloads
// its pictures itself does it once for all of them), cropped, converted to
// the analysis format (the deeper video's) and, for a test video of the
// other range, to the source's range (VideoInfo::full_range), scaled to the
// compared size, and its frames numbered.
struct VideoChains {
    std::vector<std::string> test;    // the test video's (input 0)
    std::vector<std::string> source;  // the source's
};

VideoChains video_chains(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                         const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                         const HardwareDecoding& decoding, bool split) {
    const auto [distorted_w, distorted_h] = content_size(distorted, distorted_crop);
    const auto [source_w, source_h] = content_size(source, source_crop);
    const bool sizes_differ = source_w != distorted_w || source_h != distorted_h;
    const bool upscale_distorted = sizes_differ && options.scale_direction == ScaleDirection::distorted_to_source;
    const std::string analysis_format = analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt});
    const std::vector<std::string> numbering = pairing_filters(frame_rate(distorted));
    // A GPU decoder's 8-bit picture (NV12) widened to a deeper analysis
    // format is made planar first: FFmpeg widens NV12 with a plain shift, but
    // a planar picture, as its software decoding gives it, with full-range
    // luma's top bits repeated, as the app's own decoders widen (widen_to).
    const auto planar_first = [&analysis_format](const std::optional<std::string>& hwaccel, const VideoInfo& info) {
        return hwaccel.has_value() && bit_depth(info.pix_fmt) <= 8 && bit_depth(analysis_format) > 8;
    };
    VideoChains chains;

    if (decoding.distorted && !(split && decoder_downloads(decoding.distorted))) {
        // Frames come as GPU surfaces and must come back to system memory
        // before any other filter can use them.
        chains.test.emplace_back("hwdownload");
        chains.test.push_back("format=" + hw_native_format(distorted.pix_fmt));
    }
    if (const std::optional<std::string> scale = size_change_scale(distorted, options.scale_algorithm)) {
        chains.test.push_back(*scale);
    }
    if (distorted_crop && !distorted_crop->is_noop(distorted.frame_size().first, distorted.frame_size().second)) {
        chains.test.push_back(distorted_crop->as_filter());
    }
    if (planar_first(decoding.distorted, distorted)) {
        chains.test.emplace_back("format=yuv420p");
    }
    chains.test.push_back("format=" + analysis_format);
    if (distorted.full_range() != source.full_range()) {
        // As they are, a full-range video's samples would all be off from a
        // limited one's: black is 0 in one and 16 in the other.
        chains.test.push_back(std::string("scale=out_range=") + (source.full_range() ? "pc" : "tv"));
    }
    if (upscale_distorted) {
        chains.test.push_back("scale=" + std::to_string(source_w) + ":" + std::to_string(source_h) +
                              ":flags=" + options.scale_algorithm);
    }
    chains.test.insert(chains.test.end(), numbering.begin(), numbering.end());

    // The source is cropped before the format conversion, as the test video
    // is, so chroma at a crop's edge is not filtered with the bars' samples.
    if (decoding.source && !(split && decoder_downloads(decoding.source))) {
        chains.source.emplace_back("hwdownload");
        chains.source.push_back("format=" + hw_native_format(source.pix_fmt));
    }
    if (const std::optional<std::string> scale = size_change_scale(source, options.scale_algorithm)) {
        chains.source.push_back(*scale);
    }
    if (source_crop && !source_crop->is_noop(source.frame_size().first, source.frame_size().second)) {
        chains.source.push_back(source_crop->as_filter());
    }
    if (planar_first(decoding.source, source)) {
        chains.source.emplace_back("format=yuv420p");
    }
    chains.source.push_back("format=" + analysis_format);
    if (sizes_differ && !upscale_distorted) {
        chains.source.push_back("scale=" + std::to_string(distorted_w) + ":" + std::to_string(distorted_h) +
                                ":flags=" + options.scale_algorithm);
    }
    chains.source.insert(chains.source.end(), numbering.begin(), numbering.end());
    return chains;
}

// A folder of its own for one run's logs, deleted with it.
TempFolder run_folder(std::string_view prefix) {
    return TempFolder::make<VmafRunError>(prefix, "A folder for the run's logs could not be made.");
}

// When a model= value names a custom file, copies it next to the log and
// names it by its bare file name (see libvmaf_options).
std::string model_for_run_folder(const std::string& model, const std::filesystem::path& folder) {
    const std::optional<std::string_view> named = model_file_of(model);
    if (!named) {
        return model;
    }
    const std::filesystem::path file = platform::path_from_text(*named);
    std::filesystem::copy_file(file, folder / file.filename(), std::filesystem::copy_options::overwrite_existing);
    return model_option_for_file(platform::path_text(file.filename()));
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
    std::size_t command = 0;  // which of the run's commands it is
};

// FFmpeg's -progress output, which writes each update as key=value lines:
// the latest fps is reported with every frame= line.
class ProgressLines {
public:
    void read(const std::string& line, long long total_frames, const ProgressCallback& on_progress) {
        static const std::regex frame_line(R"(^frame=(\d+))");
        static const std::regex fps_line(R"(^fps=\s*([\d.]+))");
        std::smatch match;
        if (std::regex_search(line, match, fps_line)) {
            last_fps_ = python_float(match[1].str()).value_or(last_fps_);
        } else if (std::regex_search(line, match, frame_line) && on_progress) {
            on_progress(std::stoll(match[1].str()), total_frames, last_fps_);
        }
    }

private:
    double last_fps_ = 0.0;
};

// The options of an FFmpeg run in `folder` whose error output is watched for
// FFmpeg rebuilding its filters (a video's size changed: size_changes.h),
// which sets `rebuilt`.
platform::ProcessOptions watched_options(const std::filesystem::path& folder,
                                         const std::shared_ptr<std::atomic<bool>>& rebuilt) {
    platform::ProcessOptions options{folder};
    options.on_error_line = rebuild_watcher(rebuilt);
    return options;
}

// Runs one FFmpeg command, reporting its progress. Throws Cancelled, and
// VideoSizeChanged when FFmpeg rebuilt its filters (it is ended then: what it
// scores after is not the run's).
FfmpegOutcome run_ffmpeg(const std::vector<std::string>& command, long long total_frames,
                         const ProgressCallback& on_progress, const CancelToken& cancel,
                         const std::filesystem::path& folder) {
    cancel.throw_if_cancelled();  // not one more FFmpeg after a Cancel
    const auto rebuilt = std::make_shared<std::atomic<bool>>(false);
    platform::ChildProcess ffmpeg = platform::ChildProcess::start(command, watched_options(folder, rebuilt));
    const CancelToken::Attached attached = cancel.attach(ffmpeg);
    ProgressLines progress;
    while (const std::optional<std::string> line = ffmpeg.read_line()) {
        if (cancel.cancelled() || *rebuilt) {
            ffmpeg.terminate();
            break;
        }
        progress.read(*line, total_frames, on_progress);
    }
    const int code = ffmpeg.wait();
    cancel.throw_if_cancelled();
    std::string errors = ffmpeg.standard_error();  // waits for the error output's end, and so its lines
    if (*rebuilt) {
        throw VideoSizeChanged();
    }
    return FfmpegOutcome{code, std::move(errors)};
}

// Runs the two FFmpegs of VMAF on the GPU side by side, as run_ffmpeg runs
// one: the first reports the progress, and one that fails ends the other.
// Returns the outcome of the first to fail of those whose pipe the app did
// not close (`closed_by_app`, by command), else of the first to fail, or the
// first's. Throws Cancelled, platform::ProcessError when one cannot start,
// and VideoSizeChanged when one rebuilt its filters, naming its video of
// `videos`, the one each command decodes (both are ended then).
FfmpegOutcome run_ffmpeg_pair(const std::vector<std::vector<std::string>>& commands,
                              const std::vector<std::filesystem::path>& videos,
                              const std::function<bool(std::size_t)>& closed_by_app, long long total_frames,
                              const ProgressCallback& on_progress, const CancelToken& cancel,
                              const std::filesystem::path& folder) {
    cancel.throw_if_cancelled();
    std::vector<std::shared_ptr<std::atomic<bool>>> rebuilt;
    std::vector<platform::ChildProcess> processes;
    for (const std::vector<std::string>& command : commands) {
        rebuilt.push_back(std::make_shared<std::atomic<bool>>(false));
        processes.push_back(platform::ChildProcess::start(command, watched_options(folder, rebuilt.back())));
    }
    const auto any_rebuilt = [&rebuilt] {
        return std::any_of(rebuilt.begin(), rebuilt.end(), [](const auto& flag) { return flag->load(); });
    };
    // Attached once all are in place: the vector moves them as it grows.
    std::vector<CancelToken::Attached> attached;
    for (platform::ChildProcess& process : processes) {
        attached.push_back(cancel.attach(process));
    }
    std::vector<FfmpegOutcome> outcomes(processes.size());
    std::vector<char> done(processes.size(), 0);
    std::vector<char> ended_here(processes.size(), 0);  // ended because of the other, or Cancel
    std::mutex mutex;
    std::condition_variable changed;
    // The FFmpegs that failed, in the order they did. The first is the
    // cause, unless the app closed its pipe: once one video's frames stop,
    // the app closes both, and the other FFmpeg can fail of that before the
    // one that stopped them has ended.
    std::vector<std::size_t> failed;
    std::vector<std::thread> readers;
    for (std::size_t index = 0; index < processes.size(); ++index) {
        readers.emplace_back([&, index] {
            platform::ChildProcess& ffmpeg = processes[index];
            ProgressLines progress;
            while (const std::optional<std::string> line = ffmpeg.read_line()) {
                if (index == 0) {
                    progress.read(*line, total_frames, on_progress);
                }
            }
            const int code = ffmpeg.wait();
            std::string errors = ffmpeg.standard_error();
            {
                const std::lock_guard lock(mutex);
                outcomes[index] = FfmpegOutcome{code, std::move(errors), index};
                done[index] = 1;
                if (code != 0) {
                    failed.push_back(index);
                }
            }
            changed.notify_all();
        });
    }
    {
        // Cancel ends both, and one failing ends the other: its frames would
        // have nothing to be paired with.
        std::unique_lock lock(mutex);
        while (std::find(done.begin(), done.end(), 0) != done.end()) {
            changed.wait_for(lock, std::chrono::milliseconds(100));
            if (cancel.cancelled() || !failed.empty() || any_rebuilt()) {
                for (std::size_t index = 0; index < processes.size(); ++index) {
                    if (done[index] == 0 && ended_here[index] == 0) {
                        processes[index].terminate();
                        ended_here[index] = 1;
                    }
                }
            }
        }
    }
    for (std::thread& reader : readers) {
        reader.join();
    }
    cancel.throw_if_cancelled();
    for (std::size_t index = 0; index < processes.size(); ++index) {
        if (*rebuilt[index]) {
            throw VideoSizeChanged(videos.at(index));
        }
    }
    const auto cause = std::find_if(failed.begin(), failed.end(),
                                    [&closed_by_app](std::size_t index) { return !closed_by_app(index); });
    return outcomes[cause != failed.end() ? *cause : failed.empty() ? 0 : failed.front()];
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
    return join(std::vector<std::string>(lines.begin() + static_cast<std::ptrdiff_t>(first), lines.end()), "\n");
}

const app_log::Logger logger("vml.vmaf_runner");

// Says that an FFmpeg of a pair ended because a write into the app's pipe
// failed (refused_frame_write), and that its attempt is made again. The
// pair's first command decodes the test video (build_stream_commands).
void report_refused_write(const FfmpegOutcome& outcome, const HardwareDecoding& plan,
                          const StatusCallback& on_status) {
    const std::string side = outcome.command == 0 ? "test video" : "reference";
    logger.warning("FFmpeg ended partway through a " + side + " frame: a write into the app's pipe failed (GPU "
                   "decode: " + plan.describe() + "); decoding the same way again. Last output:\n" +
                   last_lines(outcome.errors, 8));
    if (on_status) {
        // Worded as Vship's retry (remade_vship_pass), which the window translates.
        on_status({"Reading the videos failed (FFmpeg ended partway through a " + side +
                   " frame.); trying again\xE2\x80\xA6"});
    }
}

// An FFmpeg that failed, in the log: its exit code, the decode plan, whether
// another plan is tried, and the end of its error output.
void log_failed_attempt(const FfmpegOutcome& outcome, const HardwareDecoding& plan, bool retrying) {
    logger.warning("FFmpeg exited with code " + platform::exit_code_text(outcome.exit_code) + " (GPU decode: " +
                   plan.describe() + ")" + (retrying ? "; retrying" : "") + ". Last output:\n" +
                   last_lines(outcome.errors, 25));
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

// The inputs of a run's two videos: a fed video's pictures (grain_feed.h),
// the other's file.
InputOptions run_inputs(const PairFeeds& feeds, const VideoInfo& source, const VideoInfo& distorted) {
    InputOptions inputs;
    if (const PictureFeed* fed = feeds.distorted()) {
        inputs.distorted_fed = fed->input_arguments();
    } else {
        inputs.distorted = distorted.decode_options();
    }
    if (const PictureFeed* fed = feeds.source()) {
        inputs.source_fed = fed->input_arguments();
    } else {
        inputs.source = source.decode_options();
    }
    return inputs;
}

// The FFmpeg command of a run for a decode plan, the run's model= value (by
// its bare file name for a file), XPSNR's stats file name, if any, and the
// run's feeds of its videos' pictures.
using CommandBuilder =
    std::function<std::vector<std::string>(const HardwareDecoding& plan, const std::string& run_model,
                                           const std::optional<std::string>& xpsnr_log_name, const PairFeeds& feeds)>;

// Runs the FFmpeg command `build_command` makes to its end and parses its
// logs, falling back to other decode plans when hardware decoding fails.
// The logs are kept in a folder named from `folder_prefix`. Each FFmpeg gets
// the pictures of those of `source` and `distorted` (either may be nothing)
// the app feeds it (grain_feed.h).
ParsedScores execute_run(const CommandBuilder& build_command, const VmafOptions& options, const std::string& model,
                         double fps, long long total_frames, const HardwareDecoding& decoding,
                         const std::string& folder_prefix, const ProgressCallback& on_progress,
                         const StatusCallback& on_status, const CancelToken& cancel,
                         const RunEnvironment& environment, const VideoInfo* source, const VideoInfo* distorted) {
    if (options.requested_metrics().empty()) {
        throw VmafRunError("Select at least one metric to calculate.");
    }
    const TempFolder folder = run_folder(folder_prefix);
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
    const std::vector<HardwareDecoding> ladder = fallback_ladder(decoding);
    for (std::size_t attempt = 0; attempt < ladder.size(); ++attempt) {
        // Before the FFmpeg, which it outlives.
        PairFeeds feeds(source, distorted, environment.ffmpeg, environment.ffprobe, cancel);
        const HardwareDecoding plan = feeds.decoding(ladder[attempt]);
        if (on_status) {
            on_status(attempt == 0 ? decoding_status("Running ffmpeg", plan, StatusKind::starting)
                                   : decoding_status("GPU decode failed, retrying", plan));
        }
        const std::vector<std::string> command = build_command(plan, run_model, xpsnr_log_name, feeds);
        logger.info("FFmpeg: " + python_list2cmdline(command));
        try {
            outcome = run_ffmpeg(command, total_frames, on_progress, cancel, folder.path());
        } catch (const platform::ProcessError& error) {
            throw VmafRunError(std::string("FFmpeg could not be started: ") + error.what());
        }
        feeds.finish();
        if (outcome.exit_code == 0) {
            break;
        }
        log_failed_attempt(outcome, plan, attempt + 1 < ladder.size());
        // A failed attempt can leave a partial log, which would be read as
        // the retry's.
        std::error_code ignored;
        std::filesystem::remove(log_path, ignored);
        if (xpsnr_log_path) {
            std::filesystem::remove(*xpsnr_log_path, ignored);
        }
    }
    if (outcome.exit_code != 0) {
        throw VmafRunError("ffmpeg exited with code " + platform::exit_code_text(outcome.exit_code),
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
                           join(missing.empty() ? options.requested_metrics() : missing, ", "));
    }
    return scores;
}

// execute_run for a comparison of two videos (build_filtergraph and
// build_ffmpeg_command).
ParsedScores execute_run(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                         const VmafOptions& options, const std::optional<CropBox>& source_crop,
                         const std::optional<CropBox>& distorted_crop, const std::string& model, double fps,
                         long long total_frames, const HardwareDecoding& decoding, const ProgressCallback& on_progress,
                         const StatusCallback& on_status, const CancelToken& cancel) {
    const auto build_command = [&](const HardwareDecoding& plan, const std::string& run_model,
                                   const std::optional<std::string>& xpsnr_log_name, const PairFeeds& feeds) {
        const std::string graph = build_filtergraph(source, distorted, options, source_crop, distorted_crop, plan,
                                                    "vmaf_log.json", run_model, xpsnr_log_name, environment.cores);
        return build_ffmpeg_command(environment.ffmpeg, distorted.path, source.path, graph, plan,
                                    options.duration_limit, run_inputs(feeds, source, distorted));
    };
    return execute_run(build_command, options, model, fps, total_frames, decoding, "vmaf_run_", on_progress, on_status,
                       cancel, environment, &source, &distorted);
}

// The duration limit of a GPU run, in seconds (0: none): one frame more
// than the run's, because FFmpeg's libvmaf filter also scores the first
// frame at or past the limit before its output stops there, and the GPU must
// be given the same frames.
double gpu_duration_limit(const VmafOptions& options, double fps) {
    return options.duration_limit > 0 && fps > 0 ? options.duration_limit + 1 / fps : 0.0;
}

// The GPU's scores as the run's (the Python app's _gpu_frame_scores): each
// kept as a float, as the Python app keeps FFmpeg's, the frames timed at
// `fps`. Throws when no frame was scored or a requested metric has none.
ParsedScores gpu_scores(const PairScores& scored, double fps, const VmafOptions& options) {
    if (scored.frames.empty()) {
        throw LibvmafError("GPU VMAF scored no frames");
    }
    ParsedScores scores;
    scores.frame.assign(scored.frames.begin(), scored.frames.end());
    scores.time = times_of(scores.frame, fps);
    for (const auto& [key, values] : scored.by_metric) {
        std::vector<float>& column = scores.metrics[key];
        for (const double value : values) {
            column.push_back(static_cast<float>(value));
        }
    }
    for (const char* key : column_keys) {
        if (scores.has(key)) {
            scores.order.emplace_back(key);
        }
    }
    std::vector<std::string> missing;
    for (const std::string& key : options.requested_metrics()) {
        if (!scores.has(key)) {
            missing.push_back(key);
        }
    }
    if (!missing.empty()) {
        throw VmafRunError("No results for requested metrics: " + join(missing, ", "));
    }
    return scores;
}

// The plan's metrics in the run's own words: "PSNR, SSIM and XPSNR".
std::string metric_names(const GpuPlan& plan) {
    std::vector<std::string> names;
    for (const auto& [key, model] : plan.models) {
        names.push_back(key == "psnr" ? "PSNR" : key == "ssim" ? "SSIM" : key == "xpsnr" ? "XPSNR" : key);
    }
    std::string text;
    for (std::size_t i = 0; i < names.size(); ++i) {
        text += (i == 0 ? "" : i + 1 == names.size() ? " and " : ", ") + names[i];
    }
    return text;
}

// What a plan's run is doing, for its statuses: "VMAF on the GPU", or
// "PSNR and SSIM in the app".
std::string doing(const GpuPlan& plan) {
    return plan.backend == "cpu" ? metric_names(plan) + " in the app" : "VMAF on the GPU";
}

// The plan's metrics scored outside FFmpeg's filters, each video decoded by
// an FFmpeg of its own whose frames the app pairs and the plan's scorer
// scores (GpuAttempt): VMAF, NEG and VMAF v1 on the GPU, or PSNR, SSIM and
// XPSNR in the app. Falls back to other decode plans when hardware decoding
// fails.
ParsedScores execute_gpu_run(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                             const VmafOptions& options, const std::optional<CropBox>& source_crop,
                             const std::optional<CropBox>& distorted_crop, const GpuPlan& plan, double fps,
                             long long total_frames, const HardwareDecoding& decoding,
                             const ProgressCallback& on_progress, const StatusCallback& on_status,
                             const CancelToken& cancel) {
    if (options.requested_metrics().empty()) {
        throw VmafRunError("Select at least one metric to calculate.");
    }
    const TempFolder folder = run_folder(plan.backend == "cpu" ? "vmaf_cpu_run_" : "vmaf_gpu_run_");
    const double limit = gpu_duration_limit(options, fps);
    const std::vector<HardwareDecoding> ladder = fallback_ladder(decoding);
    std::optional<PairScores> scored;
    FfmpegOutcome outcome;
    std::size_t attempt_number = 0;
    // A write into the app's pipe that Windows refused is made again once a
    // run, with the same decoding: it may not happen again, and the next
    // plan would decode on the CPU for nothing.
    bool refusal_retried = false;
    bool same_again = false;  // this attempt is the last one's plan again
    while (attempt_number < ladder.size()) {
        // Before the FFmpegs, which it outlives.
        PairFeeds feeds(&source, &distorted, environment.ffmpeg, environment.ffprobe, cancel);
        const HardwareDecoding decode = feeds.decoding(ladder[attempt_number]);
        if (on_status && !same_again) {
            on_status(attempt_number == 0
                          ? decoding_status("Running ffmpeg, " + doing(plan), decode, StatusKind::starting)
                          : decoding_status("GPU decode failed, retrying", decode));
        }
        same_again = false;
        // Each attempt has its own scorer and pipes: a failed attempt's
        // cannot be used again.
        GpuAttempt attempt(plan, options.n_subsample);
        const auto abandon = [&attempt] {
            try {
                attempt.finish(false);
            } catch (const std::exception&) {
                // The failure that ends the attempt is the one to report.
            }
        };
        try {
            const std::vector<std::vector<std::string>> commands = build_stream_commands(
                environment.ffmpeg, distorted.path, source.path,
                build_stream_graphs(source, distorted, options, source_crop, distorted_crop, decode), decode,
                attempt.output_args(limit), run_inputs(feeds, source, distorted));
            for (const std::vector<std::string>& command : commands) {
                logger.info("FFmpeg: " + python_list2cmdline(command));
            }
            outcome = run_ffmpeg_pair(commands, {distorted.path, source.path},
                                      [&attempt](std::size_t command) { return attempt.closed_pipe_of(command); },
                                      total_frames, on_progress, cancel, folder.path());
        } catch (const platform::ProcessError& error) {
            abandon();
            throw VmafRunError(std::string("FFmpeg could not be started: ") + error.what());
        } catch (...) {
            abandon();
            throw;
        }
        try {
            feeds.finish();  // a fed video decoded in part is no score
        } catch (...) {
            abandon();
            throw;
        }
        // It throws when the scoring failed: FFmpeg's failure is then caused
        // by it, and no other decode plan would help.
        scored = attempt.finish(outcome.exit_code == 0);
        if (outcome.exit_code == 0) {
            break;
        }
        if (!refusal_retried && refused_frame_write(outcome.exit_code, outcome.errors)) {
            refusal_retried = true;
            same_again = true;
            report_refused_write(outcome, decode, on_status);
            continue;
        }
        log_failed_attempt(outcome, decode, attempt_number + 1 < ladder.size());
        ++attempt_number;
    }
    if (outcome.exit_code != 0) {
        throw VmafRunError("ffmpeg exited with code " + platform::exit_code_text(outcome.exit_code),
                           last_lines(outcome.errors, 25));
    }
    if (!scored) {
        throw LibvmafError("GPU VMAF scored no frames");
    }
    return gpu_scores(*scored, fps, options);
}

// The keys of the metrics a plan scores.
std::vector<std::string> plan_keys(const GpuPlan& plan) {
    std::vector<std::string> keys;
    for (const auto& [key, model] : plan.models) {
        keys.push_back(key);
    }
    return keys;
}

// The plan's metrics with both videos decoded in this process (the Python
// app's _score_decoded_on_gpu): VMAF and NEG on the GPU (score_decoded),
// with VMAF v1 from whole frames (score_decoded_v1), or PSNR, SSIM and XPSNR
// in the app (score_decoded_cpu), over the same frames execute_gpu_run's
// FFmpegs would give the scorer.
ParsedScores score_decoded_here(const RunEnvironment& environment, const VideoInfo& source,
                                const VideoInfo& distorted, const VmafOptions& options,
                                const std::optional<CropBox>& source_crop,
                                const std::optional<CropBox>& distorted_crop, const GpuPlan& plan, double fps,
                                long long total_frames, const HardwareDecoding& decoding,
                                const std::pair<std::string, std::string>& decoders,
                                const ProgressCallback& on_progress, const StatusCallback& on_status,
                                const CancelToken& cancel) {
    const double limit = gpu_duration_limit(options, fps);
    const std::optional<std::string> limit_text = limit > 0 ? std::optional(format_fixed(limit, 3)) : std::nullopt;
    if (on_status) {
        on_status(decoding_status("Running " + doing(plan), decoding));
    }
    const std::vector<std::string> keys = plan_keys(plan);
    const PairScores scored =
        plan.backend == "cpu"
            ? score_decoded_cpu(source, distorted, source_crop, distorted_crop, plan.width, plan.height,
                                plan.bit_depth, keys, options.n_subsample, limit_text, total_frames, decoders,
                                plan.threads, plan.frame_rate, options.scale_algorithm, environment.ffmpeg,
                                environment.ffprobe, on_progress, cancel)
        : std::find(keys.begin(), keys.end(), "vmaf_v1") != keys.end()
            ? score_decoded_v1(source, distorted, source_crop, distorted_crop, plan, options.n_subsample, limit_text,
                               total_frames, options.scale_algorithm, decoders, environment.ffmpeg,
                               environment.ffprobe, on_progress, cancel)
            : score_decoded(source, distorted, source_crop, distorted_crop, plan, options.n_subsample, limit_text,
                            total_frames, options.scale_algorithm, decoders, environment.ffmpeg, environment.ffprobe,
                            on_progress, cancel);
    return gpu_scores(scored, fps, options);
}

// The plan's metrics outside FFmpeg's filters (the Python app's
// _score_on_gpu and _score_cpu_metrics, which choose alike for the Vulkan
// backend): where this process has a decoder for each video
// (decoders_here), the videos are decoded here, by the decoder FFmpeg would
// have used, or by the software decoder for a video FFmpeg would decode in
// software. The frames are the same, without FFmpeg's decoding and the CPU
// copies and pipes behind it. Otherwise, or when that decoding fails after
// it has started, FFmpeg decodes them. libvmaf's CUDA code takes only
// NVIDIA's decoder's pictures and the software decoder's, so with it VMAF
// and NEG without VMAF v1 are decoded here only by those.
ParsedScores score_plan(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                        const VmafOptions& options, const std::optional<CropBox>& source_crop,
                        const std::optional<CropBox>& distorted_crop, const GpuPlan& plan, double fps,
                        long long total_frames, const HardwareDecoding& decoding,
                        const ProgressCallback& on_progress, const StatusCallback& on_status,
                        const CancelToken& cancel) {
    // A video whose size changes is decoded by FFmpeg, which scales each of
    // its frames (size_changes.h); the app's decoders read one size.
    const std::optional<std::pair<std::string, std::string>> decoders =
        source.scaled_to || distorted.scaled_to ? std::nullopt : decoders_here(decoding);
    const auto cuda_takes = [](const std::string& decoder) {
        return decoder == "nvidia" || decoder == software_decoder;
    };
    const std::vector<std::string> keys = plan_keys(plan);
    if (decoders && (plan.backend != "cuda" || std::find(keys.begin(), keys.end(), "vmaf_v1") != keys.end() ||
                     (cuda_takes(decoders->first) && cuda_takes(decoders->second)))) {
        try {
            return score_decoded_here(environment, source, distorted, options, source_crop, distorted_crop, plan,
                                      fps, total_frames, decoding, *decoders, on_progress, on_status, cancel);
        } catch (const GpuDecodeUnavailableError& error) {
            // FFmpeg decodes the videos.
            logger.info(doing(plan) + ": the videos are decoded by FFmpeg (" + error.what() + ")");
        } catch (const GpuDecodeFailedError& error) {
            logger.warning("Decoding for " + (plan.backend == "cpu" ? metric_names(plan) : "VMAF on the GPU") +
                           " failed; decoding through FFmpeg instead: " + error.what());
            if (on_status) {
                on_status({decoding_failed_status(error)});
            }
        }
    }
    return execute_gpu_run(environment, source, distorted, options, source_crop, distorted_crop, plan, fps,
                           total_frames, decoding, on_progress, on_status, cancel);
}

// ----------------------------------------------- PSNR, SSIM and XPSNR in the app

// The environment variable that, set to "ffmpeg", leaves PSNR, SSIM and
// XPSNR to FFmpeg's filters instead of the app's own libvmaf, to compare
// the two.
constexpr std::string_view cpu_metrics_variable = "VML_CPU_METRICS";

// The metric FFmpeg's libvmaf feature is, where the app scores it itself.
std::optional<std::string> cpu_metric_of_feature(const std::string& feature) {
    if (feature == "name=psnr") {
        return "psnr";
    }
    if (feature == "name=float_ssim") {
        return "ssim";
    }
    return std::nullopt;
}


// XPSNR's frame rate, in whole frames per second, as FFmpeg's xpsnr filter
// takes it from its second input, the test video: FFmpeg gives an input the
// rate av_guess_frame_rate() returns, r_frame_rate when the average rate is
// known and within a tenth of it. Otherwise that rate cannot be predicted
// here, and FFmpeg's filter scores XPSNR.
std::optional<int> xpsnr_frame_rate(const VideoInfo& info) {
    const double average = info.average_fps, nominal = info.nominal_fps;
    if (average <= 0 || nominal <= 0 || std::abs(1.0 - average / nominal) >= 0.09) {
        return std::nullopt;
    }
    return static_cast<int>(nominal);
}

// Whether libvmaf-fast's port of FFmpeg's xpsnr filter gives exactly
// FFmpeg's score for this comparison: up to 12 bits, and an even size above
// 2048x1152 (FFmpeg's filter reads beyond the picture of an odd one).
bool xpsnr_in_app(int width, int height, int bit_depth) {
    const bool odd = width % 2 != 0 || height % 2 != 0;
    if (bit_depth > 12 || (static_cast<long long>(width) * height > 2048LL * 1152 && odd)) {
        return false;
    }
    return cpu_scores_xpsnr();
}

// Whether the app will decode both videos itself (score_decoded_cpu), as far
// as can be known before it asks the decoders: a video compared at another
// size only with a decoder that scales it as `ffmpeg` would (scaling_refusal;
// the software decoder scales only what it alone decodes, software_only).
bool decoded_in_app(const VideoInfo& source, const VideoInfo& distorted, const std::optional<CropBox>& source_crop,
                    const std::optional<CropBox>& distorted_crop, std::pair<int, int> size,
                    const std::string& scale_algorithm, const HardwareDecoding& decoding, const std::string& ffmpeg) {
    const std::optional<std::pair<std::string, std::string>> decoders =
        source.scaled_to || distorted.scaled_to ? std::nullopt : decoders_here(decoding);  // as score_plan
    if (!decoders || !decodes_codec(ascii_lower(source.codec_name), decoders->first) ||
        !decodes_codec(ascii_lower(distorted.codec_name), decoders->second) ||
        source.full_range() != distorted.full_range()) {  // as require_same_range
        return false;
    }
    const auto decodes_here = [&ffmpeg](const std::string& decoder, const DecodePlan& plan) {
        return !plan.scaled() || ((decoder != software_decoder || plan.software_only) && !scaling_refusal(plan, ffmpeg));
    };
    try {
        return decodes_here(decoders->first, plan_decode(source, source_crop, 6, true, size, scale_algorithm)) &&
               decodes_here(decoders->second,
                            plan_decode(distorted, distorted_crop, 6, true, size, scale_algorithm));
    } catch (const GpuDecodeUnavailableError&) {
        return false;
    }
}

// The plan for scoring PSNR, SSIM and XPSNR with the bundled libvmaf in the
// app (run_cpu_metrics) instead of with FFmpeg's libvmaf and xpsnr filters,
// or nothing to leave them to FFmpeg's filters. The app scores them when
// they are the run's only metrics (VMAF on the CPU keeps them in FFmpeg's
// filters, which it runs anyway). XPSNR stays with FFmpeg where libvmaf-fast's
// score would not be FFmpeg's exactly (xpsnr_frame_rate, xpsnr_in_app),
// while PSNR and SSIM are still scored in the app. `cores` are "Auto"'s
// threads; `ffmpeg` scales a video the decoders cannot (decoded_in_app).
std::optional<GpuPlan> cpu_metrics_plan(const VmafOptions& options, std::pair<int, int> size,
                                        const VideoInfo& source, const VideoInfo& distorted,
                                        const HardwareDecoding& decoding,
                                        const std::optional<CropBox>& source_crop,
                                        const std::optional<CropBox>& distorted_crop, unsigned cores,
                                        const std::string& ffmpeg) {
    if (!cpu_metrics_in_app() || uses_vmaf_model(options)) {
        return std::nullopt;
    }
    std::vector<std::string> metrics;
    for (const std::string& feature : options.extra_features) {
        const std::optional<std::string> metric = cpu_metric_of_feature(feature);
        if (!metric) {
            return std::nullopt;
        }
        if (std::find(metrics.begin(), metrics.end(), *metric) == metrics.end()) {
            metrics.push_back(*metric);
        }
    }
    const auto [width, height] = size;
    const int depth = bit_depth(analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt}));
    const std::optional<int> frame_rate = options.compute_xpsnr ? xpsnr_frame_rate(distorted) : std::nullopt;
    if (frame_rate && xpsnr_in_app(width, height, depth)) {
        metrics.emplace_back("xpsnr");
    } else if (options.compute_xpsnr && !(decoding.source && decoding.distorted)) {
        // XPSNR's FFmpeg beside them would decode the videos again, and a
        // video the CPU decodes would then be decoded twice on the CPU: 70%
        // more CPU per frame for a 4K film against its encode, and slower for
        // a 4K film against a 1080p encode (97 fps against 112). FFmpeg's
        // single run decodes each video once for both libvmaf and XPSNR.
        return std::nullopt;
    }
    if (metrics.empty()) {
        return std::nullopt;
    }
    if (metrics == std::vector<std::string>{"xpsnr"} &&
        !decoded_in_app(source, distorted, source_crop, distorted_crop, size, options.scale_algorithm, decoding,
                        ffmpeg)) {
        // XPSNR alone, with its frames from FFmpeg's pipes: FFmpeg's own
        // filter keeps up with FFmpeg's decoding and uses far less (the
        // Python app measured twice the CPU and 825 MB against 453 for a 4K
        // film against a 1080p encode).
        return std::nullopt;
    }
    GpuPlan plan;
    for (const std::string& metric : metrics) {
        plan.models.emplace_back(metric, metric);
    }
    plan.width = width;
    plan.height = height;
    plan.bit_depth = depth;
    plan.backend = "cpu";
    plan.threads = options.n_threads > 0 ? options.n_threads : static_cast<int>(cores);
    plan.frame_rate = frame_rate.value_or(0);
    return plan;
}

// Reports two runs' progress as one: that of whichever is behind (the
// Python app's _Progress).
class SlowerProgress {
public:
    SlowerProgress(ProgressCallback on_progress, std::size_t parts)
        : on_progress_(std::move(on_progress)), latest_(parts) {}

    // The progress callback of run number `index`.
    ProgressCallback part(std::size_t index) {
        if (!on_progress_) {
            return nullptr;
        }
        return [this, index](long long done, long long whole, double fps) {
            Progress behind;
            {
                const std::lock_guard lock(mutex_);
                latest_[index] = Progress{done, whole, fps};
                if (std::any_of(latest_.begin(), latest_.end(), [](const auto& latest) { return !latest; })) {
                    return;
                }
                behind = **std::min_element(latest_.begin(), latest_.end(),
                                            [](const auto& a, const auto& b) { return a->done < b->done; });
            }
            on_progress_(behind.done, behind.whole, behind.fps);
        };
    }

private:
    struct Progress {
        long long done = 0;
        long long whole = 0;
        double fps = 0;
    };
    ProgressCallback on_progress_;
    std::mutex mutex_;
    std::vector<std::optional<Progress>> latest_;
};

// ------------------------------------------- the GPU's run in a process of its own

nlohmann::json hwaccel_json(const std::optional<std::string>& hwaccel) {
    return hwaccel ? nlohmann::json(*hwaccel) : nlohmann::json();
}

std::optional<std::string> hwaccel_from(const nlohmann::json& text) {
    return text.is_null() ? std::nullopt : std::optional(text.get<std::string>());
}

// The scores the GPU's process leaves for its parent: the frame numbers,
// then each metric's values in `order`, as this PC stores them.
constexpr const char* scores_file = "scores.bin";

ParsedScores read_scores(const nlohmann::json& result, const std::filesystem::path& folder, double fps) {
    ParsedScores scores;
    const auto count = result.at("frames").get<std::size_t>();
    scores.order = result.at("order").get<std::vector<std::string>>();
    std::ifstream file(folder / scores_file, std::ios::binary);
    scores.frame.resize(count);
    file.read(reinterpret_cast<char*>(scores.frame.data()), static_cast<std::streamsize>(count * sizeof(int)));
    for (const std::string& key : scores.order) {
        std::vector<float>& values = scores.metrics[key];
        values.resize(count);
        file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    }
    if (!file) {
        throw std::runtime_error("The scores could not be read back from the process that made them.");
    }
    scores.time = times_of(scores.frame, fps);
    return scores;
}

nlohmann::json plan_json(const GpuPlan& plan) {
    nlohmann::json models = nlohmann::json::array();
    for (const auto& [key, version] : plan.models) {
        models.push_back({key, version});
    }
    return {{"models", models},           {"width", plan.width},     {"height", plan.height},
            {"bit_depth", plan.bit_depth}, {"device", plan.device},   {"backend", plan.backend},
            {"beside", plan.beside},       {"threads", plan.threads}, {"frame_rate", plan.frame_rate}};
}

GpuPlan plan_from(const nlohmann::json& text) {
    GpuPlan plan;
    for (const nlohmann::json& model : text.at("models")) {
        plan.models.emplace_back(model.at(0).get<std::string>(), model.at(1).get<std::string>());
    }
    text.at("width").get_to(plan.width);
    text.at("height").get_to(plan.height);
    text.at("bit_depth").get_to(plan.bit_depth);
    text.at("device").get_to(plan.device);
    text.at("backend").get_to(plan.backend);
    text.at("beside").get_to(plan.beside);
    text.at("threads").get_to(plan.threads);
    text.at("frame_rate").get_to(plan.frame_rate);
    return plan;
}

// The plan's metrics scored in a process of its own (score_plan in the
// child, `task` "gpu-run" or "cpu-metrics"), so that a crash in libvmaf, the
// GPU's code or its driver ends that process and not the run (the Python
// app's run_isolated(..., what="libvmaf")). Throws what run_isolated throws.
ParsedScores score_plan_apart(std::string_view task, const RunEnvironment& environment, const VideoInfo& source,
                              const VideoInfo& distorted, const VmafOptions& options,
                              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                              const GpuPlan& plan, double fps, long long total_frames,
                              const HardwareDecoding& decoding, const ProgressCallback& on_progress,
                              const StatusCallback& on_status, const CancelToken& cancel) {
    const nlohmann::json request{
        {"ffmpeg", environment.ffmpeg},
        {"ffprobe", environment.ffprobe},
        {"source", video_json(source)},
        {"distorted", video_json(distorted)},
        {"options", options_json(options)},
        {"source_crop", crop_json(source_crop)},
        {"distorted_crop", crop_json(distorted_crop)},
        {"plan", plan_json(plan)},
        {"fps", fps},
        {"total_frames", total_frames},
        {"decoding", {{"source", hwaccel_json(decoding.source)}, {"distorted", hwaccel_json(decoding.distorted)}}}};
    ParsedScores scores;
    run_isolated(
        task, "libvmaf", request,
        [&](const nlohmann::json& message) {
            if (message.contains("progress") && on_progress) {
                const nlohmann::json& counts = message["progress"];
                on_progress(counts.at(0).get<long long>(), counts.at(1).get<long long>(), counts.at(2).get<double>());
            } else if (message.contains("status") && on_status) {
                on_status(status_from_json(message["status"]));
            }
        },
        cancel,
        [&](const nlohmann::json& result, const std::filesystem::path& folder) {
            scores = read_scores(result, folder, fps);
        });
    return scores;
}

// VMAF, NEG and VMAF v1 on the GPU (score_plan), with the plan's CPU metrics
// beside them, one GPU pass at a time, in a process of its own where a
// program to run it is set (score_plan_apart). Throws what they throw.
ParsedScores score_on_gpu(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                          const VmafOptions& options, const std::optional<CropBox>& source_crop,
                          const std::optional<CropBox>& distorted_crop, const GpuPlan& plan, double fps,
                          long long total_frames, const HardwareDecoding& decoding,
                          const ProgressCallback& on_progress, const StatusCallback& on_status,
                          const CancelToken& cancel) {
    std::string models;
    for (const auto& [key, model] : plan.models) {
        models += (models.empty() ? "" : ", ") + model;
    }
    std::string beside;
    if (!plan.beside.empty()) {
        GpuPlan cpu;
        for (const std::string& key : plan.beside) {
            cpu.models.emplace_back(key, key);
        }
        beside = ", with " + metric_names(cpu) + " beside it from the same frames (" + cpu_library_build() + ", " +
                 std::to_string(plan.threads) + " threads)";
    }
    logger.info("VMAF on the GPU (" + models +
                "): " + (plan.backend == "vulkan" ? vulkan_library_build() : cuda_library_build()) + beside);
    const std::unique_lock pass = take_gpu_turn(
        [&on_status] {
            if (on_status) {
                on_status({std::string(gpu_wait_message), StatusKind::gpu_wait});
            }
        },
        cancel);
    if (!isolation_program()) {
        return score_plan(environment, source, distorted, options, source_crop, distorted_crop, plan, fps,
                          total_frames, decoding, on_progress, on_status, cancel);
    }
    return score_plan_apart("gpu-run", environment, source, distorted, options, source_crop, distorted_crop, plan, fps,
                            total_frames, decoding, on_progress, on_status, cancel);
}

// score_on_gpu without metrics beside, or nothing when it fails for any
// reason but Cancel: the run is then made again with VMAF on the CPU, as on a
// PC without a GPU that scores it.
std::optional<ParsedScores> run_on_gpu(const RunEnvironment& environment, const VideoInfo& source,
                                       const VideoInfo& distorted, const VmafOptions& options,
                                       const std::optional<CropBox>& source_crop,
                                       const std::optional<CropBox>& distorted_crop, const GpuPlan& plan, double fps,
                                       long long total_frames, const HardwareDecoding& decoding,
                                       const ProgressCallback& on_progress, const StatusCallback& on_status,
                                       const CancelToken& cancel) {
    try {
        return score_on_gpu(environment, source, distorted, options, source_crop, distorted_crop, plan, fps,
                            total_frames, decoding, on_progress, on_status, cancel);
    } catch (const Cancelled&) {
        throw;
    } catch (const VideoSizeChanged&) {
        throw;  // the run is made again, the video marked (run_vmaf)
    } catch (const FilmGrainRefusal&) {
        throw;  // final: the CPU's FFmpeg gets its pictures from the same decoder (grain_feed.h)
    } catch (const std::exception& error) {
        logger.error(std::string("VMAF on the GPU failed; calculating it on the CPU: ") + error.what());
        if (on_status) {
            on_status({std::string(vmaf_gpu_failed) + " (" + error.what() + "); calculating it on the CPU…",
                       StatusKind::gpu_vmaf_failed});
        }
        return std::nullopt;
    }
}

// Scores made on a thread beside the run's own (`run`: XPSNR by an FFmpeg of
// its own beside the app's PSNR and SSIM, or the app's CPU metrics beside the
// GPU's), with the run's Cancel and Pause passed on to it for as long as it
// runs. Its token is its own, so that it can be stopped alone when the run's
// own scoring fails. (With only a Cancel passed on, and only once that
// scoring had ended, it went on scoring through a Pause; the Python app hands
// it the run's own process handle.) Stopped and waited for when it goes.
class ScoresBeside {
public:
    ScoresBeside(std::function<ParsedScores(const CancelToken&)> run, const CancelToken& cancel) : cancel_(cancel) {
        run_ = std::thread([this, run = std::move(run)] {
            try {
                ParsedScores found = run(stop_);
                const std::lock_guard lock(mutex_);
                scores_ = std::move(found);
            } catch (...) {
                const std::lock_guard lock(mutex_);
                error_ = std::current_exception();
            }
            const std::lock_guard lock(mutex_);
            done_ = true;
            ended_.notify_all();
        });
        relay_ = std::thread([this] { relay(); });
    }
    ~ScoresBeside() {
        stop();
        join();
    }
    ScoresBeside(const ScoresBeside&) = delete;
    ScoresBeside& operator=(const ScoresBeside&) = delete;

    void stop() { stop_.cancel(); }
    // Waits for its run to end.
    void join() {
        if (relay_.joinable()) {
            relay_.join();
        }
        if (run_.joinable()) {
            run_.join();
        }
    }
    // Its scores, once joined; throws what its run threw.
    std::optional<ParsedScores> result() {
        if (error_) {
            std::rethrow_exception(error_);
        }
        return std::move(scores_);
    }

private:
    // Passes the run's Cancel and Pause on until the run ends.
    void relay() {
        std::unique_lock lock(mutex_);
        while (!ended_.wait_for(lock, std::chrono::milliseconds(50), [this] { return done_; })) {
            lock.unlock();
            if (cancel_.cancelled()) {
                stop_.cancel();
            }
            if (cancel_.paused() && !stop_.paused()) {
                stop_.pause();
            } else if (!cancel_.paused() && stop_.paused()) {
                stop_.resume();
            }
            lock.lock();
        }
    }

    const CancelToken& cancel_;
    CancelToken stop_;
    std::mutex mutex_;
    std::condition_variable ended_;
    bool done_ = false;
    std::optional<ParsedScores> scores_;
    std::exception_ptr error_;
    std::thread run_;
    std::thread relay_;
};

// XPSNR's scores (`xpsnr`, from its own FFmpeg) taken into `frames`, for the
// frames PSNR and SSIM were scored for: NaN for a frame it has none for.
void merge_xpsnr(ParsedScores& frames, const ParsedScores& xpsnr) {
    std::map<int, float> by_frame;
    const std::vector<float>& values = xpsnr.metrics.at("xpsnr");
    for (std::size_t i = 0; i < xpsnr.frame.size(); ++i) {
        by_frame[xpsnr.frame[i]] = values[i];
    }
    std::vector<float>& column = frames.metrics["xpsnr"];
    column.clear();
    for (const int frame : frames.frame) {
        const auto found = by_frame.find(frame);
        column.push_back(found == by_frame.end() ? std::numeric_limits<float>::quiet_NaN() : found->second);
    }
    frames.order.clear();
    for (const char* key : column_keys) {
        if (frames.has(key)) {
            frames.order.emplace_back(key);
        }
    }
}

// PSNR, SSIM and XPSNR with the bundled libvmaf's CPU extractors (the Python
// app's _run_cpu_metrics), in a process of its own where a program to run it
// is set (score_plan_apart), on the frames that would reach FFmpeg's
// filters, paired as those filters pair them. XPSNR that the plan leaves to
// FFmpeg is scored by an FFmpeg of its own alongside, and its scores are
// taken for the frames PSNR and SSIM were scored for. XPSNR alone is scored
// for every frame, as FFmpeg's filter scores it whatever n_subsample is.
// Nothing when it fails for any reason but Cancel: FFmpeg's filters then
// calculate the metrics.
std::optional<ParsedScores> run_cpu_metrics(const RunEnvironment& environment, const VideoInfo& source,
                                            const VideoInfo& distorted, const VmafOptions& options,
                                            const std::optional<CropBox>& source_crop,
                                            const std::optional<CropBox>& distorted_crop, const GpuPlan& plan,
                                            double fps, long long total_frames, const HardwareDecoding& decoding,
                                            const ProgressCallback& on_progress, const StatusCallback& on_status,
                                            const CancelToken& cancel) {
    logger.info(metric_names(plan) + " in the app: " + cpu_library_build() + ", " + std::to_string(plan.threads) +
                " threads");
    const std::vector<std::string> keys = plan_keys(plan);
    const bool beside = options.compute_xpsnr && std::find(keys.begin(), keys.end(), "xpsnr") == keys.end();
    SlowerProgress progress(on_progress, beside ? 2 : 1);
    std::optional<ScoresBeside> xpsnr;
    if (beside) {
        VmafOptions xpsnr_options = options;
        xpsnr_options.extra_features.clear();  // only XPSNR: PSNR and SSIM would be scored there again
        xpsnr.emplace(
            [&, xpsnr_options, xpsnr_progress = progress.part(1)](const CancelToken& stop) {
                return execute_run(environment, source, distorted, xpsnr_options, source_crop, distorted_crop, "",
                                   fps, total_frames, decoding, xpsnr_progress, nullptr, stop);
            },
            cancel);
    }
    VmafOptions scored = options;
    if (beside) {
        scored.compute_xpsnr = false;
    } else if (options.extra_features.empty()) {  // XPSNR alone
        scored.n_subsample = 1;
    }
    std::optional<ParsedScores> frames;
    try {
        frames = isolation_program()
                     ? score_plan_apart("cpu-metrics", environment, source, distorted, scored, source_crop,
                                        distorted_crop, plan, fps, total_frames, decoding, progress.part(0),
                                        on_status, cancel)
                     : score_plan(environment, source, distorted, scored, source_crop, distorted_crop, plan, fps,
                                  total_frames, decoding, progress.part(0), on_status, cancel);
    } catch (const Cancelled&) {
        xpsnr.reset();  // stopped and waited for
        throw;
    } catch (const VideoSizeChanged&) {
        xpsnr.reset();
        throw;  // the run is made again, the video marked (run_vmaf)
    } catch (const FilmGrainRefusal&) {
        xpsnr.reset();
        throw;  // final: FFmpeg gets its pictures from the same decoder (grain_feed.h)
    } catch (const std::exception& error) {
        xpsnr.reset();
        logger.error(metric_names(plan) + " in the app failed; FFmpeg calculates them: " + error.what());
        if (on_status) {
            on_status(
                {metric_names(plan) + " in the app failed (" + error.what() + "); calculating them with FFmpeg…"});
        }
        return std::nullopt;
    }
    if (xpsnr) {
        xpsnr->join();
    }
    cancel.throw_if_cancelled();
    if (xpsnr) {
        if (const std::optional<ParsedScores> found = xpsnr->result()) {
            merge_xpsnr(*frames, *found);
        }
    }
    return frames;
}

// The VMAF model a run uses, from the size its frames are compared at, which
// is known only once black bars are: Auto's model comes from it, while an
// explicit or custom choice is the user's. VMAF v1's model is resolved into
// `options` the same way.
std::string resolve_run_models(VmafOptions& options, int width, int height) {
    if (options.compute_vmaf_v1) {
        options.model_v1 = resolve_v1_model(options, width, height);
    }
    if (!options.compute_vmaf) {
        return {};
    }
    return options.model_choice != auto_model_choice ? options.model : model_for_resolution(width, height);
}

}  // namespace

std::string fallback_note(const std::string& status_text) {
    constexpr std::string_view ellipsis = "\xE2\x80\xA6";
    return status_text.ends_with(ellipsis) ? status_text.substr(0, status_text.size() - ellipsis.size())
                                           : status_text;
}

RunStatus decoding_status(const std::string& brief, const HardwareDecoding& decoding, StatusKind kind,
                          std::string_view ending) {
    return {brief + " (GPU decode: " + decoding.describe() + ")" + std::string(ending), kind, brief, decoding};
}

nlohmann::json status_json(const RunStatus& status) {
    nlohmann::json json{{"text", status.text}, {"kind", static_cast<int>(status.kind)}, {"brief", status.brief}};
    if (status.decoding) {
        json["decoding"] = {{"source", hwaccel_json(status.decoding->source)},
                            {"distorted", hwaccel_json(status.decoding->distorted)}};
    }
    return json;
}

RunStatus status_from_json(const nlohmann::json& json) {
    RunStatus status{json.at("text").get<std::string>(), static_cast<StatusKind>(json.at("kind").get<int>()),
                     json.at("brief").get<std::string>()};
    if (const auto decoding = json.find("decoding"); decoding != json.end()) {
        status.decoding =
            HardwareDecoding{hwaccel_from(decoding->at("source")), hwaccel_from(decoding->at("distorted"))};
    }
    return status;
}

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
    if (const std::optional<std::string> problem = shape_problem(source, distorted, source_crop, distorted_crop)) {
        throw VmafRunError(*problem);
    }
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

bool refused_frame_write(int exit_code, std::string_view ffmpeg_errors) {
    // EINVAL as ChildProcess::wait gives FFmpeg's exit code on Windows
    // (0xFFFFFFEA): a write that Windows refused for want of resources
    // (ERROR_NO_SYSTEM_RESOURCES, 1450) reaches FFmpeg as EINVAL.
    constexpr int einval_exit_code = -22;
    // What FFmpeg writes as it closes an output whose writes failed (each of
    // the 39 refused writes logged under load had both lines).
    return exit_code == einval_exit_code &&
           (ffmpeg_errors.find("Error writing trailer: Invalid argument") != std::string_view::npos ||
            ffmpeg_errors.find("Error closing file: Invalid argument") != std::string_view::npos);
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

bool cpu_metrics_in_app() {
    std::error_code ignored;
    return ascii_lower(platform::environment_variable(cpu_metrics_variable).value_or("")) != "ffmpeg" &&
           std::filesystem::is_regular_file(libvmaf_folder() / platform::shared_library_name("libvmaf"), ignored);
}

bool is_cpu_metric(std::string_view key) { return key == "psnr" || key == "ssim" || key == "xpsnr"; }

bool shares_decode(const VideoInfo& source, const VideoInfo& test, const VmafOptions& options,
                   const DecodingCapabilities& capabilities) {
    if (!cpu_metrics_in_app()) {
        return false;
    }
    if (!options.gpu_decode) {
        return true;
    }
    for (const VideoInfo* video : {&source, &test}) {
        if (only_the_app_adds(effective_film_grain(*video))) {
            return true;
        }
    }
    const HardwareDecoding decoding = plan_hardware_decoding(capabilities, options.gpu_vendor, source, test);
    return !decoding.source || !decoding.distorted;
}

std::string build_filtergraph(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                              const HardwareDecoding& decoding, const std::string& log_name, const std::string& model,
                              const std::optional<std::string>& xpsnr_log_name, unsigned cores) {
    // XPSNR beside libvmaf's metrics: each in a graph of its own, which
    // FFmpeg runs on a thread of its own (44 fps at 4K, against 31 chained).
    const bool split = options.compute_xpsnr && xpsnr_log_name &&
                       (uses_vmaf_model(options) || !options.extra_features.empty());
    VideoChains chains = video_chains(source, distorted, options, source_crop, distorted_crop, decoding, split);
    // Both videos' frames with the same colour tags: where they differ,
    // FFmpeg 9 converts one video's colours to the other's before a filter
    // that takes both (for an untagged test video and a BT.2020 source,
    // PSNR 43.7 rather than 52.1), and libvmaf and xpsnr compare samples
    // as they are. The ranges are the same by now (video_chains).
    const std::string same_tags =
        std::string("setparams=colorspace=unknown:range=") + (source.full_range() ? "pc" : "tv");
    chains.test.push_back(same_tags);
    chains.source.push_back(same_tags);
    const std::string stream = std::string(video_stream);
    if (split) {
        // The graphs decode nothing themselves: FFmpeg gives every graph
        // that reads an input the same decoded frames.
        const auto both = [&](const std::string& suffix) {
            return "[0:" + stream + "]" + join(chains.test, ",") + "[main_" + suffix + "];[1:" + stream + "]" +
                   join(chains.source, ",") + "[ref_" + suffix + "]";
        };
        return both("v") + ";[main_v][ref_v]libvmaf=" + join(libvmaf_options(options, log_name, model, cores), ":") +
               "[graph0]" + graph_separator + both("x") + ";" + xpsnr_filter("ref_x", "main_x", *xpsnr_log_name) +
               "[graph1]";
    }
    return "[0:" + stream + "]" + join(chains.test, ",") + "[main];[1:" + stream + "]" + join(chains.source, ",") +
           "[ref];" + libvmaf_stage(options, log_name, model, xpsnr_log_name, cores);
}

std::vector<std::string> build_ffmpeg_command(const std::string& ffmpeg, const std::filesystem::path& distorted,
                                              const std::filesystem::path& source, const std::string& filtergraph,
                                              const HardwareDecoding& decoding, double duration_limit,
                                              const InputOptions& inputs) {
    const bool several_graphs = filtergraph.find(graph_separator) != std::string::npos;
    const auto decode = [several_graphs](const std::optional<std::string>& hwaccel) {
        return several_graphs ? split_hwaccel_args(hwaccel) : hwaccel_args(hwaccel);
    };
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-y"};
    for (const auto& [hwaccel, path, own, fed] :
         {std::tuple{decoding.distorted, distorted, &inputs.distorted, &inputs.distorted_fed},
          std::tuple{decoding.source, source, &inputs.source, &inputs.source_fed}}) {
        if (*fed) {
            command.insert(command.end(), (*fed)->begin(), (*fed)->end());
            continue;
        }
        const std::vector<std::string> options = decode(hwaccel);
        command.insert(command.end(), options.begin(), options.end());
        command.insert(command.end(), own->begin(), own->end());
        command.insert(command.end(), {"-i", input_path(path)});
    }
    const std::vector<std::string> outputs = output_args(filtergraph, duration_limit);
    command.insert(command.end(), outputs.begin(), outputs.end());
    return command;
}

std::string build_resample_filtergraph(const VideoInfo& source, const VmafOptions& options,
                                       const std::optional<CropBox>& source_crop,
                                       const std::optional<std::string>& hwaccel, const std::string& log_name,
                                       const std::string& model, const std::optional<std::string>& xpsnr_log_name,
                                       unsigned cores) {
    if (!options.resample_test) {
        throw std::invalid_argument("not a resolution test");
    }
    const bool cropped = source_crop && !source_crop->is_noop(source.width, source.height);
    const int original_w = cropped ? source_crop->w : source.width;
    const int original_h = cropped ? source_crop->h : source.height;
    const int down_w = options.resample_test->width;
    const int down_h = resample_down_height(down_w, {original_w, original_h});
    // One input, so the analysis format comes from the source alone.
    const std::string analysis_format = analysis_pix_fmt({source.pix_fmt});

    std::vector<std::string> base;
    if (hwaccel) {
        // As in build_filtergraph: hwdownload can only give the hardware
        // surface's own format, not the analysis format.
        base.emplace_back("hwdownload");
        base.push_back("format=" + hw_native_format(source.pix_fmt));
    }
    if (cropped) {
        base.push_back(source_crop->as_filter());  // before the conversion, as build_filtergraph's
    }
    base.push_back("format=" + analysis_format);
    const std::string numbering = join(pairing_filters(frame_rate(source)), ",");
    const std::string scale_flags = ":flags=" + options.scale_algorithm;
    return join({"[0:" + std::string(video_stream) + "]" + join(base, ",") + "[base]",
                   "[base]split=2[ref_src][dist_src]", "[ref_src]" + numbering + "[ref]",
                   "[dist_src]scale=" + std::to_string(down_w) + ":" + std::to_string(down_h) + scale_flags +
                       ",scale=" + std::to_string(original_w) + ":" + std::to_string(original_h) + scale_flags + "," +
                       numbering + "[main]",
                   libvmaf_stage(options, log_name, model, xpsnr_log_name, cores)},
                  ";");
}

std::vector<std::string> build_resample_command(const std::string& ffmpeg, const std::filesystem::path& source,
                                                const std::string& filtergraph,
                                                const std::optional<std::string>& hwaccel, double duration_limit,
                                                const std::vector<std::string>& source_options,
                                                const std::optional<std::vector<std::string>>& source_fed) {
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-y"};
    if (source_fed) {
        command.insert(command.end(), source_fed->begin(), source_fed->end());
    } else {
        const std::vector<std::string> decode = hwaccel_args(hwaccel);
        command.insert(command.end(), decode.begin(), decode.end());
        command.insert(command.end(), source_options.begin(), source_options.end());
        command.insert(command.end(), {"-i", input_path(source)});
    }
    const std::vector<std::string> outputs = output_args(filtergraph, duration_limit);
    command.insert(command.end(), outputs.begin(), outputs.end());
    return command;
}

std::filesystem::path synthetic_resample_distorted_path(const std::filesystem::path& source,
                                                        const ResampleTarget& target) {
    std::filesystem::path name = source.stem();
    name += " [downscale-" + target.label + "-upscale]";
    name += source.extension();
    return source.parent_path() / name;
}

nlohmann::json scoring_task(const nlohmann::json& request, const std::filesystem::path& folder) {
    RunEnvironment environment;
    environment.ffmpeg = request.at("ffmpeg").get<std::string>();
    environment.ffprobe = request.at("ffprobe").get<std::string>();
    const nlohmann::json& decoding = request.at("decoding");
    const ParsedScores scores = score_plan(
        environment, video_from(request.at("source")), video_from(request.at("distorted")),
        options_from(request.at("options")), crop_from(request.at("source_crop")),
        crop_from(request.at("distorted_crop")), plan_from(request.at("plan")), request.at("fps").get<double>(),
        request.at("total_frames").get<long long>(),
        HardwareDecoding{hwaccel_from(decoding.at("source")), hwaccel_from(decoding.at("distorted"))},
        [](long long done, long long whole, double fps) {
            send_isolated_message({{"progress", {done, whole, fps}}});
        },
        [](const RunStatus& status) { send_isolated_message({{"status", status_json(status)}}); },
        CancelToken());  // the parent cancels by ending this process
    std::ofstream file(folder / scores_file, std::ios::binary);
    file.write(reinterpret_cast<const char*>(scores.frame.data()),
               static_cast<std::streamsize>(scores.frame.size() * sizeof(int)));
    for (const std::string& key : scores.order) {
        const std::vector<float>& values = scores.metrics.at(key);
        file.write(reinterpret_cast<const char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    if (!file) {
        throw std::runtime_error("The scores could not be written.");
    }
    return {{"frames", scores.frame.size()}, {"order", scores.order}};
}

std::pair<std::string, std::string> build_stream_graphs(const VideoInfo& source, const VideoInfo& distorted,
                                                        const VmafOptions& options,
                                                        const std::optional<CropBox>& source_crop,
                                                        const std::optional<CropBox>& distorted_crop,
                                                        const HardwareDecoding& decoding) {
    const VideoChains chains = video_chains(source, distorted, options, source_crop, distorted_crop, decoding, false);
    const std::string input = "[0:" + std::string(video_stream) + "]";  // each FFmpeg has its video alone
    if (by_timestamp()) {
        // A copy of the frames, which costs nothing, for FFmpeg to list their
        // timestamps (timed_gpu_output_args).
        return {input + join(chains.test, ",") + ",split=2[vmaf_dist][vmaf_dist_ts]",
                input + join(chains.source, ",") + ",split=2[vmaf_ref][vmaf_ref_ts]"};
    }
    return {input + join(chains.test, ",") + "[vmaf_dist]", input + join(chains.source, ",") + "[vmaf_ref]"};
}

std::vector<std::vector<std::string>> build_stream_commands(
    const std::string& ffmpeg, const std::filesystem::path& distorted, const std::filesystem::path& source,
    const std::pair<std::string, std::string>& graphs, const HardwareDecoding& decoding,
    const std::pair<std::vector<std::string>, std::vector<std::string>>& outputs, const InputOptions& inputs) {
    const auto command = [&ffmpeg](const std::filesystem::path& path, const std::optional<std::string>& hwaccel,
                                   const std::vector<std::string>& input,
                                   const std::optional<std::vector<std::string>>& fed, const std::string& graph,
                                   const std::vector<std::string>& output) {
        std::vector<std::string> made{ffmpeg, "-nostdin", "-hide_banner", "-y"};
        if (fed) {
            made.insert(made.end(), fed->begin(), fed->end());
        } else {
            const std::vector<std::string> decode = hwaccel_args(hwaccel);
            made.insert(made.end(), decode.begin(), decode.end());
            made.insert(made.end(), input.begin(), input.end());
            made.insert(made.end(), {"-i", input_path(path)});
        }
        made.insert(made.end(), {"-lavfi", graph, "-progress", "pipe:1", "-nostats"});
        made.insert(made.end(), output.begin(), output.end());
        return made;
    };
    return {command(distorted, decoding.distorted, inputs.distorted, inputs.distorted_fed, graphs.first,
                    outputs.first),
            command(source, decoding.source, inputs.source, inputs.source_fed, graphs.second, outputs.second)};
}

namespace {

// The crops of a run: none, or (CropMode::automatic) the black bars found in
// both videos, decoded the way the run will (`decoding`), as the picture
// both show.
std::pair<std::optional<CropBox>, std::optional<CropBox>> run_crops(const RunEnvironment& environment,
                                                                    const VideoInfo& source, const VideoInfo& distorted,
                                                                    const VmafOptions& options,
                                                                    const HardwareDecoding& decoding,
                                                                    const StatusCallback& on_status,
                                                                    const CancelToken& cancel) {
    if (options.crop_mode != CropMode::automatic) {
        return {};
    }
    // Said, or the run line would keep the step before: a long video's
    // detection takes a while.
    if (on_status) {
        on_status({"Detecting black bars in source and distorted..."});
    }
    const auto [source_box, distorted_box] =
        detect_pair(environment.ffmpeg, source, distorted, cancel, decoding.source, decoding.distorted,
                    options.scale_algorithm);
    return common_picture(source, distorted, source_box, distorted_box);
}

// The VMAF models the GPU scores (`model` VMAF's, at `depth` bits and
// `width` x `height`). VMAF is scored on the GPU only when VMAF, NEG and
// VMAF v1 are all `options` ask for: the command line puts them in a run of
// their own, apart from FFmpeg's other metrics, and a run that scores both
// asks for its GPU's part alone (score_gpu_and_cpu). None when one is
// calculated on the CPU: FFmpeg's libvmaf calculates them all together. A
// check of the GPU it asks for again (forget_failed_gpu_vmaf_probe) runs
// under the run's `cancel`.
GpuModels models_on_gpu(const VmafOptions& options, const std::string& model, int depth, int width, int height,
                        const CancelToken& cancel) {
    const std::vector<std::string> requested = options.requested_metrics();
    if (!std::all_of(requested.begin(), requested.end(), [](const std::string& key) {
            return key == "vmaf" || key == "vmaf_neg" || key == "vmaf_v1";
        })) {
        return {};
    }
    GpuModels gpu = scores_on_gpu(options.compute_vmaf, options.compute_vmaf_neg, model, options.vmaf_on_gpu, depth,
                                  std::pair(width, height), cancel, options.compute_vmaf_v1, options.model_v1);
    if (gpu.size() != requested.size()) {
        gpu.clear();
    }
    return gpu;
}

// Where a run's `key` scores come from: FFmpeg (`ffmpeg_version`), with
// `model` VMAF's model; or the bundled libvmaf, where it calculated them, on
// the CPU (`cpu_keys`) or the GPU (`gpu_keys`, with `gpu_backend`).
MetricProvenance run_provenance(const std::string& key, const std::string& model, const VmafOptions& options,
                                const std::string& ffmpeg_version, const std::set<std::string, std::less<>>& cpu_keys,
                                const std::set<std::string, std::less<>>& gpu_keys, const std::string& gpu_backend) {
    // Each parameter's JSON text, as every provenance keeps them
    // (results.h): the saved scores write them out as JSON.
    std::map<std::string, std::string> parameters;
    if (key == "vmaf_neg") {
        parameters["model"] = nlohmann::json("version=vmaf_v0.6.1neg").dump();
    } else if (key == "vmaf") {
        parameters["model"] = nlohmann::json(model).dump();
    } else if (key == "vmaf_v1") {
        parameters["model"] = nlohmann::json(options.model_v1).dump();
    }
    MetricProvenance provenance = current_ffmpeg_provenance(key, ffmpeg_version, std::move(parameters));
    if (cpu_keys.contains(key)) {
        // FFmpeg's libvmaf's scores and request, calculated by the
        // bundled build's own extractors.
        provenance.implementation = "libvmaf";
        provenance.implementation_version = cpu_library_build();
    }
    if (gpu_keys.contains(key)) {
        // The same request as FFmpeg's libvmaf (the scores agree to
        // within 4e-5), made on the GPU by the bundled build: by its
        // CUDA code or the Vulkan port of it, and VMAF v1 with Vulkan on
        // every GPU.
        const std::string backend = key == "vmaf_v1" ? "vulkan" : gpu_backend;
        provenance.implementation = "libvmaf/" + backend;
        provenance.compute_backend = "gpu";
        provenance.implementation_version = key == "vmaf_v1"        ? v1_library_build()
                                            : backend == "vulkan" ? vulkan_library_build()
                                                                  : cuda_library_build();
    }
    return provenance;
}

// What every part of a run scores: the videos as they are compared (their
// crops, and the size and depth compared at), and how they are decoded.
struct RunFrames {
    const RunEnvironment& environment;
    const VideoInfo& source;
    const VideoInfo& distorted;
    const std::optional<CropBox>& source_crop;
    const std::optional<CropBox>& distorted_crop;
    int width = 0;
    int height = 0;
    int depth = 8;
    long long total_frames = 0;
    const HardwareDecoding& decoding;
    const CancelToken& cancel;
};

// A run's scores, in parts with frames of their own (in a run of the GPU's
// metrics and the CPU's, the CPU's first, as two tasks' are merged), and what
// calculated each metric (run_provenance).
struct RunScores {
    std::vector<ParsedScores> parts;
    std::set<std::string, std::less<>> cpu_keys;  // by the bundled libvmaf on the CPU
    std::set<std::string, std::less<>> gpu_keys;  // by the bundled libvmaf on the GPU
    std::string gpu_backend = "cuda";
};

void insert_keys(std::set<std::string, std::less<>>& keys, const GpuPlan& plan) {
    for (const auto& [key, model] : plan.models) {
        keys.insert(key);
    }
}

// The plan for the run's VMAF, NEG and VMAF v1 on the GPU, or nothing when the
// CPU calculates them (models_on_gpu). VMAF v0.6.1 and NEG are scored by what
// the GPU's check found: the Vulkan engine on its GPU, or libvmaf's CUDA code.
// VMAF v1 runs on Vulkan, on the best GPU unless the engine scores VMAF v0.6.1
// here; alone, where neither scores it, it goes as with CUDA (the Python
// app's default backend).
std::optional<GpuPlan> gpu_plan(const RunFrames& run, const VmafOptions& options, const std::string& model) {
    const GpuModels gpu = models_on_gpu(options, model, run.depth, run.width, run.height, run.cancel);
    if (gpu.empty()) {
        return std::nullopt;
    }
    const GpuVmafBackend backend = gpu_vmaf_backend(run.cancel);
    GpuPlan plan{gpu, run.width, run.height, run.depth};
    plan.backend = backend.backend == "vulkan" ? "vulkan" : "cuda";
    plan.device = backend.backend == "vulkan" && backend.device ? *backend.device : -1;
    return plan;
}

// The plan for the run's PSNR, SSIM and XPSNR in the app (cpu_metrics_plan).
std::optional<GpuPlan> cpu_plan(const RunFrames& run, const VmafOptions& options) {
    return cpu_metrics_plan(options, std::pair(run.width, run.height), run.source, run.distorted, run.decoding,
                            run.source_crop, run.distorted_crop, run.environment.cores, run.environment.ffmpeg);
}

// The scores of a run in one part: VMAF, NEG and VMAF v1 on the GPU when they
// are all it scores and the GPU scores them; PSNR, SSIM and XPSNR in the app
// when they are all it scores; otherwise, or when either fails, FFmpeg's.
RunScores score_alone(const RunFrames& run, const VmafOptions& options, const std::string& model,
                      const ProgressCallback& on_progress, const StatusCallback& on_status) {
    RunScores scores;
    std::optional<ParsedScores> scored;
    if (const std::optional<GpuPlan> plan = gpu_plan(run, options, model)) {
        scores.gpu_backend = plan->backend;
        scored = run_on_gpu(run.environment, run.source, run.distorted, options, run.source_crop, run.distorted_crop,
                            *plan, run.distorted.fps, run.total_frames, run.decoding, on_progress, on_status,
                            run.cancel);
        if (scored) {
            insert_keys(scores.gpu_keys, *plan);
        }
    }
    if (!scored) {
        if (const std::optional<GpuPlan> cpu = cpu_plan(run, options)) {
            scored = run_cpu_metrics(run.environment, run.source, run.distorted, options, run.source_crop,
                                     run.distorted_crop, *cpu, run.distorted.fps, run.total_frames, run.decoding,
                                     on_progress, on_status, run.cancel);
            if (scored) {
                insert_keys(scores.cpu_keys, *cpu);
            }
        }
    }
    if (!scored) {
        scored = execute_run(run.environment, run.source, run.distorted, options, run.source_crop, run.distorted_crop,
                             model, run.distorted.fps, run.total_frames, run.decoding, on_progress, on_status,
                             run.cancel);
    }
    scores.parts.push_back(std::move(*scored));
    return scores;
}

// `options` asking for its CPU metrics only (is_cpu_metric), or for its
// others only.
VmafOptions part_of(const VmafOptions& options, bool cpu_metrics) {
    VmafOptions part = options;
    for (const std::string& key : options.requested_metrics()) {
        part.set_metric_enabled(key, is_cpu_metric(key) == cpu_metrics);
    }
    return part;
}

// Whether the GPU's pass (`gpu`) can score the run's CPU metrics (`cpu`)
// beside its own, from the frames it decodes, exactly as their own run
// would: both videos decoded in this process (as score_plan decodes them) by
// the software decoder or NVIDIA's, whose pictures the CPU's scorer reads the
// same way for either pass (the other GPUs' decoders hand pictures over to
// the GPU's pass another way); every CPU metric of the run in `cpu`, none left
// to an FFmpeg of its own (run_cpu_metrics's XPSNR beside); scored for the
// GPU's frames, which XPSNR alone with a frame subsample is not
// (run_cpu_metrics scores it for every frame); and not to libvmaf's CUDA code
// without VMAF v1 (score_decoded_cuda).
bool shares_pass(const RunFrames& run, const GpuPlan& gpu, const GpuPlan& cpu, const VmafOptions& cpu_options) {
    const std::optional<std::pair<std::string, std::string>> decoders =
        run.source.scaled_to || run.distorted.scaled_to ? std::nullopt : decoders_here(run.decoding);
    const auto reads_alike = [](const std::string& decoder) {
        return decoder == software_decoder || decoder == "nvidia";
    };
    if (!decoders || !reads_alike(decoders->first) || !reads_alike(decoders->second)) {
        return false;
    }
    const std::vector<std::string> gpu_keys = plan_keys(gpu);
    if (gpu.backend == "cuda" && std::find(gpu_keys.begin(), gpu_keys.end(), "vmaf_v1") == gpu_keys.end()) {
        return false;
    }
    const std::vector<std::string> keys = plan_keys(cpu);
    if (cpu_options.compute_xpsnr && std::find(keys.begin(), keys.end(), "xpsnr") == keys.end()) {
        return false;
    }
    return !(keys == std::vector<std::string>{"xpsnr"} && cpu_options.n_subsample > 1);
}

// Moves the scores of `keys` out of `scores` into scores of their own, for the
// same frames.
ParsedScores take_metrics(ParsedScores& scores, const std::vector<std::string>& keys) {
    ParsedScores taken;
    taken.frame = scores.frame;
    taken.time = scores.time;
    std::vector<std::string> kept;
    for (const std::string& key : scores.order) {
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
            kept.push_back(key);
            continue;
        }
        taken.metrics[key] = std::move(scores.metrics.at(key));
        scores.metrics.erase(key);
        taken.order.push_back(key);
    }
    scores.order = std::move(kept);
    return taken;
}

// The run's GPU metrics (`gpu`) with its CPU metrics (`cpu`) beside them in
// one GPU pass, from one decode of each video (shares_pass). Throws what
// score_on_gpu throws.
RunScores score_shared(const RunFrames& run, const VmafOptions& options, const GpuPlan& gpu, const GpuPlan& cpu,
                       const ProgressCallback& on_progress, const StatusCallback& on_status) {
    GpuPlan plan = gpu;
    plan.beside = plan_keys(cpu);
    plan.threads = cpu.threads;
    plan.frame_rate = cpu.frame_rate;
    ParsedScores scored = score_on_gpu(run.environment, run.source, run.distorted, options, run.source_crop,
                                       run.distorted_crop, plan, run.distorted.fps, run.total_frames, run.decoding,
                                       on_progress, on_status, run.cancel);
    RunScores scores;
    scores.parts.push_back(take_metrics(scored, plan.beside));
    scores.parts.push_back(std::move(scored));
    insert_keys(scores.cpu_keys, cpu);
    insert_keys(scores.gpu_keys, gpu);
    scores.gpu_backend = gpu.backend;
    return scores;
}

// The run's GPU metrics (`gpu`, with `gpu_options`) and its CPU metrics (with
// `cpu_options`: in the app by `cpu`, or else by FFmpeg's filters), each from
// a decode of its own and at the same time, as two tasks score them: the
// CPU's on a thread beside. Each falls back as its task would: VMAF to
// FFmpeg's libvmaf on the CPU, PSNR, SSIM and XPSNR to FFmpeg's filters. The
// statuses are the GPU's.
RunScores score_apart(const RunFrames& run, const VmafOptions& gpu_options, const VmafOptions& cpu_options,
                      const GpuPlan& gpu, const std::optional<GpuPlan>& cpu, const std::string& model,
                      const ProgressCallback& on_progress, const StatusCallback& on_status) {
    SlowerProgress progress(on_progress, 2);
    bool in_app = false;  // set on the CPU's thread, read once it has ended
    ScoresBeside cpu_part(
        [&, cpu_progress = progress.part(1)](const CancelToken& stop) {
            if (cpu) {
                if (std::optional<ParsedScores> scored = run_cpu_metrics(
                        run.environment, run.source, run.distorted, cpu_options, run.source_crop, run.distorted_crop,
                        *cpu, run.distorted.fps, run.total_frames, run.decoding, cpu_progress, nullptr, stop)) {
                    in_app = true;
                    return std::move(*scored);
                }
            }
            return execute_run(run.environment, run.source, run.distorted, cpu_options, run.source_crop,
                               run.distorted_crop, "", run.distorted.fps, run.total_frames, run.decoding, cpu_progress,
                               nullptr, stop);
        },
        run.cancel);
    RunScores scores;
    scores.gpu_backend = gpu.backend;
    std::optional<ParsedScores> gpu_scores =
        run_on_gpu(run.environment, run.source, run.distorted, gpu_options, run.source_crop, run.distorted_crop, gpu,
                   run.distorted.fps, run.total_frames, run.decoding, progress.part(0), on_status, run.cancel);
    if (gpu_scores) {
        insert_keys(scores.gpu_keys, gpu);
    } else {
        gpu_scores = execute_run(run.environment, run.source, run.distorted, gpu_options, run.source_crop,
                                 run.distorted_crop, model, run.distorted.fps, run.total_frames, run.decoding,
                                 progress.part(0), on_status, run.cancel);
    }
    cpu_part.join();
    run.cancel.throw_if_cancelled();
    std::optional<ParsedScores> cpu_scores = cpu_part.result();
    if (!cpu_scores) {
        throw VmafRunError("PSNR, SSIM and XPSNR gave no scores");
    }
    if (in_app) {
        insert_keys(scores.cpu_keys, *cpu);
    }
    scores.parts.push_back(std::move(*cpu_scores));
    scores.parts.push_back(std::move(*gpu_scores));
    return scores;
}

// The scores of a run of VMAF, NEG or VMAF v1 with PSNR, SSIM or XPSNR whose
// GPU scores the former (a task plan_tasks made, with share_decode, for
// videos the app decodes in software), or nothing for any other run
// (score_alone). Where it can (shares_pass), one GPU pass scores both from one
// decode of each video. Otherwise, or when that pass fails, the two are
// scored apart, at the same time, as two tasks would score them
// (score_apart).
std::optional<RunScores> score_gpu_and_cpu(const RunFrames& run, const VmafOptions& options, const std::string& model,
                                           const ProgressCallback& on_progress, const StatusCallback& on_status) {
    const std::vector<std::string> requested = options.requested_metrics();
    const auto cpu_metric = [](const std::string& key) { return is_cpu_metric(key); };
    if (std::none_of(requested.begin(), requested.end(), cpu_metric) ||
        std::all_of(requested.begin(), requested.end(), cpu_metric)) {
        return std::nullopt;
    }
    const VmafOptions gpu_options = part_of(options, false);
    const std::optional<GpuPlan> gpu = gpu_plan(run, gpu_options, model);
    if (!gpu) {
        return std::nullopt;  // FFmpeg's libvmaf calculates them all on the CPU
    }
    const VmafOptions cpu_options = part_of(options, true);
    const std::optional<GpuPlan> cpu = cpu_plan(run, cpu_options);
    if (cpu && shares_pass(run, *gpu, *cpu, cpu_options)) {
        try {
            return score_shared(run, options, *gpu, *cpu, on_progress, on_status);
        } catch (const Cancelled&) {
            throw;
        } catch (const VideoSizeChanged&) {
            throw;  // the run is made again, the video marked (run_vmaf)
        } catch (const FilmGrainRefusal&) {
            throw;  // final: every route decodes it with the same decoder (grain_feed.h)
        } catch (const std::exception& error) {
            logger.warning("VMAF on the GPU with " + metric_names(*cpu) +
                           " beside it failed; scoring them apart: " + error.what());
        }
    }
    return score_apart(run, gpu_options, cpu_options, *gpu, cpu, model, on_progress, on_status);
}

// run_vmaf's run of the pair as it is marked (size_changes.h). Throws
// VideoSizeChanged when FFmpeg rebuilt its filters for one of them.
ComparisonResult run_vmaf_once(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                               VmafOptions options, const ProgressCallback& on_progress,
                               const StatusCallback& on_status, const CancelToken& cancel,
                               const std::optional<std::filesystem::path>& result_distorted_path) {
    validate_video_pair(source, distorted, options);
    // The decode plan comes before black-bar detection, so the detection
    // decodes the way the run will.
    HardwareDecoding decoding;
    if (options.gpu_decode) {
        decoding = plan_hardware_decoding(environment.decoding, options.gpu_vendor, source, distorted);
    }
    const auto [source_crop, distorted_crop] = run_crops(environment, source, distorted, options, decoding, on_status, cancel);
    // After cropping: removing a letterbox is what makes a letterboxed source
    // and a cropped encode the same shape.
    validate_display_geometry(source, distorted, source_crop, distorted_crop);

    const auto [width, height] = analysis_dimensions(source, distorted, options, source_crop, distorted_crop);
    const std::string model = resolve_run_models(options, width, height);
    const long long total_frames = estimate_total_frames(distorted, options, &source);

    const int depth = bit_depth(analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt}));
    const RunFrames run{environment, source, distorted, source_crop, distorted_crop, width, height,
                        depth,       total_frames, decoding, cancel};
    std::optional<RunScores> scores = score_gpu_and_cpu(run, options, model, on_progress, on_status);
    if (!scores) {
        scores = score_alone(run, options, model, on_progress, on_status);
    }
    for (const ParsedScores& part : scores->parts) {
        const long long compared = part.frame.empty() ? 0 : static_cast<long long>(part.frame.back()) + 1;
        if (const std::optional<std::string> short_run =
                short_comparison(total_frames, compared, distorted.fps, options.n_subsample)) {
            throw VmafRunError(*short_run);
        }
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
    for (const ParsedScores& part : scores->parts) {
        for (const std::string& key : part.order) {
            result.metric_results.add(FrameMetricResult{
                key, part.frame, part.time, part.metrics.at(key),
                run_provenance(key, model, options, environment.ffmpeg_version, scores->cpu_keys, scores->gpu_keys,
                               scores->gpu_backend)});
        }
    }
    return result;
}

}  // namespace

ComparisonResult run_vmaf(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                          VmafOptions options, const ProgressCallback& on_progress, const StatusCallback& on_status,
                          const CancelToken& cancel,
                          const std::optional<std::filesystem::path>& result_distorted_path) {
    VideoInfo scored_source = source;
    VideoInfo scored_test = distorted;
    SizeChangeMarks marks(scored_source, scored_test);
    while (true) {
        try {
            ComparisonResult result = run_vmaf_once(environment, scored_source, scored_test, options, on_progress,
                                                    on_status, cancel, result_distorted_path);
            marks.finished();
            return result;
        } catch (const VideoSizeChanged& changed) {
            cancel.throw_if_cancelled();  // not made again after a Cancel
            if (!marks.after(changed, options.scale_algorithm)) {
                throw VmafRunError(size_change_unfixed);
            }
        }
    }
}

namespace {

// run_resample_test's run. Throws VideoSizeChanged when FFmpeg rebuilt its
// filters (size_changes.h).
ComparisonResult resample_test_run(const RunEnvironment& environment, const VideoInfo& source, VmafOptions options,
                                   const ProgressCallback& on_progress, const StatusCallback& on_status,
                                   const CancelToken& cancel) {
    if (!options.resample_test) {
        throw std::invalid_argument("not a resolution test");
    }
    // One file: there is no test video to plan GPU decoding for.
    HardwareDecoding decoding;
    if (options.gpu_decode) {
        decoding.source = pick_decode(environment.decoding, options.gpu_vendor, source);
    }
    std::optional<CropBox> source_crop;
    if (options.crop_mode == CropMode::automatic) {
        if (on_status) {
            on_status({"Detecting black bars in source..."});
        }
        source_crop = detect_crop(environment.ffmpeg, source, cancel, decoding.source, options.scale_algorithm);
    }
    // The comparison is at the source's own size, the downscale being undone
    // before it: Auto's VMAF models come from that.
    const auto [width, height] = resample_analysis_dimensions(source, source_crop);
    const std::string model = resolve_run_models(options, width, height);
    const long long total_frames = estimate_total_frames(source, options, nullptr);
    const auto build_command = [&](const HardwareDecoding& plan, const std::string& run_model,
                                   const std::optional<std::string>& xpsnr_log_name, const PairFeeds& feeds) {
        const std::string graph = build_resample_filtergraph(source, options, source_crop, plan.source,
                                                             "vmaf_log.json", run_model, xpsnr_log_name,
                                                             environment.cores);
        if (const PictureFeed* fed = feeds.source()) {
            return build_resample_command(environment.ffmpeg, source.path, graph, plan.source, options.duration_limit,
                                          {}, fed->input_arguments());
        }
        return build_resample_command(environment.ffmpeg, source.path, graph, plan.source, options.duration_limit,
                                      source.decode_options());
    };
    const ParsedScores scores =
        execute_run(build_command, options, model, source.fps, total_frames, decoding, "vmaf_resample_", on_progress,
                    on_status, cancel, environment, &source, nullptr);

    ComparisonResult result;
    result.source = source.path;
    result.distorted = synthetic_resample_distorted_path(source.path, *options.resample_test);
    result.fps = source.fps;
    result.model = model;
    result.source_crop = source_crop;
    result.distorted_crop = source_crop;  // both branches come from the same source
    result.source_info = source;
    result.distorted_info = source;  // after the round trip, it is back at the source's own size
    result.scale_algorithm = options.scale_algorithm;
    result.resample_target = options.resample_test;
    result.compared_frame_count = total_frames;
    result.model_choice = options.model_choice;
    result.model_v1 = options.model_v1;
    if (options.compute_vmaf_v1) {
        result.model_choice_v1 = options.model_choice_v1;
    }
    for (const std::string& key : scores.order) {
        std::map<std::string, std::string> parameters;
        if (key == "vmaf_neg") {
            parameters["model"] = nlohmann::json("version=vmaf_v0.6.1neg").dump();
        } else if (key == "vmaf") {
            parameters["model"] = nlohmann::json(model).dump();
        } else if (key == "vmaf_v1") {
            parameters["model"] = nlohmann::json(options.model_v1).dump();
        }
        result.metric_results.add(
            FrameMetricResult{key, scores.frame, scores.time, scores.metrics.at(key),
                              current_ffmpeg_provenance(key, environment.ffmpeg_version, std::move(parameters))});
    }
    return result;
}

}  // namespace

ComparisonResult run_resample_test(const RunEnvironment& environment, const VideoInfo& source, VmafOptions options,
                                   const ProgressCallback& on_progress, const StatusCallback& on_status,
                                   const CancelToken& cancel) {
    try {
        return resample_test_run(environment, source, std::move(options), on_progress, on_status, cancel);
    } catch (const VideoSizeChanged&) {
        // Its frames are scaled down and up again at the size they come in.
        throw VmafRunError("The video's picture size changes partway through; a resolution round-trip test needs "
                           "a video of one size.");
    }
}

}  // namespace vml
