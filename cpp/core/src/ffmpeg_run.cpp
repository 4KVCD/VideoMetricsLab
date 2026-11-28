#include "vml/ffmpeg_run.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <thread>
#include <variant>

#include "ffmpeg_run_parts.h"
#include "vml/app_log.h"
#include "vml/chroma_vmaf.h"
#include "vml/crop_detect.h"
#include "vml/film_grain_scoring.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_checks.h"
#include "vml/gpu_cpu_metrics.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/grain_feed.h"
#include "vml/isolated.h"
#include "vml/libvmaf_cpu.h"
#include "vml/memory_fit.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/run_errors.h"
#include "vml/run_json.h"
#include "vml/size_changes.h"
#include "vml/temp_files.h"
#include "vml/time_format.h"
#include "vml/vmaf_cuda.h"
#include "vml/vmaf_hdr.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vmaf_vulkan.h"

namespace vml {

namespace {

using namespace run_parts;

// A folder of its own for one run's logs, deleted with it.
TempFolder run_folder(std::string_view prefix) {
    return TempFolder::make<VmafRunError>(prefix, "A folder for the run's logs could not be made.");
}

// When a model= value names a custom file, copies it next to the log under a
// name of its own: FFmpeg's filter graph would take a bracket or comma of the
// user's name for its syntax, and a bundled VMAF v1 model of the same name is
// copied there too (see libvmaf_options).
std::string model_for_run_folder(const std::string& model, const std::filesystem::path& folder) {
    const std::optional<std::string_view> named = model_file_of(model);
    if (!named) {
        return model;
    }
    constexpr std::string_view name = "custom_vmaf_model.json";
    std::filesystem::copy_file(platform::path_from_text(*named), folder / name,
                               std::filesystem::copy_options::overwrite_existing);
    return model_option_for_file(name);
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
// told to `rebuild`, and which inherits `inherited` (its pipes' ends).
platform::ProcessOptions watched_options(const std::filesystem::path& folder,
                                         const std::shared_ptr<FilterRebuild>& rebuild,
                                         std::vector<int> inherited = {}) {
    platform::ProcessOptions options{folder};
    options.on_error_line = rebuild_watcher(rebuild);
    options.inherited_descriptors = std::move(inherited);
    return options;
}

// Runs one FFmpeg command, reporting its progress. Throws Cancelled, and
// VideoSizeChanged when FFmpeg rebuilt its filters (it is ended then: what it
// scores after is not the run's).
FfmpegOutcome run_ffmpeg(const std::vector<std::string>& command, long long total_frames,
                         const ProgressCallback& on_progress, const CancelToken& cancel,
                         const std::filesystem::path& folder) {
    cancel.throw_if_cancelled();  // not one more FFmpeg after a Cancel
    const auto rebuilt = std::make_shared<FilterRebuild>();
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
        throw VideoSizeChanged(std::nullopt, rebuilt->size());
    }
    return FfmpegOutcome{code, std::move(errors)};
}

}  // namespace

FfmpegOutcome run_ffmpeg_pair(const std::vector<std::vector<std::string>>& commands,
                              const std::vector<std::filesystem::path>& videos,
                              const std::function<std::vector<int>(std::size_t)>& take_descriptors,
                              const std::function<bool(std::size_t)>& closed_by_app, long long total_frames,
                              const ProgressCallback& on_progress, const CancelToken& cancel,
                              const std::filesystem::path& folder) {
    cancel.throw_if_cancelled();
    std::vector<std::shared_ptr<FilterRebuild>> rebuilt;
    std::vector<platform::ChildProcess> processes;
    for (std::size_t index = 0; index < commands.size(); ++index) {
        rebuilt.push_back(std::make_shared<FilterRebuild>());
        processes.push_back(platform::ChildProcess::start(
            commands[index], watched_options(folder, rebuilt.back(), take_descriptors(index))));
    }
    const auto any_rebuilt = [&rebuilt] {
        return std::any_of(rebuilt.begin(), rebuilt.end(), [](const auto& rebuild) { return static_cast<bool>(*rebuild); });
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
            throw VideoSizeChanged(videos.at(index), rebuilt[index]->size());
        }
    }
    const auto cause = std::find_if(failed.begin(), failed.end(),
                                    [&closed_by_app](std::size_t index) { return !closed_by_app(index); });
    return outcomes[cause != failed.end() ? *cause : failed.empty() ? 0 : failed.front()];
}

namespace {

// What every part of a run scores: the videos as they are compared (their
// crops, and the size and depth compared at), how many frames, how they are
// decoded, the token that cancels and pauses it, and the place its plan gave
// the metrics that follow VMAF's (none: the GPU's check is asked,
// run_engine_place). Each part's scores are timed at the test video's rate
// (distorted.fps).
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
    std::optional<EnginePlace> engine;

    // The same run for a part on a thread beside it, stopped by a token of its own (ScoresBeside).
    RunFrames with_cancel(const CancelToken& stop) const {
        return {environment, source, distorted, source_crop, distorted_crop, width, height, depth, total_frames,
                decoding, stop, engine};
    }
};

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

// The inputs of a run's two videos: a fed video's pictures (grain_feed.h),
// the other's file.
InputOptions run_inputs(const PairFeeds& feeds, const VideoInfo& source, const VideoInfo& distorted,
                        const RangeSeeks& seeks) {
    InputOptions inputs;
    if (const PictureFeed* fed = feeds.distorted()) {
        inputs.distorted_fed = fed->input_arguments();
    } else {
        inputs.distorted = with_seek(distorted.decode_options(), seeks.distorted);
    }
    if (const PictureFeed* fed = feeds.source()) {
        inputs.source_fed = fed->input_arguments();
    } else {
        inputs.source = with_seek(source.decode_options(), seeks.source);
    }
    return inputs;
}

// `options` with the places FFmpeg starts reading the two videos of their range.
VmafOptions reading_near_range(const VmafOptions& options, const RunEnvironment& environment, const VideoInfo& source,
                               const VideoInfo& distorted, const CancelToken& cancel) {
    const long long decode_start = run_range(options, source).decode_start;
    VmafOptions seeking = options;
    seeking.range_seeks.source =
        ffmpeg_range_seek(source, decode_start, environment.ffmpeg, environment.ffprobe, cancel);
    seeking.range_seeks.distorted =
        ffmpeg_range_seek(distorted, decode_start, environment.ffmpeg, environment.ffprobe, cancel);
    return seeking;
}

// The FFmpeg command of a run for a decode plan, the run's model= value (by
// its bare file name for a file), XPSNR's stats file name, if any, and the
// run's feeds of its videos' pictures.
using CommandBuilder =
    std::function<std::vector<std::string>(const HardwareDecoding& plan, const std::string& run_model,
                                           const std::optional<std::string>& xpsnr_log_name, const PairFeeds& feeds)>;

// Whether FFmpeg's xpsnr filter scores pictures of `bit_depth` rightly. It
// reads a test picture of more than 8 bits from its frame with the rows the
// picture's width apart (vf_xpsnr.c, FFmpeg 9.0 and its master in November
// 2026: s_rec = plane_width) instead of the frame's linesize, so wherever
// FFmpeg pads the rows (a crop narrower than the decoded picture, a picture
// whose width is not a multiple of its alignment) the score is garbage: a
// 10-bit 3348x1608 crop against itself gives 16.8 dB, not infinity. Pictures
// of 8 bits it first copies into rows of their own width.
bool ffmpeg_scores_xpsnr(int bit_depth) { return bit_depth <= 8; }

const char* const xpsnr_refusal =
    "XPSNR of videos of more than 8 bits is calculated by the app alone, which cannot calculate it for this "
    "comparison (FFmpeg's xpsnr filter reads such pictures wrongly at some widths)";

// The CAMBI options of a comparison's test video (CambiInput): its own size
// once cropped (a resolution test's: the size the source is scaled down to),
// its bit depth and whether it is PQ.
CambiInput cambi_input(const VideoInfo& distorted, const std::optional<CropBox>& distorted_crop) {
    const auto [width, height] =
        distorted.resample_via ? *distorted.resample_via : content_size(distorted, distorted_crop);
    return {width, height, bit_depth(distorted.pix_fmt), ascii_lower(distorted.color_transfer) == "smpte2084"};
}

// The metrics of `options` that FFmpeg's filters cannot score for a
// comparison at `bit_depth`, each with why: the app's own (scored_in_app), and XPSNR
// where FFmpeg would score it wrongly (ffmpeg_scores_xpsnr). The app scores
// them, or nobody does.
std::map<std::string, std::string> app_only_metrics(const VmafOptions& options, int bit_depth) {
    std::map<std::string, std::string> refused;
    if (options.compute_xpsnr && !ffmpeg_scores_xpsnr(bit_depth)) {
        refused["xpsnr"] = xpsnr_refusal;
    }
    for (const MetricDefinition& metric : metrics()) {
        const std::string_view key = metric.key;
        if (scored_in_app(key) && options.metric_enabled(key)) {
            const bool yuv = scored_by(key) == "psnr_yuv" || scored_by(key) == "psnr_yuv_mse" ||
                             scored_by(key) == "ssim_yuv";
            refused[std::string(key)] =
                std::string(metric_definition(key).label) +
                " is calculated by the app alone, which cannot calculate it for this comparison (" +
                (yuv ? "the 6:1:1 metrics are the app's own: FFmpeg's filters have no SSIM of the chroma planes)"
                     : "it is scored by the app's own libvmaf, not by FFmpeg's filters)");
        }
    }
    return refused;
}

// `options` without the metrics of `keys` (a map's).
VmafOptions without_metrics(const VmafOptions& options, const std::map<std::string, std::string>& keys) {
    VmafOptions others = options;
    for (const auto& [key, reason] : keys) {
        others.set_metric_enabled(key, false);
    }
    return others;
}

// Thrown where an FFmpeg run would score only metrics FFmpeg cannot score
// (app_only_metrics): each of them with why.
class AppOnlyUnavailable : public VmafRunError {
public:
    explicit AppOnlyUnavailable(const std::map<std::string, std::string>& refused)
        : VmafRunError(refused.begin()->second), refused(refused) {}
    std::map<std::string, std::string> refused;
};

// `run`'s scores of `options`, the FFmpeg run of a comparison at `bit_depth`,
// without the metrics FFmpeg cannot score (app_only_metrics): the run's
// others, those failed in them. Throws AppOnlyUnavailable where they are all
// it scores.
ParsedScores without_app_only(const VmafOptions& options, int bit_depth,
                              const std::function<ParsedScores(const VmafOptions&)>& run) {
    const std::map<std::string, std::string> refused = app_only_metrics(options, bit_depth);
    if (refused.empty()) {
        return run(options);
    }
    const VmafOptions others = without_metrics(options, refused);
    if (others.requested_metrics().empty()) {
        throw AppOnlyUnavailable(refused);
    }
    ParsedScores scores = run(others);
    scores.failed.insert(refused.begin(), refused.end());
    return scores;
}

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
    // A failed attempt can leave a partial log, which would be read as the
    // retry's.
    const auto remove_logs = [&] {
        std::error_code ignored;
        std::filesystem::remove(log_path, ignored);
        if (xpsnr_log_path) {
            std::filesystem::remove(*xpsnr_log_path, ignored);
        }
    };
    bool same_again = false;  // this attempt is the last one's plan again
    for (std::size_t attempt = 0; attempt < ladder.size(); ++attempt) {
        // Before the FFmpeg, which it outlives.
        PairFeeds feeds(source, distorted, environment.ffmpeg, environment.ffprobe, cancel,
                        vvdec_wanted(options.gpu_decode, options.vvc_on_gpu), on_status);
        const HardwareDecoding plan = feeds.decoding(ladder[attempt]);
        if (on_status && !same_again) {
            on_status(attempt == 0 ? decoding_status("Running ffmpeg", plan, StatusKind::starting)
                                   : decoding_status("GPU decode failed, retrying", plan));
        }
        same_again = false;
        const std::vector<std::string> command = build_command(plan, run_model, xpsnr_log_name, feeds);
        logger.info("FFmpeg: " + python_list2cmdline(command));
        try {
            outcome = run_ffmpeg(command, total_frames, on_progress, cancel, folder.path());
        } catch (const platform::ProcessError& error) {
            throw VmafRunError(std::string("FFmpeg could not be started: ") + error.what());
        }
        try {
            feeds.finish();
        } catch (const VvdecFeedFailed& error) {
            // vvdec-vulkan is off now (it cannot fail twice): the same attempt again, VVC decoded by VVdeC
            report_vvdec_feed_failed(error, on_status);
            remove_logs();
            same_again = true;
            --attempt;
            continue;
        }
        if (outcome.exit_code == 0) {
            break;
        }
        log_failed_attempt(outcome, plan, attempt + 1 < ladder.size());
        remove_logs();
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
    if (source != nullptr) {
        keep_range(scores, run_range(options, *source), fps);
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

ParsedScores score_watched(const RunFrames& run, const std::vector<std::string>& keys,
                           MemoryWatch::GpuReader gpu_memory, const ProgressCallback& on_progress,
                           const std::function<ParsedScores(const RunFrames&, const ProgressCallback&)>& score);

// execute_run for a comparison of two videos (build_filtergraph and
// build_ffmpeg_command), without the metrics FFmpeg cannot score
// (without_app_only). Watched for the PC's memory (score_watched): FFmpeg's
// libvmaf on the CPU is the biggest pass of a CPU run. Its GPU memory
// (libvmaf's CUDA code in FFmpeg) is another process's, not watched.
ParsedScores execute_run(const RunFrames& run, const VmafOptions& options, const std::string& model,
                         const ProgressCallback& on_progress, const StatusCallback& on_status) {
    const VideoInfo& source = run.source;
    const VideoInfo& distorted = run.distorted;
    const int depth = bit_depth(analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt}));
    return without_app_only(options, depth, [&](const VmafOptions& scored) {
        return score_watched(
            run, scored.requested_metrics(), nullptr, on_progress,
            [&](const RunFrames& watched, const ProgressCallback& progress) {
                const VmafOptions seeking =
                    reading_near_range(scored, watched.environment, source, distorted, watched.cancel);
                const auto build_command = [&](const HardwareDecoding& plan, const std::string& run_model,
                                               const std::optional<std::string>& xpsnr_log_name,
                                               const PairFeeds& feeds) {
                    const std::string graph =
                        build_filtergraph(source, distorted, seeking, run.source_crop, run.distorted_crop, plan,
                                          "vmaf_log.json", run_model, xpsnr_log_name, run.environment.cores);
                    return build_ffmpeg_command(run.environment.ffmpeg, distorted.path, source.path, graph, plan,
                                                run_inputs(feeds, source, distorted, seeking.range_seeks));
                };
                return execute_run(build_command, scored, model, distorted.fps, run.total_frames, run.decoding,
                                   "vmaf_run_", progress, on_status, watched.cancel, run.environment, &source,
                                   &distorted);
            });
    });
}

// The GPU's scores as the run's (the Python app's _gpu_frame_scores): each
// kept as a float, as the Python app kept FFmpeg's, the frames timed at
// `fps`, with a ranged run's frames numbered as in the whole videos and only
// those in `range` kept (keep_range). Throws when no frame was scored or a
// requested metric has none.
ParsedScores gpu_scores(const PairScores& scored, double fps, const VmafOptions& options, const FrameRange& range) {
    if (scored.frames.empty()) {
        throw LibvmafError("GPU VMAF scored no frames");
    }
    ParsedScores scores;
    scores.frame.assign(scored.frames.begin(), scored.frames.end());
    scores.time = times_of(scores.frame, fps);
    scores.features_on_gpu = scored.features_on_gpu;
    scores.exact = scored.exact;
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
    keep_range(scores, range, fps);
    std::vector<std::string> missing;
    for (const std::string& key : options.requested_metrics()) {
        // Only the columns in `order` are kept (a child sends no others): a key column_keys lacks counts as missing.
        if (std::find(scores.order.begin(), scores.order.end(), key) == scores.order.end()) {
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
        names.push_back(key == "psnr"        ? "PSNR"
                        : key == "ssim"      ? "SSIM"
                        : key == "xpsnr"     ? "XPSNR"
                        : key == "vmaf_hdr"  ? "HDR VMAF"
                        : scored_in_app(key) ? std::string(metric_definition(key).short_label)
                                             : key);
    }
    std::string text;
    for (std::size_t i = 0; i < names.size(); ++i) {
        text += (i == 0 ? "" : i + 1 == names.size() ? " and " : ", ") + names[i];
    }
    return text;
}

// What a plan's run is doing, for its statuses: "VMAF on the GPU", or
// "PSNR and SSIM in the app", or "HDR VMAF in the app".
std::string doing(const GpuPlan& plan) {
    return plan.backend == "cpu" || plan.backend == "hdr" ? metric_names(plan) + " in the app" : "VMAF on the GPU";
}

// `plan`, with its CPU scorers saying which of their metrics the GPU
// calculates as each starts, as a metrics_on_gpu status.
GpuPlan reporting_gpu_metrics(const GpuPlan& plan, const StatusCallback& on_status) {
    GpuPlan reporting = plan;
    if (on_status) {
        reporting.on_gpu_metrics = [on_status](const std::vector<std::string>& metrics,
                                               const std::vector<std::string>& on_gpu) {
            on_status({metrics_on_gpu_text(metrics, on_gpu), StatusKind::metrics_on_gpu});
        };
    }
    return reporting;
}

// The plan's metrics scored outside FFmpeg's filters, each video decoded by
// an FFmpeg of its own whose frames the app pairs and the plan's scorer
// scores (GpuAttempt): VMAF, NEG and VMAF v1 on the GPU, or PSNR, SSIM and
// XPSNR in the app. Falls back to other decode plans when hardware decoding
// fails.
ParsedScores execute_gpu_run(const RunFrames& run, const VmafOptions& options, const GpuPlan& planned,
                             const ProgressCallback& on_progress, const StatusCallback& on_status) {
    const RunEnvironment& environment = run.environment;
    const VideoInfo& source = run.source;
    const VideoInfo& distorted = run.distorted;
    const CancelToken& cancel = run.cancel;
    const GpuPlan plan = reporting_gpu_metrics(planned, on_status);
    if (options.requested_metrics().empty()) {
        throw VmafRunError("Select at least one metric to calculate.");
    }
    const TempFolder folder =
        run_folder(plan.backend == "cpu" || plan.backend == "hdr" ? "vmaf_cpu_run_" : "vmaf_gpu_run_");
    const VmafOptions seeking = reading_near_range(options, environment, source, distorted, cancel);
    const std::vector<HardwareDecoding> ladder = fallback_ladder(run.decoding);
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
        PairFeeds feeds(&source, &distorted, environment.ffmpeg, environment.ffprobe, cancel,
                        vvdec_wanted(options.gpu_decode, options.vvc_on_gpu), on_status);
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
                build_stream_graphs(source, distorted, seeking, run.source_crop, run.distorted_crop, decode), decode,
                attempt.output_args(), run_inputs(feeds, source, distorted, seeking.range_seeks));
            for (const std::vector<std::string>& command : commands) {
                logger.info("FFmpeg: " + python_list2cmdline(command));
            }
            outcome = run_ffmpeg_pair(commands, {distorted.path, source.path},
                                      [&attempt](std::size_t command) {
                                          return attempt.take_output_descriptors(command);
                                      },
                                      [&attempt](std::size_t command) { return attempt.closed_pipe_of(command); },
                                      run.total_frames, on_progress, cancel, folder.path());
            // The scorer is built while the FFmpegs run; when it could not
            // start, that is the failure to report, not theirs.
            attempt.wait_started();
        } catch (const platform::ProcessError& error) {
            abandon();
            throw VmafRunError(std::string("FFmpeg could not be started: ") + error.what());
        } catch (...) {
            abandon();
            throw;
        }
        try {
            feeds.finish();  // a fed video decoded in part is no score
        } catch (const VvdecFeedFailed& error) {
            // vvdec-vulkan is off now (it cannot fail twice): the same attempt again, VVC decoded by VVdeC
            abandon();
            report_vvdec_feed_failed(error, on_status);
            same_again = true;
            continue;
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
    return gpu_scores(*scored, distorted.fps, options, run_range(options, source));
}

// The keys of the metrics a plan scores.
std::vector<std::string> plan_keys(const GpuPlan& plan) {
    std::vector<std::string> keys;
    for (const auto& [key, model] : plan.models) {
        keys.push_back(key);
    }
    return keys;
}

// What the passes that decode in the app call as vvdec-vulkan opens: a vvdec_used status, before it opens, so that a
// crash while it opens counts as its own (score_plan_apart).
std::function<void()> on_vvdec_opening(const StatusCallback& on_status) {
    return [on_status] {
        if (on_status) {
            on_status({"VVC decoded on the GPU by vvdec-vulkan", StatusKind::vvdec_used});
        }
    };
}

// The plan's metrics with both videos decoded in this process (the Python
// app's _score_decoded_on_gpu): VMAF and NEG on the GPU (score_decoded),
// with VMAF v1 from whole frames (score_decoded_v1), or PSNR, SSIM and XPSNR
// in the app (score_decoded_cpu), over the same frames execute_gpu_run's
// FFmpegs would give the scorer.
ParsedScores score_decoded_here(const RunFrames& run, const VmafOptions& options, const GpuPlan& planned,
                                const std::pair<std::string, std::string>& decoders,
                                const ProgressCallback& on_progress, const StatusCallback& on_status) {
    const VideoInfo& source = run.source;
    const VideoInfo& distorted = run.distorted;
    const std::string& ffmpeg = run.environment.ffmpeg;
    const std::string& ffprobe = run.environment.ffprobe;
    const GpuPlan plan = reporting_gpu_metrics(planned, on_status);
    const FrameRange range = run_range(options, source);
    if (on_status) {
        on_status(decoding_status("Running " + doing(plan), run.decoding));
    }
    const std::vector<std::string> keys = plan_keys(plan);
    const bool want_vvdec = vvdec_wanted(options.gpu_decode, options.vvc_on_gpu);
    const PairScores scored =
        plan.backend == "cpu"
            ? score_decoded_cpu(source, distorted, run.source_crop, run.distorted_crop, plan.width, plan.height,
                                plan.bit_depth, keys, options.n_subsample, range, run.total_frames, decoders,
                                plan.threads, plan.frame_rate, plan.cambi, plan.extra_gpu, plan.vmaf_model,
                                plan.vmaf_on_vulkan, options.scale_algorithm, ffmpeg, ffprobe, want_vvdec,
                                on_vvdec_opening(on_status), plan.on_gpu_metrics, on_progress, run.cancel)
        : plan.backend == "hdr"
            ? score_decoded_hdr(source, distorted, run.source_crop, run.distorted_crop, plan, options.n_subsample,
                                range, run.total_frames, options.scale_algorithm, decoders, ffmpeg, ffprobe,
                                want_vvdec, on_vvdec_opening(on_status), on_progress, run.cancel)
        : std::find(keys.begin(), keys.end(), "vmaf_v1") != keys.end()
            ? score_decoded_v1(source, distorted, run.source_crop, run.distorted_crop, plan, options.n_subsample,
                               range, run.total_frames, options.scale_algorithm, decoders, ffmpeg, ffprobe,
                               want_vvdec, on_vvdec_opening(on_status), on_progress, run.cancel)
            : score_decoded(source, distorted, run.source_crop, run.distorted_crop, plan, options.n_subsample, range,
                            run.total_frames, options.scale_algorithm, decoders, ffmpeg, ffprobe, want_vvdec,
                            on_vvdec_opening(on_status), on_progress, run.cancel);
    return gpu_scores(scored, distorted.fps, options, range);
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
// and NEG without VMAF v1 are decoded here only by those. CIEDE2000 and
// CAMBI on the GPU (extra_gpu) that fail partway make the pass again with
// them in libvmaf on the CPU. score_plan watches it for memory.
ParsedScores score_plan_unwatched(const RunFrames& run, const VmafOptions& options, const GpuPlan& plan,
                                  const ProgressCallback& on_progress, const StatusCallback& on_status);

// score_plan_unwatched, without making the pass again for CIEDE2000, CAMBI and MS-SSIM.
ParsedScores score_plan_as_planned(const RunFrames& run, const VmafOptions& options, const GpuPlan& plan,
                                   const ProgressCallback& on_progress, const StatusCallback& on_status) {
    // A video whose size changes is decoded by FFmpeg, which scales each of
    // its frames (size_changes.h); the app's decoders read one size.
    const std::optional<std::pair<std::string, std::string>> decoders =
        run.source.scaled_to || run.distorted.scaled_to ? std::nullopt : decoders_here(run.decoding);
    const std::vector<std::string> keys = plan_keys(plan);
    if (decoders && (plan.backend != "cuda" || std::find(keys.begin(), keys.end(), "vmaf_v1") != keys.end() ||
                     (libvmaf_cuda_takes(decoders->first) && libvmaf_cuda_takes(decoders->second)))) {
        try {
            return score_decoded_here(run, options, plan, *decoders, on_progress, on_status);
        } catch (const SeekRefused& error) {
            // Nothing scored yet: the same pass from the videos' start, with the decoders it had
            turn_off_range_seek(error.what());
            return score_plan_unwatched(run, options, plan, on_progress, on_status);
        } catch (const GpuDecodeUnavailableError& error) {
            if (error.backend() == vvdec_decoder) {
                // vvdec-vulkan refused: off for the run, and the same pass with the app's VVdeC (vvdec_wanted is
                // false now, so this runs once)
                turn_off_vvdec(error.what(), on_status);
                return score_plan_unwatched(run, options, plan, on_progress, on_status);
            }
            // FFmpeg decodes the videos.
            logger.info(doing(plan) + ": the videos are decoded by FFmpeg (" + error.what() + ")");
        } catch (const GpuDecodeFailedError& error) {
            if (error.backend() == vvdec_decoder) {
                // vvdec-vulkan failed partway: the pass again from its first frame with the app's VVdeC, not FFmpeg's
                // slower route; the other GPU decoders stay trusted
                logger.warning("vvdec-vulkan failed for " + doing(plan) +
                               "; decoding with VVdeC on the CPU instead: " + std::string(error.what()));
                if (on_status) {
                    on_status({decoding_failed_status(error, "with VVdeC on the CPU")});
                }
                turn_off_vvdec(error.what(), on_status);
                return score_plan_unwatched(run, options, plan, on_progress, on_status);
            }
            logger.warning("Decoding for " + (plan.backend == "cpu" ? metric_names(plan) : "VMAF on the GPU") +
                           " failed; decoding through FFmpeg instead: " + error.what());
            if (on_status) {
                on_status({decoding_failed_status(error, "through FFmpeg")});
            }
            distrust_failed_decoder(error, on_status);
        }
    }
    return execute_gpu_run(run, options, plan, on_progress, on_status);
}

ParsedScores score_plan_unwatched(const RunFrames& run, const VmafOptions& options, const GpuPlan& plan,
                                  const ProgressCallback& on_progress, const StatusCallback& on_status) {
    try {
        return score_plan_as_planned(run, options, plan, on_progress, on_status);
    } catch (const GpuMetricsFailed& error) {
        // Here, in the process that scores, so that the caller's fallbacks (vvdec-vulkan turned off, FFmpeg's
        // filters, which cannot score the app's own metrics) are not taken for a failure of these two alone.
        logger.warning(std::string(error.what()) + "; the pass is made again with them on the CPU");
        if (on_status) {
            on_status({std::string(error.what()) + "; calculating them on the CPU\xE2\x80\xA6"});
        }
        GpuPlan on_cpu = plan;
        on_cpu.extra_gpu.reset();
        return score_plan_as_planned(run, options, on_cpu, on_progress, on_status);
    }
}

// A token of its own for a part of the run that may be stopped alone, which
// follows the run's Cancel and Pause (`cancel`) for as long as it lives,
// looking every 50 ms. (With only a Cancel passed on, and only once the
// run's own scoring had ended, a part beside it went on scoring through a
// Pause; the Python app hands it the run's own process handle.)
class FollowingToken {
public:
    explicit FollowingToken(const CancelToken& cancel) : cancel_(cancel), relay_([this] { relay(); }) {}
    ~FollowingToken() {
        {
            const std::lock_guard lock(mutex_);
            ended_ = true;
        }
        wake_.notify_all();
        relay_.join();
    }
    FollowingToken(const FollowingToken&) = delete;
    FollowingToken& operator=(const FollowingToken&) = delete;

    const CancelToken& token() const { return stop_; }
    // Stops the part alone.
    void stop() { stop_.cancel(); }

private:
    void relay() {
        std::unique_lock lock(mutex_);
        while (!wake_.wait_for(lock, std::chrono::milliseconds(50), [this] { return ended_; })) {
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
    std::condition_variable wake_;
    bool ended_ = false;
    std::thread relay_;  // last: it starts once the others are made
};

// The metrics a pass scores: its plan's own and those beside them.
std::vector<std::string> pass_keys(const GpuPlan& plan) {
    std::vector<std::string> keys = plan_keys(plan);
    for (const std::string& key : plan.beside) {
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
            keys.push_back(key);
        }
    }
    return keys;
}

// The pass's GPU's memory (MemoryWatch): the GPU the Vulkan engine scores
// its metrics on. None where it uses none (libvmaf's CUDA code alone, whose
// memory Vulkan's budget may not see, and the CPU).
MemoryWatch::GpuReader pass_gpu_memory(const GpuPlan& plan) {
    const std::optional<int> device = plan.backend == "vulkan" ? std::optional(plan.device)
                                      : plan.extra_gpu         ? plan.extra_gpu
                                                               : plan.hdr_gpu;
    if (!device) {
        return nullptr;
    }
    return [device = *device] { return vulkan_memory(device); };
}

// A pass of `keys` that `score` scores (given the run and the progress
// callback to score it with), watched for memory (memory_fit.h) where its
// metrics could be split: a pass found not to fit as it starts is stopped,
// and it and one refused memory throw MemoryShort, for the run to score its
// metrics in halves (score_fitting). Its own token, which follows the run's
// Cancel and Pause, stops it alone. `gpu_memory`: the pass's GPU's, if any.
ParsedScores score_watched(const RunFrames& run, const std::vector<std::string>& keys,
                           MemoryWatch::GpuReader gpu_memory, const ProgressCallback& on_progress,
                           const std::function<ParsedScores(const RunFrames&, const ProgressCallback&)>& score) {
    MemoryWatch watch(keys, std::move(gpu_memory), [] { return platform::memory_headroom(); });
    if (!watch.active()) {
        return score(run, on_progress);
    }
    FollowingToken pass(run.cancel);
    const ProgressCallback progress = [&](long long done, long long whole, double fps) {
        // FFmpeg's reader threads report progress, which must not throw: a check that fails is no shortage.
        try {
            if (watch.check(done)) {
                pass.stop();
            }
        } catch (const std::exception&) {
        }
        if (on_progress) {
            on_progress(done, whole, fps);
        }
    };
    try {
        return score(run.with_cancel(pass.token()), progress);
    } catch (const std::exception& error) {
        if (run.cancel.cancelled() || dynamic_cast<const MemoryShort*>(&error) != nullptr) {
            throw;
        }
        std::optional<MemoryShortage> shortage = watch.shortage();
        if (!shortage) {
            const std::optional<bool> on_gpu = memory_refused(error);
            if (!on_gpu || memory_split_groups(keys, *on_gpu) < 2) {
                throw;
            }
            shortage = MemoryShortage{*on_gpu, error.what()};
        }
        throw MemoryShort(*shortage);
    }
}

// score_plan_unwatched, watched for memory (score_watched).
ParsedScores score_plan(const RunFrames& run, const VmafOptions& options, const GpuPlan& plan,
                        const ProgressCallback& on_progress, const StatusCallback& on_status) {
    return score_watched(run, pass_keys(plan), pass_gpu_memory(plan), on_progress,
                         [&](const RunFrames& watched, const ProgressCallback& progress) {
                             return score_plan_unwatched(watched, options, plan, progress, on_status);
                         });
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
// here, and FFmpeg's filter scores XPSNR (of 8 bits; of more, nothing does:
// ffmpeg_scores_xpsnr).
std::optional<int> xpsnr_frame_rate(const VideoInfo& info) {
    const double average = info.average_fps, nominal = info.nominal_fps;
    if (average <= 0 || nominal <= 0 || std::abs(1.0 - average / nominal) >= 0.09) {
        return std::nullopt;
    }
    return static_cast<int>(nominal);
}

// Whether libvmaf-fast's port of FFmpeg's xpsnr filter gives exactly
// FFmpeg's score for this comparison, as FFmpeg gives it for pictures whose
// rows are not padded (ffmpeg_scores_xpsnr): up to 12 bits, and an even size
// above 2048x1152 (FFmpeg's filter reads beyond the picture of an odd one).
bool xpsnr_in_app(int width, int height, int bit_depth) {
    if (bit_depth > 12 || (static_cast<long long>(width) * height > 2048LL * 1152 && is_odd_size(width, height))) {
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
// filters, which it runs anyway). XPSNR of 8 bits stays with FFmpeg where
// libvmaf-fast's score would not be FFmpeg's exactly (xpsnr_frame_rate,
// xpsnr_in_app), while PSNR and SSIM are still scored in the app; of more
// bits, FFmpeg's would be wrong (ffmpeg_scores_xpsnr): it is the app's, or
// nobody's, as the app's own metrics are always (app_only_metrics). `cores`
// are "Auto"'s threads; `ffmpeg` scales a video the decoders cannot
// (decoded_in_app).
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
    bool app_only = false;  // a metric FFmpeg's filters do not score
    for (const MetricDefinition& metric : vml::metrics()) {  // a plane right after its metric
        if (scored_in_app(metric.key) && options.metric_enabled(metric.key)) {
            metrics.emplace_back(metric.key);
            app_only = true;
        }
    }
    const auto [width, height] = size;
    const int depth = bit_depth(analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt}));
    const std::optional<int> frame_rate = options.compute_xpsnr ? xpsnr_frame_rate(distorted) : std::nullopt;
    // Where FFmpeg's filter would score XPSNR wrongly, the speed it would
    // give is no reason to leave XPSNR to it.
    const bool ffmpeg_may_score = ffmpeg_scores_xpsnr(depth);
    if (frame_rate && xpsnr_in_app(width, height, depth)) {
        metrics.emplace_back("xpsnr");
    } else if (options.compute_xpsnr && ffmpeg_may_score && !(decoding.source && decoding.distorted) &&
               !app_only) {
        // XPSNR's FFmpeg beside them would decode the videos again, and a
        // video the CPU decodes would then be decoded twice on the CPU: 70%
        // more CPU per frame for a 4K film against its encode, and slower for
        // a 4K film against a 1080p encode (97 fps against 112). FFmpeg's
        // single run decodes each video once for both libvmaf and XPSNR. The
        // app's own metrics are the app's alone, so XPSNR's FFmpeg is then run
        // beside them.
        return std::nullopt;
    }
    if (metrics.empty()) {
        return std::nullopt;
    }
    if (metrics == std::vector<std::string>{"xpsnr"} && ffmpeg_may_score &&
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
    plan.cambi = cambi_input(distorted, distorted_crop);
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
// then each metric's values in `order`, then the values as calculated of
// those in `exact` (ParsedScores::exact), as this PC stores them.
constexpr const char* scores_file = "scores.bin";

ParsedScores read_scores(const nlohmann::json& result, const std::filesystem::path& folder, double fps) {
    ParsedScores scores;
    const auto count = result.at("frames").get<std::size_t>();
    scores.order = result.at("order").get<std::vector<std::string>>();
    scores.features_on_gpu = result.value("features_on_gpu", std::vector<std::string>{});
    std::ifstream file(folder / scores_file, std::ios::binary);
    scores.frame.resize(count);
    file.read(reinterpret_cast<char*>(scores.frame.data()), static_cast<std::streamsize>(count * sizeof(int)));
    for (const std::string& key : scores.order) {
        std::vector<float>& values = scores.metrics[key];
        values.resize(count);
        file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    }
    for (const std::string& key : result.value("exact", std::vector<std::string>{})) {
        std::vector<double>& values = scores.exact[key];
        values.resize(count);
        file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(double)));
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
    // HDR VMAF's colours, each video's as libvmaf's numbers: range, primaries, transfer, matrix.
    nlohmann::json colours = nlohmann::json::array();
    for (const VmafColor& colour : {plan.hdr_colours.source, plan.hdr_colours.distorted}) {
        colours.push_back({colour.range, colour.primaries, colour.trc, colour.matrix});
    }
    return {{"models", models},           {"width", plan.width},     {"height", plan.height},
            {"bit_depth", plan.bit_depth}, {"device", plan.device},   {"backend", plan.backend},
            {"beside", plan.beside},       {"threads", plan.threads}, {"frame_rate", plan.frame_rate},
            {"cambi", {plan.cambi.width, plan.cambi.height, plan.cambi.bit_depth, plan.cambi.pq}},
            {"hdr_colours", colours},      {"hdr_model_file", plan.hdr_model_file},
            {"hdr_threads", plan.hdr_threads},
            {"hdr_gpu", plan.hdr_gpu ? nlohmann::json(*plan.hdr_gpu) : nlohmann::json(nullptr)},
            {"extra_gpu", plan.extra_gpu ? nlohmann::json(*plan.extra_gpu) : nlohmann::json(nullptr)},
            {"vmaf_model", plan.vmaf_model}, {"vmaf_on_vulkan", plan.vmaf_on_vulkan}};
}

VmafColor colour_from(const nlohmann::json& text) {
    return {static_cast<VmafColorRange>(text.at(0).get<int>()),
            static_cast<VmafColorPrimaries>(text.at(1).get<int>()),
            static_cast<VmafColorTransferCharacteristic>(text.at(2).get<int>()),
            static_cast<VmafColorMatrixCoefficients>(text.at(3).get<int>())};
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
    const nlohmann::json& cambi = text.at("cambi");
    plan.cambi = {cambi.at(0).get<int>(), cambi.at(1).get<int>(), cambi.at(2).get<int>(), cambi.at(3).get<bool>()};
    plan.hdr_colours = {colour_from(text.at("hdr_colours").at(0)), colour_from(text.at("hdr_colours").at(1))};
    text.at("hdr_model_file").get_to(plan.hdr_model_file);
    text.at("hdr_threads").get_to(plan.hdr_threads);
    if (const nlohmann::json& gpu = text.at("hdr_gpu"); !gpu.is_null()) {
        plan.hdr_gpu = gpu.get<int>();
    }
    if (const nlohmann::json& gpu = text.at("extra_gpu"); !gpu.is_null()) {
        plan.extra_gpu = gpu.get<int>();
    }
    text.at("vmaf_model").get_to(plan.vmaf_model);
    text.at("vmaf_on_vulkan").get_to(plan.vmaf_on_vulkan);
    return plan;
}

// One attempt of score_plan_apart: the child's run, `vvdec_used` set where it opened vvdec-vulkan.
ParsedScores run_plan_child(std::string_view task, const RunFrames& run, const VmafOptions& options,
                            const GpuPlan& plan, const ProgressCallback& on_progress, const StatusCallback& on_status,
                            bool& vvdec_used) {
    const nlohmann::json request{
        {"ffmpeg", run.environment.ffmpeg},
        {"ffprobe", run.environment.ffprobe},
        {"source", video_json(run.source)},
        {"distorted", video_json(run.distorted)},
        {"options", options_json(child_options(options))},
        {"source_crop", crop_json(run.source_crop)},
        {"distorted_crop", crop_json(run.distorted_crop)},
        {"plan", plan_json(plan)},
        {"total_frames", run.total_frames},
        {"decoding",
         {{"source", hwaccel_json(run.decoding.source)}, {"distorted", hwaccel_json(run.decoding.distorted)}}}};
    ParsedScores scores;
    run_isolated(
        task, "libvmaf", request,
        [&](const nlohmann::json& message) {
            if (message.contains("progress") && on_progress) {
                const nlohmann::json& counts = message["progress"];
                on_progress(counts.at(0).get<long long>(), counts.at(1).get<long long>(), counts.at(2).get<double>());
            } else if (message.contains("status")) {
                take_child_status(status_from_json(message["status"]), on_status, vvdec_used);
            }
        },
        run.cancel,
        [&](const nlohmann::json& result, const std::filesystem::path& folder) {
            scores = read_scores(result, folder, run.distorted.fps);
        });
    return scores;
}

// The plan's metrics scored in a process of its own (score_plan in the
// child, `task` "gpu-run" or "cpu-metrics", HDR VMAF's too), so that a crash in libvmaf, the
// GPU's code or its driver ends that process and not the run (the Python
// app's run_isolated(..., what="libvmaf")). A child that opened vvdec-vulkan
// and then failed or crashed, where it did not turn vvdec-vulkan off itself,
// is run once more with it off (the app's VVdeC on the CPU), still in the
// caller's GPU turn. Throws what run_isolated throws.
ParsedScores score_plan_apart(std::string_view task, const RunFrames& run, const VmafOptions& options,
                              const GpuPlan& plan, const ProgressCallback& on_progress,
                              const StatusCallback& on_status) {
    for (bool retried = false;; retried = true) {
        bool vvdec_used = false;
        try {
            return run_plan_child(task, run, options, plan, on_progress, on_status, vvdec_used);
        } catch (const std::runtime_error& error) {
            if (ends_the_run(error) || retried || !vvdec_used || vvdec_turned_off()) {
                throw;
            }
            turn_off_vvdec(std::string("its pass failed: ") + error.what(), on_status);
        }
    }
}

// The plan's CPU metrics (and HDR VMAF's) scored in the app: in a process of its own where a program to run it is
// set (score_plan_apart's "cpu-metrics"), else here. Throws what they throw.
ParsedScores score_cpu_plan(const RunFrames& run, const VmafOptions& options, const GpuPlan& plan,
                            const ProgressCallback& on_progress, const StatusCallback& on_status) {
    return isolation_program() ? score_plan_apart("cpu-metrics", run, options, plan, on_progress, on_status)
                               : score_plan(run, options, plan, on_progress, on_status);
}

// VMAF, NEG and VMAF v1 on the GPU (score_plan), with the plan's CPU metrics
// beside them, one GPU pass at a time, in a process of its own where a
// program to run it is set (score_plan_apart). Throws what they throw.
ParsedScores score_on_gpu(const RunFrames& run, const VmafOptions& options, const GpuPlan& plan,
                          const ProgressCallback& on_progress, const StatusCallback& on_status) {
    std::string models;
    for (const auto& [key, model] : plan.models) {
        models += (models.empty() ? "" : ", ") + model;
    }
    std::string beside;
    if (!plan.beside.empty()) {
        // CIEDE2000, CAMBI and MS-SSIM may go to the GPU with one given them, VMAF YUV's U and V on the Vulkan engine's
        // (cpu_plan); libvmaf on the CPU scores the others, HDR VMAF's on threads of its own (log_hdr_vmaf).
        const std::vector<std::string> maybe_gpu =
            may_run_on_gpu(plan.beside, plan.extra_gpu.has_value(), plan.vmaf_on_vulkan);
        const auto on_cpu = [&maybe_gpu](const std::string& key) {
            return std::find(maybe_gpu.begin(), maybe_gpu.end(), key) == maybe_gpu.end();
        };
        const bool cpu_library = std::any_of(plan.beside.begin(), plan.beside.end(), on_cpu);
        const bool cpu_metrics = std::any_of(plan.beside.begin(), plan.beside.end(), [&on_cpu](const std::string& key) {
            return is_cpu_metric(key) && on_cpu(key);
        });
        beside = ", with " + cpu_metrics_text(plan.beside, maybe_gpu) + " beside it from the same frames";
        if (cpu_library) {
            beside += " (" + cpu_library_build() +
                      (cpu_metrics ? ", the CPU's metrics on " + std::to_string(plan.threads) + " threads" : "") + ")";
        }
    }
    logger.info("VMAF on the GPU (" + models +
                "): " + (plan.backend == "vulkan" ? vulkan_library_build() : cuda_library_build()) + beside);
    const std::unique_lock pass = take_gpu_turn(
        [&on_status] {
            if (on_status) {
                on_status({std::string(gpu_wait_message), StatusKind::gpu_wait});
            }
        },
        run.cancel);
    if (!isolation_program()) {
        return score_plan(run, options, plan, on_progress, on_status);
    }
    return score_plan_apart("gpu-run", run, options, plan, on_progress, on_status);
}

// score_on_gpu without metrics beside, or nothing when it fails for any
// reason but Cancel: the run is then made again with VMAF on the CPU, as on a
// PC without a GPU that scores it.
std::optional<ParsedScores> run_on_gpu(const RunFrames& run, const VmafOptions& options, const GpuPlan& plan,
                                       const ProgressCallback& on_progress, const StatusCallback& on_status) {
    try {
        return score_on_gpu(run, options, plan, on_progress, on_status);
    } catch (const std::exception& error) {
        if (ends_the_run(error)) {
            throw;
        }
        logger.error(std::string("VMAF on the GPU failed; calculating it on the CPU: ") + error.what());
        forget_gpu_checks();  // the saved check said this GPU works: check it again next start
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
// runs. Its token is its own (FollowingToken), so that it can be stopped
// alone when the run's own scoring fails. Stopped and waited for when it goes.
class ScoresBeside {
public:
    ScoresBeside(std::function<ParsedScores(const CancelToken&)> run, const CancelToken& cancel) : stop_(cancel) {
        run_ = std::thread([this, run = std::move(run)] {
            try {
                ParsedScores found = run(stop_.token());
                const std::lock_guard lock(mutex_);
                scores_ = std::move(found);
            } catch (...) {
                const std::lock_guard lock(mutex_);
                error_ = std::current_exception();
            }
        });
    }
    ~ScoresBeside() {
        stop();
        join();
    }
    ScoresBeside(const ScoresBeside&) = delete;
    ScoresBeside& operator=(const ScoresBeside&) = delete;

    void stop() { stop_.stop(); }
    // Waits for its run to end.
    void join() {
        if (run_.joinable()) {
            run_.join();
        }
    }
    // Its scores, once joined. Where its run failed, `keys` are failed, so
    // that the scores beside them stand; what ends the whole run goes on up.
    ParsedScores scores_or_failed(const std::vector<std::string>& keys) {
        try {
            if (error_) {
                std::rethrow_exception(error_);
            }
        } catch (const std::exception& error) {
            if (ends_the_run(error)) {
                throw;
            }
            logger.error(std::string("A part of the run failed, the others' scores stand: ") + error.what());
            ParsedScores none;
            for (const std::string& key : keys) {
                none.failed[key] = error.what();
            }
            return none;
        }
        return std::move(scores_).value_or(ParsedScores{});
    }

private:
    FollowingToken stop_;
    std::mutex mutex_;
    std::optional<ParsedScores> scores_;
    std::exception_ptr error_;
    std::thread run_;
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
// taken for the frames PSNR and SSIM were scored for; above 8 bits, where
// FFmpeg would score it wrongly, it fails instead. XPSNR alone is scored
// for every frame, as FFmpeg's filter scores it whatever n_subsample is.
// Nothing when it fails for any reason but Cancel: FFmpeg's filters then
// calculate the metrics.
std::optional<ParsedScores> run_cpu_metrics(const RunFrames& run, const VmafOptions& options, const GpuPlan& plan,
                                            const ProgressCallback& on_progress, const StatusCallback& on_status,
                                            std::string* failure = nullptr) {
    logger.info(metric_names(plan) + " in the app: " + cpu_library_build() + ", " + std::to_string(plan.threads) +
                " threads");
    const std::vector<std::string> keys = plan_keys(plan);
    const bool left_to_ffmpeg = options.compute_xpsnr && std::find(keys.begin(), keys.end(), "xpsnr") == keys.end();
    const bool beside = left_to_ffmpeg && ffmpeg_scores_xpsnr(plan.bit_depth);
    SlowerProgress progress(on_progress, beside ? 2 : 1);
    std::optional<ScoresBeside> xpsnr;
    if (beside) {
        VmafOptions xpsnr_options = options;  // only XPSNR: the others would be scored there again
        for (const std::string& key : options.requested_metrics()) {
            xpsnr_options.set_metric_enabled(key, key == "xpsnr");
        }
        xpsnr.emplace(
            [&, xpsnr_options, xpsnr_progress = progress.part(1)](const CancelToken& stop) {
                return execute_run(run.with_cancel(stop), xpsnr_options, "", xpsnr_progress, nullptr);
            },
            run.cancel);
    }
    VmafOptions scored = options;
    if (left_to_ffmpeg) {
        scored.compute_xpsnr = false;
    } else if (keys == std::vector<std::string>{"xpsnr"}) {  // XPSNR alone
        scored.n_subsample = 1;
    }
    std::optional<ParsedScores> frames;
    try {
        frames = score_cpu_plan(run, scored, plan, progress.part(0), on_status);
    } catch (const std::exception& error) {
        xpsnr.reset();  // stopped and waited for
        if (ends_the_run(error)) {
            throw;
        }
        logger.error(metric_names(plan) + " in the app failed; FFmpeg calculates them: " + error.what());
        if (failure != nullptr) {
            *failure = error.what();
        }
        if (on_status) {
            on_status(
                {metric_names(plan) + " in the app failed (" + error.what() + "); calculating them with FFmpeg…"});
        }
        return std::nullopt;
    }
    if (xpsnr) {
        xpsnr->join();
    }
    run.cancel.throw_if_cancelled();
    if (xpsnr) {
        const ParsedScores found = xpsnr->scores_or_failed({"xpsnr"});
        if (found.has("xpsnr") && !frames->frame.empty() && found.frame.back() < frames->frame.back()) {
            frames->failed["xpsnr"] = "XPSNR's FFmpeg stopped at frame " + std::to_string(found.frame.back()) +
                                      " of " + std::to_string(frames->frame.back());
        } else if (found.has("xpsnr")) {
            merge_xpsnr(*frames, found);
        } else {
            frames->failed.insert(found.failed.begin(), found.failed.end());
        }
    } else if (left_to_ffmpeg) {
        frames->failed["xpsnr"] = xpsnr_refusal;
    }
    return frames;
}

// The VMAF model a run uses, from the size its frames are compared at, which
// is known only once black bars are: Auto's model comes from it, while an
// explicit or custom choice is the user's. VMAF v1's and HDR VMAF's models
// are resolved into `options` the same way.
// The model= value of the run's VMAF at the size compared: the one chosen, or, for Auto, the one for that size.
std::string run_vmaf_model(const VmafOptions& options, int width, int height) {
    return options.model_choice != auto_model_choice ? options.model : model_for_resolution(width, height);
}

std::string resolve_run_models(VmafOptions& options, int width, int height) {
    if (options.compute_vmaf_v1) {
        options.model_v1 = resolve_v1_model(options, width, height);
    }
    if (options.compute_vmaf_hdr) {
        options.model_hdr = resolve_hdr_model(options, width, height);
    }
    if (!options.compute_vmaf) {
        return {};
    }
    return run_vmaf_model(options, width, height);
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

RunStatus decoding_why_status(const HardwareDecoding& plan, const VideoInfo& source, const VideoInfo& distorted,
                              MacGpuDecoding mac) {
    RunStatus status{"", StatusKind::decoding_why};
    status.decoding_why = software_decoding_why(plan, source, distorted, true, std::nullopt, mac);
    return status;
}

void distrust_failed_decoder(const GpuDecodeFailedError& error, const StatusCallback& on_status) {
    if (!error.gpu_decoder()) {
        return;  // the file's reading failed, or the app's own software decoder: nothing says the GPU is wrong
    }
    distrust_gpu_decoding(error.what());
    if (on_status) {
        on_status({error.what(), StatusKind::gpu_decoder_failed});
    }
}

void turn_off_vvdec(const std::string& reason, const StatusCallback& on_status) {
    turn_off_vvdec_vulkan(reason);
    if (on_status) {
        on_status({reason, StatusKind::vvdec_off});
    }
}

void take_child_status(const RunStatus& status, const StatusCallback& on_status, bool& vvdec_used) {
    if (status.kind == StatusKind::gpu_decoder_failed) {
        distrust_gpu_decoding(status.text);
    } else if (status.kind == StatusKind::vvdec_off) {
        turn_off_vvdec_vulkan(status.text);
    } else if (status.kind == StatusKind::vvdec_used) {
        vvdec_used = true;
    }
    if (on_status) {
        on_status(status);
    }
}

VmafOptions child_options(VmafOptions options) {
    options.vvc_on_gpu = options.vvc_on_gpu && !vvdec_turned_off();
    return options;
}

RunStatus gpu_decoding_off_status(const std::string& reason) {
    return {"The GPU's video decoding failed (" + reason + "); the videos were decoded on the CPU\xE2\x80\xA6",
            StatusKind::gpu_decoding_off};
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

std::string metrics_on_gpu_text(const std::vector<std::string>& metrics, const std::vector<std::string>& on_gpu) {
    const auto joined = [](const std::vector<std::string>& keys) {
        std::string text;
        for (const std::string& key : keys) {
            text += (text.empty() ? "" : ",") + key;
        }
        return text;
    };
    return joined(metrics) + ";" + joined(on_gpu);
}

std::pair<std::vector<std::string>, std::vector<std::string>> metrics_on_gpu_from(std::string_view text) {
    const auto keys = [](std::string_view list) {
        std::vector<std::string> found;
        while (!list.empty()) {
            const std::size_t comma = list.find(',');
            if (const std::string_view key = list.substr(0, comma); !key.empty()) {
                found.emplace_back(key);
            }
            list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
        }
        return found;
    };
    const std::size_t semicolon = text.find(';');
    return {keys(text.substr(0, semicolon)),
            keys(semicolon == std::string_view::npos ? std::string_view() : text.substr(semicolon + 1))};
}

void validate_video_pair(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options) {
    if (const std::optional<std::string> problem =
            pair_problem(source, distorted, options.compared_range())) {
        throw VmafRunError(*problem);
    }
}

void validate_display_geometry(const VideoInfo& source, const VideoInfo& distorted,
                               const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop) {
    if (const std::optional<std::string> problem = shape_problem(source, distorted, source_crop, distorted_crop)) {
        throw VmafRunError(*problem);
    }
}

FrameRange run_range(const VmafOptions& options, const VideoInfo& source) {
    // VMAF's motion (HDR VMAF's and VMAF U and V's too) and XPSNR look at the frames around the one they
    // score. U and V can be in a pass without VMAF (plan_tasks puts them in one of their own on the CPU).
    const bool looks_around = options.compute_vmaf || options.compute_vmaf_neg || options.compute_vmaf_v1 ||
                              options.compute_vmaf_hdr || options.compute_vmaf_yuv_u ||
                              options.compute_vmaf_yuv_v || options.compute_xpsnr;
    return frame_range(options.compared_range(), source.fps, options.n_subsample, looks_around);
}

namespace {

// The frames of the shorter video, as far as they are known.
long long shorter_frame_count(const VideoInfo& reference, const VideoInfo* other) {
    const long long frames = reference.estimated_frame_count();
    return other != nullptr ? std::min(frames, other->estimated_frame_count()) : frames;
}

}  // namespace

long long estimate_total_frames(const VideoInfo& reference, const VmafOptions& options, const VideoInfo* other) {
    const long long frames = shorter_frame_count(reference, other);
    const FrameRange range = run_range(options, reference);
    return std::max(0LL, std::min(frames, range.end.value_or(frames)) - range.start);
}

long long estimate_decoded_frames(const VideoInfo& reference, const VmafOptions& options, const VideoInfo* other) {
    const long long frames = shorter_frame_count(reference, other);
    const FrameRange range = run_range(options, reference);
    return std::max(0LL, std::min(frames, range.decode_end.value_or(frames)) - range.decode_start);
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

std::optional<std::string> short_comparison(long long expected, long long compared, long long first_frame,
                                            double fps, int step) {
    // Short by at most half a second (at least three frames), plus one step
    // of subsampling, still covers the video.
    const long long slack = std::max(3LL, static_cast<long long>(std::ceil(0.5 * fps))) + std::max(1, step);
    if (expected <= 0 || compared >= expected - slack) {
        return std::nullopt;
    }
    return "Only " + std::to_string(compared) + " of the " + std::to_string(expected) +
           " frames expected could be compared: one of the videos ends after " +
           format_hms(static_cast<double>(first_frame + compared) / std::max(fps, 1.0), 1) +
           ", though its file says it is longer. It may have been cut short: an encode that stopped early, or a "
           "copy that did not finish.";
}

bool cpu_metrics_in_app() {
    std::error_code ignored;
    return ascii_lower(platform::environment_variable(cpu_metrics_variable).value_or("")) != "ffmpeg" &&
           std::filesystem::is_regular_file(libvmaf_folder() / platform::shared_library_name("libvmaf"), ignored);
}

bool is_cpu_metric(std::string_view key) {
    return key == "psnr" || key == "ssim" || key == "xpsnr" || scored_in_app(key);
}

bool cpu_metrics_need_every_frame(const std::vector<std::string>& keys, int n_subsample) {
    return keys == std::vector<std::string>{"xpsnr"} && n_subsample > 1;
}

namespace {

// Whether a GPU pass's pictures reach the CPU's scorer as they would in the
// CPU metrics' own run, so that it may score them beside its own. Both videos
// decoded in this process (a video whose size changes is FFmpeg's, in either
// pass): the software decoder's and Intel's pictures, which are never handed
// over (can_hand_over: nvf_download writes them out in both passes), NVIDIA's,
// which always are, and AMD's, which are not with these metrics beside
// (pass_may_hand_over): AMF's download in both. (Beside VMAF YUV's U and V
// or HDR VMAF alone, AMD's are handed over, pass_may_hand_over_beside: those
// take Windows' decoder's pictures on the GPU.) Or FFmpeg decoding them, Apple's
// VideoToolbox having no library here: both passes read the same pipes.
bool pass_feeds_cpu_metrics(const VideoInfo& source, const VideoInfo& distorted, const HardwareDecoding& decoding) {
    const auto ffmpeg_decodes = [](const std::optional<std::string>& hwaccel) {
        const GpuDecoder* decoder = hwaccel ? decoder_of_hwaccel(*hwaccel) : nullptr;
        return decoder && decoder->library.empty();
    };
    if (ffmpeg_decodes(decoding.source) || ffmpeg_decodes(decoding.distorted)) {
        return true;
    }
    return !source.scaled_to && !distorted.scaled_to && decoders_here(decoding).has_value();
}

}  // namespace

bool shares_decode(const VideoInfo& source, const VideoInfo& test, const VmafOptions& options,
                   const DecodingCapabilities& capabilities) {
    if (!cpu_metrics_in_app()) {
        return false;
    }
    if (!options.gpu_decode) {
        return true;
    }
    for (const VideoInfo* video : {&source, &test}) {
        if (decoded_by_the_app(*video)) {
            return true;
        }
    }
    const HardwareDecoding decoding = plan_hardware_decoding(capabilities, options.gpu_vendor, source, test);
    return !decoding.source || !decoding.distorted || pass_feeds_cpu_metrics(source, test, decoding);
}

namespace {

// Whether scoring `plan` opens the Vulkan engine: VMAF on it, VMAF v1 (Vulkan's on every GPU), HDR VMAF's features
// on the GPU, or CIEDE2000, CAMBI, MS-SSIM or VMAF YUV's U and V on it (extra_gpu, set only
// for those).
bool plan_opens_vulkan(const GpuPlan& plan) {
    const std::vector<std::string> keys = plan_keys(plan);
    return plan.backend == "vulkan" || plan.extra_gpu || plan.hdr_gpu ||
           std::find(keys.begin(), keys.end(), "vmaf_v1") != keys.end();
}

}  // namespace

nlohmann::json scoring_task(const nlohmann::json& request, const std::filesystem::path& folder) {
    const GpuPlan plan = plan_from(request.at("plan"));
    // The GPU's Vulkan driver loaded on a thread of its own while the decoders start, where the plan opens the
    // engine: its first instance in a process took 194 ms (RTX 5090, driver 617.24), every later one 15-20 ms, and
    // each pass is a process of its own. The engine makes its one instance once, whichever thread asks first. The
    // future waits for the thread as it goes (std::jthread is not in the macOS CI's library).
    std::future<void> vulkan_warmup;
    if (plan_opens_vulkan(plan)) {
        vulkan_warmup = std::async(std::launch::async, [] {
            try {
                static_cast<void>(vulkan_devices());
            } catch (const std::exception&) {
                // The pass's own first use reports it.
            }
        });
    }
    RunEnvironment environment;
    environment.ffmpeg = request.at("ffmpeg").get<std::string>();
    environment.ffprobe = request.at("ffprobe").get<std::string>();
    const nlohmann::json& decoders = request.at("decoding");
    const VideoInfo source = video_from(request.at("source"));
    const VideoInfo distorted = video_from(request.at("distorted"));
    const std::optional<CropBox> source_crop = crop_from(request.at("source_crop"));
    const std::optional<CropBox> distorted_crop = crop_from(request.at("distorted_crop"));
    const HardwareDecoding decoding{hwaccel_from(decoders.at("source")), hwaccel_from(decoders.at("distorted"))};
    const CancelToken cancel;  // the parent cancels by ending this process
    // The plan was made in the parent, with the metrics that follow VMAF's place already placed.
    const RunFrames run{environment,   source,      distorted,      source_crop,
                        distorted_crop, plan.width, plan.height,    plan.bit_depth,
                        request.at("total_frames").get<long long>(), decoding, cancel, std::nullopt};
    const ParsedScores scores = score_plan(
        run, options_from(request.at("options")), plan,
        [](long long done, long long whole, double fps) {
            send_isolated_message({{"progress", {done, whole, fps}}});
        },
        [](const RunStatus& status) { send_isolated_message({{"status", status_json(status)}}); });
    std::ofstream file(folder / scores_file, std::ios::binary);
    file.write(reinterpret_cast<const char*>(scores.frame.data()),
               static_cast<std::streamsize>(scores.frame.size() * sizeof(int)));
    for (const std::string& key : scores.order) {
        const std::vector<float>& values = scores.metrics.at(key);
        file.write(reinterpret_cast<const char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    std::vector<std::string> exact;
    for (const auto& [key, values] : scores.exact) {
        if (std::find(scores.order.begin(), scores.order.end(), key) != scores.order.end() &&
            values.size() == scores.frame.size()) {
            file.write(reinterpret_cast<const char*>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(double)));
            exact.push_back(key);
        }
    }
    if (!file) {
        throw std::runtime_error("The scores could not be written.");
    }
    return {{"frames", scores.frame.size()},
            {"order", scores.order},
            {"features_on_gpu", scores.features_on_gpu},
            {"exact", exact}};
}

bool follows_vulkan_engine(std::string_view key) {
    return engine_takes_cpu_metric(key) || key == "vmaf_yuv" || key == vmaf_u_key || key == vmaf_v_key ||
           key == "vmaf_hdr";
}

EnginePlace engine_place(const CancelToken& cancel) {
    const GpuVmafBackend backend = gpu_vmaf_backend(cancel);
    if (backend.backend != "vulkan") {
        return {};
    }
    return {true, backend.device.value_or(-1)};
}

namespace {

// The crops of a run as its black-bar handling asks (crops_for), each video
// decoded the way the run will (`decoding`).
std::pair<std::optional<CropBox>, std::optional<CropBox>> run_crops(const RunEnvironment& environment,
                                                                    const VideoInfo& source, const VideoInfo& distorted,
                                                                    const VmafOptions& options,
                                                                    const HardwareDecoding& decoding,
                                                                    const StatusCallback& on_status,
                                                                    const CancelToken& cancel) {
    if (options.crop_mode == CropMode::none) {
        return {};
    }
    // Said, or the run line would keep the step before: a long video's
    // detection takes a while.
    if (on_status && detects_black_bars(options.crop_mode)) {
        on_status({options.crop_mode == CropMode::source_only ? "Detecting black bars in source..."
                                                               : "Detecting black bars in source and distorted..."});
    }
    try {
        return crops_for(options.crop_choice(), environment.ffmpeg, source, distorted, cancel, decoding.source,
                         decoding.distorted, options.scale_algorithm);
    } catch (const CropEntryError& error) {
        throw VmafRunError(error.what());
    }
}

// The VMAF models the GPU scores (`model` VMAF's, at `depth` bits and
// `width` x `height`). VMAF is scored on the GPU only when VMAF, NEG and
// VMAF v1 are all `options` ask for: the command line puts them in a run of
// their own, apart from FFmpeg's other metrics, and a run that scores both
// asks for its GPU's part alone (score_gpu_and_cpu). None when one is
// calculated on the CPU: FFmpeg's libvmaf calculates them all together. A
// check of the GPU not made yet runs under the run's `cancel`.
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
    } else if (uses_vmaf_model(key)) {
        parameters["model"] = nlohmann::json(model).dump();
    } else if (key == "vmaf_v1") {
        parameters["model"] = nlohmann::json(options.model_v1).dump();
    } else if (key == "vmaf_hdr") {
        parameters["model"] = nlohmann::json(options.model_hdr).dump();
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
        // every GPU, as HDR VMAF's features (its pictures converted by libvmaf on the CPU), CIEDE2000,
        // CAMBI and MS-SSIM (VMAF v1's engine without VMAF's features); VMAF YUV's U and V
        // on VMAF's Vulkan engine, beside VMAF there (ChromaVmaf).
        const bool v1_engine = key == "vmaf_v1" || key == "vmaf_hdr" || engine_takes_cpu_metric(key);
        const bool chroma_vmaf = key == "vmaf_yuv_u" || key == "vmaf_yuv_v";
        const std::string backend = v1_engine || chroma_vmaf ? "vulkan" : gpu_backend;
        provenance.implementation = "libvmaf/" + backend;
        provenance.compute_backend = "gpu";
        provenance.implementation_version = v1_engine               ? v1_library_build()
                                            : backend == "vulkan" ? vulkan_library_build()
                                                                  : cuda_library_build();
    }
    return provenance;
}

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

// Files the CPU metrics the GPU calculated (engine_takes_cpu_metric's and VMAF YUV's U and V, CpuScorer's
// features_on_gpu) under the GPU.
void file_gpu_metrics(RunScores& scores) {
    for (const ParsedScores& part : scores.parts) {
        for (const std::string& key : part.features_on_gpu) {
            if (engine_takes_cpu_metric(key) || key == "vmaf_yuv_u" || key == "vmaf_yuv_v") {
                scores.cpu_keys.erase(key);
                scores.gpu_keys.insert(key);
            }
        }
    }
}

// Files HDR VMAF's scores (in `scored`) under where its features were calculated: VMAF v1's engine on the GPU, or
// the bundled libvmaf on the CPU, where HdrScorer fell back to it or was not asked for the GPU.
void insert_hdr_key(RunScores& scores, const ParsedScores& scored) {
    const bool on_gpu = std::find(scored.features_on_gpu.begin(), scored.features_on_gpu.end(), "vmaf_hdr") !=
                        scored.features_on_gpu.end();
    (on_gpu ? scores.gpu_keys : scores.cpu_keys).insert("vmaf_hdr");
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
    GpuPlan plan;
    plan.models = gpu;
    plan.width = run.width;
    plan.height = run.height;
    plan.bit_depth = run.depth;
    plan.backend = backend.backend == "vulkan" ? "vulkan" : "cuda";
    plan.device = backend.backend == "vulkan" && backend.device ? *backend.device : -1;
    return plan;
}

// Where the run's metrics that follow VMAF's place go (follows_vulkan_engine): where its plan counted them, or,
// for a run without one, where the GPU's check says.
EnginePlace run_engine_place(const RunFrames& run) {
    return run.engine ? *run.engine : engine_place(run.cancel);
}

// The plan for the run's CPU metrics in the app (cpu_metrics_plan).
// VMAF YUV's U and V are scored with the run's VMAF model. They, CIEDE2000, CAMBI and MS-SSIM (with MS-SSIM YUV)
// go to the GPU VMAF runs on
// where the Vulkan engine scores VMAF there (CpuScorer falls back to libvmaf where the GPU cannot take them): with
// the GPU there they took a fraction of the time and memory libvmaf takes on the CPU. On a GPU whose VMAF is
// libvmaf's CUDA code the Vulkan engine's would not be checked, so they stay on the CPU there.
std::optional<GpuPlan> cpu_plan(const RunFrames& run, const VmafOptions& options) {
    std::optional<GpuPlan> plan =
        cpu_metrics_plan(options, std::pair(run.width, run.height), run.source, run.distorted, run.decoding,
                         run.source_crop, run.distorted_crop, run.environment.cores, run.environment.ffmpeg);
    if (!plan) {
        return plan;
    }
    const bool chroma_vmaf = options.metric_enabled(vmaf_u_key) || options.metric_enabled(vmaf_v_key);
    if (chroma_vmaf) {
        plan->vmaf_model = run_vmaf_model(options, run.width, run.height);
    }
    // What the engine's place serves in this plan: U and V (vmaf_on_vulkan), engine_takes_cpu_metric's
    // (extra_gpu). VMAF YUV with its Y alone needs neither.
    const bool engine_metrics = std::any_of(plan->models.begin(), plan->models.end(),
                                            [](const auto& model) { return engine_takes_cpu_metric(model.first); });
    if (options.vmaf_on_gpu && (chroma_vmaf || engine_metrics)) {
        if (const EnginePlace place = run_engine_place(run); place.on_engine) {
            plan->vmaf_on_vulkan = chroma_vmaf;
            plan->extra_gpu = place.device;
        }
    }
    return plan;
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

// The scores of a run with metrics FFmpeg cannot score (app_only_metrics:
// PSNR YUV, SSIM YUV, XPSNR above 8 bits) beside metrics FFmpeg scores (VMAF
// on the CPU, say): those in the app (run_cpu_metrics) on a thread beside
// FFmpeg's run of the others, as score_apart scores two parts. Nothing where
// the app cannot score them (FFmpeg's run then reports them failed:
// without_app_only).
std::optional<RunScores> score_app_only_beside(const RunFrames& run, const VmafOptions& options,
                                               const std::string& model, const ProgressCallback& on_progress,
                                               const StatusCallback& on_status) {
    const std::map<std::string, std::string> refused = app_only_metrics(options, run.depth);
    if (refused.empty()) {
        return std::nullopt;
    }
    const VmafOptions others = without_metrics(options, refused);
    VmafOptions app_only = options;
    for (const std::string& key : options.requested_metrics()) {
        app_only.set_metric_enabled(key, refused.contains(key));
    }
    const std::optional<GpuPlan> plan = cpu_plan(run, app_only);
    if (others.requested_metrics().empty() || !plan) {
        return std::nullopt;
    }
    SlowerProgress progress(on_progress, 2);
    bool in_app = false;  // set on the app's thread, read once it has ended
    ScoresBeside app_part(
        [&, app_progress = progress.part(1)](const CancelToken& stop) {
            std::string failure;
            std::optional<ParsedScores> scored =
                run_cpu_metrics(run.with_cancel(stop), app_only, *plan, app_progress, nullptr, &failure);
            if (!scored) {
                scored.emplace();  // it failed in the app, and FFmpeg cannot score them
                for (const auto& [key, reason] : refused) {
                    scored->failed[key] = std::string(metric_definition(key).label) + " in the app failed (" +
                                          failure + "). " + reason;
                }
            }
            in_app = !scored->frame.empty();
            return std::move(*scored);
        },
        run.cancel);
    RunScores scores;
    scores.parts.push_back(execute_run(run, others, model, progress.part(0), on_status));
    app_part.join();
    run.cancel.throw_if_cancelled();
    scores.parts.push_back(app_part.scores_or_failed(app_only.requested_metrics()));
    if (in_app) {
        insert_keys(scores.cpu_keys, *plan);
    }
    return scores;
}

// The scores of a run in one part: VMAF, NEG and VMAF v1 on the GPU when they
// are all it scores and the GPU scores them; PSNR, SSIM and XPSNR in the app
// when they are all it scores; otherwise, or when either fails, FFmpeg's
// (the metrics FFmpeg cannot score in the app beside it: score_app_only_beside).
RunScores score_alone(const RunFrames& run, const VmafOptions& options, const std::string& model,
                      const ProgressCallback& on_progress, const StatusCallback& on_status) {
    RunScores scores;
    std::optional<ParsedScores> scored;
    if (const std::optional<GpuPlan> plan = gpu_plan(run, options, model)) {
        scores.gpu_backend = plan->backend;
        scored = run_on_gpu(run, options, *plan, on_progress, on_status);
        if (scored) {
            insert_keys(scores.gpu_keys, *plan);
        }
    }
    if (!scored) {
        if (const std::optional<GpuPlan> cpu = cpu_plan(run, options)) {
            scored = run_cpu_metrics(run, options, *cpu, on_progress, on_status);
            if (scored) {
                insert_keys(scores.cpu_keys, *cpu);
            }
        }
    }
    if (!scored) {
        if (std::optional<RunScores> apart = score_app_only_beside(run, options, model, on_progress, on_status)) {
            return std::move(*apart);
        }
        scored = execute_run(run, options, model, on_progress, on_status);
    }
    scores.parts.push_back(std::move(*scored));
    return scores;
}

// Whether the GPU's pass (`gpu`) can score the run's CPU metrics (`cpu`)
// beside its own, from the frames it decodes, exactly as their own run
// would: pictures that reach the CPU's scorer as in that run
// (pass_feeds_cpu_metrics); every CPU metric of the run in `cpu`, none left
// to an FFmpeg of its own (run_cpu_metrics's XPSNR beside); scored for the
// GPU's frames, which XPSNR alone with a frame subsample is not
// (run_cpu_metrics scores it for every frame); and not to libvmaf's CUDA code
// without VMAF v1 (score_decoded_cuda).
bool shares_pass(const RunFrames& run, const GpuPlan& gpu, const GpuPlan& cpu, const VmafOptions& cpu_options) {
    if (!pass_feeds_cpu_metrics(run.source, run.distorted, run.decoding)) {
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
    return !cpu_metrics_need_every_frame(keys, cpu_options.n_subsample);
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
        if (auto exact = scores.exact.extract(key)) {
            taken.exact.insert(std::move(exact));
        }
        taken.order.push_back(key);
        if (const auto on_gpu = std::find(scores.features_on_gpu.begin(), scores.features_on_gpu.end(), key);
            on_gpu != scores.features_on_gpu.end()) {
            taken.features_on_gpu.push_back(key);
            scores.features_on_gpu.erase(on_gpu);
        }
    }
    scores.order = std::move(kept);
    return taken;
}

// The GPU for the CPU metrics' CIEDE2000, CAMBI and MS-SSIM (cpu.extra_gpu) when they are scored beside `pass`: the
// GPU VMAF's own on Vulkan, else the one VMAF would run on.
std::optional<int> beside_gpu(const GpuPlan& cpu, const GpuPlan& pass) {
    if (cpu.extra_gpu && pass.backend == "vulkan" && pass.device >= 0) {
        return pass.device;
    }
    return cpu.extra_gpu;
}

// Puts the CPU metrics' plan (`cpu`) beside the GPU pass's (`pass`): its metrics and how CpuScorer scores them.
void put_beside(GpuPlan& pass, const GpuPlan& cpu) {
    pass.beside = plan_keys(cpu);
    pass.threads = cpu.threads;
    pass.frame_rate = cpu.frame_rate;
    pass.cambi = cpu.cambi;
    pass.extra_gpu = beside_gpu(cpu, pass);
    pass.vmaf_model = cpu.vmaf_model;
    pass.vmaf_on_vulkan = cpu.vmaf_on_vulkan;
}

// The run's GPU metrics (`gpu`) with its CPU metrics (`cpu`) beside them in
// one GPU pass, from one decode of each video (shares_pass). Throws what
// score_on_gpu throws.
RunScores score_shared(const RunFrames& run, const VmafOptions& options, const GpuPlan& gpu, const GpuPlan& cpu,
                       const ProgressCallback& on_progress, const StatusCallback& on_status) {
    GpuPlan plan = gpu;
    put_beside(plan, cpu);
    ParsedScores scored = score_on_gpu(run, options, plan, on_progress, on_status);
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
                if (std::optional<ParsedScores> scored =
                        run_cpu_metrics(run.with_cancel(stop), cpu_options, *cpu, cpu_progress, nullptr)) {
                    in_app = true;
                    return std::move(*scored);
                }
            }
            try {
                return execute_run(run.with_cancel(stop), cpu_options, "", cpu_progress, nullptr);
            } catch (const AppOnlyUnavailable& error) {
                ParsedScores none;  // they were all: they fail, the GPU's metrics stand
                none.failed.insert(error.refused.begin(), error.refused.end());
                return none;
            }
        },
        run.cancel);
    RunScores scores;
    scores.gpu_backend = gpu.backend;
    std::optional<ParsedScores> gpu_scores = run_on_gpu(run, gpu_options, gpu, progress.part(0), on_status);
    if (gpu_scores) {
        insert_keys(scores.gpu_keys, gpu);
    } else {
        gpu_scores = execute_run(run, gpu_options, model, progress.part(0), on_status);
    }
    cpu_part.join();
    run.cancel.throw_if_cancelled();
    scores.parts.push_back(cpu_part.scores_or_failed(cpu_options.requested_metrics()));
    if (in_app) {
        insert_keys(scores.cpu_keys, *cpu);
    }
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
        } catch (const std::exception& error) {
            if (ends_the_run(error)) {
                throw;
            }
            logger.warning("VMAF on the GPU with " + metric_names(*cpu) +
                           " beside it failed; scoring them apart: " + error.what());
        }
    }
    return score_apart(run, gpu_options, cpu_options, *gpu, cpu, model, on_progress, on_status);
}

// HDR VMAF's plan for the run (backend "hdr"), or why it fails for this pair: videos it does not take (not PQ, say),
// or a model it cannot load or that refuses the size. Neither decodes anything.
std::variant<GpuPlan, std::string> hdr_plan(const RunFrames& run, const VmafOptions& options) {
    const std::variant<HdrColours, std::string> colours = hdr_colours(run.source, run.distorted);
    if (const auto* refusal = std::get_if<std::string>(&colours)) {
        return *refusal;
    }
    GpuPlan plan;
    plan.backend = "hdr";
    plan.width = run.width;
    plan.height = run.height;
    plan.bit_depth = run.depth;
    plan.hdr_colours = std::get<HdrColours>(colours);
    // libvmaf's threads when its features are on the CPU: half the cores by default, beside its conversion threads.
    // Each keeps its own 4K feature buffers: on the Core Ultra 9 285K, 12 threads scored 4K as fast as 24 (8.5 s
    // against 9.0 s for 480 frames) with 3.2 GB less memory (5.0 GB against 8.1 GB).
    plan.hdr_threads =
        options.n_threads > 0 ? options.n_threads : std::max(1, static_cast<int>(run.environment.cores) / 2);
    if (options.vmaf_on_gpu) {
        // Its features on VMAF v1's Vulkan engine, on the GPU VMAF runs on, where the Vulkan engine scores VMAF
        // there (HdrScorer falls back to the CPU where it cannot). Where the engine failed its check, or VMAF is
        // libvmaf's CUDA code, it is not checked on this GPU: the features stay on the CPU, as the run's plan
        // counts them (heavy_cpu_unit), as do CIEDE2000, CAMBI and MS-SSIM (cpu_plan).
        if (const EnginePlace place = run_engine_place(run); place.on_engine) {
            plan.hdr_gpu = place.device;
        }
    }
    try {
        const std::optional<std::string_view> file = model_file_of(options.model_hdr);
        if (!file) {
            throw std::invalid_argument("Not an HDR VMAF model: " + options.model_hdr);
        }
        plan.models.emplace_back("vmaf_hdr", std::string(*file));
        plan.hdr_model_file = std::string(*file);
        // Which model scores HDR VMAF: Auto's pick for the size compared, or the one chosen.
        logger.info("HDR VMAF model: " + model_shown(options.model_hdr) +
                    (options.model_choice_hdr == auto_model_choice
                         ? " (Auto's choice for " + std::to_string(run.width) + "x" + std::to_string(run.height) + ")"
                         : " (chosen in the options)"));
        const HdrModelTarget model = hdr_model_target(platform::path_from_text(*file));
        hdr_targets(static_cast<unsigned>(run.width), static_cast<unsigned>(run.height),
                    static_cast<unsigned>(run.width), static_cast<unsigned>(run.height), model);
    } catch (const std::exception& error) {
        logger.error(std::string("HDR VMAF failed: ") + error.what());
        return std::string("HDR VMAF failed: ") + error.what();
    }
    return plan;
}

// The log line of HDR VMAF's libvmaf and threads, as a run starts it (HdrScorer logs where its features and its
// pictures' conversion go).
void log_hdr_vmaf(const GpuPlan& plan) {
    const std::string converters =
        std::to_string(hdr_conversion_threads(std::thread::hardware_concurrency(), plan.hdr_gpu.has_value()));
    if (plan.hdr_gpu) {
        logger.info("HDR VMAF in the app: its features on the GPU where the engine takes them, its pictures "
                    "converted on the GPU where it can, else on " + converters + " CPU threads; " +
                    cpu_library_build() + ", " + std::to_string(plan.hdr_threads) + " threads");
    } else {
        logger.info("HDR VMAF in the app: " + cpu_library_build() + ", " + std::to_string(plan.hdr_threads) +
                    " threads, its pictures converted on " + converters);
    }
}

// HDR VMAF's part of a run (vmaf_hdr.h, `planned` by hdr_plan): scored in the app, in a process of its own where one
// is set, on the frames the other metrics compare (cropped, and at the size they are compared at), decoded by the
// app's decoders or by FFmpeg as theirs are. A pair it cannot score (`planned`'s failure), or a run of it that fails,
// fails HDR VMAF alone (the part's `failed`); Cancel and what makes the whole run again or refuses it end the run.
ParsedScores score_hdr(const RunFrames& run, const VmafOptions& options,
                       const std::variant<GpuPlan, std::string>& planned, const ProgressCallback& on_progress,
                       const StatusCallback& on_status) {
    ParsedScores refused;
    if (const auto* failure = std::get_if<std::string>(&planned)) {
        refused.failed["vmaf_hdr"] = *failure;
        return refused;
    }
    const GpuPlan& plan = std::get<GpuPlan>(planned);
    try {
        log_hdr_vmaf(plan);
        return score_cpu_plan(run, options, plan, on_progress, on_status);
    } catch (const std::exception& error) {
        if (ends_the_run(error)) {
            throw;
        }
        logger.error(std::string("HDR VMAF failed: ") + error.what());
        refused.failed["vmaf_hdr"] = std::string("HDR VMAF failed: ") + error.what();
        return refused;
    }
}

// HDR VMAF (`hdr`, hdr_plan's) beside the run's other metrics (`others`: VMAF, NEG and VMAF v1 on the GPU, PSNR, SSIM
// and XPSNR in the app), in one pass from one decode of each video, its scores the same as in a part of its own (it
// reads the frames that part would: shares_pass's rules, as for the CPU's metrics). The GPU's pass where the GPU scores
// VMAF, else HDR VMAF's own with the CPU's metrics beside. Nothing where they cannot share a pass, or where the pass
// fails for a reason that would not end the parts too: the parts are then scored one after the other.
std::optional<RunScores> score_hdr_beside(const RunFrames& run, const VmafOptions& others, const std::string& model,
                                          const GpuPlan& hdr, const ProgressCallback& on_progress,
                                          const StatusCallback& on_status) {
    const std::vector<std::string> requested = others.requested_metrics();
    const bool gpu_wanted = std::any_of(requested.begin(), requested.end(),
                                        [](const std::string& key) { return !is_cpu_metric(key); });
    const bool cpu_wanted = std::any_of(requested.begin(), requested.end(),
                                        [](const std::string& key) { return is_cpu_metric(key); });
    std::optional<GpuPlan> gpu;
    if (gpu_wanted) {
        gpu = gpu_plan(run, part_of(others, false), model);
        if (!gpu) {
            return std::nullopt;  // FFmpeg's libvmaf calculates them on the CPU
        }
    }
    const VmafOptions cpu_options = part_of(others, true);
    std::optional<GpuPlan> cpu;
    if (cpu_wanted) {
        cpu = cpu_plan(run, cpu_options);
        if (!cpu) {
            return std::nullopt;  // FFmpeg's filters score them
        }
    }
    if (!shares_pass(run, gpu ? *gpu : hdr, cpu ? *cpu : GpuPlan{}, cpu_options)) {
        return std::nullopt;
    }
    GpuPlan plan = gpu ? *gpu : hdr;
    if (cpu) {
        put_beside(plan, *cpu);
    }
    if (gpu) {
        plan.beside.push_back("vmaf_hdr");
        plan.hdr_colours = hdr.hdr_colours;
        plan.hdr_model_file = hdr.hdr_model_file;
        plan.hdr_threads = hdr.hdr_threads;
        plan.hdr_gpu = hdr.hdr_gpu;
    }
    try {
        log_hdr_vmaf(hdr);
        ParsedScores scored = gpu ? score_on_gpu(run, others, plan, on_progress, on_status)
                                  : score_cpu_plan(run, others, plan, on_progress, on_status);
        RunScores scores;
        if (cpu) {
            scores.parts.push_back(take_metrics(scored, plan_keys(*cpu)));
            insert_keys(scores.cpu_keys, *cpu);
        }
        insert_hdr_key(scores, scored);
        if (gpu) {
            scores.parts.push_back(take_metrics(scored, {"vmaf_hdr"}));
            insert_keys(scores.gpu_keys, *gpu);
            scores.gpu_backend = gpu->backend;
        }
        scores.parts.push_back(std::move(scored));
        return scores;
    } catch (const std::exception& error) {
        if (ends_the_run(error)) {
            throw;
        }
        logger.warning("HDR VMAF beside the run's other metrics failed; scoring them apart: " +
                       std::string(error.what()));
        return std::nullopt;
    }
}

// The frames one part's scores span; 0 for a part with none. Its step is
// that of its frame numbers (some metrics score every frame of a subsampled
// run), or the run's for a single score. Scores of every step-th frame
// span up to step - 1 frames past the last score, which may not be there:
// the estimate caps those, unless the container gave no length (the
// estimate is then 1).
long long scored_span(const ParsedScores& part, long long estimate, int run_step) {
    if (part.frame.empty()) {
        return 0;
    }
    const long long step = part.frame.size() > 1 ? part.frame[1] - part.frame[0] : run_step;
    const long long scored = static_cast<long long>(part.frame.back()) - part.frame.front() + 1;
    if (step <= 1 || estimate <= 1) {
        return scored;
    }
    return std::max(scored, std::min(scored - 1 + step, estimate));
}

// The frames a run compared, for its result, its pass's fps and Video
// Compare's frame box, once its parts' scores are in: the most any part
// spans, not the estimate it started from (estimate_total_frames), which a
// video shorter or longer than its length says gets wrong. The estimate
// stays where no part has scores.
void count_scored_frames(ComparisonResult& result, std::span<const ParsedScores> parts, int run_step) {
    const long long estimate = result.compared_frame_count;
    long long compared = 0;
    for (const ParsedScores& part : parts) {
        compared = std::max(compared, scored_span(part, estimate, run_step));
    }
    if (compared > 0) {
        result.compared_frame_count = compared;
    }
}

// What a comparison's result says of the videos and the options, before its scores.
ComparisonResult result_of_pair(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                                const std::string& model, const std::optional<CropBox>& source_crop,
                                const std::optional<CropBox>& distorted_crop, long long total_frames) {
    ComparisonResult result;
    result.source = source.path;
    result.fps = distorted.fps;
    result.model = model;
    result.source_crop = source_crop;
    result.distorted_crop = distorted_crop;
    result.source_info = source;
    result.distorted_info = distorted;
    result.scale_algorithm = options.scale_algorithm;
    result.compared_frame_count = total_frames;
    result.model_choice = options.model_choice;
    result.model_v1 = options.model_v1;
    if (options.compute_vmaf_v1) {
        result.model_choice_v1 = options.model_choice_v1;
    }
    if (options.compute_vmaf_hdr) {
        result.model_hdr = options.model_hdr;
        result.model_choice_hdr = options.model_choice_hdr;
    }
    return result;
}

// The run's metrics in `options` (score_pair_once's): HDR VMAF in the
// others' pass where it can (score_hdr_beside); else a part of its own, after
// the others (score_hdr), they scored as they would be without it.
RunScores score_run_metrics(const RunFrames& run, const VmafOptions& options, const std::string& model,
                            const ProgressCallback& on_progress, const StatusCallback& on_status) {
    VmafOptions others = options;
    others.compute_vmaf_hdr = false;
    const bool with_others = !others.requested_metrics().empty();
    std::optional<std::variant<GpuPlan, std::string>> hdr;
    if (options.compute_vmaf_hdr) {
        hdr = hdr_plan(run, options);
    }
    std::optional<RunScores> scores;
    if (with_others && hdr && std::holds_alternative<GpuPlan>(*hdr)) {
        scores = score_hdr_beside(run, others, model, std::get<GpuPlan>(*hdr), on_progress, on_status);
    }
    const bool hdr_apart = hdr && !scores;
    const long long parts = (with_others ? 1 : 0) + (hdr_apart ? 1 : 0);
    if (!scores && with_others) {
        const ProgressCallback progress = part_in_sequence(on_progress, 0, parts);
        scores = score_gpu_and_cpu(run, others, model, progress, on_status);
        if (!scores) {
            scores = score_alone(run, others, model, progress, on_status);
        }
    } else if (!scores) {
        scores.emplace();
    }
    if (hdr_apart) {
        VmafOptions hdr_only = options;
        for (const std::string& key : others.requested_metrics()) {
            hdr_only.set_metric_enabled(key, false);
        }
        ParsedScores hdr_part =
            score_hdr(run, hdr_only, *hdr, part_in_sequence(on_progress, parts - 1, parts), on_status);
        if (hdr_part.has("vmaf_hdr")) {
            insert_hdr_key(*scores, hdr_part);
        }
        scores->parts.push_back(std::move(hdr_part));
    }
    return std::move(*scores);
}

// `options` with only `keys` of its metrics asked for.
VmafOptions with_metrics_only(const VmafOptions& options, const std::vector<std::string>& keys) {
    VmafOptions only = options;
    for (const std::string& key : options.requested_metrics()) {
        only.set_metric_enabled(key, std::find(keys.begin(), keys.end(), key) != keys.end());
    }
    return only;
}

// score_run_metrics; where a pass of it does not fit in memory (MemoryShort,
// memory_fit.h), its metrics in two passes of half of them each, one after
// the other, a half that does not fit halved again. Each metric is scored
// from the same frames by the same code as in one pass.
RunScores score_fitting(const RunFrames& run, const VmafOptions& options, const std::string& model,
                        const ProgressCallback& on_progress, const StatusCallback& on_status) {
    std::optional<std::pair<std::vector<std::string>, std::vector<std::string>>> halves;
    try {
        return score_run_metrics(run, options, model, on_progress, on_status);
    } catch (const MemoryShort& error) {
        halves = halve_metrics(options.requested_metrics(), error.gpu());
        if (!halves) {
            throw;  // only a pass that could be split throws it (score_watched): never here
        }
        logger.warning(std::string(error.what()) + ": scoring the metrics in two passes, " + join(halves->first, ", ") +
                       ", then " + join(halves->second, ", "));
        if (on_status) {
            on_status({"Not enough memory for all the metrics at once (" + std::string(error.what()) +
                       "); calculating them in two passes\xE2\x80\xA6"});
        }
    }
    RunScores scores = score_fitting(run, with_metrics_only(options, halves->first), model,
                                     part_in_sequence(on_progress, 0, 2), on_status);
    RunScores second = score_fitting(run, with_metrics_only(options, halves->second), model,
                                     part_in_sequence(on_progress, 1, 2), on_status);
    if (scores.gpu_keys.empty()) {
        scores.gpu_backend = second.gpu_backend;
    }
    std::move(second.parts.begin(), second.parts.end(), std::back_inserter(scores.parts));
    scores.cpu_keys.merge(second.cpu_keys);
    scores.gpu_keys.merge(second.gpu_keys);
    return scores;
}

// score_pair's run, as its reading of the videos is set.
ComparisonResult score_pair_once(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                                 VmafOptions options, const std::optional<CropBox>& source_crop,
                                 const std::optional<CropBox>& distorted_crop, const HardwareDecoding& decoding,
                                 const ProgressCallback& on_progress, const StatusCallback& on_status,
                                 const CancelToken& cancel,
                                 const std::optional<std::filesystem::path>& result_distorted_path,
                                 const std::optional<EnginePlace>& engine) {
    if (options.compute_vmaf_yuv || options.compute_vmaf_yuv_y) {
        // Made from VMAF and its U and V by the video's job (JobRun::merged), which asks for those instead.
        throw std::invalid_argument("VMAF YUV and its Y are not scored by a run: they are made from VMAF and "
                                    "its U and V");
    }
    const auto [width, height] = analysis_dimensions(source, distorted, options, source_crop, distorted_crop);
    const std::string model = resolve_run_models(options, width, height);
    // The range's frames counted at the source's rate, as the run cuts them (run_range).
    const long long total_frames = estimate_total_frames(source, options, &distorted);
    const long long decoded_frames = estimate_decoded_frames(source, options, &distorted);
    const FrameRange range = run_range(options, source);
    const long long first_frame = range.start;

    const int depth = bit_depth(analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt}));
    const RunFrames run{environment, source,         distorted, source_crop, distorted_crop, width,
                        height,      depth,          decoded_frames, decoding, cancel,         engine};
    std::optional<RunScores> scores = score_fitting(run, options, model, on_progress, on_status);
    for (const ParsedScores& part : scores->parts) {
        if (part.order.empty() && !part.failed.empty()) {
            continue;  // a part whose only metric failed: the others stand
        }
        const long long compared = part.frame.empty() ? 0 : static_cast<long long>(part.frame.back()) + 1 - first_frame;
        if (const std::optional<std::string> short_run =
                short_comparison(total_frames, compared, first_frame, distorted.fps, options.n_subsample)) {
            throw VmafRunError(*short_run);
        }
    }

    file_gpu_metrics(*scores);
    ComparisonResult result =
        result_of_pair(source, distorted, options, model, source_crop, distorted_crop, total_frames);
    result.distorted = result_distorted_path.value_or(distorted.path);
    result.scale_direction = options.scale_direction;
    for (const ParsedScores& part : scores->parts) {
        for (const std::string& key : part.order) {
            // VMAF YUV's U and V have VMAF's model also in a task without VMAF (`model` empty).
            const std::string key_model = uses_vmaf_model(key) ? run_vmaf_model(options, width, height) : model;
            MetricProvenance provenance = run_provenance(key, key_model, options, environment.ffmpeg_version,
                                                         scores->cpu_keys, scores->gpu_keys, scores->gpu_backend);
            mark_range(provenance, range);
            FrameMetricResult metric{key, part.frame, part.time, part.metrics.at(key), std::move(provenance)};
            if (const auto exact = part.exact.find(key);
                exact != part.exact.end() && exact->second.size() == metric.values.size()) {
                metric.exact = exact->second;
            }
            result.metric_results.add(std::move(metric));
        }
        result.metric_failures.insert(part.failed.begin(), part.failed.end());
    }
    count_scored_frames(result, scores->parts, options.n_subsample);
    if (!result.metric_results.frame("vmaf_hdr")) {
        result.model_hdr.clear();  // no HDR VMAF score: no model to record with it
        result.model_choice_hdr.reset();
    }
    return result;
}

// The pair's scores with its crops known (run_vmaf_once), and its result. A run
// that fails where FFmpeg may have read a range from a seek, which it cannot check
// picture by picture (ffmpeg_range_seek), is made again from the videos' start: a
// video really cut short then fails for good.
ComparisonResult score_pair(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                            const VmafOptions& options, const std::optional<CropBox>& source_crop,
                            const std::optional<CropBox>& distorted_crop, const HardwareDecoding& decoding,
                            const ProgressCallback& on_progress, const StatusCallback& on_status,
                            const CancelToken& cancel,
                            const std::optional<std::filesystem::path>& result_distorted_path,
                            const std::optional<EnginePlace>& engine) {
    try {
        return score_pair_once(environment, source, distorted, options, source_crop, distorted_crop, decoding,
                               on_progress, on_status, cancel, result_distorted_path, engine);
    } catch (const VmafRunError& error) {
        // (The proofs are kept: this asks for what the run was given.)
        const RangeSeeks seeks = reading_near_range(options, environment, source, distorted, cancel).range_seeks;
        if (!seeks.source && !seeks.distorted) {
            throw;  // read from the videos' start already: the same again would fail the same way
        }
        turn_off_range_seek(error.what());
    }
    return score_pair_once(environment, source, distorted, options, source_crop, distorted_crop, decoding, on_progress,
                           on_status, cancel, result_distorted_path, engine);
}

// Logs why each video the plan leaves off the GPU is decoded in software, as
// the run's status line says only that it is ("cpu"). Not VVC: the plan never
// takes it, as VVdeC or vvdec-vulkan decodes it, and they log that themselves.
// `mac` is the plan's.
void log_software_decoding(const HardwareDecoding& decoding, const VideoInfo& source, const VideoInfo& distorted,
                           MacGpuDecoding mac) {
    const auto log_video = [mac](const std::optional<std::string>& hwaccel, const VideoInfo& video,
                                 const char* label) {
        if (!hwaccel && ascii_lower(video.codec_name) != "vvc") {
            logger.info(std::string(label) + " is decoded in software: " +
                        software_decoding_reason(video, true, mac));
        }
    };
    log_video(decoding.source, source, "The source");
    log_video(decoding.distorted, distorted, "The test video");
}

// Throws VmafRunError when pictures compared at `size` are too small for one
// of the run's metrics (too_small_problem): libvmaf's VMAF would read past
// them on the CPU, and SSIM has no score under its window.
void refuse_too_small(const VmafOptions& options, Size size) {
    if (const std::optional<std::string> problem =
            too_small_problem(options.requested_metrics(), size.first, size.second)) {
        throw VmafRunError(*problem);
    }
}

// The app's metrics that cannot score pictures of this size and bit depth
// (cpu_metrics_unfit: too small, too large or too deep for them), turned off
// in `options` and logged, to be reported failed alone: in libvmaf one
// failing would fail the others scored with it. Throws VmafRunError when no
// metric is left.
std::map<std::string, std::string> refuse_unfit_metrics(VmafOptions& options, Size size, int bit_depth,
                                                        const CambiInput& cambi) {
    const std::map<std::string, std::string> unfit =
        cpu_metrics_unfit(options.requested_metrics(), size.first, size.second, bit_depth, cambi);
    for (const auto& [key, reason] : unfit) {
        logger.warning(reason);
        options.set_metric_enabled(key, false);
    }
    if (!unfit.empty() && options.requested_metrics().empty()) {
        throw VmafRunError(unfit.begin()->second);
    }
    return unfit;
}

// run_vmaf's run of the pair as it is marked (size_changes.h). Throws
// VideoSizeChanged when FFmpeg rebuilt its filters for one of them.
ComparisonResult run_vmaf_once(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                               VmafOptions options, const ProgressCallback& on_progress,
                               const StatusCallback& on_status, const CancelToken& cancel,
                               const std::optional<std::filesystem::path>& result_distorted_path,
                               const std::optional<EnginePlace>& engine) {
    validate_video_pair(source, distorted, options);
    // The decode plan comes before black-bar detection, so the detection
    // decodes the way the run will.
    HardwareDecoding decoding;
    if (options.gpu_decode) {
        // A Mac decodes only video deeper than 8 bits on its GPU here (MacGpuDecoding)
        constexpr MacGpuDecoding mac = MacGpuDecoding::deep;
        decoding = plan_hardware_decoding(environment.decoding, options.gpu_vendor, source, distorted, mac);
        log_software_decoding(decoding, source, distorted, mac);
        if (on_status) {
            on_status(decoding_why_status(decoding, source, distorted, mac));
        }
    }
    const auto [source_crop, distorted_crop] = run_crops(environment, source, distorted, options, decoding, on_status, cancel);
    // After cropping: removing a letterbox is what makes a letterboxed source
    // and a cropped encode the same shape.
    validate_display_geometry(source, distorted, source_crop, distorted_crop);
    const Size size = analysis_dimensions(source, distorted, options, source_crop, distorted_crop);
    refuse_too_small(options, size);
    const std::map<std::string, std::string> unfit =
        refuse_unfit_metrics(options, size, bit_depth(analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt})),
                             cambi_input(distorted, distorted_crop));
    ComparisonResult result = score_pair(environment, source, distorted, std::move(options), source_crop,
                                         distorted_crop, decoding, on_progress, on_status, cancel,
                                         result_distorted_path, engine);
    result.metric_failures.insert(unfit.begin(), unfit.end());
    return result;
}

// `options` with the videos decoded on the CPU, the GPU's video decoding
// having been found wrong in this run (`reason`: gpu_decoding_distrusted):
// their black-bar answers, which may come from its pictures, forgotten, and
// the video's note sent.
void decode_on_cpu(VmafOptions& options, const std::string& reason, std::initializer_list<const VideoInfo*> videos,
                   const StatusCallback& on_status) {
    options.gpu_decode = false;
    for (const VideoInfo* video : videos) {
        forget_crop_answer(*video);
    }
    if (on_status) {
        on_status(gpu_decoding_off_status(reason));
    }
}

}  // namespace

ComparisonResult run_vmaf(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                          VmafOptions options, const ProgressCallback& on_progress, const StatusCallback& on_status,
                          const CancelToken& cancel,
                          const std::optional<std::filesystem::path>& result_distorted_path,
                          const std::optional<EnginePlace>& engine) {
    VideoInfo scored_source = source;
    VideoInfo scored_test = distorted;
    SizeChangeMarks marks(scored_source, scored_test);
    while (true) {
        // The GPU decodes the pair: by FFmpeg's -hwaccel, or by the app's decoders, which take the videos FFmpeg
        // would decode there
        const bool on_gpu = options.gpu_decode && plan_hardware_decoding(environment.decoding, options.gpu_vendor,
                                                                         scored_source, scored_test)
                                                      .uses_gpu();
        const std::optional<std::string> distrusted = gpu_decoding_distrusted();
        if (on_gpu && distrusted) {  // found wrong earlier in the run
            decode_on_cpu(options, *distrusted, {&scored_source, &scored_test}, on_status);
            continue;
        }
        try {
            ComparisonResult result = run_vmaf_once(environment, scored_source, scored_test, options, on_progress,
                                                    on_status, cancel, result_distorted_path, engine);
            if (!on_gpu || !gpu_decoding_distrusted()) {
                marks.finished();
                return result;
            }
        } catch (const VideoSizeChanged& changed) {
            cancel.throw_if_cancelled();  // not made again after a Cancel
            if (!marks.after(changed, options.scale_algorithm)) {
                throw VmafRunError(size_change_unfixed);
            }
            continue;
        } catch (const Cancelled&) {
            throw;
        } catch (const std::exception&) {
            // Its failure may come from the GPU's wrong pictures: made again on the CPU where they were found wrong
            if (!on_gpu || !gpu_decoding_distrusted()) {
                throw;
            }
        }
        // The GPU's video decoding was found wrong while the pair was scored, maybe by another video: its scores
        // may come from wrong pictures (FFmpeg's -hwaccel does not notice them), so the pair is scored again
        cancel.throw_if_cancelled();
        decode_on_cpu(options, *gpu_decoding_distrusted(), {&scored_source, &scored_test}, on_status);
    }
}

namespace {

// A resolution test scored as a comparison of the source with a test video made
// from it (VideoInfo::resample_via), so that VMAF runs on the GPU and PSNR, SSIM
// and XPSNR in the app, from the same frames. NVIDIA's decoder makes its pictures,
// scaling as FFmpeg's scale filter does; with any other maker's, FFmpeg's scale
// filters make them, the same pictures, in the comparison's pipes (video_chains).
// Nothing where it cannot be, or where only FFmpeg's filters could make it and the
// CPU scores VMAF: then FFmpeg makes the test, decoding the video once
// (resample_test_run).
std::optional<ComparisonResult> resample_test_on_gpu(const RunEnvironment& environment, const VideoInfo& source,
                                                     const VmafOptions& options,
                                                     const std::optional<CropBox>& source_crop,
                                                     const ProgressCallback& on_progress,
                                                     const StatusCallback& on_status, const CancelToken& cancel) {
    const ResampleTarget& target = *options.resample_test;
    if (source.scaled_to) {
        return std::nullopt;
    }
    const auto [width, height] = resample_analysis_dimensions(source, source_crop);
    VideoInfo test = source;
    test.resample_via = resample_via_size(source, source_crop, target.width);
    const HardwareDecoding decoding = options.gpu_decode ? plan_hardware_decoding(environment.decoding,
                                                                                  options.gpu_vendor, source, source)
                                                         : HardwareDecoding{};
    const std::optional<std::pair<std::string, std::string>> decoders = decoders_here(decoding);
    if (decoders && gpu_decoder(decoders->first).scales_in_two_steps &&
        gpu_decoder(decoders->second).scales_in_two_steps) {
        try {
            const DecodePlan plan =
                plan_decode(test, source_crop, 6, true, std::pair(width, height), options.scale_algorithm);
            if (const std::optional<std::string> refusal = scaling_refusal(plan, environment.ffmpeg)) {
                logger.info("Resolution test made by FFmpeg: " + *refusal);
                return std::nullopt;
            }
        } catch (const GpuDecodeUnavailableError& error) {
            logger.info(std::string("Resolution test made by FFmpeg: ") + error.what());
            return std::nullopt;
        }
    } else if (!options.vmaf_on_gpu || gpu_vmaf_backend(cancel).backend.empty()) {
        return std::nullopt;
    }
    try {
        ComparisonResult result =
            score_pair(environment, source, test, options, source_crop, source_crop, decoding, on_progress, on_status,
                       cancel, synthetic_resample_distorted_path(source.path, target), std::nullopt);
        result.distorted_info = source;  // after the round trip, it is back at the source's own size
        result.resample_target = options.resample_test;
        return result;
    } catch (const Cancelled&) {
        throw;
    } catch (const VideoSizeChanged&) {
        throw;  // run_resample_test refuses it
    } catch (const std::exception& error) {
        logger.warning(std::string("The resolution test on the GPU failed; FFmpeg makes it: ") + error.what());
        return std::nullopt;
    }
}

// resample_test_run's run through FFmpeg, as its reading of the video is set.
ComparisonResult resample_test_by_ffmpeg(const RunEnvironment& environment, const VideoInfo& source,
                                         VmafOptions options, const std::optional<CropBox>& source_crop,
                                         const HardwareDecoding& decoding, const ProgressCallback& on_progress,
                                         const StatusCallback& on_status, const CancelToken& cancel) {
    // The comparison is at the source's own size, the downscale being undone
    // before it: Auto's VMAF models come from that.
    const auto [width, height] = resample_analysis_dimensions(source, source_crop);
    const std::string model = resolve_run_models(options, width, height);
    const long long total_frames = estimate_total_frames(source, options, nullptr);
    const long long decoded_frames = estimate_decoded_frames(source, options, nullptr);
    const int depth = bit_depth(analysis_pix_fmt({source.pix_fmt}));
    const FrameRange range = run_range(options, source);
    const ParsedScores scores = without_app_only(options, depth, [&](const VmafOptions& scored) {
        VmafOptions seeking = scored;
        seeking.range_seeks.source =
            ffmpeg_range_seek(source, range.decode_start, environment.ffmpeg, environment.ffprobe, cancel);
        const auto build_command = [&](const HardwareDecoding& plan, const std::string& run_model,
                                       const std::optional<std::string>& xpsnr_log_name, const PairFeeds& feeds) {
            const std::string graph = build_resample_filtergraph(source, seeking, source_crop, plan.source,
                                                                 "vmaf_log.json", run_model, xpsnr_log_name,
                                                                 environment.cores);
            if (const PictureFeed* fed = feeds.source()) {
                return build_resample_command(environment.ffmpeg, source.path, graph, plan.source, {},
                                              fed->input_arguments());
            }
            return build_resample_command(environment.ffmpeg, source.path, graph, plan.source,
                                          with_seek(source.decode_options(), seeking.range_seeks.source));
        };
        return execute_run(build_command, scored, model, source.fps, decoded_frames, decoding, "vmaf_resample_",
                           on_progress, on_status, cancel, environment, &source, nullptr);
    });

    if (!scores.order.empty() || scores.failed.empty()) {
        const long long compared = scores.frame.empty() ? 0 : static_cast<long long>(scores.frame.back()) + 1 - range.start;
        if (const std::optional<std::string> short_run =
                short_comparison(total_frames, compared, range.start, source.fps, options.n_subsample)) {
            throw VmafRunError(*short_run);
        }
    }
    // Both branches come from the same source, and after the round trip the test
    // video is back at the source's own size.
    ComparisonResult result = result_of_pair(source, source, options, model, source_crop, source_crop, total_frames);
    result.distorted = synthetic_resample_distorted_path(source.path, *options.resample_test);
    result.resample_target = options.resample_test;
    result.metric_failures = scores.failed;
    count_scored_frames(result, std::span(&scores, 1), options.n_subsample);
    for (const std::string& key : scores.order) {
        std::map<std::string, std::string> parameters;
        if (key == "vmaf_neg") {
            parameters["model"] = nlohmann::json("version=vmaf_v0.6.1neg").dump();
        } else if (uses_vmaf_model(key)) {
            parameters["model"] = nlohmann::json(model).dump();
        } else if (key == "vmaf_v1") {
            parameters["model"] = nlohmann::json(options.model_v1).dump();
        }
        MetricProvenance provenance = current_ffmpeg_provenance(key, environment.ffmpeg_version, std::move(parameters));
        mark_range(provenance, run_range(options, source));
        result.metric_results.add(
            FrameMetricResult{key, scores.frame, scores.time, scores.metrics.at(key), std::move(provenance)});
    }
    return result;
}

// run_resample_test's run. Throws VideoSizeChanged when FFmpeg rebuilt its
// filters (size_changes.h).
ComparisonResult resample_test_run(const RunEnvironment& environment, const VideoInfo& source, VmafOptions options,
                                   const ProgressCallback& on_progress, const StatusCallback& on_status,
                                   const CancelToken& cancel) {
    if (!options.resample_test) {
        throw std::invalid_argument("not a resolution test");
    }
    if (!options.compared_range().whole()) {
        if (const std::optional<std::string> problem = pair_problem(source, source, options.compared_range())) {
            throw VmafRunError(*problem);
        }
    }
    // Every scaling of the run, and its result, with the test's own
    // scaling algorithm.
    options.scale_algorithm = options.resample_test->scaler;
    // One file, for the black bars and FFmpeg's route: there is no test
    // video file to plan GPU decoding for.
    HardwareDecoding decoding;
    if (options.gpu_decode) {
        decoding.source = pick_decode(environment.decoding, options.gpu_vendor, source);
    }
    if (on_status && detects_black_bars(options.crop_mode)) {
        on_status({"Detecting black bars in source..."});
    }
    std::optional<CropBox> source_crop;
    try {
        source_crop = source_own_crop(options.crop_choice(), environment.ffmpeg, source, cancel, decoding.source,
                                      options.scale_algorithm);
    } catch (const CropEntryError& error) {
        throw VmafRunError(error.what());
    }
    // A size chosen against the whole frame: with its bars cut, the picture
    // may be no larger, and the test would scale it up or not at all.
    if (options.resample_test->width >= resample_analysis_dimensions(source, source_crop).first) {
        throw VmafRunError(resample_not_smaller);
    }
    refuse_too_small(options, resample_analysis_dimensions(source, source_crop));
    VideoInfo test = source;  // as resample_test_on_gpu makes it
    test.resample_via = resample_via_size(source, source_crop, options.resample_test->width);
    const std::map<std::string, std::string> unfit =
        refuse_unfit_metrics(options, resample_analysis_dimensions(source, source_crop),
                             bit_depth(analysis_pix_fmt({source.pix_fmt})), cambi_input(test, source_crop));
    const auto with_unfit = [&unfit](ComparisonResult result) {
        result.metric_failures.insert(unfit.begin(), unfit.end());
        return result;
    };
    if (std::optional<ComparisonResult> on_gpu =
            resample_test_on_gpu(environment, source, options, source_crop, on_progress, on_status, cancel)) {
        return with_unfit(std::move(*on_gpu));
    }
    // A run that fails where FFmpeg may have read the range from a seek, which it
    // cannot check picture by picture (ffmpeg_range_seek), is made again from the
    // video's start, as score_pair's: a video really cut short then fails for good.
    try {
        return with_unfit(resample_test_by_ffmpeg(environment, source, options, source_crop, decoding, on_progress,
                                                  on_status, cancel));
    } catch (const VmafRunError& error) {
        if (!ffmpeg_range_seek(source, run_range(options, source).decode_start, environment.ffmpeg,
                               environment.ffprobe, cancel)) {
            throw;  // read from the video's start already: the same again would fail the same way
        }
        turn_off_range_seek(error.what());
    }
    return with_unfit(
        resample_test_by_ffmpeg(environment, source, options, source_crop, decoding, on_progress, on_status, cancel));
}

}  // namespace

ComparisonResult run_resample_test(const RunEnvironment& environment, const VideoInfo& source, VmafOptions options,
                                   const ProgressCallback& on_progress, const StatusCallback& on_status,
                                   const CancelToken& cancel) {
    // A resolution test makes its test video in FFmpeg's filters, which HDR
    // VMAF's conversion is not part of: it fails alone, the others stand.
    const bool hdr_refused = std::exchange(options.compute_vmaf_hdr, false);
    if (hdr_refused && options.requested_metrics().empty()) {
        throw VmafRunError("HDR VMAF is not calculated in resolution tests");
    }
    const auto with_hdr_refusal = [hdr_refused](ComparisonResult result) {
        if (hdr_refused) {
            result.metric_failures["vmaf_hdr"] = "HDR VMAF is not calculated in resolution tests";
        }
        return result;
    };
    while (true) {
        const bool on_gpu =
            options.gpu_decode && pick_decode(environment.decoding, options.gpu_vendor, source).has_value();
        const std::optional<std::string> distrusted = gpu_decoding_distrusted();
        if (on_gpu && distrusted) {  // found wrong earlier in the run (run_vmaf)
            decode_on_cpu(options, *distrusted, {&source}, on_status);
            continue;
        }
        try {
            ComparisonResult result = resample_test_run(environment, source, options, on_progress, on_status, cancel);
            if (!on_gpu || !gpu_decoding_distrusted()) {
                return with_hdr_refusal(std::move(result));
            }
        } catch (const VideoSizeChanged&) {
            throw VmafRunError(size_change_in_resample_test);
        } catch (const Cancelled&) {
            throw;
        } catch (const std::exception&) {
            if (!on_gpu || !gpu_decoding_distrusted()) {
                throw;
            }
        }
        // Found wrong while it ran: scored again on the CPU (run_vmaf)
        cancel.throw_if_cancelled();
        decode_on_cpu(options, *gpu_decoding_distrusted(), {&source}, on_status);
    }
}

ProgressCallback part_in_sequence(const ProgressCallback& on_progress, long long index, long long parts) {
    if (!on_progress) {
        return nullptr;
    }
    return [on_progress, index, parts](long long done, long long whole, double fps) {
        on_progress(index * whole + done, parts * whole, fps);
    };
}

}  // namespace vml
