#include "vml/perceptual_cpu.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

#include "vml/app_log.h"
#include "vml/append.h"
#include "vml/build_info.h"
#include "vml/colour.h"
#include "vml/film_grain_scoring.h"
#include "vml/ffmpeg_tools.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_vmaf.h"
#include "vml/gpu_frames.h"
#include "vml/grain_feed.h"
#include "vml/isolated.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/cpu_pair_scorer.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/run_json.h"
#include "vml/stats.h"
#include "vml/score_identity.h"
#include "vml/size_changes.h"
#include "vml/temp_files.h"
#include "vml/threads.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.perceptual_cpu");

// The share of the free memory the workers may fill between them.
constexpr double memory_share = 0.6;

// How often the progress is reported at most, after the first pair: small
// pictures are scored many times a second.
constexpr std::chrono::milliseconds progress_interval{250};

// The memory of one worker of the app's engines (in_app_schedule): its copy
// of the pair (two 16-bit RGB pictures), its Butteraugli's planes, and its
// SSIMULACRA2, which keeps only the rows its filters still need. Measured as
// the peak working set added by each further worker, at 3840x2160 and
// 1920x1080: Butteraugli 133-134 bytes per pixel besides the pictures (here
// with a margin), SSIMULACRA2 about 3 MB at either size.
constexpr long long in_app_picture_bytes_per_pixel = 12;
constexpr long long in_app_butteraugli_bytes_per_pixel = 160;
constexpr long long in_app_ssimulacra2_bytes = 8LL << 20;

// A program's output as Python reads it in text mode: its line endings
// (CR LF, or a lone CR) as LF, and bytes that are not UTF-8 replaced.
std::string as_text(std::string_view output) {
    std::string text;
    text.reserve(output.size());
    for (std::size_t at = 0; at < output.size(); ++at) {
        if (output[at] == '\r') {
            text.push_back('\n');
            if (at + 1 < output.size() && output[at + 1] == '\n') {
                ++at;
            }
        } else {
            text.push_back(output[at]);
        }
    }
    return replace_invalid_utf8(text);
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Whether `c` counts as a word character before a score (Python's \w): an
// ASCII letter, digit or underscore, or a byte of a character past ASCII,
// most of which are letters.
bool is_word_character(char c) {
    const auto byte = static_cast<unsigned char>(c);
    return is_digit(c) || (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || c == '_' || byte >= 0x80;
}

// Where the number that starts at `start` ends (the pattern
// [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?), or `start` when none starts there.
std::size_t number_end(std::string_view text, std::size_t start) {
    std::size_t at = start;
    if (at < text.size() && (text[at] == '+' || text[at] == '-')) {
        ++at;
    }
    if (at < text.size() && is_digit(text[at])) {
        while (at < text.size() && is_digit(text[at])) {
            ++at;
        }
        if (at < text.size() && text[at] == '.') {
            ++at;
            while (at < text.size() && is_digit(text[at])) {
                ++at;
            }
        }
    } else if (at + 1 < text.size() && text[at] == '.' && is_digit(text[at + 1])) {
        at += 2;
        while (at < text.size() && is_digit(text[at])) {
            ++at;
        }
    } else {
        return start;
    }
    if (at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
        std::size_t exponent = at + 1;
        if (exponent < text.size() && (text[exponent] == '+' || text[exponent] == '-')) {
            ++exponent;
        }
        if (exponent < text.size() && is_digit(text[exponent])) {
            while (exponent < text.size() && is_digit(text[exponent])) {
                ++exponent;
            }
            at = exponent;
        }
    }
    return at;
}

// ------------------------------------------------------------ the pictures

// The crop filter that removes a video's black bars, if there are any.
std::vector<std::string> crop_filter(const std::optional<CropBox>& crop, const VideoInfo& info) {
    if (!crop || crop->is_noop(info.frame_size().first, info.frame_size().second)) {
        return {};
    }
    return {crop->as_filter()};
}

// The filters that crop a video's black bars, and scale it to `target`
// when it is scaled; a video whose size changes is scaled to one size first
// (size_changes.h). It is scaled in its own pixel format, as Vship's
// pictures are, and converted to RGB only after its colour tags are set
// (to_rgb): left to make the RGB itself, the scale filter would do so
// before the tags are set, and convert an untagged HD source with BT.601's
// matrix instead of BT.709's.
std::vector<std::string> prepare(const std::optional<CropBox>& crop, const VideoInfo& info,
                                 const std::optional<Size>& target, const std::string& scale_algorithm) {
    std::vector<std::string> filters;
    if (const std::optional<std::string> scale = size_change_scale(info, scale_algorithm)) {
        filters.push_back(*scale);
    }
    for (std::string& filter : crop_filter(crop, info)) {
        filters.push_back(std::move(filter));
    }
    const auto scale_to = [&](const Size& size) {
        filters.push_back("scale=" + std::to_string(size.first) + ":" + std::to_string(size.second) +
                          ":flags=" + scale_algorithm);
    };
    if (info.resample_via) {
        scale_to(*info.resample_via);  // a resolution test's picture: down, then back
        scale_to(target.value_or(content_size(info, crop)));
    } else if (target) {
        scale_to(*target);
    }
    if (target || info.resample_via) {
        filters.push_back("format=" + info.pix_fmt);
    }
    return filters;
}

// The filters that convert a video to RGB with the matrix and range Vship
// reads it with (colour.h). For an untagged video FFmpeg would choose
// BT.601's matrix, where Vship takes an HD video as BT.709. The scale filter
// converts it here, after the tags are set, rather than wherever FFmpeg's
// format negotiation would put a conversion. accurate_rnd keeps an 8-bit
// 4:2:0 or 4:2:2 video off swscale's table converter, which works in 8 bits
// and comes out about 2 steps of 255 too dark: with it, the same picture
// gives the same RGB in 8 bits as in 10 (and a deeper video's is unchanged),
// so a pair of an 8-bit and a 10-bit video is not scored for that offset.
std::vector<std::string> to_rgb(const VideoInfo& info) {
    std::vector<std::string> filters;
    if (const std::optional<VideoColour> colour = colour_of(info)) {
        if (const std::optional<std::string_view> matrix = ffmpeg_matrix_name(colour->matrix)) {
            filters.push_back("setparams=colorspace=" + std::string(*matrix) +
                              ":range=" + (colour->full_range ? "pc" : "tv"));
        }
    }
    filters.push_back("scale=flags=bicubic+accurate_rnd");
    filters.push_back("format=rgb48le");
    return filters;
}

// The filter chains of image_filtergraph, one per video: each from the
// frames of its input to its lossless RGB48 pictures.
class ImageChains {
public:
    ImageChains(const VideoInfo& source_info, const VideoInfo& distorted_info, const VmafOptions& options,
                const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop, int step)
        // Read with the same colours as Vship reads them (with_shared_colours).
        : videos_(with_shared_colours(source_info, distorted_info)),
          source_crop_(source_crop),
          distorted_crop_(distorted_crop),
          scale_algorithm_(options.scale_algorithm),
          seeks_(options.range_seeks),
          // Both videos' frames are first kept to those a range decodes
          // (counted as decoded), then numbered at the test video's frame
          // rate, so that the frames of both are paired; then sampled, so
          // that the frames select drops are not cropped or scaled.
          range_(cpu_range(options, videos_.first, step)),
          numbering_(numbering_filters(frame_rate(videos_.second))) {
        const Size source_size = content_size(videos_.first, source_crop);
        const Size distorted_size = content_size(videos_.second, distorted_crop);
        if (source_size != distorted_size) {
            if (options.scale_direction == ScaleDirection::distorted_to_source) {
                distorted_target_ = source_size;
            } else {
                source_target_ = distorted_size;
            }
        }
        if (step > 1) {
            sample_.push_back("select=not(mod(n\\," + std::to_string(step) + "))");
        }
    }

    // The test video's chain, from FFmpeg's stream `input` to the label
    // `output`.
    std::string distorted(const std::string& input, const std::string& output) const {
        return chain(input, output, distorted_crop_, videos_.second, distorted_target_, seeks_.distorted);
    }

    // The source's chain, likewise.
    std::string source(const std::string& input, const std::string& output) const {
        return chain(input, output, source_crop_, videos_.first, source_target_, seeks_.source);
    }

private:
    std::string chain(const std::string& input, const std::string& output, const std::optional<CropBox>& crop,
                      const VideoInfo& info, const std::optional<Size>& target,
                      const std::optional<RangeSeek>& seek) const {
        std::vector<std::string> filters;
        if (const std::optional<std::string> trim = range_trim_filter(range_, seek)) {
            filters.push_back(*trim);
        }
        append(filters, numbering_);
        append(filters, sample_);
        append(filters, prepare(crop, info, target, scale_algorithm_));
        append(filters, to_rgb(info));
        return "[" + input + "]" + join(filters, ",") + "[" + output + "]";
    }

    std::pair<VideoInfo, VideoInfo> videos_;  // the source and the test video, with shared colours
    std::optional<CropBox> source_crop_;
    std::optional<CropBox> distorted_crop_;
    std::string scale_algorithm_;
    RangeSeeks seeks_;
    FrameRange range_;
    std::vector<std::string> numbering_;
    std::optional<Size> source_target_;     // the size the source is scaled to, if it is
    std::optional<Size> distorted_target_;  // the size the test video is scaled to, if it is
    std::vector<std::string> sample_;       // the filter keeping every step-th frame, if any
};

// The scores of a task's pairs, gathered from the threads that score them,
// and the progress they make: pairs of `step` frames each, of about
// `total_units` frames, at `rate_of` frames a second. The first failure is
// kept, and stops the others.
class ScoreBook {
public:
    ScoreBook(int step, long long total_units, ProgressCallback on_progress, std::function<double(long long)> rate_of)
        : step_(step), total_units_(total_units), on_progress_(std::move(on_progress)), rate_of_(std::move(rate_of)) {}
    ScoreBook(const ScoreBook&) = delete;
    ScoreBook& operator=(const ScoreBook&) = delete;

    // Keeps pair `number`'s scores (each in the keys' order). The progress
    // counts how many pairs are scored, in whatever order they finish. The
    // estimated frame count can be lower than the real one, so the total
    // grows with the work rather than going past 100%.
    void keep(long long number, std::vector<double> values) {
        const std::lock_guard lock(mutex_);
        if (by_pair_.size() <= static_cast<std::size_t>(number)) {
            by_pair_.resize(static_cast<std::size_t>(number) + 1);
        }
        by_pair_[static_cast<std::size_t>(number)] = std::move(values);
        ++finished_;
        const auto now = std::chrono::steady_clock::now();
        if (on_progress_ && (finished_ == 1 || now - reported_ >= progress_interval)) {
            reported_ = now;
            const long long units = finished_ * step_;
            on_progress_(units, std::max(total_units_, units + step_), rate_of_(units));
        }
    }

    // Keeps the first failure, and says the scoring is to stop.
    void fail(std::exception_ptr error) {
        const std::lock_guard lock(mutex_);
        if (!failure_) {
            failure_ = std::move(error);
        }
        stop_ = true;
    }

    // Set once the scoring is to stop (a failure).
    const std::atomic<bool>& stopping() const { return stop_; }

    std::exception_ptr failure() const {
        const std::lock_guard lock(mutex_);
        return failure_;
    }

    // Each pair's scores, in the pairs' order, once every worker is done.
    std::vector<std::vector<double>> take() { return std::move(by_pair_); }

private:
    const int step_;
    const long long total_units_;
    const ProgressCallback on_progress_;
    const std::function<double(long long)> rate_of_;
    mutable std::mutex mutex_;
    std::vector<std::vector<double>> by_pair_;
    long long finished_ = 0;
    std::exception_ptr failure_;
    std::atomic<bool> stop_{false};
    std::chrono::steady_clock::time_point reported_{};
};

}  // namespace

// ----------------------------------------------------------------- public

InAppSchedule in_app_schedule(long long pixels, const std::vector<std::string>& metrics, int concurrent_tasks,
                              unsigned cores, std::optional<unsigned long long> free_memory) {
    const long long task_cores =
        std::max(1LL, static_cast<long long>(std::max(1u, cores)) / std::max(1, concurrent_tasks));
    const bool butteraugli = std::find(metrics.begin(), metrics.end(), "butteraugli") != metrics.end();
    const bool ssimulacra2 = std::find(metrics.begin(), metrics.end(), "ssimulacra2") != metrics.end();
    long long workers = 1;
    if (butteraugli && ssimulacra2) {
        workers = task_cores / 6;
    } else if (butteraugli) {
        workers = task_cores / 12;
    } else {
        workers = task_cores / 2;
    }
    workers = std::max(1LL, workers);
    if (free_memory) {
        const long long per_worker =
            std::max(1LL, pixels) *
                (in_app_picture_bytes_per_pixel + (butteraugli ? in_app_butteraugli_bytes_per_pixel : 0)) +
            (ssimulacra2 ? in_app_ssimulacra2_bytes : 0);
        const auto share = static_cast<long long>(static_cast<double>(*free_memory) * memory_share /
                                                  std::max(1, concurrent_tasks));
        workers = std::max(1LL, std::min(workers, share / per_worker));
    }
    // With both metrics, each pair's SSIMULACRA2 runs on a thread of its own
    // beside its Butteraugli (CpuPairScorer::score).
    const long long threads = std::max(1LL, task_cores / workers - (butteraugli && ssimulacra2 ? 1 : 0));
    return InAppSchedule{static_cast<int>(workers), static_cast<std::size_t>(threads)};
}

double parse_score(std::string_view metric, std::string_view output) {
    // The last number not straight after a word character or a point.
    std::optional<std::string_view> last;
    std::size_t at = 0;
    while (at < output.size()) {
        if (at > 0 && (is_word_character(output[at - 1]) || output[at - 1] == '.')) {
            ++at;
            continue;
        }
        const std::size_t end = number_end(output, at);
        if (end > at) {
            last = output.substr(at, end - at);
            at = end;
        } else {
            ++at;
        }
    }
    if (!last) {
        throw PerceptualRunError(std::string(metric) + " did not produce a numeric score: " +
                                 first_characters(python_strip(output), 300));
    }
    const std::optional<double> score = python_float(*last);
    if (!score) {
        throw PerceptualRunError("Could not parse " + std::string(metric) + " score.");
    }
    return *score;
}

namespace {

// The 3-norm of `count` values, the Ith of them in the file's order
// value_at(I).
template <typename ValueAt>
double norm3_in_file_order(std::size_t count, const ValueAt& value_at) {
    // Each cube is pow's, as numpy's ** makes it: x * x * x differs from it
    // in the last bits.
    const double sum = numpy_sum_of(
        count, [&value_at](std::size_t i) { return std::pow(std::abs(static_cast<double>(value_at(i))), 3.0); });
    // numpy's mean: its pairwise sum, divided by the count (NaN for none).
    const double mean = sum / static_cast<double>(count);
    return std::pow(mean, 1.0 / 3.0);
}

}  // namespace

double butteraugli_norm3_top_row_first(std::span<const float> distortion_map, std::size_t width) {
    const std::size_t rows = width == 0 ? 0 : distortion_map.size() / width;
    // The file's Ith value is in its row I / width counted from the bottom.
    return norm3_in_file_order(distortion_map.size(), [distortion_map, width, rows](std::size_t i) {
        return distortion_map[(rows - 1 - i / width) * width + i % width];
    });
}

FrameRange cpu_range(const VmafOptions& options, const VideoInfo& source, int step) {
    return frame_range(options.compared_range(), source.fps, step, false);
}

std::string image_filtergraph(const VideoInfo& source_info, const VideoInfo& distorted_info,
                              const VmafOptions& options, const std::optional<CropBox>& source_crop,
                              const std::optional<CropBox>& distorted_crop, int step) {
    const ImageChains chains(source_info, distorted_info, options, source_crop, distorted_crop, step);
    return chains.distorted("0:" + std::string(video_stream), "distorted") + ";" +
           chains.source("1:" + std::string(video_stream), "reference");
}

namespace {

// FFmpeg's options for reading `video`: its feed's input where the app feeds
// its pictures (grain_feed.h), else its file's, from `seek`.
std::vector<std::string> input_options(const VideoInfo& video, const std::optional<std::vector<std::string>>& fed,
                                       const std::optional<RangeSeek>& seek) {
    if (fed) {
        return *fed;
    }
    std::vector<std::string> input = with_seek(video.decode_options(), seek);
    input.insert(input.end(), {as_stored, "-i", input_path(video.path)});
    return input;
}

// The start of every FFmpeg command of these metrics. At watched_log_level:
// the app reads FFmpeg's word that it rebuilt its filters.
std::vector<std::string> command_start(const std::string& ffmpeg) {
    return {ffmpeg, "-nostdin", "-hide_banner", "-loglevel", watched_log_level, "-nostats", "-y"};
}

}  // namespace

std::vector<std::vector<std::string>> rgb_stream_commands(
    const std::string& ffmpeg, const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
    const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop, int step,
    const std::string& test_pipe, const std::string& source_pipe,
    const std::optional<std::vector<std::string>>& distorted_fed,
    const std::optional<std::vector<std::string>>& source_fed) {
    const ImageChains chains(source, distorted, options, source_crop, distorted_crop, step);
    const std::string input = "0:" + std::string(video_stream);  // each command's only input
    const auto command = [&ffmpeg](std::vector<std::string> reading, const std::string& graph,
                                   const std::string& pipe) {
        std::vector<std::string> made = command_start(ffmpeg);
        append(made, reading);
        made.insert(made.end(), {"-filter_complex", graph, "-map", "[frames]", "-fps_mode", "passthrough",
                                 "-flush_packets", "1", "-f", "rawvideo", "-pix_fmt", "rgb48le", pipe});
        return made;
    };
    return {command(input_options(distorted, distorted_fed, options.range_seeks.distorted),
                    chains.distorted(input, "frames"), test_pipe),
            command(input_options(source, source_fed, options.range_seeks.source), chains.source(input, "frames"),
                    source_pipe)};
}

namespace {

// The one frame coverage (every how many frames) `keys`' metrics share;
// throws when they differ.
int common_step(const PerceptualRequest& request, const std::vector<std::string>& keys) {
    std::set<int> steps;
    for (const std::string& key : keys) {
        steps.insert(request.step_of(key));
    }
    if (steps.size() != 1) {
        throw PerceptualRunError("Perceptual metrics in one task must use the same frame coverage.");
    }
    return *steps.begin();
}

// How many frames the comparison is expected to decode: the shorter video's,
// those of the range (cpu_range).
long long expected_frames(const VideoInfo& source, const VideoInfo& distorted, const FrameRange& range) {
    const long long frames = std::min(source.estimated_frame_count(), distorted.estimated_frame_count());
    return std::max(1LL, std::min(frames, range.decode_end.value_or(frames)) - range.decode_start);
}

// A video's colour tags as a refusal names them.
std::string colour_tags_text(const VideoInfo& video) {
    const auto tag = [](const std::string& value) { return value.empty() ? std::string("untagged") : value; };
    return "matrix " + tag(video.color_space) + ", transfer " + tag(video.color_transfer) + ", primaries " +
           tag(video.color_primaries) + ", range " + tag(video.color_range);
}

// The colours the source's and the test's pictures are scored with, each
// read with its own tags, or with the other's where it has none
// (with_shared_colours). Throws PerceptualRunError for tags the app cannot
// read, as the GPU's Vship refuses them, naming each video read with such
// tags (both where one took the other's: the command line lends them before
// the request is made, so which video they came from is not known here).
std::pair<VideoColour, VideoColour> scored_colours(const VideoInfo& source, const VideoInfo& distorted) {
    const auto [source_read, distorted_read] = with_shared_colours(source, distorted);
    const std::optional<VideoColour> source_colour = colour_of(source_read);
    const std::optional<VideoColour> distorted_colour = colour_of(distorted_read);
    if (source_colour && distorted_colour) {
        return {*source_colour, *distorted_colour};
    }
    const auto named = [](const VideoInfo& video, const VideoInfo& read) {
        return platform::path_text(video.path.filename()) + " (" + colour_tags_text(read) + ")";
    };
    std::string refused;
    if (!source_colour) refused = named(source, source_read);
    if (!distorted_colour) refused += (refused.empty() ? "" : " and ") + named(distorted, distorted_read);
    throw PerceptualRunError("SSIMULACRA2 and Butteraugli do not support the colours of " + refused + ".");
}

// ------------------------------------------------- scored in this process

// How many frames each video's pipe holds: the one a worker copies and two
// read ahead. FFmpeg waits once they are full, which holds it to the
// scoring's pace.
constexpr std::size_t pipe_frames = 3;

// The bytes of one 16-bit RGB picture of `width` x `height`.
std::size_t rgb48_bytes(int width, int height) {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 6;
}

// What a task's pairs are scored with in this process.
struct InAppScoring {
    const std::vector<std::string>& keys;
    VideoColour source_colour;
    VideoColour distorted_colour;
    int width;
    int height;
    InAppSchedule schedule;
};

// The frame pairs of one CPU perceptual task, scored in this process by
// the app's engines (cpu_pair_scorer.h), `scoring.schedule.workers` pairs at
// a time. Two FFmpegs, one per video, write its pictures as raw 16-bit RGB
// into pipes of the app's (rgb_stream_commands). Frame N of the test video
// is paired with frame N of the source until either ends. Each worker takes
// the next pair from the pipes, copies it into its own scorer, gives the
// buffers back for FFmpeg's next frames, and scores it. A failure stops the
// pipes, which ends both FFmpegs.
class InAppPairs {
public:
    // Opens the pipes and starts the workers, who wait for FFmpeg's frames.
    InAppPairs(const InAppScoring& scoring, ScoreBook& book)
        : scoring_(scoring),
          book_(book),
          frame_bytes_(rgb48_bytes(scoring.width, scoring.height)),
          test_("perceptual-test", frame_bytes_, StreamReader::Framing::raw, pipe_frames),
          source_("perceptual-source", frame_bytes_, StreamReader::Framing::raw, pipe_frames) {
        test_.start();
        source_.start();
        try {
            for (int worker = 0; worker < std::max(1, scoring.schedule.workers); ++worker) {
                workers_.emplace_back([this] { work(); });
            }
        } catch (...) {
            fail(std::current_exception());  // no thread for another worker: the others stop
        }
    }

    // Stops what still runs: a run that threw before finish().
    ~InAppPairs() {
        stop_pipes();
        join_workers();
    }

    InAppPairs(const InAppPairs&) = delete;
    InAppPairs& operator=(const InAppPairs&) = delete;

    // What the test video's FFmpeg (command 0) and the source's (1) write to.
    const std::string& test_pipe() const { return test_.path(); }
    const std::string& source_pipe() const { return source_.path(); }

    // What command `command` inherits (run_ffmpeg_pair's take_descriptors).
    std::vector<int> take_descriptors(std::size_t command) {
        const int writer = reader(command).take_writer();
        return writer >= 0 ? std::vector<int>{writer} : std::vector<int>{};
    }

    // Whether the app closed command `command`'s pipe while it still wrote.
    bool closed_by_app(std::size_t command) { return reader(command).stopped_while_reading(); }

    // Once the FFmpegs have ended: waits for the workers, and returns the
    // error of a pipe whose stream was not what FFmpeg writes (a frame cut
    // short), if any.
    std::exception_ptr finish(bool ffmpeg_succeeded) {
        if (ffmpeg_succeeded) {
            test_.ffmpeg_ended();
            source_.ffmpeg_ended();
        } else {
            stop_pipes();
        }
        join_workers();
        std::exception_ptr error;
        for (StreamReader* stream : {&test_, &source_}) {
            try {
                stream->join();
            } catch (...) {
                if (!error) {
                    error = std::current_exception();
                }
            }
        }
        return error;
    }

    // How many pairs the workers took.
    long long pairs() const { return handed_; }

private:
    StreamReader& reader(std::size_t command) { return command == 0 ? test_ : source_; }

    void work() {
        std::optional<CpuPairScorer> scorer;
        try {
            scorer.emplace(scoring_.keys, scoring_.source_colour, scoring_.distorted_colour, scoring_.width,
                           scoring_.height, scoring_.schedule.butteraugli_threads);
        } catch (...) {
            fail(std::current_exception());
            return;
        }
        while (!book_.stopping()) {
            std::vector<std::uint8_t>* test_frame = nullptr;
            std::vector<std::uint8_t>* source_frame = nullptr;
            long long number = 0;
            {
                const std::lock_guard lock(pairs_mutex_);
                if (ended_ || !take_pair(test_frame, source_frame)) {
                    return;
                }
                number = handed_++;
            }
            // Copied outside the lock, so that the next worker can take its
            // pair meanwhile; the buffers go back at once for FFmpeg's next
            // frames.
            std::memcpy(scorer->test_samples(), test_frame->data(), frame_bytes_);
            std::memcpy(scorer->source_samples(), source_frame->data(), frame_bytes_);
            test_.release(test_frame);
            source_.release(source_frame);
            try {
                book_.keep(number, scorer->score());
            } catch (...) {
                fail(std::current_exception());
                return;
            }
        }
    }

    // The next pair, or false at the end of either video's frames, where
    // the comparison ends: what FFmpeg still writes is dropped. A stream
    // that is not what FFmpeg writes ends it too; finish() reports it.
    // Called under pairs_mutex_.
    bool take_pair(std::vector<std::uint8_t>*& test_frame, std::vector<std::uint8_t>*& source_frame) {
        try {
            test_frame = test_.pull();
            source_frame = test_frame != nullptr ? source_.pull() : nullptr;
        } catch (const std::exception&) {
            source_frame = nullptr;
        }
        if (source_frame == nullptr) {
            if (test_frame != nullptr) {
                test_.release(test_frame);
            }
            ended_ = true;
            test_.drop_rest();
            source_.drop_rest();
            return false;
        }
        return true;
    }

    // Keeps the first failure and closes the pipes, which ends both FFmpegs
    // and every worker's wait for a frame.
    void fail(std::exception_ptr error) {
        book_.fail(std::move(error));
        stop_pipes();
    }

    void stop_pipes() {
        test_.stop();
        source_.stop();
    }

    void join_workers() {
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    const InAppScoring& scoring_;
    ScoreBook& book_;
    const std::size_t frame_bytes_;
    StreamReader test_;
    StreamReader source_;
    std::mutex pairs_mutex_;
    bool ended_ = false;              // either video's frames have ended
    std::atomic<long long> handed_{0};  // how many pairs workers have taken
    std::vector<std::thread> workers_;  // last: they use the rest
};

// The scores of every pair of `commands` (the test video's FFmpeg and the
// source's, made by `make_commands` for the pipes' paths), scored in this
// process, as FFmpeg writes them. A write into the app's pipe that Windows
// refused (refused_frame_write) is made again once. Throws Cancelled,
// VideoSizeChanged and PerceptualRunError.
std::vector<std::vector<double>> score_in_app(
    const InAppScoring& scoring,
    const std::function<std::vector<std::vector<std::string>>(const std::string&, const std::string&)>& make_commands,
    const std::vector<std::filesystem::path>& videos, const std::function<std::unique_ptr<ScoreBook>()>& new_book,
    const std::filesystem::path& folder, const CancelToken& cancel) {
    for (bool retried = false;; retried = true) {
        const std::unique_ptr<ScoreBook> book = new_book();  // a new progress for an attempt made again
        InAppPairs pairs(scoring, *book);
        FfmpegOutcome outcome;
        try {
            const std::vector<std::vector<std::string>> commands =
                make_commands(pairs.test_pipe(), pairs.source_pipe());
            for (const std::vector<std::string>& command : commands) {
                logger.info("FFmpeg: " + python_list2cmdline(command));
            }
            outcome = run_ffmpeg_pair(
                commands, videos, [&pairs](std::size_t command) { return pairs.take_descriptors(command); },
                [&pairs](std::size_t command) { return pairs.closed_by_app(command); }, 0, nullptr, cancel, folder);
        } catch (const platform::ProcessError& error) {
            throw PerceptualRunError(std::string("FFmpeg could not be started: ") + error.what());
        }
        const bool succeeded = outcome.exit_code == 0;
        const std::exception_ptr stream_error = pairs.finish(succeeded);
        const auto ffmpeg_errors = [&outcome] {
            return last_characters(python_strip(as_text(error_lines_of(outcome.errors))), 2000);
        };
        // A scoring that failed closed the pipes, and so ended the FFmpegs:
        // its failure is the cause.
        if (const std::exception_ptr failure = book->failure()) {
            std::rethrow_exception(failure);
        }
        if (!succeeded) {
            if (!retried && refused_frame_write(outcome.exit_code, outcome.errors)) {
                logger.warning("FFmpeg's write into the app's pipe was refused; the frames are made again");
                continue;
            }
            // A frame cut short where FFmpeg failed is that failure.
            throw PerceptualRunError("FFmpeg could not prepare lossless perceptual-metric frames.", ffmpeg_errors());
        }
        if (stream_error) {
            throw PerceptualRunError("FFmpeg's frames could not be read: " + message_of(stream_error),
                                     ffmpeg_errors());
        }
        if (pairs.pairs() == 0) {
            throw PerceptualRunError("FFmpeg produced no frame pairs for perceptual metrics.", ffmpeg_errors());
        }
        return book->take();
    }
}

// The scores `by_pair` (each pair's, in `keys`' order) as the task's
// results: frames `step` apart from range.decode_start, those in range kept,
// timed at `fps`, with how they were made.
MetricResultSet cpu_results(const std::vector<std::string>& keys, const std::vector<std::vector<double>>& by_pair,
                            int step, double fps, bool hdr, const FrameRange& range) {
    const auto total = static_cast<long long>(by_pair.size());
    std::vector<int> frames;
    std::vector<double> times;
    std::vector<std::size_t> kept;  // the pairs in range
    for (long long index = 0; index < total; ++index) {
        const long long frame = range.decode_start + index * step;
        if (frame < range.start) {
            continue;  // decoded for the subsampling's grid, before the range
        }
        frames.push_back(static_cast<int>(frame));
        times.push_back(static_cast<double>(frame) / std::max(fps, 1.0));
        kept.push_back(static_cast<std::size_t>(index));
    }
    MetricResultSet results;
    for (std::size_t position = 0; position < keys.size(); ++position) {
        const std::string& key = keys[position];
        std::vector<float> values;
        for (const std::size_t pair : kept) {
            values.push_back(static_cast<float>(by_pair[pair][position]));
        }
        // The colours are read as Vship reads them, with its Butteraugli
        // 3-norm and display brightness.
        std::map<std::string, std::string> parameters{
            {"intermediate", nlohmann::json("png/rgb48le").dump()},
            {"coverage_step", std::to_string(step)},
            {"color_tags", nlohmann::json(score_identity::cpu_colour_tags).dump()},
        };
        if (key == "butteraugli") {
            parameters["butteraugli_norm"] = nlohmann::json("3-norm").dump();
            parameters["intensity_target"] =
                hdr ? nlohmann::json("picture's own").dump() : nlohmann::json(butteraugli_intensity_nits).dump();
        }
        MetricProvenance provenance{key, std::string(in_app_scoring_version), "cpu",
                                    score_identity::libjxl_cpu(key), std::move(parameters), build_stamp()};
        mark_range(provenance, range);
        results.add(FrameMetricResult{key, frames, times, std::move(values), std::move(provenance)});
    }
    return results;
}

// run_cpu_perceptual_task's work in this process, with the black bars
// `crops` found.
PerceptualTaskOutput score_cpu_task(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                    const CropPair& crops, const ProgressCallback& on_progress,
                                    const StatusCallback& on_status, const CancelToken& cancel) {
    const VideoInfo& source = request.source;
    VmafOptions options = request.options;
    const int step = common_step(request, keys);
    const auto& [source_crop, distorted_crop] = crops;
    const VideoInfo distorted = with_resample_test_video(request, source_crop).distorted;
    const FrameRange range = cpu_range(options, source, step);
    for (const auto& [video, seek] : {std::pair{&source, &options.range_seeks.source},
                                      std::pair{&distorted, &options.range_seeks.distorted}}) {
        *seek = ffmpeg_range_seek(*video, range.decode_start, request.environment.ffmpeg,
                                  request.environment.ffprobe, cancel);
    }
    const long long expected = expected_frames(source, distorted, range);
    const long long total_units = std::max(1LL, (expected + step - 1) / step) * step;
    if (on_status) {
        on_status(decoding_status("Calculating SSIMULACRA2/Butteraugli on the CPU as frames are extracted",
                                  HardwareDecoding{}, StatusKind::step, "\xE2\x80\xA6"));
    }
    auto started = std::chrono::steady_clock::now();
    const auto rate_of = [&started](long long units) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return static_cast<double>(units) / std::max(seconds, 1e-6);
    };
    const auto [width, height] =
        compared_dimensions(source, distorted, options.scale_direction, source_crop, distorted_crop);
    // The pipes' frames take their memory beside the workers'.
    std::optional<unsigned long long> free_memory = platform::available_memory();
    const unsigned long long pipes = 2 * pipe_frames * rgb48_bytes(width, height);
    if (free_memory) {
        *free_memory = *free_memory > pipes ? *free_memory - pipes : 0;
    }
    const long long pixels = static_cast<long long>(width) * height;
    const InAppSchedule schedule =
        in_app_schedule(pixels, keys, request.concurrent_cpu_tasks, platform::logical_cores(), free_memory);
    const bool with_butteraugli = std::find(keys.begin(), keys.end(), "butteraugli") != keys.end();
    logger.info(join(keys, " and ") + ": " + std::to_string(schedule.workers) + " frame pair" +
                (schedule.workers == 1 ? "" : "s") + " at a time" +
                (with_butteraugli
                     ? ", Butteraugli on " + std::to_string(schedule.butteraugli_threads) + " threads each"
                     : std::string()));
    const auto [source_colour, distorted_colour] = scored_colours(source, distorted);
    const InAppScoring scoring{keys, source_colour, distorted_colour, width, height, schedule};
    const bool hdr = scoring.source_colour.hdr() || scoring.distorted_colour.hdr();
    const auto new_book = [&] {
        started = std::chrono::steady_clock::now();
        return std::make_unique<ScoreBook>(step, total_units, on_progress, rate_of);
    };

    std::vector<std::vector<double>> by_pair;
    // Made again, with new pictures and progress, where a feeding's pictures from vvdec-vulkan failed
    // (VvdecFeedFailed: vvdec-vulkan off then, so once)
    for (bool again = true; again;) {
        again = false;
        try {
            // Before FFmpeg, which it outlives.
            PairFeeds feeds(&source, &distorted, request.environment.ffmpeg, request.environment.ffprobe, cancel,
                            vvdec_wanted(options.gpu_decode, options.vvc_on_gpu), on_status);
            const auto fed_input = [](const PictureFeed* feed) {
                return feed != nullptr ? std::optional(feed->input_arguments()) : std::nullopt;
            };
            try {
                // A folder of its own for the task's FFmpegs and pictures, deleted with them.
                const TempFolder folder = TempFolder::make<PerceptualRunError>(
                    "videometricslab-perceptual-", "A folder for the frame pictures could not be made.");
                by_pair = score_in_app(
                    scoring,
                    [&](const std::string& test_pipe, const std::string& source_pipe) {
                        return rgb_stream_commands(request.environment.ffmpeg, source, distorted, options,
                                                   source_crop, distorted_crop, step, test_pipe, source_pipe,
                                                   fed_input(feeds.distorted()), fed_input(feeds.source()));
                    },
                    {distorted.path, source.path}, new_book, folder.path(), cancel);
            } catch (const Cancelled&) {
                throw;
            } catch (...) {
                feeds.finish();  // a feeding that failed is why: its failure is the one to see
                throw;
            }
            feeds.finish();  // a fed video decoded in part is no score
        } catch (const VvdecFeedFailed& error) {
            report_vvdec_feed_failed(error, on_status);
            again = true;
        }
    }
    const auto total = static_cast<long long>(by_pair.size());
    if (const std::optional<std::string> short_reason =
            short_comparison(expected, total * step, range.decode_start, source.fps, step)) {
        if (options.range_seeks.source || options.range_seeks.distorted) {
            // A video really cut short, or a seek that gave less than it promised: from the start, to tell
            turn_off_range_seek(*short_reason);
            return score_cpu_task(request, keys, crops, on_progress, on_status, cancel);
        }
        throw ComparisonCutShortError(*short_reason);
    }
    if (on_progress) {
        on_progress(total * step, total * step, rate_of(total * step));
    }
    MetricResultSet results = cpu_results(keys, by_pair, step, source.fps, hdr, range);
    const long long compared = static_cast<long long>(results.frame(keys.front())->frame.size()) * step;
    return PerceptualTaskOutput{std::move(results), source_crop, distorted_crop, compared, {}};
}

// What a "cpu-perceptual" task's process is given.
nlohmann::json cpu_task_request(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                const CropPair& crops) {
    return {{"ffmpeg", request.environment.ffmpeg},
            {"ffprobe", request.environment.ffprobe},
            {"source", video_json(request.source)},
            {"distorted", video_json(request.distorted)},
            {"options", options_json(child_options(request.options))},
            {"backends", request.backends},
            {"concurrent_cpu_tasks", request.concurrent_cpu_tasks},
            {"keys", keys},
            {"source_crop", crop_json(crops.first)},
            {"distorted_crop", crop_json(crops.second)}};
}

// score_cpu_task in this program's own process (isolated.h), in the app:
// a crash in the engines then ends only that process, failing the task;
// Pause holds that process and its FFmpegs at once, and Cancel ends them.
PerceptualTaskOutput score_cpu_task_apart(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                          const CropPair& crops, const ProgressCallback& on_progress,
                                          const StatusCallback& on_status, const CancelToken& cancel) {
    PerceptualTaskOutput output;
    std::optional<std::string> cut_short;
    std::optional<PerceptualRunError> failed;
    // The process deals with a failure of vvdec-vulkan's pictures itself (VvdecFeedFailed).
    bool vvdec_used = false;
    try {
        run_isolated(
            "cpu-perceptual", "SSIMULACRA2/Butteraugli", cpu_task_request(request, keys, crops),
            [&](const nlohmann::json& message) {
                if (message.contains("progress") && on_progress) {
                    const nlohmann::json& counts = message["progress"];
                    on_progress(counts.at(0).get<long long>(), counts.at(1).get<long long>(),
                                counts.at(2).get<double>());
                } else if (message.contains("status")) {
                    take_child_status(status_from_json(message["status"]), on_status, vvdec_used);
                }
            },
            cancel, [&](const nlohmann::json& result, const std::filesystem::path& folder) {
                if (result.contains("cut_short")) {
                    cut_short = result.at("cut_short").get<std::string>();
                } else if (result.contains("failed")) {
                    failed.emplace(result.at("failed").get<std::string>(), result.value("stderr_tail", ""));
                } else {
                    output = perceptual_output_from(result, folder);
                }
            });
    } catch (const Cancelled&) {
        throw;
    } catch (const VideoSizeChanged&) {
        throw;  // the process's, kept by run_isolated: the task is made again
    } catch (const FilmGrainRefusal&) {
        throw;
    } catch (const IsolatedCrashError& error) {
        logger.error(join(keys, " and ") + " on the CPU failed: " + error.what());
        throw PerceptualRunError(error.what());
    } catch (const std::runtime_error& error) {
        throw PerceptualRunError(error.what());  // the process's own error, as its message
    }
    if (cut_short) {
        throw ComparisonCutShortError(*cut_short);
    }
    if (failed) {
        throw *failed;
    }
    return output;
}

}  // namespace

PerceptualTaskOutput run_cpu_perceptual_task(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                             const std::optional<CropPair>& crops,
                                             const ProgressCallback& on_progress, const StatusCallback& on_status,
                                             const CancelToken& cancel) {
    if (keys.empty() || std::any_of(keys.begin(), keys.end(), [](const std::string& key) {
            return key != "ssimulacra2" && key != "butteraugli";
        })) {
        throw std::invalid_argument("a CPU perceptual task takes SSIMULACRA2 and Butteraugli only");
    }
    validate_pair(request.source, request.distorted, request.options);
    common_step(request, keys);  // before the black bars are looked for: one coverage for the task
    // Colours the engines cannot read are refused before the black bars are
    // looked for or a process is started (resampling keeps the tags).
    scored_colours(request.source, request.distorted);
    const CropPair found = crops ? *crops : perceptual_crops(request, on_status, cancel);
    if (isolation_program()) {
        return score_cpu_task_apart(request, keys, found, on_progress, on_status, cancel);
    }
    return score_cpu_task(request, keys, found, on_progress, on_status, cancel);
}

nlohmann::json cpu_perceptual_task(const nlohmann::json& request, const std::filesystem::path& folder) {
    PerceptualRequest perceptual;
    perceptual.environment.ffmpeg = request.at("ffmpeg").get<std::string>();
    perceptual.environment.ffprobe = request.at("ffprobe").get<std::string>();
    perceptual.source = video_from(request.at("source"));
    perceptual.distorted = video_from(request.at("distorted"));
    perceptual.options = options_from(request.at("options"));
    request.at("backends").get_to(perceptual.backends);
    request.at("concurrent_cpu_tasks").get_to(perceptual.concurrent_cpu_tasks);
    const auto keys = request.at("keys").get<std::vector<std::string>>();
    const CropPair crops{crop_from(request.at("source_crop")), crop_from(request.at("distorted_crop"))};
    try {
        const PerceptualTaskOutput output = score_cpu_task(
            perceptual, keys, crops,
            [](long long done, long long whole, double fps) {
                send_isolated_message({{"progress", {done, whole, fps}}});
            },
            [](const RunStatus& status) { send_isolated_message({{"status", status_json(status)}}); },
            CancelToken());  // the parent cancels by ending this process
        return perceptual_output_json(output, folder);
    } catch (const ComparisonCutShortError& error) {
        return {{"cut_short", error.what()}};
    } catch (const PerceptualRunError& error) {
        return {{"failed", error.what()}, {"stderr_tail", error.stderr_tail()}};
    }
}

}  // namespace vml
