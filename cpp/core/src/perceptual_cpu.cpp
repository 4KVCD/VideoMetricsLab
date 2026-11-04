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
#include "vml/ffmpeg_tools.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_frames.h"
#include "vml/grain_feed.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/stats.h"
#include "vml/score_identity.h"
#include "vml/size_changes.h"
#include "vml/temp_files.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.perceptual_cpu");

// How many frame pairs FFmpeg may write ahead of the scoring before it is
// suspended. FFmpeg writes the pictures far faster than the tools score them
// (1-2 s per 4K pair), so without a limit it would write the whole video out
// first: about 10 MB of 16-bit PNG per 4K frame, terabytes for a film, with
// no progress shown until it finished. 24 pairs of 4K take about 0.5 GB.
// FFmpeg resumes once half of them are scored.
constexpr int backlog_pairs = 24;

// The longest one tool may run on one frame pair. The bundled tools take
// 1-2 s for a 4K pair.
constexpr std::chrono::seconds tool_timeout{120};

// The share of the free memory the tools may fill between them.
constexpr double memory_share = 0.6;

// How often the progress is reported at most, after the first pair: small
// pictures are scored many times a second.
constexpr std::chrono::milliseconds progress_interval{250};

// The memory one tool takes at its peak, in bytes per pixel of the pair it
// scores, with a margin. On a 3840x2160 pair, SSIMULACRA2 takes 1.19 GB (143
// bytes per pixel) and Butteraugli 1.85 GB (223 bytes per pixel).
long long tool_bytes_per_pixel(const std::string& metric) { return metric == "ssimulacra2" ? 160 : 250; }

std::filesystem::path libjxl_folder() { return resource_folder() / "tools" / "libjxl"; }

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

// The tool's first line for --version, -version or -V, or "unknown" (the
// Python app's _tool_version).
std::string tool_version(const std::filesystem::path& executable) {
    for (const char* flag : {"--version", "-version", "-V"}) {
        const std::optional<platform::ProcessOutput> output =
            platform::finished_program({platform::path_text(executable), flag}, std::chrono::seconds(5));
        if (!output) {
            return "unknown";
        }
        if (output->exit_code == 0) {
            const std::string text =
                as_text(!output->standard_output.empty() ? output->standard_output : output->standard_error);
            const std::string_view stripped = python_strip(text);
            if (stripped.empty()) {
                return "unknown";
            }
            return first_characters(stripped.substr(0, stripped.find_first_of("\n\v\f\x1C\x1D\x1E")), 160);
        }
    }
    return "unknown";
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
    if (target) {
        filters.push_back("scale=" + std::to_string(target->first) + ":" + std::to_string(target->second) +
                          ":flags=" + scale_algorithm);
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

// What image_filtergraph's chains need of the two videos.
struct GraphPlan {
    const VideoInfo& source;
    const VideoInfo& distorted;
    const std::optional<CropBox>& source_crop;
    const std::optional<CropBox>& distorted_crop;
    std::optional<Size> source_target;     // the size the source is scaled to, if it is
    std::optional<Size> distorted_target;  // the size the test video is scaled to, if it is
    Size source_size;                      // the source's size after its crop
    std::vector<std::string> sample;       // the filter keeping every step-th frame, if any
    std::string scale_algorithm;
};

// ------------------------------------------------------------ the tools

// Deletes a file when it goes, if it is there.
class Deleted {
public:
    explicit Deleted(std::filesystem::path path) : path_(std::move(path)) {}
    ~Deleted() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    Deleted(const Deleted&) = delete;
    Deleted& operator=(const Deleted&) = delete;

private:
    std::filesystem::path path_;
};

// Runs a still-image tool to its end, as platform::run_program does, with
// the tool attached to the run: Pause holds it, as it holds FFmpeg, and
// Cancel ends it at once. Without that, the tools would score the pairs
// already written while the window says "Paused". `stop` ends it too, and
// so does tool_timeout of running, the time paused not counted.
platform::ProcessOutput run_tool(const std::vector<std::string>& command, const std::function<bool()>& stop,
                                 const CancelToken& cancel) {
    platform::ChildProcess tool = platform::ChildProcess::start(command);
    const CancelToken::Attached attached = cancel.attach(tool);
    // Read on a thread of its own, so that the tool is never held up by a
    // full pipe while this one waits for it to end.
    std::string output;
    std::thread reader([&tool, &output] {
        while (const std::optional<std::string> line = tool.read_line()) {
            output += *line;
            output += '\n';
        }
    });
    platform::ProcessOutput result;
    constexpr std::chrono::milliseconds step(100);
    std::chrono::milliseconds running(0);
    while (!tool.wait_for(step)) {
        if (stop()) {
            tool.terminate();
            result.stopped = true;
            break;
        }
        if (!cancel.paused()) {
            running += step;
        }
        if (running >= tool_timeout) {
            tool.terminate();
            result.timed_out = true;
            break;
        }
    }
    result.exit_code = tool.wait();
    reader.join();
    // Ended by the run's Cancel (which ends attached programs itself), it was
    // stopped, not failed: its exit code says only that it was ended.
    if (!result.timed_out && cancel.cancelled()) {
        result.stopped = true;
    }
    result.standard_output = std::move(output);
    result.standard_error = tool.standard_error();
    return result;
}

// Scores one frame pair with a still-image tool and returns the score (the
// Python app's _run_metric). `stop` ends the tool at once: the user's
// Cancel, or another pair's failure, which throws Cancelled here.
double run_metric(const std::filesystem::path& executable, const std::string& metric,
                  const std::filesystem::path& reference, const std::filesystem::path& test, bool hdr,
                  const std::function<bool()>& stop, const CancelToken& cancel) {
    std::vector<std::string> command{platform::path_text(executable), platform::path_text(reference),
                                     platform::path_text(test)};
    std::optional<std::filesystem::path> distortion_map;
    if (metric == "butteraugli") {
        distortion_map =
            test.parent_path() / platform::path_from_text(platform::path_text(test.stem()) + "-distortion.pfm");
        if (!hdr) {
            // As Python's {:g} writes it: "203".
            command.insert(command.end(), {"--intensity_target", format_general(butteraugli_intensity_nits)});
        }
        command.insert(command.end(), {"--rawdistmap", platform::path_text(*distortion_map)});
    }
    std::optional<Deleted> map_deleted;
    if (distortion_map) {
        map_deleted.emplace(*distortion_map);
    }
    platform::ProcessOutput output;
    try {
        output = run_tool(command, stop, cancel);
    } catch (const platform::ProcessError& error) {
        throw PerceptualRunError("Could not run " + metric + ": " + error.what());
    }
    if (output.stopped) {
        throw Cancelled();
    }
    if (output.timed_out) {
        throw PerceptualRunError(metric + " did not finish a frame within " + std::to_string(tool_timeout.count()) +
                                 " s.");
    }
    const std::string combined = as_text(output.standard_output) + "\n" + as_text(output.standard_error);
    if (output.exit_code != 0) {
        throw PerceptualRunError(metric + " failed for a frame.", last_characters(combined, 2000));
    }
    if (distortion_map) {
        try {
            return butteraugli_norm3(*distortion_map);
        } catch (const PerceptualRunError&) {
            throw;
        } catch (const std::exception&) {
            throw PerceptualRunError("butteraugli wrote no readable distortion map.", last_characters(combined, 2000));
        }
    }
    return parse_score(metric, combined);
}

// The frame pairs FFmpeg writes into a folder, handed out as they are
// complete (the Python app's _png_pairs).
//
// One FFmpeg decodes both videos and writes lossless 16-bit RGB PNG files
// (png_pairs_command). Each picture is written under a temporary name and
// renamed when complete, so a picture that exists is whole; a pair is handed
// out once FFmpeg has let go of both its pictures too (ready).
//
// FFmpeg is suspended while it has `backlog` complete pairs ready ahead of
// the caller, and resumed when it has half as many. A thread of its own
// checks this every 10 ms, because checking only as each pair is handed out
// would let FFmpeg run far ahead while one slow pair is scored. Only
// complete pairs count. The two picture sequences come from two decoders,
// and one can run well ahead of the other (a 4K AV1 test video ran 24
// frames ahead of its HEVC source), so suspending because of the side that
// is ahead would stop the side that is behind too, and the pair the caller
// waits for would never come. So while the caller waits, FFmpeg always
// runs.
//
// Each output runs to its own input's end. The comparison is the frames
// both videos have (libvmaf's shortest=1), so the pairs end when either
// sequence does.
class PngPairs {
public:
    // A pair: the source's picture, then the test video's.
    using Pair = std::pair<std::filesystem::path, std::filesystem::path>;

    // Starts FFmpeg with `command`, which writes into `folder`. `stop` makes
    // a caller waiting for a pair give up (another pair failed). Pause holds
    // FFmpeg as the backlog does: each of the two holds it while it wants it
    // held (ChildProcess::suspend).
    PngPairs(const std::vector<std::string>& command, std::filesystem::path folder, int backlog,
             const std::atomic<bool>& stop, const CancelToken& cancel)
        : folder_(std::move(folder)),
          backlog_(std::max(2, backlog)),
          stop_(stop),
          cancel_(cancel),
          ffmpeg_(platform::ChildProcess::start(command, watched(rebuilt_))),
          attached_(cancel.attach(ffmpeg_)),
          regulator_([this] { regulate(); }) {}

    // Stops FFmpeg if the pairs ended early.
    ~PngPairs() {
        ending_ = true;
        regulator_.join();
        throttle(false);
        if (!ffmpeg_.wait_for(std::chrono::milliseconds(0))) {
            ffmpeg_.terminate();
            ffmpeg_.wait();
        }
    }

    PngPairs(const PngPairs&) = delete;
    PngPairs& operator=(const PngPairs&) = delete;

    // The next pair once both its pictures are whole and FFmpeg has let go
    // of them; nothing once the shorter video has ended, or when `stop` is
    // set. Throws Cancelled, PerceptualRunError when FFmpeg fails, and
    // VideoSizeChanged when it rebuilt its filters (size_changes.h): its
    // pictures from then on are not what the task scores.
    std::optional<Pair> next() {
        if (ended_) {
            return std::nullopt;
        }
        consumed_ = index_;
        Pair found = pair(index_);
        while (!ready(index_)) {
            cancel_.throw_if_cancelled();
            if (stop_) {
                return std::nullopt;
            }
            if (*rebuilt_) {
                throw VideoSizeChanged();
            }
            if (ffmpeg_.wait_for(std::chrono::milliseconds(0))) {
                if (complete(index_)) {
                    break;  // written just before FFmpeg ended
                }
                // Ended by the run's Cancel since the check above, it was
                // stopped, not failed.
                cancel_.throw_if_cancelled();
                static_cast<void>(ffmpeg_.standard_error());  // waits for the error output's end, and so its lines
                if (*rebuilt_) {
                    throw VideoSizeChanged();
                }
                if (ffmpeg_.wait() != 0) {
                    throw failure("FFmpeg could not prepare lossless perceptual-metric frames.");
                }
                if (index_ == 1) {
                    throw failure("FFmpeg produced no frame pairs for perceptual metrics.");
                }
                ended_ = true;  // the shorter input has ended
                return std::nullopt;
            }
            throttle(false);  // the caller is waiting, so FFmpeg must run
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (*rebuilt_) {
            throw VideoSizeChanged();
        }
        ++index_;
        return found;
    }

    // After a pair's tool failed: whether FFmpeg had rebuilt its filters, and
    // so the failure was a pair of pictures of two sizes, made as FFmpeg's
    // word of it was on its way. FFmpeg is ended first, its pictures no
    // longer wanted, so that all it wrote has been read.
    bool rebuilt_before_failure() {
        ffmpeg_.terminate();
        ffmpeg_.wait();
        static_cast<void>(ffmpeg_.standard_error());  // waits for the error output's end, and so its lines
        return *rebuilt_;
    }

private:
    // The options of an FFmpeg watched for rebuilding its filters, which
    // sets `rebuilt`.
    static platform::ProcessOptions watched(const std::shared_ptr<std::atomic<bool>>& rebuilt) {
        platform::ProcessOptions options;
        options.on_error_line = rebuild_watcher(rebuilt);
        return options;
    }

    Pair pair(long long number) const {
        const std::string digits = std::to_string(number);
        const std::string name = std::string(digits.size() < 8 ? 8 - digits.size() : 0, '0') + digits + ".png";
        return {folder_ / ("reference-" + name), folder_ / ("test-" + name)};
    }

    // Whether both pictures of pair `number` have been written.
    bool complete(long long number) const {
        const auto [reference, test] = pair(number);
        std::error_code error;
        return std::filesystem::exists(reference, error) && std::filesystem::exists(test, error);
    }

    // Whether both pictures of pair `number` have been written and can be
    // opened, as describe_png and the tools open them. On Windows, FFmpeg
    // can still hold a picture for a moment after giving it its name (up to
    // 77 ms was seen), and an FFmpeg ended by Cancel until it is gone
    // (2-4 ms). A picture opened then could not be read, so the task failed,
    // even when it had been cancelled.
    bool ready(long long number) const {
        if (!complete(number)) {
            return false;
        }
        const auto [reference, test] = pair(number);
        return std::ifstream(reference, std::ios::binary).is_open() &&
               std::ifstream(test, std::ios::binary).is_open();
    }

    // Suspends FFmpeg (`on`) or lets it run again.
    void throttle(bool on) {
        const std::lock_guard lock(throttle_mutex_);
        if (on == throttled_) {
            return;
        }
        if (on) {
            ffmpeg_.suspend();
        } else {
            ffmpeg_.resume();
        }
        throttled_ = on;
    }

    // Suspends FFmpeg when it is `backlog` pairs ahead of the caller, and
    // lets it run again at half that.
    void regulate() {
        // This thread does very little, but must do it on time. At normal
        // priority, with every core busy, it wakes late and FFmpeg writes far
        // past the limit (a median of 70-125 pictures against 52, with 22
        // busy processes on 24 cores). A higher priority lets it run ahead of
        // the load.
        platform::raise_thread_priority();
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (ending_) {
                return;
            }
            const long long index = consumed_;
            if (complete(index + backlog_)) {
                throttle(true);
            } else if (!complete(index + backlog_ / 2)) {
                throttle(false);
            }
        }
    }

    // A PerceptualRunError with `message` and the end of FFmpeg's errors.
    PerceptualRunError failure(const std::string& message) {
        const std::string errors = error_lines_of(ffmpeg_.standard_error());
        return PerceptualRunError(message, last_characters(python_strip(as_text(errors)), 2000));
    }

    std::filesystem::path folder_;
    int backlog_;
    const std::atomic<bool>& stop_;
    const CancelToken& cancel_;
    std::shared_ptr<std::atomic<bool>> rebuilt_ = std::make_shared<std::atomic<bool>>(false);  // before ffmpeg_
    platform::ChildProcess ffmpeg_;
    CancelToken::Attached attached_;  // after ffmpeg_, so that it goes first
    long long index_ = 1;  // the number of the next pair
    bool ended_ = false;
    std::atomic<long long> consumed_{1};  // the number of the pair the caller is on, read by the regulator
    std::mutex throttle_mutex_;
    bool throttled_ = false;
    std::atomic<bool> ending_{false};
    std::thread regulator_;  // last, so that it starts once the rest is set up
};

// The scores of a task's pairs, gathered from the threads that score them.
struct Scores {
    std::mutex mutex;
    std::vector<std::vector<double>> by_pair;  // each pair's scores, in the keys' order
    long long finished = 0;
    std::exception_ptr failure;  // the first, which ends the others' tools
    std::atomic<bool> stop{false};
    std::chrono::steady_clock::time_point reported{};
};

}  // namespace

// ----------------------------------------------------------------- public

std::optional<std::filesystem::path> find_metric_executable(std::string_view metric) {
    if (metric != "ssimulacra2" && metric != "butteraugli") {
        throw std::invalid_argument("not a CPU tool's metric: " + std::string(metric));
    }
    const std::string variable = metric == "ssimulacra2" ? "SSIMULACRA2_PATH" : "BUTTERAUGLI_PATH";
    const std::string chosen(python_strip(platform::environment_variable(variable).value_or("")));
    if (!chosen.empty()) {
        const std::filesystem::path path = platform::path_from_text(chosen);
        std::error_code error;
        return std::filesystem::exists(path, error) ? std::optional(path) : std::nullopt;
    }
    // libjxl names the tool butteraugli_main. The short name is accepted too,
    // for tools the user installed. Elsewhere than Windows, never the .exe
    // files: the repository's resources folder has Windows' tools.
#ifdef _WIN32
    const std::vector<std::string> names =
        metric == "ssimulacra2"
            ? std::vector<std::string>{"ssimulacra2.exe", "ssimulacra2"}
            : std::vector<std::string>{"butteraugli_main.exe", "butteraugli.exe", "butteraugli_main", "butteraugli"};
#else
    const std::vector<std::string> names = metric == "ssimulacra2"
                                               ? std::vector<std::string>{"ssimulacra2"}
                                               : std::vector<std::string>{"butteraugli_main", "butteraugli"};
#endif
    for (const std::string& name : names) {
        const std::filesystem::path candidate = libjxl_folder() / name;
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) {
            return candidate;
        }
    }
    for (const std::string& name : names) {
        if (std::optional<std::filesystem::path> found = platform::find_on_path(name)) {
            return found;
        }
    }
    return std::nullopt;
}

std::string implementation_version(const std::filesystem::path& executable) {
    const std::string version = tool_version(executable);
    std::error_code error;
    const std::filesystem::path bundle = std::filesystem::weakly_canonical(libjxl_folder(), error);
    const bool bundled = !error && std::filesystem::weakly_canonical(executable, error).parent_path() == bundle && !error;
    if (version == "unknown" && bundled) {
        if (const std::optional<std::string> text = platform::read_file(bundle / "LIBJXL-VERSION.txt")) {
            return std::string(python_strip(std::string_view(*text).substr(0, text->find_first_of("\r\n"))));
        }
    }
    return version;
}

int scoring_workers(long long pixels, const std::vector<std::string>& metrics, int concurrent_tasks, unsigned cores,
                    std::optional<unsigned long long> free_memory) {
    concurrent_tasks = std::max(1, concurrent_tasks);
    const long long by_cores = static_cast<long long>(std::max(1u, cores)) / 2 / concurrent_tasks;
    long long bytes_per_pixel = 0;
    for (const std::string& metric : metrics) {
        bytes_per_pixel = std::max(bytes_per_pixel, tool_bytes_per_pixel(metric));
    }
    const long long per_worker = std::max(1LL, pixels) * std::max(1LL, bytes_per_pixel);
    long long workers = by_cores;
    if (free_memory) {
        const auto share = static_cast<long long>(static_cast<double>(*free_memory) * memory_share / concurrent_tasks);
        workers = std::min(workers, share / per_worker);
    }
    return static_cast<int>(std::max(1LL, workers));
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

double butteraugli_norm3(const std::filesystem::path& distortion_map) {
    const std::optional<std::string> read = platform::read_file(distortion_map);
    if (!read) {
        throw std::runtime_error("the distortion map could not be read");
    }
    const std::string& data = *read;
    // Three lines, "Pf", "<width> <height>" and the scale, whose sign gives
    // the byte order, then the values, as 32-bit floats.
    std::array<std::string_view, 3> lines;
    std::size_t start = 0;
    for (std::string_view& line : lines) {
        const std::size_t end = data.find('\n', start);
        if (end == std::string::npos) {
            throw std::runtime_error("the distortion map has no PFM header");
        }
        line = std::string_view(data).substr(start, end - start);
        start = end + 1;
    }
    if (python_strip(lines[0]) != "Pf") {
        throw PerceptualRunError("butteraugli wrote a distortion map that is not a greyscale PFM.");
    }
    std::vector<std::string_view> size;
    for (std::size_t at = 0; at < lines[1].size();) {
        const std::size_t word = lines[1].find_first_not_of(" \t\n\r\v\f", at);
        if (word == std::string_view::npos) {
            break;
        }
        const std::size_t word_end = std::min(lines[1].find_first_of(" \t\n\r\v\f", word), lines[1].size());
        size.push_back(lines[1].substr(word, word_end - word));
        at = word_end;
    }
    // -1 for a size that is not a whole number.
    const long long width = size.size() == 2 ? python_int(size[0]).value_or(-1) : -1;
    const long long height = size.size() == 2 ? python_int(size[1]).value_or(-1) : -1;
    const std::optional<double> scale = python_float(lines[2]);
    if (!scale || width < 0 || height < 0 || width > (1LL << 20) || height > (1LL << 20)) {
        throw std::runtime_error("the distortion map's header is not a PFM's");
    }
    const auto count = static_cast<std::size_t>(width * height);
    const std::string_view values = std::string_view(data).substr(start);
    if (values.size() < count * 4) {
        throw std::runtime_error("the distortion map is cut short");
    }
    const bool little_endian = *scale < 0;
    std::vector<double> cubes(count);
    // Each cube is pow's, as numpy's ** makes it: x * x * x differs from it
    // in the last bits.
    for (std::size_t i = 0; i < count; ++i) {
        const auto byte = [&values, i](std::size_t k) {
            return static_cast<std::uint32_t>(static_cast<unsigned char>(values[i * 4 + k]));
        };
        const std::uint32_t bits = little_endian ? byte(0) | (byte(1) << 8) | (byte(2) << 16) | (byte(3) << 24)
                                                 : (byte(0) << 24) | (byte(1) << 16) | (byte(2) << 8) | byte(3);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof value);
        cubes[i] = std::pow(std::abs(static_cast<double>(value)), 3.0);
    }
    // numpy's mean: its pairwise sum, divided by the count (NaN for none).
    const double mean = numpy_sum(cubes) / static_cast<double>(count);
    return std::pow(mean, 1.0 / 3.0);
}

FrameRange cpu_range(const VmafOptions& options, const VideoInfo& source, int step) {
    return frame_range(options.range_start, options.range_end, source.fps, step, false);
}

std::string image_filtergraph(const VideoInfo& source_info, const VideoInfo& distorted_info,
                              const VmafOptions& options, const std::optional<CropBox>& source_crop,
                              const std::optional<CropBox>& distorted_crop, int step) {
    // Read with one colour matrix, as Vship reads them (with_shared_matrix).
    const auto [source, distorted] = with_shared_matrix(source_info, distorted_info);
    GraphPlan plan{source, distorted, source_crop, distorted_crop, std::nullopt, std::nullopt,
                   content_size(source, source_crop), {}, options.scale_algorithm};
    const Size distorted_size = content_size(distorted, distorted_crop);
    if (plan.source_size != distorted_size) {
        if (options.scale_direction == ScaleDirection::distorted_to_source) {
            plan.distorted_target = plan.source_size;
        } else {
            plan.source_target = distorted_size;
        }
    }
    if (step > 1) {
        plan.sample.push_back("select=not(mod(n\\," + std::to_string(step) + "))");
    }
    // Both videos' frames are first kept to those a range decodes (counted
    // as decoded), then numbered at the test video's frame rate, so that the
    // frames of both are paired; then sampled, so that the frames select
    // drops are not cropped or scaled.
    const FrameRange range = cpu_range(options, source, step);
    const std::vector<std::string> numbering = numbering_filters(frame_rate(distorted));
    const auto chain = [&plan, &range, &numbering](const std::string& input, const std::string& output,
                                                   const std::optional<CropBox>& crop, const VideoInfo& info,
                                                   const std::optional<Size>& target,
                                                   const std::optional<RangeSeek>& seek) {
        std::vector<std::string> filters;
        if (const std::optional<std::string> trim = range_trim_filter(range, seek)) {
            filters.push_back(*trim);
        }
        append(filters, numbering);
        append(filters, plan.sample);
        for (std::string& filter : prepare(crop, info, target, plan.scale_algorithm)) {
            filters.push_back(std::move(filter));
        }
        for (std::string& filter : to_rgb(info)) {
            filters.push_back(std::move(filter));
        }
        return "[" + input + "]" + join(filters, ",") + "[" + output + "]";
    };
    return chain("0:" + std::string(video_stream), "distorted", distorted_crop, distorted, plan.distorted_target,
                 options.range_seeks.distorted) +
           ";" +
           chain("1:" + std::string(video_stream), "reference", source_crop, source, plan.source_target,
                 options.range_seeks.source);
}

std::vector<std::string> png_pairs_command(const std::string& ffmpeg, const VideoInfo& source,
                                           const VideoInfo& distorted, const VmafOptions& options,
                                           const std::optional<CropBox>& source_crop,
                                           const std::optional<CropBox>& distorted_crop, int step,
                                           const std::filesystem::path& folder,
                                           const std::optional<std::vector<std::string>>& distorted_fed,
                                           const std::optional<std::vector<std::string>>& source_fed) {
    // At watched_log_level: PngPairs reads FFmpeg's word that it rebuilt its filters.
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-loglevel", watched_log_level,
                                     "-nostats", "-y"};
    struct Input {
        const VideoInfo* video;
        const std::optional<std::vector<std::string>>* fed;
        const std::optional<RangeSeek>* seek;
    };
    for (const auto& [video, fed, seek] : {Input{&distorted, &distorted_fed, &options.range_seeks.distorted},
                                           Input{&source, &source_fed, &options.range_seeks.source}}) {
        if (*fed) {
            command.insert(command.end(), (*fed)->begin(), (*fed)->end());
            continue;
        }
        const std::vector<std::string> input = with_seek(video->decode_options(), *seek);
        append(command, input);
        command.insert(command.end(), {as_stored, "-i", input_path(video->path)});
    }
    command.insert(command.end(), {"-filter_complex", image_filtergraph(source, distorted, options, source_crop,
                                                                        distorted_crop, step)});
    const std::vector<std::string> output_args{"-fps_mode", "passthrough", "-pix_fmt",
                                               "rgb48le",    "-atomic_writing", "1"};
    for (const auto& [label, pattern] : {std::pair{"[distorted]", "test-%08d.png"},
                                         std::pair{"[reference]", "reference-%08d.png"}}) {
        command.insert(command.end(), {"-map", label});
        append(command, output_args);
        command.push_back(platform::path_text(folder / pattern));
    }
    return command;
}

namespace {

// The reference executables of `keys`' metrics; throws, naming the first
// one not installed.
std::vector<std::filesystem::path> metric_executables(const std::vector<std::string>& keys) {
    std::vector<std::filesystem::path> executables;
    for (const std::string& key : keys) {
        std::optional<std::filesystem::path> executable = find_metric_executable(key);
        if (!executable) {
            throw PerceptualRunError(key + " is not installed. Install the reference " + key + " executable or set " +
                                     (key == "ssimulacra2" ? "SSIMULACRA2" : "BUTTERAUGLI") + "_PATH.");
        }
        executables.push_back(std::move(*executable));
    }
    return executables;
}

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

// The frame pairs of one CPU perceptual task, scored `workers` pairs at a
// time (the calling thread among them): each pair by every chosen tool in
// turn. The first failure stops the others' tools, and is thrown.
class CpuScoringPool {
public:
    CpuScoringPool(const std::vector<std::string>& keys, const std::vector<std::filesystem::path>& executables,
                   const VideoInfo& source, const VideoInfo& distorted, const CancelToken& cancel)
        : keys_(keys),
          executables_(executables),
          source_colour_(colour_of(source)),
          distorted_colour_(colour_of(distorted)),
          hdr_((source_colour_ && source_colour_->hdr()) || (distorted_colour_ && distorted_colour_->hdr())),
          cancel_(cancel) {}
    CpuScoringPool(const CpuScoringPool&) = delete;
    CpuScoringPool& operator=(const CpuScoringPool&) = delete;

    // Whether either video is HDR: the tools score it so.
    bool hdr() const { return hdr_; }
    // Set when the scoring is to stop (a failure, or Cancel): FFmpeg's
    // extraction stops with it.
    const std::atomic<bool>& stopping() const { return scores_.stop; }
    // Scores every pair `pairs` gives, `workers` at a time, reporting how
    // many are done (`step` frames each, of about `total_units`) to
    // `on_progress`, at `rate_of` frames a second. Returns each pair's
    // scores, in the order FFmpeg gave the pairs.
    std::vector<std::vector<double>> score_all(PngPairs& pairs, int workers, int step, long long total_units,
                                               const ProgressCallback& on_progress,
                                               const std::function<double(long long)>& rate_of);

private:
    std::vector<double> score(const PngPairs::Pair& pair);
    void work(PngPairs& pairs);
    void keep(long long number, std::vector<double> values);
    void fail(std::exception_ptr error);

    const std::vector<std::string>& keys_;
    const std::vector<std::filesystem::path>& executables_;
    // Each picture is tagged so that the tools read its values as Vship
    // reads the video's (describe_png). FFmpeg would tag a BT.709 picture
    // with H.273's BT.709 curve, the camera's, where Vship and a display use
    // a 2.4 gamma, and an untagged picture not at all (sRGB). A tagged BT.709
    // film then scored 40.4 SSIMULACRA2 here against 55.1 on the GPU.
    const std::optional<VideoColour> source_colour_;
    const std::optional<VideoColour> distorted_colour_;
    const bool hdr_;
    const CancelToken& cancel_;
    Scores scores_;
    const std::function<bool()> stopped_ = [this] { return scores_.stop || cancel_.cancelled(); };
    std::mutex pairs_mutex_;
    long long handed_ = 0;  // how many pairs workers have taken
    // How score_all reports.
    int step_ = 1;
    long long total_units_ = 0;
    ProgressCallback on_progress_;
    std::function<double(long long)> rate_of_;
};

std::vector<std::vector<double>> CpuScoringPool::score_all(PngPairs& pairs, int workers, int step,
                                                           long long total_units, const ProgressCallback& on_progress,
                                                           const std::function<double(long long)>& rate_of) {
    step_ = step;
    total_units_ = total_units;
    on_progress_ = on_progress;
    rate_of_ = rate_of;
    std::vector<std::thread> threads;
    try {
        for (int worker = 1; worker < workers; ++worker) {
            threads.emplace_back([this, &pairs] { work(pairs); });
        }
    } catch (...) {
        fail(std::current_exception());  // no thread for another worker: the others stop
    }
    work(pairs);
    // The tools end before FFmpeg and the folder go: a tool still reading
    // its pair would hold the folder.
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (scores_.failure) {
        if (pairs.rebuilt_before_failure()) {
            throw VideoSizeChanged();  // the task is made again, the video scaled first
        }
        std::rethrow_exception(scores_.failure);
    }
    return std::move(scores_.by_pair);
}

// One pair's scores, one tool after the other, in the keys' order. Each pair
// is deleted once every chosen tool has scored it; as FFmpeg is held to a
// small backlog, the folder never holds more than the backlog and the pairs
// being scored.
std::vector<double> CpuScoringPool::score(const PngPairs::Pair& pair) {
    const auto& [reference, test] = pair;
    const Deleted reference_deleted(reference);
    const Deleted test_deleted(test);
    if (source_colour_) {
        describe_png(reference, *source_colour_);
    }
    if (distorted_colour_) {
        describe_png(test, *distorted_colour_);
    }
    std::vector<double> values;
    for (std::size_t position = 0; position < keys_.size(); ++position) {
        values.push_back(run_metric(executables_[position], keys_[position], reference, test, hdr_, stopped_, cancel_));
    }
    return values;
}

// One worker: takes a pair from FFmpeg only when free for it, so that the
// pairs not yet started count in FFmpeg's backlog, which holds it back.
void CpuScoringPool::work(PngPairs& pairs) {
    while (!scores_.stop) {
        std::optional<PngPairs::Pair> pair;
        long long number = 0;
        try {
            const std::lock_guard lock(pairs_mutex_);
            cancel_.throw_if_cancelled();
            pair = pairs.next();
            if (!pair) {
                return;
            }
            number = handed_++;
        } catch (...) {
            fail(std::current_exception());
            return;
        }
        try {
            keep(number, score(*pair));
        } catch (...) {
            fail(std::current_exception());
            return;
        }
    }
}

// Keeps pair `number`'s scores. The progress counts how many pairs are
// scored, in whatever order they finish. The estimated frame count can be
// lower than the real one, so the total grows with the work rather than
// going past 100%.
void CpuScoringPool::keep(long long number, std::vector<double> values) {
    const std::lock_guard lock(scores_.mutex);
    if (scores_.by_pair.size() <= static_cast<std::size_t>(number)) {
        scores_.by_pair.resize(static_cast<std::size_t>(number) + 1);
    }
    scores_.by_pair[static_cast<std::size_t>(number)] = std::move(values);
    ++scores_.finished;
    const auto now = std::chrono::steady_clock::now();
    if (on_progress_ && (scores_.finished == 1 || now - scores_.reported >= progress_interval)) {
        scores_.reported = now;
        const long long units = scores_.finished * step_;
        on_progress_(units, std::max(total_units_, units + step_), rate_of_(units));
    }
}

// Keeps the first failure, and ends the tools of every pair in progress.
void CpuScoringPool::fail(std::exception_ptr error) {
    const std::lock_guard lock(scores_.mutex);
    if (!scores_.failure) {
        scores_.failure = std::move(error);
    }
    scores_.stop = true;
}

// The scores `by_pair` (each pair's, in `keys`' order, scored by
// `executables`) as the task's results: frames `step` apart from
// range.decode_start, those in range kept, timed at `fps`, with how they
// were made.
MetricResultSet cpu_results(const std::vector<std::string>& keys, const std::vector<std::filesystem::path>& executables,
                            const std::vector<std::vector<double>>& by_pair, int step, double fps, bool hdr,
                            const FrameRange& range) {
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
        MetricProvenance provenance{key, implementation_version(executables[position]), "cpu",
                                    score_identity::libjxl_cpu(key), std::move(parameters), build_stamp()};
        mark_range(provenance, range);
        results.add(FrameMetricResult{key, frames, times, std::move(values), std::move(provenance)});
    }
    return results;
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
    const VideoInfo& source = request.source;
    const VideoInfo& distorted = request.distorted;
    VmafOptions options = request.options;
    if (options.resample_test) {
        throw PerceptualRunError("Perceptual CPU metrics do not support resolution round-trip tests yet.");
    }
    validate_pair(source, distorted, options);
    const std::vector<std::filesystem::path> executables = metric_executables(keys);
    const int step = common_step(request, keys);
    const auto [source_crop, distorted_crop] =
        crops ? *crops
              : resolve_crops(request.environment.ffmpeg, source, distorted, options.crop_mode,
                              options.scale_algorithm, cancel, on_status);
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
    // Tagged with the colours the pictures are read with (with_shared_matrix).
    const auto [source_read, distorted_read] = with_shared_matrix(source, distorted);
    const auto [width, height] =
        compared_dimensions(source, distorted, options.scale_direction, source_crop, distorted_crop);
    const int workers = scoring_workers(static_cast<long long>(width) * height, keys, request.concurrent_cpu_tasks,
                                        platform::logical_cores(), platform::available_memory());
    logger.info(join(keys, " and ") + " on the CPU: " + std::to_string(workers) + " frame pair" + (workers == 1 ? "" : "s") +
                " at a time");

    std::optional<CpuScoringPool> pool;
    std::vector<std::vector<double>> by_pair;
    // Made again, with a new pool, pictures and progress, where a feeding's pictures from vvdec-vulkan failed
    // (VvdecFeedFailed: vvdec-vulkan off then, so once)
    for (bool again = true; again;) {
        again = false;
        pool.emplace(keys, executables, source_read, distorted_read, cancel);
        started = std::chrono::steady_clock::now();
        try {
            // Before FFmpeg, which it outlives.
            PairFeeds feeds(&source, &distorted, request.environment.ffmpeg, request.environment.ffprobe, cancel,
                            vvdec_wanted(options.gpu_decode, options.vvc_on_gpu), on_status);
            const auto fed_input = [](const PictureFeed* feed) {
                return feed != nullptr ? std::optional(feed->input_arguments()) : std::nullopt;
            };
            try {
                // A folder of its own for the task's pictures, deleted with them.
                const TempFolder folder = TempFolder::make<PerceptualRunError>(
                    "videometricslab-perceptual-", "A folder for the frame pictures could not be made.");
                // Room for every worker's next pair while FFmpeg wakes up.
                PngPairs pairs(png_pairs_command(request.environment.ffmpeg, source, distorted, options, source_crop,
                                                 distorted_crop, step, folder.path(), fed_input(feeds.distorted()),
                                                 fed_input(feeds.source())),
                               folder.path(), std::max(backlog_pairs, 2 * workers), pool->stopping(), cancel);
                by_pair = pool->score_all(pairs, workers, step, total_units, on_progress, rate_of);
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
            return run_cpu_perceptual_task(request, keys, crops, on_progress, on_status, cancel);
        }
        throw ComparisonCutShortError(*short_reason);
    }
    if (on_progress) {
        on_progress(total * step, total * step, rate_of(total * step));
    }
    MetricResultSet results = cpu_results(keys, executables, by_pair, step, source.fps, pool->hdr(), range);
    const long long compared = static_cast<long long>(results.frame(keys.front())->frame.size()) * step;
    return PerceptualTaskOutput{std::move(results), source_crop, distorted_crop, compared, {}};
}

}  // namespace vml
