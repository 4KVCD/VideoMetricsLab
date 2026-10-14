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
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

#include "vml/colour.h"
#include "vml/ffmpeg_tools.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/stats.h"
#include "vml/temp_files.h"

namespace vml {

namespace {

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

// How the pictures' colours are read: as Vship reads the video's (colour.h),
// recorded with the scores, as the Python app records it (metric_cache's
// CPU_COLOR_TAGS).
constexpr std::string_view cpu_color_tags = "vship-5.1.1/2";

// How often the progress is reported at most, after the first pair: small
// pictures are scored many times a second.
constexpr std::chrono::milliseconds progress_interval{250};

// The memory one tool takes at its peak, in bytes per pixel of the pair it
// scores, with a margin. On a 3840x2160 pair, SSIMULACRA2 took 1.19 GB (143
// bytes per pixel) and Butteraugli 1.85 GB (223 bytes per pixel).
long long tool_bytes_per_pixel(const std::string& metric) { return metric == "ssimulacra2" ? 160 : 250; }

std::filesystem::path libjxl_folder() { return resource_folder() / "tools" / "libjxl"; }

// Whether `c` is a byte of the first character of a UTF-8 sequence (or of an
// ASCII one), so that text can be cut by characters, as Python cuts it.
bool starts_character(char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }

// The first `count` characters of UTF-8 text (Python's text[:count]).
std::string first_characters(std::string_view text, std::size_t count) {
    std::size_t seen = 0;
    for (std::size_t at = 0; at < text.size(); ++at) {
        if (starts_character(text[at]) && seen++ == count) {
            return std::string(text.substr(0, at));
        }
    }
    return std::string(text);
}

// The last `count` characters of UTF-8 text (Python's text[-count:]).
std::string last_characters(std::string_view text, std::size_t count) {
    std::size_t seen = 0;
    for (std::size_t at = text.size(); at > 0; --at) {
        if (starts_character(text[at - 1]) && ++seen == count) {
            return std::string(text.substr(at - 1));
        }
    }
    return std::string(text);
}

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
        platform::ProcessOutput output;
        try {
            output = platform::run_program({platform::path_text(executable), flag}, std::chrono::seconds(5));
        } catch (const platform::ProcessError&) {
            return "unknown";
        }
        if (output.timed_out) {
            return "unknown";
        }
        if (output.exit_code == 0) {
            const std::string text =
                as_text(!output.standard_output.empty() ? output.standard_output : output.standard_error);
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
    if (!crop || crop->is_noop(info.width, info.height)) {
        return {};
    }
    return {crop->as_filter()};
}

std::string joined(const std::vector<std::string>& filters, std::string_view separator = ",") {
    std::string text;
    for (const std::string& filter : filters) {
        text += (text.empty() ? "" : std::string(separator)) + filter;
    }
    return text;
}

// The filters that crop a video's black bars, and scale it to `target`
// when it is scaled. It is scaled in its own pixel format, as Vship's
// pictures are, and converted to RGB only after its colour tags are set
// (to_rgb): left to make the RGB itself, the scale filter would do so
// before the tags are set, and convert an untagged HD source with BT.601's
// matrix instead of BT.709's.
std::vector<std::string> prepare(const std::optional<CropBox>& crop, const VideoInfo& info,
                                 const std::optional<Size>& target, const std::string& scale_algorithm) {
    std::vector<std::string> filters = crop_filter(crop, info);
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
// format negotiation would put a conversion.
std::vector<std::string> to_rgb(const VideoInfo& info) {
    std::vector<std::string> filters;
    if (const std::optional<VideoColour> colour = colour_of(info)) {
        if (const std::optional<std::string_view> matrix = ffmpeg_matrix_name(colour->matrix)) {
            filters.push_back("setparams=colorspace=" + std::string(*matrix) +
                              ":range=" + (colour->full_range ? "pc" : "tv"));
        }
    }
    filters.push_back("scale");
    filters.push_back("format=rgb48le");
    return filters;
}

// Whether FFmpeg's blend filter takes frames of this pixel format as they
// are, so that it can carry a source frame through unchanged
// (pairs_by_time): the planar YUV, GBR and gray formats,
// (yuv(420|422|444)p|gbrp|gray)(9|10|12|14|16)?(le)?. Each was checked frame
// by frame against the source's own frames.
bool blend_takes(std::string_view format) {
    std::string_view rest = format;
    bool known = false;
    for (const std::string_view family : {"yuv420p", "yuv422p", "yuv444p", "gbrp", "gray"}) {
        if (rest.starts_with(family)) {
            rest.remove_prefix(family.size());
            known = true;
            break;
        }
    }
    if (!known) {
        return false;
    }
    for (const std::string_view depth : {"9", "10", "12", "14", "16"}) {
        if (rest.starts_with(depth)) {
            rest.remove_prefix(depth.size());
            break;
        }
    }
    return rest.empty() || rest == "le";
}

// The names setparams knows for each colour tag, as ffprobe prints them.
const std::map<std::string, std::set<std::string, std::less<>>, std::less<>> setparams_names{
    {"range", {"unknown", "tv", "pc"}},
    {"color_primaries",
     {"bt709", "unknown", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "bt2020", "smpte428", "smpte431",
      "smpte432", "jedec-p22", "ebu3213", "vgamut"}},
    {"color_trc",
     {"bt709", "unknown", "bt470m", "bt470bg", "smpte170m", "smpte240m", "linear", "log100", "log316",
      "iec61966-2-4", "bt1361e", "iec61966-2-1", "bt2020-10", "bt2020-12", "smpte2084", "smpte428", "arib-std-b67",
      "vlog"}},
    {"colorspace",
     {"gbr", "bt709", "unknown", "fcc", "bt470bg", "smpte170m", "smpte240m", "ycgco", "ycgco-re", "ycgco-ro",
      "bt2020nc", "bt2020c", "smpte2085", "chroma-derived-nc", "chroma-derived-c", "ictcp", "ipt-c2"}},
    {"chroma_location", {"unspecified", "left", "center", "topleft", "top", "bottomleft", "bottom"}},
};

// A setparams filter that gives frames `info`'s colour tags again, or
// nothing when setparams has no name for one of them.
std::optional<std::string> frame_tags(const VideoInfo& info) {
    const std::array<std::pair<std::string, std::string>, 5> tags{{
        {"range", info.color_range.empty() ? "unknown" : info.color_range},
        {"color_primaries", info.color_primaries.empty() ? "unknown" : info.color_primaries},
        {"color_trc", info.color_transfer.empty() ? "unknown" : info.color_transfer},
        {"colorspace", info.color_space.empty() ? "unknown" : info.color_space},
        {"chroma_location", info.chroma_location.empty() ? "unspecified" : info.chroma_location},
    }};
    std::vector<std::string> options;
    for (const auto& [option, value] : tags) {
        const std::string name = ascii_lower(value);
        if (!setparams_names.at(option).contains(name)) {
            return std::nullopt;
        }
        options.push_back(option + "=" + name);
    }
    return "setparams=" + joined(options, ":");
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

// image_filtergraph's streams when frames are paired by timestamp.
//
// The two streams' Nth pictures are a pair: the test video's Nth frame, and
// the source frame that libvmaf pairs it with by timestamp (FFmpeg's frame
// sync, framesync_opts). The blend filter makes the pairs with the same
// frame sync. It ORs a "clock" of all-zero frames, one at each test frame's
// time, with the source's frames, which gives one source frame per test
// frame with the source's pixels unchanged. The source's own colour tags are
// then set again, because blend's output takes the clock's tags, which are
// the test video's. Every step-th pair is kept. A source in a pixel format
// that blend does not take as it is (packed RGB, NV12, full-range "yuvj")
// is paired by position instead.
std::string pairs_by_time(const GraphPlan& plan) {
    const std::string source_format = ascii_lower(plan.source.pix_fmt);
    const std::optional<std::string> restore = frame_tags(plan.source);
    const auto prepared = [&plan](const std::optional<CropBox>& crop, const VideoInfo& info,
                                  const std::optional<Size>& target) {
        return prepare(crop, info, target, plan.scale_algorithm);
    };
    if (!blend_takes(source_format) || !restore) {
        // select comes first, so that the frames it drops are not cropped or
        // scaled.
        const auto chain = [&plan, &prepared](const std::string& input, const std::string& output,
                                              const std::optional<CropBox>& crop, const VideoInfo& info,
                                              const std::optional<Size>& target) {
            std::vector<std::string> filters = plan.sample;
            for (std::string& filter : prepared(crop, info, target)) {
                filters.push_back(std::move(filter));
            }
            filters.emplace_back("setpts=PTS-STARTPTS");
            for (std::string& filter : to_rgb(info)) {
                filters.push_back(std::move(filter));
            }
            return "[" + input + "]" + joined(filters) + "[" + output + "]";
        };
        return chain("0:" + std::string(video_stream), "distorted", plan.distorted_crop, plan.distorted,
                     plan.distorted_target) +
               ";" +
               chain("1:" + std::string(video_stream), "reference", plan.source_crop, plan.source,
                     plan.source_target);
    }

    const auto [width, height] = plan.source_target.value_or(plan.source_size);
    // pad keeps a subsampled picture on whole chroma samples: asked for an
    // odd size, it gives the even size below, and blend then refuses the two
    // inputs ("size 852x480 do not match ... 853x480"). So an odd-sized clock
    // is padded to the even size above, then cropped to the picture's size
    // exactly.
    std::string clock_size = "pad=" + std::to_string(width + (width & 1)) + ":" + std::to_string(height + (height & 1));
    if ((width & 1) != 0 || (height & 1) != 0) {
        clock_size += ",crop=" + std::to_string(width) + ":" + std::to_string(height) + ":0:0:exact=1";
    }
    std::vector<std::string> test_chain = prepared(plan.distorted_crop, plan.distorted, plan.distorted_target);
    test_chain.emplace_back("setpts=PTS-STARTPTS");
    test_chain.emplace_back("split=2[test_frames][test_times]");
    std::vector<std::string> source_chain = prepared(plan.source_crop, plan.source, plan.source_target);
    if (!plan.source_target) {
        source_chain.push_back("format=" + source_format);
    }
    source_chain.emplace_back("setpts=PTS-STARTPTS");
    std::vector<std::string> reference_end = plan.sample;
    for (std::string& filter : to_rgb(plan.source)) {
        reference_end.push_back(std::move(filter));
    }
    std::vector<std::string> distorted_end = plan.sample;
    for (std::string& filter : to_rgb(plan.distorted)) {
        distorted_end.push_back(std::move(filter));
    }
    const std::string stream = std::string(video_stream);
    return joined({
        "[0:" + stream + "]" + joined(test_chain),
        // The clock has a scale filter of its own, because it is made in the
        // source's format, while the test frames reach the split (and the
        // test's pictures) as they are decoded.
        "[test_times]crop=2:2:0:0,scale,format=" + source_format + "," + clock_size +
            ",lut=c0=0:c1=0:c2=0:c3=0[clock]",
        "[1:" + stream + "]" + joined(source_chain) + "[source_frames]",
        "[clock][source_frames]blend=all_mode=or:" + joined(framesync_opts(), ":") + "," + *restore + "," +
            joined(reference_end) + "[reference]",
        "[test_frames]" + joined(distorted_end) + "[distorted]",
    }, ";");
}

// ------------------------------------------------------------ the tools

// A folder of its own for a task's pictures, deleted with everything in it.
class PictureFolder {
public:
    PictureFolder() {
        std::random_device random;
        for (int attempt = 0; attempt < 100; ++attempt) {
            std::filesystem::path candidate = std::filesystem::temp_directory_path() /
                                              (temp_prefix("videometricslab-perceptual-") + std::to_string(random()));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                path_ = candidate;
                return;
            }
        }
        throw PerceptualRunError("A folder for the frame pictures could not be made.");
    }
    ~PictureFolder() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    PictureFolder(const PictureFolder&) = delete;
    PictureFolder& operator=(const PictureFolder&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

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
            std::string nits = python_repr(butteraugli_intensity_nits);
            if (nits.ends_with(".0")) {
                nits.resize(nits.size() - 2);
            }
            command.insert(command.end(), {"--intensity_target", nits});
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
// renamed when complete, so a picture that exists is whole.
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
          ffmpeg_(platform::ChildProcess::start(command)),
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

    // The next pair once both its pictures are whole; nothing once the
    // shorter video has ended, or when `stop` is set. Throws Cancelled, and
    // PerceptualRunError when FFmpeg fails.
    std::optional<Pair> next() {
        if (ended_) {
            return std::nullopt;
        }
        consumed_ = index_;
        Pair found = pair(index_);
        while (!complete(index_)) {
            cancel_.throw_if_cancelled();
            if (stop_) {
                return std::nullopt;
            }
            if (ffmpeg_.wait_for(std::chrono::milliseconds(0))) {
                if (complete(index_)) {
                    break;  // written just before FFmpeg ended
                }
                // Ended by the run's Cancel since the check above, it was
                // stopped, not failed.
                cancel_.throw_if_cancelled();
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
        ++index_;
        return found;
    }

private:
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
        // priority, with every core busy, it woke late and FFmpeg wrote far
        // past the limit (a median of 70-125 pictures against 52, with 22
        // busy processes on 24 cores, in the Python app). A higher priority
        // lets it run ahead of the load.
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

    // A PerceptualRunError with `message` and the end of FFmpeg's error
    // output.
    PerceptualRunError failure(const std::string& message) {
        return PerceptualRunError(message, last_characters(python_strip(as_text(ffmpeg_.standard_error())), 2000));
    }

    std::filesystem::path folder_;
    int backlog_;
    const std::atomic<bool>& stop_;
    const CancelToken& cancel_;
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
    // for tools the user installed.
    const std::vector<std::string> names =
        metric == "ssimulacra2"
            ? std::vector<std::string>{"ssimulacra2.exe", "ssimulacra2"}
            : std::vector<std::string>{"butteraugli_main.exe", "butteraugli.exe", "butteraugli_main", "butteraugli"};
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

std::string image_filtergraph(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                              int step) {
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
    if (by_timestamp()) {
        return pairs_by_time(plan);
    }
    // Both videos' frames are first numbered at the test video's frame rate,
    // so that the duration limit (each output's -t) keeps the same frames of
    // both; then sampled, so that the frames select drops are not cropped or
    // scaled.
    const std::vector<std::string> numbering = numbering_filters(frame_rate(distorted));
    const auto chain = [&plan, &numbering](const std::string& input, const std::string& output,
                                           const std::optional<CropBox>& crop, const VideoInfo& info,
                                           const std::optional<Size>& target) {
        std::vector<std::string> filters = numbering;
        filters.insert(filters.end(), plan.sample.begin(), plan.sample.end());
        for (std::string& filter : prepare(crop, info, target, plan.scale_algorithm)) {
            filters.push_back(std::move(filter));
        }
        for (std::string& filter : to_rgb(info)) {
            filters.push_back(std::move(filter));
        }
        return "[" + input + "]" + joined(filters) + "[" + output + "]";
    };
    return chain("0:" + std::string(video_stream), "distorted", distorted_crop, distorted, plan.distorted_target) +
           ";" + chain("1:" + std::string(video_stream), "reference", source_crop, source, plan.source_target);
}

std::vector<std::string> png_pairs_command(const std::string& ffmpeg, const VideoInfo& source,
                                           const VideoInfo& distorted, const VmafOptions& options,
                                           const std::optional<CropBox>& source_crop,
                                           const std::optional<CropBox>& distorted_crop, int step,
                                           const std::filesystem::path& folder) {
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                                     "-i", input_path(distorted.path), "-i", input_path(source.path),
                                     "-filter_complex",
                                     image_filtergraph(source, distorted, options, source_crop, distorted_crop, step)};
    std::vector<std::string> output_args;
    if (options.duration_limit > 0) {
        output_args = duration_args(format_fixed(options.duration_limit, 3));
    }
    for (const char* arg : {"-fps_mode", "passthrough", "-pix_fmt", "rgb48le", "-atomic_writing", "1"}) {
        output_args.emplace_back(arg);
    }
    for (const auto& [label, pattern] : {std::pair{"[distorted]", "test-%08d.png"},
                                         std::pair{"[reference]", "reference-%08d.png"}}) {
        command.insert(command.end(), {"-map", label});
        command.insert(command.end(), output_args.begin(), output_args.end());
        command.push_back(platform::path_text(folder / pattern));
    }
    return command;
}

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
    const VmafOptions& options = request.options;
    validate_pair(source, distorted, options.duration_limit);
    std::vector<std::filesystem::path> executables;
    for (const std::string& key : keys) {
        std::optional<std::filesystem::path> executable = find_metric_executable(key);
        if (!executable) {
            throw PerceptualRunError(key + " is not installed. Install the reference " + key + " executable or set " +
                                     (key == "ssimulacra2" ? "SSIMULACRA2" : "BUTTERAUGLI") + "_PATH.");
        }
        executables.push_back(std::move(*executable));
    }
    std::set<int> steps;
    for (const std::string& key : keys) {
        steps.insert(request.step_of(key));
    }
    if (steps.size() != 1) {
        throw PerceptualRunError("Perceptual metrics in one task must use the same frame coverage.");
    }
    const int step = *steps.begin();
    const auto [source_crop, distorted_crop] =
        crops ? *crops
              : resolve_crops(request.environment.ffmpeg, source, distorted, options.crop_mode, cancel, on_status);
    long long expected = std::min(source.estimated_frame_count(), distorted.estimated_frame_count());
    if (options.duration_limit > 0) {
        expected = std::min(expected,
                            std::max(1LL, static_cast<long long>(std::ceil(options.duration_limit * source.fps))));
    }
    const long long total_units = std::max(1LL, (expected + step - 1) / step) * step;
    if (on_status) {
        on_status(decoding_status("Calculating SSIMULACRA2/Butteraugli on the CPU as frames are extracted",
                                  HardwareDecoding{}, StatusKind::step, "\xE2\x80\xA6"));
    }
    const auto started = std::chrono::steady_clock::now();
    const auto rate_of = [&started](long long units) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return static_cast<double>(units) / std::max(seconds, 1e-6);
    };
    // Each picture is tagged so that the tools read its values as Vship
    // reads the video's (describe_png). FFmpeg would tag a BT.709 picture
    // with H.273's BT.709 curve, the camera's, where Vship and a display use
    // a 2.4 gamma, and an untagged picture not at all (sRGB). A tagged BT.709
    // film then scored 40.4 SSIMULACRA2 here against 55.1 on the GPU.
    const std::optional<VideoColour> source_colour = colour_of(source);
    const std::optional<VideoColour> distorted_colour = colour_of(distorted);
    const bool hdr = (source_colour && source_colour->hdr()) || (distorted_colour && distorted_colour->hdr());
    const auto [width, height] =
        compared_dimensions(source, distorted, options.scale_direction, source_crop, distorted_crop);
    const int workers = scoring_workers(static_cast<long long>(width) * height, keys, request.concurrent_cpu_tasks,
                                        platform::logical_cores(), platform::available_memory());

    Scores scores;
    const std::function<bool()> stopped = [&scores, &cancel] { return scores.stop || cancel.cancelled(); };
    // Keeps the first failure, and ends the tools of every pair in progress.
    const auto fail = [&scores](std::exception_ptr error) {
        const std::lock_guard lock(scores.mutex);
        if (!scores.failure) {
            scores.failure = std::move(error);
        }
        scores.stop = true;
    };
    // One pair's scores, one tool after the other, in the keys' order. Each
    // pair is deleted once every chosen tool has scored it; as FFmpeg is
    // held to a small backlog, the folder never holds more than the backlog
    // and the pairs being scored.
    const auto score = [&](const PngPairs::Pair& pair) {
        const auto& [reference, test] = pair;
        const Deleted reference_deleted(reference);
        const Deleted test_deleted(test);
        if (source_colour) {
            describe_png(reference, *source_colour);
        }
        if (distorted_colour) {
            describe_png(test, *distorted_colour);
        }
        std::vector<double> values;
        for (std::size_t position = 0; position < keys.size(); ++position) {
            values.push_back(
                run_metric(executables[position], keys[position], reference, test, hdr, stopped, cancel));
        }
        return values;
    };

    {
        const PictureFolder folder;
        // Room for every worker's next pair while FFmpeg wakes up.
        PngPairs pairs(png_pairs_command(request.environment.ffmpeg, source, distorted, options, source_crop,
                                         distorted_crop, step, folder.path()),
                       folder.path(), std::max(backlog_pairs, 2 * workers), scores.stop, cancel);
        std::mutex pairs_mutex;
        long long handed = 0;  // how many pairs workers have taken
        // A pair is taken from FFmpeg only when a worker is free for it, so
        // the pairs not yet started count in FFmpeg's backlog, which holds it
        // back.
        const auto work = [&] {
            while (!scores.stop) {
                std::optional<PngPairs::Pair> pair;
                long long number = 0;
                try {
                    const std::lock_guard lock(pairs_mutex);
                    cancel.throw_if_cancelled();
                    pair = pairs.next();
                    if (!pair) {
                        return;
                    }
                    number = handed++;
                } catch (...) {
                    fail(std::current_exception());
                    return;
                }
                try {
                    std::vector<double> values = score(*pair);
                    const std::lock_guard lock(scores.mutex);
                    if (scores.by_pair.size() <= static_cast<std::size_t>(number)) {
                        scores.by_pair.resize(static_cast<std::size_t>(number) + 1);
                    }
                    scores.by_pair[static_cast<std::size_t>(number)] = std::move(values);
                    ++scores.finished;
                    // The progress counts how many pairs are scored, in
                    // whatever order they finish. The estimated frame count
                    // can be lower than the real one, so the total grows with
                    // the work rather than going past 100%.
                    const auto now = std::chrono::steady_clock::now();
                    if (on_progress && (scores.finished == 1 || now - scores.reported >= progress_interval)) {
                        scores.reported = now;
                        const long long units = scores.finished * step;
                        on_progress(units, std::max(total_units, units + step), rate_of(units));
                    }
                } catch (...) {
                    fail(std::current_exception());
                    return;
                }
            }
        };
        std::vector<std::thread> threads;
        try {
            for (int worker = 1; worker < workers; ++worker) {
                threads.emplace_back(work);
            }
        } catch (...) {
            fail(std::current_exception());  // no thread for another worker: the others stop
        }
        work();
        // The tools end before FFmpeg and the folder go: a tool still reading
        // its pair would hold the folder.
        for (std::thread& thread : threads) {
            thread.join();
        }
    }
    if (scores.failure) {
        std::rethrow_exception(scores.failure);
    }

    const auto total = static_cast<long long>(scores.by_pair.size());
    if (const std::optional<std::string> short_reason = short_comparison(expected, total * step, source.fps, step)) {
        throw ComparisonCutShortError(*short_reason);
    }
    if (on_progress) {
        on_progress(total * step, total * step, rate_of(total * step));
    }
    std::vector<int> frames;
    std::vector<double> times;
    for (long long index = 0; index < total; ++index) {
        const auto frame = static_cast<int>(index * step);
        frames.push_back(frame);
        times.push_back(static_cast<double>(frame) / std::max(source.fps, 1.0));
    }
    MetricResultSet results;
    for (std::size_t position = 0; position < keys.size(); ++position) {
        const std::string& key = keys[position];
        std::vector<float> values;
        for (const std::vector<double>& pair : scores.by_pair) {
            values.push_back(static_cast<float>(pair[position]));
        }
        // The colours are read as Vship reads them, with its Butteraugli
        // 3-norm and display brightness.
        std::map<std::string, std::string> parameters{
            {"intermediate", nlohmann::json("png/rgb48le").dump()},
            {"coverage_step", std::to_string(step)},
            {"color_tags", nlohmann::json(cpu_color_tags).dump()},
        };
        if (key == "butteraugli") {
            parameters["butteraugli_norm"] = nlohmann::json("3-norm").dump();
            parameters["intensity_target"] =
                hdr ? nlohmann::json("picture's own").dump() : nlohmann::json(butteraugli_intensity_nits).dump();
        }
        results.add(FrameMetricResult{key, frames, times, std::move(values),
                                      MetricProvenance{key, implementation_version(executables[position]), "cpu",
                                                       key + "-libjxl-cpu-v1", std::move(parameters)}});
    }
    return PerceptualTaskOutput{std::move(results), source_crop, distorted_crop, total * step, {}};
}

}  // namespace vml
