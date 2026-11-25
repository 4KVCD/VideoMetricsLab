#include "compare.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <ctime>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>

#include "batch_file.h"
#include "codes.h"
#include "common.h"
#include "file_names.h"
#include "notify.h"
#include "report.h"
#include "stills.h"
#include "watch.h"
#include "weak_report.h"
#include "vml/app_log.h"
#include "vml/append.h"
#include "vml/bd_rate_report.h"
#include "vml/cancel.h"
#include "vml/colour.h"
#include "vml/crop_detect.h"
#include "vml/cvvdp.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/vmaf_hdr.h"
#include "vml/file_fingerprint.h"
#include "vml/film_grain_detect.h"
#include "vml/film_grain_scoring.h"
#include "vml/frame_pairing.h"
#include "vml/frame_range.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_choice.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
#include "vml/isolated.h"
#include "vml/job_runner.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/run_line.h"
#include "vml/settings.h"
#include "vml/tasks_ahead.h"
#include "vml/temp_files.h"
#include "vml/time_format.h"
#include "vml/vship.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vship_run.h"

namespace vml::cli {

namespace {

using Clock = std::chrono::steady_clock;

// How many of a video's weak segments --segments prints; its files and the
// --json have them all.
constexpr std::size_t shown_segments = 20;

// ----------------------------------------------------------------- the paths

bool has_magic(std::string_view text) { return text.find_first_of("*?[") != std::string_view::npos; }

// The first character of UTF-8 text. A byte that starts no whole sequence is
// a character of its own: a name on Linux need not be UTF-8.
std::string_view first_character(std::string_view text) {
    const auto lead = static_cast<unsigned char>(text.front());
    const std::size_t length = lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    if (length > text.size()) {
        return text.substr(0, 1);
    }
    for (std::size_t i = 1; i < length; ++i) {
        if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
            return text.substr(0, 1);
        }
    }
    return text.substr(0, length);
}

// Whether a character is in the set of a [seq]: its characters, and ranges
// such as a-z. (UTF-8 text sorts in the order of its code points.)
bool in_character_set(std::string_view character, std::string_view set) {
    for (std::size_t i = 0; i < set.size();) {
        const std::string_view low = first_character(set.substr(i));
        i += low.size();
        if (i + 1 < set.size() && set[i] == '-') {
            const std::string_view high = first_character(set.substr(i + 1));
            i += 1 + high.size();
            if (low <= character && character <= high) {
                return true;
            }
        } else if (character == low) {
            return true;
        }
    }
    return false;
}

// Whether `name` matches a shell pattern (* ? [seq] [!seq]), as Python's
// fnmatch matches it, character by character; both as the system compares
// names (platform::name_as_compared).
bool name_matches(std::string_view name, std::string_view pattern) {
    if (pattern.empty()) {
        return name.empty();
    }
    if (pattern.front() == '*') {
        for (std::size_t skipped = 0;; skipped += first_character(name.substr(skipped)).size()) {
            if (name_matches(name.substr(skipped), pattern.substr(1))) {
                return true;
            }
            if (skipped == name.size()) {
                return false;
            }
        }
    }
    if (name.empty()) {
        return false;
    }
    const std::string_view character = first_character(name);
    if (pattern.front() == '?') {
        return name_matches(name.substr(character.size()), pattern.substr(1));
    }
    if (pattern.front() == '[') {
        // A "]" first in the set (after any "!") is one of its characters.
        std::size_t first = 1;
        if (first < pattern.size() && pattern[first] == '!') {
            ++first;
        }
        if (first < pattern.size() && pattern[first] == ']') {
            ++first;
        }
        const std::size_t close = pattern.find(']', first);
        if (close != std::string_view::npos) {
            std::string_view set = pattern.substr(1, close - 1);
            const bool negated = set.front() == '!';
            if (negated) {
                set.remove_prefix(1);
            }
            return in_character_set(character, set) != negated &&
                   name_matches(name.substr(character.size()), pattern.substr(close + 1));
        }
    }
    return name.front() == pattern.front() && name_matches(name.substr(1), pattern.substr(1));
}

// The files and folders a pattern names, as Python's glob.glob finds them:
// a name starting with "." only for a pattern part that does too.
std::vector<std::filesystem::path> glob(const std::string& pattern) {
    const std::filesystem::path whole = python_path(pattern);
    std::vector<std::filesystem::path> found{whole.root_path()};
    for (const std::filesystem::path& part : whole.relative_path()) {
        const std::string text = platform::path_text(part);
        const std::string compared_text = platform::name_as_compared(text);
        std::vector<std::filesystem::path> next;
        for (const std::filesystem::path& folder : found) {
            // A folder really named so ("[Group] Show") is not a pattern. Python's
            // glob would read it as a set of letters, and find nothing.
            std::error_code named;
            if (!has_magic(text) || std::filesystem::exists(folder / part, named)) {
                next.push_back(folder / part);
                continue;
            }
            std::error_code error;
            const std::filesystem::path listed = folder.empty() ? std::filesystem::path(".") : folder;
            for (auto entry = std::filesystem::directory_iterator(listed, error);
                 !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
                const std::string name = platform::path_text(entry->path().filename());
                if (name.starts_with(".") && !text.starts_with(".")) {
                    continue;
                }
                if (name_matches(platform::name_as_compared(name), compared_text)) {
                    next.push_back(folder / entry->path().filename());
                }
            }
        }
        found = std::move(next);
    }
    std::erase_if(found, [](const std::filesystem::path& path) {
        std::error_code error;
        return !std::filesystem::exists(path, error);
    });
    return found;
}

// A path's text as Python compares paths: on Windows ignoring case.
std::string comparable(const std::filesystem::path& path) {
    return platform::name_as_compared(platform::path_text(path));
}

// Whether `a` sorts before `b` as Python sorts paths: part by part, so that
// enc/b.mkv comes before "enc a/a.mkv".
bool sorts_before(const std::filesystem::path& a, const std::filesystem::path& b) {
    const auto parts = [](const std::filesystem::path& path) {
        const std::string text = comparable(path);
        const char separator = static_cast<char>(std::filesystem::path::preferred_separator);
        std::vector<std::string> split;
        std::size_t start = 0;
        for (std::size_t end = text.find(separator); end != std::string::npos; end = text.find(separator, start)) {
            split.push_back(text.substr(start, end - start));
            start = end + 1;
        }
        split.push_back(text.substr(start));
        return split;
    };
    return parts(a) < parts(b);
}

// A path resolved, as Python's path.resolve() resolves it: a path that
// cannot be followed (under a folder that cannot be read, on a drive Windows
// cannot resolve) stays as it is, made absolute. weakly_canonical gives an
// empty path then, which made every such video the same one.
std::filesystem::path resolved_path(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(path, error);
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(absolute, error);
    return error ? absolute : resolved;
}

}  // namespace

std::string resolved(const std::filesystem::path& path) { return comparable(resolved_path(path)); }

namespace {

// ------------------------------------------------------------------- colours

// The note for a video read without colour tags: that it was read as SDR
// (colour.h's video_colour's guess), as HDR10 without its tags is too, with
// the colour matrix the pair is read with (with_shared_matrix): its own, the
// other video's, or the reference's guess. `source` is the reference.
std::string guessed_colour_note(const VideoInfo& info, const std::string& which, const VideoInfo& source,
                                const VideoInfo& distorted) {
    const bool reference = &info == &source;
    const VideoInfo& other = reference ? distorted : source;
    const std::string guess =
        !is_untagged_tag(info.color_space)   ? "with its colour matrix, " + info.color_space
        : !is_untagged_tag(other.color_space) ? "with the other video's colour matrix, " + other.color_space
        : source.height > 650 ? std::string(reference ? "BT.709, as it is over 650 lines"
                                                      : "BT.709, as the reference is over 650 lines")
                              : std::string(reference ? "BT.601, as it is 650 lines or fewer"
                                                      : "BT.601, as the reference is 650 lines or fewer");
    return which + " has no colour tags, so it was read as SDR (" + guess +
           "): HDR10 without its tags is read as SDR too, which changes CVVDP, SSIMULACRA2 and Butteraugli. If it "
           "is HDR10, give --colour pq-bt2020.";
}

// `info` with the colours --colour gives, when it has no colour tags of its
// own: the transfer, primaries and matrix, as ffprobe names them. A matrix
// it does say is kept.
VideoInfo with_assumed_colour(VideoInfo info, const std::optional<std::string>& colour) {
    if (!colour || has_colour_tags(info)) {
        return info;
    }
    struct Colours {
        const char* transfer;
        const char* primaries;
        const char* matrix;
    };
    static const std::map<std::string, Colours, std::less<>> assumed{
        {"pq-bt2020", {"smpte2084", "bt2020", "bt2020nc"}},
        {"hlg-bt2020", {"arib-std-b67", "bt2020", "bt2020nc"}},
        {"bt709", {"bt709", "bt709", "bt709"}},
    };
    const Colours& colours = assumed.at(*colour);
    info.color_transfer = colours.transfer;
    info.color_primaries = colours.primaries;
    if (is_untagged_tag(info.color_space)) {
        info.color_space = colours.matrix;
    }
    return info;
}

// The note for a video whose result scaled one picture to the other's by a
// few lines (near_miss_crop): crops a few lines apart. None otherwise.
std::optional<std::string> near_miss_note(const TestVideo& video) {
    if (!video.result) {
        return std::nullopt;
    }
    const ComparisonResult& result = *video.result;
    const std::optional<NearMissCrop> near = near_miss_crop(result.source_info, result.distorted_info,
                                                            result.source_crop, result.distorted_crop,
                                                            result.scale_direction);
    if (!near) {
        return std::nullopt;
    }
    const auto size = [](Size of) { return std::to_string(of.first) + "x" + std::to_string(of.second); };
    return std::string(near->source_scaled ? "the reference's picture (" : "the test video's picture (") +
           size(near->scaled) + ") was scaled to the " + (near->source_scaled ? "test video's" : "reference's") +
           " (" + size(near->target) + "): the same picture cropped a few lines apart, which lowers every score. "
           "If the test video was cropped otherwise, give both crops: --source-crop and --test-crop T:B:L:R";
}

// The note for a video that took the other's colour tags (take_colour_tags).
std::string taken_colour_note(const VideoInfo& info, const std::string& which, const std::string& other) {
    return which + " has no colour tags, so it was taken to have " + other + "'s (transfer " + info.color_transfer +
           ", primaries " + info.color_primaries +
           "). CVVDP, SSIMULACRA2 and Butteraugli depend on them: --colour says what they are, if they are not "
           "those.";
}

// The video at `path` as the command reads it: a raw YUV file with --size,
// --pix-fmt and --fps (ffprobe.h's probe_raw_yuv), a raw stream timed at
// --fps or else the rate it stores (timed_raw_stream), any other video as
// ffprobe reads it, with the film grain its stream describes and whether it
// is monochrome H.264 (scan_stream_start), scored with its grain or without
// as `with_film_grain` says
// (film_grain_choice). Nothing, with `wrong` saying why, when the command
// does not give what it needs (`which` names the video). Throws ProbeError
// when it cannot be read. With RateSample::later the rate of a video that
// states none stays the estimate probe_video gives (marked as one).
std::optional<VideoInfo> read_video(const std::filesystem::path& path, const CompareArguments& arguments,
                                    std::optional<bool> with_film_grain, const std::string& ffprobe,
                                    const std::string& ffmpeg, const std::string& which, RateSample rate_sample,
                                    std::string& wrong) {
    if (is_raw_yuv(path)) {
        if (!arguments.size || !arguments.pix_fmt || !arguments.fps) {
            wrong = which + " is a raw YUV file, which records no picture size, pixel format or frame rate: give "
                            "them with --size, --pix-fmt and --fps, such as --size 1920x1080 --pix-fmt yuv420p10le "
                            "--fps 24000/1001.";
            return std::nullopt;
        }
        VideoInfo info = probe_raw_yuv(path, *arguments.size, *arguments.pix_fmt, *arguments.fps, ffprobe);
        info.film_grain = 0u;  // raw pictures carry no film grain message
        return info;
    }
    VideoInfo info = probe_video(path, ffprobe, CancelToken(), rate_sample);
    const StreamStart start = scan_stream_start(info, ffmpeg, CancelToken());
    info.film_grain = start.film_grain;
    info.monochrome_h264 = start.monochrome_h264;
    info.with_film_grain = with_film_grain;
    if (!info.is_raw_stream()) {
        return info;
    }
    const std::optional<double> fps =
        arguments.fps ? arguments.fps : frame_rate_in_stream(info, ffprobe, CancelToken());
    if (!fps) {
        wrong = raw_stream_needs_fps(which, info);
        return std::nullopt;
    }
    return timed_raw_stream(std::move(info), *fps, ffprobe);
}

// A video as read_video read it, or why the command does not say how to read it (`wrong`)
struct VideoRead {
    std::optional<VideoInfo> info;
    std::string wrong;
};

// ------------------------------------------------------------------- options

// Where the command says VMAF is calculated: --vmaf-on, else --cpu; nothing
// when it says nothing.
std::optional<std::string> vmaf_where(const CompareArguments& arguments) {
    if (arguments.vmaf_on) {
        return arguments.vmaf_on;
    }
    return arguments.cpu ? std::optional<std::string>("cpu") : std::nullopt;
}

// The range compared (an end of 0: to the end): --frames', --range's, or
// --duration's first seconds; both 0: the whole videos.
ComparedRange range_of(const CompareArguments& arguments) {
    if (arguments.frames) {
        return {arguments.frames->first, arguments.frames->second, true};
    }
    if (arguments.range) {
        return {arguments.range->first, arguments.range->second, false};
    }
    return {0, arguments.duration.value_or(0), false};
}

// The Vship build SSIMULACRA2, Butteraugli and CVVDP run on: --vship's, or
// the one the window gives new videos.
VshipBuild vship_build(const CompareArguments& arguments, const Settings& settings) {
    return arguments.vship ? vship_build_from(*arguments.vship) : default_vship_build(settings.default_vship_build);
}

// The maker --gpu-decode names: automatic for its "auto", and without it.
GpuVendor gpu_maker(const std::optional<std::string>& word) {
    return word ? gpu_vendor_named(*word).value_or(GpuVendor::automatic) : GpuVendor::automatic;
}

}  // namespace

std::optional<std::string> missing_gpu_maker(const std::optional<std::string>& gpu_decode,
                                             const std::vector<GpuVendor>& found, bool listed) {
    const GpuVendor maker = gpu_maker(gpu_decode);
    if (maker == GpuVendor::automatic || !listed || std::find(found.begin(), found.end(), maker) != found.end()) {
        return std::nullopt;
    }
    std::string have;
    for (const GpuVendor vendor : found) {
        have += (have.empty() ? "" : ", ") + std::string(maker_name(vendor));
    }
    return "--gpu-decode " + *gpu_decode + ": this PC has no " + std::string(maker_name(maker)) + " GPU" +
           (have.empty() ? "." : " (it has " + have + ").");
}

#ifdef __APPLE__
std::optional<std::string> mac_gpu_decode_refusal(const std::optional<std::string>& gpu_decode) {
    if (!gpu_decode || *gpu_decode == "auto") {
        return std::nullopt;
    }
    return "--gpu-decode " + *gpu_decode + ": a Mac decodes with its own GPU (VideoToolbox); use --gpu-decode auto.";
}
#endif

std::optional<bool> film_grain_choice(const CompareArguments& arguments, const Settings& settings) {
    if (arguments.film_grain) {
        return *arguments.film_grain == "with";
    }
    return film_grain_default_choice(settings.film_grain_default);
}

VmafOptions row_options(const CompareArguments& arguments, const Settings& settings,
                        const std::vector<std::string>& metric_keys) {
    const std::optional<std::string> vmaf_on = vmaf_where(arguments);
    VmafOptions options;
    // --gpu-decode turns GPU decoding on, whatever Settings say.
    options.gpu_decode = arguments.gpu_decode || (settings.default_gpu_decode && !arguments.no_gpu_decode);
    options.gpu_vendor = gpu_maker(arguments.gpu_decode);
    options.vvc_on_gpu = arguments.vvc_decoder ? *arguments.vvc_decoder == "gpu" : settings.vvc_decode_on_gpu;
    options.vmaf_on_gpu = vmaf_on ? *vmaf_on == "gpu" : settings.default_vmaf_on_gpu;
    options.vship_build = vship_build(arguments, settings);
    options.n_threads = arguments.threads.value_or(0);
    options.n_subsample = arguments.subsample;
    options.scale_algorithm = arguments.scaler;
    options.scale_direction = arguments.scale_to == "reference" ? ScaleDirection::distorted_to_source
                                                                : ScaleDirection::source_to_distorted;
    const ComparedRange range = range_of(arguments);
    options.range_start = range.start;
    options.range_end = range.end;
    options.range_in_frames = range.in_frames;
    options.crop_mode = arguments.black_bars == "auto"     ? CropMode::automatic
                        : arguments.black_bars == "source" ? CropMode::source_only
                                                           : CropMode::none;
    if (arguments.source_crop || arguments.test_crop) {
        options.crop_mode = CropMode::manual;
        options.manual_source_crop = arguments.source_crop.value_or(CropEdges{});
        options.manual_test_crop = arguments.test_crop.value_or(CropEdges{});
    }
    options.compute_vmaf = false;
    if (arguments.model == "auto") {
        options.model_choice = std::string(auto_model_choice);
    } else if (arguments.model == "standard") {
        options.model_choice = std::string(default_model);
    } else if (arguments.model == "4k") {
        options.model_choice = std::string(uhd_model);
    } else {
        options.model_choice = std::string(custom_model_choice);
        options.custom_model_path = platform::path_text(resolved_path(python_path(arguments.model)));
    }
    for (const ModelChoice& model : v1_model_choices()) {
        if (model.word == arguments.model_v1) {
            options.model_choice_v1 = model.choice;
        }
    }
    for (const ModelChoice& model : hdr_model_choices()) {
        if (model.word == arguments.model_hdr) {
            options.model_choice_hdr = model.choice;
        }
    }
    for (const std::string& key : metric_keys) {
        if (metric_definition(key).ffmpeg_binding) {
            options.set_metric_enabled(key, true);
        }
    }
    return options;
}

namespace {

// Where SSIMULACRA2 and Butteraugli run: as the window runs them, on the GPU
// where one can and on the CPU without one (so that a score of theirs saved
// from the CPU is calculated again where the GPU can); on the CPU with --cpu.
PerceptualBackends perceptual_backends(const CompareArguments& arguments) {
    if (!arguments.cpu) {
        return {automatic_perceptual_backends.begin(), automatic_perceptual_backends.end()};
    }
    return {{"ssimulacra2", "cpu"}, {"butteraugli", "cpu"}};
}

// Starts the GPU checks a run of `metrics` needs, together and in the
// background, rather than one after another as the videos are set up: each
// starts a process of its own (Vship's takes 1.8 s, VMAF's and VMAF v1's
// 0.5 s each), and where the videos may be decoded on the GPU, which
// decoders `ffmpeg` and the GPUs have. Their answers are kept, and a caller
// that asks while one runs waits for it. The checks still running when the
// futures go are waited for.
std::vector<std::future<void>> start_gpu_checks(const CompareArguments& arguments, const Settings& settings,
                                                const std::vector<std::string>& metrics, const std::string& ffmpeg) {
    const auto asked = [&metrics](std::string_view key) {
        return std::find(metrics.begin(), metrics.end(), key) != metrics.end();
    };
    const std::optional<std::string> vmaf_on = vmaf_where(arguments);
    const bool vmaf_on_gpu = vmaf_on ? *vmaf_on == "gpu" : settings.default_vmaf_on_gpu;
    std::vector<std::future<void>> checks;
    if (asked("cvvdp") || (!arguments.cpu && (asked("ssimulacra2") || asked("butteraugli")))) {
        checks.push_back(std::async(std::launch::async,
                                    [build = vship_build(arguments, settings)] { detect_vship_device(build, gpu_probes_token()); }));
    }
    if (vmaf_on_gpu && (asked("vmaf") || asked("vmaf_neg"))) {
        checks.push_back(std::async(std::launch::async, [] { gpu_vmaf_backend(gpu_probes_token()); }));
    }
    if (vmaf_on_gpu && asked("vmaf_v1")) {
        checks.push_back(std::async(std::launch::async, [] { v1_available(gpu_probes_token()); }));
    }
    if (row_options(arguments, settings, metrics).gpu_decode) {
        // The GPUs' decoders and FFmpeg's (0.05 to 0.25 s), asked while the
        // videos are read, not as the first video is planned
        checks.push_back(std::async(std::launch::async, [ffmpeg] { DecodingCapabilities::remembered(ffmpeg); }));
    }
    return checks;
}

// A preset's display, the compared frames scaled up to fill it with `fill`
// (--cvvdp-fill), else not, as for a video added to the window.
CvvdpDisplayChoice on_display(const CvvdpPreset& preset, bool fill) {
    CvvdpSettings settings{preset.settings.display};
    settings.resize_to_display = fill;
    return {preset.name, settings};
}

}  // namespace

std::vector<CvvdpDisplayChoice> cvvdp_displays(const CompareArguments& arguments, const Settings& settings) {
    if (arguments.cvvdp_displays.empty()) {
        if (const std::optional<CvvdpPreset> preset =
                cvvdp_preset_named(settings.cvvdp_default_preset, settings.cvvdp_presets_json)) {
            return {on_display(*preset, arguments.cvvdp_fill)};
        }
        return {};
    }
    std::vector<CvvdpDisplayChoice> displays;
    std::set<std::string> named;
    for (const std::string& name : arguments.cvvdp_displays) {
        if (!named.insert(name).second) {
            continue;
        }
        const std::optional<CvvdpPreset> preset = cvvdp_preset_named(name, settings.cvvdp_presets_json);
        if (!preset) {
            std::string names;
            for (const CvvdpPreset& known : cvvdp_presets(settings.cvvdp_presets_json)) {
                names += (names.empty() ? "" : ", ") + python_string_repr(known.name);
            }
            throw std::invalid_argument("No CVVDP display preset is called " + python_string_repr(name) +
                                        ". The presets: " + names + ".");
        }
        displays.push_back(on_display(*preset, arguments.cvvdp_fill));
    }
    return displays;
}

CvvdpDisplayChoice display_for(const std::vector<CvvdpDisplayChoice>& displays, const VideoInfo& reference,
                               bool fill) {
    if (!displays.empty()) {
        return displays.front();
    }
    return on_display(automatic_cvvdp_preset(&reference), fill);
}

namespace {

// The note for HDR video whose CVVDP modelled the SDR display chosen as the
// app's default in Settings, or nothing.
std::optional<std::string> sdr_display_note(const CompareArguments& arguments, const VideoInfo& source,
                                            const TestVideo& video, const std::vector<CvvdpDisplayChoice>& displays) {
    const std::optional<VideoColour> colour = colour_of(source);
    if (std::find(video.metrics.begin(), video.metrics.end(), "cvvdp") == video.metrics.end() ||
        !arguments.cvvdp_displays.empty() || displays.empty() || displays.front().second.display.hdr || !colour ||
        !colour->hdr()) {
        return std::nullopt;
    }
    return "the video is HDR, and CVVDP modelled the SDR display chosen as the app's default, " +
           python_string_repr(displays.front().first) + ". --cvvdp-display " +
           python_string_repr(std::string(hdr_cvvdp_preset)) +
           " (or another HDR preset, or Automatic in the app's Settings) models an HDR one.";
}

// What a video added to the window starts with ticked, less the metrics its
// Metrics... picker hides (they are not calculated there), with the planes
// of those that have them where the window shows them (show_yuv_planes).
std::vector<std::string> default_metrics(const Settings& settings) {
    VmafOptions options;
    settings.take_default_metrics(options);
    const std::vector<std::string> requested = options.requested_metrics();
    std::set<std::string, std::less<>> chosen(requested.begin(), requested.end());
    const std::pair<const char*, bool> perceptual[] = {{"ssimulacra2", settings.default_compute_ssimulacra2},
                                                       {"butteraugli", settings.default_compute_butteraugli},
                                                       {"cvvdp", settings.default_compute_cvvdp}};
    for (const auto& [key, ticked] : perceptual) {
        if (ticked) {
            chosen.insert(key);
        }
    }
    std::set<std::string, std::less<>> hidden;
    for (const std::string& key : settings.hidden_metrics) {
        if (find_metric(key) != nullptr) {
            hidden.insert(key);
        }
    }
    if (hidden.size() < metrics().size()) {  // a settings file that hides everything hides nothing
        for (const std::string& key : hidden) {
            chosen.erase(key);
        }
    }
    for (const MetricDefinition& metric : metrics()) {
        if (settings.show_yuv_planes && metric.is_plane() && chosen.contains(metric.plane_of)) {
            chosen.emplace(metric.key);
        }
    }
    std::vector<std::string> keys;
    for (const MetricDefinition& metric : metrics()) {
        if (chosen.contains(metric.key)) {
            keys.emplace_back(metric.key);
        }
    }
    return keys;
}

// What the command asks for that this C++ version cannot do yet, or
// nothing.
std::optional<std::string> not_yet(const CompareArguments& arguments, const Settings& settings,
                                   const std::vector<std::string>& asked) {
    const bool vmaf = std::any_of(asked.begin(), asked.end(), [](const std::string& key) {
        return key == "vmaf" || key == "vmaf_neg" || key == "vmaf_v1";
    });
    const std::optional<std::string> vmaf_on = vmaf_where(arguments);
    const bool cpu_metrics = std::any_of(asked.begin(), asked.end(), [](const std::string& key) {
        return is_cpu_metric(key);
    });
    // The app scores PSNR, SSIM and XPSNR (and their 6:1:1 forms) itself
    // unless VMAF is calculated on the CPU beside them, in FFmpeg; with
    // FFmpeg's frames, the C++ app pairs them itself only.
    if (cpu_metrics && (!vmaf || (vmaf_on ? *vmaf_on == "gpu" : settings.default_vmaf_on_gpu)) &&
        ascii_lower(platform::environment_variable("VML_VMAF_PIPES").value_or("")) == "canvas") {
        return "PSNR, SSIM or XPSNR with FFmpeg pairing the frames (VML_VMAF_PIPES=canvas)";
    }
    return std::nullopt;
}

// What this C++ version cannot do yet of a video's VMAF on the GPU (a task
// of its own, plan_tasks), or nothing.
std::optional<std::string> gpu_not_yet(const std::vector<MetricTask>& tasks) {
    // VMAF's task alone: Vship's, in the GPU's queue too, pairs its frames
    // itself whatever VML_VMAF_PIPES says.
    if (std::none_of(tasks.begin(), tasks.end(), [](const MetricTask& task) { return task.backend == gpu_vmaf_task; })) {
        return std::nullopt;
    }
    if (ascii_lower(platform::environment_variable("VML_VMAF_PIPES").value_or("")) == "canvas") {
        return "VMAF on the GPU with FFmpeg pairing the frames (VML_VMAF_PIPES=canvas)";
    }
    return std::nullopt;
}

// ------------------------------------------------------------------- the run

// --timeout and --video-timeout, as the run goes: a thread that checks the
// deadlines (watch.h) ten times a second, and cancels the run, or the video
// whose run is due, when one is. With --video-timeout each job runs on a
// cancel token of its own (token), cancelled with its video's deadline or
// with the run. Nothing runs without either option.
class RunWatch {
    using Clock = std::chrono::steady_clock;

public:
    RunWatch(const CompareArguments& arguments, CancelToken cancel, std::shared_ptr<Progress> progress,
             std::shared_ptr<std::atomic<RunStage>> stage)
        : timeout_(arguments.timeout),
          video_timeout_(arguments.video_timeout),
          cancel_(std::move(cancel)),
          progress_(std::move(progress)),
          stage_(std::move(stage)),
          began_(Clock::now()),
          deadlines_(0.0, timeout_ ? std::optional<double>(static_cast<double>(*timeout_)) : std::nullopt,
                     video_timeout_ ? std::optional<double>(static_cast<double>(*video_timeout_)) : std::nullopt) {
        if (timeout_ || video_timeout_) {
            thread_ = std::thread([this] { watch(); });
        }
    }
    ~RunWatch() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        woken_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    RunWatch(const RunWatch&) = delete;
    RunWatch& operator=(const RunWatch&) = delete;

    // Before the run: a token for each of its `count` jobs, with --video-timeout.
    void jobs(std::size_t count) {
        const std::lock_guard lock(mutex_);
        if (video_timeout_) {
            tokens_ = separate_tokens(count);
            timed_out_jobs_.assign(count, false);
        }
    }

    // What job `index` runs on: its own token, else the run's.
    const CancelToken& token(std::size_t index, const CancelToken& run) const {
        const std::lock_guard lock(mutex_);
        return index < tokens_.size() ? tokens_[index] : run;
    }

    void started(std::size_t index, const std::string& label) {
        const std::lock_guard lock(mutex_);
        labels_[index] = label;
        deadlines_.started(index, seconds());
    }
    void ended(std::size_t index) {
        const std::lock_guard lock(mutex_);
        deadlines_.ended(index);
    }

    // The run is over: the stage is outside it from now, set under the lock
    // the watch decides under, so that a --timeout due as the run ends has
    // either cancelled it already or ends the command.
    void run_ended() {
        const std::lock_guard lock(mutex_);
        *stage_ = RunStage::outside;
    }

    // Whether --timeout cancelled the run; whether --video-timeout stopped job `index`.
    bool timed_out() const { return timed_out_; }
    bool timed_out(std::size_t index) const {
        const std::lock_guard lock(mutex_);
        return index < timed_out_jobs_.size() && timed_out_jobs_[index];
    }
    bool any_job_timed_out() const {
        const std::lock_guard lock(mutex_);
        return std::find(timed_out_jobs_.begin(), timed_out_jobs_.end(), true) != timed_out_jobs_.end();
    }
    std::optional<long long> video_timeout() const { return video_timeout_; }

private:
    double seconds() const { return std::chrono::duration<double>(Clock::now() - began_).count(); }

    void watch() {
        std::unique_lock lock(mutex_);
        bool run_cancel_passed_on = false;
        while (!woken_.wait_for(lock, std::chrono::milliseconds(100), [this] { return stopping_; })) {
            const Deadlines::Due due = deadlines_.due(seconds());
            const TimeoutAction action =
                due.run ? on_run_timeout(*stage_, cancel_.cancelled()) : TimeoutAction::nothing;
            if (action == TimeoutAction::end_command) {
                // Before the run, or while the results are written, as Ctrl+C then: the command ends at once.
                progress_->problem("Timed out after --timeout's " + std::to_string(*timeout_) + " s");
                platform::end_process_now(exit_timeout);
            }
            if (action == TimeoutAction::cancel_run) {
                timed_out_ = true;
                progress_->message("Timed out after --timeout's " + std::to_string(*timeout_) + " s: cancelling...");
                app_log::write(app_log::Level::info, "vml.job_runner", "Run: cancel requested (--timeout)");
                cancel_.cancel();
                stop_gpu_probes();
            }
            for (const std::size_t index : due.jobs) {
                timed_out_jobs_[index] = true;
                progress_->finished(index,
                                    labels_[index] + ": timed out after --video-timeout's " +
                                        std::to_string(*video_timeout_) + " s, stopped",
                                    "timed_out");
                tokens_[index].cancel();
            }
            if (!run_cancel_passed_on && cancel_.cancelled()) {
                run_cancel_passed_on = true;  // Ctrl+C or --timeout: every job's token too
                for (CancelToken& token : tokens_) {
                    token.cancel();
                }
            }
        }
    }

    const std::optional<long long> timeout_;
    const std::optional<long long> video_timeout_;
    CancelToken cancel_;
    const std::shared_ptr<Progress> progress_;
    const std::shared_ptr<std::atomic<RunStage>> stage_;
    const Clock::time_point began_;
    mutable std::mutex mutex_;
    std::condition_variable woken_;
    bool stopping_ = false;
    Deadlines deadlines_;
    std::vector<CancelToken> tokens_;
    std::vector<bool> timed_out_jobs_;
    std::map<std::size_t, std::string> labels_;
    std::atomic<bool> timed_out_ = false;
    std::thread thread_;  // last: it starts once the rest is made
};

// Each video that --video-timeout stopped, as the results say it: what it
// had not finished failed with the code timed_out, where the run would say
// only that it was cancelled.
void say_timed_out_videos(const std::vector<TestVideo*>& targets, const RunWatch& watch) {
    if (!watch.video_timeout()) {
        return;
    }
    const std::string why =
        "its run took more than --video-timeout's " + std::to_string(*watch.video_timeout()) + " s, and was stopped";
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (!watch.timed_out(index)) {
            continue;
        }
        TestVideo& video = *targets[index];
        if (!video.result) {
            video.error = why;
            video.error_code = codes::timed_out;
        }
        std::vector<std::pair<std::string, std::string>> failures;
        for (const std::string& key : video.metrics) {
            if (!video.result || !video.result->has_metric(key)) {
                failures.emplace_back(key, why);
                video.failure_codes[key] = codes::timed_out;
            }
        }
        video.failures = std::move(failures);
        if (video.facts) {
            for (auto& [key, outcome] : video.facts->metrics) {
                if (outcome.status == metric_failed || outcome.status == metric_cancelled) {
                    outcome = MetricOutcome{metric_failed, why, codes::timed_out};
                }
            }
        }
    }
}

// Runs the jobs (job_runner.h), keeping each video's result, or why it or
// some of its metrics failed, in its TestVideo (`targets` holds each job's),
// saving it in the saved scores under `cache_folder` (none: nothing is saved,
// --no-save) unless the video's save_scores is false, and showing the run's
// progress (Progress). A video
// whose run scored fewer frame pairs than its length holds has both videos'
// frames counted (frame_pairing.h): where they state them, and with
// `read_whole_files` by reading those that do not. A port of the Python
// app's command line's cli.run.
void run(const RunEnvironment& environment, const std::vector<TestVideo*>& targets, const std::vector<VmafJob>& jobs,
         int parallel, bool gpu_metrics_together, bool read_whole_files,
         const std::optional<std::filesystem::path>& cache_folder, Progress& progress, const CancelToken& cancel,
         RunWatch& watch) {
    // The line being redrawn is drawn with new figures once a second while
    // the run lasts, and ends with the run, however the run ends.
    progress.start();
    struct EndsTheLine {
        Progress& progress;
        ~EndsTheLine() { progress.done(); }
    } const ends_the_line{progress};
    const auto video = [&targets](std::size_t index) -> TestVideo& { return *targets[index]; };
    // A video's CVVDP on other displays is saved in the same comparison
    // folder, and two jobs replacing its context.json at once can fail on
    // Windows, losing a score: one save at a time.
    std::mutex saving;
    const auto keep = [&](std::size_t index, const ComparisonResult& result) {
        TestVideo& target = video(index);
        target.result = result;
        if (!target.save_scores || !cache_folder) {
            return;
        }
        try {
            const std::lock_guard lock(saving);
            store_result(*cache_folder, result.source, target.path, target.request.recipe, result,
                         platform::path_text(target.path.stem()), target.request.metrics);
        } catch (const std::exception& error) {
            progress.message(target.label() + ": the scores could not be saved (" + error.what() + ")");
        }
    };
    // A metric that fell back from the GPU, to the CPU or to stock Vship
    // 5.1.2, as its task's status says: a note, so that the results (and the
    // JSON) say where it was calculated and why, not the CPU column alone. A
    // video's tasks report on their lanes' threads.
    std::mutex noting;
    const auto note_fallback = [&](std::size_t index, const RunStatus& step) {
        if (!is_gpu_fallback(step.kind)) {
            return;
        }
        // The status as it is, less its "...": it names the metrics or the
        // pass, and why, where it knows.
        std::string note = fallback_note(step.text);
        if (note.find(" (") == std::string::npos) {
            note += " (the log, -v, says why)";
        }
        const std::lock_guard lock(noting);
        if (video(index).note(codes::gpu_fallback, note)) {
            if (progress.jsonl()) {
                progress.event({{"event", "note"},
                                {"code", codes::gpu_fallback},
                                {"text", note},
                                {"videos", {platform::path_text(video(index).path)}}});
            } else {
                progress.message(video(index).label() + ": " + note);
            }
        }
    };
    RunEvents events;
    watch.jobs(jobs.size());
    events.job_started = [&](std::size_t index) {
        watch.started(index, video(index).label());
        if (progress.jsonl()) {
            nlohmann::ordered_json who{{"path", platform::path_text(video(index).path)}};
            if (video(index).for_display_only) {
                who["cvvdp_display"] = video(index).display;
            }
            progress.name_job(index, who);
            nlohmann::ordered_json started{{"event", "video_started"}, {"job", index}};
            started.update(who);  // its path, and its display (a further one)
            progress.event(std::move(started));
            return;
        }
        progress.message(video(index).label() + ": started");
    };
    events.task_progress = [&](std::size_t index, const std::vector<TaskSnapshot>& tasks) {
        for (const TaskSnapshot& task : tasks) {
            note_fallback(index, task.step);
        }
        if (progress.jsonl()) {
            nlohmann::ordered_json listed = nlohmann::ordered_json::array();
            for (const TaskSnapshot& task : tasks) {
                listed.push_back({{"metrics", task.metric_keys},
                                  {"state", task.state},
                                  {"done", task.current},
                                  {"total", task.total},
                                  {"fps", task.fps}});
            }
            progress.job_event(index, {{"event", "progress"}, {"tasks", std::move(listed)}});
            return;
        }
        std::string line;
        for (const std::string& part : metric_line(tasks, false).parts) {
            line += (line.empty() ? "" : "; ") + part;
        }
        if (!line.empty()) {
            progress.job(index, video(index).label() + ": " + line, line_shape(tasks));
        }
    };
    // A video's result so far is kept, and saved, when one of its tasks
    // finishes while another still runs: kept if the rest is lost.
    events.result_updated = keep;
    events.job_finished = [&](std::size_t index, const ComparisonResult& result) {
        keep(index, result);
        // A cancelled run also sends what a video had finished, which the
        // tables and the JSON call cancelled. Judged by this job's own
        // result: the jobs of its other displays run on their own threads.
        const std::vector<std::string>& asked = video(index).metrics;
        const bool whole = std::all_of(asked.begin(), asked.end(),
                                       [&result](const std::string& key) { return result.has_metric(key); });
        watch.ended(index);
        if (watch.timed_out(index)) {
            return;  // said as it was stopped
        }
        const bool done = whole || !cancel.cancelled();
        progress.finished(index, video(index).label() + (done ? ": done" : ": cancelled"), done ? "done" : "cancelled");
    };
    events.job_facts = [&](std::size_t index, const RunFacts& facts) {
        video(index).facts = facts;
        append(video(index).facts->metrics, video(index).not_applicable);
    };
    // The counts files state, always (they cost nothing); whole files read only when asked
    events.count_frames =
        counting_each_file_once(container_frame_counter(environment.ffprobe, cancel, read_whole_files));
    events.job_failed = [&](std::size_t index, const std::string& message, const std::string& /*stderr_tail*/) {
        video(index).error = message;
        video(index).error_code = codes::run_failed;
        watch.ended(index);
        if (!watch.timed_out(index)) {
            progress.finished(index, video(index).label() + ": FAILED: " + message, "failed");
        }
    };
    events.job_partially_failed = [&](std::size_t index, const ComparisonResult& result, const std::string& message,
                                      const std::string& /*stderr_tail*/, const MetricFailures& failures) {
        keep(index, result);
        TestVideo& target = video(index);
        append(target.failures, failures.empty() ? MetricFailures{{"", message}} : failures);
        watch.ended(index);
        if (!watch.timed_out(index)) {
            progress.finished(index, target.label() + ": some metrics FAILED: " + message, "partial");
        }
    };
    run_jobs(
        jobs, parallel, events,
        // Each job on its own token with --video-timeout (RunWatch): the
        // runner is given the job of `jobs` itself, so its place is its index.
        [&environment, &jobs, &watch](const VmafJob& job, const VmafOptions& options,
                                      const ProgressCallback& on_progress, const StatusCallback& on_status,
                                      const CancelToken& token) {
            const CancelToken& own = watch.token(static_cast<std::size_t>(&job - jobs.data()), token);
            return run_vmaf(environment, job.source_info, job.distorted_info, options, on_progress, on_status, own,
                            job.result_distorted_path, job.gpu_answers.engine);
        },
        [&environment, &jobs, &watch](const VmafJob& job, const std::vector<std::string>& keys, bool together,
                                      int concurrent_cpu_tasks, const PerceptualHooks& hooks,
                                      const CancelToken& token) {
            const PerceptualRequest request{environment,        job.source_info,         job.distorted_info,
                                            job.options,        job.perceptual_backends, job.cvvdp,
                                            concurrent_cpu_tasks};
            return run_perceptual_task(request, keys, together, hooks,
                                       watch.token(static_cast<std::size_t>(&job - jobs.data()), token));
        },
        gpu_metrics_together, environment.cores, cancel);
}


// Adds the job that calculates what `video` asks for, less what is saved
// under `cache_folder` (when `use_saved`; a saved comparison's files are
// recorded there when `save`); with everything saved, takes the
// saved result and adds none (the Python app's cli._queue). `options` are
// the run's, with VMAF's model resolved; `share_decode` is plan_tasks's.
// Returns what this version cannot do yet of the job's VMAF on the GPU
// (gpu_not_yet), or nothing.
std::optional<std::string> queue(TestVideo& video, const VideoInfo& source, VmafOptions options,
                                 const PerceptualBackends& backends, const CvvdpSettings& cvvdp, bool use_saved,
                                 bool save, const std::filesystem::path& cache_folder, std::vector<VmafJob>& jobs,
                                 bool share_decode = false) {
    // The request the window makes for a row: what the saved scores are
    // looked up and saved under, so that the window and vml-cli find each other's.
    const std::map<std::string, std::string> chosen(backends.begin(), backends.end());
    const auto asks = [&video](std::string_view key) {
        return std::find(video.metrics.begin(), video.metrics.end(), key) != video.metrics.end();
    };
    const auto on_gpu = [&asks, &chosen](const char* key) {
        const auto backend = chosen.find(key);
        return asks(key) && (backend == chosen.end() || backend->second == "gpu");
    };
    if (video.options.vship_build == VshipBuild::fast && vship_build_bundled(VshipBuild::stock) &&
        (asks("cvvdp") || on_gpu("ssimulacra2") || on_gpu("butteraugli"))) {
        // Vship-fast's probe, started with the run (start_gpu_checks),
        // answered first: where only 5.1.2 found a GPU, the request asks for
        // 5.1.2's scores (vship_build_in_use), which is what is calculated.
        detect_vship_device(VshipBuild::fast, gpu_probes_token());
    }
    video.request = analysis_request(video.options, video.metrics, chosen, cvvdp, source, *video.info);
    std::optional<ComparisonResult> saved;
    if (use_saved) {
        if (auto found = load_result(cache_folder, source.path, video.path, video.request.recipe,
                                     video.request.metrics, video.request.perceptual_backends)) {
            saved = std::move(found->first);
            // The bitrates as read now, as the window takes them
            // (on_saved_found): an older version may have taken the whole
            // file's.
            saved->source_info.take_bit_rate(source);
            saved->distorted_info.take_bit_rate(*video.info);
            // Saved by an older version: its files recorded now, so that it
            // is found when they move (not with --no-save).
            if (save) {
                record_saved_files(cache_folder, source.path, video.path, video.request.recipe);
            }
        }
    }
    // The saved scores the run keeps, by the window's rule.
    MetricResultSet kept;
    if (saved) {
        // A metric set to the GPU keeps a CPU score only without a GPU the
        // video's Vship build can use; the probe runs only if a score needs
        // that answer.
        const VshipBuild build = video.options.vship_build;
        kept = reusable(*saved, video.request.metrics, chosen,
                        [build](const std::string&) { return detect_vship_device(build, gpu_probes_token()).first.has_value(); },
                        colour_tags(source, *video.info));
    }
    if (std::all_of(video.metrics.begin(), video.metrics.end(), [&kept](const std::string& key) { return kept.has(key); })) {
        // Everything asked for is saved: nothing to run, as the window runs
        // nothing for a row that has its scores.
        video.result = std::move(saved);
        video.from_cache = true;
        RunFacts facts;
        facts.settings = VideoSettings{source, *video.info, options, video.metrics, backends, cvvdp};
        for (const std::string& key : video.metrics) {
            facts.metrics.emplace_back(key, MetricOutcome{metric_saved, ""});
        }
        append(facts.metrics, video.not_applicable);
        video.facts = std::move(facts);
        return std::nullopt;
    }
    const GpuAnswers gpu_answers = plan_gpu_answers(options, video.metrics, backends);
    std::vector<MetricTask> tasks =
        plan_tasks(source, *video.info, options, video.metrics, backends, kept, share_decode, gpu_answers);
    std::optional<std::string> missing = gpu_not_yet(tasks);
    if (kept.empty()) {
        saved.reset();
    }
    jobs.push_back(VmafJob{source, *video.info, std::move(options), video.path, video.metrics, std::move(tasks), backends,
                           cvvdp, std::move(saved), std::move(kept), platform::path_text(video.path.stem()),
                           share_decode, gpu_answers});
    return missing;
}

std::string ffmpeg_problem(const ToolsStatus& status) {
    std::string text = "FFmpeg cannot be used:";
    for (const std::string& problem : status.problems()) {
        text += "\n  " + problem;
    }
#ifdef _WIN32
    // The window downloads it into a folder this program uses too (the
    // settings' FFmpeg folder).
    return text + "\nOpen the VideoMetricsLab window once and click \"Download FFmpeg\", put ffmpeg and ffprobe on "
                  "PATH, or set FFmpeg's folder in the app's Settings tab.";
#else
    return text + "\nPut ffmpeg and ffprobe on PATH, or set FFmpeg's folder in the app's Settings tab.";
#endif
}

}  // namespace

std::filesystem::path python_path(const std::string& text) {
#ifdef _WIN32
    constexpr char separator = '\\';
    std::string normal = text;
    std::replace(normal.begin(), normal.end(), '/', '\\');
#else
    constexpr char separator = '/';
    const std::string& normal = text;
#endif
    // The root (a drive, \\server\share, or a separator) is kept as it is;
    // after it, empty and "." parts are dropped.
    std::string root;
    std::size_t at = 0;
#ifdef _WIN32
    if (normal.size() >= 2 && normal[1] == ':') {
        root = normal.substr(0, 2);
        at = 2;
    } else if (normal.starts_with("\\\\") && normal.size() > 2 && normal[2] != '\\') {
        const std::size_t server_end = normal.find('\\', 2);
        const std::size_t share_end =
            server_end == std::string::npos ? std::string::npos : normal.find('\\', server_end + 1);
        root = normal.substr(0, share_end);
        at = share_end == std::string::npos ? normal.size() : share_end;
    }
#else
    if (normal.starts_with("//") && !normal.starts_with("///")) {
        root = "/";  // Python keeps exactly two leading slashes
    }
#endif
    if (at < normal.size() && normal[at] == separator) {
        root += separator;
        while (at < normal.size() && normal[at] == separator) {
            ++at;
        }
    }
    std::string rest;
    while (at < normal.size()) {
        std::size_t end = normal.find(separator, at);
        if (end == std::string::npos) {
            end = normal.size();
        }
        const std::string part = normal.substr(at, end - at);
        if (!part.empty() && part != ".") {
            rest += (rest.empty() ? "" : std::string(1, separator)) + part;
        }
        at = end + 1;
    }
    const std::string whole = root + rest;
    return platform::path_from_text(whole.empty() ? "." : whole);
}

std::filesystem::path absolute_path(const std::filesystem::path& path) {
    if (path.is_absolute()) {
        return path;
    }
    // Under the working folder, as Python's path.absolute() puts it: ".."
    // kept, which std::filesystem::absolute (GetFullPathNameW) drops, though
    // a link before it can make it another folder. A path with a drive or a
    // root but not both ("C:name", "\name") takes the working folder's.
    std::error_code error;
    std::filesystem::path absolute = path.has_root_name() || path.has_root_directory()
                                         ? std::filesystem::absolute(path, error)
                                         : std::filesystem::current_path(error) / path;
    return error ? path : absolute;
}

std::vector<std::filesystem::path> expand_tests(const std::vector<std::string>& patterns,
                                                const std::filesystem::path& reference) {
    return expand_videos(patterns, reference);
}

namespace {

// Why `gpu` names no one GPU of `adapters` (more than one, or none), or
// nothing where it names one: the reason both --gpu's refusal and a chosen
// GPU's problem give, after what chose it.
std::optional<std::string> gpu_name_problem(const std::string& gpu, const std::vector<platform::GpuAdapter>& adapters) {
    std::vector<std::string> names;
    for (const platform::GpuAdapter& adapter : adapters) {
        names.push_back(adapter.name);
    }
    const std::vector<std::size_t> found = gpus_named(names, gpu);
    if (found.size() == 1) {
        return std::nullopt;
    }
    std::string listed;
    for (const std::size_t index : found) {
        listed += (listed.empty() ? "" : ", ") + names[index];
    }
    if (!found.empty()) {
        return "more than one GPU here has that in its name: " + listed + ". Give more of the name.";
    }
    for (const std::string& name : names) {
        listed += (listed.empty() ? "" : ", ") + name;
    }
    return "no GPU here has that name. The GPUs here: " + listed + "; or auto.";
}

}  // namespace

std::optional<std::string> gpu_refusal(const std::optional<std::string>& gpu,
                                       const std::vector<platform::GpuAdapter>& adapters) {
    if (!gpu || *gpu == "auto" || adapters.empty()) {
        return std::nullopt;
    }
    if (const std::optional<std::string> why = gpu_name_problem(*gpu, adapters)) {
        return "--gpu " + *gpu + ": " + *why;
    }
    return std::nullopt;
}

std::optional<std::string> chosen_gpu_problem(const std::optional<std::string>& gpu, bool from_settings,
                                              const std::vector<platform::GpuAdapter>& adapters) {
    if (!gpu || *gpu == "auto" || adapters.empty()) {
        return std::nullopt;
    }
    if (const std::optional<std::string> why = gpu_name_problem(*gpu, adapters)) {
        return std::string(from_settings ? "The GPU the settings choose, " : "The GPU VML_GPU chooses, ") + *gpu +
               ": " + *why + " No GPU is used for the GPU metrics: choose one with --gpu.";
    }
    return std::nullopt;
}

std::optional<std::string> difference_refusal(std::size_t test_count) {
    if (test_count == 2) {
        return std::nullopt;
    }
    return "argument --difference: needs exactly two test videos, got " + std::to_string(test_count);
}

std::vector<std::filesystem::path> expand_videos(const std::vector<std::string>& patterns,
                                                 const std::optional<std::filesystem::path>& excluded) {
    std::vector<std::filesystem::path> found;
    std::set<std::string> found_files;  // each path found, resolved()
    const auto add = [&found, &found_files](const std::filesystem::path& path, const std::string& file) {
        if (found_files.insert(file).second) {
            found.push_back(path);
        }
    };
    const std::optional<std::string> excluded_file = excluded ? std::optional(resolved(*excluded)) : std::nullopt;
    for (const std::string& text : patterns) {
        const std::filesystem::path path = python_path(text);
        std::error_code error;
        if (std::filesystem::exists(path, error) || !has_magic(text)) {
            add(path, resolved(path));
            continue;
        }
        std::vector<std::filesystem::path> matches;
        for (const std::filesystem::path& match : glob(text)) {
            if (std::filesystem::is_regular_file(match, error)) {
                matches.push_back(python_path(platform::path_text(match)));
            }
        }
        std::sort(matches.begin(), matches.end(), sorts_before);
        if (matches.empty()) {
            found.push_back(path);  // named in the results as not read (found_nothing)
        }
        for (const std::filesystem::path& match : matches) {
            if (const std::string file = resolved(match); file != excluded_file) {
                add(match, file);
            }
        }
    }
    // Made absolute: the paths are recorded (the saved scores' context.json,
    // the saved bitrate, the JSON) and read again from other folders.
    for (std::filesystem::path& path : found) {
        path = absolute_path(path);
    }
    return found;
}

std::optional<std::string> found_nothing(const std::filesystem::path& path) {
    std::error_code error;
    if (has_magic(platform::path_text(path)) && !std::filesystem::exists(path, error)) {
        return "no file matches it";
    }
    return std::nullopt;
}

namespace {

// Ctrl+C: outside the run, the command ends at once with exit code 130 and
// nothing more, as the Python app's KeyboardInterrupt ends it. An FFmpeg
// check or ffprobe under way ends with it, as this process's job ends.
// During the run (RunStage::running), the run is cancelled and what finished is
// reported, and the GPU checks started with the command are ended (a task
// of the run may be waiting for one: nothing after the run asks for one);
// a second Ctrl+C (or SIGTERM) ends the command at once, for a
// run whose cancel does not come back (a GPU driver that hangs). Closing the
// console window (or End task on it) is a Ctrl+C. Set up
// first: the checks and probes before the run can take a while (60 s for
// ffprobe on a slow share). The handler outlives the command, so it shares
// what it uses.
// Returns the same, for other ends of the run (--cancel-on-stdin-eof),
// taking what to say ("Cancelling..." for Ctrl+C).
std::function<void(const std::string&)> cancel_on_interrupt(CancelToken cancel,
                                                            const std::shared_ptr<Progress>& progress,
                                                            const std::shared_ptr<std::atomic<RunStage>>& stage) {
    auto asked = std::make_shared<std::atomic<bool>>(false);
    const std::function<void(const std::string&)> interrupt = [cancel, progress, stage,
                                                               asked](const std::string& saying) mutable {
        if (*stage != RunStage::running || asked->exchange(true)) {
            platform::end_process_now(exit_cancelled);
        }
        progress->message(saying);
        app_log::write(app_log::Level::info, "vml.job_runner", "Run: cancel requested");
        cancel.cancel();
        stop_gpu_probes();
    };
    platform::on_interrupt([interrupt] { interrupt("Cancelling..."); });
    return interrupt;
}

// --cancel-on-stdin-eof: standard input read to its end on a thread of its
// own, which then cancels the run as Ctrl+C does (`interrupt`): unless the
// results are given already (`stage`), when a program that has read them
// closes it and the command ends as the run did. Whatever is written to it
// is read and dropped.
void cancel_at_end_of_input(const std::function<void(const std::string&)>& interrupt,
                            const std::shared_ptr<std::atomic<RunStage>>& stage) {
    std::thread([interrupt, stage] {
        char buffer[4096];
        while (std::fread(buffer, 1, sizeof buffer, stdin) > 0) {
        }
        if (*stage != RunStage::results_given) {
            interrupt("Standard input ended: cancelling...");
        }
    }).detach();
}

// Sets up where the black bars found are stored, as `settings` say. Returns
// the saved scores' folder.
// The black bars found, kept with the saved scores as they are used:
// read and written, read only (--no-save), or neither (--no-cache).
void use_crop_store(const SavedScoresUse& saved) {
    if (!saved.read) {
        set_crop_store({});
        return;
    }
    const std::filesystem::path folder = saved.folder;
    const bool save = saved.save;
    set_crop_store(
        [folder, save] {
            if (save) {
                std::filesystem::create_directories(folder);
            }
            return folder / black_bars_file;
        },
        save);
}

// Whether looking for `video`'s film grain failed (scan_stream_start).
bool film_grain_not_known(const VideoInfo& video) {
    const std::string codec = ascii_lower(video.codec_name);
    return !video.film_grain && (codec == "h264" || codec == "hevc" || codec == "vvc" || codec == "av1");
}

// Plans each test video's comparison with the reference: the metrics it can
// be compared on, notes on what the reader should know, and the jobs that
// calculate what is not saved: its own, and one of CVVDP alone for each
// further display.
class ComparisonPlanner {
public:
    ComparisonPlanner(const CompareArguments& arguments, const Settings& settings, const VideoInfo& source,
                      const std::vector<CvvdpDisplayChoice>& displays, std::vector<std::string> asked,
                      const SavedScoresUse& saved)
        : arguments_(arguments),
          settings_(settings),
          source_(source),
          displays_(displays),
          asked_(std::move(asked)),
          backends_(perceptual_backends(arguments)),
          // The saved scores know a video by its file, not by the colours it
          // was read with, so --colour scores are neither read nor saved.
          use_saved_(saved.read && !arguments.colour),
          save_(saved.save),
          cache_folder_(saved.folder) {
        note_argument_use(source);
    }

    // Plans the comparison of the video at `path` (as `read` gives it: it
    // throws ProbeError where the video cannot be read), kept in `videos`
    // with why it is not compared, if it is not. Returns what the command
    // line cannot do yet of it, or nothing.
    std::optional<std::string> plan(const std::filesystem::path& path, const std::function<VideoRead()>& read,
                                    const std::string& ffmpeg);
    // Each job's video, once every video is planned (the lists no longer grow).
    std::vector<TestVideo*> job_targets();
    // What the command asked for that no video used (--fps with no raw
    // stream, say), once every video is planned: about the command, not
    // about any video's scores, so not among its notes. Nothing when no test
    // video was read (its error says why).
    std::vector<std::string> unused_argument_notes() const;

    std::vector<TestVideo> videos;
    std::vector<VmafJob> jobs;
    // Why the command is wrong, once a test video turns out to be a raw
    // stream or raw YUV file that it does not give what it needs for
    // (read_video).
    std::optional<std::string> command_wrong;

private:
    void leave_out_cvvdp_if_unable(TestVideo& video) const;
    void leave_out_hdr_vmaf_if_not_pq(TestVideo& video) const;
    void add_colour_notes(TestVideo& video) const;
    void add_film_grain_notes(TestVideo& video) const;
    std::optional<std::string> queue_jobs(TestVideo& video, const std::string& ffmpeg);
    void queue_other_displays(TestVideo& video);
    // Whether `video`'s scores may be read from the saved scores: those it
    // would save (TestVideo::save_scores).
    bool reads_saved(const TestVideo& video) const { return use_saved_ && video.save_scores; }
    // Marks the arguments that only some videos use as used by `video`.
    void note_argument_use(const VideoInfo& video);

    const CompareArguments& arguments_;
    const Settings& settings_;
    const VideoInfo& source_;
    const std::vector<CvvdpDisplayChoice>& displays_;
    const std::vector<std::string> asked_;
    const PerceptualBackends backends_;
    const bool use_saved_;
    const bool save_;  // scores and records saved (not with --no-save or --no-cache)
    const std::filesystem::path cache_folder_;
    // Each job's video: its index, and for CVVDP on a further display, which.
    std::vector<std::pair<std::size_t, std::optional<std::size_t>>> job_videos_;
    // Whether the reference or a test video read so far is a raw stream or
    // raw YUV file (--fps), a raw YUV file (--size, --pix-fmt), or has film
    // grain (--film-grain); and whether any test video was read.
    bool raw_stream_used_ = false;
    bool raw_yuv_used_ = false;
    bool film_grain_used_ = false;
    bool vvc_used_ = false;
    bool test_video_read_ = false;
};

// What the command could ask for when the two videos last differently: the
// end that compares to the shorter video's end, its last second partial
// (end_second_of), from the range's start where it has one (a range with
// no end is all it can have here), or any part of both. A range of frames
// ends at the shorter video's last frame.
std::string different_durations_hint(const VideoInfo& source, const VideoInfo& test, const ComparedRange& range) {
    if (source.fps <= 0 || source.duration <= 0 || test.duration <= 0) {
        return " Here: --duration SECONDS, --range START-END or --frames FIRST-LAST.";
    }
    const long long shorter_frames = std::min(source.estimated_frame_count(), test.estimated_frame_count());
    if (range.in_frames) {
        return " Here: --frames " + std::to_string(range.start) + "-" + std::to_string(shorter_frames - 1) +
               " compares to the shorter video's last frame.";
    }
    const long long range_start = range.start;
    const long long end = end_second_of(shorter_frames, source.fps);
    const std::string to_the_end = range_start > 0
                                       ? "--range " + std::to_string(range_start) + "-" + std::to_string(end)
                                       : "--duration " + std::to_string(end);
    return " Here: " + to_the_end + " compares to the shorter video's end; --range START-END compares a part of both.";
}

std::optional<std::string> ComparisonPlanner::plan(const std::filesystem::path& path,
                                                   const std::function<VideoRead()>& read,
                                                   const std::string& ffmpeg) {
    TestVideo& video = videos.emplace_back();
    video.path = path;
    if (std::optional<std::string> nothing = found_nothing(path)) {
        video.error = *std::move(nothing);
        video.error_code = codes::no_match;
        return std::nullopt;
    }
    try {
        VideoRead found = read();
        if (!found.info) {
            video.error = "the command does not say how to read it";
            command_wrong = std::move(found.wrong);
            return std::nullopt;
        }
        video.info = with_assumed_colour(std::move(*found.info), arguments_.colour);
    } catch (const ProbeError& error) {
        video.error = std::string("it could not be read: ") + error.what();
        video.error_code = codes::unreadable;
        return std::nullopt;
    }
    try {  // the crops given, refused before anything runs where they cannot be cut
        box_from_edges(arguments_.source_crop.value_or(CropEdges{}), source_, "the reference");
        box_from_edges(arguments_.test_crop.value_or(CropEdges{}), *video.info, platform::path_text(path.filename()));
    } catch (const CropEntryError& error) {
        video.error = error.what();
        command_wrong = error.what();
        return std::nullopt;
    }
    video.source = source_;
    if (!arguments_.colour) {
        if (take_colour_tags(*video.info, source_)) {
            video.colours_taken = "the test video";
        } else if (take_colour_tags(video.source, *video.info)) {
            video.colours_taken = "the reference";
        }
    }
    // A raw test stream's scores are saved under the rate it was read at
    // (ComparisonRecipe::test_rate); a raw reference's are not, nor a raw YUV
    // file's, read at a size and pixel format the saved scores do not know.
    video.save_scores = !arguments_.colour && video.colours_taken.empty() && !source_.timed_raw &&
                        video.info->format_name != "rawvideo";
    note_argument_use(*video.info);
    test_video_read_ = true;
    add_film_grain_notes(video);
    video.metrics = asked_;
    leave_out_cvvdp_if_unable(video);
    leave_out_hdr_vmaf_if_not_pq(video);
    if (video.metrics.empty()) {
        video.error = "none of the metrics asked for can be calculated for it";
        video.error_code = codes::no_metric;
        return std::nullopt;
    }
    add_colour_notes(video);
    if (const std::optional<std::string> note = sdr_display_note(arguments_, video.source, video, displays_)) {
        video.note(codes::sdr_display, *note);
    }
    try {
        return queue_jobs(video, ffmpeg);
    } catch (const std::exception& error) {
        video.error = error.what();
        video.error_code = codes::not_comparable;
        if (video.error.starts_with("Durations do not match")) {
            video.error += different_durations_hint(video.source, *video.info, range_of(arguments_));
        }
        return std::nullopt;
    }
}

std::vector<TestVideo*> ComparisonPlanner::job_targets() {
    std::vector<TestVideo*> targets;
    for (const auto& [index, display] : job_videos_) {
        TestVideo& video = videos[index];
        targets.push_back(display ? &video.other_displays[*display] : &video);
    }
    return targets;
}

// CVVDP left out of `video`'s metrics, with a note why, where it cannot be
// calculated: with --subsample above 1, or with no GPU the run's Vship build
// can use.
void ComparisonPlanner::leave_out_cvvdp_if_unable(TestVideo& video) const {
    if (std::find(video.metrics.begin(), video.metrics.end(), "cvvdp") == video.metrics.end()) {
        return;
    }
    const VshipBuild build = vship_build(arguments_, settings_);
    std::optional<std::string> reason;
    const char* code = codes::needs_every_frame;
    if (arguments_.subsample > 1) {
        reason = "CVVDP needs every frame: it is not calculated with --subsample above 1";
    } else if (!detect_vship_device(build, gpu_probes_token()).first) {
        reason = "CVVDP is calculated on the GPU only, and no GPU that " + std::string(vship_build_label(build)) +
                 " can use was found";
        code = codes::no_gpu;
    }
    if (reason) {
        std::erase(video.metrics, "cvvdp");
        // Asked for by name, its absence is a failure of the run; ticked by
        // default in the settings, a note.
        if (arguments_.metrics) {
            video.failures.emplace_back("cvvdp", *reason);
            video.failure_codes["cvvdp"] = code;
        } else {
            video.note(codes::metric_left_out, *reason);
        }
    }
}

// HDR VMAF left out of `video`'s metrics where it does not apply: a pair HDR
// VMAF cannot score, as it scores PQ (HDR10) video only (hdr_colours). Not a
// failure, whether asked for by name or by the settings: the other metrics
// are the video's, and its outcome says why (metric_not_applicable).
void ComparisonPlanner::leave_out_hdr_vmaf_if_not_pq(TestVideo& video) const {
    if (std::find(video.metrics.begin(), video.metrics.end(), "vmaf_hdr") == video.metrics.end()) {
        return;
    }
    const std::variant<HdrColours, std::string> colours = hdr_colours(video.source, *video.info);
    if (const std::string* why = std::get_if<std::string>(&colours)) {
        std::erase(video.metrics, "vmaf_hdr");
        video.not_applicable.emplace_back("vmaf_hdr",
                                          MetricOutcome{metric_not_applicable, *why, codes::hdr_vmaf_colours});
        video.note(codes::not_applicable, "HDR VMAF is not calculated: " + *why);
    }
}

// What the reader should know about how the colours were read, where a
// perceptual metric depends on them: a video's taken from the other video,
// or guessed.
void ComparisonPlanner::add_colour_notes(TestVideo& video) const {
    if (arguments_.colour ||
        std::none_of(video.metrics.begin(), video.metrics.end(), [](const std::string& key) {
            return is_perceptual_metric(key);
        })) {
        return;
    }
    const struct {
        const VideoInfo* info;
        const char* which;
        const char* other;
    } sides[] = {{&video.source, "the reference", "the test video"}, {&*video.info, "the test video", "the reference"}};
    for (const auto& side : sides) {
        if (video.colours_taken == side.which) {
            video.note(codes::colours_taken, taken_colour_note(*side.info, side.which, side.other));
        } else if (!has_colour_tags(*side.info)) {
            video.note(codes::colours_guessed, guessed_colour_note(*side.info, side.which, video.source, *video.info));
        }
    }
}

// Film grain asked for but left out (FilmGrainScoring::why_without).
void ComparisonPlanner::add_film_grain_notes(TestVideo& video) const {
    const std::pair<const VideoInfo*, const char*> sides[] = {{&video.source, "the reference"},
                                                              {&*video.info, "the test video"}};
    for (const auto& [info, which] : sides) {
        if (film_grain_not_known(*info)) {
            video.note(codes::film_grain_unknown, std::string("the film grain of ") + which +
                                  " could not be looked for (FFmpeg failed or took too long): whether its score "
                                  "includes any is not known");
        }
        const FilmGrainScoring grain = effective_film_grain(*info);
        if (!grain.why_without.empty()) {
            video.note(codes::film_grain_left_out,
                       std::string("the film grain of ") + which + " is left out: " + grain.why_without);
        }
    }
}

void ComparisonPlanner::note_argument_use(const VideoInfo& video) {
    raw_stream_used_ = raw_stream_used_ || video.is_raw_stream();
    raw_yuv_used_ = raw_yuv_used_ || video.format_name == "rawvideo";
    film_grain_used_ = film_grain_used_ || effective_film_grain(video).scoring != GrainScoring::none ||
                       film_grain_not_known(video);
    vvc_used_ = vvc_used_ || ascii_lower(video.codec_name) == "vvc";
}

std::vector<std::string> ComparisonPlanner::unused_argument_notes() const {
    std::vector<std::string> notes;
    if (!test_video_read_) {
        return notes;
    }
    if (arguments_.fps && !raw_stream_used_) {
        notes.emplace_back("--fps is not used: it is for raw streams and raw YUV files, and neither the reference "
                           "nor any test video is one.");
    }
    if ((arguments_.size || arguments_.pix_fmt) && !raw_yuv_used_) {
        notes.emplace_back("--size and --pix-fmt are not used: they are for raw YUV files, and neither the "
                           "reference nor any test video is one.");
    }
    const auto asked = [this](const char* metric) {
        return std::find(asked_.begin(), asked_.end(), metric) != asked_.end();
    };
    if ((!arguments_.cvvdp_displays.empty() || arguments_.cvvdp_fill) && !asked("cvvdp")) {
        notes.emplace_back("--cvvdp-display and --cvvdp-fill are not used: CVVDP is not among the metrics.");
    }
    if (arguments_.model_v1 != "auto" && !asked("vmaf_v1")) {
        notes.emplace_back("--model-v1 is not used: VMAF v1 is not among the metrics.");
    }
    if (arguments_.model_hdr != "auto" && !asked("vmaf_hdr")) {
        notes.emplace_back("--model-hdr is not used: HDR VMAF is not among the metrics.");
    }
    if (arguments_.film_grain && !film_grain_used_) {
        notes.emplace_back("--film-grain is not used: neither the reference nor any test video has film grain.");
    }
    if (arguments_.vvc_decoder == "gpu" && vvc_used_ && !decoder_available(vvdec_decoder)) {
        notes.emplace_back("--vvc-decoder gpu is not used: this package has no vvdec-vulkan, so VVC is decoded on the "
                           "CPU.");
    }
    return notes;
}

// Queues the jobs that compare `video`, checked against the reference
// first: its own, unless everything is saved, and CVVDP on each further
// display. `ffmpeg` is asked which decoders this PC has, where a video may be
// decoded on the GPU. Returns what the command line cannot do yet of its own
// job, or nothing; throws when the two videos cannot be compared as asked.
std::optional<std::string> ComparisonPlanner::queue_jobs(TestVideo& video, const std::string& ffmpeg) {
    video.options = row_options(arguments_, settings_, video.metrics);
    validate_video_pair(video.source, *video.info, video.options);
    const auto [width, height] = analysis_dimensions(video.source, *video.info, video.options);
    VmafOptions options = video.options;
    options.model = video.options.compute_vmaf ? resolve_model(video.options, width, height) : "";
    const DecodingCapabilities capabilities =
        options.gpu_decode ? DecodingCapabilities::remembered(ffmpeg) : DecodingCapabilities{};
    const bool share_decode = shares_decode(video.source, *video.info, options, capabilities);
    const bool cvvdp = std::find(video.metrics.begin(), video.metrics.end(), "cvvdp") != video.metrics.end();
    // By the reference as it is read here: with the test video's tags where
    // it has none, or --colour's.
    const CvvdpDisplayChoice display = display_for(displays_, video.source, arguments_.cvvdp_fill);
    if (cvvdp) {
        video.display = display.first;
    }
    const std::size_t queued = jobs.size();
    if (std::optional<std::string> missing = queue(video, video.source, std::move(options), backends_,
                                                   display.second, reads_saved(video), save_, cache_folder_,
                                                   jobs, share_decode)) {
        return missing;
    }
    if (jobs.size() > queued) {
        job_videos_.emplace_back(videos.size() - 1, std::nullopt);
    }
    if (cvvdp) {
        queue_other_displays(video);
    }
    return std::nullopt;
}

// CVVDP on each further display: a comparison of its own, of CVVDP alone.
void ComparisonPlanner::queue_other_displays(TestVideo& video) {
    for (std::size_t display = 1; display < displays_.size(); ++display) {
        TestVideo& other = video.other_displays.emplace_back();
        other.path = video.path;
        other.info = video.info;
        other.source = video.source;
        other.colours_taken = video.colours_taken;
        other.metrics = {"cvvdp"};
        other.display = displays_[display].first;
        other.for_display_only = true;
        other.save_scores = video.save_scores;
        other.options = row_options(arguments_, settings_, other.metrics);
        const std::size_t before = jobs.size();
        queue(other, other.source, other.options, backends_, displays_[display].second, reads_saved(other), save_,
              cache_folder_, jobs);
        if (jobs.size() > before) {
            job_videos_.emplace_back(videos.size() - 1, video.other_displays.size() - 1);
        }
    }
}

// What the planning found for each video: why it is not compared, its
// notes, or that every score is saved.
// Videos as a note names them: up to three by name, then how many more
// ("a.mkv, b.mkv, c.mkv and 7 more").
std::string videos_named(const std::vector<const TestVideo*>& videos) {
    constexpr std::size_t named = 3;
    std::string text;
    for (std::size_t v = 0; v < videos.size() && v < named; ++v) {
        text += (v == 0 ? "" : v + 1 == videos.size() ? " and " : ", ") + videos[v]->name();
    }
    if (videos.size() > named) {
        text += " and " + std::to_string(videos.size() - named) + " more";
    }
    return text;
}

void report_plans(const std::vector<TestVideo>& videos, Progress& progress) {
    // Each note once, with the videos it is of, in the order first said: the
    // same note of ten videos (their colour tags, say) is one line.
    std::vector<std::pair<std::string, std::string>> notes;  // text and code
    std::map<std::string, std::vector<const TestVideo*>> of;
    for (const TestVideo& video : videos) {
        for (std::size_t n = 0; n < video.notes.size(); ++n) {
            std::vector<const TestVideo*>& listed = of[video.notes[n]];
            if (listed.empty()) {
                notes.emplace_back(video.notes[n], n < video.note_codes.size() ? video.note_codes[n] : "");
            }
            listed.push_back(&video);
        }
    }
    for (const TestVideo& video : videos) {
        if (!video.error.empty()) {
            progress.message(video.name() + ": not compared: " + video.error);
        }
    }
    for (const auto& [text, code] : notes) {
        const std::vector<const TestVideo*>& listed = of[text];
        if (progress.jsonl()) {
            nlohmann::ordered_json paths = nlohmann::ordered_json::array();
            for (const TestVideo* video : listed) {
                paths.push_back(platform::path_text(video->path));
            }
            progress.event({{"event", "note"}, {"code", code}, {"text", text}, {"videos", std::move(paths)}});
        } else {
            progress.message(videos_named(listed) + ": " + text);
        }
    }
    for (const TestVideo& video : videos) {
        if (video.from_cache) {
            progress.message(video.name() + ": every score is saved; nothing to calculate");
        }
    }
}

// What the jobs run with. Which decoders this FFmpeg and these GPUs have is
// only asked when a video may be decoded on the GPU (start_gpu_checks asked
// it already).
RunEnvironment run_environment(const ToolsStatus& tools, const std::vector<VmafJob>& jobs) {
    const bool gpu_decode = std::any_of(jobs.begin(), jobs.end(), [](const VmafJob& job) { return job.options.gpu_decode; });
    return RunEnvironment{tools.ffmpeg.path, format_version(tools.ffmpeg.version), tools.ffprobe.path,
                          gpu_decode ? DecodingCapabilities::remembered(tools.ffmpeg.path) : DecodingCapabilities{},
                          platform::logical_cores()};
}

// Why --model's file cannot be VMAF's model, when VMAF is asked for and it
// names one that is not there: said before anything is calculated.
std::optional<std::string> model_file_wrong(const CompareArguments& arguments, const std::vector<std::string>& asked) {
    if (arguments.model == "auto" || arguments.model == "standard" || arguments.model == "4k" ||
        std::find(asked.begin(), asked.end(), "vmaf") == asked.end()) {
        return std::nullopt;
    }
    std::error_code error;
    const std::filesystem::path path = python_path(arguments.model);
    if (std::filesystem::is_regular_file(path, error)) {
        return std::nullopt;
    }
    return "--model: no VMAF model file at " + platform::path_text(path) +
           " (give auto, standard, 4k, or a model .json file)";
}

// One reference's comparisons: the command's REFERENCE and TESTs, or the
// rows of one reference in a --batch list. It does not move once made: its
// planner keeps a reference to `source`.
struct Comparison {
    std::filesystem::path reference;
    std::vector<std::filesystem::path> tests;
    // Each test video's row in the --batch list, by its index in `tests`
    // (and in the planner's videos); empty without --batch.
    std::vector<const BatchRow*> rows;
    VideoInfo source;
    std::optional<ComparisonPlanner> planner;  // once the reference is read
    std::string error;  // why it is not compared (--batch: the others still are)
    std::string error_code{};  // error's (codes.h)
};

// With --test-remuxes on (or Settings'), each test video calculated and
// saved, its stream read once more after the results are written, for a
// remux of it to keep its scores (record_test_stream): said first, as it
// takes as long as reading the file, and stopped by Ctrl+C. So is each one
// whose saved scores were found without a stream recorded (saved by an older
// version: test_stream_wanted).
void record_test_streams(const std::vector<std::unique_ptr<Comparison>>& comparisons,
                         const std::filesystem::path& cache_folder, Progress& progress, const CancelToken& cancel) {
    std::vector<const TestVideo*> recorded;
    std::set<std::filesystem::path> paths;
    for (const std::unique_ptr<Comparison>& comparison : comparisons) {
        if (comparison->planner) {
            for (const TestVideo& video : comparison->planner->videos) {
                const bool calculated = video.save_scores && video.result && !video.from_cache;
                const bool found_without_one = video.result && video.from_cache &&
                                               test_stream_wanted(cache_folder, video.path);
                if ((calculated || found_without_one) && paths.insert(video.path).second) {
                    recorded.push_back(&video);
                }
            }
        }
    }
    if (recorded.empty()) {
        return;
    }
    progress.message("Reading the stream of each test video calculated, or scored by an older version, so "
                     "that a remux of it keeps its saved scores (--test-remuxes off skips this; Ctrl+C stops it):");
    for (const TestVideo* video : recorded) {
        if (cancel.cancelled()) {
            return;
        }
        const auto started = std::chrono::steady_clock::now();
        bool read = false;
        try {
            read = record_test_stream(cache_folder, video->path, cancel);
        } catch (const std::exception&) {
            // a record only: the scores are saved
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        progress.message("  " + platform::path_text(video->path.filename()) + ": " +
                         (cancel.cancelled() ? std::string("stopped")
                          : read             ? format_fixed(seconds, 1) + " s"
                                             : std::string("recorded before")));
    }
}

using Comparisons = std::vector<std::unique_ptr<Comparison>>;

// Every comparison's reference and test videos, read ahead on several
// threads at once (TasksAhead) rather than one after another as each is
// reached: a read mostly waits for ffprobe and FFmpeg runs, 0.05 to 0.5 s
// each. They are taken in the command's order (plan_comparison,
// ComparisonPlanner::plan), so its messages, errors and exit codes are those
// of reading each as it is reached. A reference's rate is not sampled
// (RateSample::later): the command shows only the test videos' rates.
class VideoReads {
public:
    VideoReads(const Comparisons& comparisons, const CompareArguments& arguments, const Settings& settings,
               const ToolsStatus& tools)
        : videos_(videos_of(comparisons)),
          firsts_(firsts_of(comparisons)),
          tasks_(videos_.size(), read_threads(),
                 [this, &arguments, &tools, grain = film_grain_choice(arguments, settings)](std::size_t index) {
                     return read(videos_[index], arguments, grain, tools);
                 }) {}

    // Comparison `comparison`'s reference, and its test video `test`, as
    // read_video read them; ProbeError thrown where they could not be read.
    VideoRead reference(std::size_t comparison) { return tasks_.get(firsts_[comparison]); }
    VideoRead test(std::size_t comparison, std::size_t test) { return tasks_.get(firsts_[comparison] + 1 + test); }

private:
    struct Video {
        std::filesystem::path path;
        bool reference = false;
    };

    // Each comparison's reference, then its test videos
    static std::vector<Video> videos_of(const Comparisons& comparisons) {
        std::vector<Video> videos;
        for (const std::unique_ptr<Comparison>& comparison : comparisons) {
            videos.push_back({comparison->reference, true});
            for (const std::filesystem::path& test : comparison->tests) {
                videos.push_back({test, false});
            }
        }
        return videos;
    }
    // Where each comparison's videos start in videos_of's list
    static std::vector<std::size_t> firsts_of(const Comparisons& comparisons) {
        std::vector<std::size_t> firsts;
        std::size_t first = 0;
        for (const std::unique_ptr<Comparison>& comparison : comparisons) {
            firsts.push_back(first);
            first += 1 + comparison->tests.size();
        }
        return firsts;
    }
    // A few at once: more only slow each other down reading the disk
    static std::size_t read_threads() { return std::clamp(std::thread::hardware_concurrency(), 1u, 8u); }

    static VideoRead read(const Video& video, const CompareArguments& arguments, std::optional<bool> grain,
                          const ToolsStatus& tools) {
        VideoRead found;
        if (video.reference) {
            found.info = read_video(video.path, arguments, grain, tools.ffprobe.path, tools.ffmpeg.path,
                                    "The reference, " + platform::path_text(video.path.filename()) + ",",
                                    RateSample::later, found.wrong);
        } else if (!found_nothing(video.path)) {  // (ComparisonPlanner::plan says what was found)
            found.info = read_video(video.path, arguments, grain, tools.ffprobe.path, tools.ffmpeg.path,
                                    platform::path_text(video.path.filename()), RateSample::now, found.wrong);
        }
        return found;
    }

    std::vector<Video> videos_;
    std::vector<std::size_t> firsts_;
    TasksAhead<VideoRead> tasks_;  // last: its threads read videos_
};

// A --batch group's test videos, each with its row: each row's pattern found
// as TEST's are (expand_tests), a file found twice taken once.
std::unique_ptr<Comparison> batch_comparison(const BatchGroup& group) {
    auto comparison = std::make_unique<Comparison>();
    comparison->reference = group.reference;
    std::set<std::string> files;  // each test video's, resolved()
    for (const BatchRow& row : group.rows) {
        for (std::filesystem::path& path : expand_tests({row.test}, group.reference)) {
            if (files.insert(resolved(path)).second) {
                comparison->tests.push_back(std::move(path));
                comparison->rows.push_back(&row);
            }
        }
    }
    return comparison;
}

// The comparisons the command asks for, their test videos found (which reads
// only the file system): those of --batch's list (`groups` keeps its rows),
// or REFERENCE's. Throws std::runtime_error for a list that cannot be read.
Comparisons comparisons_asked(const CompareArguments& arguments, std::vector<BatchGroup>& groups) {
    Comparisons comparisons;
    if (!arguments.batch) {
        auto comparison = std::make_unique<Comparison>();
        comparison->reference = absolute_path(python_path(*arguments.reference));
        comparison->tests = expand_tests(arguments.tests, comparison->reference);
        comparisons.push_back(std::move(comparison));
        return comparisons;
    }
    const std::filesystem::path list = python_path(*arguments.batch);
    groups = read_batch_file(read_whole_file(list), absolute_path(list).parent_path(), platform::path_text(list));
    for (const BatchGroup& group : groups) {
        comparisons.push_back(batch_comparison(group));
    }
    return comparisons;
}

// Why an output file the command names would replace one of the videos
// compared (it is written after the run), or nothing.
std::optional<std::string> output_names_a_video(const CompareArguments& arguments, const Comparisons& comparisons) {
    const std::pair<const std::optional<std::string>*, const char*> outputs[] = {
        {&arguments.json_path, "--json"},
        {&arguments.scores_csv, "--scores-csv"},
        {&arguments.segments_report, "--segments-report"},
        {&arguments.log_file, "--log-file"}};
    for (const auto& [output, option] : outputs) {
        if (!*output || **output == "-") {
            continue;
        }
        const std::string file = resolved(python_path(**output));
        const auto is_it = [&file](const std::filesystem::path& video) { return resolved(video) == file; };
        for (const std::unique_ptr<Comparison>& comparison : comparisons) {
            if (is_it(comparison->reference) || std::ranges::any_of(comparison->tests, is_it)) {
                return std::string(option) + " names one of the videos: " + **output;
            }
        }
    }
    return std::nullopt;
}

// What the command asked for that no comparison used (unused_argument_notes),
// of those that read a test video.
std::vector<std::string> unused_argument_notes(const Comparisons& comparisons) {
    std::optional<std::vector<std::string>> unused;
    for (const std::unique_ptr<Comparison>& comparison : comparisons) {
        const auto was_read = [](const TestVideo& video) { return video.info.has_value(); };
        if (!comparison->planner || std::ranges::none_of(comparison->planner->videos, was_read)) {
            continue;
        }
        std::vector<std::string> notes = comparison->planner->unused_argument_notes();
        if (!unused) {
            unused = std::move(notes);
        } else {
            std::erase_if(*unused, [&notes](const std::string& note) { return std::ranges::count(notes, note) == 0; });
        }
    }
    return unused.value_or(std::vector<std::string>{});
}

// Finds each job's black bars ahead of the run, in the run's order, while it
// runs: a video's detection then goes on while the GPU scores the video
// before it, rather than hold up its own job. A job whose bars are being
// found waits for that answer, and takes it (job_crops), so the crops, and
// the scores, are those it would find itself. What fails is left for the job,
// which meets it again and says why. Stops, ending a detection under way, as
// it goes.
class BlackBarsAhead {
public:
    BlackBarsAhead(const std::vector<VmafJob>& jobs, std::string ffmpeg)
        : thread_([this, &jobs, ffmpeg = std::move(ffmpeg)] {
              for (const VmafJob& job : jobs) {
                  if (stop_.cancelled()) {
                      return;
                  }
                  if (!detects_black_bars(job.options.crop_mode)) {
                      continue;
                  }
                  try {
                      job_crops(job, ffmpeg, stop_);
                  } catch (const std::exception&) {  // Cancelled too
                  }
              }
          }) {}
    BlackBarsAhead(const BlackBarsAhead&) = delete;
    BlackBarsAhead& operator=(const BlackBarsAhead&) = delete;
    ~BlackBarsAhead() {
        stop_.cancel();
        thread_.join();
    }

private:
    CancelToken stop_;  // before thread_, which uses it
    std::thread thread_;
};

// --scores-csv: a row for each test video with a result, in the columns
// bdrate reads (rate_encode: its sequence, its rate in kb/s and each metric's
// score), then the test video, its reference and its --batch row's other
// cells; the row's sequence is its list's where the list gives one. A video
// whose scores have no rate (rate_of_scores) is left out, said. False, said,
// when the file cannot be written.
bool write_scores_csv(const std::filesystem::path& file, const Comparisons& comparisons, Progress& progress) {
    RateTable table;
    std::vector<std::string> list_columns;  // the --batch list's other columns, in the order first met
    const auto is_scores_column = [](const std::string& column) {
        return column == "kbps" || find_metric(column) != nullptr;
    };
    for (const std::unique_ptr<Comparison>& comparison : comparisons) {
        if (!comparison->planner) {
            continue;
        }
        const std::vector<TestVideo>& videos = comparison->planner->videos;
        for (std::size_t index = 0; index < videos.size(); ++index) {
            const TestVideo& video = videos[index];
            if (!video.result) {
                continue;  // the tables say why
            }
            std::optional<RateTable::Encode> encode = rate_encode(*video.result);
            if (!encode) {
                progress.problem(video.name() + ": not in the scores file: its bitrate is not known, or only a part"
                                                " of it was compared");
                continue;
            }
            std::erase_if(encode->scores,
                          [&video](const auto& score) { return std::ranges::count(video.metrics, score.first) == 0; });
            encode->text["test"] = platform::path_text(video.path);
            encode->text["reference"] = encode->reference;
            if (const BatchRow* row = comparison->rows.empty() ? nullptr : comparison->rows[index]) {
                if (!row->sequence.empty()) {
                    encode->sequence = row->sequence;
                }
                for (const auto& [column, cell] : row->text) {
                    if (is_scores_column(column)) {
                        continue;  // the scores' own
                    }
                    encode->text[column] = cell;
                    if (std::ranges::count(list_columns, column) == 0) {
                        list_columns.push_back(column);
                    }
                }
            }
            table.encodes.push_back(std::move(*encode));
        }
    }
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        const auto has_it = [&key](const RateTable::Encode& encode) { return encode.scores.contains(key); };
        if (std::ranges::any_of(table.encodes, has_it)) {
            table.metric_keys.push_back(key);
        }
    }
    table.columns = {"sequence", "kbps"};
    append(table.columns, table.metric_keys);
    append(table.columns, std::vector<std::string>{"test", "reference"});
    append(table.columns, list_columns);
    if (const std::optional<std::string> failure = write_bytes_file(file, rate_table_csv(table))) {
        print_error("The scores file could not be written: " + *failure);
        return false;
    }
    progress.message("Rates and scores for bdrate in " + platform::path_text(file));
    return true;
}

// --name: each test video's files named after `name`, its fields those of
// the run (`fields`), the video, its reference and its --batch row (whose
// sequence, where the list gives none, is the reference's name, as
// --scores-csv's is).
void name_files(std::vector<TestVideo>& videos, const Comparison& comparison, const std::string& name,
                NameFields fields) {
    fields.reference = platform::path_text(comparison.reference.stem());
    for (std::size_t index = 0; index < videos.size(); ++index) {
        fields.test = platform::path_text(videos[index].path.stem());
        const BatchRow* row = comparison.rows.empty() ? nullptr : comparison.rows[index];
        fields.sequence = row != nullptr && !row->sequence.empty() ? row->sequence : *fields.reference;
        videos[index].file_name = expand_names(name, fields);
    }
}

// One comparison's results: its run, CSV and segment files written
// (`written` false where one could not be), its tables added to `tables`
// and its JSON returned.
nlohmann::ordered_json comparison_results(const CompareArguments& arguments, Comparison& comparison, bool cancelled,
                                          const RunRecord& record, const NameFields& fields, std::string& tables,
                                          bool& written, Progress& progress) {
    std::vector<TestVideo>& videos = comparison.planner->videos;
    const std::vector<Summaries> summaries = summarize(videos);
    const std::optional<nlohmann::ordered_json> difference =
        arguments.difference ? difference_metrics(videos, summaries) : std::nullopt;
    if (arguments.difference && !difference) {
        const auto missing = std::find_if(videos.begin(), videos.end(), [](const TestVideo& v) { return !v.result; });
        progress.problem("No difference: " + (missing != videos.end() ? missing->name() : "a video") +
                         " has no result.");
    }
    const auto message = [&progress](const std::string& text) { progress.message(text); };
    const auto failed = [&progress](const std::string& text) { progress.problem(text); };
    if (arguments.name) {
        name_files(videos, comparison, *arguments.name, fields);
    }
    if (arguments.run_dir) {
        written = write_run_files(videos, python_path(*arguments.run_dir), message, failed) && written;
    }
    if (arguments.frames_json) {
        written = write_csvs(videos, python_path(*arguments.frames_json), message, failed, true) && written;
    }
    if (arguments.score_summary) {
        written = write_score_summary(videos, comparison.reference, python_path(*arguments.score_summary), arguments,
                                      message, failed) &&
                  written;
    }
    if (arguments.csv) {
        const std::filesystem::path folder = python_path(*arguments.csv);
        written = write_csvs(videos, folder, message, failed) && written;
        if (arguments.difference) {
            written = write_difference_csv(videos, folder, message, failed) && written;
        }
    }
    const SegmentRules rules = segment_rules(arguments);
    if (!rules.conditions.empty() && arguments.segments_dir) {
        written = write_segment_files(videos, rules, python_path(*arguments.segments_dir), message, failed) && written;
    }
    // The quality gate (--require, --fail-on-segments): each video's checks,
    // which its JSON and the text after the tables give.
    if (!arguments.require.empty() || arguments.fail_on_segments) {
        const bool on_segments = arguments.fail_on_segments && !rules.conditions.empty();
        for (std::size_t v = 0; v < videos.size(); ++v) {
            TestVideo& video = videos[v];
            video.checks = gate_checks(video, summaries[v], arguments.require, on_segments ? &rules : nullptr);
            if (!gate_passed(video.checks)) {
                std::string failing;
                for (const GateCheck& check : video.checks) {
                    if (check.passed && !*check.passed) {
                        failing += (failing.empty() ? "" : ", ") + check.rule;
                    }
                }
                progress.message(video.name() + ": quality gate FAILED: " + failing);
            }
        }
    }
    if (arguments.stills) {
        written = write_stills(videos, arguments.stills_from.value_or("worst"),
                               static_cast<std::size_t>(arguments.stills_count.value_or(5)), rules,
                               python_path(*arguments.stills), message, failed) &&
                  written;
    }
    // After the CSV files: it says where each one went.
    nlohmann::ordered_json document =
        results_document(comparison.source, videos, summaries, cancelled, difference, &record);
    if (!rules.conditions.empty()) {
        add_segments_json(document, videos, rules);
    }
    if (arguments.log_file) {
        document["log_file"] = platform::path_text(python_path(*arguments.log_file));
    }
    tables += (tables.empty() ? "" : "\n") +
              results_table(comparison.source, videos, summaries, cancelled, difference);
    if (arguments.worst) {
        tables += "\n" + worst_text(videos, static_cast<std::size_t>(*arguments.worst));
    }
    if (!rules.conditions.empty()) {
        tables += "\n" + segments_text(videos, rules, shown_segments);
    }
    if (const std::string gates = gate_text(videos); !gates.empty()) {
        tables += "\n" + gates;
    }
    return document;
}

// Gives the results: printed as tables, or as JSON alone for "--json -",
// which is for a program to read; and written to the CSV files, the scores
// file and the JSON file asked for (the tables are printed as well then).
// With --batch, each reference's in turn, and the JSON a list of them. False
// when a file could not be written.
bool write_results(const CompareArguments& arguments, const Comparisons& comparisons, bool cancelled,
                   const std::optional<bool>& timed_out, const RunRecord& record,
                   const nlohmann::ordered_json& profile, const NameFields& fields,
                   Progress& progress) {
    bool written = true;
    std::string tables;
    nlohmann::ordered_json batch = nlohmann::ordered_json::array();
    nlohmann::ordered_json document;
    for (const std::unique_ptr<Comparison>& comparison : comparisons) {
        if (!comparison->error.empty()) {
            const std::string reference = platform::path_text(comparison->reference);
            tables += (tables.empty() ? "" : "\n") + reference + ": not compared: " + comparison->error + "\n";
            batch.push_back(
                {{"reference", reference}, {"error", comparison->error}, {"error_code", comparison->error_code}});
            if (!arguments.batch) {
                document = error_document(reference, comparison->error, comparison->error_code, &record);
                document["profile"] = profile;
            }
            continue;
        }
        document = comparison_results(arguments, *comparison, cancelled, record, fields, tables, written, progress);
        document["profile"] = profile;
        if (timed_out) {
            document["timed_out"] = *timed_out;  // --timeout or --video-timeout given
        }
        batch.push_back(document);
    }
    if (arguments.batch) {
        document = {{"batch", std::move(batch)}};
    }
    if (arguments.scores_csv) {
        written = write_scores_csv(python_path(*arguments.scores_csv), comparisons, progress) && written;
    }
    if (arguments.segments_report) {
        // Every reference's test videos, in one report.
        std::vector<std::pair<const TestVideo*, std::filesystem::path>> videos;
        for (const std::unique_ptr<Comparison>& comparison : comparisons) {
            if (comparison->error.empty() && comparison->planner) {
                for (const TestVideo& video : comparison->planner->videos) {
                    videos.emplace_back(&video, comparison->reference);
                }
            }
        }
        const std::filesystem::path file = python_path(*arguments.segments_report);
        if (const std::optional<std::string> failure = write_segments_report(videos, segment_rules(arguments), file)) {
            progress.problem("The weak segments report could not be written: " + *failure);
            written = false;
        } else {
            progress.message("Weak segments of every test video in " + platform::path_text(file));
        }
    }
    const std::optional<std::filesystem::path> json_file =
        arguments.json_path ? std::optional(python_path(*arguments.json_path)) : std::nullopt;
    if (json_file && platform::path_text(*json_file) == "-") {
        return print_results(python_json_dumps(document) + "\n") && written;
    }
    written = print_results(tables + "\n") && written;
    if (json_file) {
        if (const std::optional<std::string> failure = write_json_file(*json_file, document)) {
            print_error("The JSON file could not be written: " + *failure);
            return false;
        }
        progress.message("Results as JSON in " + platform::path_text(*json_file));
    }
    return written;
}

// --notify: the run as its notification tells it (notify.h): how it ended
// (cancelled; with failures where a video or a metric failed; else
// finished), how long the command took, the references, and each test video
// compared; sent as Settings > Notifications say, and said on `progress`.
void notify_of_run(const Settings& settings, const Comparisons& comparisons, bool cancelled, bool every_ok,
                   double seconds, Progress& progress) {
    NotifiedRun run;
    run.outcome = cancelled ? RunOutcome::cancelled
                  : every_ok ? RunOutcome::finished
                             : RunOutcome::finished_with_failures;
    run.outcome_text = cancelled ? "Cancelled" : every_ok ? "Finished" : "Finished with failures";
    run.seconds = seconds;
    run.app = std::string("VideoMetricsLab ") + VML_VERSION;
    const std::tm now = platform::local_time(std::time(nullptr));
    char finished[32];
    std::strftime(finished, sizeof finished, "%Y-%m-%d %H:%M", &now);
    run.finished_at = finished;
    for (const std::unique_ptr<Comparison>& comparison : comparisons) {
        run.reference += (run.reference.empty() ? "" : ", ") + platform::path_text(comparison->reference.filename());
        if (!comparison->planner) {
            continue;
        }
        for (const TestVideo& video : comparison->planner->videos) {
            if (!video.for_display_only) {
                run.videos.push_back(notified_video(video, cancelled));
            }
        }
    }
    for (const NotifyLine& line : notify_run_end(settings, run, post_notification,
                                                 utc_text(std::chrono::system_clock::now()))) {
        if (line.problem) {
            progress.problem(line.text);
        } else {
            progress.message(line.text);
        }
    }
}

// The fields of --json's, --scores-csv's and --log-file's paths replaced
// (file_names.h): the run's, and with one test video its own and its
// reference's. Why one cannot be (a video's field with several test videos,
// or {sequence}), or nothing.
std::optional<std::string> name_paths(CompareArguments& arguments, const Comparisons& comparisons,
                                      NameFields fields) {
    const bool one_video = comparisons.size() == 1 && comparisons.front()->tests.size() == 1;
    if (one_video) {
        fields.reference = platform::path_text(comparisons.front()->reference.stem());
        fields.test = platform::path_text(comparisons.front()->tests.front().stem());
    }
    const std::pair<std::optional<std::string>*, const char*> paths[] = {
        {&arguments.json_path, "--json"},
        {&arguments.scores_csv, "--scores-csv"},
        {&arguments.segments_report, "--segments-report"},
        {&arguments.log_file, "--log-file"}};
    for (const auto& [path, option] : paths) {
        if (!*path || **path == "-") {
            continue;
        }
        if (const std::optional<std::string> field = video_field_in(**path)) {
            if (*field == "{sequence}") {
                return std::string(option) + " " + **path + ": {sequence} names a test video's files (--name), not "
                                                            "the run's";
            }
            if (!one_video) {
                return std::string(option) + " " + **path + ": " + *field +
                       " is a test video's, and this run compares more than one";
            }
        }
        **path = expand_names(**path, fields);
    }
    return std::nullopt;
}

// --log-file: the app's log written to `file` as the window writes its own,
// headed as each of its sessions is, with the command line, and where it
// goes said. False, said, when the file cannot be opened.
bool start_log_file(const std::filesystem::path& file, const std::vector<std::string>& command_line,
                    Progress& progress) {
    if (!app_log::start_logging_to(file)) {
        print_error("--log-file: " + platform::path_text(file) + " could not be opened");
        return false;
    }
    progress.message("The log in " + platform::path_text(file));
    const app_log::Logger log("vml.cli");
    log.info("==== VideoMetricsLab-cli" + std::string(app_log::session_start));
    for (const std::string& line : app_log::environment_lines(VML_VERSION)) {
        log.info(line);
    }
    std::string words;
    for (const std::string& word : command_line) {
        const bool quoted = word.empty() || word.find(' ') != std::string::npos;
        words += (words.empty() ? "" : " ") + (quoted ? "\"" + word + "\"" : word);
    }
    log.info("Command line: " + words);
    return true;
}

// Takes `comparison`'s reference as read (reads, the comparison's `index`)
// and plans its test videos. Returns the exit code of a wrong command, or
// nothing. A reference that cannot be read is the comparison's error (with
// --batch, the others are still made), which the results give.
std::optional<int> plan_comparison(Comparison& comparison, std::size_t index, VideoReads& reads,
                                   const CompareArguments& arguments, const Settings& settings,
                                   const ToolsStatus& tools, const std::vector<CvvdpDisplayChoice>& displays,
                                   const std::vector<std::string>& asked, const SavedScoresUse& saved,
                                   Progress& progress) {
    try {
        VideoRead read = reads.reference(index);
        if (!read.info) {
            print_error(read.wrong);
            return exit_usage;
        }
        comparison.source = with_assumed_colour(std::move(*read.info), arguments.colour);
    } catch (const ProbeError& error) {
        comparison.error = std::string("The reference could not be read: ") + error.what();
        comparison.error_code = codes::reference_unreadable;
        progress.problem(platform::path_text(comparison.reference) + ": not compared: " + comparison.error);
        return std::nullopt;
    }
    ComparisonPlanner& planner =
        comparison.planner.emplace(arguments, settings, comparison.source, displays, asked, saved);
    for (std::size_t test = 0; test < comparison.tests.size(); ++test) {
        const auto read = [&reads, index, test] { return reads.test(index, test); };
        if (const std::optional<std::string> missing = planner.plan(comparison.tests[test], read, tools.ffmpeg.path)) {
            print_error("The C++ command line cannot do this yet: " + *missing + ".");
            return exit_usage;
        }
        if (planner.command_wrong) {
            print_error(*planner.command_wrong);
            return exit_usage;
        }
    }
    if (planner.videos.empty()) {
        if (!arguments.batch) {
            print_error("No test videos: the patterns find only the reference.");
            return exit_usage;
        }
        comparison.error = "its rows find only the reference";
        comparison.error_code = codes::no_test_video;
        comparison.planner.reset();
        progress.problem(platform::path_text(comparison.reference) + ": not compared: " + comparison.error);
        return std::nullopt;
    }
    report_plans(planner.videos, progress);
    return std::nullopt;
}

}  // namespace

std::vector<std::string> options_from_settings(const CompareArguments& arguments, bool gpu_from_settings) {
    std::vector<std::string> taken;
    const auto left_out = [&taken](bool given, const char* option) {
        if (!given) {
            taken.emplace_back(option);
        }
    };
    left_out(arguments.metrics.has_value(), "--metrics");
    left_out(arguments.vmaf_on || arguments.cpu, "--vmaf-on");
    left_out(arguments.vship.has_value(), "--vship");
    left_out(arguments.vship_pass.has_value(), "--vship-pass");
    left_out(arguments.film_grain.has_value(), "--film-grain");
    left_out(!arguments.cvvdp_displays.empty(), "--cvvdp-display");
    left_out(arguments.gpu_decode || arguments.no_gpu_decode, "--gpu-decode");
    left_out(arguments.vvc_decoder || arguments.no_gpu_decode, "--vvc-decoder");
    left_out(!gpu_from_settings, "--gpu");
    left_out(arguments.parallel.has_value(), "--parallel");
    left_out(arguments.frame_counts.has_value(), "--frame-counts");
    left_out(arguments.test_remuxes.has_value(), "--test-remuxes");
    left_out(arguments.profile.cache_dir || arguments.profile.no_cache, "--cache-dir");
    if (arguments.segments_saved && !arguments.segments_profile && !arguments.segments_preset) {
        taken.emplace_back("--segments");  // "saved": the profile in use
    }
    if (arguments.notify) {
        taken.emplace_back("--notify");  // its services and message
    }
    return taken;
}

void write_error_json(const CompareArguments& given, const std::string& error, const std::string& code) {
    CompareArguments arguments = given;
    const NameFields fields = run_fields(platform::local_time(std::time(nullptr)));
    if (!arguments.json_path || name_paths(arguments, {}, fields)) {
        return;
    }
    const std::optional<std::string> reference =
        arguments.batch || !arguments.reference
            ? std::nullopt
            : std::optional(platform::path_text(absolute_path(python_path(*arguments.reference))));
    const nlohmann::ordered_json document = error_document(reference, error, code);
    if (*arguments.json_path == "-") {
        print_results(python_json_dumps(document) + "\n");
    } else if (const std::optional<std::string> failure =
                   write_json_file(python_path(*arguments.json_path), document)) {
        print_error("The JSON file could not be written: " + *failure);
    }
}

int compare(const CompareArguments& given, const ChosenSettings& chosen, bool gpu_from_settings) {
    // The command's start, for the results' record and the files' names
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    const std::string started_utc = utc_text(start);
    const NameFields fields = run_fields(platform::local_time(std::chrono::system_clock::to_time_t(start)));
    const Clock::time_point began = Clock::now();
    CancelToken cancel;
    const auto progress = std::make_shared<Progress>(given.quiet, given.progress == "jsonl");
    // --progress jsonl: every line on standard error an event, the app's log
    // (its failures, everything with -v) and the errors said as well, until
    // the command returns.
    const struct LinesAsEvents {
        bool on;
        ~LinesAsEvents() {
            if (on) {
                app_log::console_lines_to({});
                print_errors_to({});
            }
        }
    } lines_as_events{progress->jsonl()};
    if (progress->jsonl()) {
        app_log::console_lines_to([progress](std::string_view level, std::string_view name, std::string_view text) {
            progress->event(
                {{"event", "log"}, {"level", std::string(level)}, {"logger", std::string(name)}, {"text", std::string(text)}});
        });
        print_errors_to([progress](const std::string& text) { progress->problem(text); });
    }
    const auto stage = std::make_shared<std::atomic<RunStage>>(RunStage::outside);
    const std::function<void(const std::string&)> interrupt = cancel_on_interrupt(cancel, progress, stage);
    if (given.cancel_on_stdin_eof) {
        cancel_at_end_of_input(interrupt, stage);
    }
    RunWatch watch(given, cancel, progress, stage);  // --timeout, --video-timeout
    // The test videos, found first (it reads only the file system), so that
    // a wrong command is said before any work.
    std::vector<BatchGroup> groups;  // --batch's rows, which the comparisons point to
    Comparisons comparisons;
    try {
        comparisons = comparisons_asked(given, groups);
    } catch (const std::runtime_error& error) {
        const std::string why = "--batch: " + std::string(error.what());
        print_error(why);
        write_error_json(given, why, codes::batch_unreadable);
        return exit_input;
    }
    CompareArguments arguments = given;  // with the fields in its paths replaced
    if (const std::optional<std::string> wrong = name_paths(arguments, comparisons, fields)) {
        print_error(*wrong);
        return exit_usage;
    }
    if (const std::optional<std::string> wrong = output_names_a_video(arguments, comparisons)) {
        print_error(*wrong);
        return exit_usage;
    }
    if (arguments.difference) {
        if (const std::optional<std::string> refused = difference_refusal(comparisons.front()->tests.size())) {
            print_error(*refused);
            return exit_usage;
        }
    }
    if (const std::optional<std::string> refused = gpu_refusal(arguments.gpu, platform::gpu_adapters())) {
        print_error(*refused);
        return exit_usage;
    }
    if (!arguments.gpu) {
        if (const std::optional<std::string> problem =
                chosen_gpu_problem(chosen_gpu(), gpu_from_settings, platform::gpu_adapters())) {
            progress->problem(*problem);
        }
    }
    if (arguments.black_bars_given && (arguments.source_crop || arguments.test_crop)) {
        print_error("--black-bars cannot be given with --source-crop or --test-crop: the crops given are cut, and no "
                    "black bars are detected.");
        return exit_usage;
    }
    const Settings& settings = chosen.settings;
    arguments.score_summary_columns = settings.saved_details_columns;  // --score-summary' columns, as the window's
    arguments.cvvdp_presets_json = settings.cvvdp_presets_json;
    const SavedScoresUse saved = saved_scores_use(arguments.profile, settings);
    // Saved scores read unless asked to calculate again (--recalculate, or
    // the settings' "Reuse a saved result when a video is added again" off);
    // the black bars found are still read: they are not scores.
    const bool read_scores = saved.read && !arguments.recalculate && settings.use_cache;
    use_crop_store(saved);
    SavedScoresUse scores = saved;  // the scores': read as read_scores says
    scores.read = read_scores;
    const std::filesystem::path& cache_folder = saved.folder;
    const ToolsStatus tools = check_tools(settings);
    if (!tools.ok()) {
        print_error(ffmpeg_problem(tools));
        return exit_usage;
    }
    // A remuxed or moved reference, and a remuxed test video, keep their
    // saved scores (file_fingerprint.h).
    set_video_fingerprinter(fingerprinter_using(tools.ffmpeg.path, tools.ffprobe.path));
    // A remuxed test video, with --test-remuxes on (or Settings'): read
    // nothing of test streams otherwise.
    const bool test_remuxes =
        arguments.test_remuxes ? *arguments.test_remuxes == "on" : settings.recognise_test_remuxes;
    set_stream_fingerprinter(test_remuxes ? stream_fingerprinter_using(tools.ffmpeg.path, tools.ffprobe.path)
                                          : StreamFingerprinter());
    std::vector<CvvdpDisplayChoice> displays;
    try {
        displays = cvvdp_displays(arguments, settings);
    } catch (const std::invalid_argument& error) {
        print_error(error.what());
        return exit_usage;
    }
    // VMAF with VMAF YUV, which is made from its score.
    const std::vector<std::string> asked = with_metric_inputs(arguments.metrics.value_or(default_metrics(settings)));
    if (const std::optional<std::string> wrong = model_file_wrong(arguments, asked)) {
        print_error(*wrong);
        return exit_usage;
    }
    std::vector<std::string> saved_left_out;  // --segments saved's conditions on metrics not compared
    std::optional<WeakCondition> saved_fallback;  // the one taken for a profile with none on them
    if (arguments.segments_saved) {
        try {
            saved_left_out = take_saved_segments(arguments, settings, asked, &saved_fallback);
        } catch (const std::invalid_argument& error) {
            print_error(error.what());
            return exit_usage;
        }
    }
    if (const std::optional<std::string> refused = segments_refusal(arguments.segments, asked)) {
        print_error(*refused);
        return exit_usage;
    }
    for (const GateRule& rule : arguments.require) {
        if (std::find(asked.begin(), asked.end(), rule.metric) == asked.end()) {
            print_error("--require: " + rule.metric + " is not one of the metrics compared (-m)");
            return exit_usage;
        }
    }
    if (arguments.vship && !vship_build_bundled(vship_build_from(*arguments.vship))) {
        print_error("--vship " + *arguments.vship + ": " +
                    std::string(vship_build_label(vship_build_from(*arguments.vship))) +
                    " is not bundled with this package.");
        return exit_usage;
    }
#ifdef __APPLE__
    if (const std::optional<std::string> refused = mac_gpu_decode_refusal(arguments.gpu_decode)) {
        print_error(*refused);
        return exit_usage;
    }
#endif
    if (const std::optional<std::string> missing =
            missing_gpu_maker(arguments.gpu_decode, detected_gpu_vendors(), platform::lists_gpus())) {
        print_error(*missing);
        return exit_usage;
    }
    if (const std::optional<std::string> missing = not_yet(arguments, settings, asked)) {
        print_error("The C++ command line cannot do this yet: " + *missing + ".");
        return exit_usage;
    }
    if (arguments.log_file && !start_log_file(python_path(*arguments.log_file), arguments.command_line, *progress)) {
        return exit_usage;
    }
    VideoReads reads(comparisons, arguments, settings, tools);  // all started now, taken in order below
    const std::vector<std::future<void>> gpu_checks = start_gpu_checks(arguments, settings, asked, tools.ffmpeg.path);

    {
        // The saved scores' lookups may read a test video's whole stream to
        // tell a remux of a saved encode (said first): a run saved by it
        // takes longer than the reading.
        const StreamReads whole_streams(cancel, true, [&progress](const std::filesystem::path& file) {
            progress->message("Checking whether " + platform::path_text(file.filename()) +
                              " is a remux of a test video already scored: reading its stream...");
        });
        for (std::size_t index = 0; index < comparisons.size(); ++index) {
            if (const std::optional<int> wrong = plan_comparison(*comparisons[index], index, reads, arguments,
                                                                 settings, tools, displays, asked, scores,
                                                                 *progress)) {
                return *wrong;
            }
        }
    }
    for (const std::string& note : unused_argument_notes(comparisons)) {
        progress->message(note);
    }
    for (const std::string& condition : saved_left_out) {
        progress->message("note: --segments " + segments_source(arguments) + ": " + condition +
                          " is left out: its metric is not compared (-m)");
    }
    if (saved_fallback) {
        progress->message("note: --segments " + segments_source(arguments) +
                          ": the profile has no condition on the metrics compared, so " + saved_fallback->text() +
                          " is taken, as the Worst Frames tab starts (a condition given after it takes its place)");
    }
    // Every comparison's jobs in one run: one start of the GPU's work for all.
    std::vector<VmafJob> jobs;
    std::vector<TestVideo*> targets;
    for (const std::unique_ptr<Comparison>& comparison : comparisons) {
        if (comparison->planner) {
            append(targets, comparison->planner->job_targets());
            std::vector<VmafJob>& planned = comparison->planner->jobs;
            jobs.insert(jobs.end(), std::make_move_iterator(planned.begin()), std::make_move_iterator(planned.end()));
            planned.clear();
        }
    }

    if (progress->jsonl()) {
        progress->event({{"event", "run_started"}, {"jobs", jobs.size()}});
    }
    bool cancelled = false;
    if (!jobs.empty()) {
        const Clock::time_point started = Clock::now();
        *stage = RunStage::running;
        // What runs ended outright (End task, a crash) left in the temporary
        // folder goes while this one runs; waited for as the block ends.
        const std::future<void> leftovers = std::async(std::launch::async, [] {
            std::error_code error;
            const std::filesystem::path folder = std::filesystem::temp_directory_path(error);
            if (!error) {
                remove_leftovers(folder);
            }
        });
        trust_gpu_decoding();  // a run of its own (distrust_gpu_decoding)
        calculate_saved_scores_of_other_bars(jobs, tools.ffmpeg.path, cancel);
        {
            const BlackBarsAhead bars_ahead(jobs, tools.ffmpeg.path);
            run(run_environment(tools, jobs), targets, jobs, arguments.parallel.value_or(settings.parallel_jobs),
                arguments.vship_pass ? *arguments.vship_pass == "one" : settings.gpu_metrics_together,
                arguments.frame_counts ? *arguments.frame_counts == "check" : settings.check_frame_counts,
                scores.save ? std::optional(cache_folder) : std::nullopt, *progress, cancel, watch);
        }
        cancelled = cancel.cancelled();
        say_timed_out_videos(targets, watch);
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        progress->message(std::string(cancelled ? "Cancelled" : "Finished") + " after " + format_hms(seconds));
    } else {
        // Nothing to calculate: the GPU checks started with the command
        // would only be waited for as it ends.
        stop_gpu_probes();
    }

    // A video answered by its saved scores alone had no job to count its
    // frames in (frame_pairing.h): counted here, and saved with its scores.
    // Part of the run: reading whole files can take minutes, and Ctrl+C or
    // --timeout then cancels the counting, and the results are still given.
    *stage = RunStage::running;
    const bool read_whole_files =
        arguments.frame_counts ? *arguments.frame_counts == "check" : settings.check_frame_counts;
    // Each file once: every saved test video's pairing asks for the reference's.
    const FrameCounter count_saved_frames =
        counting_each_file_once(container_frame_counter(tools.ffprobe.path, cancel, read_whole_files));
    const auto count_saved = [&](TestVideo& video) {
        std::optional<FramesShort>& frames_short = video.result->frames_short;
        frames_short = count_frames_short(*frames_short, video.result->source_info, video.result->distorted_info,
                                          count_saved_frames);
        if (video.save_scores && scores.save && frames_short->counted()) {
            try {
                store_result(cache_folder, video.result->source, video.path, video.request.recipe, *video.result,
                             platform::path_text(video.path.stem()), video.request.metrics);
            } catch (const std::exception&) {
                // the counts are still this run's; the next counts again
            }
        }
    };
    bool every_ok = true;
    bool input_unread = false;  // a reference or a test video could not be read, or a pattern found nothing
    for (const std::unique_ptr<Comparison>& comparison : comparisons) {
        every_ok = every_ok && comparison->error.empty();
        // (one whose --batch rows find only the reference is a failure: without
        // --batch, a wrong command)
        input_unread = input_unread || comparison->error_code == codes::reference_unreadable;
        if (!comparison->planner) {
            continue;
        }
        for (TestVideo& video : comparison->planner->videos) {
            input_unread =
                input_unread || video.error_code == codes::no_match || video.error_code == codes::unreadable;
            if (const std::optional<std::string> note = near_miss_note(video)) {
                video.note(codes::near_miss_crop, *note);
            }
            if (!cancelled && video.from_cache && video.result && video.result->frames_short &&
                !video.result->frames_short->counted()) {
                try {
                    count_saved(video);
                } catch (const Cancelled&) {
                    cancelled = true;
                }
            }
            if (video.result && video.result->frames_short && !video.result->frames_short->in_step()) {
                video.note(codes::frame_counts, frames_short_text(*video.result->frames_short));
            }
            every_ok = every_ok && video_status(video, false) == "ok";
        }
    }
    watch.run_ended();
    cancelled = cancelled || cancel.cancelled();  // as the frames were counted, or right after
    // The command line cannot pause: no paused time
    RunRecord record{"cli",
                     arguments.command_line,
                     started_utc,
                     utc_text(std::chrono::system_clock::now()),
                     std::chrono::duration<double>(Clock::now() - began).count(),
                     0.0,
                     1,
                     this_machine(tools.ffmpeg.path, format_version(tools.ffmpeg.version))};
    const bool timed_out = watch.timed_out() || watch.any_job_timed_out();
    const bool written = write_results(arguments, comparisons, cancelled,
                                       arguments.timeout || arguments.video_timeout ? std::optional(timed_out)
                                                                                    : std::nullopt,
                                       record,
                                       profile_json(chosen, scores, options_from_settings(arguments, gpu_from_settings)),
                                       fields, *progress);
    *stage = RunStage::results_given;
    // A video below a --require bar, or with a weak segment (--fail-on-segments): its checks are made as the
    // results are written.
    bool gate_failed = false;
    for (const std::unique_ptr<Comparison>& comparison : comparisons) {
        if (comparison->planner) {
            for (const TestVideo& video : comparison->planner->videos) {
                gate_failed = gate_failed || !gate_passed(video.checks);
            }
        }
    }
    if (arguments.notify) {
        notify_of_run(settings, comparisons, cancelled, every_ok, record.wall_seconds.value_or(0.0), *progress);
    }
    if (test_remuxes && scores.save && !cancelled) {
        record_test_streams(comparisons, cache_folder, *progress, cancel);  // after the results: nothing waits
    }
    const int exit_code = cancelled               ? (watch.timed_out() ? exit_timeout : exit_cancelled)
                          : timed_out             ? exit_timeout
                          : input_unread          ? exit_input
                          : !every_ok || !written ? exit_failed
                          : gate_failed           ? exit_gate_failed
                                                  : exit_ok;
    app_log::Logger("vml.cli").info("Command line finished: exit code " + std::to_string(exit_code));
    if (progress->jsonl()) {
        progress->event(
            {{"event", "run_done"}, {"exit_code", exit_code}, {"cancelled", cancelled}, {"timed_out", timed_out}});
    }
    return exit_code;
}

}  // namespace vml::cli
