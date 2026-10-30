#include "compare.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <thread>

#include "report.h"
#include "vml/app_log.h"
#include "vml/cancel.h"
#include "vml/colour.h"
#include "vml/crop_detect.h"
#include "vml/cvvdp.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/film_grain_detect.h"
#include "vml/film_grain_scoring.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
#include "vml/isolated.h"
#include "vml/job_runner.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/run_line.h"
#include "vml/settings.h"
#include "vml/temp_files.h"
#include "vml/time_format.h"
#include "vml/vship.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vship_run.h"

namespace vml::cli {

namespace {

using Clock = std::chrono::steady_clock;

// ----------------------------------------------------------------- the paths

bool has_magic(std::string_view text) { return text.find_first_of("*?[") != std::string_view::npos; }

// A character as Windows compares names: ignoring case. Elsewhere, as it is.
char folded(char c) {
#ifdef _WIN32
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#else
    return c;
#endif
}

// Whether `name` matches a shell pattern (* ? [seq] [!seq]), as Python's
// fnmatch matches it.
bool name_matches(std::string_view name, std::string_view pattern) {
    if (pattern.empty()) {
        return name.empty();
    }
    if (pattern.front() == '*') {
        for (std::size_t skipped = 0; skipped <= name.size(); ++skipped) {
            if (name_matches(name.substr(skipped), pattern.substr(1))) {
                return true;
            }
        }
        return false;
    }
    if (name.empty()) {
        return false;
    }
    if (pattern.front() == '?') {
        return name_matches(name.substr(1), pattern.substr(1));
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
            const char c = folded(name.front());
            bool in_set = false;
            for (std::size_t i = 0; i < set.size(); ++i) {
                if (i + 2 < set.size() && set[i + 1] == '-') {
                    in_set = in_set || (c >= folded(set[i]) && c <= folded(set[i + 2]));
                    i += 2;
                } else {
                    in_set = in_set || c == folded(set[i]);
                }
            }
            return in_set != negated && name_matches(name.substr(1), pattern.substr(close + 1));
        }
    }
    return folded(name.front()) == folded(pattern.front()) && name_matches(name.substr(1), pattern.substr(1));
}

// The files and folders a pattern names, as Python's glob.glob finds them:
// a name starting with "." only for a pattern part that does too.
std::vector<std::filesystem::path> glob(const std::string& pattern) {
    const std::filesystem::path whole = python_path(pattern);
    std::vector<std::filesystem::path> found{whole.root_path()};
    for (const std::filesystem::path& part : whole.relative_path()) {
        const std::string text = platform::path_text(part);
        std::vector<std::filesystem::path> next;
        for (const std::filesystem::path& folder : found) {
            if (!has_magic(text)) {
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
                if (name_matches(name, text)) {
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

// A path's text as Python compares paths: on Windows ignoring the case of
// ASCII letters (Python's lower() folds other letters too).
std::string comparable(const std::filesystem::path& path) {
    std::string text = platform::path_text(path);
    std::transform(text.begin(), text.end(), text.begin(), folded);
    return text;
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

// Two paths name the same file when these are equal, as Python compares
// path.resolve() with another.
std::string resolved(const std::filesystem::path& path) { return comparable(resolved_path(path)); }

// ------------------------------------------------------------------- colours

// Whether the video says its transfer or its primaries.
bool has_colour_tags(const VideoInfo& info) {
    return !is_untagged_tag(info.color_transfer) || !is_untagged_tag(info.color_primaries);
}

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

// Gives `to` the colour tags of `from` when only `from` has them (its
// transfer and primaries, and its matrix unless `to` says its own), as
// --colour would give them: an encode keeps its source's colours, so a y4m
// encode of a tagged HDR10 source is HDR10 too. Returns whether it did.
bool take_colour_tags(VideoInfo& to, const VideoInfo& from) {
    if (has_colour_tags(to) || !has_colour_tags(from)) {
        return false;
    }
    to.color_transfer = from.color_transfer;
    to.color_primaries = from.color_primaries;
    if (is_untagged_tag(to.color_space)) {
        to.color_space = from.color_space;
    }
    return true;
}

// The note for a video that took the other's colour tags (take_colour_tags).
std::string taken_colour_note(const VideoInfo& info, const std::string& which, const std::string& other) {
    return which + " has no colour tags, so it was taken to have " + other + "'s (transfer " + info.color_transfer +
           ", primaries " + info.color_primaries +
           "). CVVDP, SSIMULACRA2 and Butteraugli depend on them: --colour says what they are, if they are not "
           "those.";
}

// Whether `path` is a raw YUV file (.yuv): pictures and nothing else, which
// FFmpeg reads only when told their size, pixel format and frame rate.
bool is_raw_yuv(const std::filesystem::path& path) { return ascii_lower(platform::path_text(path.extension())) == ".yuv"; }

// The video at `path` as the command reads it: a raw YUV file with --size,
// --pix-fmt and --fps (ffprobe.h's probe_raw_yuv), a raw stream timed at
// --fps (timed_raw_stream), any other video as ffprobe reads it, with the
// film grain its stream describes (detect_film_grain), scored with it or
// without as `with_film_grain` says (film_grain_choice). Nothing, with
// `wrong` saying why, when the command does not give what it needs (`which`
// names the video). Throws ProbeError when it cannot be read.
std::optional<VideoInfo> read_video(const std::filesystem::path& path, const CompareArguments& arguments,
                                    std::optional<bool> with_film_grain, const std::string& ffprobe,
                                    const std::string& ffmpeg, const std::string& which, std::string& wrong) {
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
    VideoInfo info = probe_video(path, ffprobe);
    info.film_grain = detect_film_grain(info, ffmpeg, CancelToken());
    info.with_film_grain = with_film_grain;
    if (!info.is_raw_stream()) {
        return info;
    }
    if (!arguments.fps) {
        // FFmpeg would read it at 25 frames a second.
        wrong = which + " is a raw " + codec_label(info.codec_name) +
                " stream, which records no frame rate: give its frame rate with --fps, such as --fps 24000/1001.";
        return std::nullopt;
    }
    return timed_raw_stream(std::move(info), *arguments.fps, ffprobe);
}

// ------------------------------------------------------------------- options

// Where the command says VMAF is calculated: --vmaf-on, else --cpu; nothing
// when it says nothing.
std::optional<std::string> vmaf_where(const CompareArguments& arguments) {
    if (arguments.vmaf_on) {
        return arguments.vmaf_on;
    }
    return arguments.cpu ? std::optional<std::string>("cpu") : std::nullopt;
}

// The duration limit, in seconds, of --duration or --frames; 0 for none.
//
// N frames are the frames that start before N/fps, numbered at the test
// video's frame rate as every metric numbers them. The limit is half a frame
// short of that, so that writing it to the millisecond (-t) can neither drop
// the last frame nor take one more. Throws std::invalid_argument for
// --frames on a video without a frame rate.
double duration_limit(const CompareArguments& arguments, const VideoInfo& test) {
    if (!arguments.frames) {
        return arguments.duration.value_or(0.0);
    }
    if (test.fps <= 0) {
        throw std::invalid_argument("its frame rate is unknown, so --frames cannot count its frames; use --duration");
    }
    return (static_cast<double>(*arguments.frames) - 0.5) / test.fps;
}

// The Vship build SSIMULACRA2, Butteraugli and CVVDP run on: --vship's, or
// the one the window gives new videos.
VshipBuild vship_build(const CompareArguments& arguments, const Settings& settings) {
    return arguments.vship ? vship_build_from(*arguments.vship) : default_vship_build(settings.default_vship_build);
}

// The maker --gpu-decode names: automatic for its "auto", and without it.
GpuVendor gpu_maker(const std::optional<std::string>& word) {
    if (word == "nvidia") {
        return GpuVendor::nvidia;
    }
    if (word == "intel") {
        return GpuVendor::intel;
    }
    if (word == "amd") {
        return GpuVendor::amd;
    }
    return GpuVendor::automatic;
}

// A GPU maker's name, as the window's GPU decode dropdown has it.
std::string maker_name(GpuVendor vendor) {
    return vendor == GpuVendor::nvidia ? "NVIDIA" : vendor == GpuVendor::intel ? "Intel" : "AMD";
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
        have += (have.empty() ? "" : ", ") + maker_name(vendor);
    }
    return "--gpu-decode " + *gpu_decode + ": this PC has no " + maker_name(maker) + " GPU" +
           (have.empty() ? "." : " (it has " + have + ").");
}

std::optional<bool> film_grain_choice(const CompareArguments& arguments, const Settings& settings) {
    if (arguments.film_grain) {
        return *arguments.film_grain == "with";
    }
    return film_grain_default_choice(settings.film_grain_default);
}

VmafOptions row_options(const CompareArguments& arguments, const Settings& settings,
                        const std::vector<std::string>& metric_keys, const VideoInfo& test) {
    const std::optional<std::string> vmaf_on = vmaf_where(arguments);
    VmafOptions options;
    // --gpu-decode turns GPU decoding on, whatever Settings say.
    options.gpu_decode = arguments.gpu_decode || (settings.default_gpu_decode && !arguments.no_gpu_decode);
    options.gpu_vendor = gpu_maker(arguments.gpu_decode);
    options.vmaf_on_gpu = vmaf_on ? *vmaf_on == "gpu" : settings.default_vmaf_on_gpu;
    options.vship_build = vship_build(arguments, settings);
    options.n_threads = arguments.threads.value_or(0);
    options.n_subsample = arguments.subsample;
    options.scale_algorithm = arguments.scaler;
    options.scale_direction = arguments.scale_to == "reference" ? ScaleDirection::distorted_to_source
                                                                : ScaleDirection::source_to_distorted;
    options.duration_limit = duration_limit(arguments, test);
    options.crop_mode = arguments.black_bars == "auto" ? CropMode::automatic : CropMode::none;
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
// 0.5 s each). Their answers are kept, and a caller that asks while one
// runs waits for it. The checks still running when the futures go are
// waited for.
std::vector<std::future<void>> start_gpu_checks(const CompareArguments& arguments, const Settings& settings,
                                                const std::vector<std::string>& metrics) {
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
// Metrics... picker hides (they are not calculated there).
std::vector<std::string> default_metrics(const Settings& settings) {
    VmafOptions options;
    options.extra_features = settings.default_extra_features();
    options.compute_xpsnr = settings.default_compute_xpsnr;
    options.compute_vmaf = settings.default_compute_vmaf;
    options.compute_vmaf_neg = settings.default_compute_vmaf_neg;
    options.compute_vmaf_v1 = settings.default_compute_vmaf_v1;
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
        return key == "psnr" || key == "ssim" || key == "xpsnr";
    });
    // The app scores PSNR, SSIM and XPSNR itself unless VMAF is calculated
    // on the CPU beside them, in FFmpeg; with FFmpeg's frames, the C++ app
    // pairs them itself only.
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

// ------------------------------------------------------------------ progress

// The run's progress on stderr, nothing with --quiet: lines that stay, and
// a line for each video. On a terminal the videos share one line, redrawn
// in place as the window draws its lines: at once when what it says apart
// from its figures changes (line_shape), and with new figures once a
// second, the videos with new figures taking the line in turn. Elsewhere
// each video's line is written out, at most every ten seconds.
class Progress {
public:
    explicit Progress(bool quiet) : quiet_(quiet), redraw_(platform::is_terminal(platform::Stream::error)) {}
    ~Progress() { stop_ticking(); }

    // Writes a line that stays.
    void message(const std::string& text) {
        const std::lock_guard lock(mutex_);
        write(text);
    }

    // Writes a line that stays, with -q too: a failure the user must see.
    void problem(const std::string& text) {
        const std::lock_guard lock(mutex_);
        if (quiet_ || !redraw_) {
            platform::write_text(platform::Stream::error, text + "\n");
            return;
        }
        platform::write_text(platform::Stream::error, wipe() + text + "\n" + shown_);
    }

    // Starts drawing new figures once a second, on a terminal, until done().
    void start() {
        const std::lock_guard lock(mutex_);
        if (redraw_ && !quiet_ && !ticker_.joinable()) {
            stopping_ = false;
            ticker_ = std::thread([this] { tick(); });
        }
    }

    // Shows `text`, the line of the video of job `index`; `shape` is what it
    // says apart from its figures (line_shape).
    void job(std::size_t index, const std::string& text, const LineShape& shape) {
        if (quiet_) {
            return;
        }
        const std::lock_guard lock(mutex_);
        if (!redraw_) {
            const Clock::time_point now = Clock::now();
            const auto last = last_written_.find(index);
            if (last == last_written_.end() || now - last->second >= write_every_) {
                last_written_[index] = now;
                write(text);
            }
            return;
        }
        const auto [found, added] = lines_.try_emplace(index);
        Line& line = found->second;
        line.text = text;
        if (added || line.shape != shape) {
            line.shape = shape;
            draw(index);
        } else {
            line.due = true;
        }
    }

    // Writes `text`, a line that stays, for the video of job `index`, which
    // has ended: its line is not drawn again.
    void finished(std::size_t index, const std::string& text) {
        const std::lock_guard lock(mutex_);
        lines_.erase(index);
        if (shown_index_ == index) {
            platform::write_text(platform::Stream::error, wipe());
            shown_.clear();
            shown_index_.reset();
        }
        write(text);
    }

    // Wipes the line being redrawn, at the end of the run.
    void done() {
        stop_ticking();
        const std::lock_guard lock(mutex_);
        if (redraw_ && !shown_.empty() && !quiet_) {
            platform::write_text(platform::Stream::error, wipe());
        }
        shown_.clear();
        shown_index_.reset();
        lines_.clear();
    }

private:
    struct Line {
        std::string text;
        LineShape shape;
        bool due = false;  // it has new figures, not drawn yet
    };

    // Once a second until the run ends: the line of the next video with new
    // figures, after the one shown, so that videos running at once take the
    // line in turn.
    void tick() {
        std::unique_lock lock(mutex_);
        while (!woken_.wait_for(lock, redraw_every_, [this] { return stopping_; })) {
            if (const std::optional<std::size_t> next = next_due()) {
                draw(*next);
            }
        }
    }

    std::optional<std::size_t> next_due() const {
        const auto is_due = [](const auto& line) { return line.second.due; };
        const auto after = shown_index_ ? lines_.upper_bound(*shown_index_) : lines_.begin();
        auto found = std::find_if(after, lines_.end(), is_due);
        if (found == lines_.end()) {
            found = std::find_if(lines_.begin(), after, is_due);
            if (found == after) {
                return std::nullopt;
            }
        }
        return found->first;
    }

    // Stops the ticks; the thread ends before this returns.
    void stop_ticking() {
        std::thread ticker;
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
            ticker = std::move(ticker_);
        }
        woken_.notify_all();
        if (ticker.joinable()) {
            ticker.join();
        }
    }

    // Draws the line of the video of job `index` in place of the one shown.
    void draw(std::size_t index) {
        Line& line = lines_.at(index);
        line.due = false;
        const std::string wiped = wipe();
        shown_ = first_characters(line.text, 118);
        shown_index_ = index;
        platform::write_text(platform::Stream::error, wiped + shown_);
    }

    // Writes a line that stays: on a terminal, above the line being redrawn.
    void write(const std::string& text) {
        if (quiet_) {
            return;
        }
        if (!redraw_) {
            platform::write_text(platform::Stream::error, text + "\n");
            return;
        }
        platform::write_text(platform::Stream::error, wipe() + text + "\n" + shown_);
    }

    // What wipes the line being redrawn, leaving the cursor at its start.
    std::string wipe() const { return "\r" + std::string(python_length(shown_), ' ') + "\r"; }

    static constexpr std::chrono::seconds redraw_every_{1};
    static constexpr std::chrono::seconds write_every_{10};
    const bool quiet_;
    const bool redraw_;
    std::mutex mutex_;
    // On a terminal: each running video's line, the one being redrawn, and
    // the thread that draws new figures once a second.
    std::map<std::size_t, Line> lines_;
    std::string shown_;
    std::optional<std::size_t> shown_index_;
    std::thread ticker_;
    std::condition_variable woken_;
    bool stopping_ = false;
    // Elsewhere: when each video's line was last written.
    std::map<std::size_t, Clock::time_point> last_written_;
};

// ------------------------------------------------------------------- the run

// Runs the jobs (job_runner.h), keeping each video's result, or why it or
// some of its metrics failed, in its TestVideo (`targets` holds each job's),
// saving it in the saved scores under `cache_folder` unless the video's
// save_scores is false, and showing the run's progress (Progress). A port of
// the Python app's command line's cli.run.
void run(const RunEnvironment& environment, const std::vector<TestVideo*>& targets, const std::vector<VmafJob>& jobs,
         int parallel, bool gpu_metrics_together, const std::filesystem::path& cache_folder, Progress& progress,
         const CancelToken& cancel) {
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
        if (!target.save_scores) {
            return;
        }
        try {
            const std::lock_guard lock(saving);
            store_result(cache_folder, result.source, target.path, target.request.recipe, result,
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
        std::vector<std::string>& notes = video(index).notes;
        if (std::find(notes.begin(), notes.end(), note) == notes.end()) {
            notes.push_back(note);
            progress.message(video(index).label() + ": " + note);
        }
    };
    RunEvents events;
    events.job_started = [&](std::size_t index) { progress.message(video(index).label() + ": started"); };
    events.task_progress = [&](std::size_t index, const std::vector<TaskSnapshot>& tasks) {
        for (const TaskSnapshot& task : tasks) {
            note_fallback(index, task.step);
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
        progress.finished(index, video(index).label() + (whole || !cancel.cancelled() ? ": done" : ": cancelled"));
    };
    events.job_failed = [&](std::size_t index, const std::string& message, const std::string& /*stderr_tail*/) {
        video(index).error = message;
        progress.finished(index, video(index).label() + ": FAILED: " + message);
    };
    events.job_partially_failed = [&](std::size_t index, const ComparisonResult& result, const std::string& message,
                                      const std::string& /*stderr_tail*/, const MetricFailures& failures) {
        keep(index, result);
        TestVideo& target = video(index);
        target.failures = failures.empty() ? MetricFailures{{"", message}} : failures;
        progress.finished(index, target.label() + ": some metrics FAILED: " + message);
    };
    run_jobs(
        jobs, parallel, events,
        [&environment](const VmafJob& job, const VmafOptions& options, const ProgressCallback& on_progress,
                       const StatusCallback& on_status, const CancelToken& token) {
            return run_vmaf(environment, job.source_info, job.distorted_info, options, on_progress, on_status, token,
                            job.result_distorted_path);
        },
        [&environment](const VmafJob& job, const std::vector<std::string>& keys, bool together,
                       int concurrent_cpu_tasks, const PerceptualHooks& hooks, const CancelToken& token) {
            const PerceptualRequest request{environment,        job.source_info,         job.distorted_info,
                                            job.options,        job.perceptual_backends, job.cvvdp,
                                            concurrent_cpu_tasks};
            return run_perceptual_task(request, keys, together, hooks, token);
        },
        gpu_metrics_together, environment.cores, cancel);
}

// Adds the job that calculates what `video` asks for, less what is saved
// under `cache_folder` (when `use_saved`); with everything saved, takes the
// saved result and adds none (the Python app's cli._queue). `options` are
// the run's, with VMAF's model resolved; `share_decode` is plan_tasks's.
// Returns what this version cannot do yet of the job's VMAF on the GPU
// (gpu_not_yet), or nothing.
std::optional<std::string> queue(TestVideo& video, const VideoInfo& source, VmafOptions options,
                                 const PerceptualBackends& backends, const CvvdpSettings& cvvdp, bool use_saved,
                                 const std::filesystem::path& cache_folder, std::vector<VmafJob>& jobs,
                                 bool share_decode = false) {
    // The request the window makes for a row: what the saved scores are
    // looked up and saved under, so that the two apps find each other's.
    const std::map<std::string, std::string> chosen(backends.begin(), backends.end());
    const auto asks = [&video](std::string_view key) {
        return std::find(video.metrics.begin(), video.metrics.end(), key) != video.metrics.end();
    };
    const auto on_gpu = [&asks, &chosen](const char* key) {
        const auto backend = chosen.find(key);
        return asks(key) && (backend == chosen.end() || backend->second == "gpu");
    };
    if (video.options.vship_build == VshipBuild::fast && vship51_bundled() &&
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
        return std::nullopt;
    }
    std::vector<MetricTask> tasks =
        plan_tasks(source, *video.info, options, video.metrics, backends, kept, share_decode);
    std::optional<std::string> missing = gpu_not_yet(tasks);
    if (kept.empty()) {
        saved.reset();
    }
    jobs.push_back(VmafJob{source, *video.info, std::move(options), video.path, video.metrics, std::move(tasks), backends,
                           cvvdp, std::move(saved), std::move(kept), platform::path_text(video.path.stem())});
    return missing;
}

std::string ffmpeg_problem(const ToolsStatus& status) {
    std::string text = "FFmpeg cannot be used:";
    for (const std::string& problem : status.problems()) {
        text += "\n  " + problem;
    }
    return text + "\nPut ffmpeg and ffprobe on PATH, or set FFmpeg's folder in the app's Settings tab.";
}

void print_error(const std::string& text) { platform::write_text(platform::Stream::error, text + "\n"); }

// `text` with its lines ended as Python writes a text file on this system:
// CR LF on Windows.
std::string text_file(std::string text) {
#ifdef _WIN32
    std::string crlf;
    crlf.reserve(text.size() + text.size() / 16);
    for (const char c : text) {
        if (c == '\n') {
            crlf += '\r';
        }
        crlf += c;
    }
    return crlf;
#else
    return text;
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

std::vector<std::filesystem::path> expand_tests(const std::vector<std::string>& patterns,
                                                const std::filesystem::path& reference) {
    std::vector<std::filesystem::path> found;
    std::set<std::string> found_files;  // each path found, resolved()
    const auto add = [&found, &found_files](const std::filesystem::path& path, const std::string& file) {
        if (found_files.insert(file).second) {
            found.push_back(path);
        }
    };
    const std::string reference_file = resolved(reference);
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
            found.push_back(path);  // named in the results as not readable
        }
        for (const std::filesystem::path& match : matches) {
            if (const std::string file = resolved(match); file != reference_file) {
                add(match, file);
            }
        }
    }
    return found;
}

namespace {

// Ctrl+C: outside the run, the command ends at once with exit code 130 and
// nothing more, as the Python app's KeyboardInterrupt ends it. An FFmpeg
// check or ffprobe under way ends with it, as this process's job ends.
// During the run (`running`), the run is cancelled and what finished is
// reported, and the GPU checks started with the command are ended (a task
// of the run may be waiting for one: nothing after the run asks for one);
// a second Ctrl+C (or SIGTERM) ends the command at once, for a
// run whose cancel does not come back (a GPU driver that hangs). Closing the
// console window (or End task on it) is a Ctrl+C. Set up
// first: the checks and probes before the run can take a while (60 s for
// ffprobe on a slow share). The handler outlives the command, so it shares
// what it uses.
void cancel_on_interrupt(CancelToken cancel, const std::shared_ptr<Progress>& progress,
                         const std::shared_ptr<std::atomic<bool>>& running) {
    auto asked = std::make_shared<std::atomic<bool>>(false);
    platform::on_interrupt([cancel, progress, running, asked]() mutable {
        if (!*running || asked->exchange(true)) {
            platform::end_process_now(exit_cancelled);
        }
        progress->message("Cancelling...");
        app_log::write(app_log::Level::info, "vml.job_runner", "Run: cancel requested");
        cancel.cancel();
        stop_gpu_probes();
    });
}

// Sets up where the black bars found are stored, as `settings` say. Returns
// the saved scores' folder.
std::filesystem::path apply_settings(const Settings& settings) {
    const std::filesystem::path cache_folder = settings.cache_dir_path().value_or(default_cache_dir());
    set_crop_store([cache_folder] {
        std::filesystem::create_directories(cache_folder);
        return cache_folder / black_bars_file;
    });
    return cache_folder;
}

// Plans each test video's comparison with the reference: the metrics it can
// be compared on, notes on what the reader should know, and the jobs that
// calculate what is not saved: its own, and one of CVVDP alone for each
// further display.
class ComparisonPlanner {
public:
    ComparisonPlanner(const CompareArguments& arguments, const Settings& settings, const VideoInfo& source,
                      const std::vector<CvvdpDisplayChoice>& displays, std::vector<std::string> asked,
                      std::filesystem::path cache_folder)
        : arguments_(arguments),
          settings_(settings),
          source_(source),
          displays_(displays),
          asked_(std::move(asked)),
          backends_(perceptual_backends(arguments)),
          // The saved scores know a video by its file, not by the colours it
          // was read with, so --colour scores are neither read nor saved.
          use_saved_(!arguments.recalculate && settings.use_cache && !arguments.colour),
          cache_folder_(std::move(cache_folder)) {}

    // Plans the comparison of the video at `path` (read with `ffprobe`), kept
    // in `videos` with why it is not compared, if it is not. Returns what the
    // command line cannot do yet of it, or nothing.
    std::optional<std::string> plan(const std::filesystem::path& path, const std::string& ffprobe,
                                    const std::string& ffmpeg);
    // Each job's video, once every video is planned (the lists no longer grow).
    std::vector<TestVideo*> job_targets();

    std::vector<TestVideo> videos;
    std::vector<VmafJob> jobs;
    // Why the command is wrong, once a test video turns out to be a raw
    // stream or raw YUV file that it does not give what it needs for
    // (read_video).
    std::optional<std::string> command_wrong;

private:
    void leave_out_cvvdp_if_unable(TestVideo& video) const;
    void add_colour_notes(TestVideo& video) const;
    void add_film_grain_notes(TestVideo& video) const;
    std::optional<std::string> queue_jobs(TestVideo& video, const std::string& ffmpeg);
    void queue_other_displays(TestVideo& video);
    // Whether `video`'s scores may be read from the saved scores: those it
    // would save (TestVideo::save_scores).
    bool reads_saved(const TestVideo& video) const { return use_saved_ && video.save_scores; }

    const CompareArguments& arguments_;
    const Settings& settings_;
    const VideoInfo& source_;
    const std::vector<CvvdpDisplayChoice>& displays_;
    const std::vector<std::string> asked_;
    const PerceptualBackends backends_;
    const bool use_saved_;
    const std::filesystem::path cache_folder_;
    // Each job's video: its index, and for CVVDP on a further display, which.
    std::vector<std::pair<std::size_t, std::optional<std::size_t>>> job_videos_;
};

std::optional<std::string> ComparisonPlanner::plan(const std::filesystem::path& path, const std::string& ffprobe,
                                                   const std::string& ffmpeg) {
    TestVideo& video = videos.emplace_back();
    video.path = path;
    try {
        std::string wrong;
        std::optional<VideoInfo> read = read_video(path, arguments_, film_grain_choice(arguments_, settings_), ffprobe,
                                                   ffmpeg, platform::path_text(path.filename()), wrong);
        if (!read) {
            video.error = "the command does not say how to read it";
            command_wrong = wrong;
            return std::nullopt;
        }
        video.info = with_assumed_colour(std::move(*read), arguments_.colour);
    } catch (const ProbeError& error) {
        video.error = std::string("it could not be read: ") + error.what();
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
    video.save_scores =
        !arguments_.colour && video.colours_taken.empty() && !source_.timed_raw && !video.info->timed_raw;
    if (arguments_.fps && !source_.is_raw_stream() && !video.info->is_raw_stream()) {
        video.notes.emplace_back("--fps is not used: it is for raw streams and raw YUV files, and neither this "
                                 "video nor the reference is one");
    }
    if ((arguments_.size || arguments_.pix_fmt) && source_.format_name != "rawvideo" &&
        video.info->format_name != "rawvideo") {
        video.notes.emplace_back("--size and --pix-fmt are not used: they are for raw YUV files, and neither this "
                                 "video nor the reference is one");
    }
    add_film_grain_notes(video);
    video.metrics = asked_;
    leave_out_cvvdp_if_unable(video);
    if (video.metrics.empty()) {
        video.error = "none of the metrics asked for can be calculated for it";
        return std::nullopt;
    }
    add_colour_notes(video);
    if (const std::optional<std::string> note = sdr_display_note(arguments_, video.source, video, displays_)) {
        video.notes.push_back(*note);
    }
    try {
        return queue_jobs(video, ffmpeg);
    } catch (const std::exception& error) {
        video.error = error.what();
        if (video.error.starts_with("Durations do not match")) {
            video.error += " Here: --frames N or --duration SECONDS.";
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
    const char* reason = arguments_.subsample > 1
                             ? "CVVDP needs every frame: it is not calculated with --subsample above 1"
                         : !detect_vship_device(vship_build(arguments_, settings_), gpu_probes_token()).first
                             ? "CVVDP is calculated on the GPU only, and no GPU that Vship can use was found"
                             : nullptr;
    if (reason != nullptr) {
        std::erase(video.metrics, "cvvdp");
        video.notes.emplace_back(reason);
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
            video.notes.push_back(taken_colour_note(*side.info, side.which, side.other));
        } else if (!has_colour_tags(*side.info)) {
            video.notes.push_back(guessed_colour_note(*side.info, side.which, video.source, *video.info));
        }
    }
}

// Film grain asked for but left out (FilmGrainScoring::why_without), and
// --film-grain given for two videos with no film grain.
void ComparisonPlanner::add_film_grain_notes(TestVideo& video) const {
    const std::pair<const VideoInfo*, const char*> sides[] = {{&video.source, "the reference"},
                                                              {&*video.info, "the test video"}};
    bool any_grain = false;
    for (const auto& [info, which] : sides) {
        const FilmGrainScoring grain = effective_film_grain(*info);
        any_grain = any_grain || grain.scoring != GrainScoring::none;
        if (!grain.why_without.empty()) {
            video.notes.push_back(std::string("the film grain of ") + which + " is left out: " + grain.why_without);
        }
    }
    if (arguments_.film_grain && !any_grain) {
        video.notes.emplace_back("--film-grain is not used: neither this video nor the reference has film grain");
    }
}

// Queues the jobs that compare `video`, checked against the reference
// first: its own, unless everything is saved, and CVVDP on each further
// display. `ffmpeg` is asked which decoders this PC has, where a video may be
// decoded on the GPU. Returns what the command line cannot do yet of its own
// job, or nothing; throws when the two videos cannot be compared as asked.
std::optional<std::string> ComparisonPlanner::queue_jobs(TestVideo& video, const std::string& ffmpeg) {
    video.options = row_options(arguments_, settings_, video.metrics, *video.info);
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
                                                   display.second, reads_saved(video), cache_folder_, jobs,
                                                   share_decode)) {
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
        other.options = row_options(arguments_, settings_, other.metrics, *video.info);
        const std::size_t before = jobs.size();
        queue(other, other.source, other.options, backends_, displays_[display].second, reads_saved(other),
              cache_folder_, jobs);
        if (jobs.size() > before) {
            job_videos_.emplace_back(videos.size() - 1, video.other_displays.size() - 1);
        }
    }
}

// What the planning found for each video: why it is not compared, its
// notes, or that every score is saved.
void report_plans(const std::vector<TestVideo>& videos, Progress& progress) {
    for (const TestVideo& video : videos) {
        if (!video.error.empty()) {
            progress.message(video.name() + ": not compared: " + video.error);
        }
        for (const std::string& note : video.notes) {
            progress.message(video.name() + ": " + note);
        }
        if (video.from_cache) {
            progress.message(video.name() + ": every score is saved; nothing to calculate");
        }
    }
}

// What the jobs run with. Which decoders this FFmpeg and these GPUs have is
// only asked when a video may be decoded on the GPU.
RunEnvironment run_environment(const ToolsStatus& tools, const std::vector<VmafJob>& jobs) {
    const bool gpu_decode = std::any_of(jobs.begin(), jobs.end(), [](const VmafJob& job) { return job.options.gpu_decode; });
    return RunEnvironment{tools.ffmpeg.path, format_version(tools.ffmpeg.version), tools.ffprobe.path,
                          gpu_decode ? DecodingCapabilities::detect(tools.ffmpeg.path) : DecodingCapabilities{},
                          platform::logical_cores()};
}

// Writes `document` to `path`, its folder made first, as the Python app
// writes it. Why it could not be written, as the Python app's OSError says
// it (the path alone left a user guessing), or nothing.
std::optional<std::string> write_json_file(const std::filesystem::path& path, const nlohmann::ordered_json& document) {
    if (path.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            return python_os_error(error, path.parent_path());
        }
    }
    errno = 0;
    std::ofstream file(path, std::ios::binary);
    const int opened = errno;
    file << text_file(python_json_dumps(document) + "\n");
    file.close();
    if (!file) {
        const int code = opened != 0 ? opened : (errno != 0 ? errno : EIO);
        return python_os_error(std::error_code(code, std::generic_category()), path);
    }
    return std::nullopt;
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

// Gives the results: printed as tables, or as JSON alone for "--json -",
// which is for a program to read; and written to the CSV files and the JSON
// file asked for (the tables are printed as well then). False when a file
// could not be written.
bool write_results(const CompareArguments& arguments, const VideoInfo& source, const std::vector<TestVideo>& videos,
                   bool cancelled, Progress& progress) {
    const std::vector<Summaries> summaries = summarize(videos);
    const nlohmann::ordered_json document = results_document(source, videos, summaries, cancelled);
    bool written = true;
    if (arguments.csv) {
        written = write_csvs(
            videos, python_path(*arguments.csv), [&progress](const std::string& text) { progress.message(text); },
            [&progress](const std::string& text) { progress.problem(text); });
    }
    const std::optional<std::filesystem::path> json_file =
        arguments.json_path ? std::optional(python_path(*arguments.json_path)) : std::nullopt;
    if (json_file && platform::path_text(*json_file) == "-") {
        platform::write_text(platform::Stream::output, python_json_dumps(document) + "\n");
        return written;
    }
    platform::write_text(platform::Stream::output, results_table(source, videos, summaries, cancelled) + "\n");
    if (json_file) {
        if (const std::optional<std::string> failure = write_json_file(*json_file, document)) {
            print_error("The JSON file could not be written: " + *failure);
            return false;
        }
        progress.message("Results as JSON in " + platform::path_text(*json_file));
    }
    return written;
}

}  // namespace

int compare(const CompareArguments& arguments) {
    CancelToken cancel;
    const auto progress = std::make_shared<Progress>(arguments.quiet);
    const auto running = std::make_shared<std::atomic<bool>>(false);
    cancel_on_interrupt(cancel, progress, running);
    const Settings settings = Settings::load();
    const std::filesystem::path cache_folder = apply_settings(settings);
    const ToolsStatus tools = check_tools(settings);
    if (!tools.ok()) {
        print_error(ffmpeg_problem(tools));
        return exit_usage;
    }
    std::vector<CvvdpDisplayChoice> displays;
    try {
        displays = cvvdp_displays(arguments, settings);
    } catch (const std::invalid_argument& error) {
        print_error(error.what());
        return exit_usage;
    }
    const std::vector<std::string> asked = arguments.metrics.value_or(default_metrics(settings));
    if (const std::optional<std::string> wrong = model_file_wrong(arguments, asked)) {
        print_error(*wrong);
        return exit_usage;
    }
    if (arguments.vship && vship_build_from(*arguments.vship) == VshipBuild::stock && !vship51_bundled()) {
        print_error("--vship vship: stock Vship 5.1.2 is not bundled with this package.");
        return exit_usage;
    }
    if (const std::optional<std::string> missing =
            missing_gpu_maker(arguments.gpu_decode, detected_gpu_vendors(), platform::lists_gpus())) {
        print_error(*missing);
        return exit_usage;
    }
    if (const std::optional<std::string> missing = not_yet(arguments, settings, asked)) {
        print_error("The C++ command line cannot do this yet: " + *missing + ".");
        return exit_usage;
    }
    const std::vector<std::future<void>> gpu_checks = start_gpu_checks(arguments, settings, asked);

    VideoInfo source;
    try {
        const std::filesystem::path reference = python_path(arguments.reference);
        std::string wrong;
        std::optional<VideoInfo> read =
            read_video(reference, arguments, film_grain_choice(arguments, settings), tools.ffprobe.path,
                       tools.ffmpeg.path, "The reference, " + platform::path_text(reference.filename()) + ",", wrong);
        if (!read) {
            print_error(wrong);
            return exit_usage;
        }
        source = with_assumed_colour(std::move(*read), arguments.colour);
    } catch (const ProbeError& error) {
        print_error(std::string("The reference could not be read: ") + error.what());
        return exit_usage;
    }
    ComparisonPlanner planner(arguments, settings, source, displays, asked, cache_folder);
    for (const std::filesystem::path& path : expand_tests(arguments.tests, python_path(arguments.reference))) {
        if (const std::optional<std::string> missing = planner.plan(path, tools.ffprobe.path, tools.ffmpeg.path)) {
            print_error("The C++ command line cannot do this yet: " + *missing + ".");
            return exit_usage;
        }
        if (planner.command_wrong) {
            print_error(*planner.command_wrong);
            return exit_usage;
        }
    }
    if (planner.videos.empty()) {
        print_error("No test videos: the patterns find only the reference.");
        return exit_usage;
    }
    report_plans(planner.videos, *progress);

    bool cancelled = false;
    if (!planner.jobs.empty()) {
        const Clock::time_point started = Clock::now();
        *running = true;
        // What runs ended outright (End task, a crash) left in the temporary
        // folder goes while this one runs; waited for as the block ends.
        const std::future<void> leftovers = std::async(std::launch::async, [] {
            std::error_code error;
            const std::filesystem::path folder = std::filesystem::temp_directory_path(error);
            if (!error) {
                remove_leftovers(folder);
            }
        });
        run(run_environment(tools, planner.jobs), planner.job_targets(), planner.jobs,
            arguments.parallel.value_or(settings.parallel_jobs),
            arguments.vship_pass ? *arguments.vship_pass == "one" : settings.gpu_metrics_together, cache_folder,
            *progress, cancel);
        cancelled = cancel.cancelled();
        *running = false;
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        progress->message(std::string(cancelled ? "Cancelled" : "Finished") + " after " + format_hms(seconds));
    }

    const bool written = write_results(arguments, source, planner.videos, cancelled, *progress);
    if (cancelled) {
        return exit_cancelled;
    }
    const bool every_ok = std::all_of(planner.videos.begin(), planner.videos.end(),
                                      [](const TestVideo& video) { return video_status(video, false) == "ok"; });
    return every_ok && written ? exit_ok : exit_failed;
}

}  // namespace vml::cli
