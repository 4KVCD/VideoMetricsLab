#include "compare.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>

#include "report.h"
#include "vml/app_log.h"
#include "vml/cancel.h"
#include "vml/colour.h"
#include "vml/crop_detect.h"
#include "vml/cvvdp.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
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

// The note for a video read without colour tags: what the app took its
// colours to be (colour.h's video_colour's guess).
std::string guessed_colour_note(const VideoInfo& info, const std::string& which) {
    const std::string guess = !is_untagged_tag(info.color_space) ? "from its colour matrix, " + info.color_space
                              : info.height > 650         ? "BT.709 SDR, as it is over 650 lines"
                                                          : "BT.601 SDR, as it is 650 lines or fewer";
    return which + " has no colour tags, so its colours were guessed (" + guess +
           "). CVVDP, SSIMULACRA2 and Butteraugli depend on them: --colour says what they are, such as pq-bt2020 "
           "for HDR10.";
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

// Why a raw stream cannot be compared without --fps: it records no frame
// rate, and FFmpeg would read it at 25 frames a second. `which` names it.
std::string frame_rate_needed(const VideoInfo& info, const std::string& which) {
    return which + " is a raw " + codec_label(info.codec_name) +
           " stream, which records no frame rate: give its frame rate with --fps, such as --fps 24000/1001.";
}

// `info`, a raw stream timed at --fps's rate (ffprobe.h's timed_raw_stream),
// or any other video as it is; nothing for a raw stream when the command
// gives no --fps. Throws ProbeError when a raw stream's frames cannot be
// counted.
std::optional<VideoInfo> timed_by_arguments(VideoInfo info, const CompareArguments& arguments,
                                            const std::string& ffprobe) {
    if (!info.is_raw_stream()) {
        return info;
    }
    if (!arguments.fps) {
        return std::nullopt;
    }
    return timed_raw_stream(std::move(info), *arguments.fps, ffprobe);
}

// ------------------------------------------------------------------- options

// Where the command says `key` is calculated: its own option, else --cpu;
// nothing when it says nothing.
std::optional<std::string> where(const CompareArguments& arguments, std::string_view key) {
    const std::optional<std::string>& own = key == "vmaf"          ? arguments.vmaf_on
                                            : key == "ssimulacra2" ? arguments.ssimulacra2_on
                                                                   : arguments.butteraugli_on;
    if (own) {
        return own;
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

// The options the window's row would hold for these choices, for the test
// video `test`.
VmafOptions row_options(const CompareArguments& arguments, const Settings& settings,
                        const std::vector<std::string>& metric_keys, const VideoInfo& test) {
    const std::optional<std::string> vmaf_on = where(arguments, "vmaf");
    VmafOptions options;
    options.gpu_decode = settings.default_gpu_decode && !arguments.no_gpu_decode;
    options.vmaf_on_gpu = vmaf_on ? *vmaf_on == "gpu" : settings.default_vmaf_on_gpu;
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
    for (const std::string& key : metric_keys) {
        if (metric_definition(key).ffmpeg_binding) {
            options.set_metric_enabled(key, true);
        }
    }
    return options;
}

// Where SSIMULACRA2 and Butteraugli run: as asked on the command line, else
// as saved in Settings.
PerceptualBackends perceptual_backends(const CompareArguments& arguments, const Settings& settings) {
    PerceptualBackends chosen;
    for (const std::string key : {"ssimulacra2", "butteraugli"}) {
        const std::optional<std::string> asked = where(arguments, key);
        const std::string& saved =
            key == "ssimulacra2" ? settings.default_ssimulacra2_backend : settings.default_butteraugli_backend;
        chosen[key] = asked ? *asked : saved == "cpu" ? "cpu" : "gpu";
    }
    return chosen;
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
    const PerceptualBackends backends = perceptual_backends(arguments, settings);
    const std::optional<std::string> vmaf_on = where(arguments, "vmaf");
    const bool vmaf_on_gpu = vmaf_on ? *vmaf_on == "gpu" : settings.default_vmaf_on_gpu;
    std::vector<std::future<void>> checks;
    if (asked("cvvdp") || (asked("ssimulacra2") && backends.at("ssimulacra2") == "gpu") ||
        (asked("butteraugli") && backends.at("butteraugli") == "gpu")) {
        checks.push_back(std::async(std::launch::async, [] { detect_vship_device(); }));
    }
    if (vmaf_on_gpu && (asked("vmaf") || asked("vmaf_neg"))) {
        checks.push_back(std::async(std::launch::async, [] { gpu_vmaf_backend(); }));
    }
    if (vmaf_on_gpu && asked("vmaf_v1")) {
        checks.push_back(std::async(std::launch::async, [] { v1_available(); }));
    }
    return checks;
}

// A display CVVDP models, by its preset's name.
using CvvdpDisplayChoice = std::pair<std::string, CvvdpSettings>;

// The CVVDP displays to score on: those --cvvdp-display names, each once in
// the order given, else the app's default. Throws std::invalid_argument for
// a name that is no preset.
std::vector<CvvdpDisplayChoice> cvvdp_displays(const CompareArguments& arguments, const Settings& settings) {
    // A preset is a display; scaling to fill it starts off, as for a video
    // added to the window.
    if (arguments.cvvdp_displays.empty()) {
        const std::optional<CvvdpPreset> preset =
            cvvdp_preset_named(settings.cvvdp_default_preset, settings.cvvdp_presets_json);
        const CvvdpPreset& chosen = preset ? *preset : builtin_cvvdp_presets().front();
        return {{chosen.name, CvvdpSettings{chosen.settings.display}}};
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
        displays.emplace_back(preset->name, CvvdpSettings{preset->settings.display});
    }
    return displays;
}

// The note for HDR video whose CVVDP modelled the app's default display,
// which is SDR, or nothing.
std::optional<std::string> sdr_display_note(const CompareArguments& arguments, const VideoInfo& source,
                                            const TestVideo& video, const CvvdpDisplayChoice& display) {
    const std::string transfer = ascii_lower(source.color_transfer);
    if (std::find(video.metrics.begin(), video.metrics.end(), "cvvdp") == video.metrics.end() ||
        !arguments.cvvdp_displays.empty() || display.second.display.hdr ||
        (transfer != "smpte2084" && transfer != "arib-std-b67")) {
        return std::nullopt;
    }
    return "the video is HDR, and CVVDP modelled the app's default display, " + python_string_repr(display.first) +
           ", which is SDR. --cvvdp-display " + python_string_repr("30-inch 4K HDR monitor, dim room") +
           " (or another HDR preset) models an HDR one.";
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
    const std::optional<std::string> vmaf_on = where(arguments, "vmaf");
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
// a line for each video, redrawn in place on a terminal and elsewhere
// repeated every ten seconds.
class Progress {
public:
    explicit Progress(bool quiet) : quiet_(quiet), redraw_(platform::is_terminal(platform::Stream::error)) {}

    // Writes a line that stays.
    void message(const std::string& text) {
        const std::lock_guard lock(mutex_);
        write(text, true);
    }

    // Shows the line of the video of job `index`.
    void job(std::size_t index, const std::string& text) {
        const std::lock_guard lock(mutex_);
        if (redraw_) {
            write(text, false);
            return;
        }
        const Clock::time_point now = Clock::now();
        const auto last = last_.find(index);
        if (last == last_.end() || now - last->second >= every_) {
            last_[index] = now;
            write(text, true);
        }
    }

    // Wipes the line being redrawn, at the end of the run.
    void done() {
        const std::lock_guard lock(mutex_);
        if (redraw_ && !shown_.empty() && !quiet_) {
            platform::write_text(platform::Stream::error, "\r" + std::string(python_length(shown_), ' ') + "\r");
        }
        shown_.clear();
    }

private:
    void write(const std::string& text, bool keep) {
        if (quiet_) {
            return;
        }
        if (!redraw_) {
            if (keep) {
                platform::write_text(platform::Stream::error, text + "\n");
            }
            return;
        }
        const std::string wipe = "\r" + std::string(python_length(shown_), ' ') + "\r";
        if (keep) {
            platform::write_text(platform::Stream::error, wipe + text + "\n" + shown_);
        } else {
            shown_ = first_characters(text, 118);
            platform::write_text(platform::Stream::error, wipe + shown_);
        }
    }

    static constexpr std::chrono::seconds every_{10};
    const bool quiet_;
    const bool redraw_;
    std::mutex mutex_;
    std::string shown_;  // the line being redrawn
    std::map<std::size_t, Clock::time_point> last_;
};

// ------------------------------------------------------------------- the run

// Runs the jobs (job_runner.h), keeping each video's result, or why it or
// some of its metrics failed, in its TestVideo (`targets` holds each job's),
// saving it in the saved scores under `cache_folder` unless the video's
// save_scores is false, and showing the run's progress, as the Python app's
// command line does (cli.run).
void run(const RunEnvironment& environment, const std::vector<TestVideo*>& targets, const std::vector<VmafJob>& jobs,
         int parallel, bool gpu_metrics_together, const std::filesystem::path& cache_folder, Progress& progress,
         const CancelToken& cancel) {
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
    RunEvents events;
    events.job_started = [&](std::size_t index) { progress.message(video(index).label() + ": started"); };
    events.task_progress = [&](std::size_t index, const std::vector<TaskSnapshot>& tasks) {
        std::string line;
        for (const std::string& part : metric_line(tasks, false).parts) {
            line += (line.empty() ? "" : "; ") + part;
        }
        if (!line.empty()) {
            progress.job(index, video(index).label() + ": " + line);
        }
    };
    // A video's result so far is kept, and saved, when one of its tasks
    // finishes while another still runs: kept if the rest is lost.
    events.result_updated = keep;
    events.job_finished = [&](std::size_t index, const ComparisonResult& result) {
        keep(index, result);
        progress.message(video(index).label() + ": done");
    };
    events.job_failed = [&](std::size_t index, const std::string& message, const std::string& /*stderr_tail*/) {
        video(index).error = message;
        progress.message(video(index).label() + ": FAILED: " + message);
    };
    events.job_partially_failed = [&](std::size_t index, const ComparisonResult& result, const std::string& message,
                                      const std::string& /*stderr_tail*/, const MetricFailures& failures) {
        keep(index, result);
        TestVideo& target = video(index);
        target.failures = failures.empty() ? MetricFailures{{"", message}} : failures;
        progress.message(target.label() + ": some metrics FAILED: " + message);
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
// the run's, with VMAF's model resolved. Returns what this version cannot
// do yet of the job's VMAF on the GPU (gpu_not_yet), or nothing.
std::optional<std::string> queue(TestVideo& video, const VideoInfo& source, VmafOptions options,
                                 const PerceptualBackends& backends, const CvvdpSettings& cvvdp, bool use_saved,
                                 const std::filesystem::path& cache_folder, std::vector<VmafJob>& jobs) {
    // The request the window makes for a row: what the saved scores are
    // looked up and saved under, so that the two apps find each other's.
    const std::map<std::string, std::string> chosen(backends.begin(), backends.end());
    video.request = analysis_request(video.options, video.metrics, chosen, cvvdp);
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
        // A metric set to the GPU keeps a CPU score only without a GPU Vship
        // can use; the probe runs only if a score needs that answer.
        kept = reusable(*saved, video.request.metrics, chosen,
                        [](const std::string&) { return detect_vship_device().first.has_value(); },
                        colour_tags(source, *video.info));
    }
    if (std::all_of(video.metrics.begin(), video.metrics.end(), [&kept](const std::string& key) { return kept.has(key); })) {
        // Everything asked for is saved: nothing to run, as the window runs
        // nothing for a row that has its scores.
        video.result = std::move(saved);
        video.from_cache = true;
        return std::nullopt;
    }
    std::vector<MetricTask> tasks = plan_tasks(source, *video.info, options, video.metrics, backends, kept);
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
// reported. Set up first: the checks and probes before the run can take a
// while (60 s for ffprobe on a slow share). The handler outlives the
// command, so it shares what it uses.
void cancel_on_interrupt(CancelToken cancel, const std::shared_ptr<Progress>& progress,
                         const std::shared_ptr<std::atomic<bool>>& running) {
    platform::on_interrupt([cancel, progress, running]() mutable {
        if (!*running) {
            platform::end_process_now(exit_cancelled);
        }
        progress->message("Cancelling...");
        app_log::write(app_log::Level::info, "vml.job_runner", "Run: cancel requested");
        cancel.cancel();
    });
}

// Sets up what the run reads from `settings` where it is kept: the GPU
// backend, and where the black bars found are stored. Returns the saved
// scores' folder.
std::filesystem::path apply_settings(const Settings& settings) {
    set_gpu_backend(settings.gpu_backend);
    set_vship_backend(settings.gpu_backend);
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
          backends_(perceptual_backends(arguments, settings)),
          // The saved scores know a video by its file, not by the colours it
          // was read with, so --colour scores are neither read nor saved.
          use_saved_(!arguments.recalculate && settings.use_cache && !arguments.colour),
          cache_folder_(std::move(cache_folder)) {}

    // Plans the comparison of the video at `path` (read with `ffprobe`), kept
    // in `videos` with why it is not compared, if it is not. Returns what the
    // command line cannot do yet of it, or nothing.
    std::optional<std::string> plan(const std::filesystem::path& path, const std::string& ffprobe);
    // Each job's video, once every video is planned (the lists no longer grow).
    std::vector<TestVideo*> job_targets();

    std::vector<TestVideo> videos;
    std::vector<VmafJob> jobs;
    // Why the command is wrong, once a test video turns out to be a raw
    // stream and it gives no --fps.
    std::optional<std::string> frame_rate_missing;

private:
    void leave_out_cvvdp_if_unable(TestVideo& video) const;
    void add_colour_notes(TestVideo& video) const;
    std::optional<std::string> queue_jobs(TestVideo& video);
    void queue_other_displays(TestVideo& video);
    // Whether `video`'s scores may be read from the saved scores. The saved
    // scores know a video by its file, not by the frame rate --fps gave a
    // raw stream, so a pair with one neither reads nor saves them.
    bool reads_saved(const TestVideo& video) const {
        return use_saved_ && !source_.timed_raw && !video.info->timed_raw;
    }

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

std::optional<std::string> ComparisonPlanner::plan(const std::filesystem::path& path, const std::string& ffprobe) {
    TestVideo& video = videos.emplace_back();
    video.path = path;
    try {
        VideoInfo info = probe_video(path, ffprobe);
        std::optional<VideoInfo> timed = timed_by_arguments(info, arguments_, ffprobe);
        if (!timed) {
            video.error = "it is a raw stream with no frame rate";
            frame_rate_missing = frame_rate_needed(info, platform::path_text(path.filename()));
            return std::nullopt;
        }
        video.info = with_assumed_colour(std::move(*timed), arguments_.colour);
    } catch (const ProbeError& error) {
        video.error = std::string("it could not be read: ") + error.what();
        return std::nullopt;
    }
    video.save_scores = !arguments_.colour && !source_.timed_raw && !video.info->timed_raw;
    if (arguments_.fps && !source_.is_raw_stream() && !video.info->is_raw_stream()) {
        video.notes.emplace_back("--fps is not used: it is for raw streams, and neither this video nor the "
                                 "reference is one");
    }
    video.metrics = asked_;
    leave_out_cvvdp_if_unable(video);
    if (video.metrics.empty()) {
        video.error = "none of the metrics asked for can be calculated for it";
        return std::nullopt;
    }
    add_colour_notes(video);
    if (const std::optional<std::string> note = sdr_display_note(arguments_, source_, video, displays_.front())) {
        video.notes.push_back(*note);
    }
    try {
        return queue_jobs(video);
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
// calculated: with --subsample above 1, or with no GPU Vship can use.
void ComparisonPlanner::leave_out_cvvdp_if_unable(TestVideo& video) const {
    if (std::find(video.metrics.begin(), video.metrics.end(), "cvvdp") == video.metrics.end()) {
        return;
    }
    const char* reason = arguments_.subsample > 1
                             ? "CVVDP needs every frame: it is not calculated with --subsample above 1"
                         : !detect_vship_device().first
                             ? "CVVDP is calculated on the GPU only, and no GPU that Vship can use was found"
                             : nullptr;
    if (reason != nullptr) {
        std::erase(video.metrics, "cvvdp");
        video.notes.emplace_back(reason);
    }
}

// What the reader should know about how the colours were read, where a
// perceptual metric depends on them: a video's guessed.
void ComparisonPlanner::add_colour_notes(TestVideo& video) const {
    if (arguments_.colour ||
        std::none_of(video.metrics.begin(), video.metrics.end(), [](const std::string& key) {
            return is_perceptual_metric(key);
        })) {
        return;
    }
    for (const auto& [info, which] : {std::pair<const VideoInfo*, const char*>(&source_, "the reference"),
                                      std::pair<const VideoInfo*, const char*>(&*video.info, "the test video")}) {
        if (!has_colour_tags(*info)) {
            video.notes.push_back(guessed_colour_note(*info, which));
        }
    }
}

// Queues the jobs that compare `video`, checked against the reference
// first: its own, unless everything is saved, and CVVDP on each further
// display. Returns what the command line cannot do yet of its own job, or
// nothing; throws when the two videos cannot be compared as asked.
std::optional<std::string> ComparisonPlanner::queue_jobs(TestVideo& video) {
    video.options = row_options(arguments_, settings_, video.metrics, *video.info);
    validate_video_pair(source_, *video.info, video.options);
    const auto [width, height] = analysis_dimensions(source_, *video.info, video.options);
    VmafOptions options = video.options;
    options.model = video.options.compute_vmaf ? resolve_model(video.options, width, height) : "";
    const bool cvvdp = std::find(video.metrics.begin(), video.metrics.end(), "cvvdp") != video.metrics.end();
    if (cvvdp) {
        video.display = displays_.front().first;
    }
    const std::size_t queued = jobs.size();
    if (std::optional<std::string> missing = queue(video, source_, std::move(options), backends_,
                                                   displays_.front().second, reads_saved(video), cache_folder_, jobs)) {
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
        other.metrics = {"cvvdp"};
        other.display = displays_[display].first;
        other.for_display_only = true;
        other.save_scores = video.save_scores;
        other.options = row_options(arguments_, settings_, other.metrics, *video.info);
        const std::size_t before = jobs.size();
        queue(other, source_, other.options, backends_, displays_[display].second, reads_saved(other), cache_folder_,
              jobs);
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
        written = write_csvs(videos, python_path(*arguments.csv),
                             [&progress](const std::string& text) { progress.message(text); });
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
    if (const std::optional<std::string> missing = not_yet(arguments, settings, asked)) {
        print_error("The C++ command line cannot do this yet: " + *missing + ".");
        return exit_usage;
    }
    const std::vector<std::future<void>> gpu_checks = start_gpu_checks(arguments, settings, asked);

    VideoInfo source;
    try {
        const std::filesystem::path reference = python_path(arguments.reference);
        const VideoInfo read = probe_video(reference, tools.ffprobe.path);
        std::optional<VideoInfo> timed = timed_by_arguments(read, arguments, tools.ffprobe.path);
        if (!timed) {
            print_error(frame_rate_needed(read, "The reference, " + platform::path_text(reference.filename()) + ","));
            return exit_usage;
        }
        source = with_assumed_colour(std::move(*timed), arguments.colour);
    } catch (const ProbeError& error) {
        print_error(std::string("The reference could not be read: ") + error.what());
        return exit_usage;
    }
    ComparisonPlanner planner(arguments, settings, source, displays, asked, cache_folder);
    for (const std::filesystem::path& path : expand_tests(arguments.tests, python_path(arguments.reference))) {
        if (const std::optional<std::string> missing = planner.plan(path, tools.ffprobe.path)) {
            print_error("The C++ command line cannot do this yet: " + *missing + ".");
            return exit_usage;
        }
        if (planner.frame_rate_missing) {
            print_error(*planner.frame_rate_missing);
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
            arguments.parallel.value_or(settings.parallel_jobs), settings.gpu_metrics_together, cache_folder,
            *progress, cancel);
        progress->done();
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
