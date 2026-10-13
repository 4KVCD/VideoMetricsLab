#include "compare.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>

#include "report.h"
#include "vml/cancel.h"
#include "vml/crop_detect.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
#include "vml/job_runner.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/run_line.h"
#include "vml/settings.h"
#include "vml/time_format.h"

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

// A path resolved, as Python compares path.resolve() with another: two
// paths name the same file when these are equal.
std::string resolved(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(path, error);
    return comparable(std::filesystem::weakly_canonical(absolute, error));
}

// ------------------------------------------------------------------- colours

// Colour tag values that say nothing (the Python app's colour.UNTAGGED).
bool untagged(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value.empty() || value == "unknown" || value == "unspecified" || value == "reserved";
}

// Whether the video says its transfer or its primaries.
bool has_colour_tags(const VideoInfo& info) {
    return !untagged(info.color_transfer) || !untagged(info.color_primaries);
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
    if (untagged(info.color_space)) {
        info.color_space = colours.matrix;
    }
    return info;
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
        std::error_code error;
        const std::filesystem::path absolute = std::filesystem::absolute(python_path(arguments.model), error);
        options.custom_model_path = platform::path_text(std::filesystem::weakly_canonical(absolute, error));
    }
    for (const std::string& key : metric_keys) {
        if (metric_definition(key).ffmpeg_binding) {
            options.set_metric_enabled(key, true);
        }
    }
    return options;
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
    std::string others;
    for (const std::string& key : asked) {
        const MetricDefinition& definition = metric_definition(key);
        if (!definition.ffmpeg_binding) {
            others += (others.empty() ? "" : ", ") + std::string(definition.label);
        }
    }
    if (!others.empty()) {
        return others + " (only FFmpeg's metrics so far)";
    }
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
    if (!arguments.cvvdp_displays.empty()) {
        return "--cvvdp-display";
    }
    return std::nullopt;
}

// What this C++ version cannot do yet of a video's VMAF on the GPU (a task
// of its own, plan_tasks), or nothing.
std::optional<std::string> gpu_not_yet(const std::vector<MetricTask>& tasks) {
    const auto on_gpu = std::find_if(tasks.begin(), tasks.end(), [](const MetricTask& task) { return task.on_gpu(); });
    if (on_gpu == tasks.end()) {
        return std::nullopt;
    }
    // VMAF v1 runs on Vulkan, whatever scores VMAF v0.6.1 and NEG.
    const bool v1_alone = on_gpu->metric_keys == std::vector<std::string>{"vmaf_v1"};
    if (!v1_alone && gpu_vmaf_backend().backend != "vulkan") {
        return "VMAF on the GPU with libvmaf's CUDA code (the GPU backend setting is CUDA, or Vulkan cannot score "
               "VMAF on this GPU)";
    }
    if (by_timestamp()) {
        return "VMAF on the GPU with frames paired by timestamp (VML_FRAME_PAIRING)";
    }
    if (ascii_lower(platform::environment_variable("VML_VMAF_PIPES").value_or("")) == "canvas") {
        return "VMAF on the GPU with FFmpeg pairing the frames (VML_VMAF_PIPES=canvas)";
    }
    return std::nullopt;
}

// ------------------------------------------------------------------ progress

// How many characters UTF-8 text has (Python's len() of the string).
std::size_t length_of(std::string_view text) {
    return static_cast<std::size_t>(std::count_if(
        text.begin(), text.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }));
}

// The first `count` characters of UTF-8 text.
std::string first_characters(const std::string& text, std::size_t count) {
    std::size_t seen = 0;
    for (std::size_t at = 0; at < text.size(); ++at) {
        if ((static_cast<unsigned char>(text[at]) & 0xC0) != 0x80 && seen++ == count) {
            return text.substr(0, at);
        }
    }
    return text;
}

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
            platform::write_text(platform::Stream::error, "\r" + std::string(length_of(shown_), ' ') + "\r");
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
        const std::string wipe = "\r" + std::string(length_of(shown_), ' ') + "\r";
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
// some of its metrics failed, in its TestVideo (`video_of_job` says which
// video each job is), and showing the run's progress, as the Python app's
// command line does (cli.run).
void run(const RunEnvironment& environment, std::vector<TestVideo>& videos, const std::vector<VmafJob>& jobs,
         const std::vector<std::size_t>& video_of_job, int parallel, Progress& progress, const CancelToken& cancel) {
    const auto video = [&videos, &video_of_job](std::size_t index) -> TestVideo& {
        return videos[video_of_job[index]];
    };
    RunEvents events;
    events.job_started = [&](std::size_t index) { progress.message(video(index).name() + ": started"); };
    events.task_progress = [&](std::size_t index, const std::vector<TaskSnapshot>& tasks) {
        std::string line;
        for (const std::string& part : metric_line(tasks, false)) {
            line += (line.empty() ? "" : "; ") + part;
        }
        if (!line.empty()) {
            progress.job(index, video(index).name() + ": " + line);
        }
    };
    // A video's result so far is kept when one of its tasks finishes while
    // another still runs: kept if the rest is lost.
    events.result_updated = [&](std::size_t index, const ComparisonResult& result) { video(index).result = result; };
    events.job_finished = [&](std::size_t index, const ComparisonResult& result) {
        video(index).result = result;
        progress.message(video(index).name() + ": done");
    };
    events.job_failed = [&](std::size_t index, const std::string& message, const std::string& /*stderr_tail*/) {
        video(index).error = message;
        progress.message(video(index).name() + ": FAILED: " + message);
    };
    events.job_partially_failed = [&](std::size_t index, const ComparisonResult& result, const std::string& message,
                                      const std::string& /*stderr_tail*/, const MetricFailures& failures) {
        TestVideo& target = video(index);
        target.result = result;
        target.failures = failures.empty() ? MetricFailures{{"", message}} : failures;
        progress.message(target.name() + ": some metrics FAILED: " + message);
    };
    run_jobs(
        jobs, parallel, events,
        [&environment](const VmafJob& job, const VmafOptions& options, const ProgressCallback& on_progress,
                       const StatusCallback& on_status, const CancelToken& token) {
            return run_vmaf(environment, job.source_info, job.distorted_info, options, on_progress, on_status, token,
                            job.result_distorted_path);
        },
        environment.cores, cancel);
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

int compare(const CompareArguments& arguments) {
    const Settings settings = Settings::load();
    set_gpu_backend(settings.gpu_backend);
    const std::filesystem::path cache_folder =
        settings.cache_dir_path().value_or(platform::home_directory() / ".videometricslab" / "results_cache");
    set_crop_store([cache_folder] {
        std::filesystem::create_directories(cache_folder);
        return cache_folder / "black_bars.json";
    });
    const ToolsStatus tools = check_tools(settings);
    if (!tools.ok()) {
        print_error(ffmpeg_problem(tools));
        return exit_usage;
    }
    const std::vector<std::string> asked = arguments.metrics.value_or(default_metrics(settings));
    if (const std::optional<std::string> missing = not_yet(arguments, settings, asked)) {
        print_error("The C++ command line cannot do this yet: " + *missing + ".");
        return exit_usage;
    }

    // Ctrl+C: before the run, the command stops with exit code 130 and
    // nothing more; during it, the run is cancelled and what finished is
    // reported. The handler outlives this function, so it shares what it
    // uses.
    CancelToken cancel;
    const auto progress = std::make_shared<Progress>(arguments.quiet);
    const auto running = std::make_shared<std::atomic<bool>>(false);
    platform::on_interrupt([cancel, progress, running]() mutable {
        if (*running) {
            progress->message("Cancelling...");
        }
        cancel.cancel();
    });

    VideoInfo source;
    try {
        source = with_assumed_colour(probe_video(python_path(arguments.reference), tools.ffprobe.path),
                                     arguments.colour);
    } catch (const ProbeError& error) {
        print_error(std::string("The reference could not be read: ") + error.what());
        return exit_usage;
    }

    std::vector<TestVideo> videos;
    std::vector<VmafJob> jobs;
    std::vector<std::size_t> video_of_job;
    for (const std::filesystem::path& path : expand_tests(arguments.tests, python_path(arguments.reference))) {
        if (cancel.cancelled()) {
            return exit_cancelled;
        }
        TestVideo& video = videos.emplace_back();
        video.path = path;
        try {
            video.info = with_assumed_colour(probe_video(path, tools.ffprobe.path), arguments.colour);
        } catch (const ProbeError& error) {
            video.error = std::string("it could not be read: ") + error.what();
            continue;
        }
        video.metrics = asked;
        if (video.metrics.empty()) {
            video.error = "none of the metrics asked for can be calculated for it";
            continue;
        }
        try {
            video.options = row_options(arguments, settings, video.metrics, *video.info);
            validate_video_pair(source, *video.info, video.options);
            const auto [width, height] = analysis_dimensions(source, *video.info, video.options);
            VmafOptions options = video.options;
            options.model = video.options.compute_vmaf ? resolve_model(video.options, width, height) : "";
            std::vector<MetricTask> tasks = plan_tasks(source, *video.info, options, video.metrics);
            if (const std::optional<std::string> missing = gpu_not_yet(tasks)) {
                print_error("The C++ command line cannot do this yet: " + *missing + ".");
                return exit_usage;
            }
            jobs.push_back(VmafJob{source, *video.info, std::move(options), video.path, video.metrics, std::move(tasks)});
            video_of_job.push_back(videos.size() - 1);
        } catch (const std::exception& error) {
            video.error = error.what();
            if (video.error.starts_with("Durations do not match")) {
                video.error += " Here: --frames N or --duration SECONDS.";
            }
        }
    }
    if (cancel.cancelled()) {
        return exit_cancelled;
    }
    if (videos.empty()) {
        print_error("No test videos: the patterns find only the reference.");
        return exit_usage;
    }
    for (const TestVideo& video : videos) {
        if (!video.error.empty()) {
            progress->message(video.name() + ": not compared: " + video.error);
        }
        for (const std::string& note : video.notes) {
            progress->message(video.name() + ": " + note);
        }
    }

    bool cancelled = false;
    if (!jobs.empty()) {
        const Clock::time_point started = Clock::now();
        *running = true;
        // Which decoders this FFmpeg and these GPUs have is only asked when a
        // video may be decoded on the GPU.
        const bool gpu_decode =
            std::any_of(jobs.begin(), jobs.end(), [](const VmafJob& job) { return job.options.gpu_decode; });
        const RunEnvironment environment{tools.ffmpeg.path, format_version(tools.ffmpeg.version), tools.ffprobe.path,
                                         gpu_decode ? DecodingCapabilities::detect(tools.ffmpeg.path)
                                                    : DecodingCapabilities{},
                                         platform::logical_cores()};
        run(environment, videos, jobs, video_of_job, arguments.parallel.value_or(settings.parallel_jobs), *progress,
            cancel);
        progress->done();
        cancelled = cancel.cancelled();
        *running = false;
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        progress->message(std::string(cancelled ? "Cancelled" : "Finished") + " after " + format_hms(seconds));
    }

    const std::vector<Summaries> summaries = summarize(videos);
    const nlohmann::ordered_json document = results_document(source, videos, summaries, cancelled);
    bool written = true;
    if (arguments.csv) {
        written = write_csvs(videos, python_path(*arguments.csv),
                             [&progress](const std::string& text) { progress->message(text); });
    }
    // "--json -" is for a program to read, so it prints the JSON alone.
    // "--json FILE" writes the file, and the tables are printed as well.
    const std::optional<std::filesystem::path> json_file =
        arguments.json_path ? std::optional(python_path(*arguments.json_path)) : std::nullopt;
    if (json_file && platform::path_text(*json_file) == "-") {
        platform::write_text(platform::Stream::output, python_json_dumps(document) + "\n");
    } else {
        platform::write_text(platform::Stream::output, results_table(source, videos, summaries, cancelled) + "\n");
        if (json_file) {
            std::error_code error;
            if (json_file->has_parent_path()) {
                std::filesystem::create_directories(json_file->parent_path(), error);
            }
            std::ofstream file(*json_file, std::ios::binary);
            file << text_file(python_json_dumps(document) + "\n");
            file.close();
            if (error || !file) {
                print_error("The JSON file could not be written: " + platform::path_text(*json_file));
                written = false;
            } else {
                progress->message("Results as JSON in " + platform::path_text(*json_file));
            }
        }
    }
    if (cancelled) {
        return exit_cancelled;
    }
    const bool every_ok = std::all_of(videos.begin(), videos.end(),
                                      [](const TestVideo& video) { return video_status(video, false) == "ok"; });
    return every_ok && written ? exit_ok : exit_failed;
}

}  // namespace vml::cli
