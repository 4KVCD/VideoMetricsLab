#include "bitrate.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <system_error>

#include "common.h"
#include "compare.h"
#include "vml/cancel.h"
#include "vml/export_csv.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/geometry.h"
#include "vml/number_format.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/settings.h"
#include "vml/time_format.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

// What the JSON is: its own format, not compare's results (results_format).
constexpr const char* bitrate_format = "bitrate-1";

std::string name_of(const std::filesystem::path& path) { return platform::path_text(path.filename()); }

// kb/s as the table shows it: Mb/s, to a thousandth.
std::string mbps_text(double kbps) { return format_fixed(kbps / 1000, 3); }

// kb/s in the JSON: to a tenth (the summary comes from float rates, exact to
// a tenth at film bitrates and no finer).
double kbps_value(double kbps) { return std::round(kbps * 10) / 10; }

// Each video named (files, or what patterns find), each read by ffprobe
// first, so that a command that cannot be followed (a raw stream without
// --fps) is refused before any packets are read. Nothing, with the refusal
// printed, for such a command.
std::optional<std::vector<BitrateVideo>> read_videos(const BitrateArguments& arguments, const std::string& ffprobe,
                                                     Progress& progress) {
    std::vector<BitrateVideo> videos;
    bool any_raw = false;
    for (const std::filesystem::path& path : expand_videos(arguments.videos, std::nullopt)) {
        BitrateVideo& video = videos.emplace_back(BitrateVideo{path, std::nullopt, std::nullopt, {}});
        if (is_raw_yuv(path)) {
            video.error = "a raw YUV file has pictures and no encoded packets, so it has no bitrate";
            continue;
        }
        try {
            VideoInfo info = probe_video(path, ffprobe);
            if (info.is_raw_stream()) {
                any_raw = true;
                if (!arguments.fps) {
                    print_error(raw_stream_needs_fps(name_of(path), info));
                    return std::nullopt;
                }
                // Its packets read at that rate (VideoInfo::probe_options); its length and frames are counted from
                // them, not from ffprobe's guess at 25 frames a second.
                info.fps = *arguments.fps;
                info.timed_raw = true;
            }
            video.info = std::move(info);
        } catch (const std::exception& error) {
            video.error = error.what();
        }
    }
    if (arguments.fps && !any_raw) {
        progress.message("--fps is not used: it is for raw streams (.266, .265, .264, .obu), and none of these "
                         "videos is one.");
    }
    return videos;
}

// Reads each video's packets, showing how far it has got.
void read_packets(std::vector<BitrateVideo>& videos, const std::string& ffprobe, Progress& progress) {
    for (std::size_t index = 0; index < videos.size(); ++index) {
        BitrateVideo& video = videos[index];
        const std::string name = name_of(video.path);
        if (!video.info) {
            progress.problem(name + ": not read: " + video.error);
            continue;
        }
        // The frames expected only where the file says (a raw stream's are counted as its packets are read)
        const bool total_known = video.info->nb_frames > 0;
        const auto on_progress = [&](long long done, long long total) {
            progress.job(index, name + ": " + with_thousands(done) + (total_known ? "/" + with_thousands(total) : "") +
                                    " packets read",
                         LineShape{});
        };
        try {
            video.data = analyze_video_bitrate(ffprobe, *video.info, CancelToken(), on_progress);
            progress.finished(index, name + ": " + with_thousands(video.data->frame_count()) + " packets read");
        } catch (const std::exception& error) {
            video.error = error.what();
            progress.problem(name + ": not read: " + video.error);
        }
    }
    progress.done();
}

// Writes `text` to `path` as it is (its lines ended already). Throws
// std::runtime_error saying why it could not be, as the Python app's OSError.
void write_file(const std::filesystem::path& path, const std::string& text) {
    errno = 0;
    std::ofstream file(path, std::ios::binary);
    const int opened = errno;
    file << text;
    file.close();
    if (!file) {
        const int code = opened != 0 ? opened : (errno != 0 ? errno : EIO);
        throw std::runtime_error(python_os_error(std::error_code(code, std::generic_category()), path));
    }
}

// --csv: each read video's files. Whether every one was written.
bool write_csvs(const std::vector<BitrateVideo>& videos, const BitrateArguments& arguments, Progress& progress) {
    const std::filesystem::path folder = python_path(*arguments.csv);
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error) {
        progress.problem("The CSV folder could not be made: " + python_os_error(error, folder));
        return false;
    }
    bool written = true;
    std::set<std::filesystem::path> reserved;
    for (const BitrateVideo& video : videos) {
        if (!video.data) {
            continue;
        }
        try {
            const std::vector<std::filesystem::path> paths =
                write_bitrate_csvs(video, folder, !arguments.no_adjust_start, reserved);
            progress.message(name_of(video.path) + ": bitrate figures in " + platform::path_text(paths.front()) +
                             " and beside it");
        } catch (const std::exception& failure) {
            written = false;
            progress.problem(name_of(video.path) + ": the CSV files could not be written (" + failure.what() + ")");
        }
    }
    return written;
}

}  // namespace

std::string bitrate_table(const std::vector<BitrateVideo>& videos) {
    std::vector<std::vector<std::string>> rows{
        {"File", "Codec", "Duration", "Frames", "Average Mb/s", "Minimum Mb/s", "Maximum Mb/s"}};
    for (const BitrateVideo& video : videos) {
        if (!video.data) {
            continue;
        }
        const BitrateData& data = *video.data;
        rows.push_back({name_of(video.path), codec_label(ascii_lower(video.info->codec_name)),
                        format_hms(data.duration(), 2), with_thousands(data.frame_count()),
                        mbps_text(data.summary.average_kbps), mbps_text(data.summary.minimum_kbps),
                        mbps_text(data.summary.maximum_kbps)});
    }
    if (rows.size() == 1) {
        return {};
    }
    std::vector<std::string> lines;
    add_table(lines, rows, false);
    std::string text;
    for (const std::string& line : lines) {
        text += line + "\n";
    }
    return text;
}

Json bitrate_json(const std::vector<BitrateVideo>& videos) {
    Json entries = Json::array();
    for (const BitrateVideo& video : videos) {
        Json entry{{"path", platform::path_text(video.path)}};
        if (!video.data) {
            entry["error"] = video.error;
            entries.push_back(entry);
            continue;
        }
        const VideoInfo& info = *video.info;
        const BitrateData& data = *video.data;
        entry["codec"] = info.codec_name;
        entry["width"] = info.width;
        entry["height"] = info.height;
        entry["fps"] = info.fps;
        entry["duration_s"] = data.duration();
        entry["frames"] = data.frame_count();
        entry["bytes"] = data.total_bytes();
        entry["average_kbps"] = kbps_value(data.summary.average_kbps);
        entry["minimum_kbps"] = kbps_value(data.summary.minimum_kbps);
        entry["maximum_kbps"] = kbps_value(data.summary.maximum_kbps);
        entries.push_back(entry);
    }
    return Json{{"app", "VideoMetricsLab"}, {"version", VML_VERSION}, {"format", bitrate_format}, {"videos", entries}};
}

std::vector<std::filesystem::path> write_bitrate_csvs(const BitrateVideo& video, const std::filesystem::path& folder,
                                                      bool adjust_start, std::set<std::filesystem::path>& reserved) {
    const BitrateData& data = *video.data;
    const std::vector<std::filesystem::path> paths =
        unique_output_paths(folder, platform::path_text(video.path.stem()),
                            {"_bitrate_frames.csv", "_bitrate_seconds.csv", "_bitrate_gops.csv"}, reserved);
    write_file(paths[0], frames_csv(data, adjust_start));
    write_file(paths[1], seconds_csv(data, adjust_start));
    write_file(paths[2], gops_csv(data, adjust_start));
    return paths;
}

int bitrate(const BitrateArguments& arguments) {
    // Ctrl+C ends the command at once: nothing it makes is left half done but a file being written, and ffprobe
    // ends with this process's job.
    platform::on_interrupt([] { platform::end_process_now(exit_cancelled); });
    const Settings settings = Settings::load();
    const ToolStatus ffprobe = check_tool("ffprobe", settings);
    if (!ffprobe.runnable) {
        print_error("ffprobe cannot be used: " + platform::executable_name(ffprobe.name) + " could not be run (" +
                    (ffprobe.error.empty() ? "not found" : ffprobe.error) +
                    ").\nPut ffprobe on PATH, or set FFmpeg's folder in the app's Settings tab.");
        return exit_usage;
    }
    Progress progress(arguments.quiet);
    std::optional<std::vector<BitrateVideo>> videos = read_videos(arguments, ffprobe.path, progress);
    if (!videos) {
        return exit_usage;
    }
    progress.start();
    read_packets(*videos, ffprobe.path, progress);

    bool ok = std::all_of(videos->begin(), videos->end(),
                          [](const BitrateVideo& video) { return video.data.has_value(); });
    if (arguments.csv && !write_csvs(*videos, arguments, progress)) {
        ok = false;
    }
    const Json document = bitrate_json(*videos);
    if (arguments.json_path && *arguments.json_path == "-") {
        platform::write_text(platform::Stream::output, python_json_dumps(document) + "\n");
    } else {
        platform::write_text(platform::Stream::output, bitrate_table(*videos));
        if (arguments.json_path) {
            const std::optional<std::string> failure = write_json_file(python_path(*arguments.json_path), document);
            if (failure) {
                print_error("The JSON file could not be written: " + *failure);
                ok = false;
            }
        }
    }
    return ok ? exit_ok : exit_failed;
}

}  // namespace vml::cli
