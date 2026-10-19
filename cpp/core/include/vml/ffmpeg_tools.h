// Finding FFmpeg and ffprobe, and checking that they run and are new enough.
// A port of vmaf_app/core/ffmpeg_locate.py.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/settings.h"

namespace vml {

// The stream every command reads from a file: the first video stream that
// is not a picture. FFmpeg lists cover art as a video stream too ("v:0"
// could be the cover); "V" leaves pictures out.
inline constexpr std::string_view video_stream = "V:0";

// A video's path for FFmpeg's -i: absolute, as Python's Path.resolve()
// makes it, because FFmpeg may run in a folder of its own (a plain
// argument, so a Windows path with its drive works).
std::string input_path(const std::filesystem::path& path);

// The oldest FFmpeg the app's filter graphs work with.
inline const std::vector<int> minimum_ffmpeg_version{9};

// The program to run for `name` ("ffmpeg" or "ffprobe"): in the folder chosen
// in Settings, then on the PATH, then in common install folders (a newly
// installed winget package is not on the PATH of programs already running;
// on macOS, Homebrew's folder is not on the PATH of an app the Finder
// opens); else the bare name, in case it is on the PATH by the time it runs.
std::string find_tool(std::string_view name, const Settings& settings);

// On macOS, the folders find_tool looks in after the PATH, separated by ':',
// instead of Homebrew's and MacPorts': for tests, which need a Mac where
// FFmpeg is missing.
inline constexpr std::string_view ffmpeg_folders_variable = "VML_FFMPEG_FOLDERS";

// The version in a tool's -version output, such as {9, 0, 1}; nothing for a
// development build ("N-113411-g1234567"), which has no usable number.
std::optional<std::vector<int>> parse_version(std::string_view version_output);

// A version as text, such as "9.0.1", or "unknown".
std::string format_version(const std::optional<std::vector<int>>& version);

// The result of checking one tool.
struct ToolStatus {
    std::string name;
    std::string path;
    bool runnable = false;
    std::optional<std::vector<int>> version;
    std::string error;
};

// The result of checking FFmpeg and ffprobe.
struct ToolsStatus {
    ToolStatus ffmpeg;
    ToolStatus ffprobe;

    // Why the app cannot run, for the user; empty if nothing is wrong.
    std::vector<std::string> problems() const;
    bool ok() const { return problems().empty(); }
};

// Runs `<tool> -version`, the only way to know that the tool is there and
// runs.
ToolStatus check_tool(std::string_view name, const Settings& settings);
ToolsStatus check_tools(const Settings& settings);

}  // namespace vml
