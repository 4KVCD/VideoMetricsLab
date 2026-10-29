#include "vml/ffmpeg_tools.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <future>
#include <regex>
#include <string>
#include <utility>

#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

#ifdef _WIN32
// Whether `name` matches `pattern`, whose only wildcard is "*" (any run of
// characters), ignoring case as Windows does.
bool matches(std::string_view name, std::string_view pattern) {
    if (pattern.empty()) {
        return name.empty();
    }
    if (pattern.front() == '*') {
        for (std::size_t skipped = 0; skipped <= name.size(); ++skipped) {
            if (matches(name.substr(skipped), pattern.substr(1))) {
                return true;
            }
        }
        return false;
    }
    return !name.empty() &&
           std::tolower(static_cast<unsigned char>(name.front())) ==
               std::tolower(static_cast<unsigned char>(pattern.front())) &&
           matches(name.substr(1), pattern.substr(1));
}

// The folders inside `folder` whose names match `pattern`, by name.
std::vector<std::filesystem::path> matching_folders(const std::filesystem::path& folder, const std::string& pattern) {
    std::vector<std::filesystem::path> found;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
        if (entry.is_directory(error) && matches(platform::path_text(entry.path().filename()), pattern)) {
            found.push_back(entry.path());
        }
    }
    std::sort(found.begin(), found.end());
    return found;
}
#endif

// The common install folders to look in. Windows: winget's FFmpeg
// packages, then C:\ffmpeg\bin and C:\Program Files\ffmpeg\bin. macOS:
// Homebrew's (Apple silicon, then Intel) and MacPorts' -- an app opened
// from the Finder does not have them on its PATH.
std::vector<std::filesystem::path> candidate_folders() {
    std::vector<std::filesystem::path> candidates;
#ifdef _WIN32
    const std::filesystem::path packages =
        platform::home_directory() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages";
    for (const auto& gyan : matching_folders(packages, "Gyan.FFmpeg*")) {
        for (const auto& build : matching_folders(gyan, "ffmpeg-*-full_build")) {
            candidates.push_back(build / "bin");
        }
    }
    for (const auto& btbn : matching_folders(packages, "BtbN.FFmpeg*")) {
        candidates.push_back(btbn / "bin");
    }
    candidates.emplace_back(L"C:\\ffmpeg\\bin");
    candidates.emplace_back(L"C:\\Program Files\\ffmpeg\\bin");
#elif defined(__APPLE__)
    if (const std::optional<std::string> named = platform::environment_variable(ffmpeg_folders_variable)) {
        std::string_view rest = *named;
        while (!rest.empty()) {
            const std::size_t end = std::min(rest.find(':'), rest.size());
            if (end > 0) {
                candidates.emplace_back(std::string(rest.substr(0, end)));
            }
            rest.remove_prefix(std::min(end + 1, rest.size()));
        }
        return candidates;
    }
    candidates.emplace_back("/opt/homebrew/bin");
    candidates.emplace_back("/usr/local/bin");
    candidates.emplace_back("/opt/local/bin");
#endif
    return candidates;
}

bool is_file(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

}  // namespace

std::string input_path(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(std::filesystem::absolute(path), error);
    return platform::path_text(error ? std::filesystem::absolute(path) : resolved);
}

std::string find_tool(std::string_view name, const Settings& settings) {
    const std::string program = platform::executable_name(name);
    if (const std::optional<std::filesystem::path> folder = settings.ffmpeg_dir_path()) {
        if (const std::filesystem::path candidate = *folder / platform::path_from_text(program); is_file(candidate)) {
            return platform::path_text(candidate);
        }
    }
    if (const std::optional<std::filesystem::path> on_path = platform::find_on_path(name)) {
        return platform::path_text(*on_path);
    }
    for (const std::filesystem::path& folder : candidate_folders()) {
        if (const std::filesystem::path candidate = folder / platform::path_from_text(program); is_file(candidate)) {
            return platform::path_text(candidate);
        }
    }
    return std::string(name);
}

std::optional<std::vector<int>> parse_version(std::string_view version_output) {
    // The version line both tools print, such as "ffmpeg version
    // 9.0.1-full_build-www.gyan.dev" or "ffprobe version n7.1".
    static const std::regex version(R"(\b(?:ffmpeg|ffprobe) version n?(\d+)(?:\.(\d+))?(?:\.(\d+))?)");
    std::match_results<std::string_view::const_iterator> match;
    if (!std::regex_search(version_output.begin(), version_output.end(), match, version)) {
        return std::nullopt;
    }
    std::vector<int> parts;
    for (std::size_t group = 1; group < match.size(); ++group) {
        if (match[group].matched) {
            parts.push_back(std::stoi(match[group].str()));
        }
    }
    return parts;
}

std::string format_version(const std::optional<std::vector<int>>& version) {
    if (!version || version->empty()) {
        return "unknown";
    }
    std::string text;
    for (const int part : *version) {
        text += (text.empty() ? "" : ".") + std::to_string(part);
    }
    return text;
}

std::vector<std::string> ToolsStatus::problems() const {
    std::vector<std::string> issues;
    for (const ToolStatus* tool : {&ffmpeg, &ffprobe}) {
        if (!tool->runnable) {
            issues.push_back(platform::executable_name(tool->name) + " could not be run (" +
                             (tool->error.empty() ? "not found" : tool->error) + ").");
        }
    }
    if (ffmpeg.runnable && ffmpeg.version && *ffmpeg.version < minimum_ffmpeg_version) {
        issues.push_back("ffmpeg " + format_version(ffmpeg.version) + " is too old -- version " +
                         format_version(minimum_ffmpeg_version) + " or newer is required.");
    }
    return issues;
}

// How long `<tool> -version` may take. It takes tens of milliseconds, even
// with every core busy, but on a PC busy with encodes and comparisons a
// start of ffprobe has taken over 15 seconds. FFmpeg and ffprobe are checked
// at once, so a tool that hangs holds up what waits for the check (the
// window as it opens) no longer than the two checked in turn at 15 s did.
constexpr std::chrono::seconds tool_check_timeout{30};

ToolStatus check_tool(std::string_view name, const Settings& settings) {
    ToolStatus status;
    status.name = name;
    status.path = find_tool(name, settings);
    platform::ProcessOutput output;
    try {
        output = platform::run_program({status.path, "-version"}, tool_check_timeout);
    } catch (const platform::ProcessError& error) {
        status.error = error.what();
        return status;
    }
    if (output.timed_out) {
        status.error = "timed out after " + std::to_string(tool_check_timeout.count()) + " seconds";
        return status;
    }
    if (output.exit_code != 0) {
        const std::string said(
            python_strip(output.standard_error.empty() ? output.standard_output : output.standard_error).substr(0, 300));
        status.error = said.empty() ? "exited with code " + platform::exit_code_text(output.exit_code) : said;
        return status;
    }
    status.runnable = true;
    status.version = parse_version(output.standard_output);
    return status;
}

ToolsStatus check_tools(const Settings& settings) {
    // Both at once: the window waits for them as it opens, and each may take
    // up to tool_check_timeout.
    std::future<ToolStatus> ffprobe =
        std::async(std::launch::async, [&settings] { return check_tool("ffprobe", settings); });
    ToolStatus ffmpeg = check_tool("ffmpeg", settings);
    return ToolsStatus{std::move(ffmpeg), ffprobe.get()};
}

}  // namespace vml
