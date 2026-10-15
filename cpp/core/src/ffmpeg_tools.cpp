#include "vml/ffmpeg_tools.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <regex>

#include "vml/platform/process.h"
#include "vml/platform/system.h"

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

// The common install folders to look in (Windows): winget's FFmpeg
// packages, then C:\ffmpeg\bin and C:\Program Files\ffmpeg\bin.
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
#endif
    return candidates;
}

bool is_file(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::exists(path, error);
}

std::string strip(std::string text) {
    const std::size_t start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return {};
    }
    return text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1);
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

ToolStatus check_tool(std::string_view name, const Settings& settings) {
    ToolStatus status;
    status.name = name;
    status.path = find_tool(name, settings);
    platform::ProcessOutput output;
    try {
        output = platform::run_program({status.path, "-version"}, std::chrono::seconds(15));
    } catch (const platform::ProcessError& error) {
        status.error = error.what();
        return status;
    }
    if (output.timed_out) {
        status.error = "timed out after 15 seconds";
        return status;
    }
    if (output.exit_code != 0) {
        std::string said = strip(output.standard_error.empty() ? output.standard_output : output.standard_error);
        said = said.substr(0, 300);
        status.error = said.empty() ? "exited with code " + platform::exit_code_text(output.exit_code) : said;
        return status;
    }
    status.runnable = true;
    status.version = parse_version(output.standard_output);
    return status;
}

ToolsStatus check_tools(const Settings& settings) {
    return ToolsStatus{check_tool("ffmpeg", settings), check_tool("ffprobe", settings)};
}

}  // namespace vml
