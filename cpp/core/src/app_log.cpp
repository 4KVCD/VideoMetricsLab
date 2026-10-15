#include "vml/app_log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>

#include <miniz.h>

#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml::app_log {

namespace {

// Rotated before a line would take the file to 5 MB, with four older files
// kept: 25 MB at most.
constexpr std::uintmax_t max_bytes = 5 * 1024 * 1024;
constexpr int backups = 4;

// Each line ends as a text file's lines do on the system, as Python writes
// them.
#ifdef _WIN32
constexpr std::string_view newline = "\r\n";
#else
constexpr std::string_view newline = "\n";
#endif

constexpr int info = static_cast<int>(Level::info);

struct State {
    std::mutex mutex;
    std::ofstream file;  // the log, while it is written
    std::filesystem::path path;
    std::uintmax_t size = 0;  // the log's length in bytes
    Forward forward;
    std::optional<int> console;  // the lowest level written to standard error
};

State& state() {
    static State instance;
    return instance;
}

// The lowest level a line must have to go anywhere, read without the lock
// by enabled(); above every level while lines go nowhere.
constexpr int nowhere = std::numeric_limits<int>::max();
std::atomic<int> threshold{nowhere};

// threshold, after a change to where lines go; with the lock held.
void update_threshold(const State& log) {
    int lowest = log.forward || log.file.is_open() ? info : nowhere;
    if (log.console) {
        lowest = std::min(lowest, *log.console);
    }
    threshold = lowest;
}

// The calling thread's name, in plain characters, with nothing to destroy
// when the thread ends: MinGW's emulated thread-local storage corrupted the
// heap as threads holding a std::string here ended (6 runs in 20 of a test
// whose lanes end at once).
constexpr std::size_t name_room = 64;
thread_local std::array<char, name_room> thread_name{};
std::atomic<int> unnamed_threads{0};

void name_this_thread(std::string_view name) {
    const std::size_t length = std::min(name.size(), name_room - 1);
    std::copy_n(name.data(), length, thread_name.data());
    thread_name[length] = '\0';
}

std::string current_thread_name() {
    if (thread_name[0] == '\0') {
        name_this_thread("Thread-" + std::to_string(++unnamed_threads));
    }
    return thread_name.data();
}

std::string level_name(int level) {
    switch (level) {
        case 10: return "DEBUG";
        case 20: return "INFO";
        case 30: return "WARNING";
        case 40: return "ERROR";
        case 50: return "CRITICAL";
        default: return "Level " + std::to_string(level);
    }
}

// The local time now, as "2026-10-14 10:21:36" and its milliseconds.
std::pair<std::string, int> timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    const std::tm local = platform::local_time(static_cast<std::time_t>(since_epoch / 1000));
    char stamp[64];
    std::snprintf(stamp, sizeof stamp, "%04d-%02d-%02d %02d:%02d:%02d", local.tm_year + 1900, local.tm_mon + 1,
                  local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec);
    return {stamp, static_cast<int>(since_epoch % 1000)};
}

std::string three_digits(int milliseconds) {
    char digits[8];
    std::snprintf(digits, sizeof digits, "%03d", milliseconds);
    return digits;
}

// "2026-10-14 10:21:36.123 INFO    [MainThread] vml.job_runner: text", as
// the Python app's logging writes a line to its file ("%(asctime)s.%(msecs)03d
// %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"), the line breaks
// in `text` the system's own.
std::string file_line(int level, std::string_view name, std::string_view text) {
    const auto [stamp, milliseconds] = timestamp();
    std::string level_text = level_name(level);
    if (level_text.size() < 7) {
        level_text.resize(7, ' ');
    }
    std::string line = stamp + "." + three_digits(milliseconds) + " " + level_text + " [" + current_thread_name() +
                       "] " + std::string(name) + ": ";
    for (const char c : text) {
        if (c == '\n') {
            line += newline;
        } else {
            line += c;
        }
    }
    line += newline;
    return line;
}

// "2026-10-14 10:21:36,123 ERROR vml.job_runner: text", as the command line's
// logging.basicConfig writes a line to standard error ("%(asctime)s
// %(levelname)s %(name)s: %(message)s").
std::string console_line(int level, std::string_view name, std::string_view text) {
    const auto [stamp, milliseconds] = timestamp();
    return stamp + "," + three_digits(milliseconds) + " " + level_name(level) + " " + std::string(name) + ": " +
           std::string(text) + "\n";
}

std::filesystem::path numbered(const std::filesystem::path& path, int number) {
    std::filesystem::path rotated = path;
    rotated += "." + std::to_string(number);
    return rotated;
}

// Python's RotatingFileHandler.doRollover: each older file one number up,
// the oldest dropped, the log the first of them, and a new log started. A
// file another program holds stays where it is, and the log goes on in it.
void rotate(State& log) {
    log.file.close();
    std::error_code error;
    for (int number = backups - 1; number > 0; --number) {
        const std::filesystem::path from = numbered(log.path, number);
        if (std::filesystem::exists(from, error)) {
            const std::filesystem::path to = numbered(log.path, number + 1);
            std::filesystem::remove(to, error);
            std::filesystem::rename(from, to, error);
        }
    }
    const std::filesystem::path first = numbered(log.path, 1);
    std::filesystem::remove(first, error);
    std::filesystem::rename(log.path, first, error);
    log.file.open(log.path, std::ios::binary | std::ios::app);
    const std::uintmax_t size = std::filesystem::file_size(log.path, error);
    log.size = error ? 0 : size;
}

void write_line(int level, std::string_view name, std::string_view text) {
    State& log = state();
    const std::lock_guard lock(log.mutex);
    if (log.forward) {
        if (level >= info) {
            log.forward(level, name, text);
        }
        return;
    }
    if (log.console && level >= *log.console) {
        platform::write_text(platform::Stream::error, console_line(level, name, text));
    }
    if (!log.file.is_open() || level < info) {
        return;
    }
    const std::string line = file_line(level, name, text);
    // As Python's shouldRollover: never an empty file.
    if (log.size > 0 && log.size + line.size() >= max_bytes) {
        rotate(log);
    }
    log.file.write(line.data(), static_cast<std::streamsize>(line.size()));
    log.file.flush();
    log.size += line.size();
}

std::string compiler() {
#if defined(_MSC_VER) && !defined(__clang__)
    return "MSVC " + std::to_string(_MSC_VER / 100) + "." + std::to_string(_MSC_VER % 100);
#elif defined(__apple_build_version__)
    return "Apple Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    return "an unknown compiler";
#endif
}

}  // namespace

std::filesystem::path log_dir() { return platform::home_directory() / ".videometricslab" / "logs"; }

std::optional<std::filesystem::path> start_logging(const std::optional<std::filesystem::path>& directory) {
    State& log = state();
    const std::lock_guard lock(log.mutex);
    if (log.file.is_open()) {
        return log.path;
    }
    const std::filesystem::path folder = directory.value_or(log_dir());
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    log.path = folder / log_file_name;
    log.file.open(log.path, std::ios::binary | std::ios::app);
    if (!log.file.is_open()) {
        return std::nullopt;
    }
    const std::uintmax_t size = std::filesystem::file_size(log.path, error);
    log.size = error ? 0 : size;
    name_this_thread("MainThread");
    update_threshold(log);
    return log.path;
}

void stop_logging() {
    State& log = state();
    const std::lock_guard lock(log.mutex);
    log.file.close();
    update_threshold(log);
}

void forward_lines(Forward forward) {
    State& log = state();
    const std::lock_guard lock(log.mutex);
    log.forward = std::move(forward);
    update_threshold(log);
}

void log_to_console(Level level) {
    State& log = state();
    const std::lock_guard lock(log.mutex);
    log.console = static_cast<int>(level);
    update_threshold(log);
}

bool enabled(Level level) { return static_cast<int>(level) >= threshold; }

void write(Level level, std::string_view name, std::string_view text) { write(static_cast<int>(level), name, text); }

void write(int level, std::string_view name, std::string_view text) {
    if (level >= threshold) {
        write_line(level, name, text);
    }
}

void set_thread_name(std::string name) { name_this_thread(name); }

std::vector<std::filesystem::path> log_files(const std::optional<std::filesystem::path>& directory) {
    // As the Python app's glob("*.log*"): the log, its rotations, the crash
    // report.
    const std::filesystem::path folder = directory.value_or(log_dir());
    std::vector<std::filesystem::path> files;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
        const std::string name = platform::path_text(entry.path().filename());
        std::error_code size_error;
        if (name.find(".log") != std::string::npos && entry.is_regular_file(size_error) &&
            entry.file_size(size_error) > 0 && !size_error) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> export_logs(const std::filesystem::path& destination,
                                     const std::optional<std::filesystem::path>& directory) {
    const std::vector<std::filesystem::path> files = log_files(directory);
    if (files.empty()) {
        return {};
    }
    mz_zip_archive zip{};
    if (!mz_zip_writer_init_heap(&zip, 0, 0)) {
        throw std::runtime_error("the zip file could not be made");
    }
    std::vector<std::string> names;
    std::string failure;
    for (const std::filesystem::path& file : files) {
        const std::string name = platform::path_text(file.filename());
        const std::optional<std::string> bytes = platform::read_file(file);
        if (!bytes) {
            failure = platform::path_text(file) + " could not be read";
            break;
        }
        // Deflated, as the Python app's zipfile.ZIP_DEFLATED.
        if (!mz_zip_writer_add_mem(&zip, name.c_str(), bytes->data(), bytes->size(), MZ_DEFAULT_LEVEL)) {
            failure = name + " could not be added to the zip file";
            break;
        }
        names.push_back(name);
    }
    void* archive = nullptr;
    std::size_t size = 0;
    if (failure.empty() && !mz_zip_writer_finalize_heap_archive(&zip, &archive, &size)) {
        failure = "the zip file could not be made";
    }
    if (failure.empty()) {
        std::ofstream out(destination, std::ios::binary | std::ios::trunc);
        out.write(static_cast<const char*>(archive), static_cast<std::streamsize>(size));
        if (!out) {
            failure = platform::path_text(destination) + " could not be written";
        }
    }
    mz_zip_writer_end(&zip);
    mz_free(archive);
    if (!failure.empty()) {
        throw std::runtime_error(failure);
    }
    return names;
}

std::optional<std::pair<std::string, bool>> log_text_to_share(const std::optional<std::filesystem::path>& directory,
                                                              std::size_t limit) {
    const std::filesystem::path folder = directory.value_or(log_dir());
    const std::filesystem::path current = folder / log_file_name;
    std::vector<std::filesystem::path> files;
    for (int number = backups; number > 0; --number) {  // the oldest first
        files.push_back(numbered(current, number));
    }
    files.push_back(current);
    std::vector<std::string> lines;
    for (const std::filesystem::path& file : files) {
        std::error_code ignored;
        if (!std::filesystem::is_regular_file(file, ignored)) {
            continue;
        }
        const std::optional<std::string> content = platform::read_file(file);
        if (!content) {
            throw std::runtime_error(platform::path_text(file) + " could not be read");
        }
        const std::string text = replace_invalid_utf8(*content);
        for (const std::string_view line : python_splitlines(text)) {
            lines.emplace_back(line);
        }
    }
    if (lines.empty()) {
        return std::nullopt;
    }
    return text_to_share(lines, limit);
}

std::pair<std::string, bool> text_to_share(const std::vector<std::string>& all_lines, std::size_t limit) {
    const auto joined = [](auto begin, auto end) {
        std::string text;
        for (auto line = begin; line != end; ++line) {
            if (line != begin) {
                text += '\n';
            }
            text += *line;
        }
        return text;
    };
    if (all_lines.empty()) {
        return {std::string(), false};
    }
    // From the session that started the latest run (or, with no run, the
    // latest line's session).
    std::size_t last_run = all_lines.size() - 1;
    for (std::size_t i = all_lines.size(); i-- > 0;) {
        if (all_lines[i].find(run_start) != std::string::npos) {
            last_run = i;
            break;
        }
    }
    std::size_t start = 0;
    for (std::size_t i = last_run + 1; i-- > 0;) {
        if (all_lines[i].ends_with(session_start)) {
            start = i;
            break;
        }
    }
    const std::vector<std::string> lines(all_lines.begin() + static_cast<std::ptrdiff_t>(start), all_lines.end());
    std::string text = joined(lines.begin(), lines.end());
    if (python_length(text) <= limit) {
        return {std::move(text), false};
    }
    // The session's first lines (what the app ran on: version, system, CPU,
    // FFmpeg, GPUs), within a quarter of the limit.
    std::size_t head = 0;
    while (head < std::min<std::size_t>(lines.size(), 12) &&
           python_length(joined(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(head) + 1)) <= limit / 4) {
        ++head;
    }
    // Then as many of the last lines as fit.
    const std::size_t head_length = python_length(joined(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(head)));
    long long budget = static_cast<long long>(limit) - static_cast<long long>(head_length) - 120;
    std::size_t tail_start = lines.size();
    while (tail_start > head) {
        const auto length = static_cast<long long>(python_length(lines[tail_start - 1])) + 1;
        if (length > budget) {
            break;
        }
        budget -= length;
        --tail_start;
    }
    const std::size_t left_out = tail_start - head;
    std::vector<std::string> kept(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(head));
    kept.push_back("... " + std::to_string(left_out) + " lines left out here; Export log in Settings saves all of it ...");
    kept.insert(kept.end(), lines.begin() + static_cast<std::ptrdiff_t>(tail_start), lines.end());
    return {joined(kept.begin(), kept.end()), true};
}

std::vector<std::string> environment_lines(std::string_view version) {
    std::string processor = platform::processor_description();
    std::vector<std::string> lines{
        "VideoMetricsLab " + std::string(version),
        platform::system_description() + ", built with " + compiler(),
        "CPU: " + (processor.empty() ? std::string("unknown") : processor) + ", " +
            std::to_string(platform::logical_cores()) + " logical processors",
    };
    if (const std::optional<unsigned long long> memory = platform::total_memory()) {
        // Whole gigabytes, a half to the even one as Python's format rounds.
        const double gigabytes = static_cast<double>(*memory) / (1024.0 * 1024.0 * 1024.0);
        lines.push_back("Memory: " + std::to_string(static_cast<long long>(std::nearbyint(gigabytes))) + " GB");
    }
    return lines;
}

}  // namespace vml::app_log
