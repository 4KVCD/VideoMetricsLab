// The app's log file (the Python app's core/app_log.py): what ran, how it
// went, and every failure in full, so that a failure during an overnight run
// can still be understood after the app is restarted (the window's tooltips
// are gone by then). It is written to <user data>/logs/VideoMetricsLab.log,
// in the Python app's form, and rotated at 5 MB with four older files kept.
//
// Only the app turns it on (start_logging); the command line writes its
// lines to standard error (log_to_console), and tests log nowhere. The
// scoring process the app starts (isolated.h) hands its lines to the app,
// which writes them.
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vml::app_log {

inline constexpr std::string_view log_file_name = "VideoMetricsLab.log";
// Where a crash is reported: the process is gone before anything could
// reach the main log.
inline constexpr std::string_view crash_file_name = "native-crashes.log";
// What ends the line opening each session, and starts the line opening
// each run.
inline constexpr std::string_view session_start = " starting ====";
inline constexpr std::string_view run_start = "Run started:";
// The most text "Copy log" puts on the clipboard: about what a forum post or
// chat message takes. A session's log is usually a few kilobytes (a run of
// four films is about 30 KB), so only an unusually long one is cut.
inline constexpr std::size_t share_limit = 60'000;

// A line's level, numbered as Python's logging numbers them. Lines below
// INFO are not written, as in the Python app.
enum class Level { debug = 10, info = 20, warning = 30, error = 40, critical = 50 };

// The folder the log files are in: <user data>/logs.
std::filesystem::path log_dir();

// Starts writing the log (in `directory`, by default log_dir()), and names
// the calling thread "MainThread". Returns the log file's path, or nothing
// when it could not be opened: the app runs the same without a log.
std::optional<std::filesystem::path> start_logging(const std::optional<std::filesystem::path>& directory = {});
// Undoes start_logging (for tests).
void stop_logging();

// Hands each line to `forward` rather than writing it: how the scoring
// process gives its lines to the app. An empty function stops it.
using Forward = std::function<void(int level, std::string_view name, std::string_view text)>;
void forward_lines(Forward forward);

// Writes each line at `level` or above to standard error as well, in the
// command line's form ("<date> <time>,<ms> <LEVEL> <name>: <text>", the
// Python app's logging.basicConfig): its failures always, everything with
// --verbose.
void log_to_console(Level level);

// Whether a line at `level` goes anywhere, for a caller to skip putting
// together a line no one keeps.
bool enabled(Level level);

// A line from the part of the app called `name` (Python's logger names, with
// vml for vmaf_app: "vml.job_runner"). `text` may run over several lines.
void write(Level level, std::string_view name, std::string_view text);
// The same with Python's level number (a line from the scoring process).
void write(int level, std::string_view name, std::string_view text);

// Names the calling thread in the lines it writes, as Python names its
// threads ("vmaf-cpu-0"). A thread not named is "Thread-" and a number.
void set_thread_name(std::string name);

// One part of the app's log (Python's logging.getLogger(__name__)).
class Logger {
public:
    explicit constexpr Logger(std::string_view name) : name_(name) {}

    void info(std::string_view text) const { write(Level::info, name_, text); }
    void warning(std::string_view text) const { write(Level::warning, name_, text); }
    void error(std::string_view text) const { write(Level::error, name_, text); }
    void critical(std::string_view text) const { write(Level::critical, name_, text); }

private:
    std::string_view name_;
};

// The log files that have anything in them (the current log, its older
// rotations and the crash report), sorted by name, for Export log to put in
// a zip; none when there is no log yet.
std::vector<std::filesystem::path> log_files(const std::optional<std::filesystem::path>& directory = {});

// Zips those files (log_files) into `destination`, to attach to a report,
// and returns their names; when there is no log yet, nothing is written and
// the list is empty. Throws std::runtime_error when a file cannot be read
// or the zip cannot be written.
std::vector<std::string> export_logs(const std::filesystem::path& destination,
                                     const std::optional<std::filesystem::path>& directory = {});

// The part of the log to paste into a post or chat, and whether it was
// shortened; nothing when there is no log yet. Throws std::runtime_error
// when a log file there cannot be read.
//
// The text runs from the start of the latest session that calculated
// metrics to the end of the log, so a run that failed before the app was
// restarted is still included. If that is longer than `limit` characters,
// the session's first lines (what the app ran on) are kept with as much of
// the end as fits, and a line says how many lines were left out. Export log
// saves all of it.
std::optional<std::pair<std::string, bool>> log_text_to_share(const std::optional<std::filesystem::path>& directory = {},
                                                              std::size_t limit = share_limit);
// The same, of the log's lines.
std::pair<std::string, bool> text_to_share(const std::vector<std::string>& lines, std::size_t limit = share_limit);

// What this session runs on, for the top of each session's log: the app and
// `version`, the system and the compiler, the CPU and the memory.
std::vector<std::string> environment_lines(std::string_view version);

}  // namespace vml::app_log
