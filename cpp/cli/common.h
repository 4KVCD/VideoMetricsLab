// What the command line's commands share (compare.h, bitrate.h): their progress on stderr, errors, JSON and
// text files, tables, and how raw inputs are read.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "vml/platform/console.h"
#include "vml/python_text.h"
#include "vml/run_line.h"
#include "vml/video.h"

namespace vml::cli {

// Writes `text` and a line end to stderr.
void print_error(const std::string& text);

// `text` with its lines ended as Python writes a text file on this system:
// CR LF on Windows.
std::string text_file(std::string text);

// Writes `document` to `path`, its folder made first, as the Python app
// writes it. Why it could not be written, as the Python app's OSError says
// it (the path alone left a user guessing), or nothing.
std::optional<std::string> write_json_file(const std::filesystem::path& path, const nlohmann::ordered_json& document);

// A whole number with commas between thousands ("28,800").
std::string with_thousands(long long number);

// Table rows as lines: the first column left-justified, the others right,
// the last as it is, two spaces between, trailing spaces cut.
void add_table(std::vector<std::string>& lines, const std::vector<std::vector<std::string>>& rows,
               bool last_column_free);

// Whether `path` is a raw YUV file (.yuv): pictures and nothing else, which
// FFmpeg reads only when told their size, pixel format and frame rate.
bool is_raw_yuv(const std::filesystem::path& path);

// Why a raw stream (VideoInfo::is_raw_stream: a .266, .265, .264 or .obu file) cannot be read without --fps:
// FFmpeg and ffprobe would read it at 25 frames a second. `which` names the video, from its first word.
std::string raw_stream_needs_fps(const std::string& which, const VideoInfo& info);

// The run's progress on stderr, nothing with --quiet: lines that stay, and
// a line for each video. On a terminal the videos share one line, redrawn
// in place as the window draws its lines: at once when what it says apart
// from its figures changes (line_shape), and with new figures once a
// second, the videos with new figures taking the line in turn. Elsewhere
// each video's line is written out, at most every ten seconds.
class Progress {
    using Clock = std::chrono::steady_clock;

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

}  // namespace vml::cli
