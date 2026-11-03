// The scores FFmpeg writes: libvmaf's JSON log, read as it streams past,
// and xpsnr's stats file, as one run's per-frame scores (ParsedScores), and
// a ranged run's cut to its range.
#include "ffmpeg_run_parts.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <map>
#include <regex>

#include "vml/ffmpeg_run.h"
#include "vml/metrics.h"
#include "vml/python_text.h"

namespace vml {

namespace {

using namespace run_parts;

// Reads libvmaf's JSON log as it streams past, keeping only the per-frame
// scores the app uses. A feature-length log runs to hundreds of megabytes,
// and a parsed copy of the whole of it to gigabytes; this keeps four bytes
// per score. It reads what the Python app's _parse_log reads: the root's
// "frames", each frame's "frameNum" and its "metrics", the last of a key
// given twice.
class LogReader final : public nlohmann::json::json_sax_t {
public:
    explicit LogReader(const std::map<int, double>& xpsnr) : xpsnr_(xpsnr) {}

    std::vector<int> frames;
    std::array<std::vector<float>, column_count> columns;
    std::array<bool, column_count> scored{};  // whether any frame has a score in the column

    bool null() override { return value(std::nullopt); }
    bool boolean(bool) override { return value(std::nullopt); }
    bool number_integer(number_integer_t number) override { return value(static_cast<double>(number)); }
    bool number_unsigned(number_unsigned_t number) override { return value(static_cast<double>(number)); }
    bool number_float(number_float_t number, const string_t&) override { return value(number); }
    bool string(string_t&) override { return value(std::nullopt); }
    bool binary(binary_t&) override { return value(std::nullopt); }

    bool key(string_t& name) override {
        next_ = Next::other;
        switch (open_.empty() ? Place::other : open_.back()) {
            case Place::root:
                next_ = name == "frames" ? Next::frames : Next::other;
                break;
            case Place::frame:
                if (name == "frameNum") {
                    next_ = Next::frame_number;
                } else if (name == "metrics") {
                    next_ = Next::metrics;
                    frame_.metrics_is_object = false;  // until it is seen to be one
                    frame_.present.fill(false);
                    frame_.number_of.fill(std::nullopt);
                }
                break;
            case Place::metrics:
                metric_ = index_of(name);
                if (metric_) {
                    frame_.present[*metric_] = true;
                    frame_.number_of[*metric_].reset();
                }
                break;
            default:
                break;
        }
        return true;
    }

    bool start_object(std::size_t) override {
        Place place = Place::other;
        if (open_.empty()) {
            place = Place::root;
        } else if (open_.back() == Place::frames) {
            place = Place::frame;
            frame_ = Frame{};
        } else if (open_.back() == Place::frame && next_ == Next::metrics) {
            place = Place::metrics;
            frame_.metrics_is_object = true;
        }
        open_.push_back(place);
        next_ = Next::other;
        return true;
    }

    bool start_array(std::size_t) override {
        const bool frames_array = !open_.empty() && open_.back() == Place::root && next_ == Next::frames;
        if (frames_array) {
            // The last "frames" is the one read, as by a parser that keeps the
            // last of a key given twice.
            frames.clear();
            for (std::vector<float>& column : columns) {
                column.clear();
            }
            scored.fill(false);
        }
        open_.push_back(frames_array ? Place::frames : Place::other);
        next_ = Next::other;
        return true;
    }

    bool end_object() override { return close(); }
    bool end_array() override { return close(); }

    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& error) override {
        throw std::runtime_error(error.what());
    }

private:
    // What an open object or array is.
    enum class Place { root, frames, frame, metrics, other };
    // What the value after a key is.
    enum class Next { other, frames, frame_number, metrics };

    // The metric keys whose presence and numbers a frame is read for.
    static constexpr std::array<const char*, 7> metric_keys{"vmaf",   "vmaf_neg",   "vmaf_v1", "psnr_y",
                                                            "psnr",   "float_ssim", "ssim"};
    enum Metric { vmaf, vmaf_neg, vmaf_v1, psnr_y, psnr, float_ssim, ssim };

    static std::optional<std::size_t> index_of(const std::string& name) {
        for (std::size_t i = 0; i < metric_keys.size(); ++i) {
            if (name == metric_keys[i]) {
                return i;
            }
        }
        return std::nullopt;
    }

    struct Frame {
        std::optional<int> number;
        bool metrics_is_object = false;
        std::array<bool, metric_keys.size()> present{};
        std::array<std::optional<double>, metric_keys.size()> number_of{};
    };

    bool value(std::optional<double> number) {
        constexpr double int_min = std::numeric_limits<int>::min();
        constexpr double int_max = std::numeric_limits<int>::max();
        if (!open_.empty() && open_.back() == Place::frame && next_ == Next::frame_number && number &&
            *number >= int_min && *number <= int_max) {
            frame_.number = static_cast<int>(*number);
        } else if (!open_.empty() && open_.back() == Place::metrics && metric_ && number) {
            frame_.number_of[*metric_] = number;
        }
        next_ = Next::other;
        return true;
    }

    bool close() {
        const Place closed = open_.back();
        open_.pop_back();
        if (closed == Place::frame) {
            finish_frame();
        }
        return true;
    }

    // Keeps a frame's scores, if it has any of the metrics the app reads.
    void finish_frame() {
        const bool has_any =
            frame_.metrics_is_object && std::any_of(frame_.present.begin(), frame_.present.end(), [](bool p) { return p; });
        if (!has_any) {
            return;
        }
        const int frame_number = frame_.number.value_or(static_cast<int>(frames.size()));
        // A real 0.0 is a score: "psnr" is read only where "psnr_y" has no
        // number, not where it is zero.
        const auto first = [this](Metric a, Metric b) {
            return frame_.number_of[a] ? frame_.number_of[a] : frame_.number_of[b];
        };
        const auto at = xpsnr_.find(frame_number);
        const std::array<std::optional<double>, column_count> values{
            frame_.number_of[vmaf], first(psnr_y, psnr), first(float_ssim, ssim),
            at == xpsnr_.end() ? std::nullopt : std::optional<double>(at->second), frame_.number_of[vmaf_neg],
            frame_.number_of[vmaf_v1]};
        frames.push_back(frame_number);
        for (std::size_t column = 0; column < column_count; ++column) {
            columns[column].push_back(values[column] ? static_cast<float>(*values[column]) : std::nanf(""));
            scored[column] = scored[column] || values[column].has_value();
        }
    }

    const std::map<int, double>& xpsnr_;
    std::vector<Place> open_;  // the objects and arrays open, outermost first
    Next next_ = Next::other;
    std::optional<std::size_t> metric_;  // the metric key just read, if it is one
    Frame frame_;
};

}  // namespace

namespace run_parts {

std::map<int, double> parse_xpsnr_log(const std::filesystem::path& path) {
    std::map<int, double> scores;
    std::ifstream file(path, std::ios::binary);
    static const std::regex line_pattern(
        R"(^n:\s*(\d+)\s+XPSNR y:\s*([+-]?(?:inf|nan|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)))", std::regex::icase);
    for (std::string line; std::getline(file, line);) {
        std::smatch match;
        if (std::regex_search(line, match, line_pattern)) {
            scores[std::stoi(match[1].str()) - 1] = python_float(match[2].str()).value_or(std::nan(""));
        }
    }
    return scores;
}

std::vector<double> times_of(const std::vector<int>& frames, double fps) {
    std::vector<double> times;
    for (const int frame : frames) {
        times.push_back(fps > 0 ? frame / fps : 0.0);
    }
    return times;
}

void keep_range(ParsedScores& scores, const FrameRange& range, double fps) {
    if (range.whole()) {
        return;
    }
    ParsedScores kept;
    kept.order = scores.order;
    for (std::size_t i = 0; i < scores.frame.size(); ++i) {
        const long long frame = scores.frame[i] + range.decode_start;
        if (frame < range.start || (range.end && frame >= *range.end)) {
            continue;
        }
        kept.frame.push_back(static_cast<int>(frame));
        for (const auto& [key, values] : scores.metrics) {
            kept.metrics[key].push_back(i < values.size() ? values[i] : std::nanf(""));
        }
    }
    kept.time = times_of(kept.frame, fps);
    scores = std::move(kept);
}

ParsedScores parse_log(const std::filesystem::path& log_path, double fps,
                       const std::optional<std::filesystem::path>& xpsnr_log_path) {
    const std::map<int, double> xpsnr = xpsnr_log_path ? parse_xpsnr_log(*xpsnr_log_path) : std::map<int, double>{};
    LogReader reader(xpsnr);
    std::ifstream log(log_path, std::ios::binary);
    nlohmann::json::sax_parse(log, &reader);
    ParsedScores parsed;
    if (reader.frames.empty()) {
        return parsed;
    }
    parsed.frame = std::move(reader.frames);
    parsed.time = times_of(parsed.frame, fps);
    for (std::size_t column = 0; column < column_count; ++column) {
        if (!reader.scored[column]) {
            continue;  // the run did not ask for this metric
        }
        parsed.metrics[column_keys[column]] = std::move(reader.columns[column]);
        parsed.order.emplace_back(column_keys[column]);
    }
    return parsed;
}

}  // namespace run_parts

FrameScores parse_libvmaf_log(const std::filesystem::path& log, double fps,
                              const std::optional<std::filesystem::path>& xpsnr_log) {
    ParsedScores parsed = parse_log(log, fps, xpsnr_log);
    return FrameScores{std::move(parsed.frame), std::move(parsed.time), std::move(parsed.metrics)};
}

}  // namespace vml
