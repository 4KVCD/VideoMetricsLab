#include "vml/export_csv.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

#include "vml/append.h"
#include "vml/difference.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/temp_files.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// The columns the export always had come first, in their old order, so a
// new metric never moves them.
constexpr std::array<std::string_view, 5> first_columns{"vmaf", "vmaf_neg", "psnr", "ssim", "xpsnr"};

// The frame metrics in the CSV's order.
std::vector<std::string> column_keys() {
    std::vector<std::string> keys;
    for (const std::string_view first : first_columns) {
        keys.emplace_back(first);
    }
    for (const MetricDefinition& metric : metrics()) {
        if (metric.kind == MetricKind::frame &&
            std::find(first_columns.begin(), first_columns.end(), metric.key) == first_columns.end()) {
            keys.emplace_back(metric.key);
        }
    }
    return keys;
}

// A result's metrics as they are exported: an old result whose VMAF was
// scored with a bundled VMAF v1 model (from before VMAF v1 had a column of
// its own) has that VMAF in the vmaf_v1 column.
MetricResultSet portable_metric_results(const ComparisonResult& result) {
    const FrameMetricResult* vmaf = result.metric_results.frame("vmaf");
    const bool v1_choice = result.model_choice && is_builtin_model_choice(*result.model_choice);
    if (!v1_choice || vmaf == nullptr || result.metric_results.has("vmaf_v1")) {
        return result.metric_results;
    }
    MetricResultSet moved;
    for (const std::string& key : result.metric_results.keys()) {
        if (key == "vmaf") {
            FrameMetricResult renamed = *vmaf;
            renamed.key = "vmaf_v1";
            moved.add(renamed);
        } else if (const FrameMetricResult* frame = result.metric_results.frame(key)) {
            moved.add(*frame);
        } else if (const SequenceMetricResult* sequence = result.metric_results.sequence(key)) {
            moved.add(*sequence);
        }
    }
    return moved;
}

// A result's scores as export_csv lays them out: every frame any metric
// scored, in order, with its time (the first column's that has the frame),
// and each column's cell on each row (NaN: blank).
struct CsvTable {
    std::vector<int> frames;
    std::vector<double> times;
    std::vector<std::string> names;  // the columns after frame and time_s
    std::map<std::string, std::vector<double>> values;

    std::size_t row_of(int frame) const {
        return static_cast<std::size_t>(std::lower_bound(frames.begin(), frames.end(), frame) - frames.begin());
    }
};

constexpr double blank = std::numeric_limits<double>::quiet_NaN();

CsvTable csv_table(const MetricResultSet& metrics) {
    CsvTable table;
    const std::vector<std::string> keys = column_keys();
    const SequenceMetricResult* cvvdp = metrics.sequence("cvvdp");
    for (const std::string& key : keys) {
        if (const FrameMetricResult* column = metrics.frame(key)) {
            append(table.frames, column->frame);
        }
    }
    if (cvvdp != nullptr && cvvdp->has_timeline()) {
        append(table.frames, cvvdp->frame);
    }
    std::sort(table.frames.begin(), table.frames.end());
    table.frames.erase(std::unique(table.frames.begin(), table.frames.end()), table.frames.end());
    // A CVVDP score alone, with no per-second values: a row to hold it.
    const bool score_alone = cvvdp != nullptr && table.frames.empty();
    if (score_alone) {
        table.frames.push_back(0);
    }

    table.times.assign(table.frames.size(), blank);
    if (score_alone) {
        table.times.front() = 0.0;
    }
    table.names = keys;
    const auto fill = [&table](const std::vector<int>& at, const std::vector<double>& their_times,
                               const auto& their_values, std::vector<double>& column) {
        for (std::size_t i = 0; i < at.size(); ++i) {
            const std::size_t row = table.row_of(at[i]);
            column[row] = static_cast<double>(their_values[i]);
            if (std::isnan(table.times[row])) {
                table.times[row] = their_times[i];
            }
        }
    };
    for (const std::string& key : keys) {
        std::vector<double>& column = table.values[key] = std::vector<double>(table.frames.size(), blank);
        if (const FrameMetricResult* result_column = metrics.frame(key)) {
            fill(result_column->frame, result_column->time, result_column->values, column);
        }
    }
    if (cvvdp != nullptr) {
        // The JOD of the second starting at each row's frame, and the whole
        // video's JOD on the same rows.
        std::vector<double>& second = table.values["cvvdp_second_jod"] =
            std::vector<double>(table.frames.size(), blank);
        std::vector<double>& video = table.values["cvvdp_video_jod"] = std::vector<double>(table.frames.size(), blank);
        if (cvvdp->has_timeline()) {
            fill(cvvdp->frame, cvvdp->time, cvvdp->values, second);
            for (const int frame : cvvdp->frame) {
                video[table.row_of(frame)] = cvvdp->score;
            }
        } else {
            video.front() = cvvdp->score;  // no seconds to put it on: the first row
        }
        table.names.emplace_back("cvvdp_second_jod");
        table.names.emplace_back("cvvdp_video_jod");
    }
    return table;
}

void write_csv_table(const CsvTable& table, const std::filesystem::path& path) {
    // Python's csv.writer: comma-separated, rows ended with CR LF; none of
    // these cells needs quotes.
    write_file_whole(path, [&table](std::ostream& file) {
        std::string header = "frame,time_s";
        for (const std::string& name : table.names) {
            header += "," + name;
        }
        file << header << "\r\n";
        for (std::size_t row = 0; row < table.frames.size(); ++row) {
            std::string line = std::to_string(table.frames[row]) + "," + format_fixed(table.times[row], 6);
            for (const std::string& name : table.names) {
                const double value = table.values.at(name)[row];
                line += "," + (std::isnan(value) ? std::string() : python_repr(value));
            }
            file << line << "\r\n";
        }
    });
}

}  // namespace

void export_csv(const ComparisonResult& result, const std::filesystem::path& path) {
    write_csv_table(csv_table(portable_metric_results(result)), path);
}

void export_difference_csv(const ComparisonResult& first, const ComparisonResult& second,
                           const std::filesystem::path& path) {
    const MetricResultSet ours = portable_metric_results(first);
    const MetricResultSet theirs = portable_metric_results(second);
    CsvTable table = csv_table(ours);  // the first's rows, times and columns
    for (const std::string& key : column_keys()) {
        std::vector<double>& column = table.values.at(key);
        std::fill(column.begin(), column.end(), blank);
        const FrameMetricResult* mine = ours.frame(key);
        const FrameMetricResult* other = theirs.frame(key);
        if (mine != nullptr && other != nullptr) {
            const std::vector<double> differences =
                frame_difference(mine->frame, mine->values, other->frame, other->values);
            for (std::size_t i = 0; i < differences.size(); ++i) {
                column[table.row_of(mine->frame[i])] = differences[i];
            }
        }
    }
    if (const SequenceMetricResult* mine = ours.sequence("cvvdp")) {
        std::vector<double>& seconds = table.values.at("cvvdp_second_jod");
        std::vector<double>& video = table.values.at("cvvdp_video_jod");
        std::fill(seconds.begin(), seconds.end(), blank);
        std::fill(video.begin(), video.end(), blank);
        const SequenceMetricResult* other = theirs.sequence("cvvdp");
        if (mine->has_timeline() && other != nullptr && other->has_timeline()) {
            const std::vector<double> differences =
                second_difference(mine->time, mine->values, other->time, other->values);
            for (std::size_t i = 0; i < differences.size(); ++i) {
                const std::size_t row = table.row_of(mine->frame[i]);
                seconds[row] = differences[i];
                video[row] = mine->score - other->score;
            }
        }
    }
    write_csv_table(table, path);
}

std::string safe_filename_stem(const std::string& label) {
    std::string stem;
    for (std::size_t i = 0; i < label.size(); ++i) {
        const auto c = static_cast<unsigned char>(label[i]);
        if (c >= 0x80) {
            // A character past ASCII is kept whole. Python's isalnum() keeps
            // the letters and digits of every script, but would replace the
            // rest, such as a dash or a symbol, with "_".
            stem += label[i];
        } else if (std::isalnum(c) != 0 || c == '-' || c == '_' || c == '.') {
            stem += static_cast<char>(c);
        } else {
            stem += '_';
        }
    }
    return stem;
}

std::string difference_label(const std::string& first, const std::string& second) {
    // Cut at a character's first byte, never inside a UTF-8 sequence.
    const auto cut = [](const std::string& name) {
        constexpr int longest = 60;
        int characters = 0;
        for (std::size_t i = 0; i < name.size(); ++i) {
            const bool continues = (static_cast<unsigned char>(name[i]) & 0xC0) == 0x80;
            if (!continues && ++characters > longest) {
                return name.substr(0, i);
            }
        }
        return name;
    };
    return cut(first) + "_minus_" + cut(second);
}

std::string output_stem(const std::string& label) {
    std::string stem = safe_filename_stem(label);
    return stem.empty() ? "run" : stem;
}

std::filesystem::path unique_output_path(const std::filesystem::path& folder, const std::string& label,
                                         const std::string& suffix, std::set<std::filesystem::path>& reserved) {
    return unique_output_paths(folder, label, {suffix}, reserved).front();
}

std::vector<std::filesystem::path> unique_output_paths(const std::filesystem::path& folder, const std::string& label,
                                                       const std::vector<std::string>& suffixes,
                                                       std::set<std::filesystem::path>& reserved) {
    const std::string stem = output_stem(label);
    const auto paths_for = [&](const std::string& name) {
        std::vector<std::filesystem::path> paths;
        for (const std::string& suffix : suffixes) {
            paths.push_back(folder / platform::path_from_text(name + suffix));
        }
        return paths;
    };
    // Reserved as the system compares names: "Clip" and "clip", still to be
    // written, are one file on Windows.
    const auto as_compared = [](const std::filesystem::path& path) {
        return path.parent_path() /
               platform::path_from_text(platform::name_as_compared(platform::path_text(path.filename())));
    };
    const auto taken = [&reserved, &as_compared](const std::vector<std::filesystem::path>& paths) {
        std::error_code error;
        return std::any_of(paths.begin(), paths.end(), [&](const std::filesystem::path& path) {
            return reserved.contains(as_compared(path)) || std::filesystem::exists(path, error);
        });
    };
    std::vector<std::filesystem::path> candidates = paths_for(stem);
    for (int counter = 2; taken(candidates); ++counter) {
        candidates = paths_for(stem + "_" + std::to_string(counter));
    }
    for (const std::filesystem::path& path : candidates) {
        reserved.insert(as_compared(path));
    }
    return candidates;
}

}  // namespace vml
