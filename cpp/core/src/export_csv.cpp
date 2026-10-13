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

#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"

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

}  // namespace

void export_csv(const ComparisonResult& result, const std::filesystem::path& path) {
    const MetricResultSet metrics = portable_metric_results(result);
    const std::vector<std::string> keys = column_keys();
    // Every frame any metric scored, in order.
    std::vector<int> frames;
    const SequenceMetricResult* cvvdp = metrics.sequence("cvvdp");
    for (const std::string& key : keys) {
        if (const FrameMetricResult* column = metrics.frame(key)) {
            frames.insert(frames.end(), column->frame.begin(), column->frame.end());
        }
    }
    if (cvvdp != nullptr && cvvdp->has_timeline()) {
        frames.insert(frames.end(), cvvdp->frame.begin(), cvvdp->frame.end());
    }
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    const auto row_of = [&frames](int frame) {
        return static_cast<std::size_t>(std::lower_bound(frames.begin(), frames.end(), frame) - frames.begin());
    };

    const double none = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> times(frames.size(), none);
    std::map<std::string, std::vector<double>> values;
    std::vector<std::string> names = keys;
    // A frame's time is the first column's that has the frame.
    const auto fill = [&](const std::vector<int>& at, const std::vector<double>& their_times,
                          const auto& their_values, std::vector<double>& column) {
        for (std::size_t i = 0; i < at.size(); ++i) {
            const std::size_t row = row_of(at[i]);
            column[row] = static_cast<double>(their_values[i]);
            if (std::isnan(times[row])) {
                times[row] = their_times[i];
            }
        }
    };
    for (const std::string& key : keys) {
        std::vector<double>& column = values[key] = std::vector<double>(frames.size(), none);
        if (const FrameMetricResult* result_column = metrics.frame(key)) {
            fill(result_column->frame, result_column->time, result_column->values, column);
        }
    }
    if (cvvdp != nullptr) {
        // The JOD of the second starting at each row's frame, and the whole
        // video's JOD on the same rows.
        std::vector<double>& second = values["cvvdp_second_jod"] = std::vector<double>(frames.size(), none);
        std::vector<double>& video = values["cvvdp_video_jod"] = std::vector<double>(frames.size(), none);
        if (cvvdp->has_timeline()) {
            fill(cvvdp->frame, cvvdp->time, cvvdp->values, second);
            for (const int frame : cvvdp->frame) {
                video[row_of(frame)] = cvvdp->score;
            }
        }
        names.emplace_back("cvvdp_second_jod");
        names.emplace_back("cvvdp_video_jod");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot write " + platform::path_text(path));
    }
    // Python's csv.writer: comma-separated, rows ended with CR LF; none of
    // these cells needs quotes.
    std::string header = "frame,time_s";
    for (const std::string& name : names) {
        header += "," + name;
    }
    file << header << "\r\n";
    for (std::size_t row = 0; row < frames.size(); ++row) {
        std::string line = std::to_string(frames[row]) + "," + format_fixed(times[row], 6);
        for (const std::string& name : names) {
            const double value = values[name][row];
            line += "," + (std::isnan(value) ? std::string() : python_repr(value));
        }
        file << line << "\r\n";
    }
    if (!file) {
        throw std::runtime_error("cannot write " + platform::path_text(path));
    }
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

std::filesystem::path unique_output_path(const std::filesystem::path& folder, const std::string& label,
                                         const std::string& suffix, std::set<std::filesystem::path>& reserved) {
    std::string stem = safe_filename_stem(label);
    if (stem.empty()) {
        stem = "run";
    }
    std::filesystem::path candidate = folder / platform::path_from_text(stem + suffix);
    std::error_code error;
    for (int counter = 2; reserved.contains(candidate) || std::filesystem::exists(candidate, error); ++counter) {
        candidate = folder / platform::path_from_text(stem + "_" + std::to_string(counter) + suffix);
    }
    reserved.insert(candidate);
    return candidate;
}

}  // namespace vml
