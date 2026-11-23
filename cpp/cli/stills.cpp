#include "stills.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "common.h"
#include "vml/cancel.h"
#include "vml/export_csv.h"
#include "vml/ffmpeg_tools.h"
#include "vml/frame_extract.h"
#include "vml/metrics.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/results_summary.h"
#include "vml/settings.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

// `frame` as the files' names have it: six digits, as Video Compare's.
std::string frame_digits(int frame) {
    char text[16];
    std::snprintf(text, sizeof(text), "%06d", frame);
    return text;
}

// The worst sample of `segment` by `condition`'s metric: only the segment's
// own samples are looked at. They are in frame order (score_at relies on it
// too), so the first is found by a binary search, and the scan stops after
// the segment's last frame.
std::optional<WorstFrame> segment_worst(const ScoreSamples& samples, const WeakCondition& condition,
                                        const WeakSegment& segment) {
    std::optional<WorstFrame> worst;
    const auto first = std::lower_bound(samples.frame.begin(), samples.frame.end(), segment.first_frame);
    for (auto index = static_cast<std::size_t>(first - samples.frame.begin());
         index < samples.frame.size() && samples.frame[index] <= segment.last_frame; ++index) {
        const int frame = samples.frame[index];
        const double score = samples.values[index];
        if (std::isnan(score)) {
            continue;
        }
        const bool worse = !worst || (condition.comparison == '>' ? score > worst->score : score < worst->score);
        if (worse) {
            worst = WorstFrame{frame, samples.time[index], score};
        }
    }
    return worst;
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) {
        throw std::runtime_error("could not write " + platform::path_text(path));
    }
}

// The two pictures side by side, the reference on the left, made by FFmpeg
// from the two files (the same size: both are as the metrics compared them).
void write_side_by_side(const std::string& ffmpeg, const std::filesystem::path& reference,
                        const std::filesystem::path& test, const std::filesystem::path& out) {
    std::vector<std::string> command = ffmpeg_command_start(ffmpeg, {"-loglevel", "error", "-y"});
    command.insert(command.end(), {"-i", platform::path_text(reference), "-i", platform::path_text(test),
                                   "-filter_complex", "hstack=inputs=2", "-frames:v", "1", "-compression_level", "2",
                                   "-pred", "paeth", platform::path_text(out)});
    platform::ChildProcess process = platform::ChildProcess::start(command);
    platform::read_to_end(process);
    if (process.wait() != 0) {
        const std::string error(python_strip(replace_invalid_utf8(process.standard_error())));
        throw std::runtime_error("FFmpeg could not put them side by side: " + error);
    }
}

// FFmpeg, as compare finds it; checked once.
const std::string& stills_ffmpeg() {
    static const ToolStatus tool = check_tool("ffmpeg", Settings::load());
    if (!tool.runnable) {
        throw std::runtime_error("FFmpeg is needed for the stills: " + tool.error);
    }
    return tool.path;
}

}  // namespace

std::vector<StillFrame> still_frames(const ComparisonResult& result, const std::string& from, std::size_t count,
                                     const SegmentRules& rules) {
    std::map<int, StillFrame> frames;
    if (from == "segments") {
        const SegmentRules used = scored_rules(result.metric_results, rules);
        const std::vector<WeakSegment> segments = weak_segments(result, used);
        // Each condition's samples, taken once
        std::vector<std::optional<ScoreSamples>> samples_of;
        for (const WeakCondition& condition : used.conditions) {
            samples_of.push_back(score_samples(result.metric_results, condition.key));
        }
        // Each segment's worst frame, by the first condition with scores
        std::vector<StillFrame> found;
        for (std::size_t index = 0; index < segments.size(); ++index) {
            for (std::size_t c = 0; c < used.conditions.size(); ++c) {
                const WeakCondition& condition = used.conditions[c];
                const std::optional<WorstFrame> worst =
                    samples_of[c] ? segment_worst(*samples_of[c], condition, segments[index]) : std::nullopt;
                if (!worst) {
                    continue;
                }
                found.push_back(StillFrame{worst->frame, worst->time, "segment", {condition.key}, worst->score,
                                           static_cast<int>(index)});
                break;
            }
        }
        // The weakest: by how far past its condition's threshold
        const auto margin = [&used](const StillFrame& still) {
            const auto condition = std::find_if(used.conditions.begin(), used.conditions.end(),
                                                [&still](const WeakCondition& c) { return c.key == still.metrics[0]; });
            const double past = *still.score - condition->threshold;
            return condition->comparison == '>' ? -past : past;
        };
        std::stable_sort(found.begin(), found.end(),
                         [&margin](const StillFrame& a, const StillFrame& b) { return margin(a) < margin(b); });
        found.resize(std::min(found.size(), count));
        for (StillFrame& still : found) {
            frames.emplace(still.frame, std::move(still));
        }
    } else {
        for (const MetricDefinition& metric : metrics()) {
            const std::string key(metric.key);
            const std::optional<ScoreSamples> samples = score_samples(result.metric_results, key);
            if (!samples || samples->frame.empty()) {
                continue;
            }
            for (const WorstFrame& worst : worst_frames(*samples, metric.direction, count)) {
                auto [at, added] = frames.try_emplace(worst.frame);
                if (added) {
                    at->second = StillFrame{worst.frame, worst.time, "worst", {key}, worst.score, std::nullopt};
                } else {
                    at->second.metrics.push_back(key);
                }
            }
        }
    }
    std::vector<StillFrame> ordered;
    for (auto& [frame, still] : frames) {
        ordered.push_back(std::move(still));
    }
    return ordered;
}

bool write_stills(std::vector<TestVideo>& videos, const std::string& from, std::size_t count,
                  const SegmentRules& rules, const std::filesystem::path& folder,
                  const std::function<void(const std::string&)>& message,
                  const std::function<void(const std::string&)>& failed) {
    bool all = true;
    std::set<std::filesystem::path> reserved;
    std::error_code no_folder;  // a folder that cannot be made fails at the first write
    std::filesystem::create_directories(folder, no_folder);
    for (TestVideo& video : videos) {
        if (!video.result) {
            continue;
        }
        video.stills.clear();
        const std::vector<StillFrame> frames = still_frames(*video.result, from, count, rules);
        if (frames.empty()) {
            continue;
        }
        try {
            const std::string& ffmpeg = stills_ffmpeg();
            const FrameComparison comparison = FrameComparison::from_result(*video.result);
            // As Video Compare's Export frame as PNG: HDR mapped to SDR white
            // (100 nits), SDR as it is; a file outlives the display.
            const PreviewColorSettings colors{PreviewColorMode::display_aware, std::nullopt, std::nullopt};
            for (const StillFrame& frame : frames) {
                const std::string label = video.file_stem() + ".frame" + frame_digits(frame.frame);
                const std::vector<std::filesystem::path> paths = unique_output_paths(
                    folder, label, {".reference.png", ".test.png", ".side-by-side.png"}, reserved);
                for (const auto& [side, path] :
                     {std::pair{FrameSide::source, paths[0]}, std::pair{FrameSide::distorted, paths[1]}}) {
                    write_file(path, extract_frame_png(ffmpeg, comparison, side, frame.frame, CancelToken(), colors,
                                                       PngEncoding::compressed));
                }
                write_side_by_side(ffmpeg, paths[0], paths[1], paths[2]);
                video.stills.push_back(VideoStill{frame, paths[0], paths[1], paths[2]});
            }
            const std::size_t made = video.stills.size();
            message("Wrote " + std::to_string(made) + (made == 1 ? " still of " : " stills of ") + video.name() +
                    " to " + platform::path_text(folder));
        } catch (const std::exception& error) {
            failed(video.name() + ": its stills could not all be made: " + error.what());
            all = false;
        }
    }
    return all;
}

Json stills_json(const std::vector<VideoStill>& stills) {
    Json list = Json::array();
    for (const VideoStill& still : stills) {
        const StillFrame& frame = still.frame;
        Json entry{{"frame", frame.frame},
                   {"time", frame.time},
                   {"from", frame.from},
                   {"metrics", frame.metrics},
                   {"score", json_number(frame.score)}};
        entry["segment"] = frame.segment ? Json(*frame.segment) : Json(nullptr);
        entry["reference"] = platform::path_text(still.reference);
        entry["test"] = platform::path_text(still.test);
        entry["side_by_side"] = platform::path_text(still.side_by_side);
        list.push_back(std::move(entry));
    }
    return list;
}

}  // namespace vml::cli
