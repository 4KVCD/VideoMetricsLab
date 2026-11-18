#include "vml/frame_pairing.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <future>
#include <utility>

#include "vml/app_log.h"
#include "vml/ffprobe.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.frame_pairing");

// 132985 as "132,985".
std::string with_thousands(long long number) {
    std::string digits = std::to_string(std::llabs(number));
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) {
        digits.insert(static_cast<std::size_t>(at), ",");
    }
    return number < 0 ? "-" + digits : digits;
}

// The frames a video's length holds at its frame rate; 0 when not known.
long long length_frames(const VideoInfo& video) {
    if (video.duration <= 0 || video.fps <= 0) {
        return 0;
    }
    return std::llround(video.duration * video.fps);
}

}  // namespace

long long scored_frame_pairs(const ComparisonResult& result) {
    long long reach = 0;
    for (const std::string& key : result.metric_results.keys()) {
        const FrameMetricResult* metric = result.metric_results.frame(key);
        if (metric == nullptr || metric->frame.empty()) {
            continue;
        }
        const std::vector<int>& frames = metric->frame;
        const long long step = frames.size() > 1 ? std::max(1, frames[1] - frames[0]) : 1;
        reach = std::max(reach, frames.back() + step);
    }
    return reach;
}

std::optional<FramesShort> frames_short_of_lengths(const ComparisonResult& result, bool to_the_end) {
    if (!to_the_end || result.resample_target) {
        return std::nullopt;
    }
    const long long source = length_frames(result.source_info);
    const long long test = length_frames(result.distorted_info);
    const long long pairs = scored_frame_pairs(result);
    if (source <= 0 || test <= 0 || pairs <= 0) {
        return std::nullopt;
    }
    const long long shorter = std::min(source, test);
    if (shorter - pairs < 2) {
        return std::nullopt;
    }
    return FramesShort{pairs, shorter};
}

FramesShort count_frames_short(const FramesShort& short_of, const VideoInfo& source, const VideoInfo& test,
                               const FrameCounter& count) {
    // Both at once: on a network drive each count may read a whole film.
    std::future<std::optional<long long>> source_count =
        std::async(std::launch::async, [&count, &source] { return count(source); });
    const std::optional<long long> test_frames = count(test);
    const std::optional<long long> source_frames = source_count.get();
    const auto plausible = [](const std::optional<long long>& counted, const VideoInfo& video) {
        const long long expected = length_frames(video);
        return counted && *counted > 0 &&
               std::llabs(*counted - expected) <= std::max(10LL, static_cast<long long>(expected / 100));
    };
    if (!plausible(source_frames, source) || !plausible(test_frames, test)) {
        return short_of;
    }
    FramesShort counted = short_of;
    counted.source_frames = source_frames;
    counted.test_frames = test_frames;
    return counted;
}

std::string frames_short_text(const FramesShort& short_of) {
    if (short_of.in_step()) {
        return {};
    }
    std::string text = frames_short_note(short_of);
    text.front() = static_cast<char>(std::tolower(static_cast<unsigned char>(text.front())));
    if (text.back() == '.') {
        text.pop_back();  // a note on the command line ends without one
    }
    return text;
}

std::string frames_short_note(const FramesShort& short_of) {
    const std::string compared = "Fewer frames were compared (" + with_thousands(short_of.scored_pairs) +
                                 ") than the videos' length holds (" + with_thousands(short_of.length_frames) + ")";
    if (short_of.in_step()) {
        return compared + ", but both videos have " + with_thousands(*short_of.source_frames) +
               " frames: their frames line up.";
    }
    if (short_of.counted()) {
        return "The reference has " + with_thousands(*short_of.source_frames) + " frames and the test video " +
               with_thousands(*short_of.test_frames) +
               " in the same length, so from where they part their frames are compared out of step and the scores "
               "from there on are wrong. Frames repeated or dropped as the test video was encoded do this (an "
               "encode at a constant frame rate of a source with gaps in its timing, say); encoding at a variable "
               "frame rate, as the source has it, keeps its frames as they are.";
    }
    return compared +
           ": if the scores drop suddenly partway through and stay low, the two may not line up frame for frame. "
           "Not both videos state their frame count, so to tell for sure their frames must be counted by reading "
           "the whole files: --frame-counts check, or in the window Settings > Storage > Read whole files to count "
           "frames when they may not line up, then calculate the video again (in the window: right-click it > "
           "Recalculate selected metrics).";
}

nlohmann::ordered_json frame_pairing_json(const ComparisonResult& result) {
    const auto count = [](const std::optional<long long>& value) {
        return value ? nlohmann::ordered_json(*value) : nlohmann::ordered_json(nullptr);
    };
    const long long pairs = scored_frame_pairs(result);
    const long long source = length_frames(result.source_info);
    const long long test = length_frames(result.distorted_info);
    const long long shorter = source > 0 && test > 0 ? std::min(source, test) : 0;
    std::string status;
    std::string note;
    if (const std::optional<FramesShort>& short_of = result.frames_short) {
        status = !short_of->counted() ? "may_be_out_of_step" : short_of->in_step() ? "in_step" : "out_of_step";
        note = frames_short_note(*short_of);
        return {{"status", status},
                {"note", note},
                {"compared_pairs", short_of->scored_pairs},
                {"length_frames", short_of->length_frames},
                {"reference_frames", count(short_of->source_frames)},
                {"test_frames", count(short_of->test_frames)}};
    }
    if (result.resample_target) {
        status = "not_checked";
        note = "Not checked: the test video is made from the reference, frame for frame.";
    } else if (pairs <= 0) {
        status = "not_checked";
        note = "Not checked: no metric scored single frames.";
    } else if (shorter <= 0) {
        status = "not_checked";
        note = "Not checked: a video's length is not known.";
    } else if (shorter - pairs >= 2) {
        status = "not_checked";
        note = "Not checked: only a part of the videos was compared.";
    } else {
        status = "no_mismatch_seen";
        note = "As many frames were compared (" + with_thousands(pairs) + ") as the videos' length holds (" +
               with_thousands(shorter) + "): nothing suggests their frames are out of step.";
    }
    return {{"status", status},
            {"note", note},
            {"compared_pairs", pairs},
            {"length_frames", shorter > 0 ? nlohmann::ordered_json(shorter) : nlohmann::ordered_json(nullptr)},
            {"reference_frames", nullptr},
            {"test_frames", nullptr}};
}

FrameCounter container_frame_counter(std::string ffprobe, CancelToken cancel, bool read_whole_files) {
    return [ffprobe = std::move(ffprobe), cancel = std::move(cancel),
            read_whole_files](const VideoInfo& video) -> std::optional<long long> {
        try {
            if (const std::optional<long long> stated = trusted_frame_count(video.path, ffprobe, cancel)) {
                return stated;
            }
            if (!read_whole_files) {
                return std::nullopt;
            }
            return count_shown_video_packets(video.path, ffprobe, cancel);
        } catch (const Cancelled&) {
            throw;
        } catch (const std::exception& error) {
            logger.warning("The frames of " + platform::path_text(video.path) + " could not be counted: " +
                           error.what());
            return std::nullopt;
        }
    };
}

}  // namespace vml
