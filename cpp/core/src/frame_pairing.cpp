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

bool FramesShort::different_lengths() const {
    return source_length && test_length && std::llabs(*source_length - *test_length) > 1;
}

bool FramesShort::in_step() const {
    if (!counted()) {
        return false;
    }
    if (*source_frames == *test_frames) {
        return true;
    }
    if (!different_lengths()) {
        return false;
    }
    const bool source_fewer = *source_frames < *test_frames;
    const long long fewer_lacks = source_fewer ? *source_length - *source_frames : *test_length - *test_frames;
    const long long other_lacks = source_fewer ? *test_length - *test_frames : *source_length - *source_frames;
    return other_lacks >= fewer_lacks;
}

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
    return FramesShort{pairs, shorter, std::nullopt, std::nullopt, source, test};
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
    if (short_of.in_step() && *short_of.source_frames == *short_of.test_frames) {
        return "Only " + with_thousands(short_of.scored_pairs) + " frames were compared, while videos of this " +
               "length should have " + with_thousands(short_of.length_frames) + ", but both videos have " +
               with_thousands(*short_of.source_frames) + " frames, so they are in sync.";
    }
    if (short_of.in_step()) {
        return "Only " + with_thousands(short_of.scored_pairs) + " frames were compared, while the shorter video's " +
               "length should have " + with_thousands(short_of.length_frames) + ". The two videos are different " +
               "lengths (this video has " + with_thousands(*short_of.test_frames) + " frames, the reference " +
               with_thousands(*short_of.source_frames) + "), and the one with fewer frames is missing no more of " +
               "its length than the other, so its missing frames can be gaps both videos share: nothing suggests "
               "they are out of sync.";
    }
    if (short_of.counted() && short_of.different_lengths()) {
        return "This video has " + with_thousands(*short_of.test_frames) + " frames for a length that should have " +
               with_thousands(*short_of.test_length) + ", and the reference " +
               with_thousands(*short_of.source_frames) + " for " + with_thousands(*short_of.source_length) +
               ". The one with fewer frames is missing more of its length than the other, so it lacks frames the "
               "other has. From the first such frame, each frame is compared with the wrong reference frame, so "
               "every score after that point is wrong. This usually happens when frames are duplicated or dropped "
               "during encoding; encoding with a variable frame rate that matches the source avoids it.";
    }
    if (short_of.counted()) {
        return "This video has " + with_thousands(*short_of.test_frames) + " frames, but the reference has " +
               with_thousands(*short_of.source_frames) +
               ", even though both are the same length. From the first point where they differ, each frame is "
               "compared with the wrong reference frame, so every score after that point is wrong. This usually "
               "happens when frames are duplicated or dropped during encoding; encoding with a variable frame rate "
               "that matches the source avoids it.";
    }
    return "This video may be out of sync with the reference: only " + with_thousands(short_of.scored_pairs) +
           " frames were compared, but videos of this length should have " + with_thousands(short_of.length_frames) +
           ". If the scores drop suddenly partway through and stay low, this is the likely cause. At least one of "
           "the files doesn't store its frame count, so to find out for sure it has to be read from start to "
           "finish: use --frame-counts check, or in the app turn on Settings > Storage > Read entire files to check "
           "frame counts when needed, then recalculate the video (right-click it > Recalculate selected metrics).";
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
        status = !short_of->counted() ? "may_be_out_of_sync" : short_of->in_step() ? "in_sync" : "out_of_sync";
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
        note = "Not checked: the test video is made from the reference, frame by frame.";
    } else if (pairs <= 0) {
        status = "not_checked";
        note = "Not checked: none of the metrics scores individual frames.";
    } else if (shorter <= 0) {
        status = "not_checked";
        note = "Not checked: the length of one of the videos isn't known.";
    } else if (shorter - pairs >= 2) {
        status = "not_checked";
        note = "Not checked: only part of the videos was compared.";
    } else {
        status = "no_mismatch_seen";
        note = "The number of frames compared (" + with_thousands(pairs) +
               ") matches the length of the videos, so nothing suggests they are out of sync.";
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
