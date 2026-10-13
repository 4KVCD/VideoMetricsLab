#include "vml/perceptual.h"

#include <algorithm>

#include "vml/crop_detect.h"
#include "vml/geometry.h"

namespace vml {

std::string PerceptualRequest::backend_of(std::string_view key) const {
    if (key == "cvvdp") {
        return "gpu";
    }
    const auto found = backends.find(key);
    return found != backends.end() ? found->second : "gpu";
}

int PerceptualRequest::step_of(std::string_view key) const { return perceptual_step(key, options.n_subsample); }

int perceptual_step(std::string_view key, int n_subsample) {
    return key == "cvvdp" ? 1 : std::max(1, n_subsample);
}

void set_failure(PerceptualFailures& failures, const std::string& key, const std::string& reason) {
    const auto found =
        std::find_if(failures.begin(), failures.end(), [&key](const auto& failure) { return failure.first == key; });
    if (found != failures.end()) {
        found->second = reason;
    } else {
        failures.emplace_back(key, reason);
    }
}

const std::string* failure_of(const PerceptualFailures& failures, const std::string& key) {
    const auto found =
        std::find_if(failures.begin(), failures.end(), [&key](const auto& failure) { return failure.first == key; });
    return found != failures.end() ? &found->second : nullptr;
}

double compared_seconds(const VideoInfo& source, const VideoInfo& distorted, double duration_limit) {
    double seconds = std::min(source.duration, distorted.duration);
    if (duration_limit > 0) {
        seconds = std::min(seconds, duration_limit);
    }
    return seconds;
}

void validate_pair(const VideoInfo& source, const VideoInfo& distorted, double duration_limit) {
    if (const std::optional<std::string> problem = pair_problem(source, distorted, duration_limit)) {
        throw PerceptualRunError(*problem);
    }
}

CropPair resolve_crops(const std::string& ffmpeg, const VideoInfo& source, const VideoInfo& distorted,
                       CropMode crop_mode, const CancelToken& cancel, const StatusCallback& on_status) {
    if (crop_mode == CropMode::none) {
        return {std::nullopt, std::nullopt};
    }
    if (on_status) {
        // Worded as for FFmpeg's metrics, because a video's tasks share one
        // detection (crop_detect waits for the other task's answer), and
        // their parts of the run line sit side by side.
        on_status({"Detecting black bars in source and distorted\xE2\x80\xA6"});
    }
    // Black bars are detected from samples across the whole file, not only
    // the part compared: a short duration limit may fall entirely in a dark
    // intro, which must not decide the crop. Decoded in software, as the
    // Python app's perceptual metrics detect them.
    const auto [source_box, distorted_box] =
        detect_pair(ffmpeg, source, distorted, cancel, std::nullopt, std::nullopt);
    return common_picture(source, distorted, source_box, distorted_box);
}

}  // namespace vml
