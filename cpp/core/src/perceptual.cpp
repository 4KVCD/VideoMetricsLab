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

void validate_pair(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options) {
    if (const std::optional<std::string> problem =
            pair_problem(source, distorted, options.range_start, options.range_end)) {
        throw PerceptualRunError(*problem);
    }
}

CropPair resolve_crops(const std::string& ffmpeg, const VideoInfo& source, const VideoInfo& distorted,
                       CropMode crop_mode, const std::string& scale_algorithm, const CancelToken& cancel,
                       const StatusCallback& on_status) {
    CropPair crops{std::nullopt, std::nullopt};
    if (crop_mode != CropMode::none) {
        if (on_status) {
            // Worded as for FFmpeg's metrics, because a video's tasks share
            // one detection (crop_detect waits for the other task's answer),
            // and their parts of the run line sit side by side.
            on_status({crop_mode == CropMode::source_only ? "Detecting black bars in source\xE2\x80\xA6"
                                                          : "Detecting black bars in source and distorted\xE2\x80\xA6"});
        }
        // Black bars are detected from samples across the whole file, not
        // only the part compared: a short range may fall entirely in
        // a dark intro, which must not decide the crop. Decoded in software,
        // as the Python app's perceptual metrics detect them.
        crops = crops_for(crop_mode, ffmpeg, source, distorted, cancel, std::nullopt, std::nullopt, scale_algorithm);
    }
    // As FFmpeg's metrics refuse them (validate_display_geometry).
    if (const std::optional<std::string> problem = shape_problem(source, distorted, crops.first, crops.second)) {
        throw PerceptualRunError(*problem);
    }
    return crops;
}

CropPair perceptual_crops(const PerceptualRequest& request, const StatusCallback& on_status,
                          const CancelToken& cancel) {
    const VmafOptions& options = request.options;
    if (!options.resample_test) {
        return resolve_crops(request.environment.ffmpeg, request.source, request.distorted, options.crop_mode,
                             options.scale_algorithm, cancel, on_status);
    }
    // The source's own bars, whether Auto or Source only, as VMAF's part of the test (resample_test_run)
    std::optional<CropBox> crop;
    if (options.crop_mode != CropMode::none) {
        if (on_status) {
            on_status({"Detecting black bars in source..."});
        }
        crop = crops_for(CropMode::source_only, request.environment.ffmpeg, request.source, request.source, cancel,
                         std::nullopt, std::nullopt, options.scale_algorithm)
                   .first;
    }
    if (options.resample_test->width >= resample_analysis_dimensions(request.source, crop).first) {
        throw PerceptualRunError(resample_not_smaller);
    }
    return {crop, crop};
}

PerceptualRequest with_resample_test_video(PerceptualRequest request, const std::optional<CropBox>& crop) {
    if (request.options.resample_test) {
        request.distorted.resample_via = resample_via_size(request.source, crop, request.options.resample_test->width);
    }
    return request;
}

}  // namespace vml
