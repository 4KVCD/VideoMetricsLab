#include "vml/perceptual.h"

#include <algorithm>
#include <cmath>

#include "vml/crop_detect.h"
#include "vml/geometry.h"
#include "vml/metrics.h"

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

namespace {

// The seconds per megapixel of a compared frame pair for SSIMULACRA2, with
// Butteraugli or without, on the CPU, measured with the app's engines on a
// 24-core CPU (Core Ultra 9 285K), 48 pairs of a 3840x2160 (8.29 MP) video:
// 0.34 s a pair with Butteraugli (16.2 s alone, 16.4 s with SSIMULACRA2
// beside it), 0.17 s for SSIMULACRA2 alone (8.2 s). A rough guide, as CPUs
// differ.
double cpu_seconds_per_megapixel(bool butteraugli) { return (butteraugli ? 0.34 : 0.17) / 8.29; }

}  // namespace

std::optional<LongCpuRun> long_cpu_run(const std::vector<std::string>& keys, const VideoInfo& source,
                                       const VideoInfo& test, const VmafOptions& options) {
    LongCpuRun run;
    for (const char* key : {"ssimulacra2", "butteraugli"}) {
        if (std::find(keys.begin(), keys.end(), key) != keys.end()) {
            run.keys.emplace_back(key);
        }
    }
    if (run.keys.empty()) {
        return std::nullopt;
    }
    double seconds = std::min(test.duration, source.duration);
    if (const double fps = source.fps; fps > 0) {  // the range's part of them
        const auto [first, end] = compared_frames(options.compared_range(), fps);
        if (end) {
            seconds = std::min(seconds, static_cast<double>(*end) / fps);
        }
        seconds -= static_cast<double>(first) / fps;
    }
    if (seconds <= long_cpu_run_seconds) {
        return std::nullopt;
    }
    const double frames = seconds * (test.fps > 0 ? test.fps : 24.0) / std::max(1, options.n_subsample);
    const double megapixels = static_cast<double>(test.width) * test.height / 1e6;
    run.video_seconds = seconds;
    run.scoring_seconds = frames * megapixels * cpu_seconds_per_megapixel(run.keys.back() == "butteraugli");
    return run;
}

std::pair<long long, std::string_view> rough_time(double seconds) {
    if (seconds >= 2 * 86400) {
        return {static_cast<long long>(std::nearbyint(seconds / 86400)), "day"};
    }
    if (seconds >= 2 * 3600) {
        return {static_cast<long long>(std::nearbyint(seconds / 3600)), "hour"};
    }
    return {std::max(1LL, static_cast<long long>(std::nearbyint(seconds / 60))), "minute"};
}

void validate_pair(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options) {
    if (const std::optional<std::string> problem =
            pair_problem(source, distorted, options.compared_range())) {
        throw PerceptualRunError(*problem);
    }
}

CropPair resolve_crops(const std::string& ffmpeg, const VideoInfo& source, const VideoInfo& distorted,
                       const CropChoice& crop, const std::string& scale_algorithm, const CancelToken& cancel,
                       const StatusCallback& on_status) {
    if (on_status && detects_black_bars(crop.mode)) {
        // Worded as for FFmpeg's metrics, because a video's tasks share
        // one detection (crop_detect waits for the other task's answer),
        // and their parts of the run line sit side by side.
        on_status({crop.mode == CropMode::source_only ? "Detecting black bars in source\xE2\x80\xA6"
                                                      : "Detecting black bars in source and distorted\xE2\x80\xA6"});
    }
    // Black bars are detected from samples across the whole file, not
    // only the part compared: a short range may fall entirely in
    // a dark intro, which must not decide the crop. Decoded in software,
    // as the Python app's perceptual metrics detect them.
    CropPair crops{std::nullopt, std::nullopt};
    try {
        crops = crops_for(crop, ffmpeg, source, distorted, cancel, std::nullopt, std::nullopt, scale_algorithm);
    } catch (const CropEntryError& error) {
        throw PerceptualRunError(error.what());
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
        return resolve_crops(request.environment.ffmpeg, request.source, request.distorted, options.crop_choice(),
                             options.scale_algorithm, cancel, on_status);
    }
    // The source's own crop, as VMAF's part of the test (resample_test_run)
    if (on_status && detects_black_bars(options.crop_mode)) {
        on_status({"Detecting black bars in source..."});
    }
    std::optional<CropBox> crop;
    try {
        crop = source_own_crop(options.crop_choice(), request.environment.ffmpeg, request.source, cancel,
                               std::nullopt, options.scale_algorithm);
    } catch (const CropEntryError& error) {
        throw PerceptualRunError(error.what());
    }
    if (options.resample_test->width >= resample_analysis_dimensions(request.source, crop).first) {
        throw PerceptualRunError(resample_not_smaller);
    }
    return {crop, crop};
}

void refuse_too_small(const PerceptualRequest& request, const std::vector<std::string>& keys, const CropPair& crops) {
    const auto [width, height] = compared_dimensions(request.source, request.distorted,
                                                     request.options.scale_direction, crops.first, crops.second);
    if (const std::optional<std::string> problem = too_small_problem(keys, width, height)) {
        throw PerceptualRunError(*problem);
    }
}

PerceptualRequest with_resample_test_video(PerceptualRequest request, const std::optional<CropBox>& crop) {
    if (request.options.resample_test) {
        request.distorted.resample_via = resample_via_size(request.source, crop, request.options.resample_test->width);
    }
    return request;
}

}  // namespace vml
