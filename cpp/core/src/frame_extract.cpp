#include "vml/frame_extract.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <tuple>

#include "vml/app_log.h"
#include "vml/colour.h"
#include "vml/ffmpeg_tools.h"
#include "vml/film_grain_scoring.h"
#include "vml/grain_feed.h"
#include "vml/hardware_decoding.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.frame_extract");

// The filters that convert a frame to 8-bit RGB for the window, mapping HDR
// to SDR where the mode says.
std::vector<std::string> tone_map_filters(const VideoInfo& info, const PreviewColorSettings& colors) {
    const std::optional<std::string> kind = hdr_kind(info);
    if (colors.mode == PreviewColorMode::unmanaged ||
        (colors.mode == PreviewColorMode::display_aware && !kind)) {
        return {"format=rgb24"};
    }
    // HDR → SDR doubles as the way to show an HDR file whose container lost
    // its colour tags: it assumes PQ (its label says so). Display-aware
    // never guesses from the bit depth, as 10-bit SDR is common.
    std::vector<std::string> options;
    if (!kind) {
        options = {"pin=bt2020", "tin=smpte2084", "min=bt2020nc", "rin=tv"};
    } else {
        // A remux can keep the PQ or HLG tag and lose the primaries, the
        // matrix or the range, and zscale refuses an unspecified conversion:
        // BT.2020's HDR defaults stand in for only what is missing.
        const std::string primaries = is_untagged_tag(info.color_primaries) ? "bt2020" : info.color_primaries;
        const std::string matrix = is_untagged_tag(info.color_space) ? "bt2020nc" : info.color_space;
        static const std::set<std::string, std::less<>> ranges{"tv", "pc", "limited", "full"};
        const std::string range = ranges.contains(info.color_range) ? info.color_range : "tv";
        options = {"pin=" + primaries, "tin=" + info.color_transfer, "min=" + matrix, "rin=" + range};
    }
    options.push_back("t=linear");
    options.push_back("npl=" + format_general(colors.target_nits()));
    std::string linear;
    for (const std::string& option : options) {
        linear += (linear.empty() ? "" : ":") + option;
    }
    return {"zscale=" + linear, "format=gbrpf32le", "tonemap=mobius:desat=2",
            "zscale=p=bt709:t=bt709:m=bt709:r=tv:dither=error_diffusion", "format=rgb24"};
}

}  // namespace

std::string_view preview_color_mode_value(PreviewColorMode mode) {
    switch (mode) {
        case PreviewColorMode::hdr_to_sdr: return "hdr_to_sdr";
        case PreviewColorMode::unmanaged: return "unmanaged";
        case PreviewColorMode::display_aware: break;
    }
    return "display_aware";
}

PreviewColorMode preview_color_mode_from(std::string_view value) {
    if (value == "hdr_to_sdr") {
        return PreviewColorMode::hdr_to_sdr;
    }
    if (value == "unmanaged") {
        return PreviewColorMode::unmanaged;
    }
    return PreviewColorMode::display_aware;
}

double PreviewColorSettings::target_nits() const {
    // Implausible values from the driver are ignored rather than given to
    // FFmpeg.
    if (mode == PreviewColorMode::display_aware && display_hdr_enabled == true && display_sdr_white_nits &&
        *display_sdr_white_nits >= 40 && *display_sdr_white_nits <= 1000) {
        return *display_sdr_white_nits;
    }
    return 100.0;
}

std::string PreviewColorSettings::cache_token() const {
    const std::string hdr = !display_hdr_enabled ? "unknown" : *display_hdr_enabled ? "on" : "off";
    return std::string(preview_color_mode_value(mode)) + "|" + hdr + "|" + format_fixed(target_nits(), 3);
}

FrameComparison FrameComparison::from_result(const ComparisonResult& result) {
    FrameComparison comparison;
    // A video scored scaled to one size (size_changes.h) is shown as decoded.
    std::tie(comparison.source_info, comparison.source_crop) = at_own_size(result.source_info, result.source_crop);
    std::tie(comparison.distorted_info, comparison.distorted_crop) =
        at_own_size(result.distorted_info, result.distorted_crop);
    comparison.scale_direction = result.scale_direction;
    comparison.scale_algorithm = result.scale_algorithm;
    comparison.resample_target = result.resample_target;
    comparison.fps = result.fps;
    comparison.frame_count = result.compared_frame_count;
    return comparison;
}

Size comparison_dimensions(const FrameComparison& comparison) {
    const Size source = content_size(comparison.source_info, comparison.source_crop);
    if (comparison.resample_target) {
        return source;
    }
    const Size distorted = content_size(comparison.distorted_info, comparison.distorted_crop);
    if (source == distorted || comparison.scale_direction == ScaleDirection::distorted_to_source) {
        return source;
    }
    return distorted;
}

const std::filesystem::path& frame_input_path(const FrameComparison& comparison, FrameSide side) {
    return frame_video_info(comparison, side).path;
}

const VideoInfo& frame_video_info(const FrameComparison& comparison, FrameSide side) {
    return side == FrameSide::source || comparison.resample_target ? comparison.source_info
                                                                   : comparison.distorted_info;
}

double seek_seconds(const VideoInfo& info, long long frame, double fps) {
    return std::max(0.0, info.start_offset + (static_cast<double>(frame) - 0.125) / fps);
}

double keyframe_lead(const std::string& ffmpeg, const std::filesystem::path& path, const CancelToken& cancel) {
    // Kept by the file's path, size and time: a file written again is read again.
    std::error_code error;
    const std::string key = platform::path_text(path) + '|' + std::to_string(std::filesystem::file_size(path, error)) +
                            '|' + std::to_string(platform::modified_ns(path).value_or(0));
    static std::mutex mutex;
    static std::map<std::string, double> leads;
    {
        const std::lock_guard lock(mutex);
        if (const auto found = leads.find(key); found != leads.end()) {
            return found->second;
        }
    }
    // FFmpeg's framecrc lists each packet copied as "stream, dts, pts, ...",
    // in the time base its "#tb 0: N/D" line gives.
    platform::ChildProcess process = platform::ChildProcess::start(
        {ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", input_path(path), "-map",
         "0:" + std::string(video_stream), "-c", "copy", "-frames:v", "200", "-f", "framecrc", "-"});
    std::string listing;
    {
        const CancelToken::Attached attached = cancel.attach(process);
        std::vector<char> buffer(1 << 16);
        while (const std::size_t count = process.read(buffer.data(), buffer.size())) {
            listing.append(buffer.data(), count);
        }
        process.wait();
    }
    cancel.throw_if_cancelled();
    double time_base = 0.0;
    long long ticks = 0;
    std::istringstream lines(listing);
    for (std::string line; std::getline(lines, line);) {
        const std::string_view text(line);
        if (text.starts_with("#tb 0:")) {
            const std::string_view rate = text.substr(6);
            const std::size_t slash = rate.find('/');
            const std::optional<long long> numerator = python_int(rate.substr(0, slash));
            const std::optional<long long> denominator =
                slash == std::string_view::npos ? std::nullopt : python_int(rate.substr(slash + 1));
            if (numerator && denominator && *denominator > 0) {
                time_base = static_cast<double>(*numerator) / static_cast<double>(*denominator);
            }
        } else if (text.starts_with("0,")) {
            const std::size_t dts_end = text.find(',', 2);
            const std::size_t pts_end = dts_end == std::string_view::npos ? dts_end : text.find(',', dts_end + 1);
            if (pts_end != std::string_view::npos) {
                const std::optional<long long> dts = python_int(text.substr(2, dts_end - 2));
                const std::optional<long long> pts = python_int(text.substr(dts_end + 1, pts_end - dts_end - 1));
                if (dts && pts) {
                    ticks = std::max(ticks, *pts - *dts);
                }
            }
        }
    }
    // A file FFmpeg cannot read this way (0) is not read again either.
    const double lead = static_cast<double>(ticks) * time_base;
    const std::lock_guard lock(mutex);
    leads.emplace(key, lead);
    return lead;
}

SeekPoint seek_point(double seconds, double lead) {
    // FFmpeg's command line already seeks a video with B-frames 3/23 s
    // (0.13 s) earlier, enough for a lead up to 0.1 s. A longer one (VVenC's
    // random access, an open GOP of 32 frames: 1.2 s) lands -ss on the
    // keyframe after the frame, whose leading frames, the frame among them,
    // cannot be decoded from it: this seeks that much earlier, and the trim
    // filter drops the frames before the frame.
    constexpr double allowance = 0.1;
    const double early = lead > allowance ? std::min(seconds, lead) : 0.0;
    return {seconds - early, early};
}

std::optional<std::string> trim_filter(const SeekPoint& start) {
    if (start.skip <= 0.0) {
        return std::nullopt;
    }
    return "trim=start=" + format_fixed(start.skip, 9);
}

std::optional<std::string> hdr_kind(const VideoInfo& info) {
    const std::string transfer = ascii_lower(python_strip(info.color_transfer));
    if (transfer == "smpte2084" || transfer == "smpte-st-2084") {
        return "HDR10 / PQ";
    }
    if (transfer == "arib-std-b67" || transfer == "hlg") {
        return "HLG";
    }
    return std::nullopt;
}

bool would_distort(const FrameComparison& comparison, const VideoInfo& info, const std::optional<CropBox>& crop,
                   int output_width, int output_height) {
    // Only for black bars still to be detected: once a run has measured
    // them, the two sides agree by construction.
    if (!comparison.auto_crop_pending || output_height <= 0) {
        return false;
    }
    const double shape = display_aspect_ratio(info, crop);
    const double target = static_cast<double>(output_width) / output_height;
    if (shape <= 0 || target <= 0) {
        return false;
    }
    return std::abs(shape - target) > 0.01 * std::max(shape, target);
}

std::string frame_filter(const FrameComparison& comparison, FrameSide side, const PreviewColorSettings& colors,
                         std::optional<Size> output_size) {
    const VideoInfo& info = side == FrameSide::source ? comparison.source_info : comparison.distorted_info;
    const std::optional<CropBox>& crop =
        side == FrameSide::source ? comparison.source_crop : comparison.distorted_crop;
    const auto [output_width, output_height] = output_size.value_or(comparison_dimensions(comparison));
    std::vector<std::string> filters;
    // The scorer's order: each side cropped before the common pixel format.
    if (crop && !crop->is_noop(info.width, info.height)) {
        filters.push_back(crop->as_filter());
    }
    std::vector<std::string> formats{comparison.source_info.pix_fmt};
    if (!comparison.resample_target) {
        formats.push_back(comparison.distorted_info.pix_fmt);
    }
    filters.push_back("format=" + analysis_pix_fmt(formats));

    const Size content = content_size(info, crop);
    const std::string algorithm = ":flags=" + comparison.scale_algorithm;
    const std::string output = std::to_string(output_width) + ":" + std::to_string(output_height);
    if (comparison.resample_target && side == FrameSide::distorted) {
        const int width = comparison.resample_target->width;
        filters.push_back("scale=" + std::to_string(width) + ":" +
                          std::to_string(resample_down_height(width, content)) + algorithm);
        filters.push_back("scale=" + output + algorithm);
    } else if (content != Size{output_width, output_height}) {
        if (would_distort(comparison, info, crop, output_width, output_height)) {
            // Fitted, not stretched, while the black bars are still being
            // detected: a letterboxed source is 16:9 until its bars come off,
            // and scaled to the shape of a cropped 2.35:1 encode it would be
            // squashed. Padded, both stay on one canvas to flip between.
            filters.push_back("scale=" + output + algorithm + ":force_original_aspect_ratio=decrease");
            filters.push_back("pad=" + output + ":(ow-iw)/2:(oh-ih)/2");
        } else {
            filters.push_back("scale=" + output + algorithm);
        }
    }
    // A PNG has square pixels: reset after the geometry, so that both sides
    // fill the same canvas when a file is stored anamorphic.
    filters.push_back("setsar=1");
    for (std::string& filter : tone_map_filters(frame_video_info(comparison, side), colors)) {
        filters.push_back(std::move(filter));
    }
    std::string chain;
    for (const std::string& filter : filters) {
        chain += (chain.empty() ? "" : ",") + filter;
    }
    return chain;
}

namespace {

// Where `frame` of `side` is decoded from (seek_point). Throws as
// build_frame_command does.
SeekPoint frame_start(const FrameComparison& comparison, FrameSide side, long long frame, double lead) {
    if (frame < 0) {
        throw std::invalid_argument("frame number must be non-negative");
    }
    if (comparison.fps <= 0) {
        throw FrameExtractError("This video has no usable frame rate.");
    }
    return seek_point(seek_seconds(frame_video_info(comparison, side), frame, comparison.fps), lead);
}

// A still's command after its input: the first frame through `filters`, as a
// PNG on standard output.
void add_png_output(std::vector<std::string>& command, const std::string& filters) {
    command.insert(command.end(), {"-map", "0:" + std::string(video_stream), "-an", "-sn", "-dn", "-vf", filters,
                                   "-frames:v", "1", "-f", "image2pipe",
                                   // Stored, not compressed: the same pixels, 25 MB for 4K
                                   // rather than 2-8, but written 85-340 ms sooner and read
                                   // in 27 ms rather than 64-104.
                                   "-c:v", "png", "-compression_level", "0", "-pred", "none", "pipe:1"});
}

// What a still's FFmpeg wrote: its standard output, exit code and error text.
struct StillOutput {
    std::string png;
    int exit_code = 0;
    std::string error;
};

StillOutput run_still(const std::vector<std::string>& command, const CancelToken& cancel) {
    platform::ChildProcess process = platform::ChildProcess::start(command);
    StillOutput output;
    const CancelToken::Attached attached = cancel.attach(process);
    std::vector<char> buffer(1 << 20);
    while (const std::size_t count = process.read(buffer.data(), buffer.size())) {
        output.png.append(buffer.data(), count);
    }
    output.exit_code = process.wait();
    if (output.exit_code != 0) {
        output.error = std::string(python_strip(replace_invalid_utf8(process.standard_error())));
    }
    return output;
}

// The PNG; FrameExtractError with FFmpeg's error, or for no frame there.
std::string still_png(StillOutput output, long long frame, const CancelToken& cancel) {
    if (output.exit_code != 0) {
        cancel.throw_if_cancelled();
        throw FrameExtractError(output.error.empty()
                                    ? "ffmpeg exited with code " + platform::exit_code_text(output.exit_code) + "."
                                    : last_characters(output.error, 2000));
    }
    if (output.png.empty()) {
        throw FrameExtractError("No frame was returned at frame " + std::to_string(frame) +
                                "; it may be beyond the end of the video.");
    }
    return std::move(output.png);
}

// The still made from the app's decoder's pictures, with the video's grain
// as it is scored: the pictures from the seek point fed to FFmpeg, those
// before the frame left out as the trim filter leaves them out, and the
// first one after shown. Throws FilmGrainRefusal where the app could not
// feed them.
std::string fed_still(const std::string& ffmpeg, const FrameComparison& comparison, FrameSide side, long long frame,
                      const CancelToken& cancel, const PreviewColorSettings& colors, const SeekPoint& start) {
    // VVC's decoder asks ffprobe nothing (gpu_frames.cpp's configuration).
    PictureFeed feed(frame_video_info(comparison, side), ffmpeg, "", cancel, FeedWindow{start.seek, 0, start.skip});
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"};
    command.insert(command.end(), feed.input_arguments().begin(), feed.input_arguments().end());
    add_png_output(command, frame_filter(comparison, side, colors));
    StillOutput output = run_still(command, cancel);
    feed.finish();  // a feeding that failed gives no still
    return still_png(std::move(output), frame, cancel);
}

}  // namespace

bool still_from_the_app(const VideoInfo& info) { return windows_from_the_app(info) && !decoding_refusal(info); }

std::string still_grain(const VideoInfo& info) {
    if (still_from_the_app(info)) {
        return "the app's decoder, grain " + std::to_string(decoder_film_grain(effective_film_grain(info)));
    }
    return picture_grain_options(info).empty() ? "FFmpeg, grain added" : "FFmpeg, grain left out";
}

std::vector<std::string> build_frame_command(const std::string& ffmpeg, const FrameComparison& comparison,
                                             FrameSide side, long long frame, const PreviewColorSettings& colors,
                                             double lead) {
    const SeekPoint start = frame_start(comparison, side, frame, lead);
    std::string filters = frame_filter(comparison, side, colors);
    if (const std::optional<std::string> trim = trim_filter(start)) {
        filters = *trim + "," + filters;
    }
    std::error_code error;
    const std::filesystem::path input = std::filesystem::weakly_canonical(frame_input_path(comparison, side), error);
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"};
    const std::vector<std::string> grain = picture_grain_options(frame_video_info(comparison, side));
    command.insert(command.end(), grain.begin(), grain.end());
    command.insert(command.end(), {"-ss", format_fixed(start.seek, 9), "-i",
                                   platform::path_text(error ? frame_input_path(comparison, side) : input)});
    add_png_output(command, filters);
    return command;
}

std::string extract_frame_png(const std::string& ffmpeg, const FrameComparison& comparison, FrameSide side,
                              long long frame, const CancelToken& cancel, const PreviewColorSettings& colors) {
    const std::filesystem::path& path = frame_input_path(comparison, side);
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        throw FrameExtractError("Video file is missing: " + platform::path_text(path));
    }
    cancel.throw_if_cancelled();
    const double lead = keyframe_lead(ffmpeg, path, cancel);
    const VideoInfo& info = frame_video_info(comparison, side);
    if (still_from_the_app(info)) {
        const SeekPoint start = frame_start(comparison, side, frame, lead);
        try {
            return fed_still(ffmpeg, comparison, side, frame, cancel, colors, start);
        } catch (const FilmGrainRefusal& refusal) {
            // Not FFmpeg's picture instead: the still is labelled and kept as
            // the app's decoder's, and scoring the video would fail too.
            throw FrameExtractError(std::string("The app's decoder could not decode this frame: ") + refusal.what());
        }
    }
    if (windows_from_the_app(info)) {  // refused before any feeding: decoding_refusal
        const FilmGrainRefusal refusal(decoding_refusal(info).value_or("the app's decoder cannot decode it here"));
        if (const std::optional<std::string> failure = unfed_failure(info, refusal)) {
            throw FrameExtractError(*failure);
        }
        logger.info("A still of " + platform::path_text(path.filename()) +
                    " decoded by FFmpeg, as the app's decoder could not feed it (" + refusal.what() + ")");
    }
    return still_png(run_still(build_frame_command(ffmpeg, comparison, side, frame, colors, lead), cancel), frame,
                     cancel);
}

}  // namespace vml
