#include "vml/video_playback.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string_view>

#include "vml/colour.h"
#include "vml/ffmpeg_tools.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// A file as FFmpeg is given it: its full path.
std::string resolved_text(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(path, error);
    return platform::path_text(error ? path : resolved);
}

// A number as Python's f"{value:.12g}" writes it.
std::string twelve_digits(double value) { return format_general(value, 12); }

}  // namespace

FrameComparison source_playback_comparison(const FrameComparison& comparison, bool native) {
    const VideoInfo& source = comparison.source_info;
    const std::optional<CropBox>& crop = comparison.source_crop;
    int width = crop ? crop->w : source.width;
    int height = crop ? crop->h : source.height;
    if (!native) {
        const VideoInfo& encoded = comparison.distorted_info;
        const std::optional<CropBox>& encoded_crop = comparison.distorted_crop;
        const int encoded_width = encoded_crop ? encoded_crop->w : encoded.width;
        const int encoded_height = encoded_crop ? encoded_crop->h : encoded.height;
        const double factor = std::min({1.0, static_cast<double>(encoded_width) / width,
                                        static_cast<double>(encoded_height) / height});
        if (factor < 1) {
            width = std::max(2, static_cast<int>(width * factor) / 2 * 2);
            height = std::max(2, static_cast<int>(height * factor) / 2 * 2);
        }
    }
    FrameComparison played = comparison;
    played.distorted_info = source;
    played.distorted_info.width = width;
    played.distorted_info.height = height;
    played.distorted_crop = std::nullopt;
    played.scale_direction = ScaleDirection::source_to_distorted;
    played.resample_target = std::nullopt;
    return played;
}

std::vector<int> neighbour_indices(int count, int selected, int width) {
    if (count <= 0 || selected < 0 || selected >= count) {
        return {};
    }
    std::vector<int> order{selected};
    const auto add = [&order](int index) {
        if (std::find(order.begin(), order.end(), index) == order.end()) {
            order.push_back(index);
        }
    };
    for (int step = 1; step < count; ++step) {
        add((selected + step) % count);
        add(((selected - step) % count + count) % count);
    }
    order.resize(std::min(order.size(), static_cast<std::size_t>(std::max(1, width))));
    return order;
}

Size playback_dimensions(const FrameComparison& comparison, std::optional<Size> maximum) {
    const auto [width, height] = comparison_dimensions(comparison);
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("comparison has no usable display dimensions");
    }
    if (!maximum) {
        return {std::max(2, width / 2 * 2), std::max(2, height / 2 * 2)};
    }
    const auto [max_width, max_height] = *maximum;
    const double scale = std::min({1.0, static_cast<double>(max_width) / width,
                                   static_cast<double>(max_height) / height});
    return {std::max(2, static_cast<int>(width * scale) / 2 * 2), std::max(2, static_cast<int>(height * scale) / 2 * 2)};
}

std::vector<PlaybackProcessing> playback_modes(const std::optional<std::string>& hwaccel) {
    // The GPU's own decoder first. Don't use FFmpeg's Vulkan decoder: on an
    // RTX 5090 it corrupts 2-3% of HEVC frames (FFmpeg 9.0.1, driver 616.92).
    if (!hwaccel) {
        return {PlaybackProcessing::software, PlaybackProcessing::cpu};
    }
    if (*hwaccel == "cuda") {
        return {PlaybackProcessing::interop, PlaybackProcessing::transfer, PlaybackProcessing::software,
                PlaybackProcessing::cpu};
    }
    return {PlaybackProcessing::transfer, PlaybackProcessing::software, PlaybackProcessing::cpu};
}

std::vector<std::string> build_playback_command(const std::string& ffmpeg, const FrameComparison& comparison,
                                                FrameSide side, long long start_frame,
                                                const PreviewColorSettings& colors, const HardwareDecoding& plan,
                                                std::optional<Size> maximum, PlaybackProcessing processing) {
    const double fps = comparison.fps;
    if (fps <= 0 || start_frame < 0) {
        throw std::invalid_argument("invalid playback frame rate or start frame");
    }
    const Size size = playback_dimensions(comparison, maximum);
    const bool gpu = processing != PlaybackProcessing::cpu;
    const bool interop = processing == PlaybackProcessing::interop;
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"};
    if (interop) {
        // NVDEC's pictures handed to a Vulkan device made from the CUDA one.
        command.insert(command.end(), {"-init_hw_device", "cuda=decode", "-init_hw_device", "vulkan=preview@decode",
                                       "-filter_hw_device", "preview"});
    } else if (gpu) {
        command.insert(command.end(), {"-init_hw_device", "vulkan=preview", "-filter_hw_device", "preview"});
    }
    std::optional<std::string> accel = side == FrameSide::source ? plan.source : plan.distorted;
    if (processing == PlaybackProcessing::software || processing == PlaybackProcessing::cpu) {
        accel.reset();
    }
    const VideoInfo& info = frame_video_info(comparison, side);
    command.insert(command.end(), {"-ss", format_fixed(seek_seconds(info, start_frame, fps), 9)});
    const std::vector<std::string> accel_args = hwaccel_args(accel);
    command.insert(command.end(), accel_args.begin(), accel_args.end());
    if (interop && accel == "cuda") {
        command.insert(command.end(), {"-hwaccel_device", "decode"});  // the device the Vulkan one is made from
    }
    command.insert(command.end(), {"-i", resolved_text(frame_input_path(comparison, side))});

    const std::optional<CropBox>& crop = side == FrameSide::source ? comparison.source_crop : comparison.distorted_crop;
    std::vector<std::string> ops;
    if (gpu) {
        // Only missing HDR descriptions are filled in; a transfer declared
        // SDR is never overwritten.
        const bool assumed_hdr = colors.mode == PreviewColorMode::hdr_to_sdr && info.color_transfer.empty();
        if (hdr_kind(info) || assumed_hdr) {
            struct Param {
                const char* name;
                const std::string& value;
                const char* fallback;
            };
            const Param described[] = {{"color_primaries", info.color_primaries, "bt2020"},
                                       {"color_trc", info.color_transfer, "smpte2084"},
                                       {"colorspace", info.color_space, "bt2020nc"},
                                       {"range", info.color_range, "limited"}};
            std::vector<std::string> params;
            for (const Param& param : described) {
                if (is_untagged_tag(param.value)) {
                    params.push_back(std::string(param.name) + "=" + param.fallback);
                }
            }
            if (!params.empty()) {
                std::string joined;
                for (const std::string& param : params) {
                    joined += (joined.empty() ? "" : ":") + param;
                }
                ops.push_back("setparams=" + joined);
            }
        }
        if (accel && !(interop && accel == "cuda")) {
            ops.push_back("hwdownload");
            ops.push_back("format=" + hw_native_format(info.pix_fmt));
        }
        ops.push_back("hwupload");  // NVDEC's CUDA frames, in interop: on the GPU
        std::vector<std::string> options{"w=" + std::to_string(size.first), "h=" + std::to_string(size.second),
                                         "format=rgba", "colorspace=gbr", "range=pc"};
        if (crop) {
            options.insert(options.end(), {"crop_x=" + std::to_string(crop->x), "crop_y=" + std::to_string(crop->y),
                                           "crop_w=" + std::to_string(crop->w), "crop_h=" + std::to_string(crop->h)});
        }
        if (comparison.auto_crop_pending) {
            options.insert(options.end(), {"normalize_sar=1", "fit_mode=contain"});
        }
        if (colors.mode != PreviewColorMode::unmanaged) {
            // One mapping for both sides: detecting each scene's peak on its
            // own would render an encode differently from its source.
            options.insert(options.end(), {"color_primaries=bt709", "color_trc=iec61966-2-1", "tonemapping=bt.2390",
                                           "peak_detect=0"});
        }
        std::string joined;
        for (const std::string& option : options) {
            joined += (joined.empty() ? "" : ":") + option;
        }
        ops.insert(ops.end(), {"libplacebo=" + joined, "hwdownload", "format=rgba"});
    } else {
        ops.insert(ops.end(), {frame_filter(comparison, side, colors, size), "format=rgba"});
    }
    const std::string rate = twelve_digits(fps);
    ops.insert(ops.end(), {"fps=" + rate, "setpts=N/(" + rate + "*TB)", "setsar=1"});
    std::string chain;
    for (const std::string& op : ops) {
        chain += (chain.empty() ? "" : ",") + op;
    }
    const std::string graph = "[0:" + std::string(video_stream) + "]split=1[in0];[in0]" + chain + "[tile0];[tile0]null[out]";
    command.insert(command.end(), {"-filter_complex", graph, "-map", "[out]", "-an", "-sn", "-dn", "-pix_fmt", "rgba",
                                   "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"});
    return command;
}

std::optional<std::filesystem::path> ffplay_beside(const std::string& ffmpeg) {
    const std::filesystem::path program = platform::path_from_text(ffmpeg);
    const bool exe = ascii_lower(platform::path_text(program.extension())) == ".exe";
    const std::filesystem::path candidate = program.parent_path() / (exe ? "ffplay.exe" : "ffplay");
    std::error_code error;
    return std::filesystem::is_regular_file(candidate, error) ? std::optional(candidate) : std::nullopt;
}

std::optional<std::vector<std::string>> build_audio_command(const std::filesystem::path& ffplay,
                                                            const FrameComparison& comparison,
                                                            long long start_frame) {
    if (comparison.fps <= 0) {
        return std::nullopt;
    }
    // From the file's start, as the frames are (seek_seconds).
    const double timestamp = comparison.source_info.start_offset + static_cast<double>(start_frame) / comparison.fps;
    return std::vector<std::string>{platform::path_text(ffplay), "-nodisp", "-autoexit", "-loglevel", "error", "-ss",
                                    format_fixed(timestamp, 9), "-i",
                                    resolved_text(frame_input_path(comparison, FrameSide::source)), "-vn", "-sn"};
}

}  // namespace vml
