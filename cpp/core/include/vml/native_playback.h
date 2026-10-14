// Video Compare's native playback, as the Python app's GStreamer playback
// decides it (vmaf_app/core/gstreamer_playback.py, locked_presentation.py,
// d3d11_tonemap.py): which comparisons play natively, and how the
// presenter's shader shows each video's colours. The decoding and the
// presenting themselves are the window's (cpp/app/native_*).
#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>

#include "vml/frame_extract.h"

namespace vml {

// Why a comparison cannot play natively, in the Python app's words, or
// nothing when it can: unmanaged colours, HDR forced onto untagged video,
// and HDR primaries the shader does not convert play through FFmpeg.
std::optional<std::string> native_playback_problem(const FrameComparison& comparison,
                                                   const PreviewColorSettings& colors);

// Whether HDR video is mapped to SDR: when the user chose HDR to SDR, or when
// the display is not known to be in HDR mode.
bool needs_sdr_tonemap(const PreviewColorSettings& colors);

// An HDR video's primaries by FFmpeg's name, untagged taken as BT.2020.
std::string hdr_primaries(std::string_view tag);

// Whether the shader converts HDR video of these primaries (BT.709, BT.2020
// and Display P3, all with D65 white).
bool shader_supports(std::string_view primaries);

// What the shader does to one video's frames: nothing; HDR mapped to SDR
// BT.709; or HDR kept for an HDR display and converted to the PQ in BT.2020
// it takes (BT.2020 PQ itself is passed through as it comes).
enum class NativeShading { none, sdr, hdr };
NativeShading native_shading_mode(const VideoInfo& info, const PreviewColorSettings& colors);

// The shader's parameters for HDR video of `kind` ("HDR10 / PQ" or "HLG") in
// `primaries`, as vmaf_present_texture takes them: 1 to map to SDR BT.709 or
// 2 to keep HDR as PQ in BT.2020; the transfer, 1 PQ or 2 HLG; the source
// primaries' luminance weights (3); linear source RGB to the target's by rows
// (9); 1000 nits mapped to SDR white (HLG's display peak too); SDR white, 100.
std::array<float, 16> shading_parameters(std::string_view kind, std::string_view primaries, bool hdr);

// The presenter's YUV-to-RGB conversion for a frame's planes as its shader
// samples them (NV12's 8-bit codes / 255, P010's 10-bit codes in the top of
// 16 bits / 65535): three rows of Y, U and V coefficients and an offset.
// `matrix` is GStreamer's name for the YUV matrix (bt709, bt601, bt2020,
// smpte240m, fcc; BT.709 for any other).
std::array<float, 12> yuv_to_rgb(std::string_view matrix, bool full_range, bool ten_bit);

// GStreamer's name for a frame's YUV matrix, from FFmpeg's name for it
// (AVFrame's colorspace); an unknown one taken as GStreamer takes it for a
// frame `height` rows high: BT.601 up to 576 rows, BT.709 above.
std::string_view yuv_matrix_name(std::string_view ffmpeg_space, int height);

}  // namespace vml
