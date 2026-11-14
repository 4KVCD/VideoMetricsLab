// Detecting black bars (letterbox and pillarbox) as HandBrake does
// (libhb/scan.c, black_bars_rule.h): ten previews across the video, each
// judged on one picture, line by line: a line is a bar's when it is dark and
// even (not a bright detail in black, as a packed alpha matte is), and the
// median of each edge's bars wins. A video's previews run one after another;
// the two videos of a comparison are detected at the same time. The answer
// for a file is kept, in this process and between sessions (black_bars.json
// beside the saved scores, the Python app's file).
// A video of a codec the app's own decoder decodes (app_decoders.h: VVC, by
// VVdeC) is sampled from that decoder's pictures (samples_through_the_app).
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "vml/cancel.h"
#include "vml/video.h"

namespace vml {

// Black bars could not be detected; the message says why, as the Python
// app's does.
class CropDetectError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Keeps the answers between sessions in `store` (a folder can move, so it is
// asked for each time); nothing keeps them only in this process.
void set_crop_store(std::function<std::filesystem::path()> store);

// The store's file, kept with the saved scores.
inline constexpr std::string_view black_bars_file = "black_bars.json";

// Forgets the answers this process holds, and reads the store again when
// next asked: clearing the saved results, which deletes the store too, then
// starts afresh.
void forget_crop_answers();

// Forgets the answer kept for one video, in the store too: a run that found
// it from the GPU's pictures while the GPU's video decoding was found wrong
// (gpu_decoding_distrusted) finds it again on the CPU.
void forget_crop_answer(const VideoInfo& info);

// The box that cuts the video's black bars. `hwaccel` is the decoder the
// run will use for this video (only how fast the answer comes depends on
// it). A video whose size changes (VideoInfo::scaled_to) is sampled scaled
// with `scale_algorithm`, as the run scales it, and its box is not kept.
// Throws CropDetectError when it cannot be found, Cancelled, and
// VideoSizeChanged (size_changes.h) when FFmpeg rebuilt its filters partway
// through a sample.
CropBox detect_crop(const std::string& ffmpeg, const VideoInfo& info, const CancelToken& cancel,
                    const std::optional<std::string>& hwaccel, const std::string& scale_algorithm);

// Whether a video's black bars are sampled from the pictures the app's own
// decoder makes, fed to FFmpeg (grain_feed.h), rather than from
// FFmpeg decoding the file: a codec app_decoders.h lists (FFmpeg's vvc fails
// on VVenC's --fga 3 encodes at 10 bits, so their bars were not found). Not a
// raw stream (sampled from bytes, not seeked to). A preview the app cannot
// feed is sampled by FFmpeg where FFmpeg decodes the codec, and fails where
// it does not.
bool samples_through_the_app(const VideoInfo& info);

// Both videos' boxes, detected at the same time.
std::pair<CropBox, CropBox> detect_pair(const std::string& ffmpeg, const VideoInfo& source,
                                        const VideoInfo& distorted, const CancelToken& cancel,
                                        const std::optional<std::string>& source_hwaccel,
                                        const std::optional<std::string>& distorted_hwaccel,
                                        const std::string& scale_algorithm);

// The two videos' boxes, made to cover the same picture where the two show
// the same picture at one size or two (an encode's soft bar edge can put its
// box a row from the source's): the boxes' overlap, in each video's own
// pixels (as scaled_to, where set), on even samples. Where the sizes are one exact fraction apart
// (3840x2160 and 2560x1440), the source's box is the test video's scaled by it, so that the scaled
// test picture lines up with the source's row for row. Boxes that agree, or that are not the same
// picture, are kept.
std::pair<std::optional<CropBox>, std::optional<CropBox>> common_picture(const VideoInfo& source,
                                                                         const VideoInfo& distorted,
                                                                         std::optional<CropBox> source_crop,
                                                                         std::optional<CropBox> distorted_crop);

// What Source only cuts from the test video, which is compared at its full frame:
// nothing, but for the odd line a box on even samples leaves out of an odd-sized
// frame. Kept, it would stretch the source by a pixel to fit, moving every line.
std::optional<CropBox> test_crop_beside_source_box(const CropBox& source_box, const VideoInfo& test);

// The box a crop entered by hand (`edges`) leaves of `video`'s frame (as scaled_to, where set): none where nothing
// is cut. Its size is rounded down to even, as a detected box's is. Throws CropEntryError, naming the video as
// `which` ("the source"), where an edge is negative or odd (4:2:0's colour comes in pairs of lines, and FFmpeg's
// crop would move an odd one silently) or less than 16x16 of the picture is left.
std::optional<CropBox> box_from_edges(const CropEdges& edges, const VideoInfo& video, std::string_view which);

// What `box` cuts from each edge of `video`'s frame, each rounded down to even: the crop box_from_edges gives
// `box` back from (an odd-sized frame's box leaves an odd line at its right or bottom).
CropEdges edges_of_box(const CropBox& box, const VideoInfo& video);

// A crop as the command line takes it, "top:bottom:left:right" in lines (HandBrake's order): none where it is not
// four whole numbers of 0 or more.
std::optional<CropEdges> parse_crop_edges(std::string_view text);

// A crop written as parse_crop_edges reads it ("22:22:0:0").
std::string crop_edges_text(const CropEdges& edges);

// The source's own crop, for a resolution test (both of whose videos come
// from the source): the crop entered for it (manual), its detected box (Auto
// and Source only: detect_crop), or none.
std::optional<CropBox> source_own_crop(const CropChoice& choice, const std::string& ffmpeg, const VideoInfo& source,
                                       const CancelToken& cancel, const std::optional<std::string>& hwaccel,
                                       const std::string& scale_algorithm);

// The crop entered for a video cannot be cut from it (box_from_edges).
struct CropEntryError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// The two videos' crops as `choice` asks: none for CropMode::none; for automatic, both videos' boxes (detect_pair)
// made to cover the same picture (common_picture); for source_only, the source's box (detect_crop) and the test
// video's full frame, as an encode cropped when it was made shows the source's picture without its bars; for
// manual, each video's box from the crop entered for it (box_from_edges), nothing detected and the two boxes kept
// as asked.
std::pair<std::optional<CropBox>, std::optional<CropBox>> crops_for(const CropChoice& choice, const std::string& ffmpeg,
                                                                    const VideoInfo& source,
                                                                    const VideoInfo& distorted,
                                                                    const CancelToken& cancel,
                                                                    const std::optional<std::string>& source_hwaccel,
                                                                    const std::optional<std::string>& distorted_hwaccel,
                                                                    const std::string& scale_algorithm);

}  // namespace vml
