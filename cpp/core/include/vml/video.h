// A video's details (VideoInfo), black-bar crops (CropBox), and the choices
// that say how two videos are lined up. A port of the parts of
// vmaf_app/core/models.py they come from.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vml {

// Whether black bars are detected and cut (automatic), kept (none), detected and cut in the source alone, the
// test video's full frame compared (source_only: an encode cropped when it was made), or cut as entered by hand,
// nothing detected (manual: VmafOptions' manual crops). New ones go last: a mode goes between processes as its
// number.
enum class CropMode { automatic, none, source_only, manual };

// The mode as the saved scores' identity and the log name it: "auto", "none", "source" or "manual".
const char* crop_mode_text(CropMode mode);

// Whether the mode finds black bars by looking at the pictures (Auto and Source only).
inline bool detects_black_bars(CropMode mode) {
    return mode == CropMode::automatic || mode == CropMode::source_only;
}

// The GPU maker whose decoder to use: automatic picks one found on this PC,
// and none decodes in software.
enum class GpuVendor { automatic, nvidia, intel, amd, none };

// The maker of a GPU with this PCI vendor ID: none for a maker not known.
GpuVendor vendor_of_pci_id(unsigned id);

// The maker's name as the logs and the window write it: "NVIDIA", "INTEL" or
// "AMD" (empty for automatic and none).
std::string_view vendor_label(GpuVendor vendor);
// The maker's name as the window's GPU decode list and the command line's
// messages give it: "NVIDIA", "Intel" or "AMD" (empty for automatic and none).
std::string_view maker_name(GpuVendor vendor);

// Which video is scaled when the two sizes differ.
enum class ScaleDirection {
    // The default: the source is scaled to the test video's size, which
    // judges quality at the size actually delivered.
    source_to_distorted,
    // The test video is scaled up to the source's size, which judges quality
    // as if it were played at the source's size.
    distorted_to_source,
};

// A crop rectangle in the video's own pixels.
struct CropBox {
    int w = 0;
    int h = 0;
    int x = 0;
    int y = 0;

    // The FFmpeg crop filter, "crop=w:h:x:y".
    std::string as_filter() const;
    // Whether it keeps the whole of a frame this size.
    bool is_noop(int frame_width, int frame_height) const;
    bool operator==(const CropBox&) const = default;
};

// A crop entered by hand (CropMode::manual): the lines cut from each edge of the frame, as HandBrake's crop fields
// take them. Edges rather than a box, so that one crop fits videos of any size.
struct CropEdges {
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;

    bool none() const { return top == 0 && bottom == 0 && left == 0 && right == 0; }
    bool operator==(const CropEdges&) const = default;
};

// How a run crops its two videos: the mode and, for CropMode::manual, the crops entered for each (nothing cut
// where they are none).
struct CropChoice {
    CropMode mode = CropMode::automatic;
    CropEdges source;
    CropEdges test;

    CropChoice() = default;
    explicit CropChoice(CropMode crop_mode, CropEdges source_crop = {}, CropEdges test_crop = {})
        : mode(crop_mode), source(source_crop), test(test_crop) {}
};

// A video file's details, as ffprobe reports them (ffprobe.h).
struct VideoInfo {
    std::filesystem::path path;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    double duration = 0.0;
    long long nb_frames = 0;
    std::string codec_name;
    // The codec's profile as ffprobe names it ("High", "High 4:4:4
    // Predictive"); empty where it gives none.
    std::string profile;
    int level = 0;  // the codec's level as ffprobe gives it (HEVC's 153 is 5.1); 0 or less: not given
    std::string sar = "1:1";
    std::string pix_fmt;
    long long bit_rate = 0;  // bits per second, 0 if unknown
    // The file's bitrate, soundtrack included, where the video's own is not
    // recorded and the other streams' rates are not known either.
    bool bit_rate_whole_file = false;
    // An estimate: the file's bitrate minus its other streams' (audio,
    // subtitles), its attachments (fonts) and the container's own bytes,
    // where the video's own is not recorded (a Matroska file without
    // statistics tags it can trust, an MPEG-TS file).
    bool bit_rate_estimated = false;
    // The video stream's own size in bytes, from Matroska's statistics tags
    // where they are the file's (as bit_rate trusts them); 0 if unknown.
    long long stream_bytes = 0;
    // What wrote the file and encoded the video, as their tags say: the
    // container's "encoder" (Lavf61.7.100), the stream's (Lavc61.19.100
    // libx265), and an encoder settings tag (ENCODER_SETTINGS) where one is;
    // empty where none is. Settings carried only inside the stream (x265's
    // SEI) are not read.
    std::string writing_library;
    std::string encoder;
    std::string encoder_settings;
    // ffprobe's r_frame_rate. A real difference from the average frame rate
    // warns that the frame rate varies.
    double nominal_fps = 0.0;
    // ffprobe's avg_frame_rate as it gives it: 0 where it gives none.
    double average_fps = 0.0;
    // The stream's colour tags; empty where the file does not say.
    std::string color_range;
    std::string color_space;
    std::string color_transfer;
    std::string color_primaries;
    std::string chroma_location;
    // The container, or a raw stream's demuxer ("hevc", "h264", ...).
    std::string format_name;
    // How long after the file's start the video's first frame comes, in
    // seconds.
    double start_offset = 0.0;
    // The part of `duration` before the video's first frame, in seconds: a
    // Matroska file's length counts from the file's zero, so a first frame
    // at 0.125 s adds 0.125 s that holds no frame. 0 for other files.
    double lead_in = 0.0;
    // A raw stream or raw YUV file whose frame rate is known: found in its
    // headers, entered in the window or given by the command line's --fps
    // (ffprobe.h's raw_stream_as_found, timed_raw_stream, probe_raw_yuv). Its
    // frame count, length and bitrate come from its counted frames.
    bool timed_raw = false;
    // The degrees its rotation tag says a player turns it by (0, 90, 180 or
    // 270). The app does not: width and height, and every picture, are as stored.
    int rotation = 0;
    // Whether its first pictures are not the size above: AV1 may code its
    // frames smaller than its sequence header says, which is the size above.
    // FFmpeg rebuilds nothing for a stream that starts so, and its hardware
    // decoders give pictures of the header's size that are not the video
    // (SizeChangeMarks marks such a video from the start).
    bool coded_at_another_size = false;
    // Whether it is monochrome H.264 (film_grain_detect.h's
    // h264_monochrome_in_annex_b), which ffprobe reports as 4:2:0: decoded in
    // software on every GPU (pick_decode), as NVIDIA's decoder leaves its grey
    // chroma unset (AMD's gives it; Intel's and Apple's are not checked).
    // False where its stream was not read, as film_grain is then unknown.
    bool monochrome_h264 = false;
    // Set for a video whose picture size changes partway through (FFmpeg
    // rebuilt its filters for it): the size every one of its frames is
    // scaled to first, so that the whole video is scored at one size
    // (size_changes.h). Nothing for any other video.
    std::optional<std::pair<int, int>> scaled_to;
    // Set for a resolution test's test video, which the app's decoders
    // (DecodePlan::via_w) or FFmpeg's filters (video_chains) make from the
    // source: each frame scaled to this size, then back to the size compared
    // at. Nothing for any other video.
    std::optional<std::pair<int, int>> resample_via;
    // The film grain its stream describes for the player to add, as
    // film_grain_detect.h's FilmGrainKind bits, 0 for none; nothing where it
    // was not read (a run saved before 2.3, a scan that failed).
    std::optional<unsigned> film_grain;
    // Whether it is scored with that film grain (true) or without (false),
    // as the user chose; nothing where no choice was made: then with
    // (film_grain_scoring.h's effective_film_grain).
    std::optional<bool> with_film_grain;

    // Its frames' size as they reach the comparison's filters: scaled_to,
    // or its own.
    std::pair<int, int> frame_size() const { return scaled_to.value_or(std::pair(width, height)); }
    // Its length from its first frame to its end: duration less lead_in.
    double length_seconds() const { return duration - lead_in; }
    // The frame count ffprobe reports, or else length_seconds times fps.
    long long estimated_frame_count() const;
    // Whether this is a raw stream whose frame rate nobody knows: its headers
    // have none and none was entered. Its frames cannot be placed in time, so
    // it cannot be compared until one is.
    bool frame_rate_unknown() const { return is_raw_stream() && !timed_raw; }
    // A raw stream's rate is now `frames_per_second` (above 0): its length
    // and bitrate follow from its counted frames (nb_frames).
    void set_frame_rate(double frames_per_second);
    // A raw stream's rate is not known: FFmpeg's guess at 25 frames a second,
    // and the length and bitrate worked out from it, are forgotten.
    void forget_frame_rate();
    // Takes the bitrate `from` gives (with where it comes from), the same file
    // read again: a saved result's copy of a video read since, which an
    // older rule gave another rate.
    void take_bit_rate(const VideoInfo& from);
    // A bare elementary stream (a .hevc or .264 file) or raw pictures (a
    // .yuv file), not a container: it has no timestamps, so its frame rate
    // cannot be known.
    bool is_raw_stream() const;
    // Whether its samples use the full range (0-255 at 8 bits) rather than
    // the limited one (16-235 for luma): tagged so, or untagged in a "yuvj"
    // pixel format.
    bool full_range() const;
    // FFmpeg's options before the video's -i: probe_options(); and for one
    // scaled_to a size, -reinit_filter 0, so that FFmpeg keeps its filters
    // (and their scale) as the size changes. Nothing for any other video.
    std::vector<std::string> input_options() const;
    // How the input is read, for ffprobe as for FFmpeg (ffprobe has no
    // -reinit_filter): a timed raw stream's frame rate (-framerate), so that
    // the times of its frames are right, after a raw YUV file's format,
    // picture size and pixel format. Nothing for any other video.
    std::vector<std::string> probe_options() const;
    // input_options() for an FFmpeg that decodes its pictures to score them,
    // with -export_side_data film_grain for a video scored without its film
    // grain (FFmpeg then adds none). Throws FilmGrainRefusal for one scored
    // with grain FFmpeg does not add as the app does (H.274's: FFmpeg adds it
    // at 8 bits only, rounded down): it is decoded in the app or not at all.
    // Stream copies and black-bar detection take input_options(), ffprobe
    // probe_options().
    std::vector<std::string> decode_options() const;
    // Whether the nominal and the average frame rate differ by more than
    // rounding.
    bool is_variable_frame_rate() const;
    // Every field the same, as the Python app's dataclass compares them.
    bool operator==(const VideoInfo&) const = default;
};

}  // namespace vml
