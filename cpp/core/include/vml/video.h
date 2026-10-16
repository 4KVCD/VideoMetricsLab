// A video's details (VideoInfo), black-bar crops (CropBox), and the choices
// that say how two videos are lined up. A port of the parts of
// vmaf_app/core/models.py they come from.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace vml {

// Whether black bars are detected and cut (automatic) or kept (none).
enum class CropMode { automatic, none };

// The GPU maker whose decoder to use: automatic picks one found on this PC,
// and none decodes in software.
enum class GpuVendor { automatic, nvidia, intel, amd, none };

// The maker of a GPU with this PCI vendor ID: none for a maker not known.
GpuVendor vendor_of_pci_id(unsigned id);

// The maker's name as the logs and the window write it: "NVIDIA", "INTEL" or
// "AMD" (empty for automatic and none).
std::string_view vendor_label(GpuVendor vendor);

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

// A video file's details, as ffprobe reports them (ffprobe.h).
struct VideoInfo {
    std::filesystem::path path;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    double duration = 0.0;
    long long nb_frames = 0;
    std::string codec_name;
    std::string sar = "1:1";
    std::string pix_fmt;
    long long bit_rate = 0;  // bits per second, 0 if unknown
    // The file's bitrate, soundtrack included, where the video's own is not
    // recorded (a Matroska file without statistics tags).
    bool bit_rate_whole_file = false;
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

    // The frame count ffprobe reports, or else duration times fps.
    long long estimated_frame_count() const;
    // A bare elementary stream (a .hevc or .264 file), not a container: it
    // has no timestamps, so its frame rate cannot be known.
    bool is_raw_stream() const;
    // Whether the nominal and the average frame rate differ by more than
    // rounding.
    bool is_variable_frame_rate() const;
    // Every field the same, as the Python app's dataclass compares them.
    bool operator==(const VideoInfo&) const = default;
};

}  // namespace vml
