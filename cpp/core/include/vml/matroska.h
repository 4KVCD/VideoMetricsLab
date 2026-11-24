#pragma once

#include <filesystem>
#include <istream>
#include <optional>
#include <string>

namespace vml {

// The program that wrote a Matroska (or WebM) file: its Segment Info's
// WritingApp, such as "mkvmerge v94.0 ('Initiate') 64-bit", "Lavf61.7.100" or
// "HandBrake 1.9.0", read from the start of `in`, where every muxer writes it
// (before the first cluster). ffprobe does not give it: it gives the file's
// ENCODER tag, which mkvmerge copies from its input, so a file mkvmerge wrote
// from one FFmpeg made says "Lavf". Nothing for a file that isn't Matroska, has
// none, or whose headers are damaged.
std::optional<std::string> matroska_writing_app(std::istream& in);
std::optional<std::string> matroska_writing_app(const std::filesystem::path& path);

}  // namespace vml
