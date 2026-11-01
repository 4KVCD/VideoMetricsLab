// How Video Compare's native playback probes a file (avformat_find_stream_info,
// which decodes a few pictures of each stream with the codec's default
// decoder): never with FFmpeg's own VVC decoder. The app decodes VVC with
// VVdeC (FFmpeg's decoder "libvvdec"), and FFmpeg's own crashes on a 4:0:0
// video whose AFGS1 film grain set has chroma (FFmpeg 9.0.2: it adds the
// chroma grain into planes a monochrome picture has not). The probe then
// decodes no VVC picture: the time base, frame rates and start time, all the
// app reads from the probe, come from the container and the VVC parser.
#pragma once

#include <string>

struct AVFormatContext;

namespace vml::app {

// Every decoder of the bundled FFmpeg but FFmpeg's own "vvc", comma-separated
// (an AVFormatContext's codec_whitelist). Made once; the bundled libraries
// must be loaded first (load_ffmpeg_libraries).
const std::string& probe_decoder_whitelist();

// Keeps the probe of `format` (before avformat_open_input) from FFmpeg's own
// VVC decoder. False when out of memory.
bool restrict_probe_decoders(AVFormatContext& format);

}  // namespace vml::app
