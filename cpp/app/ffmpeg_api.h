// FFmpeg's headers as Video Compare's native playback uses them: the
// bundled libraries (vmaf_app/tools/ffmpeg, scripts/build_ffmpeg_decoders.ps1,
// which vendors these headers and what they include into native/ffmpeg by
// reading this file). Windows only, for now: Direct3D 11's video decoding.
#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
