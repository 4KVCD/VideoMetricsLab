// FFmpeg's headers as Video Compare's native playback uses them: the
// bundled libraries (resources/tools/ffmpeg, scripts/build_ffmpeg_decoders.ps1,
// which vendors these headers and what they include into native/ffmpeg by
// reading this file). Windows only, for now: Direct3D 11's video decoding,
// and WASAPI for the soundtrack.
#pragma once

// Direct3D 11's C++ header first: FFmpeg's D3D11VA header includes it, and
// inside extern "C" its operators could not be declared.
#include <d3d11.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
#include <libavutil/channel_layout.h>
#include <libavutil/film_grain_params.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
