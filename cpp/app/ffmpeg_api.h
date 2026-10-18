// FFmpeg's headers as Video Compare's native playback uses them: the
// bundled libraries (resources/tools/ffmpeg, scripts/build_ffmpeg_decoders.ps1,
// which vendors these headers and what they include into native/ffmpeg by
// reading this file). On Windows, Direct3D 11's video decoding and WASAPI
// for the soundtrack; on macOS (scripts/build_ffmpeg_decoders_macos.sh, the
// same headers), VideoToolbox's and CoreAudio.
#pragma once

#ifdef _WIN32
// Direct3D 11's C++ header first: FFmpeg's D3D11VA header includes it, and
// inside extern "C" its operators could not be declared.
#include <d3d11.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#ifdef _WIN32
#include <libavutil/hwcontext_d3d11va.h>
#endif
#include <libavutil/channel_layout.h>
#include <libavutil/film_grain_params.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
