// A VVC video's film grain in Video Compare's native playback, on macOS
// (native_grain.h): not drawn yet. The Windows drawer is Direct3D 11 compute
// (native/d3d11_film_grain.cpp); a Metal port of it is still to be made.
// Meanwhile FFmpeg draws H.274 grain on 8-bit 4:2:0 VVC video itself, and
// 10-bit video is shown without it.
#include "native_grain.h"

namespace vml::app {

std::optional<QString> NativeGrain::unavailable() {
    return QStringLiteral("film grain for VVC video is not drawn on macOS yet");
}

}  // namespace vml::app
