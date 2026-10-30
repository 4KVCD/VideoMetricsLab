// The codecs the app's own decoder decodes on every route that can take its
// pictures, rather than FFmpeg's decoder: FFmpeg's decodes some of their
// streams wrongly, or none at all. Each route asks here (decoded_by_the_app
// in film_grain_scoring.h), never for a codec by name, so that another codec
// is one more entry: the software decoder (native/software_frames.cpp) then
// needs its decoder, and everything else follows.
#pragma once

#include <array>
#include <string_view>

namespace vml {

struct AppDecodedCodec {
    std::string_view codec;     // ffprobe's name of it, lower case
    std::string_view decoder;  // the app's decoder, for messages
    // Whether FFmpeg decodes it where its pictures cannot reach a route from
    // the app (the log says why); a codec FFmpeg cannot decode refuses the
    // metric there instead. Its scores are the same ones whichever decoder
    // made them: a saved score does not record the decoder.
    bool ffmpeg_decodes;
};

inline constexpr std::array<AppDecodedCodec, 1> app_decoded_codecs{{
    // FFmpeg's vvc fails on, and decodes wrongly, VVenC encodes with H.274
    // model 1 grain at 10 bits (VMAF 26.92 where VVdeC's pictures score 95.23).
    {"vvc", "VVdeC", true},
}};

// The entry of `codec` (ffprobe's name, lower case), or nullptr where FFmpeg
// decodes it as ever.
inline const AppDecodedCodec* app_decoded_codec(std::string_view codec) {
    for (const AppDecodedCodec& entry : app_decoded_codecs) {
        if (entry.codec == codec) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace vml
