#include "vml/film_grain_scoring.h"

#include "vml/app_decoders.h"
#include "vml/film_grain_detect.h"
#include "vml/geometry.h"
#include "vml/hardware_decoding.h"
#include "vml/python_text.h"
#include "vml/score_identity.h"

namespace vml {

FilmGrainScoring effective_film_grain(const VideoInfo& info) {
    FilmGrainScoring result;
    if (!info.film_grain || *info.film_grain == 0) {
        return result;
    }
    result.kinds = *info.film_grain;
    result.scoring = info.with_film_grain.value_or(true) ? GrainScoring::with : GrainScoring::without;
    // Grain added in part is not what viewers see either: a video with any
    // grain nothing here adds is scored without all of it, whatever is chosen.
    const std::string codec = ascii_lower(info.codec_name);
    const bool h274 = (result.kinds & (film_grain_h274 | film_grain_h274_model_1)) != 0;
    if (h274 && codec != "vvc") {
        // Only VVdeC synthesizes H.274's grain as the standard says, and it
        // decodes VVC only.
        result.why_without = "H.274 film grain is added to VVC only (by VVdeC), not to " + codec_label(codec);
    } else if ((result.kinds & film_grain_afgs1) && codec == "h264") {
        result.why_without = "AFGS1 film grain in H.264 is added by no decoder here";
    } else if ((result.kinds & film_grain_afgs1) && h274) {
        result.why_without =
            "it describes two kinds of film grain, H.274's and AFGS1, which no decoder here adds together";
    }
    if (!result.why_without.empty()) {
        result.scoring = GrainScoring::without;
    }
    return result;
}

bool film_grain_choosable(const VideoInfo& info) {
    const FilmGrainScoring grain = effective_film_grain(info);
    return grain.scoring != GrainScoring::none && grain.why_without.empty();
}

std::optional<bool> film_grain_default_choice(std::string_view setting) {
    if (setting == "with" || setting == "without") {
        return setting == "with";
    }
    return std::nullopt;
}

bool only_the_app_adds(const FilmGrainScoring& grain) {
    return grain.scoring == GrainScoring::with && (grain.kinds & (film_grain_h274 | film_grain_h274_model_1)) != 0;
}

bool decoded_by_the_app(const VideoInfo& info) {
    return app_decoded_codec(ascii_lower(info.codec_name)) != nullptr || only_the_app_adds(effective_film_grain(info));
}

bool only_the_app_decodes(const VideoInfo& info) {
    const AppDecodedCodec* codec = app_decoded_codec(ascii_lower(info.codec_name));
    return (codec != nullptr && !codec->ffmpeg_decodes) || only_the_app_adds(effective_film_grain(info));
}

int decoder_film_grain(const FilmGrainScoring& grain) {
    return grain.scoring == GrainScoring::with ? 1 : grain.scoring == GrainScoring::without ? 0 : -1;
}

GrainDecoding video_grain_decoding(const VideoInfo& info) {
    const FilmGrainScoring grain = effective_film_grain(info);
    GrainDecoding decoding = grain_decoding(ascii_lower(info.codec_name) == "vvc", decoder_film_grain(grain));
    decoding.only_decoder_adds = only_the_app_adds(grain);
    return decoding;
}

std::vector<std::string> picture_grain_options(const VideoInfo& info) {
    const FilmGrainScoring grain = effective_film_grain(info);
    if (grain.scoring == GrainScoring::without || only_the_app_adds(grain)) {
        return {"-export_side_data", "film_grain"};
    }
    return {};
}

std::optional<std::string> film_grain_identity(const VideoInfo& info) {
    const FilmGrainScoring grain = effective_film_grain(info);
    if (grain.scoring == GrainScoring::none) {
        return std::nullopt;
    }
    if (grain.scoring == GrainScoring::without) {
        // Grain some route added before: FFmpeg H.274's model 0 at 8 bits,
        // AFGS1 outside H.264 and AV1's; GPU decoders AV1's, or not. AV2's,
        // which dav2d adds unless told not to, is marked left out too.
        const bool h274_8_bit = (grain.kinds & film_grain_h274) != 0 && bit_depth(info.pix_fmt) == 8;
        const bool afgs1_added = (grain.kinds & film_grain_afgs1) != 0 && ascii_lower(info.codec_name) != "h264";
        const bool added = h274_8_bit || afgs1_added || (grain.kinds & (film_grain_av1 | film_grain_av2)) != 0;
        return added ? std::optional(std::string(score_identity::film_grain_left_out)) : std::nullopt;
    }
    if (only_the_app_adds(grain)) {
        return std::string(score_identity::film_grain_h274);
    }
    if ((grain.kinds & film_grain_av2) != 0) {
        return std::string(score_identity::film_grain_av2);
    }
    if ((grain.kinds & film_grain_av1) == 0) {
        return std::string(score_identity::film_grain_afgs1);
    }
#ifdef _WIN32
    // The Windows routes added AV1's grain before, and alike: dav1d, FFmpeg's
    // libdav1d, CUDA and QSV decoding, and NVIDIA's, Intel's and AMD's
    // decoders in the app (the same scores frame for frame).
    return std::nullopt;
#else
    return std::string(score_identity::film_grain_av1);  // VideoToolbox's is not known
#endif
}

void take_film_grain(VideoInfo& into, const VideoInfo& from) {
    into.film_grain = from.film_grain;
    into.with_film_grain = from.with_film_grain;
}

}  // namespace vml
