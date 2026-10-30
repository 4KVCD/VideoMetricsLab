// How a video's pictures are decoded with the film grain its stream describes,
// as the video is scored (the app's film_grain_scoring.h): which decoder, and
// whether the grain is added or left out. One rule for the decoder scores are
// made from (native/software_frames.cpp) and for Video Compare's (the app's
// native_decoder.cpp and native_dav1d.cpp), so that what is shown is what was
// scored. Plain C++: no FFmpeg or dav1d header.
#pragma once

// What film grain a stream describes: the app's FilmGrainKind bits
// (film_grain_detect.h).
enum GrainKind { GRAIN_H274 = 1, GRAIN_H274_MODEL_1 = 2, GRAIN_AFGS1 = 4, GRAIN_AV1 = 8 };

struct GrainDecoding {
    // FFmpeg's decoder to open by name, or nothing for the stream's own.
    const char* decoder = nullptr;
    // FFmpeg's decoders give the grain's parameters as side data instead of
    // adding it (AV_CODEC_EXPORT_DATA_FILM_GRAIN).
    bool export_grain = false;
    // dav1d adds AV1's grain (Dav1dSettings::apply_grain).
    bool apply_grain = true;
};

// The decoding of a video (`vvc`: a VVC stream) whose grain is `film_grain`:
// 1 added, 0 left out, -1 as the decoder adds it by itself (the grain not
// known); `kinds` is what the stream describes.
inline GrainDecoding grain_decoding(bool vvc, int film_grain, int kinds) {
    GrainDecoding decoding;
    if (vvc && film_grain == 1 && (kinds & (GRAIN_H274 | GRAIN_H274_MODEL_1)) != 0) {
        // VVdeC adds H.274's grain, of either model at any depth; FFmpeg's vvc
        // adds model 0 at 8 bits only, rounded down.
        decoding.decoder = "libvvdec";
    }
    decoding.export_grain = film_grain == 0;
    decoding.apply_grain = film_grain != 0;
    return decoding;
}
