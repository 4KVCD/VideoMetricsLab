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
    // The grain added is one only `decoder` adds (H.274's, by VVdeC): an
    // FFmpeg without that decoder leaves it out rather than add it otherwise.
    // Set by the app (film_grain_scoring.h's video_grain_decoding).
    bool only_decoder_adds = false;
};

// The decoding of a video (`vvc`: a VVC stream) whose grain is `film_grain`:
// 1 added, 0 left out, -1 as the decoder adds it by itself (the grain not
// known).
inline GrainDecoding grain_decoding(bool vvc, int film_grain) {
    GrainDecoding decoding;
    if (vvc) {
        // VVdeC decodes every VVC stream. It adds H.274's
        // grain, of either model at any depth, where FFmpeg's vvc adds model 0
        // at 8 bits only, rounded down; and FFmpeg's vvc fails on streams
        // VVdeC decodes (H.274 model 1 at 10 bits: "Invalid data found").
        decoding.decoder = "libvvdec";
    }
    decoding.export_grain = film_grain == 0;
    decoding.apply_grain = film_grain != 0;
    return decoding;
}

// How many pictures VVdeC parses ahead by default when it decodes with
// `threads` (its DecLib: (threads * 24) >> 4, at most 48). A decoder holds
// that many before its first comes out, and each seek starts a new one.
inline unsigned vvdec_default_parse_delay(unsigned threads) {
    const unsigned delay = (threads * 24) >> 4;
    return delay < 48 ? delay : 48;
}

// Video Compare's (libvvdec's parse_delay): half the default, at least 2.
// Half decodes as fast, where 2 did not: a 149 Mbit/s 4K60 stream on a
// 32-thread CPU at 142-143 fps (the default 143-145, 2 36-37), and on 16
// threads at 84 fps, as the default. A 4K 10-bit grain video then seeks
// 0.23-0.33 s sooner and its decoder holds 1,750 MB, not 2,697 (32 threads).
// The pictures are the same. 0 would decode strictly in order, waiting at
// every picture's suffix SEI.
inline unsigned video_compare_parse_delay(unsigned threads) {
    const unsigned half = vvdec_default_parse_delay(threads) / 2;
    return half > 2 ? half : 2;
}

// Scoring's (software_frames.cpp, the decoder scores are made from): one per
// thread, at least 2, at most VVdeC's 48. Scoring has no seeks and waits for
// the decoding wherever the app decodes for a GPU metric, so it must decode
// as fast as with the default: one per thread did, where Video Compare's did
// not quite (the same 149 Mbit/s stream on 32 threads: 145 fps, 24 142-143,
// the default 143-145; on 16 threads 84 fps, as the default), holding a fifth
// to a quarter less (1,977 MB, not 2,630; 1,310, not 1,638). The threads are
// VVdeC's own: std::thread::hardware_concurrency in the same process, from
// the same C++ library.
inline unsigned scoring_parse_delay(unsigned threads) {
    const unsigned delay = threads > 2 ? threads : 2;
    return delay < 48 ? delay : 48;
}
