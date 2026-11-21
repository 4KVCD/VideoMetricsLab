// splot's C API, as mirrored: the structures, constants and calls of its
// splot.h that the app's AV2 decoding uses, laid out as that header lays them
// out (as dav1d/dav1d_api.h mirrors dav1d's), so that the app needs neither
// the header nor an import library and loads splot by its path
// (splot_library_path). splot (github.com/splotlabs/splot) is an AV2 decoder
// in Rust; scripts/build_splot.ps1 pins its commit. Plain C++, shared by the
// app, its probe and the software frame decoder.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

// splot 0.0.0 (commit 19ba2a57, 2026-11-21).
constexpr const char *kSplotVersion = "0.0.0";

constexpr int kSplotOk = 0;
// splot_send_data: full, take pictures and send the same data again;
// splot_get_picture: send more data first.
constexpr int kSplotAgain = -1;
// splot_get_picture: drained, no picture follows.
constexpr int kSplotEof = -2;
constexpr int kSplotInvalid = -3;
// The stream failed to decode; the decoder takes no more data.
constexpr int kSplotDecodeError = -4;
constexpr int kSplotStateError = -5;

// SplotPixelLayout's values (dav1d's too).
constexpr int kSplotI400 = 0;
constexpr int kSplotI420 = 1;
constexpr int kSplotI422 = 2;
constexpr int kSplotI444 = 3;

struct SplotSettings {
    uint32_t n_threads;        // 0: one per logical core
    uint32_t max_frame_delay;  // 0: from n_threads
    uint8_t skip_film_grain;   // 1 leaves the film grain out
    uint8_t reserved[31];
};
struct SplotDataProps {
    int64_t pts, duration;
    uint64_t user_data;
};
// H.273's codes (AV2's content interpretation, its sets made codes); 2, 2, 2
// and limited range where the stream says none.
struct SplotColor {
    uint8_t primaries, transfer, matrix, full_range;
};
struct SplotPicture {
    // Each plane at its first stored sample; the visible area starts crop_top
    // rows and crop_left samples in (shifted by the chroma subsampling for
    // U and V). 10-bit samples are uint16_t, in the low bits.
    const void *data[3];
    ptrdiff_t stride[3];
    uint32_t width, height;  // visible, in luma samples
    uint32_t crop_left, crop_top;
    uint32_t bit_depth;  // 8 or 10
    int layout;          // kSplotI400 to kSplotI444
    SplotColor color;
    SplotDataProps props;    // of the unit that coded or showed it
    uint8_t has_film_grain;  // 1 where the stream signals film grain (added or left out)
    size_t reserved[3];
    void *internal;
};
// As splot.h lays them out (LP64 and LLP64 alike).
static_assert(sizeof(SplotSettings) == 40, "SplotSettings");
static_assert(offsetof(SplotPicture, width) == 48 && offsetof(SplotPicture, layout) == 68 &&
                  offsetof(SplotPicture, color) == 72 && offsetof(SplotPicture, props) == 80 &&
                  offsetof(SplotPicture, has_film_grain) == 104 && offsetof(SplotPicture, internal) == 136 &&
                  sizeof(SplotPicture) == 144,
              "SplotPicture");

// splot's calls, as the app binds them by name.
struct SplotCalls {
    const char *(*version)() = nullptr;
    void (*default_settings)(SplotSettings *) = nullptr;
    int (*open)(void **, const SplotSettings *) = nullptr;
    int (*send_data)(void *, const uint8_t *, size_t, const SplotDataProps *) = nullptr;
    void (*drain)(void *) = nullptr;
    int (*get_picture)(void *, SplotPicture *) = nullptr;
    void (*picture_unref)(SplotPicture *) = nullptr;
    void (*close)(void **) = nullptr;

    // Binds each call with `find(name)`, which gives a function's address or
    // nullptr. Returns the name of the first one missing, or nullptr.
    template <typename Find>
    const char *bind(Find &&find) {
        const char *missing = nullptr;
        const auto one = [&](auto &target, const char *name) {
            // Through void *: GetProcAddress's FARPROC, cast straight to
            // another function type, draws GCC's -Wcast-function-type.
            const auto address = reinterpret_cast<void *>(find(name));
            target = reinterpret_cast<std::remove_reference_t<decltype(target)>>(address);
            if (!target && !missing) missing = name;
        };
        one(version, "splot_version");
        one(default_settings, "splot_default_settings");
        one(open, "splot_open");
        one(send_data, "splot_send_data");
        one(drain, "splot_drain");
        one(get_picture, "splot_get_picture");
        one(picture_unref, "splot_picture_unref");
        one(close, "splot_close");
        return missing;
    }
};

// The first visible sample of `plane` (0 Y, 1 U, 2 V) of a picture.
inline const uint8_t *splot_visible_plane(const SplotPicture &picture, int plane) {
    const bool chroma = plane > 0;
    const unsigned shift_x = chroma && (picture.layout == kSplotI420 || picture.layout == kSplotI422) ? 1 : 0;
    const unsigned shift_y = chroma && picture.layout == kSplotI420 ? 1 : 0;
    const ptrdiff_t sample = picture.bit_depth > 8 ? 2 : 1;
    return static_cast<const uint8_t *>(picture.data[plane]) +
           static_cast<ptrdiff_t>(picture.crop_top >> shift_y) * picture.stride[plane] +
           static_cast<ptrdiff_t>(picture.crop_left >> shift_x) * sample;
}

// Whether a picture's colour is said by its stream: codes other than
// unspecified, or full range. (The stream's colour description may also say
// unspecified and limited, which tells nothing either.)
inline bool splot_says_colour(const SplotColor &color) {
    return color.primaries != 2 || color.transfer != 2 || color.matrix != 2 || color.full_range != 0;
}
