// dav2d's structures, as mirrored: the few of its public API that the app's
// AV2 decoding uses, laid out as dav2d's headers lay them out (as
// dav1d_api.h mirrors dav1d's), so that the app needs neither those headers
// nor an import library and loads dav2d.dll by its path. dav2d (VideoLAN's
// AV2 decoder, based on dav1d) has dav1d's API with its names changed, but
// its picture carries two more things (the film grain model and the content
// interpretation, with their references). Its API is not stable yet: the app
// checks dav2d's API version, and scripts/build_dav2d.ps1 pins its commit.
#pragma once

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

// dav2d's API 1.0.0 (commit 3dde00d0, 2026-10-30).
struct Dav2dUserData {
    const uint8_t *data;
    void *ref;
};
struct Dav2dDataProps {
    int64_t timestamp, duration, offset;
    size_t size;
    Dav2dUserData user_data;
};
struct Dav2dData {
    const uint8_t *data;
    size_t sz;
    void *ref;
    Dav2dDataProps m;
};
struct Dav2dPictureParameters {
    int w, h;
    int layout;  // Dav2dPixelLayout: 1 is I420
    int bpc;
};
// The colour of the content (AV2's content interpretation OBU): as H.273's
// codes, or one of AV2's descriptions that stands for a set of them.
struct Dav2dContentInterpretation {
    uint8_t scan_type, color_description_present, chroma_sample_position_present, aspect_ratio_info_present,
        timing_info_present, extension_present;
    uint8_t chroma_sample_position[2];
    struct {
        uint8_t type;  // Dav2dColorDescription: 0 explicit, else a set below
        uint8_t primaries, transfer, matrix, range;
    } color;
    // (more follows, which the app does not read)
};
struct Dav2dPicture {
    void *seq_hdr, *frame_hdr;
    void *data[3];
    ptrdiff_t stride[2];
    Dav2dPictureParameters p;
    Dav2dDataProps m;
    void *content_light, *mastering_display, *itut_t35;
    const struct Dav2dFilmGrainData *film_grain;  // nothing where the picture has no grain model
    const Dav2dContentInterpretation *content;  // nothing where the stream has none
    size_t n_itut_t35;
    uintptr_t reserved[4];
    void *frame_hdr_ref, *seq_hdr_ref, *content_light_ref, *mastering_display_ref, *itut_t35_ref, *film_grain_ref,
        *content_ref;
    uintptr_t reserved_ref[4];
    void *ref;
    void *allocator_data;
};
struct Dav2dFilmGrainData {
    int chroma_scaling_from_luma;
    int num_points[3];
    uint8_t points[3][14][2];
    int scaling_shift, ar_coeff_lag;
    int8_t ar_coeffs[3][28];
    uint64_t ar_coeff_shift;
    int grain_scale_shift;
    int uv_mult[2], uv_luma_mult[2], uv_offset[2];
    int overlap_flag, clip_to_restricted_range, mc_identity, block_size;
};
struct Dav2dSettings {
    int n_threads, max_frame_delay, apply_grain, operating_point, all_layers;
    unsigned frame_size_limit;
    void *allocator_cookie, *alloc_picture_callback, *release_picture_callback;
    void *logger_cookie, *logger_callback;
    int strict_std_compliance, output_invisible_frames, inloop_filters, decode_frame_type;
    uint8_t reserved[16];
};
// As dav2d's own headers give them (checked with its headers on x86-64).
static_assert(offsetof(Dav2dPicture, p) == 56 && offsetof(Dav2dPicture, m) == 72, "Dav2dPicture");
static_assert(offsetof(Dav2dPicture, content) == 152 && offsetof(Dav2dPicture, allocator_data) == 296 &&
                  sizeof(Dav2dPicture) == 304,
              "Dav2dPicture");
static_assert(offsetof(Dav2dSettings, logger_callback) == 56 && sizeof(Dav2dSettings) == 96, "Dav2dSettings");
static_assert(sizeof(Dav2dData) == 72, "Dav2dData");
static_assert(offsetof(Dav2dContentInterpretation, color) == 8, "Dav2dContentInterpretation");
static_assert(offsetof(Dav2dFilmGrainData, clip_to_restricted_range) == 232, "Dav2dFilmGrainData");

// Whether a picture's grain model adds grain, as dav2d's has_grain() decides
// before it applies one: points to scale it, or chroma scaled from luma with
// the picture clipped to the restricted range.
inline bool dav2d_adds_grain(const Dav2dFilmGrainData *grain) {
    return grain && (grain->num_points[0] || grain->num_points[1] || grain->num_points[2] ||
                     (grain->clip_to_restricted_range && grain->chroma_scaling_from_luma));
}
// Dav2dPixelLayout's values.
constexpr int kDav2dI400 = 0;
constexpr int kDav2dI420 = 1;
constexpr int kDav2dI422 = 2;
constexpr int kDav2dI444 = 3;
// Dav2dColorDescription's values: explicit codes, or these sets (H.273's
// primaries, transfer and matrix codes), as AOM's AVM decoder, the reference,
// sets them. dav2d (3dde00d, and its main branch on 2026-10-31) fills its
// picture's codes otherwise for two: matrix 5 for BT.709 SDR and transfer 14
// (BT.2020 10-bit, an SDR curve) for HLG; the app reads the set, not those.
constexpr int kDav2dColorExplicit = 0;
constexpr int kDav2dColorBt709Sdr = 1;    // 1, 1, 1
constexpr int kDav2dColorBt2100Pq = 2;    // 9, 16, 9
constexpr int kDav2dColorBt2100Hlg = 3;   // 9, 18, 9
constexpr int kDav2dColorSrgb = 4;        // 1, 13, 0
constexpr int kDav2dColorSrgbSycc = 5;    // 1, 13, 5
// H.273's primaries, transfer and matrix codes of a content interpretation:
// its own, or those of the description it names (2, unspecified, for one
// not known).
inline void dav2d_colour_codes(const Dav2dContentInterpretation &content, int codes[3]) {
    static const int sets[6][3] = {{0, 0, 0}, {1, 1, 1}, {9, 16, 9}, {9, 18, 9}, {1, 13, 0}, {1, 13, 5}};
    const int type = content.color.type;
    if (type == kDav2dColorExplicit) {
        codes[0] = content.color.primaries;
        codes[1] = content.color.transfer;
        codes[2] = content.color.matrix;
    } else {
        for (int i = 0; i < 3; i++) codes[i] = type > 0 && type < 6 ? sets[type][i] : 2;
    }
}
constexpr unsigned kDav2dApiMajor = 1;
// DAV2D_ERR(EAGAIN), as dav1d's.
constexpr int kDav2dAgain = -EAGAIN;
// DAV2D_EOF, which dav1d does not have: get_picture's answer once the
// pictures it held at the end are all given (after send_data(NULL), which
// dav2d needs to give them: dav1d gives them when asked twice).
constexpr int kDav2dEof = -('E' | ('O' << 8) | ('F' << 16));
