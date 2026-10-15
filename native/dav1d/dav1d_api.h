// dav1d's structures, as mirrored: the few of its public API that the app's
// AV1 decoding uses, laid out as dav1d's headers lay them out, so that the
// app needs neither those headers nor an import library and loads dav1d.dll
// by its path (the Python app's from its GStreamer wheel; the C++ app's
// package carries the same file). Used by software_frames.cpp (the GPU
// metrics' software decoder) and by the C++ app's Video Compare playback
// (cpp/app/native_dav1d.cpp); each checks dav1d's API version before use.
#pragma once

#include <stddef.h>
#include <stdint.h>

// dav1d's API 7 (dav1d 1.x from 1.3 on; GStreamer 1.28 ships 1.5).
struct Dav1dUserData {
    const uint8_t *data;
    void *ref;
};
struct Dav1dDataProps {
    int64_t timestamp, duration, offset;
    size_t size;
    Dav1dUserData user_data;
};
struct Dav1dData {
    const uint8_t *data;
    size_t sz;
    void *ref;
    Dav1dDataProps m;
};
struct Dav1dPictureParameters {
    int w, h;
    int layout;  // Dav1dPixelLayout: 1 is I420
    int bpc;
};
struct Dav1dPicture {
    void *seq_hdr, *frame_hdr;
    void *data[3];
    ptrdiff_t stride[2];
    Dav1dPictureParameters p;
    Dav1dDataProps m;
    void *content_light, *mastering_display, *itut_t35;
    size_t n_itut_t35;
    uintptr_t reserved[4];
    void *frame_hdr_ref, *seq_hdr_ref, *content_light_ref, *mastering_display_ref, *itut_t35_ref;
    uintptr_t reserved_ref[4];
    void *ref;
    void *allocator_data;
};
struct Dav1dSettings {
    int n_threads, max_frame_delay, apply_grain, operating_point, all_layers;
    unsigned frame_size_limit;
    void *allocator_cookie, *alloc_picture_callback, *release_picture_callback;
    void *logger_cookie, *logger_callback;
    int strict_std_compliance, output_invisible_frames, inloop_filters, decode_frame_type;
    uint8_t reserved[16];
};
static_assert(offsetof(Dav1dPicture, p) == 56 && offsetof(Dav1dPicture, m) == 72, "Dav1dPicture");
static_assert(offsetof(Dav1dSettings, logger_callback) == 56, "Dav1dSettings");
// Dav1dPixelLayout's values.
constexpr int kDav1dI400 = 0;
constexpr int kDav1dI420 = 1;
constexpr int kDav1dI422 = 2;
constexpr int kDav1dI444 = 3;
constexpr unsigned kDav1dApiMajor = 7;
constexpr int kDav1dAgain = -11;  // DAV1D_ERR(EAGAIN)
