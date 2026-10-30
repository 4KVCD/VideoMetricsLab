// AV1 decoded in software for Video Compare's native playback, by dav1d:
// FFmpeg's own AV1 decoder only drives a GPU's, and the app's FFmpeg has no
// dav1d wrapper. dav1d.dll is loaded by its path (dav1d_library_path()),
// through the structures native/dav1d/dav1d_api.h mirrors, as the GPU
// metrics' software decoder loads it.
#pragma once

#include <QString>

#include <memory>
#include <optional>

struct AVCodecParameters;
struct AVFrame;
struct AVPacket;

namespace vml::app {

class Dav1dVideo {
public:
    // Why AV1 cannot be decoded in software here, or nothing when it can.
    static std::optional<QString> unavailable();

    // A decoder for the AV1 stream `stream` describes: its configuration's
    // sequence header is given to dav1d before the first packet, and its
    // colours are its pictures'; its film grain added where `apply_grain`
    // (grain_decoding.h). Throws std::runtime_error when dav1d is
    // unavailable or does not start.
    Dav1dVideo(const AVCodecParameters& stream, bool apply_grain);
    ~Dav1dVideo();
    Dav1dVideo(const Dav1dVideo&) = delete;
    Dav1dVideo& operator=(const Dav1dVideo&) = delete;

    // As avcodec_send_packet: a packet, with its presentation time, or
    // nullptr at the end of the stream. 0, or a negative error (dav1d's
    // errors are AVERROR's: negative errno values).
    int send(const AVPacket* packet);
    // As avcodec_receive_frame: the next picture into `frame`, its planes
    // dav1d's own, held until release(); AVERROR(EAGAIN) when it needs more
    // packets, AVERROR_EOF when it has given every one after the end.
    int receive(AVFrame* frame);
    // Lets the picture last received go.
    void release();
    // Forgets every packet and picture, for a seek.
    void flush();

private:
    // dav1d's decoder and what it was given (native_dav1d.cpp).
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace vml::app
