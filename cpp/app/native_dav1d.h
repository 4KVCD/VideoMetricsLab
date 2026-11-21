// AV1 and AV2 decoded in software for Video Compare's native playback, by
// dav1d and splot: FFmpeg's own AV1 decoder only drives a GPU's, the app's
// FFmpeg has no dav1d wrapper, and FFmpeg has no AV2 decoder at all. dav1d is
// loaded by its path (dav1d_library_path()), through the structures
// native/dav1d/dav1d_api.h mirrors, as the GPU metrics' software decoder
// loads it; splot through splot_library() (vml/splot_library.h).
#pragma once

#include <QString>

#include <memory>
#include <optional>

struct AVCodecParameters;
struct AVFrame;
struct AVPacket;

namespace vml::app {

// The calls both decoders take, as avcodec's.
class SoftwareVideo {
public:
    virtual ~SoftwareVideo() = default;
    // As avcodec_send_packet: a packet, with its presentation time, or
    // nullptr at the end of the stream. 0, or a negative AVERROR.
    virtual int send(const AVPacket* packet) = 0;
    // As avcodec_receive_frame: the next picture into `frame`, its planes the
    // decoder's own, held until release(); AVERROR(EAGAIN) when it needs
    // more packets, AVERROR_EOF when it has given every one after the end.
    virtual int receive(AVFrame* frame) = 0;
    // Lets the picture last received go.
    virtual void release() = 0;
    // Forgets every packet and picture, for a seek.
    virtual void flush() = 0;
};

class Dav1dVideo : public SoftwareVideo {
public:
    // Why AV1 cannot be decoded in software here, or nothing when it can.
    static std::optional<QString> unavailable();

    // A decoder for the AV1 stream `stream` describes: its configuration's
    // sequence header is given to dav1d before the first packet, and its
    // colours are its pictures'; its film grain added where `apply_grain`
    // (grain_decoding.h). Throws std::runtime_error when dav1d is
    // unavailable or does not start.
    Dav1dVideo(const AVCodecParameters& stream, bool apply_grain);
    ~Dav1dVideo() override;
    Dav1dVideo(const Dav1dVideo&) = delete;
    Dav1dVideo& operator=(const Dav1dVideo&) = delete;

    int send(const AVPacket* packet) override;
    int receive(AVFrame* frame) override;
    void release() override;
    void flush() override;

private:
    struct State;
    std::unique_ptr<State> state_;
};

class SplotVideo : public SoftwareVideo {
public:
    // Why AV2 cannot be decoded here, or nothing when it can.
    static std::optional<QString> unavailable();

    // A decoder for an AV2 stream, its colours its pictures' own (AV2's
    // content interpretation; the container says none), its film grain added
    // where `apply_grain`. Throws std::runtime_error when splot is
    // unavailable or does not start.
    explicit SplotVideo(bool apply_grain);
    ~SplotVideo() override;
    SplotVideo(const SplotVideo&) = delete;
    SplotVideo& operator=(const SplotVideo&) = delete;

    int send(const AVPacket* packet) override;
    int receive(AVFrame* frame) override;
    void release() override;
    void flush() override;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace vml::app
