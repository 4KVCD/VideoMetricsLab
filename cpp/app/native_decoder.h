// One video of a comparison decoded for Video Compare's native playback (the
// Python app's GstComparePipeline, on FFmpeg's libraries instead of
// GStreamer): read with libavformat, decoded by Direct3D 11's video decoding
// on the presenter's device where the GPU decodes the video, else by
// FFmpeg's software decoder (dav1d for AV1, native_dav1d.h) and uploaded,
// frames timed from the video's first frame. It decodes freely, at most
// three frames ahead of what is taken; seeks are made on its thread, only
// the latest one asked for.
#pragma once

#include <QString>
#include <QThread>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "vml/frame_extract.h"
#include "vml/native_playback.h"

struct ID3D11Texture2D;

namespace vml::app {

class NativeDevice;

// A decoded frame on the GPU: a slice of a decoder's texture array, or the
// upload of a picture decoded in software, in NV12 or P010. It stays valid
// while anything holds it.
struct NativeFrame {
    // From the video's first frame.
    std::int64_t time_ns = 0;
    ID3D11Texture2D* texture = nullptr;
    unsigned subresource = 0;
    // The picture's size; its texture can be larger (a decoder pads it).
    int width = 0;
    int height = 0;
    bool ten_bit = false;
    // GStreamer's name for its YUV matrix (yuv_matrix_name), and its range.
    std::string matrix = "bt709";
    bool full_range = false;
    // PQ video shown as it comes, in BT.2020 (on an HDR display): the
    // swapchain then takes HDR10.
    bool hdr10 = false;
    // Already cropped and scaled to the size it is shown at (the source
    // scaled down to the test video's size): drawn whole.
    bool prepared = false;
    // What keeps the texture: the decoder's frame, or the uploaded texture.
    std::shared_ptr<void> keep;
};
using NativeFramePtr = std::shared_ptr<const NativeFrame>;

// What changed since the last poll: a status line, an error, the video's end.
struct NativeUpdate {
    std::optional<QString> status;
    std::optional<QString> error;
    bool ended = false;
};

// The decoder as the playback pool uses it.
class NativeVideo : public QThread {
    Q_OBJECT

public:
    using QThread::QThread;
    // Asks for the frames from `position_ms` (counted from the video's first
    // frame) on: the latest seek asked for replaces any not yet made.
    virtual void seek(int position_ms) = 0;
    // Whether the last seek asked for is still to be made: until it is,
    // frames from before it can still come.
    virtual bool seeking() const = 0;
    // The next frame decoded, or nothing yet.
    virtual NativeFramePtr try_pull() = 0;
    virtual NativeUpdate poll() = 0;
    // Where the video's first frame is on the file's timeline, once known.
    virtual std::optional<int> first_frame_ms() const = 0;
    // How the presenter's shader shows this video's frames.
    virtual const std::optional<std::array<float, 16>>& shading() const = 0;
    // Stops it soon; it finishes on its own thread.
    virtual void cancel() = 0;
};

class FfmpegNativeVideo : public NativeVideo {
    Q_OBJECT

public:
    // `comparison`'s `side`, from `position_ms` on, decoded on `device`.
    FfmpegNativeVideo(std::shared_ptr<NativeDevice> device, FrameComparison comparison, FrameSide side,
                      const PreviewColorSettings& colors, int position_ms, QObject* parent = nullptr);
    ~FfmpegNativeVideo() override;

    void seek(int position_ms) override;
    bool seeking() const override;
    NativeFramePtr try_pull() override;
    NativeUpdate poll() override;
    std::optional<int> first_frame_ms() const override;
    const std::optional<std::array<float, 16>>& shading() const override { return shading_; }
    void cancel() override;

    // For the tests: decodes in software from the start, as on a GPU without
    // a decoder for the video. Before start().
    void decode_in_software() { gpu_decoding_ = false; }

    // One opening of the video (native_decoder.cpp).
    struct Decoding;

protected:
    void run() override;

private:
    // Waits while three frames wait to be taken; false once cancelled or a
    // seek is asked for that can be made (once the first frame is known).
    bool wait_for_room();
    bool cancelled() const;
    // FFmpeg's reading asks it whether to stop: once cancelled (a file on a
    // slow disk would otherwise hold Cancel until its read ends).
    static int interrupted(void* self);
    void deliver(NativeFramePtr frame);
    void fail(const QString& error);

    std::shared_ptr<NativeDevice> device_;
    FrameComparison comparison_;
    FrameSide side_;
    std::optional<std::array<float, 16>> shading_;
    NativeShading shaded_ = NativeShading::none;
    // The size the source is scaled down to, when it is.
    std::optional<Size> scale_to_;
    bool gpu_decoding_ = true;

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<NativeFramePtr> frames_;
    bool cancelled_ = false;
    std::optional<int> wanted_ms_;  // a seek asked for, not yet made
    int seeks_asked_ = 0;
    int seeks_made_ = 0;
    std::optional<int> first_frame_ms_;  // set with the first frame: seeks can then be made
    std::optional<QString> status_;
    std::optional<QString> error_;
    bool ended_ = false;
};

}  // namespace vml::app
