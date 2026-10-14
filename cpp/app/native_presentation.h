// How Video Compare's native playback shows its frames (the Python app's
// LockedPresentation): the app's own Direct3D 11 presenter
// (vmaf_app/native/d3d11_tonemap.dll, native/d3d11_tonemap.cpp) draws them
// into a swapchain in a window of its own over the view, on the decoders'
// device, taking one frame per display refresh.
//
// A thread waits, holding no lock, until the swapchain can take a frame, and
// only then takes the device's lock to draw the frame that is due. The
// decoders need the same lock for each frame, so it is never held while
// waiting for the display: GStreamer's d3d11videosink did that, stalling the
// decoders by up to 40 ms, so 120 fps video showed about 100 frames a second.
#pragma once

#include <QString>

#include <array>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "native_decoder.h"

struct HWND__;

namespace vml::app {

class NativeDevice;

// What native playback shows its frames with: the presenter (or the tests'
// stand-in).
class NativeOutput {
public:
    // The part of a frame drawn (x, y, width, height in its pixels), and where
    // in the window (device pixels).
    struct Placement {
        std::array<int, 4> source{};
        std::array<int, 4> target{};
        bool operator==(const Placement&) const = default;
    };

    virtual ~NativeOutput() = default;
    // Shows `frame` at the next display refresh, as `placement` says, shaded
    // as `shading` says (shading_parameters) or as it comes.
    virtual void present(NativeFramePtr frame, const Placement& placement,
                         const std::optional<std::array<float, 16>>& shading) = 0;
    // Why drawing stopped, if it did.
    virtual std::optional<QString> error() const = 0;
};

class NativePresentation : public NativeOutput {
public:
    // Whether the presenter is bundled; why not, when it is not.
    static std::optional<QString> unavailable();
    // Compiles the presenter's shaders ahead of the first presenter, once per
    // process: otherwise the first opening of a pair takes 10 ms longer.
    static void prepare();

    // Draws on `device` over `window`, into a 10-bit swapchain for an HDR
    // display. Throws std::runtime_error when the presenter cannot start.
    NativePresentation(std::shared_ptr<NativeDevice> device, HWND__* window, bool hdr_display);
    ~NativePresentation() override;
    NativePresentation(const NativePresentation&) = delete;
    NativePresentation& operator=(const NativePresentation&) = delete;

    void present(NativeFramePtr frame, const Placement& placement,
                 const std::optional<std::array<float, 16>>& shading) override;
    std::optional<QString> error() const override;
    // The presenter drawing, for the status line.
    static QString name() { return QStringLiteral("Direct3D 11"); }

private:
    struct Due {
        NativeFramePtr frame;
        Placement placement;
        std::optional<std::array<float, 16>> shading;
    };
    void present_paced();
    void draw(const Due& due);

    std::shared_ptr<NativeDevice> device_;
    HWND__* window_ = nullptr;
    void* presenter_ = nullptr;
    mutable std::mutex mutex_;
    std::condition_variable due_ready_;
    std::deque<Due> due_;
    bool stopping_ = false;
    std::optional<QString> error_;
    std::thread thread_;
};

}  // namespace vml::app
