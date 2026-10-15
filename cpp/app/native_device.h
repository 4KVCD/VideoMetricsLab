// What Video Compare's native playback runs on (Windows): the bundled FFmpeg
// libraries, loaded once, and the one Direct3D 11 device the decoders decode
// on and the presenter draws with (the Python app's d3d11_device(), which
// GStreamer made). Both decoding and drawing take the device's lock, never
// held while waiting for the display.
#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct AVBufferRef;

namespace vml::app {

// Loads the bundled FFmpeg libraries (tools/ffmpeg), once per process, by
// their paths, so that the app's calls into them, which load them by name
// only when first made, find them loaded. Returns why they cannot be used,
// or nothing when they can.
std::optional<std::string> load_ffmpeg_libraries();

class NativeDevice {
public:
    // The session's device on the first GPU (adapter 0, as the Python app's
    // GStreamer took; VML_GPU_VENDOR's first, platform::first_gpu_adapter),
    // made the first time and again once lost (a GPU reset). Making one
    // takes about 0.14 s while the driver loads, so it is not made per
    // comparison. Throws std::runtime_error when there is none.
    static std::shared_ptr<NativeDevice> shared();

    // A device on the GPU DXGI lists `adapter`th (the tests try each).
    explicit NativeDevice(unsigned adapter = 0);
    ~NativeDevice();
    NativeDevice(const NativeDevice&) = delete;
    NativeDevice& operator=(const NativeDevice&) = delete;

    ID3D11Device* device() const { return device_.Get(); }
    ID3D11DeviceContext* context() const { return context_.Get(); }
    // FFmpeg's D3D11VA device over it, for the decoders (a reference of
    // their own each).
    AVBufferRef* hardware() const { return hardware_; }
    // The lock every use of the device's immediate context takes: FFmpeg's
    // decoding (given it as its lock), uploads and the presenter's drawing.
    std::recursive_mutex& lock() { return mutex_; }
    bool lost() const;
    // The codecs the GPU's Direct3D 11 video decoding takes, as the self-test
    // names them ("H.264", "HEVC 10-bit"...).
    std::vector<std::string> gpu_decoders() const;

private:
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
    AVBufferRef* hardware_ = nullptr;
    std::recursive_mutex mutex_;
};

}  // namespace vml::app
