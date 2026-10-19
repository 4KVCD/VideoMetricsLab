// What Video Compare's native playback runs on: the bundled FFmpeg
// libraries, loaded once, and the one GPU device the decoders decode on and
// the presenter draws with, Direct3D 11's on Windows, Metal's on macOS. Both
// decoding and drawing take the device's lock, never held while waiting for
// the display.
#pragma once

#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
#endif

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct AVBufferRef;
#ifndef _WIN32
struct __CVBuffer;
#endif

namespace vml::app {

// Loads the bundled FFmpeg libraries (tools/ffmpeg), once per process, by
// their paths, so that the app's calls into them, which load them by name
// only when first made, find them loaded. Returns why they cannot be used,
// or nothing when they can.
std::optional<std::string> load_ffmpeg_libraries();

#ifdef _WIN32
class NativeDevice {
public:
    // The session's device on the first GPU (adapter 0, or VML_GPU_VENDOR's
    // first: platform::first_gpu_adapter), made the first time and again once
    // lost (a GPU reset). Making one
    // takes about 0.14 s while the driver loads, so it is not made per
    // comparison. Throws std::runtime_error when there is none.
    static std::shared_ptr<NativeDevice> shared();
    // Lets go of the session's device: it goes once nothing else holds it,
    // and shared() makes another.
    static void release_shared();

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
#else
// A picture on macOS: a CoreVideo pixel buffer (CVPixelBufferRef), on an
// IOSurface the GPU reads, NV12 or P010's layout ('420v'/'420f', 'x420'/'xf20').
using PixelBuffer = ::__CVBuffer*;
// Holds a pixel buffer for as long as it lives.
using HeldPixelBuffer = std::shared_ptr<::__CVBuffer>;

class NativeDevice {
public:
    // The session's device, the Mac's GPU, made the first time. Throws
    // std::runtime_error when there is none.
    static std::shared_ptr<NativeDevice> shared();
    static void release_shared();

    NativeDevice();
    ~NativeDevice();
    NativeDevice(const NativeDevice&) = delete;
    NativeDevice& operator=(const NativeDevice&) = delete;

    // id<MTLDevice>, and a CVMetalTextureCacheRef over it: what the presenter
    // and the scaler read pictures with (native_device_mac.mm).
    void* metal() const { return metal_; }
    void* texture_cache() const { return texture_cache_; }
    // FFmpeg's VideoToolbox decoding device, for the decoders (a reference
    // of their own each).
    AVBufferRef* hardware() const { return hardware_; }
    // The lock the device's work takes, as on Windows: the scaler's and the
    // presenter's.
    std::recursive_mutex& lock() { return mutex_; }
    bool lost() const { return false; }
    // The codecs VideoToolbox decodes on this Mac's GPU, as the self-test
    // names them ("H.264", "AV1"...).
    std::vector<std::string> gpu_decoders() const;
    // A pixel buffer `width` x `height` (even) for a picture, 10-bit
    // (P010's layout) or 8 (NV12's), from pools kept for each size and
    // format; nothing when none can be made.
    HeldPixelBuffer take_buffer(int width, int height, bool ten_bit);
    // The part `crop` (x, y, width, height) of `picture` scaled to `size`, by
    // the GPU (an area filter, each output pixel the average of the source
    // pixels it covers); nothing when it cannot be.
    HeldPixelBuffer scale(PixelBuffer picture, const std::array<int, 4>& crop, std::pair<int, int> size);

private:
    struct Pools;
    void* metal_ = nullptr;
    void* texture_cache_ = nullptr;
    std::unique_ptr<Pools> pools_;
    AVBufferRef* hardware_ = nullptr;
    std::recursive_mutex mutex_;
};
#endif

}  // namespace vml::app
