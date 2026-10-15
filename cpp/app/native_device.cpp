#include "native_device.h"

#include <dxgi1_2.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "ffmpeg_api.h"
#include "vml/model_select.h"
#include "vml/platform/library.h"
#include "vml/platform/system.h"

using Microsoft::WRL::ComPtr;

namespace vml::app {

std::optional<std::string> load_ffmpeg_libraries() {
    static std::mutex mutex;
    static std::vector<platform::DynamicLibrary> loaded;
    static std::optional<std::string> problem;
    static bool tried = false;
    const std::lock_guard lock(mutex);
    if (tried) {
        return problem;
    }
    tried = true;
    // In the order they import each other: each finds the ones before it
    // already loaded.
    constexpr std::array<const char*, 5> names{"avutil-61.dll", "swresample-7.dll", "avcodec-63.dll",
                                               "avformat-63.dll", "swscale-10.dll"};
    const std::filesystem::path folder = resource_folder() / "tools" / "ffmpeg";
    try {
        for (const char* name : names) {
            loaded.emplace_back(folder / name);
        }
    } catch (const std::exception& error) {
        loaded.clear();
        problem = std::string("FFmpeg's libraries are not bundled (") + error.what() + ")";
        return problem;
    }
    if (avutil_version() >> 16 != LIBAVUTIL_VERSION_MAJOR || avcodec_version() >> 16 != LIBAVCODEC_VERSION_MAJOR ||
        avformat_version() >> 16 != LIBAVFORMAT_VERSION_MAJOR || swscale_version() >> 16 != LIBSWSCALE_VERSION_MAJOR) {
        problem = "FFmpeg's libraries are not the version this app was built for";
        return problem;
    }
    // FFmpeg says nothing on the window's console: a GPU that cannot decode
    // a video (FFmpeg logs it) is told in the status line, which says which
    // decoder decodes it.
    av_log_set_level(AV_LOG_QUIET);
    return problem;
}

std::shared_ptr<NativeDevice> NativeDevice::shared() {
    static std::mutex mutex;
    static std::shared_ptr<NativeDevice> device;
    const std::lock_guard lock(mutex);
    if (!device || device->lost()) {
        device = std::make_shared<NativeDevice>(platform::first_gpu_adapter());
    }
    return device;
}

NativeDevice::NativeDevice(unsigned adapter_index) {
    ComPtr<IDXGIFactory1> factory;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumAdapters1(adapter_index, &adapter))) {
        throw std::runtime_error("D3D11 device unavailable");
    }
    // Video support for the decoders, BGRA for the presenter's swapchain.
    constexpr std::array levels{D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    const HRESULT made = D3D11CreateDevice(
        adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        levels.data(), static_cast<UINT>(levels.size()), D3D11_SDK_VERSION, &device_, nullptr, &context_);
    if (FAILED(made)) {
        throw std::runtime_error("D3D11 device unavailable");
    }
    ComPtr<ID3D10Multithread> multithread;
    if (SUCCEEDED(device_.As(&multithread))) {
        multithread->SetMultithreadProtected(TRUE);
    }
    hardware_ = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA);
    if (hardware_ == nullptr) {
        throw std::runtime_error("FFmpeg's D3D11 decoding device could not be made");
    }
    auto* context = reinterpret_cast<AVHWDeviceContext*>(hardware_->data);
    auto* d3d11 = static_cast<AVD3D11VADeviceContext*>(context->hwctx);
    device_->AddRef();  // FFmpeg's, released when its device goes
    d3d11->device = device_.Get();
    d3d11->lock = [](void* mutex) { static_cast<std::recursive_mutex*>(mutex)->lock(); };
    d3d11->unlock = [](void* mutex) { static_cast<std::recursive_mutex*>(mutex)->unlock(); };
    d3d11->lock_ctx = &mutex_;
    if (av_hwdevice_ctx_init(hardware_) < 0) {
        av_buffer_unref(&hardware_);
        throw std::runtime_error("FFmpeg's D3D11 decoding device could not be made");
    }
}

NativeDevice::~NativeDevice() { av_buffer_unref(&hardware_); }

bool NativeDevice::lost() const { return device_->GetDeviceRemovedReason() != S_OK; }

std::vector<std::string> NativeDevice::gpu_decoders() const {
    // The profiles FFmpeg's D3D11VA decoders use, in this order.
    const std::array<std::pair<GUID, const char*>, 8> known{{
        {D3D11_DECODER_PROFILE_H264_VLD_NOFGT, "H.264"},
        {D3D11_DECODER_PROFILE_HEVC_VLD_MAIN, "HEVC"},
        {D3D11_DECODER_PROFILE_HEVC_VLD_MAIN10, "HEVC 10-bit"},
        {D3D11_DECODER_PROFILE_AV1_VLD_PROFILE0, "AV1"},
        {D3D11_DECODER_PROFILE_VP9_VLD_PROFILE0, "VP9"},
        {D3D11_DECODER_PROFILE_VP9_VLD_10BIT_PROFILE2, "VP9 10-bit"},
        {D3D11_DECODER_PROFILE_MPEG2_VLD, "MPEG-2"},
        {D3D11_DECODER_PROFILE_MPEG2and1_VLD, "MPEG-2"},
    }};
    std::vector<std::string> found;
    ComPtr<ID3D11VideoDevice> video;
    if (FAILED(device_.As(&video))) {
        return found;
    }
    std::vector<GUID> profiles(video->GetVideoDecoderProfileCount());
    for (UINT index = 0; index < profiles.size(); ++index) {
        video->GetVideoDecoderProfile(index, &profiles[index]);
    }
    for (const auto& [profile, name] : known) {
        const bool taken = std::any_of(profiles.begin(), profiles.end(),
                                       [&profile](const GUID& offered) { return offered == profile; });
        if (taken && std::find(found.begin(), found.end(), name) == found.end()) {
            found.emplace_back(name);
        }
    }
    return found;
}

}  // namespace vml::app
