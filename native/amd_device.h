// The GPU AMD's decoders (amf_frames.cpp: AMF; mf_frames.cpp: Windows' own)
// decode on. The app names it as DirectX numbers it (Params::device, and
// nvf_supports' device): the GPU chosen to score on where that is an AMD one,
// else the first AMD GPU, the adapter FFmpeg's D3D11VA decodes on too
// (hardware_decoding.h's d3d11va_adapter). Its pictures are then handed over to
// Vulkan on the same GPU the metrics score on, and a GPU the choice keeps free
// decodes nothing. A negative number (the app's tests, which name none) takes
// the first AMD GPU DirectX lists.

#pragma once

#include <d3d11.h>
#include <dxgi.h>

#include <string>

// A Direct3D 11 device with video support on that GPU, or nullptr with
// `error` set.
inline ID3D11Device *amd_video_device(int adapter_index, std::string &error) {
    IDXGIFactory1 *factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&factory)))) {
        error = "DirectX could not list the GPUs";
        return nullptr;
    }
    ID3D11Device *device = nullptr;
    IDXGIAdapter1 *adapter = nullptr;
    bool named_is_amd = false;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; index++) {
        DXGI_ADAPTER_DESC1 desc;
        const bool amd = SUCCEEDED(adapter->GetDesc1(&desc)) && (desc.VendorId == 0x1002 || desc.VendorId == 0x1022);
        const bool wanted = adapter_index < 0 ? amd : static_cast<int>(index) == adapter_index;
        if (wanted && amd) {
            named_is_amd = true;
            D3D_FEATURE_LEVEL level;
            ID3D11DeviceContext *context = nullptr;
            HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                                           nullptr, 0, D3D11_SDK_VERSION, &device, &level, &context);
            if (context) context->Release();
            if (FAILED(hr)) device = nullptr;
        }
        adapter->Release();
        if (device || (wanted && adapter_index >= 0)) break;
    }
    factory->Release();
    if (!device) {
        if (adapter_index < 0) {
            error = "no AMD GPU with a Direct3D 11 video device";
        } else if (named_is_amd) {
            error = "DirectX's GPU " + std::to_string(adapter_index) + " (AMD's) has no Direct3D 11 video device";
        } else {
            error = "DirectX lists no AMD GPU numbered " + std::to_string(adapter_index);
        }
    }
    return device;
}
