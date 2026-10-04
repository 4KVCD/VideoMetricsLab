// Scaling on the GPU for the decoders whose pictures are Direct3D 11 textures
// (Intel's through oneVPL, AMD's through AMF): scale_filter.h's filters as a
// compute shader on the decoded NV12/P010 texture, so only the scaled picture
// comes to system memory.
//
// The picture is scale_plane's, sample for sample: the same weights (made on
// the CPU by plane_filter and handed to the shader), summed in the same order
// in IEEE single precision (`precise`: no fused multiply-add, nothing
// reordered), the same pass first, the same cap between the passes, the same
// rounding. The decoders check the first pictures against the CPU's and
// scale on the CPU if a driver's ever differ (GPU_SCALE_CHECKED).
//
// One picture: the decoded texture is copied on the GPU to a texture of the
// same format the shader may read (a decoder's own often may not be), two
// passes filter the luma plane and two the chroma plane (U and V together),
// and the results are read back from staging textures. D3DCompiler_47.dll is
// Windows' own (System32), loaded at run time.

#pragma once

#include <d3d10.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu_frames.h"

namespace gpu_scale {

using Microsoft::WRL::ComPtr;

// How many of a session's first pictures the decoders also scale on the CPU,
// to compare.
constexpr int GPU_SCALE_CHECKED = 3;

// COMPONENTS: 1 for luma, 2 for chroma (U and V at once). FIRST: the pass
// from the picture to floats; otherwise the pass from those to the output.
const char kShader[] = R"(
#if COMPONENTS == 1
#define VALUE float
#define SAMPLE uint
#else
#define VALUE float2
#define SAMPLE uint2
#endif
cbuffer Constants : register(b0) {
    int taps;        // weights for each output sample
    int vertical;    // the direction filtered
    int limit;       // the input's last sample that way
    int origin_x;    // the crop's corner in the picture (first pass)
    int origin_y;
    float scale;     // a normalised sample's largest value: 255 or 65535
    int in_shift;
    float cap;       // swscale's intermediate saturates here
    float top;       // the largest sample
    int out_shift;
    int out_w;
    int out_h;
};
Texture2D<VALUE> source : register(t0);
Buffer<float> weights : register(t1);
Buffer<int> starts : register(t2);
#if FIRST
RWTexture2D<VALUE> target : register(u0);
#else
RWTexture2D<SAMPLE> target : register(u0);
#endif

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if ((int)id.x >= out_w || (int)id.y >= out_h) return;
    int along = vertical ? (int)id.y : (int)id.x;
    int start = starts[along];
    int base = along * taps;
    precise VALUE sum = 0;
    for (int k = 0; k < taps; k++) {
        int at = clamp(start + k, 0, limit);
        int2 where = vertical ? int2(id.x, at) : int2(at, id.y);
#if FIRST
        SAMPLE whole = (SAMPLE)(source.Load(int3(where + int2(origin_x, origin_y), 0)) * scale + 0.5);
        VALUE value = (VALUE)(whole >> in_shift);
#else
        VALUE value = source.Load(int3(where, 0));
#endif
        precise VALUE term = weights[base + k] * value;
        sum = sum + term;
    }
#if FIRST
    target[id.xy] = min(sum, cap);
#else
    target[id.xy] = ((SAMPLE)clamp(round(sum), 0, top)) << out_shift;
#endif
}
)";

struct PassConstants {
    int32_t taps, vertical, limit, origin_x, origin_y;
    float scale;
    int32_t in_shift;
    float cap, top;
    int32_t out_shift, out_w, out_h;
};

typedef HRESULT(WINAPI *CompileFunction)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO *, ID3DInclude *, LPCSTR,
                                         LPCSTR, UINT, UINT, ID3DBlob **, ID3DBlob **);

inline std::string hresult_text(const char *what, HRESULT hr) {
    char text[160];
    snprintf(text, sizeof text, "%s (0x%08lX)", what, static_cast<unsigned long>(hr));
    return text;
}

// One plane's two passes: luma, or U and V together.
struct PlanePasses {
    int components = 1;
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0, mid_w = 0, mid_h = 0;
    bool vertical_first = true;
    PassConstants constants[2]{};
    ComPtr<ID3D11ShaderResourceView> picture;  // this plane of the copy
    ComPtr<ID3D11Texture2D> mid, out, staging;
    ComPtr<ID3D11UnorderedAccessView> mid_write, out_write;
    ComPtr<ID3D11ShaderResourceView> mid_read;
    ComPtr<ID3D11Buffer> constant_buffers[2];
    ComPtr<ID3D11ShaderResourceView> weights[2], starts[2];
    ComPtr<ID3D11ComputeShader> shaders[2];
};

class Scaler {
public:
    // Made for the session's first picture: its device, format and size.
    bool start(ID3D11Texture2D *picture, const Params &params, std::string &error) {
        p_ = params;
        wide_ = params.bit_depth > 8;
        picture->GetDevice(&device_);
        device_->GetImmediateContext(&context_);
        if (device_->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0) {
            error = "the GPU has no Direct3D 11 compute shaders";
            return false;
        }
        // The decoder uses this device from its own threads.
        if (FAILED(device_.As(&lock_))) {
            error = "the Direct3D device cannot be shared between threads";
            return false;
        }
        lock_->SetMultithreadProtected(TRUE);
        if (params.crop_w % 2 || params.crop_h % 2 || params.widen) {
            error = "this crop is scaled on the CPU";
            return false;
        }
        // Loaded once, and kept.
        static HMODULE compiler = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        compile_ = compiler ? reinterpret_cast<CompileFunction>(
                                  reinterpret_cast<void *>(GetProcAddress(compiler, "D3DCompile")))
                            : nullptr;
        if (!compile_) {
            error = "Windows' shader compiler (d3dcompiler_47.dll) is missing";
            return false;
        }
        if (!make_copy(picture, error)) return false;
        const int ow = out_width(p_), oh = out_height(p_);
        luma_.components = 1;
        if (!make_plane(luma_, p_.crop_w, p_.crop_h, ow, oh, error)) return false;
        if (!p_.luma_only) {
            chroma_.components = 2;
            if (!make_plane(chroma_, p_.crop_w / 2, p_.crop_h / 2, (ow + 1) / 2, (oh + 1) / 2, error)) return false;
        }
        started_ = true;
        return true;
    }

    bool started() const { return started_; }

    // Scales `picture`'s slice (the crop's corner `offset_x`, `offset_y`
    // further in, where a picture starts inside its texture) into `out`,
    // laid out as scale_frame lays it out.
    bool scale(ID3D11Texture2D *picture, UINT slice, int offset_x, int offset_y, uint8_t *out, std::string &error) {
        D3D11_TEXTURE2D_DESC desc;
        picture->GetDesc(&desc);
        if (desc.Width != copy_desc_.Width || desc.Height != copy_desc_.Height || desc.Format != copy_desc_.Format) {
            error = "the decoder's pictures changed";
            return false;
        }
        if (slice >= desc.ArraySize) {
            error = "the decoder named a picture its texture does not have";
            return false;
        }
        const int x = p_.crop_x + offset_x, y = p_.crop_y + offset_y;
        if (x % 2 || y % 2 || x + p_.crop_w > static_cast<int>(desc.Width)
            || y + p_.crop_h > static_cast<int>(desc.Height)) {
            error = "the crop is outside the decoder's picture";
            return false;
        }
        lock_->Enter();
        set_origin(luma_, x, y);
        if (!p_.luma_only) set_origin(chroma_, x / 2, y / 2);
        context_->CopySubresourceRegion(copy_.Get(), 0, 0, 0, 0, picture, slice, nullptr);
        run(luma_);
        if (!p_.luma_only) run(chroma_);
        ID3D11ShaderResourceView *no_views[3] = {};
        ID3D11UnorderedAccessView *no_target[1] = {};
        context_->CSSetShaderResources(0, 3, no_views);
        context_->CSSetUnorderedAccessViews(0, 1, no_target, nullptr);
        context_->CSSetShader(nullptr, nullptr, 0);
        context_->CopyResource(luma_.staging.Get(), luma_.out.Get());
        if (!p_.luma_only) context_->CopyResource(chroma_.staging.Get(), chroma_.out.Get());
        D3D11_MAPPED_SUBRESOURCE luma{}, chroma{};
        HRESULT hr = context_->Map(luma_.staging.Get(), 0, D3D11_MAP_READ, 0, &luma);
        if (SUCCEEDED(hr) && !p_.luma_only) {
            hr = context_->Map(chroma_.staging.Get(), 0, D3D11_MAP_READ, 0, &chroma);
            if (FAILED(hr)) context_->Unmap(luma_.staging.Get(), 0);
        }
        lock_->Leave();
        if (FAILED(hr)) {
            error = hresult_text("reading the scaled picture failed", hr);
            return false;
        }
        const size_t sample = wide_out(p_) ? 2 : 1;
        const size_t ow = luma_.out_w, oh = luma_.out_h;
        for (size_t row = 0; row < oh; row++)
            memcpy(out + row * ow * sample, static_cast<const uint8_t *>(luma.pData) + row * luma.RowPitch,
                   ow * sample);
        if (!p_.luma_only) {
            const size_t cw = chroma_.out_w, ch = chroma_.out_h;
            uint8_t *u = out + ow * oh * sample, *v = u + cw * ch * sample;
            for (size_t row = 0; row < ch; row++) {
                const uint8_t *pairs = static_cast<const uint8_t *>(chroma.pData) + row * chroma.RowPitch;
                if (sample == 2) {
                    const uint16_t *from = reinterpret_cast<const uint16_t *>(pairs);
                    uint16_t *ur = reinterpret_cast<uint16_t *>(u) + row * cw;
                    uint16_t *vr = reinterpret_cast<uint16_t *>(v) + row * cw;
                    for (size_t i = 0; i < cw; i++) {
                        ur[i] = from[2 * i];
                        vr[i] = from[2 * i + 1];
                    }
                } else {
                    uint8_t *ur = u + row * cw, *vr = v + row * cw;
                    for (size_t i = 0; i < cw; i++) {
                        ur[i] = pairs[2 * i];
                        vr[i] = pairs[2 * i + 1];
                    }
                }
            }
        }
        lock_->Enter();
        context_->Unmap(luma_.staging.Get(), 0);
        if (!p_.luma_only) context_->Unmap(chroma_.staging.Get(), 0);
        lock_->Leave();
        return true;
    }

private:
    Params p_{};
    bool wide_ = false, started_ = false;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D10Multithread> lock_;
    CompileFunction compile_ = nullptr;
    ComPtr<ID3D11Texture2D> copy_;
    D3D11_TEXTURE2D_DESC copy_desc_{};
    PlanePasses luma_, chroma_;

    // The texture the pictures are copied to: the decoder's format and size,
    // readable by a shader.
    bool make_copy(ID3D11Texture2D *picture, std::string &error) {
        D3D11_TEXTURE2D_DESC desc;
        picture->GetDesc(&desc);
        if (desc.Format != (wide_ ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12)) {
            error = "the decoder's pictures are not NV12 or P010";
            return false;
        }
        copy_desc_ = desc;
        desc.ArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;
        HRESULT hr = device_->CreateTexture2D(&desc, nullptr, &copy_);
        if (FAILED(hr)) error = hresult_text("the GPU could not make a picture to scale from", hr);
        return SUCCEEDED(hr);
    }

    bool make_view(ID3D11Buffer *buffer, DXGI_FORMAT format, UINT count, ComPtr<ID3D11ShaderResourceView> &view) {
        D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Format = format;
        desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = count;
        return SUCCEEDED(device_->CreateShaderResourceView(buffer, &desc, &view));
    }

    bool make_filter_views(const Filter &filter, ComPtr<ID3D11ShaderResourceView> &weights,
                           ComPtr<ID3D11ShaderResourceView> &starts) {
        D3D11_BUFFER_DESC desc{};
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data{};
        ComPtr<ID3D11Buffer> weight_buffer, start_buffer;
        desc.ByteWidth = static_cast<UINT>(filter.weights.size() * sizeof(float));
        data.pSysMem = filter.weights.data();
        if (FAILED(device_->CreateBuffer(&desc, &data, &weight_buffer))) return false;
        desc.ByteWidth = static_cast<UINT>(filter.starts.size() * sizeof(int32_t));
        data.pSysMem = filter.starts.data();
        if (FAILED(device_->CreateBuffer(&desc, &data, &start_buffer))) return false;
        return make_view(weight_buffer.Get(), DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(filter.weights.size()), weights)
               && make_view(start_buffer.Get(), DXGI_FORMAT_R32_SINT, static_cast<UINT>(filter.starts.size()), starts);
    }

    bool make_texture(int width, int height, DXGI_FORMAT format, bool staging, ComPtr<ID3D11Texture2D> &texture) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = static_cast<UINT>(width);
        desc.Height = static_cast<UINT>(height);
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.Usage = staging ? D3D11_USAGE_STAGING : D3D11_USAGE_DEFAULT;
        desc.BindFlags = staging ? 0 : D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = staging ? D3D11_CPU_ACCESS_READ : 0;
        return SUCCEEDED(device_->CreateTexture2D(&desc, nullptr, &texture));
    }

    bool make_shader(int components, bool first, ComPtr<ID3D11ComputeShader> &shader, std::string &error) {
        const D3D_SHADER_MACRO macros[] = {{"COMPONENTS", components == 1 ? "1" : "2"},
                                           {"FIRST", first ? "1" : "0"},
                                           {nullptr, nullptr}};
        ComPtr<ID3DBlob> code, messages;
        // D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS
        HRESULT hr = compile_(kShader, strlen(kShader), "scale", macros, nullptr, "main", "cs_5_0",
                              (1 << 15) | (1 << 13), 0, &code, &messages);
        if (SUCCEEDED(hr))
            hr = device_->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader);
        if (FAILED(hr)) {
            error = hresult_text("the scaling shader could not be made", hr);
            if (messages && messages->GetBufferSize())
                error += ": " + std::string(static_cast<const char *>(messages->GetBufferPointer()),
                                            messages->GetBufferSize());
        }
        return SUCCEEDED(hr);
    }

    // One plane's passes, as scale_plane makes them: down the columns first
    // where that shrinks the picture.
    bool make_plane(PlanePasses &plane, int w, int h, int ow, int oh, std::string &error) {
        plane.in_w = w;
        plane.in_h = h;
        plane.out_w = ow;
        plane.out_h = oh;
        plane.vertical_first = oh <= h;
        plane.mid_w = plane.vertical_first ? w : ow;
        plane.mid_h = plane.vertical_first ? oh : h;
        const Filter horizontal = plane_filter(w, ow, p_.scaler), vertical = plane_filter(h, oh, p_.scaler);
        const Filter &first = plane.vertical_first ? vertical : horizontal;
        const Filter &second = plane.vertical_first ? horizontal : vertical;
        const bool two = plane.components == 2;
        const float cap = wide_ ? kIntermediateCap10 : kIntermediateCap8;
        PassConstants &a = plane.constants[0], &b = plane.constants[1];
        a.taps = first.taps;
        a.vertical = plane.vertical_first;
        a.limit = (plane.vertical_first ? h : w) - 1;
        a.scale = wide_ ? 65535.0f : 255.0f;
        a.in_shift = wide_ ? 6 : 0;  // P010 keeps its 10 bits at the top
        a.cap = cap;
        a.out_w = plane.mid_w;
        a.out_h = plane.mid_h;
        b.taps = second.taps;
        b.vertical = !plane.vertical_first;
        b.limit = (plane.vertical_first ? w : h) - 1;
        b.top = wide_ ? 1023.0f : 255.0f;
        b.out_shift = wide_ && p_.shift == 0 ? 6 : 0;
        b.out_w = ow;
        b.out_h = oh;

        D3D11_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = wide_ ? (two ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R16_UNORM)
                            : (two ? DXGI_FORMAT_R8G8_UNORM : DXGI_FORMAT_R8_UNORM);
        view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        view.Texture2D.MipLevels = 1;
        HRESULT hr = device_->CreateShaderResourceView(copy_.Get(), &view, &plane.picture);
        if (FAILED(hr)) {
            error = hresult_text("the GPU cannot read the decoder's picture format in a shader", hr);
            return false;
        }
        const DXGI_FORMAT mid_format = two ? DXGI_FORMAT_R32G32_FLOAT : DXGI_FORMAT_R32_FLOAT;
        const DXGI_FORMAT out_format = wide_out(p_) ? (two ? DXGI_FORMAT_R16G16_UINT : DXGI_FORMAT_R16_UINT)
                                                    : (two ? DXGI_FORMAT_R8G8_UINT : DXGI_FORMAT_R8_UINT);
        bool made = make_texture(plane.mid_w, plane.mid_h, mid_format, false, plane.mid)
                    && make_texture(ow, oh, out_format, false, plane.out)
                    && make_texture(ow, oh, out_format, true, plane.staging)
                    && SUCCEEDED(device_->CreateUnorderedAccessView(plane.mid.Get(), nullptr, &plane.mid_write))
                    && SUCCEEDED(device_->CreateUnorderedAccessView(plane.out.Get(), nullptr, &plane.out_write))
                    && SUCCEEDED(device_->CreateShaderResourceView(plane.mid.Get(), nullptr, &plane.mid_read))
                    && make_filter_views(first, plane.weights[0], plane.starts[0])
                    && make_filter_views(second, plane.weights[1], plane.starts[1]);
        for (int pass = 0; made && pass < 2; pass++) {
            D3D11_BUFFER_DESC desc{};
            desc.ByteWidth = sizeof(PassConstants);
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            D3D11_SUBRESOURCE_DATA data{};
            data.pSysMem = &plane.constants[pass];
            made = SUCCEEDED(device_->CreateBuffer(&desc, &data, &plane.constant_buffers[pass]));
        }
        if (!made) {
            error = "the GPU could not make the scaler's textures";
            return false;
        }
        return make_shader(plane.components, true, plane.shaders[0], error)
               && make_shader(plane.components, false, plane.shaders[1], error);
    }

    void set_origin(PlanePasses &plane, int x, int y) {
        PassConstants &first = plane.constants[0];
        if (first.origin_x == x && first.origin_y == y) return;
        first.origin_x = x;
        first.origin_y = y;
        context_->UpdateSubresource(plane.constant_buffers[0].Get(), 0, nullptr, &first, 0, 0);
    }

    void run(PlanePasses &plane) {
        for (int pass = 0; pass < 2; pass++) {
            const bool first = pass == 0;
            ID3D11UnorderedAccessView *target[1] = {first ? plane.mid_write.Get() : plane.out_write.Get()};
            ID3D11ShaderResourceView *views[3] = {first ? plane.picture.Get() : plane.mid_read.Get(),
                                                  plane.weights[pass].Get(), plane.starts[pass].Get()};
            ID3D11Buffer *constants[1] = {plane.constant_buffers[pass].Get()};
            // The target first: the first pass's is the second's input.
            ID3D11ShaderResourceView *no_view[1] = {};
            context_->CSSetShaderResources(0, 1, no_view);
            context_->CSSetUnorderedAccessViews(0, 1, target, nullptr);
            context_->CSSetShaderResources(0, 3, views);
            context_->CSSetConstantBuffers(0, 1, constants);
            context_->CSSetShader(plane.shaders[pass].Get(), nullptr, 0);
            const int w = first ? plane.mid_w : plane.out_w, h = first ? plane.mid_h : plane.out_h;
            context_->Dispatch(static_cast<UINT>((w + 7) / 8), static_cast<UINT>((h + 7) / 8), 1);
        }
    }
};

// For the tests: scales one NV12/P010 picture (`picture`: the luma rows, then
// the interleaved chroma rows, `params.width` samples each, 10-bit samples at
// the top of 16) on the CPU, or on the first GPU of `vendor` (PCI vendor id;
// 0x1414 is Windows' software device). 0 scaled; 1 there is no such GPU;
// -1 failed, with the reason.
inline int scale_test(int vendor, const Params &params, const uint8_t *picture, int on_gpu, uint8_t *out,
                      std::string &error) {
    const size_t sample = params.bit_depth > 8 ? 2 : 1;
    const size_t pitch = static_cast<size_t>(params.width) * sample;
    if (!on_gpu) {
        PlaneScaler scaler;
        prepare_scaler(params, scaler);
        scale_frame(params, scaler, picture, picture + pitch * params.height, pitch, true, out);
        return 0;
    }
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(factory.GetAddressOf())))) {
        error = "DirectX could not list the GPUs";
        return -1;
    }
    ComPtr<ID3D11Device> device;
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT index = 0; !device && factory->EnumAdapters1(index, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND;
         index++) {
        DXGI_ADAPTER_DESC1 desc;
        if (FAILED(adapter->GetDesc1(&desc)) || desc.VendorId != static_cast<UINT>(vendor)) continue;
        D3D_FEATURE_LEVEL level;
        ComPtr<ID3D11DeviceContext> context;
        if (FAILED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                     &device, &level, &context)))
            device.Reset();
    }
    if (!device) return 1;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = static_cast<UINT>(params.width);
    desc.Height = static_cast<UINT>(params.height);
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = params.bit_depth > 8 ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA data{};
    data.pSysMem = picture;
    data.SysMemPitch = static_cast<UINT>(pitch);
    ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = device->CreateTexture2D(&desc, &data, &texture);
    if (FAILED(hr)) {
        error = hresult_text("this GPU has no NV12/P010 textures", hr);
        return 1;
    }
    Scaler scaler;
    if (!scaler.start(texture.Get(), params, error) || !scaler.scale(texture.Get(), 0, 0, 0, out, error)) return -1;
    return 0;
}

}  // namespace gpu_scale
