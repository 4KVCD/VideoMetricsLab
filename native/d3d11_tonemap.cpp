// Native playback: private RGBA16 GPU surface processing for the GStreamer
// branch, and the frames' presentation. No GStreamer ABI dependencies, pixel
// readback, or CPU frame allocation.
#include <d3d11.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <new>
using Microsoft::WRL::ComPtr;

static const char shader[] = R"(
Texture2D<float4> inputFrame : register(t0);
cbuffer Parameters : register(b0) {
    float kind; float peak; float white; float hdrOut;
    float4 luma;         // the source primaries' luminance weights
    float4 toOutput[3];  // linear source RGB -> linear BT.709 (SDR) or BT.2020 (hdrOut), by rows
};
float4 vs(uint id : SV_VertexID) : SV_Position {
    return float4(id == 2 ? 3 : -1, id == 1 ? 3 : -1, 0, 1);
}
float3 pq(float3 x) {
    float3 p = pow(max(x, 0), 1.0 / 78.84375);
    return 10000 * pow(max(p - 0.8359375, 0) / max(18.8515625 - 18.6875*p, 1e-6), 1.0 / 0.1593017578125);
}
float3 pq_code(float3 nits) {
    float3 y = pow(saturate(nits / 10000), 0.1593017578125);
    return pow((0.8359375 + 18.8515625*y) / (1 + 18.6875*y), 78.84375);
}
float3 hlg_scene(float3 x) {
    float3 lo = x*x/3;
    float3 hi = (exp((x-0.55991073)/0.17883277)+0.28466892)/12;
    return float3(x.r<=0.5?lo.r:hi.r, x.g<=0.5?lo.g:hi.g, x.b<=0.5?lo.b:hi.b);
}
float3 hlg(float3 x) {
    float3 scene = hlg_scene(x);
    return scene * pow(max(dot(scene, luma.rgb), 1e-8), .2) * peak;
}
float3 converted(float3 x) {
    return float3(dot(x,toOutput[0].rgb), dot(x,toOutput[1].rgb), dot(x,toOutput[2].rgb));
}
float3 srgb(float3 x) {
    float3 lo=12.92*x, hi=1.055*pow(max(x,0),1.0/2.4)-.055;
    return float3(x.r<=.0031308?lo.r:hi.r, x.g<=.0031308?lo.g:hi.g, x.b<=.0031308?lo.b:hi.b);
}
float4 ps(float4 pos : SV_Position) : SV_Target {
    float3 encoded = inputFrame.Load(int3(pos.xy,0)).rgb;
    float3 light = kind < 1.5 ? pq(encoded) : hlg(encoded);
    if (hdrOut > 0.5) {
        // HDR kept, for an HDR display, which takes PQ in BT.2020: the light
        // converted to BT.2020 and coded as PQ (HLG's at its 1000-nit peak).
        return float4(pq_code(converted(light)), 1);
    }
    // Fixed extended-Reinhard luminance curve, shared by all comparison sides.
    float y = max(dot(light,luma.rgb)/white,0);
    float w = peak/white;
    float mapped = y*(1+y/(w*w))/(1+y);
    light = light/white * (y > 1e-8 ? mapped/y : 0);
    // Linear source primaries -> BT.709, then gamut clipping and sRGB encoding.
    return float4(srgb(saturate(converted(light))),1);
}
)";

struct Mapper {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> immediate, deferred;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11Buffer> parameters;
    ComPtr<ID3D11Texture2D> scratch;
    ComPtr<ID3D11ShaderResourceView> srv;
    UINT width=0, height=0;
};

static void* create(ID3D11Resource* resource, int kind, bool hdr_out, const float* luma, const float* to_output) {
    if (!luma || !to_output) return nullptr;
    auto m = new(std::nothrow) Mapper;
    if (!m || !resource) { delete m; return nullptr; }
    resource->GetDevice(&m->device);
    m->device->GetImmediateContext(&m->immediate);
    ComPtr<ID3DBlob> vs, ps, errors;
    HRESULT hr = m->device->CreateDeferredContext(0,&m->deferred);
    if (SUCCEEDED(hr)) hr=D3DCompile(shader,strlen(shader),nullptr,nullptr,nullptr,"vs","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vs,&errors);
    if (SUCCEEDED(hr)) hr=D3DCompile(shader,strlen(shader),nullptr,nullptr,nullptr,"ps","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&ps,&errors);
    if (SUCCEEDED(hr)) hr=m->device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&m->vs);
    if (SUCCEEDED(hr)) hr=m->device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&m->ps);
    const float* t = to_output;
    float params[20]={float(kind),1000,100,hdr_out?1.f:0.f, luma[0],luma[1],luma[2],0,
                      t[0],t[1],t[2],0, t[3],t[4],t[5],0, t[6],t[7],t[8],0};
    D3D11_BUFFER_DESC bd={}; bd.ByteWidth=sizeof(params); bd.Usage=D3D11_USAGE_IMMUTABLE; bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial={}; initial.pSysMem=params;
    if (SUCCEEDED(hr)) hr=m->device->CreateBuffer(&bd,&initial,&m->parameters);
    if (FAILED(hr)) { delete m; return nullptr; }
    return m;
}

// HDR to SDR. `luma`: the source primaries' luminance weights (3);
// `to_bt709`: linear source RGB to linear BT.709, by rows (9).
extern "C" __declspec(dllexport) void* vmaf_tonemap_create_primaries(ID3D11Resource* resource, int kind,
                                                                     const float* luma, const float* to_bt709) {
    return create(resource, kind, false, luma, to_bt709);
}

// HDR kept, for an HDR display, which takes PQ in BT.2020: `luma`, the source
// primaries' luminance weights (3, for HLG's display light); `to_bt2020`,
// linear source RGB to linear BT.2020, by rows (9). Coded as PQ.
extern "C" __declspec(dllexport) void* vmaf_hdr_convert_create(ID3D11Resource* resource, int kind,
                                                               const float* luma, const float* to_bt2020) {
    return create(resource, kind, true, luma, to_bt2020);
}

// BT.2020 primaries: what the mapper took before it was given any.
extern "C" __declspec(dllexport) void* vmaf_tonemap_create(ID3D11Resource* resource, int kind) {
    static const float luma[3]={.2627f,.678f,.0593f};
    static const float to_bt709[9]={1.660491f,-.587641f,-.072850f, -.124550f,1.132900f,-.008349f,
                                    -.018151f,-.100579f,1.118730f};
    return vmaf_tonemap_create_primaries(resource, kind, luma, to_bt709);
}

// Caller holds the owning GstD3D11Device lock throughout this operation.
extern "C" __declspec(dllexport) int vmaf_tonemap_render(void* opaque, ID3D11Resource* resource) {
    auto m=static_cast<Mapper*>(opaque);
    if (!m || !resource) return E_INVALIDARG;
    ComPtr<ID3D11Texture2D> frame;
    HRESULT hr=resource->QueryInterface(__uuidof(ID3D11Texture2D),reinterpret_cast<void**>(frame.GetAddressOf()));
    if (FAILED(hr)) return hr;
    D3D11_TEXTURE2D_DESC desc; frame->GetDesc(&desc);
    ComPtr<ID3D11Device> owner; resource->GetDevice(&owner);
    if (owner.Get()!=m->device.Get() || desc.Format!=DXGI_FORMAT_R16G16B16A16_UNORM || desc.ArraySize!=1 || desc.SampleDesc.Count!=1 || desc.MipLevels!=1)
        return E_INVALIDARG;
    if (m->width!=desc.Width || m->height!=desc.Height) {
        m->srv.Reset(); m->scratch.Reset();
        auto sd=desc; sd.BindFlags=D3D11_BIND_SHADER_RESOURCE; sd.MiscFlags=0; sd.CPUAccessFlags=0; sd.Usage=D3D11_USAGE_DEFAULT;
        hr=m->device->CreateTexture2D(&sd,nullptr,&m->scratch);
        if (SUCCEEDED(hr)) hr=m->device->CreateShaderResourceView(m->scratch.Get(),nullptr,&m->srv);
        if (FAILED(hr)) return hr;
        m->width=desc.Width; m->height=desc.Height;
    }
    ComPtr<ID3D11RenderTargetView> rtv;
    hr=m->device->CreateRenderTargetView(frame.Get(),nullptr,&rtv);
    if (FAILED(hr)) return hr;
    auto c=m->deferred.Get();
    c->CopyResource(m->scratch.Get(),frame.Get());
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->VSSetShader(m->vs.Get(),nullptr,0); c->PSSetShader(m->ps.Get(),nullptr,0);
    c->PSSetShaderResources(0,1,m->srv.GetAddressOf()); c->PSSetConstantBuffers(0,1,m->parameters.GetAddressOf());
    c->OMSetRenderTargets(1,rtv.GetAddressOf(),nullptr);
    D3D11_VIEWPORT viewport={0,0,float(desc.Width),float(desc.Height),0,1}; c->RSSetViewports(1,&viewport);
    c->Draw(3,0);
    ComPtr<ID3D11CommandList> commands;
    hr=c->FinishCommandList(FALSE,&commands);
    if (SUCCEEDED(hr)) m->immediate->ExecuteCommandList(commands.Get(),TRUE);
    return hr;
}

extern "C" __declspec(dllexport) void vmaf_tonemap_destroy(void* opaque) { delete static_cast<Mapper*>(opaque); }

// ---------------------------------------------------------------- presenter
// Native playback's own presentation: a waitable flip-model swapchain on the
// decoders' D3D11 device, in a child window of the view, two frames deep. Its
// caller waits for the swapchain (vmaf_present_wait) holding nothing, then
// draws and presents with the device lock held (vmaf_present_frame): Present
// then has no frame before it to wait for. GStreamer's d3d11videosink presented
// at the display's refresh waiting for it with the device lock held, and the
// decoders, which need that lock for each frame, stalled up to 40 ms. One frame
// deep, the swapchain lost a refresh to each repaint of a window on the desktop
// (the app's own position display among them, 21 a second): 120 fps video
// showed 98 frames a second.

static const char present_shader[] = R"(
Texture2D<float4> rgbFrame : register(t0);
Texture2D<float> lumaPlane : register(t1);
Texture2D<float2> chromaPlane : register(t2);
SamplerState linearClamp : register(s0);
cbuffer Draw : register(b0) {
    float4 source;    // the part of the frame drawn: left, top, width, height in texture coordinates
    float4 toRgb[3];  // YUV frames: rows of (Y, U, V) coefficients, the offset in w
    float yuv; float3 unused;
};
struct Vertex { float4 position : SV_Position; float2 uv : TEXCOORD0; };
Vertex vs(uint id : SV_VertexID) {
    float2 corner = float2(id & 1, id >> 1);
    Vertex v;
    v.position = float4(corner.x * 2 - 1, 1 - corner.y * 2, 0, 1);
    v.uv = source.xy + corner * source.zw;
    return v;
}
float4 ps(Vertex v) : SV_Target {
    if (yuv < 0.5) return float4(rgbFrame.Sample(linearClamp, v.uv).rgb, 1);
    float3 c = float3(lumaPlane.Sample(linearClamp, v.uv), chromaPlane.Sample(linearClamp, v.uv));
    float3 rgb = float3(dot(c, toRgb[0].xyz), dot(c, toRgb[1].xyz), dot(c, toRgb[2].xyz))
               + float3(toRgb[0].w, toRgb[1].w, toRgb[2].w);
    return float4(saturate(rgb), 1);
}
)";

struct PresenterViews {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> rgb, luma, chroma;
};

struct Presenter {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> immediate, deferred;
    HWND window=nullptr;
    ComPtr<IDXGISwapChain3> swapchain;
    HANDLE waitable=nullptr;
    ComPtr<ID3D11RenderTargetView> target;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11Buffer> constants;
    UINT width=0, height=0;
    int asked=DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;  // the colour space last asked for, given or not
    // The frames' views, kept: the decoders' pools hand the same textures round.
    PresenterViews views[16];
    int next=0;
    // Frames a shader cannot read, copied (present_source).
    ComPtr<ID3D11Texture2D> copy;
};

static LRESULT CALLBACK present_window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    if (message==WM_ERASEBKGND) return 1;
    if (message==WM_PAINT) { ValidateRect(window,nullptr); return 0; }
    return DefWindowProcW(window,message,w,l);
}

static HRESULT present_target(Presenter* p) {
    ComPtr<ID3D11Texture2D> buffer;
    HRESULT hr=p->swapchain->GetBuffer(0,IID_PPV_ARGS(&buffer));
    if (SUCCEEDED(hr)) hr=p->device->CreateRenderTargetView(buffer.Get(),nullptr,&p->target);
    return hr;
}

// On the thread the window `parent` belongs to, with the device lock held.
// `hdr_display`: 10-bit buffers, for PQ BT.2020 frames as well as SDR ones.
extern "C" __declspec(dllexport) void* vmaf_present_create(ID3D11Device* device, HWND parent, int hdr_display) {
    static ATOM window_class=0;
    HINSTANCE module=GetModuleHandleW(nullptr);
    if (!window_class) {
        WNDCLASSEXW wc={}; wc.cbSize=sizeof(wc); wc.lpfnWndProc=present_window_proc; wc.hInstance=module;
        wc.lpszClassName=L"VideoMetricsLabNativeVideo";
        window_class=RegisterClassExW(&wc);
        if (!window_class) return nullptr;
    }
    auto p=new(std::nothrow) Presenter;
    if (!p || !device || !parent) { delete p; return nullptr; }
    p->device=device;
    device->GetImmediateContext(&p->immediate);
    RECT client={}; GetClientRect(parent,&client);
    p->width=client.right>1?client.right:1; p->height=client.bottom>1?client.bottom:1;
    // Disabled: the mouse is the view's (Windows' hit testing skips it).
    p->window=CreateWindowExW(0,L"VideoMetricsLabNativeVideo",L"",WS_CHILD|WS_VISIBLE|WS_DISABLED|WS_CLIPSIBLINGS,
                              0,0,p->width,p->height,parent,nullptr,module,nullptr);
    ComPtr<IDXGIDevice> dxgi; ComPtr<IDXGIAdapter> adapter; ComPtr<IDXGIFactory2> factory;
    HRESULT hr=p->window?device->QueryInterface(IID_PPV_ARGS(&dxgi)):E_FAIL;
    if (SUCCEEDED(hr)) hr=dxgi->GetAdapter(&adapter);
    if (SUCCEEDED(hr)) hr=adapter->GetParent(IID_PPV_ARGS(&factory));
    DXGI_SWAP_CHAIN_DESC1 desc={};
    desc.Width=p->width; desc.Height=p->height;
    desc.Format=hdr_display?DXGI_FORMAT_R10G10B10A2_UNORM:DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count=1; desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT; desc.BufferCount=3;
    desc.Scaling=DXGI_SCALING_STRETCH; desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.AlphaMode=DXGI_ALPHA_MODE_IGNORE; desc.Flags=DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    ComPtr<IDXGISwapChain1> swapchain;
    if (SUCCEEDED(hr)) hr=factory->CreateSwapChainForHwnd(device,p->window,&desc,nullptr,nullptr,&swapchain);
    if (SUCCEEDED(hr)) factory->MakeWindowAssociation(p->window,DXGI_MWA_NO_ALT_ENTER|DXGI_MWA_NO_WINDOW_CHANGES);
    if (SUCCEEDED(hr)) hr=swapchain.As(&p->swapchain);
    if (SUCCEEDED(hr)) hr=p->swapchain->SetMaximumFrameLatency(2);
    if (SUCCEEDED(hr)) { p->waitable=p->swapchain->GetFrameLatencyWaitableObject(); hr=present_target(p); }
    ComPtr<ID3DBlob> vs, ps, errors;
    if (SUCCEEDED(hr)) hr=device->CreateDeferredContext(0,&p->deferred);
    if (SUCCEEDED(hr)) hr=D3DCompile(present_shader,strlen(present_shader),nullptr,nullptr,nullptr,"vs","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vs,&errors);
    if (SUCCEEDED(hr)) hr=D3DCompile(present_shader,strlen(present_shader),nullptr,nullptr,nullptr,"ps","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&ps,&errors);
    if (SUCCEEDED(hr)) hr=device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&p->vs);
    if (SUCCEEDED(hr)) hr=device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&p->ps);
    D3D11_SAMPLER_DESC sd={}; sd.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU=sd.AddressV=sd.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP; sd.MaxLOD=D3D11_FLOAT32_MAX;
    if (SUCCEEDED(hr)) hr=device->CreateSamplerState(&sd,&p->sampler);
    D3D11_BUFFER_DESC bd={}; bd.ByteWidth=80; bd.Usage=D3D11_USAGE_DYNAMIC;
    bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER; bd.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    if (SUCCEEDED(hr)) hr=device->CreateBuffer(&bd,nullptr,&p->constants);
    if (FAILED(hr)) {
        if (p->waitable) CloseHandle(p->waitable);
        p->swapchain.Reset();
        if (p->window) DestroyWindow(p->window);
        delete p;
        return nullptr;
    }
    return p;
}

// The window follows the view's size (on its thread); the buffers follow the
// window at the next frame drawn.
extern "C" __declspec(dllexport) void vmaf_present_follow(void* opaque, HWND parent) {
    auto p=static_cast<Presenter*>(opaque);
    if (!p || !p->window) return;
    RECT client={}, own={};
    GetClientRect(parent,&client); GetClientRect(p->window,&own);
    if (client.right!=own.right || client.bottom!=own.bottom)
        SetWindowPos(p->window,nullptr,0,0,client.right,client.bottom,SWP_NOZORDER|SWP_NOACTIVATE|SWP_NOMOVE);
}

// Until the swapchain takes a frame without waiting: 0, or 1 at `timeout_ms`.
// Holding no lock.
extern "C" __declspec(dllexport) int vmaf_present_wait(void* opaque, unsigned timeout_ms) {
    auto p=static_cast<Presenter*>(opaque);
    if (!p || !p->waitable) return -1;
    return WaitForSingleObjectEx(p->waitable,timeout_ms,TRUE)==WAIT_OBJECT_0?0:1;
}

static PresenterViews* present_views(Presenter* p, ID3D11Texture2D* texture, const D3D11_TEXTURE2D_DESC& desc) {
    for (auto& v : p->views) if (v.texture.Get()==texture) return &v;
    PresenterViews& v=p->views[p->next]; p->next=(p->next+1)%16;
    v=PresenterViews{}; v.texture=texture;
    D3D11_SHADER_RESOURCE_VIEW_DESC d={}; d.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D; d.Texture2D.MipLevels=1;
    HRESULT hr=S_OK;
    if (desc.Format==DXGI_FORMAT_NV12 || desc.Format==DXGI_FORMAT_P010) {
        bool deep=desc.Format==DXGI_FORMAT_P010;
        d.Format=deep?DXGI_FORMAT_R16_UNORM:DXGI_FORMAT_R8_UNORM;
        hr=p->device->CreateShaderResourceView(texture,&d,&v.luma);
        d.Format=deep?DXGI_FORMAT_R16G16_UNORM:DXGI_FORMAT_R8G8_UNORM;
        if (SUCCEEDED(hr)) hr=p->device->CreateShaderResourceView(texture,&d,&v.chroma);
    } else {
        d.Format=desc.Format;
        hr=p->device->CreateShaderResourceView(texture,&d,&v.rgb);
    }
    if (FAILED(hr)) { v=PresenterViews{}; return nullptr; }
    return &v;
}

// The texture a shader reads `frame`'s `subresource` from, and its
// description (`desc`): the frame's own, or -- a slice of a decoder's texture
// array, or a texture bound for decoding only, as d3d11h264dec's are -- a
// copy in a texture of the presenter's. Drawn from the decoder's own, such
// frames failed (E_FAIL) and playback fell back to FFmpeg. With the device
// lock held; null if the copy could not be made.
static ID3D11Texture2D* present_source(Presenter* p, ID3D11Texture2D* frame, UINT subresource,
                                       D3D11_TEXTURE2D_DESC& desc) {
    frame->GetDesc(&desc);
    if (desc.ArraySize==1 && subresource==0 && (desc.BindFlags & D3D11_BIND_SHADER_RESOURCE)) return frame;
    D3D11_TEXTURE2D_DESC own=desc;
    own.MipLevels=1; own.ArraySize=1; own.SampleDesc.Count=1; own.SampleDesc.Quality=0;
    own.Usage=D3D11_USAGE_DEFAULT; own.BindFlags=D3D11_BIND_SHADER_RESOURCE; own.CPUAccessFlags=0; own.MiscFlags=0;
    D3D11_TEXTURE2D_DESC have={};
    if (p->copy) p->copy->GetDesc(&have);
    if (!p->copy || have.Width!=own.Width || have.Height!=own.Height || have.Format!=own.Format) {
        p->copy.Reset();
        if (FAILED(p->device->CreateTexture2D(&own,nullptr,&p->copy))) return nullptr;
    }
    p->immediate->CopySubresourceRegion(p->copy.Get(),0,0,0,0,frame,subresource,nullptr);
    desc=own;
    return p->copy.Get();
}

// With the device lock held. `frame`: the texture, `subresource` the frame's
// in it. `source`: the frame's part shown (x, y, width, height in its
// pixels), or null for all of it; `target`: where in the window (device
// pixels), or null to fit it, letterboxed. `to_rgb`: 12 floats for a YUV
// frame (rows of Y, U, V coefficients and an offset), null for RGB.
// `space`: the frames' DXGI colour space; `clear`: the colour beside them.
extern "C" __declspec(dllexport) int vmaf_present_frame(void* opaque, ID3D11Resource* frame, unsigned subresource,
                                                        const int* source, const int* target, const float* to_rgb,
                                                        int space, const float* clear) {
    auto p=static_cast<Presenter*>(opaque);
    if (!p || !frame || !clear) return E_INVALIDARG;
    RECT own={}; GetClientRect(p->window,&own);
    UINT width=own.right>1?own.right:1, height=own.bottom>1?own.bottom:1;
    HRESULT hr=S_OK;
    if (width!=p->width || height!=p->height) {
        p->target.Reset();
        p->immediate->OMSetRenderTargets(0,nullptr,nullptr);
        p->immediate->Flush();
        hr=p->swapchain->ResizeBuffers(0,width,height,DXGI_FORMAT_UNKNOWN,DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
        if (SUCCEEDED(hr)) hr=present_target(p);
        if (FAILED(hr)) return hr;
        p->width=width; p->height=height;
    }
    if (space!=p->asked) {  // asked once: a display without HDR refuses it at every frame
        p->asked=space;
        UINT support=0;
        if (SUCCEEDED(p->swapchain->CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE(space),&support))
            && (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT))
            p->swapchain->SetColorSpace1(DXGI_COLOR_SPACE_TYPE(space));
    }
    ComPtr<ID3D11Texture2D> texture;
    hr=frame->QueryInterface(IID_PPV_ARGS(&texture));
    if (FAILED(hr)) return hr;
    D3D11_TEXTURE2D_DESC desc;
    ID3D11Texture2D* drawn=present_source(p,texture.Get(),subresource,desc);
    if (!drawn) return E_OUTOFMEMORY;
    PresenterViews* views=present_views(p,drawn,desc);
    if (!views) return E_FAIL;
    float s[4]={0,0,float(desc.Width),float(desc.Height)};
    if (source) for (int i=0;i<4;i++) s[i]=float(source[i]);
    float t[4];
    if (target) { for (int i=0;i<4;i++) t[i]=float(target[i]); }
    else {  // fitted, on whole pixels
        float scale=std::min(width/s[2],height/s[3]);
        t[2]=std::round(s[2]*scale); t[3]=std::round(s[3]*scale);
        t[0]=std::floor((width-t[2])/2); t[1]=std::floor((height-t[3])/2);
    }
    auto c=p->deferred.Get();
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr=c->Map(p->constants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped);
    if (FAILED(hr)) return hr;
    float* k=static_cast<float*>(mapped.pData);
    k[0]=s[0]/desc.Width; k[1]=s[1]/desc.Height; k[2]=s[2]/desc.Width; k[3]=s[3]/desc.Height;
    for (int i=0;i<12;i++) k[4+i]=to_rgb?to_rgb[i]:0;
    k[16]=to_rgb?1.f:0.f; k[17]=k[18]=k[19]=0;
    c->Unmap(p->constants.Get(),0);
    c->ClearRenderTargetView(p->target.Get(),clear);
    c->OMSetRenderTargets(1,p->target.GetAddressOf(),nullptr);
    D3D11_VIEWPORT viewport={t[0],t[1],t[2],t[3],0,1}; c->RSSetViewports(1,&viewport);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    c->IASetInputLayout(nullptr);
    c->VSSetShader(p->vs.Get(),nullptr,0); c->PSSetShader(p->ps.Get(),nullptr,0);
    c->VSSetConstantBuffers(0,1,p->constants.GetAddressOf()); c->PSSetConstantBuffers(0,1,p->constants.GetAddressOf());
    ID3D11ShaderResourceView* srvs[3]={views->rgb.Get(),views->luma.Get(),views->chroma.Get()};
    c->PSSetShaderResources(0,3,srvs);
    c->PSSetSamplers(0,1,p->sampler.GetAddressOf());
    c->Draw(4,0);
    ComPtr<ID3D11CommandList> commands;
    hr=c->FinishCommandList(FALSE,&commands);
    if (FAILED(hr)) return hr;
    p->immediate->ExecuteCommandList(commands.Get(),TRUE);
    return p->swapchain->Present(1,0);
}

// With the device lock held, on any thread: the window goes with the view.
extern "C" __declspec(dllexport) void vmaf_present_destroy(void* opaque) {
    auto p=static_cast<Presenter*>(opaque);
    if (!p) return;
    if (p->waitable) CloseHandle(p->waitable);
    for (auto& v : p->views) v=PresenterViews{};
    p->copy.Reset(); p->target.Reset(); p->swapchain.Reset();
    p->immediate->Flush();
    delete p;
}
