// Private RGBA16 GPU surface processing for the GStreamer playback branch.
// No GStreamer ABI dependencies, pixel readback, or CPU frame allocation.
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>
#include <new>
using Microsoft::WRL::ComPtr;

static const char shader[] = R"(
Texture2D<float4> inputFrame : register(t0);
cbuffer Parameters : register(b0) { float kind; float peak; float white; float unused; };
float4 vs(uint id : SV_VertexID) : SV_Position {
    return float4(id == 2 ? 3 : -1, id == 1 ? 3 : -1, 0, 1);
}
float3 pq(float3 x) {
    float3 p = pow(max(x, 0), 1.0 / 78.84375);
    return 10000 * pow(max(p - 0.8359375, 0) / max(18.8515625 - 18.6875*p, 1e-6), 1.0 / 0.1593017578125);
}
float3 hlg(float3 x) {
    float3 lo = x*x/3;
    float3 hi = (exp((x-0.55991073)/0.17883277)+0.28466892)/12;
    float3 scene = float3(x.r<=0.5?lo.r:hi.r, x.g<=0.5?lo.g:hi.g, x.b<=0.5?lo.b:hi.b);
    return scene * pow(max(dot(scene, float3(.2627,.678,.0593)), 1e-8), .2) * peak;
}
float3 srgb(float3 x) {
    float3 lo=12.92*x, hi=1.055*pow(max(x,0),1.0/2.4)-.055;
    return float3(x.r<=.0031308?lo.r:hi.r, x.g<=.0031308?lo.g:hi.g, x.b<=.0031308?lo.b:hi.b);
}
float4 ps(float4 pos : SV_Position) : SV_Target {
    float3 encoded = inputFrame.Load(int3(pos.xy,0)).rgb;
    float3 light = kind < 1.5 ? pq(encoded) : hlg(encoded);
    // Fixed extended-Reinhard luminance curve, shared by all comparison sides.
    float y = max(dot(light,float3(.2627,.678,.0593))/white,0);
    float w = peak/white;
    float mapped = y*(1+y/(w*w))/(1+y);
    light = light/white * (y > 1e-8 ? mapped/y : 0);
    // Linear BT.2020 -> BT.709, then gamut clipping and sRGB encoding.
    float3 rgb = float3(dot(light,float3(1.660491,-.587641,-.072850)),
                        dot(light,float3(-.124550,1.132900,-.008349)),
                        dot(light,float3(-.018151,-.100579,1.118730)));
    return float4(srgb(saturate(rgb)),1);
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

extern "C" __declspec(dllexport) void* vmaf_tonemap_create(ID3D11Resource* resource, int kind) {
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
    float params[4]={float(kind),1000,100,0};
    D3D11_BUFFER_DESC bd={}; bd.ByteWidth=sizeof(params); bd.Usage=D3D11_USAGE_IMMUTABLE; bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial={}; initial.pSysMem=params;
    if (SUCCEEDED(hr)) hr=m->device->CreateBuffer(&bd,&initial,&m->parameters);
    if (FAILED(hr)) { delete m; return nullptr; }
    return m;
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
