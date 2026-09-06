// Standalone GPU numerical check. Readback is TEST ONLY, never playback.
#include "d3d11_tonemap.cpp"
#include <cmath>
#include <iostream>
#include <vector>
int main() {
    ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context;
    if (FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context))) return 2;
    constexpr int width=256;
    for (int kind=1;kind<=2;++kind) {
        std::vector<unsigned short> pixels(width*4);
        for (int i=0;i<width;i++) {
            for (int c=0;c<3;c++) pixels[4*i+c]=i*257;
            pixels[4*i+3]=65535;
        }
        D3D11_TEXTURE2D_DESC desc={}; desc.Width=width; desc.Height=1; desc.MipLevels=1; desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_R16G16B16A16_UNORM; desc.SampleDesc.Count=1; desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={}; initial.pSysMem=pixels.data(); initial.SysMemPitch=width*8;
        ComPtr<ID3D11Texture2D> texture;
        if(FAILED(device->CreateTexture2D(&desc,&initial,&texture)))return 3;
        void* mapper=vmaf_tonemap_create(texture.Get(),kind);
        if(!mapper)return 4;
        int hr=vmaf_tonemap_render(mapper,texture.Get());
        vmaf_tonemap_destroy(mapper);
        if(hr<0)return 5;
        desc.Usage=D3D11_USAGE_STAGING; desc.BindFlags=0; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> readback;
        if(FAILED(device->CreateTexture2D(&desc,nullptr,&readback)))return 6;
        context->CopyResource(readback.Get(),texture.Get()); D3D11_MAPPED_SUBRESOURCE mapped;
        if(FAILED(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped)))return 7;
        auto values=static_cast<unsigned short*>(mapped.pData);
        double largest=0;
        for(int i=0;i<width;i++) {
            double x=double(i)/255, light;
            if(kind==1) { double p=std::pow(x,1/78.84375); light=10000*std::pow(std::max(p-.8359375,0.0)/(18.8515625-18.6875*p),1/.1593017578125); }
            else { double scene=x<=.5?x*x/3:(std::exp((x-.55991073)/.17883277)+.28466892)/12; light=1000*std::pow(scene,1.2); }
            double y=light/100, linear=std::min(y*(1+y/100)/(1+y),1.0);
            double expected=linear<=.0031308?12.92*linear:1.055*std::pow(linear,1/2.4)-.055;
            for(int c=0;c<3;c++) largest=std::max(largest,std::abs(double(values[i*4+c])/65535-expected));
            if(values[i*4+3]!=65535)return 8;
        }
        context->Unmap(readback.Get(),0);
        std::cout<<(kind==1?"PQ":"HLG")<<" max normalized error: "<<largest<<std::endl;
        if(largest>.0002)return 9;
    }
    return 0;
}
