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
    // Saturated PQ colours through each primaries' weights and matrix
    // (vmaf_tonemap_create_primaries): BT.2020's, as the mapper always
    // assumed, and Display P3's (SMPTE EG 432-1, D65).
    struct Primaries { const char* name; float luma[3]; float to_bt709[9]; };
    const Primaries primaries[]={
        {"BT.2020",{.2627f,.677998f,.059302f},{1.660491f,-.587641f,-.072850f,-.124550f,1.132900f,-.008349f,-.018151f,-.100579f,1.118730f}},
        {"Display P3",{.228975f,.691739f,.079287f},{1.224940f,-.224940f,0.f,-.042057f,1.042057f,0.f,-.019638f,-.078636f,1.098274f}},
    };
    for (const auto& prim : primaries) {
        std::vector<unsigned short> pixels(width*4);
        for (int i=0;i<width;i++) {
            pixels[4*i]=i*257; pixels[4*i+1]=(255-i)*128; pixels[4*i+2]=(i*7%256)*257; pixels[4*i+3]=65535;
        }
        D3D11_TEXTURE2D_DESC desc={}; desc.Width=width; desc.Height=1; desc.MipLevels=1; desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_R16G16B16A16_UNORM; desc.SampleDesc.Count=1; desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={}; initial.pSysMem=pixels.data(); initial.SysMemPitch=width*8;
        ComPtr<ID3D11Texture2D> texture;
        if(FAILED(device->CreateTexture2D(&desc,&initial,&texture)))return 10;
        void* mapper=vmaf_tonemap_create_primaries(texture.Get(),1,prim.luma,prim.to_bt709);
        if(!mapper)return 11;
        int hr=vmaf_tonemap_render(mapper,texture.Get());
        vmaf_tonemap_destroy(mapper);
        if(hr<0)return 12;
        desc.Usage=D3D11_USAGE_STAGING; desc.BindFlags=0; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> readback;
        if(FAILED(device->CreateTexture2D(&desc,nullptr,&readback)))return 13;
        context->CopyResource(readback.Get(),texture.Get()); D3D11_MAPPED_SUBRESOURCE mapped;
        if(FAILED(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped)))return 14;
        auto values=static_cast<unsigned short*>(mapped.pData);
        double largest=0;
        for(int i=0;i<width;i++) {
            double light[3];
            for(int c=0;c<3;c++) {
                double p=std::pow(pixels[4*i+c]/65535.0,1/78.84375);
                light[c]=10000*std::pow(std::max(p-.8359375,0.0)/(18.8515625-18.6875*p),1/.1593017578125);
            }
            double y=(prim.luma[0]*light[0]+prim.luma[1]*light[1]+prim.luma[2]*light[2])/100;
            double mapped_y=y*(1+y/100)/(1+y), gain=y>1e-8?mapped_y/y:0;
            for(int c=0;c<3;c++) {
                double linear=0;
                for(int k=0;k<3;k++) linear+=prim.to_bt709[3*c+k]*light[k]/100*gain;
                linear=std::min(std::max(linear,0.0),1.0);
                double expected=linear<=.0031308?12.92*linear:1.055*std::pow(linear,1/2.4)-.055;
                largest=std::max(largest,std::abs(double(values[i*4+c])/65535-expected));
            }
        }
        context->Unmap(readback.Get(),0);
        std::cout<<prim.name<<" colours max normalized error: "<<largest<<std::endl;
        if(largest>.0005)return 15;
    }
    // HDR kept (vmaf_hdr_convert_create): Display P3 colours converted to
    // BT.2020 and coded again, PQ as absolute light, HLG as scene light.
    const float p3_to_bt2020[9]={.753833f,.198597f,.047570f, .045744f,.941777f,.012479f, -.001210f,.017602f,.983609f};
    for (int kind=1;kind<=2;++kind) {
        std::vector<unsigned short> pixels(width*4);
        for (int i=0;i<width;i++) {
            pixels[4*i]=i*257; pixels[4*i+1]=(255-i)*128; pixels[4*i+2]=(i*7%256)*257; pixels[4*i+3]=65535;
        }
        D3D11_TEXTURE2D_DESC desc={}; desc.Width=width; desc.Height=1; desc.MipLevels=1; desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_R16G16B16A16_UNORM; desc.SampleDesc.Count=1; desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={}; initial.pSysMem=pixels.data(); initial.SysMemPitch=width*8;
        ComPtr<ID3D11Texture2D> texture;
        if(FAILED(device->CreateTexture2D(&desc,&initial,&texture)))return 16;
        void* mapper=vmaf_hdr_convert_create(texture.Get(),kind,p3_to_bt2020);
        if(!mapper)return 17;
        int hr=vmaf_tonemap_render(mapper,texture.Get());
        vmaf_tonemap_destroy(mapper);
        if(hr<0)return 18;
        desc.Usage=D3D11_USAGE_STAGING; desc.BindFlags=0; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> readback;
        if(FAILED(device->CreateTexture2D(&desc,nullptr,&readback)))return 19;
        context->CopyResource(readback.Get(),texture.Get()); D3D11_MAPPED_SUBRESOURCE mapped;
        if(FAILED(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped)))return 20;
        auto values=static_cast<unsigned short*>(mapped.pData);
        double largest=0;
        for(int i=0;i<width;i++) {
            double light[3];
            for(int c=0;c<3;c++) {
                double x=pixels[4*i+c]/65535.0;
                if(kind==1) { double p=std::pow(x,1/78.84375); light[c]=10000*std::pow(std::max(p-.8359375,0.0)/(18.8515625-18.6875*p),1/.1593017578125); }
                else light[c]=x<=.5?x*x/3:(std::exp((x-.55991073)/.17883277)+.28466892)/12;
            }
            for(int c=0;c<3;c++) {
                double out=0;
                for(int k=0;k<3;k++) out+=p3_to_bt2020[3*c+k]*light[k];
                double expected;
                if(kind==1) { double y=std::pow(std::min(std::max(out/10000,0.0),1.0),.1593017578125);
                              expected=std::pow((.8359375+18.8515625*y)/(1+18.6875*y),78.84375); }
                else { double e=std::min(std::max(out,0.0),1.0);
                       expected=e<=1.0/12?std::sqrt(3*e):.17883277*std::log(std::max(12*e-.28466892,1e-6))+.55991073; }
                largest=std::max(largest,std::abs(double(values[i*4+c])/65535-expected));
            }
        }
        context->Unmap(readback.Get(),0);
        std::cout<<(kind==1?"PQ":"HLG")<<" Display P3 to BT.2020, HDR kept: max normalized error: "<<largest<<std::endl;
        if(largest>.0005)return 21;
    }
    return 0;
}
