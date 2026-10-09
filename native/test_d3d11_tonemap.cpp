// Standalone GPU numerical check of the presenter's shader: frames drawn 1:1
// into a float texture by its own draw code, and read back. Readback is TEST
// ONLY, never playback.
#include "d3d11_tonemap.cpp"
#include <cmath>
#include <iostream>
#include <vector>

static ComPtr<ID3D11Device> device;
static ComPtr<ID3D11DeviceContext> context;
static Presenter presenter;

// `frame`'s `subresource` (`width` x `height`, its sampled rows: the first)
// drawn by the presenter's shader with `to_rgb` and `shading`; RGBA floats
// read back.
static bool draw(ID3D11Texture2D* frame, int width, int height, const float* to_rgb, const float* shading,
                 std::vector<float>& out, UINT subresource=0) {
    D3D11_TEXTURE2D_DESC desc;
    ID3D11Texture2D* drawn=present_source(&presenter,frame,subresource,desc);
    if (!drawn) return false;
    PresenterViews* views=present_views(&presenter,drawn,desc);
    if (!views) return false;
    D3D11_TEXTURE2D_DESC td={}; td.Width=width; td.Height=height; td.MipLevels=1; td.ArraySize=1;
    td.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; td.SampleDesc.Count=1; td.Usage=D3D11_USAGE_DEFAULT;
    td.BindFlags=D3D11_BIND_RENDER_TARGET;
    ComPtr<ID3D11Texture2D> target; ComPtr<ID3D11RenderTargetView> rtv;
    if (FAILED(device->CreateTexture2D(&td,nullptr,&target))) return false;
    if (FAILED(device->CreateRenderTargetView(target.Get(),nullptr,&rtv))) return false;
    float source[4]={0,0,float(width),float(height)};
    float constants[kConstants]; present_constants(constants,desc,source,to_rgb,shading);
    D3D11_VIEWPORT viewport={0,0,float(width),float(height),0,1};
    const float clear[4]={0,0,0,1};
    if (FAILED(present_draw(&presenter,rtv.Get(),viewport,views,constants,clear))) return false;
    td.Usage=D3D11_USAGE_STAGING; td.BindFlags=0; td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> readback;
    if (FAILED(device->CreateTexture2D(&td,nullptr,&readback))) return false;
    context->CopyResource(readback.Get(),target.Get());
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped))) return false;
    out.assign(static_cast<float*>(mapped.pData),static_cast<float*>(mapped.pData)+width*4);
    context->Unmap(readback.Get(),0);
    return true;
}

static ComPtr<ID3D11Texture2D> rgba16(const std::vector<unsigned short>& pixels, int width) {
    D3D11_TEXTURE2D_DESC desc={}; desc.Width=width; desc.Height=1; desc.MipLevels=1; desc.ArraySize=1;
    desc.Format=DXGI_FORMAT_R16G16B16A16_UNORM; desc.SampleDesc.Count=1; desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial={}; initial.pSysMem=pixels.data(); initial.SysMemPitch=width*8;
    ComPtr<ID3D11Texture2D> texture;
    device->CreateTexture2D(&desc,&initial,&texture);
    return texture;
}

static double pq_light(double x) {
    double p=std::pow(x,1/78.84375);
    return 10000*std::pow(std::max(p-.8359375,0.0)/(18.8515625-18.6875*p),1/.1593017578125);
}
static double hlg_scene(double x) { return x<=.5?x*x/3:(std::exp((x-.55991073)/.17883277)+.28466892)/12; }
static double srgb(double linear) { return linear<=.0031308?12.92*linear:1.055*std::pow(linear,1/2.4)-.055; }
static double pq_code(double nits) {
    double y=std::pow(std::min(std::max(nits/10000,0.0),1.0),.1593017578125);
    return std::pow((.8359375+18.8515625*y)/(1+18.6875*y),78.84375);
}

// The shading as present_constants takes it.
static std::vector<float> shading(int mode, int kind, const float* luma, const float* matrix) {
    std::vector<float> s={float(mode),float(kind),luma[0],luma[1],luma[2]};
    s.insert(s.end(),matrix,matrix+9);
    s.push_back(1000); s.push_back(100);
    return s;
}

int main() {
    if (FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context))) return 2;
    presenter.device=device; presenter.immediate=context;
    if (FAILED(present_resources(&presenter))) return 3;
    constexpr int width=256;
    const float bt2020_luma[3]={.2627f,.678f,.0593f};
    const float bt2020_to_bt709[9]={1.660491f,-.587641f,-.072850f, -.124550f,1.132900f,-.008349f, -.018151f,-.100579f,1.118730f};
    std::vector<float> out;
    // Grey ramps, PQ and HLG, BT.2020, mapped to SDR.
    for (int kind=1;kind<=2;++kind) {
        std::vector<unsigned short> pixels(width*4);
        for (int i=0;i<width;i++) { for (int c=0;c<3;c++) pixels[4*i+c]=i*257; pixels[4*i+3]=65535; }
        auto frame=rgba16(pixels,width);
        auto s=shading(1,kind,bt2020_luma,bt2020_to_bt709);
        if (!frame || !draw(frame.Get(),width,1,nullptr,s.data(),out)) return 4;
        double largest=0;
        for (int i=0;i<width;i++) {
            double x=double(i)/255;
            double light=kind==1?pq_light(x):1000*std::pow(hlg_scene(x),1.2);
            double y=light/100, linear=std::min(y*(1+y/100)/(1+y),1.0);
            for (int c=0;c<3;c++) largest=std::max(largest,std::abs(out[i*4+c]-srgb(linear)));
            if (out[i*4+3]!=1.f) return 5;
        }
        std::cout<<(kind==1?"PQ":"HLG")<<" to SDR, max error: "<<largest<<std::endl;
        if (largest>.0002) return 6;
    }
    // Saturated PQ colours through each primaries' weights and matrix:
    // BT.2020's and Display P3's (SMPTE EG 432-1, D65).
    struct Primaries { const char* name; float luma[3]; float to_bt709[9]; };
    const Primaries primaries[]={
        {"BT.2020",{.2627f,.677998f,.059302f},{1.660491f,-.587641f,-.072850f,-.124550f,1.132900f,-.008349f,-.018151f,-.100579f,1.118730f}},
        {"Display P3",{.228975f,.691739f,.079287f},{1.224940f,-.224940f,0.f,-.042057f,1.042057f,0.f,-.019638f,-.078636f,1.098274f}},
    };
    std::vector<unsigned short> colours(width*4);
    for (int i=0;i<width;i++) {
        colours[4*i]=i*257; colours[4*i+1]=(255-i)*128; colours[4*i+2]=(i*7%256)*257; colours[4*i+3]=65535;
    }
    auto coloured=rgba16(colours,width);
    if (!coloured) return 7;
    for (const auto& prim : primaries) {
        auto s=shading(1,1,prim.luma,prim.to_bt709);
        if (!draw(coloured.Get(),width,1,nullptr,s.data(),out)) return 8;
        double largest=0;
        for (int i=0;i<width;i++) {
            double light[3];
            for (int c=0;c<3;c++) light[c]=pq_light(colours[4*i+c]/65535.0);
            double y=(prim.luma[0]*light[0]+prim.luma[1]*light[1]+prim.luma[2]*light[2])/100;
            double gain=y>1e-8?(y*(1+y/100)/(1+y))/y:0;
            for (int c=0;c<3;c++) {
                double linear=0;
                for (int k=0;k<3;k++) linear+=prim.to_bt709[3*c+k]*light[k]/100*gain;
                largest=std::max(largest,std::abs(out[i*4+c]-srgb(std::min(std::max(linear,0.0),1.0))));
            }
        }
        std::cout<<prim.name<<" colours to SDR, max error: "<<largest<<std::endl;
        if (largest>.0005) return 9;
    }
    // HDR kept: Display P3 colours converted to BT.2020 and coded as PQ --
    // HLG's display light at its 1000-nit peak.
    const float p3_to_bt2020[9]={.753833f,.198597f,.047570f, .045744f,.941777f,.012479f, -.001210f,.017602f,.983609f};
    const float p3_luma[3]={.228975f,.691739f,.079287f};
    for (int kind=1;kind<=2;++kind) {
        auto s=shading(2,kind,p3_luma,p3_to_bt2020);
        if (!draw(coloured.Get(),width,1,nullptr,s.data(),out)) return 10;
        double largest=0;
        for (int i=0;i<width;i++) {
            double light[3];
            for (int c=0;c<3;c++) {
                double x=colours[4*i+c]/65535.0;
                light[c]=kind==1?pq_light(x):hlg_scene(x);
            }
            if (kind==2) {  // HLG's display light: the system gamma (1.2) at a 1000-nit peak
                double y=std::max(p3_luma[0]*light[0]+p3_luma[1]*light[1]+p3_luma[2]*light[2],1e-8);
                for (int c=0;c<3;c++) light[c]*=std::pow(y,.2)*1000;
            }
            for (int c=0;c<3;c++) {
                double nits=0;
                for (int k=0;k<3;k++) nits+=p3_to_bt2020[3*c+k]*light[k];
                largest=std::max(largest,std::abs(out[i*4+c]-pq_code(nits)));
            }
        }
        std::cout<<(kind==1?"PQ":"HLG")<<" Display P3 to BT.2020 PQ, HDR kept, max error: "<<largest<<std::endl;
        if (largest>.0005) return 11;
    }
    // A P010 frame, as the decoders give them: grey levels in BT.2020's
    // limited range, through the YUV rows, then mapped to SDR as PQ.
    {
        constexpr int w=16, h=2;
        std::vector<unsigned short> planes(w*h + w*h/2);
        const int levels[w]={64,100,200,300,400,450,500,550,600,650,700,750,800,850,900,940};
        for (int y=0;y<h;y++) for (int x=0;x<w;x++) planes[y*w+x]=levels[x]<<6;
        for (int i=0;i<w*h/2;i++) planes[w*h+i]=512<<6;  // U and V: neutral
        D3D11_TEXTURE2D_DESC desc={}; desc.Width=w; desc.Height=h; desc.MipLevels=1; desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_P010; desc.SampleDesc.Count=1; desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={}; initial.pSysMem=planes.data(); initial.SysMemPitch=w*2;
        ComPtr<ID3D11Texture2D> frame;
        if (FAILED(device->CreateTexture2D(&desc,&initial,&frame))) return 12;
        // The rows locked_presentation.yuv_to_rgb gives BT.2020, limited, 10 bits.
        const double kr=.2627, kb=.0593, kg=1-kr-kb, code=65535.0/64;
        const double ay=code/876, by=-64.0/876, ac=code/896, bc=-512.0/896;
        const double gu=-2*kb*(1-kb)/kg, gv=-2*kr*(1-kr)/kg;
        const float rows[12]={float(ay),0,float(2*(1-kr)*ac),float(by+2*(1-kr)*bc),
                              float(ay),float(gu*ac),float(gv*ac),float(by+(gu+gv)*bc),
                              float(ay),float(2*(1-kb)*ac),0,float(by+2*(1-kb)*bc)};
        auto s=shading(1,1,bt2020_luma,bt2020_to_bt709);
        if (!draw(frame.Get(),w,h,rows,s.data(),out)) return 13;
        double largest=0;
        for (int x=0;x<w;x++) {
            double encoded=std::min(std::max((levels[x]-64)/876.0,0.0),1.0);
            double y=pq_light(encoded)/100, linear=std::min(y*(1+y/100)/(1+y),1.0);
            for (int c=0;c<3;c++) largest=std::max(largest,std::abs(out[x*4+c]-srgb(linear)));
        }
        std::cout<<"P010 grey PQ to SDR, max error: "<<largest<<std::endl;
        if (largest>.0005) return 14;
        // The same frame as the second slice of a texture array bound for
        // nothing a shader can read, as d3d11h264dec gives them (copied by
        // present_source): the same light, not the first slice's black.
        std::vector<unsigned short> black(planes.size());
        for (int i=0;i<w*h;i++) black[i]=64<<6;
        for (int i=0;i<w*h/2;i++) black[w*h+i]=512<<6;
        desc.ArraySize=2; desc.BindFlags=0;
        D3D11_SUBRESOURCE_DATA slices[2]={{black.data(),UINT(w*2),0},{planes.data(),UINT(w*2),0}};
        ComPtr<ID3D11Texture2D> array;
        if (FAILED(device->CreateTexture2D(&desc,slices,&array))) return 15;
        std::vector<float> sliced;
        if (!draw(array.Get(),w,h,rows,s.data(),sliced,1)) return 16;
        double apart=0;
        for (int i=0;i<w*4;i++) apart=std::max(apart,double(std::abs(sliced[i]-out[i])));
        std::cout<<"P010 from a texture array's second slice, unreadable by a shader: max difference "<<apart<<std::endl;
        if (apart>1e-6) return 17;
    }
    {   // The shaders compiled once a process (present_shaders): every
        // presenter after the first takes the same bytecode.
        ID3DBlob *vs1=nullptr,*ps1=nullptr,*vs2=nullptr,*ps2=nullptr;
        if (vmaf_present_prepare()<0 || FAILED(present_shaders(&vs1,&ps1)) || FAILED(present_shaders(&vs2,&ps2)))
            return 18;
        std::cout<<"presenter shaders compiled once: "<<(vs1==vs2 && ps1==ps2 ? "yes" : "no")<<std::endl;
        if (vs1!=vs2 || ps1!=ps2) return 19;
    }
    return 0;
}
