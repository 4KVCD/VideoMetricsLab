// What Video Compare's native playback runs on, on macOS (native_device.h):
// the bundled FFmpeg libraries, the Mac's GPU (Metal) with FFmpeg's
// VideoToolbox device beside it, the pixel buffers pictures decoded in
// software are written to, the scaler the source is scaled down with, and the
// Metal shaders both it and the presenter draw with (native_metal.h).
#include "native_device.h"

#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

#include "ffmpeg_api.h"
#include "native_metal.h"
#include "vml/app_log.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"

namespace vml::app {

namespace {

const app_log::Logger logger("vml.native_playback");

// The shaders (native_metal.h). The presenter's are native/d3d11_tonemap.cpp's,
// the same arithmetic in the same order, for the YUV frames native playback
// shows: its Draw constants are laid out as that cbuffer is. The scaler
// gives each output pixel the average of the source it covers: taps spread
// evenly over it, each read through the bilinear sampler.
NSString* const shader_source = @R"(
#include <metal_stdlib>
using namespace metal;

struct Draw {
    float4 source;       // the part of the frame drawn: left, top, width, height in texture coordinates
    float4 toRgb[3];     // rows of (Y, U, V) coefficients, the offset in w
    float yuv;           // always 1 here: the frames are YUV
    float shade;         // 0: as it comes; 1: HDR mapped to SDR BT.709; 2: HDR kept, PQ in BT.2020
    float kind;          // HDR: 1 PQ, 2 HLG
    float peak;          // HDR: the light mapped to SDR white, and HLG's display peak (nits)
    float4 luma;         // HDR: the source primaries' luminance weights; w: SDR white (nits)
    float4 toOutput[3];  // HDR: linear source RGB -> linear BT.709 (1) or BT.2020 (2), by rows
};

struct Vertex {
    float4 position [[position]];
    float2 uv;
};

vertex Vertex present_vertex(uint id [[vertex_id]], constant Draw& draw [[buffer(0)]]) {
    float2 corner = float2(float(id & 1), float(id >> 1));
    Vertex v;
    v.position = float4(corner.x * 2 - 1, 1 - corner.y * 2, 0, 1);
    v.uv = draw.source.xy + corner * draw.source.zw;
    return v;
}

float3 pq(float3 x) {
    float3 p = precise::pow(max(x, float3(0)), 1.0f / 78.84375f);
    return 10000 * precise::pow(max(p - 0.8359375f, float3(0)) / max(18.8515625f - 18.6875f * p, float3(1e-6f)),
                                1.0f / 0.1593017578125f);
}

float3 pq_code(float3 nits) {
    float3 y = precise::pow(saturate(nits / 10000), 0.1593017578125f);
    return precise::pow((0.8359375f + 18.8515625f * y) / (1 + 18.6875f * y), 78.84375f);
}

float3 hlg_scene(float3 x) {
    float3 lo = x * x / 3;
    float3 hi = (precise::exp((x - 0.55991073f) / 0.17883277f) + 0.28466892f) / 12;
    return float3(x.r <= 0.5f ? lo.r : hi.r, x.g <= 0.5f ? lo.g : hi.g, x.b <= 0.5f ? lo.b : hi.b);
}

float3 hlg(float3 x, constant Draw& draw) {
    float3 scene = hlg_scene(x);
    return scene * precise::pow(max(dot(scene, draw.luma.rgb), 1e-8f), 0.2f) * draw.peak;
}

float3 converted(float3 x, constant Draw& draw) {
    return float3(dot(x, draw.toOutput[0].rgb), dot(x, draw.toOutput[1].rgb), dot(x, draw.toOutput[2].rgb));
}

float3 srgb(float3 x) {
    float3 lo = 12.92f * x, hi = 1.055f * precise::pow(max(x, float3(0)), 1.0f / 2.4f) - 0.055f;
    return float3(x.r <= 0.0031308f ? lo.r : hi.r, x.g <= 0.0031308f ? lo.g : hi.g, x.b <= 0.0031308f ? lo.b : hi.b);
}

fragment float4 present_fragment(Vertex v [[stage_in]], constant Draw& draw [[buffer(0)]],
                                 texture2d<float> lumaPlane [[texture(0)]], texture2d<float> chromaPlane [[texture(1)]],
                                 sampler linearClamp [[sampler(0)]]) {
    float3 c = float3(lumaPlane.sample(linearClamp, v.uv).r, chromaPlane.sample(linearClamp, v.uv).rg);
    float3 rgb = saturate(float3(dot(c, draw.toRgb[0].xyz), dot(c, draw.toRgb[1].xyz), dot(c, draw.toRgb[2].xyz))
                          + float3(draw.toRgb[0].w, draw.toRgb[1].w, draw.toRgb[2].w));
    if (draw.shade < 0.5f) return float4(rgb, 1);
    float3 light = draw.kind < 1.5f ? pq(rgb) : hlg(rgb, draw);
    if (draw.shade > 1.5f) {
        // HDR kept, for an HDR display, which takes PQ in BT.2020: the light
        // converted to BT.2020 and coded as PQ (HLG's at its display peak).
        return float4(pq_code(converted(light, draw)), 1);
    }
    // Fixed extended-Reinhard luminance curve, shared by all comparison sides.
    float white = draw.luma.w;
    float y = max(dot(light, draw.luma.rgb) / white, 0.0f);
    float w = draw.peak / white;
    float mapped = y * (1 + y / (w * w)) / (1 + y);
    light = light / white * (y > 1e-8f ? mapped / y : 0);
    // Linear source primaries -> BT.709, then gamut clipping and sRGB encoding.
    return float4(srgb(saturate(converted(light, draw))), 1);
}

struct Scale {
    float4 source;  // the part read: left, top, width, height in texture coordinates
    float2 output;  // the output's size, in pixels
    float2 taps;    // taps across one output pixel, and down it
};

vertex float4 scale_vertex(uint id [[vertex_id]]) {
    float2 corner = float2(float(id & 1), float(id >> 1));
    return float4(corner.x * 2 - 1, 1 - corner.y * 2, 0, 1);
}

fragment float4 scale_fragment(float4 at [[position]], constant Scale& scale [[buffer(0)]],
                               texture2d<float> plane [[texture(0)]], sampler linearClamp [[sampler(0)]]) {
    float2 cell = scale.source.zw / scale.output;
    float2 corner = scale.source.xy + floor(at.xy) * cell;
    int across = int(scale.taps.x), down = int(scale.taps.y);
    float4 sum = float4(0);
    for (int j = 0; j < down; j++) {
        for (int i = 0; i < across; i++) {
            sum += plane.sample(linearClamp, corner + (float2(float(i), float(j)) + 0.5f) / scale.taps * cell);
        }
    }
    return sum / float(across * down);
}
)";

struct SharedDevice {
    std::mutex mutex;
    std::shared_ptr<NativeDevice> device;
};

SharedDevice& shared_device() {
    static SharedDevice shared;
    return shared;
}

// A pixel buffer's planes as Metal reads and writes them: R and RG, of 8
// bits (NV12's layout) or 16 (P010's); nothing for another layout.
std::optional<std::pair<MTLPixelFormat, MTLPixelFormat>> plane_formats(CVPixelBufferRef buffer) {
    switch (CVPixelBufferGetPixelFormatType(buffer)) {
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
        return std::pair(MTLPixelFormatR8Unorm, MTLPixelFormatRG8Unorm);
    case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
        return std::pair(MTLPixelFormatR16Unorm, MTLPixelFormatRG16Unorm);
    default:
        return std::nullopt;
    }
}

// A plane of `buffer` as a Metal texture, through `cache`; held by the
// CVMetalTexture returned with it (released by the caller once the GPU is
// done with it), or nothing.
std::pair<id<MTLTexture>, CVMetalTextureRef> plane_texture(CVMetalTextureCacheRef cache, CVPixelBufferRef buffer,
                                                         size_t plane, MTLPixelFormat format) {
    CVMetalTextureRef texture = nullptr;
    if (CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, buffer, nullptr, format,
                                                  CVPixelBufferGetWidthOfPlane(buffer, plane),
                                                  CVPixelBufferGetHeightOfPlane(buffer, plane), plane,
                                                  &texture) != kCVReturnSuccess) {
        return {nil, nullptr};
    }
    return {CVMetalTextureGetTexture(texture), texture};
}

}  // namespace

std::optional<std::string> load_ffmpeg_libraries() {
    static std::mutex mutex;
    static std::optional<std::string> problem;
    static bool tried = false;
    const std::lock_guard lock(mutex);
    if (tried) {
        return problem;
    }
    tried = true;
    // Linked weakly by @rpath (cpp/app/CMakeLists.txt): loaded as the window
    // started where they were found, else their functions are null.
    constexpr std::array<const char*, 5> names{"libavutil.61.dylib", "libswresample.7.dylib", "libavcodec.63.dylib",
                                               "libavformat.63.dylib", "libswscale.10.dylib"};
    const std::filesystem::path folder = resource_folder() / "tools" / "ffmpeg";
    for (const char* name : names) {
        std::error_code ignored;
        if (!std::filesystem::is_regular_file(folder / name, ignored)) {
            problem = std::string("FFmpeg's libraries are not bundled (") + name + " is missing)";
            return problem;
        }
    }
    if (&avutil_version == nullptr || &avcodec_version == nullptr || &avformat_version == nullptr ||
        &swscale_version == nullptr || &swr_alloc == nullptr) {
        problem = "FFmpeg's libraries did not load as the app started";
        return problem;
    }
    if (avutil_version() >> 16 != LIBAVUTIL_VERSION_MAJOR || avcodec_version() >> 16 != LIBAVCODEC_VERSION_MAJOR ||
        avformat_version() >> 16 != LIBAVFORMAT_VERSION_MAJOR || swscale_version() >> 16 != LIBSWSCALE_VERSION_MAJOR) {
        problem = "FFmpeg's libraries are not the version this app was built for";
        return problem;
    }
    // FFmpeg says nothing on the window's console, as on Windows.
    av_log_set_level(AV_LOG_QUIET);
    return problem;
}

id<MTLLibrary> native_metal_library(id<MTLDevice> device, NSError** error) {
    static std::mutex mutex;
    static NSMapTable<id<MTLDevice>, id<MTLLibrary>>* made = [NSMapTable weakToStrongObjectsMapTable];
    const std::lock_guard lock(mutex);
    if (id<MTLLibrary> library = [made objectForKey:device]) {
        return library;
    }
    MTLCompileOptions* options = [MTLCompileOptions new];
    // As the HLSL's arithmetic: no fast-math reassociation.
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
    } else {
        options.fastMathEnabled = NO;
    }
    const auto start = std::chrono::steady_clock::now();
    id<MTLLibrary> library = [device newLibraryWithSource:shader_source options:options error:error];
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (library == nil) {
        logger.error(std::string("The Metal shaders did not compile: ") +
                     (error && *error ? [[*error localizedDescription] UTF8String] : "no reason given"));
        return nil;
    }
    logger.info("Metal shaders compiled in " + std::to_string(static_cast<int>(std::lround(ms))) + " ms");
    [made setObject:library forKey:device];
    return library;
}

// The pixel buffer pools (take_buffer), and the scaler's pipelines.
struct NativeDevice::Pools {
    std::mutex mutex;
    std::map<std::tuple<int, int, bool>, CVPixelBufferPoolRef> pools;
    std::map<MTLPixelFormat, id<MTLRenderPipelineState>> scalers;
    id<MTLCommandQueue> queue = nil;
    id<MTLSamplerState> sampler = nil;
    ~Pools() {
        for (auto& [key, pool] : pools) {
            CVPixelBufferPoolRelease(pool);
        }
    }
};

void NativeDevice::release_shared() {
    std::shared_ptr<NativeDevice> going;
    {
        SharedDevice& shared = shared_device();
        const std::lock_guard lock(shared.mutex);
        going = std::move(shared.device);
    }
}

std::shared_ptr<NativeDevice> NativeDevice::shared() {
    SharedDevice& shared = shared_device();
    const std::lock_guard lock(shared.mutex);
    if (!shared.device) {
        shared.device = std::make_shared<NativeDevice>();
    }
    return shared.device;
}

NativeDevice::NativeDevice() : pools_(std::make_unique<Pools>()) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("Metal device unavailable");
    }
    CVMetalTextureCacheRef cache = nullptr;
    if (CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, device, nullptr, &cache) != kCVReturnSuccess) {
        throw std::runtime_error("Metal's texture cache could not be made");
    }
    if (av_hwdevice_ctx_create(&hardware_, AV_HWDEVICE_TYPE_VIDEOTOOLBOX, nullptr, nullptr, 0) < 0) {
        CFRelease(cache);
        throw std::runtime_error("FFmpeg's VideoToolbox decoding device could not be made");
    }
    pools_->queue = [device newCommandQueue];
    MTLSamplerDescriptor* sampling = [MTLSamplerDescriptor new];
    sampling.minFilter = sampling.magFilter = MTLSamplerMinMagFilterLinear;
    sampling.sAddressMode = sampling.tAddressMode = MTLSamplerAddressModeClampToEdge;
    pools_->sampler = [device newSamplerStateWithDescriptor:sampling];
    metal_ = (__bridge_retained void*)device;
    texture_cache_ = cache;
}

NativeDevice::~NativeDevice() {
    pools_.reset();
    av_buffer_unref(&hardware_);
    if (texture_cache_ != nullptr) {
        CFRelease(static_cast<CVMetalTextureCacheRef>(texture_cache_));
    }
    if (metal_ != nullptr) {
        CFRelease(metal_);
    }
}

std::vector<std::string> NativeDevice::gpu_decoders() const {
    // The codecs FFmpeg's VideoToolbox decoders take, in the order Windows'
    // self-test names them.
    const std::array<std::pair<CMVideoCodecType, const char*>, 7> known{{
        {kCMVideoCodecType_H264, "H.264"},
        {kCMVideoCodecType_HEVC, "HEVC"},
        {kCMVideoCodecType_HEVC, "HEVC 10-bit"},
        {kCMVideoCodecType_AV1, "AV1"},
        {kCMVideoCodecType_VP9, "VP9"},
        {kCMVideoCodecType_MPEG2Video, "MPEG-2"},
        {kCMVideoCodecType_AppleProRes422, "ProRes"},
    }};
    std::vector<std::string> found;
    for (const auto& [codec, name] : known) {
        if (VTIsHardwareDecodeSupported(codec)) {
            found.emplace_back(name);
        }
    }
    return found;
}

HeldPixelBuffer NativeDevice::take_buffer(int width, int height, bool ten_bit) {
    const std::lock_guard lock(pools_->mutex);
    const auto key = std::make_tuple(width, height, ten_bit);
    auto found = pools_->pools.find(key);
    if (found == pools_->pools.end()) {
        // A few sizes and formats at once: a source decoded in software and
        // scaled down takes two in turn, each frame (one pool each, as
        // Windows' decoders keep an UploadTextures each). Past that, they go:
        // the comparison changed. A pool's buffers still held outlive it.
        constexpr std::size_t kept_pools = 4;
        if (pools_->pools.size() >= kept_pools) {
            for (auto& [other, pool] : pools_->pools) {
                CVPixelBufferPoolRelease(pool);
            }
            pools_->pools.clear();
        }
        NSDictionary* attributes = @{
            (id)kCVPixelBufferPixelFormatTypeKey : @(ten_bit ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
                                                              : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            (id)kCVPixelBufferWidthKey : @(width),
            (id)kCVPixelBufferHeightKey : @(height),
            (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
            (id)kCVPixelBufferMetalCompatibilityKey : @YES,
        };
        CVPixelBufferPoolRef pool = nullptr;
        if (CVPixelBufferPoolCreate(kCFAllocatorDefault, nullptr, (__bridge CFDictionaryRef)attributes, &pool) !=
            kCVReturnSuccess) {
            return nullptr;
        }
        found = pools_->pools.emplace(key, pool).first;
    }
    CVPixelBufferRef buffer = nullptr;
    if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, found->second, &buffer) != kCVReturnSuccess) {
        return nullptr;
    }
    return HeldPixelBuffer(buffer, [](PixelBuffer held) { CVPixelBufferRelease(held); });
}

HeldPixelBuffer NativeDevice::scale(PixelBuffer picture, const std::array<int, 4>& crop, std::pair<int, int> size) {
    const std::optional<std::pair<MTLPixelFormat, MTLPixelFormat>> formats = plane_formats(picture);
    if (!formats || size.first <= 0 || size.second <= 0) {
        return nullptr;
    }
    const bool ten_bit = formats->first == MTLPixelFormatR16Unorm;
    HeldPixelBuffer output = take_buffer((size.first + 1) & ~1, (size.second + 1) & ~1, ten_bit);
    if (!output) {
        return nullptr;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)metal_;
    NSError* error = nil;
    id<MTLLibrary> library = native_metal_library(device, &error);
    if (library == nil) {
        return nullptr;
    }
    const std::lock_guard lock(mutex_);
    const auto pipeline = [&](MTLPixelFormat format) -> id<MTLRenderPipelineState> {
        __strong id<MTLRenderPipelineState>& made = pools_->scalers[format];
        if (made == nil) {
            MTLRenderPipelineDescriptor* description = [MTLRenderPipelineDescriptor new];
            description.vertexFunction = [library newFunctionWithName:@"scale_vertex"];
            description.fragmentFunction = [library newFunctionWithName:@"scale_fragment"];
            description.colorAttachments[0].pixelFormat = format;
            made = [device newRenderPipelineStateWithDescriptor:description error:nil];
        }
        return made;
    };
    auto* cache = static_cast<CVMetalTextureCacheRef>(texture_cache_);
    std::vector<CVMetalTextureRef> held;
    const auto release_held = [&held] {
        for (CVMetalTextureRef texture : held) {
            CFRelease(texture);
        }
    };
    id<MTLCommandBuffer> commands = [pools_->queue commandBuffer];
    for (size_t plane = 0; plane < 2; ++plane) {
        const MTLPixelFormat format = plane == 0 ? formats->first : formats->second;
        auto [source, source_held] = plane_texture(cache, picture, plane, format);
        auto [target, target_held] = plane_texture(cache, output.get(), plane, format);
        if (source_held != nullptr) {
            held.push_back(source_held);
        }
        if (target_held != nullptr) {
            held.push_back(target_held);
        }
        id<MTLRenderPipelineState> state = pipeline(format);
        if (source == nil || target == nil || state == nil) {
            release_held();
            return nullptr;
        }
        // The chroma plane has half the luma's pixels each way.
        const float shrink = plane == 0 ? 1.0f : 2.0f;
        const float width = static_cast<float>(CVPixelBufferGetWidthOfPlane(picture, plane));
        const float height = static_cast<float>(CVPixelBufferGetHeightOfPlane(picture, plane));
        const float out_width = static_cast<float>(target.width);
        const float out_height = static_cast<float>(target.height);
        struct {
            float source[4];
            float output[2];
            float taps[2];
        } constants{{static_cast<float>(crop[0]) / shrink / width, static_cast<float>(crop[1]) / shrink / height,
                     static_cast<float>(crop[2]) / shrink / width, static_cast<float>(crop[3]) / shrink / height},
                    {out_width, out_height},
                    {std::clamp(std::ceil(static_cast<float>(crop[2]) / shrink / out_width), 1.0f, 16.0f),
                     std::clamp(std::ceil(static_cast<float>(crop[3]) / shrink / out_height), 1.0f, 16.0f)}};
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
        [encoder setRenderPipelineState:state];
        [encoder setFragmentBytes:&constants length:sizeof constants atIndex:0];
        [encoder setFragmentTexture:source atIndex:0];
        [encoder setFragmentSamplerState:pools_->sampler atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [encoder endEncoding];
    }
    [commands commit];
    [commands waitUntilCompleted];
    release_held();
    if (commands.status != MTLCommandBufferStatusCompleted) {
        return nullptr;
    }
    return output;
}

}  // namespace vml::app
