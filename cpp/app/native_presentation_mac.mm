// How Video Compare's native playback shows its frames on macOS
// (native_presentation.h): Metal draws them into a CAMetalLayer, in a view of
// its own over the view (which the mouse passes through), on the decoders'
// device, taking one frame per display refresh, with the shading of
// native/d3d11_tonemap.cpp (native_metal.h).
//
// As on Windows, a thread waits, holding no lock, until the layer has a
// drawable free (two of them: one shown, one drawn), and only then takes the
// frame that is due and draws it: frames chosen within one refresh are drawn
// one per refresh rather than lost. The drawable is presented at the display's
// refresh (displaySyncEnabled).
#include "native_presentation.h"

#import <AppKit/AppKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "native_device.h"
#include "native_metal.h"
#include "vml/native_playback.h"

// The presenter's view: a CAMetalLayer, which the mouse passes through to the
// view under it (the frame dragged when zoomed).
@interface VMLNativeVideoView : NSView
@end

@implementation VMLNativeVideoView
- (BOOL)wantsUpdateLayer {
    return YES;
}
- (CALayer*)makeBackingLayer {
    return [CAMetalLayer layer];
}
- (NSView*)hitTest:(NSPoint)point {
    (void)point;
    return nil;
}
@end

namespace vml::app {

namespace {

// The colour around a frame: the views' background (#171717).
constexpr std::array<float, 4> background{0x17 / 255.0f, 0x17 / 255.0f, 0x17 / 255.0f, 1.0f};

// What the layer is told its colours are: SDR (sRGB-coded BT.709), or HDR10
// (PQ in BT.2020, extended dynamic range), as Windows' swapchain is.
enum class Space { sdr, hdr10 };

struct Presenter {
    VMLNativeVideoView* view = nil;
    CAMetalLayer* layer = nil;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLRenderPipelineState> pipeline = nil;
    id<MTLSamplerState> sampler = nil;
    CVMetalTextureCacheRef cache = nullptr;
    std::optional<Space> asked;
    // Whether the view is on screen, as follow() or the window's
    // notifications last found it: its window shown and not wholly covered.
    // The drawing thread draws only then.
    std::atomic<bool> shown{true};
    // The window's notifications that it was covered, uncovered, minimised
    // or restored (removed with the presenter).
    std::vector<id> observers;
};

// YUV-to-RGB rows for frames of one kind, worked out once each.
const std::array<float, 12>& rows_for(const NativeFrame& frame) {
    static std::mutex mutex;
    static std::map<std::tuple<std::string, bool, bool>, std::array<float, 12>> made;
    const std::lock_guard lock(mutex);
    const auto key = std::make_tuple(frame.matrix, frame.full_range, frame.ten_bit);
    auto found = made.find(key);
    if (found == made.end()) {
        found = made.emplace(key, yuv_to_rgb(frame.matrix, frame.full_range, frame.ten_bit)).first;
    }
    return found->second;
}

// The shader's constants, as native/d3d11_tonemap.cpp's present_constants
// makes them: `source` the part drawn in the picture's pixels, `width` and
// `height` its buffer's.
void make_constants(float* k, float width, float height, const float* source, const float* to_rgb,
                    const float* shading) {
    std::fill(k, k + native_metal_constants, 0.f);
    k[0] = source[0] / width;
    k[1] = source[1] / height;
    k[2] = source[2] / width;
    k[3] = source[3] / height;
    for (int i = 0; i < 12; i++) {
        k[4 + i] = to_rgb[i];
    }
    k[16] = 1.f;
    if (!shading) {
        return;
    }
    k[17] = shading[0];
    k[18] = shading[1];
    k[19] = shading[14];
    k[20] = shading[2];
    k[21] = shading[3];
    k[22] = shading[4];
    k[23] = shading[15];
    for (int row = 0; row < 3; row++) {
        for (int i = 0; i < 3; i++) {
            k[24 + 4 * row + i] = shading[5 + 3 * row + i];
        }
    }
}

// The layer follows the view's size: its drawables are the view's size in
// the display's pixels. On the window's thread.
// Whether the presenter's view is on screen: off screen only in a window
// that is (minimised, hidden or wholly covered); a layer in no window yet
// still gives drawables at once.
bool on_screen(const Presenter& presenter) {
    NSWindow* window = presenter.view.window;
    return window == nil || (window.visible && (window.occlusionState & NSWindowOcclusionStateVisible) != 0 &&
                             !presenter.view.hiddenOrHasHiddenAncestor);
}

void follow(Presenter& presenter) {
    NSView* parent = presenter.view.superview;
    if (parent == nil) {
        return;
    }
    presenter.shown = on_screen(presenter);
    if (!NSEqualRects(presenter.view.frame, parent.bounds)) {
        presenter.view.frame = parent.bounds;
    }
    const CGFloat scale = parent.window != nil ? parent.window.backingScaleFactor : 1.0;
    const CGSize size = CGSizeMake(std::max<CGFloat>(1, std::round(parent.bounds.size.width * scale)),
                                   std::max<CGFloat>(1, std::round(parent.bounds.size.height * scale)));
    if (presenter.layer.contentsScale != scale) {
        presenter.layer.contentsScale = scale;
    }
    if (!CGSizeEqualToSize(presenter.layer.drawableSize, size)) {
        presenter.layer.drawableSize = size;
    }
}

// The window's keys back with Qt: when a video's surface (a native view of
// its own, which had them) goes, the window keeps them itself, and none
// reach the app (Video Compare's S, Space and arrows) until something is
// clicked. Its content view, the window's Qt view, takes them again.
void keep_keys(NSWindow* window) {
    if (window != nil && window.firstResponder == window && window.contentView != nil) {
        [window makeFirstResponder:window.contentView];
    }
}

// Tells the layer what its colours are, when that changes.
void set_space(Presenter& presenter, Space space) {
    if (presenter.asked == space) {
        return;
    }
    presenter.asked = space;
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    if (space == Space::hdr10) {
        CGColorSpaceRef pq = CGColorSpaceCreateWithName(kCGColorSpaceITUR_2100_PQ);
        presenter.layer.colorspace = pq;
        CGColorSpaceRelease(pq);
        presenter.layer.wantsExtendedDynamicRangeContent = YES;
    } else {
        CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        presenter.layer.colorspace = srgb;
        CGColorSpaceRelease(srgb);
        presenter.layer.wantsExtendedDynamicRangeContent = NO;
    }
    [CATransaction commit];
}

}  // namespace

std::optional<QString> NativePresentation::unavailable() {
    static const std::optional<QString> problem = []() -> std::optional<QString> {
        if (MTLCreateSystemDefaultDevice() == nil) {
            return QStringLiteral("this Mac has no Metal device");
        }
        return std::nullopt;
    }();
    return problem;
}

void NativePresentation::prepare() {
    // The shaders compiled for the session's device, off the window's thread
    // (NativePool::load).
    try {
        NSError* error = nil;
        native_metal_library((__bridge id<MTLDevice>)NativeDevice::shared()->metal(), &error);
    } catch (const std::exception&) {
        // No device: the first pool says why.
    }
}

NativePresentation::NativePresentation(std::shared_ptr<NativeDevice> device, NativeWindowHandle window,
                                       bool hdr_display)
    : device_(std::move(device)), window_(window) {
    NSView* parent = (__bridge NSView*)window_;
    if (parent == nil) {
        throw std::runtime_error("Could not set up the native video presentation");
    }
    auto presenter = std::make_unique<Presenter>();
    presenter->device = (__bridge id<MTLDevice>)device_->metal();
    presenter->cache = static_cast<CVMetalTextureCacheRef>(device_->texture_cache());
    NSError* error = nil;
    id<MTLLibrary> library = native_metal_library(presenter->device, &error);
    if (library == nil) {
        throw std::runtime_error("Could not set up the native video presentation: the Metal shaders did not compile");
    }
    // 10-bit drawables for an HDR display, for PQ BT.2020 frames as well as
    // SDR ones, as Windows' swapchain.
    const MTLPixelFormat format = hdr_display ? MTLPixelFormatBGR10A2Unorm : MTLPixelFormatBGRA8Unorm;
    MTLRenderPipelineDescriptor* description = [MTLRenderPipelineDescriptor new];
    description.vertexFunction = [library newFunctionWithName:@"present_vertex"];
    description.fragmentFunction = [library newFunctionWithName:@"present_fragment"];
    description.colorAttachments[0].pixelFormat = format;
    presenter->pipeline = [presenter->device newRenderPipelineStateWithDescriptor:description error:&error];
    MTLSamplerDescriptor* sampling = [MTLSamplerDescriptor new];
    sampling.minFilter = sampling.magFilter = MTLSamplerMinMagFilterLinear;
    sampling.sAddressMode = sampling.tAddressMode = MTLSamplerAddressModeClampToEdge;
    presenter->sampler = [presenter->device newSamplerStateWithDescriptor:sampling];
    presenter->queue = [presenter->device newCommandQueue];
    if (presenter->pipeline == nil || presenter->sampler == nil || presenter->queue == nil) {
        throw std::runtime_error("Could not set up the native video presentation");
    }
    presenter->view = [[VMLNativeVideoView alloc] initWithFrame:parent.bounds];
    presenter->view.wantsLayer = YES;
    presenter->view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    presenter->layer = (CAMetalLayer*)presenter->view.layer;
    presenter->layer.device = presenter->device;
    presenter->layer.pixelFormat = format;
    presenter->layer.framebufferOnly = YES;
    presenter->layer.opaque = YES;
    presenter->layer.maximumDrawableCount = 2;
    presenter->layer.displaySyncEnabled = YES;
    set_space(*presenter, Space::sdr);
    [parent addSubview:presenter->view];
    keep_keys(parent.window);
    follow(*presenter);
    // Shown again (uncovered, restored) with no frame coming (paused): the
    // frame kept while it was not is drawn then (present_paced).
    Presenter* raw = presenter.get();
    for (NSNotificationName name : {NSWindowDidChangeOcclusionStateNotification, NSWindowDidDeminiaturizeNotification,
                                    NSWindowDidMiniaturizeNotification}) {
        raw->observers.push_back([[NSNotificationCenter defaultCenter]
            addObserverForName:name
                        object:nil
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(NSNotification* note) {
                      if (note.object == raw->view.window) {
                          raw->shown = on_screen(*raw);
                          {
                              const std::lock_guard lock(mutex_);
                          }
                          due_ready_.notify_all();
                      }
                    }]);
    }
    presenter_ = presenter.release();
    thread_ = std::thread([this] { present_paced(); });
}

NativePresentation::~NativePresentation() {
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    due_ready_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
    auto* presenter = static_cast<Presenter*>(presenter_);
    for (id observer : presenter->observers) {
        [[NSNotificationCenter defaultCenter] removeObserver:observer];
    }
    // The view goes on the window's thread, which AppKit's views belong to.
    VMLNativeVideoView* view = presenter->view;
    dispatch_async(dispatch_get_main_queue(), ^{
      NSWindow* window = view.window;
      [view removeFromSuperview];
      keep_keys(window);
    });
    delete presenter;
}

void NativePresentation::present(NativeFramePtr frame, const Placement& placement,
                                 const std::optional<std::array<float, 16>>& shading) {
    follow(*static_cast<Presenter*>(presenter_));
    {
        const std::lock_guard lock(mutex_);
        due_.push_back({std::move(frame), placement, shading});
        // At most presenter_waiting_frames wait; past that the oldest is dropped.
        while (due_.size() > presenter_waiting_frames) {
            due_.pop_front();
        }
    }
    due_ready_.notify_all();
}

std::optional<QString> NativePresentation::error() const {
    const std::lock_guard lock(mutex_);
    return error_;
}

void NativePresentation::present_paced() {
    // Draws the waiting frames in order, one each time the layer has a
    // drawable free (once per refresh), as on Windows. A frame that cannot be
    // drawn ends the thread, its error kept for error().
    auto* presenter = static_cast<Presenter*>(presenter_);
    while (true) {
        {
            std::unique_lock lock(mutex_);
            due_ready_.wait(lock, [this] { return stopping_ || !due_.empty(); });
            if (stopping_) {
                return;
            }
        }
        if (!presenter->shown) {
            // Not on screen (minimised, covered): nothing is drawn, as the
            // layer would give no drawable but after waiting a second, which
            // a Close (joining this thread) would wait too. The newest frame
            // is kept, to be drawn when it is shown again, a paused one too.
            std::unique_lock lock(mutex_);
            while (due_.size() > 1) {
                due_.pop_front();
            }
            due_ready_.wait_for(lock, std::chrono::milliseconds(100),
                                [this, presenter] { return stopping_ || presenter->shown.load(); });
            continue;
        }
        @autoreleasepool {
            // Waits at most a second (the layer's own limit): nil for a hidden
            // view, say, and then it waits again.
            id<CAMetalDrawable> drawable = [presenter->layer nextDrawable];
            if (drawable == nil) {
                continue;
            }
            Due due;
            {
                const std::lock_guard lock(mutex_);
                if (stopping_ || due_.empty()) {
                    return;
                }
                due = std::move(due_.front());
                due_.pop_front();
            }
            // The drawable draw() draws into.
            current_drawable_ = (__bridge void*)drawable;
            draw(due);
            current_drawable_ = nullptr;
        }
        const std::lock_guard lock(mutex_);
        if (error_) {
            return;
        }
    }
}

void NativePresentation::draw(const Due& due) {
    auto* presenter = static_cast<Presenter*>(presenter_);
    id<CAMetalDrawable> drawable = (__bridge id<CAMetalDrawable>)current_drawable_;
    const NativeFrame& frame = *due.frame;
    const auto fail = [this](const QString& why) {
        const std::lock_guard lock(mutex_);
        error_ = QStringLiteral("Native presentation failed: %1").arg(why);
    };
    CVPixelBufferRef picture = frame.picture;
    const OSType layout = CVPixelBufferGetPixelFormatType(picture);
    const bool deep = layout == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ||
                      layout == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange;
    const bool shallow = layout == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
                         layout == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
    if (!deep && !shallow) {
        fail(QStringLiteral("a picture in a layout the presenter does not draw"));
        return;
    }
    Space space = frame.hdr10 ? Space::hdr10 : Space::sdr;
    if (due.shading) {
        // The layer is told whether the shader maps the frame to SDR or keeps
        // it HDR, as PQ in BT.2020.
        space = (*due.shading)[0] == 2.0f ? Space::hdr10 : Space::sdr;
    }
    if (presenter->layer.pixelFormat == MTLPixelFormatBGR10A2Unorm) {
        set_space(*presenter, space);
    }
    std::vector<CVMetalTextureRef> held;
    id<MTLTexture> planes[2] = {nil, nil};
    {
        const std::lock_guard lock(device_->lock());
        for (size_t plane = 0; plane < 2; ++plane) {
            const MTLPixelFormat format = plane == 0 ? (deep ? MTLPixelFormatR16Unorm : MTLPixelFormatR8Unorm)
                                                     : (deep ? MTLPixelFormatRG16Unorm : MTLPixelFormatRG8Unorm);
            CVMetalTextureRef texture = nullptr;
            if (CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, presenter->cache, picture, nullptr,
                                                          format, CVPixelBufferGetWidthOfPlane(picture, plane),
                                                          CVPixelBufferGetHeightOfPlane(picture, plane), plane,
                                                          &texture) == kCVReturnSuccess) {
                held.push_back(texture);
                planes[plane] = CVMetalTextureGetTexture(texture);
            }
        }
    }
    const auto release_held = [held] {
        for (CVMetalTextureRef texture : held) {
            CFRelease(texture);
        }
    };
    if (planes[0] == nil || planes[1] == nil) {
        release_held();
        fail(QStringLiteral("the picture could not be read"));
        return;
    }
    const float picture_width = static_cast<float>(CVPixelBufferGetWidth(picture));
    const float picture_height = static_cast<float>(CVPixelBufferGetHeight(picture));
    const id<MTLTexture> target = drawable.texture;
    const float width = static_cast<float>(target.width);
    const float height = static_cast<float>(target.height);
    float s[4] = {0, 0, picture_width, picture_height};
    if (due.placement.source[2] > 0 && due.placement.source[3] > 0) {
        for (int i = 0; i < 4; i++) {
            s[i] = static_cast<float>(due.placement.source[static_cast<std::size_t>(i)]);
        }
    }
    float t[4];
    if (due.placement.target[2] > 0 && due.placement.target[3] > 0) {
        for (int i = 0; i < 4; i++) {
            t[i] = static_cast<float>(due.placement.target[static_cast<std::size_t>(i)]);
        }
    } else {  // fitted, on whole pixels
        const float scale = std::min(width / s[2], height / s[3]);
        t[2] = std::round(s[2] * scale);
        t[3] = std::round(s[3] * scale);
        t[0] = std::floor((width - t[2]) / 2);
        t[1] = std::floor((height - t[3]) / 2);
    }
    float constants[native_metal_constants];
    make_constants(constants, picture_width, picture_height, s, rows_for(frame).data(),
                   due.shading ? due.shading->data() : nullptr);

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(background[0], background[1], background[2], background[3]);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> commands = [presenter->queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:presenter->pipeline];
    [encoder setViewport:MTLViewport{t[0], t[1], t[2], t[3], 0, 1}];
    [encoder setVertexBytes:constants length:sizeof constants atIndex:0];
    [encoder setFragmentBytes:constants length:sizeof constants atIndex:0];
    [encoder setFragmentTexture:planes[0] atIndex:0];
    [encoder setFragmentTexture:planes[1] atIndex:1];
    [encoder setFragmentSamplerState:presenter->sampler atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [encoder endEncoding];
    [commands presentDrawable:drawable];
    // The picture and its textures are kept until the GPU has drawn them.
    NativeFramePtr kept = due.frame;
    [commands addCompletedHandler:^(id<MTLCommandBuffer>) {
      release_held();
      (void)kept;
    }];
    [commands commit];
}

}  // namespace vml::app
