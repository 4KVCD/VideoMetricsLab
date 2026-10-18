// Native playback's Metal shaders (macOS), for native_device_mac.mm and
// native_presentation_mac.mm: the presenter's, a port of
// native/d3d11_tonemap.cpp's HLSL with the same arithmetic in the same order,
// and the scaler's. They are compiled from their source when first needed,
// for the device they run on: the Command Line Tools have no offline Metal
// compiler (xcrun metal). NativePool::load() compiles them off the window's
// thread as Video Compare first loads (NativePresentation::prepare), so that
// the window never waits for the compiler.
#pragma once

#import <Metal/Metal.h>

namespace vml::app {

// The shaders' library on `device`, compiled the first time it is asked for
// on that device (thread-safe); nil, with `error` set, when it cannot be.
id<MTLLibrary> native_metal_library(id<MTLDevice> device, NSError** error);

// The presenter's constants, in floats, laid out as the shader's Draw (and
// as native/d3d11_tonemap.cpp's cbuffer Draw).
constexpr int native_metal_constants = 36;

}  // namespace vml::app
