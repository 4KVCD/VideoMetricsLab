// What Windows' and a Mac's native playback share (native_presentation*,
// native_device*).
#pragma once

#include <array>
#include <memory>
#include <mutex>

#include "native_decoder.h"
#include "native_device.h"

namespace vml::app {

// YUV-to-RGB rows for frames of one kind, worked out once each.
const std::array<float, 12>& rows_for(const NativeFrame& frame);

// The one device playback shares, made when first asked for.
struct SharedDevice {
    std::mutex mutex;
    std::shared_ptr<NativeDevice> device;
};
SharedDevice& shared_device();

}  // namespace vml::app
