#include "native_common.h"

#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "native_presentation.h"
#include "vml/native_playback.h"

namespace vml::app {

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

SharedDevice& shared_device() {
    static SharedDevice shared;
    return shared;
}

void NativePresentation::present(NativeFramePtr frame, const Placement& placement,
                                 const std::optional<std::array<float, 16>>& shading) {
    follow_window();
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

void NativeDevice::release_shared() {
    std::shared_ptr<NativeDevice> going;
    {
        SharedDevice& shared = shared_device();
        const std::lock_guard lock(shared.mutex);
        going = std::move(shared.device);
    }
}

}  // namespace vml::app
