#include "native_presentation.h"

#include <d3d11.h>

#include <map>
#include <stdexcept>
#include <tuple>

#include "native_device.h"
#include "vml/model_select.h"
#include "vml/native_playback.h"
#include "vml/platform/library.h"

namespace vml::app {

namespace {

// The colour around a frame: the views' background (#171717).
constexpr std::array<float, 4> background{0x17 / 255.0f, 0x17 / 255.0f, 0x17 / 255.0f, 1.0f};
// DXGI colour spaces: SDR (sRGB-coded BT.709) and HDR10 (PQ in BT.2020).
constexpr int sdr_space = 0;
constexpr int hdr10_space = 12;

// The presenter's functions (native/d3d11_tonemap.cpp).
struct PresenterApi {
    using Prepare = int (*)();
    using Create = void* (*)(ID3D11Device*, HWND, int);
    using Follow = void (*)(void*, HWND);
    using Wait = int (*)(void*, unsigned);
    using Texture = int (*)(void*, ID3D11Resource*, unsigned, const int*, const int*, const float*, const float*, int,
                            const float*);
    using Destroy = void (*)(void*);
    Prepare prepare = nullptr;
    Create create = nullptr;
    Follow follow = nullptr;
    Wait wait = nullptr;
    Texture texture = nullptr;
    Destroy destroy = nullptr;
    std::optional<QString> problem;
};

// Loads the presenter once per process.
const PresenterApi& presenter_api() {
    static const PresenterApi api = [] {
        PresenterApi loaded;
        static std::optional<platform::DynamicLibrary> library;
        try {
            library.emplace(resource_folder() / "native" / "d3d11_tonemap.dll");
        } catch (const std::exception&) {
            loaded.problem = QStringLiteral("the native video presenter is not built (d3d11_tonemap.dll)");
            return loaded;
        }
        const auto bind = [&](auto& function, const char* name) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(library->find(name));
            return function != nullptr;
        };
        if (!(bind(loaded.create, "vmaf_present_create") && bind(loaded.follow, "vmaf_present_follow") &&
              bind(loaded.wait, "vmaf_present_wait") && bind(loaded.texture, "vmaf_present_texture") &&
              bind(loaded.destroy, "vmaf_present_destroy"))) {
            loaded.problem = QStringLiteral("the native video presenter is out of date (d3d11_tonemap.dll)");
        }
        bind(loaded.prepare, "vmaf_present_prepare");  // an older build compiles them for its first presenter
        return loaded;
    }();
    return api;
}

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

}  // namespace

std::optional<QString> NativePresentation::unavailable() { return presenter_api().problem; }

void NativePresentation::prepare() {
    const PresenterApi& api = presenter_api();
    if (!api.problem && api.prepare != nullptr) {
        api.prepare();
    }
}

NativePresentation::NativePresentation(std::shared_ptr<NativeDevice> device, HWND__* window, bool hdr_display)
    : device_(std::move(device)), window_(window) {
    const PresenterApi& api = presenter_api();
    if (api.problem) {
        throw std::runtime_error(api.problem->toStdString());
    }
    {
        const std::lock_guard lock(device_->lock());
        presenter_ = api.create(device_->device(), window_, hdr_display ? 1 : 0);
    }
    if (presenter_ == nullptr) {
        throw std::runtime_error("Could not set up the native video presentation");
    }
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
    const std::lock_guard lock(device_->lock());
    presenter_api().destroy(presenter_);
}

void NativePresentation::present(NativeFramePtr frame, const Placement& placement,
                                 const std::optional<std::array<float, 16>>& shading) {
    presenter_api().follow(presenter_, window_);
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
    // Draws the waiting frames in order, one each time the swapchain takes a
    // frame (once per refresh): frames can be chosen as little as 4 ms apart,
    // and two within one refresh would otherwise lose one. A frame that
    // cannot be drawn ends the thread, its error kept for error().
    const PresenterApi& api = presenter_api();
    while (true) {
        {
            std::unique_lock lock(mutex_);
            due_ready_.wait(lock, [this] { return stopping_ || !due_.empty(); });
            if (stopping_) {
                return;
            }
        }
        const int waited = api.wait(presenter_, 100);
        if (waited < 0) {
            const std::lock_guard lock(mutex_);
            error_ = QStringLiteral("Native presentation failed: the swapchain failed");
            return;
        }
        if (waited != 0) {
            continue;  // the swapchain took no frame yet (a hidden window, say): wait again
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
        draw(due);
        const std::lock_guard lock(mutex_);
        if (error_) {
            return;
        }
    }
}

void NativePresentation::draw(const Due& due) {
    const NativeFrame& frame = *due.frame;
    const std::array<float, 12>& rows = rows_for(frame);
    int space = frame.hdr10 ? hdr10_space : sdr_space;
    if (due.shading) {
        // The swapchain is told whether the shader maps the frame to SDR or
        // keeps it HDR, as PQ in BT.2020.
        space = (*due.shading)[0] == 2.0f ? hdr10_space : sdr_space;
    }
    int result = 0;
    {
        const std::lock_guard lock(device_->lock());
        result = presenter_api().texture(presenter_, frame.texture, frame.subresource, due.placement.source.data(),
                                         due.placement.target.data(), rows.data(),
                                         due.shading ? due.shading->data() : nullptr, space, background.data());
    }
    if (result < 0) {
        const std::lock_guard lock(mutex_);
        error_ = QStringLiteral("Native presentation failed: 0x%1").arg(static_cast<unsigned>(result), 8, 16, QChar('0'));
    }
}

}  // namespace vml::app
