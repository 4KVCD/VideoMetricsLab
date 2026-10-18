#include "vml/vmaf_vulkan.h"

#include <algorithm>
#include <system_error>

#include "vml/gpu_frames.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/sha256.h"
#include "vml/vship.h"

namespace vml {

namespace {

// Positions in the engine's feature rows (vmaf_vulkan.cpp).
constexpr std::size_t vif = 0, adm2 = 4, motion2 = 5, vif_neg = 6, adm2_neg = 10;

using ModelFeatures = std::vector<std::pair<std::string, std::size_t>>;

// For VMAF and for VMAF NEG: libvmaf's names for the model's features, and
// their positions in a row.
const ModelFeatures& model_features(bool neg) {
    static const ModelFeatures plain{
        {"VMAF_integer_feature_adm2_score", adm2},
        {"VMAF_integer_feature_motion2_score", motion2},
        {"VMAF_integer_feature_vif_scale0_score", vif},
        {"VMAF_integer_feature_vif_scale1_score", vif + 1},
        {"VMAF_integer_feature_vif_scale2_score", vif + 2},
        {"VMAF_integer_feature_vif_scale3_score", vif + 3},
    };
    // VMAF NEG's features carry their enhancement gain limit in the name.
    static const ModelFeatures negative{
        {"integer_adm2_egl_1", adm2_neg},
        {"VMAF_integer_feature_motion2_score", motion2},
        {"integer_vif_scale0_egl_1", vif_neg},
        {"integer_vif_scale1_egl_1", vif_neg + 1},
        {"integer_vif_scale2_egl_1", vif_neg + 2},
        {"integer_vif_scale3_egl_1", vif_neg + 3},
    };
    return neg ? negative : plain;
}

bool is_neg_model(const std::string& version) {
    return version == "vmaf_v0.6.1neg" || version == "vmaf_4k_v0.6.1neg";
}

// Vulkan's device types for real GPUs. The other types are virtual GPUs
// and CPU renderers.
constexpr std::uint32_t type_integrated = 1, type_discrete = 2;

// Vulkan's number for the GPU to score on, instead of the one
// best_vulkan_device() picks. It is for testing an integrated GPU on a PC
// that also has a discrete one.
constexpr const char* device_variable = "VML_VMAF_VULKAN_DEVICE";

// How many of the engine's sums per frame are VMAF v0.6.1's, which the
// probe checks: motion's (1), VIF's (4 scales of 9), ADM's denominators
// (4 x 3) and its numerators (2 limits x 4 scales x 3). VMAF v1's CAMBI
// sums follow them.
constexpr std::size_t probe_slots = 1 + 4 * 9 + 4 * 3 + 2 * 4 * 3;
// The SHA-256 of the sums the probe's frames give, at 8 and then 10 bits:
// sums whose features are identical to libvmaf's CUDA code's (the Python
// app's tests/test_vmaf_vulkan.py compares them where CUDA runs).
constexpr std::array<const char*, 2> probe_digests{
    "33668b74708340de30734d996e3b60ea025c4c55a5d977707ea6482467f85dee",
    "76f6f75034298e831c97642952e3682d896b85713efddf8e32d65a19c09bc3b1",
};
constexpr int probe_width = 320, probe_height = 192;

// Throws VmafVulkanError saying that `what` failed, with the engine's
// message. The engine keeps its message per thread, so it is read on the
// thread whose call failed, straight after it.
[[noreturn]] void fail(const VulkanEngine& engine, const std::string& what) {
    const char* said = engine.error();
    throw VmafVulkanError(what + " failed: " + replace_invalid_utf8(said != nullptr ? said : ""));
}

// The engine's function `name` as a pointer of type F, or nullptr when the
// engine has none.
template <typename F>
F optional_function(const platform::DynamicLibrary& library, const char* name) {
    return reinterpret_cast<F>(library.find(name));
}

// A sample of the probe's frames, as the engine reads it: a byte at 8 bits,
// otherwise 16 bits, little-endian.
void append_sample(std::vector<std::uint8_t>& frame, std::int64_t value, bool wide) {
    frame.push_back(static_cast<std::uint8_t>(value & 0xFF));
    if (wide) {
        frame.push_back(static_cast<std::uint8_t>(value >> 8));
    }
}

}  // namespace

VulkanEngine::VulkanEngine(const std::filesystem::path& path)
    : library(path),
      device(library.function<decltype(device)>("vv_device")),
      create(library.function<decltype(create)>("vv_create")),
      destroy(library.function<decltype(destroy)>("vv_destroy")),
      error(library.function<decltype(error)>("vv_error")),
      submit(library.function<decltype(submit)>("vv_submit")),
      flush(library.function<decltype(flush)>("vv_flush")),
      features(library.function<decltype(features)>("vv_features")),
      sums(library.function<decltype(sums)>("vv_sums")),
      staging(library.function<decltype(staging)>("vv_staging")),
      commit(library.function<decltype(commit)>("vv_commit")),
      export_memory(library.function<decltype(export_memory)>("vv_export")),
      shared_next(library.function<decltype(shared_next)>("vv_shared_next")),
      create_v1(optional_function<decltype(create_v1)>(library, "vv_create_v1")),
      features_v1(optional_function<decltype(features_v1)>(library, "vv_features_v1")),
      v1_cambi(optional_function<decltype(v1_cambi)>(library, "vv_v1_cambi")),
      v1_cambi_scores(optional_function<decltype(v1_cambi_scores)>(library, "vv_v1_cambi_scores")),
      v1_speed(optional_function<decltype(v1_speed)>(library, "vv_v1_speed")),
      v1_speed_scores(optional_function<decltype(v1_speed_scores)>(library, "vv_v1_speed_scores")),
      submit_v1(optional_function<decltype(submit_v1)>(library, "vv_submit_v1")),
      chroma_staging(optional_function<decltype(chroma_staging)>(library, "vv_chroma_staging")),
      shared_chroma(optional_function<decltype(shared_chroma)>(library, "vv_shared_chroma")),
      shared_device(optional_function<decltype(shared_device)>(library, "vv_shared_device")),
      import_timeline(optional_function<decltype(import_timeline)>(library, "vv_import_timeline")),
      commit_after(optional_function<decltype(commit_after)>(library, "vv_commit_after")),
      pictures(optional_function<decltype(pictures)>(library, "vv_pictures")),
      pictures_mode(optional_function<decltype(pictures_mode)>(library, "vv_pictures_mode")) {}

void check_vulkan(const VulkanEngine& engine, int error, const std::string& what) {
    if (error < 0) {
        fail(engine, what);
    }
}

std::filesystem::path vulkan_engine_path() {
#ifdef __APPLE__
    static_cast<void>(vulkan_library_path());  // the loader it links pointed at MoltenVK before it loads
#endif
    return resource_folder() / "tools" / "vmaf_vulkan" / platform::shared_library_name("vmaf_vulkan");
}

const VulkanEngine& vulkan_engine() {
    try {
        static const VulkanEngine loaded(vulkan_engine_path());
        return loaded;
    } catch (const platform::LibraryError& error) {
        throw VmafVulkanError(error.what());
    }
}

std::string vulkan_library_build() { return "libvmaf-fast " + libvmaf_fast_version() + " (Vulkan)"; }

std::vector<VulkanDevice> vulkan_devices() {
    const VulkanEngine& engine = vulkan_engine();
    const int count = engine.device(-1, nullptr, 0, nullptr, nullptr, nullptr);
    check_vulkan(engine, count, "Listing the GPUs");
    std::vector<VulkanDevice> found;
    for (int index = 0; index < count; ++index) {
        std::array<char, 256> name{};
        std::uint32_t vendor = 0, kind = 0, flags = 0;
        engine.device(index, name.data(), static_cast<int>(name.size()), &vendor, &kind, &flags);
        found.push_back({index, replace_invalid_utf8(name.data()), vendor, kind, (flags & 1) != 0, (flags & 2) != 0});
    }
    return found;
}

std::optional<VulkanDevice> best_vulkan_device(const std::vector<VulkanDevice>& found) {
    const auto order = [](const VulkanDevice& device) { return std::pair(device.kind != type_discrete, device.index); };
    std::optional<VulkanDevice> best;
    for (const VulkanDevice& device : found) {
        if (device.usable && (device.kind == type_discrete || device.kind == type_integrated) &&
            (!best || order(device) < order(*best))) {
            best = device;
        }
    }
    return best;
}

SharedSlots::SharedSlots(const VulkanEngine& engine, VulkanContext* context, GpuFrameStream& decoder)
    : engine_(engine), context_(context), decoder_(decoder) {
    try {
        // AMD's decoder imports nothing without it (GpuFrameStream::import_memory).
        std::optional<SharedDevice> exporter;
        if (engine_.shared_device != nullptr) {
            SharedDevice described;
            check_vulkan(engine_,
                         engine_.shared_device(context_, described.device_uuid.data(), described.driver_uuid.data(),
                                               &described.memory_type),
                         "Sharing Vulkan's memory");
            exporter = described;
        }
        void* handle = nullptr;
        std::uint64_t size = 0;
        while (engine_.export_memory(context_, static_cast<int>(imports_.size()), &handle, &size) == 0) {
            const std::optional<std::pair<std::uint64_t, void*>> imported =
                decoder_.import_memory(handle, size, exporter ? &*exporter : nullptr);
            platform::close_native_handle(handle);
            if (!imported) {
                throw VmafVulkanError("the decoder cannot write into Vulkan's memory");
            }
            imports_.push_back(*imported);
        }
        if (imports_.empty()) {
            const char* said = engine_.error();
            throw VmafVulkanError("Vulkan's memory is not shared: " + replace_invalid_utf8(said != nullptr ? said : ""));
        }
    } catch (...) {
        for (const auto& [address, memory] : imports_) {
            decoder_.unimport(memory);
        }
        throw;
    }
}

SharedSlots::~SharedSlots() {
    for (const auto& [address, memory] : imports_) {
        decoder_.unimport(memory);
    }
}

SharedSlots::Next SharedSlots::next() {
    int slot = 0;
    std::uint32_t stride = 0;
    std::uint32_t distorted_offset = 0;
    check_vulkan(engine_, engine_.shared_next(context_, &slot, &stride, &distorted_offset), "Scoring a frame");
    const std::uint64_t base = imports_.at(static_cast<std::size_t>(slot)).first;
    return {base, base + distorted_offset, stride, base};
}

VulkanScorer::VulkanScorer(int width, int height, int bit_depth,
                           std::vector<std::pair<std::string, std::string>> models, int n_subsample,
                           std::optional<int> device, GpuFrameStream* shared)
    : engine_(vulkan_engine()), models_(std::move(models)), step_(static_cast<unsigned>(std::max(1, n_subsample))) {
    libvmaf();  // it predicts the scores at the end: one that cannot load fails the run now
    if (!device) {
        const std::optional<VulkanDevice> chosen = best_vulkan_device(vulkan_devices());
        if (!chosen) {
            throw VmafVulkanError("no GPU that Vulkan can calculate VMAF on");
        }
        device = chosen->index;
    }
    const int flags = (vulkan_frames_in_flight << 8) | (shared != nullptr ? vulkan_shared_flag : 0);
    check(engine_.create(&context_, *device, width, height, bit_depth, flags), "Starting Vulkan");
    if (shared != nullptr) {
        try {
            shared_.emplace(engine_, context_, *shared);
        } catch (...) {
            release();
            throw;
        }
    }
    const std::size_t sample = bit_depth <= 8 ? 1 : 2;
    const auto columns = static_cast<std::size_t>(width), rows = static_cast<std::size_t>(height);
    luma_stride_ = static_cast<std::ptrdiff_t>(columns * sample);
    luma_bytes_ = columns * rows * sample;
    frame_bytes_ = luma_bytes_ + 2 * ((columns + 1) / 2) * ((rows + 1) / 2) * sample;
}

VulkanScorer::~VulkanScorer() { release(); }

void VulkanScorer::release() {
    shared_.reset();  // the imported memory goes back before the context that allocated it
    if (context_ != nullptr) {
        engine_.destroy(context_);
        context_ = nullptr;
    }
}

void VulkanScorer::check(int error, const std::string& what) const { check_vulkan(engine_, error, what); }

void VulkanScorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    const bool score = count_ % step_ == 0;
    if (std::min(reference.size(), distorted.size()) < luma_bytes_) {  // the engine reads that much of each
        throw VmafVulkanError("frame " + std::to_string(count_) + " is shorter than its luma plane");
    }
    // Made for every frame: what failed is only put into words when it does.
    if (engine_.submit(context_, reference.data(), luma_stride_, distorted.data(), luma_stride_, score ? 1 : 0) < 0) {
        fail(engine_, "Scoring frame " + std::to_string(count_));
    }
    ++count_;
}

void VulkanScorer::add_decoded(const std::function<void(std::uint8_t*)>& reference,
                               const std::function<void(std::uint8_t*)>& distorted) {
    const bool score = count_ % step_ == 0;
    std::uint8_t* reference_plane = nullptr;
    std::uint8_t* distorted_plane = nullptr;
    std::uint32_t stride = 0;
    if (engine_.staging(context_, &reference_plane, &distorted_plane, &stride) < 0) {
        fail(engine_, "Scoring frame " + std::to_string(count_));
    }
    const auto fill = [&](const std::function<void(std::uint8_t*)>& write, std::uint8_t* plane) {
        if (static_cast<std::ptrdiff_t>(stride) == luma_stride_) {
            write(plane);
            return;
        }
        // A width whose rows the engine pads to four bytes: the plane is
        // written packed, then copied row by row.
        packed_.resize(luma_bytes_);
        write(packed_.data());
        const auto row_bytes = static_cast<std::size_t>(luma_stride_);
        for (std::size_t row = 0; row < luma_bytes_ / row_bytes; ++row) {
            std::copy_n(packed_.data() + row * row_bytes, row_bytes, plane + row * stride);
        }
    };
    fill(reference, reference_plane);
    if (score) {
        fill(distorted, distorted_plane);
    }
    if (engine_.commit(context_, score ? 1 : 0) < 0) {
        fail(engine_, "Scoring frame " + std::to_string(count_));
    }
    ++count_;
}

void VulkanScorer::add_shared(const std::function<void(std::uint64_t, std::int64_t)>& reference,
                              const std::function<void(std::uint64_t, std::int64_t)>& distorted) {
    const bool score = count_ % step_ == 0;
    const SharedSlots::Next next = shared_->next();
    reference(next.reference, next.pitch);
    if (score) {
        distorted(next.distorted, next.pitch);
    }
    if (engine_.commit(context_, score ? 1 : 0) < 0) {
        fail(engine_, "Scoring frame " + std::to_string(count_));
    }
    ++count_;
}

VulkanFeatures VulkanScorer::features() {
    check(engine_.flush(context_), "Scoring");
    VulkanFeatures result;
    for (unsigned frame = 0; frame < count_; frame += step_) {
        result.frames.push_back(frame);
    }
    for (std::vector<double>& column : result.columns) {
        column.resize(result.frames.size());
    }
    for (std::size_t slot = 0; slot < result.frames.size(); ++slot) {
        std::array<double, vulkan_feature_count> row{};
        if (engine_.features(context_, result.frames[slot], row.data()) < 0) {
            fail(engine_, "Reading frame " + std::to_string(result.frames[slot]));
        }
        for (std::size_t feature = 0; feature < vulkan_feature_count; ++feature) {
            result.columns[feature][slot] = row[feature];
        }
    }
    return result;
}

std::vector<std::uint64_t> VulkanScorer::sums(unsigned frame) const {
    std::vector<std::uint64_t> out(128);
    const int count = engine_.sums(context_, frame, out.data(), static_cast<int>(out.size()));
    check(count, "Reading frame " + std::to_string(frame));
    out.resize(std::min(static_cast<std::size_t>(count), out.size()));
    return out;
}

PairScores VulkanScorer::finish() {
    PairScores result;
    if (count_ == 0) {
        for (const auto& [name, version] : models_) {
            result.by_metric.emplace_back(name, std::vector<double>());
        }
        return result;
    }
    VulkanFeatures measured = features();
    for (const auto& [name, version] : models_) {
        std::vector<FeatureValues> given;
        for (const auto& [feature, column] : model_features(is_neg_model(version))) {
            given.emplace_back(feature, measured.columns[column]);
        }
        result.by_metric.emplace_back(name, predict_scores(version, measured.frames, given));
    }
    result.frames = std::move(measured.frames);
    return result;
}

VulkanProbeFrames vulkan_probe_frames(int bits, int count, int width, int height) {
    const std::int64_t peak = (std::int64_t{1} << bits) - 1;
    const bool wide = bits != 8;
    std::vector<std::uint8_t> chroma;
    for (int i = 0; i < 2 * (width / 2) * (height / 2); ++i) {
        append_sample(chroma, peak / 2, wide);
    }
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::int64_t> noise(pixels), reference(pixels), blurred(pixels);
    VulkanProbeFrames frames;
    for (int index = 0; index < count; ++index) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // numpy's 64-bit product wraps around, and its >> keeps the sign.
                const auto mixed = static_cast<std::uint64_t>(std::int64_t{x} * 2654435761LL +
                                                              std::int64_t{y} * 40503LL + std::int64_t{index} * 69069LL);
                const auto product = static_cast<std::int64_t>(mixed * 1103515245ULL);
                const std::int64_t grain = (product >> 16) & 0xFF;
                const std::int64_t ramp = (x * 3 + y * 2 + index * 5) % 256;
                const std::int64_t blocks = ((x / 16 + y / 16 + index) % 2) * 160 + 40;
                const std::size_t at = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                       static_cast<std::size_t>(x);
                noise[at] = grain;
                reference[at] = x < width / 3 ? blocks : x < 2 * width / 3 ? ramp : (ramp + grain) / 2;
            }
        }
        blurred = reference;
        for (int y = 0; y < height; ++y) {
            const std::int64_t* row = reference.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            std::int64_t* out = blurred.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 1; x < width - 1; ++x) {
                out[x] = (row[x - 1] + 2 * row[x] + row[x + 1]) / 4;
            }
        }
        std::vector<std::uint8_t> ref_frame, dis_frame;
        const std::size_t sample = wide ? 2 : 1;
        ref_frame.reserve(pixels * sample + chroma.size());
        dis_frame.reserve(pixels * sample + chroma.size());
        for (std::size_t at = 0; at < pixels; ++at) {
            std::int64_t distorted = 0;
            if (index % 3 == 0) {
                distorted = blurred[at] + noise[at] % 7 - 3;
            } else if (index % 3 == 1) {
                distorted = 2 * reference[at] - blurred[at];  // sharpened: a gain above 1, which VMAF NEG limits
            } else {
                distorted = (blurred[at] * 7 + noise[at]) / 8;
            }
            append_sample(ref_frame, std::clamp<std::int64_t>(reference[at], 0, 255) * peak / 255, wide);
            append_sample(dis_frame, std::clamp<std::int64_t>(distorted, 0, 255) * peak / 255, wide);
        }
        ref_frame.insert(ref_frame.end(), chroma.begin(), chroma.end());
        dis_frame.insert(dis_frame.end(), chroma.begin(), chroma.end());
        frames.reference.push_back(std::move(ref_frame));
        frames.distorted.push_back(std::move(dis_frame));
    }
    return frames;
}

std::string vulkan_probe_sums(int device, int bits) {
    const VulkanProbeFrames frames = vulkan_probe_frames(bits);
    VulkanScorer scorer(probe_width, probe_height, bits, {}, 1, device);
    for (std::size_t i = 0; i < frames.reference.size(); ++i) {
        scorer.add(frames.reference[i], frames.distorted[i]);
    }
    scorer.features();
    Sha256 digest;
    for (unsigned frame = 0; frame < frames.reference.size(); ++frame) {
        // VMAF v0.6.1's sums: the engine keeps VMAF v1's after them.
        const std::vector<std::uint64_t> sums = scorer.sums(frame);
        for (std::size_t i = 0; i < std::min(sums.size(), probe_slots); ++i) {
            std::uint8_t bytes[8];
            for (int b = 0; b < 8; ++b) {
                bytes[b] = static_cast<std::uint8_t>(sums[i] >> (8 * b));  // little-endian, as numpy's "<u8"
            }
            digest.update(bytes, sizeof bytes);
        }
    }
    return digest.hexdigest();
}

VulkanProbe probe_vulkan() {
    std::error_code error;
    if (!std::filesystem::is_regular_file(vulkan_engine_path(), error)) {
        return {false, std::nullopt, platform::shared_library_name("vmaf_vulkan") + " is not bundled"};
    }
    if (!std::filesystem::is_regular_file(libvmaf_folder() / platform::shared_library_name("libvmaf"), error)) {
        return {false, std::nullopt, "libvmaf, which predicts the score, is not bundled"};
    }
    try {
        const std::vector<VulkanDevice> found = vulkan_devices();
        const std::string chosen(python_strip(platform::environment_variable(device_variable).value_or("")));
        std::optional<VulkanDevice> device;
        if (chosen.empty()) {
            device = best_vulkan_device(found);
        } else {
            const auto named = std::find_if(found.begin(), found.end(),
                                            [&](const VulkanDevice& d) { return std::to_string(d.index) == chosen; });
            if (named != found.end()) {
                device = *named;
            }
        }
        if (!device || !device->usable) {
            return {false, std::nullopt, "no GPU that Vulkan can calculate VMAF on"};
        }
        for (std::size_t i = 0; i < probe_digests.size(); ++i) {
            const int bits = i == 0 ? 8 : 10;
            if (vulkan_probe_sums(device->index, bits) != probe_digests[i]) {
                return {false, std::nullopt,
                        device->name + "'s driver calculates VMAF wrongly (" + std::to_string(bits) + "-bit self-test)"};
            }
        }
        return {true, device->index, "Vulkan on " + device->name};
    } catch (const LibvmafError& failure) {
        return {false, std::nullopt, failure.what()};
    }
}

}  // namespace vml
