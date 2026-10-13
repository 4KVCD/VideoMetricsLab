#include "vml/gpu_vmaf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <thread>

#include "vml/frame_sync.h"
#include "vml/gpu_frames.h"
#include "vml/isolated.h"
#include "vml/libvmaf_cpu.h"
#include "vml/number_format.h"
#include "vml/platform/pipe.h"
#include "vml/platform/system.h"
#include "vml/vmaf_v1_gpu.h"

namespace vml {

namespace {

// The app's built-in VMAF models, as libvmaf names them. A custom model file
// is calculated on the CPU.
const std::map<std::string, std::string, std::less<>> built_in_gpu_models{
    {"version=vmaf_v0.6.1", "vmaf_v0.6.1"},
    {"version=vmaf_4k_v0.6.1", "vmaf_4k_v0.6.1"},
};
constexpr const char* neg_model = "vmaf_v0.6.1neg";

// How many frames each reader may hold ready before the feeder takes them.
constexpr std::size_t read_ahead = 3;
// How much each named pipe holds: a 4K frame at 16 bits is 25 MB.
constexpr std::size_t pipe_bytes = 64 * 1024 * 1024;

constexpr unsigned nvidia = 0x10DE;  // its PCI vendor ID


// A queue of frame buffers between two threads. nullptr in it says that no
// more will come.
class BufferQueue {
public:
    void put(std::vector<std::uint8_t>* buffer) {
        {
            const std::lock_guard lock(mutex_);
            items_.push_back(buffer);
        }
        ready_.notify_one();
    }

    std::vector<std::uint8_t>* get() {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return !items_.empty(); });
        std::vector<std::uint8_t>* buffer = items_.front();
        items_.pop_front();
        return buffer;
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::vector<std::uint8_t>*> items_;
};

// Fills `size` bytes at `into` from the pipe. False at the stream's end
// before the first byte; throws LibvmafError when it ends inside them.
bool read_exactly(platform::InboundPipe& pipe, std::uint8_t* into, std::size_t size) {
    std::size_t filled = 0;
    while (filled < size) {
        const std::size_t got = pipe.read(into + filled, size - filled);
        if (got == 0) {
            if (filled != 0) {
                throw LibvmafError("FFmpeg's frames ended inside one (" + std::to_string(filled) + " of " +
                                   std::to_string(size) + " bytes)");
            }
            return false;
        }
        filled += got;
    }
    return true;
}

// An exception's message.
std::string message_of(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& caught) {
        return caught.what();
    } catch (...) {
        return "an unknown error";
    }
}

struct Probe {
    std::mutex mutex;
    std::string preference = "auto";
    std::optional<GpuVmafBackend> found;
};

Probe& probe_state() {
    static Probe probe;
    return probe;
}

// The backends to try, in order, for a GPU backend setting.
std::vector<std::string> backend_order(const std::string& preference) {
    return preference == "cuda" ? std::vector<std::string>{"cuda", "vulkan"} : std::vector<std::string>{"vulkan", "cuda"};
}

// The Vulkan probe, in a process of its own where a program to run it is
// set (the Python app's run_isolated(vmaf_vulkan.probe)): a driver that
// crashes on the probe ends that process, and the GPU is taken as unable.
VulkanProbe probe_vulkan_apart() {
    if (!isolation_program()) {
        return probe_vulkan();
    }
    try {
        const nlohmann::json result =
            run_isolated("vulkan-probe", "the Vulkan VMAF probe", nlohmann::json::object(), {}, CancelToken());
        VulkanProbe probe;
        result.at("available").get_to(probe.available);
        if (!result.at("device").is_null()) {
            probe.device = result.at("device").get<int>();
        }
        result.at("text").get_to(probe.text);
        return probe;
    } catch (const IsolatedCrashError& error) {
        return {false, std::nullopt, error.what()};
    }
}

GpuVmafBackend probe_backends(const std::string& preference) {
    std::string reasons;
    const auto add_reason = [&reasons](const std::string& reason) {
        reasons += (reasons.empty() ? "" : "; ") + reason;
    };
    for (const std::string& backend : backend_order(preference)) {
        // Any failure means "not on this GPU": one that escaped would fail
        // every video's setup instead of VMAF being calculated on the CPU.
        try {
            if (backend == "cuda") {
                const std::vector<unsigned> vendors = platform::gpu_vendor_ids();
                if (std::find(vendors.begin(), vendors.end(), nvidia) == vendors.end()) {
                    add_reason("no NVIDIA GPU");
                    continue;
                }
                return {"cuda", std::nullopt, "libvmaf's CUDA code, which the C++ app does not have yet"};
            }
            const VulkanProbe probe = probe_vulkan_apart();
            if (probe.available) {
                return {"vulkan", probe.device, probe.text};
            }
            add_reason("Vulkan: " + probe.text);
        } catch (const std::exception& error) {
            add_reason(backend + ": the GPU probe failed: " + error.what());
        }
    }
    return {"", std::nullopt, reasons};
}

}  // namespace

GpuModels gpu_models(bool compute_vmaf, bool compute_vmaf_neg, const std::string& model, bool compute_vmaf_v1,
                     const std::string& model_v1) {
    GpuModels models;
    if (compute_vmaf || compute_vmaf_neg) {
        std::optional<std::string> version("");
        if (compute_vmaf) {
            const auto built_in = built_in_gpu_models.find(model);
            version = built_in != built_in_gpu_models.end() ? std::optional(built_in->second) : std::nullopt;
        }
        if (version) {
            if (compute_vmaf) {
                models.emplace_back("vmaf", *version);
            }
            if (compute_vmaf_neg) {
                models.emplace_back("vmaf_neg", neg_model);
            }
        }
    }
    if (compute_vmaf_v1) {
        models.emplace_back("vmaf_v1", model_v1);
    }
    return models;
}

void set_gpu_backend(const std::string& preference) {
    Probe& probe = probe_state();
    const std::lock_guard lock(probe.mutex);
    if (backend_order(preference) != backend_order(probe.preference)) {
        probe.found.reset();
    }
    probe.preference = preference;
}

GpuVmafBackend gpu_vmaf_backend() {
    Probe& probe = probe_state();
    const std::lock_guard lock(probe.mutex);
    if (!probe.found) {
        probe.found = probe_backends(probe.preference);
    }
    return *probe.found;
}

GpuModels scores_on_gpu(bool compute_vmaf, bool compute_vmaf_neg, const std::string& model, bool enabled,
                        int bit_depth, std::optional<std::pair<int, int>> size, bool compute_vmaf_v1,
                        const std::string& model_v1) {
    // Deeper than 10 bits, or at an odd size, the frame pairs could not
    // cross FFmpeg's overlay, which the Python app can pair them with.
    if (!enabled || bit_depth > 10 || (size && ((size->first & 1) != 0 || (size->second & 1) != 0))) {
        return {};
    }
    GpuModels models = gpu_models(compute_vmaf, compute_vmaf_neg, model, compute_vmaf_v1, model_v1);
    const auto is_v1 = [](const auto& entry) { return entry.first == "vmaf_v1"; };
    if (std::any_of(models.begin(), models.end(), is_v1) && !v1_on_gpu(model_v1, size)) {
        std::erase_if(models, is_v1);
    }
    // VMAF v1 runs on Vulkan whatever scores VMAF v0.6.1 and NEG.
    if (!std::all_of(models.begin(), models.end(), is_v1) && gpu_vmaf_backend().backend.empty()) {
        std::erase_if(models, [&](const auto& entry) { return !is_v1(entry); });
    }
    return models;
}

std::timed_mutex& gpu_pass() {
    static std::timed_mutex pass;
    return pass;
}

std::pair<std::vector<std::string>, std::vector<std::string>> gpu_output_args(double duration_limit,
                                                                              const std::string& test_pipe,
                                                                              const std::string& source_pipe) {
    const std::vector<std::string> cut =
        duration_args(duration_limit > 0 ? std::optional(format_fixed(duration_limit, 3)) : std::nullopt);
    const auto output = [&cut](const char* label, const std::string& pipe) {
        std::vector<std::string> args{"-map", "[" + std::string(label) + "]", "-fps_mode", "passthrough"};
        args.insert(args.end(), cut.begin(), cut.end());
        args.insert(args.end(), {"-flush_packets", "1", "-f", "yuv4mpegpipe", "-strict", "-1", pipe});
        return args;
    };
    return {output("vmaf_dist", test_pipe), output("vmaf_ref", source_pipe)};
}

// ------------------------------------------------------------- StreamReader

struct StreamReader::State {
    State(std::string_view name, std::size_t frame_bytes)
        : pipe(name, pipe_bytes), frame_bytes(frame_bytes), buffers(read_ahead + 1) {
        for (std::vector<std::uint8_t>& buffer : buffers) {
            buffer.resize(frame_bytes);
            free.put(&buffer);
        }
    }

    platform::InboundPipe pipe;
    std::size_t frame_bytes;
    // The frame being paired and those read ahead.
    std::vector<std::vector<std::uint8_t>> buffers;
    BufferQueue free;    // buffers to read into; nullptr: stop reading frames
    BufferQueue pixels;  // frames read; nullptr: the end
    std::atomic<bool> stopped{false};
    std::atomic<bool> writer_gone{false};
    std::atomic<bool> connected{false};  // FFmpeg opened the pipe (or ffmpeg_ended did)
    std::atomic<bool> finished{false};   // the reading thread has ended
    std::exception_ptr error;            // set before the final nullptr in pixels
    std::thread thread;
};

StreamReader::StreamReader(std::string_view name, std::size_t frame_bytes)
    : state_(std::make_unique<State>(name, frame_bytes)) {}

StreamReader::~StreamReader() {
    if (state_->thread.joinable()) {
        stop();
        state_->thread.join();
    }
}

const std::string& StreamReader::path() const { return state_->pipe.path(); }

void StreamReader::start() {
    state_->thread = std::thread([this] { read_frames(); });
}

void StreamReader::read_frames() {
    State& s = *state_;
    try {
        s.pipe.connect();
        s.connected = true;
        if (!s.stopped) {
            // The stream is a header line, "YUV4MPEG2 W.. H.. ...\n", then
            // for each frame a "FRAME\n" line and the frame's planes, packed.
            std::string header;
            std::uint8_t byte = 0;
            while (byte != '\n' && header.size() < 1024 && read_exactly(s.pipe, &byte, 1)) {
                header += static_cast<char>(byte);
            }
            if (!header.empty() && !header.starts_with("YUV4MPEG2 ")) {
                throw LibvmafError("FFmpeg's frames are not a YUV4MPEG stream");
            }
            std::array<std::uint8_t, 6> marker{};
            while (!header.empty() && !s.stopped) {
                // Each frame is left to gather in the pipe and taken in one
                // read: read as it arrived, a 4K frame would come in hundreds
                // of pieces.
                s.pipe.gather(marker.size() + s.frame_bytes, s.stopped, s.writer_gone);
                if (!read_exactly(s.pipe, marker.data(), marker.size())) {
                    break;
                }
                if (std::memcmp(marker.data(), "FRAME\n", marker.size()) != 0) {
                    throw LibvmafError("FFmpeg's frames are not where they were expected");
                }
                std::vector<std::uint8_t>* buffer = s.free.get();
                if (buffer == nullptr) {
                    break;
                }
                if (!read_exactly(s.pipe, buffer->data(), buffer->size())) {
                    throw LibvmafError("FFmpeg's frames ended after a frame's start");
                }
                s.pixels.put(buffer);
            }
            // The comparison may end before this video does: what FFmpeg
            // still writes is read and dropped, so that it can finish.
            std::vector<std::uint8_t> spare(4 * 1024 * 1024);
            while (!s.stopped) {
                s.pipe.gather(spare.size(), s.stopped, s.writer_gone);
                try {
                    if (!read_exactly(s.pipe, spare.data(), spare.size())) {
                        break;
                    }
                } catch (const LibvmafError&) {
                    break;  // the tail, shorter than the buffer
                }
            }
        }
    } catch (...) {
        s.error = std::current_exception();
    }
    s.pipe.close();
    s.finished = true;
    s.pixels.put(nullptr);
}

std::vector<std::uint8_t>* StreamReader::pull() {
    State& s = *state_;
    std::vector<std::uint8_t>* frame = s.pixels.get();
    if (frame == nullptr) {
        s.pixels.put(nullptr);  // the end, for the next pull too
        if (s.error) {
            std::rethrow_exception(s.error);
        }
    }
    return frame;
}

void StreamReader::release(std::vector<std::uint8_t>* frame) { state_->free.put(frame); }

void StreamReader::drop_rest() { state_->free.put(nullptr); }

void StreamReader::ffmpeg_ended() {
    State& s = *state_;
    s.writer_gone = true;
    // A named pipe can be opened here at once. A FIFO can be only once the
    // reading thread waits for its writer, which it does a moment after it
    // starts.
    for (int attempt = 0; attempt < 200 && s.thread.joinable() && !s.connected && !s.finished; ++attempt) {
        platform::InboundPipe::touch(s.pipe.path());
        if (s.connected || s.finished) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void StreamReader::stop() {
    state_->stopped = true;
    state_->free.put(nullptr);
    ffmpeg_ended();
}

void StreamReader::join() {
    if (state_->thread.joinable()) {
        state_->thread.join();
    }
    if (state_->error) {
        std::rethrow_exception(state_->error);
    }
}

// --------------------------------------------------------------- GpuAttempt

struct GpuAttempt::State {
    std::unique_ptr<PairScorer> scorer;
    std::unique_ptr<StreamReader> distorted;
    std::unique_ptr<StreamReader> reference;
    std::thread feeder;
    std::exception_ptr error;  // the feeder's, set before it ends
    std::optional<PairScores> result;
};

GpuAttempt::GpuAttempt(const GpuPlan& plan, int n_subsample) : state_(std::make_unique<State>()) {
    State& s = *state_;
    if (plan.backend == "cpu") {
        std::vector<std::string> metrics;
        for (const auto& [key, same] : plan.models) {
            metrics.push_back(key);
        }
        s.scorer = std::make_unique<CpuScorer>(plan.width, plan.height, plan.bit_depth, metrics, n_subsample,
                                               plan.threads, plan.frame_rate);
    } else if (std::any_of(plan.models.begin(), plan.models.end(),
                           [](const auto& model) { return model.first == "vmaf_v1"; })) {
        s.scorer = std::make_unique<MultiScorer>(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample,
                                                 plan.device >= 0 ? std::optional(plan.device) : std::nullopt);
    } else {
        s.scorer = std::make_unique<VulkanScorer>(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample,
                                                  plan.device);
    }
    s.distorted = std::make_unique<StreamReader>("distorted", s.scorer->frame_bytes());
    s.reference = std::make_unique<StreamReader>("reference", s.scorer->frame_bytes());
    s.distorted->start();
    s.reference->start();
    s.feeder = std::thread([this] { feed(); });
}

GpuAttempt::~GpuAttempt() {
    if (state_->feeder.joinable()) {
        stop();
        state_->feeder.join();
    }
}

std::pair<std::vector<std::string>, std::vector<std::string>> GpuAttempt::output_args(double duration_limit) const {
    return gpu_output_args(duration_limit, state_->distorted->path(), state_->reference->path());
}

void GpuAttempt::feed() {
    State& s = *state_;
    try {
        StreamReader& test = *s.distorted;
        StreamReader& source = *s.reference;
        // Frame N of the test video with frame N of the source, until either
        // video ends (frame_sync.frame_pairs).
        while (true) {
            std::vector<std::uint8_t>* test_frame = test.pull();
            if (test_frame == nullptr) {
                break;
            }
            std::vector<std::uint8_t>* source_frame = source.pull();
            if (source_frame == nullptr) {
                test.release(test_frame);
                break;
            }
            try {
                s.scorer->add(*source_frame, *test_frame);
            } catch (...) {
                test.release(test_frame);
                source.release(source_frame);
                throw;
            }
            test.release(test_frame);
            source.release(source_frame);
        }
        test.drop_rest();
        source.drop_rest();
        test.join();
        source.join();
        s.result = s.scorer->finish();
    } catch (...) {
        s.error = std::current_exception();
        stop();
    }
}

void GpuAttempt::stop() {
    state_->distorted->stop();
    state_->reference->stop();
}

std::optional<PairScores> GpuAttempt::finish(bool ffmpeg_succeeded) {
    State& s = *state_;
    if (!ffmpeg_succeeded) {
        stop();
    } else {
        s.distorted->ffmpeg_ended();
        s.reference->ffmpeg_ended();
    }
    if (s.feeder.joinable()) {
        s.feeder.join();
    }
    s.scorer.reset();  // its GPU memory, before anything else runs
    if (s.error) {
        throw LibvmafError("GPU VMAF failed: " + message_of(s.error));
    }
    return ffmpeg_succeeded ? std::move(s.result) : std::nullopt;
}

PairScores score_decoded(const VideoInfo& source, const VideoInfo& distorted,
                         const std::optional<CropBox>& source_crop,
                         const std::optional<CropBox>& distorted_crop, const GpuPlan& plan,
                         int n_subsample, const std::optional<std::string>& duration_limit,
                         long long total_frames, const std::string& scale_algorithm,
                         const std::pair<std::string, std::string>& decoders, const std::string& ffmpeg,
                         const std::string& ffprobe,
                         const std::function<void(long long, long long, double)>& on_progress,
                         const CancelToken& cancel) {
    const std::vector<VulkanDevice> found = vulkan_devices();
    const auto chosen =
        std::find_if(found.begin(), found.end(), [&](const VulkanDevice& device) { return device.index == plan.device; });
    if (chosen == found.end()) {
        throw VmafVulkanError("no GPU that Vulkan can calculate VMAF on");
    }
    const auto& [source_backend, test_backend] = decoders;
    for (const std::string& library : {source_backend, test_backend}) {
        if (library.empty() || !decoder_available(library)) {
            throw GpuDecodeUnavailableError("no decoder library for " + chosen->name);
        }
    }
    // Each video's plan, the test video's first: its luma alone, at the size
    // and depth compared at, 10-bit samples in the low bits as FFmpeg's
    // yuv420p10le has them.
    std::vector<DecodePlan> plans;
    for (const bool test_video : {true, false}) {
        const VideoInfo& info = test_video ? distorted : source;
        DecodePlan decoded = plan_decode(info, test_video ? distorted_crop : source_crop, 6, true,
                                         std::pair(plan.width, plan.height), scale_algorithm);
        widen_to(decoded, info, plan.bit_depth);
        const auto [supported, refusal] = decoder_supports(0, decoded, test_video ? test_backend : source_backend);
        if (!supported) {
            throw GpuDecodeUnavailableError(refusal);
        }
        plans.push_back(decoded);
    }
    // The decoders write each luma plane into memory of the size compared at.
    const std::size_t luma_bytes = static_cast<std::size_t>(plan.width) * static_cast<std::size_t>(plan.height) *
                                   (plan.bit_depth > 8 ? 2 : 1);
    if (std::any_of(plans.begin(), plans.end(), [&](const DecodePlan& p) { return p.frame_bytes() != luma_bytes; })) {
        throw GpuDecodeUnavailableError("the decoders' frames are not the size compared at");
    }
    GpuFrameStream test(distorted, plans[0], 0, 4, test_backend, ffmpeg, ffprobe);
    GpuFrameStream reference(source, plans[1], 0, 4, source_backend, ffmpeg, ffprobe);
    // Declared after the streams so that it goes first: it holds on to the
    // decoder's memory.
    std::optional<VulkanScorer> scorer;
    // The planes stay on the GPU when the decoder and Vulkan share memory,
    // which needs NVIDIA's decoder for both videos.
    bool shared = false;
    if (source_backend == "nvidia" && test_backend == "nvidia") {
        try {
            scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample, plan.device, &test);
            shared = true;
        } catch (const VmafVulkanError&) {
            // The frames go through system memory instead.
        }
    }
    if (!scorer) {
        scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample, plan.device);
    }
    test.start();
    reference.start();
    score_frame_pairs(
        test, reference, distorted, duration_limit, total_frames,
        [&](int test_slot, int reference_slot) {
            if (shared) {
                scorer->add_shared(
                    [&](std::uint64_t address, std::int64_t pitch) {
                        reference.copy_luma(reference_slot, address, pitch);
                    },
                    [&](std::uint64_t address, std::int64_t pitch) { test.copy_luma(test_slot, address, pitch); });
            } else {
                scorer->add_decoded([&](std::uint8_t* plane) { reference.download(reference_slot, plane); },
                                    [&](std::uint8_t* plane) { test.download(test_slot, plane); });
            }
        },
        on_progress, cancel);
    return scorer->finish();
}

}  // namespace vml
