#include "vml/gpu_vmaf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

#include "vml/app_log.h"
#include "vml/frame_sync.h"
#include "vml/gpu_frames.h"
#include "vml/host_frames.h"
#include "vml/isolated.h"
#include "vml/libvmaf_cpu.h"
#include "vml/number_format.h"
#include "vml/platform/pipe.h"
#include "vml/platform/system.h"
#include "vml/threads.h"
#include "vml/vmaf_cuda.h"
#include "vml/vmaf_v1_gpu.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.vmaf_gpu");

// The app's built-in VMAF models, as libvmaf names them. A custom model file
// is calculated on the CPU.
const std::map<std::string, std::string, std::less<>> built_in_gpu_models{
    {"version=vmaf_v0.6.1", "vmaf_v0.6.1"},
    {"version=vmaf_4k_v0.6.1", "vmaf_4k_v0.6.1"},
};
constexpr const char* neg_model = "vmaf_v0.6.1neg";

// How many frames each reader may hold ready before the feeder takes them.
constexpr std::size_t read_ahead = 3;

// A queue of frame buffers between two threads. nullptr in it says that no
// more will come.
using BufferQueue = Channel<std::vector<std::uint8_t>*>;

// A video's frames that end inside one: its FFmpeg stopped writing mid-frame,
// as one that fails does, or one ended because the other failed
// (run_ffmpeg_pair). FFmpeg's failure, then, not the scoring's.
class FramesCutError : public LibvmafError {
public:
    using LibvmafError::LibvmafError;
};

bool frames_cut(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const FramesCutError&) {
        return true;
    } catch (...) {
        return false;
    }
}

// Fills `size` bytes at `into` from the pipe, in a few large reads
// (read_full). False at the stream's end before the first byte; throws
// FramesCutError when it ends inside them.
bool read_exactly(platform::InboundPipe& pipe, std::uint8_t* into, std::size_t size, const std::atomic<bool>& stop,
                  const std::atomic<bool>& writer_gone) {
    const std::size_t filled = pipe.read_full(into, size, stop, writer_gone);
    if (filled == size) {
        return true;
    }
    if (filled != 0) {
        throw FramesCutError("FFmpeg's frames ended inside one (" + std::to_string(filled) + " of " +
                             std::to_string(size) + " bytes)");
    }
    return false;
}

struct Probe {
    std::mutex mutex;
    std::optional<GpuVmafBackend> found;
    std::chrono::steady_clock::time_point found_at;
};

// How long a failed probe is kept before the window's next run asks again
// (stale_failure).
constexpr std::chrono::seconds failed_probe_retry{60};

Probe& probe_state() {
    static Probe probe;
    return probe;
}

// The backends to try, in order: libvmaf-fast's Vulkan engine, the faster,
// and libvmaf's CUDA code for an NVIDIA GPU whose Vulkan cannot score VMAF.
const std::vector<std::string> backend_order{"vulkan", "cuda"};

// The Vulkan probe, in a process of its own where a program to run it is
// set (the Python app's run_isolated(vmaf_vulkan.probe)): a driver that
// crashes on the probe ends that process, and the GPU is taken as unable,
// with `crashed` set.
VulkanProbe probe_vulkan_apart(bool& crashed, const CancelToken& cancel) {
    if (!isolation_program()) {
        return probe_vulkan();
    }
    try {
        const nlohmann::json result =
            run_isolated("vulkan-probe", "the Vulkan VMAF probe", nlohmann::json::object(), {}, cancel);
        VulkanProbe probe;
        result.at("available").get_to(probe.available);
        if (!result.at("device").is_null()) {
            probe.device = result.at("device").get<int>();
        }
        result.at("text").get_to(probe.text);
        probe.failed = result.value("failed", false);
        return probe;
    } catch (const IsolatedCrashError& error) {
        crashed = true;
        return {false, std::nullopt, error.what()};
    }
}

// libvmaf's CUDA probe, in a process of its own where a program to run it is
// set (the Python app's run_isolated(probe)); `crashed` as for Vulkan's.
CudaProbe probe_cuda_apart(bool& crashed, const CancelToken& cancel) {
    if (!isolation_program()) {
        return probe_cuda();
    }
    try {
        const nlohmann::json result =
            run_isolated("cuda-probe", "libvmaf's GPU probe", nlohmann::json::object(), {}, cancel);
        return {result.at("available").get<bool>(), result.at("text").get<std::string>()};
    } catch (const IsolatedCrashError& error) {
        crashed = true;
        return {false, error.what()};
    }
}

GpuVmafBackend probe_backends(const CancelToken& cancel) {
    std::string reasons;
    bool failed = false;  // a probe crashed, threw or scored wrongly
    const auto add_reason = [&reasons](const std::string& reason) {
        reasons += (reasons.empty() ? "" : "; ") + reason;
    };
    for (const std::string& backend : backend_order) {
        // Any failure means "not on this GPU": one that escaped would fail
        // every video's setup instead of VMAF being calculated on the CPU.
        try {
            if (backend == "cuda") {
                const std::vector<unsigned> vendors = platform::gpu_vendor_ids();
                if (std::find(vendors.begin(), vendors.end(), platform::nvidia_pci_vendor) == vendors.end()) {
                    add_reason("no NVIDIA GPU");
                    continue;
                }
                const CudaProbe probe = probe_cuda_apart(failed, cancel);
                if (probe.available) {
                    return {"cuda", std::nullopt, probe.text + " with CUDA"};
                }
                add_reason("CUDA: " + probe.text);
                continue;
            }
            const VulkanProbe probe = probe_vulkan_apart(failed, cancel);
            if (probe.available) {
                return {"vulkan", probe.device, probe.text};
            }
            add_reason("Vulkan: " + probe.text);
            failed = failed || probe.failed;
        } catch (const Cancelled&) {
            throw;  // the run's Cancel, or the app ending (stop_gpu_probes): no answer, nothing kept
        } catch (const std::exception& error) {
            logger.warning("The " + backend + " VMAF probe failed: " + error.what());
            add_reason(backend + ": the GPU probe failed: " + error.what());
            failed = true;
        }
    }
    return {"", std::nullopt, reasons, failed};
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

GpuVmafBackend gpu_vmaf_backend(const CancelToken& cancel) {
    Probe& probe = probe_state();
    const std::lock_guard lock(probe.mutex);
    if (!probe.found) {
        probe.found = probe_backends(cancel);
        probe.found_at = std::chrono::steady_clock::now();
        logger.info((probe.found->backend.empty() ? "VMAF on the GPU unavailable: " : "VMAF on the GPU: ") +
                    probe.found->text);
    }
    return *probe.found;
}

bool stale_failure(bool on_cpu, std::chrono::steady_clock::duration age) {
    return on_cpu && age >= failed_probe_retry;
}

bool gpu_vmaf_probe_ready() {
    Probe& probe = probe_state();
    const std::unique_lock lock(probe.mutex, std::try_to_lock);
    return lock.owns_lock() && probe.found.has_value() &&
           !stale_failure(probe.found->backend.empty(), std::chrono::steady_clock::now() - probe.found_at);
}

void forget_failed_gpu_vmaf_probe() {
    Probe& probe = probe_state();
    const std::lock_guard lock(probe.mutex);
    if (probe.found && stale_failure(probe.found->backend.empty(), std::chrono::steady_clock::now() - probe.found_at)) {
        probe.found.reset();
    }
}

GpuModels scores_on_gpu(bool compute_vmaf, bool compute_vmaf_neg, const std::string& model, bool enabled,
                        int bit_depth, std::optional<std::pair<int, int>> size, const CancelToken& cancel,
                        bool compute_vmaf_v1, const std::string& model_v1,
                        std::vector<std::pair<std::string, std::string>>* check_failures) {
    // Deeper than 10 bits, or at an odd size, the frame pairs could not
    // cross FFmpeg's overlay, which the Python app can pair them with.
    if (!enabled || bit_depth > 10 || (size && ((size->first & 1) != 0 || (size->second & 1) != 0))) {
        return {};
    }
    GpuModels models = gpu_models(compute_vmaf, compute_vmaf_neg, model, compute_vmaf_v1, model_v1);
    const auto is_v1 = [](const auto& entry) { return entry.first == "vmaf_v1"; };
    const auto left_out = [check_failures](const std::string& key, const std::string& why) {
        if (check_failures != nullptr) {
            check_failures->emplace_back(key, why);
        }
    };
    std::string v1_failure;
    if (std::any_of(models.begin(), models.end(), is_v1) && !v1_on_gpu(model_v1, size, cancel, &v1_failure)) {
        std::erase_if(models, is_v1);
        if (!v1_failure.empty()) {
            left_out("vmaf_v1", v1_failure);
        }
    }
    // VMAF v1 runs on Vulkan whatever scores VMAF v0.6.1 and NEG.
    if (!std::all_of(models.begin(), models.end(), is_v1)) {
        if (const GpuVmafBackend backend = gpu_vmaf_backend(cancel); backend.backend.empty()) {
            for (const auto& entry : models) {
                if (backend.check_failed && !is_v1(entry)) {
                    left_out(entry.first, backend.text);
                }
            }
            std::erase_if(models, [&](const auto& entry) { return !is_v1(entry); });
        }
    }
    return models;
}

void start_beside(std::optional<CpuScorer>& scorer, const GpuPlan& plan, int n_subsample) {
    if (!plan.beside.empty()) {
        scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.beside, n_subsample, plan.threads,
                       plan.frame_rate);
    }
}

void add_beside(PairScores& scores, std::optional<CpuScorer>& beside) {
    if (!beside) {
        return;
    }
    PairScores scored = beside->finish();
    if (scored.frames != scores.frames) {
        throw LibvmafError("the CPU's metrics were scored for other frames than the GPU's (" +
                           std::to_string(scored.frames.size()) + " against " + std::to_string(scores.frames.size()) +
                           ")");
    }
    for (auto& metric : scored.by_metric) {
        scores.by_metric.push_back(std::move(metric));
    }
}

std::timed_mutex& gpu_pass() {
    static std::timed_mutex pass;
    return pass;
}

std::unique_lock<std::timed_mutex> take_gpu_turn(const std::function<void()>& on_wait, const CancelToken& cancel) {
    std::unique_lock pass(gpu_pass(), std::defer_lock);
    if (pass.try_lock()) {
        return pass;
    }
    if (on_wait) {
        on_wait();
    }
    while (!pass.try_lock_for(std::chrono::milliseconds(100))) {
        cancel.throw_if_cancelled();
    }
    return pass;
}

std::pair<std::vector<std::string>, std::vector<std::string>> gpu_output_args(const std::string& test_pipe,
                                                                              const std::string& source_pipe) {
    const auto output = [](const char* label, const std::string& pipe) {
        return std::vector<std::string>{"-map", "[" + std::string(label) + "]", "-fps_mode", "passthrough",
                                        "-flush_packets", "1", "-f", "yuv4mpegpipe", "-strict", "-1", pipe};
    };
    return {output("vmaf_dist", test_pipe), output("vmaf_ref", source_pipe)};
}

// ------------------------------------------------------------- StreamReader

struct StreamReader::State {
    State(std::string_view name, std::size_t frame_bytes)
        : name(name), pipe(name, platform::frame_pipe_bytes), frame_bytes(frame_bytes), buffers(read_ahead + 1) {
        for (std::vector<std::uint8_t>& buffer : buffers) {
            buffer.resize(frame_bytes);
            free.put(&buffer);
        }
    }

    std::string name;  // what the pipe's path ends in: "distorted", "reference"
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
    std::atomic<bool> stopped_while_reading{false};
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

void StreamReader::start() { state_->thread = std::thread([this] { read_frames(); }); }

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
            while (byte != '\n' && header.size() < 1024 && s.pipe.read(&byte, 1) == 1) {
                header += static_cast<char>(byte);
            }
            if (!header.empty() && !header.starts_with("YUV4MPEG2 ")) {
                throw LibvmafError("FFmpeg's frames are not a YUV4MPEG stream");
            }
            std::array<std::uint8_t, 6> marker{};
            while (!header.empty() && !s.stopped) {
                if (!read_exactly(s.pipe, marker.data(), marker.size(), s.stopped, s.writer_gone)) {
                    break;
                }
                if (std::memcmp(marker.data(), "FRAME\n", marker.size()) != 0) {
                    throw LibvmafError("FFmpeg's frames are not where they were expected");
                }
                std::vector<std::uint8_t>* buffer = s.free.get();
                if (buffer == nullptr) {
                    break;
                }
                if (!read_exactly(s.pipe, buffer->data(), buffer->size(), s.stopped, s.writer_gone)) {
                    throw FramesCutError("FFmpeg's frames ended after a frame's start");
                }
                s.pixels.put(buffer);
            }
            // The comparison may end before this video does: what FFmpeg
            // still writes is read and dropped, so that it can finish.
            std::vector<std::uint8_t> spare(4 * 1024 * 1024);
            while (!s.stopped) {
                try {
                    if (!read_exactly(s.pipe, spare.data(), spare.size(), s.stopped, s.writer_gone)) {
                        break;
                    }
                } catch (const LibvmafError&) {
                    break;  // the tail, shorter than the buffer
                }
            }
        }
    } catch (const FramesCutError& error) {
        // Which video's FFmpeg stopped mid-frame, for the log: the run
        // reports the FFmpeg that failed.
        logger.info("The " + s.name + " pipe: " + error.what());
        s.error = std::current_exception();
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
    if (!state_->finished) {
        state_->stopped_while_reading = true;
    }
    state_->stopped = true;
    state_->free.put(nullptr);
    ffmpeg_ended();
}

bool StreamReader::stopped_while_reading() const { return state_->stopped_while_reading; }

void StreamReader::join() {
    State& s = *state_;
    if (s.thread.joinable()) {
        s.thread.join();
    }
    if (s.error) {
        std::rethrow_exception(s.error);
    }
}

// --------------------------------------------------------------- GpuAttempt

struct GpuAttempt::State {
    std::unique_ptr<PairScorer> scorer;
    std::optional<CpuScorer> beside;  // the plan's metrics beside the GPU's
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
                                                 plan.backend,
                                                 plan.device >= 0 ? std::optional(plan.device) : std::nullopt);
    } else if (plan.backend == "cuda") {
        s.scorer = std::make_unique<CudaScorer>(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample);
    } else {
        s.scorer = std::make_unique<VulkanScorer>(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample,
                                                  plan.device);
    }
    start_beside(s.beside, plan, n_subsample);
    // Both read the frames FFmpeg writes: whole frames, luma first.
    if (s.beside && s.beside->frame_bytes() != s.scorer->frame_bytes()) {
        throw LibvmafError("the CPU's metrics take frames of another size than the GPU's");
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

std::pair<std::vector<std::string>, std::vector<std::string>> GpuAttempt::output_args() const {
    const State& s = *state_;
    return gpu_output_args(s.distorted->path(), s.reference->path());
}

// Frame N of the test video with frame N of the source, until either video
// ends (frame_sync.frame_pairs).
void GpuAttempt::feed_pairs() {
    State& s = *state_;
    StreamReader& test = *s.distorted;
    StreamReader& source = *s.reference;
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
            if (s.beside) {
                s.beside->add(*source_frame, *test_frame);
            }
            s.scorer->add(*source_frame, *test_frame);
        } catch (...) {
            test.release(test_frame);
            source.release(source_frame);
            throw;
        }
        test.release(test_frame);
        source.release(source_frame);
    }
}

void GpuAttempt::feed() {
    State& s = *state_;
    try {
        StreamReader& test = *s.distorted;
        StreamReader& source = *s.reference;
        feed_pairs();
        test.drop_rest();
        source.drop_rest();
        test.join();
        source.join();
        s.result = s.scorer->finish();
        add_beside(*s.result, s.beside);
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
    s.beside.reset();
    // Frames cut short where FFmpeg failed are that failure, which the
    // caller reports and tries another decode plan for.
    if (s.error && !(frames_cut(s.error) && !ffmpeg_succeeded)) {
        throw LibvmafError("GPU VMAF failed: " + message_of(s.error));
    }
    return ffmpeg_succeeded ? std::move(s.result) : std::nullopt;
}

bool GpuAttempt::closed_pipe_of(std::size_t output) const {
    return (output == 0 ? state_->distorted : state_->reference)->stopped_while_reading();
}

namespace {

// score_decoded with libvmaf's CUDA code (the Python app's
// vmaf_gpu.score_decoded), for the decoders' `plans` (the test video's
// first): NVIDIA's pictures are copied on the GPU into libvmaf's, and the
// software decoder's are written into the memory they are uploaded from.
PairScores score_decoded_cuda(const VideoInfo& source, const VideoInfo& distorted,
                              const std::vector<DecodePlan>& plans, const GpuPlan& plan, int n_subsample,
                              const FrameRange& range, long long total_frames,
                              const std::pair<std::string, std::string>& decoders, const std::string& ffmpeg,
                              const std::string& ffprobe,
                              const std::function<void(long long, long long, double)>& on_progress,
                              const CancelToken& cancel) {
    const auto& [source_backend, test_backend] = decoders;
    // Declared before the decoders, so that they close first and their
    // threads stop before libvmaf closes; made after them. Both are the
    // Python app's order: NVIDIA's decoder sets the GPU's CUDA context, in
    // which libvmaf works too, to sleep while it waits rather than spin.
    std::optional<CudaScorer> scorer;
    GpuFrameStream test(distorted, plans[0], 0, 4, test_backend, ffmpeg, ffprobe);
    GpuFrameStream reference(source, plans[1], 0, 4, source_backend, ffmpeg, ffprobe);
    scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample,
                   source_backend == "nvidia" && test_backend == "nvidia");
    // A decoded picture as the scorer takes it.
    const auto side = [](GpuFrameStream& stream, int slot) {
        CudaScorer::Decoded decoded;
        if (stream.backend() == "nvidia") {  // copied on the GPU
            decoded.copy = [&stream, slot](std::uint64_t address, std::int64_t pitch) {
                stream.copy_luma(slot, address, pitch);
            };
        } else {  // into the upload's memory
            decoded.write = [&stream, slot](std::uint8_t* plane) { stream.download(slot, plane); };
        }
        return decoded;
    };
    test.start();
    reference.start();
    score_frame_pairs(
        test, reference, range, total_frames,
        [&](int test_slot, int reference_slot) {
            scorer->add_decoded(side(reference, reference_slot), side(test, test_slot));
        },
        on_progress, cancel);
    return scorer->finish();
}

}  // namespace

PairScores score_decoded(const VideoInfo& source, const VideoInfo& distorted,
                         const std::optional<CropBox>& source_crop,
                         const std::optional<CropBox>& distorted_crop, const GpuPlan& plan,
                         int n_subsample, const FrameRange& range,
                         long long total_frames, const std::string& scale_algorithm,
                         const std::pair<std::string, std::string>& decoders, const std::string& ffmpeg,
                         const std::string& ffprobe, bool want_vvdec,
                         const std::function<void()>& on_vvdec_opening,
                         const std::function<void(long long, long long, double)>& on_progress,
                         const CancelToken& cancel) {
    const auto& [source_backend, test_backend] = decoders;
    const bool cuda = plan.backend == "cuda";
    if (cuda && !plan.beside.empty()) {
        throw std::invalid_argument("libvmaf's CUDA code is not given CPU metrics to score beside it");
    }
    std::string gpu;  // its name, for messages
    if (cuda) {
        // libvmaf's CUDA code takes NVIDIA's pictures on the GPU and uploads
        // the software decoder's.
        for (const std::string& library : {source_backend, test_backend}) {
            if (library != "nvidia" && library != software_decoder) {
                throw GpuDecodeUnavailableError("libvmaf's CUDA code takes NVIDIA's and the software decoder's");
            }
        }
        gpu = "NVIDIA's GPU";
    } else {
        const std::vector<VulkanDevice> found = vulkan_devices();
        const auto chosen = std::find_if(found.begin(), found.end(),
                                         [&](const VulkanDevice& device) { return device.index == plan.device; });
        if (chosen == found.end()) {
            throw VmafVulkanError("no GPU that Vulkan can calculate VMAF on");
        }
        gpu = chosen->name;
    }
    for (const std::string& library : {source_backend, test_backend}) {
        if (library.empty() || !decoder_available(library)) {
            throw GpuDecodeUnavailableError("no decoder library for " + gpu);
        }
    }
    require_same_range(source, distorted);
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
    if (cuda) {
        return score_decoded_cuda(source, distorted, plans, plan, n_subsample, range, total_frames,
                                  decoders, ffmpeg, ffprobe, on_progress, cancel);
    }
    // Where wanted (Settings' VVC decoder): an 8- or 10-bit VVC video, unscaled and scored without film grain, decoded
    // on the GPU by vvdec-vulkan (vvdec_use): a test video's pictures handed over to the engine, the reference written
    // into its slots by the CPU (mixed); otherwise copied by the GPU into host memory, which the engine and the CPU's
    // metrics read (a handed-over frame's planes may not be read, vvdec.h's hand-over).
    const VvdecUse vvdec = vvdec_use(vvdec_chosen(plans[0], want_vvdec), vvdec_chosen(plans[1], want_vvdec),
                                     !plan.beside.empty());
    const std::string test_library = vvdec.test ? std::string(vvdec_decoder) : test_backend;
    const std::string source_library = vvdec.reference ? std::string(vvdec_decoder) : source_backend;
    // AMD's decoder hands its pictures over on the GPU where neither video is scaled or widened for the other
    // (can_hand_over); NVIDIA's always does. Not beside vvdec-vulkan's pictures in host memory, where nothing hands
    // over.
    const bool hand_over = (vvdec.mixed || !vvdec.any()) && can_hand_over(plans[0], test_library) &&
                           can_hand_over(plans[1], source_library);
    if (vvdec.any() && on_vvdec_opening) {
        on_vvdec_opening();
    }
    // vvdec-vulkan on the scorer's Vulkan device (plan.device), as its pictures go into the scorer's memory; the
    // other decoders on CUDA device 0. (Where this runs in the run's own process, vvdec-vulkan keeps the device it
    // first opened on: vvdecvk_frames sets it once a process.)
    GpuFrameStream test =
        open_pass_stream(distorted, plans[0], 4, test_backend, vvdec.test, plan.device, ffmpeg, ffprobe, hand_over);
    GpuFrameStream reference = open_pass_stream(source, plans[1], 4, source_backend, vvdec.reference, plan.device,
                                                ffmpeg, ffprobe, hand_over);
    // Declared after the streams so that it goes first: it holds on to the
    // decoder's memory.
    std::optional<VulkanScorer> scorer;
    // The planes stay on the GPU when the decoders and Vulkan share memory:
    // one decoder library for both videos, NVIDIA's or AMD's handing over
    // (the scorer's memory is imported by the test video's decoder, and the
    // source's copies go into it).
    bool shared = false;
    // The test video's pictures handed over, the reference's written by the CPU (whatever decodes it: its pictures
    // are downloaded, as without the hand-over)
    bool mixed = false;
    if (vvdec.mixed) {
        try {
            scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample, plan.device, &test);
            mixed = scorer->takes_host_planes();
            if (!mixed) {
                scorer.reset();
                throw VmafVulkanError("this libvmaf-fast takes no host planes beside a decoder's");
            }
            shared = true;
            app_log::write(app_log::Level::info, "vml.vmaf_vulkan",
                           "VMAF with Vulkan: the test video decoded by vvdec-vulkan on the GPU and handed over there, "
                           "the source's frames through system memory");
        } catch (const VmafVulkanError& error) {
            // vvdec-vulkan's pictures through system memory instead (HostPairs)
            app_log::write(app_log::Level::info, "vml.vmaf_vulkan",
                           std::string("VMAF with Vulkan: vvdec-vulkan's pictures go through system memory (") +
                               error.what() + ")");
        }
    } else if (vvdec.any()) {
        app_log::write(app_log::Level::info, "vml.vmaf_vulkan",
                       std::string("VMAF with Vulkan: vvdec-vulkan's pictures go through system memory (") +
                           (plan.beside.empty() ? "only a test video's are handed over to the engine"
                                                : "the CPU's metrics read them") +
                           ")");
    } else if (test.handover() && reference.handover() && !test.shares_gpu_memory_with(reference)) {
        // Windows' own decoder for one video and AMF's for the other.
        app_log::write(app_log::Level::info, "vml.vmaf_vulkan",
                       "VMAF with Vulkan: the frames go through system memory (the videos' decoders share no GPU "
                       "memory)");
    } else if (test.shares_gpu_memory_with(reference)) {
        try {
            scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample, plan.device, &test);
            shared = true;
        } catch (const VmafVulkanError& error) {
            // The frames go through system memory instead.
            app_log::write(app_log::Level::info, "vml.vmaf_vulkan",
                           std::string("VMAF with Vulkan: the frames go through system memory (") + error.what() + ")");
        }
    }
    if (!scorer) {
        scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample, plan.device);
    }
    // After the streams, so that it goes before them: it gives back the
    // memory NVIDIA's decoder page-locked for it.
    std::optional<CpuScorer> beside;
    start_beside(beside, plan, n_subsample);
    // Without the CPU's metrics, where neither video's picture needs a copy
    // on the CPU of its own to reach the engine (the software decoder's read
    // where it holds them, NVIDIA's downloaded by its GPU into page-locked
    // memory), the engine copies them on its copy threads. Pictures of other
    // decoders are written straight into the engine's memory instead.
    // vvdec-vulkan's not handed over are always taken so, as its GPU copies
    // them only into its own buffer, and the CPU's metrics read them there
    // too. Last, so that it goes first: it unlocks memory with the streams.
    const auto as_planes = [](const GpuFrameStream& stream) {
        return stream.gives_planes() || stream.backend() == "nvidia";
    };
    std::optional<HostPairs> pairs;
    if ((vvdec.any() && !mixed) || (!shared && !beside && as_planes(reference) && as_planes(test))) {
        pairs.emplace(reference, plans[1], test, plans[0]);
    }
    test.start();
    reference.start();
    score_frame_pairs(
        test, reference, range, total_frames,
        [&](int test_slot, int reference_slot) {
            if (pairs) {
                // A frame that is not scored needs only the source's (motion), unless the CPU's metrics read it.
                pairs->score(reference_slot, test_slot, *scorer, beside ? &*beside : nullptr,
                             beside || scorer->scores_next());
                return;
            }
            if (beside && !shared) {
                // Each luma plane written out once, into the CPU's picture,
                // and read from there by the engine.
                beside->add_decoded(reference, reference_slot, test, test_slot,
                                    [&](Plane source_plane, Plane test_plane) {
                                        scorer->add_planes(FramePlanes{source_plane}, FramePlanes{test_plane});
                                    });
                return;
            }
            if (beside) {
                beside->add_decoded(reference, reference_slot, test, test_slot);
            }
            if (mixed) {
                scorer->add_shared_mixed([&](std::uint8_t* plane) { reference.download(reference_slot, plane); },
                                         [&](std::uint64_t address, std::int64_t pitch) -> std::uint64_t {
                                             if (scorer->waits_on_decoder()) {
                                                 return test.copy_luma_async(test_slot, address, pitch);
                                             }
                                             test.copy_luma(test_slot, address, pitch);
                                             return 0;
                                         });
            } else if (shared) {
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
    PairScores scores = scorer->finish();
    add_beside(scores, beside);
    return scores;
}

}  // namespace vml
