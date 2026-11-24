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
#include "vml/gpu_checks.h"
#include "vml/gpu_choice.h"
#include "vml/gpu_frames.h"
#include "vml/host_frames.h"
#include "vml/isolated.h"
#include "vml/libvmaf_cpu.h"
#include "vml/metrics.h"
#include "vml/geometry.h"
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
    std::timed_mutex mutex;  // held for the whole of a probe
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
            run_gpu_check("vulkan-probe", "the Vulkan VMAF probe", nlohmann::json::object(), cancel);
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
            run_gpu_check("cuda-probe", "libvmaf's GPU probe", nlohmann::json::object(), cancel);
        return {result.at("available").get<bool>(), result.at("text").get<std::string>()};
    } catch (const IsolatedCrashError& error) {
        crashed = true;
        return {false, error.what()};
    }
}

// The GPU chosen (gpu_choice.h), unless it names one NVIDIA GPU DirectX
// lists: CUDA would score on an NVIDIA GPU the choice keeps free. (Where the
// system lists no GPUs, nothing: CUDA is left to find its own.)
std::optional<std::string> chosen_gpu_not_nvidia() {
    const std::optional<std::string> chosen = chosen_gpu();
    if (!chosen) {
        return std::nullopt;
    }
    const std::vector<platform::GpuAdapter> adapters = platform::gpu_adapters();
    std::vector<std::string> names;
    for (const platform::GpuAdapter& adapter : adapters) {
        names.push_back(adapter.name);
    }
    if (adapters.empty()) {
        return std::nullopt;
    }
    const std::optional<std::size_t> index = chosen_among(names, *chosen);
    return index && adapters[*index].vendor_id == platform::nvidia_pci_vendor ? std::nullopt : chosen;
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
                if (const std::optional<std::string> other = chosen_gpu_not_nvidia()) {
                    add_reason("CUDA: the GPU chosen, " + *other + ", is not an NVIDIA GPU here");
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
    const std::unique_lock lock = lock_unless_cancelled(probe.mutex, cancel);
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
    if (!enabled || bit_depth > 10 || (size && is_odd_size(size->first, size->second))) {
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

std::unique_ptr<HdrScorer> make_hdr_scorer(const GpuPlan& plan, int n_subsample, GpuFrameStream* shared) {
    return std::make_unique<HdrScorer>(plan.width, plan.height, plan.bit_depth,
                                       platform::path_from_text(plan.hdr_model_file), plan.hdr_colours, n_subsample,
                                       plan.hdr_threads, plan.hdr_gpu, shared);
}

namespace {

// `scores` with `more`'s metrics after its own, the two scored for the same frames (`what` names `more` in the error).
void append_scores(PairScores& scores, PairScores&& more, const std::string& what) {
    if (more.frames != scores.frames) {
        throw LibvmafError(what + " were scored for other frames than the pass's own metrics (" +
                           std::to_string(more.frames.size()) + " against " + std::to_string(scores.frames.size()) +
                           ")");
    }
    for (auto& metric : more.by_metric) {
        scores.by_metric.push_back(std::move(metric));
    }
    scores.features_on_gpu.insert(scores.features_on_gpu.end(), more.features_on_gpu.begin(),
                                  more.features_on_gpu.end());
}

}  // namespace

BesideScorer::BesideScorer(const GpuPlan& plan, int n_subsample, GpuFrameStream* shared) {
    std::vector<std::string> cpu_metrics;
    bool hdr = false;
    for (const std::string& key : plan.beside) {
        if (key == "vmaf_hdr") {
            hdr = true;
        } else {
            cpu_metrics.push_back(key);
        }
    }
    if (!cpu_metrics.empty()) {
        cpu_.emplace(plan.width, plan.height, plan.bit_depth, cpu_metrics, n_subsample, plan.threads,
                     plan.frame_rate, plan.cambi, plan.extra_gpu);
    }
    if (hdr) {
        hdr_ = make_hdr_scorer(plan, n_subsample, shared);
    }
}

BesideScorer::~BesideScorer() = default;

std::size_t BesideScorer::frame_bytes() const { return cpu_ ? cpu_->frame_bytes() : hdr_->frame_bytes(); }

void BesideScorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    if (cpu_) {
        cpu_->add(reference, distorted);
    }
    if (hdr_) {
        hdr_->add(reference, distorted);
    }
}

void BesideScorer::add_planes(const FramePlanes& reference, const FramePlanes& distorted) {
    if (cpu_) {
        cpu_->add_planes(reference, distorted);
    }
    if (hdr_) {
        hdr_->add_planes(reference, distorted);
    }
}

CpuScorer& BesideScorer::cpu_alone() {
    if (hdr_ || !cpu_) {
        throw std::logic_error("HDR VMAF beside a pass reads whole frames, not the CPU's metrics' downloads");
    }
    return *cpu_;
}

void BesideScorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
    if (cpu_) {
        cpu_->add_decoded(reference, reference_slot, test, test_slot);
    }
    if (hdr_) {
        hdr_->add_decoded(reference, reference_slot, test, test_slot);
    }
}

void BesideScorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot,
                               const std::function<void(Plane, Plane)>& also) {
    cpu_alone().add_decoded(reference, reference_slot, test, test_slot, also);
}

void BesideScorer::add_decoded_alongside(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test,
                                         int test_slot, const std::function<void()>& alongside) {
    const auto add = [&] { add_decoded(reference, reference_slot, test, test_slot); };
    // Only NVIDIA's library downloads on a CUDA stream of its own, apart from
    // its copies on the GPU (native/nvdec_frames.cpp's download_stream). The
    // other decoders' downloads and copies of one stream are not known to run
    // at once (vvdec-vulkan's take turns under its decoder's lock).
    if (!gpu_decoder(reference.backend()).cuda_copies || !gpu_decoder(test.backend()).cuda_copies) {
        add();
        alongside();
        return;
    }
    // At the same time, the CPU's downloads wait on the PCIe bus while the
    // GPU's scorers wait on the GPU (an RTX 5090's 4K pair with all seven
    // metrics: 4.5 ms one after the other, 3.8 at once).
    if (!side_) {
        side_ = std::make_unique<SideThread>();
    }
    side_->both(add, alongside);
}

PairScores BesideScorer::finish() {
    if (!hdr_) {
        return cpu_->finish();
    }
    if (!cpu_) {
        return hdr_->finish();
    }
    PairScores scores = cpu_->finish();
    append_scores(scores, hdr_->finish(), "HDR VMAF's scores");
    return scores;
}

void start_beside(std::optional<BesideScorer>& scorer, const GpuPlan& plan, int n_subsample,
                  GpuFrameStream* shared) {
    if (plan.beside.empty()) {
        return;
    }
    scorer.emplace(plan, n_subsample, shared);
    if (const CpuScorer* cpu = scorer->cpu(); cpu != nullptr && plan.on_gpu_metrics) {
        plan.on_gpu_metrics(cpu->metric_keys(), cpu->gpu_metrics());
    }
    // Which pass took them, so that a run shows it decoded the videos once for all of its metrics.
    const std::string pass = plan.backend == "hdr"    ? "HDR VMAF"
                             : plan.backend == "cpu" ? "the CPU's metrics"
                                                     : "VMAF on the GPU (" + plan.backend + ")";
    logger.info(metric_labels(plan.beside, " and ") + " beside " + pass + ", from the same frames");
}

void add_beside(PairScores& scores, std::optional<BesideScorer>& beside) {
    if (beside) {
        append_scores(scores, beside->finish(), "The metrics beside them");
    }
}

std::timed_mutex& gpu_pass() {
    static std::timed_mutex pass;
    return pass;
}

std::unique_lock<std::timed_mutex> take_gpu_turn(const std::function<void()>& on_wait, const CancelToken& cancel) {
    std::unique_lock pass(gpu_pass(), std::try_to_lock);
    if (pass.owns_lock()) {
        return pass;
    }
    if (on_wait) {
        on_wait();
    }
    return lock_unless_cancelled(gpu_pass(), cancel);
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
    State(std::string_view name, std::size_t frame_bytes, Framing framing, std::size_t buffer_count)
        : name(name),
          pipe(name, platform::frame_pipe_bytes),
          frame_bytes(frame_bytes),
          framing(framing),
          buffers(std::max<std::size_t>(1, buffer_count)) {
        for (std::vector<std::uint8_t>& buffer : buffers) {
            buffer.resize(frame_bytes);
            free.put(&buffer);
        }
    }

    std::string name;  // what the pipe's path ends in: "distorted", "reference"
    platform::InboundPipe pipe;
    std::size_t frame_bytes;
    Framing framing;
    // The frame being paired and those read ahead.
    std::vector<std::vector<std::uint8_t>> buffers;
    BufferQueue free;    // buffers to read into; nullptr: stop reading frames
    BufferQueue pixels;  // frames read; nullptr: the end
    std::atomic<bool> stopped{false};
    std::atomic<bool> writer_gone{false};
    std::atomic<bool> connected{false};  // the wait for a writer is over (InboundPipe::connect)
    std::atomic<bool> finished{false};   // the reading thread has ended
    std::atomic<bool> stopped_while_reading{false};
    std::exception_ptr error;            // set before the final nullptr in pixels
    std::thread thread;
};

StreamReader::StreamReader(std::string_view name, std::size_t frame_bytes, Framing framing, std::size_t buffers)
    : state_(std::make_unique<State>(name, frame_bytes, framing, buffers)) {}

StreamReader::~StreamReader() {
    if (state_->thread.joinable()) {
        stop();
        state_->thread.join();
    }
}

const std::string& StreamReader::path() const { return state_->pipe.path(); }

int StreamReader::take_writer() { return state_->pipe.take_writer(); }

void StreamReader::start() { state_->thread = std::thread([this] { read_frames(); }); }

// The stream is a header line, "YUV4MPEG2 W.. H.. ...\n", then for each frame
// a "FRAME\n" line and the frame's planes, packed.
void StreamReader::read_yuv4mpeg_frames() {
    State& s = *state_;
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
            return;
        }
        if (std::memcmp(marker.data(), "FRAME\n", marker.size()) != 0) {
            throw LibvmafError("FFmpeg's frames are not where they were expected");
        }
        std::vector<std::uint8_t>* buffer = s.free.get();
        if (buffer == nullptr) {
            return;
        }
        if (!read_exactly(s.pipe, buffer->data(), buffer->size(), s.stopped, s.writer_gone)) {
            throw FramesCutError("FFmpeg's frames ended after a frame's start");
        }
        s.pixels.put(buffer);
    }
}

// The stream is the frames, one after the other: it ends between two, and a
// frame cut short is FFmpeg's failure (read_exactly).
void StreamReader::read_raw_frames() {
    State& s = *state_;
    while (!s.stopped) {
        std::vector<std::uint8_t>* buffer = s.free.get();
        if (buffer == nullptr) {
            return;
        }
        if (!read_exactly(s.pipe, buffer->data(), buffer->size(), s.stopped, s.writer_gone)) {
            s.free.put(buffer);
            return;
        }
        s.pixels.put(buffer);
    }
}

void StreamReader::read_frames() {
    State& s = *state_;
    try {
        s.pipe.connect();
        s.connected = true;
        if (!s.stopped) {
            if (s.framing == Framing::raw) {
                read_raw_frames();
            } else {
                read_yuv4mpeg_frames();
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
    s.pipe.no_writer_will_come();
    // Again until the reading thread has gone past waiting for a writer,
    // which it does a moment after it starts: for a named pipe, opened now
    // and read only after that.
    for (int attempt = 0; attempt < 200 && s.thread.joinable() && !s.connected && !s.finished; ++attempt) {
        s.pipe.no_writer_will_come();
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
    std::optional<BesideScorer> beside;  // the plan's metrics beside the pass's own
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
        auto cpu = std::make_unique<CpuScorer>(plan.width, plan.height, plan.bit_depth, metrics, n_subsample,
                                               plan.threads, plan.frame_rate, plan.cambi, plan.extra_gpu);
        if (plan.on_gpu_metrics) {
            plan.on_gpu_metrics(cpu->metric_keys(), cpu->gpu_metrics());
        }
        s.scorer = std::move(cpu);
    } else if (plan.backend == "hdr") {
        s.scorer = make_hdr_scorer(plan, n_subsample);
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
        throw LibvmafError("the metrics beside the pass take frames of another size than its own");
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

std::vector<int> GpuAttempt::take_output_descriptors(std::size_t output) {
    const int writer = (output == 0 ? state_->distorted : state_->reference)->take_writer();
    return writer >= 0 ? std::vector<int>{writer} : std::vector<int>{};
}

bool GpuAttempt::closed_pipe_of(std::size_t output) const {
    return (output == 0 ? state_->distorted : state_->reference)->stopped_while_reading();
}

bool both_copy_on_gpu(const GpuFrameStream& test, const GpuFrameStream& reference) {
    const auto copies = [](const GpuFrameStream& stream) {
        return stream.handover() && (stream.backend() == vvdec_decoder || stream.backend() == nvidia_decoder);
    };
    return copies(test) && copies(reference) &&
           (test.backend() == vvdec_decoder || reference.backend() == vvdec_decoder);
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
                   gpu_decoder(source_backend).cuda_copies && gpu_decoder(test_backend).cuda_copies);
    // A decoded picture as the scorer takes it.
    const auto side = [](GpuFrameStream& stream, int slot) {
        CudaScorer::Decoded decoded;
        if (gpu_decoder(stream.backend()).cuda_copies) {  // copied on the GPU
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
    // HDR VMAF beside reads whole frames (its colour conversion takes the chroma too).
    const bool hdr_beside = std::ranges::find(plan.beside, "vmaf_hdr") != plan.beside.end();
    // The 6:1:1 metrics beside read the chroma too: the decoders decode whole
    // frames, and the engine is given their luma.
    const bool chroma_beside = cpu_metrics_read_chroma(plan.beside);
    std::string gpu;  // its name, for messages
    if (cuda) {
        // libvmaf's CUDA code takes NVIDIA's pictures on the GPU and uploads
        // the software decoder's.
        for (const std::string& library : {source_backend, test_backend}) {
            if (!libvmaf_cuda_takes(library)) {
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
    const std::pair size(plan.width, plan.height);
    std::vector<DecodePlan> plans = plan_pair(source, distorted, source_crop, distorted_crop, size, plan.bit_depth,
                                              !hdr_beside && !chroma_beside, scale_algorithm, decoders);
    if (!hdr_beside) {
        require_frames_of_size(plans, size, plan.bit_depth);
    }
    if (cuda) {
        plan_pair_seek(plans, distorted, source, range.decode_start, ffmpeg, ffprobe, cancel);
        return score_decoded_cuda(source, distorted, plans, plan, n_subsample, range, total_frames,
                                  decoders, ffmpeg, ffprobe, on_progress, cancel);
    }
    // Where wanted (Settings' VVC decoder): an 8- or 10-bit VVC video, unscaled and scored without film grain, decoded
    // on the GPU by vvdec-vulkan (vvdec_use). Where the other video's decoder copies on the GPU too (vvdec-vulkan's or
    // NVIDIA's) and no metric runs beside (they need both pictures in system memory anyway: the extra copies on
    // the GPU made the six metrics of a 1080p pair about 30 % slower on an RTX 5090), both videos' pictures are
    // copied into the engine's memory by the GPU (both_copy_on_gpu). Otherwise a test video's pictures handed over to
    // the engine, the reference written into its slots by the CPU (mixed); else copied by the GPU into host memory,
    // which the engine and the CPU's metrics read (a handed-over frame's planes may not be read, vvdec.h's hand-over).
    const VvdecUse vvdec = vvdec_use(vvdec_chosen(plans[0], want_vvdec), vvdec_chosen(plans[1], want_vvdec),
                                     !plan.beside.empty());
    const std::string test_library = vvdec.test ? std::string(vvdec_decoder) : test_backend;
    const std::string source_library = vvdec.reference ? std::string(vvdec_decoder) : source_backend;
    // AMD's decoder hands its pictures over on the GPU where neither video is scaled or widened for the other
    // (can_hand_over); NVIDIA's always does. Not beside vvdec-vulkan's pictures in host memory, where nothing hands
    // over, nor beside the CPU's metrics or HDR VMAF (pass_may_hand_over).
    const bool hand_over = pass_may_hand_over(plan) && (vvdec.mixed || !vvdec.any()) &&
                           can_hand_over(plans[0], test_library) && can_hand_over(plans[1], source_library);
    if (vvdec.any() && on_vvdec_opening) {
        on_vvdec_opening();
    }
    // vvdec-vulkan on the scorer's Vulkan device (plan.device), as its pictures go into the scorer's memory; the
    // other decoders on CUDA device 0. (Where this runs in the run's own process, vvdec-vulkan keeps the device it
    // first opened on: vvdecvk_frames sets it once a process.)
    plan_pair_seek(plans, distorted, source, range.decode_start, ffmpeg, ffprobe, cancel);
    GpuFrameStream test =
        open_pass_stream(distorted, plans[0], 4, test_backend, vvdec.test, plan.device, ffmpeg, ffprobe, hand_over);
    GpuFrameStream reference = open_pass_stream(source, plans[1], 4, source_backend, vvdec.reference, plan.device,
                                                ffmpeg, ffprobe, hand_over);
    // The metrics beside. Where the decoders and Vulkan share memory (below), HDR VMAF's GPU takes the pictures
    // from the decoders too, if it converts them (HdrScorer's `shared`); the pass's scorer then does as well.
    // After the streams, so that it goes before them: it gives back the memory NVIDIA's decoder page-locked for
    // it, and the decoders' memory it imported.
    const bool may_share = !vvdec.any() && test.shares_gpu_memory_with(reference);
    std::optional<BesideScorer> beside;
    start_beside(beside, plan, n_subsample, may_share ? &test : nullptr);
    // Declared after the streams so that it goes first: it holds on to the
    // decoder's memory.
    std::optional<VulkanScorer> scorer;
    // The planes stay on the GPU when the decoders and Vulkan share memory:
    // one decoder library for both videos, NVIDIA's or AMD's handing over
    // (the scorer's memory is imported by the test video's decoder, and the
    // source's copies go into it), or vvdec-vulkan's beside its own or
    // NVIDIA's (each imports the scorer's memory).
    bool shared = false;
    // The test video's pictures handed over, the reference's written by the CPU (whatever decodes it: its pictures
    // are downloaded, as without the hand-over)
    bool mixed = false;
    if (vvdec.any() && plan.beside.empty() && both_copy_on_gpu(test, reference)) {
        try {
            scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample, plan.device, &test,
                           &reference);
            shared = true;
        } catch (const VmafVulkanError& error) {
            app_log::write(app_log::Level::info, "vml.vmaf_vulkan",
                           std::string("VMAF with Vulkan: the two videos' pictures are not both copied on the GPU (") +
                               error.what() + ")");
        }
    }
    if (shared) {
        app_log::write(app_log::Level::info, "vml.vmaf_vulkan",
                       "VMAF with Vulkan: vvdec-vulkan's pictures and the other video's copied on the GPU into the "
                       "engine's memory");
    } else if (vvdec.mixed) {
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
                                                : "the metrics beside it read them") +
                           ")");
    } else if (test.handover() && reference.handover() && !test.shares_gpu_memory_with(reference)) {
        // Windows' own decoder for one video and AMF's for the other.
        app_log::write(app_log::Level::info, "vml.vmaf_vulkan",
                       "VMAF with Vulkan: the frames go through system memory (the videos' decoders share no GPU "
                       "memory)");
    } else if (may_share && (!beside || beside->takes_decoded())) {
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
    if (!shared && hdr_beside && beside->takes_decoded()) {
        // HDR VMAF took the decoders' pictures, the pass's scorer could not: both read them in system memory.
        beside.reset();
        start_beside(beside, plan, n_subsample);
    }
    if (beside && beside->reads_whole_frames() &&
        std::any_of(plans.begin(), plans.end(),
                    [&](const DecodePlan& decoded) { return decoded.frame_bytes() != beside->frame_bytes(); })) {
        throw LibvmafError("the decoders' frames are not the size HDR VMAF's scorer takes");
    }
    // Without the CPU's metrics, where neither video's picture needs a copy
    // on the CPU of its own to reach the engine (the software decoder's read
    // where it holds them, NVIDIA's downloaded by its GPU into page-locked
    // memory), the engine copies them on its copy threads. Pictures of other
    // decoders are written straight into the engine's memory instead.
    // vvdec-vulkan's not handed over are always taken so, as its GPU copies
    // them only into its own buffer, and the CPU's metrics read them there
    // too. So are all pictures beside HDR VMAF where the decoders do not
    // copy them on the GPU, as it reads whole frames.
    // Last, so that it goes first: it unlocks memory with the streams.
    const auto as_planes = [](const GpuFrameStream& stream) {
        return stream.gives_planes() || gpu_decoder(stream.backend()).cuda_copies;
    };
    std::optional<HostPairs> pairs;
    if (!shared && (vvdec.any() || hdr_beside || (!beside && as_planes(reference) && as_planes(test)))) {
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
            const auto add_to_scorer = [&] {
                if (mixed) {
                    scorer->add_shared_mixed(
                        [&](std::uint8_t* plane) { reference.download(reference_slot, plane); },
                        [&](std::uint64_t address, std::int64_t pitch) -> std::uint64_t {
                            if (scorer->waits_on_decoder()) {
                                return test.copy_luma_async(test_slot, address, pitch);
                            }
                            test.copy_luma(test_slot, address, pitch);
                            return 0;
                        });
                } else if (shared) {
                    scorer->add_shared(luma_copy(*scorer, reference, reference_slot),
                                       luma_copy(*scorer, test, test_slot));
                } else {
                    scorer->add_decoded([&](std::uint8_t* plane) { reference.download(reference_slot, plane); },
                                        [&](std::uint8_t* plane) { test.download(test_slot, plane); });
                }
            };
            if (beside) {
                beside->add_decoded_alongside(reference, reference_slot, test, test_slot, add_to_scorer);
            } else {
                add_to_scorer();
            }
        },
        on_progress, cancel);
    PairScores scores = scorer->finish();
    add_beside(scores, beside);
    return scores;
}

}  // namespace vml
