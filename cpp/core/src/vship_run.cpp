#include "vml/vship_run.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <system_error>
#include <thread>

#include "vml/app_log.h"
#include "vml/colour.h"
#include "vml/ffmpeg_tools.h"
#include "vml/frame_sync.h"
#include "vml/frame_sync_by_time.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
#include "vml/isolated.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/perceptual_cpu.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/run_json.h"
#include "vml/threads.h"
#include "vml/score_identity.h"
#include "vml/temp_files.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.perceptual_vship");

// Vship handlers working at once per metric. Where the GPU sets the pace,
// more keep it busy between one handler's kernels: Vship-fast's README
// measures three at 1.2-1.6x the speed of one on an RTX 3070. Where decoding
// does, they change nothing: on an RTX 5090, FFmpeg decoding 4K 10-bit, each
// metric scored about 50 pairs a second with one, two or three, with the
// same scores. On MoltenVK one: on an M1 Pro two gained at most 3%
// (1080p, 480 pairs: SSIMULACRA2 53.2 against 54.5 pairs a second,
// Butteraugli 21.7 against 22.0, CVVDP 39.1 against 39.2), for twice the
// handlers' memory; Vship 5.1.2 lost the GPU there with two Butteraugli
// handlers.
#ifdef __APPLE__
constexpr std::size_t lanes_per_metric = 1;
#else
constexpr std::size_t lanes_per_metric = 2;
#endif
// Frames in flight per video: one per lane being scored, one waiting and
// one being filled, so that decoding, transfer and GPU work overlap. A 4K
// 10-bit frame is 25 MB of pinned memory, so this is 250 MB for a 4K pair.
constexpr int ring_slots = static_cast<int>(lanes_per_metric) + 3;
// The size of the pipe between FFmpeg and this process. Windows' default is
// a few kilobytes, which holds a raw 4K stream near 55 fps however fast the
// decoder is; with 64 MB, one 4K 10-bit HEVC stream comes at 107 fps.
constexpr std::size_t frame_pipe_bytes = 64 * 1024 * 1024;
constexpr int end_of_video = -1;
constexpr std::string_view y4m_frame_marker = "FRAME\n";
// How long an FFmpeg that ended partway through a frame is given to exit by
// itself, as long as a frame's timestamp is waited for (Timestamps).
constexpr std::chrono::seconds partial_frame_exit_wait(10);
const std::vector<std::string> vship_metrics{"ssimulacra2", "butteraugli", "cvvdp"};

std::string label_of(const std::string& key) { return std::string(metric_definition(key).label); }

// An exit code as the log gives it: in hexadecimal too, as Windows' crash
// codes are known (0xC0000005).
std::string exit_code_for_log(int code) {
    std::array<char, 16> hex{};
    std::snprintf(hex.data(), hex.size(), "0x%08X", static_cast<unsigned>(code));
    return platform::exit_code_text(code) + " (" + hex.data() + ")";
}

// Thrown only when pairing by timestamp: a subsampled pass whose source
// frames do not line up with the test video's (VshipPass::check_lined_up),
// which is then made again with every source frame.
class FramesApartError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ------------------------------------------------------------ frame memory

// One frame's buffer in Vship's pinned (page-locked) memory.
class PinnedBuffer {
public:
    PinnedBuffer(const VshipLibrary& library, std::size_t bytes, int gpu_id)
        : library_(library), gpu_id_(gpu_id) {
        const int error = library_.pinned_malloc(&address_, bytes, gpu_id);
        if (error != 0 || address_ == nullptr) {
            address_ = nullptr;
            throw VshipUnavailableError("Could not allocate Vship pinned frame memory: " + library_.message(error));
        }
    }
    ~PinnedBuffer() {
        if (address_ != nullptr) {
            library_.pinned_free(address_, gpu_id_);
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    std::uint8_t* data() const { return static_cast<std::uint8_t*>(address_); }

    // The frame's three planes, in the order Vship reads them.
    std::array<const std::uint8_t*, 3> planes(const VshipImageFormat& format,
                                              const std::array<std::size_t, 3>& sizes) const {
        const std::array<std::size_t, 3> offsets{0, sizes[0], sizes[0] + sizes[1]};
        std::array<const std::uint8_t*, 3> found{};
        for (std::size_t plane = 0; plane < 3; ++plane) {
            found[plane] = data() + offsets[static_cast<std::size_t>(format.plane_order[plane])];
        }
        return found;
    }

private:
    const VshipLibrary& library_;
    void* address_ = nullptr;
    int gpu_id_;
};

// Fills `size` bytes at `into` from the program's standard output, and
// returns how many it read: fewer only at its end. Read straight into the
// pinned memory, with no copy between.
std::size_t read_exact(platform::ChildProcess& process, std::uint8_t* into, std::size_t size) {
    std::size_t filled = 0;
    while (filled < size) {
        const std::size_t got = process.read(into + filled, size - filled);
        if (got == 0) {
            break;
        }
        filled += got;
    }
    return filled;
}

// How a hardware-decoded frame crosses the pipe as it was downloaded (NV12,
// P010): a luma plane of `luma` bytes, then U and V interleaved, `plane`
// bytes each once split, 2 bytes a sample when `wide`.
struct InterleavedChroma {
    std::size_t luma = 0;
    std::size_t plane = 0;
    bool wide = false;
};

// A ring of pinned frame buffers that a thread fills with one video's
// frames, handing them over in order, and that the scoring gives each slot
// back to once Vship is done with it. The pinned memory is freed with the
// ring, once the thread that fills it has stopped (each kind of stream's
// close).
class FrameRing {
public:
    FrameRing(const VshipLibrary& library, std::size_t frame_bytes, int gpu_id) {
        // One at a time, so that a failure part-way through frees those
        // already allocated (25 MB of pinned memory each for 4K 10-bit).
        for (int slot = 0; slot < ring_slots; ++slot) {
            buffers_.push_back(std::make_unique<PinnedBuffer>(library, frame_bytes, gpu_id));
            free_.put(slot);
        }
    }
    virtual ~FrameRing() = default;
    FrameRing(const FrameRing&) = delete;
    FrameRing& operator=(const FrameRing&) = delete;

    const std::vector<std::unique_ptr<PinnedBuffer>>& buffers() const { return buffers_; }

    // Starts decoding.
    virtual void start() = 0;

    // Stops decoding and the thread that fills the ring.
    virtual void close() = 0;

    // The next filled slot, or end_of_video. Throws the filling thread's
    // error, and Cancelled; checks `cancel` every 100 ms while it waits.
    int next(const CancelToken& cancel) {
        while (true) {
            cancel.throw_if_cancelled();
            if (std::optional<Filled> item = filled_.get_for(std::chrono::milliseconds(100))) {
                if (item->error) {
                    std::rethrow_exception(item->error);
                }
                return item->slot;
            }
        }
    }

    // Gives a slot back once Vship is done with it.
    void release(int slot) { free_.put(slot); }

    // When pairing by timestamp, the timestamp of a slot next() handed out,
    // and the time base it is in.
    long long pts(int slot) const { return pts_[static_cast<std::size_t>(slot)]; }
    Rational time_base() const { return time_base_.value(); }

protected:
    struct Filled {
        int slot = end_of_video;
        std::exception_ptr error;  // the filling thread's, handed over instead of a slot
    };

    // Notes a slot's timestamp before the slot is handed over, and the time
    // base with the first one (it does not change).
    void stamp(int slot, long long pts, Rational time_base) {
        pts_[static_cast<std::size_t>(slot)] = pts;
        if (!time_base_) {
            time_base_ = time_base;
        }
    }

    std::vector<std::unique_ptr<PinnedBuffer>> buffers_;
    Channel<int> free_;
    Channel<Filled> filled_;
    std::thread thread_;

private:
    std::array<long long, ring_slots> pts_{};
    std::optional<Rational> time_base_;
};

// The timestamp lines one FFmpeg writes as it sends frames (-stats_enc_pre),
// one line a frame read (the Python app's _Timestamps). A frame's line is
// written and flushed before the frame, so it is in the file once the frame
// has been read; the only wait is for a line caught part-way through.
class Timestamps {
public:
    Timestamps(std::filesystem::path path, platform::ChildProcess& process, std::string label)
        : path_(std::move(path)), process_(process), label_(std::move(label)) {}

    // The next frame's timestamp, in time_base().
    long long next() {
        std::optional<std::chrono::steady_clock::time_point> deadline;
        while (true) {
            if (const std::size_t newline = pending_.find('\n'); newline != std::string::npos) {
                const std::string line = pending_.substr(0, newline);
                pending_.erase(0, newline + 1);
                return parse(line);
            }
            if (!file_.is_open()) {
                file_.open(path_, std::ios::binary);
            }
            if (file_.is_open()) {
                std::array<char, 4096> chunk{};
                file_.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
                const std::streamsize count = file_.gcount();
                file_.clear();  // after its end, so that what FFmpeg writes next is read
                if (count > 0) {
                    pending_.append(chunk.data(), static_cast<std::size_t>(count));
                    continue;
                }
            }
            const bool ended = process_.wait_for(std::chrono::milliseconds(0));
            const auto now = std::chrono::steady_clock::now();
            if (!deadline) {
                deadline = now + std::chrono::seconds(10);
            }
            if (ended || now >= *deadline) {
                throw missing("FFmpeg wrote none");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    Rational time_base() const { return time_base_.value(); }

private:
    // The pass's failure for a frame without a usable timestamp. The window
    // gets one fixed message (it has a translation), and the log gets
    // `what`.
    VshipUnavailableError missing(const std::string& what) const {
        logger.error("Vship: no timestamp for a " + label_ + " frame: " + what);
        return VshipUnavailableError("FFmpeg did not give the timestamp of a " + label_ + " frame.");
    }

    // What FFmpeg wrote, for the log: as Python writes bytes (b'123 1/1000').
    static std::string written(std::string_view line) {
        return "FFmpeg wrote b" + python_string_repr(line.substr(0, 80));
    }

    static std::string fraction_text(const Rational& value) {
        return std::to_string(value.numerator) +
               (value.denominator == 1 ? std::string() : "/" + std::to_string(value.denominator));
    }

    // A line's timestamp and time base ("123 1/1000"), the time base checked
    // against the first line's.
    long long parse(std::string_view line) {
        std::vector<std::string_view> words;
        for (std::size_t at = 0; at < line.size();) {
            const std::size_t start = line.find_first_not_of(" \t\r\n\v\f", at);
            if (start == std::string_view::npos) {
                break;
            }
            const std::size_t end = std::min(line.find_first_of(" \t\r\n\v\f", start), line.size());
            words.push_back(line.substr(start, end - start));
            at = end;
        }
        if (words.size() != 2) {
            throw missing(written(line));
        }
        const std::optional<long long> pts = python_int(words[0]);
        const std::size_t slash = words[1].find('/');
        const std::optional<long long> numerator = python_int(words[1].substr(0, slash));
        const std::optional<long long> denominator =
            slash == std::string_view::npos ? std::optional(1LL) : python_int(words[1].substr(slash + 1));
        if (!pts || !numerator || !denominator || *denominator == 0) {
            throw missing(written(line));
        }
        long long top = *numerator;
        long long bottom = *denominator;
        if (bottom < 0) {
            top = -top;
            bottom = -bottom;
        }
        if (*pts == std::numeric_limits<long long>::min() || top <= 0) {  // AV_NOPTS_VALUE, or no time base
            throw missing(written(line) + ": no timestamp");
        }
        const long long divisor = std::gcd(top, bottom);
        const Rational time_base{top / divisor, bottom / divisor};
        if (!time_base_) {
            time_base_ = time_base;
        } else if (time_base != *time_base_) {
            throw missing("its time base changed from " + fraction_text(*time_base_) + " to " +
                          fraction_text(time_base));
        }
        return *pts;
    }

    std::filesystem::path path_;
    platform::ChildProcess& process_;
    std::string label_;
    std::ifstream file_;
    std::string pending_;
    std::optional<Rational> time_base_;
};

// `command`, whose last argument is its output, made to also write each piped
// frame's timestamp to `path`: the picture's timestamp as it leaves the
// filter chain, in the chain's time base (-enc_time_base filter), which is
// the timestamp libvmaf's frame sync is given at the end of the same chain.
// Not the decoder's: a file that stores no presentation times (H.264 with
// B-frames in AVI) has none, as FFmpeg works its frames' times out after
// decoding.
std::vector<std::string> with_timestamps(std::vector<std::string> command, const std::filesystem::path& path) {
    command.insert(command.end() - 1, {"-enc_time_base", "filter", "-stats_enc_pre", platform::path_text(path),
                                       "-stats_enc_pre_fmt", "{pts} {tb}"});
    return command;
}

// One video decoded by FFmpeg, feeding a ring (the Python app's
// _FrameStream): a thread reads FFmpeg's frames into free slots. If hardware
// decoding fails before giving a frame, the same pictures are decoded in
// software instead (the next command), and `on_software` is called first.
// With `timestamps`, each frame's timestamp is read too, for pairing by
// timestamp, from a file FFmpeg writes in `scratch_folder`.
class FrameStream : public FrameRing {
public:
    FrameStream(const VshipLibrary& library, std::size_t frame_bytes, std::vector<std::vector<std::string>> commands,
                std::string label, std::optional<InterleavedChroma> split, std::function<void()> on_software,
                int gpu_id, bool timestamps, std::filesystem::path scratch_folder)
        : FrameRing(library, frame_bytes, gpu_id),
          frame_bytes_(frame_bytes),
          commands_(std::move(commands)),
          label_(std::move(label)),
          split_(split),
          on_software_(std::move(on_software)),
          timestamps_(timestamps),
          scratch_folder_(std::move(scratch_folder)) {
        if (split_) {
            staging_.resize(2 * split_->plane);
        }
    }

    ~FrameStream() override { close(); }

    void start() override {
        thread_ = std::thread([this] { run(); });
    }

    void close() override {
        {
            const std::lock_guard lock(process_mutex_);
            stopping_ = true;
            if (process_ != nullptr) {
                process_->terminate();
            }
        }
        free_.put(end_of_video);  // wakes a reader waiting for a slot
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:

    // The reading thread: runs each FFmpeg command in turn (hardware decoding
    // first) until one succeeds. What fails here is the pass's input
    // (VshipInputError).
    void run() {
        try {
            for (std::size_t attempt = 0; attempt < commands_.size(); ++attempt) {
                const auto [frames, code, errors] = decode(commands_[attempt]);
                if (stopping_) {
                    return;
                }
                if (code == 0) {
                    filled_.put({end_of_video, nullptr});
                    return;
                }
                if (frames == 0 && attempt + 1 < commands_.size()) {
                    // Hardware decoding refused this stream: decode it in software.
                    if (on_software_) {
                        on_software_();
                    }
                    continue;
                }
                logger.error("Vship: FFmpeg failed while decoding the " + label_ + " (exit code " +
                             exit_code_for_log(code) + ")");
                throw VshipUnavailableError("FFmpeg failed while decoding the " + label_ + " for Vship" +
                                            (errors.empty() ? "." : ": " + errors));
            }
        } catch (const VshipInputError&) {  // handed to the scoring thread, never lost
            filled_.put({end_of_video, std::current_exception()});
        } catch (const std::exception& error) {
            filled_.put({end_of_video, std::make_exception_ptr(VshipInputError(error.what()))});
        } catch (...) {
            filled_.put({end_of_video, std::current_exception()});
        }
    }

    // Runs one FFmpeg command into the ring, with a file of its timestamps
    // when pairing by timestamp: the frames it gave, its exit code, and the
    // end of its error output.
    std::tuple<long long, int, std::string> decode(const std::vector<std::string>& command) {
        if (!timestamps_) {
            return decode_with(command, nullptr);
        }
        // A new, empty file for this FFmpeg's timestamps, deleted with it.
        const TempFile stamps_file = TempFile::make<VshipUnavailableError>(
            scratch_folder_, "vml-vship-pts-", ".txt", "A file for FFmpeg's timestamps could not be made.");
        return decode_with(with_timestamps(command, stamps_file.path()), &stamps_file.path());
    }

    std::tuple<long long, int, std::string> decode_with(const std::vector<std::string>& command,
                                                        const std::filesystem::path* stamps_path) {
        platform::ProcessOptions options;
        options.output_pipe_bytes = frame_pipe_bytes;
        std::optional<platform::ChildProcess> process;
        try {
            process.emplace(platform::ChildProcess::start(command, options));
        } catch (const platform::ProcessError& error) {
            throw VshipUnavailableError("Could not start FFmpeg for Vship: " + std::string(error.what()));
        }
        {
            const std::lock_guard lock(process_mutex_);
            process_ = &*process;
            if (stopping_) {
                process->terminate();
            }
        }
        std::optional<Timestamps> stamps;
        if (stamps_path != nullptr) {
            stamps.emplace(*stamps_path, *process, label_);
        }
        long long frames = 0;
        const bool y4m = std::find(command.begin(), command.end(), "yuv4mpegpipe") != command.end();
        std::optional<std::string> cut;  // how much came of a frame FFmpeg ended partway through
        std::exception_ptr failure;
        try {
            bool reading = !y4m || y4m_header(*process);  // no header: FFmpeg wrote nothing, its code says why
            while (reading && !stopping_) {
                const int slot = free_.get();
                if (slot == end_of_video) {
                    break;
                }
                process->gather(frame_bytes_ + (y4m ? y4m_frame_marker.size() : 0), stopping_);
                if (y4m && !y4m_frame(*process, cut)) {
                    free_.put(slot);
                    break;
                }
                const std::size_t received = fill(*process, slot);
                if (received == frame_bytes_) {
                    if (stamps) {
                        try {
                            const long long pts = stamps->next();
                            stamp(slot, pts, stamps->time_base());
                        } catch (...) {
                            free_.put(slot);
                            throw;
                        }
                    }
                    filled_.put({slot, nullptr});
                    ++frames;
                    continue;
                }
                free_.put(slot);
                if (received > 0 || y4m) {  // a "FRAME" line with no frame after it
                    cut = std::to_string(received) + " of its " + std::to_string(frame_bytes_) + " bytes";
                }
                reading = false;
            }
        } catch (...) {
            failure = std::current_exception();
            process->terminate();
        }
        // An FFmpeg that ended partway through a frame has closed its output.
        // It is given a while to exit by itself, so that the exit code kept
        // is its own, but not for ever: it could still be running.
        bool still_running = false;
        if (cut && !failure && !process->wait_for(partial_frame_exit_wait)) {
            still_running = true;
            process->terminate();
        }
        const int code = process->wait();
        // The end of FFmpeg's error output, as the Python app keeps it.
        std::string errors = process->standard_error();
        if (errors.size() > 2000) {
            errors.erase(0, errors.size() - 2000);
        }
        {
            const std::lock_guard lock(process_mutex_);
            process_ = nullptr;
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        errors = std::string(python_strip(replace_invalid_utf8(errors)));
        if (cut) {
            log_partial_frame(command, frames, *cut, still_running, code, process->output_error(), errors);
            throw VshipInputError("FFmpeg ended partway through a " + label_ + " frame.");
        }
        return {frames, code, errors};
    }

    // Logs what is known of an FFmpeg that ended partway through a frame.
    // The pass's own message stays the one the window has a translation for.
    void log_partial_frame(const std::vector<std::string>& command, long long frames, const std::string& cut,
                           bool still_running, int code, int output_error, const std::string& errors) const {
        const bool hardware = std::find(command.begin(), command.end(), "-hwaccel") != command.end();
        const std::string ended =
            still_running ? "still running " + std::to_string(partial_frame_exit_wait.count()) + " s later, so ended"
                          : "exit code " + exit_code_for_log(code);
        const std::string output = output_error == 0 ? "FFmpeg closed its output"
                                                     : "reading its output failed (system error " +
                                                           std::to_string(output_error) + ")";
        logger.error("Vship: FFmpeg ended partway through the " + label_ + "'s frame " + std::to_string(frames) +
                     " (" + cut + "), decoding " + (hardware ? "on the GPU" : "in software") + "; " + output +
                     "; " + ended + (errors.empty() ? "" : "; FFmpeg said: " + errors));
    }

    // Reads a YUV4MPEG stream's first line; false if there is no stream.
    bool y4m_header(platform::ChildProcess& process) {
        std::string line;
        std::uint8_t byte = 0;
        while (process.read(&byte, 1) == 1) {
            if (byte == '\n') {
                if (!line.starts_with("YUV4MPEG2 ")) {
                    break;
                }
                return true;
            }
            line += static_cast<char>(byte);
            if (line.size() > 1024) {
                break;
            }
        }
        if (!line.empty()) {
            throw VshipUnavailableError("FFmpeg's " + label_ + " frames are not a YUV4MPEG stream.");
        }
        return false;
    }

    // Reads the "FRAME" line before a frame; false at the stream's end, with
    // `cut` set if FFmpeg ended partway through the line.
    bool y4m_frame(platform::ChildProcess& process, std::optional<std::string>& cut) {
        std::array<std::uint8_t, y4m_frame_marker.size()> marker{};
        const std::size_t received = read_exact(process, marker.data(), marker.size());
        if (received == 0) {
            return false;
        }
        if (received < marker.size()) {
            cut = std::to_string(received) + " of the " + std::to_string(marker.size()) + " bytes of its FRAME line";
            return false;
        }
        if (std::memcmp(marker.data(), y4m_frame_marker.data(), marker.size()) != 0) {
            throw VshipUnavailableError("FFmpeg's " + label_ + " frames are not where they were expected.");
        }
        return true;
    }

    // Reads one frame into `slot`, and returns the number of bytes read. A
    // frame sent as NV12 or P010 has its luma read straight into the slot,
    // and its U and V pairs into staging memory, from which they are split
    // into the slot's U and V planes.
    std::size_t fill(platform::ChildProcess& process, int slot) {
        std::uint8_t* frame = buffers_[static_cast<std::size_t>(slot)]->data();
        if (!split_) {
            return read_exact(process, frame, frame_bytes_);
        }
        const std::size_t received = read_exact(process, frame, split_->luma);
        if (received < split_->luma) {
            return received;
        }
        const std::size_t chroma = read_exact(process, staging_.data(), staging_.size());
        if (chroma == staging_.size()) {
            std::uint8_t* u = frame + split_->luma;
            std::uint8_t* v = u + split_->plane;
            const std::uint8_t* pair = staging_.data();
            if (split_->wide) {
                for (std::size_t at = 0; at < split_->plane; at += 2, pair += 4) {
                    u[at] = pair[0];
                    u[at + 1] = pair[1];
                    v[at] = pair[2];
                    v[at + 1] = pair[3];
                }
            } else {
                for (std::size_t at = 0; at < split_->plane; ++at, pair += 2) {
                    u[at] = pair[0];
                    v[at] = pair[1];
                }
            }
        }
        return received + chroma;
    }

    std::size_t frame_bytes_;
    std::vector<std::vector<std::string>> commands_;
    std::string label_;
    std::optional<InterleavedChroma> split_;
    std::function<void()> on_software_;
    bool timestamps_;
    std::filesystem::path scratch_folder_;
    std::vector<std::uint8_t> staging_;
    std::atomic<bool> stopping_{false};
    std::mutex process_mutex_;
    platform::ChildProcess* process_ = nullptr;  // the FFmpeg running, if one is
};

// One video decoded in this process by the GPU's decoder or the software
// decoder (gpu_frames.h), feeding a ring as FrameStream does (the Python
// app's _NativeFrameStream): each picture FFmpeg's chain would send goes
// straight into a free slot, copied there by the GPU from NVIDIA's decoder,
// or by the CPU from the others. FFmpeg's chain numbers the frames, keeps
// every `step`-th, and its -t keeps the first `limit` numbered ones (none:
// all). When pairing by timestamp (`by_time`), its select keeps every
// step-th picture, setpts starts their times at 0, and -t (`time_limit`'s
// text) is FFmpeg's trim filter: a picture is kept while its time is below
// the limit in the stream's time base, and the output ends at the first
// that is not (the Python app's _TimedFrameSelection). A decoding failure
// after the start is thrown from next() as GpuDecodeFailedError, and the
// pass is then made again through FFmpeg.
class NativeFrameStream : public FrameRing {
public:
    NativeFrameStream(const VshipLibrary& library, std::size_t frame_bytes, std::unique_ptr<GpuFrameStream> decoder,
                      int step, std::optional<long long> limit, bool by_time, std::optional<std::string> time_limit,
                      int gpu_id)
        : FrameRing(library, frame_bytes, gpu_id),
          decoder_(std::move(decoder)),
          frame_bytes_(frame_bytes),
          step_(step),
          limit_(limit),
          by_time_(by_time),
          time_limit_(std::move(time_limit)) {}

    ~NativeFrameStream() override { close(); }

    void start() override {
        decoder_->start();
        // NVIDIA's decoder copies each picture with the GPU: straight into
        // Vship's memory once CUDA has page-locked it too; else CUDA stages
        // the copy itself.
        if (decoder_->backend() == "nvidia") {
            for (const std::unique_ptr<PinnedBuffer>& buffer : buffers_) {
                if (decoder_->pin(buffer->data(), frame_bytes_)) {
                    pinned_.push_back(buffer->data());
                }
            }
            logger.info("Vship: " + std::to_string(pinned_.size()) + " of " + std::to_string(buffers_.size()) +
                        " frame buffers page-locked for NVIDIA's decoder");
        }
        thread_ = std::thread([this] { run(); });
    }

    void close() override {
        stopping_ = true;
        stop_.cancel();  // ends the decoder's wait for its next picture
        free_.put(end_of_video);  // wakes a copier waiting for a slot
        if (thread_.joinable()) {
            thread_.join();
        }
        for (void* host : pinned_) {
            decoder_->unpin(host);
        }
        pinned_.clear();
        decoder_->close();
    }

private:
    // What FFmpeg's chain does with a decoded picture.
    enum class Taken { piped, skipped, ended };

    // The selection of the picture numbered `at`, stamped `pts`.
    Taken select(long long at, long long pts) {
        if (!by_time_) {
            if (limit_ && at >= *limit_) {
                return Taken::ended;
            }
            return at % step_ == 0 ? Taken::piped : Taken::skipped;
        }
        if (at % step_ != 0) {
            return Taken::skipped;
        }
        if (!first_pts_) {
            first_pts_ = pts;
            decoder_time_base_ = decoder_->time_base(stop_);
            if (time_limit_) {
                time_limit_ticks_ = duration_in(*time_limit_, *decoder_time_base_);
            }
        }
        return time_limit_ticks_ && pts - *first_pts_ >= *time_limit_ticks_ ? Taken::ended : Taken::piped;
    }

    // The copying thread: copies each selected picture into a free slot.
    void run() {
        try {
            long long number = 0;  // each decoded picture's number, as FFmpeg's chain numbers it
            while (!stopping_) {
                const std::optional<GpuFrameStream::Picture> picture = decoder_->next(stop_);
                if (!picture) {
                    break;  // the end, checked
                }
                const Taken taken = select(number++, picture->pts);
                if (taken == Taken::ended) {
                    decoder_->release(picture->slot);  // the output ends here
                    break;
                }
                if (taken == Taken::skipped) {
                    decoder_->release(picture->slot);
                    continue;
                }
                const int slot = free_.get();
                if (slot == end_of_video) {
                    decoder_->release(picture->slot);
                    return;
                }
                try {
                    decoder_->download(picture->slot, buffers_[static_cast<std::size_t>(slot)]->data());
                } catch (...) {
                    decoder_->release(picture->slot);
                    throw;
                }
                decoder_->release(picture->slot);
                if (by_time_) {
                    stamp(slot, picture->pts, *decoder_time_base_);
                }
                filled_.put({slot, nullptr});
            }
            filled_.put({end_of_video, nullptr});
        } catch (...) {  // handed to the scoring thread, never lost
            filled_.put({end_of_video, std::current_exception()});
        }
    }

    std::unique_ptr<GpuFrameStream> decoder_;
    std::size_t frame_bytes_;
    std::vector<void*> pinned_;  // the buffers CUDA page-locked, unpinned before the decoder closes
    int step_;
    std::optional<long long> limit_;
    bool by_time_;
    std::optional<std::string> time_limit_;
    // By timestamp: the first piped picture's timestamp, the stream's time
    // base, and the limit in it.
    std::optional<long long> first_pts_;
    std::optional<Rational> decoder_time_base_;
    std::optional<long long> time_limit_ticks_;
    std::atomic<bool> stopping_{false};
    CancelToken stop_;  // set by close()
};

// ------------------------------------------------------------------- lanes

// Each frame's score, by frame index, as the lanes finish them out of
// order. Kept in chunks, so that a long video's scores never move under a
// lane writing one; the feeding thread adds a chunk before any lane is given
// an index in it.
class ScoreArray {
public:
    void reserve(long long index) {
        const std::lock_guard lock(mutex_);
        while (static_cast<std::size_t>(index) >= chunks_.size() * chunk) {
            chunks_.push_back(std::make_unique<std::array<double, chunk>>());
            chunks_.back()->fill(std::nan(""));
        }
    }

    void set(long long index, double value) {
        const std::lock_guard lock(mutex_);
        (*chunks_[static_cast<std::size_t>(index) / chunk])[static_cast<std::size_t>(index) % chunk] = value;
    }

    // The first `count` scores, as 32-bit floats.
    std::vector<float> values(long long count) const {
        const std::lock_guard lock(mutex_);
        std::vector<float> found;
        found.reserve(static_cast<std::size_t>(count));
        for (std::size_t index = 0; index < static_cast<std::size_t>(count); ++index) {
            found.push_back(static_cast<float>((*chunks_[index / chunk])[index % chunk]));
        }
        return found;
    }

private:
    static constexpr std::size_t chunk = 8192;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<std::array<double, chunk>>> chunks_;
};

// A frame pair handed to a lane: its index, and the two streams' slots.
struct LaneJob {
    long long index = 0;
    int source_slot = 0;
    int distorted_slot = 0;
};

using Planes = std::vector<std::array<const std::uint8_t*, 3>>;  // each slot's planes

// What a lane scores with: Vship's library and GPU, each video's colour
// space, the planes and strides of the streams' ring slots, whom to tell
// once it is done with a pair, and the pass's abort flag.
struct LaneSetup {
    const VshipLibrary& library;
    const VshipDevice& device;
    vship_api::Colorspace source;
    vship_api::Colorspace distorted;
    const Planes& source_planes;
    const Planes& distorted_planes;
    std::array<std::int64_t, 3> source_strides;
    std::array<std::int64_t, 3> distorted_strides;
    std::function<void(long long)> finished;  // given a pair's index
    const std::atomic<bool>& abort;
};

// One Vship handler on a thread of its own, scoring the pairs it is given.
// A lane whose handler fails keeps handing its pairs back unscored, so that
// the other metrics can finish.
class Lane {
public:
    explicit Lane(LaneSetup setup) : setup_(std::move(setup)) {}
    // Each kind of lane stops and joins its thread as it is destroyed: the
    // thread runs that kind's run().
    virtual ~Lane() = default;
    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;

    void start() {
        thread_ = std::thread([this] { run(); });
    }

    // Hands the lane a pair to score.
    void give(const LaneJob& job) { jobs_.put(job); }

    // Tells the lane to stop after the pairs already given.
    void stop() { jobs_.put(std::nullopt); }

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    // The handler's failure, if it failed.
    std::exception_ptr error() const {
        const std::lock_guard lock(error_mutex_);
        return error_;
    }

protected:
    void fail(std::exception_ptr error) {
        const std::lock_guard lock(error_mutex_);
        error_ = std::move(error);
    }

    // The pair's planes, in the order Vship reads them.
    const std::uint8_t* const* source_planes(const LaneJob& job) const {
        return setup_.source_planes[static_cast<std::size_t>(job.source_slot)].data();
    }
    const std::uint8_t* const* distorted_planes(const LaneJob& job) const {
        return setup_.distorted_planes[static_cast<std::size_t>(job.distorted_slot)].data();
    }

    virtual void run() = 0;

    const LaneSetup setup_;
    Channel<std::optional<LaneJob>> jobs_;

private:
    mutable std::mutex error_mutex_;
    std::exception_ptr error_;
    std::thread thread_;
};

// One of SSIMULACRA2's or Butteraugli's handlers (the Python app's
// _MetricLane). Frame pairs are dealt to a metric's lanes in turn; each lane
// writes its score by frame index, so the order they finish in does not
// matter.
class MetricLane : public Lane {
public:
    MetricLane(LaneSetup setup, std::string key, ScoreArray& scores)
        : Lane(std::move(setup)), key_(std::move(key)), scores_(scores) {}

    ~MetricLane() override {
        stop();
        join();
    }

    const std::string& key() const { return key_; }

private:
    void run() override {
        void* handler = nullptr;
        try {
            handler = make_handler();
        } catch (...) {
            fail(std::current_exception());
        }
        while (true) {
            const std::optional<LaneJob> job = jobs_.get();
            if (!job || setup_.abort) {
                break;
            }
            if (!error()) {
                try {
                    scores_.set(job->index, compute(handler, *job));
                } catch (...) {
                    fail(std::current_exception());
                }
            }
            setup_.finished(job->index);
        }
        if (handler != nullptr) {
            setup_.library.free_handler(handler);
        }
    }

    // A Vship handler for SSIMULACRA2 or Butteraugli.
    void* make_handler() {
        void* handler = nullptr;
        int error = 0;
        if (key_ == "ssimulacra2") {
            vship_api::InitSsimulacra2 init;
            init.src_colorspace = setup_.source;
            init.dis_colorspace = setup_.distorted;
            init.gpu_id = setup_.device.gpu_id;
            error = setup_.library.init_handler(&handler, &init);
        } else {
            // Vship's defaults (doc/BUTTERAUGLI.md): the 2-norm, reported
            // beside the 3-norm the app shows, for a 203-nit display.
            vship_api::InitButteraugli init;
            init.src_colorspace = setup_.source;
            init.dis_colorspace = setup_.distorted;
            init.Qnorm = 2;
            init.intensity_multiplier = 203.0f;
            init.gpu_id = setup_.device.gpu_id;
            error = setup_.library.init_handler(&handler, &init);
        }
        if (error != 0) {
            throw VshipUnavailableError("Could not initialize Vship " + key_ + ": " +
                                        setup_.library.message(std::nullopt));
        }
        return handler;
    }

    // Scores one frame pair with the handler.
    double compute(void* handler, const LaneJob& job) {
        const VshipLibrary& library = setup_.library;
        const std::string label = key_ == "ssimulacra2" ? "SSIMULACRA2" : "Butteraugli";
        double value = 0.0;
        int error = 0;
        if (key_ == "ssimulacra2") {
            vship_api::ScoreSsimulacra2 score;
            error = library.compute_handler(handler, &score, source_planes(job), distorted_planes(job),
                                            setup_.source_strides.data(), setup_.distorted_strides.data());
            value = score.score;
        } else {
            vship_api::ScoreButteraugli score;  // dstp nullptr: no distortion map
            error = library.compute_handler(handler, &score, source_planes(job), distorted_planes(job),
                                            setup_.source_strides.data(), setup_.distorted_strides.data());
            value = score.norm3;
        }
        if (error != 0) {
            throw VshipUnavailableError("Vship " + label + " failed: " + library.handler_error(handler, error));
        }
        if (!std::isfinite(value)) {
            throw VshipUnavailableError("Vship " + label + " returned a non-finite value.");
        }
        return value;
    }

    std::string key_;
    ScoreArray& scores_;
};

// CVVDP's one handler, fed every frame pair in order (the Python app's
// _CvvdpLane). CVVDP judges each frame together with the ones before it, so
// its pairs cannot be dealt in turn to two handlers. Its score is reset at
// each new second of video, which gives each second's JOD (the timeline),
// and the whole video's JOD is pooled from those seconds
// (pool_cvvdp_windows).
class CvvdpLane : public Lane {
public:
    CvvdpLane(LaneSetup setup, CvvdpSettings settings, double fps)
        : Lane(std::move(setup)), settings_(std::move(settings)), fps_(fps) {}

    ~CvvdpLane() override {
        stop();
        join();
    }

    // Each second scored, once the lane has stopped.
    const std::vector<CvvdpWindow>& windows() const { return windows_; }

private:
    void run() override {
        void* handler = nullptr;
        try {
            handler = make_handler();
        } catch (...) {
            fail(std::current_exception());
        }
        long long start = 0;
        long long count = 0;
        long long second = 0;
        double jod = std::nan("");
        while (true) {
            const std::optional<LaneJob> job = jobs_.get();
            if (!job || setup_.abort) {
                break;
            }
            if (!error()) {
                try {
                    const auto this_second = static_cast<long long>(static_cast<double>(job->index) / fps_);
                    if (count != 0 && this_second != second) {
                        windows_.push_back({start, count, jod});
                        reset(handler);
                        start = job->index;
                        count = 0;
                    }
                    second = this_second;
                    jod = compute(handler, *job);
                    ++count;
                } catch (...) {
                    fail(std::current_exception());
                }
            }
            setup_.finished(job->index);
        }
        if (!error() && count != 0) {
            windows_.push_back({start, count, jod});
        }
        if (handler != nullptr) {
            setup_.library.free_handler(handler);
        }
    }

    // A Vship handler for CVVDP, with the display model and the frame rate.
    void* make_handler() {
        if (!(std::isfinite(fps_) && fps_ > 0)) {
            // CVVDP models how the eye integrates over time, so the frame
            // rate is part of the score, and must not be quietly replaced by
            // a guess.
            throw VshipUnavailableError("CVVDP needs the video's frame rate, and none was found (it was read as " +
                                        format_general(fps_) + " fps).");
        }
        // The display as JSON text, which Vship reads itself when it starts
        // with "{". Not a file: Vship opens one with a narrow-character path,
        // which fails for a Windows user name outside the ANSI code page.
        const std::string config = vship_display_json(settings_.display);
        vship_api::InitCvvdp init;
        init.src_colorspace = setup_.source;
        init.dis_colorspace = setup_.distorted;
        init.fps = static_cast<float>(fps_);
        init.resizeToDisplay = settings_.resize_to_display;
        init.model_key_cstr = vship_model_key;
        init.model_config_json_cstr = config.c_str();
        init.gpu_id = setup_.device.gpu_id;
        void* handler = nullptr;
        if (setup_.library.init_handler(&handler, &init) != 0) {
            throw VshipUnavailableError("Could not initialize Vship CVVDP: " + setup_.library.message(std::nullopt));
        }
        return handler;
    }

    // Scores the next frame pair: the JOD of the frames since the last reset.
    double compute(void* handler, const LaneJob& job) {
        const VshipLibrary& library = setup_.library;
        vship_api::ScoreCvvdp score;  // dstp nullptr: no distortion map
        const int error = library.compute_handler(handler, &score, source_planes(job), distorted_planes(job),
                                                  setup_.source_strides.data(), setup_.distorted_strides.data());
        if (error != 0) {
            throw VshipUnavailableError("Vship CVVDP failed: " + library.handler_error(handler, error));
        }
        if (!std::isfinite(score.score)) {
            throw VshipUnavailableError("Vship CVVDP returned a non-finite value.");
        }
        return score.score;
    }

    // Starts a new second: clears the score's sum, not the handler's history.
    void reset(void* handler) {
        if (const int error = setup_.library.reset_score(handler); error != 0) {
            throw VshipUnavailableError("Vship CVVDP failed: " + setup_.library.handler_error(handler, error));
        }
    }

    CvvdpSettings settings_;
    double fps_;
    std::vector<CvvdpWindow> windows_;
};

// A pass's frame rate for its progress reports, timed from its first frame
// pair rather than from the pass's start (FFmpeg opening both videos, the
// first 4K frames, Vship's handlers set up), which says nothing about its
// speed. The rate is in the video's frames, as the progress is: with frame
// subsampling, each pair covers `step` frames.
class PassRate {
public:
    explicit PassRate(int step) : step_(step) {}

    double frames_per_second(long long pairs) {
        const auto now = std::chrono::steady_clock::now();
        if (!first_at_) {
            first_at_ = now;
            first_pairs_ = pairs;
            return 0.0;
        }
        const double span = std::chrono::duration<double>(now - *first_at_).count();
        return span >= warmup_seconds ? static_cast<double>(pairs - first_pairs_) * step_ / span : 0.0;
    }

private:
    static constexpr double warmup_seconds = 0.5;
    int step_;
    std::optional<std::chrono::steady_clock::time_point> first_at_;
    long long first_pairs_ = 0;
};

// ------------------------------------------------------------------- a pass

// How one video of a pass is decoded and handed to Vship.
struct VideoPlan {
    VideoInfo info;
    std::optional<CropBox> crop;
    std::string side;   // "source" or "distorted"
    std::string label;  // "reference" or "test video", for messages
    std::optional<std::string> hwaccel;  // how FFmpeg decodes it; nothing for software
    std::optional<std::pair<VshipImageFormat, std::string>> passthrough;
    VshipImageFormat image;
    std::pair<int, int> size;  // the size it is compared at
    vship_api::Colorspace colour;
    VshipImageFormat::Layout layout;
};

// What a pass is given, beside the request: its metrics, the GPU, the
// black-bar crops, each video's decoder, and whether it runs on stock Vship
// 5.1.2 (the fallback after Vship-fast failed) rather than Vship-fast.
struct PassPlan {
    std::vector<std::string> keys;
    VshipDevice device;
    std::optional<CropBox> source_crop;
    std::optional<CropBox> distorted_crop;
    HardwareDecoding decoding;
    bool vship51 = false;
    // Where the pass's own files go: the folder of its process's task, which
    // the run removes whatever becomes of the process; or, in the run's
    // process, the system's temporary folder.
    std::filesystem::path scratch_folder = std::filesystem::temp_directory_path();
};

// One pass of Vship's metrics over a pair of videos (the Python app's
// _VshipPass): both videos are decoded once, their frames paired, and each
// pair handed to every metric's lanes and then given back to its decoder. A
// pair's two slots are counted: a slot goes back to its stream once the
// pairing and every metric are done with it.
class VshipPass {
public:
    // `native` decodes a video in this process where a decoder of the app's
    // takes it; `every_source_frame`, when pairing by timestamp, takes every
    // source frame.
    VshipPass(const PerceptualRequest& request, const PassPlan& plan, bool native, bool every_source_frame,
              const ProgressCallback& on_progress, const StatusCallback& on_status, const CancelToken& cancel)
        : request_(request),
          plan_(plan),
          native_(native),
          on_progress_(on_progress),
          on_status_(on_status),
          cancel_(cancel),
          library_(vship_library(plan.vship51)),
          decode_(plan.decoding) {
        if (plan.keys.empty() || !std::all_of(plan.keys.begin(), plan.keys.end(), is_perceptual_metric)) {
            throw std::invalid_argument("A Vship pass scores SSIMULACRA2, Butteraugli and CVVDP only.");
        }
        validate_pair(request.source, request.distorted, request.options.duration_limit);
        cancel.throw_if_cancelled();
        plan_videos();
        step_ = request.step_of(plan.keys.front());
        for (const std::string& key : plan.keys) {
            if (request.step_of(key) != step_) {
                throw VshipUnavailableError("Perceptual metrics in one task must use the same frame coverage.");
            }
        }
        for (const std::string& key : plan.keys) {
            if (key == "cvvdp") {
                // The display as a saved score says it was scored on: its
                // identity's rounded values.
                cvvdp_ = CvvdpSettings::from_parameters(nlohmann::json(request.cvvdp.parameters()));
            } else {
                frame_keys_.push_back(key);
            }
        }
        if (cvvdp_ && step_ != 1) {
            throw VshipUnavailableError("CVVDP scores every frame; it cannot be subsampled.");
        }
        // Pairing by frame number, both videos' frames are numbered at the
        // test video's frame rate (frame_sync.h) and both are cut at the
        // duration limit, so they keep the same frames. Pairing by
        // timestamp, only the test video is cut.
        by_time_ = by_timestamp();
        numbering_rate_ = frame_rate(request.distorted);
        const double limit = request.options.duration_limit;
        if (limit > 0) {
            duration_limit_ = format_fixed(limit, 6);
        }
        expected_frames_ =
            std::min(request.source.estimated_frame_count(), request.distorted.estimated_frame_count());
        if (by_time_ && limit > 0) {
            expected_frames_ = std::min(
                expected_frames_, std::max(1LL, static_cast<long long>(std::ceil(limit * request.source.fps))));
        } else if (duration_limit_) {
            expected_frames_ = std::min(expected_frames_, std::max(1LL, *frames_within(duration_limit_, numbering_rate_)));
        }
        total_units_ = std::max(1LL, (expected_frames_ + step_ - 1) / step_) * step_;
        // Pairing by timestamp, the source gives the frames among which each
        // test frame's pair is found by time. So it is never cut at the
        // limit, and it is subsampled as the test video is while the two
        // line up (FramesApartError). A source without a frame rate gives
        // every frame.
        every_source_frame_ = by_time_ && (every_source_frame || request.source.fps <= 0);
        check_apart_ = by_time_ && step_ > 1 && !every_source_frame_;
        for (const std::string& key : frame_keys_) {
            scores_.emplace(key, std::make_unique<ScoreArray>());
        }
    }

    // Makes the pass and returns its scores. A metric whose lanes fail is in
    // the output's failures while the others keep their scores; the pass
    // fails only when every metric does.
    PerceptualTaskOutput run() {
        const auto started = std::chrono::steady_clock::now();
        try {
            const PerceptualTaskOutput output = run_pass(started);
            close();
            return output;
        } catch (const Cancelled&) {
            close();
            throw;
        } catch (const ComparisonCutShortError&) {
            // A video cut short is the file's fault, not the GPU's: the CPU
            // would stop as short.
            close();
            throw;
        } catch (const GpuDecodeFailedError&) {
            close();  // made again with FFmpeg decoding (remade_vship_pass)
            throw;
        } catch (const FramesApartError&) {
            close();  // made again with every source frame (remade_vship_pass)
            throw;
        } catch (const VshipInputError& error) {
            logger.error("Vship pass (" + metric_labels(plan_.keys, ", ") + ") failed: " + error.what());
            // Made again the same way (remade_vship_pass), but not where a
            // metric's lane failed too: then the GPU may be what failed (a
            // reset ends FFmpeg's GPU decoding as well), and nothing is run
            // on it again.
            stop_lanes();
            const bool lanes_failed = !lane_errors().empty();
            close();
            if (lanes_failed) {
                throw VshipUnavailableError(error.what());
            }
            throw;
        } catch (const VshipUnavailableError& error) {
            logger.error("Vship pass (" + metric_labels(plan_.keys, ", ") + ") failed: " + error.what());
            close();
            throw;
        } catch (const std::exception& error) {
            logger.error("Vship pass (" + metric_labels(plan_.keys, ", ") + ") failed: " + error.what());
            close();
            throw VshipUnavailableError(std::string("Vship GPU calculation failed: ") + error.what());
        }
    }

private:
    // How each video is decoded, sized and laid out for Vship. Decoding
    // follows the run's GPU decode setting, per video and per codec;
    // hardware decoding is bit-exact, so it changes only the speed. A video
    // without chroma siting is read with the other's (shared_chroma_location),
    // and one without a colour matrix with the other's, or the reference's
    // guess (with_shared_matrix).
    void plan_videos() {
        const auto [source_size, distorted_size] =
            vship_scaled_sizes(request_.source, request_.distorted, request_.options.scale_direction,
                               plan_.source_crop, plan_.distorted_crop);
        const auto [source_siting, distorted_siting] =
            shared_chroma_location(request_.source.chroma_location, request_.distorted.chroma_location);
        if (chroma_location_shared(request_.source.chroma_location, request_.distorted.chroma_location)) {
            logger.info("Vship: one video gives no chroma siting; both are read as " + source_siting);
        }
        const auto [source_read, distorted_read] = with_shared_matrix(request_.source, request_.distorted);
        if (matrix_shared(request_.source, request_.distorted)) {
            logger.info("Vship: the videos are read with one colour matrix, " + source_read.color_space);
        }
        for (const bool source : {true, false}) {
            VideoPlan& video = videos_[source ? 0 : 1];
            video.info = source ? source_read : distorted_read;
            video.crop = source ? plan_.source_crop : plan_.distorted_crop;
            video.side = source ? "source" : "distorted";
            video.label = source ? "reference" : "test video";
            video.hwaccel = source ? plan_.decoding.source : plan_.decoding.distorted;
            video.size = source ? source_size : distorted_size;
            video.passthrough = passthrough_format(video.info, video.hwaccel);
            video.image = video.passthrough ? video.passthrough->first : vship_image_format(video.info);
            video.layout = video.image.frame_layout(video.size.first, video.size.second);
            VideoInfo described = video.info;
            described.chroma_location = source ? source_siting : distorted_siting;
            video.colour = vship_colorspace(described, video.image, video.size.first, video.size.second);
        }
    }

    PerceptualTaskOutput run_pass(std::chrono::steady_clock::time_point started) {
        // One at a time, so that if the test video's buffers cannot be
        // allocated, the source's stream is already kept and freed.
        for (const VideoPlan& video : videos_) {
            streams_.push_back(open_stream(video));
        }
        if (on_status_) {
            const std::lock_guard lock(decode_mutex_);
            on_status_(decoding_status("Vship GPU (" + plan_.device.name + "): calculating " +
                                           metric_labels(plan_.keys, ", "),
                                       decode_, StatusKind::step, "\xE2\x80\xA6"));
        }
        for (const std::unique_ptr<FrameRing>& stream : streams_) {
            stream->start();
        }
        start_lanes();
        PassRate rate(step_);
        const long long frames = feed_pairs(rate);
        stop_lanes();
        return finish(frames, rate, started);
    }

    // The lanes first: they read the pinned frames the streams own, so the
    // streams may only free that memory once no lane can touch it.
    void close() {
        abort_ = true;
        stop_lanes();
        lanes_.clear();
        cvvdp_lane_.reset();
        for (const std::unique_ptr<FrameRing>& stream : streams_) {
            stream->close();
        }
    }

    // The decoder of the app's that decodes a video FFmpeg would decode with
    // `hwaccel`, in this process: the GPU's, or the software decoder for a
    // video FFmpeg would decode in software.
    std::optional<std::string> decoded_here(const std::optional<std::string>& hwaccel) const {
        if (!hwaccel) {
            return software_decoder_bundled() ? std::optional(std::string(software_decoder)) : std::nullopt;
        }
        return *hwaccel == "cuda"      ? std::optional<std::string>("nvidia")
               : *hwaccel == "qsv"     ? std::optional<std::string>("intel")
               : *hwaccel == "d3d11va" ? std::optional<std::string>("amd")
                                       : std::nullopt;
    }

    // The decoder of one video, when `backend`'s can give it in the layout
    // Vship is told: 8-bit planes, or P016's 16-bit samples as they are or
    // shifted to 10-bit (as FFmpeg converts full-range 10-bit), scaled where
    // it is to the size compared at with the comparison's filter. Nothing
    // when FFmpeg decodes the video.
    std::unique_ptr<GpuFrameStream> native_decoder(const VideoPlan& video, const std::string& backend) const {
        static const std::map<std::string, int, std::less<>> shifts{
            {"yuv420p", 0}, {"yuv420p16le", 0}, {"yuv420p10le", 6}};
        std::string reason;  // why FFmpeg decodes it, for the log
        const auto shift = shifts.find(video.image.pixel_format);
        if (shift == shifts.end()) {
            reason = "Vship reads it as " + video.image.pixel_format;
        } else {
            try {
                const DecodePlan decode = plan_decode(video.info, video.crop, shift->second, false, video.size,
                                                      request_.options.scale_algorithm);
                if (decode.frame_bytes() != video.layout.frame_bytes) {
                    throw GpuDecodeUnavailableError("its frames would be " + std::to_string(decode.frame_bytes()) +
                                                    " bytes, not " + std::to_string(video.layout.frame_bytes));
                }
                // NVIDIA's decoder on CUDA's first GPU, where FFmpeg's
                // -hwaccel cuda decodes too: Vship's number is Vulkan's, a
                // list in another order.
                const int device = backend == "nvidia" ? 0 : plan_.device.gpu_id;
                // Asked first, because a stream the decoder refuses once the
                // pass has started (10-bit H.264, say) means making the
                // whole pass again.
                const auto [supported, refusal] = decoder_supports(device, decode, backend);
                if (!supported) {
                    throw GpuDecodeUnavailableError(refusal);
                }
                auto decoder = std::make_unique<GpuFrameStream>(video.info, decode, device, 4, backend,
                                                                request_.environment.ffmpeg,
                                                                request_.environment.ffprobe);
                logger.info("Vship: the " + video.label + " is decoded by the " + backend +
                            " decoder in the scoring process");
                return decoder;
            } catch (const GpuDecodeUnavailableError& error) {
                reason = error.what();
            }
        }
        logger.info("Vship: the " + video.label + " is decoded by FFmpeg (" + reason + ")");
        return nullptr;
    }

    // One video's frame stream: decoded in this process where it can be, by
    // the decoder FFmpeg would use, else by FFmpeg. Pairing by timestamp,
    // only the test video is cut at the limit, and the source may give every
    // frame.
    std::unique_ptr<FrameRing> open_stream(const VideoPlan& video) {
        const bool cut = video.side == "distorted" || !by_time_;
        const int every = video.side == "source" && every_source_frame_ ? 1 : step_;
        bool said = false;  // native_decoder logs which decoder the video has
        if (native_) {
            if (const std::optional<std::string> backend = decoded_here(video.hwaccel)) {
                said = true;
                if (std::unique_ptr<GpuFrameStream> decoder = native_decoder(video, *backend)) {
                    return std::make_unique<NativeFrameStream>(
                        library_, video.layout.frame_bytes, std::move(decoder), every,
                        by_time_ ? std::nullopt : frames_within(duration_limit_, numbering_rate_), by_time_,
                        by_time_ && cut ? duration_limit_ : std::nullopt, plan_.device.gpu_id);
                }
            }
        }
        if (!said) {
            logger.info("Vship: the " + video.label + " is decoded by FFmpeg " +
                        (video.hwaccel ? "on the GPU (-hwaccel " + *video.hwaccel + ")"
                                       : "in software (" + software_decoding_reason(video.info) + ")"));
        }
        std::optional<InterleavedChroma> split;
        if (video.passthrough) {
            split = InterleavedChroma{video.layout.plane_bytes[0], video.layout.plane_bytes[1], video.image.wide()};
        }
        const std::string pixel_format = video.passthrough ? video.passthrough->second : video.image.pixel_format;
        const std::string side = video.side;
        return std::make_unique<FrameStream>(library_, video.layout.frame_bytes,
                                             ffmpeg_commands(video, pixel_format, every, cut), video.label, split,
                                             [this, side] { decoded_in_software(side); }, plan_.device.gpu_id,
                                             by_time_, plan_.scratch_folder);
    }

    // Why FFmpeg decodes a video in software, for the log.
    std::string software_decoding_reason(const VideoInfo& info) const {
        if (!request_.options.gpu_decode) {
            return "GPU decoding is off";
        }
        if (!downloads_from_gpu(info.pix_fmt, info.width, info.height)) {
            return "GPU decoding takes 4:2:0 at 8 or 10 bits and an even size, not " + info.pix_fmt + " at " +
                   std::to_string(info.width) + "x" + std::to_string(info.height);
        }
        if (!gpu_decodes_exactly(info.codec_name)) {
            return "the GPUs decode " + info.codec_name + " to other pictures than FFmpeg's";
        }
        if (ascii_lower(info.codec_name) == "h264" && bit_depth(info.pix_fmt) > 8) {
            return "this GPU does not decode H.264 deeper than 8 bits";
        }
        return "no GPU decoder here for " + info.codec_name;
    }

    // The FFmpeg commands to try for one video, keeping every `every`-th
    // frame, cut at the duration limit when `cut`: with hardware decoding
    // first if it has any, then in software. The software retry sends the
    // same layout as the hardware attempt, because Vship's handlers are set
    // up for one layout per video.
    std::vector<std::vector<std::string>> ffmpeg_commands(const VideoPlan& video, const std::string& pixel_format,
                                                          int every, bool cut) {
        std::vector<std::optional<std::string>> attempts{video.hwaccel};
        if (video.hwaccel) {
            attempts.emplace_back(std::nullopt);
        }
        std::vector<std::vector<std::string>> commands;
        for (const std::optional<std::string>& accel : attempts) {
            const std::string chain =
                vship_filter_chain(video.info, video.crop, video.size, pixel_format, every,
                                   request_.options.scale_algorithm, accel,
                                   by_time_ ? std::nullopt : std::optional(numbering_rate_),
                                   video.colour.colorFamily == 0 && video.colour.range == 1);
            std::vector<std::string> command{request_.environment.ffmpeg, "-hide_banner", "-loglevel", "error",
                                             "-nostdin"};
            const std::vector<std::string> decode = hwaccel_args(accel);
            command.insert(command.end(), decode.begin(), decode.end());
            const std::vector<std::string> input = video.info.input_options();
            command.insert(command.end(), input.begin(), input.end());
            command.insert(command.end(), {"-i", input_path(video.info.path), "-map",
                                           "0:" + std::string(video_stream), "-an", "-sn", "-dn", "-vf", chain});
            if (cut) {
                const std::vector<std::string> args = duration_args(duration_limit_);
                command.insert(command.end(), args.begin(), args.end());
            }
            command.insert(command.end(), {"-fps_mode", "passthrough"});
            const std::vector<std::string> piped = vship_piped_as(pixel_format);
            command.insert(command.end(), piped.begin(), piped.end());
            commands.push_back(std::move(command));
        }
        return commands;
    }

    // Reports a video falling back to software decoding. The two videos'
    // threads can both fall back.
    void decoded_in_software(const std::string& side) {
        const std::lock_guard lock(decode_mutex_);
        (side == "source" ? decode_.source : decode_.distorted).reset();
        if (on_status_) {
            on_status_(decoding_status("GPU decode failed for the " +
                                           std::string(side == "distorted" ? "test video" : side) +
                                           ", decoding it in software",
                                       decode_, StatusKind::step, "\xE2\x80\xA6"));
        }
    }

    // Starts CVVDP's lane, if it is asked for, and each frame metric's lanes,
    // on the planes of the streams' buffers.
    void start_lanes() {
        const VideoPlan& source = videos_[0];
        const VideoPlan& distorted = videos_[1];
        for (const std::unique_ptr<PinnedBuffer>& buffer : streams_.at(0)->buffers()) {
            source_planes_.push_back(buffer->planes(source.image, source.layout.plane_bytes));
        }
        for (const std::unique_ptr<PinnedBuffer>& buffer : streams_[1]->buffers()) {
            distorted_planes_.push_back(buffer->planes(distorted.image, distorted.layout.plane_bytes));
        }
        const auto setup = [&] {
            return LaneSetup{library_,
                             plan_.device,
                             source.colour,
                             distorted.colour,
                             source_planes_,
                             distorted_planes_,
                             source.layout.strides,
                             distorted.layout.strides,
                             [this](long long index) { pair_done(index); },
                             abort_};
        };
        if (cvvdp_) {
            cvvdp_lane_ = std::make_unique<CvvdpLane>(setup(), *cvvdp_, request_.source.fps);
        }
        for (const std::string& key : frame_keys_) {
            for (std::size_t lane = 0; lane < lanes_per_metric; ++lane) {
                lanes_.push_back(std::make_unique<MetricLane>(setup(), key, *scores_.at(key)));
            }
        }
        for (Lane* lane : every_lane()) {
            lane->start();
        }
    }

    // CVVDP's lane, if it has one, then the frame metrics'.
    std::vector<Lane*> every_lane() const {
        std::vector<Lane*> found;
        if (cvvdp_lane_) {
            found.push_back(cvvdp_lane_.get());
        }
        for (const std::unique_ptr<MetricLane>& lane : lanes_) {
            found.push_back(lane.get());
        }
        return found;
    }

    void stop_lanes() {
        const std::vector<Lane*> lanes = every_lane();
        for (Lane* lane : lanes) {
            lane->stop();
        }
        for (Lane* lane : lanes) {
            lane->join();
        }
    }

    // Each failed metric's first error: the frame metrics', then CVVDP's.
    // One failed lane fails its metric, because the other lane's frames
    // alone would leave holes.
    std::vector<std::pair<std::string, std::exception_ptr>> lane_errors() const {
        std::vector<std::pair<std::string, std::exception_ptr>> errors;
        for (const std::string& key : frame_keys_) {
            for (const std::unique_ptr<MetricLane>& lane : lanes_) {
                if (lane->key() == key) {
                    if (std::exception_ptr error = lane->error()) {
                        errors.emplace_back(key, error);
                        break;
                    }
                }
            }
        }
        if (cvvdp_lane_) {
            if (std::exception_ptr error = cvvdp_lane_->error()) {
                errors.emplace_back("cvvdp", error);
            }
        }
        return errors;
    }

    // ------------------------------------------------------------ the slots

    void hold(std::size_t side, int slot) {
        const std::lock_guard lock(slots_mutex_);
        ++held_[side][slot];
    }

    // One holder of stream `side`'s `slot` is done with it; the last gives it
    // back to the stream.
    void drop(std::size_t side, int slot) {
        {
            const std::lock_guard lock(slots_mutex_);
            int& holders = held_[side][slot];
            if (--holders > 0) {
                return;
            }
            held_[side].erase(slot);
        }
        streams_[side]->release(slot);
    }

    // One metric is done with pair `index`; the last one drops its slots.
    void pair_done(long long index) {
        std::array<int, 3> entry{};
        {
            const std::lock_guard lock(slots_mutex_);
            std::array<int, 3>& pending = pending_.at(index);
            if (--pending[0] > 0) {
                return;
            }
            entry = pending;
            pending_.erase(index);
        }
        drop(0, entry[1]);
        drop(1, entry[2]);
    }

    // Stream `side`'s next frame's slot, held for the pairing, or
    // end_of_video.
    int next_frame(std::size_t side) {
        const int slot = streams_[side]->next(cancel_);
        if (slot != end_of_video) {
            hold(side, slot);
        }
        return slot;
    }

    // Hands pair `frame` to each metric's lanes, which hold its slots until
    // they are done with them.
    void give_pair(long long frame, int source_slot, int distorted_slot, PassRate& rate) {
        const auto metrics = static_cast<int>(plan_.keys.size());
        const std::vector<std::pair<std::string, std::exception_ptr>> errors = lane_errors();
        if (static_cast<int>(errors.size()) == metrics) {
            // Nothing left to score: stop now rather than decode the rest of
            // the video.
            std::rethrow_exception(errors.front().second);
        }
        // No abort here: a failed lane keeps handing its frames back, so the
        // ring never fills with slots nobody gives back.
        {
            const std::lock_guard lock(slots_mutex_);
            pending_[frame] = {metrics, source_slot, distorted_slot};
            ++held_[0][source_slot];
            ++held_[1][distorted_slot];
        }
        for (const std::string& key : frame_keys_) {
            scores_.at(key)->reserve(frame);
        }
        // Each frame metric's lanes are dealt pairs in turn; CVVDP's one
        // takes every pair.
        for (std::size_t metric = 0; metric < frame_keys_.size(); ++metric) {
            lanes_[metric * lanes_per_metric + static_cast<std::size_t>(frame) % lanes_per_metric]->give(
                {frame, source_slot, distorted_slot});
        }
        if (cvvdp_lane_) {
            cvvdp_lane_->give({frame, source_slot, distorted_slot});
        }
        report_progress(std::min((frame + 1) * step_, total_units_), rate.frames_per_second(frame + 1));
    }

    // Hands every pair to each metric's lanes: each test frame with the
    // source frame of the same number, as every metric pairs them
    // (frame_sync), until either video ends; or by timestamp. Returns how
    // many pairs.
    long long feed_pairs(PassRate& rate) {
        if (by_time_) {
            return feed_timed_pairs(rate);
        }
        long long frame = 0;
        while (true) {
            const int distorted_slot = next_frame(1);
            if (distorted_slot == end_of_video) {
                break;
            }
            const int source_slot = next_frame(0);
            if (source_slot == end_of_video) {
                drop(1, distorted_slot);
                break;
            }
            try {
                give_pair(frame, source_slot, distorted_slot, rate);
                ++frame;
            } catch (...) {
                drop(1, distorted_slot);
                drop(0, source_slot);
                throw;
            }
            // The pairing is done with this pair; its metrics may still hold it.
            drop(1, distorted_slot);
            drop(0, source_slot);
        }
        return frame;
    }

    // feed_pairs when pairing by timestamp: each test frame with the source
    // frame libvmaf pairs it with by timestamp (frame_sync_by_time.h). The
    // comparison is the frames both videos have, so it ends with the test
    // video, or at its first frame after the source's end. A source frame
    // can go to several pairs; the pairing holds a frame until no later pair
    // can use it.
    long long feed_timed_pairs(PassRate& rate) {
        FrameRing& source = *streams_[0];
        FrameRing& test = *streams_[1];
        // Each video's first frame, for its time base.
        const int first_source = source.next(cancel_);
        const int first_test = test.next(cancel_);
        if (first_source == end_of_video || first_test == end_of_video) {
            if (first_source != end_of_video) {
                source.release(first_source);
            }
            if (first_test != end_of_video) {
                test.release(first_test);
            }
            return 0;
        }
        const Rational test_base = test.time_base();
        const Rational source_base = source.time_base();
        const long long source_start = source.pts(first_source);
        // Each video's next frame, held for the pairing, starting with the
        // one already taken.
        const auto puller = [this](std::size_t side, int first) {
            return [this, side, waiting = std::optional<int>(first)]() mutable -> std::optional<TimedPairs::Frame> {
                const int slot = waiting ? *waiting : streams_[side]->next(cancel_);
                waiting.reset();
                if (slot == end_of_video) {
                    return std::nullopt;
                }
                hold(side, slot);
                return TimedPairs::Frame{slot, streams_[side]->pts(slot)};
            };
        };
        TimedPairs pairs(puller(1, first_test), puller(0, first_source), test_base, source_base,
                         [this](int slot) { drop(1, slot); }, [this](int slot) { drop(0, slot); });
        long long frame = 0;
        while (const std::optional<TimedPairs::Pair> pair = pairs.next()) {
            if (!pair->source) {
                throw VshipUnavailableError("The source has no frame at the test video's first.");
            }
            if (check_apart_) {
                check_lined_up(pair->time, test_base, source.pts(*pair->source) - source_start, source_base);
            }
            give_pair(frame, *pair->source, pair->test, rate);
            ++frame;
        }
        return frame;
    }

    // With frame subsampling, the source gives only every step-th frame, as
    // the test video does, and while the two videos' frames line up, those
    // are the frames the test video's are paired with. A pair further apart
    // than a quarter of a frame (any other source frame is then at least
    // three quarters of a frame away) shows that they do not: throws
    // FramesApartError, and the pass is made again with every source frame.
    // In doubles, where the Python app takes exact fractions: the two can
    // differ only within a rounding of a quarter of a frame.
    void check_lined_up(long long test_ticks, Rational test_base, long long source_ticks, Rational source_base) const {
        const double test_time =
            static_cast<double>(test_ticks) * static_cast<double>(test_base.numerator) / test_base.denominator;
        const double source_time = static_cast<double>(source_ticks) * static_cast<double>(source_base.numerator) /
                                   source_base.denominator;
        const double apart = test_time - source_time;
        if (std::abs(apart) * request_.source.fps > 0.25) {
            throw FramesApartError("the test frame at " + format_fixed(test_time, 3) + " s is " +
                                   format_fixed(apart * 1000, 1) +
                                   " ms from the nearest of the source frames sampled with it (one in " +
                                   std::to_string(step_) + ")");
        }
    }

    // Passes the pass's progress on, at most every tenth of a second besides
    // its first and last figures: the pairs come up to hundreds a second, and
    // each report crosses to the run's process (the Python app's job queue
    // thins them the same way).
    void report_progress(long long done, double fps) {
        if (!on_progress_) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (last_report_ && done < total_units_ && now - *last_report_ < std::chrono::milliseconds(100)) {
            return;
        }
        last_report_ = now;
        on_progress_(done, total_units_, fps);
    }

    // ------------------------------------------------------------ results

    PerceptualTaskOutput finish(long long frames, PassRate& rate, std::chrono::steady_clock::time_point started) {
        if (frames == 0) {
            throw VshipUnavailableError("FFmpeg produced no frame pairs for Vship.");
        }
        if (const std::optional<std::string> short_run =
                short_comparison(expected_frames_, frames * step_, request_.source.fps, step_)) {
            throw ComparisonCutShortError(*short_run);
        }
        // A metric whose handler failed (out of GPU memory on a smaller card,
        // say) is reported on its own, and the others keep their scores.
        const std::vector<std::pair<std::string, std::exception_ptr>> errors = lane_errors();
        if (errors.size() == plan_.keys.size()) {
            std::rethrow_exception(errors.front().second);
        }
        PerceptualFailures failures;
        for (const auto& [key, error] : errors) {
            set_failure(failures, key, message_of(error));
        }
        PerceptualTaskOutput output;
        output.metrics = results(frames, failures);
        output.source_crop = videos_[0].crop;
        output.distorted_crop = videos_[1].crop;
        output.compared_frame_count = frames * step_;
        output.failures = failures;
        const double elapsed =
            std::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(), 1e-6);
        logger.info("Vship pass (" + metric_labels(plan_.keys, ", ") + ") on " + plan_.device.name + ": " +
                    std::to_string(frames) + " frame pairs in " + format_fixed(elapsed, 1) + " s, " +
                    format_fixed(static_cast<double>(frames) / elapsed, 1) + " pairs/s");
        for (const auto& [key, error] : errors) {
            logger.error("Vship " + std::string(metric_definition(key).label) + " failed in its pass: " +
                         message_of(error));
        }
        if (on_progress_) {
            // A pass too short to time from its first frame is timed over its
            // whole length instead.
            const double fps = rate.frames_per_second(frames);
            on_progress_(frames * step_, frames * step_,
                         fps != 0.0 ? fps : static_cast<double>(frames * step_) / elapsed);
        }
        return output;
    }

    // Each metric's scores that did not fail: per frame, and CVVDP's per
    // second of video.
    MetricResultSet results(long long frames, const PerceptualFailures& failures) const {
        const VshipDevice& device = plan_.device;
        std::vector<int> frame_numbers;
        std::vector<double> times;
        for (long long index = 0; index < frames; ++index) {
            const auto number = static_cast<int>(index * step_);
            frame_numbers.push_back(number);
            times.push_back(static_cast<double>(number) / std::max(request_.source.fps, 1.0));
        }
        // How the frames reached Vship and how the videos' colour tags were
        // read, as the Python app records it (its values as JSON text).
        const std::map<std::string, std::string> parameters{
            {"gpu_backend", nlohmann::json(device.backend).dump()},
            {"gpu_name", nlohmann::json(device.name).dump()},
            {"input", nlohmann::json("ffmpeg frames (hardware-decoded NV12/P010 split into planes); "
                                     "native range/transfer and primaries")
                          .dump()},
            {"coverage_step", std::to_string(step_)},
            {"butteraugli_norm", nlohmann::json("3-norm").dump()},
            {"color_tags", nlohmann::json(score_identity::vship_colour_tags).dump()},
        };
        MetricResultSet found;
        for (const std::string& key : frame_keys_) {
            if (failure_of(failures, key) != nullptr) {
                continue;
            }
            MetricProvenance provenance{"Vship/" + key, build_name(), "gpu", score_id(key), parameters};
            found.add(FrameMetricResult{key, frame_numbers, times, scores_.at(key)->values(frames),
                                        std::move(provenance)});
        }
        if (cvvdp_lane_ && !cvvdp_lane_->error()) {
            found.add(cvvdp_result());
        }
        return found;
    }

    // The build that calculated the scores, as their provenance names it:
    // Vship-fast, or stock Vship with the version it says.
    std::string build_name() const {
        if (!plan_.vship51) {
            return "Vship-fast " + plan_.device.version;
        }
        const vship_api::Version version = library_.get_version();
        return "Vship " + std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
               std::to_string(version.minorMinor);
    }

    // The id a score of `key` is saved under (score_identity.h): Vship-fast's,
    // or Vship 5.1's for stock Vship 5.1.2's.
    std::string score_id(const std::string& key) const {
        if (key == "cvvdp") {
            return std::string(plan_.vship51 ? score_identity::vship51_cvvdp : score_identity::cvvdp);
        }
        return plan_.vship51 ? score_identity::vship51_gpu(key) : score_identity::vship_gpu(key);
    }

    // CVVDP's score, with each second's JOD at the second's first frame.
    SequenceMetricResult cvvdp_result() const {
        const VshipDevice& device = plan_.device;
        const std::vector<CvvdpWindow>& windows = cvvdp_lane_->windows();
        SequenceMetricResult result;
        result.key = "cvvdp";
        result.score = pool_cvvdp_windows(windows);
        result.provenance = {"Vship/cvvdp",
                             build_name(),
                             "gpu",
                             score_id("cvvdp"),
                             {{"gpu_backend", nlohmann::json(device.backend).dump()},
                              {"gpu_name", nlohmann::json(device.name).dump()},
                              {"display", cvvdp_->display.identity().dump()},
                              {"resize_to_display", nlohmann::json(cvvdp_->resize_to_display).dump()},
                              {"timeline", nlohmann::json("JOD of each second of video").dump()},
                              {"color_tags", nlohmann::json(score_identity::vship_colour_tags).dump()}}};
        for (const CvvdpWindow& window : windows) {
            result.frame.push_back(static_cast<int>(window.first));
            result.time.push_back(static_cast<double>(static_cast<int>(window.first)) /
                                  std::max(request_.source.fps, 1.0));
            result.values.push_back(static_cast<float>(window.jod));
        }
        return result;
    }

    const PerceptualRequest& request_;
    const PassPlan& plan_;
    bool native_;
    const ProgressCallback& on_progress_;
    const StatusCallback& on_status_;
    const CancelToken& cancel_;
    const VshipLibrary& library_;
    std::array<VideoPlan, 2> videos_;
    int step_ = 1;
    std::vector<std::string> frame_keys_;  // the metrics scored per frame: all but CVVDP
    std::optional<CvvdpSettings> cvvdp_;   // with CVVDP, the display it models
    bool by_time_ = false;                 // frames paired by timestamp
    bool every_source_frame_ = false;      // by timestamp: the source's every frame, subsampled or not
    bool check_apart_ = false;             // by timestamp, subsampled: whether the pairs line up
    Rational numbering_rate_;
    std::optional<std::string> duration_limit_;
    long long expected_frames_ = 0;
    long long total_units_ = 1;
    std::mutex decode_mutex_;
    HardwareDecoding decode_;  // where each video is decoded, as statuses report it
    std::vector<std::unique_ptr<FrameRing>> streams_;  // the source's, then the test video's
    Planes source_planes_;
    Planes distorted_planes_;
    std::vector<std::unique_ptr<MetricLane>> lanes_;  // each frame metric's, in turn
    std::unique_ptr<CvvdpLane> cvvdp_lane_;
    std::map<std::string, std::unique_ptr<ScoreArray>, std::less<>> scores_;
    std::atomic<bool> abort_{false};
    std::mutex slots_mutex_;
    std::map<long long, std::array<int, 3>> pending_;  // pair -> [metrics left, source slot, test slot]
    std::array<std::map<int, int>, 2> held_;           // each stream's slots -> how many hold them
    std::optional<std::chrono::steady_clock::time_point> last_report_;
};

// ------------------------------------------------- a pass in its own process

constexpr const char* perceptual_scores_file = "perceptual.bin";

nlohmann::json provenance_json(const MetricProvenance& provenance) {
    return {{"implementation", provenance.implementation},
            {"implementation_version", provenance.implementation_version},
            {"compute_backend", provenance.compute_backend},
            {"implementation_compatibility_id", provenance.implementation_compatibility_id},
            {"parameters", provenance.parameters}};
}

MetricProvenance provenance_from(const nlohmann::json& text) {
    MetricProvenance provenance;
    text.at("implementation").get_to(provenance.implementation);
    text.at("implementation_version").get_to(provenance.implementation_version);
    text.at("compute_backend").get_to(provenance.compute_backend);
    text.at("implementation_compatibility_id").get_to(provenance.implementation_compatibility_id);
    text.at("parameters").get_to(provenance.parameters);
    return provenance;
}

nlohmann::json request_json(const PerceptualRequest& request, const PassPlan& plan) {
    return {{"ffmpeg", request.environment.ffmpeg},
            {"ffprobe", request.environment.ffprobe},
            {"source", video_json(request.source)},
            {"distorted", video_json(request.distorted)},
            {"options", options_json(request.options)},
            {"backends", request.backends},
            {"cvvdp", request.cvvdp.parameters()},
            {"keys", plan.keys},
            {"device", vship_device_json(plan.device)},
            {"source_crop", crop_json(plan.source_crop)},
            {"distorted_crop", crop_json(plan.distorted_crop)},
            {"decoding",
             {{"source", plan.decoding.source ? nlohmann::json(*plan.decoding.source) : nlohmann::json()},
              {"distorted", plan.decoding.distorted ? nlohmann::json(*plan.decoding.distorted) : nlohmann::json()}}},
            {"vship51", plan.vship51}};
}

// A metric's frames, times and values into the scores file.
void write_numbers(std::ofstream& file, const std::vector<int>& frame, const std::vector<double>& time,
                   const std::vector<float>& values) {
    file.write(reinterpret_cast<const char*>(frame.data()), static_cast<std::streamsize>(frame.size() * sizeof(int)));
    file.write(reinterpret_cast<const char*>(time.data()), static_cast<std::streamsize>(time.size() * sizeof(double)));
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

// `count` frames, times and values back from the scores file.
void read_numbers(std::ifstream& file, std::size_t count, std::vector<int>& frame, std::vector<double>& time,
                  std::vector<float>& values) {
    frame.resize(count);
    time.resize(count);
    values.resize(count);
    file.read(reinterpret_cast<char*>(frame.data()), static_cast<std::streamsize>(count * sizeof(int)));
    file.read(reinterpret_cast<char*>(time.data()), static_cast<std::streamsize>(count * sizeof(double)));
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
}

// The pass's output, with its metrics' numbers in `folder`'s
// perceptual_scores_file (too many for a message): each frame metric's, and
// CVVDP's timeline followed by its score (8 bytes: exact, and a NaN
// survives, which JSON's numbers would not).
nlohmann::json output_json(const PerceptualTaskOutput& output, const std::filesystem::path& folder) {
    nlohmann::json metrics = nlohmann::json::array();
    std::ofstream file(folder / perceptual_scores_file, std::ios::binary);
    for (const std::string& key : output.metrics.keys()) {
        if (const FrameMetricResult* result = output.metrics.frame(key)) {
            metrics.push_back(
                {{"key", key}, {"count", result->frame.size()}, {"provenance", provenance_json(result->provenance)}});
            write_numbers(file, result->frame, result->time, result->values);
            continue;
        }
        const SequenceMetricResult& result = *output.metrics.sequence(key);
        metrics.push_back({{"key", key},
                           {"count", result.values.size()},
                           {"sequence", true},
                           {"provenance", provenance_json(result.provenance)}});
        write_numbers(file, result.frame, result.time, result.values);
        file.write(reinterpret_cast<const char*>(&result.score), sizeof result.score);
    }
    if (!file) {
        throw std::runtime_error("The scores could not be written.");
    }
    return {{"metrics", metrics},
            {"source_crop", crop_json(output.source_crop)},
            {"distorted_crop", crop_json(output.distorted_crop)},
            {"compared_frame_count", output.compared_frame_count},
            {"failures", output.failures}};
}

PerceptualTaskOutput output_from(const nlohmann::json& result, const std::filesystem::path& folder) {
    PerceptualTaskOutput output;
    std::ifstream file(folder / perceptual_scores_file, std::ios::binary);
    for (const nlohmann::json& metric : result.at("metrics")) {
        const auto count = metric.at("count").get<std::size_t>();
        if (metric.value("sequence", false)) {
            SequenceMetricResult found;
            found.key = metric.at("key").get<std::string>();
            found.provenance = provenance_from(metric.at("provenance"));
            read_numbers(file, count, found.frame, found.time, found.values);
            file.read(reinterpret_cast<char*>(&found.score), sizeof found.score);
            output.metrics.add(std::move(found));
            continue;
        }
        FrameMetricResult found;
        found.key = metric.at("key").get<std::string>();
        found.provenance = provenance_from(metric.at("provenance"));
        read_numbers(file, count, found.frame, found.time, found.values);
        output.metrics.add(std::move(found));
    }
    if (!file) {
        throw std::runtime_error("The scores could not be read back from the process that made them.");
    }
    output.source_crop = crop_from(result.at("source_crop"));
    output.distorted_crop = crop_from(result.at("distorted_crop"));
    result.at("compared_frame_count").get_to(output.compared_frame_count);
    result.at("failures").get_to(output.failures);
    return output;
}

// One pass in this process (the Python app's _score_vship_pass), with the
// videos that a decoder of the app's takes decoded here, made again where
// that can give its scores (remade_vship_pass).
PerceptualTaskOutput score_vship_pass(const PerceptualRequest& request, const PassPlan& plan,
                                      const ProgressCallback& on_progress, const StatusCallback& on_status,
                                      const CancelToken& cancel) {
    return remade_vship_pass(
        [&](bool native, bool every_source_frame) {
            return VshipPass(request, plan, native, every_source_frame, on_progress, on_status, cancel).run();
        },
        on_status, cancel);
}

// One pass, in a process of its own where a program to run it is set (the
// Python app's _run_vship_pass): Vship and the GPU's driver run in the
// process that loads them, and a crash in either then ends only that
// process, failing the pass like any other GPU failure.
PerceptualTaskOutput run_vship_pass(const PerceptualRequest& request, const PassPlan& plan,
                                    const ProgressCallback& on_progress, const StatusCallback& on_status,
                                    const CancelToken& cancel) {
    if (request.options.resample_test) {
        throw VshipUnavailableError("The GPU metrics do not support resolution round-trip tests yet.");
    }
    if (!isolation_program()) {
        if (plan.vship51) {
            // This process has Vship-fast's library (vship_library).
            throw VshipUnavailableError("Vship 5.1.2 runs only in a process of its own");
        }
        return score_vship_pass(request, plan, on_progress, on_status, cancel);
    }
    PerceptualTaskOutput output;
    std::optional<std::string> cut_short;
    try {
        run_isolated(
            "vship-pass", "Vship", request_json(request, plan),
            [&](const nlohmann::json& message) {
                if (message.contains("progress") && on_progress) {
                    const nlohmann::json& counts = message["progress"];
                    on_progress(counts.at(0).get<long long>(), counts.at(1).get<long long>(),
                                counts.at(2).get<double>());
                } else if (message.contains("status") && on_status) {
                    on_status(status_from_json(message["status"]));
                }
            },
            cancel, [&](const nlohmann::json& result, const std::filesystem::path& folder) {
                if (result.contains("cut_short")) {
                    cut_short = result.at("cut_short").get<std::string>();
                } else {
                    output = output_from(result, folder);
                }
            });
    } catch (const IsolatedCrashError& error) {
        logger.error("Vship pass (" + metric_labels(plan.keys, ", ") + ") failed: " + error.what());
        throw VshipUnavailableError("Vship GPU calculation failed: " + std::string(error.what()));
    } catch (const Cancelled&) {
        throw;
    } catch (const std::runtime_error& error) {
        throw VshipUnavailableError(error.what());  // the child's own error, as its message
    }
    if (cut_short) {
        // A video cut short keeps its kind, so that it is not retried on the
        // CPU.
        throw ComparisonCutShortError(*cut_short);
    }
    return output;
}

// ------------------------------------------------------------------- a task

// The metrics of `keys` that `device` does not calculate at the size the
// videos are compared at (size_refusal), each with the reason.
PerceptualFailures size_refusals(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                 const VshipDevice& device, const std::optional<CropBox>& source_crop,
                                 const std::optional<CropBox>& distorted_crop) {
    const std::pair<int, int> size =
        vship_scaled_sizes(request.source, request.distorted, request.options.scale_direction, source_crop,
                           distorted_crop)
            .first;
    PerceptualFailures refused;
    for (const std::string& key : keys) {
        // CVVDP set to fill the display is calculated at the display's size,
        // so a 1080p comparison on a 4K display would otherwise get through
        // at the size that hangs Intel's GPU.
        const std::optional<std::pair<int, int>> computed =
            key == "cvvdp" ? std::optional(cvvdp_computed_size(
                                 size, CvvdpSettings::from_parameters(nlohmann::json(request.cvvdp.parameters()))))
                           : std::nullopt;
        if (const std::optional<std::string> reason = size_refusal(device, key, size, computed)) {
            set_failure(refused, key, *reason);
        }
    }
    return refused;
}

// Scores one task's Vship metrics, in one pass or several (the Python app's
// _VshipTask).
class VshipTask {
public:
    // `vship51`: on stock Vship 5.1.2, the fallback, rather than Vship-fast;
    // always on a device only 5.1.2 found.
    VshipTask(const PerceptualRequest& request, const std::vector<std::string>& keys, const VshipDevice& device,
              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
              bool together, const PerceptualHooks& hooks, const CancelToken& cancel, bool vship51 = false)
        : request_(request),
          keys_(keys),
          device_(device),
          source_crop_(source_crop),
          distorted_crop_(distorted_crop),
          together_(together),
          hooks_(hooks),
          cancel_(cancel),
          vship51_(vship51 || device.vship51) {}

    PerceptualTaskOutput run() {
        const PerceptualFailures refused = refused_for_size();
        if (!refused.empty()) {
            return run_without(refused);
        }
        const std::unique_lock pass = take_gpu_turn(
            [this] {
                if (hooks_.on_status) {
                    hooks_.on_status({std::string(gpu_wait_message), StatusKind::gpu_wait});
                }
            },
            cancel_);
        if (keys_.size() <= 1) {
            if (hooks_.on_pass) {
                hooks_.on_pass(1, 1, keys_);
            }
            return run_pass(keys_, hooks_.on_progress);
        }
        return run_passes();
    }

private:
    PerceptualFailures refused_for_size() const {
        return size_refusals(request_, keys_, device_, source_crop_, distorted_crop_);
    }

    // Scores the metrics not `refused`, which are reported as failed with
    // their reasons.
    PerceptualTaskOutput run_without(const PerceptualFailures& refused) {
        for (const auto& [key, reason] : refused) {
            logger.warning(std::string(metric_definition(key).label) + " not started: " + reason);
        }
        std::vector<std::string> keys;
        for (const std::string& key : keys_) {
            if (failure_of(refused, key) == nullptr) {
                keys.push_back(key);
            }
        }
        if (keys.empty()) {
            throw VshipPassesFailedError(refused.front().second, refused);
        }
        PerceptualTaskOutput output;
        try {
            output = VshipTask(request_, keys, device_, source_crop_, distorted_crop_, together_, hooks_, cancel_,
                               vship51_)
                         .run();
        } catch (const Cancelled&) {
            throw;
        } catch (const ComparisonCutShortError&) {
            throw;
        } catch (const VshipPassesFailedError& error) {
            // The others failed too. Each keeps its own reason.
            PerceptualFailures reasons = error.failures();
            for (const auto& [key, reason] : refused) {
                set_failure(reasons, key, reason);
            }
            throw VshipPassesFailedError(error.what(), reasons);
        } catch (const std::exception& error) {
            PerceptualFailures reasons;
            for (const std::string& key : keys) {
                set_failure(reasons, key, error.what());
            }
            for (const auto& [key, reason] : refused) {
                set_failure(reasons, key, reason);
            }
            throw VshipPassesFailedError(error.what(), reasons);
        }
        for (const auto& [key, reason] : refused) {
            set_failure(output.failures, key, reason);
        }
        return output;
    }

    PerceptualTaskOutput run_pass(const std::vector<std::string>& keys, const ProgressCallback& progress) const {
        // On a Mac every video VideoToolbox decodes, 8-bit ones too: the
        // pass is slower than its decoding (MacGpuDecoding).
        const PassPlan plan{keys,
                            device_,
                            source_crop_,
                            distorted_crop_,
                            request_.options.gpu_decode
                                ? plan_hardware_decoding(request_.environment.decoding, request_.options.gpu_vendor,
                                                         request_.source, request_.distorted, MacGpuDecoding::on)
                                : HardwareDecoding{},
                            vship51_};
        return run_vship_pass(request_, plan, progress, hooks_.on_status, cancel_);
    }

    // Scores the metrics in the passes vship_passes groups them into, then
    // each metric that failed in a shared pass in a pass of its own. They
    // share one pass only when Settings > GPU metrics says so: it decodes
    // each video once for all of them, which is faster only on a very
    // powerful GPU, and takes far more of the GPU's memory.
    PerceptualTaskOutput run_passes() {
        const std::vector<std::vector<std::string>> passes =
            vship_passes(keys_, request_.options.n_subsample, together_);
        run_groups(passes, 0, static_cast<int>(passes.size()));
        std::vector<std::string> shared;
        for (const std::vector<std::string>& group : passes) {
            if (group.size() > 1) {
                for (const std::string& key : group) {
                    if (failure_of(failures_, key) != nullptr) {
                        shared.push_back(key);
                    }
                }
            }
        }
        if (!shared.empty()) {
            retry_alone(shared, static_cast<int>(passes.size()));
        }
        if (metrics_.empty()) {
            if (errors_.empty()) {
                throw VshipUnavailableError("Vship produced no scores.");
            }
            throw VshipPassesFailedError(message_of(errors_.front()), failures_);
        }
        return PerceptualTaskOutput{metrics_, source_crop_, distorted_crop_, frames_, failures_};
    }

    // Scores each metric that failed in a shared pass in a pass of its own,
    // numbered after the `passes_before` passes.
    void retry_alone(const std::vector<std::string>& shared, int passes_before) {
        const std::string labels = metric_labels(shared, " and ");
        logger.warning(labels + " failed in the GPU pass shared with the other metrics; calculating " +
                       (shared.size() == 1 ? "it in a pass of its own" : "them in a pass of their own"));
        if (hooks_.on_status) {
            hooks_.on_status({labels + " failed in the shared GPU pass; calculating " +
                              (shared.size() == 1 ? "it" : "each") + " in a pass of its own\xE2\x80\xA6"});
        }
        for (const std::string& key : shared) {
            std::erase_if(failures_, [&key](const auto& failure) { return failure.first == key; });
        }
        std::vector<std::exception_ptr> failed_before;
        failed_before.swap(errors_);
        std::vector<std::vector<std::string>> alone;
        for (const std::string& key : shared) {
            alone.push_back({key});
        }
        run_groups(alone, passes_before, passes_before + static_cast<int>(shared.size()));
        errors_.insert(errors_.end(), failed_before.begin(), failed_before.end());  // the retries' own first
    }

    // Runs `passes` as passes first + 1 onwards of `count`, keeping each
    // one's scores, or its error as each of its metrics' failure.
    void run_groups(const std::vector<std::vector<std::string>>& passes, int first, int count) {
        for (std::size_t offset = 0; offset < passes.size(); ++offset) {
            const int number = first + static_cast<int>(offset);
            const std::vector<std::string>& group = passes[offset];
            if (hooks_.on_status && count > 1) {
                hooks_.on_status({"GPU metric " + std::to_string(number + 1) + "/" + std::to_string(count) + ": " +
                                      metric_labels(group, " + "),
                                  StatusKind::gpu_pass});
            }
            if (hooks_.on_pass) {
                hooks_.on_pass(number + 1, count, group);
            }
            reached_.push_back(0);
            ProgressCallback progress;
            if (hooks_.on_progress) {
                long long done = 0;
                for (const long long reached : reached_) {
                    done += reached;
                }
                progress = pass_progress(reached_.size() - 1, done, count - number);
            }
            PerceptualTaskOutput output;
            try {
                output = run_pass(group, progress);
            } catch (const Cancelled&) {
                throw;
            } catch (const ComparisonCutShortError&) {
                // A video that ends early is cut short for every metric: the
                // file's fault, the same in each pass and on the CPU.
                throw;
            } catch (const std::exception& error) {
                if (cancel_.cancelled()) {
                    throw Cancelled();
                }
                logger.error("GPU metric " + std::to_string(number + 1) + "/" + std::to_string(count) + " (" +
                             metric_labels(group, " + ") + ") failed: " + error.what());
                errors_.push_back(std::current_exception());
                for (const std::string& key : group) {
                    set_failure(failures_, key, error.what());
                }
                continue;
            }
            metrics_.add_all(output.metrics);
            for (const auto& [key, reason] : output.failures) {
                set_failure(failures_, key, reason);
            }
            frames_ = std::max(frames_, output.compared_frame_count);
            if (hooks_.on_pass_done && !output.metrics.empty()) {
                hooks_.on_pass_done(output);
            }
        }
    }

    // Reports a pass's progress as the task's: after `done` units of the
    // passes before it, with `left` passes of its length to go, this one
    // included (they score the same frames). A retry's count follows the
    // passes before it, so the task never goes from 100% back to 0%.
    ProgressCallback pass_progress(std::size_t slot, long long done, int left) {
        return [this, slot, done, left](long long current, long long total, double fps) {
            reached_[slot] = current;
            hooks_.on_progress(done + current, done + left * total, fps);
        };
    }

    const PerceptualRequest& request_;
    std::vector<std::string> keys_;
    const VshipDevice& device_;
    std::optional<CropBox> source_crop_;
    std::optional<CropBox> distorted_crop_;
    bool together_;
    const PerceptualHooks& hooks_;
    const CancelToken& cancel_;
    bool vship51_;
    MetricResultSet metrics_;
    PerceptualFailures failures_;
    std::vector<std::exception_ptr> errors_;
    long long frames_ = 0;
    std::vector<long long> reached_;
};

}  // namespace

// ----------------------------------------------------------------- public

bool is_perceptual_metric(std::string_view key) {
    return std::find(vship_metrics.begin(), vship_metrics.end(), key) != vship_metrics.end();
}

std::vector<std::vector<std::string>> vship_passes(const std::vector<std::string>& keys, int n_subsample,
                                                   bool together) {
    std::vector<std::vector<std::string>> passes;
    if (!together) {
        for (const std::string& key : keys) {
            passes.push_back({key});
        }
        return passes;
    }
    // One per set of frames scored, in the order each set first appears.
    std::vector<int> steps;
    for (const std::string& key : keys) {
        const int step = perceptual_step(key, n_subsample);
        const auto at = std::find(steps.begin(), steps.end(), step);
        if (at == steps.end()) {
            steps.push_back(step);
            passes.push_back({key});
        } else {
            passes[static_cast<std::size_t>(at - steps.begin())].push_back(key);
        }
    }
    return passes;
}

namespace {

// Calculates a task's SSIMULACRA2, Butteraugli and CVVDP each where it is
// set to run (the Python app's _MetricRouting). What Vship-fast fails on the
// GPU is calculated again with stock Vship 5.1.2 where it is bundled, and
// what that fails too moves to the CPU, whatever the video's length.
class MetricRouting {
public:
    MetricRouting(const PerceptualRequest& request, const std::vector<std::string>& keys, bool together,
                  const PerceptualHooks& hooks, const CancelToken& cancel, const PerceptualRoutes& routes)
        : request_(request), keys_(keys), together_(together), hooks_(hooks), cancel_(cancel), routes_(routes) {
        for (const std::string& key : keys) {
            (request.backend_of(key) == "gpu" ? gpu_ : cpu_).push_back(key);
            if (key == "cvvdp") {
                gpu_only_.push_back(key);
            } else {
                retryable_.push_back(key);  // what may be calculated on the CPU
            }
        }
    }

    PerceptualTaskOutput run() {
        // A choice of the CPU alone does not probe Vship or touch a GPU.
        if (gpu_.empty()) {
            return on_cpu(cpu_, {}, std::nullopt);
        }
        const auto [device, reason] = routes_.find_gpu();
        if (!device) {
            return without_gpu(reason);
        }
        device_ = *device;
        if (device_->vship51 && !device_->fast_failure.empty() && hooks_.on_status) {
            // Vship-fast failed its check on this GPU, where stock Vship
            // 5.1.2 passed: a fallback, slower, said as the GPU's failures are.
            hooks_.on_status({"Vship-fast could not use the GPU (" + device_->fast_failure + "); calculating " +
                                  metric_labels(gpu_, " and ") + " with Vship 5.1.2\xE2\x80\xA6",
                              StatusKind::gpu_perceptual_failed});
        }
        const CropPair crops = routes_.find_crops(request_, hooks_.on_status, cancel_);
        PerceptualTaskOutput gpu_output;
        try {
            gpu_output = routes_.run_gpu(request_, gpu_, *device_, crops, together_, hooks_, cancel_);
        } catch (const Cancelled&) {
            throw;
        } catch (const ComparisonCutShortError&) {
            // A video cut short (one that ends early) is not retried with
            // stock Vship or on the CPU: it would stop as short, after hours
            // instead of the GPU's minutes.
            throw;
        } catch (const std::exception& error) {
            if (cancel_.cancelled()) {
                throw Cancelled();
            }
            return after_vship_fast_failure(error, crops);
        }
        return after_gpu(std::move(gpu_output), crops, true);
    }

private:
    // ------------------------------------------------------------ the CPU

    PerceptualTaskOutput run_on_cpu(const std::vector<std::string>& keys, const std::optional<CropPair>& crops) {
        return routes_.run_cpu(request_, keys, crops, hooks_.on_progress, hooks_.on_status, cancel_);
    }

    // Reports the metrics the CPU is about to calculate (on_cpu).
    void announce_cpu(const std::vector<std::string>& handed) {
        if (hooks_.on_cpu && !handed.empty()) {
            hooks_.on_cpu(handed);
        }
    }

    // Calculates `keys` on the CPU, with `failures` added to the output.
    PerceptualTaskOutput on_cpu(const std::vector<std::string>& keys, const PerceptualFailures& failures,
                                const std::optional<CropPair>& crops) {
        announce_cpu(keys);
        PerceptualTaskOutput output = run_on_cpu(keys, crops);
        for (const auto& [key, reason] : failures) {
            set_failure(output.failures, key, reason);
        }
        return output;
    }

    // ------------------------------------------------------------ when the GPU cannot

    // The failures of the GPU-only metrics (CVVDP), or, when nothing else
    // was asked for, the PerceptualRunError with them. `no_gpu` means no
    // usable GPU was found. Otherwise the GPU pass itself failed, possibly
    // for a reason that has nothing to do with the GPU (the two videos'
    // frame rates differing, say), so the message does not blame the GPU,
    // which would send people looking for the wrong problem. `reasons` are
    // each metric's own, where the passes gave them.
    PerceptualFailures gpu_only_failed(const std::string& reason, bool no_gpu,
                                       const PerceptualFailures* reasons = nullptr) {
        PerceptualFailures messages;
        std::string lines;
        for (const std::string& key : gpu_only_) {
            const std::string* found = reasons != nullptr ? failure_of(*reasons, key) : nullptr;
            const std::string& own = found != nullptr ? *found : reason;
            const std::string message = no_gpu ? label_of(key) + " needs a GPU that Vship can use, and could not use one: " + own
                                               : label_of(key) + " could not be calculated: " + own;
            set_failure(messages, key, message);
            lines += (lines.empty() ? "" : "\n") + message;
        }
        if (retryable_.empty()) {
            throw PerceptualRunError(lines);
        }
        return messages;
    }

    // Calculates on the CPU what can be, when no usable GPU was found: a
    // fallback, said as the GPU's failures are, where a check of the GPU
    // failed rather than finding none.
    PerceptualTaskOutput without_gpu(const std::string& reason) {
        const PerceptualFailures failures = gpu_only_.empty() ? PerceptualFailures{} : gpu_only_failed(reason, true);
        if (hooks_.on_status) {
            const bool failed = routes_.gpu_check_failed && routes_.gpu_check_failed();
            hooks_.on_status({"Vship GPU unavailable (" + reason + "); using CPU reference metrics\xE2\x80\xA6",
                              failed ? StatusKind::gpu_perceptual_failed : StatusKind::step});
        }
        return on_cpu(retryable_, failures, std::nullopt);
    }

    // ------------------------------------------------------------ stock Vship 5.1.2

    // The metrics of `keys` that stock Vship 5.1.2 calculates again after
    // Vship-fast failed them: none where it is not bundled or is what failed
    // them (on a GPU only 5.1.2 found), and none refused for the size
    // compared at, which hung the GPU with it (size_refusal).
    std::vector<std::string> for_vship51(const std::vector<std::string>& keys, const CropPair& crops) const {
        if (!routes_.run_vship51 || keys.empty() || device_->vship51) {
            return {};
        }
        const PerceptualFailures refused = size_refusals(request_, keys, *device_, crops.first, crops.second);
        std::vector<std::string> kept;
        std::copy_if(keys.begin(), keys.end(), std::back_inserter(kept),
                     [&refused](const std::string& key) { return failure_of(refused, key) == nullptr; });
        return kept;
    }

    // After Vship-fast's task failed: its metrics again with stock Vship
    // 5.1.2, then on the CPU what that fails too (after_gpu); or, where Vship
    // 5.1.2 cannot take them or fails them all, on the CPU.
    PerceptualTaskOutput after_vship_fast_failure(const std::exception& error, const CropPair& crops) {
        const std::vector<std::string> keys = for_vship51(gpu_, crops);
        if (keys.empty()) {
            return after_gpu_failure(error, crops);
        }
        logger.warning("Vship-fast failed (" + std::string(error.what()) + "); calculating " +
                       metric_labels(keys, ", ") + " with Vship 5.1.2");
        if (hooks_.on_status) {
            hooks_.on_status({"Vship-fast failed (" + std::string(error.what()) + "); calculating " +
                                  metric_labels(keys, " and ") + " with Vship 5.1.2\xE2\x80\xA6",
                              StatusKind::gpu_perceptual_failed});
        }
        PerceptualTaskOutput output;
        try {
            output = routes_.run_vship51(request_, keys, *device_, crops, together_, hooks_, cancel_);
        } catch (const Cancelled&) {
            throw;
        } catch (const ComparisonCutShortError&) {
            throw;
        } catch (const std::exception& again) {
            if (cancel_.cancelled()) {
                throw Cancelled();
            }
            return after_gpu_failure(again, crops);
        }
        // A metric not retried keeps Vship-fast's reason.
        const auto* passes = dynamic_cast<const VshipPassesFailedError*>(&error);
        for (const std::string& key : gpu_) {
            if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
                const std::string* own = passes != nullptr ? failure_of(passes->failures(), key) : nullptr;
                set_failure(output.failures, key, own != nullptr ? *own : std::string(error.what()));
            }
        }
        return after_gpu(std::move(output), crops, false);
    }

    // ------------------------------------------------------------ the CPU after the GPU

    // Calculates every metric that can be on the CPU after the GPU task
    // failed.
    PerceptualTaskOutput after_gpu_failure(const std::exception& error, const CropPair& crops) {
        const auto* passes = dynamic_cast<const VshipPassesFailedError*>(&error);
        const PerceptualFailures failures =
            gpu_only_.empty() ? PerceptualFailures{}
                              : gpu_only_failed(error.what(), false, passes != nullptr ? &passes->failures() : nullptr);
        if (hooks_.on_status) {
            hooks_.on_status({"Vship GPU compute failed (" + std::string(error.what()) +
                                  "); using CPU reference metrics\xE2\x80\xA6",
                              StatusKind::gpu_perceptual_failed});
        }
        // After a GPU failure, all metrics run together on the CPU, so that
        // the frames are extracted only once and the result stays whole.
        return on_cpu(retryable_, failures, crops);
    }

    // ------------------------------------------------------------ after the GPU task

    // A metric of Vship-fast's `gpu_output` that failed while the others
    // finished, calculated again with stock Vship 5.1.2, CVVDP too: its
    // scores join the output's, or its new reason replaces Vship-fast's.
    void retry_with_vship51(PerceptualTaskOutput& gpu_output, const CropPair& crops) {
        std::vector<std::string> failed;
        for (const std::string& key : gpu_) {
            if (failure_of(gpu_output.failures, key) != nullptr) {
                failed.push_back(key);
            }
        }
        const std::vector<std::string> keys = for_vship51(failed, crops);
        if (keys.empty()) {
            return;
        }
        logger.warning(metric_labels(keys, ", ") + " failed with Vship-fast; calculating it with Vship 5.1.2");
        if (hooks_.on_status) {
            hooks_.on_status({metric_labels(keys, " and ") + " failed on the GPU; calculating it with Vship 5.1.2\xE2\x80\xA6",
                              StatusKind::gpu_perceptual_failed});
        }
        PerceptualTaskOutput again;
        try {
            again = routes_.run_vship51(request_, keys, *device_, crops, together_, hooks_, cancel_);
        } catch (const Cancelled&) {
            throw;
        } catch (const ComparisonCutShortError&) {
            throw;
        } catch (const std::exception& error) {
            if (cancel_.cancelled()) {
                throw Cancelled();
            }
            logger.error("Vship 5.1.2 failed too: " + std::string(error.what()));
            const auto* passes = dynamic_cast<const VshipPassesFailedError*>(&error);
            for (const std::string& key : keys) {
                const std::string* own = passes != nullptr ? failure_of(passes->failures(), key) : nullptr;
                set_failure(gpu_output.failures, key, own != nullptr ? *own : std::string(error.what()));
            }
            return;
        }
        for (const std::string& key : keys) {
            if (again.metrics.has(key)) {
                gpu_output.metrics.copy_from(again.metrics, key);
                std::erase_if(gpu_output.failures, [&key](const auto& failure) { return failure.first == key; });
            } else if (const std::string* reason = failure_of(again.failures, key)) {
                set_failure(gpu_output.failures, key, *reason);
            }
        }
        gpu_output.compared_frame_count = std::max(gpu_output.compared_frame_count, again.compared_frame_count);
    }

    // Calculates the CPU's metrics once the GPU task is done, with any metric
    // that failed on the GPU (after Vship 5.1.2 too, with `vship51_next`),
    // and combines the two.
    PerceptualTaskOutput after_gpu(PerceptualTaskOutput gpu_output, const CropPair& crops, bool vship51_next) {
        if (vship51_next) {
            retry_with_vship51(gpu_output, crops);
        }
        // A metric that failed on the GPU while the others finished is
        // retried on the CPU, as a failed pass is, whatever the video's length.
        PerceptualFailures failures = gpu_output.failures;
        std::vector<std::string> gpu_failed;
        for (const std::string& key : gpu_) {
            if (key != "cvvdp" && failure_of(failures, key) != nullptr) {
                gpu_failed.push_back(key);
            }
        }
        // Kept for the message if the CPU retry fails too: the GPU failure is
        // the first cause, and without it a GPU user would be sent to fix a
        // CPU tool.
        PerceptualFailures gpu_reasons;
        for (const std::string& key : gpu_failed) {
            gpu_reasons.emplace_back(key, *failure_of(failures, key));
            std::erase_if(failures, [&key](const auto& failure) { return failure.first == key; });
        }
        std::vector<std::string> cpu_keys = cpu_;
        cpu_keys.insert(cpu_keys.end(), gpu_failed.begin(), gpu_failed.end());
        if (cpu_keys.empty()) {
            gpu_output.failures = failures;
            return gpu_output;
        }
        announce_cpu(cpu_keys);
        if (hooks_.on_status && gpu_failed.empty()) {
            hooks_.on_status({"Calculating selected perceptual metric(s) on CPU\xE2\x80\xA6"});
        } else if (hooks_.on_status) {
            hooks_.on_status({metric_labels(gpu_failed, " and ") + " failed on the GPU; calculating it on the CPU\xE2\x80\xA6",
                              StatusKind::gpu_perceptual_failed});
        }
        // A CPU failure's message, with the GPU failure first if there was one.
        const auto failed_on_cpu = [&gpu_reasons](const std::string& key, const std::string& reason) {
            const std::string* gpu_reason = failure_of(gpu_reasons, key);
            return gpu_reason != nullptr
                       ? "GPU scoring failed (" + *gpu_reason + "); the CPU retry failed too: " + reason
                       : reason;
        };
        // The GPU pass has finished, so a CPU failure from here on fails only
        // the CPU metrics, and does not throw away what the GPU scored (a
        // finished CVVDP pass lost to a missing libjxl tool, say). The
        // progress from here is a stage of its own, from 0 (on_cpu).
        PerceptualTaskOutput cpu_output;
        try {
            cpu_output = run_on_cpu(cpu_keys, crops);
        } catch (const Cancelled&) {
            throw;
        } catch (const std::exception& error) {
            if (cancel_.cancelled()) {
                throw Cancelled();
            }
            if (gpu_output.metrics.empty()) {
                if (!gpu_reasons.empty()) {
                    throw PerceptualRunError(failed_on_cpu(cpu_keys.back(), error.what()));
                }
                throw;
            }
            for (const std::string& key : cpu_keys) {
                set_failure(failures, key, failed_on_cpu(key, error.what()));
            }
            gpu_output.failures = failures;
            return gpu_output;
        }
        return combined(gpu_output, cpu_output, failures);
    }

    // The GPU's and the CPU's scores as one output. Each metric keeps its own
    // frames: Vship and the CPU tools can end a frame apart (each stops where
    // the shorter video's pictures do), so a difference must not throw the
    // CPU's finished scores away.
    PerceptualTaskOutput combined(const PerceptualTaskOutput& gpu_output, const PerceptualTaskOutput& cpu_output,
                                  const PerceptualFailures& failures) const {
        PerceptualTaskOutput output{{}, gpu_output.source_crop, gpu_output.distorted_crop,
                                    std::max(gpu_output.compared_frame_count, cpu_output.compared_frame_count),
                                    failures};
        for (const std::string& key : keys_) {
            if (failure_of(failures, key) != nullptr) {
                continue;
            }
            const MetricResultSet& found = gpu_output.metrics.has(key) ? gpu_output.metrics : cpu_output.metrics;
            if (const FrameMetricResult* frame = found.frame(key)) {
                output.metrics.add(*frame);
            } else if (const SequenceMetricResult* sequence = found.sequence(key)) {
                output.metrics.add(*sequence);
            } else {
                throw PerceptualRunError("Selected backend did not produce " + key + ".");
            }
        }
        return output;
    }

    const PerceptualRequest& request_;
    const std::vector<std::string>& keys_;
    bool together_;
    const PerceptualHooks& hooks_;
    const CancelToken& cancel_;
    const PerceptualRoutes& routes_;
    std::optional<VshipDevice> device_;   // the GPU Vship runs on, once found
    std::vector<std::string> gpu_;        // the metrics set to run on the GPU
    std::vector<std::string> cpu_;        // the metrics set to run on the CPU
    std::vector<std::string> gpu_only_;   // the GPU's metrics that run on no CPU (CVVDP)
    std::vector<std::string> retryable_;  // the metrics the CPU can take over
};

}  // namespace

PerceptualRoutes app_perceptual_routes() {
    PerceptualRoutes routes;
    routes.find_gpu = [] {
        forget_failed_vship_probe();
        return detect_vship_device();
    };
    routes.gpu_check_failed = vship_check_failed;
    routes.find_crops = [](const PerceptualRequest& request, const StatusCallback& on_status,
                           const CancelToken& cancel) {
        return resolve_crops(request.environment.ffmpeg, request.source, request.distorted, request.options.crop_mode,
                             cancel, on_status);
    };
    routes.run_gpu = [](const PerceptualRequest& request, const std::vector<std::string>& keys,
                        const VshipDevice& device, const CropPair& crops, bool together, const PerceptualHooks& hooks,
                        const CancelToken& cancel) {
        return VshipTask(request, keys, device, crops.first, crops.second, together, hooks, cancel).run();
    };
    if (vship51_bundled()) {
        routes.run_vship51 = [](const PerceptualRequest& request, const std::vector<std::string>& keys,
                                const VshipDevice& device, const CropPair& crops, bool together,
                                const PerceptualHooks& hooks, const CancelToken& cancel) {
            return VshipTask(request, keys, device, crops.first, crops.second, together, hooks, cancel, true).run();
        };
    }
    routes.run_cpu = [](const PerceptualRequest& request, const std::vector<std::string>& keys,
                        const std::optional<CropPair>& crops, const ProgressCallback& on_progress,
                        const StatusCallback& on_status, const CancelToken& cancel) {
        return run_cpu_perceptual_task(request, keys, crops, on_progress, on_status, cancel);
    };
    return routes;
}

PerceptualTaskOutput run_perceptual_task(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                         bool together, const PerceptualHooks& hooks, const CancelToken& cancel,
                                         const PerceptualRoutes& routes) {
    return MetricRouting(request, keys, together, hooks, cancel, routes).run();
}

PerceptualTaskOutput remade_vship_pass(
    const std::function<PerceptualTaskOutput(bool native, bool every_source_frame)>& attempt,
    const StatusCallback& on_status, const CancelToken& cancel) {
    bool native = true;
    bool every_source_frame = false;
    bool input_retried = false;
    while (true) {
        try {
            return attempt(native, every_source_frame);
        } catch (const VshipInputError& error) {
            // A failure of the input alone, such as an FFmpeg that ended
            // partway through a frame, may not come again; stock Vship 5.1.2
            // would read the same frames, and score them otherwise.
            if (input_retried || cancel.cancelled()) {
                throw;
            }
            logger.warning(std::string("Vship pass: ") + error.what() + " It is made again.");
            if (on_status) {
                on_status({"Reading the videos failed (" + std::string(error.what()) + "); trying again\xE2\x80\xA6"});
            }
            input_retried = true;
        } catch (const GpuDecodeFailedError& error) {
            if (!native) {
                throw VshipUnavailableError(error.what());
            }
            logger.warning(std::string("Decoding in the Vship pass failed; decoding through FFmpeg instead: ") +
                           error.what());
            if (on_status) {
                on_status({decoding_failed_status(error)});
            }
            native = false;
        } catch (const FramesApartError& error) {
            if (every_source_frame) {
                throw VshipUnavailableError(error.what());
            }
            logger.info(std::string("Vship pass: ") + error.what() + "; it is made again with every source frame");
            every_source_frame = true;
        }
    }
}

nlohmann::json vship_pass_task(const nlohmann::json& request, const std::filesystem::path& folder) {
    PerceptualRequest perceptual;
    perceptual.environment.ffmpeg = request.at("ffmpeg").get<std::string>();
    perceptual.environment.ffprobe = request.at("ffprobe").get<std::string>();
    perceptual.source = video_from(request.at("source"));
    perceptual.distorted = video_from(request.at("distorted"));
    perceptual.options = options_from(request.at("options"));
    request.at("backends").get_to(perceptual.backends);
    perceptual.cvvdp = CvvdpSettings::from_parameters(request.at("cvvdp"));
    PassPlan plan;
    request.at("keys").get_to(plan.keys);
    plan.device = vship_device_from(request.at("device")).value();
    plan.source_crop = crop_from(request.at("source_crop"));
    plan.distorted_crop = crop_from(request.at("distorted_crop"));
    const nlohmann::json& decoding = request.at("decoding");
    plan.decoding.source =
        decoding.at("source").is_null() ? std::nullopt : std::optional(decoding.at("source").get<std::string>());
    plan.decoding.distorted = decoding.at("distorted").is_null()
                                  ? std::nullopt
                                  : std::optional(decoding.at("distorted").get<std::string>());
    plan.vship51 = request.value("vship51", false);
    // The task's folder, which the run removes even when this process is
    // ended (Ctrl+C reaches it too): no file of FFmpeg's timestamps is left.
    plan.scratch_folder = folder;
    try {
        const PerceptualTaskOutput output = score_vship_pass(
            perceptual, plan,
            [](long long done, long long whole, double fps) {
                send_isolated_message({{"progress", {done, whole, fps}}});
            },
            [](const RunStatus& status) { send_isolated_message({{"status", status_json(status)}}); },
            CancelToken());  // the parent cancels by ending this process
        return output_json(output, folder);
    } catch (const ComparisonCutShortError& error) {
        return {{"cut_short", error.what()}};
    }
}

}  // namespace vml
