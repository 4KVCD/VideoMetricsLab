#include "vml/gpu_frames.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <regex>
#include <system_error>
#include <thread>

#include "vml/app_log.h"
#include "vml/ffmpeg_tools.h"
#include "vml/frame_sync.h"
#include "vml/frame_sync_by_time.h"
#include "vml/model_select.h"
#include "vml/platform/library.h"
#include "vml/platform/pipe.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// For each codec FFmpeg names: the decoders' number for it (NVIDIA's
// cudaVideoCodec; for VVC and FFV1, which NVIDIA's decoder does not have,
// the app's own), the bitstream filters that turn the container's packets
// into what the decoder's parser reads, and whether the GPUs' decoders are
// used for it (NVIDIA's would take VP9 and MPEG-2, but they are untried).
//
// H.264, HEVC and VVC get Annex B start codes, with the parameter sets from
// the stream's configuration before every keyframe (mp4toannexb adds them
// only before an IDR, and an open-GOP H.264 stream cut at a recovery point
// has none, so NVIDIA's decoder decoded nothing of it). AV1's packets are
// already in its low-overhead OBU format, and its sequence header goes to
// the parser separately. MPEG-2's sequence header, which a container may
// keep only in its configuration, is put before every keyframe too. VP9's
// packets need nothing, and FFV1's configuration goes to its decoder
// separately (configuration()).
//
// VVC's packets are given decode times that count up from far below any
// presentation time (setts). Matroska stores presentation times only, and
// FFmpeg's demuxer gives most of a VVC stream's packets decode times equal
// to their presentation times, out of order, and the rest none. FFmpeg's
// stream copy then "repairs" them by overwriting the presentation times,
// but the packet listing needs those as they are. Nothing reads these
// made-up decode times.
struct Codec {
    std::string_view name;
    int number;
    std::string_view filters;
    bool gpu;
};
constexpr Codec codecs[] = {
    {"h264", 4, "h264_mp4toannexb,dump_extra=freq=keyframe", true},
    {"hevc", 8, "hevc_mp4toannexb,dump_extra=freq=keyframe", true},
    {"av1", 11, "", true},
    {"vvc", 100, "vvc_mp4toannexb,dump_extra=freq=keyframe,setts=pts=PTS:dts=N-1000000000", false},
    {"vp9", 10, "", false},
    {"mpeg2video", 1, "dump_extra=freq=keyframe", false},
    {"ffv1", 101, "", false},
};

const Codec* find_codec(std::string_view name) {
    const auto found =
        std::find_if(std::begin(codecs), std::end(codecs), [&](const Codec& codec) { return codec.name == name; });
    return found == std::end(codecs) ? nullptr : found;
}

// The bit depths of the pixel formats FFmpeg decodes these codecs to: 4:2:0
// only, as 4:2:2 and 4:4:4 are not decoded here. 0 for any other.
int depth_of(std::string_view pix_fmt) {
    if (pix_fmt == "yuv420p" || pix_fmt == "yuvj420p") {
        return 8;
    }
    if (pix_fmt == "yuv420p10le" || pix_fmt == "yuv420p10be") {
        return 10;
    }
    return 0;
}

// The app's scaling algorithms, as the decoders' filters number them
// (native/scale_filter.h).
int scaler_number(std::string_view algorithm) {
    if (algorithm == "bilinear") {
        return 0;
    }
    if (algorithm == "lanczos") {
        return 2;
    }
    if (algorithm == "spline") {
        return 3;
    }
    return 1;  // bicubic
}

bool is_scaler(std::string_view algorithm) {
    return algorithm == "bilinear" || algorithm == "bicubic" || algorithm == "lanczos" || algorithm == "spline";
}

// What the decoder libraries answer (native/gpu_frames.h).
constexpr int nvf_frame = 1, nvf_end = 0, nvf_aborted = -2, nvf_timeout = 2;
// Packet flags (AV_PKT_FLAG_*).
constexpr int key_flag = 0x1, discard_flag = 0x4;
// The timestamp FFmpeg writes for none (AV_NOPTS_VALUE).
constexpr long long no_pts = std::numeric_limits<long long>::min();
// How much the pipes of FFmpeg's packets and of its listing hold.
constexpr std::size_t packet_pipe_bytes = 8 * 1024 * 1024;
constexpr std::size_t listing_pipe_bytes = 1024 * 1024;
// How long a decoder library may take to close a decoder. It normally takes
// milliseconds, but AMD's can fail to return: on a Radeon 780M, closing a
// decoder while Vulkan is in use in the same process can stop part-way.
constexpr std::chrono::seconds close_timeout(10);
// How long closing a stream waits for its feeding thread to stop.
constexpr std::chrono::seconds feeder_timeout(30);

std::atomic<int> stuck_closes{0};

const app_log::Logger logger("vml.gpu_frames");

// ------------------------------------------------------------------ binding

// The decoder libraries' Params and Info (native/gpu_frames.h), field for
// field.
struct NativeParams {
    int device;
    int codec;
    int bit_depth;
    int width;
    int height;
    int crop_x;
    int crop_y;
    int crop_w;
    int crop_h;
    int shift;
    int luma_only;
    int pool;
    const unsigned char* extradata;
    int extradata_size;
    int out_w;
    int out_h;
    int scaler;
    int widen;
    int cpu_scaling;
    int handover;
};

// A function of `library` that some builds of it lack: nullptr where it has
// none.
template <typename F>
F optional_function(const platform::DynamicLibrary& library, const char* name) {
    return reinterpret_cast<F>(library.find(name));
}

// A decoder library and the functions of it the app calls.
struct DecoderLibrary {
    DecoderLibrary(const std::filesystem::path& path, std::string_view backend)
        : library(path),
          open(library.function<decltype(open)>("nvf_open")),
          push(library.function<decltype(push)>("nvf_push")),
          finish(library.function<decltype(finish)>("nvf_finish")),
          pop(library.function<decltype(pop)>("nvf_pop")),
          download(library.function<decltype(download)>("nvf_download")),
          copy_luma(library.function<decltype(copy_luma)>("nvf_copy_luma")),
          release(library.function<decltype(release)>("nvf_release")),
          abort(library.function<decltype(abort)>("nvf_abort")),
          close(library.function<decltype(close)>("nvf_close")),
          error(library.function<decltype(error)>("nvf_error")),
          supports(library.function<decltype(supports)>("nvf_supports")) {
        const bool amd = backend == "amd" || backend == media_foundation_decoder;
        if (backend == "nvidia" || amd) {  // pictures handed over with no CPU copy
            unimport = library.function<decltype(unimport)>("nvf_unimport");
            pin = library.function<decltype(pin)>("nvf_pin");
            unpin = library.function<decltype(unpin)>("nvf_unpin");
            download_planes = library.function<decltype(download_planes)>("nvf_download_planes");
            copy_planes = library.function<decltype(copy_planes)>("nvf_copy_planes");
            // NVIDIA's CUDA imports any export; AMD's Vulkan only one from its
            // own GPU and driver.
            if (amd) {
                import_vulkan = library.function<decltype(import_vulkan)>("nvf_import_vulkan");
            } else {
                import_memory = library.function<decltype(import_memory)>("nvf_import");
            }
        }
        // AMD's: copies the GPU waits for, rather than the CPU; and a slot's
        // picture in place (Windows' decoder).
        timeline = optional_function<decltype(timeline)>(library, "nvf_timeline");
        copy_luma_async = optional_function<decltype(copy_luma_async)>(library, "nvf_copy_luma_async");
        copy_planes_async = optional_function<decltype(copy_planes_async)>(library, "nvf_copy_planes_async");
        slot_texture = optional_function<decltype(slot_texture)>(library, "nvf_slot_texture");
        if (backend == software_decoder) {
            set_libraries = library.function<decltype(set_libraries)>("nvf_set_libraries");
            libraries_error = library.function<decltype(libraries_error)>("nvf_libraries_error");
        }
        if (backend == "intel" || amd) {
            scale_note = library.function<decltype(scale_note)>("nvf_scale_note");
        }
    }

    platform::DynamicLibrary library;
    void* (*open)(const NativeParams* params, char* error, int error_size);
    int (*push)(void* decoder, const unsigned char* data, int size, long long pts);
    int (*finish)(void* decoder);
    int (*pop)(void* decoder, int timeout_ms, int* slot, long long* pts);
    int (*download)(void* decoder, int slot, void* host);
    int (*copy_luma)(void* decoder, int slot, unsigned long long address, long long pitch);
    void (*release)(void* decoder, int slot);
    void (*abort)(void* decoder);
    void (*close)(void* decoder);
    int (*error)(void* decoder, char* out, int size);
    int (*supports)(int device, int codec, int bit_depth, int width, int height, char* error, int error_size);
    // NVIDIA's and AMD's, whose pictures can stay on the GPU: GPU memory
    // another API allocated, as memory copy_luma writes into (NVIDIA's
    // through CUDA, AMD's through a Vulkan device of its own, which must be
    // the exporter's GPU and driver); memory of this process page-locked for
    // its copies; a picture's planes copied each to its own place, in this
    // process's memory or in imported GPU memory.
    int (*import_memory)(void* decoder, void* handle, unsigned long long bytes, int dedicated,
                         unsigned long long* address, void** memory) = nullptr;
    int (*import_vulkan)(void* decoder, void* handle, unsigned long long bytes, unsigned memory_type,
                         const unsigned char* device_uuid, const unsigned char* driver_uuid,
                         unsigned long long* address, void** memory) = nullptr;
    void (*unimport)(void* decoder, void* memory) = nullptr;
    int (*pin)(void* decoder, void* host, unsigned long long bytes) = nullptr;
    int (*unpin)(void* decoder, void* host) = nullptr;
    int (*download_planes)(void* decoder, int slot, void* const* planes, const long long* pitches) = nullptr;
    int (*copy_planes)(void* decoder, int slot, const unsigned long long* addresses, const long long* pitches) = nullptr;
    // AMD's: the timeline semaphore its copies signal, and the copies that
    // give the value it reaches once they are done instead of waiting; a
    // slot's picture in place, as a texture (Windows' decoder only).
    int (*timeline)(void* decoder, void** win32) = nullptr;
    int (*copy_luma_async)(void* decoder, int slot, unsigned long long address, long long pitch,
                           unsigned long long* value) = nullptr;
    int (*copy_planes_async)(void* decoder, int slot, const unsigned long long* addresses, const long long* pitches,
                             unsigned long long* value) = nullptr;
    int (*slot_texture)(void* decoder, int slot, void** texture, unsigned* width, unsigned* height, int* x,
                        int* y) = nullptr;
    // The software decoder's: the libraries it decodes with (their handles),
    // and why it cannot use one of them.
    int (*set_libraries)(void* avutil, void* avcodec, void* dav1d) = nullptr;
    int (*libraries_error)(char* out, int size) = nullptr;
    // Intel's and AMD's, which scale the pictures on the GPU when they can:
    // why they were scaled on the CPU instead.
    int (*scale_note)(void* decoder, char* out, int size) = nullptr;
};

// Text a library wrote into a buffer of ours, up to its terminating zero.
std::string buffer_text(const char* buffer, std::size_t size) {
    const char* const end = std::find(buffer, buffer + size, '\0');
    return replace_invalid_utf8(std::string_view(buffer, static_cast<std::size_t>(end - buffer)));
}

bool software_decode_off() {
    return ascii_lower(platform::environment_variable(software_decode_variable).value_or("")) == "ffmpeg";
}

std::filesystem::path library_path(std::string_view backend) {
    const char* name = backend == "nvidia"                   ? "nvdec_frames"
                       : backend == "intel"                  ? "vpl_frames"
                       : backend == "amd"                    ? "amf_frames"
                       : backend == media_foundation_decoder ? "mf_frames"
                       : backend == software_decoder         ? "software_frames"
                                                             : nullptr;
    if (name == nullptr) {
        throw std::invalid_argument("there is no decoder " + std::string(backend));
    }
    return resource_folder() / "native" / platform::shared_library_name(name);
}

// The software decoder's libraries: FFmpeg's libavutil and libavcodec with
// only the decoders it needs (LGPL, built by
// scripts/build_ffmpeg_decoders.ps1), and dav1d for AV1 (dav1d_library_path,
// built by scripts/build_dav1d.ps1), all beside each other. Without dav1d,
// FFmpeg decodes the AV1 videos FFmpeg would decode in software.
std::filesystem::path software_folder() { return resource_folder() / "tools" / "ffmpeg"; }
constexpr std::array<const char*, 2> ffmpeg_libraries{"avutil-61.dll", "avcodec-63.dll"};

// Loads a decoder library once, and keeps it for the life of the process.
// Throws GpuDecodeUnavailableError when it is not bundled or will not load.
const DecoderLibrary& load_decoder(std::string_view backend) {
    static std::mutex mutex;
    static std::map<std::string, std::unique_ptr<DecoderLibrary>, std::less<>> loaded;
    // The libraries the software decoder is given, kept loaded with it. Each
    // is loaded by its path, which finds the libraries it needs beside it.
    static std::vector<platform::DynamicLibrary> software_libraries;
    const std::lock_guard lock(mutex);
    if (const auto found = loaded.find(backend); found != loaded.end()) {
        return *found->second;
    }
    if (backend == software_decoder && software_decode_off()) {
        throw GpuDecodeUnavailableError(std::string(software_decode_variable) + " is set to ffmpeg");
    }
    const std::filesystem::path path = library_path(backend);
    std::error_code ignored;
    if (!std::filesystem::is_regular_file(path, ignored)) {
        throw GpuDecodeUnavailableError("the frame decoder (" + platform::path_text(path.filename()) +
                                        ") is not bundled");
    }
    std::unique_ptr<DecoderLibrary> library;
    try {
        library = std::make_unique<DecoderLibrary>(path, backend);
    } catch (const platform::LibraryError& error) {
        throw GpuDecodeUnavailableError(std::string("the GPU frame decoder could not be loaded: ") + error.what());
    }
    if (backend == software_decoder) {
        software_libraries.reserve(3);  // never moved: the decoder keeps their handles
        std::array<void*, 3> handles{};
        const std::array<std::filesystem::path, 3> paths{software_folder() / ffmpeg_libraries[0],
                                                         software_folder() / ffmpeg_libraries[1],
                                                         dav1d_library_path()};
        for (std::size_t i = 0; i < paths.size(); ++i) {
            if (std::filesystem::is_regular_file(paths[i], ignored)) {
                try {
                    software_libraries.emplace_back(paths[i]);
                    handles[i] = software_libraries.back().native_handle();
                } catch (const platform::LibraryError& error) {
                    // The decoder then refuses what it would need it for (nvf_supports).
                    logger.info("The software decoder cannot use " + platform::path_text(paths[i].filename()) + ": " +
                                error.what());
                }
            }
        }
        // What it cannot use, it refuses.
        if (library->set_libraries(handles[0], handles[1], handles[2]) != 0) {
            std::array<char, 512> said{};
            library->libraries_error(said.data(), static_cast<int>(said.size()));
            logger.info("The software decoder: " + buffer_text(said.data(), said.size()));
        }
    }
    return *loaded.emplace(std::string(backend), std::move(library)).first->second;
}

// The plan as the decoder libraries take it; `handover` asks AMD's decoder
// to keep its pictures on the GPU (NVIDIA's always does).
NativeParams native_params(const DecodePlan& plan, int device, int pool, const std::vector<unsigned char>& extradata,
                           bool handover = false) {
    const auto [out_w, out_h] = plan.output_size();
    return NativeParams{device,
                        find_codec(plan.codec)->number,
                        plan.bit_depth,
                        plan.width,
                        plan.height,
                        plan.crop_x,
                        plan.crop_y,
                        plan.crop_w,
                        plan.crop_h,
                        plan.shift,
                        plan.luma_only ? 1 : 0,
                        pool,
                        extradata.empty() ? nullptr : extradata.data(),
                        static_cast<int>(extradata.size()),
                        out_w,
                        out_h,
                        scaler_number(plan.scaler),
                        plan.widen,
                        0,  // cpu_scaling: Intel's and AMD's tests only
                        handover ? 1 : 0};
}

// Closes a decoder on a thread of its own, waited for only so long: a close
// that never returned would hang the whole run at its end, scores and all
// (AMD's decoder did, after failing on 10-bit H.264, which it is no longer
// given). A decoder that does not close is left open, and counted. Whether
// it closed.
bool close_decoder(const DecoderLibrary& library, void* handle) {
    struct Closing {
        std::mutex mutex;
        std::condition_variable closed_signal;
        bool closed = false;
    };
    const auto closing = std::make_shared<Closing>();
    std::thread closer([&library, handle, closing] {
        library.close(handle);
        const std::lock_guard lock(closing->mutex);
        closing->closed = true;
        closing->closed_signal.notify_all();
    });
    std::unique_lock lock(closing->mutex);
    const bool closed = closing->closed_signal.wait_for(lock, close_timeout, [&] { return closing->closed; });
    lock.unlock();
    if (closed) {
        closer.join();
    } else {
        closer.detach();  // the library stays loaded, so the thread may go on
        ++stuck_closes;
    }
    return closed;
}

// ------------------------------------------------------------- the packets

// A whole number in base 16 as Python's int(text, 16) reads FFmpeg's flags
// ("0x4"): its "0x" is optional.
std::optional<long long> hex_int(std::string_view text) {
    text = python_strip(text);
    if (text.starts_with("0x") || text.starts_with("0X")) {
        text.remove_prefix(2);
    }
    long long value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, 16);
    if (text.empty() || error != std::errc() || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

long long listing_int(std::string_view text) {
    if (const std::optional<long long> value = python_int(text)) {
        return *value;
    }
    throw std::invalid_argument("invalid literal for int() with base 10: '" + std::string(text) + "'");
}

// FFmpeg's warnings that mean the timestamps it copied may not be the ones
// its decoding would give a picture (it rewrites non-monotonic ones as it
// muxes).
const std::regex& timestamp_warning_pattern() {
    static const std::regex pattern(R"(timestamp|\bdts\b|\bpts\b|monoton)",
                                    std::regex::ECMAScript | std::regex::icase);
    return pattern;
}

// An FFmpeg that copies the video's first video stream out of its
// container. It writes the packets to its standard output (-f data), and
// each packet's timestamp and size to a framecrc listing on a pipe served
// here. A thread reads the listing as it arrives, so neither of FFmpeg's
// outputs waits for the other.
class PacketReader {
public:
    PacketReader(const std::string& ffmpeg, const std::filesystem::path& video, std::string_view codec)
        : listing_pipe_("gpu-frames", listing_pipe_bytes),
          command_(packet_reader_command(ffmpeg, video, codec, listing_pipe_.path())) {}

    ~PacketReader() {
        close();
        if (listing_thread_.joinable()) {
            listing_thread_.join();
        }
    }

    PacketReader(const PacketReader&) = delete;
    PacketReader& operator=(const PacketReader&) = delete;

    // Starts FFmpeg, and the thread that reads its listing. Throws
    // platform::ProcessError.
    void start() {
        platform::ProcessOptions options;
        options.output_pipe_bytes = packet_pipe_bytes;
        options.on_error_line = [this](std::string_view line) { note_error_line(line); };
        ffmpeg_.emplace(platform::ChildProcess::start(command_, options));
        listing_thread_ = std::thread([this] { read_listing(); });
    }

    // The next packet, in decode order: its bytes into `data`, its timestamp
    // and its flags. False at the end, once FFmpeg has succeeded without
    // rewriting any timestamp. Throws GpuDecodeFailedError.
    bool next(std::vector<std::uint8_t>& data, long long& pts, int& flags) {
        std::optional<Listed> item;
        {
            std::unique_lock lock(mutex_);
            while (listing_.empty() && !listing_ended_) {
                if (!listed_.wait_for(lock, std::chrono::milliseconds(200),
                                      [this] { return !listing_.empty() || listing_ended_; })) {
                    // If FFmpeg ended without opening the listing's pipe, the
                    // pipe is opened here, so that the wait for it ends.
                    lock.unlock();
                    if (ffmpeg_ && ffmpeg_->wait_for(std::chrono::milliseconds(0))) {
                        platform::InboundPipe::touch(listing_pipe_.path());
                    }
                    lock.lock();
                }
            }
            if (!listing_.empty()) {
                item = listing_.front();
                listing_.pop_front();
            } else if (listing_error_) {
                try {
                    std::rethrow_exception(listing_error_);
                } catch (const GpuDecodeFailedError&) {
                    throw;
                } catch (const std::exception& error) {
                    throw GpuDecodeFailedError(std::string("reading FFmpeg's packet list failed: ") + error.what());
                }
            }
        }
        if (!item) {
            return end_of_packets();
        }
        if (item->pts == no_pts) {
            throw GpuDecodeFailedError("a packet has no timestamp");
        }
        if (first_ && !(item->flags & key_flag)) {
            throw GpuDecodeFailedError("the video does not start with a keyframe");
        }
        if (const std::optional<std::string> warning = timestamp_warning()) {
            throw GpuDecodeFailedError("FFmpeg rewrote the video's timestamps: " + *warning);
        }
        first_ = false;
        if (item->size < 0 || item->size > INT_MAX) {
            throw GpuDecodeFailedError("FFmpeg listed a packet of " + std::to_string(item->size) + " bytes");
        }
        data.resize(static_cast<std::size_t>(item->size));
        for (std::size_t filled = 0; filled < data.size();) {
            const std::size_t count = ffmpeg_->read(data.data() + filled, data.size() - filled);
            if (count == 0) {
                throw GpuDecodeFailedError("FFmpeg's packets ended before its packet list");
            }
            filled += count;
        }
        pts = item->pts;
        flags = item->flags;
        return true;
    }

    // Ends FFmpeg, and with it the listing's thread; callable from another
    // thread than next()'s, and twice. next() then ends without an error.
    void close() {
        closed_ = true;
        if (ffmpeg_) {
            ffmpeg_->terminate();
            ffmpeg_->wait_for(std::chrono::seconds(10));
        }
        platform::InboundPipe::touch(listing_pipe_.path());  // releases a listing thread still waiting for FFmpeg
    }

    // The packets' time base, once the listing has said it: nothing when it
    // ended without saying it, or after a minute. Throws Cancelled.
    std::optional<Rational> wait_time_base(const CancelToken& cancel) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        std::unique_lock lock(mutex_);
        while (!time_base_ && !listing_ended_ && std::chrono::steady_clock::now() < deadline) {
            if (!listed_.wait_for(lock, std::chrono::milliseconds(100),
                                  [this] { return time_base_.has_value() || listing_ended_; })) {
                // As in next(): FFmpeg may have ended without opening the
                // listing's pipe.
                lock.unlock();
                cancel.throw_if_cancelled();
                if (ffmpeg_ && ffmpeg_->wait_for(std::chrono::milliseconds(0))) {
                    platform::InboundPipe::touch(listing_pipe_.path());
                }
                lock.lock();
            }
        }
        return time_base_;
    }

private:
    struct Listed {
        long long pts;
        long long size;
        int flags;
    };

    // The listing's thread: reads each packet's timestamp, size and flags,
    // and checks that the decode times go up.
    void read_listing() {
        try {
            listing_pipe_.connect();
            std::optional<long long> last_dts;
            std::string unread;
            std::vector<char> buffer(64 * 1024);
            while (true) {
                const std::size_t count = listing_pipe_.read(buffer.data(), buffer.size());
                unread.append(buffer.data(), count);
                std::size_t start = 0;
                for (std::size_t end = unread.find('\n'); end != std::string::npos; end = unread.find('\n', start)) {
                    take_listing_line(std::string_view(unread).substr(start, end - start), last_dts);
                    start = end + 1;
                }
                unread.erase(0, start);
                if (count == 0) {
                    if (!unread.empty()) {
                        take_listing_line(unread, last_dts);  // a last line without its line ending
                    }
                    break;
                }
            }
        } catch (...) {  // passed on to next(), which throws it
            const std::lock_guard lock(mutex_);
            listing_error_ = std::current_exception();
        }
        listing_pipe_.close();
        const std::lock_guard lock(mutex_);
        listing_ended_ = true;
        listed_.notify_all();
    }

    // One line of the listing: "stream, dts, pts, duration, size, crc", then
    // the packet's flags ("F=0x4") when they are not just "keyframe".
    void take_listing_line(std::string_view line, std::optional<long long>& last_dts) {
        line = python_strip(line);
        if (line.starts_with('#')) {
            take_header_line(line);
            return;
        }
        std::vector<std::string_view> fields;
        for (std::size_t start = 0;;) {
            const std::size_t comma = line.find(',', start);
            fields.push_back(python_strip(line.substr(start, comma == std::string_view::npos ? comma : comma - start)));
            if (comma == std::string_view::npos) {
                break;
            }
            start = comma + 1;
        }
        if (fields.size() < 6 || fields[0].empty() ||
            !std::all_of(fields[0].begin(), fields[0].end(), [](char c) { return c >= '0' && c <= '9'; })) {
            return;
        }
        int flags = key_flag;
        const auto printed = std::find_if(fields.begin() + 6, fields.end(),
                                          [](std::string_view field) { return field.starts_with("F="); });
        if (printed != fields.end()) {
            const std::optional<long long> value = hex_int(printed->substr(2));
            if (!value) {
                throw std::invalid_argument("invalid literal for int() with base 16: '" +
                                            std::string(printed->substr(2)) + "'");
            }
            flags = static_cast<int>(*value);
        }
        // A decode time that does not go up is one FFmpeg's muxer rewrote (to
        // the previous one, and the presentation time with it). It is caught
        // here, with the packet, because FFmpeg's warning about it can come
        // only after a run that stopped early has already used the packet.
        const long long dts = listing_int(fields[1]);
        if (dts != no_pts) {
            if (last_dts && dts <= *last_dts) {
                throw GpuDecodeFailedError("the video's decode timestamps do not go up");
            }
            last_dts = dts;
        }
        const Listed listed{listing_int(fields[2]), listing_int(fields[4]), flags};
        const std::lock_guard lock(mutex_);
        listing_.push_back(listed);
        listed_.notify_all();
    }

    // A line of the listing's header: "#tb 0: 1/1000" says the time base of
    // the packets' timestamps.
    void take_header_line(std::string_view line) {
        static const std::regex time_base_line(R"(#tb 0: (\d+)/(\d+))");
        std::match_results<std::string_view::const_iterator> match;
        if (!std::regex_search(line.begin(), line.end(), match, time_base_line,
                               std::regex_constants::match_continuous)) {
            return;
        }
        const std::optional<long long> numerator = python_int(match.str(1));
        const std::optional<long long> denominator = python_int(match.str(2));
        if (!numerator || !denominator || *denominator == 0) {
            throw GpuDecodeFailedError("FFmpeg reported a time base of " + match.str(1) + "/" + match.str(2));
        }
        const long long divisor = std::gcd(*numerator, *denominator);
        const std::lock_guard lock(mutex_);
        time_base_ = Rational{*numerator / divisor, *denominator / divisor};
        listed_.notify_all();
    }

    // FFmpeg's error output, as its thread reads it: its start is kept for
    // messages, and its first warning about timestamps noted.
    void note_error_line(std::string_view written) {
        const std::string text = replace_invalid_utf8(written);
        const std::string line(python_rstrip(text));
        const std::lock_guard lock(mutex_);
        if (error_lines_.size() < 50) {
            error_lines_.push_back(line);
        }
        if (!timestamp_warning_ && std::regex_search(line, timestamp_warning_pattern())) {
            timestamp_warning_ = line;
        }
    }

    std::optional<std::string> timestamp_warning() {
        const std::lock_guard lock(mutex_);
        return timestamp_warning_;
    }

    // The end of the start of FFmpeg's error output, for messages.
    std::string error_text() {
        const std::lock_guard lock(mutex_);
        std::string joined;
        for (std::size_t i = 0; i < error_lines_.size(); ++i) {
            joined += (i > 0 ? "\n" : "") + error_lines_[i];
        }
        return last_characters(std::string(python_strip(joined)), 500);
    }

    // After the listing's last packet: checks that FFmpeg succeeded and did
    // not rewrite any timestamps.
    bool end_of_packets() {
        const int code = ffmpeg_ ? ffmpeg_->wait() : 0;
        if (ffmpeg_) {
            ffmpeg_->standard_error();  // waits for its error output to end, so that every warning has been read
        }
        if (closed_) {
            return false;
        }
        if (const std::optional<std::string> warning = timestamp_warning()) {
            throw GpuDecodeFailedError("FFmpeg rewrote the video's timestamps: " + *warning);
        }
        if (code != 0 || first_) {
            // No packets at all is a failure too: FFmpeg can exit with 0 when
            // it refused to open an output.
            const std::string errors = error_text();
            throw GpuDecodeFailedError("FFmpeg failed copying the video's packets (exit code " + platform::exit_code_text(code) +
                                       ")" + (errors.empty() ? "" : ": " + errors));
        }
        return false;
    }

    platform::InboundPipe listing_pipe_;
    std::vector<std::string> command_;
    std::mutex mutex_;
    std::condition_variable listed_;
    std::deque<Listed> listing_;
    bool listing_ended_ = false;
    std::optional<Rational> time_base_;  // once the listing's header has said it
    std::exception_ptr listing_error_;
    std::vector<std::string> error_lines_;
    std::optional<std::string> timestamp_warning_;
    bool first_ = true;
    std::atomic<bool> closed_{false};
    // After what its error thread uses (note_error_line), so that it goes first.
    std::optional<platform::ChildProcess> ffmpeg_;
    std::thread listing_thread_;
};

// The video stream's configuration (FFmpeg's extradata), as ffprobe shows
// it; empty when there is none or it cannot be read.
std::vector<unsigned char> extradata(const std::string& ffprobe, const std::filesystem::path& video) {
    platform::ProcessOutput output;
    try {
        output = platform::run_program({ffprobe, "-v", "error", "-select_streams", std::string(video_stream),
                                        "-show_entries", "stream=extradata", "-show_data", "-of",
                                        "default=noprint_wrappers=1", input_path(video)},
                                       std::chrono::seconds(60));
    } catch (const platform::ProcessError&) {
        return {};
    }
    if (output.timed_out) {
        return {};
    }
    const auto is_hex = [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); };
    const auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    std::vector<unsigned char> data;
    const std::string_view text = output.standard_output;
    for (std::size_t start = 0; start < text.size();) {
        const std::size_t end = std::min(text.find('\n', start), text.size());
        // A line of the dump: "00000010: 0160 0000 0300 0000  .`......".
        const std::string_view line = python_strip(text.substr(start, end - start));
        start = end + 1;
        if (line.size() < 10 || !std::all_of(line.begin(), line.begin() + 8, is_hex) || line.substr(8, 2) != ": ") {
            continue;
        }
        std::string hex;
        for (std::size_t at = 10;;) {
            std::size_t run = 0;
            while (run < 4 && at + run < line.size() && is_hex(line[at + run])) {
                ++run;
            }
            if (run < 2) {
                break;
            }
            hex.append(line.substr(at, run));
            at += run;
            if (at < line.size() && line[at] == ' ') {
                ++at;
            }
        }
        for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
            data.push_back(static_cast<unsigned char>(nibble(hex[i]) * 16 + nibble(hex[i + 1])));
        }
    }
    return data;
}

// The part of the stream's configuration a decoder is given besides the
// packets: AV1's sequence header OBUs (its av1C box without the box's own
// 4-byte header, as FFmpeg's NVDEC decoder gives them to NVIDIA's parser),
// or FFV1's configuration record (which a version 3 stream needs to read its
// frames).
std::vector<unsigned char> configuration(const std::string& ffprobe, const std::filesystem::path& video,
                                         std::string_view codec) {
    if (codec == "av1") {
        std::vector<unsigned char> data = extradata(ffprobe, video);
        if (data.size() > 4 && (data[0] & 0x80)) {
            return std::vector<unsigned char>(data.begin() + 4, data.end());
        }
        return {};
    }
    return codec == "ffv1" ? extradata(ffprobe, video) : std::vector<unsigned char>();
}

}  // namespace

std::string decoding_failed_status(const GpuDecodeFailedError& error) {
    const std::string what = error.backend() == software_decoder ? "Decoding in the app" : "GPU decoding";
    return what + " failed (" + error.what() + "); decoding through FFmpeg instead…";
}

std::size_t DecodePlan::frame_bytes() const {
    const auto [columns, rows] = output_size();
    const auto sample = static_cast<std::size_t>(bytes_per_sample());
    const auto w = static_cast<std::size_t>(columns), h = static_cast<std::size_t>(rows);
    const std::size_t luma = w * h * sample;
    if (luma_only) {
        return luma;
    }
    return luma + 2 * ((w + 1) / 2) * ((h + 1) / 2) * sample;
}

DecodePlan plan_decode(const VideoInfo& info, const std::optional<CropBox>& crop, int shift, bool luma_only,
                       const std::optional<std::pair<int, int>>& size, const std::string& algorithm) {
    const std::string codec = ascii_lower(info.codec_name);
    if (find_codec(codec) == nullptr) {
        throw GpuDecodeUnavailableError((info.codec_name.empty() ? "this codec" : info.codec_name) +
                                        " is decoded by FFmpeg");
    }
    if (info.is_raw_stream()) {
        // Decoding here needs each packet's timestamp, and a raw stream's
        // packets have none: the pass would start only to fail.
        throw GpuDecodeUnavailableError("a raw stream is decoded by FFmpeg");
    }
    const int depth = depth_of(ascii_lower(info.pix_fmt));
    if (depth == 0) {
        throw GpuDecodeUnavailableError((info.pix_fmt.empty() ? "this pixel format" : info.pix_fmt) +
                                        " is decoded by FFmpeg");
    }
    if (info.width <= 0 || info.height <= 0) {
        throw GpuDecodeUnavailableError("the video's size is unknown");
    }
    if (info.width % 2 != 0 || info.height % 2 != 0) {
        // The decoders refuse it only once started (NVIDIA's decodes to an
        // even size), so the pass would start, fail and be made again.
        throw GpuDecodeUnavailableError("an odd-sized video is decoded by FFmpeg");
    }
    DecodePlan plan;
    plan.codec = codec;
    plan.bit_depth = depth;
    plan.width = info.width;
    plan.height = info.height;
    if (!crop || crop->is_noop(info.width, info.height)) {
        plan.crop_w = info.width;
        plan.crop_h = info.height;
    } else {
        // FFmpeg's crop filter keeps a 4:2:0 crop on whole chroma samples:
        // its left and top edges move to the even sample at or before them,
        // and an odd width or height loses its last column or row.
        plan.crop_x = crop->x & ~1;
        plan.crop_y = crop->y & ~1;
        plan.crop_w = crop->w & ~1;
        plan.crop_h = crop->h & ~1;
    }
    if (plan.crop_w <= 0 || plan.crop_h <= 0 || plan.crop_x + plan.crop_w > info.width ||
        plan.crop_y + plan.crop_h > info.height) {
        throw GpuDecodeUnavailableError("the crop is outside the picture");
    }
    std::tie(plan.out_w, plan.out_h) = size.value_or(std::pair(plan.crop_w, plan.crop_h));
    if (plan.out_w <= 0 || plan.out_h <= 0) {
        throw GpuDecodeUnavailableError("the size it is scaled to is empty");
    }
    plan.shift = depth > 8 ? shift : 0;
    plan.luma_only = luma_only;
    plan.scaler = is_scaler(algorithm) ? algorithm : "bicubic";
    return plan;
}

bool decodes_codec(std::string_view codec, std::string_view backend) {
    const Codec* found = find_codec(codec);
    return found != nullptr && (backend == software_decoder || found->gpu);
}

bool software_decoder_bundled() {
    std::error_code ignored;
    if (software_decode_off() || !std::filesystem::is_regular_file(library_path(software_decoder), ignored)) {
        return false;
    }
    return std::all_of(ffmpeg_libraries.begin(), ffmpeg_libraries.end(), [&](const char* name) {
        return std::filesystem::is_regular_file(software_folder() / name, ignored);
    });
}

bool decoder_available(std::string_view backend) {
    try {
        load_decoder(backend);
        return true;
    } catch (const GpuDecodeUnavailableError&) {
        return false;
    }
}

bool can_hand_over(const DecodePlan& plan, std::string_view backend) {
    if (backend == "nvidia") {
        return true;
    }
    return backend == "amd" && !plan.scaled() && plan.widen == 0;
}

bool hands_over_textures(std::string_view backend) {
    if (backend != "amd") {
        return false;
    }
    try {
        return load_decoder(media_foundation_decoder).slot_texture != nullptr;
    } catch (const GpuDecodeUnavailableError&) {
        return false;
    }
}

std::pair<std::vector<std::string>, std::string> software_codecs() {
    const DecoderLibrary* library = nullptr;
    try {
        library = &load_decoder(software_decoder);
    } catch (const GpuDecodeUnavailableError& error) {
        return {{}, error.what()};
    }
    std::vector<std::string> decodes;
    std::string why;
    for (const Codec& codec : codecs) {
        std::array<char, 512> error{};
        if (library->supports(0, codec.number, 8, 64, 64, error.data(), static_cast<int>(error.size())) != 0) {
            decodes.emplace_back(codec.name);
        } else if (why.empty()) {
            why = buffer_text(error.data(), error.size());
        }
    }
    return {decodes, decodes.empty() ? why : std::string()};
}

std::filesystem::path dav1d_library_path() { return software_folder() / "dav1d.dll"; }

std::pair<bool, std::string> decoder_supports(int device, const DecodePlan& plan, std::string_view backend) {
    if (!decodes_codec(plan.codec, backend)) {
        return {false, plan.codec + " is not decoded by the " + std::string(backend) + " decoder"};
    }
    if (backend == software_decoder && plan.scaled()) {
        return {false, "the software decoder does not scale; FFmpeg scales it"};
    }
    const DecoderLibrary* library = nullptr;
    try {
        library = &load_decoder(backend);
    } catch (const GpuDecodeUnavailableError& error) {
        return {false, error.what()};
    }
    std::array<char, 512> error{};
    const int ok = library->supports(device, find_codec(plan.codec)->number, plan.bit_depth, plan.width, plan.height,
                                     error.data(), static_cast<int>(error.size()));
    return {ok != 0, buffer_text(error.data(), error.size())};
}

std::optional<std::pair<std::string, std::string>> decoders_here(const HardwareDecoding& decoding) {
    std::vector<std::string> decoders;
    for (const std::optional<std::string>& accel : {decoding.source, decoding.distorted}) {
        std::optional<std::string> decoder;
        if (!accel) {
            if (software_decoder_bundled()) {
                decoder = std::string(software_decoder);
            }
        } else if (*accel == "cuda") {
            decoder = "nvidia";
        } else if (*accel == "qsv") {
            decoder = "intel";
        } else if (*accel == "d3d11va") {
            decoder = "amd";
        }
        if (!decoder) {
            return std::nullopt;
        }
        decoders.push_back(*decoder);
    }
    if (decoders[0] != decoders[1] && decoders[0] != software_decoder && decoders[1] != software_decoder) {
        return std::nullopt;  // two GPU makers' decoders
    }
    return std::pair(decoders[0], decoders[1]);
}

std::vector<std::string> packet_reader_command(const std::string& ffmpeg, const std::filesystem::path& video,
                                               std::string_view codec, const std::string& listing_pipe) {
    const Codec* found = find_codec(codec);
    if (found == nullptr) {
        throw std::invalid_argument(std::string(codec) + " is not decoded here");
    }
    std::vector<std::string> filters;
    if (!found->filters.empty()) {
        filters = {"-bsf:v", std::string(found->filters)};
    }
    const std::string stream = "0:" + std::string(video_stream);
    // -y: the listing's pipe already exists (this process serves it).
    // -copyinkf copies every packet, as a decoder would be given them (the
    // reader checks that the first is a keyframe).
    std::vector<std::string> command{ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
                                     "-i", input_path(video), "-map", stream, "-c:v", "copy", "-copyinkf"};
    command.insert(command.end(), filters.begin(), filters.end());
    command.insert(command.end(), {"-f", "data", "pipe:1", "-map", stream, "-c:v", "copy", "-copyinkf"});
    command.insert(command.end(), filters.begin(), filters.end());
    command.insert(command.end(), {"-flush_packets", "1", "-f", "framecrc", listing_pipe});
    return command;
}

int stuck_decoders() { return stuck_closes; }

// ------------------------------------------------------------------ checks

void PictureChecks::fed(long long pts, bool discard) {
    const std::lock_guard lock(mutex_);
    waiting_.push(pts);
    ++fed_count_;
    if (discard) {
        discard_.insert(pts);
    }
}

bool PictureChecks::take(long long pts) {
    if (last_shown_ && pts <= *last_shown_) {
        throw GpuDecodeFailedError(decoder_ + " gave pictures out of order");
    }
    bool discarded = false;
    {
        const std::lock_guard lock(mutex_);
        // An earlier-stamped packet is waiting only if it was fed, so its
        // picture was due before this one. A packet fed after this picture
        // came out would come out of order, which fails above.
        if (!waiting_.empty() && waiting_.top() < pts) {
            throw GpuDecodeFailedError(decoder_ + " gave " + std::to_string(shown_count_) + " pictures for " +
                                       std::to_string(shown_count_ + 1) + " packets");
        }
        if (waiting_.empty() || waiting_.top() != pts) {
            throw GpuDecodeFailedError(decoder_ + " gave pictures the packets do not have");
        }
        waiting_.pop();
        discarded = discard_.contains(pts);
    }
    ++shown_count_;
    last_shown_ = pts;
    return discarded;
}

void PictureChecks::check_end() {
    long long fed = 0;
    {
        const std::lock_guard lock(mutex_);
        fed = fed_count_;
    }
    if (shown_count_ != fed) {
        throw GpuDecodeFailedError(decoder_ + " gave " + std::to_string(shown_count_) + " pictures for " +
                                   std::to_string(fed) + " packets");
    }
}

// ----------------------------------------------------------------- streams

struct GpuFrameStream::State {
    explicit State(std::string_view backend_name)
        : backend(backend_name),
          decoder(backend_name == software_decoder ? "the software decoder" : "the GPU's decoder"),
          checks(decoder) {}

    // The decoder's last error message.
    std::string error_text() const {
        std::array<char, 1024> text{};
        library->error(handle, text.data(), static_cast<int>(text.size()));
        return buffer_text(text.data(), text.size());
    }

    std::string backend;
    std::string decoder;    // for messages
    std::string file_name;  // the video's, for the log
    PictureChecks checks;
    const DecoderLibrary* library = nullptr;
    // AV1's sequence header or FFV1's configuration, which the decoder may
    // read for as long as it runs.
    std::vector<unsigned char> configuration;
    void* handle = nullptr;
    // Whether the pictures are handed over on the GPU (handover()).
    bool handover = false;
    std::unique_ptr<PacketReader> reader;
    std::atomic<bool> closing{false};

    // Opens the decoder as `params` say with `opening`'s library, or returns
    // nullptr and puts what the library said into `said`.
    void* open(const DecoderLibrary& opening, const NativeParams& params, std::string& said) {
        std::array<char, 1024> error{};
        void* const opened = opening.open(&params, error.data(), static_cast<int>(error.size()));
        said = opened == nullptr ? buffer_text(error.data(), error.size()) : std::string();
        return opened;
    }

    // What ended the feeding thread, when it failed.
    std::mutex feed_mutex;
    std::optional<std::string> feed_error;

    std::thread feeder;
    std::mutex feeder_mutex;
    std::condition_variable feeder_ended_signal;
    bool feeder_ended = false;

    // The feeding thread: gives FFmpeg's packets to the decoder, noting each
    // packet's timestamp for the checks of the pictures. It keeps its own
    // copy of the decoder's handle, which a close that gives up on this
    // thread clears.
    void feed() {
        void* const decoder_handle = handle;
        try {
            std::vector<std::uint8_t> data;
            long long pts = 0;
            int flags = 0;
            while (reader->next(data, pts, flags)) {
                checks.fed(pts, (flags & discard_flag) != 0);
                if (library->push(decoder_handle, data.data(), static_cast<int>(data.size()), pts) != 0) {
                    return;  // a decoder error, or a close: next() reports it
                }
            }
            library->finish(decoder_handle);
        } catch (const std::exception& error) {
            stop_feeding(decoder_handle, error.what());
        } catch (...) {
            stop_feeding(decoder_handle, "feeding the decoder failed");
        }
    }

    // The feeding failed: next() says why, unless the stream is closing.
    void stop_feeding(void* decoder_handle, const std::string& why) {
        if (!closing) {
            const std::lock_guard lock(feed_mutex);
            feed_error = why;
        }
        library->abort(decoder_handle);
    }
};

GpuFrameStream::GpuFrameStream(const VideoInfo& info, const DecodePlan& plan, int device, int pool,
                               std::string_view backend, const std::string& ffmpeg, const std::string& ffprobe,
                               bool handover)
    : state_(std::make_shared<State>(backend)) {
    State& state = *state_;
    state.file_name = platform::path_text(info.path.filename());
    if (!decodes_codec(plan.codec, backend)) {
        throw GpuDecodeUnavailableError(plan.codec + " is not decoded by the " + std::string(backend) + " decoder");
    }
    state.library = &load_decoder(backend);
    state.reader = std::make_unique<PacketReader>(ffmpeg, info.path, plan.codec);
    state.configuration = configuration(ffprobe, info.path, plan.codec);
    state.handover = backend == "nvidia" || (handover && can_hand_over(plan, backend));
    std::string said;
    if (state.handover && backend == "amd") {
        // Windows' own decoder first (native/mf_frames.cpp): the same decoder
        // hardware as AMF's, faster and with less CPU. Where it cannot be used
        // (not bundled, or no decoder for the codec on the GPU; HEVC's and
        // AV1's come with Windows' video extensions), AMF's is.
        try {
            const DecoderLibrary& windows = load_decoder(media_foundation_decoder);
            state.handle = state.open(windows, native_params(plan, device, pool, state.configuration, true), said);
            if (state.handle != nullptr) {
                state.library = &windows;
            } else {
                logger.info(state.file_name + ": decoded by AMF, not Windows' decoder (" + said + ")");
            }
        } catch (const GpuDecodeUnavailableError& error) {
            logger.info(state.file_name + ": decoded by AMF (" + error.what() + ")");
        }
    }
    if (state.handover && backend != "nvidia" && state.handle == nullptr) {
        state.handle = state.open(*state.library, native_params(plan, device, pool, state.configuration, true), said);
        if (state.handle == nullptr) {
            logger.info(state.file_name + ": the GPU's pictures go through system memory (" + said + ")");
            state.handover = false;
        }
    }
    if (state.handle == nullptr) {
        state.handle = state.open(*state.library, native_params(plan, device, pool, state.configuration), said);
    }
    if (state.handle == nullptr) {
        throw GpuDecodeUnavailableError(said.empty() ? state.decoder + " could not start" : said);
    }
}

GpuFrameStream::~GpuFrameStream() {
    try {
        close();
    } catch (...) {
        // Closing gives up on what does not stop; nothing is left to report.
    }
}

void GpuFrameStream::start() {
    state_->reader->start();
    state_->feeder = std::thread([state = state_] {
        state->feed();
        const std::lock_guard lock(state->feeder_mutex);
        state->feeder_ended = true;
        state->feeder_ended_signal.notify_all();
    });
}

std::optional<GpuFrameStream::Picture> GpuFrameStream::next(const CancelToken& cancel) {
    State& state = *state_;
    while (true) {
        cancel.throw_if_cancelled();
        int slot = 0;
        long long pts = 0;
        int code = 0;
        while ((code = state.library->pop(state.handle, 100, &slot, &pts)) == nvf_frame) {
            bool discarded = false;
            try {
                discarded = state.checks.take(pts);
            } catch (const GpuDecodeFailedError& error) {
                release(slot);
                throw GpuDecodeFailedError(error.what(), state.backend);
            }
            if (!discarded) {
                return Picture{slot, pts};
            }
            release(slot);
        }
        if (code == nvf_timeout) {
            continue;
        }
        if (code == nvf_end) {
            try {
                state.checks.check_end();
            } catch (const GpuDecodeFailedError& error) {
                throw GpuDecodeFailedError(error.what(), state.backend);
            }
            return std::nullopt;
        }
        {
            const std::lock_guard lock(state.feed_mutex);
            if (state.feed_error) {
                throw GpuDecodeFailedError(*state.feed_error, state.backend);
            }
        }
        if (code == nvf_aborted) {
            throw GpuDecodeFailedError("decoding was stopped", state.backend);
        }
        const std::string said = state.error_text();
        throw GpuDecodeFailedError(said.empty() ? state.decoder + " failed" : said, state.backend);
    }
}

void GpuFrameStream::release(int slot) {
    if (state_->handle != nullptr) {
        state_->library->release(state_->handle, slot);
    }
}

Rational GpuFrameStream::time_base(const CancelToken& cancel) {
    const std::optional<Rational> found = state_->reader->wait_time_base(cancel);
    if (!found) {
        throw GpuDecodeFailedError("FFmpeg did not report the stream's time base");
    }
    return *found;
}

void GpuFrameStream::download(int slot, void* host) {
    State& state = *state_;
    if (state.library->download(state.handle, slot, host) != 0) {
        const std::string said = state.error_text();
        throw GpuDecodeFailedError(said.empty() ? state.decoder + " failed" : said, state.backend);
    }
}

void GpuFrameStream::copy_luma(int slot, std::uint64_t address, std::int64_t pitch) {
    State& state = *state_;
    if (state.library->copy_luma(state.handle, slot, address, pitch) != 0) {
        const std::string said = state.error_text();
        throw GpuDecodeFailedError(said.empty() ? "the GPU's decoder failed" : said);
    }
}

bool GpuFrameStream::copies_planes() const { return state_->handover && state_->library->copy_planes != nullptr; }

void GpuFrameStream::copy_planes(int slot, const std::array<std::uint64_t, 3>& addresses,
                                 const std::array<std::int64_t, 3>& pitches) {
    State& state = *state_;
    const std::array<unsigned long long, 3> places{addresses[0], addresses[1], addresses[2]};
    const std::array<long long, 3> rows{pitches[0], pitches[1], pitches[2]};
    if (state.library->copy_planes == nullptr ||
        state.library->copy_planes(state.handle, slot, places.data(), rows.data()) != 0) {
        const std::string said = state.library->copy_planes == nullptr ? std::string() : state.error_text();
        throw GpuDecodeFailedError(said.empty() ? "the GPU's decoder failed" : said);
    }
}

void* GpuFrameStream::timeline() {
    State& state = *state_;
    if (!state.handover || state.library->timeline == nullptr || state.handle == nullptr) {
        return nullptr;
    }
    void* handle = nullptr;
    if (state.library->timeline(state.handle, &handle) != 0) {
        return nullptr;
    }
    return handle;
}

std::uint64_t GpuFrameStream::copy_luma_async(int slot, std::uint64_t address, std::int64_t pitch) {
    State& state = *state_;
    unsigned long long value = 0;
    if (state.library->copy_luma_async == nullptr ||
        state.library->copy_luma_async(state.handle, slot, address, pitch, &value) != 0) {
        const std::string said = state.library->copy_luma_async == nullptr ? std::string() : state.error_text();
        throw GpuDecodeFailedError(said.empty() ? "the GPU's decoder failed" : said);
    }
    return value;
}

std::uint64_t GpuFrameStream::copy_planes_async(int slot, const std::array<std::uint64_t, 3>& addresses,
                                                const std::array<std::int64_t, 3>& pitches) {
    State& state = *state_;
    const std::array<unsigned long long, 3> places{addresses[0], addresses[1], addresses[2]};
    const std::array<long long, 3> rows{pitches[0], pitches[1], pitches[2]};
    unsigned long long value = 0;
    if (state.library->copy_planes_async == nullptr ||
        state.library->copy_planes_async(state.handle, slot, places.data(), rows.data(), &value) != 0) {
        const std::string said = state.library->copy_planes_async == nullptr ? std::string() : state.error_text();
        throw GpuDecodeFailedError(said.empty() ? "the GPU's decoder failed" : said);
    }
    return value;
}

bool GpuFrameStream::gives_textures() const { return state_->handover && state_->library->slot_texture != nullptr; }

GpuFrameStream::Texture GpuFrameStream::slot_texture(int slot) {
    State& state = *state_;
    Texture texture;
    if (state.library->slot_texture == nullptr ||
        state.library->slot_texture(state.handle, slot, &texture.handle, &texture.width, &texture.height, &texture.x,
                                    &texture.y) != 0) {
        const std::string said = state.library->slot_texture == nullptr ? std::string() : state.error_text();
        throw GpuDecodeFailedError(said.empty() ? "the decoder's picture is not a texture" : said);
    }
    return texture;
}

std::optional<std::pair<std::uint64_t, void*>> GpuFrameStream::import_memory(void* handle, std::uint64_t size,
                                                                             const SharedDevice* exporter) {
    State& state = *state_;
    if (!state.handover || state.handle == nullptr) {
        return std::nullopt;
    }
    unsigned long long address = 0;
    void* memory = nullptr;
    int result = -1;
    if (state.library->import_memory != nullptr) {  // NVIDIA's
        result = state.library->import_memory(state.handle, handle, size, 1, &address, &memory);
    } else if (state.library->import_vulkan != nullptr && exporter != nullptr) {  // AMD's
        result = state.library->import_vulkan(state.handle, handle, size, exporter->memory_type,
                                              exporter->device_uuid.data(), exporter->driver_uuid.data(), &address,
                                              &memory);
    }
    if (result != 0) {
        return std::nullopt;
    }
    return std::pair(static_cast<std::uint64_t>(address), memory);
}

void GpuFrameStream::unimport(void* memory) {
    if (state_->handle != nullptr && state_->library->unimport != nullptr) {
        state_->library->unimport(state_->handle, memory);
    }
}

const std::string& GpuFrameStream::backend() const { return state_->backend; }

bool GpuFrameStream::handover() const { return state_->handover; }

bool GpuFrameStream::pin(void* host, std::uint64_t size) {
    State& state = *state_;
    return state.handover && state.library->pin != nullptr && state.handle != nullptr &&
           state.library->pin(state.handle, host, size) == 0;
}

void GpuFrameStream::unpin(void* host) {
    if (state_->handle != nullptr && state_->library->unpin != nullptr) {
        state_->library->unpin(state_->handle, host);
    }
}

void GpuFrameStream::download_planes(int slot, const std::array<void*, 3>& planes,
                                     const std::array<std::int64_t, 3>& pitches) {
    State& state = *state_;
    const std::array<long long, 3> rows{pitches[0], pitches[1], pitches[2]};
    if (state.library->download_planes == nullptr ||
        state.library->download_planes(state.handle, slot, planes.data(), rows.data()) != 0) {
        const std::string said = state.library->download_planes == nullptr ? std::string() : state.error_text();
        throw GpuDecodeFailedError(said.empty() ? "the GPU's decoder failed" : said);
    }
}

void GpuFrameStream::close() {
    State& state = *state_;
    if (state.handle == nullptr) {
        return;
    }
    state.closing = true;
    if (state.library->scale_note != nullptr) {
        std::array<char, 1024> note{};
        state.library->scale_note(state.handle, note.data(), static_cast<int>(note.size()));
        if (const std::string text = buffer_text(note.data(), note.size()); !text.empty()) {
            logger.warning(state.file_name + ": scaled on the CPU, not the GPU: " + text);
        }
    }
    state.library->abort(state.handle);
    state.reader->close();
    bool feeder_stopped = true;
    if (state.feeder.joinable()) {
        std::unique_lock lock(state.feeder_mutex);
        feeder_stopped = state.feeder_ended_signal.wait_for(lock, feeder_timeout, [&] { return state.feeder_ended; });
        lock.unlock();
        if (feeder_stopped) {
            state.feeder.join();
        } else {
            state.feeder.detach();  // it holds the state, which lives on with it
        }
    }
    if (!feeder_stopped) {
        // Never seen: leaking the decoder is safer than freeing it under the
        // running thread.
        logger.error("The " + state.backend + " decoder's feeding thread did not stop; its decoder is left open");
    } else if (!close_decoder(*state.library, state.handle)) {
        logger.error(state.file_name + ": the " + state.backend + " decoder did not close in " +
                     std::to_string(close_timeout.count()) + " s; it is left open");
    }
    state.handle = nullptr;
}

void widen_to(DecodePlan& plan, const VideoInfo& info, int bit_depth) {
    if (plan.bit_depth < bit_depth) {
        const std::string range = ascii_lower(info.color_range);
        const bool full = range == "pc" || range == "jpeg" || range == "full" ||
                          (range.empty() && ascii_lower(info.pix_fmt).starts_with("yuvj"));
        plan.widen = full ? widen_repeat : widen_shift;
    }
    if (plan.bit_depth > bit_depth) {
        throw GpuDecodeUnavailableError("the videos are compared at " + std::to_string(bit_depth) +
                                        " bits and one is " + std::to_string(plan.bit_depth) + "-bit");
    }
}

void score_frame_pairs(GpuFrameStream& test, GpuFrameStream& reference, const VideoInfo& distorted,
                       const std::optional<std::string>& duration_limit, long long total_frames,
                       const std::function<void(int, int)>& score,
                       const std::function<void(long long, long long, double)>& on_progress,
                       const CancelToken& cancel, const SlotReleases& releases) {
    SlotReleases release = releases;
    if (!release.test) {
        release.test = [&test](int slot) { test.release(slot); };
    }
    if (!release.reference) {
        release.reference = [&reference](int slot) { reference.release(slot); };
    }
    if (by_timestamp()) {
        score_timed_pairs(test, reference, duration_limit, total_frames, score, on_progress, cancel, release);
        return;
    }
    const std::optional<long long> limit = frames_within(duration_limit, frame_rate(distorted));
    const auto started = std::chrono::steady_clock::now();
    auto reported = started;
    long long count = 0;
    while (!limit || count < *limit) {
        const std::optional<GpuFrameStream::Picture> test_picture = test.next(cancel);
        if (!test_picture) {
            break;
        }
        const std::optional<GpuFrameStream::Picture> reference_picture = reference.next(cancel);
        if (!reference_picture) {
            release.test(test_picture->slot);
            break;
        }
        score(test_picture->slot, reference_picture->slot);
        release.test(test_picture->slot);
        release.reference(reference_picture->slot);
        ++count;
        const auto now = std::chrono::steady_clock::now();
        if (on_progress && now - reported >= std::chrono::milliseconds(250)) {
            reported = now;
            const double seconds = std::chrono::duration<double>(now - started).count();
            on_progress(count, total_frames, static_cast<double>(count) / seconds);
        }
    }
}

}  // namespace vml
