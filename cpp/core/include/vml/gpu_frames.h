// Decoding videos in the app, without FFmpeg decoding them, for the metrics
// that take the decoded pictures: VMAF on the GPU (Vulkan, CUDA and VMAF
// v1), Vship's, and PSNR, SSIM and XPSNR in the app (the Python app's
// vmaf_app/core/gpu_frames.py).
//
// On NVIDIA's GPUs the pictures are decoded into GPU memory
// (resources/native/nvdec_frames.dll); Intel's and AMD's decoders hand them
// over in system memory (vpl_frames.dll, amf_frames.dll), but for VMAF v1
// on the GPU AMD's pictures stay on the GPU too: decoded by Windows' own
// decoder (Media Foundation's, through Direct3D 11: mf_frames.dll) where it
// opens, else by AMF's, and handed over to Vulkan there. A video FFmpeg
// would decode in software is decoded on the CPU by FFmpeg's own decoders,
// built for the app (software_frames.dll). The libraries share one C API
// (native/gpu_frames.h) and are built by scripts/build_gpu_frames.ps1.
//
// Why: when FFmpeg decodes for VMAF on the GPU, it decodes each picture,
// copies it to system memory, converts it and writes it to a pipe, which
// the app reads and copies again: most of the CPU a GPU metric uses, and at
// 4K more than a pipe carries. Here FFmpeg only copies the compressed stream
// out of its container (-c:v copy, a few MB a second), and the decoder
// decodes it in this process.
//
// The pictures are the same, sample for sample, as FFmpeg's decoding gives,
// and a picture scaled here as FFmpeg's scale filter gives it
// (native/scale/scale_filter.h: swscale's arithmetic, as FFmpeg 9's x86 builds
// run it). The pictures that
// come out, and their timestamps, are checked against the packets that went
// in (PictureChecks): a picture the decoder dropped or added fails the run,
// and FFmpeg then decodes the videos. Whatever is not decoded here (a codec
// or format a decoder does not take, another GPU's decoder) is decoded by
// FFmpeg: GpuDecodeUnavailableError comes before the first picture, and
// GpuDecodeFailedError after it.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/frame_sync.h"
#include "vml/hardware_decoding.h"
#include "vml/pair_scorer.h"
#include "vml/video.h"

namespace vml {

// This video cannot be decoded here, so FFmpeg decodes it.
class GpuDecodeUnavailableError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Decoding failed after it had started; the caller decodes the video again
// through FFmpeg. backend() names the decoder, where it is known.
class GpuDecodeFailedError : public std::runtime_error {
public:
    explicit GpuDecodeFailedError(const std::string& message, std::string backend = "")
        : std::runtime_error(message), backend_(std::move(backend)) {}
    const std::string& backend() const { return backend_; }

private:
    std::string backend_;
};

// The status shown when decoding in the app failed after it had started,
// naming the GPU's decoder or the app's own.
std::string decoding_failed_status(const GpuDecodeFailedError& error);

// The decoder for the videos FFmpeg would decode in software.
inline constexpr std::string_view software_decoder = "software";

// The environment variable that, set to "ffmpeg", sends the videos FFmpeg
// decodes in software through FFmpeg's pipes instead of the software
// decoder, to compare the two.
inline constexpr std::string_view software_decode_variable = "VML_SOFTWARE_DECODE";

// How an 8-bit video is widened to the 10 bits it is compared at, as FFmpeg
// converts it: shifted left by 2, or, for full-range luma, also with its top
// bits repeated in the bottom ones.
inline constexpr int widen_shift = 1;
inline constexpr int widen_repeat = 2;

// How one video is decoded here: its codec and bit depth, the rectangle
// handed on (FFmpeg's crop, rounded as FFmpeg rounds it), the size it is
// scaled to and with which filter, and the sample layout.
struct DecodePlan {
    std::string codec;  // FFmpeg's name, lower case
    int bit_depth = 8;
    int width = 0;  // the video's own size
    int height = 0;
    int crop_x = 0;
    int crop_y = 0;
    int crop_w = 0;
    int crop_h = 0;
    // The right shift of 16-bit samples: 6 for yuv420p10le, 0 to keep P016's
    // layout (Vship reads it as 16-bit).
    int shift = 0;
    bool luma_only = false;  // only the luma plane, which VMAF reads
    int out_w = 0;           // the size the crop is scaled to (0: not scaled)
    int out_h = 0;
    std::string scaler = "bicubic";  // the app's scaling algorithm
    int widen = 0;  // widen_shift or widen_repeat: 8-bit pictures handed back as 10-bit (0: no)
    // The stream's film grain as the video is scored (film_grain_scoring.h):
    // 1 added, 0 left out, -1 as the decoder adds it by itself (none, or not
    // known); and the kinds the stream describes (FilmGrainKind bits).
    int film_grain = -1;
    unsigned film_grain_kinds = 0;
    // Whether only the software decoder may make the pictures, as it alone
    // adds the H.274 film grain the video is scored with (only_the_app_adds).
    // FFmpeg could not decode the video instead, so the software decoder then
    // also scales, and decodes a raw stream.
    bool software_only = false;
    // A raw stream: its packets have no timestamps, so each is given the sum
    // of the frame durations before it, and each picture, in display order,
    // the earliest of those still waiting (frames are paired by number).
    bool untimed = false;

    // The bytes per sample of the pictures handed back.
    int bytes_per_sample() const { return bit_depth > 8 || widen ? 2 : 1; }
    // The size of the pictures handed back.
    std::pair<int, int> output_size() const { return {out_w ? out_w : crop_w, out_h ? out_h : crop_h}; }
    bool scaled() const { return output_size() != std::pair(crop_w, crop_h); }
    // The bytes of one picture handed back, its planes packed.
    std::size_t frame_bytes() const;
};

// The plan for decoding the video here, cropped to `crop` and scaled to
// `size` with `algorithm`. Throws GpuDecodeUnavailableError, with the
// reason, when FFmpeg must decode it instead.
DecodePlan plan_decode(const VideoInfo& info, const std::optional<CropBox>& crop, int shift = 0,
                       bool luma_only = false, const std::optional<std::pair<int, int>>& size = std::nullopt,
                       const std::string& algorithm = "bicubic");

// Whether `backend`'s decoder ("nvidia", "intel", "amd", "software") is used
// for `codec` (FFmpeg's name): the GPUs' for H.264, HEVC and AV1, the
// software decoder for those and VVC, VP9, MPEG-2 and FFV1.
bool decodes_codec(std::string_view codec, std::string_view backend);

// Whether the software decoder can be used, judged from its files alone:
// its library and FFmpeg's libraries are there, and VML_SOFTWARE_DECODE is
// not set to ffmpeg. decoder_supports finds out for sure.
bool software_decoder_bundled();

// Whether `backend`'s decoder library is bundled and loads (the GPU is not
// asked).
bool decoder_available(std::string_view backend);

// Windows' own decoders (Media Foundation's) on AMD's GPUs, which can only
// hand their pictures over on the GPU: opened first where AMD's decoder
// would hand over, and AMD's own (AMF) where they cannot be (GpuFrameStream).
inline constexpr std::string_view media_foundation_decoder = "amd-mf";

// Whether `backend`'s decoder hands the plan's pictures over on the GPU
// (GpuFrameStream's handover): NVIDIA's always, and AMD's when they are
// neither scaled nor widened (native/amf_handover.h).
bool can_hand_over(const DecodePlan& plan, std::string_view backend);

// Whether `backend`'s decoder may let the plan's pictures be read where it
// holds them in system memory (GpuFrameStream::slot_planes): the software
// decoder's, when a download would only copy them (neither scaled nor
// widened, 10-bit samples kept in the low bits, as they are decoded).
bool can_lend_planes(const DecodePlan& plan, std::string_view backend);

// Whether `backend`'s decoder may give its pictures in place, as textures
// (GpuFrameStream::slot_texture): AMD's, through Windows' own decoder, where
// that is bundled. Whether a given stream's pictures come that way depends
// on its codec (GpuFrameStream::gives_textures).
bool hands_over_textures(std::string_view backend);

// What a Vulkan device importing memory that another exported must match,
// and the memory type to allocate it as: the Vulkan VMAF engine's
// (vv_shared_device), for AMD's decoder, whose own Vulkan device imports it
// (GpuFrameStream::import_memory).
struct SharedDevice {
    std::array<std::uint8_t, 16> device_uuid{};
    std::array<std::uint8_t, 16> driver_uuid{};
    std::uint32_t memory_type = 0;
};

// The codecs (FFmpeg's names) the software decoder can decode on this PC,
// after loading its libraries, and why when it can decode none. For the
// self-test: AV1 is among them only where dav1d is beside FFmpeg's libraries.
std::pair<std::vector<std::string>, std::string> software_codecs();

// dav1d.dll, which decodes AV1 in software, for the software decoder and
// for Video Compare's playback: beside FFmpeg's libraries, where a packaged
// build has the Python app's and a development build has none.
std::filesystem::path dav1d_library_path();

// Whether GPU `device`'s decoder takes the plan's codec, bit depth and size,
// and why not; for the software decoder, whether its libraries decode the
// codec. The software decoder is never asked to scale (FFmpeg scales on
// several threads, the decoder would on one).
std::pair<bool, std::string> decoder_supports(int device, const DecodePlan& plan, std::string_view backend);

// Why a GPU's decoder cannot scale the plan's pictures exactly as this
// FFmpeg's scale filter does, or nothing (nothing too for a plan that is not
// scaled): so large a downscale that swscale makes it in two steps, or an
// FFmpeg whose scale filter gives other pictures (checked once for each FFmpeg
// and algorithm, at 8 and 10 bits). FFmpeg then scales the video.
std::optional<std::string> scaling_refusal(const DecodePlan& plan, const std::string& ffmpeg);

// The decoders the app would decode the source and the test video with
// itself, or nothing when FFmpeg must decode them: each video by the decoder
// of the GPU FFmpeg would use (cuda: NVIDIA's, qsv: Intel's, d3d11va:
// AMD's), or by the software decoder where FFmpeg would decode in software.
// Two GPU makers' decoders are never used together.
std::optional<std::pair<std::string, std::string>> decoders_here(const HardwareDecoding& decoding);

// Whether pictures of two streams of these decoders may be downloaded at the
// same time, from two threads: the software decoder's and NVIDIA's keep each
// stream's state apart (native/software_frames.cpp, nvdec_frames.cpp); the
// other libraries are not known to (AMD's shares memory between its
// streams).
bool downloads_at_once(const std::string& source_decoder, const std::string& test_decoder);

// FFmpeg's command that copies the video's first video stream out of its
// container: the packets to its standard output, and each packet's
// timestamp and size to a framecrc listing on the pipe at `listing_pipe`.
// `input_options` go before the video (a raw stream's frame rate).
std::vector<std::string> packet_reader_command(const std::string& ffmpeg, const std::filesystem::path& video,
                                               std::string_view codec, const std::string& listing_pipe,
                                               const std::vector<std::string>& input_options = {});

// How many decoders this process gave up closing because their library
// never returned. Such a process cannot be trusted to end by itself (the
// stuck call holds the driver's locks), so a process of its own that ran a
// GPU run ends itself at once.
int stuck_decoders();

// The pictures at a video's start that FFmpeg's decoding drops, so that the
// packet reader does not give them to the decoders either (which may decode
// them, and could not be checked: their packets have no timestamps). An HEVC
// stream that starts at a CRA or BLA picture (a stream copy cut at an
// open-GOP random access point) can have RASL pictures after it, which refer
// to pictures before the cut; FFmpeg's decoder drops them, and reads only the
// parameter sets in their packets (H.265's NoRaslOutputFlag). They come
// before the first picture that is neither RASL nor RADL, in decode order.
// Only RASL pictures refer to them. Other codecs' packets are all fed.
class LeadingPictures {
public:
    enum class Verdict { feed, skip, refuse };

    explicit LeadingPictures(std::string_view codec) : open_(codec == "hevc") {}

    // Decides the next packet (Annex B, in decode order): skip one of the
    // first picture's RASL pictures; refuse one that carries parameter sets
    // too, which FFmpeg's decoder would read (FFmpeg then decodes the video).
    Verdict take(const std::vector<std::uint8_t>& packet);

    // How many packets take() has said to skip.
    int skipped() const { return skipped_; }

private:
    bool open_;  // still deciding: the video's first picture, or its leading pictures
    bool first_ = true;
    int skipped_ = 0;
};

// Whether a line of FFmpeg's error output while it copies a video's packets
// warns that the timestamps it copied may not be the ones its decoding would
// give a picture (it rewrites ones that do not go up as it muxes). Not the
// note that a packet has none: the reader refuses such a packet itself,
// unless it is one LeadingPictures skips.
bool warns_of_changed_timestamps(std::string_view line);

// The check of a decoder's pictures against the packets it was fed: they
// must come out in timestamp order, each the next packet's (every packet
// stamped earlier has had its picture), and at the end every packet must
// have had one. Checked as each picture comes out, a picture the decoder
// dropped or made up fails at once, not a whole pass later.
class PictureChecks {
public:
    // `decoder` names it in messages ("the GPU's decoder").
    explicit PictureChecks(std::string decoder) : decoder_(std::move(decoder)) {}

    // A packet fed to the decoder; `discard`: one the demuxer marked so (the
    // frames an MP4 edit list cuts off), decoded but never handed out, as
    // FFmpeg's decoding drops them. Called by the thread that feeds it.
    void fed(long long pts, bool discard);

    // Checks the picture stamped `pts` as it comes out. Returns whether it
    // is one to drop. Throws GpuDecodeFailedError.
    bool take(long long pts);

    // The smallest timestamp of the packets fed whose pictures have not come
    // out; nothing when there is none. An untimed stream's picture is
    // stamped with it (DecodePlan::untimed).
    std::optional<long long> earliest_waiting();

    // Throws GpuDecodeFailedError unless every packet fed has had its
    // picture. Call it at the stream's end.
    void check_end();

private:
    std::string decoder_;
    std::mutex mutex_;  // fed() comes from another thread than take()
    // The timestamps of the packets fed whose pictures have not come out
    // yet, smallest first.
    std::priority_queue<long long, std::vector<long long>, std::greater<>> waiting_;
    long long fed_count_ = 0;
    std::set<long long> discard_;
    long long shown_count_ = 0;
    std::optional<long long> last_shown_;
};

// One video decoded by `backend`'s decoder on GPU `device`, its pictures held
// in a pool of `pool` slots, in display order. A thread feeds FFmpeg's
// packets to the decoder; next() hands out the pictures, and each slot is
// given back with release() once its picture has been copied.
//
// NVIDIA's decoder keeps its pictures on the GPU, for import_memory, pin,
// copy_luma and the rest. With `handover`, AMD's does too where it can
// (can_hand_over): through Windows' own decoder where that opens, and
// otherwise through AMF (but AV1 not: AMF hands its pictures over wrongly).
// Where it cannot, it decodes without handing over, and handover() says so.
class GpuFrameStream {
public:
    // Throws GpuDecodeUnavailableError when the decoder cannot start.
    GpuFrameStream(const VideoInfo& info, const DecodePlan& plan, int device, int pool, std::string_view backend,
                   const std::string& ffmpeg, const std::string& ffprobe, bool handover = false);
    ~GpuFrameStream();
    GpuFrameStream(const GpuFrameStream&) = delete;
    GpuFrameStream& operator=(const GpuFrameStream&) = delete;

    // The decoder: "nvidia", "intel", "amd" or "software".
    const std::string& backend() const;

    // Whether the pictures are handed over on the GPU (import_memory and pin
    // work): NVIDIA's always, AMD's when opened with handover where it can.
    bool handover() const;

    // Whether this stream and `other` both hand over through the same
    // decoder library, so that memory one imports or pins is the other's to
    // copy into, and one timeline signals both streams' copies. NVIDIA's
    // always are; two AMD streams need not be: each falls back from Windows'
    // own decoder (mf_frames.dll) to AMF's (amf_frames.dll) on its own, and
    // each library has a Vulkan device, and imports, of its own.
    bool shares_gpu_memory_with(const GpuFrameStream& other) const;

    // Starts FFmpeg and the thread that feeds its packets to the decoder.
    void start();

    // A picture handed out: its slot, and its timestamp.
    struct Picture {
        int slot = 0;
        long long pts = 0;
    };

    // The next picture, or nothing at the video's end (which is then
    // checked, see PictureChecks). Checks `cancel` every 100 ms while it
    // waits. Throws GpuDecodeFailedError when decoding failed or the
    // pictures do not match the packets (FilmGrainRefusal for a plan only the
    // software decoder makes the pictures of: DecodePlan::software_only), and
    // Cancelled.
    std::optional<Picture> next(const CancelToken& cancel);

    // Gives a slot back to the decoder once its picture has been copied.
    void release(int slot);

    // The time base of the pictures' timestamps, once FFmpeg has reported it
    // (before the first packet), for pairing by timestamp. Waits up to a
    // minute. Throws GpuDecodeFailedError when FFmpeg gave none, and
    // Cancelled.
    Rational time_base(const CancelToken& cancel);

    // Copies the slot's picture, planes packed, to `host` (frame_bytes()
    // of the plan long).
    void download(int slot, void* host);

    // Whether slot_planes gives this stream's pictures (can_lend_planes).
    bool gives_planes() const;

    // The slot's picture where the decoder holds it, cropped, to be read in
    // place of a download: each plane's first sample and the bytes from one
    // row to the next. Valid until the slot is given back.
    FramePlanes slot_planes(int slot);

    // Copies the slot's luma plane to GPU memory at `address` that
    // import_memory imported, with rows `pitch` bytes apart (a decoder that
    // hands over).
    void copy_luma(int slot, std::uint64_t address, std::int64_t pitch);

    // Whether copy_planes can copy this decoder's pictures (one that hands
    // over).
    bool copies_planes() const;

    // As copy_luma, for the slot's Y, U and V planes, each to its own
    // address in imported memory with rows `pitches` bytes apart; an address
    // of 0 skips that plane.
    void copy_planes(int slot, const std::array<std::uint64_t, 3>& addresses,
                     const std::array<std::int64_t, 3>& pitches);

    // The timeline semaphore that copy_luma_async's and copy_planes_async's
    // copies signal (AMD's hand-over), as a Win32 handle for another Vulkan
    // device on the GPU to import; the caller closes it. nullptr where there
    // is none: the copies are then waited for instead.
    void* timeline();

    // As copy_luma and copy_planes, without waiting for the copy: the value
    // the timeline semaphore reaches once it is done, for the GPU to wait
    // for (0 when the copy was waited for after all).
    std::uint64_t copy_luma_async(int slot, std::uint64_t address, std::int64_t pitch);
    std::uint64_t copy_planes_async(int slot, const std::array<std::uint64_t, 3>& addresses,
                                    const std::array<std::int64_t, 3>& pitches);

    // Whether slot_texture gives this stream's pictures: those of Windows'
    // own decoder, handing over.
    bool gives_textures() const;

    // A slot's picture where the decoder left it: its texture's KMT handle (a
    // Direct3D 11 texture shared without a mutex), the texture's size, and
    // the place (x, y) of the picture's first sample in it. Valid until the
    // slot is given back.
    struct Texture {
        void* handle = nullptr;
        unsigned width = 0;
        unsigned height = 0;
        int x = 0;
        int y = 0;
    };
    Texture slot_texture(int slot);

    // Imports GPU memory that Vulkan allocated on this GPU and exported (a
    // Win32 handle to one buffer's allocation of `size` bytes), so that
    // copy_luma can write into it: its address here, and what unimport
    // takes. Nothing where the pictures are not handed over, or the driver
    // cannot share it. AMD's decoder imports into a Vulkan device of its own,
    // which must be `exporter`'s GPU and driver: without it, nothing is
    // imported. With AMD's decoder, memory one stream imports is copied into
    // by every stream of the process that hands over through the same
    // library (shares_gpu_memory_with).
    std::optional<std::pair<std::uint64_t, void*>> import_memory(void* handle, std::uint64_t size,
                                                                 const SharedDevice* exporter = nullptr);

    // Undoes import_memory.
    void unimport(void* memory);

    // Page-locks `size` bytes of this process's memory at `host`, so that a
    // decoder that hands over copies pictures into it directly (otherwise the
    // driver copies them through memory of its own). False where it cannot;
    // downloads then work as before. With AMD's decoder, memory pinned by one
    // stream is written into by every stream of the process that hands over
    // through the same library (shares_gpu_memory_with).
    bool pin(void* host, std::uint64_t size);

    // Undoes pin.
    void unpin(void* host);

    // Copies the slot's Y, U and V planes each to its own address, with rows
    // `pitches` bytes apart (a libvmaf picture's planes); a null address
    // skips that plane. A decoder that hands over only.
    void download_planes(int slot, const std::array<void*, 3>& planes, const std::array<std::int64_t, 3>& pitches);

    // Stops decoding and frees everything; safe to call twice.
    void close();

private:
    std::optional<Picture> next_picture(const CancelToken& cancel);
    // A failed decoding of a plan only this decoder makes the pictures of
    // (DecodePlan::software_only) as FilmGrainRefusal, which the routes'
    // fallbacks pass on: no other route would decode them.
    void throw_final_if_software_only(const GpuDecodeFailedError& error) const;

    struct State;
    std::shared_ptr<State> state_;  // shared with the feeding thread, which may outlive a close that gives up on it
};

// Throws GpuDecodeUnavailableError when one video is full range and the
// other limited (VideoInfo::full_range): the app's decoders give samples as
// they are, and FFmpeg brings the test video's to the source's range
// (ffmpeg_run.cpp's video_chains).
void require_same_range(const VideoInfo& source, const VideoInfo& distorted);

// Readies `plan` for frames compared at `bit_depth` bits: an 8-bit video is
// widened as FFmpeg converts it, full-range luma with its top bits repeated
// in the bottom ones. Throws GpuDecodeUnavailableError for a video deeper
// than the comparison.
void widen_to(DecodePlan& plan, const VideoInfo& info, int bit_depth);

// How the slots of the pairs scored are given back: each to its decoder, by
// default, as soon as no later pair uses it; or, for VMAF v1 reading the
// pictures where they are, once the GPU has read them
// (MultiScorer::release_later).
struct SlotReleases {
    std::function<void(int)> test;       // nothing: the test video's decoder's release()
    std::function<void(int)> reference;  // nothing: the source's decoder's release()
};

// Scores two videos decoded in the app pair by pair, as the Python app's
// frame_sync.decoded_frame_pairs pairs them: frame N of the test video with
// frame N of the source, until either video ends, or as far as
// `duration_limit` (-t's text) keeps frames at the test video's rate.
// `score` is called with each pair's slots (the test video's, then the
// source's), which are given back (`releases`) once it returns. Progress
// goes to `on_progress` every quarter of a second. When frames are paired by
// timestamp (by_timestamp), frame_sync_by_time.h's score_timed_pairs pairs
// them instead.
void score_frame_pairs(GpuFrameStream& test, GpuFrameStream& reference, const VideoInfo& distorted,
                       const std::optional<std::string>& duration_limit, long long total_frames,
                       const std::function<void(int, int)>& score,
                       const std::function<void(long long, long long, double)>& on_progress,
                       const CancelToken& cancel, const SlotReleases& releases = {});

}  // namespace vml
