// PSNR, SSIM and XPSNR calculated in the app's own process, with the bundled
// libvmaf's CPU feature extractors (libvmaf.h): the same extractors FFmpeg's
// libvmaf filter runs, and libvmaf-fast's port of FFmpeg's xpsnr filter. The
// run (ffmpeg_run.h) decides when to use them instead of FFmpeg's filters.
// A port of vmaf_app/core/libvmaf_cpu.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/frame_range.h"
#include "vml/libvmaf.h"
#include "vml/pair_scorer.h"
#include "vml/threads.h"
#include "vml/video.h"

namespace vml {

class GpuFrameStream;  // gpu_frames.h

// The libvmaf CPU extractor of each metric the app scores this way: the
// extractor, the feature that is the metric's score, and the decimals kept.
// They match what FFmpeg's libvmaf filter is asked for ("name=psnr",
// "name=float_ssim") and how its log names and rounds the scores the app
// reads; XPSNR is rounded as FFmpeg's xpsnr filter writes its stats file
// ("%3.4f"). A 6:1:1 metric (PSNR YUV, SSIM YUV, MS-SSIM YUV) is
// (6 Y + Cb + Cr) / 8 of its extractor's luma feature and its two chroma
// features, rounded once. CIEDE2000 is shown as the mean colour difference
// ΔE00 of the frame's pixels (0: identical), from libvmaf's
// 45 - 20 log10(mean ΔE00), which is +infinity for identical frames.
struct CpuFeature {
    enum class Value {
        feature,     // the feature as libvmaf gives it
        planes_611,  // (6 feature + cb + cr) / 8
        delta_e,     // 10^((45 - feature) / 20)
    };
    std::string_view key;  // the app's metric key: "psnr", "ssim", "cambi", "ciede2000", ...
    std::string_view extractor;
    std::string_view feature;
    int decimals;
    bool chroma = false;  // reads the chroma planes
    Value value = Value::feature;
    std::string_view cb = {};  // a 6:1:1 metric's chroma features; empty otherwise
    std::string_view cr = {};
};

// The extractor of metric `key`, or nullptr for one not scored this way.
const CpuFeature* cpu_feature(std::string_view key);

// Whether `metrics` (cpu_feature keys) read the chroma planes: the 6:1:1
// metrics, PSNR-HVS-M and CIEDE2000 do; the others read the luma alone.
bool cpu_metrics_read_chroma(const std::vector<std::string>& metrics);

// What CAMBI (the banding of the test video alone) needs to know of the test
// video when the pictures compared are not its own: its own size (a video
// scaled up to the source's is banded as it was encoded: CAMBI scales it
// back), its own bit depth (an 8-bit encode widened to the source's 10 bits
// is still dithered as 8 bits), and whether it is PQ (HDR10), whose levels
// CAMBI reads with the PQ curve. 0 or false: as the pictures.
struct CambiInput {
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    bool pq = false;
};

// The metrics of `metrics` (cpu_feature keys) that libvmaf cannot calculate
// for pictures of this size and bit depth, each with why. They are left out
// of the run and reported failed, so the others are still calculated: libvmaf
// starts its extractors at the first frame, and one that fails then fails
// them all. MS-SSIM needs 176 samples on each side of each plane it scores,
// CAMBI a picture at least 216 wide or high and at most 6092 wide plus high
// (its window outgrows its tables: 3840x2160 is the largest 16:9 size),
// PSNR-HVS-M 16 (two 8x8 chroma blocks) and at most 12 bits, CIEDE2000 8, 10,
// 12 or 16 bits.
std::map<std::string, std::string> cpu_metrics_unfit(const std::vector<std::string>& metrics, int width,
                                                     int height, int bit_depth, const CambiInput& cambi);

// What a score calculated this way records as having calculated it:
// "libvmaf-fast v1 (CPU)".
std::string cpu_library_build();

// Scores the CPU metrics (cpu_feature) of frame pairs given in order, with
// libvmaf-fast's own CPU extractors, in this process, on libvmaf's
// preallocated pictures. Running them here is faster than in FFmpeg, whose
// filter allocates, zeroes and copies two new pictures for every pair on its
// single filter thread before libvmaf's threads see them.
//
// The pictures hold luma only unless a 6:1:1 metric is asked for
// (cpu_metrics_read_chroma): PSNR's luma score (psnr_y), SSIM and XPSNR read
// nothing else, and libvmaf's PSNR skips the chroma planes of a picture
// without them. With a 6:1:1 metric they are 4:2:0, as the frames given; an
// odd size's last chroma column or row is then not compared (libvmaf's
// chroma planes are half the size rounded down, FFmpeg's rounded up).
// Frames are given packed as FFmpeg's rawvideo writes them (add), as planes
// wherever they are (add_planes), or as a decoder's pictures (add_decoded).
class CpuScorer : public PairScorer {
public:
    // `metrics` are cpu_feature keys. Every `n_subsample`th frame is scored,
    // on `threads` libvmaf threads (0: libvmaf's own choice). `frame_rate`
    // is XPSNR's, as FFmpeg's filter takes it: whole frames per second.
    // `cambi`: CAMBI's test video. Throws LibvmafError.
    CpuScorer(int width, int height, int bit_depth, std::vector<std::string> metrics, int n_subsample = 1,
              int threads = 0, int frame_rate = 0, const CambiInput& cambi = {});
    ~CpuScorer() override;
    CpuScorer(const CpuScorer&) = delete;
    CpuScorer& operator=(const CpuScorer&) = delete;

    std::size_t frame_bytes() const override { return frame_bytes_; }

    void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) override;
    void add_planes(const FramePlanes& reference, const FramePlanes& distorted) override;

    // Scores one more pair held by the app's decoders, which decode the
    // planes the pictures hold (luma only, or whole 4:2:0 frames). NVIDIA's
    // decoder copies them into libvmaf's picture itself, with the picture's
    // row spacings, into memory it page-locks the first time each of the
    // pool's pictures comes round. The software decoder's are copied in from
    // where the decoder holds them (GpuFrameStream::slot_planes). The others
    // write them straight into the picture when its planes lie as a packed
    // frame's, as libvmaf lays them out at most widths (multiples of 32
    // samples; 64 with chroma), and otherwise into a buffer that is then
    // copied in, as NVIDIA's are for an odd size with chroma. Where the decoders allow (downloads_at_once), or one of the
    // two is copied from where the software decoder holds it, the two come
    // at the same time, the source's on a thread of the scorer's.
    void add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot);

    // add_decoded, giving `also` the pair's two luma planes (the source's,
    // then the test video's) before they are scored: another scorer of the
    // same pair reads them (add_planes) rather than having the decoders write
    // them out again.
    void add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot,
                     const std::function<void(Plane, Plane)>& also);

    // The frames scored (every n_subsample-th) and each metric's scores, in
    // the order the metrics were given, rounded as the files the app reads
    // FFmpeg's scores from round them.
    PairScores finish() override;

private:
    VmafPicture take_picture();
    void give_back(VmafPicture& reference, VmafPicture& distorted);
    void score(VmafPicture& reference, VmafPicture& distorted);
    void download(VmafPicture& picture, GpuFrameStream& stream, int slot, std::size_t which);
    // Both pictures of a pair, at the same time where add_decoded says.
    void download_both(VmafPicture& reference_picture, GpuFrameStream& reference, int reference_slot,
                       VmafPicture& distorted_picture, GpuFrameStream& test, int test_slot);
    // Copies the planes the pictures hold into the picture.
    void fill(VmafPicture& picture, const FramePlanes& frame) const;
    // Whether the picture's planes lie as those of a packed frame, so that a
    // decoder can write the frame straight into it.
    bool laid_out_packed(const VmafPicture& picture) const;
    void release();

    const Libvmaf& library_;
    std::vector<std::string> metrics_;
    std::string cambi_feature_;  // CAMBI's score's name, which libvmaf builds from its options
    unsigned step_ = 1;
    unsigned count_ = 0;
    VmafContext* context_ = nullptr;
    bool chroma_ = false;  // the pictures hold the chroma planes too (a 6:1:1 metric)
    bool odd_ = false;     // an odd width or height
    std::size_t sample_ = 1;
    PackedFrame packed_;            // the frames given, planes packed
    std::size_t decoded_bytes_ = 0;  // what a decoder hands over: the luma, or the whole frame with chroma_
    std::size_t frame_bytes_ = 0;
    // The pictures' memory page-locked for NVIDIA's decoder, by address,
    // with the stream that locked it (nullptr where it could not).
    std::map<void*, GpuFrameStream*> pinned_;
    std::mutex pinned_mutex_;
    // A decoder's picture, written packed for a picture of libvmaf's whose
    // planes do not lie so: the source's and the test video's.
    std::vector<std::uint8_t> staging_[2];
    SideThread side_;  // takes in the source's picture beside the test video's
};

// Whether the bundled libvmaf-fast has the xpsnr extractor with the "psnr"
// option CpuScorer uses (a build after 3.2.0-fast.1). libvmaf ignores
// options it does not know, so the check scores a pair of tiny pictures and
// looks for both scores. Asked once.
bool cpu_scores_xpsnr();

// Whether `metrics` (cpu_feature keys) include one that takes far longer per
// pair than the decoders take to feed it: MS-SSIM, MS-SSIM YUV, CAMBI,
// CIEDE2000.
bool cpu_metrics_slow(const std::vector<std::string>& metrics);

// How many pairs of pictures CpuScorer's pool holds: one pair per libvmaf
// thread plus one being filled, counting at most 8 threads (every thread with
// a `slow` metric), within 200 MB (270 MB with XPSNR, 800 MB with a slow
// metric; half as much again for pictures with `chroma`, so that they come to
// as many pairs), but at least two pairs; XPSNR gets one more, for the test
// pictures it keeps from the previous frames.
int cpu_pairs(int width, int height, int bit_depth, int threads, bool xpsnr, bool chroma, bool slow);

// Copies the given planes (0 Y, 1 U, 2 V) of a frame into a libvmaf picture of
// its size, `sample` bytes a sample. Its planes are the frame's, but for an odd
// size's chroma planes, which libvmaf rounds down where FFmpeg rounds up: the
// picture's part of them is copied.
void fill_picture(VmafPicture& picture, const FramePlanes& frame, std::size_t sample,
                  std::initializer_list<int> planes);

// The CPU metrics (CpuScorer) with both videos decoded in this process,
// luma only unless a metric reads the chroma, as score_decoded (gpu_vmaf.h) does for VMAF: by `decoders`
// (the source's and the test video's), cropped as FFmpeg's crop filter
// crops and widened as FFmpeg converts, paired frame N with frame N, those
// `range` decodes (score_frame_pairs). A video compared at another
// size is scaled by its GPU decoder as FFmpeg's scale filter scales it; the
// software decoder does not scale, so such a video it would decode is
// refused (GpuDecodeUnavailableError, from decoder_supports) and FFmpeg
// scales it. A VVC video is decoded on the GPU by vvdec-vulkan where
// `want_vvdec` and it can (vvdec_chosen), on the GPU it picks,
// `on_vvdec_opening` called before it opens, its pictures copied by the GPU
// into host memory. Throws GpuDecodeUnavailableError before the first
// picture (backend() vvdec_decoder where vvdec-vulkan refused),
// GpuDecodeFailedError when decoding fails after it started, LibvmafError
// and Cancelled.
PairScores score_decoded_cpu(const VideoInfo& source, const VideoInfo& distorted,
                             const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                             int width, int height, int bit_depth, const std::vector<std::string>& metrics,
                             int n_subsample, const FrameRange& range, long long total_frames,
                             const std::pair<std::string, std::string>& decoders, int threads, int frame_rate,
                             const CambiInput& cambi, const std::string& scale_algorithm, const std::string& ffmpeg,
                             const std::string& ffprobe, bool want_vvdec,
                             const std::function<void()>& on_vvdec_opening,
                             const std::function<void(long long, long long, double)>& on_progress,
                             const CancelToken& cancel);

}  // namespace vml
