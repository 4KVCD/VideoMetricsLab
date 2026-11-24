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
// ("%3.4f"). A 6:1:1 metric (PSNR YUV, SSIM YUV) is (6 Y + Cb + Cr) / 8 of
// its extractor's luma feature and its two chroma features, rounded once.
struct CpuFeature {
    std::string_view key;  // the app's metric key: "psnr", "ssim", "xpsnr", "psnr_yuv", "ssim_yuv"
    std::string_view extractor;
    std::string_view feature;
    int decimals;
    std::string_view cb = {};  // a 6:1:1 metric's chroma features; empty otherwise
    std::string_view cr = {};
};

// The extractor of metric `key`, or nullptr for one not scored this way.
const CpuFeature* cpu_feature(std::string_view key);

// Whether `metrics` (cpu_feature keys) read the chroma planes: a 6:1:1
// metric does; the others read the luma alone.
bool cpu_metrics_read_chroma(const std::vector<std::string>& metrics);

// What a score calculated this way records as having calculated it:
// "libvmaf-fast v1 (CPU)".
std::string cpu_library_build();

// Scores PSNR, SSIM and XPSNR of frame pairs given in order, with
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
    // Throws LibvmafError.
    CpuScorer(int width, int height, int bit_depth, std::vector<std::string> metrics, int n_subsample = 1,
              int threads = 0, int frame_rate = 0);
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

// How many pairs of pictures CpuScorer's pool holds: one pair per libvmaf
// thread plus one being filled, counting at most 8 threads, within 200 MB
// (270 MB with XPSNR; half as much again for pictures with `chroma`, so that
// they come to as many pairs), but at least two pairs; XPSNR gets one more,
// for the test pictures it keeps from the previous frames.
int cpu_pairs(int width, int height, int bit_depth, int threads, bool xpsnr, bool chroma);

// Copies the given planes (0 Y, 1 U, 2 V) of a frame into a libvmaf picture of
// its size, `sample` bytes a sample. Its planes are the frame's, but for an odd
// size's chroma planes, which libvmaf rounds down where FFmpeg rounds up: the
// picture's part of them is copied.
void fill_picture(VmafPicture& picture, const FramePlanes& frame, std::size_t sample,
                  std::initializer_list<int> planes);

// PSNR, SSIM and XPSNR (CpuScorer) with both videos decoded in this process,
// luma only unless a 6:1:1 metric reads the chroma, as score_decoded (gpu_vmaf.h) does for VMAF: by `decoders`
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
                             const std::string& scale_algorithm, const std::string& ffmpeg,
                             const std::string& ffprobe, bool want_vvdec,
                             const std::function<void()>& on_vvdec_opening,
                             const std::function<void(long long, long long, double)>& on_progress,
                             const CancelToken& cancel);

}  // namespace vml
