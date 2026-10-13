// PSNR, SSIM and XPSNR calculated in the app's own process, with the bundled
// libvmaf's CPU feature extractors (libvmaf.h): the same extractors FFmpeg's
// libvmaf filter runs, and libvmaf-fast's port of FFmpeg's xpsnr filter. The
// run (ffmpeg_run.h) decides when to use them instead of FFmpeg's filters.
// A port of vmaf_app/core/libvmaf_cpu.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/libvmaf.h"
#include "vml/pair_scorer.h"
#include "vml/video.h"

namespace vml {

class GpuFrameStream;  // gpu_frames.h

// The libvmaf CPU extractor of each metric the app scores this way: the
// extractor, the feature that is the metric's score, and the decimals kept.
// They match what FFmpeg's libvmaf filter is asked for ("name=psnr",
// "name=float_ssim") and how its log names and rounds the scores the app
// reads; XPSNR is rounded as FFmpeg's xpsnr filter writes its stats file
// ("%3.4f").
struct CpuFeature {
    std::string_view key;  // the app's metric key: "psnr", "ssim", "xpsnr"
    std::string_view extractor;
    std::string_view feature;
    int decimals;
};

// The extractor of metric `key`, or nullptr for one not scored this way.
const CpuFeature* cpu_feature(std::string_view key);

// What a score calculated this way records as having calculated it:
// "libvmaf-fast v1 (CPU)".
std::string cpu_library_build();

// Scores PSNR, SSIM and XPSNR of frame pairs given in order, with
// libvmaf-fast's own CPU extractors, in this process, on libvmaf's
// preallocated pictures. Running them here is faster than in FFmpeg, whose
// filter allocates, zeroes and copies two new pictures for every pair on its
// single filter thread before libvmaf's threads see them.
//
// The pictures hold luma only: the app keeps PSNR's luma score (psnr_y) and
// SSIM, which read nothing else, and libvmaf's PSNR skips the chroma planes
// of a picture without them. Frames are given packed as FFmpeg's rawvideo
// writes them (add; only the luma is read), or as a decoder's pictures
// (add_decoded).
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
    // The bytes of a frame's luma, which is what the decoders hand over.
    std::size_t luma_bytes() const { return luma_bytes_; }

    void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) override;

    // Scores one more pair held by the app's decoders, which decode luma
    // only. NVIDIA's decoder copies the luma into libvmaf's picture itself,
    // with the picture's row spacing, into memory it page-locks the first
    // time each of the pool's pictures comes round. The others write it
    // straight into the picture when the picture's rows are packed, as
    // libvmaf lays them out at most widths (multiples of 32 samples), and
    // otherwise into a buffer that is then copied in.
    void add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot);

    // The frames scored (every n_subsample-th) and each metric's scores, in
    // the order the metrics were given, rounded as the files the app reads
    // FFmpeg's scores from round them.
    PairScores finish() override;

private:
    VmafPicture take_picture();
    void give_back(VmafPicture& reference, VmafPicture& distorted);
    void score(VmafPicture& reference, VmafPicture& distorted);
    void download(VmafPicture& picture, GpuFrameStream& stream, int slot, std::size_t which);
    void fill(VmafPicture& picture, std::span<const std::uint8_t> frame, std::size_t size);
    void release();

    const Libvmaf& library_;
    std::vector<std::string> metrics_;
    unsigned step_ = 1;
    unsigned count_ = 0;
    VmafContext* context_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t row_bytes_ = 0;
    std::size_t luma_bytes_ = 0;
    std::size_t frame_bytes_ = 0;
    // The pictures' memory page-locked for NVIDIA's decoder, by address,
    // with the stream that locked it (nullptr where it could not).
    std::map<void*, GpuFrameStream*> pinned_;
    // A decoder's plane, written packed for a picture whose rows are padded:
    // the source's and the test video's.
    std::vector<std::uint8_t> staging_[2];
};

// Whether the bundled libvmaf-fast has the xpsnr extractor with the "psnr"
// option CpuScorer uses (a build after 3.2.0-fast.1). libvmaf ignores
// options it does not know, so the check scores a pair of tiny pictures and
// looks for both scores. Asked once.
bool cpu_scores_xpsnr();

// How many pairs of pictures CpuScorer's pool holds: one pair per libvmaf
// thread plus one being filled, counting at most 8 threads, within 200 MB
// (270 MB with XPSNR), but at least two pairs; XPSNR gets one more, for the
// test pictures it keeps from the previous frames.
int cpu_pairs(int width, int height, int bit_depth, int threads, bool xpsnr);

// PSNR, SSIM and XPSNR (CpuScorer) with both videos decoded in this process,
// luma only, as score_decoded (gpu_vmaf.h) does for VMAF: by `decoders`
// (the source's and the test video's), cropped as FFmpeg's crop filter
// crops and widened as FFmpeg converts, paired frame N with frame N as far
// as `duration_limit` (-t's text) keeps them. A comparison that is scaled
// is refused with GpuDecodeUnavailableError: the decoders do not scale
// exactly as FFmpeg does, and unlike VMAF's, these scores must be FFmpeg's.
// Throws GpuDecodeFailedError when decoding fails after it started,
// LibvmafError and Cancelled.
PairScores score_decoded_cpu(const VideoInfo& source, const VideoInfo& distorted,
                             const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                             int width, int height, int bit_depth, const std::vector<std::string>& metrics,
                             int n_subsample, const std::optional<std::string>& duration_limit, long long total_frames,
                             const std::pair<std::string, std::string>& decoders, int threads, int frame_rate,
                             const std::string& scale_algorithm, const std::string& ffmpeg,
                             const std::string& ffprobe,
                             const std::function<void(long long, long long, double)>& on_progress,
                             const CancelToken& cancel);

}  // namespace vml
