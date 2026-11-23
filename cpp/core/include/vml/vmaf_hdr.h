// HDR VMAF: Netflix's HDR VMAF models, scored in the app's own process. Each
// model asks for its pictures in ICtCp (PQ, BT.2020, limited range, 4:2:0,
// 16 bits) at the display size it was trained on; the pictures the videos
// give are converted to that first, with libvmaf's own conversion
// (vmaf_picture_convert, zimg).
//
// libvmaf would convert them itself, but one picture after another on the
// thread that hands them over, which took 87 % of a 4K run (11.5 fps on a
// Core Ultra 9 285K). The converted pictures' features are calculated on
// the GPU by VMAF v1's Vulkan engine (HDR VMAF has VMAF v1's four features,
// with other options), which gives libvmaf's values; or, without it, by
// libvmaf, which takes pictures already in the model's colours and size as
// they are. Where the engine calculates all four, it converts the pictures
// too, from a plan of libvmaf's (vmaf_picture_convert_gpu_plan: zimg's very
// steps, filters and tables), giving libvmaf's samples bit for bit, which
// is checked on the first frame. Otherwise HdrScorer converts them on
// threads of its own, both videos' and several frames' at once, exactly as
// libvmaf would.
#pragma once

#include <libvmaf/picture.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "vml/cancel.h"
#include "vml/frame_range.h"
#include "vml/libvmaf.h"
#include "vml/pair_scorer.h"
#include "vml/video.h"

namespace vml {

class GpuFrameStream;  // gpu_frames.h
struct GpuPlan;        // gpu_vmaf.h

// What an HDR VMAF model asks its pictures to be converted to (its JSON's
// "conversion_target"): the colours, pixel format and bit depth, and the
// short side of the display size (1080 or 2160).
struct HdrModelTarget {
    VmafColor colour{};
    VmafPixelFormat pixel_format = VMAF_PIX_FMT_UNKNOWN;
    unsigned bit_depth = 0;
    unsigned display = 0;
};

// The target of the model in `file`. Throws LibvmafError when the file is
// not an HDR VMAF model (no conversion target, or one the app does not read).
HdrModelTarget hdr_model_target(const std::filesystem::path& file);

// How a pair's pictures are coded, the source's and the test video's, for
// libvmaf's conversion.
struct HdrColours {
    VmafColor source{};
    VmafColor distorted{};
};

// The colours of a pair as HDR VMAF reads them, or why it cannot score the
// pair. The videos are read as the other metrics read them
// (with_shared_colours: an untagged video takes the other's tags). The
// transfer must say PQ: HDR VMAF is made for PQ video, and an HLG or SDR
// video's score would mean nothing. With PQ, an untagged matrix or untagged
// primaries are BT.2020's, as PQ video almost always is; matrices other than
// BT.2020 NCL and BT.709, and primaries other than BT.2020, P3-D65 and
// BT.709, are refused (libvmaf converts no others). Both are read in the
// source's range: FFmpeg's chain brings the test video to it, and the app's
// decoders decode only pairs of the same range (require_same_range).
std::variant<HdrColours, std::string> hdr_colours(VideoInfo source, VideoInfo distorted);

// What each picture of a pair is converted to, the source's and the test
// video's: libvmaf's own choice (its conversion_policy.c), so that it takes
// them as they are. Both are scaled to the size the source's short side
// scales to at the model's display size, the long side keeping its aspect
// ratio (rounded to even); a picture already at the display size keeps its
// size. Each picture's filter is its own: none when its size is kept,
// bicubic to upscale, lanczos to downscale. Throws std::invalid_argument
// where libvmaf refuses: a long side over 4x the display size, or two
// pictures whose orientations or aspect ratios differ (beyond 2 pixels of
// rounding).
std::array<VmafPictureConvertTarget, 2> hdr_targets(unsigned source_width, unsigned source_height,
                                                    unsigned distorted_width, unsigned distorted_height,
                                                    const HdrModelTarget& model);

// Whether libvmaf takes `picture` as it is for `target` (its
// vmaf_conversion_policy_picture_matches).
bool picture_matches(const VmafPicture& picture, const VmafPictureConvertTarget& target);

// The threads HdrScorer converts pictures on, for a PC with `cores` logical
// cores. With the features on the CPU, a third of them (a picture's
// conversion takes about a third of the CPU time its features take), at
// least 1 and at most 12; with them on the GPU, where the conversion is all
// the CPU does, two thirds, at least 1 and at most 16.
int hdr_conversion_threads(unsigned cores, bool features_on_gpu);

// Scores HDR VMAF of frame pairs given in order (PairScorer): on the GPU,
// both pictures converted and scored by the Vulkan engine where it can (the
// first pair's conversion held to libvmaf's, and a picture with every luma
// code, with the CPU's conversion from there on at any difference);
// otherwise both pictures copied into libvmaf pictures tagged with the
// pair's colours, converted on hdr_conversion_threads threads, one frame on
// each, each thread with its own conversion of each video, then given to the
// features' scorer (the GPU's, else libvmaf) in their order.
class HdrScorer : public PairScorer {
public:
    // Frames of `width` x `height` at `bit_depth`, 4:2:0, coded in
    // `colours`; the model in `model_file`; every `n_subsample`th frame
    // scored. The features on the GPU numbered `gpu` (Vulkan's number, -1
    // for the best one) where VMAF v1's engine can calculate them, else
    // (and without `gpu`) by libvmaf on `threads` threads (at least 1); the
    // log says which, where the pictures are converted, and why. Throws LibvmafError, and std::invalid_argument
    // for a size the model refuses (hdr_targets).
    HdrScorer(int width, int height, int bit_depth, const std::filesystem::path& model_file,
              const HdrColours& colours, int n_subsample, int threads, std::optional<int> gpu = std::nullopt);
    ~HdrScorer() override;
    HdrScorer(const HdrScorer&) = delete;
    HdrScorer& operator=(const HdrScorer&) = delete;

    std::size_t frame_bytes() const override;
    void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) override;
    void add_planes(const FramePlanes& reference, const FramePlanes& distorted) override;

    // The frames scored (every n_subsample-th) and their scores under
    // "vmaf_hdr", each kept to 6 decimals.
    PairScores finish() override;

private:
    struct State;
    std::unique_ptr<State> state_;
};

// HDR VMAF of a plan with backend "hdr" (GpuPlan::hdr_colours; its model the
// file named by its one model), both videos decoded in this process as
// whole frames, as score_decoded_v1 (vmaf_v1_gpu.h) decodes them, the
// pictures read in system memory. Throws as score_decoded_v1 does.
PairScores score_decoded_hdr(const VideoInfo& source, const VideoInfo& distorted,
                             const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                             const GpuPlan& plan, int n_subsample, const FrameRange& range, long long total_frames,
                             const std::string& scale_algorithm, const std::pair<std::string, std::string>& decoders,
                             const std::string& ffmpeg, const std::string& ffprobe, bool want_vvdec,
                             const std::function<void()>& on_vvdec_opening,
                             const std::function<void(long long, long long, double)>& on_progress,
                             const CancelToken& cancel);

}  // namespace vml
