// The app's decoders' pictures for scorers that read them in system memory
// (PairScorer::add_planes): where the software decoder holds them, or
// downloaded into a buffer. The VMAF passes whose pictures do not stay on the
// GPU feed their scorers this way (score_decoded_v1 in vmaf_v1_gpu.cpp,
// score_decoded in gpu_vmaf.cpp).
#pragma once

#include <cstdint>
#include <vector>

#include "vml/gpu_frames.h"
#include "vml/pair_scorer.h"
#include "vml/threads.h"

namespace vml {

// One video's pictures: read where the software decoder holds them
// (GpuFrameStream::gives_planes), or downloaded into a buffer of this one's,
// which for NVIDIA's decoder is page-locked while this lives
// (GpuFrameStream::pin): the GPU then copies a picture into it by itself,
// where into pageable memory the driver copies it again on the CPU (a 4K
// 10-bit frame: 5.3 ms). Made after its stream, so that it goes first.
class HostFrames {
public:
    // `plan` is the stream's.
    HostFrames(GpuFrameStream& stream, const DecodePlan& plan);
    ~HostFrames();
    HostFrames(const HostFrames&) = delete;
    HostFrames& operator=(const HostFrames&) = delete;

    // Whether the pictures are downloaded, rather than read where they are.
    bool downloads() const { return !buffer_.empty(); }

    // The planes of the slot's picture (its luma alone, for a plan of the luma
    // alone), downloaded first where it is not read in place. They are valid
    // until the slot is given back, or the next take.
    FramePlanes take(int slot);

private:
    GpuFrameStream& stream_;
    PackedFrame packed_;
    bool luma_only_ = false;
    std::vector<std::uint8_t> buffer_;
    bool pinned_ = false;
};

// Both videos' pictures, pair by pair, for a GPU scorer and the CPU's metrics
// beside it: the two downloads at the same time where both are downloaded and
// the decoders allow it (downloads_at_once), the source's on a side thread;
// then the two scorers read the planes at the same time, the CPU's on the
// side thread.
class HostPairs {
public:
    // The plans are the streams'.
    HostPairs(GpuFrameStream& reference, const DecodePlan& reference_plan, GpuFrameStream& test,
              const DecodePlan& test_plan);

    // Scores the pair in the slots with `gpu`, and with `beside` too where it
    // is given. Without `test_read` (a frame `gpu` does not score, with
    // nothing beside), the test video's picture is not taken, and `gpu` is
    // given no planes for it.
    void score(int reference_slot, int test_slot, PairScorer& gpu, PairScorer* beside, bool test_read = true);

private:
    HostFrames reference_;
    HostFrames test_;
    bool at_once_ = false;
    SideThread side_;  // last, so that it goes first: its jobs use the frames
};

}  // namespace vml
