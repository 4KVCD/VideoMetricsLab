#include "vml/host_frames.h"

#include <stdexcept>

#include "vml/app_log.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.host_frames");

}  // namespace

HostFrames::HostFrames(GpuFrameStream& stream, const DecodePlan& plan)
    : stream_(stream), luma_only_(plan.luma_only) {
    const auto [width, height] = plan.output_size();
    packed_ = PackedFrame(width, height, 8 * plan.bytes_per_sample());
    if (stream.gives_planes()) {
        return;
    }
    buffer_.resize(plan.frame_bytes());
    if (stream.backend() == "nvidia") {
        pinned_ = stream.pin(buffer_.data(), buffer_.size());
        if (!pinned_) {
            logger.info("NVIDIA's decoder could not page-lock the memory it downloads the pictures to: the driver "
                        "copies each one again on the CPU");
        }
    }
}

HostFrames::~HostFrames() {
    if (pinned_) {
        stream_.unpin(buffer_.data());
    }
}

FramePlanes HostFrames::take(int slot) {
    if (!downloads()) {
        FramePlanes planes = stream_.slot_planes(slot);
        if (luma_only_) {
            planes[1] = planes[2] = Plane{};
        }
        return planes;
    }
    stream_.download(slot, buffer_.data());
    if (luma_only_) {
        return FramePlanes{packed_.luma(buffer_.data())};
    }
    return packed_.planes(buffer_.data());
}

HostPairs::HostPairs(GpuFrameStream& reference, const DecodePlan& reference_plan, GpuFrameStream& test,
                     const DecodePlan& test_plan)
    : reference_(reference, reference_plan),
      test_(test, test_plan),
      at_once_(reference_.downloads() && test_.downloads() && downloads_at_once(reference.backend(), test.backend())) {}

void HostPairs::score(int reference_slot, int test_slot, PairScorer& gpu, PairScorer* beside, bool test_read) {
    if (!test_read && beside != nullptr) {
        throw std::logic_error("the CPU's metrics read every test picture");
    }
    FramePlanes reference;
    FramePlanes test;
    const auto take_reference = [&] { reference = reference_.take(reference_slot); };
    const auto take_test = [&] {
        if (test_read) {
            test = test_.take(test_slot);
        }
    };
    if (at_once_ && test_read) {
        side_.both(take_reference, take_test);
    } else {
        take_reference();
        take_test();
    }
    const auto to_gpu = [&] { gpu.add_planes(reference, test); };
    if (beside != nullptr) {
        side_.both([&] { beside->add_planes(reference, test); }, to_gpu);
    } else {
        to_gpu();
    }
}

}  // namespace vml
