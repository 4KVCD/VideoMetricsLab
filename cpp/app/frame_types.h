// Each frame's type (I, P or B), for the Bitrate tab's frame graph. Read by
// FFmpeg's parsers from the video's packets, without decoding: ffprobe's
// frame=pict_type decodes, which took 27.6 s for a 60 s 4K clip, where reading
// its packets took 0.15 s.
#pragma once

#include <atomic>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include "vml/bitrate.h"
#include "vml/cancel.h"
#include "vml/video.h"

namespace vml::app {

enum FrameType : std::uint8_t { frame_type_unknown, frame_type_i, frame_type_p, frame_type_b };

// A packet as it was read.
struct ReadFrame {
    double time = 0;
    long long position = 0;
    long long size = 0;
    std::uint8_t type = frame_type_unknown;
};

// The packets' types, one for each of `sizes` (BitrateData's, in presentation
// order). Put in that order as PacketList::finish (bitrate.cpp) puts ffprobe's
// packets: by time, then position, or left as they were read with
// `keep_read_order` (a raw stream's, whose times are only counted). Empty unless
// every packet then has the frame's size, so that no type goes with a wrong frame.
std::vector<std::uint8_t> types_matching_frames(std::vector<ReadFrame> frames, bool keep_read_order,
                                                std::span<const long long> sizes);

// Reads a video's packets on a thread of its own, from when it is made. Made
// before ffprobe reads the same file, so that they share what the system has
// cached: read after ffprobe, a 64 GB remux takes as long again.
class FrameTypeReader {
public:
    // `cancel` must outlive this.
    FrameTypeReader(VideoInfo info, const CancelToken& cancel);
    // Stops the reading.
    ~FrameTypeReader();
    FrameTypeReader(const FrameTypeReader&) = delete;
    FrameTypeReader& operator=(const FrameTypeReader&) = delete;

    // Waits for the reading to end. A type for each frame of `data`; none where
    // they cannot be known: without FFmpeg's libraries, for a format they do not
    // read (AV2), once cancelled, or if the packets do not match `data`
    // (types_matching_frames). Once only.
    std::vector<std::uint8_t> types_of(const BitrateData& data);

    // Whether the reading is to end now (FFmpeg asks, from the thread).
    bool stopping() const { return stop_ || cancel_.cancelled(); }

private:
    VideoInfo info_;
    const CancelToken& cancel_;
    std::atomic<bool> stop_ = false;
    std::vector<ReadFrame> frames_;  // the thread's, until it has ended
    std::thread thread_;
};

}  // namespace vml::app
