// The display times of an AV2 video's pictures, as dav2d gives them.
//
// dav2d shows pictures in display order (its output queue orders them by
// their frame offset), but each keeps the timestamp of the data it was decoded
// from: a picture decoded ahead, as AV1 would decode a hidden frame and show
// it later, comes out after pictures decoded after it, with an earlier stamp.
// Every temporal unit decodes one picture that is shown once, so the k-th
// picture shown is due at the k-th smallest stamp fed: each picture is given
// the smallest stamp not yet given. With the units stamped by their number
// (gpu_frames.cpp's Av2PacketSource, native_decoder.cpp), a picture's stamp is
// then its place in display order.
#pragma once

#include <functional>
#include <optional>
#include <queue>
#include <vector>

class Dav2dDisplayOrder {
public:
    // The stamp of data given to dav2d.
    void fed(long long stamp) { waiting_.push(stamp); }

    // The stamp of the next picture dav2d shows; nothing where it shows more
    // pictures than it was given units.
    std::optional<long long> shown() {
        if (waiting_.empty()) return std::nullopt;
        const long long stamp = waiting_.top();
        waiting_.pop();
        return stamp;
    }

    // Whether every stamp fed has been given: at the end, one picture shown
    // for each unit.
    bool empty() const { return waiting_.empty(); }

    // Forgets every stamp, for a seek.
    void clear() { waiting_ = {}; }

private:
    std::priority_queue<long long, std::vector<long long>, std::greater<long long>> waiting_;
};
