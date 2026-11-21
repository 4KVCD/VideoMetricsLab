// The display times of an AV2 video's pictures, as splot gives them.
//
// splot gives pictures in display order. The stamp each carries (that of the
// unit that coded it, or showed it) is not needed: every temporal unit
// decodes one picture that is shown once, so the k-th picture shown is due at
// the k-th smallest stamp fed, and each picture is given the smallest stamp
// not yet given. With the units stamped by their number (gpu_frames.cpp's
// Av2PacketSource, native_decoder.cpp), a picture's stamp is then its place
// in display order.
#pragma once

#include <functional>
#include <optional>
#include <queue>
#include <vector>

class Av2DisplayOrder {
public:
    // The stamp of a unit the decoder took.
    void fed(long long stamp) { waiting_.push(stamp); }

    // The stamp of the next picture the decoder gives; nothing where it gives
    // more pictures than it was given units.
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
