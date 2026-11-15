// Where to start decoding a raw VVC or HEVC stream (Annex B: NAL units after
// start codes, as VVenC and x265 write .266 and .265 files) to reach a frame.
//
// Such a stream has no timestamps and no index: its frames are numbered by
// counting them from its start. The only number each picture carries is its
// picture order count's low bits, which wrap (every 256 pictures in VVenC's
// streams), so a jump to a keyframe somewhere in the file cannot say which
// frame it is. Counting the pictures needs only their NAL headers, not
// decoding: this map reads the file's headers as far as a seek needs, once
// for the life of the process, and keeps each keyframe's place and number.
//
// A keyframe is an IRAP picture (VVC's IDR and CRA, HEVC's IDR, CRA and BLA),
// at the start of its access unit (its parameter sets and prefix SEI with
// it). Every picture before it in decoding order is shown before it, and so
// are its leading pictures, which follow it in decoding order: its frame
// number is the count of both. A decoder starting at it drops its RASL
// pictures (they need pictures before it) and shows its RADL ones, so the
// first frame it gives is the keyframe's number less its RADL pictures.
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace vml {

enum class RawCodec { vvc, hevc };

struct RawKeyframe {
    std::int64_t offset = 0;        // where its access unit starts in the file
    std::int64_t first_output = 0;  // the number of the first frame a decoder starting there gives
};

class RawKeyframes {
public:
    RawKeyframes(std::filesystem::path path, RawCodec codec);

    // The last keyframe from which decoding gives frame `frame` (its
    // first_output at or before it), reading the file's headers further as
    // needed. Nothing when the file cannot be read or `stop` says to stop
    // (asked between blocks of the file).
    std::optional<RawKeyframe> start_for(std::int64_t frame, const std::function<bool()>& stop);

    // Reads the headers in `bytes`, the file's next bytes, and the
    // keyframes found so far: for the unit test, which feeds hand-made
    // streams in pieces (neither takes the lock start_for takes).
    void scan(const std::uint8_t* bytes, std::size_t size);
    const std::vector<RawKeyframe>& keyframes() const { return keyframes_; }

private:
    // Reads the next block of `file` and its headers: false at its end.
    bool read_block(std::FILE* file);
    // Reads the headers in buffer_: the bytes carried from the block before
    // and `added` new ones after them.
    void scan_buffer(std::size_t added);
    void on_nal(std::int64_t offset, int type, std::uint8_t first_payload_byte);
    void on_picture(std::int64_t offset, int type);
    // Ends the counting of the open keyframe's leading pictures and keeps it.
    void close_keyframe();

    std::filesystem::path path_;
    RawCodec codec_;
    std::mutex mutex_;
    std::int64_t scanned_to_ = 0;  // the file's bytes read so far
    bool file_ended_ = false;
    // The bytes being read: those carried from the block before (a start
    // code and header may straddle two), then a block of the file. Only the
    // carried bytes are kept between reads.
    std::vector<std::uint8_t> buffer_;
    std::size_t carried_ = 0;
    std::int64_t pictures_ = 0;          // pictures seen, in decoding order
    std::optional<std::int64_t> unit_;   // where the access unit being read started, before its picture
    bool picture_header_ = false;        // VVC: a picture header NAL started the picture being read
    // The keyframe whose leading pictures are being counted, and their count.
    std::optional<std::int64_t> open_offset_;
    std::int64_t open_before_ = 0;
    std::int64_t open_rasl_ = 0;
    std::int64_t open_radl_ = 0;
    // Whether its RASL pictures are dropped, never shown: the first keyframe's,
    // an HEVC BLA's, and a keyframe's right after an end of sequence.
    bool open_drops_rasl_ = false;
    bool after_end_of_sequence_ = false;  // an end of sequence NAL since the last keyframe
    bool seen_keyframe_ = false;
    // False where the stream starts with pictures a decoder drops (no
    // keyframe first): its frames cannot be numbered from the headers.
    bool usable_ = true;
    std::vector<RawKeyframe> keyframes_;  // completed: their leading pictures counted
};

// The map of the raw stream at `path`, shared for the life of the process
// by the file's path, size and time (a file written again is read again).
std::shared_ptr<RawKeyframes> raw_keyframes(const std::filesystem::path& path, RawCodec codec);

}  // namespace vml
