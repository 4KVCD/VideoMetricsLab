// AV2 video files, which FFmpeg 9 cannot read: it names no AV2 codec, marks
// no picture of them as a keyframe, and its OBU reader takes AV1's headers.
// Three forms are read here, as AOM's AV2 encoder avmenc writes them: WebM
// (its default; any Matroska file with a "V_AV2" track), IVF ("DKIF" with the
// fourcc "AV02", .ivf) and a Section 5 stream of OBUs (.obu), each as
// temporal units in decoding order: a unit's bytes, its number and whether
// decoding can start at it.
//
// An AV2 OBU is its size (LEB128, the header included), then its header:
// the extension flag (bit 7), the type (bits 6-2) and the temporal layer
// (bits 1-0) in one byte, and an extension byte where the flag is set. A
// temporal unit starts at a temporal delimiter OBU and shows one picture.
// avmenc's IVF frames and Matroska blocks are not units: one holds several
// (its time is that of the first picture it shows, and the times of the
// others are not said), so the units are split out and numbered, and a
// picture is known by its number, timed by the stream's frame rate
// (av2_probe.h finds it).
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "vml/frame_sync.h"

namespace vml::av2 {

// The OBU types this reader looks at (AV2's, as dav2d's headers.h names them).
enum class ObuType : int {
    sequence_header = 1,
    temporal_delimiter = 2,
    closed_loop_key = 4,     // a keyframe: decoding starts here
    open_loop_key = 5,       // a keyframe whose leading pictures need the ones before it
    leading_tile_group = 6,  // a leading picture of an open-loop keyframe
    leading_sef = 11,
    leading_tip = 13,
};

// What a temporal unit's OBUs say about decoding from it.
struct UnitKind {
    bool closed_key = false;  // holds a closed-loop keyframe
    bool open_key = false;    // holds an open-loop keyframe
    bool leading = false;     // holds a leading picture
};

// The unit's OBUs read as they come, up to the first that does not fit
// (a damaged unit says no more than its whole OBUs say).
UnitKind kind_of(std::span<const std::uint8_t> unit);

// A file that is not AV2 in a form read here, or damaged before its first unit.
class Av2StreamError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class Container { ivf, matroska, section5 };

// Which form `path` holds, by its first bytes: nothing where neither.
std::optional<Container> container_of(const std::filesystem::path& path);

struct TemporalUnit {
    std::vector<std::uint8_t> data;
    // Its place in the stream from 0, in decoding order. Each unit decodes one
    // picture, shown once, but not always in this order: the k-th picture
    // shown is the k-th smallest number (dav2d/display_order.h).
    long long number = 0;
    // Its IVF frame's or Matroska block's time (time_base()), the same for
    // every unit the frame holds; a Section 5 unit's number.
    long long container_pts = 0;
    UnitKind kind;
    // Where its IVF frame or Matroska block starts (its header), and its
    // place among the frame's units; a Section 5 unit's first OBU, and 0:
    // where seek() goes.
    std::int64_t offset = 0;
    int index = 0;
};

// What a reader gives of each unit: its bytes; or, to scan a file for its
// units' numbers, times and keyframes, only its OBUs' headers, each as an
// OBU of that type with no payload, the payloads passed over in the file
// (a few bytes read of each picture's).
enum class Payloads { read, skipped };

// One file's temporal units, one after another.
class Reader {
public:
    // Throws Av2StreamError where the file is not AV2 in any of the forms.
    explicit Reader(const std::filesystem::path& path, Payloads payloads = Payloads::read);

    Container container() const { return container_; }
    // An IVF file's or a Matroska track's picture size and time base, as its
    // headers give them; a Section 5 stream has none (0x0, 1/1: each unit a
    // tick).
    int width() const { return width_; }
    int height() const { return height_; }
    Rational time_base() const { return time_base_; }

    // The next unit, or nothing at the end of the file. A damaged unit ends
    // the file there: nothing comes after it but after a seek.
    std::optional<TemporalUnit> next();

    // Goes back (or on) to `unit`, one this reader or another of the same
    // file gave: next() gives it again, numbered as it was.
    void seek(const TemporalUnit& unit);

private:
    // The next unit of an IVF file or a Matroska file: the frame's next, or
    // the next frame's first.
    std::optional<TemporalUnit> next_framed();
    std::optional<TemporalUnit> next_section5();
    // The next IVF frame or Matroska block into frame_: false at the end, or
    // where it is cut short or damaged.
    bool read_frame() { return container_ == Container::ivf ? read_ivf_frame() : read_matroska_block(); }
    bool read_ivf_frame();
    bool read_matroska_block();
    // The `size` bytes of a frame's OBUs from the file, into `into`, or with
    // Payloads::skipped only their headers: false where the file ends first.
    bool read_payload(std::uint64_t size, std::vector<std::uint8_t>& into);
    // One OBU of a Section 5 stream, its size field included: false at the end.
    bool read_obu(std::vector<std::uint8_t>& obu);

    std::ifstream file_;
    std::uint64_t file_size_ = 0;
    Payloads payloads_ = Payloads::read;
    bool ended_ = false;  // at the end, or at damage, until a seek
    Container container_ = Container::ivf;
    int width_ = 0;
    int height_ = 0;
    Rational time_base_{1, 1};
    long long units_read_ = 0;
    // Matroska: the AV2 track's number, the time of the cluster being read,
    // and after a seek the time of the block sought (from which the
    // cluster's is found again).
    std::uint64_t track_ = 0;
    long long cluster_time_ = 0;
    std::optional<long long> sought_time_;
    // IVF and Matroska: the frame being split into units, its time and where
    // it starts, where its next unit starts and how many it has given.
    std::vector<std::uint8_t> frame_;
    long long frame_pts_ = 0;
    std::int64_t frame_offset_ = 0;
    std::size_t frame_at_ = 0;
    int frame_units_ = 0;
    // Section 5: the OBU that starts the next unit, and where it starts.
    std::vector<std::uint8_t> pending_;
    std::int64_t pending_offset_ = 0;
};

// A file's closed-loop keyframes, in decoding order, as a scan of its OBUs'
// headers finds them (Payloads::skipped): scanned once and kept for the life
// of the process, by the file's path, size and time, so that a file written
// again is scanned again (a 2 GB file of 280,000 pictures took about 2 s).
// Throws Av2StreamError where the file is not AV2.
std::shared_ptr<const std::vector<TemporalUnit>> closed_keyframes(const std::filesystem::path& path);

}  // namespace vml::av2
