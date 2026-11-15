#include "vml/raw_keyframes.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <utility>

#include "vml/platform/system.h"

namespace vml {

namespace {

// Read a block at a time: large enough for the disk, small enough that a
// seek asked for meanwhile stops the reading soon (4 MB is ~20 ms from a USB
// disk).
constexpr std::size_t block_size = 4 << 20;

// A start code (00 00 01), then the NAL header (two bytes in VVC and HEVC)
// and the first byte of the payload: what is read of each NAL unit.
constexpr std::size_t nal_bytes_read = 6;

enum class Kind { keyframe, rasl, radl, other };

Kind vvc_kind(int type) {
    // VVC's NAL unit types (H.266 table 5): RADL 2, RASL 3, IDR_W_RADL 7,
    // IDR_N_LP 8, CRA 9.
    if (type == 7 || type == 8 || type == 9) {
        return Kind::keyframe;
    }
    return type == 3 ? Kind::rasl : type == 2 ? Kind::radl : Kind::other;
}

Kind hevc_kind(int type) {
    // HEVC's (H.265 table 7-1): RADL 6-7, RASL 8-9, IRAP 16-23 (BLA, IDR, CRA
    // and two reserved IRAP types).
    if (type >= 16 && type <= 23) {
        return Kind::keyframe;
    }
    return type == 8 || type == 9 ? Kind::rasl : type == 6 || type == 7 ? Kind::radl : Kind::other;
}

}  // namespace

RawKeyframes::RawKeyframes(std::filesystem::path path, RawCodec codec)
    : path_(std::move(path)), codec_(codec) {}

void RawKeyframes::scan(const std::uint8_t* bytes, std::size_t size) {
    buffer_.resize(nal_bytes_read + block_size);
    while (size > 0) {
        const std::size_t piece = std::min(size, block_size);
        std::memcpy(buffer_.data() + carried_, bytes, piece);
        scan_buffer(piece);
        bytes += piece;
        size -= piece;
    }
    buffer_.resize(carried_);
}

bool RawKeyframes::read_block(std::FILE* file) {
    const std::size_t got = std::fread(buffer_.data() + carried_, 1, block_size, file);
    if (got == 0) {
        return false;
    }
    scan_buffer(got);
    return true;
}

void RawKeyframes::scan_buffer(std::size_t added) {
    const std::uint8_t* bytes = buffer_.data();
    const std::size_t size = carried_ + added;
    const std::int64_t base = scanned_to_ - static_cast<std::int64_t>(carried_);
    // Annex B's start code search, a byte in three looked at where none is
    // near: a third byte above 1 rules out a start code at any of the three
    // places that would need it to be 00 or 01; a nonzero second byte, at the
    // first two.
    std::size_t at = 0;
    while (at + nal_bytes_read <= size) {
        if (bytes[at + 2] > 1) {
            at += 3;
        } else if (bytes[at + 1] != 0) {
            at += 2;
        } else if (bytes[at] != 0 || bytes[at + 2] != 1) {
            at += 1;
        } else {
            // VVC's header: forbidden zero, reserved, layer (6 bits), then
            // type (5 bits) and temporal id (3); HEVC's: forbidden zero, type
            // (6 bits), layer and temporal id.
            const int type = codec_ == RawCodec::vvc ? bytes[at + 4] >> 3 : (bytes[at + 3] >> 1) & 0x3F;
            on_nal(base + static_cast<std::int64_t>(at), type, bytes[at + 5]);
            at += 3;
        }
    }
    scanned_to_ += static_cast<std::int64_t>(added);
    // Where the search stopped, a start code may straddle this block and the
    // next: the bytes from there go first in the next.
    carried_ = size - at;
    std::memmove(buffer_.data(), bytes + at, carried_);
}

void RawKeyframes::on_nal(std::int64_t offset, int type, std::uint8_t first_payload_byte) {
    if (codec_ == RawCodec::vvc) {
        if (type == 19) {  // a picture header: a picture starts, its type comes with its first slice
            unit_ = unit_.value_or(offset);
            picture_header_ = true;
            return;
        }
        if (type <= 11) {  // a slice: a picture starts where it has the picture header in it
            if (picture_header_ || (first_payload_byte & 0x80) != 0) {
                on_picture(unit_.value_or(offset), type);
            }
            return;
        }
        if (type == 21 || type == 22) {  // end of sequence, end of bitstream
            after_end_of_sequence_ = true;
            return;
        }
        // What comes before a picture in its access unit: OPI, DCI, VPS,
        // SPS, PPS, prefix APS, the delimiter and prefix SEI.
        if ((type >= 12 && type <= 17) || type == 20 || type == 23) {
            unit_ = unit_.value_or(offset);
        }
        return;
    }
    if (type <= 31) {  // a slice segment: the first of its picture starts it
        if ((first_payload_byte & 0x80) != 0) {
            on_picture(unit_.value_or(offset), type);
        }
        return;
    }
    if (type == 36 || type == 37) {  // end of sequence, end of bitstream
        after_end_of_sequence_ = true;
        return;
    }
    // VPS, SPS, PPS, the delimiter, prefix SEI and the types that may start
    // an access unit (H.265 7.4.2.4.4).
    if ((type >= 32 && type <= 35) || type == 39 || (type >= 41 && type <= 44) || (type >= 48 && type <= 55)) {
        unit_ = unit_.value_or(offset);
    }
}

void RawKeyframes::on_picture(std::int64_t offset, int type) {
    unit_.reset();
    picture_header_ = false;
    const Kind kind = codec_ == RawCodec::vvc ? vvc_kind(type) : hevc_kind(type);
    if (kind == Kind::keyframe) {
        close_keyframe();
        if (!seen_keyframe_ && pictures_ > 0) {
            usable_ = false;  // pictures before the first keyframe, which decoders drop: not counted as they show
        }
        // Where a decoder drops the RASL pictures even from the start (the
        // standards' NoRaslOutputFlag): the pictures they refer to before
        // the keyframe are not there to decode them.
        const bool hevc_bla = codec_ == RawCodec::hevc && type >= 16 && type <= 18;
        open_drops_rasl_ = !seen_keyframe_ || after_end_of_sequence_ || hevc_bla;
        after_end_of_sequence_ = false;
        seen_keyframe_ = true;
        open_offset_ = offset;
        open_before_ = pictures_;
        open_rasl_ = 0;
        open_radl_ = 0;
    } else if (open_offset_ && kind == Kind::rasl) {
        if (open_drops_rasl_) {
            return;  // not shown, decoded from the start either
        }
        ++open_rasl_;
    } else if (open_offset_ && kind == Kind::radl) {
        ++open_radl_;
    } else {
        close_keyframe();  // a trailing picture: its keyframe's leading pictures are all counted
    }
    ++pictures_;
}

void RawKeyframes::close_keyframe() {
    if (!open_offset_) {
        return;
    }
    // Started at it, a decoder gives the RADL pictures and the keyframe, the
    // RASL ones it drops shown before both: with both kinds the frames given
    // would have gaps, so it is not a place to start (no encoder known here
    // writes both).
    if (open_rasl_ == 0 || open_radl_ == 0) {
        keyframes_.push_back({*open_offset_, open_before_ + open_rasl_});
    }
    open_offset_.reset();
}

std::optional<RawKeyframe> RawKeyframes::start_for(std::int64_t frame, const std::function<bool()>& stop) {
    const std::lock_guard lock(mutex_);
    const auto known = [&] { return !keyframes_.empty() && keyframes_.back().first_output > frame; };
    if (usable_ && !file_ended_ && !known()) {
        // Opened for this reading only, where the last stopped: an open file
        // could not be deleted or written again on Windows. Unbuffered, the
        // blocks go straight into buffer_ (an ifstream's buffering made the
        // reading three times slower).
        const std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(platform::open_for_reading(path_, scanned_to_),
                                                                   &std::fclose);
        if (!file) {
            return std::nullopt;
        }
        std::setvbuf(file.get(), nullptr, _IONBF, 0);
        buffer_.resize(nal_bytes_read + block_size);
        bool stopped = false;
        while (usable_ && !known()) {
            if (stop()) {
                stopped = true;
                break;
            }
            if (!read_block(file.get())) {
                file_ended_ = true;
                close_keyframe();
                break;
            }
        }
        buffer_.resize(carried_);
        buffer_.shrink_to_fit();
        if (stopped) {
            return std::nullopt;
        }
    }
    if (!usable_) {
        return std::nullopt;
    }
    const auto after = std::upper_bound(keyframes_.begin(), keyframes_.end(), frame,
                                        [](std::int64_t number, const RawKeyframe& key) { return number < key.first_output; });
    if (after == keyframes_.begin()) {
        return std::nullopt;
    }
    return *std::prev(after);
}

std::shared_ptr<RawKeyframes> raw_keyframes(const std::filesystem::path& path, RawCodec codec) {
    std::error_code ignored;
    const std::string key = platform::path_text(path) + '|' + std::to_string(std::filesystem::file_size(path, ignored)) +
                            '|' + std::to_string(platform::modified_ns(path).value_or(0)) + '|' +
                            std::to_string(static_cast<int>(codec));
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<RawKeyframes>> maps;
    const std::lock_guard lock(mutex);
    std::shared_ptr<RawKeyframes>& map = maps[key];
    if (!map) {
        map = std::make_shared<RawKeyframes>(path, codec);
    }
    return map;
}

}  // namespace vml
