#include "vml/av2_stream.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <string>

#include "vml/append.h"
#include "vml/platform/system.h"

#include "matroska_ebml.h"

namespace vml::av2 {

namespace {

// IVF: a 32-byte header ("DKIF", version, header size, fourcc, width,
// height, the time base's denominator then numerator, frame count, unused),
// then each frame: its size (4 bytes), its time (8), its bytes; little-endian.
constexpr std::size_t ivf_header_bytes = 32;
constexpr std::size_t ivf_frame_header_bytes = 12;
// Larger than any picture's units: a size field past it is damage, not data.
constexpr std::uint64_t largest_unit = 1ULL << 30;

std::uint32_t le32(const std::uint8_t* bytes) {
    return bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

// A LEB128 number at `bytes` (at most 8 bytes of it, as AV2 allows): its
// value and its length, or nothing where it runs past `size` or is too long.
std::optional<std::pair<std::uint64_t, std::size_t>> leb128(const std::uint8_t* bytes, std::size_t size) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < size && i < 8; i++) {
        value |= static_cast<std::uint64_t>(bytes[i] & 0x7f) << (7 * i);
        if (!(bytes[i] & 0x80)) {
            return std::pair{value, i + 1};
        }
    }
    return std::nullopt;
}

ObuType type_of(std::uint8_t header) { return static_cast<ObuType>((header >> 2) & 0x1f); }

using namespace vml::matroska;  // the EBML elements and their reading (matroska_ebml.h)

// The headers of a Matroska file's AV2 track, as read before its first
// cluster: the track's number, picture size, the file's ticks and where its
// clusters start.
struct MatroskaTrack {
    std::uint64_t number = 0;
    int width = 0;
    int height = 0;
    long long timecode_scale = mkv::default_timecode_scale;
    std::int64_t first_cluster = 0;
};

// A track entry's number and picture size, where its codec is AV2's and its
// frames are stored as they are (no ContentEncodings: header stripping or
// compression would give splot other bytes).
std::optional<MatroskaTrack> av2_track_entry(std::istream& in, std::int64_t end) {
    MatroskaTrack track;
    std::string codec;
    bool encoded = false;
    const bool read = each_child(in, end, [&](const Element& element) {
        if (element.id == mkv::content_encodings) {
            encoded = true;
        } else if (element.id == mkv::track_number) {
            track.number = read_unsigned(in, element.size).value_or(0);
        } else if (element.id == mkv::codec_id && element.size <= 64) {
            codec.resize(element.size);
            in.read(codec.data(), static_cast<std::streamsize>(element.size));
        } else if (element.id == mkv::video) {
            return each_child(in, end_of(in, element), [&](const Element& field) {
                if (field.id == mkv::pixel_width || field.id == mkv::pixel_height) {
                    const int value = static_cast<int>(read_unsigned(in, field.size).value_or(0));
                    (field.id == mkv::pixel_width ? track.width : track.height) = value;
                }
                return true;
            });
        }
        return true;
    });
    // A string element may be padded with zero bytes.
    codec.erase(std::find(codec.begin(), codec.end(), '\0'), codec.end());
    return read && !encoded && codec == mkv::av2_codec && track.number > 0 ? std::optional(track) : std::nullopt;
}

// The first AV2 track of the Matroska file `in` holds (read from where it
// is, the file's start), or nothing where it has none or its headers are
// damaged. The segment's info and tracks come before its first cluster, as
// every muxer writes them (a file with its tracks after, found only through
// its SeekHead, is not read here).
std::optional<MatroskaTrack> matroska_av2_track(std::istream& in) {
    const std::optional<Element> header = read_element(in);
    if (!header || header->id != mkv::ebml || header->size == mkv::unknown_size) {
        return std::nullopt;
    }
    in.seekg(end_of(in, *header));
    const std::optional<Element> segment = read_element(in);
    if (!segment || segment->id != mkv::segment) {
        return std::nullopt;
    }
    std::optional<MatroskaTrack> found;
    long long scale = mkv::default_timecode_scale;
    while (true) {
        const std::int64_t start = position(in);
        const std::optional<Element> element = read_element(in);
        if (!element) {
            return std::nullopt;
        }
        if (element->id == mkv::cluster) {
            if (found) {
                found->timecode_scale = scale;
                found->first_cluster = start;
            }
            return found;
        }
        if (element->size == mkv::unknown_size) {
            return std::nullopt;
        }
        const std::int64_t end = end_of(in, *element);
        bool read = true;
        if (element->id == mkv::info) {
            read = each_child(in, end, [&](const Element& field) {
                if (field.id == mkv::timecode_scale) {
                    scale = static_cast<long long>(read_unsigned(in, field.size).value_or(0));
                }
                return scale > 0;
            });
        } else if (element->id == mkv::tracks) {
            read = each_child(in, end, [&](const Element& entry) {
                if (entry.id == mkv::track_entry && !found) {
                    found = av2_track_entry(in, end_of(in, entry));
                }
                return true;
            });
            if (!found) {
                return std::nullopt;  // tracks, none AV2's: a file FFmpeg reads
            }
        }
        in.clear();
        in.seekg(end);
        if (!read || !in) {
            return std::nullopt;
        }
    }
}

// Calls `visit(begin, header, end)` for each whole OBU of `unit`, in order:
// where it starts (its size field), where its header is and where it ends.
// The first that does not fit ends the walk (a damaged unit says no more
// than its whole OBUs say).
template <typename Visit>
void each_obu(std::span<const std::uint8_t> unit, Visit&& visit) {
    std::size_t at = 0;
    while (at < unit.size()) {
        const auto size = leb128(unit.data() + at, unit.size() - at);
        if (!size || size->first == 0 || size->first > unit.size() - at - size->second) {
            return;
        }
        const std::size_t header = at + size->second;
        visit(at, header, header + size->first);
        at = header + size->first;
    }
}

// Whether an OBU of this type sends a setting a decoder keeps (TemporalUnit::carried).
bool is_kept_setting(ObuType type) {
    return type == ObuType::film_grain || type == ObuType::quantization_matrix || type == ObuType::multi_frame_header;
}

// AV2's bits, most significant first, as its syntax reads them: f(n) and uvlc().
class BitReader {
public:
    explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    // The next `count` bits (at most 32), or nothing past the end.
    std::optional<std::uint32_t> bits(int count) {
        std::uint32_t value = 0;
        for (int i = 0; i < count; i++, at_++) {
            if (at_ / 8 >= bytes_.size()) {
                return std::nullopt;
            }
            value = value << 1 | ((bytes_[at_ / 8] >> (7 - at_ % 8)) & 1);
        }
        return value;
    }

    // uvlc(): n zeros, a one, then n bits.
    std::optional<std::uint32_t> uvlc() {
        int zeros = 0;
        while (true) {
            const std::optional<std::uint32_t> bit = bits(1);
            if (!bit || zeros == 32) {
                return std::nullopt;
            }
            if (*bit) {
                break;
            }
            zeros++;
        }
        const std::optional<std::uint32_t> rest = bits(zeros);
        return rest ? std::optional(static_cast<std::uint32_t>((1ULL << zeros) - 1 + *rest)) : std::nullopt;
    }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t at_ = 0;
};

// The slots a kept setting's OBU sets, a bit each, as its payload's first
// fields say (as AOM's read_fgm_obu, read_qm_obu and
// av2_read_multi_frame_header read them): a film grain OBU's 8
// (fgm_bit_map), a quantization matrix OBU's 15 (qm_bit_map; 0 sets them all,
// to the predefined matrices), a multi-frame header's id (1 to 15). 0 where
// the payload is too short.
std::uint32_t slots_of(ObuType type, std::span<const std::uint8_t> payload) {
    BitReader read(payload);
    switch (type) {
        case ObuType::film_grain:
            return read.bits(8).value_or(0);
        case ObuType::quantization_matrix: {
            constexpr std::uint32_t every_slot = (1U << 15) - 1;
            const std::optional<std::uint32_t> map = read.bits(15);
            return !map ? 0 : *map == 0 ? every_slot : *map;
        }
        case ObuType::multi_frame_header: {
            const std::optional<std::uint32_t> sequence_header_id = read.uvlc();
            const std::optional<std::uint32_t> id_minus_1 = read.uvlc();
            return sequence_header_id && id_minus_1 && *id_minus_1 < 15 ? 1U << (*id_minus_1 + 1) : 0;
        }
        default:
            return 0;
    }
}

// Calls `visit(type, slots, obu)` for each kept setting `unit` sends, in
// order: its type, the slots it sets and the whole OBU.
template <typename Visit>
void each_kept_setting(std::span<const std::uint8_t> unit, Visit&& visit) {
    each_obu(unit, [&](std::size_t begin, std::size_t header, std::size_t end) {
        const ObuType type = type_of(unit[header]);
        // The payload follows the header and, where its first bit says so, its extension byte.
        const std::size_t payload = header + 1 + (unit[header] >> 7);
        if (!is_kept_setting(type) || payload > end) {
            return;
        }
        if (const std::uint32_t slots = slots_of(type, unit.subspan(payload, end - payload)); slots != 0) {
            visit(type, slots, unit.subspan(begin, end - begin));
        }
    });
}

// The kept settings a decoder has been sent, as a file is scanned in
// decoding order: what each closed-loop keyframe carries (TemporalUnit::carried).
class KeptSettings {
public:
    // What a decoder started at the closed-loop keyframe `unit` needs first,
    // from those sent before it: each setting whose slots are all still its
    // own, and that `unit` does not send again. Nothing where none.
    std::shared_ptr<const std::vector<std::uint8_t>> carried_into(std::span<const std::uint8_t> unit) {
        std::vector<std::pair<ObuType, std::uint32_t>> sent_again;
        each_kept_setting(unit, [&](ObuType type, std::uint32_t slots, std::span<const std::uint8_t>) {
            sent_again.emplace_back(type, slots);
        });
        std::vector<std::uint8_t> carried;
        for (const Setting& setting : settings_) {
            const bool overlaps = std::ranges::any_of(sent_again, [&](const auto& again) {
                return again.first == setting.type && (again.second & setting.slots) != 0;
            });
            if (setting.live == setting.slots && !overlaps) {
                carried.insert(carried.end(), setting.obu.begin(), setting.obu.end());
            }
        }
        if (carried.empty()) {
            return nullptr;
        }
        if (!last_ || *last_ != carried) {
            last_ = std::make_shared<const std::vector<std::uint8_t>>(std::move(carried));
        }
        return last_;
    }

    // The settings `unit` sends, in order, each slot's last.
    void add(std::span<const std::uint8_t> unit) {
        each_kept_setting(unit, [&](ObuType type, std::uint32_t slots, std::span<const std::uint8_t> obu) {
            for (Setting& setting : settings_) {
                if (setting.type == type) {
                    setting.live &= ~slots;
                }
            }
            std::erase_if(settings_, [](const Setting& setting) { return setting.live == 0; });
            settings_.push_back({type, slots, slots, {obu.begin(), obu.end()}});
        });
    }

private:
    struct Setting {
        ObuType type;
        std::uint32_t slots;  // the slots it set
        std::uint32_t live;   // those no later one has set (never 0: one is dropped then)
        std::vector<std::uint8_t> obu;
    };
    // In the order sent, at most one a slot (8 + 15 + 15).
    std::vector<Setting> settings_;
    // The last keyframe's, shared by those after it that carry the same.
    std::shared_ptr<const std::vector<std::uint8_t>> last_;
};

// Puts `settings` (whole OBUs) into `unit` after its sequence header, or
// after its temporal delimiter where it has none.
void put_in(std::vector<std::uint8_t>& unit, const std::vector<std::uint8_t>& settings) {
    std::size_t at = 0;
    each_obu(unit, [&](std::size_t, std::size_t header, std::size_t end) {
        const ObuType type = type_of(unit[header]);
        if (type == ObuType::sequence_header || (type == ObuType::temporal_delimiter && at == 0)) {
            at = end;
        }
    });
    unit.insert(unit.begin() + static_cast<std::ptrdiff_t>(at), settings.begin(), settings.end());
}

}  // namespace

UnitKind kind_of(std::span<const std::uint8_t> unit) {
    UnitKind kind;
    each_obu(unit, [&](std::size_t, std::size_t header, std::size_t) {
        switch (type_of(unit[header])) {
            case ObuType::closed_loop_key:
                kind.closed_key = true;
                break;
            case ObuType::open_loop_key:
                kind.open_key = true;
                break;
            case ObuType::leading_tile_group:
            case ObuType::leading_sef:
            case ObuType::leading_tip:
                kind.leading = true;
                break;
            default:
                break;
        }
    });
    return kind;
}

std::optional<Container> container_of(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::array<std::uint8_t, 16> start{};
    file.read(reinterpret_cast<char*>(start.data()), start.size());
    const std::size_t got = static_cast<std::size_t>(file.gcount());
    if (got >= 12 && std::memcmp(start.data(), "DKIF", 4) == 0) {
        return std::memcmp(start.data() + 8, "AV02", 4) == 0 ? std::optional(Container::ivf) : std::nullopt;
    }
    if (got >= 4 && std::memcmp(start.data(), "\x1A\x45\xDF\xA3", 4) == 0) {
        // Matroska or WebM: AV2 where a track is (FFmpeg reads the others).
        file.clear();
        file.seekg(0);
        return matroska_av2_track(file) ? std::optional(Container::matroska) : std::nullopt;
    }
    // A Section 5 stream starts with its first unit's temporal delimiter, an
    // OBU of its header alone (and its extension byte, where its flag is
    // set), then the sequence header: a raw YUV file can start with any one
    // byte of that, hardly with all of it.
    const auto delimiter = leb128(start.data(), got);
    if (!delimiter || delimiter->second >= got) {
        return std::nullopt;
    }
    const std::uint8_t header = start[delimiter->second];
    const std::size_t header_bytes = (header & 0x80) ? 2 : 1;
    const std::size_t next = delimiter->second + delimiter->first;
    if (type_of(header) != ObuType::temporal_delimiter || delimiter->first != header_bytes || next >= got) {
        return std::nullopt;
    }
    const auto sequence = leb128(start.data() + next, got - next);
    if (sequence && sequence->first > 0 && next + sequence->second < got &&
        type_of(start[next + sequence->second]) == ObuType::sequence_header) {
        return Container::section5;
    }
    return std::nullopt;
}

Reader::Reader(const std::filesystem::path& path, Payloads payloads)
    : file_(path, std::ios::binary), payloads_(payloads) {
    const std::optional<Container> container = container_of(path);
    std::error_code error;
    file_size_ = std::filesystem::file_size(path, error);
    if (!file_ || !container || error) {
        throw Av2StreamError("not an AV2 IVF, Matroska or OBU stream");
    }
    container_ = *container;
    if (container_ == Container::ivf) {
        std::array<std::uint8_t, ivf_header_bytes> header{};
        file_.read(reinterpret_cast<char*>(header.data()), header.size());
        const std::uint32_t header_size = header[6] | (header[7] << 8);
        if (file_.gcount() != static_cast<std::streamsize>(header.size()) || header_size < ivf_header_bytes) {
            throw Av2StreamError("its IVF header is cut short");
        }
        width_ = header[12] | (header[13] << 8);
        height_ = header[14] | (header[15] << 8);
        const std::uint32_t denominator = le32(header.data() + 16);
        const std::uint32_t numerator = le32(header.data() + 20);
        if (denominator > 0 && numerator > 0) {
            time_base_ = {numerator, denominator};
        }
        file_.seekg(header_size);
    } else if (container_ == Container::matroska) {
        const std::optional<MatroskaTrack> track = matroska_av2_track(file_);
        if (!track) {
            throw Av2StreamError("its Matroska headers could not be read");
        }
        track_ = track->number;
        width_ = track->width;
        height_ = track->height;
        // Ticks of so many nanoseconds.
        const long long divisor = std::gcd(track->timecode_scale, 1'000'000'000LL);
        time_base_ = {track->timecode_scale / divisor, 1'000'000'000LL / divisor};
        file_.clear();
        file_.seekg(track->first_cluster);
    }
}

std::optional<TemporalUnit> Reader::next() {
    if (ended_) {
        return std::nullopt;
    }
    std::optional<TemporalUnit> unit = container_ == Container::section5 ? next_section5() : next_framed();
    if (!unit) {
        ended_ = true;
    } else if (carried_) {
        put_in(unit->data, *carried_);
    }
    carried_.reset();
    return unit;
}

namespace {

// Where the temporal unit starting at `from` in `bytes` ends: at the next
// temporal delimiter (the one it starts with passed over), or at the end.
std::size_t unit_end(const std::vector<std::uint8_t>& bytes, std::size_t from) {
    std::size_t at = from;
    while (at < bytes.size()) {
        const auto size = leb128(bytes.data() + at, bytes.size() - at);
        if (!size || size->first == 0 || size->first > bytes.size() - at - size->second) {
            return bytes.size();  // damaged: the rest is one unit
        }
        if (at > from && type_of(bytes[at + size->second]) == ObuType::temporal_delimiter) {
            return at;
        }
        at += size->second + size->first;
    }
    return bytes.size();
}

}  // namespace

bool Reader::read_ivf_frame() {
    frame_offset_ = static_cast<std::int64_t>(file_.tellg());
    std::array<std::uint8_t, ivf_frame_header_bytes> header{};
    file_.read(reinterpret_cast<char*>(header.data()), header.size());
    if (file_.gcount() != static_cast<std::streamsize>(header.size())) {
        return false;
    }
    const std::uint32_t size = le32(header.data());
    if (size == 0 || size > largest_unit) {
        return false;
    }
    frame_pts_ = static_cast<long long>(le32(header.data() + 4)) | (static_cast<long long>(le32(header.data() + 8)) << 32);
    frame_at_ = 0;
    frame_units_ = 0;
    // Cut short: the file ends at the frame before.
    return read_payload(size, frame_);
}

bool Reader::read_payload(std::uint64_t size, std::vector<std::uint8_t>& into) {
    const std::int64_t end = position(file_) + static_cast<std::int64_t>(size);
    if (!file_ || end > static_cast<std::int64_t>(file_size_)) {
        return false;
    }
    if (payloads_ == Payloads::read) {
        into.resize(size);
        file_.read(reinterpret_cast<char*>(into.data()), static_cast<std::streamsize>(size));
        return file_.gcount() == static_cast<std::streamsize>(size);
    }
    // Each OBU's size field and header: the header kept as a payload-less
    // OBU (size 1), the rest passed over; a kept setting kept whole. A
    // damaged OBU ends the frame's.
    into.clear();
    while (position(file_) < end) {
        std::array<std::uint8_t, 8> field{};
        std::size_t length = 0;
        while (length < field.size()) {
            const int byte = file_.get();
            if (byte == std::char_traits<char>::eof()) {
                return false;
            }
            field[length++] = static_cast<std::uint8_t>(byte);
            if (!(byte & 0x80)) {
                break;
            }
        }
        const auto obu_size = leb128(field.data(), length);
        const std::int64_t payload = position(file_);
        if (!obu_size || obu_size->first == 0 || payload + static_cast<std::int64_t>(obu_size->first) > end) {
            break;
        }
        const auto header = static_cast<std::uint8_t>(file_.get());
        if (is_kept_setting(type_of(header))) {
            const std::size_t begin = into.size();
            into.insert(into.end(), field.begin(), field.begin() + static_cast<std::ptrdiff_t>(length));
            into.push_back(header);
            const std::size_t rest = into.size();
            const auto payload_size = static_cast<std::streamsize>(obu_size->first - 1);
            into.resize(rest + obu_size->first - 1);
            file_.read(reinterpret_cast<char*>(into.data() + rest), payload_size);
            if (file_.gcount() != payload_size) {
                into.resize(begin);  // cut short by the file's end: no setting from it
                return false;
            }
        } else {
            into.push_back(1);
            into.push_back(header);
        }
        file_.seekg(payload + static_cast<std::int64_t>(obu_size->first));
    }
    file_.clear();
    file_.seekg(end);
    return static_cast<bool>(file_);
}

// A Matroska file's elements are read as they come, not nested: a cluster
// or a block group is entered (its size not needed: a live stream's cluster
// does not give it), a cluster's time kept, and every other element but a
// block of the AV2 track passed over by its size, so that a seek can start
// at any block. A laced block (several frames in one), which avmenc does not
// write, or a damaged element ends the file there.
bool Reader::read_matroska_block() {
    while (true) {
        const std::int64_t start = position(file_);
        const std::optional<Element> element = read_element(file_);
        if (!element || element->id == mkv::segment || element->id == mkv::ebml) {
            return false;  // the end, or another segment
        }
        if (element->id == mkv::cluster || element->id == mkv::block_group) {
            continue;
        }
        if (element->size == mkv::unknown_size) {
            return false;
        }
        if (element->id == mkv::timecode) {
            const std::optional<std::uint64_t> time = read_unsigned(file_, element->size);
            if (!time) {
                return false;
            }
            cluster_time_ = static_cast<long long>(*time);
            continue;
        }
        if (element->id != mkv::simple_block && element->id != mkv::block) {
            file_.seekg(end_of(file_, *element));
            if (!file_) {
                return false;
            }
            continue;
        }
        const std::int64_t end = end_of(file_, *element);
        if (element->size > largest_unit || end > static_cast<std::int64_t>(file_size_)) {
            return false;  // damaged, or cut short
        }
        // A block: its track's number (coded as a size is), its time from the
        // cluster's (16 bits, signed), its flags (bits 2-1: lacing), its frame.
        const std::optional<std::uint64_t> track = ebml_number(
            [this] {
                const int byte = file_.get();
                return byte == std::char_traits<char>::eof() ? -1 : byte;
            },
            false, 8);
        std::array<std::uint8_t, 3> fields{};
        file_.read(reinterpret_cast<char*>(fields.data()), fields.size());
        if (!track || !file_ || position(file_) > end) {
            return false;
        }
        if (*track != track_) {
            file_.seekg(end);
            continue;  // another track's, such as a soundtrack's
        }
        const long long relative = static_cast<std::int16_t>((fields[0] << 8) | fields[1]);
        if ((fields[2] & 0x06) != 0) {
            return false;
        }
        if (sought_time_) {
            // Sought into the cluster: its time is found again from the block's.
            cluster_time_ = *sought_time_ - relative;
            sought_time_.reset();
        }
        if (!read_payload(static_cast<std::uint64_t>(end - position(file_)), frame_)) {
            return false;
        }
        frame_pts_ = cluster_time_ + relative;
        frame_offset_ = start;
        frame_at_ = 0;
        frame_units_ = 0;
        if (!frame_.empty()) {
            return true;
        }
    }
}

std::optional<TemporalUnit> Reader::next_framed() {
    if (frame_at_ >= frame_.size()) {
        frame_.clear();
        if (!read_frame()) {
            frame_.clear();
            return std::nullopt;
        }
    }
    TemporalUnit unit;
    const std::size_t end = unit_end(frame_, frame_at_);
    unit.data.assign(frame_.begin() + static_cast<std::ptrdiff_t>(frame_at_),
                     frame_.begin() + static_cast<std::ptrdiff_t>(end));
    unit.number = units_read_++;
    unit.container_pts = frame_pts_;
    unit.offset = frame_offset_;
    unit.index = frame_units_++;
    unit.kind = kind_of(unit.data);
    frame_at_ = end;
    return unit;
}

void Reader::seek(const TemporalUnit& unit) {
    ended_ = false;
    carried_ = unit.carried;
    sought_time_.reset();
    file_.clear();
    file_.seekg(unit.offset);
    units_read_ = unit.number - unit.index;
    if (container_ != Container::section5) {
        frame_.clear();
        frame_at_ = 0;
        if (container_ == Container::matroska) {
            sought_time_ = unit.container_pts;
        }
        if (!read_frame()) {
            frame_.clear();
            return;
        }
        // The frame's units before it are passed over, keeping their numbers.
        for (int skipped = 0; skipped < unit.index && frame_at_ < frame_.size(); skipped++) {
            frame_at_ = unit_end(frame_, frame_at_);
            frame_units_++;
            units_read_++;
        }
    } else {
        pending_.clear();
    }
}

bool Reader::read_obu(std::vector<std::uint8_t>& obu) {
    obu.clear();
    // The size field, a byte at a time (at most 8 of them).
    for (int i = 0; i < 8; i++) {
        const int byte = file_.get();
        if (byte == std::char_traits<char>::eof()) {
            return false;
        }
        obu.push_back(static_cast<std::uint8_t>(byte));
        if (!(byte & 0x80)) {
            break;
        }
    }
    const auto size = leb128(obu.data(), obu.size());
    if (!size || size->first == 0 || size->first > largest_unit) {
        return false;
    }
    const std::size_t field = obu.size();
    if (payloads_ == Payloads::skipped && !is_kept_setting(type_of(static_cast<std::uint8_t>(file_.peek())))) {
        // Its header alone, as an OBU of size 1.
        const std::int64_t next = position(file_) + static_cast<std::int64_t>(size->first);
        obu = {1, static_cast<std::uint8_t>(file_.get())};
        file_.seekg(next);
        return file_ && next <= static_cast<std::int64_t>(file_size_);
    }
    obu.resize(field + size->first);
    file_.read(reinterpret_cast<char*>(obu.data() + field), static_cast<std::streamsize>(size->first));
    return file_.gcount() == static_cast<std::streamsize>(size->first);
}

std::optional<TemporalUnit> Reader::next_section5() {
    TemporalUnit unit;
    std::vector<std::uint8_t> obu;
    if (pending_.empty()) {
        pending_offset_ = static_cast<std::int64_t>(file_.tellg());
        if (!read_obu(obu)) {
            return std::nullopt;
        }
        pending_ = std::move(obu);
    }
    unit.offset = pending_offset_;
    unit.data = std::move(pending_);
    pending_.clear();
    // The OBUs up to the next temporal delimiter, which starts the next unit.
    // The end of the file, or damage, ends it and the file.
    while (true) {
        const std::int64_t at = position(file_);
        if (!read_obu(obu)) {
            ended_ = true;
            break;
        }
        const std::size_t header = leb128(obu.data(), obu.size())->second;
        if (type_of(obu[header]) == ObuType::temporal_delimiter) {
            pending_ = std::move(obu);
            pending_offset_ = at;
            break;
        }
        append(unit.data, obu);
    }
    unit.number = units_read_++;
    unit.container_pts = unit.number;
    unit.kind = kind_of(unit.data);
    return unit;
}

std::shared_ptr<const std::vector<TemporalUnit>> closed_keyframes(const std::filesystem::path& path) {
    std::error_code ignored;
    const std::string key = platform::path_text(path) + '|' + std::to_string(std::filesystem::file_size(path, ignored)) + '|' +
                            std::to_string(platform::modified_ns(path).value_or(0));
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const std::vector<TemporalUnit>>> scanned;
    {
        const std::lock_guard lock(mutex);
        if (const auto found = scanned.find(key); found != scanned.end()) {
            return found->second;
        }
    }
    // Scanned outside the lock: another file's scan need not wait (the same
    // file scanned twice at once finds the same keyframes).
    auto keyframes = std::make_shared<std::vector<TemporalUnit>>();
    Reader scan(path, Payloads::skipped);
    KeptSettings settings;
    while (std::optional<TemporalUnit> unit = scan.next()) {
        if (unit->kind.closed_key) {
            unit->carried = settings.carried_into(unit->data);
        }
        // A leading picture's settings are left out: AOM's decoder drops
        // them at the next picture that is not one, unless sent again.
        if (!unit->kind.leading) {
            settings.add(unit->data);
        }
        if (unit->kind.closed_key) {
            keyframes->push_back(std::move(*unit));
        }
    }
    const std::lock_guard lock(mutex);
    return scanned.emplace(key, std::move(keyframes)).first->second;
}

}  // namespace vml::av2
