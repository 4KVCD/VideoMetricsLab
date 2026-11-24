#pragma once
// Matroska (EBML) reading, shared by the AV2 reader (av2_stream.cpp) and the
// writing application's (matroska.cpp). Internal to the core library.

#include <cstdint>
#include <istream>
#include <optional>
#include <string_view>

namespace vml::matroska {

// Matroska (EBML): each element is its ID, its size and its contents. The ID
// and the size are variable-length integers: the first byte's leading zero
// bits say how many bytes follow; an ID keeps that length marker, a size
// does not, and a size of all ones is unknown (a live stream's cluster). The
// elements read here, by ID:
namespace mkv {
constexpr std::uint64_t ebml = 0x1A45DFA3, segment = 0x18538067, info = 0x1549A966, timecode_scale = 0x2AD7B1,
                        writing_app = 0x5741,
                        tracks = 0x1654AE6B, track_entry = 0xAE, track_number = 0xD7, codec_id = 0x86, video = 0xE0,
                        content_encodings = 0x6D80, pixel_width = 0xB0, pixel_height = 0xBA,
                        cluster = 0x1F43B675, timecode = 0xE7,
                        block_group = 0xA0, block = 0xA1, simple_block = 0xA3;
constexpr std::uint64_t unknown_size = ~0ULL;
constexpr std::string_view av2_codec = "V_AV2";
constexpr long long default_timecode_scale = 1'000'000;  // nanoseconds a tick: a millisecond
}  // namespace mkv

// A variable-length integer from `next` (each call a byte, or -1 at the
// end), at most `longest` bytes: an ID with its marker, a size without it
// (mkv::unknown_size where all its bits are ones). Nothing where it is cut
// short or longer.
template <typename NextByte>
std::optional<std::uint64_t> ebml_number(NextByte next, bool is_id, int longest) {
    const int first = next();
    if (first <= 0) {
        return std::nullopt;  // the end, or a length past 8 bytes
    }
    int length = 1;
    while (!(first & (0x80 >> (length - 1)))) {
        length++;
    }
    if (length > longest) {
        return std::nullopt;
    }
    const std::uint64_t value_bits = static_cast<std::uint64_t>(first) & (0xFFu >> length);
    std::uint64_t value = is_id ? static_cast<std::uint64_t>(first) : value_bits;
    bool all_ones = value_bits == (0xFFu >> length);
    for (int i = 1; i < length; i++) {
        const int byte = next();
        if (byte < 0) {
            return std::nullopt;
        }
        value = (value << 8) | static_cast<std::uint64_t>(byte);
        all_ones = all_ones && byte == 0xFF;
    }
    return !is_id && all_ones ? mkv::unknown_size : value;
}

struct Element {
    std::uint64_t id = 0;
    std::uint64_t size = 0;  // mkv::unknown_size where not given
};

inline std::optional<Element> read_element(std::istream& in) {
    const auto next = [&in] {
        const int byte = in.get();
        return byte == std::char_traits<char>::eof() ? -1 : byte;
    };
    const std::optional<std::uint64_t> id = ebml_number(next, true, 4);
    const std::optional<std::uint64_t> size = id ? ebml_number(next, false, 8) : std::nullopt;
    if (!size) {
        return std::nullopt;
    }
    return Element{*id, *size};
}

// An unsigned integer element's contents (at most 8 bytes, big-endian).
inline std::optional<std::uint64_t> read_unsigned(std::istream& in, std::uint64_t size) {
    if (size > 8) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    for (std::uint64_t i = 0; i < size; i++) {
        const int byte = in.get();
        if (byte == std::char_traits<char>::eof()) {
            return std::nullopt;
        }
        value = (value << 8) | static_cast<std::uint64_t>(byte);
    }
    return value;
}

inline std::int64_t position(std::istream& in) { return static_cast<std::int64_t>(in.tellg()); }

// Where an element whose contents start at the stream's position ends.
inline std::int64_t end_of(std::istream& in, const Element& element) {
    return position(in) + static_cast<std::int64_t>(element.size);
}


// Calls `read(child)` for each element inside one that ends at `end`, the
// stream at the child's contents, then goes past the child. False where one
// is damaged or `read` says so.
template <typename Read>
bool each_child(std::istream& in, std::int64_t end, Read read) {
    while (in && position(in) < end) {
        const std::optional<Element> child = read_element(in);
        if (!child || child->size == mkv::unknown_size) {
            return false;
        }
        const std::int64_t child_end = end_of(in, *child);
        if (child_end > end || !read(*child)) {
            return false;
        }
        in.clear();
        in.seekg(child_end);
    }
    return static_cast<bool>(in);
}


}  // namespace vml::matroska
