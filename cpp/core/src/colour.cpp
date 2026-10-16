#include "vml/colour.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string_view>
#include <utility>

#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// FFmpeg's tag names (as ffprobe prints them, with their aliases), mapped to
// the values of Vship's enums in VshipColor.h, which are the H.273 numbers.
// The mapping is FFVship's (src/ffvship_utility/ffmpegToVshipColorFormat.hpp
// in Vship 5.1.1). A tag missing here has no Vship value.
const std::map<std::string, int, std::less<>> matrices{
    {"gbr", 0},      {"rgb", 0},       {"bt709", 1},      {"bt470bg", 5},   {"smpte170m", 6},
    {"ycgco", 8},    {"ycocg", 8},     {"bt2020nc", 9},   {"bt2020ncl", 9}, {"bt2020c", 10},
    {"bt2020cl", 10}, {"ictcp", 14},   {"ycgco-re", 16},  {"ycgco-ro", 17},
};
// BT.2020's 10- and 12-bit curves are BT.709's curve (H.273), and FFVship
// maps them that way.
const std::map<std::string, int, std::less<>> transfers{
    {"bt709", 1},         {"bt470m", 4},         {"gamma22", 4},      {"bt470bg", 5},
    {"gamma28", 5},       {"smpte170m", 6},      {"smpte240m", 7},    {"linear", 8},
    {"iec61966-2-1", 13}, {"srgb", 13},          {"iec61966_2_1", 13}, {"bt2020-10", 1},
    {"bt2020_10bit", 1},  {"bt2020-12", 1},      {"bt2020_12bit", 1}, {"smpte2084", 16},
    {"smpte428", 17},     {"smpte428_1", 17},    {"arib-std-b67", 18},
};
const std::map<std::string, int, std::less<>> primaries_codes{
    {"bt709", 1}, {"bt470m", 4}, {"bt470bg", 5}, {"smpte170m", 6}, {"smpte240m", 7}, {"bt2020", 9}, {"smpte432", 12},
};

// The transfers whose light is relative to the display's white (SDR). The
// others (PQ, ST 428 and HLG) are HDR.
const std::set<int> sdr_transfers{1, 4, 5, 6, 7, 8, 13};

// FFmpeg's names of the matrices it can convert with, by H.273 number.
const std::map<int, std::string_view> ffmpeg_matrices{
    {1, "bt709"}, {5, "bt470bg"}, {6, "smpte170m"}, {8, "ycgco"},     {9, "bt2020nc"},
    {10, "bt2020c"}, {14, "ictcp"}, {16, "ycgco-re"}, {17, "ycgco-ro"},
};

// ------------------------------------------------------ PNG colour chunks

// Each transfer that Vship treats as a plain power curve, with its exponent.
const std::map<int, double> gammas{{1, 2.4}, {4, 2.2}, {5, 2.8}, {6, 2.8}, {7, 2.4}};
// The white, red, green and blue chromaticities of each primaries code, as
// x and y.
const std::map<int, std::array<double, 8>> chromaticities{
    {1, {0.3127, 0.3290, 0.64, 0.33, 0.30, 0.60, 0.15, 0.06}},
    {4, {0.310, 0.316, 0.67, 0.33, 0.21, 0.71, 0.14, 0.08}},
    {5, {0.3127, 0.3290, 0.64, 0.33, 0.29, 0.60, 0.15, 0.06}},
    {6, {0.3127, 0.3290, 0.630, 0.340, 0.310, 0.595, 0.155, 0.070}},
    {7, {0.3127, 0.3290, 0.630, 0.340, 0.310, 0.595, 0.155, 0.070}},
    {9, {0.3127, 0.3290, 0.708, 0.292, 0.170, 0.797, 0.131, 0.046}},
    {12, {0.3127, 0.3290, 0.680, 0.320, 0.265, 0.690, 0.150, 0.060}},
};
// The PNG chunks that say how its values are to be read. Any of these that
// FFmpeg wrote are replaced.
const std::set<std::string_view> colour_chunk_kinds{"gAMA", "cHRM", "sRGB", "iCCP", "cICP", "mDCV", "cLLI"};

// The CRC-32 a PNG chunk ends with (zlib's crc32).
std::uint32_t crc32(std::string_view bytes) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> made{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int bit = 0; bit < 8; ++bit) {
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }
            made[n] = c;
        }
        return made;
    }();
    std::uint32_t c = 0xFFFFFFFFu;
    for (const char byte : bytes) {
        c = table[(c ^ static_cast<unsigned char>(byte)) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

// `value` as four bytes, most significant first, as PNG writes numbers.
void append_big_endian(std::string& bytes, std::uint32_t value) {
    for (const int shift : {24, 16, 8, 0}) {
        bytes.push_back(static_cast<char>((value >> shift) & 0xFFu));
    }
}

// One PNG chunk: its length, type, body and CRC.
std::string png_chunk(std::string_view kind, std::string_view body) {
    std::string chunk;
    append_big_endian(chunk, static_cast<std::uint32_t>(body.size()));
    const std::string typed = std::string(kind) + std::string(body);
    chunk += typed;
    append_big_endian(chunk, crc32(typed));
    return chunk;
}

// A positive number rounded to a whole one as Python's round() does it: to
// the nearest, halves to the even one.
std::uint32_t python_round(double value) { return static_cast<std::uint32_t>(std::nearbyint(value)); }

}  // namespace

bool is_untagged_tag(std::string_view value) {
    const std::string name = ascii_lower(value);
    return name.empty() || name == "unknown" || name == "unspecified" || name == "reserved";
}

UnsupportedColourError::UnsupportedColourError(std::string kind, std::string value)
    : std::invalid_argument("unsupported " + kind + ": " + value), kind_(std::move(kind)), value_(std::move(value)) {}

bool VideoColour::hdr() const { return !sdr_transfers.contains(transfer); }

VideoColour video_colour(const VideoInfo& info, bool rgb, bool full_range_untagged) {
    VideoColour colour;
    const std::string matrix_name = ascii_lower(info.color_space);
    if (rgb) {
        colour.matrix = 0;
    } else if (is_untagged_tag(matrix_name)) {
        // The video's own height, not the size it is compared at: otherwise
        // an untagged 1080p source scaled to a 640x360 encode would be read
        // as BT.470BG.
        colour.matrix = info.height > 650 ? 1 : 5;
    } else if (const auto found = matrices.find(matrix_name); found != matrices.end()) {
        colour.matrix = found->second;
    } else {
        throw UnsupportedColourError("matrix", info.color_space);
    }
    const bool bt2020_family = colour.matrix == 9 || colour.matrix == 10 || colour.matrix == 14;

    const std::string transfer_name = ascii_lower(info.color_transfer);
    if (is_untagged_tag(transfer_name)) {
        colour.transfer = colour.matrix == 0 ? 13 : colour.matrix == 5 ? 5 : bt2020_family ? 16 : 1;
    } else if (const auto found = transfers.find(transfer_name); found != transfers.end()) {
        colour.transfer = found->second;
    } else {
        throw UnsupportedColourError("transfer", info.color_transfer);
    }

    const std::string primaries_name = ascii_lower(info.color_primaries);
    if (is_untagged_tag(primaries_name)) {
        colour.primaries = colour.matrix == 5 ? 5 : bt2020_family ? 9 : 1;
    } else if (const auto found = primaries_codes.find(primaries_name); found != primaries_codes.end()) {
        colour.primaries = found->second;
    } else {
        throw UnsupportedColourError("primaries", info.color_primaries);
    }

    const std::string range_name = ascii_lower(info.color_range);
    if (range_name == "pc" || range_name == "jpeg" || range_name == "full" ||
        (range_name.empty() && full_range_untagged)) {
        colour.full_range = true;
    } else if (range_name.empty() || range_name == "unknown" || range_name == "unspecified" || range_name == "tv" ||
               range_name == "mpeg" || range_name == "limited") {
        colour.full_range = rgb;
    } else {
        throw UnsupportedColourError("range", info.color_range);
    }
    return colour;
}

bool is_rgb_format(const std::string& pix_fmt) {
    const std::string name = ascii_lower(pix_fmt);
    for (const std::string_view prefix : {"gbr", "rgb", "bgr", "argb", "abgr", "0rgb", "0bgr"}) {
        if (name.starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

std::optional<VideoColour> colour_of(const VideoInfo& info) {
    const std::string name = ascii_lower(info.pix_fmt);
    try {
        return video_colour(info, is_rgb_format(name), name.starts_with("yuvj"));
    } catch (const UnsupportedColourError&) {
        return std::nullopt;
    }
}

std::optional<std::string_view> ffmpeg_matrix_name(int matrix) {
    const auto found = ffmpeg_matrices.find(matrix);
    return found != ffmpeg_matrices.end() ? std::optional(found->second) : std::nullopt;
}

std::string png_colour_chunks(const VideoColour& colour) {
    const auto gamma = gammas.find(colour.transfer);
    const auto points = chromaticities.find(colour.primaries);
    if (gamma != gammas.end() && points != chromaticities.end()) {
        std::string exponent;
        append_big_endian(exponent, python_round(100000.0 / gamma->second));
        std::string whitepoint_and_primaries;
        for (const double value : points->second) {
            append_big_endian(whitepoint_and_primaries, python_round(value * 100000.0));
        }
        return png_chunk("gAMA", exponent) + png_chunk("cHRM", whitepoint_and_primaries);
    }
    // Primaries, transfer, matrix (RGB) and full range.
    const std::string code{static_cast<char>(colour.primaries), static_cast<char>(colour.transfer), '\0', '\1'};
    return png_chunk("cICP", code);
}

namespace {

// The chunk that fills the room left where a picture's colour chunks were
// replaced in place: ancillary and private (its first two letters lower
// case), so that readers skip it. libjxl's tools score a 4K pair with it
// exactly as they score the pair written again whole (SSIMULACRA2, and
// Butteraugli's score, 3-norm and distortion map).
constexpr std::string_view padding_chunk_kind = "vmPd";

std::size_t big_endian_at(std::string_view bytes, std::size_t at) {
    std::size_t value = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        value = (value << 8) | static_cast<unsigned char>(bytes[at + i]);
    }
    return value;
}

std::runtime_error unreadable(const std::filesystem::path& path) {
    return std::runtime_error("The picture " + platform::path_text(path) + " could not be read.");
}

std::runtime_error unwritable(const std::filesystem::path& path) {
    return std::runtime_error("The picture " + platform::path_text(path) + " could not be written.");
}

// A picture's chunks before its first IDAT, where its colour chunks are
// (PNG's rule): the header, the others less the colour chunks, in order,
// and the bytes from the end of the signature to the first IDAT.
struct PngHead {
    std::string header;
    std::string kept;
    std::size_t room = 0;
};

// The chunks before the picture's data, read one by one; nothing when
// there is no IDAT where the chunks say it is.
std::optional<PngHead> read_png_head(std::istream& file) {
    PngHead head;
    std::string start(8, '\0');
    if (!file.read(start.data(), 8)) {
        return std::nullopt;
    }
    std::size_t position = 8;
    while (position < (1 << 20)) {  // FFmpeg's take some 150 bytes; a large ICC profile some kilobytes
        std::string chunk(8, '\0');
        if (!file.read(chunk.data(), 8)) {
            return std::nullopt;
        }
        const std::string kind = chunk.substr(4, 4);  // a copy: chunk grows below
        if (kind == "IDAT") {
            head.room = position - 8;
            return head.header.empty() ? std::nullopt : std::optional(std::move(head));
        }
        const std::size_t length = big_endian_at(chunk, 0);
        if (length > (1 << 20)) {
            return std::nullopt;
        }
        chunk.resize(12 + length);
        if (!file.read(chunk.data() + 8, static_cast<std::streamsize>(length + 4))) {
            return std::nullopt;
        }
        if (kind == "IHDR") {
            head.header = chunk;
        } else if (!colour_chunk_kinds.contains(kind)) {
            head.kept += chunk;
        }
        position += chunk.size();
    }
    return std::nullopt;
}

// Whether the picture's last 4 KB from its first IDAT (at `data_start`) on
// hold a colour chunk's name: a colour chunk after the picture's data,
// which PNG forbids and FFmpeg does not write, but which is removed all the
// same where there is one. The name inside the compressed data only costs
// the picture a whole rewrite.
bool colour_chunk_near_end(std::istream& file, std::streamoff data_start) {
    file.clear();
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    const std::streamoff from = std::max(data_start, size - 4096);
    std::string bytes(static_cast<std::size_t>(std::max<std::streamoff>(0, size - from)), '\0');
    file.seekg(from);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        return true;
    }
    return std::any_of(colour_chunk_kinds.begin(), colour_chunk_kinds.end(),
                       [&bytes](std::string_view kind) { return bytes.find(kind) != std::string::npos; });
}

// The picture written again whole, its colour chunks replaced, in one write
// (as the Python app writes it).
void rewrite_png(const std::filesystem::path& path, const VideoColour& colour) {
    const std::optional<std::string> read = platform::read_file(path);
    if (!read) {
        throw unreadable(path);
    }
    const std::string_view data = *read;
    std::string written;
    written.reserve(data.size() + 256);
    written += data.substr(0, 8);
    std::size_t position = 8;
    while (position + 8 <= data.size()) {
        const std::size_t length = big_endian_at(data, position);
        const std::string_view kind = data.substr(position + 4, 4);
        const std::string_view whole = data.substr(position, 12 + length);
        if (kind == "IHDR") {
            written += whole;
            written += png_colour_chunks(colour);
        } else if (!colour_chunk_kinds.contains(kind)) {
            written += whole;
        }
        position += 12 + length;
    }
    std::filesystem::path temporary = path;
    temporary += ".colour";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(written.data(), static_cast<std::streamsize>(written.size()));
        file.close();
        if (file.fail()) {
            throw unwritable(path);
        }
    }
    std::filesystem::rename(temporary, path);
}

}  // namespace

void describe_png(const std::filesystem::path& path, const VideoColour& colour) {
    // In place where they fit: the chunks before the picture's data are read,
    // and written again with this picture's colour chunks for FFmpeg's, the
    // room left filled with a padding chunk. The 20 MB of a 4K picture stay
    // where they are; written again whole, they made the CPU's SSIMULACRA2
    // and Butteraugli read and write 80 MB more a pair. FFmpeg 9's colour
    // chunks take 132 bytes, these at most 60.
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        if (!file) {
            throw unreadable(path);
        }
        std::optional<PngHead> head = read_png_head(file);
        if (head && !colour_chunk_near_end(file, static_cast<std::streamoff>(8 + head->room))) {
            std::string chunks = head->header + png_colour_chunks(colour) + head->kept;
            if (chunks.size() == head->room || chunks.size() + 12 <= head->room) {
                if (chunks.size() < head->room) {
                    chunks += png_chunk(padding_chunk_kind, std::string(head->room - chunks.size() - 12, '\0'));
                }
                file.clear();
                file.seekp(8);
                file.write(chunks.data(), static_cast<std::streamsize>(chunks.size()));
                file.flush();
                if (file.fail()) {
                    throw unwritable(path);
                }
                return;
            }
        }
    }
    rewrite_png(path, colour);
}

}  // namespace vml
