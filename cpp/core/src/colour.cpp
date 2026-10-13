#include "vml/colour.h"

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

bool untagged(std::string_view name) {
    return name.empty() || name == "unknown" || name == "unspecified" || name == "reserved";
}

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

UnsupportedColourError::UnsupportedColourError(std::string kind, std::string value)
    : std::invalid_argument("unsupported " + kind + ": " + value), kind_(std::move(kind)), value_(std::move(value)) {}

bool VideoColour::hdr() const { return !sdr_transfers.contains(transfer); }

VideoColour video_colour(const VideoInfo& info, bool rgb, bool full_range_untagged) {
    VideoColour colour;
    const std::string matrix_name = ascii_lower(info.color_space);
    if (rgb) {
        colour.matrix = 0;
    } else if (untagged(matrix_name)) {
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
    if (untagged(transfer_name)) {
        colour.transfer = colour.matrix == 0 ? 13 : colour.matrix == 5 ? 5 : bt2020_family ? 16 : 1;
    } else if (const auto found = transfers.find(transfer_name); found != transfers.end()) {
        colour.transfer = found->second;
    } else {
        throw UnsupportedColourError("transfer", info.color_transfer);
    }

    const std::string primaries_name = ascii_lower(info.color_primaries);
    if (untagged(primaries_name)) {
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

void describe_png(const std::filesystem::path& path, const VideoColour& colour) {
    const std::optional<std::string> read = platform::read_file(path);
    if (!read) {
        throw std::runtime_error("The picture " + platform::path_text(path) + " could not be read.");
    }
    const std::string_view data = *read;
    const auto byte = [&data](std::size_t at) {
        return static_cast<std::size_t>(static_cast<unsigned char>(data[at]));
    };
    std::filesystem::path temporary = path;
    temporary += ".colour";
    {
        // The signature, then each chunk as it is, less the colour chunks,
        // with this picture's after the header; written as they are found,
        // as a 4K picture is some 20 MB.
        std::ofstream written(temporary, std::ios::binary | std::ios::trunc);
        const auto write = [&written](std::string_view bytes) {
            written.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        };
        write(data.substr(0, 8));
        std::size_t position = 8;
        while (position + 8 <= data.size()) {
            const std::size_t length =
                (byte(position) << 24) | (byte(position + 1) << 16) | (byte(position + 2) << 8) | byte(position + 3);
            const std::string_view kind = data.substr(position + 4, 4);
            const std::string_view whole = data.substr(position, 12 + length);
            if (kind == "IHDR") {
                write(whole);
                write(png_colour_chunks(colour));
            } else if (!colour_chunk_kinds.contains(kind)) {
                write(whole);
            }
            position += 12 + length;
        }
        written.close();
        if (written.fail()) {
            throw std::runtime_error("The picture " + platform::path_text(path) + " could not be written.");
        }
    }
    std::filesystem::rename(temporary, path);
}

}  // namespace vml
