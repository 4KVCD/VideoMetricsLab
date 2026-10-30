#include "vml/film_grain_detect.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

constexpr std::array<std::pair<FilmGrainKind, std::string_view>, 4> kind_names{{
    {film_grain_h274, "h274"},
    {film_grain_h274_model_1, "h274 model 1"},
    {film_grain_afgs1, "afgs1"},
    {film_grain_av1, "av1"},
}};

// SEI payload types (H.264 Annex D, H.265 and H.266 Annex D: the same numbers).
constexpr unsigned sei_user_data_registered_itu_t_t35 = 4;
constexpr unsigned sei_film_grain_characteristics = 19;
// A T.35 message's first bytes for AFGS1: the United States, then AOM's
// provider code and its AFGS1 application identifier (FFmpeg's itut35.h).
constexpr std::string_view afgs1_t35_prefix{"\xB5\x58\x90\x01", 4};

// One NAL unit's place in an Annex B stream: its bytes, header included,
// without the start code or the zero bytes before the next one.
template <typename Visit>
void for_each_nal_unit(std::string_view stream, Visit visit) {
    constexpr std::string_view start_code{"\0\0\1", 3};
    std::size_t start = stream.find(start_code);
    while (start != std::string_view::npos) {
        start += start_code.size();
        const std::size_t next = stream.find(start_code, start);
        std::size_t end = next == std::string_view::npos ? stream.size() : next;
        while (end > start && stream[end - 1] == '\0') {
            --end;  // a 4-byte start code's first zero, or trailing_zero_8bits
        }
        visit(stream.substr(start, end - start));
        start = next;
    }
}

// Whether `nal` is a prefix SEI NAL unit of layer 0 for `codec`, and how many
// bytes its header takes.
std::optional<std::size_t> prefix_sei_header(std::string_view nal, std::string_view codec) {
    const auto byte = [&nal](std::size_t i) { return static_cast<unsigned char>(nal[i]); };
    if (codec == "h264") {
        return !nal.empty() && (byte(0) & 0x1F) == 6 ? std::optional<std::size_t>(1) : std::nullopt;
    }
    if (nal.size() < 2) {
        return std::nullopt;
    }
    if (codec == "hevc") {
        const unsigned type = (byte(0) >> 1) & 0x3F;
        const unsigned layer = ((byte(0) & 1) << 5) | (byte(1) >> 3);
        return type == 39 && layer == 0 ? std::optional<std::size_t>(2) : std::nullopt;
    }
    if (codec == "vvc") {
        const unsigned type = byte(1) >> 3;
        const unsigned layer = byte(0) & 0x3F;
        return type == 23 && layer == 0 ? std::optional<std::size_t>(2) : std::nullopt;
    }
    return std::nullopt;
}

// A NAL unit's payload without its emulation prevention bytes (the 3 in each
// 00 00 03).
std::string without_emulation_prevention(std::string_view payload) {
    std::string rbsp;
    rbsp.reserve(payload.size());
    int zeros = 0;
    for (const char c : payload) {
        if (zeros >= 2 && c == '\3') {
            zeros = 0;
            continue;
        }
        rbsp.push_back(c);
        zeros = c == '\0' ? zeros + 1 : 0;
    }
    return rbsp;
}

// The film grain the SEI messages of an SEI NAL unit's payload (`rbsp`, its
// header left out) describe. A message whose size runs past the end ends the
// reading: the rest cannot be trusted.
unsigned film_grain_in_sei(std::string_view rbsp) {
    unsigned kinds = 0;
    std::size_t at = 0;
    const auto byte = [&rbsp](std::size_t i) { return static_cast<unsigned char>(rbsp[i]); };
    // A value coded as 255s and a last byte (payloadType, payloadSize).
    const auto read_coded = [&](unsigned& value) {
        value = 0;
        while (at < rbsp.size() && byte(at) == 0xFF) {
            value += 255;
            ++at;
        }
        if (at >= rbsp.size()) {
            return false;
        }
        value += byte(at++);
        return true;
    };
    // Messages until only rbsp_trailing_bits (0x80) are left.
    while (at < rbsp.size() && !(at + 1 == rbsp.size() && byte(at) == 0x80)) {
        unsigned type = 0, size = 0;
        if (!read_coded(type) || !read_coded(size) || size > rbsp.size() - at) {
            break;
        }
        const std::string_view message = rbsp.substr(at, size);
        at += size;
        if (type == sei_film_grain_characteristics && !message.empty()) {
            const auto first = static_cast<unsigned char>(message[0]);
            const bool cancelled = (first & 0x80) != 0;
            const unsigned model = (first >> 5) & 3;
            if (!cancelled && model == 0) {
                kinds |= film_grain_h274;
            } else if (!cancelled && model == 1) {
                kinds |= film_grain_h274_model_1;
            }
        } else if (type == sei_user_data_registered_itu_t_t35 && message.starts_with(afgs1_t35_prefix)) {
            kinds |= film_grain_afgs1;
        }
    }
    return kinds;
}

// Bits of an AV1 sequence header, most significant first. Reading past its
// end sets `overrun` and gives zeros.
class BitReader {
public:
    explicit BitReader(std::string_view bytes) : bytes_(bytes) {}

    std::uint32_t read(unsigned count) {
        std::uint32_t value = 0;
        for (unsigned i = 0; i < count; ++i) {
            const std::size_t byte = position_ / 8;
            if (byte >= bytes_.size()) {
                overrun = true;
                return 0;
            }
            const unsigned bit = (static_cast<unsigned char>(bytes_[byte]) >> (7 - position_ % 8)) & 1;
            value = (value << 1) | bit;
            ++position_;
        }
        return value;
    }
    void skip(unsigned count) { position_ += count; }
    // Whether every bit left is a zero.
    bool rest_is_zero() {
        while (position_ < bytes_.size() * 8) {
            if (read(1) != 0) {
                return false;
            }
        }
        return !overrun;
    }
    // AV1's uvlc(): leading zeros, a one, then that many bits (none from 32
    // zeros on).
    void skip_uvlc() {
        unsigned zeros = 0;
        while (!overrun && read(1) == 0) {
            ++zeros;
        }
        if (zeros < 32) {
            skip(zeros);
        }
    }
    bool overrun = false;

private:
    std::string_view bytes_;
    std::size_t position_ = 0;
};

// An AV1 sequence header OBU's film_grain_params_present (AV1 spec 5.5),
// read past everything before it; nothing when the header ends first.
std::optional<bool> av1_film_grain_params_present(std::string_view header) {
    BitReader bits(header);
    const unsigned profile = bits.read(3);
    bits.skip(1);  // still_picture
    const bool reduced_still_picture_header = bits.read(1);
    if (reduced_still_picture_header) {
        bits.skip(5);  // seq_level_idx[0]
    } else {
        bool decoder_model_info = false;
        unsigned buffer_delay_length = 0;
        if (bits.read(1)) {    // timing_info_present_flag: timing_info()
            bits.skip(32 + 32);  // num_units_in_display_tick, time_scale
            if (bits.read(1)) {  // equal_picture_interval
                bits.skip_uvlc();  // num_ticks_per_picture_minus_1
            }
            decoder_model_info = bits.read(1);
            if (decoder_model_info) {  // decoder_model_info()
                buffer_delay_length = bits.read(5) + 1;
                bits.skip(32 + 5 + 5);  // num_units_in_decoding_tick, two lengths
            }
        }
        const bool initial_display_delay = bits.read(1);
        const unsigned operating_points = bits.read(5) + 1;
        for (unsigned i = 0; i < operating_points; ++i) {
            bits.skip(12);  // operating_point_idc
            if (bits.read(5) > 7) {  // seq_level_idx
                bits.skip(1);        // seq_tier
            }
            if (decoder_model_info && bits.read(1)) {  // decoder_model_present_for_this_op
                bits.skip(2 * buffer_delay_length + 1);  // operating_parameters_info()
            }
            if (initial_display_delay && bits.read(1)) {
                bits.skip(4);  // initial_display_delay_minus_1
            }
        }
    }
    const unsigned width_bits = bits.read(4) + 1;
    const unsigned height_bits = bits.read(4) + 1;
    bits.skip(width_bits + height_bits);  // max_frame_width_minus_1, max_frame_height_minus_1
    if (!reduced_still_picture_header && bits.read(1)) {  // frame_id_numbers_present_flag
        bits.skip(4 + 3);
    }
    bits.skip(3);  // use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
    if (!reduced_still_picture_header) {
        bits.skip(4);  // interintra, masked compound, warped motion, dual filter
        const bool order_hint = bits.read(1);
        if (order_hint) {
            bits.skip(2);  // enable_jnt_comp, enable_ref_frame_mvs
        }
        unsigned force_screen_content_tools = 2;
        if (!bits.read(1)) {  // seq_choose_screen_content_tools
            force_screen_content_tools = bits.read(1);
        }
        if (force_screen_content_tools > 0 && !bits.read(1)) {  // seq_choose_integer_mv
            bits.skip(1);                                       // seq_force_integer_mv
        }
        if (order_hint) {
            bits.skip(3);  // order_hint_bits_minus_1
        }
    }
    bits.skip(3);  // enable_superres, enable_cdef, enable_restoration
    // color_config()
    const bool high_bitdepth = bits.read(1);
    const bool twelve_bit = profile == 2 && high_bitdepth && bits.read(1);
    const bool mono_chrome = profile != 1 && bits.read(1);
    unsigned primaries = 2, transfer = 2, matrix = 2;
    if (bits.read(1)) {  // color_description_present_flag
        primaries = bits.read(8);
        transfer = bits.read(8);
        matrix = bits.read(8);
    }
    if (mono_chrome) {
        bits.skip(1);  // color_range; no separate_uv_delta_q
    } else {
        const bool srgb = primaries == 1 && transfer == 13 && matrix == 0;  // 4:4:4, full range: nothing read
        if (!srgb) {
            bits.skip(1);  // color_range
            bool subsampled_x = profile == 0 || (profile == 2 && !twelve_bit);
            bool subsampled_y = profile == 0;
            if (profile == 2 && twelve_bit) {
                subsampled_x = bits.read(1);
                subsampled_y = subsampled_x && bits.read(1);
            }
            if (subsampled_x && subsampled_y) {
                bits.skip(2);  // chroma_sample_position
            }
        }
        bits.skip(1);  // separate_uv_delta_q
    }
    const bool film_grain_params_present = bits.read(1);
    // trailing_bits(): a one, then zeros to the header's end. Anything else
    // means the reading strayed (or the header is damaged): no answer.
    if (bits.read(1) != 1 || !bits.rest_is_zero() || bits.overrun) {
        return std::nullopt;
    }
    return film_grain_params_present;
}

}  // namespace

std::vector<std::string> film_grain_names(unsigned kinds) {
    std::vector<std::string> names;
    for (const auto& [kind, name] : kind_names) {
        if (kinds & kind) {
            names.emplace_back(name);
        }
    }
    return names;
}

unsigned film_grain_from_names(const std::vector<std::string>& names) {
    unsigned kinds = 0;
    for (const std::string& name : names) {
        for (const auto& [kind, known] : kind_names) {
            if (name == known) {
                kinds |= kind;
            }
        }
    }
    return kinds;
}

unsigned film_grain_in_annex_b(std::string_view stream, std::string_view codec) {
    unsigned kinds = 0;
    for_each_nal_unit(stream, [&](std::string_view nal) {
        if (const std::optional<std::size_t> header = prefix_sei_header(nal, codec)) {
            kinds |= film_grain_in_sei(without_emulation_prevention(nal.substr(*header)));
        }
    });
    return kinds;
}

std::optional<unsigned> film_grain_in_av1(std::string_view obus) {
    constexpr unsigned obu_sequence_header = 1;
    std::size_t at = 0;
    while (at < obus.size()) {
        const auto header = static_cast<unsigned char>(obus[at++]);
        const unsigned type = (header >> 3) & 0xF;
        const bool has_extension = (header >> 2) & 1;
        const bool has_size = (header >> 1) & 1;
        if (has_extension) {
            ++at;
        }
        std::size_t size = obus.size() > at ? obus.size() - at : 0;  // without a size: the rest
        if (has_size) {
            std::uint64_t coded = 0;  // leb128()
            for (int i = 0; i < 8; ++i) {
                if (at >= obus.size()) {
                    return std::nullopt;
                }
                const auto byte = static_cast<unsigned char>(obus[at++]);
                coded |= static_cast<std::uint64_t>(byte & 0x7F) << (7 * i);
                if (!(byte & 0x80)) {
                    break;
                }
            }
            size = static_cast<std::size_t>(coded);
        }
        if (at > obus.size() || size > obus.size() - at) {
            return std::nullopt;
        }
        if (type == obu_sequence_header) {
            const std::optional<bool> grain = av1_film_grain_params_present(obus.substr(at, size));
            if (!grain) {
                return std::nullopt;
            }
            return *grain ? film_grain_av1 : 0u;
        }
        at += size;
    }
    return std::nullopt;
}

std::vector<std::string> film_grain_scan_command(const std::string& ffmpeg, const VideoInfo& info) {
    std::vector<std::string> command{ffmpeg, "-nostdin", "-v", "error"};
    const std::vector<std::string> input = info.input_options();
    command.insert(command.end(), input.begin(), input.end());
    // The first video stream (V: not a cover picture), copied from its first
    // key frame on.
    command.insert(command.end(), {"-i", platform::path_text(info.path), "-map", "0:V:0", "-c:v", "copy",
                                   "-frames:v", "3"});
    // In MKV and MP4 an H.264, HEVC or VVC stream's NAL units are prefixed by
    // their length, not by start codes; a raw stream already has start codes.
    const std::string codec = ascii_lower(info.codec_name);
    if (codec != "av1" && !info.is_raw_stream()) {
        command.insert(command.end(), {"-bsf:v", codec + "_mp4toannexb"});
    }
    command.insert(command.end(), {"-f", "data", "pipe:1"});
    return command;
}

std::optional<unsigned> detect_film_grain(const VideoInfo& info, const std::string& ffmpeg,
                                          const CancelToken& cancel) {
    const std::string codec = ascii_lower(info.codec_name);
    if (codec != "h264" && codec != "hevc" && codec != "vvc" && codec != "av1") {
        return 0u;
    }
    platform::ProcessOutput output;
    try {
        // Seconds even for 4K from a slow disk; a file whose packets are never
        // marked as key frames would be read to its end, which the limit stops.
        output = platform::run_program(film_grain_scan_command(ffmpeg, info), std::chrono::seconds(10), {},
                                       [&cancel] { return cancel.cancelled(); });
    } catch (const platform::ProcessError&) {
        return std::nullopt;  // no FFmpeg to run: unknown, as for a failed reading
    }
    if (output.stopped) {
        throw Cancelled();
    }
    // Only the exit code: FFmpeg's VVC parser logs errors for --fga 3's
    // messages while the copy itself is fine.
    if (output.timed_out || output.exit_code != 0) {
        return std::nullopt;
    }
    if (codec == "av1") {
        return film_grain_in_av1(output.standard_output);
    }
    return film_grain_in_annex_b(output.standard_output, codec);
}

}  // namespace vml
