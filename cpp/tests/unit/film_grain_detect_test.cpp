// Film grain read from a video's stream (film_grain_detect.h): the SEI
// messages of VVenC's --fga encodes, AV1 sequence headers, and messages made
// to reach each of the scan's rules.
#include "vml/film_grain_detect.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "vml/result_json.h"
#include "vml/run_json.h"

namespace vml {
namespace {

std::string bytes(const std::string& hex) {
    std::string out;
    for (std::size_t at = 0; at + 1 < hex.size(); at += 2) {
        out.push_back(static_cast<char>(std::stoi(hex.substr(at, 2), nullptr, 16)));
    }
    return out;
}

// NAL units, as hex, each after a start code (4 bytes for the first, as
// encoders write it, 3 for the rest).
std::string annex_b(std::initializer_list<std::string> nal_units) {
    std::string stream;
    for (const std::string& nal : nal_units) {
        stream += stream.empty() ? bytes("00000001") : bytes("000001");
        stream += bytes(nal);
    }
    return stream;
}

// VVenC's prefix SEI NAL units, as its streams hold them (emulation
// prevention bytes in).
const std::string vvenc_8_bit_grain = "00b9131700b8020afa0681c0e0020afa0d070300082be8341c0e6080";  // --fga 1
const std::string vvenc_grain_without_components = "00b91302008680";  // a 4K --fga 1 encode's first
const std::string vvenc_model_1_grain =                               // --fga 3, which FFmpeg's parser rejects
    "00b9136221f8250a0c0138023b03f07c34dc054008ec0fc1f38490138023b03f07d2974047008ec0fc1f5eed0100023b03f07c0e8541"
    "8e02c580618f090e1402c580618f11180880b1601863c6f683002c580618e03429f0110152700d937ded1c0549c0364f80";
const std::string vvenc_mastering_display = "00b9891833c286c41d4c0bb884d03e803d13404200989680000003000180";
const std::string vvc_sps = "00790000";  // another NAL unit: skipped

TEST(FilmGrainDetect, VvencsGrainMessagesAreFound) {
    EXPECT_EQ(film_grain_in_annex_b(annex_b({vvc_sps, vvenc_8_bit_grain}), "vvc"), film_grain_h274);
    // A message without components still says the stream has grain messages.
    EXPECT_EQ(film_grain_in_annex_b(annex_b({vvenc_mastering_display, vvenc_grain_without_components}), "vvc"),
              film_grain_h274);
    EXPECT_EQ(film_grain_in_annex_b(annex_b({vvenc_model_1_grain}), "vvc"), film_grain_h274_model_1);
    EXPECT_EQ(film_grain_in_annex_b(annex_b({vvc_sps, vvenc_mastering_display}), "vvc"), 0u);
}

// The mastering display message holds an emulation prevention byte (its
// 00 00 03): taken as part of its 24 bytes, the message after it would be
// misread.
TEST(FilmGrainDetect, MessagesAfterOneWithAnEmulationPreventionByteAreFound) {
    const std::string mastering_then_grain = "00b9891833c286c41d4c0bb884d03e803d1340420098968000000300011302008680";
    EXPECT_EQ(film_grain_in_annex_b(annex_b({mastering_then_grain}), "vvc"), film_grain_h274);
}

TEST(FilmGrainDetect, OnlyGrainThatIsNotCancelledInPrefixMessagesOfTheFirstLayerCounts) {
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"00b913018080"}), "vvc"), 0u);      // cancelled
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"00c11302008680"}), "vvc"), 0u);    // a suffix SEI
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"01b91302008680"}), "vvc"), 0u);    // layer 1
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"4e011302008680"}), "hevc"), film_grain_h274);
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"4e091302008680"}), "hevc"), 0u);   // layer 1
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"50011302008680"}), "hevc"), 0u);   // a suffix SEI
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"061302200080"}), "h264"), film_grain_h274_model_1);
    // H.264's SEI type in an HEVC stream is another NAL unit.
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"061302008680"}), "hevc"), 0u);
}

TEST(FilmGrainDetect, Afgs1IsAomsT35Message) {
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"00b90405b55890010080"}), "vvc"), film_grain_afgs1);
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"00b90405b55890020080"}), "vvc"), 0u);  // another AOM message
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"00b90406b5003c00010480"}), "vvc"), 0u);  // HDR10+
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"00b90405b558900100" "1302008680"}), "vvc"),
              film_grain_afgs1 | film_grain_h274);
}

TEST(FilmGrainDetect, LongTypesAndSizesAndTruncatedMessages) {
    // Type 300 (FF 2D), 256 bytes (FF 01), then a grain message.
    std::string long_message = "00b9ff2dff01";
    for (int i = 0; i < 256; ++i) {
        long_message += "11";
    }
    EXPECT_EQ(film_grain_in_annex_b(annex_b({long_message + "1302008680"}), "vvc"), film_grain_h274);
    // A size past the end ends the scan of that NAL unit, finding nothing in it.
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"00b913100086"}), "vvc"), 0u);
    EXPECT_EQ(film_grain_in_annex_b(annex_b({"00b9ff"}), "vvc"), 0u);
    EXPECT_EQ(film_grain_in_annex_b(std::string(), "vvc"), 0u);
}

// An AV1 sequence header's bits, written field by field.
class Bits {
public:
    Bits& put(std::uint32_t value, unsigned count) {
        for (unsigned i = count; i-- > 0;) {
            bits_.push_back(((value >> i) & 1) != 0);
        }
        return *this;
    }
    // trailing_bits(): a one, then zeros to the byte's end.
    std::string bytes_with_trailing_bits() const {
        std::vector<bool> bits = bits_;
        bits.push_back(true);
        while (bits.size() % 8 != 0) {
            bits.push_back(false);
        }
        std::string out;
        for (std::size_t at = 0; at < bits.size(); at += 8) {
            unsigned byte = 0;
            for (std::size_t i = 0; i < 8; ++i) {
                byte = (byte << 1) | (bits[at + i] ? 1u : 0u);
            }
            out.push_back(static_cast<char>(byte));
        }
        return out;
    }

private:
    std::vector<bool> bits_;
};

// A sequence header OBU with a size field.
std::string sequence_header_obu(const std::string& payload) {
    return bytes("0a") + static_cast<char>(payload.size()) + payload;
}

// The two AV1 clips' sequence headers: 1080p 10-bit and 8-bit, film grain
// present. Their last byte's 0x10 is film_grain_params_present (0x08 the
// trailing one), so a reading that strays by a bit finds the grain of a
// header with that bit cleared, or none in the clip's. Before them a
// temporal delimiter, after them the start of a frame.
TEST(FilmGrainDetect, Av1ClipsSequenceHeadersHaveGrain) {
    for (const char* header : {"02000042955dfe1b895f2818", "02000042955dfe1b895f2058"}) {
        const std::string with_grain = bytes(header);
        std::string without = with_grain;
        without.back() = static_cast<char>(static_cast<unsigned char>(without.back()) & ~0x10u);
        const std::string frame = bytes("32f9af011000");
        EXPECT_EQ(film_grain_in_av1(bytes("1200") + sequence_header_obu(with_grain) + frame), film_grain_av1);
        EXPECT_EQ(film_grain_in_av1(bytes("1200") + sequence_header_obu(without) + frame), 0u);
    }
}

// Headers made to take the branches the clips do not: a reduced still
// picture header; timing, decoder model and display delay info with two
// operating points (12-bit 4:2:0 with a chroma position); monochrome; sRGB.
TEST(FilmGrainDetect, Av1SequenceHeaderBranches) {
    const auto reduced = [](unsigned grain) {
        return Bits()
            .put(0, 3).put(1, 1).put(1, 1).put(8, 5)            // profile, still, reduced, seq_level_idx
            .put(10, 4).put(10, 4).put(1919, 11).put(1079, 11)  // frame size bits and maximum
            .put(0b011, 3)                                      // superblock, filter intra, intra edge
            .put(0b011, 3)                                      // superres, cdef, restoration
            .put(0, 1).put(0, 1).put(0, 1).put(0, 1)            // high bitdepth, mono, no description, range
            .put(0, 2).put(0, 1)                                // chroma position, separate uv delta q
            .put(grain, 1).bytes_with_trailing_bits();
    };
    const auto timed = [](unsigned grain) {
        Bits bits;
        bits.put(2, 3).put(0, 1).put(0, 1)                // profile 2, not still, not reduced
            .put(1, 1).put(1001, 32).put(24000, 32)       // timing info
            .put(1, 1).put(0b011, 3)                      // equal picture interval, uvlc(2)
            .put(1, 1).put(9, 5).put(1, 32).put(4, 5).put(4, 5)  // decoder model info: delays of 10 bits
            .put(1, 1).put(1, 5)                          // initial display delay, 2 operating points
            .put(0x101, 12).put(9, 5).put(1, 1)           // point 0: idc, level 9, tier
            .put(1, 1).put(5, 10).put(6, 10).put(1, 1)    //   decoder model: buffer delays, low delay
            .put(1, 1).put(3, 4)                          //   initial display delay
            .put(0x100, 12).put(4, 5).put(0, 1).put(0, 1)  // point 1: no tier, no model, no delay
            .put(11, 4).put(11, 4).put(3839, 12).put(2159, 12)
            .put(1, 1).put(5, 4).put(2, 3)                // frame id numbers
            .put(0b101, 3)
            .put(0b1101, 4).put(1, 1).put(0b11, 2)        // compound and filter tools, order hint, jnt, mvs
            .put(0, 1).put(1, 1).put(0, 1).put(1, 1)      // screen content tools forced on, integer mv forced
            .put(6, 3)                                    // order hint bits
            .put(0b110, 3)
            .put(1, 1).put(1, 1).put(0, 1)                // high bitdepth, twelve bit, not mono
            .put(1, 1).put(9, 8).put(16, 8).put(9, 8)     // BT.2020 PQ
            .put(0, 1).put(1, 1).put(1, 1).put(2, 2)      // range, 4:2:0 subsampling, chroma position
            .put(1, 1)                                    // separate uv delta q
            .put(grain, 1);
        return bits.bytes_with_trailing_bits();
    };
    const auto mono = [](unsigned grain) {
        return Bits()
            .put(0, 3).put(0, 1).put(0, 1).put(0, 1).put(0, 1).put(0, 5)  // no timing, no delay, 1 point
            .put(0, 12).put(8, 5).put(0, 1)                                // level 8: a tier
            .put(10, 4).put(10, 4).put(1919, 11).put(1079, 11)
            .put(0, 1).put(0b011, 3).put(0, 4).put(0, 1)  // no frame ids; tools; no order hint
            .put(1, 1).put(1, 1)                          // screen content tools chosen, integer mv chosen
            .put(0, 3)
            .put(0, 1).put(1, 1).put(0, 1).put(1, 1)      // 8 bits, mono, no description, range
            .put(grain, 1).bytes_with_trailing_bits();
    };
    const auto srgb = [](unsigned grain) {
        return Bits()
            .put(1, 3).put(0, 1).put(0, 1).put(0, 1).put(0, 1).put(0, 5)  // profile 1 (4:4:4)
            .put(0, 12).put(4, 5)
            .put(10, 4).put(10, 4).put(1919, 11).put(1079, 11)
            .put(0, 1).put(0, 3).put(0, 4).put(0, 1)
            .put(0, 1).put(0, 1)                          // screen content tools off: no integer mv
            .put(0, 3)
            .put(0, 1).put(1, 1).put(1, 8).put(13, 8).put(0, 8)  // no mono bit; sRGB: nothing more
            .put(0, 1)                                    // separate uv delta q
            .put(grain, 1).bytes_with_trailing_bits();
    };
    for (const auto& header : {+reduced, +timed, +mono, +srgb}) {
        EXPECT_EQ(film_grain_in_av1(sequence_header_obu(header(1))), film_grain_av1);
        EXPECT_EQ(film_grain_in_av1(sequence_header_obu(header(0))), 0u);
    }
    // An OBU with an extension byte and no size field: the rest is its payload.
    EXPECT_EQ(film_grain_in_av1(bytes("0c00") + mono(1)), film_grain_av1);
}

TEST(FilmGrainDetect, Av1WithoutAWholeSequenceHeaderIsUnknown) {
    EXPECT_EQ(film_grain_in_av1(bytes("1200") + bytes("32021000")), std::nullopt);     // no header
    EXPECT_EQ(film_grain_in_av1(sequence_header_obu(bytes("020000"))), std::nullopt);  // cut short
    EXPECT_EQ(film_grain_in_av1(bytes("0a20020000")), std::nullopt);                   // size past the end
    EXPECT_EQ(film_grain_in_av1(std::string()), std::nullopt);
}

TEST(FilmGrainDetect, TheScanCopiesTheFirstFramesAsTheirStream) {
    VideoInfo info;
    info.path = "clip.mkv";
    info.codec_name = "hevc";
    info.format_name = "matroska,webm";
    const std::vector<std::string> mkv = film_grain_scan_command("ffmpeg", info);
    const std::vector<std::string> expected{"ffmpeg", "-nostdin", "-v", "error", "-i", "clip.mkv", "-map", "0:V:0",
                                            "-c:v", "copy", "-frames:v", "3", "-bsf:v", "hevc_mp4toannexb",
                                            "-f", "data", "pipe:1"};
    EXPECT_EQ(mkv, expected);
    info.codec_name = "vvc";
    info.format_name = "vvc";  // a raw stream has its start codes
    const std::vector<std::string> raw = film_grain_scan_command("ffmpeg", info);
    EXPECT_EQ(std::count(raw.begin(), raw.end(), "-bsf:v"), 0);
    info.codec_name = "av1";
    info.format_name = "matroska,webm";
    const std::vector<std::string> av1 = film_grain_scan_command("ffmpeg", info);
    EXPECT_EQ(std::count(av1.begin(), av1.end(), "-bsf:v"), 0);
}

TEST(FilmGrainDetect, OtherCodecsHaveNoneAndAMissingFfmpegLeavesItUnknown) {
    VideoInfo info;
    info.path = "clip.webm";
    info.codec_name = "vp9";
    EXPECT_EQ(detect_film_grain(info, "", CancelToken()), 0u);  // not run at all
    info.codec_name = "h264";
    EXPECT_EQ(detect_film_grain(info, "no-such-ffmpeg-for-the-test", CancelToken()), std::nullopt);
}

// Run files and saved scores keep the kinds by name, only where known.
TEST(FilmGrainDetect, RunFilesKeepTheFilmGrainOnlyWhereItWasRead) {
    VideoInfo info;
    info.path = "clip.mkv";
    info.codec_name = "vvc";
    EXPECT_FALSE(video_json(info).contains("film_grain"));
    EXPECT_FALSE(video_info_to_dict(info).contains("film_grain"));
    EXPECT_EQ(video_from(video_json(info)).film_grain, std::nullopt);
    for (const unsigned kinds : {0u, unsigned(film_grain_h274_model_1 | film_grain_afgs1)}) {
        info.film_grain = kinds;
        EXPECT_EQ(video_from(video_json(info)), info);
        EXPECT_EQ(video_info_from_dict(nlohmann::json(video_info_to_dict(info))).film_grain, kinds);
    }
    EXPECT_EQ(film_grain_names(film_grain_h274 | film_grain_av1), (std::vector<std::string>{"h274", "av1"}));
    EXPECT_EQ(film_grain_from_names({"h274 model 1", "a later kind"}), film_grain_h274_model_1);
}

}  // namespace
}  // namespace vml
