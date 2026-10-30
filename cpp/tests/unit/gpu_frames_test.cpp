// The app's own decoding of videos (gpu_frames.h) held to the Python app's
// (decoding.json, from gpu_frames.py and vmaf_runner.py): the plans, FFmpeg's
// command and the choice of decoders on every system; and, where the PC has
// a decoder, its pictures against FFmpeg's decoding, and VMAF on the GPU
// from them against VMAF from FFmpeg's frames.
#include "vml/gpu_frames.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/film_grain_detect.h"
#include "vml/ffprobe.h"
#include "vml/gpu_vmaf.h"
#include "vml/model_select.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/vmaf_v1_gpu.h"

namespace vml {
namespace {

using Json = nlohmann::json;

// VML_SOFTWARE_DECODE as a test sets it, put back as it was.
class SoftwareDecode : public test::ScopedVariable {
public:
    explicit SoftwareDecode(const std::string& value)
        : test::ScopedVariable(std::string(software_decode_variable), value) {}
};

TEST(GpuFrames, PlansAsThePythonAppDoes) {
    const Json reference = test::load_reference("decoding.json");
    ASSERT_FALSE(reference["plans"].empty());
    for (const Json& item : reference["plans"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        VideoInfo info;
        info.path = "video.mkv";
        info.codec_name = item["codec_name"].get<std::string>();
        info.pix_fmt = item["pix_fmt"].get<std::string>();
        info.width = item["width"].get<int>();
        info.height = item["height"].get<int>();
        std::optional<CropBox> crop;
        if (!item["crop"].is_null()) {
            const std::vector<int> box = item["crop"].get<std::vector<int>>();
            crop = CropBox{box[0], box[1], box[2], box[3]};
        }
        std::optional<std::pair<int, int>> size;
        if (!item["size"].is_null()) {
            size = std::pair(item["size"][0].get<int>(), item["size"][1].get<int>());
        }
        const auto plan = [&] {
            return plan_decode(info, crop, item["shift"].get<int>(), item["luma_only"].get<bool>(), size,
                               item["algorithm"].get<std::string>());
        };
        if (item.contains("error")) {
            try {
                plan();
                ADD_FAILURE() << "planned to decode it in the app";
            } catch (const GpuDecodeUnavailableError& error) {
                EXPECT_EQ(error.what(), item["error"].get<std::string>());
            }
            continue;
        }
        const DecodePlan made = plan();
        const Json& expected = item["plan"];
        EXPECT_EQ(made.codec, expected["codec"].get<std::string>());
        EXPECT_EQ(made.bit_depth, expected["bit_depth"].get<int>());
        EXPECT_EQ(made.width, expected["width"].get<int>());
        EXPECT_EQ(made.height, expected["height"].get<int>());
        EXPECT_EQ((std::vector<int>{made.crop_x, made.crop_y, made.crop_w, made.crop_h}),
                  expected["crop"].get<std::vector<int>>());
        EXPECT_EQ(made.shift, expected["shift"].get<int>());
        EXPECT_EQ(made.luma_only, expected["luma_only"].get<bool>());
        EXPECT_EQ((std::vector<int>{made.out_w, made.out_h}), expected["out"].get<std::vector<int>>());
        EXPECT_EQ(made.scaler, expected["scaler"].get<std::string>());
        EXPECT_EQ(made.scaled(), expected["scaled"].get<bool>());
        EXPECT_EQ(made.frame_bytes(), expected["frame_bytes"].get<std::size_t>());
    }
}

// The pictures read where the software decoder holds them are those its
// download would only copy, and its planes are those of the packed frame it
// would download.
TEST(GpuFrames, OnlyTheSoftwareDecoderLendsThePicturesItWouldOnlyCopy) {
    DecodePlan plan;
    plan.codec = "vvc";
    plan.bit_depth = 10;
    plan.width = plan.crop_w = 1919;
    plan.height = plan.crop_h = 1079;
    plan.shift = 6;
    EXPECT_TRUE(can_lend_planes(plan, software_decoder));
    EXPECT_EQ(PackedFrame(plan.crop_w, plan.crop_h, plan.bit_depth).bytes(), plan.frame_bytes());
    for (const std::string_view backend : {"nvidia", "intel", "amd"}) {
        EXPECT_FALSE(can_lend_planes(plan, backend)) << backend;
    }
    DecodePlan moved = plan;  // samples moved to the top bits (P016's layout)
    moved.shift = 0;
    EXPECT_FALSE(can_lend_planes(moved, software_decoder));
    DecodePlan scaled = plan;
    scaled.out_w = 1280;
    scaled.out_h = 720;
    EXPECT_FALSE(can_lend_planes(scaled, software_decoder));
    DecodePlan eight = plan;
    eight.bit_depth = 8;
    eight.shift = 0;
    EXPECT_TRUE(can_lend_planes(eight, software_decoder));
    eight.widen = widen_shift;
    EXPECT_FALSE(can_lend_planes(eight, software_decoder));
}

TEST(GpuFrames, CopiesThePacketsWithThePythonAppsCommand) {
    const Json reference = test::load_reference("decoding.json");
    ASSERT_FALSE(reference["packet_commands"].empty());
    for (const auto& [codec, expected] : reference["packet_commands"].items()) {
        SCOPED_TRACE(codec);
        std::vector<std::string> command = packet_reader_command("<ffmpeg>", "video.mkv", codec, "<listing>");
        for (std::string& part : command) {
            if (part == input_path("video.mkv")) {
                part = "<video>";
            }
        }
        EXPECT_EQ(command, expected.get<std::vector<std::string>>());
    }
}

// An HEVC packet in Annex B: each unit (type, layer) after a 00 00 01 start
// code, with its two header bytes and a byte of payload.
std::vector<std::uint8_t> hevc_packet(const std::vector<std::pair<int, int>>& units) {
    std::vector<std::uint8_t> packet;
    for (const auto& [type, layer] : units) {
        packet.insert(packet.end(), {0, 0, 1, static_cast<std::uint8_t>((type << 1) | (layer >> 5)),
                                     static_cast<std::uint8_t>(((layer & 31) << 3) | 1), 0x80});
    }
    return packet;
}

constexpr int vps = 32, sps = 33, pps = 34, sei = 39, trail_r = 1, radl_n = 6, rasl_n = 8, rasl_r = 9, bla_w_lp = 16,
              idr_w_radl = 19, cra = 21;

// A stream copy cut at a CRA starts with the CRA's RASL pictures, which
// FFmpeg's decoding drops: they are passed over, up to the first picture
// that is neither RASL nor RADL, and only HEVC's.
TEST(GpuFrames, PassesOverTheRaslPicturesOfTheCraAVideoStartsWith) {
    using Verdict = LeadingPictures::Verdict;
    LeadingPictures cut("hevc");
    EXPECT_EQ(cut.take(hevc_packet({{vps, 0}, {sps, 0}, {pps, 0}, {sei, 0}, {cra, 0}})), Verdict::feed);
    EXPECT_EQ(cut.take(hevc_packet({{sei, 0}})), Verdict::feed);  // no picture: decides nothing
    EXPECT_EQ(cut.take(hevc_packet({{rasl_r, 0}})), Verdict::skip);
    EXPECT_EQ(cut.take(hevc_packet({{radl_n, 0}})), Verdict::feed);
    // A 4-byte start code (00 00 00 01) is a 00 00 01 one after a zero.
    std::vector<std::uint8_t> long_start = hevc_packet({{rasl_n, 0}});
    long_start.insert(long_start.begin(), 0);
    EXPECT_EQ(cut.take(long_start), Verdict::skip);
    // An enhancement layer's slice first: the base layer's picture decides.
    EXPECT_EQ(cut.take(hevc_packet({{rasl_n, 1}, {trail_r, 0}})), Verdict::feed);
    EXPECT_EQ(cut.take(hevc_packet({{rasl_n, 0}})), Verdict::feed);  // a later RASL picture is decoded
    EXPECT_EQ(cut.skipped(), 2);

    LeadingPictures bla("hevc");
    EXPECT_EQ(bla.take(hevc_packet({{bla_w_lp, 0}})), Verdict::feed);
    EXPECT_EQ(bla.take(hevc_packet({{rasl_n, 0}})), Verdict::skip);
    // One with parameter sets, which FFmpeg's decoder would read, is refused.
    EXPECT_EQ(bla.take(hevc_packet({{pps, 0}, {rasl_n, 0}})), Verdict::refuse);

    LeadingPictures idr("hevc");  // an IDR picture has no RASL pictures
    EXPECT_EQ(idr.take(hevc_packet({{idr_w_radl, 0}})), Verdict::feed);
    EXPECT_EQ(idr.take(hevc_packet({{rasl_n, 0}})), Verdict::feed);

    LeadingPictures h264("h264");  // other codecs' units are not HEVC's
    EXPECT_EQ(h264.take(hevc_packet({{cra, 0}})), Verdict::feed);
    EXPECT_EQ(h264.take(hevc_packet({{rasl_n, 0}})), Verdict::feed);
    EXPECT_EQ(h264.skipped(), 0);
}

// A VVC packet in Annex B: each unit (type, layer) after a 00 00 01 start
// code, with its two header bytes (layer first, then type) and a payload byte.
std::vector<std::uint8_t> vvc_packet(const std::vector<std::pair<int, int>>& units) {
    std::vector<std::uint8_t> packet;
    for (const auto& [type, layer] : units) {
        packet.insert(packet.end(), {0, 0, 1, static_cast<std::uint8_t>(layer & 0x3f),
                                     static_cast<std::uint8_t>((type << 3) | 1), 0x80});
    }
    return packet;
}

// A VVC stream started at a CRA picture (a seek into an open-GOP encode) has
// its RASL pictures given to VVdeC, which drops them as FFmpeg's decoding does,
// but may read their parameter sets: hidden, not passed over. One started at a
// GDR picture is refused; an IDR picture has no RASL pictures.
TEST(GpuFrames, HidesTheRaslPicturesOfTheCraAVvcStreamStartsWith) {
    using Verdict = LeadingPictures::Verdict;
    constexpr int vvc_sps = 15, vvc_aps = 17, vvc_sei = 23, vvc_radl = 2, vvc_rasl = 3, vvc_trail = 0, vvc_idr = 7,
                  vvc_cra = 9, vvc_gdr = 10;
    LeadingPictures seek("vvc");
    EXPECT_EQ(seek.take(vvc_packet({{vvc_sps, 0}, {vvc_sei, 0}, {vvc_cra, 0}})), Verdict::feed);
    EXPECT_EQ(seek.take(vvc_packet({{vvc_aps, 0}, {vvc_rasl, 0}})), Verdict::hide);  // its APS read by VVdeC
    EXPECT_EQ(seek.take(vvc_packet({{vvc_radl, 0}})), Verdict::feed);
    EXPECT_EQ(seek.take(vvc_packet({{vvc_rasl, 0}})), Verdict::hide);
    EXPECT_EQ(seek.take(vvc_packet({{vvc_trail, 0}})), Verdict::feed);
    EXPECT_EQ(seek.take(vvc_packet({{vvc_rasl, 0}})), Verdict::feed);  // a later RASL picture is shown
    EXPECT_EQ(seek.skipped(), 2);
    LeadingPictures enhancement("vvc");  // the base layer's slice decides
    EXPECT_EQ(enhancement.take(vvc_packet({{vvc_cra, 0}})), Verdict::feed);
    EXPECT_EQ(enhancement.take(vvc_packet({{vvc_rasl, 1}, {vvc_trail, 0}})), Verdict::feed);
    LeadingPictures from_idr("vvc");
    EXPECT_EQ(from_idr.take(vvc_packet({{vvc_idr, 0}})), Verdict::feed);
    EXPECT_EQ(from_idr.take(vvc_packet({{vvc_rasl, 0}})), Verdict::feed);
    LeadingPictures from_gdr("vvc");
    EXPECT_EQ(from_gdr.take(vvc_packet({{vvc_gdr, 0}})), Verdict::refuse);
}

// FFmpeg's warnings that it changed timestamps as it copied the packets, but
// not framecrc's note of a packet without any, which the reader handles.
TEST(GpuFrames, TellsFfmpegsTimestampWarningsFromItsNoteOfNone) {
    EXPECT_TRUE(warns_of_changed_timestamps(
        "[out#1/framecrc @ 000001] Non-monotonic DTS; previous: 245, current: 0; changing to 246."));
    EXPECT_TRUE(warns_of_changed_timestamps("Invalid timestamps stream=0, pts=7, dts=9"));
    EXPECT_FALSE(warns_of_changed_timestamps(
        "[framecrc @ 000001] Timestamps are unset in a packet for stream 0. This is deprecated and will stop "
        "working in the future. Fix your code to set the timestamps properly"));
    EXPECT_FALSE(warns_of_changed_timestamps("Stream #0:0: Video: hevc (Main 10), yuv420p10le"));
}

// Each video by its GPU's decoder or the software decoder, as the Python app
// chooses. With the software decoder not bundled (or VML_SOFTWARE_DECODE
// set to ffmpeg) everywhere, and bundled where it is.
TEST(GpuFrames, DecodesTheVideosThePythonAppDecodesItself) {
    const Json reference = test::load_reference("decoding.json");
    int checked = 0;
    for (const Json& item : reference["decoders"]) {
        const SoftwareDecode software(item["software_decode"].get<std::string>());
        if (software_decoder_bundled() != item["bundled"].get<bool>()) {
            continue;  // the software decoder is not bundled for this system
        }
        const auto accel = [](const Json& value) {
            return value.is_null() ? std::nullopt : std::optional(value.get<std::string>());
        };
        const HardwareDecoding decoding{accel(item["source"]), accel(item["test"])};
        SCOPED_TRACE(decoding.describe() + ", " + item["software_decode"].get<std::string>());
        const std::optional<std::pair<std::string, std::string>> decoders = decoders_here(decoding);
        if (item["decoders"].is_null()) {
            EXPECT_FALSE(decoders.has_value());
        } else {
            ASSERT_TRUE(decoders.has_value());
            EXPECT_EQ((std::vector<std::string>{decoders->first, decoders->second}),
                      item["decoders"].get<std::vector<std::string>>());
        }
        ++checked;
    }
    EXPECT_GE(checked, 20);  // at least the cases without the software decoder
}

// A video scored with H.274's film grain, which only the software decoder
// adds as it should, is decoded by it alone, which then also decodes it as
// a raw stream and scales it: FFmpeg could not decode it instead. Every VVC
// video's raw stream is decoded by VVdeC too; another codec's is left to
// FFmpeg, and so is scaling.
TEST(GpuFrames, OnlyTheSoftwareDecoderAddsH274GrainAndThenDecodesRawStreamsAndScales) {
    VideoInfo info;
    info.path = "encode.266";
    info.format_name = "vvc";
    info.codec_name = "vvc";
    info.pix_fmt = "yuv420p10le";
    info.width = 1920;
    info.height = 1080;
    info.film_grain = film_grain_h274;
    const DecodePlan raw = plan_decode(info, std::nullopt);
    EXPECT_TRUE(raw.software_only);
    EXPECT_TRUE(raw.untimed);
    EXPECT_EQ(raw.film_grain, 1);
    info.with_film_grain = false;  // VVdeC decodes the raw stream still, the grain left out
    const DecodePlan raw_without = plan_decode(info, std::nullopt);
    EXPECT_FALSE(raw_without.software_only);
    EXPECT_TRUE(raw_without.untimed);
    EXPECT_EQ(raw_without.film_grain, 0);
    info.format_name = "matroska,webm";
    const std::string not_scaled = "the software decoder does not scale; FFmpeg scales it";
    const DecodePlan without = plan_decode(info, std::nullopt, 0, false, std::pair(1280, 720));
    EXPECT_FALSE(without.software_only || without.untimed);
    // Scaled by FFmpeg as before (VVdeC's pictures fed to it: grain_feed.h).
    EXPECT_EQ(decoder_supports(0, without, software_decoder).second, not_scaled);
    info.with_film_grain = true;
    EXPECT_NE(decoder_supports(0, plan_decode(info, std::nullopt, 0, false, std::pair(1280, 720)), software_decoder)
                  .second,
              not_scaled);
    // HEVC: its raw stream decoded and its scaling made by FFmpeg.
    VideoInfo hevc = info;
    hevc.codec_name = "hevc";
    hevc.film_grain.reset();
    hevc.with_film_grain.reset();
    EXPECT_EQ(decoder_supports(0, plan_decode(hevc, std::nullopt, 0, false, std::pair(1280, 720)), software_decoder)
                  .second,
              not_scaled);
    hevc.path = "encode.hevc";
    hevc.format_name = "hevc";
    EXPECT_THROW(plan_decode(hevc, std::nullopt), GpuDecodeUnavailableError);
}

// The decoder's pictures must match the packets fed: in timestamp order,
// each the next packet's, and one for every packet by the end. Those an MP4
// edit list cuts off are dropped.
TEST(GpuFrames, PicturesThatDoNotMatchThePacketsFailTheRun) {
    PictureChecks checks("the GPU's decoder");
    for (const long long pts : {0, 3000, 1000, 2000}) {  // in decode order
        checks.fed(pts, pts == 0);
    }
    EXPECT_TRUE(checks.take(0));  // discarded
    EXPECT_FALSE(checks.take(1000));
    EXPECT_FALSE(checks.take(2000));
    const auto message = [](const auto& call) {
        try {
            call();
        } catch (const GpuDecodeFailedError& error) {
            return std::string(error.what());
        }
        return std::string();
    };
    EXPECT_EQ(message([&] { checks.check_end(); }), "the GPU's decoder gave 3 pictures for 4 packets");
    checks.take(3000);
    EXPECT_EQ(message([&] { checks.check_end(); }), "");
    EXPECT_EQ(message([&] { checks.take(2500); }), "the GPU's decoder gave pictures out of order");

    PictureChecks skipped("the software decoder");
    for (const long long pts : {0, 1000, 2000}) {
        skipped.fed(pts, false);
    }
    skipped.take(0);
    EXPECT_EQ(message([&] { skipped.take(2000); }), "the software decoder gave 1 pictures for 2 packets");
    PictureChecks made_up("the GPU's decoder");
    made_up.fed(1000, false);
    EXPECT_EQ(message([&] { made_up.take(500); }), "the GPU's decoder gave pictures the packets do not have");
}

}  // namespace
}  // namespace vml
