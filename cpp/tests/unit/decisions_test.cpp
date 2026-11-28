// The run's pure decisions, held to the Python app's own (decisions.json):
// pixel formats and GPU decoding, frame-rate fractions and frame counts,
// the filters that pair frames, why two videos cannot be compared, the
// shapes they are compared at, and the VMAF model.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "reference.h"
#include "vml/crop_detect.h"
#include "vml/ffmpeg_run.h"
#include "vml/film_grain_detect.h"
#include "vml/frame_sync.h"
#include "vml/colour.h"
#include "vml/geometry.h"
#include "vml/number_format.h"
#include "vml/gpu_frames.h"
#include "vml/hardware_decoding.h"
#include "vml/model_select.h"
#include "vml/perceptual.h"
#include "vml/perceptual_cpu.h"
#include "vml/platform/system.h"
#include "vml/vship.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef near  // a 16-bit pointer word, which a test below uses as a name
#endif

namespace vml {
namespace {

using Json = nlohmann::json;

const Json& decisions() {
    static const Json reference = test::load_reference("decisions.json");
    return reference;
}

GpuVendor vendor_named(const std::string& name) {
    if (name == "nvidia") {
        return GpuVendor::nvidia;
    }
    if (name == "intel") {
        return GpuVendor::intel;
    }
    if (name == "amd") {
        return GpuVendor::amd;
    }
    return name == "none" ? GpuVendor::none : GpuVendor::automatic;
}

TEST(Decisions, PixelFormatsAreReadAsThePythonAppReadsThem) {
    for (const Json& item : decisions()["pixel_formats"]) {
        const std::string format = item["format"].get<std::string>();
        EXPECT_EQ(bit_depth(format), item["bit_depth"].get<int>()) << format;
        EXPECT_EQ(hw_native_format(format), item["hw_native"].get<std::string>()) << format;
        EXPECT_EQ(downloads_from_gpu(format, 1920, 1080), item["downloads"][0].get<bool>()) << format;
        EXPECT_EQ(downloads_from_gpu(format, 853, 480), item["downloads"][1].get<bool>()) << format;
        EXPECT_EQ(downloads_from_gpu(format, 0, 0), item["downloads"][2].get<bool>()) << format;
    }
    for (const Json& item : decisions()["analysis"]) {
        EXPECT_EQ(analysis_pix_fmt(item["formats"].get<std::vector<std::string>>()), item["format"].get<std::string>());
    }
}

// As the Python app chose, but for MPEG-2, which is decoded in software now
// (gpu_decodes_exactly). Its NVIDIA GPUs took 10-bit H.264, as the RTX 50
// series' decoder does (H264In444ProfilesIsGpuDecodedOnlyWhereItIsExact has
// the earlier ones).
TEST(Decisions, GpuDecodingIsChosenAsThePythonAppChoosesIt) {
    ASSERT_FALSE(decisions()["decode"].empty());
    for (const Json& item : decisions()["decode"]) {
        DecodingCapabilities capabilities;
        capabilities.nvidia_decodes_10bit_h264 = true;
        for (const Json& hwaccel : item["hwaccels"]) {
            capabilities.hwaccels.insert(hwaccel.get<std::string>());
        }
        for (const Json& vendor : item["vendors"]) {
            capabilities.vendors.push_back(vendor_named(vendor.get<std::string>()));
        }
        const std::optional<std::string> chosen =
            pick_decode(capabilities, vendor_named(item["vendor"].get<std::string>()), item["codec"].get<std::string>(),
                        item["pix_fmt"].get<std::string>(), item["size"][0].get<int>(), item["size"][1].get<int>());
        const std::optional<std::string> expected =
            item["hwaccel"].is_null() ? std::nullopt : std::optional<std::string>(item["hwaccel"].get<std::string>());
        EXPECT_EQ(chosen, expected) << item.dump();
    }
    for (const Json& item : decisions()["hwaccel_args"]) {
        const std::optional<std::string> hwaccel =
            item["hwaccel"].is_null() ? std::nullopt : std::optional<std::string>(item["hwaccel"].get<std::string>());
        EXPECT_EQ(hwaccel_args(hwaccel), item["args"].get<std::vector<std::string>>()) << item.dump();
    }
}

// A Mac's GPU decodes a video for a caller that allows it (the Vship pass
// every video; by default, as the libvmaf routes, only one deeper than 8
// bits; not playback's FFmpeg fallback) where FFmpeg has VideoToolbox and
// the GPU decodes the codec, in the 4:2:0 8- or 10-bit pictures at an even
// size the download takes; not otherwise.
// A resolution test's size from a width or a height typed: even, the picture's
// shape, within the widths the reference has, and none on a reference too small
// or too flat. A listed width keeps its name, and so its saved scores.
TEST(Decisions, AResolutionTestsSizeFollowsAWidthOrHeightTyped) {
    const Size film(3840, 2160);
    EXPECT_EQ(resample_size_for_height(720, film), Size(1280, 720));
    EXPECT_EQ(resample_size_for_width(1921, film), Size(1920, 1080));
    EXPECT_EQ(resample_size_for_width(99999, film), Size(3838, 2158));

    const Size portrait(1080, 1920);
    EXPECT_EQ(resample_size_for_height(1280, portrait), Size(720, 1280));
    EXPECT_EQ(resample_size_for_width(1, portrait), Size(16, 28));

    const Size scope(4096, 1712);
    EXPECT_EQ(resample_smallest_width(scope), 38);
    EXPECT_EQ(resample_size_for_width(1, scope), Size(38, 16));
    EXPECT_EQ(resample_size_for_height(418, scope), Size(1000, 418));

    EXPECT_GT(resample_smallest_width({17, 10}), resample_largest_width({17, 10}));
    EXPECT_GT(resample_smallest_width({300, 10}), resample_largest_width({300, 10}));

    EXPECT_EQ(resample_target_of_width(1920).label, "1080p");
    EXPECT_EQ(resample_target_of_width(1002).label, "1002w");
}

// Python's round() from the number's exact binary value: 2.5 is a true half, to
// even; 2.675 is just under its half.
TEST(Decisions, ABundledModelIsShownByItsVersion) {
    // VMAF v1's model is a file of the install's: shown as the version it
    // is, as libvmaf's own models are, whatever the path or its separators.
    EXPECT_EQ(model_shown("path=C:/Apps/VideoMetricsLab/resources/models/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json"),
              "version=vmaf_v1.0.16_3d0h");
    EXPECT_EQ(model_shown("path=C:\\Apps\\models\\vmaf_v1.0.16_hfr\\vmaf_v1.0.16_hfr_3d0h_2160.json"),
              "version=vmaf_v1.0.16_hfr_3d0h_2160");
    EXPECT_EQ(model_shown("version=vmaf_v0.6.1neg"), "version=vmaf_v0.6.1neg");
    // A model file of one's own keeps its path: it is what names it.
    EXPECT_EQ(model_shown("path=Y:/models/mine.json"), "path=Y:/models/mine.json");
    EXPECT_EQ(model_shown(""), "");
}

TEST(Decisions, RoundingIsPythons) {
    EXPECT_EQ(python_round(2.5, 0), 2.0);
    EXPECT_EQ(python_round(3.5, 0), 4.0);
    EXPECT_EQ(python_round(2.675, 2), 2.67);
}

// The names of full range are one test; an untagged video's rule differs by where
// it is read: for a video's own picture "unknown" counts as untagged.
TEST(Decisions, FullRangeByNameAndUntagged) {
    EXPECT_TRUE(is_full_range_name("jpeg"));
    EXPECT_FALSE(is_full_range_name("unknown"));
    VideoInfo video;
    video.pix_fmt = "yuvj420p";
    video.color_range = "unknown";
    EXPECT_TRUE(video.full_range());
    video.pix_fmt = "yuv420p";
    EXPECT_FALSE(video.full_range());
    video.color_range = "pc";
    EXPECT_TRUE(video.full_range());
}

TEST(Decisions, AMacsGpuDecodesWhereTheCallerAllowsIt) {
    DecodingCapabilities mac;
    mac.hwaccels = {"videotoolbox"};
    mac.videotoolbox_codecs = {"h264", "hevc", "mpeg2video"};
    const auto pick = [&mac](const std::string& codec, const std::string& pix_fmt, int width, MacGpuDecoding allowed,
                             GpuVendor vendor = GpuVendor::automatic) {
        return pick_decode(mac, vendor, codec, pix_fmt, width, 1080, allowed);
    };
    EXPECT_EQ(pick("hevc", "yuv420p10le", 1920, MacGpuDecoding::on), "videotoolbox");
    EXPECT_EQ(pick("h264", "yuv420p", 1920, MacGpuDecoding::on), "videotoolbox");
    EXPECT_EQ(pick_decode(mac, GpuVendor::automatic, "hevc", "yuv420p10le", 1920, 1080), "videotoolbox");  // by default
    EXPECT_EQ(pick_decode(mac, GpuVendor::automatic, "hevc", "yuv420p", 1920, 1080), std::nullopt);  // 8-bit: not
    EXPECT_EQ(pick("h264", "yuv420p", 1920, MacGpuDecoding::deep), std::nullopt);
    EXPECT_EQ(pick("hevc", "yuv420p", 1920, MacGpuDecoding::off), std::nullopt);
    EXPECT_EQ(pick("hevc", "yuv420p", 1920, MacGpuDecoding::on, GpuVendor::none), std::nullopt);
    EXPECT_EQ(pick("vp9", "yuv420p", 1920, MacGpuDecoding::on), std::nullopt);   // not this GPU's
    EXPECT_EQ(pick("ffv1", "yuv420p", 1920, MacGpuDecoding::on), std::nullopt);
    EXPECT_EQ(pick("mpeg2video", "yuv420p", 1920, MacGpuDecoding::on), std::nullopt);  // gpu_decodes_exactly
    EXPECT_EQ(pick("h264", "yuv420p10le", 1920, MacGpuDecoding::on), std::nullopt);  // 10-bit H.264
    EXPECT_EQ(pick("hevc", "yuv444p", 1920, MacGpuDecoding::on), std::nullopt);
    EXPECT_EQ(pick("hevc", "yuv420p", 1919, MacGpuDecoding::on), std::nullopt);
    mac.hwaccels.clear();  // an FFmpeg without VideoToolbox
    EXPECT_EQ(pick("hevc", "yuv420p", 1920, MacGpuDecoding::on), std::nullopt);
    EXPECT_EQ(hwaccel_args("videotoolbox"),
              (std::vector<std::string>{"-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"}));
}

// D3D11VA, AMD's route, decodes a video only where the GPU offers a decoder
// profile for its codec, as a Radeon 8060S offers none for VC-1: FFmpeg's
// D3D11VA decoding would fail to start, and the run would retry it in
// software. Where the profiles are not known, every codec D3D11VA decodes.
TEST(Decisions, D3d11vaDecodesOnlyTheCodecsTheGpuHasADecoderFor) {
    DecodingCapabilities pc;
    pc.hwaccels = {"d3d11va"};
    pc.vendors = {GpuVendor::amd};
    EXPECT_EQ(pick_decode(pc, GpuVendor::amd, "vc1", "yuv420p", 1920, 1080), "d3d11va");  // not known
    pc.d3d11va_codecs = std::set<std::string>{"h264", "hevc", "vp9", "av1"};
    EXPECT_EQ(pick_decode(pc, GpuVendor::amd, "vc1", "yuv420p", 1920, 1080), std::nullopt);
    EXPECT_EQ(pick_decode(pc, GpuVendor::automatic, "vc1", "yuv420p", 1920, 1080), std::nullopt);
    EXPECT_EQ(pick_decode(pc, GpuVendor::amd, "h264", "yuv420p", 1920, 1080), "d3d11va");
    EXPECT_EQ(pick_decode(pc, GpuVendor::automatic, "hevc", "yuv420p10le", 1920, 1080), "d3d11va");
}

// H.264 in a 4:4:4 profile (x264's lossless -qp 0 writes "High 4:4:4
// Predictive", in 4:2:0 too) is decoded on the GPU only by Apple's decoder and
// NVIDIA's of the RTX 50 series (the one that decodes 10-bit H.264), both
// checked bit-exact; Intel's and an RTX 3070's gave other pictures, and
// FFmpeg's D3D11VA refuses the profile. Other H.264 as before, and 10-bit
// lossless on the RTX 50 series' alone.
TEST(Decisions, H264In444ProfilesIsGpuDecodedOnlyWhereItIsExact) {
    DecodingCapabilities pc;
    pc.hwaccels = {"cuda", "qsv", "d3d11va"};
    pc.nvidia_decodes_10bit_h264 = true;
    VideoInfo video;
    video.codec_name = "h264";
    video.pix_fmt = "yuv420p";
    video.width = 1920;
    video.height = 1080;
    video.profile = "High 4:4:4 Predictive";
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), "cuda");
    for (const char* profile : {"High 4:4:4 Predictive", "High 4:4:4", "High 4:4:4 Intra", "CAVLC 4:4:4"}) {
        video.profile = profile;
        EXPECT_EQ(pick_decode(pc, GpuVendor::intel, video), std::nullopt) << profile;
        EXPECT_EQ(pick_decode(pc, GpuVendor::amd, video), std::nullopt) << profile;
    }
    EXPECT_EQ(software_decoding_reason(video, true),
              "only Apple's GPUs and NVIDIA's of the RTX 50 series decode H.264's CAVLC 4:4:4 profile exactly");
    video.profile = "High";
    EXPECT_EQ(pick_decode(pc, GpuVendor::intel, video), "qsv");
    EXPECT_EQ(pick_decode(pc, GpuVendor::amd, video), "d3d11va");
    video.profile.clear();  // not probed: taken as 4:4:4
    EXPECT_EQ(pick_decode(pc, GpuVendor::intel, video), std::nullopt);
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), "cuda");
    EXPECT_EQ(software_decoding_reason(video, true),
              "its H.264 profile is not known (a run saved before 2.3) and may be a lossless 4:4:4 one, which only "
              "Apple's GPUs and NVIDIA's of the RTX 50 series decode exactly");
    video.profile = "High 4:4:4 Predictive";
    video.pix_fmt = "yuv420p10le";
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), "cuda");
    EXPECT_EQ(software_decoding_reason(video, true),
              "only NVIDIA's GPUs (the RTX 50 series) decode H.264 deeper than 8 bits");
    // An NVIDIA decoder before the RTX 50 series (8-bit H.264 only), or one not
    // asked: lossless and 10-bit H.264 in software, other H.264 on it still.
    for (const std::optional<bool> known : {std::optional<bool>(false), std::optional<bool>()}) {
        pc.nvidia_decodes_10bit_h264 = known;
        video.pix_fmt = "yuv420p10le";
        EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), std::nullopt);
        video.pix_fmt = "yuv420p";
        EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), std::nullopt);
        video.profile.clear();
        EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), std::nullopt);
        video.profile = "High";
        EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), "cuda");
        video.profile = "High 4:4:4 Predictive";
    }
    DecodingCapabilities mac;
    mac.hwaccels = {"videotoolbox"};
    mac.videotoolbox_codecs = {"h264"};
    video.pix_fmt = "yuv420p";
    EXPECT_EQ(pick_decode(mac, GpuVendor::automatic, video, MacGpuDecoding::on), "videotoolbox");
}

// Monochrome H.264, reported as 4:2:0, is decoded in software on every
// maker's GPU: an RTX 5090 leaves its chroma unset.
TEST(Decisions, MonochromeH264IsDecodedInSoftware) {
    DecodingCapabilities pc;
    pc.hwaccels = {"cuda", "qsv", "d3d11va"};
    pc.nvidia_decodes_10bit_h264 = true;
    DecodingCapabilities mac;
    mac.hwaccels = {"videotoolbox"};
    mac.videotoolbox_codecs = {"h264"};
    VideoInfo video;
    video.codec_name = "h264";
    video.pix_fmt = "yuv420p";
    video.width = 1920;
    video.height = 1080;
    video.profile = "High";
    video.monochrome_h264 = true;
    for (const GpuVendor vendor : {GpuVendor::nvidia, GpuVendor::intel, GpuVendor::amd}) {
        EXPECT_EQ(pick_decode(pc, vendor, video), std::nullopt);
    }
    EXPECT_EQ(pick_decode(mac, GpuVendor::automatic, video, MacGpuDecoding::on), std::nullopt);
    EXPECT_EQ(software_decoding_reason(video, true),
              "it is monochrome H.264: GPU decoders are not known to give its grey chroma (an RTX 5090's leaves it "
              "unset)");
}

// A pass says why each video it decoded in software was: VVdeC's for VVC, a
// rule, the GPU's video decoding having failed in the run, or no decoder
// here; nothing for a video on the GPU, or for a pass without a plan.
TEST(Decisions, APassSaysWhyEachVideoItDecodedInSoftwareWas) {
    VideoInfo h264;
    h264.codec_name = "h264";
    h264.pix_fmt = "yuv420p";
    h264.width = 1920;
    h264.height = 1080;
    h264.profile = "High";
    VideoInfo vvc = h264;
    vvc.codec_name = "vvc";
    VideoInfo monochrome = h264;
    monochrome.monochrome_h264 = true;
    const HardwareDecoding software{};
    EXPECT_EQ(software_decoding_why(std::nullopt, h264, monochrome, true, std::nullopt), SoftwareDecodingWhy{});
    // A rule comes before the run's failure.
    EXPECT_EQ(software_decoding_why(HardwareDecoding{"cuda", std::nullopt}, h264, monochrome, true, "it hung"),
              (SoftwareDecodingWhy{"", software_decoding_reason(monochrome, true)}));
    EXPECT_EQ(software_decoding_why(software, vvc, h264, true, "it hung"),
              (SoftwareDecodingWhy{"VVdeC decodes it on the CPU",
                                   "the GPU's video decoding failed in this run (it hung)"}));
    EXPECT_EQ(software_decoding_why(software, h264, h264, true, std::nullopt).distorted,
              "no GPU decoder here for h264");
    EXPECT_EQ(software_decoding_why(software, h264, h264, false, "it hung"),
              (SoftwareDecodingWhy{"GPU decoding is off", "GPU decoding is off"}));
}

// A saved run's video without a profile takes the one its file is read with
// again, and nothing else; one with a profile, or another file or codec,
// keeps what it had.
TEST(Decisions, ASavedVideoTakesOnlyTheProfileReadFromItsFile) {
    VideoInfo saved;
    saved.path = "E:/videos/source.mkv";
    saved.codec_name = "h264";
    saved.width = 1920;
    VideoInfo read = saved;
    read.profile = "High";
    read.width = 3840;  // the file's details now: not taken
    read.start_offset = 0.5;
    EXPECT_TRUE(take_read_details(saved, read));
    EXPECT_EQ(saved.profile, "High");
    EXPECT_EQ(saved.start_offset, 0.5);
    EXPECT_EQ(saved.width, 1920);
    read.profile = "High 4:4:4 Predictive";
    EXPECT_FALSE(take_read_details(saved, read));  // it has one
    EXPECT_EQ(saved.profile, "High");

    saved.profile.clear();
    VideoInfo other = read;
    other.path = "E:/videos/test.mkv";
    EXPECT_FALSE(take_read_details(saved, other));
    other = read;
    other.codec_name = "hevc";
    EXPECT_FALSE(take_read_details(saved, other));
    EXPECT_EQ(saved.profile, "");

    // Monochrome, which a run saved before 2.5 does not say, is taken; a
    // reading that did not look (false) never clears it.
    read.profile.clear();
    read.monochrome_h264 = true;
    EXPECT_TRUE(take_read_details(saved, read));
    EXPECT_TRUE(saved.monochrome_h264);
    read.monochrome_h264 = false;
    EXPECT_FALSE(take_read_details(saved, read));
    EXPECT_TRUE(saved.monochrome_h264);
}

TEST(Decisions, FrameRatesBecomeThePythonAppsFractions) {
    for (const Json& item : decisions()["fractions"]) {
        const double rate = *test::reference_number(item["value"]);
        const Rational fraction = limit_denominator(rate);
        EXPECT_EQ(fraction.numerator, item["fraction"][0].get<long long>()) << item["value"];
        EXPECT_EQ(fraction.denominator, item["fraction"][1].get<long long>()) << item["value"];
    }
}

TEST(Decisions, AFrameLimitKeepsThePythonAppsFrames) {
    for (const Json& item : decisions()["within"]) {
        const std::optional<std::string> limit =
            item["limit"].is_null() ? std::nullopt : std::optional<std::string>(item["limit"].get<std::string>());
        const Rational rate{item["rate"][0].get<long long>(), item["rate"][1].get<long long>()};
        const std::optional<long long> frames = frames_within(limit, rate);
        EXPECT_EQ(frames.has_value(), !item["frames"].is_null()) << item.dump();
        if (frames) {
            EXPECT_EQ(*frames, item["frames"].get<long long>()) << item.dump();
        }
    }
}

TEST(Decisions, FramesArePairedAsThePythonAppPairsThem) {
    const Json& expected = decisions()["frame_filters"];
    const Rational rate{24000, 1001};
    EXPECT_EQ(framesync_opts(), expected["framesync"].get<std::vector<std::string>>());
    EXPECT_EQ(numbering_filters(rate), expected["numbering"].get<std::vector<std::string>>());
}

VideoInfo video_from(const Json& item) {
    VideoInfo info;
    info.path = platform::path_from_text(item["path"].get<std::string>());
    info.width = 1920;
    info.height = 1080;
    info.fps = *test::reference_number(item["fps"]);
    info.duration = *test::reference_number(item["duration"]);
    info.nominal_fps = *test::reference_number(item["nominal_fps"]);
    info.format_name = item["format_name"].get<std::string>();
    info.codec_name = item["codec_name"].get<std::string>();
    return info;
}

TEST(Decisions, TwoVideosAreRefusedAsThePythonAppRefusesThem) {
    for (const Json& item : decisions()["pairs"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        const std::optional<std::string> problem =
            pair_problem(video_from(item["source"]), video_from(item["test"]), 0, item["range_end"].get<long long>());
        EXPECT_EQ(problem.value_or("<none>"),
                  item["problem"].is_null() ? std::string("<none>") : item["problem"].get<std::string>());
    }
}

// A range must end after it starts and start within both videos, and needs
// their frame rates to count its frames; with no end, the videos must last
// as long.
TEST(Decisions, ARangeIsRefusedWhereItCannotBeCompared) {
    VideoInfo source;
    source.fps = 24.0;
    source.duration = 60.0;
    VideoInfo shorter = source;
    shorter.duration = 30.0;
    EXPECT_EQ(pair_problem(source, source, 20, 10), "The range ends before it starts.");
    EXPECT_EQ(pair_problem(source, shorter, 40, 50), "The range starts after the end of one of the videos.");
    EXPECT_EQ(pair_problem(source, shorter, 10, 20), std::nullopt);
    EXPECT_TRUE(pair_problem(source, shorter, 10, 0)->starts_with("Durations do not match"));
    VideoInfo no_rate = source;
    no_rate.fps = 0.0;
    EXPECT_EQ(pair_problem(no_rate, no_rate, 10, 20), "A range needs the videos' frame rates, which are not known.");
}

// A range of frames: it must end after it starts, start within both videos
// and end no later than the shorter one's last frame.
TEST(Decisions, ARangeOfFramesIsRefusedWhereItCannotBeCompared) {
    VideoInfo source;
    source.fps = 50.0;
    source.duration = 10.0;  // 500 frames
    VideoInfo shorter = source;
    shorter.duration = 4.0;  // 200 frames
    EXPECT_EQ(pair_problem(source, shorter, ComparedRange{100, 50, true}), "The range ends before it starts.");
    EXPECT_EQ(pair_problem(source, shorter, ComparedRange{200, 0, true}),
              "The range starts after the end of one of the videos.");
    EXPECT_EQ(pair_problem(source, shorter, ComparedRange{100, 201, true}),
              "The range extends beyond the end of one of the videos.");
    EXPECT_EQ(pair_problem(source, shorter, ComparedRange{100, 200, true}), std::nullopt);
    EXPECT_EQ(pair_problem(source, shorter, ComparedRange{0, 1, true}), std::nullopt);
    EXPECT_TRUE(pair_problem(source, shorter, ComparedRange{100, 0, true})->starts_with("Durations do not match"));
}

// Video Compare places a raw stream's frames by its rate: it refuses one
// whose rate is not known or differs from the reference's, but a raw stream
// shorter than the reference is shown, as any shorter pair is.
TEST(Decisions, VideoCompareRefusesARawStreamOnlyForItsFrameRate) {
    VideoInfo reference;
    reference.fps = reference.nominal_fps = 60.0;
    reference.duration = 10.0;
    VideoInfo raw;
    raw.format_name = "vvc";
    raw.nb_frames = 80;
    EXPECT_EQ(untimed_pair_problem(reference, raw), frame_rate_unknown_problem(raw));
    raw.set_frame_rate(60.0);
    EXPECT_EQ(untimed_pair_problem(reference, raw), std::nullopt);
    raw.set_frame_rate(50.0);
    EXPECT_TRUE(untimed_pair_problem(reference, raw)->starts_with("Frame rates do not match"));
}

// A range is counted in frames: one ending within the shorter video's last,
// partial second compares to its end (a 300-frame encode, 12.5 s at
// 24000/1001, to 13 s), whichever of the two is shorter, and one whose
// last second has none of its frames (288 frames last 12.012 s, none in
// second 12) is refused. Below 1 fps a second may have no frame at all.
TEST(Decisions, ARangeEndingInAVideosLastPartialSecondComparesToItsEnd) {
    VideoInfo source;
    source.fps = 24000.0 / 1001.0;
    source.duration = 60.0;
    VideoInfo encode = source;
    encode.duration = 300 * 1001.0 / 24000.0;
    EXPECT_EQ(pair_problem(source, encode, 0, 13), std::nullopt);
    EXPECT_EQ(pair_problem(source, encode, 12, 13), std::nullopt);
    EXPECT_EQ(pair_problem(encode, source, 0, 13), std::nullopt);
    EXPECT_EQ(pair_problem(source, encode, 0, 14), "The range extends beyond the end of one of the videos.");
    VideoInfo whole_seconds = source;
    whole_seconds.duration = 288 * 1001.0 / 24000.0;
    EXPECT_EQ(pair_problem(source, whole_seconds, 0, 12), std::nullopt);
    EXPECT_EQ(pair_problem(source, whole_seconds, 0, 13), "The range extends beyond the end of one of the videos.");
    EXPECT_EQ(pair_problem(source, whole_seconds, 12, 13), "The range starts after the end of one of the videos.");
    EXPECT_EQ(pair_problem(source, whole_seconds, 12, 0), "The range starts after the end of one of the videos.");
    VideoInfo slides;
    slides.fps = 0.5;
    slides.duration = 6.0;  // frames at 0, 2 and 4 s
    EXPECT_EQ(pair_problem(slides, slides, 1, 2), "The range contains no frame of the videos.");
    EXPECT_EQ(pair_problem(slides, slides, 1, 3), std::nullopt);
    // Milliseconds, as the frames in them: between two frames, none; past
    // the shorter video's frames, beyond its end (288 frames end at 12.012 s).
    EXPECT_EQ(pair_problem(slides, slides, ComparedRange{500, 1500, false, true}),
              "The range contains no frame of the videos.");
    EXPECT_EQ(pair_problem(slides, slides, ComparedRange{500, 2500, false, true}), std::nullopt);
    EXPECT_EQ(pair_problem(source, whole_seconds, ComparedRange{11500, 12011, false, true}), std::nullopt);
    EXPECT_EQ(pair_problem(source, whole_seconds, ComparedRange{11500, 12013, false, true}),
              "The range extends beyond the end of one of the videos.");
    EXPECT_EQ(pair_problem(source, whole_seconds, ComparedRange{2500, 1500, false, true}),
              "The range ends before it starts.");
}

TEST(Decisions, ShapesAndComparedSizesAreThePythonAppsOwn) {
    for (const Json& item : decisions()["shapes"]) {
        VideoInfo info;
        info.width = item["width"].get<int>();
        info.height = item["height"].get<int>();
        info.sar = item["sar"].get<std::string>();
        EXPECT_EQ(display_aspect_ratio(info), *test::reference_number(item["aspect"])) << item.dump();
    }
    for (const Json& item : decisions()["compared"]) {
        VideoInfo source;
        source.width = item["source"][0].get<int>();
        source.height = item["source"][1].get<int>();
        VideoInfo test;
        test.width = item["test"][0].get<int>();
        test.height = item["test"][1].get<int>();
        const ScaleDirection direction = item["direction"].get<std::string>() == "distorted_to_source"
                                             ? ScaleDirection::distorted_to_source
                                             : ScaleDirection::source_to_distorted;
        const Size size = compared_dimensions(source, test, direction);
        EXPECT_EQ(size.first, item["size"][0].get<int>()) << item.dump();
        EXPECT_EQ(size.second, item["size"][1].get<int>()) << item.dump();
    }
}

// The Python app's cases, but for the 853x480 source against its 1706x960 test video, whose test box is now the
// source's scaled exactly (AScaledTestVideosBoxLinesUpWithTheSourcesRowForRow).
TEST(Decisions, BlackBarBoxesCoverTheSamePicture) {
    ASSERT_FALSE(decisions()["common_picture"].empty());
    const auto box = [](const Json& values) {
        return values.is_null() ? std::nullopt
                                : std::optional<CropBox>(CropBox{values[0].get<int>(), values[1].get<int>(),
                                                                 values[2].get<int>(), values[3].get<int>()});
    };
    for (const Json& item : decisions()["common_picture"]) {
        VideoInfo source;
        source.width = item["source"][0].get<int>();
        source.height = item["source"][1].get<int>();
        VideoInfo test;
        test.width = item["test"][0].get<int>();
        test.height = item["test"][1].get<int>();
        const auto [shared_source, shared_test] =
            common_picture(source, test, box(item["source_box"]), box(item["test_box"]));
        EXPECT_EQ(shared_source, box(item["shared"][0])) << item.dump();
        EXPECT_EQ(shared_test, box(item["shared"][1])) << item.dump();
    }
}

// A crop entered by hand (Manual, --source-crop): read as top:bottom:left:right, cut as entered with nothing
// detected, refused where it cannot be cut, and given back from a detected box to start from.
TEST(Decisions, ACropEnteredByHandIsCutAsEntered) {
    EXPECT_EQ(parse_crop_edges("22:22:0:0"), (CropEdges{22, 22, 0, 0}));
    for (const char* wrong : {"22:22:0", "22:22:0:0:0", "a:b:c:d", "-2:0:0:0", ""}) {
        EXPECT_EQ(parse_crop_edges(wrong), std::nullopt) << wrong;
    }
    VideoInfo hd;
    hd.width = 1920;
    hd.height = 1080;
    EXPECT_EQ(box_from_edges({22, 22, 0, 0}, hd, "the source"), (CropBox{1920, 1036, 0, 22}));
    EXPECT_EQ(box_from_edges({}, hd, "the source"), std::nullopt);
    EXPECT_THROW(box_from_edges({21, 21, 0, 0}, hd, "the source"), CropEntryError);   // odd lines
    EXPECT_THROW(box_from_edges({540, 530, 0, 0}, hd, "the source"), CropEntryError);  // under 16 rows left
    // An odd-sized frame's box is even-sized, and its edges give it back
    VideoInfo odd;
    odd.width = 853;
    odd.height = 481;
    const std::optional<CropBox> box = box_from_edges({2, 2, 0, 0}, odd, "the source");
    EXPECT_EQ(box, (CropBox{852, 476, 0, 2}));
    EXPECT_EQ(box_from_edges(edges_of_box(*box, odd), odd, "the source"), box);
    // Nothing entered still leaves out its odd lines, as every other crop entered does
    EXPECT_EQ(box_from_edges({}, odd, "the source"), (CropBox{852, 480, 0, 0}));
    // Nothing is detected (no FFmpeg is run): the test video, given no crop, is compared whole
    const auto [source_crop, test_crop] = crops_for(CropChoice(CropMode::manual, {22, 22, 0, 0}), "no ffmpeg", hd, hd,
                                                    CancelToken(), std::nullopt, std::nullopt, "bicubic");
    EXPECT_EQ(source_crop, (CropBox{1920, 1036, 0, 22}));
    EXPECT_EQ(test_crop, std::nullopt);
}

// A crop a few lines off from the encode's (issue #12: 1920x1040 against a 1920x1036 encode) is said; a real
// difference in size, or two pictures that were not cropped, is not.
TEST(Decisions, ACropAFewLinesOffIsSaid) {
    VideoInfo source;
    source.width = 1920;
    source.height = 1080;
    VideoInfo encode;
    encode.width = 1920;
    encode.height = 1036;
    const std::optional<NearMissCrop> near = near_miss_crop(source, encode, CropBox{1920, 1040, 0, 20}, std::nullopt,
                                                            ScaleDirection::source_to_distorted);
    ASSERT_TRUE(near);
    EXPECT_TRUE(near->source_scaled);
    EXPECT_EQ(near->scaled, (Size{1920, 1040}));
    EXPECT_EQ(near->target, (Size{1920, 1036}));
    EXPECT_FALSE(near_miss_crop(source, encode, CropBox{1920, 1036, 0, 22}, std::nullopt,
                                ScaleDirection::source_to_distorted));  // the same picture
    VideoInfo smaller = encode;
    smaller.width = 1280;
    smaller.height = 690;
    EXPECT_FALSE(near_miss_crop(source, smaller, CropBox{1920, 1036, 0, 22}, std::nullopt,
                                ScaleDirection::source_to_distorted));  // a real downscale
    VideoInfo uncropped = encode;
    uncropped.height = 1072;
    EXPECT_FALSE(near_miss_crop(source, uncropped, std::nullopt, std::nullopt, ScaleDirection::source_to_distorted));
}

TEST(Decisions, AScaledTestVideosBoxLinesUpWithTheSourcesRowForRow) {
    const auto frame = [](int width, int height) {
        VideoInfo info;
        info.width = width;
        info.height = height;
        return info;
    };
    // 2/3 of the size: each box rounded on its own put the test picture's top at row 46, the source's row 69, not
    // 68; on one grid both are cut to rows the scaling maps onto each other.
    const auto [source, test] = common_picture(frame(3840, 2160), frame(2560, 1440), CropBox{3840, 2024, 0, 68},
                                               CropBox{2560, 1352, 0, 44});
    const CropBox source_rows{3840, 2016, 0, 72};
    const CropBox test_rows{2560, 1344, 0, 48};
    EXPECT_EQ(source, source_rows);
    EXPECT_EQ(test, test_rows);
    // Sizes only nearly one ratio apart: each box rounded on its own, as before.
    const CropBox bars{1920, 800, 0, 140};
    const auto [near_source, near_test] =
        common_picture(frame(1920, 1080), frame(1366, 768), bars, CropBox{1366, 570, 0, 99});
    const CropBox near_rows{1366, 568, 0, 100};
    EXPECT_EQ(near_source, bars);
    EXPECT_EQ(near_test, near_rows);
}

// Issue #18: an encode cropped when it was made (HandBrake's crop, FFmpeg's crop filter) is the source's picture at
// the same size, though its frame is a fraction of a percent from the source's in shape: compared as it is, never
// read as the source scaled.
TEST(Decisions, AnEncodeCroppedWhenItWasMadeIsComparedAtItsOwnSize) {
    const auto frame = [](int width, int height) {
        VideoInfo info;
        info.width = width;
        info.height = height;
        return info;
    };
    const VideoInfo hd = frame(1920, 1080);
    // A 1-row line at the bottom, cut to HandBrake's 0/2/0/0: the test's whole frame, not 1920x1076.
    const CropBox picture{1920, 1078, 0, 0};
    const VideoInfo cut = frame(1920, 1078);
    EXPECT_EQ(common_picture(hd, cut, picture, picture), std::pair(std::optional(picture), std::optional(picture)));
    // As black bars Source only cuts it, so the two share their saved scores (auto_folder_for_source_only).
    EXPECT_EQ(test_crop_beside_source_box(picture, cut), std::nullopt);
    // Bars of its own left after 2 rows were cut at the bottom, or at the top: each box as detected.
    const CropBox bars{1920, 800, 0, 140};
    EXPECT_EQ(common_picture(hd, cut, bars, bars), std::pair(std::optional(bars), std::optional(bars)));
    const CropBox raised{1920, 800, 0, 138};
    EXPECT_EQ(common_picture(hd, cut, bars, raised), std::pair(std::optional(bars), std::optional(raised)));
    // A box further off than the rows cut cannot be this encode's: the picture both show, as before.
    const CropBox too_high{1920, 800, 0, 137};
    EXPECT_NE(common_picture(hd, cut, bars, too_high).second, std::optional(too_high));
}

// An encode cut to the source's bars when it was made and then scaled shows the source's box: both compared as
// detected, never the source's bars cut from it again. A whole-frame scale still has them cut, and one that cannot
// be told from a cut (its frame within a sample of the source's) is read so.
TEST(Decisions, AnEncodeCutToTheSourcesBarsAndScaledShowsTheSourcesBox) {
    const auto frame = [](int width, int height) {
        VideoInfo info;
        info.width = width;
        info.height = height;
        return info;
    };
    const auto kept = [](const CropBox& source, const CropBox& test) {
        return std::pair(std::optional(source), std::optional(test));
    };
    const CropBox thin{1920, 1078, 0, 0};
    const CropBox scaled{1280, 718, 0, 0};
    EXPECT_EQ(common_picture(frame(1920, 1080), frame(1280, 718), thin, scaled), kept(thin, scaled));
    const CropBox uhd{3840, 2156, 0, 2};
    const CropBox hd{1920, 1078, 0, 0};
    EXPECT_EQ(common_picture(frame(3840, 2160), frame(1920, 1078), uhd, hd), kept(uhd, hd));
    const CropBox pillars{1912, 1080, 4, 0};
    const CropBox narrow{1276, 720, 0, 0};
    EXPECT_EQ(common_picture(frame(1920, 1080), frame(1276, 720), pillars, narrow), kept(pillars, narrow));
    // Cut to the bars at the same size, its own dark row found too: that row cut from the source as well.
    const auto [source_rows, test_rows] = common_picture(frame(1920, 1080), frame(1920, 800), CropBox{1920, 800, 0, 140},
                                                         CropBox{1920, 798, 0, 2});
    EXPECT_EQ(source_rows, (CropBox{1920, 798, 0, 142}));
    EXPECT_EQ(test_rows, (CropBox{1920, 798, 0, 2}));
    // The whole frame scaled: the source's thin bars cut from the test video too.
    const CropBox thin_bars{1920, 1076, 0, 2};
    EXPECT_NE(common_picture(frame(1920, 1080), frame(1280, 720), thin_bars, CropBox{1280, 720, 0, 0}).second,
              std::optional(CropBox{1280, 720, 0, 0}));
    EXPECT_NE(common_picture(frame(1920, 1080), frame(720, 404), thin_bars, CropBox{720, 404, 0, 0}).second,
              std::optional(CropBox{720, 404, 0, 0}));
}

// Issue #20: a thin dark edge of the source, cut 2 columns wide, that an encode cut to the bars kept (1920x816
// beside a 1918x816 box): both compared at the source's box column for column, never 1918 columns scaled to 1920.
TEST(Decisions, AnEncodeThatKeptTheSourcesThinEdgeIsComparedColumnForColumn) {
    const auto frame = [](int width, int height) {
        VideoInfo info;
        info.width = width;
        info.height = height;
        return info;
    };
    const CropBox whole_test{1920, 816, 0, 0};
    const auto [source_box, test_box] =
        common_picture(frame(1920, 1080), frame(1920, 816), CropBox{1918, 816, 0, 132}, whole_test);
    EXPECT_EQ(source_box, (CropBox{1918, 816, 0, 132}));
    EXPECT_EQ(test_box, (CropBox{1918, 816, 0, 0}));
    // The edge at the left, and a thin edge on both sides of a frame as wide as the test.
    EXPECT_EQ(common_picture(frame(1920, 1080), frame(1920, 816), CropBox{1918, 816, 2, 132}, whole_test).second,
              std::optional(CropBox{1918, 816, 2, 0}));
    EXPECT_EQ(common_picture(frame(1920, 1080), frame(1920, 816), CropBox{1916, 816, 2, 132}, whole_test).second,
              std::optional(CropBox{1916, 816, 2, 0}));
    // Pillars at the left and a thin edge at the right: the box at no edge of the frame on that axis is not
    // grown, as it could grow either way.
    const CropBox pillar_box{1638, 816, 140, 132};
    const auto [pillar_source, pillar_test] =
        common_picture(frame(1920, 1080), frame(1640, 816), pillar_box, CropBox{1640, 816, 0, 0});
    EXPECT_EQ(pillar_source, std::optional(pillar_box));
    EXPECT_EQ(pillar_test, std::optional(CropBox{1640, 816, 0, 0}));
    // A thick edge (4 columns) is not one dark line rounded up: not grown.
    EXPECT_EQ(common_picture(frame(1920, 1080), frame(1920, 816), CropBox{1916, 816, 0, 132}, whole_test).second,
              std::optional(whole_test));
}

// Sizes not one fraction apart (854x480 against 1920x1080): the test video's picture rounded to even rows, and
// the source's box those rows scaled to its size, so the two show the same rows. Each rounded on its own, the
// source's box was 8 rows taller than the test's showed, and the pair was refused for its shapes.
TEST(Decisions, SizesNotOneFractionApartStillShowTheSameRows) {
    const auto frame = [](int width, int height) {
        VideoInfo info;
        info.width = width;
        info.height = height;
        return info;
    };
    const auto [source, test] =
        common_picture(frame(1920, 1080), frame(854, 480), CropBox{1920, 800, 0, 140}, CropBox{854, 356, 0, 62});
    EXPECT_EQ(source, (CropBox{1920, 792, 0, 144}));  // the test's rows 64 to 416 at the source's size
    EXPECT_EQ(test, (CropBox{854, 352, 0, 64}));
    EXPECT_FALSE(shape_problem(frame(1920, 1080), frame(854, 480), source, test));
    const auto [uhd_source, uhd_test] =
        common_picture(frame(3840, 2160), frame(854, 480), CropBox{3840, 1600, 0, 280}, CropBox{854, 356, 0, 62});
    EXPECT_EQ(uhd_source, (CropBox{3840, 1584, 0, 288}));
    EXPECT_EQ(uhd_test, (CropBox{854, 352, 0, 64}));
    EXPECT_FALSE(shape_problem(frame(3840, 2160), frame(854, 480), uhd_source, uhd_test));
}

// Black bars Source only takes Auto's saved scores where Auto kept the test
// video's whole frame (metric_cache.cpp, auto_folder_for_source_only): so a
// box of the whole frame must be no box to every route that crops, and Auto
// must give one only with the source's box as detected (an encode cropped
// when it was made, or a picture with no bars at all).
TEST(Decisions, AWholeFrameBoxIsNoBoxOnEveryRoute) {
    VideoInfo source;
    source.path = "source.mkv";
    source.codec_name = "hevc";
    source.pix_fmt = "yuv420p10le";
    source.width = 3840;
    source.height = 2160;
    source.fps = 24.0;
    source.duration = 10.0;
    VideoInfo test = source;
    test.path = "test.mkv";
    test.height = 1608;
    const CropBox bars{3840, 1608, 0, 276};
    const CropBox whole{3840, 1608, 0, 0};
    const VmafOptions options;
    const auto graph = [&](const std::optional<CropBox>& test_box) {
        return build_filtergraph(source, test, options, bars, test_box, HardwareDecoding{}, "vmaf.json",
                                 "version=vmaf_v0.6.1", "xpsnr.log", 8);
    };
    EXPECT_EQ(graph(whole), graph(std::nullopt));
    EXPECT_EQ(image_filtergraph(source, test, options, bars, whole, 1),
              image_filtergraph(source, test, options, bars, std::nullopt, 1));
    const auto chain = [&](const std::optional<CropBox>& box) {
        return vship_filter_chain(test, box, {3840, 1608}, "p010le", 1, "bicubic", std::nullopt, Rational{24, 1}, false);
    };
    EXPECT_EQ(chain(whole), chain(std::nullopt));
    const auto planned = [&](const std::optional<CropBox>& box) {
        const DecodePlan plan = plan_decode(test, box);
        return std::tuple(plan.crop_x, plan.crop_y, plan.crop_w, plan.crop_h, plan.out_w, plan.out_h);
    };
    EXPECT_EQ(planned(whole), planned(std::nullopt));

    // Sizes not one factor apart (the cropped encode): both boxes as detected.
    EXPECT_EQ(common_picture(source, test, bars, whole), std::pair(std::optional(bars), std::optional(whole)));
    // One factor apart: the source's bars cut from the test video too, unless the source has none.
    VideoInfo half = source;
    half.width = 1920;
    half.height = 1080;
    const CropBox half_whole{1920, 1080, 0, 0};
    const std::optional<CropBox> test_cut = common_picture(source, half, bars, half_whole).second;
    ASSERT_TRUE(test_cut);
    EXPECT_FALSE(test_cut->is_noop(1920, 1080));
    const CropBox source_whole{3840, 2160, 0, 0};
    EXPECT_EQ(common_picture(source, half, source_whole, half_whole),
              std::pair(std::optional(source_whole), std::optional(half_whole)));
}

TEST(Decisions, SourceOnlyCutsOnlyTheTestVideosOddLine) {
    const auto frame = [](int width, int height) {
        VideoInfo info;
        info.width = width;
        info.height = height;
        return info;
    };
    const CropBox whole_on_even_samples{852, 480, 0, 0};
    const CropBox letterboxed{852, 400, 0, 40};
    EXPECT_EQ(test_crop_beside_source_box(whole_on_even_samples, frame(853, 481)), whole_on_even_samples);
    EXPECT_EQ(test_crop_beside_source_box(letterboxed, frame(853, 401)), (CropBox{852, 400, 0, 0}));
    EXPECT_EQ(test_crop_beside_source_box(letterboxed, frame(852, 400)), std::nullopt);
    EXPECT_EQ(test_crop_beside_source_box(letterboxed, frame(853, 481)), std::nullopt);
}

// The black bars of a video whose codec the app's own decoder decodes
// (app_decoders.h: VVC) are sampled from that decoder's pictures, grain or
// not; every other video's as before, by FFmpeg, and so a raw stream's.
TEST(Decisions, VideosTheAppDecodesAreSampledFromTheAppsDecoder) {
    VideoInfo vvc;
    vvc.codec_name = "vvc";
    vvc.format_name = "matroska,webm";
    vvc.pix_fmt = "yuv420p10le";
    EXPECT_TRUE(samples_through_the_app(vvc));
    vvc.film_grain = film_grain_h274_model_1;  // VVenC's --fga 3, which FFmpeg's vvc fails on
    EXPECT_TRUE(samples_through_the_app(vvc));
    VideoInfo other = vvc;
    other.codec_name = "hevc";
    EXPECT_FALSE(samples_through_the_app(other));
    other = vvc;
    other.format_name = "vvc";  // a raw stream, sampled from its bytes
    EXPECT_FALSE(samples_through_the_app(other));
}

// A model= value with the resource folder written as "<resources>/", as
// decisions.json has it (the folder differs between PCs).
Json portable_model(const std::string& value) {
    // The folder with its trailing separator, as the system writes it.
    const std::string folder = platform::path_text(resource_folder() / "");
    if (value.starts_with("path=" + folder)) {
        std::string relative = value.substr(5 + folder.size());
        std::replace(relative.begin(), relative.end(), '\\', '/');
        return "path=<resources>/" + relative;
    }
    return value;
}

TEST(Decisions, TheVmafModelIsChosenAsThePythonAppChoosesIt) {
    ASSERT_FALSE(decisions()["models"].empty());
    for (const Json& item : decisions()["models"]) {
        VmafOptions options;
        options.model_choice = item["choice"].get<std::string>();
        options.model_choice_v1 = item["choice_v1"].get<std::string>();
        options.custom_model_path = "C:/m/custom.json";
        const int width = item["size"][0].get<int>();
        const int height = item["size"][1].get<int>();
        SCOPED_TRACE(item.dump());
        try {
            EXPECT_EQ(portable_model(resolve_model(options, width, height)), item["model"]);
        } catch (const std::exception& error) {
            EXPECT_EQ(Json({{"error", error.what()}}), item["model"]);
        }
        try {
            EXPECT_EQ(portable_model(resolve_v1_model(options, width, height)), item["model_v1"]);
        } catch (const std::exception& error) {
            EXPECT_EQ(Json({{"error", error.what()}}), item["model_v1"]);
        }
    }
    for (const Json& item : decisions()["resolutions"]) {
        EXPECT_EQ(model_for_resolution(item["size"][0].get<int>(), item["size"][1].get<int>()),
                  item["model"].get<std::string>());
    }
}

// An empty VML_RESOURCES names no folder: the models are found where they are
// with it unset, not in the working folder (PowerShell 7 leaves a variable
// "removed" with SetEnvironmentVariable($name, $null) set, and empty).
TEST(Decisions, AnEmptyResourcesVariableIsTakenAsUnset) {
    const std::optional<std::string> previous = platform::environment_variable("VML_RESOURCES");
    platform::set_environment_variable("VML_RESOURCES", "");  // removes it
    const std::filesystem::path unset = resource_folder();
#ifdef _WIN32
    SetEnvironmentVariableW(L"VML_RESOURCES", L"");  // set, and empty
#else
    setenv("VML_RESOURCES", "", 1);
#endif
    const std::optional<std::string> value = platform::environment_variable("VML_RESOURCES");
    const std::filesystem::path empty = resource_folder();
    platform::set_environment_variable("VML_RESOURCES", previous.value_or(""));
    ASSERT_EQ(value, std::optional<std::string>(""));
    EXPECT_FALSE(unset.empty());
    EXPECT_EQ(empty, unset);
}

// D3D11VA decodes on the adapter the app names as it starts (the AMD GPU's):
// FFmpeg given no -hwaccel_device would take D3D11's default adapter, which on
// a PC with an NVIDIA card and an AMD iGPU is NVIDIA's. The other decoders
// pick their maker's GPU themselves.
TEST(Decisions, D3d11vaDecodesOnTheAdapterNamed) {
    use_d3d11va_adapter(1);
    const std::vector<std::string> d3d11va = hwaccel_args("d3d11va");
    const std::vector<std::string> cuda = hwaccel_args("cuda");
    use_d3d11va_adapter(std::nullopt);
    EXPECT_EQ(d3d11va, (std::vector<std::string>{"-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11",
                                                 "-hwaccel_device", "1"}));
    EXPECT_EQ(cuda, (std::vector<std::string>{"-threads", "1", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"}));
    EXPECT_EQ(hwaccel_args("d3d11va"),
              (std::vector<std::string>{"-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11"}));
}

// The warning before SSIMULACRA2 or Butteraugli on the CPU: only over 10 minutes of the range compared, the
// estimate of a 4K pair (0.17 s for SSIMULACRA2, 0.34 s with Butteraugli), given roughly.
TEST(LongCpuRun, CountsTheRangeAndTheMetricsCalculated) {
    VideoInfo video;
    video.width = 3840;
    video.height = 2160;
    video.fps = 24;
    video.duration = 3600;
    VmafOptions options;
    EXPECT_FALSE(long_cpu_run({"vmaf", "psnr"}, video, video, options));
    const std::optional<LongCpuRun> both = long_cpu_run({"butteraugli", "vmaf", "ssimulacra2"}, video, video, options);
    ASSERT_TRUE(both);
    EXPECT_EQ(both->keys, (std::vector<std::string>{"ssimulacra2", "butteraugli"}));
    EXPECT_DOUBLE_EQ(both->video_seconds, 3600);
    EXPECT_NEAR(both->scoring_seconds, 3600 * 24 * 0.34, 30);  // a 4K pair: 0.34 s, within 0.1 %
    const std::optional<LongCpuRun> one = long_cpu_run({"ssimulacra2"}, video, video, options);
    ASSERT_TRUE(one);
    EXPECT_NEAR(one->scoring_seconds, 3600 * 24 * 0.17, 15);
    options.set_time_range(0, 600'000);  // 10 minutes: not long
    EXPECT_FALSE(long_cpu_run({"ssimulacra2"}, video, video, options));
    options.set_time_range(1'800'000, 0);  // the last 30 minutes
    ASSERT_TRUE(long_cpu_run({"ssimulacra2"}, video, video, options));
    EXPECT_NEAR(long_cpu_run({"ssimulacra2"}, video, video, options)->video_seconds, 1800, 1e-9);

    EXPECT_EQ(rough_time(3 * 86400.0), (std::pair<long long, std::string_view>{3, "day"}));
    EXPECT_EQ(rough_time(5.5 * 3600), (std::pair<long long, std::string_view>{6, "hour"}));  // halves to even
    EXPECT_EQ(rough_time(4.5 * 3600), (std::pair<long long, std::string_view>{4, "hour"}));
    EXPECT_EQ(rough_time(10), (std::pair<long long, std::string_view>{1, "minute"}));
}

}  // namespace
}  // namespace vml
