// The run's pure decisions, held to the Python app's own (decisions.json):
// pixel formats and GPU decoding, frame-rate fractions and frame counts,
// the two ways of pairing frames, why two videos cannot be compared, the
// shapes they are compared at, and the VMAF model.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include "reference.h"
#include "vml/crop_detect.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/hardware_decoding.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"

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

// Sets VML_FRAME_PAIRING for one test, and puts it back after.
class PairingVariable {
public:
    explicit PairingVariable(const std::string& value) {
        previous_ = platform::environment_variable(pairing_variable);
        set(value);
    }
    ~PairingVariable() { set(previous_.value_or("")); }

private:
    static void set(const std::string& value) {
#ifdef _WIN32
        _putenv_s(std::string(pairing_variable).c_str(), value.c_str());
#else
        setenv(std::string(pairing_variable).c_str(), value.c_str(), 1);
#endif
    }
    std::optional<std::string> previous_;
};

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
// (gpu_decodes_exactly).
TEST(Decisions, GpuDecodingIsChosenAsThePythonAppChoosesIt) {
    ASSERT_FALSE(decisions()["decode"].empty());
    for (const Json& item : decisions()["decode"]) {
        DecodingCapabilities capabilities;
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
// Predictive", in 4:2:0 too) is decoded on the GPU only by NVIDIA's and
// Apple's decoders, both checked bit-exact; Intel's gave other pictures, and
// FFmpeg's D3D11VA refuses the profile. Other H.264 as before, and 10-bit
// lossless on NVIDIA's.
TEST(Decisions, H264In444ProfilesIsGpuDecodedOnlyWhereItIsExact) {
    DecodingCapabilities pc;
    pc.hwaccels = {"cuda", "qsv", "d3d11va"};
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
    EXPECT_EQ(software_decoding_reason(video, true), "this GPU does not decode H.264's CAVLC 4:4:4 profile exactly");
    video.profile = "High";
    EXPECT_EQ(pick_decode(pc, GpuVendor::intel, video), "qsv");
    EXPECT_EQ(pick_decode(pc, GpuVendor::amd, video), "d3d11va");
    video.profile.clear();  // not probed: taken as 4:4:4
    EXPECT_EQ(pick_decode(pc, GpuVendor::intel, video), std::nullopt);
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), "cuda");
    video.profile = "High 4:4:4 Predictive";
    video.pix_fmt = "yuv420p10le";
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, video), "cuda");
    DecodingCapabilities mac;
    mac.hwaccels = {"videotoolbox"};
    mac.videotoolbox_codecs = {"h264"};
    video.pix_fmt = "yuv420p";
    EXPECT_EQ(pick_decode(mac, GpuVendor::automatic, video, MacGpuDecoding::on), "videotoolbox");
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
    for (const auto& [mode, expected] : decisions()["pairing"].items()) {
        const PairingVariable variable(mode);
        SCOPED_TRACE("VML_FRAME_PAIRING='" + mode + "'");
        const Rational rate{24000, 1001};
        EXPECT_EQ(framesync_opts(), expected["framesync"].get<std::vector<std::string>>());
        EXPECT_EQ(pairing_filters(rate), expected["pairing"].get<std::vector<std::string>>());
        EXPECT_EQ(numbering_filters(rate), expected["numbering"].get<std::vector<std::string>>());
        EXPECT_EQ(duration_args("30.042"), expected["duration"].get<std::vector<std::string>>());
        EXPECT_EQ(duration_args(std::nullopt), expected["no_duration"].get<std::vector<std::string>>());
    }
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
            pair_problem(video_from(item["source"]), video_from(item["test"]), *test::reference_number(item["limit"]));
        EXPECT_EQ(problem.value_or("<none>"),
                  item["problem"].is_null() ? std::string("<none>") : item["problem"].get<std::string>());
    }
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

TEST(Decisions, BlackBarBoxesCoverTheSamePictureAsThePythonAppMakesThem) {
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

}  // namespace
}  // namespace vml
