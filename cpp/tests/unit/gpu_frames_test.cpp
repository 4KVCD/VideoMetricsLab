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
