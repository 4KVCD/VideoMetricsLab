#include "frame_types.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "native_device.h"

namespace vml::app {
namespace {

TEST(FrameTypes, GoInPresentationOrderOnlyIfEveryFrameMatches) {
    // Decode order I P B, shown I B P, as ffprobe's packets are sorted.
    const std::vector<ReadFrame> frames{
        {0.0, 0, 100, frame_type_i}, {2.0, 100, 50, frame_type_p}, {1.0, 150, 20, frame_type_b}};
    const std::vector<long long> shown{100, 20, 50};
    EXPECT_EQ(types_matching_frames(frames, false, shown),
              (std::vector<std::uint8_t>{frame_type_i, frame_type_b, frame_type_p}));
    // A raw stream's order is what it was read in: these sizes then do not match.
    EXPECT_TRUE(types_matching_frames(frames, true, shown).empty());
    const std::vector<long long> other_size{100, 21, 50};
    EXPECT_TRUE(types_matching_frames(frames, false, other_size).empty());
    const std::vector<long long> fewer{100, 20};
    EXPECT_TRUE(types_matching_frames(frames, false, fewer).empty());
}

TEST(FrameTypes, ReadWithoutTheDecoderThatCrashesOnThisStream) {
    if (load_ffmpeg_libraries().has_value()) {
        GTEST_SKIP() << "the bundled FFmpeg libraries are not there";
    }
    const std::filesystem::path path = std::string(VML_TEST_MEDIA_DIR) + "/vvc_gray_afgs1_chroma.266";
    VideoInfo info;
    info.path = path;
    const auto bytes = static_cast<long long>(std::filesystem::file_size(path));
    const BitrateData data = bitrate_data(path, {0.0}, {1.0 / 24}, {bytes}, {true});
    const CancelToken cancel;
    FrameTypeReader reader(info, cancel);
    EXPECT_EQ(reader.types_of(data), std::vector<std::uint8_t>{frame_type_i});
}

// #205: a reading that failed (a file that would not open, a read cut short
// on a share, out of memory) gave no types, as a file read whole whose packets
// do not match does, and the Bitrate tab saved that as the file's types: its
// frame types were never shown again until Clear. Only a file read to its end
// gives an answer; the rest give nothing, to be read again.
TEST(FrameTypes, AReadingThatDidNotGetToTheEndGivesNoAnswer) {
    if (load_ffmpeg_libraries().has_value()) {
        GTEST_SKIP() << "the bundled FFmpeg libraries are not there";
    }
    const std::filesystem::path path = std::string(VML_TEST_MEDIA_DIR) + "/vvc_gray_afgs1_chroma.266";
    const auto bytes = static_cast<long long>(std::filesystem::file_size(path));
    const BitrateData data = bitrate_data(path, {0.0}, {1.0 / 24}, {bytes}, {true});
    VideoInfo gone;
    gone.path = path.parent_path() / "no such file.266";
    const CancelToken cancel;
    FrameTypeReader unopened(gone, cancel);
    EXPECT_EQ(unopened.types_of(data), std::nullopt);

    // Read whole, its packets not those of `data`: no types, for good.
    VideoInfo info;
    info.path = path;
    const BitrateData other = bitrate_data(path, {0.0}, {1.0 / 24}, {bytes + 1}, {true});
    FrameTypeReader mismatched(info, cancel);
    EXPECT_EQ(mismatched.types_of(other), std::vector<std::uint8_t>{});

    CancelToken cancelled;
    cancelled.cancel();
    FrameTypeReader stopped(info, cancelled);
    EXPECT_EQ(stopped.types_of(data), std::nullopt);
}

}  // namespace
}  // namespace vml::app
