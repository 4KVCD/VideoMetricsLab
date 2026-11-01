// Video Compare's soundtrack on macOS (native_soundtrack.cpp): the channels it
// is played in for a device's speakers, the device's latency, and the
// resampler putting each channel in its speaker's place -- no sound played and
// no device needed.
#include "native_soundtrack.h"

#include <AudioToolbox/AudioToolbox.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ffmpeg_api.h"

namespace vml::app {
namespace {

using Labels = std::vector<unsigned>;

const Labels five_one{kAudioChannelLabel_Left,      kAudioChannelLabel_Right,     kAudioChannelLabel_Center,
                      kAudioChannelLabel_LFEScreen, kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround};

// A device with speakers FFmpeg names plays in them, in its order; one whose
// channels name none (the Mac's own), or a speaker FFmpeg has no name for,
// or one twice, or too few or too many, plays in stereo as before.
TEST(Soundtrack, AMacDeviceIsPlayedInTheSpeakersItNames) {
    const unsigned unknown = kAudioChannelLabel_Unknown;
    EXPECT_TRUE(soundtrack_output_labels({}).empty());
    EXPECT_TRUE(soundtrack_output_labels({unknown, unknown}).empty());
    EXPECT_TRUE(soundtrack_output_labels({kAudioChannelLabel_Left, kAudioChannelLabel_Right}).empty());
    EXPECT_EQ(soundtrack_output_labels(five_one), five_one);
    Labels seven_one = five_one;
    seven_one.insert(seven_one.end(), {kAudioChannelLabel_RearSurroundLeft, kAudioChannelLabel_RearSurroundRight});
    EXPECT_EQ(soundtrack_output_labels(seven_one), seven_one);
    Labels with_unknown = five_one;
    with_unknown.insert(with_unknown.end(), {unknown, unknown});
    EXPECT_EQ(soundtrack_output_labels(with_unknown), five_one);  // the unlabelled channels left out
    Labels two_and_unknown{kAudioChannelLabel_Left, kAudioChannelLabel_Right};
    two_and_unknown.insert(two_and_unknown.end(), 6, unknown);
    EXPECT_TRUE(soundtrack_output_labels(two_and_unknown).empty());
    Labels unnamed = five_one;
    unnamed[4] = kAudioChannelLabel_LeftSurroundDirect;
    EXPECT_TRUE(soundtrack_output_labels(unnamed).empty());
    Labels nine = seven_one;
    nine.push_back(kAudioChannelLabel_CenterSurround);
    EXPECT_TRUE(soundtrack_output_labels(nine).empty());
    // A speaker named twice (two pairs, L R L R): stereo, not a silent pair.
    EXPECT_TRUE(soundtrack_output_labels({kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_Left,
                                          kAudioChannelLabel_Right})
                    .empty());

    // FFmpeg's AVChannel numbers (libavutil/channel_layout.h): front left 0,
    // LFE 3, back left 4, side left 9.
    EXPECT_EQ(ffmpeg_channel_of(kAudioChannelLabel_Left), 0);
    EXPECT_EQ(ffmpeg_channel_of(kAudioChannelLabel_LeftSurround), 9);
    EXPECT_EQ(ffmpeg_channel_of(kAudioChannelLabel_RearSurroundLeft), 4);
    EXPECT_EQ(ffmpeg_channel_of(kAudioChannelLabel_LFEScreen), 3);
    EXPECT_EQ(ffmpeg_channel_of(unknown), -1);

    // This Mac's headphones: 72 frames the device's, 24 its stream's.
    EXPECT_DOUBLE_EQ(output_latency_ms(72, 24, 48000), 2.0);
    EXPECT_DOUBLE_EQ(output_latency_ms(72, 24, 0), 0.0);
}

// The soundtrack's channels reach the device's speakers in the device's
// order: a 5.1 soundtrack on a device ordered L C R Ls Rs LFE, each channel
// holding its own value, comes out with each value where its speaker is.
TEST(Soundtrack, EachChannelIsResampledToItsSpeaker) {
    const Labels device{kAudioChannelLabel_Left,         kAudioChannelLabel_Center,        kAudioChannelLabel_Right,
                        kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround, kAudioChannelLabel_LFEScreen};
    AVChannelLayout out{};
    soundtrack_output_layout(soundtrack_output_labels(device), &out);
    ASSERT_EQ(out.nb_channels, 6);
    AVChannelLayout in{};
    av_channel_layout_from_mask(&in, AV_CH_LAYOUT_5POINT1);  // FL FR FC LFE SL SR
    SwrContext* resampler = nullptr;
    ASSERT_GE(swr_alloc_set_opts2(&resampler, &out, AV_SAMPLE_FMT_FLT, 48000, &in, AV_SAMPLE_FMT_FLT, 48000, 0,
                                  nullptr),
              0);
    ASSERT_GE(swr_init(resampler), 0);
    constexpr int frames = 16;
    std::vector<float> given(frames * 6);
    for (int frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < 6; ++channel) {
            given[static_cast<std::size_t>(frame * 6 + channel)] = 0.1f * static_cast<float>(channel + 1);
        }
    }
    std::vector<float> got(frames * 6);
    const auto* from = reinterpret_cast<const std::uint8_t*>(given.data());
    auto* to = reinterpret_cast<std::uint8_t*>(got.data());
    ASSERT_EQ(swr_convert(resampler, &to, frames, &from, frames), frames);
    // FL FR FC LFE SL SR hold 0.1 .. 0.6: L C R Ls Rs LFE are 0.1 0.3 0.2 0.5 0.6 0.4.
    const std::vector<float> expected{0.1f, 0.3f, 0.2f, 0.5f, 0.6f, 0.4f};
    for (int channel = 0; channel < 6; ++channel) {
        EXPECT_NEAR(got[static_cast<std::size_t>((frames - 1) * 6 + channel)], expected[static_cast<std::size_t>(channel)],
                    1e-6f)
            << channel;
    }
    swr_free(&resampler);
    av_channel_layout_uninit(&in);
    av_channel_layout_uninit(&out);

    // A device played in stereo: FFmpeg's stereo.
    AVChannelLayout stereo{};
    soundtrack_output_layout({}, &stereo);
    AVChannelLayout expected_stereo = AV_CHANNEL_LAYOUT_STEREO;
    EXPECT_EQ(av_channel_layout_compare(&stereo, &expected_stereo), 0);
    av_channel_layout_uninit(&stereo);
}

}  // namespace
}  // namespace vml::app
