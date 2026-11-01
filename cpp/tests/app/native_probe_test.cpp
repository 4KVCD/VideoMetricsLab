// How Video Compare's native playback probes a file (native_probe.h): never
// with FFmpeg's own VVC decoder, which crashes on a 4:0:0 video whose AFGS1
// film grain set has chroma.
#include "native_probe.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "ffmpeg_api.h"
#include "native_device.h"

namespace vml::app {

namespace {

// The bundled FFmpeg libraries, loaded first (the app's calls into them load them by name only when first made);
// false, the test skipped, where they are not there (a macOS build without them)
bool ffmpeg_loaded() { return !load_ffmpeg_libraries().has_value(); }

bool lists(const std::string& whitelist, const std::string& name) {
    return ("," + whitelist + ",").find("," + name + ",") != std::string::npos;
}

}  // namespace

TEST(NativeProbe, ListsEveryDecoderButFfmpegsOwnVvc) {
    if (!ffmpeg_loaded()) {
        GTEST_SKIP() << "the bundled FFmpeg libraries are not there";
    }
    const std::string& whitelist = probe_decoder_whitelist();
    EXPECT_FALSE(lists(whitelist, "vvc"));
    EXPECT_TRUE(lists(whitelist, "libvvdec"));
    EXPECT_TRUE(lists(whitelist, "hevc"));
    EXPECT_TRUE(lists(whitelist, "aac"));
}

// FFmpeg's own decoder would crash in avformat_find_stream_info on this stream (tests/media/README.md); restricted,
// the probe reads it from its parser alone: its size, and no pixel format (no decoder ran)
TEST(NativeProbe, AMonochromeVvcWithChromaGrainProbesWithoutFfmpegsDecoder) {
    if (!ffmpeg_loaded()) {
        GTEST_SKIP() << "the bundled FFmpeg libraries are not there";
    }
    AVFormatContext* format = avformat_alloc_context();
    ASSERT_NE(format, nullptr);
    // Closed whatever the test ends on (avformat_open_input frees the context and nulls it when it fails)
    const std::unique_ptr<AVFormatContext*, void (*)(AVFormatContext**)> closing(&format, avformat_close_input);
    ASSERT_TRUE(restrict_probe_decoders(*format));
    const std::string path = std::string(VML_TEST_MEDIA_DIR) + "/vvc_gray_afgs1_chroma.266";
    ASSERT_GE(avformat_open_input(&format, path.c_str(), nullptr, nullptr), 0);
    EXPECT_GE(avformat_find_stream_info(format, nullptr), 0);
    ASSERT_EQ(format->nb_streams, 1u);
    const AVCodecParameters* stream = format->streams[0]->codecpar;
    EXPECT_EQ(stream->codec_id, AV_CODEC_ID_VVC);
    EXPECT_EQ(stream->width, 64);
    EXPECT_EQ(stream->height, 64);
    EXPECT_EQ(stream->format, AV_PIX_FMT_NONE);
}

}  // namespace vml::app
