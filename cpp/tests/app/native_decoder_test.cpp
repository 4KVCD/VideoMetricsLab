// Video Compare's native decoding (native_decoder.h) on real clips: the
// frames it gives are FFmpeg's own, counted from each video's first frame,
// whether the GPU decodes the video or FFmpeg's software decoder does; and
// the source scaled down to the test video's size is the source's picture.
// It needs a Direct3D 11 GPU and the bundled FFmpeg libraries.
#include "native_decoder.h"

#include <d3d11.h>
#include <wrl/client.h>

#include <QApplication>
#include <QDeadlineTimer>
#include <QThread>

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "native_device.h"
#include "test_videos.h"
#include "vml/ffprobe.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/video_playback.h"

namespace vml::app {
namespace {

// The frame `wanted` of `comparison`'s `side`, decoded natively after a seek
// to the frame before it, as the playback pool seeks.
NativeFramePtr decode_frame(const std::shared_ptr<NativeDevice>& device, const FrameComparison& comparison,
                            FrameSide side, long long wanted, QString& status) {
    const double fps = comparison.fps;
    FfmpegNativeVideo video(device, comparison, side, {},
                            static_cast<int>(std::nearbyint(static_cast<double>(wanted - 1) * 1000 / fps)));
    video.start();
    NativeFramePtr got;
    const QDeadlineTimer deadline(30'000);
    while (!got && !deadline.hasExpired()) {
        const NativeUpdate update = video.poll();
        if (update.status) {
            status = *update.status;
        }
        EXPECT_FALSE(update.error) << update.error->toStdString();
        if (!video.seeking()) {
            while (NativeFramePtr frame = video.try_pull()) {
                if (std::llround(static_cast<double>(frame->time_ns) / 1e9 * fps) == wanted) {
                    got = frame;
                }
            }
        }
        QThread::msleep(2);
    }
    return got;
}

// The frame's luma, read back from the GPU: 8-bit samples, or 10-bit ones.
std::vector<int> read_luma(NativeDevice& device, const NativeFrame& frame) {
    D3D11_TEXTURE2D_DESC description{};
    frame.texture->GetDesc(&description);
    D3D11_TEXTURE2D_DESC staged = description;
    staged.MipLevels = 1;
    staged.ArraySize = 1;
    staged.Usage = D3D11_USAGE_STAGING;
    staged.BindFlags = 0;
    staged.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staged.MiscFlags = 0;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    std::vector<int> luma;
    if (FAILED(device.device()->CreateTexture2D(&staged, nullptr, &staging))) {
        ADD_FAILURE() << "no staging texture";
        return luma;
    }
    const std::lock_guard lock(device.lock());
    device.context()->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, frame.texture, frame.subresource, nullptr);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(device.context()->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        ADD_FAILURE() << "the frame could not be read back";
        return luma;
    }
    for (int row = 0; row < frame.height; ++row) {
        const auto* line = static_cast<const unsigned char*>(mapped.pData) + static_cast<std::size_t>(mapped.RowPitch) * row;
        for (int x = 0; x < frame.width; ++x) {
            luma.push_back(frame.ten_bit ? (line[2 * x] | (line[2 * x + 1] << 8)) >> 6 : line[x]);
        }
    }
    device.context()->Unmap(staging.Get(), 0);
    return luma;
}

// FFmpeg's own luma of frame `wanted` of `path`, through `filter` after it.
std::vector<int> ffmpeg_luma(const test::Tools& found, const std::filesystem::path& path, long long wanted,
                             bool ten_bit, const std::string& filter, const std::filesystem::path& raw) {
    std::string chain = "select=eq(n\\," + std::to_string(wanted) + ")";
    if (!filter.empty()) {
        chain += "," + filter;
    }
    platform::ChildProcess process = platform::ChildProcess::start(
        {found.ffmpeg, "-v", "error", "-y", "-i", platform::path_text(path), "-vf", chain, "-frames:v", "1", "-f",
         "rawvideo", "-pix_fmt", ten_bit ? "p010le" : "nv12", platform::path_text(raw)});
    EXPECT_EQ(process.wait(), 0);
    std::ifstream file(raw, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<int> luma;
    const std::size_t sample = ten_bit ? 2 : 1;
    for (std::size_t index = 0; index + sample <= bytes.size(); index += sample) {
        luma.push_back(ten_bit ? (bytes[index] | (bytes[index + 1] << 8)) >> 6 : bytes[index]);
    }
    return luma;
}

class NativeDecoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        found_ = test::tools();
        if (!found_) {
            GTEST_SKIP() << "no FFmpeg";
        }
        if (const std::optional<std::string> problem = load_ffmpeg_libraries()) {
            GTEST_SKIP() << *problem;
        }
        try {
            device_ = NativeDevice::shared();
        } catch (const std::exception& error) {
            GTEST_SKIP() << error.what();
        }
    }

    FrameComparison clip(const std::string& name, const std::string& encoder, const std::string& pix_fmt) {
        FrameComparison comparison;
        comparison.source_info =
            probe_video(test::make_clip(*found_, folder_ / name, encoder, pix_fmt, 2.0, {}, "640x360"), found_->ffprobe);
        comparison.distorted_info = comparison.source_info;
        comparison.fps = comparison.source_info.fps;
        return comparison;
    }

    test::Folder folder_;
    std::optional<test::Tools> found_;
    std::shared_ptr<NativeDevice> device_;
};

}  // namespace

// Frame 30 of each clip, after a seek, is FFmpeg's frame 30 sample for
// sample: an MP4 with B-frames and 8-bit H.264 (which the GPU decodes), 10-bit
// HEVC in Matroska (the GPU too), and 10-bit H.264, which no GPU decodes
// (FFmpeg's software decoder, the picture uploaded).
TEST_F(NativeDecoderTest, TheFramesAreFfmpegsCountedFromTheFirst) {
    struct Case {
        std::string name;
        std::string encoder;
        std::string pix_fmt;
        std::string decoder;
    };
    for (const Case& item : {Case{"h264.mp4", "libx264", "yuv420p", "GPU: d3d11va h264"},
                             Case{"hevc.mkv", "libx265", "yuv420p10le", "GPU: d3d11va hevc"},
                             Case{"high10.mkv", "libx264", "yuv420p10le", "software: h264"}}) {
        SCOPED_TRACE(item.name);
        const FrameComparison comparison = clip(item.name, item.encoder, item.pix_fmt);
        QString status;
        const NativeFramePtr frame = decode_frame(device_, comparison, FrameSide::source, 30, status);
        ASSERT_NE(frame, nullptr) << "frame 30 never came";
        EXPECT_TRUE(status.contains(QString::fromStdString("source " + item.decoder))) << status.toStdString();
        const std::vector<int> native = read_luma(*device_, *frame);
        std::vector<int> wanted =
            ffmpeg_luma(*found_, comparison.source_info.path, 30, frame->ten_bit, "", folder_ / "frame.raw");
        wanted.resize(native.size());
        EXPECT_EQ(native, wanted) << "not FFmpeg's frame 30";
    }
    // A video that starts 0.25 s (six frames) after its file does, behind
    // its soundtrack: frames are counted from its first frame, not the file's
    // start.
    const FrameComparison pattern = clip("pattern.mp4", "libx264", "yuv420p");
    const std::filesystem::path late = folder_ / "late.mp4";
    test::run_ffmpeg(*found_, {"-itsoffset", "0.25", "-i", platform::path_text(pattern.source_info.path), "-f", "lavfi",
                               "-i", "anullsrc=r=48000:cl=mono", "-t", "2.3", "-map", "0:v", "-map", "1:a", "-c:v",
                               "copy", "-c:a", "aac", platform::path_text(late)});
    FrameComparison delayed;
    delayed.source_info = probe_video(late, found_->ffprobe);
    ASSERT_NEAR(delayed.source_info.start_offset, 0.25, 0.01);
    delayed.distorted_info = delayed.source_info;
    delayed.fps = delayed.source_info.fps;
    QString status;
    const NativeFramePtr frame = decode_frame(device_, delayed, FrameSide::source, 30, status);
    ASSERT_NE(frame, nullptr);
    const std::vector<int> native = read_luma(*device_, *frame);
    std::vector<int> wanted = ffmpeg_luma(*found_, late, 30, frame->ten_bit, "", folder_ / "late.raw");
    wanted.resize(native.size());
    EXPECT_EQ(native, wanted) << "not the video's frame 30";
}

// Played scaled down (Reference playback), the source comes at the test
// video's size, its picture scaled: within what two scalers' filters differ by
// (40 dB), where another frame would be far off.
TEST_F(NativeDecoderTest, TheSourceScaledDownIsTheSourcesPicture) {
    FrameComparison pair = clip("source.mkv", "libx265", "yuv420p10le");
    FrameComparison smaller;
    smaller.source_info = probe_video(
        test::make_clip(*found_, folder_ / "encode.mkv", "libx264", "yuv420p", 2.0, {}, "320x180"), found_->ffprobe);
    pair.distorted_info = smaller.source_info;
    const FrameComparison recipe = source_playback_comparison(pair, false);
    QString status;
    const NativeFramePtr frame = decode_frame(device_, recipe, FrameSide::source, 30, status);
    ASSERT_NE(frame, nullptr);
    EXPECT_EQ(frame->width, 320);
    EXPECT_EQ(frame->height, 180);
    EXPECT_TRUE(frame->prepared);
    const std::vector<int> native = read_luma(*device_, *frame);
    const auto psnr = [&native](const std::vector<int>& other) {
        double squares = 0;
        for (std::size_t index = 0; index < native.size(); ++index) {
            const double difference = native[index] - other[index];
            squares += difference * difference;
        }
        const double mse = squares / static_cast<double>(native.size());
        return mse == 0 ? 100.0 : 10 * std::log10(1023.0 * 1023.0 / mse);
    };
    const std::vector<int> same =
        ffmpeg_luma(*found_, pair.source_info.path, 30, true, "scale=320:180:flags=bilinear", folder_ / "a.raw");
    const std::vector<int> other =
        ffmpeg_luma(*found_, pair.source_info.path, 33, true, "scale=320:180:flags=bilinear", folder_ / "b.raw");
    ASSERT_GE(same.size(), native.size());
    ASSERT_GE(other.size(), native.size());
    EXPECT_GT(psnr(same), 40.0);
    EXPECT_GT(psnr(same), psnr(other) + 3) << "not frame 30";
}

}  // namespace vml::app
