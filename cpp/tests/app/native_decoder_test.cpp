// Video Compare's native decoding (native_decoder.h) on real clips: the
// frames it gives are FFmpeg's own, luma and chroma, counted from each
// video's first frame (a raw stream's from its start), whether the GPU
// decodes the video, FFmpeg's software decoder does or dav1d; a seek back
// gives the frames from it on; the source scaled down to the test video's
// size is the source's picture; and a VVC video's film grain is drawn as the
// Python app draws it, or by FFmpeg where that is AV1's. The tests run on
// every GPU of the PC (NVIDIA's and Intel's on the development PC). They
// need a Direct3D 11 GPU (on macOS, the Mac's, with VideoToolbox) and the
// bundled FFmpeg libraries.
#include "native_decoder.h"

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#else
#include <CoreVideo/CoreVideo.h>
#include <sys/sysctl.h>
#endif

#include <QApplication>
#include <QDeadlineTimer>
#include <QThread>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "grain_params.h"
#include "native_dav1d.h"
#include "native_device.h"
#include "native_grain.h"
#include "test_videos.h"
#include "vml/ffprobe.h"
#include "vml/film_grain.h"
#include "vml/model_select.h"
#include "vml/npz.h"
#include "vml/platform/library.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/video_playback.h"

namespace vml::app {
namespace {

// The frame `wanted` of `comparison`'s `side`, decoded natively after a seek
// to the frame before it, as the playback pool seeks; in `software`, as on a
// GPU without a decoder for it.
NativeFramePtr decode_frame(const std::shared_ptr<NativeDevice>& device, const FrameComparison& comparison,
                            FrameSide side, long long wanted, QString& status, bool software = false) {
    const double fps = comparison.fps;
    FfmpegNativeVideo video(device, comparison, side, {},
                            static_cast<int>(std::nearbyint(static_cast<double>(wanted - 1) * 1000 / fps)));
    if (software) {
        video.decode_in_software();
    }
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

// The frame's picture, read back from the GPU as FFmpeg writes NV12 or P010
// (ffmpeg_picture): the luma's rows, then the chroma's, its two samples side
// by side; 8-bit samples, or 10-bit ones. The luma alone unless `chroma`.
std::vector<int> read_picture(NativeDevice& device, const NativeFrame& frame, bool chroma = true) {
#ifndef _WIN32
    // The pixel buffer's planes, read where they are (an IOSurface).
    static_cast<void>(device);
    std::vector<int> samples;
    if (CVPixelBufferLockBaseAddress(frame.picture, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
        ADD_FAILURE() << "the frame could not be read back";
        return samples;
    }
    const auto rows = [&](std::size_t plane, int count, int per_row) {
        const auto* base = static_cast<const unsigned char*>(CVPixelBufferGetBaseAddressOfPlane(frame.picture, plane));
        const std::size_t pitch = CVPixelBufferGetBytesPerRowOfPlane(frame.picture, plane);
        for (int row = 0; row < count; ++row) {
            const unsigned char* line = base + pitch * static_cast<std::size_t>(row);
            for (int x = 0; x < per_row; ++x) {
                samples.push_back(frame.ten_bit ? (line[2 * x] | (line[2 * x + 1] << 8)) >> 6 : line[x]);
            }
        }
    };
    rows(0, frame.height, frame.width);
    if (chroma) {
        rows(1, (frame.height + 1) / 2, 2 * ((frame.width + 1) / 2));
    }
    CVPixelBufferUnlockBaseAddress(frame.picture, kCVPixelBufferLock_ReadOnly);
    return samples;
#else
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
    std::vector<int> samples;
    if (FAILED(device.device()->CreateTexture2D(&staged, nullptr, &staging))) {
        ADD_FAILURE() << "no staging texture";
        return samples;
    }
    const std::lock_guard lock(device.lock());
    device.context()->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, frame.texture, frame.subresource, nullptr);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(device.context()->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        ADD_FAILURE() << "the frame could not be read back";
        return samples;
    }
    const auto* base = static_cast<const unsigned char*>(mapped.pData);
    const auto rows = [&](const unsigned char* plane, int count, int per_row) {
        for (int row = 0; row < count; ++row) {
            const unsigned char* line = plane + static_cast<std::size_t>(mapped.RowPitch) * row;
            for (int x = 0; x < per_row; ++x) {
                samples.push_back(frame.ten_bit ? (line[2 * x] | (line[2 * x + 1] << 8)) >> 6 : line[x]);
            }
        }
    };
    rows(base, frame.height, frame.width);
    if (chroma) {
        // The chroma's rows follow the luma's, after the texture's whole
        // height of them (a decoder's texture can be taller than its picture).
        rows(base + static_cast<std::size_t>(mapped.RowPitch) * description.Height, (frame.height + 1) / 2,
             2 * ((frame.width + 1) / 2));
    }
    device.context()->Unmap(staging.Get(), 0);
    return samples;
#endif
}

// FFmpeg's own picture of frame `wanted` of `path`, through `filter` after
// it, in NV12 or P010: the luma's rows, then the chroma's.
std::vector<int> ffmpeg_picture(const test::Tools& found, const std::filesystem::path& path, long long wanted,
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

// Whether `found`'s FFmpeg has the encoder `name` (Homebrew's has no
// libvvenc, say).
bool has_encoder(const test::Tools& found, const std::string& name) {
    platform::ChildProcess process = platform::ChildProcess::start({found.ffmpeg, "-hide_banner", "-encoders"});
    std::string listed;
    while (const std::optional<std::string> line = process.read_line()) {
        listed += *line + "\n";
    }
    process.wait();
    return listed.find(" " + name + " ") != std::string::npos;
}

// The numbers of the first `count` frames `video` gives once its seek asked
// for is made, as the playback pool takes them; fewer if they do not come.
std::vector<long long> frames_after_seek(FfmpegNativeVideo& video, double fps, std::size_t count) {
    std::vector<long long> numbers;
    const QDeadlineTimer deadline(30'000);
    while (numbers.size() < count && !deadline.hasExpired()) {
        const NativeUpdate update = video.poll();
        EXPECT_FALSE(update.error) << update.error->toStdString();
        if (!video.seeking()) {
            while (NativeFramePtr frame = video.try_pull()) {
                numbers.push_back(std::llround(static_cast<double>(frame->time_ns) / 1e9 * fps));
            }
        }
        QThread::msleep(2);
    }
    numbers.resize(std::min(numbers.size(), count));
    return numbers;
}

// The PC's GPUs, by DXGI's order: each's number and name. Windows' software
// renderer is left out. (The Mac's one GPU on macOS.)
std::vector<std::pair<unsigned, std::string>> gpus() {
#ifndef _WIN32
    return {{0, "Metal"}};
#else
    std::vector<std::pair<unsigned, std::string>> found;
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        return found;
    }
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    for (unsigned index = 0; SUCCEEDED(factory->EnumAdapters1(index, &adapter)); ++index) {
        DXGI_ADAPTER_DESC1 description{};
        adapter->GetDesc1(&description);
        if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            found.emplace_back(index, QString::fromWCharArray(description.Description).toStdString());
        }
        adapter.Reset();
    }
    return found;
#endif
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
#ifdef __APPLE__
        // A virtual Mac's (GitHub's runners'): its VideoToolbox gives H.264's
        // and VP9's pictures converted on the way, not exactly the decoded
        // ones a Mac's GPU gives, which these tests hold to FFmpeg's.
        int virtual_machine = 0;
        std::size_t size = sizeof virtual_machine;
        if (sysctlbyname("kern.hv_vmm_present", &virtual_machine, &size, nullptr, 0) == 0 && virtual_machine != 0) {
            GTEST_SKIP() << "a virtual machine's GPU decoders are not a Mac's";
        }
#endif
        for (const auto& [index, name] : gpus()) {
            try {
#ifdef _WIN32
                devices_.emplace_back(name, std::make_shared<NativeDevice>(index));
#else
                static_cast<void>(index);
                devices_.emplace_back(name, std::make_shared<NativeDevice>());
#endif
            } catch (const std::exception& error) {
                ADD_FAILURE() << name << ": " << error.what();
            }
        }
        if (devices_.empty()) {
            GTEST_SKIP() << "no GPU device";
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
    std::vector<std::pair<std::string, std::shared_ptr<NativeDevice>>> devices_;
};

}  // namespace

// Frame 30 of each clip, after a seek, is FFmpeg's frame 30 sample for
// sample, for each codec native playback plays: an MP4 with B-frames and
// 8-bit H.264 and 10-bit HEVC in Matroska, which every Direct3D 11 GPU
// decodes; AV1, VP9 and MPEG-2, which most do; 10-bit H.264 and VVC, which
// none does (FFmpeg's software decoder, the picture uploaded); and 10-bit
// ProRes 4:2:2 (software too, its picture converted to 4:2:0 as GStreamer
// converted it).
// MPEG-2's standard leaves its inverse transform's last bit to each decoder,
// so a GPU's MPEG-2 frame is FFmpeg's to within a few codes (RTX 5090: 1.7%
// of samples, at most 4 codes, 0.018 mean square; Intel's graphics: 0.4%, 2,
// 0.004): it is held to 0.5, where another frame is far more.
TEST_F(NativeDecoderTest, TheFramesAreFfmpegsCountedFromTheFirst) {
    struct Case {
        std::string name;
        std::vector<std::string> encoding;  // FFmpeg's options after the input
        std::string decoder;                // the decoder every GPU uses, or ""
        bool exact = true;                  // false: MPEG-2's own leeway
    };
#ifdef _WIN32
    const std::string gpu_decoding = "GPU: d3d11va ";
#else
    const std::string gpu_decoding = "GPU: videotoolbox ";
#endif
    std::vector<Case> cases{
        {"h264.mp4", {"-c:v", "libx264", "-preset", "veryfast", "-bf", "3", "-pix_fmt", "yuv420p"}, gpu_decoding + "h264"},
        {"hevc.mkv", {"-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "bframes=4:log-level=error",
                      "-pix_fmt", "yuv420p10le"},
         gpu_decoding + "hevc"},
        {"av1.mkv", {"-c:v", "libsvtav1", "-preset", "12", "-pix_fmt", "yuv420p"}, ""},
        {"vp9.webm", {"-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p"}, ""},
        {"mpeg2.mkv", {"-c:v", "mpeg2video", "-q:v", "4", "-pix_fmt", "yuv420p"}, "", false},
        // Apple's GPUs decode 10-bit H.264 (an M1 Pro does).
#ifdef _WIN32
        {"high10.mkv", {"-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p10le"}, "software: h264"},
#else
        {"high10.mkv", {"-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p10le"}, gpu_decoding + "h264"},
#endif
        {"vvc.mkv", {"-c:v", "libvvenc", "-preset", "faster", "-pix_fmt", "yuv420p10le"}, "software: vvc"},
        {"prores.mov", {"-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"}, "software: prores"},
    };
    // A clip whose encoder this FFmpeg lacks is left out, and said so.
    std::erase_if(cases, [this](const Case& item) {
        const std::string& encoder = item.encoding[1];
        if (has_encoder(*found_, encoder)) {
            return false;
        }
        std::cerr << "[   NOTE   ] " << item.name << " left out: this FFmpeg has no " << encoder << " encoder\n";
        return true;
    });
    std::vector<FrameComparison> clips;
    for (const Case& item : cases) {
        std::vector<std::string> arguments{"-f", "lavfi", "-i", "testsrc2=s=640x360:r=24000/1001:d=2"};
        arguments.insert(arguments.end(), item.encoding.begin(), item.encoding.end());
        arguments.push_back(platform::path_text(folder_ / item.name));
        test::run_ffmpeg(*found_, arguments);
        FrameComparison comparison;
        comparison.source_info = probe_video(folder_ / item.name, found_->ffprobe);
        comparison.distorted_info = comparison.source_info;
        comparison.fps = comparison.source_info.fps;
        clips.push_back(comparison);
    }
    for (const auto& [gpu, device] : devices_) {
        SCOPED_TRACE(gpu);
        for (std::size_t index = 0; index < cases.size(); ++index) {
            const Case& item = cases[index];
            SCOPED_TRACE(item.name);
            QString status;
            const NativeFramePtr frame = decode_frame(device, clips[index], FrameSide::source, 30, status);
            ASSERT_NE(frame, nullptr) << "frame 30 never came";
            if (!item.decoder.empty()) {
                EXPECT_TRUE(status.contains(QString::fromStdString("source " + item.decoder))) << status.toStdString();
            }
            const std::vector<int> native = read_picture(*device, *frame);
            std::vector<int> wanted =
                ffmpeg_picture(*found_, clips[index].source_info.path, 30, frame->ten_bit, "", folder_ / "frame.raw");
            ASSERT_EQ(wanted.size(), native.size());
            if (item.exact) {
                EXPECT_EQ(native, wanted) << "not FFmpeg's frame 30; " << status.toStdString();
            } else {
                double squares = 0;
                for (std::size_t sample = 0; sample < native.size(); ++sample) {
                    const double difference = native[sample] - wanted[sample];
                    squares += difference * difference;
                }
                EXPECT_LT(squares / static_cast<double>(native.size()), 0.5)
                    << "not FFmpeg's frame 30; " << status.toStdString();
            }
        }
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
    std::vector<int> wanted;
    for (const auto& [gpu, device] : devices_) {
        SCOPED_TRACE(gpu);
        QString status;
        const NativeFramePtr frame = decode_frame(device, delayed, FrameSide::source, 30, status);
        ASSERT_NE(frame, nullptr);
        const std::vector<int> native = read_picture(*device, *frame);
        if (wanted.empty()) {
            wanted = ffmpeg_picture(*found_, late, 30, frame->ten_bit, "", folder_ / "late.raw");
            ASSERT_EQ(wanted.size(), native.size());
        }
        EXPECT_EQ(native, wanted) << "not the video's frame 30";
    }
}

// Played scaled down (Reference playback), the source comes at the test
// video's size, its picture scaled: within what two scalers' filters differ by,
// where another frame would be far off. Each GPU's video processor has a
// filter of its own: 41.9 dB from FFmpeg's bilinear scale on an RTX 5090,
// 40.0 on Intel's Arrow Lake graphics, against 23 for another frame; a range
// or colour converted on the way would cost far more.
TEST_F(NativeDecoderTest, TheSourceScaledDownIsTheSourcesPicture) {
    FrameComparison pair = clip("source.mkv", "libx265", "yuv420p10le");
    FrameComparison smaller;
    smaller.source_info = probe_video(
        test::make_clip(*found_, folder_ / "encode.mkv", "libx264", "yuv420p", 2.0, {}, "320x180"), found_->ffprobe);
    pair.distorted_info = smaller.source_info;
    const FrameComparison recipe = source_playback_comparison(pair, false);
    const std::vector<int> same =
        ffmpeg_picture(*found_, pair.source_info.path, 30, true, "scale=320:180:flags=bilinear", folder_ / "a.raw");
    const std::vector<int> other =
        ffmpeg_picture(*found_, pair.source_info.path, 33, true, "scale=320:180:flags=bilinear", folder_ / "b.raw");
    for (const auto& [gpu, device] : devices_) {
        SCOPED_TRACE(gpu);
        QString status;
        const NativeFramePtr frame = decode_frame(device, recipe, FrameSide::source, 30, status);
        ASSERT_NE(frame, nullptr);
        EXPECT_EQ(frame->width, 320);
        EXPECT_EQ(frame->height, 180);
        EXPECT_TRUE(frame->prepared);
        const std::vector<int> native = read_picture(*device, *frame, false);
        const auto psnr = [&native](const std::vector<int>& with) {
            double squares = 0;
            for (std::size_t index = 0; index < native.size(); ++index) {
                const double difference = native[index] - with[index];
                squares += difference * difference;
            }
            const double mse = squares / static_cast<double>(native.size());
            return mse == 0 ? 100.0 : 10 * std::log10(1023.0 * 1023.0 / mse);
        };
        ASSERT_GE(same.size(), native.size());
        ASSERT_GE(other.size(), native.size());
        EXPECT_GT(psnr(same), 35.0);
        EXPECT_GT(psnr(same), psnr(other) + 3) << "not frame 30";
    }
}

// On a GPU without an AV1 decoder, dav1d decodes AV1, as GStreamer's dav1ddec
// did: frame 30 of an 8-bit clip tagged BT.709 and of a 10-bit full-range
// one with film grain, after a seek, is FFmpeg's (whose AV1 decoder in
// software is dav1d too, the grain added), sample for sample, in the video's
// colours; and played from the start, every frame comes,
// the last ones too (dav1d gives the frames its threads still decode only
// when drained at the end). It needs dav1d.dll beside FFmpeg's libraries, as
// in the package: a development build has none unless VML_RESOURCES names
// the package's resources.
TEST_F(NativeDecoderTest, AV1InSoftwareIsDecodedByDav1d) {
    if (const std::optional<QString> problem = Dav1dVideo::unavailable()) {
        GTEST_SKIP() << problem->toStdString();
    }
    const std::shared_ptr<NativeDevice>& device = devices_.front().second;
    for (const auto& [name, options] :
         {std::pair<std::string, std::vector<std::string>>{"av1.mkv", {"-pix_fmt", "yuv420p", "-colorspace", "bt709"}},
          std::pair<std::string, std::vector<std::string>>{
              "av1-10-bit.mkv",
              {"-pix_fmt", "yuv420p10le", "-color_range", "pc", "-svtav1-params", "film-grain=8"}}}) {
        SCOPED_TRACE(name);
        std::vector<std::string> arguments{"-f", "lavfi", "-i", "testsrc2=s=640x360:r=24000/1001:d=2", "-c:v",
                                           "libsvtav1", "-preset", "12"};
        arguments.insert(arguments.end(), options.begin(), options.end());
        arguments.push_back(platform::path_text(folder_ / name));
        test::run_ffmpeg(*found_, arguments);
        FrameComparison comparison;
        comparison.source_info = probe_video(folder_ / name, found_->ffprobe);
        comparison.distorted_info = comparison.source_info;
        comparison.fps = comparison.source_info.fps;
        QString status;
        const NativeFramePtr frame = decode_frame(device, comparison, FrameSide::source, 30, status, true);
        ASSERT_NE(frame, nullptr) << "frame 30 never came";
        EXPECT_TRUE(status.contains(QStringLiteral("source software: libdav1d"))) << status.toStdString();
        const bool ten_bit = name == "av1-10-bit.mkv";
        EXPECT_EQ(frame->ten_bit, ten_bit);
        // Untagged at 360 lines, BT.601's matrix, as GStreamer guessed it.
        EXPECT_EQ(frame->matrix, ten_bit ? "bt601" : "bt709");
        EXPECT_EQ(frame->full_range, ten_bit);
        const std::vector<int> native = read_picture(*device, *frame);
        std::vector<int> wanted =
            ffmpeg_picture(*found_, comparison.source_info.path, 30, frame->ten_bit, "", folder_ / "frame.raw");
        ASSERT_EQ(wanted.size(), native.size());
        EXPECT_EQ(native, wanted) << "not FFmpeg's frame 30";

        FfmpegNativeVideo video(device, comparison, FrameSide::source, {}, 0);
        video.decode_in_software();
        video.start();
        std::vector<long long> numbers;
        bool ended = false;
        const QDeadlineTimer deadline(30'000);
        while (!ended && !deadline.hasExpired()) {
            const NativeUpdate update = video.poll();
            EXPECT_FALSE(update.error) << update.error->toStdString();
            ended = update.ended;
            while (NativeFramePtr given = video.try_pull()) {
                numbers.push_back(std::llround(static_cast<double>(given->time_ns) / 1e9 * comparison.fps));
            }
            QThread::msleep(2);
        }
        while (NativeFramePtr given = video.try_pull()) {
            numbers.push_back(std::llround(static_cast<double>(given->time_ns) / 1e9 * comparison.fps));
        }
        std::vector<long long> all(48);  // two seconds at 24000/1001: frames 0 to 47
        std::iota(all.begin(), all.end(), 0LL);
        EXPECT_TRUE(ended);
        EXPECT_EQ(numbers, all) << "not every frame, in order";
    }
}

// A seek back, once frames past it were given, gives the frames from it on
// and none decoded before it: each decoder forgets what it holds (a GPU's
// and FFmpeg's software decoder decode frames ahead, and dav1d's threads
// several at once). On each GPU's decoder, FFmpeg's software decoder, and
// dav1d where it is beside FFmpeg's libraries; and in a raw stream, whose
// frames are counted (it is decoded again from its start).
TEST_F(NativeDecoderTest, ASeekBackGivesTheFramesFromItOn) {
    struct Case {
        std::string name;
        FrameComparison comparison;
        std::shared_ptr<NativeDevice> device;
        bool software = false;
    };
    std::vector<Case> cases;
    const FrameComparison h264 = clip("seek.mkv", "libx264", "yuv420p");
    test::run_ffmpeg(*found_, {"-i", platform::path_text(h264.source_info.path), "-c", "copy", "-f", "h264",
                               platform::path_text(folder_ / "seek.264")});
    FrameComparison raw;
    raw.source_info = probe_video(folder_ / "seek.264", found_->ffprobe);
    raw.distorted_info = raw.source_info;
    raw.fps = raw.source_info.fps;
    for (const auto& [gpu, device] : devices_) {
        cases.push_back({"H.264 on " + gpu, h264, device});
        cases.push_back({"raw H.264 on " + gpu, raw, device});
    }
    cases.push_back({"H.264 in software", h264, devices_.front().second, true});
    cases.push_back({"raw H.264 in software", raw, devices_.front().second, true});
    if (!Dav1dVideo::unavailable()) {
        test::run_ffmpeg(*found_, {"-f", "lavfi", "-i", "testsrc2=s=640x360:r=24000/1001:d=2", "-c:v", "libsvtav1",
                                   "-preset", "12", "-pix_fmt", "yuv420p", platform::path_text(folder_ / "seek.webm")});
        FrameComparison av1;
        av1.source_info = probe_video(folder_ / "seek.webm", found_->ffprobe);
        av1.distorted_info = av1.source_info;
        av1.fps = av1.source_info.fps;
        cases.push_back({"AV1 by dav1d", av1, devices_.front().second, true});
    }
    for (const Case& item : cases) {
        SCOPED_TRACE(item.name);
        const double fps = item.comparison.fps;
        FfmpegNativeVideo video(item.device, item.comparison, FrameSide::source, {}, 0);
        if (item.software) {
            video.decode_in_software();
        }
        video.start();
        // Played on to frame 40, the decoder holding the frames after it.
        std::vector<long long> played;
        const QDeadlineTimer deadline(30'000);
        while ((played.empty() || played.back() < 40) && !deadline.hasExpired()) {
            EXPECT_FALSE(video.poll().error);
            if (NativeFramePtr frame = video.try_pull()) {
                played.push_back(std::llround(static_cast<double>(frame->time_ns) / 1e9 * fps));
            }
            QThread::msleep(2);
        }
        ASSERT_FALSE(played.empty());
        ASSERT_EQ(played.back(), 40);
        QThread::msleep(200);  // time to decode what it may
        video.seek(static_cast<int>(std::llround(10 * 1000 / fps)));
        EXPECT_EQ(frames_after_seek(video, fps, 5), (std::vector<long long>{10, 11, 12, 13, 14}));
    }
}

// A raw stream, with no timestamps (H.264, HEVC or VVC as x264, x265 and
// VVenC write them: .264, .265 and .266 files), plays as the Python app's
// GStreamer played it: its frames counted from its start, at the
// comparison's frame rate. A frame after a seek (which decodes the stream
// again from its start) is FFmpeg's frame of that number, luma and chroma;
// played from the start, every frame comes, in order. Raw H.264 with
// B-frames and 10-bit HEVC on each GPU's decoder, H.264 in software too, and
// the film grain test's VVC stream (fixtures/vvc_grain.266) in software.
TEST_F(NativeDecoderTest, ARawStreamsFramesAreCountedFromItsStart) {
    struct Case {
        std::string name;
        std::filesystem::path path;
        std::shared_ptr<NativeDevice> device;
        bool software = false;
        long long wanted = 30;  // the frame looked at after a seek
        std::size_t frames = 48;
    };
    const std::filesystem::path h264 = folder_ / "raw.264";
    test::run_ffmpeg(*found_, {"-f", "lavfi", "-i", "testsrc2=s=640x360:r=24000/1001:d=2", "-c:v", "libx264",
                               "-preset", "veryfast", "-bf", "3", "-pix_fmt", "yuv420p", "-f", "h264",
                               platform::path_text(h264)});
    const std::filesystem::path hevc = folder_ / "raw.265";
    test::run_ffmpeg(*found_, {"-f", "lavfi", "-i", "testsrc2=s=640x360:r=24000/1001:d=2", "-c:v", "libx265",
                               "-preset", "ultrafast", "-x265-params", "bframes=4:log-level=error", "-pix_fmt",
                               "yuv420p10le", "-f", "hevc", platform::path_text(hevc)});
    const std::filesystem::path vvc = std::filesystem::path(VML_FIXTURES_DIR) / "vvc_grain.266";
    std::vector<Case> cases;
    for (const auto& [gpu, device] : devices_) {
        cases.push_back({"H.264 on " + gpu, h264, device});
        cases.push_back({"HEVC on " + gpu, hevc, device});
    }
    cases.push_back({"H.264 in software", h264, devices_.front().second, true});
    cases.push_back({"VVC in software", vvc, devices_.front().second, true, 10, 17});
    for (const Case& item : cases) {
        SCOPED_TRACE(item.name);
        FrameComparison comparison;
        comparison.source_info = probe_video(item.path, found_->ffprobe);
        comparison.distorted_info = comparison.source_info;
        comparison.fps = comparison.source_info.fps;
        ASSERT_GT(comparison.fps, 0);
        QString status;
        const NativeFramePtr frame =
            decode_frame(item.device, comparison, FrameSide::source, item.wanted, status, item.software);
        ASSERT_NE(frame, nullptr) << "frame " << item.wanted << " never came";
        const std::vector<int> native = read_picture(*item.device, *frame);
        const std::vector<int> wanted =
            ffmpeg_picture(*found_, item.path, item.wanted, frame->ten_bit, "", folder_ / "raw.yuv");
        ASSERT_EQ(native.size(), wanted.size());
        EXPECT_EQ(native, wanted) << "not FFmpeg's frame " << item.wanted << "; " << status.toStdString();

        FfmpegNativeVideo video(item.device, comparison, FrameSide::source, {}, 0);
        if (item.software) {
            video.decode_in_software();
        }
        video.start();
        std::vector<long long> numbers;
        bool ended = false;
        const QDeadlineTimer deadline(30'000);
        while (!ended && !deadline.hasExpired()) {
            const NativeUpdate update = video.poll();
            EXPECT_FALSE(update.error) << update.error->toStdString();
            ended = update.ended;
            while (NativeFramePtr given = video.try_pull()) {
                numbers.push_back(std::llround(static_cast<double>(given->time_ns) / 1e9 * comparison.fps));
            }
            QThread::msleep(2);
        }
        while (NativeFramePtr given = video.try_pull()) {
            numbers.push_back(std::llround(static_cast<double>(given->time_ns) / 1e9 * comparison.fps));
        }
        std::vector<long long> all(item.frames);
        std::iota(all.begin(), all.end(), 0LL);
        EXPECT_TRUE(ended);
        EXPECT_EQ(numbers, all) << "not every frame, in order";
        EXPECT_EQ(video.first_frame_ms(), 0);
    }
}

// A VVC video's film grain is drawn as the Python app draws it: played from
// the start, every frame of the film grain test's 8-bit stream
// (fixtures/vvc_grain.266) and of its 10-bit one (vvc_grain_10bit.266) is
// the Python app's frame (vvc_grain_app.npz, which the Python app drew),
// luma and chroma. That is FFmpeg's own frame where FFmpeg draws the grain
// (8-bit, a message that persists), and the grain library's where it does
// not (10-bit, a message for its own frame only), and the frame as decoded
// where there is none. It needs the grain library
// (resources/native/d3d11_film_grain.dll, scripts/build_d3d11_film_grain.ps1).
TEST_F(NativeDecoderTest, VvcFilmGrainIsDrawnAsThePythonAppDrawsIt) {
    if (const std::optional<QString> problem = NativeGrain::unavailable()) {
        GTEST_SKIP() << problem->toStdString();
    }
    const std::filesystem::path fixtures = VML_FIXTURES_DIR;
    const std::optional<std::map<std::string, NpyArray>> expected = read_npz(fixtures / "vvc_grain_app.npz");
    ASSERT_TRUE(expected.has_value());
    constexpr int width = 64;
    constexpr int height = 48;
    constexpr std::size_t samples = width * height * 3 / 2;
    for (const auto& [name, frames, ten_bit] :
         {std::tuple<std::string, std::string, bool>{"vvc_grain.266", "grained_8", false},
          std::tuple<std::string, std::string, bool>{"vvc_grain_10bit.266", "grained_10", true}}) {
        SCOPED_TRACE(name);
        const NpyArray& wanted = expected->at(frames);
        ASSERT_EQ(wanted.shape.size(), 2u);
        ASSERT_EQ(wanted.shape[1], samples);
        ASSERT_EQ(wanted.descr, ten_bit ? "<u2" : "|u1");
        FrameComparison comparison;
        comparison.source_info = probe_video(fixtures / name, found_->ffprobe);
        comparison.distorted_info = comparison.source_info;
        comparison.fps = comparison.source_info.fps;
        for (const auto& [gpu, device] : devices_) {
            SCOPED_TRACE(gpu);
            FfmpegNativeVideo video(device, comparison, FrameSide::source, {}, 0);
            video.start();
            std::vector<std::vector<int>> pictures;
            QString status;
            bool ended = false;
            const QDeadlineTimer deadline(30'000);
            while (!ended && !deadline.hasExpired()) {
                const NativeUpdate update = video.poll();
                EXPECT_FALSE(update.error) << update.error->toStdString();
                if (update.status) {
                    status = *update.status;
                }
                ended = update.ended;
                while (NativeFramePtr frame = video.try_pull()) {
                    pictures.push_back(read_picture(*device, *frame));
                }
                QThread::msleep(2);
            }
            while (NativeFramePtr frame = video.try_pull()) {
                pictures.push_back(read_picture(*device, *frame));
            }
            EXPECT_TRUE(status.contains(QStringLiteral("film grain synthesized"))) << status.toStdString();
            ASSERT_EQ(pictures.size(), wanted.shape[0]);
            for (std::size_t index = 0; index < pictures.size(); ++index) {
                // The Python app's planes, Y, Cb and Cr, in the order the
                // read-back gives them: Cb and Cr side by side.
                std::vector<int> planar(samples);
                for (std::size_t at = 0; at < samples; ++at) {
                    const std::size_t offset = (index * samples + at) * (ten_bit ? 2 : 1);
                    const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(wanted.data[i]); };
                    planar[at] = ten_bit ? byte(offset) | byte(offset + 1) << 8 : byte(offset);
                }
                std::vector<int> picture(planar.begin(), planar.begin() + width * height);
                const std::size_t chroma = (width / 2) * (height / 2);
                for (std::size_t at = 0; at < chroma; ++at) {
                    picture.push_back(planar[width * height + at]);
                    picture.push_back(planar[width * height + chroma + at]);
                }
                EXPECT_EQ(pictures[index], picture) << "frame " << index;
            }
        }
    }
}

#ifdef _WIN32
// A texture's samples read back as stored: an NV12 or P010 texture's luma
// rows (`height` of `width`), then its chroma rows, two samples side by side.
std::vector<int> read_stored(NativeDevice& device, ID3D11Texture2D* texture, int width, int height) {
    D3D11_TEXTURE2D_DESC description{};
    texture->GetDesc(&description);
    D3D11_TEXTURE2D_DESC staged = description;
    staged.Usage = D3D11_USAGE_STAGING;
    staged.BindFlags = 0;
    staged.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    std::vector<int> samples;
    if (FAILED(device.device()->CreateTexture2D(&staged, nullptr, &staging))) {
        ADD_FAILURE() << "no staging texture";
        return samples;
    }
    const std::lock_guard lock(device.lock());
    device.context()->CopyResource(staging.Get(), texture);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(device.context()->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        ADD_FAILURE() << "the texture could not be read back";
        return samples;
    }
    const bool wide = description.Format == DXGI_FORMAT_P010;
    const auto* base = static_cast<const unsigned char*>(mapped.pData);
    const auto rows = [&](const unsigned char* plane, int count, int per_row) {
        for (int row = 0; row < count; ++row) {
            const unsigned char* line = plane + static_cast<std::size_t>(mapped.RowPitch) * row;
            for (int x = 0; x < per_row; ++x) {
                samples.push_back(wide ? line[2 * x] | (line[2 * x + 1] << 8) : line[x]);
            }
        }
    };
    rows(base, height, width);
    rows(base + static_cast<std::size_t>(mapped.RowPitch) * description.Height, (height + 1) / 2,
         2 * ((width + 1) / 2));
    device.context()->Unmap(staging.Get(), 0);
    return samples;
}

// VVC film grain on video other than 4:2:0 of 8 or 10 bits, which VVenC
// does not make, so the frames are made here: 4:2:2, 4:4:4 (of an odd size),
// 4:0:0 and 12-bit. Each plane gets the grain library's own drawing of it in
// memory (vmaf_grain_planes, the Python app's drawing), and the texture shown
// holds it as NV12 or P010: the luma as drawn, the chroma averaged down to
// 4:2:0, the last column and row repeated past an odd edge, and neutral
// chroma for 4:0:0.
TEST_F(NativeDecoderTest, FilmGrainOnOtherFormatsIsTheLibrarysDrawingShownAs420) {
    if (const std::optional<QString> problem = NativeGrain::unavailable()) {
        GTEST_SKIP() << problem->toStdString();
    }
    const platform::DynamicLibrary library(resource_folder() / "native" / "d3d11_film_grain.dll");
    const auto reference =
        library.function<int (*)(const void* const*, void* const*, int, const GrainParams*, int)>("vmaf_grain_planes");
    FilmGrainMessage message;
    message.log2_scale_factor = 3;
    message.components = {std::vector<FilmGrainInterval>{{0, 255, {120, 8, 14}}},
                          std::vector<FilmGrainInterval>{{0, 255, {60, 14, 2}}},
                          std::vector<FilmGrainInterval>{{0, 255, {60, 2, 14}}}};
    struct Format {
        const char* name;
        int width;
        int height;
        int depth;
        int count;
        std::pair<int, int> shift;
    };
    for (const Format& format : {Format{"4:2:2, 10-bit", 64, 48, 10, 3, {1, 0}},
                                 Format{"4:4:4, 12-bit, 65x47", 65, 47, 12, 3, {0, 0}},
                                 Format{"4:2:0, 12-bit", 64, 48, 12, 3, {1, 1}},
                                 Format{"4:0:0, 8-bit", 64, 48, 8, 1, {0, 0}},
                                 Format{"4:2:0, 8-bit", 64, 48, 8, 3, {1, 1}}}) {
        SCOPED_TRACE(format.name);
        const std::vector<std::pair<int, int>> sizes =
            grain_plane_sizes(format.width, format.height, format.shift, format.count);
        const std::optional<GrainParams> params = grain_params(message, 37, format.depth, sizes, format.shift);
        ASSERT_TRUE(params.has_value());
        const bool wide = format.depth > 8;
        const std::size_t sample = wide ? 2 : 1;
        // The frame's planes: a pattern of codes, the same in every run; and
        // the library's drawing of the grain on them.
        std::vector<std::vector<unsigned char>> planes(sizes.size()), drawn(sizes.size());
        std::vector<const void*> in;
        std::vector<void*> out;
        for (std::size_t c = 0; c < sizes.size(); ++c) {
            const auto [plane_width, plane_height] = sizes[c];
            for (int y = 0; y < plane_height; ++y) {
                for (int x = 0; x < plane_width; ++x) {
                    const int code = ((x * 7 + y * 13 + static_cast<int>(c) * 29) * 37) % (1 << format.depth);
                    planes[c].push_back(static_cast<unsigned char>(code & 0xFF));
                    if (wide) {
                        planes[c].push_back(static_cast<unsigned char>(code >> 8));
                    }
                }
            }
            drawn[c].resize(planes[c].size());
            in.push_back(planes[c].data());
            out.push_back(drawn[c].data());
        }
        ASSERT_EQ(reference(in.data(), out.data(), format.count, &*params, 0), 0);
        ASSERT_NE(drawn[0], planes[0]) << "no grain was drawn";
        const auto code = [&](std::size_t c, int x, int y) {
            const std::size_t at = (static_cast<std::size_t>(y) * sizes[c].first + x) * sample;
            return wide ? drawn[c][at] | (drawn[c][at + 1] << 8) : drawn[c][at];
        };
        const int scale = wide ? 1 << (16 - format.depth) : 1;
        for (const auto& [gpu, device] : devices_) {
            SCOPED_TRACE(gpu);
            NativeGrain grain(device->device());
            std::array<Microsoft::WRL::ComPtr<ID3D11Texture2D>, 3> textures;
            for (std::size_t c = 0; c < sizes.size(); ++c) {
                D3D11_TEXTURE2D_DESC description{};
                description.Width = static_cast<UINT>(sizes[c].first);
                description.Height = static_cast<UINT>(sizes[c].second);
                description.MipLevels = 1;
                description.ArraySize = 1;
                description.Format = wide ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
                description.SampleDesc.Count = 1;
                description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                const D3D11_SUBRESOURCE_DATA data{planes[c].data(), static_cast<UINT>(sizes[c].first * sample), 0};
                ASSERT_TRUE(SUCCEEDED(device->device()->CreateTexture2D(&description, &data, &textures[c])));
            }
            D3D11_TEXTURE2D_DESC shown{};
            shown.Width = static_cast<UINT>((format.width + 1) & ~1);
            shown.Height = static_cast<UINT>((format.height + 1) & ~1);
            shown.MipLevels = 1;
            shown.ArraySize = 1;
            shown.Format = wide ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
            shown.SampleDesc.Count = 1;
            shown.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
            Microsoft::WRL::ComPtr<ID3D11Texture2D> target;
            ASSERT_TRUE(SUCCEEDED(device->device()->CreateTexture2D(&shown, nullptr, &target)));
            {
                const std::lock_guard lock(device->lock());
                ID3D11Texture2D* const given[3] = {textures[0].Get(), textures[1].Get(), textures[2].Get()};
                const std::optional<QString> failed =
                    grain.draw(given, format.count, *params, format.shift, target.Get());
                ASSERT_FALSE(failed.has_value()) << failed->toStdString();
            }
            const std::vector<int> stored = read_stored(*device, target.Get(), format.width, format.height);
            ASSERT_EQ(stored.size(), static_cast<std::size_t>(format.width * format.height +
                                                              (format.height + 1) / 2 * 2 * ((format.width + 1) / 2)));
            int wrong = 0;
            for (int y = 0; y < format.height; ++y) {
                for (int x = 0; x < format.width; ++x) {
                    wrong += stored[static_cast<std::size_t>(y * format.width + x)] != code(0, x, y) * scale;
                }
            }
            EXPECT_EQ(wrong, 0) << "luma samples differ";
            // Each chroma sample: the average of the chroma samples its 2x2
            // luma square covers, rounded either way, or the middle code.
            const int step_x = 2 >> format.shift.first;
            const int step_y = 2 >> format.shift.second;
            wrong = 0;
            const std::size_t chroma_at = static_cast<std::size_t>(format.width * format.height);
            const int chroma_width = (format.width + 1) / 2;
            for (int y = 0; y < (format.height + 1) / 2; ++y) {
                for (int x = 0; x < chroma_width; ++x) {
                    for (std::size_t c = 1; c < 3; ++c) {
                        double expected = static_cast<double>(1 << (format.depth - 1)) * scale;
                        if (format.count == 3) {
                            double sum = 0;
                            for (int dy = 0; dy < step_y; ++dy) {
                                for (int dx = 0; dx < step_x; ++dx) {
                                    sum += code(c, std::min(x * step_x + dx, sizes[c].first - 1),
                                                std::min(y * step_y + dy, sizes[c].second - 1));
                                }
                            }
                            expected = sum / (step_x * step_y) * scale;
                        }
                        const int got = stored[chroma_at + static_cast<std::size_t>(y * 2 * chroma_width + 2 * x) + c - 1];
                        wrong += std::abs(got - expected) > 0.5;
                    }
                }
            }
            EXPECT_EQ(wrong, 0) << "chroma samples differ";
        }
    }
}
#endif

// AV1's film grain in a VVC video (AFGS1), which the grain library does not
// draw, is left to FFmpeg, which does: frames 0 and 9 of a VVC clip with it
// (fixtures/vvc_afgs1.mkv: FFmpeg's test pattern, 10-bit, its grain
// from the AFGS1 Grain Workbench) are ffmpeg.exe's, luma and chroma, the
// grain drawn on them. (The Python app's GStreamer draws none.)
TEST_F(NativeDecoderTest, Afgs1FilmGrainIsLeftToFfmpeg) {
    FrameComparison comparison;
    comparison.source_info = probe_video(std::filesystem::path(VML_FIXTURES_DIR) / "vvc_afgs1.mkv", found_->ffprobe);
    comparison.distorted_info = comparison.source_info;
    comparison.fps = comparison.source_info.fps;
    for (const long long wanted : {0LL, 9LL}) {
        SCOPED_TRACE(wanted);
        QString status;
        const NativeFramePtr frame = decode_frame(devices_.front().second, comparison, FrameSide::source, wanted, status);
        ASSERT_NE(frame, nullptr);
        const std::vector<int> native = read_picture(*devices_.front().second, *frame);
        const std::vector<int> ffmpeg =
            ffmpeg_picture(*found_, comparison.source_info.path, wanted, frame->ten_bit, "", folder_ / "afgs1.yuv");
        ASSERT_EQ(native.size(), ffmpeg.size());
        EXPECT_EQ(native, ffmpeg) << status.toStdString();
    }
}

}  // namespace vml::app
