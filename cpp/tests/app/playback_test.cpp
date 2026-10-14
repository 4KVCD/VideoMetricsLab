// Video Compare's playback with FFmpeg: the Python app's tests of the same
// behaviour (tests/test_playback_recovery.py, test_playback_pool.py), for
// what is ported (playback_worker.h, video_compare_view.h).
#include "video_compare_view.h"

#include <QApplication>
#include <QDeadlineTimer>
#include <QPointer>
#include <QThread>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "test_videos.h"
#include "vml/ffprobe.h"
#include "vml/platform/system.h"

namespace vml::app {

namespace {

// A decoder whose frames the test hands it: its thread waits until it is
// cancelled or finished.
class FakeDecoder : public PlaybackDecoder {
public:
    FakeDecoder(long long start, QObject* parent) : PlaybackDecoder(parent), start_frame(start) {}
    ~FakeDecoder() override {
        finish();
        wait();
    }

    void cancel() override {
        cancelled = true;
        finish();
    }
    void finish() {
        {
            const std::lock_guard lock(mutex_);
            stop_ = true;
        }
        stopped_.notify_all();
    }
    std::vector<PlaybackFrame> drain_through(long long frame) override {
        std::vector<PlaybackFrame> due;
        std::vector<PlaybackFrame> later;
        for (PlaybackFrame& item : frames) {
            (item.number <= frame ? due : later).push_back(std::move(item));
        }
        frames = std::move(later);
        return due;
    }
    void recycle(FramePixels /*pixels*/) override {}
    std::optional<long long> latest_frame_number() const override {
        if (frames.empty()) {
            return std::nullopt;
        }
        return std::max_element(frames.begin(), frames.end(), [](const auto& a, const auto& b) {
                   return a.number < b.number;
               })->number;
    }

    long long start_frame;
    bool cancelled = false;
    std::vector<PlaybackFrame> frames;

protected:
    void run() override {
        std::unique_lock lock(mutex_);
        stopped_.wait(lock, [this] { return stop_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable stopped_;
    bool stop_ = false;
};

}  // namespace

// The view's parts, for its tests.
class PlaybackTest : public ::testing::Test {
protected:
    static void tick(VideoCompareView& view) { view.tick(); }
    static long long presented(const VideoCompareView& view) { return view.presented_; }
    static const QString& last_status(const VideoCompareView& view) { return view.last_status_; }
    static void set_clock_frame(VideoCompareView& view, long long frame) { view.clock_frame_ = frame; }
    static FakeDecoder& decoder(VideoCompareView& view, const std::string& prefix) {
        for (const auto& [key, worker] : view.pool_) {
            if (key.starts_with(prefix)) {
                return *static_cast<FakeDecoder*>(worker);
            }
        }
        throw std::logic_error("no decoder " + prefix);
    }
    static std::vector<FakeDecoder*> decoders(VideoCompareView& view) {
        std::vector<FakeDecoder*> all;
        for (const auto& [key, worker] : view.pool_) {
            all.push_back(static_cast<FakeDecoder*>(worker));
        }
        return all;
    }
    static const FramePixels& shown(const StreamSurface& surface) { return surface.pixels_; }
    static const StreamSurface& source_surface(const VideoCompareView& view) { return *view.source_surface_; }
    static const StreamSurface& distorted_surface(const VideoCompareView& view) { return *view.distorted_surface_; }

    // Lets the events run until `done`, for at most 30 seconds.
    static bool wait_until(const std::function<bool()>& done) {
        const QDeadlineTimer deadline(30'000);
        while (!done()) {
            if (deadline.hasExpired()) {
                return false;
            }
            QApplication::processEvents(QEventLoop::AllEvents, 20);
            QThread::msleep(5);
        }
        return true;
    }

    // `count` test videos of 16x12 HDR10 at 24 fps beside one source, their
    // files there (empty: nothing reads them here).
    std::vector<FrameComparison> series(int count) const {
        VideoInfo source;
        source.path = folder_ / "source.mkv";
        source.width = 16;
        source.height = 12;
        source.fps = 24;
        source.duration = 10;
        source.nb_frames = 240;
        source.codec_name = "hevc";
        source.pix_fmt = "yuv420p10le";
        source.color_transfer = "smpte2084";
        source.color_primaries = "bt2020";
        source.color_space = "bt2020nc";
        std::ofstream(source.path).put('\0');
        std::vector<FrameComparison> items;
        for (int index = 0; index < count; ++index) {
            FrameComparison comparison;
            comparison.source_info = source;
            comparison.distorted_info = source;
            comparison.distorted_info.path = folder_ / (std::to_string(index) + ".mkv");
            std::ofstream(comparison.distorted_info.path).put('\0');
            comparison.fps = 24;
            comparison.frame_count = 240;
            items.push_back(comparison);
        }
        return items;
    }

    // A view of `count` test videos at 1 s, its decoders stand-ins.
    std::unique_ptr<VideoCompareView> make_view(int count) {
        auto view = std::make_unique<VideoCompareView>([] { return std::string("ffmpeg"); }, zoom_);
        view->make_decoder = [](const FrameComparison&, FrameSide, long long start, const PreviewColorSettings&,
                                std::optional<Size>, QObject* parent) { return new FakeDecoder(start, parent); };
        view->set_audio_enabled(false);
        items_ = series(count);
        EXPECT_TRUE(view->load(items_.front(), 1000, false, {}, items_, true));
        return view;
    }

    test::Folder folder_;
    Zoom zoom_;
    std::vector<FrameComparison> items_;
};

namespace {

FramePixels frame_pixels() { return std::make_shared<std::vector<unsigned char>>(16 * 12 * 4); }

}  // namespace

// A frame is shown only when the source and the test video both have it.
TEST_F(PlaybackTest, APairIsShownOnlyAtAFrameBothVideosHave) {
    const auto view = make_view(1);
    const FramePixels pixels = frame_pixels();
    decoder(*view, "s:").frames = {{24, pixels}};
    decoder(*view, "d:0").frames = {{25, pixels}};
    tick(*view);
    EXPECT_EQ(presented(*view), -1);
    decoder(*view, "d:0").frames.insert(decoder(*view, "d:0").frames.begin(), {24, pixels});
    tick(*view);
    EXPECT_EQ(presented(*view), 24);
    EXPECT_EQ(shown(source_surface(*view)), pixels);
    EXPECT_EQ(shown(distorted_surface(*view)), pixels);
}

// A seek starts every decoder again at the new frame; a pause keeps them,
// and says so at once (paused by minimizing the window, it went on saying
// "Playing").
TEST_F(PlaybackTest, ASeekReplacesEveryDecoderAndAPauseDoesNot) {
    const auto view = make_view(2);
    const std::vector<FakeDecoder*> old = decoders(*view);
    view->set_playing(true);
    EXPECT_TRUE(last_status(*view).startsWith("Playing")) << last_status(*view).toStdString();
    view->set_playing(false);
    EXPECT_EQ(decoders(*view), old);
    EXPECT_TRUE(last_status(*view).startsWith("Paused")) << last_status(*view).toStdString();
    view->set_position(2000);
    for (const FakeDecoder* worker : old) {
        EXPECT_TRUE(worker->cancelled);
    }
    ASSERT_FALSE(decoders(*view).empty());
    for (const FakeDecoder* worker : decoders(*view)) {
        EXPECT_EQ(std::find(old.begin(), old.end(), worker), old.end()) << "an old decoder kept";
        EXPECT_EQ(worker->start_frame, 48);
    }
}

// A faster video's frames are not dropped while the other is still decoding
// them: the pair goes on frame by frame.
TEST_F(PlaybackTest, AFasterVideosFramesWaitForItsPartner) {
    const auto view = make_view(1);
    set_clock_frame(*view, 30);
    const FramePixels pixels = frame_pixels();
    FakeDecoder& source = decoder(*view, "s:");
    FakeDecoder& test = decoder(*view, "d:0");
    source.frames = {{24, pixels}, {25, pixels}, {26, pixels}};
    test.frames = {{24, pixels}};
    tick(*view);
    EXPECT_EQ(presented(*view), 24);
    ASSERT_EQ(source.frames.size(), 2U);
    EXPECT_EQ(source.frames[0].number, 25);
    EXPECT_EQ(source.frames[1].number, 26);
    test.frames = {{25, pixels}};
    tick(*view);
    EXPECT_EQ(presented(*view), 25);
}

// A failed decoder's message names a test video by its number in the list,
// and a failed source as the pair shown's (the Python app gave its own key
// for the decoder: "Video ('distorted', 0) could not play").
TEST_F(PlaybackTest, AVideoThatCannotPlayIsNamedAsTheUserKnowsIt) {
    const auto view = make_view(2);
    std::vector<QString> said;
    QObject::connect(view.get(), &VideoCompareView::status_changed, view.get(),
                     [&said](const QString& status) { said.push_back(status); });
    emit decoder(*view, "d:0").failed("device lost");
    emit decoder(*view, "s:").failed("no such file");
    EXPECT_EQ(said, (std::vector<QString>{"Video 1 could not play: device lost", "Selected video failed: no such file"}));
}

// A decoder that has finished (at the end of a short clip, or failed) is
// deleted once the test video shown no longer needs it. It was kept until
// the window closed, with the frames it had queued: one more decoder at
// every switch of test video (in the Python app too).
TEST_F(PlaybackTest, AFinishedDecoderNoLongerWantedIsLetGo) {
    const auto view = make_view(3);
    view->set_decoded_videos(1);
    const QPointer<PlaybackDecoder> finished = &decoder(*view, "d:0");
    decoder(*view, "d:0").finish();
    ASSERT_TRUE(wait_until([&] { return !finished->isRunning(); }));
    QApplication::processEvents();  // the view hears that it has finished
    EXPECT_TRUE(view->load(items_[1], 1000, false, {}, items_, true));
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    EXPECT_TRUE(finished.isNull()) << "a finished decoder kept";
}

// Zoomed, playback's frames are drawn at the tab's zoom: a source decoded at
// twice the size at the same size, one of another shape keeping its shape.
TEST_F(PlaybackTest, BothVideosArePlacedByTheSharedZoom) {
    const auto view = make_view(1);
    view->resize(20, 10);
    zoom_.factor = 2.0;  // the 16 x 12 comparison at 32 x 24
    const double ratio = view->devicePixelRatioF();
    const std::optional<Placement> place = view->zoom_placement({16, 12});
    ASSERT_TRUE(place);
    EXPECT_DOUBLE_EQ(place->width, 32 / ratio);
    EXPECT_DOUBLE_EQ(place->height, 24 / ratio);
    const std::optional<Placement> larger = view->zoom_placement({32, 24});
    ASSERT_TRUE(larger);
    EXPECT_DOUBLE_EQ(larger->width, place->width);
    EXPECT_DOUBLE_EQ(larger->height, place->height);
    const std::optional<Placement> taller = view->zoom_placement({16, 16});
    ASSERT_TRUE(taller);
    EXPECT_DOUBLE_EQ(taller->width, 32 / ratio);
    EXPECT_DOUBLE_EQ(taller->height, 32 / ratio);
}

namespace {

// The PSNR of two RGBA frames' colours, in dB.
double psnr(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    double squares = 0;
    std::size_t count = 0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        if (index % 4 == 3) {
            continue;  // the alpha
        }
        const double difference = static_cast<double>(a[index]) - b[index];
        squares += difference * difference;
        ++count;
    }
    const double mse = squares / static_cast<double>(count);
    return mse == 0 ? 100.0 : 10 * std::log10(255.0 * 255.0 / mse);
}

}  // namespace

// Two real clips of FFmpeg's test pattern (an MP4 source, whose video starts
// after the file's start, and an MKV encode), played by FFmpeg: the frame
// shown of each is the same frame of the picture, counted from each video's
// first frame, at a frame paused on and after a seek. The pair at frame N
// matches far better than the source at N against the encode at N + 3.
TEST_F(PlaybackTest, ARealPairShowsTheSameFrameOfBothVideos) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    // The source's video starts 0.25 s (six frames) after its soundtrack.
    const std::filesystem::path pattern =
        test::make_clip(*found, folder_ / "pattern.mp4", "libx264", "yuv420p", 1.0, {}, "160x90");
    const std::filesystem::path source = folder_ / "source.mp4";
    test::run_ffmpeg(*found, {"-itsoffset", "0.25", "-i", platform::path_text(pattern), "-f", "lavfi", "-i",
                              "anullsrc=r=48000:cl=mono", "-t", "1.3", "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                              "-c:a", "aac", platform::path_text(source)});
    FrameComparison comparison;
    comparison.source_info = probe_video(source, found->ffprobe);
    ASSERT_NEAR(comparison.source_info.start_offset, 0.25, 0.01);
    comparison.distorted_info = probe_video(
        test::make_clip(*found, folder_ / "encode.mkv", "libx265", "yuv420p", 1.0, {"-crf", "30"}, "160x90"),
        found->ffprobe);
    comparison.fps = comparison.distorted_info.fps;
    comparison.frame_count = 23;
    auto view = std::make_unique<VideoCompareView>([&found] { return found->ffmpeg; }, zoom_);
    view->set_audio_enabled(false);
    view->resize(160, 90);
    const auto frame_at = [&](long long number) {
        const int position = static_cast<int>(std::nearbyint(number / comparison.fps * 1000));
        if (presented(*view) < 0) {
            EXPECT_TRUE(view->load(comparison, position, false, {}, {comparison}, true));
        } else {
            view->set_position(position);
        }
        EXPECT_TRUE(wait_until([&] {
            tick(*view);
            return presented(*view) == number;
        })) << last_status(*view).toStdString();
        return std::pair(*shown(source_surface(*view)), *shown(distorted_surface(*view)));
    };
    const auto [source_10, test_10] = frame_at(10);
    const auto [source_13, test_13] = frame_at(13);
    const double same = psnr(source_10, test_10);
    EXPECT_GT(same, psnr(source_10, test_13) + 3) << "the encode's frame is not the source's";
    EXPECT_GT(psnr(source_13, test_13), psnr(source_13, test_10) + 3);
    EXPECT_GT(same, 25);
}

namespace {

// What each decoding attempt of a worker was: its way and first frame.
std::vector<std::string> ways_tried(const std::filesystem::path& log) {
    std::vector<std::string> ways;
    for (const auto& entry : std::filesystem::directory_iterator(log)) {
        std::ifstream file(entry.path());
        std::string way;
        std::getline(file, way);
        ways.push_back(way);
    }
    // In the order tried: each starts after the last one's frames.
    std::sort(ways.begin(), ways.end(), [](const std::string& a, const std::string& b) {
        return std::stoll(a.substr(a.find(' ') + 1)) < std::stoll(b.substr(b.find(' ') + 1));
    });
    return ways;
}

}  // namespace

// A decoder that fails gives way to the next way of decoding, from the first
// frame not decoded: the frames queued are kept, none is decoded twice, and
// the ways tried are bounded: NVIDIA's decoder handing pictures to the GPU,
// then through memory, then the CPU's decoder, then the CPU for everything.
// FFmpeg here is a stand-in that writes numbered frames and fails on cue.
TEST_F(PlaybackTest, AFailedDecoderResumesDownABoundedLadderWithoutReplaying) {
    FrameComparison comparison = series(1).front();
    comparison.source_info.width = comparison.source_info.height = 2;
    comparison.distorted_info.width = comparison.distorted_info.height = 2;
    const test::ScopedVariable bytes("VML_TEST_FRAME_BYTES", "16");
    const test::ScopedVariable fps("VML_TEST_FPS", "24");
    struct Run {
        std::vector<PlaybackFrame> frames;
        std::vector<QString> failures;
        bool ended = false;
        std::vector<QString> errors;
    };
    int runs = 0;
    const auto run = [&](const std::string& spec, std::optional<std::string> hwaccel) {
        const std::filesystem::path log = folder_ / ("log " + std::to_string(++runs));
        std::filesystem::create_directories(log);
        const test::ScopedVariable logged_to("VML_TEST_TOOL_LOG", platform::path_text(log));
        const test::ScopedVariable playback("VML_TEST_PLAYBACK", spec);
        StreamDecodeWorker worker(VML_TEST_CHILD, comparison, FrameSide::source, 100, {},
                                  HardwareDecoding{hwaccel, std::nullopt}, std::nullopt);
        Run result;
        QObject::connect(&worker, &PlaybackDecoder::failed, &worker,
                         [&result](const QString& error) { result.failures.push_back(error); });
        worker.start();
        const bool done = wait_until([&] {
            for (PlaybackFrame& frame : worker.drain_through(LLONG_MAX)) {
                result.frames.push_back(std::move(frame));
            }
            return worker.isFinished();
        });
        EXPECT_TRUE(done) << "the worker did not end";
        for (PlaybackFrame& frame : worker.drain_through(LLONG_MAX)) {
            result.frames.push_back(std::move(frame));
        }
        QApplication::processEvents();  // the failure signal
        result.ended = worker.ended();
        result.errors = worker.attempt_errors();
        return std::pair(result, ways_tried(log));
    };
    {
        SCOPED_TRACE("the device lost after frames");
        const auto [result, ways] = run("interop=1:1:2:VK_ERROR_DEVICE_LOST;transfer=2:0:0:", "cuda");
        EXPECT_EQ(ways, (std::vector<std::string>{"interop 100", "transfer 101"}));
        ASSERT_EQ(result.frames.size(), 3U);
        for (std::size_t index = 0; index < 3; ++index) {
            EXPECT_EQ(result.frames[index].number, static_cast<long long>(100 + index));
            EXPECT_EQ((*result.frames[index].pixels)[0], static_cast<unsigned char>((100 + index) % 251));
        }
        EXPECT_TRUE(result.failures.empty());
        EXPECT_TRUE(result.ended);
        EXPECT_EQ(result.errors, std::vector<QString>{"VK_ERROR_DEVICE_LOST"});
    }
    for (const auto& [hwaccel, ladder] :
         {std::pair<std::optional<std::string>, std::vector<std::string>>{"cuda", {"interop", "transfer", "software", "cpu"}},
          {"d3d11va", {"transfer", "software", "cpu"}},
          {std::nullopt, {"software", "cpu"}}}) {
        SCOPED_TRACE("every way fails, " + hwaccel.value_or("no hardware decoder"));
        const auto [result, ways] =
            run("interop=1:1:0:device lost;transfer=1:1:0:device lost;software=1:1:0:device lost;cpu=1:1:0:device lost",
                hwaccel);
        std::vector<std::string> expected;
        for (std::size_t index = 0; index < ladder.size(); ++index) {
            expected.push_back(ladder[index] + " " + std::to_string(100 + index));
        }
        EXPECT_EQ(ways, expected);
        ASSERT_EQ(result.frames.size(), ladder.size());
        for (std::size_t index = 0; index < ladder.size(); ++index) {
            EXPECT_EQ(result.frames[index].number, static_cast<long long>(100 + index));
        }
        EXPECT_EQ(result.failures, std::vector<QString>{"device lost"});
        EXPECT_FALSE(result.ended);
    }
}

}  // namespace vml::app
