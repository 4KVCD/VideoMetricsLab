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

}  // namespace

namespace {

}  // namespace

}  // namespace vml::app
