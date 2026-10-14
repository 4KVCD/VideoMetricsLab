// Video Compare's native playback pool: the Python app's tests of the same
// behaviour (tests/test_locked_native_pool.py), for what is ported
// (native_pool.h), with stand-ins for the decoders, the presenter and the
// soundtrack.
#include "native_pool.h"

#include <gtest/gtest.h>

#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "test_videos.h"
#include "video_compare_view.h"

namespace vml::app {

namespace {

// A decoder whose frames the test hands it: its thread waits until cancelled.
class FakeVideo : public NativeVideo {
public:
    FakeVideo(std::optional<std::array<float, 16>> shading, QObject* parent)
        : NativeVideo(parent), shading_(std::move(shading)) {}
    ~FakeVideo() override {
        cancel();
        wait();
    }
    void seek(int position_ms) override { sought.push_back(position_ms); }
    bool seeking() const override { return seeking_now; }
    NativeFramePtr try_pull() override {
        ++pulls;
        if (queue.empty()) {
            return nullptr;
        }
        NativeFramePtr frame = queue.front();
        queue.pop_front();
        return frame;
    }
    NativeUpdate poll() override { return {}; }
    std::optional<int> first_frame_ms() const override { return std::nullopt; }
    const std::optional<std::array<float, 16>>& shading() const override { return shading_; }
    void cancel() override {
        {
            const std::lock_guard lock(mutex_);
            stop_ = true;
        }
        stopped_.notify_all();
    }

    std::vector<int> sought;
    bool seeking_now = false;
    int pulls = 0;
    std::deque<NativeFramePtr> queue;

protected:
    void run() override {
        std::unique_lock lock(mutex_);
        stopped_.wait(lock, [this] { return stop_; });
    }

private:
    std::optional<std::array<float, 16>> shading_;
    std::mutex mutex_;
    std::condition_variable stopped_;
    bool stop_ = false;
};

// The presenter: what it was given to draw.
struct Presented {
    NativeFramePtr frame;
    NativeOutput::Placement placement;
    std::optional<std::array<float, 16>> shading;
};

class FakeOutput : public NativeOutput {
public:
    explicit FakeOutput(std::vector<Presented>& presented) : presented_(presented) {}
    void present(NativeFramePtr frame, const Placement& placement,
                 const std::optional<std::array<float, 16>>& shading) override {
        presented_.push_back({std::move(frame), placement, shading});
    }
    std::optional<QString> error() const override { return std::nullopt; }

private:
    std::vector<Presented>& presented_;
};

// A soundtrack whose clock the test sets; what it was asked to do.
class FakeSound : public Soundtrack {
public:
    std::optional<double> poll() override {
        reading = position;  // its clock's own reading: no carrying on here
        return position;
    }
    std::optional<double> reading_ms() const override { return reading; }
    bool ready() const override { return is_ready; }
    std::optional<QString> failed() const override { return failure; }
    void set_failed(const QString& why) override { failure = why; }
    int offset_ms() const override { return 0; }
    void set_offset(int /*first_ms*/, int /*position_ms*/) override {}
    void set_playing(bool playing) override { calls.emplace_back("playing", playing ? 1 : 0); }
    void seek(int position_ms) override {
        calls.emplace_back("seek", position_ms);
        is_ready = false;
        position = position_ms;
    }
    void set_enabled(bool /*enabled*/) override {}

    double position = 520;
    std::optional<double> reading;
    bool is_ready = true;
    std::optional<QString> failure;
    std::vector<std::pair<std::string, int>> calls;
};

NativeFramePtr frame(int width = 16, int height = 12) {
    auto made = std::make_shared<NativeFrame>();
    made->width = width;
    made->height = height;
    return made;
}

}  // namespace

class NativePoolTest : public ::testing::Test {
protected:
    // A pool of `count` test videos of 16x12 at 25 fps, at 400 ms (frame 10),
    // paused unless `playing`, on stand-ins: made by the view, as it plays a
    // pair natively.
    NativePool* make_pool(int count, bool playing = false) {
        view_ = std::make_unique<VideoCompareView>([] { return std::string("ffmpeg"); }, zoom_);
        view_->clock = [this] { return now_; };
        std::ofstream(folder_ / "source.mkv").put('\0');
        for (int index = 0; index < count; ++index) {
            FrameComparison comparison;
            comparison.source_info.path = folder_ / "source.mkv";
            comparison.source_info.width = 16;
            comparison.source_info.height = 12;
            comparison.source_info.fps = 25;
            comparison.distorted_info = comparison.source_info;
            comparison.distorted_info.path = folder_ / (std::to_string(index) + ".mkv");
            std::ofstream(comparison.distorted_info.path).put('\0');
            if (index == 1 && smaller_second_) {
                // Half the source's size, scaled up to it to be compared.
                comparison.distorted_info.width = 8;
                comparison.distorted_info.height = 6;
                comparison.scale_direction = ScaleDirection::distorted_to_source;
            }
            comparison.fps = 25;
            comparison.frame_count = 250;
            series_.push_back(comparison);
        }
        NativePool::Parts parts;
        parts.make_video = [this](const FrameComparison&, FrameSide side, const PreviewColorSettings&, int,
                                  QObject* owner) -> NativeVideo* {
            auto* video = new FakeVideo(side == FrameSide::distorted ? distorted_shading_ : std::nullopt, owner);
            videos_.push_back(video);
            return video;
        };
        parts.make_presentation = [this](PairedFrameWidget&) -> std::unique_ptr<NativeOutput> {
            return std::make_unique<FakeOutput>(presented_);
        };
        parts.make_soundtrack = [this](const FrameComparison&, int, bool) -> std::unique_ptr<Soundtrack> {
            auto sound = std::make_unique<FakeSound>();
            sound_ = sound.get();
            return sound;
        };
        view_->native_parts = [parts](const PreviewColorSettings&, QString&) { return std::optional(parts); };
        EXPECT_TRUE(view_->load(series_.front(), 400, playing, {}, series_, true));
        EXPECT_NE(view_->native_pool_, nullptr) << "not played natively";
        return view_->native_pool_.get();
    }

    static std::map<long long, NativeFramePtr>& frames(NativePool& pool, const std::string& key) {
        return pool.frames_[key];
    }
    static std::map<long long, NativeFramePtr>& source_frames(NativePool& pool) {
        return pool.frames_[pool.source_key_];
    }
    static bool choose_pair(NativePool& pool, long long target) { return pool.choose_pair(target); }
    static std::pair<NativeFramePtr, NativeFramePtr> pair(const NativePool& pool) { return *pool.pair_; }
    static long long shown_frame(const NativePool& pool) { return pool.frame_; }
    static void set_frame(NativePool& pool, long long frame) { pool.frame_ = frame; }
    static bool audio_running(const NativePool& pool) { return pool.audio_running_; }
    static void play_on_the_clock(NativePool& pool, double now) {
        pool.playing_ = true;
        pool.buffering_ = false;
        pool.audio_running_ = true;
        pool.anchor_ = now;
        pool.anchor_frame_ = pool.frame_;
        pool.audio_deadline_ = now + 15;
    }
    static void set_buffering(NativePool& pool, bool buffering) { pool.buffering_ = buffering; }
    static std::map<std::string, std::optional<std::array<float, 16>>>& shadings(NativePool& pool) {
        return pool.shadings_;
    }

    // The view stops its pool and waits for the stand-ins.
    void TearDown() override { view_.reset(); }

    test::Folder folder_;
    Zoom zoom_;
    double now_ = 100.0;
    std::unique_ptr<VideoCompareView> view_;
    std::vector<FrameComparison> series_;
    std::vector<FakeVideo*> videos_;
    std::vector<Presented> presented_;
    FakeSound* sound_ = nullptr;
    std::optional<std::array<float, 16>> distorted_shading_;
    bool smaller_second_ = false;
    NativePool* pool_ = nullptr;
};

// A source decoded ahead of its test video never runs past the frame both
// have: the pair goes on frame by frame, the test video's frame drawn.
TEST_F(NativePoolTest, AFastSourceNeverAdvancesPastTheTestVideosFrame) {
    pool_ = make_pool(1);
    const NativeFramePtr s10 = frame(), s11 = frame(), s12 = frame(), d10 = frame(), d11 = frame();
    source_frames(*pool_) = {{10, s10}, {11, s11}, {12, s12}};
    frames(*pool_, "d:0") = {{10, d10}};
    ASSERT_TRUE(choose_pair(*pool_, 12));
    EXPECT_EQ(shown_frame(*pool_), 10);
    EXPECT_EQ(pair(*pool_), std::pair(s10, d10));
    ASSERT_EQ(presented_.size(), 1U);
    EXPECT_EQ(presented_.back().frame, d10);
    frames(*pool_, "d:0")[11] = d11;
    ASSERT_TRUE(choose_pair(*pool_, 12));
    EXPECT_EQ(shown_frame(*pool_), 11);
    EXPECT_EQ(pair(*pool_), std::pair(s11, d11));
}

// Switching to another test video shows the same frame from it; Hold S then
// draws the source's frame of the pair.
TEST_F(NativePoolTest, SwitchingShowsTheSameFrameOfTheOtherTestVideo) {
    pool_ = make_pool(2);
    const NativeFramePtr s10 = frame(), d10 = frame(), other10 = frame();
    source_frames(*pool_) = {{10, s10}};
    frames(*pool_, "d:0") = {{10, d10}};
    ASSERT_TRUE(choose_pair(*pool_, 10));
    frames(*pool_, "d:1") = {{10, other10}};
    pool_->sync(1);
    EXPECT_EQ(pair(*pool_), std::pair(s10, other10));
    pool_->show_source(true);
    std::vector<NativeFramePtr> drawn;
    for (const Presented& item : presented_) {
        drawn.push_back(item.frame);
    }
    EXPECT_EQ(drawn, (std::vector<NativeFramePtr>{d10, other10, s10}));
}

// Zoomed, the part of the frame in view is drawn where the zoom puts it, in
// the frame's own pixels (a test video of half the size, scaled up to the
// source to be compared, gives half the pixels); fitted, the whole frame
// fills the view's width or height on whole pixels. A test video keeps its
// shading after its decoder went (the Python app's showed HDR as SDR when the
// encode was chosen again before its decoder was back).
TEST_F(NativePoolTest, AZoomedFrameIsPresentedWithThePartInView) {
    distorted_shading_ = std::array<float, 16>{1.0f, 1.0f};
    smaller_second_ = true;
    pool_ = make_pool(2);
    view_->resize(20, 10);
    if (view_->devicePixelRatioF() != 1.0) {
        GTEST_SKIP() << "the numbers are for one device pixel per pixel";
    }
    zoom_.factor = 2.0;  // the 16 x 12 frame at 32 x 24, centred in 20 x 10
    const NativeFramePtr s10 = frame(), d10 = frame();
    source_frames(*pool_) = {{10, s10}};
    frames(*pool_, "d:0") = {{10, d10}};
    ASSERT_TRUE(choose_pair(*pool_, 12));
    ASSERT_EQ(presented_.size(), 1U);
    EXPECT_EQ(presented_.back().frame, d10);
    EXPECT_EQ(presented_.back().placement.source, (std::array<int, 4>{3, 4, 10, 5}));
    EXPECT_EQ(presented_.back().placement.target, (std::array<int, 4>{0, 0, 20, 10}));
    zoom_.factor.reset();
    pool_->place();  // the zoom changed: the same frame again, fitted
    EXPECT_EQ(presented_.back().placement.source, (std::array<int, 4>{0, 0, 16, 12}));
    EXPECT_EQ(presented_.back().placement.target, (std::array<int, 4>{3, 0, 13, 10}));
    EXPECT_EQ(presented_.back().shading, distorted_shading_);
    // The smaller test video, zoomed as before: the same part, in its pixels.
    zoom_.factor = 2.0;
    frames(*pool_, "d:1") = {{10, frame(8, 6)}};
    pool_->sync(1);
    EXPECT_EQ(presented_.back().placement.source, (std::array<int, 4>{2, 2, 5, 2}));
    EXPECT_EQ(presented_.back().placement.target, (std::array<int, 4>{0, 0, 20, 10}));
    // The first test video's decoder gone (one decoded at a time, the second
    // shown), its shading stays.
    view_->set_decoded_videos(1);
    pool_->sync(1);
    ASSERT_TRUE(shadings(*pool_).contains("d:0")) << "the shading went with the decoder";
    EXPECT_EQ(shadings(*pool_).at("d:0"), distorted_shading_);
}

// Behind its soundtrack for a moment (the window's thread held up, the
// decoders waiting with their queues full), playback catches up with the
// sound playing on: stopped and sought again each time, it broke off every
// 15 s. Behind for longer, it stops the sound, waits for the decoders and
// seeks the sound to the frame shown.
TEST_F(NativePoolTest, AStalledDecoderStopsTheSoundAndWaitsBeforePlayingOn) {
    pool_ = make_pool(1, true);
    play_on_the_clock(*pool_, now_);
    const auto f = [](int) { return frame(); };
    std::map<long long, NativeFramePtr>& source = source_frames(*pool_);
    std::map<long long, NativeFramePtr>& test = frames(*pool_, "d:0");
    source = {{10, f(10)}, {11, f(11)}, {12, f(12)}};
    test = {{10, f(10)}};
    pool_->poll();  // frame 13 due, 10 the newest pair
    EXPECT_TRUE(sound_->calls.empty());
    EXPECT_FALSE(pool_->buffering());
    EXPECT_EQ(shown_frame(*pool_), 10);
    now_ += 0.1;
    source[13] = f(13);
    for (const long long number : {11, 12, 13}) {
        test[number] = f(static_cast<int>(number));
    }
    pool_->poll();  // caught up, the sound never stopped
    EXPECT_TRUE(sound_->calls.empty());
    EXPECT_FALSE(pool_->buffering());
    EXPECT_EQ(shown_frame(*pool_), 13);
    // Frame 14 is chosen from 540 ms on the clock (13.5 frames of 40 ms):
    // the next tick 1 ms past it, not every half frame.
    EXPECT_EQ(pool_->next_tick_ms(), 21);
    now_ += 0.015;
    EXPECT_EQ(pool_->next_tick_ms(), 6);
    sound_->position = 640;  // frame 16 due; the decoders stalled at 13
    pool_->poll();
    EXPECT_TRUE(sound_->calls.empty());
    EXPECT_FALSE(pool_->next_tick_ms());  // due, not decoded: the view's busy pace
    now_ += 0.1;
    pool_->poll();
    EXPECT_TRUE(sound_->calls.empty()) << "stopped within the moment to catch up";
    now_ += 0.15;
    pool_->poll();
    EXPECT_EQ(sound_->calls, (std::vector<std::pair<std::string, int>>{{"playing", 0}, {"seek", 520}}));
    EXPECT_TRUE(pool_->buffering());
    EXPECT_FALSE(audio_running(*pool_));
    EXPECT_EQ(shown_frame(*pool_), 13);
    pool_->poll();
    EXPECT_EQ(sound_->calls.size(), 2U);  // the sound's seek still under way
    sound_->is_ready = true;
    pool_->poll();
    EXPECT_EQ(sound_->calls.back(), (std::pair<std::string, int>{"playing", 1}));
    EXPECT_TRUE(audio_running(*pool_));
    EXPECT_FALSE(pool_->buffering());
    source[14] = f(14);
    test[14] = f(14);
    sound_->position = 560;
    pool_->poll();
    EXPECT_EQ(shown_frame(*pool_), 14);
    // The test video decoded ahead of its source, the clock ahead of both:
    // its queue keeps the frames the source has yet to reach. Trimmed from
    // the clock, it dropped them, and the pair waited for never came.
    source = {{13, f(13)}, {14, f(14)}};
    test.clear();
    for (long long number = 14; number <= 18; ++number) {
        test[number] = f(static_cast<int>(number));
    }
    sound_->position = 720;  // frame 18 due
    pool_->poll();
    EXPECT_TRUE(test.contains(14));
    // Paused on the pair on screen it is locked, not buffering, though still
    // set to wait for the sound's seek, as after one made while paused.
    pool_->set_playing(false);
    set_buffering(*pool_, true);
    pool_->poll();
    EXPECT_TRUE(pool_->description().startsWith("Frame-locked GPU pair")) << pool_->description().toStdString();
    set_frame(*pool_, 15);  // sought to a frame not decoded yet
    pool_->poll();
    EXPECT_TRUE(pool_->description().startsWith("Buffering locked pair")) << pool_->description().toStdString();
    // A video whose seek is still to be made gives the pool no frames: until
    // then it can still give frames from before the seek.
    for (FakeVideo* video : videos_) {
        video->seeking_now = true;
        video->pulls = 0;
    }
    pool_->poll();
    for (FakeVideo* video : videos_) {
        EXPECT_EQ(video->pulls, 0);
    }
}

}  // namespace vml::app
