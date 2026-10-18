// Video Compare's still frames: the Python app's tests of the same
// behaviour (tests/test_frame_compare_panel.py, test_zoom.py), for what is
// ported (frame_compare_panel.h), and real frames of real clips through the
// panel.
#include "frame_compare_panel.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDeadlineTimer>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTest>
#include <QThread>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_videos.h"
#include "video_compare_view.h"
#include "vml/ffprobe.h"
#include "widgets.h"

namespace vml::app {

// The panel's parts, for its tests.
class FramePanelTest : public ::testing::Test {
protected:
    static bool showing_source(const FrameComparePanel& panel) { return panel.showing_source_; }
    static Zoom& zoom(FrameComparePanel& panel) { return panel.zoom_; }
    static void cache_frame(FrameComparePanel& panel, FrameSide side, const QImage& image) {
        panel.on_frame_ready(panel.generation_, static_cast<int>(side), image);
    }
    static bool idle(const FrameComparePanel& panel) { return panel.live_workers().empty(); }
    static std::size_t frames_kept(const FrameComparePanel& panel) { return panel.cache_.size(); }
    static const QImage* current_image(const FrameComparePanel& panel) { return panel.current_image(); }
    static QWidget* shown_page(const FrameComparePanel& panel) { return panel.content_stack_->currentWidget(); }
    // What plays the view's videos: its native playback, or its source's
    // FFmpeg decoder; another object once the view has started them again.
    static const void* decoders(const VideoCompareView& view) {
#ifdef VML_NATIVE_PLAYBACK
        if (view.native_pool_) {
            return view.native_pool_.get();
        }
#endif
        for (const auto& [key, worker] : view.pool_) {
            if (key.starts_with("s:")) {
                return worker;
            }
        }
        return nullptr;
    }
    // The sides whose current frame is kept, as they are now made.
    static std::vector<std::string> kept(const FrameComparePanel& panel) {
        std::vector<std::string> sides;
        for (const FrameSide side : {FrameSide::source, FrameSide::distorted}) {
            const auto key = panel.cache_key(side);
            if (std::any_of(panel.cache_.begin(), panel.cache_.end(),
                            [&key](const auto& item) { return key && item.first == *key; })) {
                sides.push_back(side == FrameSide::source ? "source" : "distorted");
            }
        }
        return sides;
    }
    // The black bars of a pair's two files, found.
    static void crops_found(FrameComparePanel& panel, const FrameComparisonEntry& pair, const DetectedCrops& crops) {
        panel.on_auto_crop_ready(FrameComparePanel::pair_key(pair), pair.comparison.source_info,
                                 pair.comparison.distorted_info, crops);
    }

    // Lets the panel's events run until `done`, for at most 30 seconds.
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
};

namespace {

VideoInfo video(const std::string& name, int width, int height, long long frames) {
    VideoInfo info;
    info.path = name;
    info.width = width;
    info.height = height;
    info.fps = 24.0;
    info.duration = 5.0;
    info.nb_frames = frames;
    info.codec_name = "h264";
    info.pix_fmt = "yuv420p";
    return info;
}

// A scored pair of `count` frames: VMAF `score` at the first, each frame's
// own after it (frame 24 scores `score` + 0.24).
FrameComparisonEntry scored(const std::string& name, double score = 90.0, long long count = 120) {
    auto result = std::make_shared<ComparisonResult>();
    result->source_info = video("source.mkv", 1920, 1080, count);
    result->distorted_info = video(name + ".mkv", 1920, 1080, count);
    result->fps = 24.0;
    result->compared_frame_count = count;
    FrameMetricResult vmaf;
    vmaf.key = "vmaf";
    for (int frame = 0; frame < count; ++frame) {
        vmaf.frame.push_back(frame);
        vmaf.time.push_back(frame / 24.0);
        vmaf.values.push_back(static_cast<float>(score + frame / 100.0));
    }
    result->metric_results.add(std::move(vmaf));
    return {"run:" + name, QString::fromStdString(name), FrameComparison::from_result(*result), result,
            "row:" + name};
}

// A pair that has only been read: no run, no scores.
FrameComparisonEntry unscored(const std::string& name, long long count = 120) {
    FrameComparison comparison;
    comparison.source_info = video("source.mkv", 1920, 1080, count);
    comparison.distorted_info = video(name + ".mkv", 1280, 720, count);
    comparison.fps = 24.0;
    comparison.frame_count = count;
    return {"row:" + name, QString::fromStdString(name), comparison, nullptr, "row:" + name};
}

std::unique_ptr<FrameComparePanel> make_panel(std::string ffmpeg = "ffmpeg") {
    return std::make_unique<FrameComparePanel>([ffmpeg] { return ffmpeg; }, PreviewColorMode::display_aware, 3);
}

TEST(FrameCompare, TimestampsAreReadInEveryFormTheUserMayType) {
    EXPECT_EQ(parse_timestamp("12.5"), 12.5);
    EXPECT_EQ(parse_timestamp("1:02.5"), 62.5);
    EXPECT_EQ(parse_timestamp("2:03:04.25"), 7384.25);
    for (const char* text : {"", "1:60", "1:60:00", "nope", "-1", "nan", "inf", "1.5:02", "1:2:3:4"}) {
        EXPECT_THROW(parse_timestamp(text), std::invalid_argument) << text;
    }
}

}  // namespace

// The pairs are listed, on the frames they all have; a frame shown is in
// every seek widget, with its score; a switch to another test video keeps
// the frame and wraps around, and stays when that video's scores go; an
// unscored pair is shown, and says so.
TEST_F(FramePanelTest, ThePairsAreListedOnTheirCommonFramesAndStepTogether) {
    const auto panel = make_panel();
    panel->set_runs({scored("first", 90.0, 120), scored("second", 80.0, 90)});
    EXPECT_EQ(panel->video_combo->count(), 2);
    EXPECT_EQ(panel->video_combo->currentText().toStdString(), "first");
    EXPECT_EQ(panel->frame_spin->maximum(), 89);
    EXPECT_EQ(panel->timeline->maximum(), 89);
    EXPECT_TRUE(panel->showing_label->text().contains("TEST 1 of 2")) << panel->showing_label->text().toStdString();

    panel->set_frame(24);
    EXPECT_EQ(panel->frame_spin->value(), 24);
    EXPECT_EQ(panel->timeline->value(), 24);
    EXPECT_EQ(panel->timestamp_edit->text().toStdString(), "0:00:01.000");
    EXPECT_TRUE(panel->detail_label->full_text().contains("Frame 24"));
    EXPECT_TRUE(panel->detail_label->full_text().contains("VMAF v0.6.1 90.24"))
        << panel->detail_label->full_text().toStdString();

    panel->set_frame(48);
    panel->cycle_distorted(-1);
    EXPECT_EQ(panel->video_combo->currentText().toStdString(), "second");
    EXPECT_EQ(panel->frame_spin->value(), 48);
    EXPECT_TRUE(panel->showing_label->text().contains("TEST 2 of 2"));
    EXPECT_TRUE(panel->detail_label->full_text().contains("VMAF v0.6.1 80.48"));
    // The video watched stays chosen when its scores go (Recalculate), as
    // when they come: its entry's identity changes, its row does not.
    panel->set_runs({scored("first", 90.0, 120), unscored("second", 90)});
    EXPECT_EQ(panel->video_combo->currentText().toStdString(), "second");

    panel->timestamp_edit->setText("0:00:02.5");
    emit panel->timestamp_edit->editingFinished();
    EXPECT_EQ(panel->frame_spin->value(), 60);
    panel->timestamp_edit->setText("1e300");  // past the end: the last frame
    emit panel->timestamp_edit->editingFinished();
    EXPECT_EQ(panel->frame_spin->value(), 89);

    panel->set_runs({unscored("encode")});
    EXPECT_EQ(panel->video_combo->count(), 1);
    EXPECT_TRUE(panel->frame_spin->isEnabled());
    EXPECT_EQ(panel->frame_spin->maximum(), 119);
    EXPECT_TRUE(panel->detail_label->full_text().contains("No metric results loaded"));
    EXPECT_FALSE(panel->detail_label->full_text().contains("VMAF"));
}

// Holding S shows the source, letting go the test video again.
TEST_F(FramePanelTest, HoldingSShowsTheSource) {
    const auto panel = make_panel();
    panel->set_runs({scored("encode")});
    panel->show();
    panel->viewer->setFocus();
    QTest::keyPress(panel->viewer, Qt::Key_S);
    EXPECT_TRUE(showing_source(*panel));
    EXPECT_TRUE(panel->showing_label->text().startsWith("SOURCE")) << panel->showing_label->text().toStdString();
    QTest::keyRelease(panel->viewer, Qt::Key_S);
    EXPECT_FALSE(showing_source(*panel));
    EXPECT_TRUE(panel->showing_label->text().startsWith("TEST"));
    panel->cancel();
}

// Fitted by default; a zoom chosen, or typed in any form; what is not a zoom
// leaves the one in force.
TEST_F(FramePanelTest, TheZoomFitsByDefaultAndTakesAChosenOrTypedPercentage) {
    const auto panel = make_panel();
    EXPECT_FALSE(zoom(*panel).factor);
    EXPECT_EQ(panel->zoom_combo->currentText().toStdString(), "Fit to window");
    const int index = panel->zoom_combo->findData(1.0);
    panel->zoom_combo->setCurrentIndex(index);
    emit panel->zoom_combo->activated(index);
    EXPECT_EQ(zoom(*panel).factor, 1.0);
    EXPECT_EQ(panel->zoom_combo->currentText().toStdString(), "100%");
    const auto typed = [&panel](const QString& text) {
        panel->zoom_combo->setEditText(text);
        emit panel->zoom_combo->lineEdit()->editingFinished();
    };
    typed("150 %");
    EXPECT_EQ(zoom(*panel).factor, 1.5);
    EXPECT_EQ(panel->zoom_combo->currentText().toStdString(), "150%");
    typed("not a zoom");
    EXPECT_EQ(zoom(*panel).factor, 1.5);
    EXPECT_EQ(panel->zoom_combo->currentText().toStdString(), "150%");
    typed("1,25");
    EXPECT_EQ(zoom(*panel).factor, 0.1);  // 1.25%, kept at the least zoom there is
    typed("Fit to window");
    EXPECT_FALSE(zoom(*panel).factor);
    EXPECT_EQ(percent_text(0.125).toStdString(), "12.5%");
}

// A frame larger than the view is dragged; a fitted one is not.
TEST_F(FramePanelTest, AZoomedStillFrameIsDraggedAndAFittedOneIsNot) {
    Zoom zoom;
    FrameView viewer(zoom);
    viewer.resize(400, 200);
    viewer.show();
    QImage image(800, 400, QImage::Format_RGB32);
    image.fill(0);
    viewer.set_image(image);
    const double ratio = viewer.devicePixelRatioF();
    const auto drag = [&viewer](QPoint from, QPoint to) {
        QTest::mousePress(&viewer, Qt::LeftButton, Qt::NoModifier, from);
        QTest::mouseMove(&viewer, to);
        QTest::mouseRelease(&viewer, Qt::LeftButton, Qt::NoModifier, to);
    };
    drag({200, 100}, {150, 100});
    EXPECT_EQ(zoom.centre, (std::pair<double, double>{0.5, 0.5}));  // fitted: nothing to move
    zoom.factor = 1.0;
    viewer.zoom_changed();
    EXPECT_EQ(viewer.cursor().shape(), Qt::OpenHandCursor);
    drag({200, 100}, {150, 100});
    EXPECT_NEAR(zoom.centre.first, 0.5 + 50 * ratio / 800, 1e-9);
}

// A pair keeps its identity while its frames change: its black bars found,
// its scaling edited, another reference chosen. The frames kept from before
// were shown again (in both apps, the uncropped one extracted while the bars
// were looked for stayed on screen, while the line no longer said anything
// was pending): they must be extracted again. A side whose frame is the
// same is kept.
TEST_F(FramePanelTest, AFrameKeptIsNotShownOnceWhatItIsMadeFromChanges) {
    using Sides = std::vector<std::string>;
    const auto with_frames_kept = [](const FrameComparisonEntry& entry) {
        auto panel = make_panel();
        panel->set_runs({entry});
        QImage image(1280, 720, QImage::Format_RGB32);
        image.fill(Qt::red);
        cache_frame(*panel, FrameSide::source, image);
        cache_frame(*panel, FrameSide::distorted, image);
        EXPECT_EQ(kept(*panel), (Sides{"source", "distorted"}));
        return panel;
    };
    // The 1080p source scaled down to the 720p encode's size, which is not
    // scaled.
    const FrameComparisonEntry pair = unscored("encode");
    {
        SCOPED_TRACE("black bars found");
        FrameComparisonEntry entry = pair;
        entry.comparison.auto_crop_pending = true;
        const auto panel = with_frames_kept(entry);
        crops_found(*panel, entry, {CropBox{1920, 816, 0, 132}, CropBox{1280, 544, 0, 88}});
        EXPECT_FALSE(panel->current_entry()->comparison.auto_crop_pending);
        EXPECT_EQ(kept(*panel), Sides{});
        EXPECT_EQ(current_image(*panel), nullptr) << "the uncropped frame stands for the cropped one";
    }
    {
        SCOPED_TRACE("scaling edited");
        const auto panel = with_frames_kept(pair);
        FrameComparisonEntry edited = pair;
        edited.comparison.scale_algorithm = "bilinear";
        panel->set_runs({edited});
        EXPECT_EQ(kept(*panel), Sides{"distorted"});
    }
    {
        SCOPED_TRACE("another reference");
        const auto panel = with_frames_kept(pair);
        FrameComparisonEntry other = pair;
        other.comparison.source_info.path = "another source.mkv";
        panel->set_runs({other});
        EXPECT_EQ(kept(*panel), Sides{"distorted"});
    }
}

// A pair keeps its identity when another reference is chosen. The bars found
// against the old one were kept for it (in both apps: a 4K reference's
// 3360x1600 crop applied to a 1080p one, which FFmpeg refuses): they must be
// looked for again. The pair's own files again, and they are known.
TEST_F(FramePanelTest, TheBlackBarsFoundAgainstOneReferenceAreNotUsedWithAnother) {
    const auto panel = make_panel();
    FrameComparisonEntry pair = unscored("encode");
    pair.comparison.auto_crop_pending = true;
    panel->set_runs({pair});
    const DetectedCrops found{CropBox{1920, 816, 0, 132}, CropBox{1280, 544, 0, 88}};
    crops_found(*panel, pair, found);
    const auto crops = [&panel] {
        const FrameComparison& shown = panel->current_entry()->comparison;
        return std::pair(shown.source_crop, shown.distorted_crop);
    };
    EXPECT_EQ(crops(), std::pair(found.source, found.distorted));

    FrameComparisonEntry other = pair;
    other.comparison.source_info.path = "another source.mkv";
    panel->set_runs({other});
    EXPECT_TRUE(panel->current_entry()->comparison.auto_crop_pending);
    EXPECT_EQ(crops(), (std::pair<std::optional<CropBox>, std::optional<CropBox>>{}));

    panel->set_runs({pair});
    EXPECT_EQ(crops(), std::pair(found.source, found.distorted));
}

// Each pair is cropped to the picture its two videos share, as a run crops
// it, from each file's own black bars. A reference with 40-row bars against a
// test video with 60-row bars shares the 240 rows between them; against a
// test video with the reference's own bars it shares all 280 rows, though the
// reference's bars were found for the first pair.
TEST_F(FramePanelTest, EachPairIsCroppedToThePictureItsOwnTwoVideosShare) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    const auto clip = [&](const std::string& name, int picture_height) {
        const std::string bars = std::to_string((360 - picture_height) / 2);
        return probe_video(test::make_clip(*found, folder / name, "libx264", "yuv420p", 1.0,
                                           {"-vf", "pad=640:360:0:" + bars + ":black"},
                                           "640x" + std::to_string(picture_height)),
                           found->ffprobe);
    };
    FrameComparison narrower;
    narrower.source_info = clip("source.mkv", 280);
    narrower.distorted_info = clip("narrower.mkv", 240);
    narrower.fps = narrower.source_info.fps;
    narrower.frame_count = 23;
    narrower.auto_crop_pending = true;
    FrameComparison same = narrower;
    same.distorted_info = clip("same.mkv", 280);
    const auto panel = make_panel(found->ffmpeg);
    panel->audio_checkbox->setChecked(false);
    panel->resize(800, 600);
    panel->show();
    panel->set_runs({{"row:narrower", "narrower", narrower, nullptr, "row:narrower"},
                     {"row:same", "same", same, nullptr, "row:same"}});
    const auto crops = [&panel] {
        const FrameComparison& shown = panel->current_entry()->comparison;
        return std::pair(shown.source_crop, shown.distorted_crop);
    };
    ASSERT_TRUE(wait_until([&panel] { return !panel->current_entry()->comparison.auto_crop_pending; }));
    EXPECT_EQ(crops(), std::pair(std::optional(CropBox{640, 240, 0, 60}), std::optional(CropBox{640, 240, 0, 60})));

    panel->cycle_distorted(1);
    ASSERT_EQ(panel->video_combo->currentText().toStdString(), "same");
    ASSERT_TRUE(wait_until([&panel] { return !panel->current_entry()->comparison.auto_crop_pending; }));
    EXPECT_EQ(crops(), std::pair(std::optional(CropBox{640, 280, 0, 40}), std::optional(CropBox{640, 280, 0, 40})));
    panel->cancel();
    ASSERT_TRUE(wait_until([&panel] { return FramePanelTest::idle(*panel); }));
}

// A pair of real clips plays in the tab: the playback view takes the still
// frames' place, Play and Space play and pause it (the button says which),
// the frame box follows the frames played, and a frame chosen while paused
// is shown there. Their black bars, found while they play, change nothing,
// and the next test video (Right) then plays on the same decoders: loaded
// afresh, every decoder started again, seconds at 4K, and the window showed
// what it had drawn there before.
TEST_F(FramePanelTest, APairThatCanPlayIsPlayedWithThePanelsControls) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    FrameComparison comparison;
    comparison.source_info = probe_video(
        test::make_clip(*found, folder / "source.mp4", "libx264", "yuv420p", 2.0, {}, "160x90"), found->ffprobe);
    comparison.distorted_info = probe_video(
        test::make_clip(*found, folder / "encode.mkv", "libx265", "yuv420p", 2.0, {}, "160x90"), found->ffprobe);
    comparison.fps = comparison.distorted_info.fps;
    comparison.frame_count = 47;
    comparison.auto_crop_pending = true;  // black bars to be found, as a pair just read
    FrameComparison other = comparison;
    other.distorted_info = probe_video(
        test::make_clip(*found, folder / "encode2.mkv", "libx265", "yuv420p", 2.0, {}, "160x90"), found->ffprobe);
    const auto panel = make_panel(found->ffmpeg);
    panel->audio_checkbox->setChecked(false);
    panel->resize(800, 600);
    panel->show();
    panel->set_runs({{"row:encode", "encode", comparison, nullptr, "row:encode"},
                     {"row:encode2", "encode2", other, nullptr, "row:encode2"}});
    ASSERT_NE(panel->video_view, nullptr);
    EXPECT_EQ(shown_page(*panel), panel->video_view);
    EXPECT_TRUE(panel->play_button->isEnabled());
    EXPECT_TRUE(panel->source_resolution_combo->isEnabled());

    QTest::keyClick(panel->video_view, Qt::Key_Space);
    EXPECT_TRUE(panel->video_view->playback_requested());
    EXPECT_EQ(panel->play_button->text().toStdString(), "\xE2\x9D\x9A\xE2\x9D\x9A Pause");
    ASSERT_TRUE(wait_until([&panel] { return panel->frame_spin->value() >= 10; }))
        << "the frame box did not follow playback";
    panel->play_button->click();
    EXPECT_FALSE(panel->video_view->playback_requested());
    EXPECT_EQ(panel->play_button->text().toStdString(), "\xE2\x96\xB6 Play");
    EXPECT_TRUE(panel->detail_label->full_text().contains("Paused")) << panel->detail_label->full_text().toStdString();

    panel->set_frame(5);
    ASSERT_TRUE(wait_until([&panel] {
        return panel->video_view->position() == static_cast<int>(std::nearbyint(5 / panel->current_entry()->comparison.fps * 1000));
    }));
    EXPECT_EQ(panel->frame_spin->value(), 5);

    ASSERT_TRUE(wait_until([&panel] { return !panel->current_entry()->comparison.auto_crop_pending; }))
        << "the black bars were not found";
    const void* playing = decoders(*panel->video_view);
    ASSERT_NE(playing, nullptr);
    panel->cycle_distorted(1);  // what Right does
    EXPECT_EQ(panel->video_combo->currentText().toStdString(), "encode2");
    EXPECT_EQ(decoders(*panel->video_view), playing) << "the decoders were started again";
    panel->cancel();
    ASSERT_TRUE(wait_until([&panel] { return FramePanelTest::idle(*panel); }));
}

// Real frames of a real clip's resolution test (still frames: it does not
// play), through the panel as the window shows it: the test's frame (the
// clip scaled down and back up), then with S the source's same frame, each of
// the size the metrics compare; a frame kept is shown again without FFmpeg;
// a file that has gone says why.
TEST_F(FramePanelTest, TheFramesShownAreFfmpegsFramesOfBothVideos) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    FrameComparison comparison;
    comparison.source_info = video("source.mkv", 640, 360, 24);
    comparison.source_info.path = test::make_clip(*found, folder / "source.mkv", "libx264", "yuv420p", 1.0);
    comparison.source_info.fps = 24000.0 / 1001;
    comparison.distorted_info = comparison.source_info;
    comparison.resample_target = ResampleTarget{320, "320 wide"};
    comparison.fps = 24000.0 / 1001;
    comparison.frame_count = 23;
    const auto panel = make_panel(found->ffmpeg);
    panel->resize(800, 600);
    panel->show();
    panel->set_runs({{"row:encode", "encode", comparison, nullptr, {}}});
    panel->set_frame(5);
    ASSERT_TRUE(wait_until([&panel] { return !panel->viewer->image().isNull(); }))
        << panel->viewer->message().toStdString();
    // The test's frame, at the size compared (the source's), converted for
    // the display.
    EXPECT_EQ(panel->viewer->image().size(), QSize(640, 360));
    const QImage test_frame = panel->viewer->image();
    // Both sides extracted and kept (the last arrives after its thread ends).
    ASSERT_TRUE(wait_until([&panel] { return FramePanelTest::frames_kept(*panel) == 2; }));

    QTest::keyPress(panel->viewer, Qt::Key_S);
    ASSERT_FALSE(panel->viewer->image().isNull()) << "the source's frame was not kept";
    EXPECT_EQ(panel->viewer->image().size(), QSize(640, 360));
    EXPECT_NE(panel->viewer->image(), test_frame);
    QTest::keyRelease(panel->viewer, Qt::Key_S);
    EXPECT_EQ(panel->viewer->image(), test_frame);

    std::filesystem::remove(comparison.source_info.path);
    panel->set_runs({{"row:encode", "encode", comparison, nullptr, {}}});  // read again, as the window does
    panel->set_frame(6);
    ASSERT_TRUE(wait_until([&panel] { return panel->viewer->message().startsWith("Could not load frame"); }))
        << panel->viewer->message().toStdString();
    EXPECT_TRUE(panel->viewer->message().contains("Video file is missing")) << panel->viewer->message().toStdString();
    panel->cancel();
    ASSERT_TRUE(wait_until([&panel] { return FramePanelTest::idle(*panel); }));
}

}  // namespace vml::app
