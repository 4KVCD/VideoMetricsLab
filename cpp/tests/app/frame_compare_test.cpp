// Video Compare's still frames: the Python app's tests of the same
// behaviour (tests/test_frame_compare_panel.py, test_zoom.py), for what is
// ported (frame_compare_panel.h), and real frames of real clips through the
// panel.
#include "frame_compare_panel.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDeadlineTimer>
#include <QDoubleSpinBox>
#include <QFont>
#include <QFontMetrics>
#include <QGuiApplication>
#include <QIcon>
#include <QImage>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalSpy>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTest>
#include <QThread>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_videos.h"
#include "video_compare_view.h"
#include "vml/ffprobe.h"
#include "vml/platform/system.h"
#include "vml/settings.h"
#include "widgets.h"

namespace vml::app {

namespace {

// A decoder that gives no frames and ends at once: playback with FFmpeg,
// without FFmpeg.
class IdleDecoder : public PlaybackDecoder {
public:
    explicit IdleDecoder(QObject* parent) : PlaybackDecoder(parent) {}
    ~IdleDecoder() override { wait(); }
    void cancel() override {}
    std::vector<PlaybackFrame> drain_through(long long /*frame*/) override { return {}; }
    void recycle(FramePixels /*pixels*/) override {}
    std::optional<long long> latest_frame_number() const override { return std::nullopt; }

protected:
    void run() override {}
};

}  // namespace

// The panel's parts, for its tests.
class FramePanelTest : public ::testing::Test {
protected:
    // The panel's playback view, made now, its decoders stand-ins.
    static VideoCompareView& idle_view(FrameComparePanel& panel) {
        VideoCompareView& view = panel.ensure_video_view();
        view.make_decoder = [](const FrameComparison&, FrameSide, long long, const PreviewColorSettings&,
                               std::optional<Size>, QObject* parent) -> PlaybackDecoder* {
            return new IdleDecoder(parent);
        };
        return view;
    }
    static bool showing_source(const FrameComparePanel& panel) { return panel.showing_source_; }
    static Zoom& zoom(FrameComparePanel& panel) { return panel.zoom_; }
    static void cache_frame(FrameComparePanel& panel, FrameSide side, const QImage& image) {
        panel.on_frame_ready(panel.generation_, static_cast<int>(side), image);
    }
    static bool idle(const FrameComparePanel& panel) { return panel.live_workers().empty(); }
    // The display's HDR state as the panel would find it, and the colour
    // status it then shows.
    static QString colour_status(FrameComparePanel& panel, const platform::DisplayHdrInfo& display) {
        panel.display_hdr_ = display;
        panel.update_color_status();
        return panel.color_status_label->text();
    }
#ifdef VML_NATIVE_PLAYBACK
    static bool plays_natively(const VideoCompareView& view) { return view.native_active(); }
#endif
    // The status line's playback part for the view's status `message`.
    static QString playback_status(FrameComparePanel& panel, const QString& message) {
        panel.on_video_status_changed(message);
        return panel.concise_playback_status();
    }
    static std::size_t frames_kept(const FrameComparePanel& panel) { return panel.cache_.size(); }
    static const QImage* current_image(const FrameComparePanel& panel) { return panel.current_image(); }
    static QWidget* shown_page(const FrameComparePanel& panel) { return panel.content_stack_->currentWidget(); }
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

// A pair of files that exist (so it plays), both tagged with `transfer` and
// `primaries`.
FrameComparisonEntry tagged(const test::Folder& folder, const std::string& transfer, const std::string& primaries) {
    FrameComparison comparison;
    comparison.source_info = video(platform::path_text(folder / "source.mkv"), 3840, 2160, 120);
    comparison.distorted_info = video(platform::path_text(folder / "hdr.mkv"), 3840, 2160, 120);
    for (VideoInfo* info : {&comparison.source_info, &comparison.distorted_info}) {
        std::ofstream(info->path).put('\0');
        info->pix_fmt = "yuv420p10le";
        info->color_transfer = transfer;
        info->color_primaries = primaries;
    }
    comparison.fps = 24.0;
    comparison.frame_count = 120;
    return {"row:hdr", QStringLiteral("hdr"), comparison, nullptr, "row:hdr"};
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

// Issue #24's loop: its first frame never before the video's, and playback
// past its end goes back to that first frame; near the video's end, the
// loop ends a frame before the last, where playback would stop.
TEST(FrameCompare, TheLoopGoesBackToItsStartPastItsEnd) {
    EXPECT_EQ(loop_start(48, 0.5, 24.0), 36);
    EXPECT_EQ(loop_start(5, 1.0, 24.0), 0);
    EXPECT_EQ(loop_restart(60, 48, 0.5, 0.5, 24.0, 119), std::nullopt);
    EXPECT_EQ(loop_restart(61, 48, 0.5, 0.5, 24.0, 119), 36);
    EXPECT_EQ(loop_restart(118, 118, 1.0, 1.0, 24.0, 119), std::nullopt);
    EXPECT_EQ(loop_restart(119, 118, 1.0, 1.0, 24.0, 119), 94);
    EXPECT_EQ(loop_restart(61, 48, 0.5, 0.5, 0.0, 119), std::nullopt);  // no frame rate: no loop
}

// A still's file says the video, frame, time, the score that listed it and
// its side, without characters a file name cannot have.
TEST(FrameCompare, StillsAreNamedAfterTheirFrame) {
    EXPECT_EQ(still_file_name("enc.mkv", 72, 3.0, "PSNR 50.89", "reference").toStdString(),
              "enc.mkv - frame 000072 - 0h00m03.000s - PSNR 50.89 - reference.png");
    EXPECT_EQ(still_file_name("a:b", 90061, 3725.5, "", "test").toStdString(),
              "a_b - frame 090061 - 1h02m05.500s - test.png");
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

// Issue #200: the Test video drop-down lists the whole names, and the title
// over the video shows the whole name where it fits; where it does not, the
// shortened one (Settings > Window > Video names), cut in the middle where
// that does not fit either. Neither sets the window's minimum width.
TEST_F(FramePanelTest, WholeNamesWhereTheyFit) {
    const auto panel = make_panel();
    const std::string shared = "Example Show - S01E03 - A Long Episode Title - 2160p HDR x265 10-bit smart slower ";
    panel->set_runs({scored(shared + "v1"), scored(shared + "v2")});
    EXPECT_EQ(panel->video_combo->itemText(0).toStdString(), shared + "v1");
    EXPECT_EQ(panel->video_combo->itemText(1).toStdString(), shared + "v2");
    const int short_names = [] {
        FrameComparePanel named([] { return std::string("ffmpeg"); }, PreviewColorMode::display_aware, 3);
        named.set_runs({scored("a"), scored("b")});
        return named.minimumSizeHint().width();
    }();
    EXPECT_EQ(panel->minimumSizeHint().width(), short_names);

    panel->resize(1800, 900);
    panel->show();
    panel->layout()->activate();
    QCoreApplication::processEvents();
    const QString whole = QString::fromUtf8("TEST 1 of 2 \xE2\x80\x94 ") + QString::fromStdString(shared + "v1");
    EXPECT_EQ(panel->showing_label->text(), whole);
    EXPECT_TRUE(panel->showing_label->toolTip().isEmpty());

    const QFontMetrics metrics = panel->showing_label->fontMetrics();
    const QString shortened = QString::fromUtf8("TEST 1 of 2 \xE2\x80\x94 \xE2\x80\xA6slower v1");
    panel->showing_label->resize(metrics.horizontalAdvance(shortened) + 40, panel->showing_label->height());
    EXPECT_EQ(panel->showing_label->text(), shortened);
    EXPECT_EQ(panel->showing_label->toolTip(), whole);  // the whole name, on hover
    panel->showing_label->resize(metrics.horizontalAdvance(shortened) / 2, panel->showing_label->height());
    EXPECT_TRUE(panel->showing_label->text().contains(QChar(0x2026)));
    EXPECT_TRUE(panel->showing_label->text().startsWith("TEST")) << panel->showing_label->text().toStdString();
    EXPECT_TRUE(panel->showing_label->text().endsWith("v1")) << panel->showing_label->text().toStdString();
}

// The rows under the playback controls (the worst frames' when shown, and
// the loop's) are centred, as the rows above and below them are.
TEST_F(FramePanelTest, TheLoopRowIsCentred) {
    const auto panel = make_panel();
    panel->set_runs({scored("first"), scored("second")});
    panel->resize(1400, 900);
    panel->show();
    const auto centre_offset = [&panel](QWidget* first, QWidget* last) {
        panel->layout()->activate();
        QCoreApplication::processEvents();
        const int left = first->mapTo(panel.get(), QPoint(0, 0)).x();
        const int right = last->mapTo(panel.get(), QPoint(last->width(), 0)).x();
        return std::abs((left + right) / 2 - panel->width() / 2);
    };
    EXPECT_LE(centre_offset(panel->loop_checkbox, panel->loop_range_label), 2);
    ASSERT_EQ(panel->open_at("run:first", 10), FrameComparePanel::OpenAt::shown);
    panel->set_worst_list({"run:first", "VMAF", {{10, "40.00"}, {70, "50.00"}}}, 0);
    // The worst frames' line, and the loop's under it, each centred.
    EXPECT_LE(centre_offset(panel->previous_worst_button, panel->next_worst_button), 2);
    EXPECT_LE(centre_offset(panel->loop_checkbox, panel->loop_range_label), 2);
    EXPECT_LT(panel->next_worst_button->mapTo(panel.get(), QPoint(0, 0)).y(),
              panel->loop_checkbox->mapTo(panel.get(), QPoint(0, 0)).y());
    panel->cycle_distorted(1);  // the list's video no longer shown: the loop alone, centred again
    EXPECT_LE(centre_offset(panel->loop_checkbox, panel->loop_range_label), 2);
    // VML Modern: the rows above the view a toolbar, the controls under it a card.
    EXPECT_EQ(panel->video_combo->parentWidget()->property("vmlRole").toString(), "toolbar");
    EXPECT_EQ(panel->color_mode_combo->parentWidget()->property("vmlRole").toString(), "toolbar");
    EXPECT_EQ(panel->play_button->parentWidget()->property("vmlRole").toString(), "card");
    EXPECT_EQ(panel->loop_checkbox->parentWidget()->property("vmlRole").toString(), "card");
}

// Issue #24: the Worst Frames tab's Worst frames list is stepped through
// while its video is shown, and hidden for another; Export frame as PNG without an
// export folder says where to choose one.
TEST_F(FramePanelTest, TheWorstFramesListIsSteppedThroughOnItsVideo) {
    const auto panel = make_panel();
    panel->set_runs({scored("first"), scored("second")});
    ASSERT_EQ(panel->open_at("run:first", 10), FrameComparePanel::OpenAt::shown);
    panel->set_worst_list({"run:first", "VMAF", {{10, "40.00"}, {70, "50.00"}}}, 0);
    EXPECT_TRUE(panel->worst_label->isVisibleTo(panel.get()));
    EXPECT_EQ(panel->worst_label->text().toStdString(), "Worst 1 of 2: VMAF 40.00");
    EXPECT_FALSE(panel->previous_worst_button->isEnabled());
    EXPECT_TRUE(panel->next_worst_button->isEnabled());

    panel->next_worst_button->click();
    EXPECT_EQ(panel->frame_spin->value(), 70);
    EXPECT_EQ(panel->worst_label->text().toStdString(), "Worst 2 of 2: VMAF 50.00");
    EXPECT_TRUE(panel->previous_worst_button->isEnabled());
    EXPECT_FALSE(panel->next_worst_button->isEnabled());

    panel->cycle_distorted(1);
    EXPECT_EQ(panel->video_combo->currentText().toStdString(), "second");
    EXPECT_FALSE(panel->worst_label->isVisibleTo(panel.get()));
    EXPECT_FALSE(panel->next_worst_button->isVisibleTo(panel.get()));

    QString failure;
    QObject::connect(panel.get(), &FrameComparePanel::still_failed, [&failure](const QString& error) { failure = error; });
    panel->export_folder = [] { return QString(); };
    EXPECT_EQ(panel->save_still_button->text().toStdString(), "Export frame as PNG");  // as the graphs' exports
    panel->save_still_button->click();
    EXPECT_EQ(failure.toStdString(), "No export folder: choose one in Settings.");
    EXPECT_TRUE(idle(*panel));
}

// Loop stays as it was left: the window gives it back (set_loop_on, without
// saying it changed), and a tick or untick is said, to save. It was
// forgotten at every start, though its seconds were kept.
TEST_F(FramePanelTest, LoopIsKeptAsItWasLeft) {
    const auto panel = make_panel();
    std::vector<bool> said;
    QObject::connect(panel.get(), &FrameComparePanel::loop_changed, [&said](bool on) { said.push_back(on); });
    panel->set_loop_on(true);
    EXPECT_TRUE(panel->loop_checkbox->isChecked());
    EXPECT_TRUE(said.empty());
    panel->loop_checkbox->setChecked(false);
    panel->loop_checkbox->setChecked(true);
    EXPECT_EQ(said, (std::vector<bool>{false, true}));

    // Saved with the settings, and read back.
    const test::Folder folder;
    Settings settings;
    settings.compare_loop = true;
    ASSERT_EQ(settings.save_to(folder / "settings.json"), std::nullopt);
    EXPECT_TRUE(Settings::load_from(folder / "settings.json").compare_loop);
    EXPECT_FALSE(Settings{}.compare_loop);  // off until ticked
}

// Loop ticked at the video's last frame, shown: playback stopped there
// plays the loop again (restart_loop_at_end), and its jump back, played with
// FFmpeg as here, says at once that playback is stopped, which called it
// again, and again, until the stack ran out (the window closed). Now the
// loop starts again once.
TEST_F(FramePanelTest, LoopingAtTheLastFrameStartsTheLoopOnce) {
    // Played with FFmpeg (the unmanaged preview), its decoders stand-ins.
    const auto panel = std::make_unique<FrameComparePanel>([] { return std::string("ffmpeg"); },
                                                           PreviewColorMode::unmanaged, 3);
    idle_view(*panel);
    const test::Folder folder;
    panel->set_runs({tagged(folder, "bt709", "bt709")});  // a pair that plays, frames 0 to 119
    panel->set_loop_seconds(1.0, 1.0);
    panel->set_loop_on(true);
    panel->set_frame(119);
    panel->resize(1400, 900);
    panel->show();  // stopped at the last frame: the loop plays again, once
    QCoreApplication::processEvents();
    EXPECT_TRUE(panel->loop_checkbox->isChecked());
    EXPECT_TRUE(panel->video_view->playback_requested());  // the loop playing
}

// The loop's seconds are to the millisecond, and beside each its frames at
// the pair's frame rate: either sets the other, and the seconds are saved.
TEST_F(FramePanelTest, TheLoopIsSetInSecondsOrFrames) {
    const auto panel = make_panel();
    panel->set_loop_seconds(1.0, 0.5);
    EXPECT_FALSE(panel->loop_before_frames_spin->isEnabled());  // no pair: no frame rate
    EXPECT_EQ(panel->loop_before_spin->decimals(), 3);
    EXPECT_DOUBLE_EQ(panel->loop_before_spin->singleStep(), 0.1);

    const test::Folder folder;
    panel->set_runs({tagged(folder, "bt709", "bt709")});  // 24 fps
    EXPECT_EQ(panel->loop_before_frames_spin->value(), 24);
    EXPECT_EQ(panel->loop_after_frames_spin->value(), 12);
    EXPECT_EQ(panel->loop_before_frames_spin->isEnabled(), panel->loop_before_spin->isEnabled());

    QSignalSpy saved(panel.get(), &FrameComparePanel::loop_seconds_changed);
    panel->loop_before_frames_spin->setValue(30);
    EXPECT_DOUBLE_EQ(panel->loop_before_spin->value(), 1.25);
    ASSERT_EQ(saved.count(), 1);
    EXPECT_DOUBLE_EQ(saved.at(0).at(0).toDouble(), 1.25);
    EXPECT_DOUBLE_EQ(saved.at(0).at(1).toDouble(), 0.5);
    panel->loop_after_frames_spin->setValue(1);
    EXPECT_DOUBLE_EQ(panel->loop_after_spin->value(), 0.042);  // 1/24 s, to the millisecond
    EXPECT_EQ(loop_start(48, panel->loop_after_spin->value(), 24.0), 47);  // still the one frame
    panel->loop_after_spin->setValue(0.125);
    EXPECT_EQ(panel->loop_after_frames_spin->value(), 3);
}

// Issue #204: without a soundtrack, the status line says so, and why, while
// Audio is ticked (it said so only in Advanced info, and the log not at all);
// a missing soundtrack is not an error of the pictures', which would show
// the whole status.
TEST_F(FramePanelTest, NoSoundIsSaidWithWhyWhileAudioIsTicked) {
    const auto panel = make_panel();
    panel->set_runs({scored("encode")});
    const QString dot = QString::fromUtf8(" \xC2\xB7 ");
    const QString silent = "Paused" + dot + "audio unavailable" + dot +
                           "the reference's audio could not be read: Invalid data found when processing input" + dot +
                           "Direct3D 11";
    panel->audio_checkbox->setChecked(true);
    EXPECT_EQ(playback_status(*panel, silent).toStdString(),
              "Paused" + dot.toStdString() +
                  "No sound: the reference's audio could not be read: Invalid data found when processing input");
    // Advanced info has every part, the reason among them.
    EXPECT_TRUE(panel->advanced_info_button->toolTip().contains("the reference&#x27;s audio could not be read") ||
                panel->advanced_info_button->toolTip().contains("the reference's audio could not be read"))
        << panel->advanced_info_button->toolTip().toStdString();
    panel->audio_checkbox->setChecked(false);
    EXPECT_EQ(playback_status(*panel, silent).toStdString(), "Paused");
    panel->audio_checkbox->setChecked(true);
    EXPECT_EQ(playback_status(*panel, "Playing" + dot + "source soundtrack" + dot + "Direct3D 11").toStdString(),
              "Playing");
}

// Issue #201: the loop's row says what it loops, as a sentence: the frame
// it loops around, each side's seconds and frames, and the frames it plays,
// which the timeline shades while Loop is ticked. Unticked, the row stays,
// greyed. Export frame as PNG is with the frame controls, not in this row.
TEST_F(FramePanelTest, TheLoopRowSaysWhatItLoops) {
    // Played with FFmpeg (the Unmanaged preview), its decoders stand-ins:
    // the row alone is looked at.
    const auto panel = std::make_unique<FrameComparePanel>([] { return std::string("no-ffmpeg-here"); },
                                                           PreviewColorMode::unmanaged, 3);
    idle_view(*panel);
    const test::Folder folder;
    panel->set_runs({tagged(folder, "bt709", "bt709")});  // a pair that plays, 24 fps, frames 0 to 119
    panel->set_loop_seconds(1.0, 1.0);
    panel->set_frame(10);
    EXPECT_EQ(panel->loop_checkbox->text().toStdString(), "Loop around frame 10:");
    const std::string dash = "\xE2\x80\x93";
    EXPECT_EQ(panel->loop_range_label->text().toStdString(), "(looping frames 0" + dash + "34)");  // cut short at 0
    EXPECT_TRUE(panel->loop_before_spin->toolTip().contains("before frame 10 "));
    EXPECT_FALSE(panel->loop_before_spin->isEnabled());  // unticked: greyed, no band
    EXPECT_FALSE(panel->timeline->band());
    panel->loop_checkbox->setChecked(true);
    EXPECT_TRUE(panel->loop_before_spin->isEnabled());
    EXPECT_TRUE(panel->loop_after_frames_spin->isEnabled());
    EXPECT_EQ(panel->timeline->band(), (std::optional<std::pair<int, int>>{{0, 34}}));
    const int box_width = panel->loop_checkbox->minimumSizeHint().width();
    const int range_width = panel->loop_range_label->minimumSizeHint().width();
    panel->set_frame(60);  // another frame gone to: the loop around it
    // The row keeps its width, so the centred row doesn't move.
    EXPECT_EQ(panel->loop_checkbox->minimumSizeHint().width(), box_width);
    EXPECT_EQ(panel->loop_range_label->minimumSizeHint().width(), range_width);
    EXPECT_EQ(panel->loop_checkbox->text().toStdString(), "Loop around frame 60:");
    EXPECT_EQ(panel->loop_range_label->text().toStdString(), "(looping frames 36" + dash + "84)");
    EXPECT_EQ(panel->timeline->band(), (std::optional<std::pair<int, int>>{{36, 84}}));
    EXPECT_EQ(loop_frames(60, 1.0, 1.0, 24.0, 119), (std::pair<long long, long long>{36, 84}));
    // The videos shorter now (runs replaced): the loop's centre comes down
    // with the frame, so the row names no frame past the end.
    FrameComparisonEntry shorter = tagged(folder, "bt709", "bt709");
    shorter.comparison.frame_count = 50;
    panel->set_runs({shorter});
    EXPECT_EQ(panel->loop_checkbox->text().toStdString(), "Loop around frame 49:");
    EXPECT_EQ(panel->loop_range_label->text().toStdString(), "(looping frames 25" + dash + "48)");
    // At the end: a frame before the last, where playback stops.
    EXPECT_EQ(loop_frames(119, 1.0, 1.0, 24.0, 119), (std::pair<long long, long long>{95, 118}));
    // A band is drawn at least a few pixels wide: a frame of a long video's.
    BandSlider slider(Qt::Horizontal);
    slider.setRange(0, 172'000);
    slider.resize(800, 24);
    slider.set_band(std::pair(86'000, 86'001));
    EXPECT_GE(slider.band_rect().width(), BandSlider::min_band_px);
    slider.set_band(std::nullopt);
    EXPECT_TRUE(slider.band_rect().isEmpty());
    // The still's button beside the frame's controls.
    EXPECT_EQ(panel->save_still_button->parentWidget(), panel->next_frame_button->parentWidget());
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
    // A part of the panel hiding while it is held (a frame's text as the frame
    // arrives) leaves it held; let go with Shift down, it ends all the same.
    QTest::keyPress(panel->viewer, Qt::Key_S);
    panel->video_combo->hide();
    EXPECT_TRUE(showing_source(*panel));
    QTest::keyRelease(panel->viewer, Qt::Key_S, Qt::ShiftModifier);
    EXPECT_FALSE(showing_source(*panel));
    // Shift let go too: QTest keeps the application's modifiers as its key
    // events left them, and a Shift still down makes the next test's
    // selectRow() a Shift-click (SavedResultsPanel's selection came back
    // empty after this test).
    QTest::keyRelease(panel->viewer, Qt::Key_Shift);
    EXPECT_EQ(QGuiApplication::keyboardModifiers(), Qt::NoModifier);
    panel->cancel();
}

// Display-aware playback says how the display shows HDR: on macOS whether it
// is an HDR or an SDR display, never Windows' HDR setting or D3D11; on Windows
// as before.
TEST_F(FramePanelTest, TheColourStatusNamesTheSystemsHdr) {
    const test::Folder folder;
    const auto panel = make_panel();
    const auto status = [&](const std::string& transfer, const std::string& primaries,
                            const platform::DisplayHdrInfo& display) {
        panel->set_runs({tagged(folder, transfer, primaries)});
        return colour_status(*panel, display).toStdString();
    };
#ifdef __APPLE__
    const platform::DisplayHdrInfo xdr = platform::display_hdr_from_edr(16.0, "Built-in Retina Display");
    const platform::DisplayHdrInfo sdr = platform::display_hdr_from_edr(1.0, "GP01");
    const std::vector<std::pair<std::string, std::string>> shown{
        {status("smpte2084", "bt2020", xdr),
         "Video playback \xC2\xB7 HDR10 / PQ \xC2\xB7 HDR display \xC2\xB7 native 10-bit Metal presentation \xC2\xB7 "
         "tone mapping off"},
        {status("arib-std-b67", "bt2020", xdr),
         "Video playback \xC2\xB7 HLG \xC2\xB7 HDR display \xC2\xB7 native 10-bit Metal presentation \xC2\xB7 tone "
         "mapping off"},
        {status("bt709", "bt2020", xdr),
         "Video playback \xC2\xB7 BT.2020 wide-gamut input \xC2\xB7 HDR display \xC2\xB7 native 10-bit Metal "
         "presentation"},
        {status("smpte2084", "bt2020", sdr),
         "Video playback \xC2\xB7 HDR10 / PQ \xC2\xB7 HDR \xE2\x86\x92 SDR preview \xC2\xB7 SDR display"},
    };
    for (const auto& [text, expected] : shown) {
        EXPECT_EQ(text, expected);
        EXPECT_EQ(text.find("Windows"), std::string::npos) << text;
        EXPECT_EQ(text.find("D3D11"), std::string::npos) << text;
    }
    EXPECT_EQ(panel->color_mode_combo->toolTip().toStdString().find("Windows"), std::string::npos);
#else
    platform::DisplayHdrInfo on;
    on.hdr_enabled = true;
    platform::DisplayHdrInfo off;
    off.hdr_enabled = false;
    EXPECT_EQ(status("smpte2084", "bt2020", on),
              "Video playback \xC2\xB7 HDR10 / PQ \xC2\xB7 Windows HDR on \xC2\xB7 native 10-bit D3D11 presentation "
              "\xC2\xB7 tone mapping off");
    EXPECT_EQ(status("bt709", "bt2020", on),
              "Video playback \xC2\xB7 BT.2020 wide-gamut input \xC2\xB7 Windows HDR on \xC2\xB7 native 10-bit D3D11 "
              "presentation");
    EXPECT_EQ(status("smpte2084", "bt2020", off),
              "Video playback \xC2\xB7 HDR10 / PQ \xC2\xB7 HDR \xE2\x86\x92 SDR preview \xC2\xB7 Windows HDR off");
#endif
    EXPECT_EQ(status("smpte2084", "bt2020", platform::DisplayHdrInfo{}),
              "Video playback \xC2\xB7 HDR10 / PQ \xC2\xB7 HDR \xE2\x86\x92 SDR preview \xC2\xB7 display HDR state "
              "unavailable");
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

// Play / Pause: its text in the button's own font (Segoe UI Symbol, which
// has no bold, chipped the triangle in VML Modern's semibold primary button)
// and its symbol drawn beside it, in the text's colour; another symbol for
// each state, greyed when it is disabled.
TEST(FrameCompare, PlayPauseDrawsItsSymbolInTheButtonsFont) {
    PlayPauseButton button;
    QFont semibold = button.font();
    semibold.setWeight(QFont::DemiBold);
    button.setFont(semibold);
    EXPECT_EQ(button.text(), "Play");
    EXPECT_NE(button.font().family(), "Segoe UI Symbol");
    EXPECT_EQ(button.font().weight(), QFont::DemiBold);
    ASSERT_FALSE(button.icon().isNull());
    const QSize size = button.iconSize();
    EXPECT_GE(size.height(), 8);
    const QImage play = button.icon().pixmap(size).toImage();
    button.set_playing(true);
    EXPECT_EQ(button.text(), "Pause");
    EXPECT_TRUE(button.playing());
    const QImage pause = button.icon().pixmap(size).toImage();
    EXPECT_NE(play, pause);
    // The colour of the text, at the middle of the first bar.
    const QColor text = button.palette().color(QPalette::Active, QPalette::ButtonText);
    const QColor drawn = pause.pixelColor(pause.width() * 25 / 100, pause.height() / 2);
    EXPECT_EQ(drawn.rgb(), text.rgb());
    const QImage disabled = button.icon().pixmap(size, QIcon::Disabled).toImage();
    EXPECT_EQ(disabled.pixelColor(disabled.width() * 25 / 100, disabled.height() / 2).rgb(),
              button.palette().color(QPalette::Disabled, QPalette::ButtonText).rgb());
}

}  // namespace vml::app
