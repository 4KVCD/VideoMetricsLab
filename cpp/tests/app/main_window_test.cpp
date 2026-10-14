// The main window's tests: the Python app's tests of the same behaviour
// (tests/test_main_window.py, tests/test_video_list_editing.py), for what
// main_window.h says has been ported.
#include "main_window.h"

#include <QApplication>
#include <QCheckBox>
#include <QDeadlineTimer>
#include <QDropEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QPushButton>
#include <QScrollArea>
#include <QTableWidgetSelectionRange>
#include <QTest>
#include <QThread>
#include <QTimer>
#include <QUrl>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "file_writes.h"
#include "i18n.h"
#include "test_videos.h"
#include "theme.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "widgets.h"

namespace vml::app {

// The window's parts, for its tests: the one class MainWindow lets in.
class WindowTest : public ::testing::Test {
protected:
    // Clips of FFmpeg's test pattern, made once for every test: a reference
    // in one folder and two test videos in another, as videos often are.
    static void SetUpTestSuite() {
        found_ = test::tools();
        if (!found_) {
            return;
        }
        clips_ = std::make_unique<test::Folder>();
        std::filesystem::create_directories(*clips_ / "sources");
        std::filesystem::create_directories(*clips_ / "encodes");
        test::make_clip(*found_, *clips_ / "sources" / "ref.mkv", "libx264", "yuv420p", 0.5);
        test::make_clip(*found_, *clips_ / "encodes" / "a.mkv", "libx264", "yuv420p", 0.5);
        test::make_clip(*found_, *clips_ / "encodes" / "b.mkv", "libx265", "yuv420p", 0.5, {}, "480x270");
    }

    static void TearDownTestSuite() { clips_.reset(); }

    // A profile of the test's own, so that the user's real settings and
    // saved scores are never read or written. Its settings.json ticks VMAF,
    // PSNR, SSIM and XPSNR, as the Python app's tests' does, with VMAF on
    // the CPU: on the GPU, a run of these half-second clips would be mostly
    // the GPU starting, in a process of its own.
    WindowTest()
        : profile_(folder_ / "profile"), user_profile_("USERPROFILE", platform::path_text(profile_)) {
        std::filesystem::create_directories(profile_ / ".videometricslab");
        std::filesystem::create_directories(folder_ / "results_cache");
        write_settings(nlohmann::json::object());
    }

    // Writes the profile's settings.json: the tests' own, with `changes`.
    void write_settings(const nlohmann::json& changes) const {
        nlohmann::json settings{
            {"cache_dir", platform::path_text(folder_ / "results_cache")},
            {"parallel_jobs", 1},
            {"default_compute_vmaf_neg", false},
            {"default_compute_ssimulacra2", false},
            {"default_compute_butteraugli", false},
            {"default_compute_cvvdp", false},
            {"default_compute_vmaf_v1", false},
            {"default_vmaf_on_gpu", false},
        };
        settings.update(changes);
        std::ofstream(profile_ / ".videometricslab" / "settings.json", std::ios::binary) << settings.dump();
    }

    // The settings.json the window saved.
    nlohmann::json saved_settings() const {
        return nlohmann::json::parse(
            platform::read_file(profile_ / ".videometricslab" / "settings.json").value_or("{}"));
    }

    // A clip made for the tests (SetUpTestSuite): "sources/ref.mkv",
    // "encodes/a.mkv" or "encodes/b.mkv".
    static std::filesystem::path clip(const std::string& name) {
        return (*clips_ / name).make_preferred();
    }

    static bool have_clips() { return clips_ != nullptr; }

    std::filesystem::path folder(const std::string& name) const { return folder_ / name; }

    // ------------------------------------------------------------- its parts
    static FillColumnTable& table(MainWindow& window) { return *window.distorted_table_; }
    static CheckableHeaderView& header(MainWindow& window) { return *window.metric_header_; }
    static std::vector<std::shared_ptr<RowData>>& rows(MainWindow& window) { return window.rows_; }
    static const VmafOptions& defaults(MainWindow& window) { return window.default_options_; }
    static QLineEdit& source_edit(MainWindow& window) { return *window.source_edit_; }
    static QLabel& source_info_label(MainWindow& window) { return *window.source_info_label_; }
    static const std::optional<VideoInfo>& source_info(MainWindow& window) { return window.source_info_; }
    static QLabel& ffmpeg_banner(MainWindow& window) { return *window.ffmpeg_banner_; }
    static QPushButton& locate_ffmpeg_button(MainWindow& window) { return *window.locate_ffmpeg_button_; }
    static QTabWidget& tabs(MainWindow& window) { return *window.tabs_; }
    static QPushButton& run_button(MainWindow& window) { return *window.run_button_; }
    static QPushButton& pause_button(MainWindow& window) { return *window.pause_button_; }
    static QPushButton& cancel_button(MainWindow& window) { return *window.cancel_button_; }
    static std::string status(MainWindow& window) { return window.status_label_->text().toStdString(); }
    // The line below the status line, from the top, of a video in progress.
    static std::string job_line(MainWindow& window, std::size_t line) {
        return window.job_progress_labels_[line]->full_text().toStdString();
    }
    static bool running(MainWindow& window) { return window.run_active_; }
    static RowState state(MainWindow& window, int row) {
        return window.row_state(*window.rows_[static_cast<std::size_t>(row)]);
    }
    // The run's clock, in seconds.
    static void set_clock(MainWindow& window, std::function<double()> clock) { window.clock_ = std::move(clock); }
    // Whether the saved scores' writes queued so far have run, within 30 s.
    static bool wait_for_writes(MainWindow& window) {
        return window.file_writes_->wait_until_idle(std::chrono::seconds(30));
    }
    // The table menu's Recalculate, for `rows`.
    static void recalculate(MainWindow& window, const std::vector<int>& rows) { window.recompute_rows(rows); }
    static QCheckBox& highlight_check(MainWindow& window) { return *window.highlight_check_; }
    // Whether the best/worst pass is still to come.
    static bool best_worst_due(MainWindow& window) { return window.best_worst_timer_->isActive(); }
    // Shows a row's scores, as a finished video or its saved scores do.
    static void show_scores(MainWindow& window, int row) { window.set_row_metrics(row); }
    static void remove_selected(MainWindow& window) { window.on_remove_distorted(); }
    static QString dialog_folder(MainWindow& window, const QString& kind) { return window.video_dialog_dir(kind); }

    // Whether a video is still being read.
    static bool reading(MainWindow& window) { return !window.probe_workers_.empty(); }

    static int add_row(MainWindow& window, const std::string& name) {
        return window.add_table_row(platform::path_from_text(name));
    }

    static std::vector<std::string> requested(MainWindow& window, int row) {
        return window.requested_metrics(*window.rows_[static_cast<std::size_t>(row)]);
    }

    static std::vector<std::string> selected(MainWindow& window, int row) {
        return MainWindow::selected_metrics(*window.rows_[static_cast<std::size_t>(row)]);
    }

    static void show_metric(MainWindow& window, const std::string& key, bool shown) {
        window.on_metric_visibility_toggled(key, shown, nullptr);
    }

    static void drag_rows(MainWindow& window, const QList<int>& moved, int target) {
        window.on_rows_dragged(moved, target);
    }

    static void choose_source(MainWindow& window, const std::filesystem::path& path) {
        window.choose_source(path);
    }

    // The reference and the two test videos, read.
    static void open_clips(MainWindow& window) {
        window.choose_source(clip("sources/ref.mkv"));
        window.add_test_videos({clip("encodes/a.mkv"), clip("encodes/b.mkv")});
        ASSERT_TRUE(wait_until([&] { return window.source_info_.has_value() && !reading(window); }));
    }

    // What reading a video found, as its thread says it.
    static void probed(MainWindow& window, const Probed& found) { window.on_probed(found); }

    // ------------------------------------------------------------ the user's
    // The file names in the table, top to bottom. The cells and the rows
    // behind them must move together.
    static std::vector<std::string> names(MainWindow& window) {
        std::vector<std::string> shown;
        for (int row = 0; row < table(window).rowCount(); ++row) {
            shown.push_back(table(window).item(row, col_path)->text().toStdString());
            EXPECT_EQ(shown.back(), platform::path_text(rows(window)[static_cast<std::size_t>(row)]->path.filename()));
        }
        return shown;
    }

    // The selected rows, top to bottom.
    static std::vector<int> selected_rows(MainWindow& window) {
        std::vector<int> selected;
        for (const QModelIndex& index : table(window).selectionModel()->selectedRows()) {
            selected.push_back(index.row());
        }
        std::sort(selected.begin(), selected.end());
        return selected;
    }

    // Files dropped on `target` from a file manager.
    static void drop(QWidget* target, const std::vector<std::filesystem::path>& paths) {
        QMimeData mime;
        QList<QUrl> urls;
        for (const std::filesystem::path& path : paths) {
            urls.append(QUrl::fromLocalFile(QString::fromStdString(platform::path_text(path))));
        }
        mime.setUrls(urls);
        // As a drag delivers it: Qt gives the drop to the widget that took
        // the drag as it entered, and to no other.
        QDragEnterEvent enter(QPointF(5, 5), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(target, &enter);
        QDropEvent drop(QPointF(5, 5), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(target, &drop);
    }

    // A click on a heading: on its tick box, or away from it.
    static void click_heading(MainWindow& window, int column, bool on_tick_box) {
        CheckableHeaderView& heading = header(window);
        const QPoint point = on_tick_box ? heading.section_indicator_rect(column).center()
                                         : QPoint(heading.sectionViewportPosition(column) +
                                                      heading.sectionSize(column) * 3 / 4,
                                                  heading.height() / 2);
        QTest::mouseClick(heading.viewport(), Qt::LeftButton, Qt::NoModifier, point);
    }

    // Lets the window's events run until `done`, or for at most 30 seconds:
    // what its threads find arrives as events.
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

    static bool contains(const std::vector<std::string>& keys, const std::string& key) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    }

private:
    static inline std::optional<test::Tools> found_;
    static inline std::unique_ptr<test::Folder> clips_;

    test::Folder folder_;
    std::filesystem::path profile_;
    test::ScopedVariable user_profile_;
};

namespace {

// ------------------------------------------------------------ the metrics

TEST_F(WindowTest, ClickingAMetricTickBoxSelectsThatMetricForTheRow) {
    MainWindow window;
    add_row(window, "a.mp4");
    add_row(window, "b.mp4");

    // Clicking a row that is not selected applies to that row alone...
    table(window).item(0, col_psnr)->setCheckState(Qt::Unchecked);
    EXPECT_FALSE(contains(rows(window)[0]->options.requested_metrics(), "psnr"));
    EXPECT_TRUE(contains(rows(window)[1]->options.requested_metrics(), "psnr"));
    // ...and is not the default for videos added later.
    EXPECT_TRUE(contains(defaults(window).requested_metrics(), "psnr"));

    // Clicking one of several selected rows applies to all of them.
    table(window).selectAll();
    table(window).item(1, col_xpsnr)->setCheckState(Qt::Unchecked);
    for (const auto& row : rows(window)) {
        EXPECT_FALSE(contains(row->options.requested_metrics(), "xpsnr"));
    }
}

TEST_F(WindowTest, TickingAMetricsHeadingEnablesItForEveryRowAndVideosAddedLater) {
    MainWindow window;
    window.show();
    add_row(window, "a.mp4");
    add_row(window, "b.mp4");

    click_heading(window, col_psnr, true);
    for (int row = 0; row < 2; ++row) {
        EXPECT_FALSE(contains(rows(window)[static_cast<std::size_t>(row)]->options.requested_metrics(), "psnr"));
        EXPECT_EQ(table(window).item(row, col_psnr)->checkState(), Qt::Unchecked);
    }
    // The heading is a statement about the table: the default too, saved.
    EXPECT_FALSE(contains(defaults(window).requested_metrics(), "psnr"));
    EXPECT_EQ(saved_settings()["default_compute_psnr"], false);

    click_heading(window, col_psnr, true);
    for (int row = 0; row < 2; ++row) {
        EXPECT_TRUE(contains(rows(window)[static_cast<std::size_t>(row)]->options.requested_metrics(), "psnr"));
        EXPECT_EQ(table(window).item(row, col_psnr)->checkState(), Qt::Checked);
    }
    EXPECT_EQ(saved_settings()["default_compute_psnr"], true);

    // A click on the heading away from its tick box changes nothing: reading
    // a heading by clicking it must not change what the next run calculates.
    click_heading(window, col_psnr, false);
    EXPECT_TRUE(header(window).is_checked(col_psnr));
    EXPECT_TRUE(contains(rows(window)[0]->options.requested_metrics(), "psnr"));
}

TEST_F(WindowTest, HidingAMetricHidesItsColumnAndLeavesItOutOfRuns) {
    MainWindow window;
    const int row = add_row(window, "a.mp4");
    ASSERT_TRUE(contains(requested(window, row), "xpsnr"));

    show_metric(window, "xpsnr", false);
    EXPECT_TRUE(table(window).isColumnHidden(col_xpsnr));
    EXPECT_FALSE(contains(requested(window, row), "xpsnr"));
    // The row's own tick is kept, so showing the metric again restores it.
    EXPECT_TRUE(contains(selected(window, row), "xpsnr"));
    EXPECT_EQ(saved_settings()["hidden_metrics"], nlohmann::json::array({"xpsnr"}));

    show_metric(window, "xpsnr", true);
    EXPECT_FALSE(table(window).isColumnHidden(col_xpsnr));
    EXPECT_TRUE(contains(requested(window, row), "xpsnr"));
    EXPECT_EQ(saved_settings()["hidden_metrics"], nlohmann::json::array());
}

// ------------------------------------------------------- arranging the rows

TEST_F(WindowTest, DraggedRowsMoveWithTheirDataAndStaySelected) {
    MainWindow window;
    for (const char* name : {"a.mkv", "b.mkv", "c.mkv", "d.mkv"}) {
        add_row(window, name);
    }
    table(window).setRangeSelected(QTableWidgetSelectionRange(1, 0, 2, table(window).columnCount() - 1), true);

    drag_rows(window, {1, 2}, 4);
    EXPECT_EQ(names(window), (std::vector<std::string>{"a.mkv", "d.mkv", "b.mkv", "c.mkv"}));
    EXPECT_EQ(selected_rows(window), (std::vector<int>{2, 3}));

    drag_rows(window, {3}, 0);
    EXPECT_EQ(names(window), (std::vector<std::string>{"c.mkv", "a.mkv", "d.mkv", "b.mkv"}));
}

TEST_F(WindowTest, AClickOnAHeadingSortsByItWithRowsWithoutAValueLastEitherWay) {
    MainWindow window;
    window.show();
    // File names sort as a person reads them, in the Python app's order: 9
    // before 10 and 007 before 9, whatever the case, digits of any script
    // as numbers, and a superscript as text.
    const std::string arabic_three = "clip ٣.mkv";
    const std::string superscript = "clip 10²5.mkv";
    for (const std::string& name :
         {std::string("clip 10.mkv"), std::string("Clip 9.mkv"), std::string("clip 1.mkv"), std::string("clip 2.mkv"),
          superscript, arabic_three, std::string("clip 007.mkv")}) {
        add_row(window, name);
    }
    click_heading(window, col_path, false);
    EXPECT_EQ(names(window), (std::vector<std::string>{"clip 1.mkv", "clip 2.mkv", arabic_three, "clip 007.mkv",
                                                       "Clip 9.mkv", "clip 10.mkv", superscript}));
    click_heading(window, col_path, false);
    EXPECT_EQ(names(window), (std::vector<std::string>{superscript, "clip 10.mkv", "Clip 9.mkv", "clip 007.mkv",
                                                       arabic_three, "clip 2.mkv", "clip 1.mkv"}));

    // A video whose bitrate is not known goes last, whichever the order.
    const std::map<std::string, long long> bitrates{
        {"clip 1.mkv", 1'000'000}, {"clip 2.mkv", 9'000'000}, {"clip 10.mkv", 4'000'000}, {"Clip 9.mkv", 0},
        {superscript, 2'000'000},  {arabic_three, 0},          {"clip 007.mkv", 6'000'000}};
    for (const auto& row : rows(window)) {
        VideoInfo info;
        info.bit_rate = bitrates.at(platform::path_text(row->path));
        row->video_info = info;
    }
    click_heading(window, col_bitrate, false);
    EXPECT_EQ(names(window), (std::vector<std::string>{"clip 1.mkv", superscript, "clip 10.mkv", "clip 007.mkv",
                                                       "clip 2.mkv", "Clip 9.mkv", arabic_three}));
    click_heading(window, col_bitrate, false);
    EXPECT_EQ(names(window), (std::vector<std::string>{"clip 2.mkv", "clip 007.mkv", "clip 10.mkv", superscript,
                                                       "clip 1.mkv", "Clip 9.mkv", arabic_three}));
}

// --------------------------------------------------------- adding videos

TEST_F(WindowTest, FilesDroppedAreAddedOnceReadOffTheWindowsThreadAndEachListOpensItsOwnFolder) {
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    // Again in the same drop, and a folder, which is left out.
    drop(table(window).viewport(),
         {clip("encodes/a.mkv"), clip("encodes/b.mkv"), clip("encodes/a.mkv"), clip("encodes")});
    ASSERT_EQ(names(window), (std::vector<std::string>{"a.mkv", "b.mkv"}));
    EXPECT_EQ(table(window).item(0, col_info)->text().toStdString(), "Probing...");  // read on a thread
    ASSERT_TRUE(wait_until([&] { return !reading(window); }));
    EXPECT_EQ(table(window).item(0, col_info)->text().toStdString(), "640x360, 23.98fps, h264");
    EXPECT_EQ(table(window).item(1, col_info)->text().toStdString(), "480x270, 23.98fps, hevc");
    EXPECT_FALSE(table(window).item(0, col_bitrate)->text().isEmpty());

    // The reference box takes one video, and reads it.
    drop(&source_edit(window), {clip("sources/ref.mkv")});
    EXPECT_FALSE(source_info(window));
    ASSERT_TRUE(wait_until([&] { return source_info(window).has_value() && !reading(window); }));
    EXPECT_EQ(source_edit(window).text().toStdString(), platform::path_text(clip("sources/ref.mkv")));
    EXPECT_TRUE(source_info_label(window).text().startsWith("640x360, 23.98fps, h264"));

    // Each list's dialog opens where its videos came from, kept for the next
    // start.
    EXPECT_EQ(dialog_folder(window, "reference").toStdString(), platform::path_text(clip("sources")));
    EXPECT_EQ(dialog_folder(window, "test").toStdString(), platform::path_text(clip("encodes")));
    EXPECT_EQ(saved_settings()["last_reference_dir"], platform::path_text(clip("sources")));
    EXPECT_EQ(saved_settings()["last_test_dir"], platform::path_text(clip("encodes")));

    // A video already in the table is not added again: on Windows, whatever
    // the case of its letters.
    std::string again = platform::path_text(clip("encodes/a.mkv"));
#ifdef _WIN32
    std::transform(again.begin(), again.end(), again.begin(), [](unsigned char c) { return std::toupper(c); });
#endif
    drop(table(window).viewport(), {platform::path_from_text(again)});
    EXPECT_EQ(table(window).rowCount(), 2);
}

TEST_F(WindowTest, AReferencePathPastedWithQuotesIsReadWhenEnterIsPressed) {
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    window.show();
    source_edit(window).setFocus();
    // As Explorer's "Copy as path" gives it.
    source_edit(window).setText(QString::fromStdString("\"" + platform::path_text(clip("sources/ref.mkv")) + "\""));
    QTest::keyClick(&source_edit(window), Qt::Key_Return);
    ASSERT_TRUE(wait_until([&] { return source_info(window).has_value() && !reading(window); }));
    EXPECT_EQ(source_edit(window).text().toStdString(), platform::path_text(clip("sources/ref.mkv")));
}

TEST_F(WindowTest, APathLeftInTheReferenceBoxIsNotReadAsTheWindowCloses) {
    // The focus leaves the box as the window hides, and the box read the
    // path left in it: in the Python app, the reading outlived the window and
    // crashed the app as it closed.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    window.show();
    window.activateWindow();
    QApplication::processEvents();
    source_edit(window).setFocus();
    QTest::keyClicks(&source_edit(window), QString::fromStdString(platform::path_text(clip("sources/ref.mkv"))));
    ASSERT_TRUE(source_edit(window).hasFocus());
    window.close();
    EXPECT_FALSE(window.isVisible());
    EXPECT_FALSE(reading(window));
}

TEST_F(WindowTest, AWindowEndedWithAPathTypedInTheReferenceBoxEndsCleanly) {
    // Ended while shown, the window loses the focus as it goes, and the box
    // reads what was typed: that ran on a half-destroyed window, which
    // started reading the video and crashed (0xC0000409).
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    auto window = std::make_unique<MainWindow>();
    window->show();
    window->activateWindow();
    QApplication::processEvents();
    source_edit(*window).setFocus();
    QTest::keyClicks(&source_edit(*window), QString::fromStdString(platform::path_text(clip("sources/ref.mkv"))));
    ASSERT_TRUE(source_edit(*window).hasFocus());
    window.reset();
}

TEST_F(WindowTest, TheVideosOpenAtClosingAreReopenedThoseStillThere) {
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    write_settings({{"remember_videos", true}});
    const std::filesystem::path a = folder("a.mkv");
    const std::filesystem::path b = folder("b.mkv");
    std::filesystem::copy_file(clip("encodes/a.mkv"), a);
    std::filesystem::copy_file(clip("encodes/b.mkv"), b);
    {
        MainWindow window;
        choose_source(window, clip("sources/ref.mkv"));
        drop(table(window).viewport(), {a, b});
        ASSERT_TRUE(wait_until([&] { return source_info(window).has_value() && !reading(window); }));
        window.close();
    }
    EXPECT_EQ(saved_settings()["remembered_reference"], platform::path_text(clip("sources/ref.mkv")));
    EXPECT_EQ(saved_settings()["remembered_tests"],
              nlohmann::json::array({platform::path_text(a), platform::path_text(b)}));
    std::filesystem::remove(a);

    MainWindow reopened;
    // Looked for on a thread of its own (a network drive can take long).
    ASSERT_TRUE(wait_until([&] { return table(reopened).rowCount() > 0 && !reading(reopened); }));
    EXPECT_EQ(names(reopened), (std::vector<std::string>{"b.mkv"}));
    EXPECT_EQ(source_edit(reopened).text().toStdString(), platform::path_text(clip("sources/ref.mkv")));
}

// ---------------------------------------------------------------- the run

// What a run asks of the PC's sleep, in order, in place of the request
// itself; the tests' own stand-in is put back as it goes.
class KeepAwakeCalls {
public:
    KeepAwakeCalls() {
        set_keep_awake([this](bool awake) {
            calls.push_back(awake);
            return true;
        });
    }
    ~KeepAwakeCalls() {
        set_keep_awake([](bool /*awake*/) { return false; });
    }
    KeepAwakeCalls(const KeepAwakeCalls&) = delete;
    KeepAwakeCalls& operator=(const KeepAwakeCalls&) = delete;

    std::vector<bool> calls;
};

// A score as the command line's JSON gives it: "inf" is text there.
std::optional<double> score_in(const nlohmann::json& score) {
    if (score.is_string()) {
        return score == "inf" ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }
    return score.is_null() ? std::nullopt : std::optional(score.get<double>());
}

TEST_F(WindowTest, ARunPausedOnTheWayScoresAsTheCommandLineAndItsTimeLeavesThePauseOut) {
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    KeepAwakeCalls awake;
    MainWindow window;
    double now = 100;
    set_clock(window, [&now] { return now; });
    open_clips(window);

    run_button(window).click();
    // Frozen at once: nothing that changes what a running video means can
    // be used, and the PC is kept awake.
    EXPECT_TRUE(running(window));
    EXPECT_FALSE(run_button(window).isEnabled());
    EXPECT_TRUE(source_edit(window).isReadOnly());
    EXPECT_EQ(awake.calls, std::vector<bool>{true});
    now = 103;
    pause_button(window).click();
    EXPECT_EQ(pause_button(window).text().toStdString(), "Resume");
    EXPECT_EQ(status(window), "Paused   ·   2 videos: 2 queued   ·   Elapsed: 0:00:03");
    now = 160;
    pause_button(window).click();
    now = 170;
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    EXPECT_EQ(status(window), "Done in 0:00:13.");
    EXPECT_EQ(awake.calls, (std::vector<bool>{true, false}));
    EXPECT_TRUE(run_button(window).isEnabled());
    EXPECT_FALSE(source_edit(window).isReadOnly());

    // The command line's scores for the same videos, with the same settings,
    // calculated afresh rather than read from what the window saved, are in
    // the metrics' cells.
    ASSERT_TRUE(wait_for_writes(window));
    const platform::ProcessOutput command_line = platform::run_program(
        {VML_CLI, "compare", platform::path_text(clip("sources/ref.mkv")), platform::path_text(clip("encodes/a.mkv")),
         platform::path_text(clip("encodes/b.mkv")), "--recalculate", "--json", "-", "--quiet"},
        std::chrono::minutes(2));
    ASSERT_EQ(command_line.exit_code, 0) << command_line.standard_error;
    const nlohmann::json results = nlohmann::json::parse(command_line.standard_output);
    for (int row = 0; row < 2; ++row) {
        const nlohmann::json& scores = results["videos"][row]["metrics"];
        ASSERT_EQ(scores.size(), 4U);  // VMAF, PSNR, SSIM and XPSNR
        for (const MetricColumn& column : metric_columns()) {
            if (scores.contains(column.key)) {
                EXPECT_EQ(table(window).item(row, column.column)->text().toStdString(),
                          metric_definition(column.key).format_value(score_in(scores[column.key]["score"])))
                    << row << " " << column.key;
            }
        }
        EXPECT_EQ(state(window, row), RowState::complete);
    }

    // Ticked again, every video has its scores: nothing is calculated.
    run_button(window).click();
    EXPECT_FALSE(running(window));
    EXPECT_EQ(status(window), "All 2 checked videos already have all requested metrics -- nothing to run.");
}

TEST_F(WindowTest, ARunsScoresAreSavedForTheNextTimeItsVideosAreAddedUnlessRecalculated) {
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    std::vector<std::string> saved;  // the first video's scores, as its cells show them
    {
        MainWindow window;
        open_clips(window);
        run_button(window).click();
        ASSERT_TRUE(wait_until([&] { return !running(window); }));
        for (const MetricColumn& column : metric_columns()) {
            saved.push_back(table(window).item(0, column.column)->text().toStdString());
        }
        // Recalculate forgets the second video's: its cells are tick boxes again.
        recalculate(window, {1});
        EXPECT_EQ(state(window, 1), RowState::not_calculated);
        EXPECT_TRUE(table(window).item(1, col_psnr)->text().isEmpty());
        ASSERT_TRUE(wait_for_writes(window));
    }

    // The command line finds the first video's scores as the window saved
    // them...
    const platform::ProcessOutput command_line = platform::run_program(
        {VML_CLI, "compare", platform::path_text(clip("sources/ref.mkv")), platform::path_text(clip("encodes/a.mkv")),
         "--json", "-", "--quiet"},
        std::chrono::minutes(2));
    ASSERT_EQ(command_line.exit_code, 0) << command_line.standard_error;
    const nlohmann::json found = nlohmann::json::parse(command_line.standard_output)["videos"][0];
    EXPECT_TRUE(found["from_saved_scores"].get<bool>());

    // ...and a window opened later shows them, and none for the second.
    MainWindow reopened;
    open_clips(reopened);
    EXPECT_EQ(state(reopened, 0), RowState::cached);
    std::vector<std::string> shown;
    for (const MetricColumn& column : metric_columns()) {
        const QTableWidgetItem* cell = table(reopened).item(0, column.column);
        shown.push_back(cell->text().toStdString());
        if (found["metrics"].contains(column.key)) {
            EXPECT_EQ(cell->text().toStdString(),
                      metric_definition(column.key).format_value(score_in(found["metrics"][column.key]["score"])))
                << column.key;
        }
    }
    EXPECT_EQ(shown, saved);
    EXPECT_EQ(state(reopened, 1), RowState::not_calculated);
    EXPECT_TRUE(table(reopened).item(1, col_psnr)->text().isEmpty());
}

TEST_F(WindowTest, HighlightBestWorstColoursEachMetricsBestAndWorstScore) {
    // A user's request in the Python app, as FFMetrics colours its table:
    // ticked, each metric's best score is green and its worst red among the
    // test videos -- by the metric's direction (Butteraugli's lowest is its
    // best), only where two or more have one, and not a score marked as the
    // other implementation's. Off by default, and kept in the settings.
    MainWindow window;
    ASSERT_FALSE(highlight_check(window).isChecked());
    const MetricProvenance libvmaf{"ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "vmaf-libvmaf-v1", {}};
    const MetricProvenance vship{"Vship/butteraugli", "Vship 5.1.2", "gpu", "butteraugli-vship-gpu-v1", {}};
    for (const auto& [name, vmaf, butteraugli] :
         {std::tuple{"a.mp4", 90.0f, 1.2f}, std::tuple{"b.mp4", 95.0f, 0.9f}, std::tuple{"c.mp4", 85.0f, 2.0f}}) {
        const int row = add_row(window, name);
        auto run = std::make_shared<CompletedRun>();
        run->result.metric_results.add(FrameMetricResult{"vmaf", {0}, {0.0}, {vmaf}, libvmaf});
        run->result.metric_results.add(FrameMetricResult{"butteraugli", {0}, {0.0}, {butteraugli}, vship});
        RowData& row_data = *rows(window)[static_cast<std::size_t>(row)];
        row_data.completed_run = run;
        row_data.metric_backends["butteraugli"] = "gpu";
        show_scores(window, row);
    }
    const auto marks = [&window](int column) {
        EXPECT_TRUE(wait_until([&] { return !best_worst_due(window); }));
        std::vector<std::string> found;
        for (int row = 0; row < table(window).rowCount(); ++row) {
            const QColor colour = table(window).item(row, column)->background().color();
            found.emplace_back(colour.alpha() == 0                  ? ""
                               : colour == theme::color("best")  ? "best"
                               : colour == theme::color("worst") ? "worst"
                                                                 : colour.name().toStdString());
        }
        return found;
    };
    using Marks = std::vector<std::string>;

    EXPECT_EQ(marks(col_vmaf), (Marks{"", "", ""}));
    highlight_check(window).setChecked(true);
    EXPECT_TRUE(saved_settings()["highlight_best_worst"].get<bool>());
    EXPECT_EQ(marks(col_vmaf), (Marks{"", "best", "worst"}));
    EXPECT_EQ(marks(col_butteraugli), (Marks{"", "best", "worst"}));
    // b.mp4 set to the CPU: its GPU score shows "(GPU)", on another scale.
    rows(window)[1]->metric_backends["butteraugli"] = "cpu";
    show_scores(window, 1);
    EXPECT_EQ(marks(col_butteraugli), (Marks{"best", "", "worst"}));
    // The best video removed: the better of the two left is the best.
    table(window).selectRow(1);
    remove_selected(window);
    EXPECT_EQ(marks(col_vmaf), (Marks{"best", "worst"}));
    // One video left: nothing to compare.
    table(window).selectRow(1);
    remove_selected(window);
    EXPECT_EQ(marks(col_vmaf), (Marks{""}));
    highlight_check(window).setChecked(false);
    EXPECT_FALSE(saved_settings()["highlight_best_worst"].get<bool>());
}

TEST_F(WindowTest, ScoresSavedSinceAVideoWasAddedComeBackWhenItsMetricsAreTickedOrShown) {
    // As in the Python app, a row's saved scores are looked up again when
    // its metrics are ticked or one is shown again. Otherwise scores saved
    // meanwhile (here by the command line) stay away, and so does every
    // score of a row ticked while its lookup was reading: the answer is for
    // the row as it was, and refused.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    const auto saved_by_command_line = [](const std::string& test_video) {
        const platform::ProcessOutput output = platform::run_program(
            {VML_CLI, "compare", platform::path_text(clip("sources/ref.mkv")), platform::path_text(clip(test_video)),
             "--json", "-", "--quiet"},
            std::chrono::minutes(2));
        ASSERT_EQ(output.exit_code, 0) << output.standard_error;
    };
    MainWindow window;
    open_clips(window);
    ASSERT_TRUE(table(window).item(0, col_psnr)->text().isEmpty());  // none saved yet

    saved_by_command_line("encodes/a.mkv");
    table(window).item(0, col_xpsnr)->setCheckState(Qt::Unchecked);
    ASSERT_TRUE(wait_until([&] { return !reading(window); }));
    EXPECT_FALSE(table(window).item(0, col_psnr)->text().isEmpty());

    show_metric(window, "ssim", false);
    saved_by_command_line("encodes/b.mkv");
    show_metric(window, "ssim", true);
    ASSERT_TRUE(wait_until([&] { return !reading(window); }));
    EXPECT_FALSE(table(window).item(1, col_psnr)->text().isEmpty());
}

TEST_F(WindowTest, ANewReferenceTakesAwayTheScoresMeasuredAgainstTheOldOne) {
    // Kept, they stood in the table as the new pair's scores, and a run
    // calculated nothing: every row "already had" its scores.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    open_clips(window);
    run_button(window).click();
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    ASSERT_EQ(state(window, 0), RowState::complete);
    const std::string score = table(window).item(0, col_psnr)->text().toStdString();

    // The reference chosen again keeps them...
    choose_source(window, clip("sources/ref.mkv"));
    ASSERT_TRUE(wait_until([&] { return !reading(window); }));
    EXPECT_EQ(state(window, 0), RowState::complete);
    EXPECT_EQ(table(window).item(0, col_psnr)->text().toStdString(), score);

    // ...and another reference takes them away: the next run calculates them.
    choose_source(window, clip("encodes/b.mkv"));
    ASSERT_TRUE(wait_until([&] { return source_info(window)->path == clip("encodes/b.mkv") && !reading(window); }));
    EXPECT_EQ(state(window, 0), RowState::not_calculated);
    EXPECT_TRUE(table(window).item(0, col_psnr)->text().isEmpty());
    run_button(window).click();
    EXPECT_TRUE(running(window));
    cancel_button(window).click();
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
}

TEST_F(WindowTest, CancelEndsAPausedRunAndMarksTheVideoItWasCalculating) {
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    open_clips(window);
    run_button(window).click();
    pause_button(window).click();
    // Paused, the first video starts and goes no further, its line saying
    // where it stopped; the second waits for its turn.
    ASSERT_TRUE(wait_until([&] { return state(window, 0) == RowState::calculating; }));
    EXPECT_TRUE(wait_until([&] {
        return job_line(window, 0) == "a — CPU metrics 1–4 of 4: VMAF v0.6.1, PSNR, SSIM, XPSNR (Detecting black bars "
                                      "in source and test video)";
    })) << job_line(window, 0);
    cancel_button(window).click();
    EXPECT_EQ(status(window), "Cancelling...");
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    EXPECT_TRUE(status(window).starts_with("Cancelled after ")) << status(window);
    EXPECT_EQ(state(window, 0), RowState::cancelled);
    EXPECT_EQ(state(window, 1), RowState::not_calculated);  // as it was
    EXPECT_FALSE(pause_button(window).isChecked());
    EXPECT_EQ(pause_button(window).text().toStdString(), "Pause");
}

TEST_F(WindowTest, ClosingTheWindowDuringARunEndsTheRunAndThenTheWindow) {
    // Closed while its thread still wrote into the widgets, the window would
    // crash the app and leave its programs running.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    window.show();
    open_clips(window);
    run_button(window).click();
    pause_button(window).click();
    ASSERT_TRUE(wait_until([&] { return state(window, 0) == RowState::calculating; }));
    window.close();
    EXPECT_TRUE(window.isVisible());
    EXPECT_EQ(status(window), "Finishing up; the window will close on its own.");
    ASSERT_TRUE(wait_until([&] { return !window.isVisible(); }));
    EXPECT_FALSE(running(window));
}

// ------------------------------------------------------------ the window

TEST_F(WindowTest, MissingToolsShowABannerWithAButtonToFindThem) {
    std::filesystem::create_directories(folder("no ffmpeg"));
    write_settings({{"ffmpeg_dir", platform::path_text(folder("no ffmpeg"))}});
    const test::ScopedVariable path("PATH", "");
    MainWindow window;
    EXPECT_TRUE(ffmpeg_banner(window).isVisibleTo(&window));
    EXPECT_TRUE(locate_ffmpeg_button(window).isVisibleTo(&window));
    EXPECT_TRUE(ffmpeg_banner(window).text().contains("ffmpeg"));
    EXPECT_TRUE(ffmpeg_banner(window).text().contains("ffprobe"));
}

TEST_F(WindowTest, TheWindowIsShownInTheLanguageChosen) {
    // The texts expected, from the catalog the window reads.
    const nlohmann::json catalog = nlohmann::json::parse(
        platform::read_file(resource_folder() / "translations" / "de.json").value_or("{}"))["strings"];
    const auto german = [&catalog](const char* english) {
        const std::string text = catalog.value(english, std::string());
        EXPECT_FALSE(text.empty() || text == english) << english;  // or the test would prove nothing
        return text;
    };
    struct English {
        ~English() { set_language("en"); }
    } english;
    ASSERT_EQ(set_language("de").toStdString(), "de");

    MainWindow window;
    add_row(window, "a.mp4");
    EXPECT_EQ(tabs(window).tabText(tab_graph).toStdString(), german("Metric Graphs"));
    EXPECT_EQ(run_button(window).text().toStdString(), german("Calculate metrics"));
    EXPECT_TRUE(table(window).item(0, col_path)->toolTip().contains(QString::fromStdString(german("Not calculated"))));
    EXPECT_TRUE(table(window).item(0, col_psnr)->toolTip().contains(
        QString::fromStdString(german("Ticked: calculated on the next run. Untick to skip it."))));
}

TEST_F(WindowTest, ASwitchOfThemeRecoloursEveryColouredCellAndLeavesTheRest) {
    MainWindow window;
    window.show();
    const int row = add_row(window, "a.mkv");
    probed(window, {"a.mkv", std::nullopt, "could not read it"});  // "Probe failed", in the failed colour
    QTableWidgetItem* info = table(window).item(row, col_info);
    QTableWidgetItem* plain = table(window).item(row, col_vmaf);
    const bool dark = theme::is_dark();
    ASSERT_EQ(info->foreground().color(), theme::color("failed", dark));
    ASSERT_NE(theme::color("failed", !dark), theme::color("failed", dark));

    // The other theme's palette, as Windows gives it on a switch (the
    // platform with no screen takes no colour scheme of its own).
    struct Usual {
        QPalette palette;
        ~Usual() { QApplication::setPalette(palette); }
    } usual{QApplication::palette()};
    QPalette other = usual.palette;
    other.setColor(QPalette::Window, dark ? QColor(240, 240, 240) : QColor(32, 32, 32));
    QApplication::setPalette(other);
    ASSERT_NE(theme::is_dark(), dark);
    // The window recolours once the switch is done.
    EXPECT_TRUE(wait_until([&] { return info->foreground().color() == theme::color("failed", !dark); }));
    EXPECT_FALSE(plain->data(Qt::ForegroundRole).isValid());
}

TEST_F(WindowTest, NoTabIsSqueezedInTheSmallestWindowWithLargeText) {
    // Issue #6: with large text, the window could be made smaller than a tab
    // needed, and Qt then squeezed its rows into each other.
    const QFont usual = QApplication::font();
    QFont large = usual;
    large.setPointSizeF(usual.pointSizeF() * 1.5);
    QApplication::setFont(large);
    struct Usual {
        QFont font;
        ~Usual() { QApplication::setFont(font); }
    } restore{usual};

    MainWindow window;
    window.resize(window.minimumSize());
    window.show();
    for (int index = 0; index < tabs(window).count(); ++index) {
        tabs(window).setCurrentIndex(index);
        QApplication::processEvents();
        QWidget* page = tabs(window).widget(index);
        auto* scrolling = qobject_cast<QScrollArea*>(page);
        QWidget* content = scrolling != nullptr ? scrolling->widget() : page;
        const QSize needed = content->minimumSizeHint();
        const QSize given = content->size();
        EXPECT_TRUE(given.width() >= needed.width() && given.height() >= needed.height())
            << tabs(window).tabText(index).toStdString();
    }
}

}  // namespace
}  // namespace vml::app
