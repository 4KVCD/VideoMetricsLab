// The main window's tests: the Python app's tests of the same behaviour
// (tests/test_main_window.py, tests/test_video_list_editing.py), for what
// main_window.h says has been ported.
#include "main_window.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDeadlineTimer>
#include <QDoubleSpinBox>
#include <QDropEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMimeData>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
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
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "bitrate_panel.h"
#include "cvvdp_display_dialog.h"
#include "file_writes.h"
#include "frame_compare_panel.h"
#include "graph_panel.h"
#include "i18n.h"
#include "options_panel.h"
#include "settings_panel.h"
#include "test_videos.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/gpu_vmaf.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/run_file.h"
#include "vml/vship.h"
#include "widgets.h"

namespace vml::app {

// The window's parts, for its tests: the one class MainWindow lets in.
class WindowTest : public ::testing::Test {
protected:
    // Clips of FFmpeg's test pattern, made once for every test: a reference
    // in one folder and two test videos in another, as videos often are, and
    // a reference wider than the narrowest resolution test (854).
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
        test::make_clip(*found_, *clips_ / "sources" / "wide.mkv", "libx264", "yuv420p", 0.5, {}, "1280x720");
    }

    static void TearDownTestSuite() { clips_.reset(); }

#ifdef _WIN32
    static constexpr const char* home_variable = "USERPROFILE";
#else
    static constexpr const char* home_variable = "HOME";
#endif

    // A profile of the test's own, so that the user's real settings and
    // saved scores are never read or written. Its settings.json ticks VMAF,
    // PSNR, SSIM and XPSNR, as the Python app's tests' does, with VMAF on
    // the CPU: on the GPU, a run of these half-second clips would be mostly
    // the GPU starting, in a process of its own.
    // (The home folder is USERPROFILE on Windows and HOME elsewhere: on macOS
    // these tests wrote the user's real settings while USERPROFILE alone was
    // set.)
    WindowTest()
        : profile_(folder_ / "profile"), user_profile_(home_variable, platform::path_text(profile_)) {
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
    // "sources/wide.mkv", "encodes/a.mkv" or "encodes/b.mkv".
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
    // A run asked for and not over: running, or waiting for the GPU checks
    // it needs (MainWindow::wait_for_gpu_checks).
    static bool running(MainWindow& window) { return window.run_active_ || window.gpu_check_watcher_ != nullptr; }
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
    // A resolution test of the reference, as Add resolution test adds the
    // one chosen.
    static void add_resample_test(MainWindow& window, const ResampleTarget& target) {
        window.add_resample_test(target);
    }
    // The window's information messages, title and text: kept, in the order
    // they come, rather than shown.
    static std::shared_ptr<std::vector<std::pair<QString, QString>>> keep_information(MainWindow& window) {
        auto kept = std::make_shared<std::vector<std::pair<QString, QString>>>();
        window.dialogs_.inform = [kept](const QString& title, const QString& text) { kept->emplace_back(title, text); };
        return kept;
    }
    // Answers the window's Yes or No questions with `yes`, keeping their
    // titles in the order they are asked.
    static std::shared_ptr<std::vector<QString>> answer_questions(MainWindow& window, bool yes) {
        auto asked = std::make_shared<std::vector<QString>>();
        window.dialogs_.ask = [asked, yes](const QString& title, const QString& /*text*/) {
            asked->push_back(title);
            return yes;
        };
        return asked;
    }
    // Answers the dialogs that open or save a file with `path`.
    static void choose_file(MainWindow& window, const std::filesystem::path& path) {
        const QString text = QString::fromStdString(platform::path_text(path));
        window.dialogs_.open_file = [text](const QString& /*title*/, const QString& /*filter*/) { return text; };
        window.dialogs_.save_file = [text](const QString& /*title*/, const QString& /*suggested*/,
                                   const QString& /*filter*/) { return text; };
    }
    static QPushButton& save_button(MainWindow& window) { return *window.save_button_; }
    static QPushButton& load_button(MainWindow& window) { return *window.load_button_; }
    // Load analysis results..., answered by choose_file.
    static void load_analysis_results(MainWindow& window) { window.on_load_saved_run(); }
    static OptionsPanel& options_panel(MainWindow& window) { return *window.options_panel_; }
    static SettingsPanel& settings_panel(MainWindow& window) { return *window.settings_panel_; }
    // GitHub's answer to the update check, or why it could not be asked.
    static void answer_update(MainWindow& window, const std::string& body, const QString& error) {
        window.on_update_answer(body, error);
    }
    // The newer release's message, while it is open.
    static QMessageBox* update_box(MainWindow& window) { return window.update_box_.data(); }
    // The window's settings now, as settings.json would hold them.
    static nlohmann::json settings_now(MainWindow& window) {
        return nlohmann::json::parse(window.settings_.text());
    }
    // Holds the queue of file writes until the promise is kept: what is
    // queued after waits, as behind a feature-length result being saved.
    static void hold_writes(MainWindow& window, std::shared_future<void> release) {
        window.file_writes_->submit("hold", [release] { release.wait(); });
    }
    // Answers the CVVDP display editor in the user's place when it opens.
    static void answer_cvvdp_dialog(MainWindow& window, std::function<int(CvvdpDisplayDialog&)> answer) {
        window.run_cvvdp_dialog_ = std::move(answer);
    }
    // Chooses the CVVDP preset called `name` in the Options panel.
    static void choose_cvvdp_preset(MainWindow& window, const std::string& name) {
        QComboBox& presets = *options_panel(window).cvvdp_preset_combo;
        presets.setCurrentIndex(presets.findData(QString::fromStdString(name)));
    }
    // Selects rows `first` to `last`, as a drag over them does.
    static void select_rows(MainWindow& window, int first, int last) {
        table(window).clearSelection();
        table(window).setRangeSelected(
            QTableWidgetSelectionRange(first, 0, last, table(window).columnCount() - 1), true);
    }
    static QString dialog_folder(MainWindow& window, const QString& kind) { return window.video_dialog_dir(kind); }
    // The test videos with a line on Metric Graphs, in the order added.
    static std::vector<std::string> graphed(MainWindow& window) {
        std::vector<std::string> names;
        for (const auto& [id, entry] : window.graph_panel_->entries_) {
            names.push_back(platform::path_text(entry.result->distorted.filename()));
        }
        return names;
    }
    // The pairs Video Compare lists, top to bottom: each one's name, and
    // whether it has scores to show.
    static std::vector<std::pair<std::string, bool>> compared(MainWindow& window) {
        std::vector<std::pair<std::string, bool>> listed;
        for (const FrameComparisonEntry& entry : window.frame_compare_panel_->entries_) {
            listed.emplace_back(entry.label.toStdString(), entry.scores != nullptr);
        }
        return listed;
    }

    // The Bitrate Viewer's list of videos.
    static const QTableWidget& bitrate_table(MainWindow& window) { return *window.bitrate_panel_->table; }

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
        // the drag as it entered, and to no other. Qt 6.12 takes the point as
        // a QPointF and deprecates the QPoint that Qt 6.11 and older take.
#if QT_VERSION >= QT_VERSION_CHECK(6, 12, 0)
        QDragEnterEvent enter(QPointF(5, 5), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
#else
        QDragEnterEvent enter(QPoint(5, 5), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
#endif
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
    // The GPU checks done, as the app starts them as it opens: a run asked
    // for before they are done waits for them (wait_for_gpu_checks).
    (void)detect_vship_device();
    (void)gpu_vmaf_backend();

    run_button(window).click();
    // Frozen at once: nothing that changes what a running video means can
    // be used, and the PC is kept awake.
    EXPECT_TRUE(running(window));
    EXPECT_FALSE(run_button(window).isEnabled());
    EXPECT_TRUE(source_edit(window).isReadOnly());
    EXPECT_FALSE(load_button(window).isEnabled());  // a saved run can bring another reference
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
    EXPECT_TRUE(load_button(window).isEnabled());

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

TEST_F(WindowTest, AnOptionEditedForSeveralVideosChangesThatOptionAloneInEach) {
    // Copying the whole panel over them would give every selected video the
    // first one's other settings.
    MainWindow window;
    const int a = add_row(window, "a.mp4");
    const int b = add_row(window, "b.mp4");
    const int c = add_row(window, "c.mp4");
    RowData& first = *rows(window)[static_cast<std::size_t>(a)];
    RowData& second = *rows(window)[static_cast<std::size_t>(b)];
    first.options.n_threads = 2;
    first.options.gpu_decode = false;
    second.options.n_threads = 11;
    select_rows(window, a, b);
    EXPECT_EQ(options_panel(window).target_label->text().toStdString(),
              "Editing settings for 2 selected videos -- editing anything below applies to all of them.");

    options_panel(window).crop_combo->setCurrentIndex(1);  // None (use full frame)

    EXPECT_EQ(first.options.crop_mode, CropMode::none);
    EXPECT_EQ(second.options.crop_mode, CropMode::none);
    EXPECT_EQ(rows(window)[static_cast<std::size_t>(c)]->options.crop_mode, CropMode::automatic);
    EXPECT_EQ(first.options.n_threads, 2);
    EXPECT_EQ(second.options.n_threads, 11);
    EXPECT_FALSE(first.options.gpu_decode);
    EXPECT_TRUE(second.options.gpu_decode);
}

TEST_F(WindowTest, TheOptionsPanelShowsTheSelectedVideosOwnSettings) {
    MainWindow window;
    EXPECT_FALSE(options_panel(window).isEnabled());  // nothing selected
    const int a = add_row(window, "a.mp4");
    const int b = add_row(window, "b.mp4");
    select_rows(window, a, a);
    EXPECT_TRUE(options_panel(window).isEnabled());
    EXPECT_EQ(options_panel(window).target_label->text().toStdString(), "Editing settings for: a.mp4");
    options_panel(window).crop_combo->setCurrentIndex(1);
    options_panel(window).subsample_spin->setValue(4);
    select_rows(window, b, b);
    EXPECT_EQ(options_panel(window).crop_combo->currentIndex(), 0);
    EXPECT_EQ(options_panel(window).subsample_spin->value(), 1);
    select_rows(window, a, a);
    EXPECT_EQ(options_panel(window).crop_combo->currentIndex(), 1);
    EXPECT_EQ(options_panel(window).subsample_spin->value(), 4);
}

TEST_F(WindowTest, AnOptionThatChangesWhatIsComparedTakesTheScoresAwayOneThatChangesHowKeepsThem) {
    // Which GPU, how many threads, and how frames are scaled change how a
    // comparison is made, not what it is.
    MainWindow window;
    const int row = add_row(window, "a.mp4");
    RowData& row_data = *rows(window)[static_cast<std::size_t>(row)];
    const auto run = std::make_shared<CompletedRun>();
    run->result.metric_results.add(FrameMetricResult{
        "vmaf", {0}, {0.0}, {90.0f}, MetricProvenance{"ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "vmaf", {}}});
    row_data.completed_run = run;
    show_scores(window, row);
    select_rows(window, row, row);

    OptionsPanel& panel = options_panel(window);
    panel.gpu_checkbox->setChecked(!panel.gpu_checkbox->isChecked());
    panel.threads_spin->setValue(7);
    panel.scale_algo_combo->setCurrentText("lanczos");
    EXPECT_EQ(row_data.options.n_threads, 7);
    EXPECT_EQ(row_data.options.scale_algorithm, "lanczos");
    EXPECT_EQ(row_data.completed_run, run);

    panel.subsample_spin->setValue(5);
    EXPECT_EQ(row_data.options.n_subsample, 5);
    EXPECT_EQ(row_data.completed_run, nullptr);
    EXPECT_TRUE(table(window).item(row, col_vmaf)->text().isEmpty());
}

TEST_F(WindowTest, WhereSsimulacra2IsCalculatedIsSetForTheSelectedVideosAndKeptForTheNextSession) {
    MainWindow window;
    const int a = add_row(window, "a.mp4");
    const int b = add_row(window, "b.mp4");
    select_rows(window, a, a);

    options_panel(window).ssimulacra2_backend_combo->setCurrentIndex(1);  // CPU

    EXPECT_EQ(rows(window)[static_cast<std::size_t>(a)]->metric_backends.at("ssimulacra2"), "cpu");
    EXPECT_EQ(rows(window)[static_cast<std::size_t>(b)]->metric_backends.at("ssimulacra2"), "gpu");
    EXPECT_EQ(saved_settings()["default_ssimulacra2_backend"], "cpu");
    EXPECT_EQ(rows(window)[static_cast<std::size_t>(add_row(window, "c.mp4"))]->metric_backends.at("ssimulacra2"),
              "cpu");
}

TEST_F(WindowTest, ChoosingAnotherDisplayDropsOnlyTheCvvdpScore) {
    MainWindow window;
    const int row = add_row(window, "test.mp4");
    RowData& row_data = *rows(window)[static_cast<std::size_t>(row)];
    row_data.extra_metric_keys.insert("cvvdp");
    const auto run = std::make_shared<CompletedRun>();
    run->result.metric_results.add(FrameMetricResult{
        "vmaf", {0}, {0.0}, {90.0f}, MetricProvenance{"ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "vmaf", {}}});
    run->result.metric_results.add(SequenceMetricResult{
        "cvvdp", 9.5, MetricProvenance{"Vship/cvvdp", "Vship 5.1.2", "gpu", "cvvdp-vship-gpu-v1", {}}, {}, {}, {}});
    row_data.completed_run = run;
    show_scores(window, row);
    select_rows(window, row, row);

    const CvvdpPreset& hdr = builtin_cvvdp_presets()[2];
    choose_cvvdp_preset(window, hdr.name);

    EXPECT_TRUE(row_data.cvvdp.same_as(hdr.settings));
    EXPECT_EQ(row_data.cvvdp_preset, hdr.name);
    ASSERT_NE(row_data.completed_run, nullptr);
    EXPECT_TRUE(row_data.completed_run->result.has_metric("vmaf"));
    EXPECT_FALSE(row_data.completed_run->result.has_metric("cvvdp"));
    EXPECT_FALSE(table(window).item(row, col_vmaf)->text().isEmpty());
}

TEST_F(WindowTest, SeveralVideosOnDifferentDisplaysShowAsMixedAndAPresetSetsThemAll) {
    // Shown with the first video's preset, choosing it would change nothing
    // (the dropdown already on it), and the second video would keep its own.
    MainWindow window;
    const int a = add_row(window, "a.mp4");
    const int b = add_row(window, "b.mp4");
    RowData& first = *rows(window)[static_cast<std::size_t>(a)];
    RowData& second = *rows(window)[static_cast<std::size_t>(b)];
    second.cvvdp.display = builtin_cvvdp_presets()[1].settings.display;
    first.cvvdp.resize_to_display = true;
    select_rows(window, a, b);
    OptionsPanel& panel = options_panel(window);
    EXPECT_EQ(panel.cvvdp_preset_combo->currentText().toStdString(),
              "Mixed (the selected videos use different displays)");
    EXPECT_EQ(panel.cvvdp_resize_check->checkState(), Qt::PartiallyChecked);

    choose_cvvdp_preset(window, builtin_cvvdp_presets()[0].name);
    EXPECT_TRUE(first.cvvdp.display.identity() == builtin_cvvdp_presets()[0].settings.display.identity());
    EXPECT_TRUE(second.cvvdp.display.identity() == builtin_cvvdp_presets()[0].settings.display.identity());
    EXPECT_TRUE(first.cvvdp.resize_to_display);  // each video's own
    EXPECT_FALSE(second.cvvdp.resize_to_display);

    panel.cvvdp_resize_check->setCheckState(Qt::Checked);
    EXPECT_TRUE(first.cvvdp.resize_to_display);
    EXPECT_TRUE(second.cvvdp.resize_to_display);
}

TEST_F(WindowTest, RenamingAPresetKeepsTheOtherVideosOnItWithoutAsking) {
    // Its values unchanged, their scores still hold: there is nothing to ask
    // (a question would be a dialog, and fail the test).
    MainWindow window;
    const int a = add_row(window, "a.mp4");
    const int b = add_row(window, "b.mp4");
    select_rows(window, a, a);
    answer_cvvdp_dialog(window, [](CvvdpDisplayDialog& dialog) {
        dialog.name_edit->setText("Desk");
        dialog.peak_spin->setValue(300);
        dialog.finish("save_new");
        return dialog.result();
    });
    options_panel(window).cvvdp_add_button->click();
    select_rows(window, a, b);
    choose_cvvdp_preset(window, "Desk");
    select_rows(window, a, a);

    answer_cvvdp_dialog(window, [](CvvdpDisplayDialog& dialog) {
        dialog.name_edit->setText("Desk monitor");
        dialog.finish("save");
        return dialog.result();
    });
    options_panel(window).cvvdp_edit_button->click();

    const RowData& second = *rows(window)[static_cast<std::size_t>(b)];
    EXPECT_EQ(second.cvvdp_preset, "Desk monitor");
    EXPECT_EQ(second.cvvdp.display.peak_luminance, 300);
    const nlohmann::json saved = saved_settings();
    EXPECT_EQ(saved["cvvdp_default_preset"], "Desk monitor");
    ASSERT_EQ(saved["cvvdp_presets"].size(), 1u);
    EXPECT_EQ(saved["cvvdp_presets"][0]["name"], "Desk monitor");
}

TEST_F(WindowTest, TheDisplayEditorKeepsTheValuesTheUserDidNotChange) {
    // Exactly, though its boxes show them rounded: "Apply without saving"
    // with nothing changed must leave the display, and its CVVDP score, as
    // they are.
    CvvdpDisplay display;
    display.viewing_distance_m = 0.74725;
    display.peak_luminance = 350.5;
    display.ambient_lux = 12.25;
    display.reflectivity = 0.0123456789;
    CvvdpDisplayDialog dialog(display, nullptr, std::nullopt, false, {});
    const CvvdpDisplay shown = dialog.display();
    EXPECT_EQ(shown.viewing_distance_m, 0.74725);
    EXPECT_EQ(shown.peak_luminance, 350.5);
    EXPECT_EQ(shown.ambient_lux, 12.25);
    EXPECT_EQ(shown.reflectivity, 0.0123456789);
    dialog.peak_spin->setValue(400);
    EXPECT_EQ(dialog.display().peak_luminance, 400);
    EXPECT_EQ(dialog.display().ambient_lux, 12.25);
}

TEST_F(WindowTest, TestBothAddsARowScoringTheSameVideoTheOtherWayRound) {
    // The test video of a size of its own (b.mkv): its companion row decodes
    // the same file, with the test video scaled up to the reference rather
    // than the reference down to it, and the row it came from is left as it
    // was.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    open_clips(window);
    select_rows(window, 1, 1);
    options_panel(window).scale_direction_combo->setCurrentIndex(2);  // Test both

    ASSERT_EQ(rows(window).size(), 3u);
    EXPECT_EQ(names(window)[2], "b [upscale-distorted-to-source].mkv");
    EXPECT_EQ(rows(window)[1]->options.scale_direction, ScaleDirection::source_to_distorted);
    EXPECT_EQ(rows(window)[2]->options.scale_direction, ScaleDirection::distorted_to_source);
    EXPECT_EQ(options_panel(window).scale_direction_combo->currentIndex(), 0);  // the row's own direction again

    table(window).item(0, col_check)->setCheckState(Qt::Unchecked);
    run_button(window).click();
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    EXPECT_EQ(state(window, 1), RowState::complete);
    EXPECT_EQ(state(window, 2), RowState::complete);
    EXPECT_NE(table(window).item(1, col_psnr)->text(), table(window).item(2, col_psnr)->text());
}

TEST_F(WindowTest, AResolutionTestScoresTheReferenceScaledDownAndBackUp) {
    // A resolution test has no test video of its own: the run scales the
    // reference down and back up, and scores that against the reference. Its
    // scores are saved under the reference, where the Python app files them,
    // and come back when the same test is added again.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    choose_source(window, clip("sources/wide.mkv"));
    ASSERT_TRUE(wait_until([&] { return source_info(window).has_value() && !reading(window); }));
    const ResampleTarget target{854, "480p"};
    add_resample_test(window, target);
    const std::string described = "Downscale to 854x480, upscale back to 1280x720";
    ASSERT_EQ(names(window), std::vector<std::string>{"wide [downscale-480p-upscale].mkv"});
    EXPECT_EQ(table(window).item(0, col_info)->text().toStdString(), described);
    EXPECT_EQ(rows(window)[0]->identity_path(), clip("sources/wide.mkv"));

    run_button(window).click();
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    ASSERT_EQ(state(window, 0), RowState::complete) << rows(window)[0]->status_detail.toStdString();
    // Not the reference against itself, which libvmaf scores 60 dB.
    const QString psnr = table(window).item(0, col_psnr)->text();
    bool number = false;
    EXPECT_LT(psnr.toDouble(&number), 50) << psnr.toStdString();
    EXPECT_TRUE(number) << psnr.toStdString();

    ASSERT_TRUE(wait_for_writes(window));
    select_rows(window, 0, 0);
    remove_selected(window);
    add_resample_test(window, target);
    ASSERT_TRUE(wait_until([&] { return !reading(window); }));
    EXPECT_EQ(state(window, 0), RowState::cached);
    EXPECT_EQ(table(window).item(0, col_psnr)->text(), psnr);
    EXPECT_EQ(table(window).item(0, col_info)->text().toStdString(), described);
}

TEST_F(WindowTest, ANewReferenceRemovesTheResolutionTestsMadeFromTheOldOne) {
    // A resolution test is the old reference scaled down and back up: left
    // in the list, it would describe that video while a run scaled the new
    // one. The test videos stay, and the user is told which tests went.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    const auto information = keep_information(window);
    choose_source(window, clip("sources/wide.mkv"));
    ASSERT_TRUE(wait_until([&] { return source_info(window).has_value() && !reading(window); }));
    add_resample_test(window, {1280, "720p"});
    drop(table(window).viewport(), {clip("encodes/a.mkv")});
    add_resample_test(window, {854, "480p"});
    ASSERT_EQ(names(window).size(), 3u);

    choose_source(window, clip("sources/ref.mkv"));
    ASSERT_TRUE(wait_until([&] { return source_info(window)->path == clip("sources/ref.mkv") && !reading(window); }));
    EXPECT_EQ(names(window), std::vector<std::string>{"a.mkv"});
    ASSERT_EQ(information->size(), 1u);
    EXPECT_EQ(information->front().first.toStdString(), "Resolution tests removed");
    const QString& text = information->front().second;
    EXPECT_TRUE(text.startsWith("2 resolution tests belonged to the previous source")) << text.toStdString();
    EXPECT_TRUE(text.contains("\n  wide [downscale-720p-upscale].mkv")) << text.toStdString();
    EXPECT_TRUE(text.contains("\n  wide [downscale-480p-upscale].mkv")) << text.toStdString();
}

// --------------------------------------------------------------- saved runs

TEST_F(WindowTest, ASavedRunLoadsIntoAnotherWindowAsAnUntickedRowWithItsScores) {
    // A row's result saved to a file, then loaded where the window knows
    // nothing of it: the run's reference becomes the window's, and the row
    // shows the same scores, its metrics' columns ticked from what the file
    // holds, unticked itself so that a run leaves it alone.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    const std::filesystem::path saved = folder("b.metrics.json");
    std::string psnr;
    std::string xpsnr;
    {
        MainWindow window;
        open_clips(window);
        table(window).item(0, col_check)->setCheckState(Qt::Unchecked);
        run_button(window).click();
        ASSERT_TRUE(wait_until([&] { return !running(window); }));
        ASSERT_EQ(state(window, 1), RowState::complete);
        psnr = table(window).item(1, col_psnr)->text().toStdString();
        xpsnr = table(window).item(1, col_xpsnr)->text().toStdString();
        select_rows(window, 1, 1);
        choose_file(window, saved);
        save_button(window).click();
        ASSERT_TRUE(wait_for_writes(window));
    }
    write_settings({{"default_compute_xpsnr", false}});  // not ticked for new rows here
    MainWindow window;
    choose_file(window, saved);
    load_analysis_results(window);
    ASSERT_EQ(names(window), std::vector<std::string>{"b.mkv"});
    EXPECT_EQ(source_info(window)->path, clip("sources/ref.mkv"));
    EXPECT_EQ(table(window).item(0, col_check)->checkState(), Qt::Unchecked);
    EXPECT_FALSE(psnr.empty());
    EXPECT_EQ(table(window).item(0, col_psnr)->text().toStdString(), psnr);
    EXPECT_EQ(table(window).item(0, col_xpsnr)->text().toStdString(), xpsnr);
    const std::vector<std::string> ticked = selected(window, 0);
    EXPECT_NE(std::find(ticked.begin(), ticked.end(), "xpsnr"), ticked.end());
    EXPECT_EQ(rows(window)[0]->status_detail.toStdString(), "Loaded from saved run");
}

TEST_F(WindowTest, ASavedRunsOtherReferenceTakenInRemovesTheResolutionTestsOfTheOldOne) {
    // Loading a run measured against another reference, the user agrees to
    // switch to it. That is a change of reference like any other: the
    // resolution tests made from the old one go, or each would describe one
    // video while its run scaled the other. (The Python app's own switch
    // left them, until release/v2.1 92b357b.)
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    const std::filesystem::path saved = folder("a.metrics.json");
    {
        MainWindow window;
        open_clips(window);
        table(window).item(1, col_check)->setCheckState(Qt::Unchecked);
        run_button(window).click();
        ASSERT_TRUE(wait_until([&] { return !running(window); }));
        select_rows(window, 0, 0);
        choose_file(window, saved);
        save_button(window).click();
        ASSERT_TRUE(wait_for_writes(window));
    }
    MainWindow window;
    const auto information = keep_information(window);
    const auto asked = answer_questions(window, true);
    choose_source(window, clip("sources/wide.mkv"));
    ASSERT_TRUE(wait_until([&] { return source_info(window).has_value() && !reading(window); }));
    add_resample_test(window, {854, "480p"});
    choose_file(window, saved);
    load_analysis_results(window);
    EXPECT_EQ(*asked, std::vector<QString>{"Different reference video"});
    EXPECT_EQ(source_info(window)->path, clip("sources/ref.mkv"));
    EXPECT_EQ(names(window), std::vector<std::string>{"a.mkv"});
    ASSERT_EQ(information->size(), 1u);
    EXPECT_EQ(information->front().first.toStdString(), "Resolution tests removed");
}

TEST_F(WindowTest, AResolutionTestLoadedFromASavedRunDescribesTheTest) {
    // As one added does: what it scales the reference to, with no bitrate of
    // its own, not the reference's details. (The Python app's own load drew
    // those, until release/v2.1 9423de8.)
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    choose_source(window, clip("sources/wide.mkv"));
    ASSERT_TRUE(wait_until([&] { return source_info(window).has_value() && !reading(window); }));
    const ResampleTarget target{854, "480p"};
    ComparisonResult result;
    result.source = clip("sources/wide.mkv");
    result.distorted = synthetic_resample_distorted_path(result.source, target);
    result.fps = source_info(window)->fps;
    result.source_info = *source_info(window);
    result.distorted_info = *source_info(window);
    result.resample_target = target;
    result.model_choice = std::string(auto_model_choice);
    result.metric_results.add(FrameMetricResult{"vmaf", {0, 1}, {0.0, 0.04}, {95.0F, 96.0F},
                                                current_ffmpeg_provenance("vmaf", "9.0.1")});
    const std::filesystem::path saved = folder("test.metrics.json");
    save_run(result, saved, "test");

    choose_file(window, saved);
    load_analysis_results(window);
    ASSERT_EQ(table(window).rowCount(), 1);
    EXPECT_EQ(table(window).item(0, col_info)->text().toStdString(), "Downscale to 854x480, upscale back to 1280x720");
    EXPECT_EQ(table(window).item(0, col_bitrate)->text().toStdString(), "N/A");
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

TEST_F(WindowTest, ANewReferenceTakesAwayHowARunAgainstTheOldOneEnded) {
    // A video's run against the old reference was cancelled. Under another
    // reference that says nothing about it: the row is not calculated, as in
    // the Python app, rather than still Cancelled.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    open_clips(window);
    run_button(window).click();
    pause_button(window).click();
    ASSERT_TRUE(wait_until([&] { return state(window, 0) == RowState::calculating; }));
    cancel_button(window).click();
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    ASSERT_EQ(state(window, 0), RowState::cancelled);

    choose_source(window, clip("encodes/b.mkv"));
    ASSERT_TRUE(wait_until([&] { return source_info(window)->path == clip("encodes/b.mkv") && !reading(window); }));
    EXPECT_EQ(state(window, 0), RowState::not_calculated);
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

// ------------------------------------------------------------- the graph

TEST_F(WindowTest, AFinishedRunIsOnTheGraphAndARemovedVideoTakesItsLineAway) {
    // Metric Graphs shows exactly the videos with a result: a run's reach it
    // without a button pressed, its tab brought to the front, listed in the
    // table's order (not the order they finished), and a video removed takes
    // its line with it. (The Python app once shipped both wrong: a graph
    // empty until Show graph was pressed, and lines left.)
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    open_clips(window);
    run_button(window).click();
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    ASSERT_EQ(state(window, 0), RowState::complete);
    ASSERT_EQ(state(window, 1), RowState::complete);
    EXPECT_EQ(tabs(window).currentIndex(), tab_graph);
    EXPECT_EQ(graphed(window), (std::vector<std::string>{"a.mkv", "b.mkv"}));
    drag_rows(window, {1}, 0);
    EXPECT_EQ(graphed(window), (std::vector<std::string>{"b.mkv", "a.mkv"}));

    select_rows(window, 1, 1);  // a.mkv, now second
    remove_selected(window);
    EXPECT_EQ(graphed(window), std::vector<std::string>{"b.mkv"});
}

// --------------------------------------------------------- Video Compare

TEST_F(WindowTest, VideoCompareListsTheVideosReadScoredOrNotAndFollowsTheTable) {
    // Frames can be compared before anything is measured: the videos read
    // are listed when the tab is opened, a run's scores join them, a video
    // removed leaves, and scores taken away leave the video listed without
    // them.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    using Listed = std::vector<std::pair<std::string, bool>>;
    MainWindow window;
    open_clips(window);
    tabs(window).setCurrentIndex(tab_frame_compare);
    EXPECT_EQ(compared(window), (Listed{{"a", false}, {"b", false}}));

    tabs(window).setCurrentIndex(tab_videos);
    run_button(window).click();
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    EXPECT_EQ(compared(window), (Listed{{"a", true}, {"b", true}}));
    tabs(window).setCurrentIndex(tab_frame_compare);
    EXPECT_EQ(compared(window), (Listed{{"a", true}, {"b", true}}));

    tabs(window).setCurrentIndex(tab_videos);
    select_rows(window, 0, 0);
    remove_selected(window);
    EXPECT_EQ(compared(window), (Listed{{"b", true}}));

    select_rows(window, 0, 0);
    options_panel(window).subsample_spin->setValue(5);
    ASSERT_EQ(rows(window)[0]->completed_run, nullptr);
    EXPECT_EQ(compared(window), (Listed{{"b", false}}));
}

// --------------------------------------------------------- Bitrate Viewer

TEST_F(WindowTest, AFinishedRunsVideosAreReadInTheBitrateViewer) {
    // Each video a run has compared is read packet by packet there, the
    // reference once for both test videos, without a button pressed.
    if (!have_clips()) {
        GTEST_SKIP() << "FFmpeg is not here";
    }
    MainWindow window;
    open_clips(window);
    run_button(window).click();
    ASSERT_TRUE(wait_until([&] { return !running(window); }));
    const QTableWidget& listed = bitrate_table(window);
    ASSERT_TRUE(wait_until([&] {
        for (int row = 0; row < listed.rowCount(); ++row) {
            if (listed.item(row, 8)->text() != QStringLiteral("Complete")) {
                return false;
            }
        }
        return listed.rowCount() > 0;
    })) << "the videos were not all read";
    const std::vector<std::string> names{"ref.mkv", "a.mkv", "b.mkv"};
    ASSERT_EQ(listed.rowCount(), static_cast<int>(names.size()));
    const std::optional<test::Tools> found = test::tools();
    for (int row = 0; row < listed.rowCount(); ++row) {
        const std::string& name = names[static_cast<std::size_t>(row)];
        EXPECT_EQ(listed.item(row, 1)->text().toStdString(), name);
        const std::filesystem::path path = clip((row == 0 ? "sources/" : "encodes/") + name);
        EXPECT_EQ(listed.item(row, 4)->text().toStdString(),
                  std::to_string(probe_video(path, found->ffprobe).estimated_frame_count()))
            << name;
    }
}

// ------------------------------------------------------------ the window

TEST_F(WindowTest, MissingToolsShowABannerWithAButtonToFindThem) {
    std::filesystem::create_directories(folder("no ffmpeg"));
    write_settings({{"ffmpeg_dir", platform::path_text(folder("no ffmpeg"))}});
    const test::ScopedVariable path("PATH", "");
    // Nor in Homebrew's folder (macOS).
    const test::ScopedVariable folders(std::string(ffmpeg_folders_variable), "");
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

// ------------------------------------------------------------ the Settings tab

// Clear saved results deletes the saved scores and black bars of the folder
// the question named, even when the setting changes before the queue of
// writes gets to it, and nothing else in that folder (the Python app's
// test_clearing_the_cache_deletes_the_folder_the_dialog_named and
// test_clear_cache_never_deletes_unrelated_json_files).
TEST_F(WindowTest, ClearingSavedResultsDeletesOnlyWhatTheQuestionNamed) {
    const std::filesystem::path named = folder("named");
    const std::filesystem::path other = folder("other");
    for (const std::filesystem::path& base : {named, other}) {
        std::filesystem::create_directories(base / "v2" / "comparison");
        std::ofstream(base / "v2" / "comparison" / "context.json", std::ios::binary) << "{}";
        std::ofstream(base / "black_bars.json", std::ios::binary) << "{}";
    }
    std::ofstream(named / "family_budget.json", std::ios::binary) << "important";
    write_settings({{"cache_dir", platform::path_text(named)}});
    MainWindow window;
    const auto asked = answer_questions(window, true);
    SettingsPanel& panel = settings_panel(window);
    std::promise<void> release;
    hold_writes(window, release.get_future().share());

    panel.clear_saved_results();
    ASSERT_EQ(*asked, std::vector<QString>{"Clear saved results"});
    EXPECT_EQ(panel.status->text().toStdString(), "Clearing saved results...");
    panel.cache_edit->setText(QString::fromStdString(platform::path_text(other)));  // edited meanwhile
    emit panel.cache_edit->editingFinished();
    release.set_value();
    ASSERT_TRUE(wait_for_writes(window));
    EXPECT_TRUE(wait_until([&] { return panel.status->text() == "Removed 1 saved result."; }))
        << panel.status->text().toStdString();

    EXPECT_FALSE(std::filesystem::exists(named / "v2"));
    EXPECT_FALSE(std::filesystem::exists(named / "black_bars.json"));
    EXPECT_EQ(platform::read_file(named / "family_budget.json"), "important");
    EXPECT_TRUE(std::filesystem::exists(other / "v2" / "comparison" / "context.json"));
    EXPECT_TRUE(std::filesystem::exists(other / "black_bars.json"));
}

// Each edit in the tab is saved at once, to the setting it shows and no
// other; a video added afterwards starts with the tab's defaults, while one
// already there keeps its own.
TEST_F(WindowTest, EachSettingIsSavedAsItIsEditedAndNewVideosTakeTheDefaults) {
    using Json = nlohmann::json;
    write_settings({{"remember_videos", true}, {"remembered_reference", "ref.mkv"}, {"remembered_tests", {"a.mkv"}}});
    MainWindow window;
    SettingsPanel& panel = settings_panel(window);
    add_row(window, "before.mkv");
    // The settings an edit changed in settings.json, with their new values.
    const auto changed_by = [&](const std::function<void()>& edit) {
        const Json before = settings_now(window);
        edit();
        const Json after = saved_settings();
        Json changed = Json::object();
        for (const auto& [key, value] : after.items()) {
            if (!before.contains(key) || before[key] != value) {
                changed[key] = value;
            }
        }
        return changed;
    };
    const auto select = [](QComboBox* box, const QVariant& data) {
        const int index = box->findData(data);
        ASSERT_GE(index, 0) << data.toString().toStdString();
        box->setCurrentIndex(index);
    };
    const auto type = [](QLineEdit* edit, const QString& text) {
        edit->setText(text);
        emit edit->editingFinished();
    };

    EXPECT_EQ(changed_by([&] { type(panel.ffmpeg_edit, "C:/no ffmpeg here"); }),
              (Json{{"ffmpeg_dir", "C:/no ffmpeg here"}}));
    const std::string saved = platform::path_text(folder("saved"));
    const std::string exports = platform::path_text(folder("exports"));
    EXPECT_EQ(changed_by([&] { type(panel.cache_edit, QString::fromStdString(saved)); }),
              (Json{{"cache_dir", saved}}));
    EXPECT_EQ(changed_by([&] { type(panel.export_edit, QString::fromStdString(exports)); }),
              (Json{{"export_dir", exports}}));
    EXPECT_EQ(changed_by([&] { panel.use_cache->setChecked(false); }), (Json{{"use_cache", false}}));
    EXPECT_EQ(changed_by([&] { panel.default_gpu->setChecked(false); }), (Json{{"default_gpu_decode", false}}));
    for (const auto& [key, box] : panel.default_metrics) {
        const bool shown = box->isChecked();
        EXPECT_EQ(changed_by([&box = box, shown] { box->setChecked(!shown); }),
                  (Json{{"default_compute_" + key, !shown}}));
        EXPECT_EQ(changed_by([&box = box, shown] { box->setChecked(shown); }),
                  (Json{{"default_compute_" + key, shown}}));
    }
    EXPECT_EQ(changed_by([&] { select(panel.cvvdp_default, "30-inch 4K HDR monitor, dim room"); }),
              (Json{{"cvvdp_default_preset", "30-inch 4K HDR monitor, dim room"}}));
    EXPECT_EQ(changed_by([&] { panel.gpu_together->setChecked(true); }), (Json{{"gpu_metrics_together", true}}));
    EXPECT_EQ(changed_by([&] { select(panel.decoded_videos, 5); }), (Json{{"compare_decoded_videos", 5}}));
    EXPECT_EQ(changed_by([&] { select(panel.theme, "dark"); }), (Json{{"theme", "dark"}}));
    EXPECT_EQ(changed_by([&] { panel.remember_size->setChecked(false); }), (Json{{"remember_window_size", false}}));
    // The paths go when they are no longer asked for.
    EXPECT_EQ(changed_by([&] { panel.remember_videos->setChecked(false); }),
              (Json{{"remember_videos", false}, {"remembered_reference", ""}, {"remembered_tests", Json::array()}}));
    EXPECT_EQ(changed_by([&] { panel.check_updates->setChecked(false); }), (Json{{"check_for_updates", false}}));
    EXPECT_EQ(panel.status->text().toStdString(), "Settings saved.");
    EXPECT_EQ(changed_by([&] { select(panel.language, "de"); }), (Json{{"language", "de"}}));
    EXPECT_EQ(panel.status->text().toStdString(),
              "Settings saved. The new language shows when the app is next started.");

    panel.default_metrics.at("psnr")->setChecked(false);
    add_row(window, "after.mkv");
    const std::vector<std::string> before = requested(window, 0);
    const std::vector<std::string> after = requested(window, 1);
    EXPECT_TRUE(contains(before, "psnr"));
    EXPECT_FALSE(contains(after, "psnr"));
    EXPECT_TRUE(contains(after, "ssim"));
    EXPECT_TRUE(rows(window)[0]->options.gpu_decode);
    EXPECT_FALSE(rows(window)[1]->options.gpu_decode);
}

// Copy log puts the latest session that ran something on the clipboard, and
// Export log saves every log file in one zip; both say what they did.
TEST_F(WindowTest, CopyLogAndExportLogShareTheLog) {
    MainWindow window;
    SettingsPanel& panel = settings_panel(window);
    panel.copy_log_button->click();
    EXPECT_EQ(panel.status->text().toStdString(), "There is no log to copy yet.");

    const std::filesystem::path logs = app_log::log_dir();
    std::filesystem::create_directories(logs);
    std::ofstream(logs / "VideoMetricsLab.log", std::ios::binary)
        << "2026-10-14 09:00:00.000 INFO    [MainThread] vml.main: ==== VideoMetricsLab starting ====\r\n"
           "2026-10-14 09:00:01.000 INFO    [Thread-1] vml.job_runner: Run started: 1 video(s)\r\n"
           "2026-10-14 09:00:02.000 INFO    [Thread-1] vml.job_runner: Run ended\r\n";
    panel.copy_log_button->click();
    EXPECT_EQ(QApplication::clipboard()->text().toStdString(), app_log::log_text_to_share()->first);
    EXPECT_EQ(panel.status->text().toStdString(), "Copied the log to the clipboard (3 lines).");

    const std::filesystem::path zip = folder("log.zip");
    choose_file(window, zip);
    panel.export_log_button->click();
    EXPECT_TRUE(std::filesystem::exists(zip));
    EXPECT_EQ(panel.status->text().toStdString(), "Log exported to " + platform::path_text(zip));
}

// The update check: a newer release than this one is offered, with its
// notes; Skip this version keeps it from being offered again, and Later only
// closes the message. Nothing is offered for a release no newer than this
// one, or when GitHub could not be asked.
TEST_F(WindowTest, ANewerReleaseIsOfferedUntilTheUserSkipsIt) {
    MainWindow window;
    const auto click = [&window](const QString& text) {
        QMessageBox* box = update_box(window);
        ASSERT_NE(box, nullptr);
        for (QAbstractButton* button : box->buttons()) {
            if (button->text() == text) {
                button->click();
                ASSERT_TRUE(wait_until([&window] { return update_box(window) == nullptr; }));
                return;
            }
        }
        FAIL() << "no button " << text.toStdString();
    };
    answer_update(window, R"({"tag_name": "v1.0", "html_url": "https://example.invalid/v1.0"})", {});
    EXPECT_EQ(update_box(window), nullptr);
    answer_update(window, "", "Host api.github.com not found");
    EXPECT_EQ(update_box(window), nullptr);

    const std::string newer = R"({"tag_name": "v99.0", "html_url": "https://example.invalid/v99.0", "body": "- Faster."})";
    answer_update(window, newer, {});
    ASSERT_NE(update_box(window), nullptr);
    EXPECT_EQ(update_box(window)->text().toStdString(), "VideoMetricsLab 99.0 is available. You have " VML_APP_VERSION ".");
    EXPECT_EQ(update_box(window)->detailedText().toStdString(), "- Faster.");
    click("Skip this version");
    EXPECT_EQ(saved_settings()["skipped_update_version"], "99.0");
    answer_update(window, newer, {});
    EXPECT_EQ(update_box(window), nullptr);

    answer_update(window, R"({"tag_name": "v100.0"})", {});
    ASSERT_NE(update_box(window), nullptr);
    click("Later");
    EXPECT_EQ(saved_settings()["skipped_update_version"], "99.0");
}

}  // namespace
}  // namespace vml::app
