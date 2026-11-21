// Metric Graphs, held to the Python app's own panel (reference/graph.json):
// the table and the readouts for runs of every kind of score (but for the
// readouts' Δ lines, cut from graph.json where the Python app paired
// different frames or three lines were shown); and what the
// graph must keep: XPSNR's infinity in the scores while it is drawn at a
// stand-in, CVVDP's seconds at their middles, one line per run, a removed
// line kept away, an image to export.
#include "graph_panel.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMouseEvent>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "chart.h"
#include "frame_controls.h"
#include "reference.h"
#include "series_colors.h"
#include "test_videos.h"
#include "vml/platform/system.h"
#include "window_helpers.h"
#include "vml/segment_profiles.h"
#include "worst_frames_panel.h"

namespace vml::app {

// The graph's parts, for its tests: the class GraphPanel, MetricPage and
// ChartWidget let in.
class GraphPanelTest : public ::testing::Test {
protected:
    static QTableWidget& table(GraphPanel& panel) { return *panel.stats_table_; }
    static QTabWidget& tabs(GraphPanel& panel) { return *panel.tabs_; }
    static QSpinBox& frame_spin(GraphPanel& panel) { return *panel.frame_controls_->frame_spin; }
    static QCheckBox& difference_box(GraphPanel& panel) { return *panel.difference_box_; }
    static QSpinBox& range_start_spin(GraphPanel& panel) { return *panel.frame_controls_->range_start_spin; }
    static QSpinBox& range_end_spin(GraphPanel& panel) { return *panel.frame_controls_->range_end_spin; }
    static QPushButton& clear_range_button(GraphPanel& panel) { return *panel.frame_controls_->clear_range_button; }
    static FrameControls& frame_controls(GraphPanel& panel) { return *panel.frame_controls_; }
    // A time typed into a time box, then Enter.
    static void type_time(QLineEdit& box, const QString& text) {
        box.setText(text);
        box.setModified(true);
        emit box.returnPressed();
        emit box.editingFinished();
    }
    static QString stats_hint(const GraphPanel& panel) { return panel.stats_hint_->text(); }
    // The statistics' right-click menu for the line labelled `label`: its
    // items' texts, in order, and those it offers disabled.
    static QStringList stats_menu(GraphPanel& panel, const QString& label, QStringList* disabled = nullptr) {
        int row = -1;
        for (int each = 0; each < panel.stats_table_->rowCount(); ++each) {
            if (panel.stats_table_->item(each, 0) != nullptr && panel.stats_table_->item(each, 0)->text() == label) {
                row = each;
            }
        }
        QMenu menu;
        GraphPanel::StatsMenu items;
        panel.fill_stats_menu(menu, row, items);
        QStringList texts;
        for (const QAction* action : menu.actions()) {
            if (!action->isSeparator()) {
                texts << action->text();
                if (disabled != nullptr && !action->isEnabled()) {
                    *disabled << action->text();
                }
            }
        }
        return texts;
    }
    static QPushButton& worst_button(GraphPanel& panel) { return *panel.worst_button_; }
    static WorstFramesPanel& worst_panel(GraphPanel& panel) { return *panel.worst_panel_; }
    // Every cell's text, row by row.
    static std::vector<QStringList> cell_texts(GraphPanel& panel) {
        std::vector<QStringList> rows;
        const QTableWidget& stats = *panel.stats_table_;
        for (int row = 0; row < stats.rowCount(); ++row) {
            QStringList cells;
            for (int column = 0; column < stats.columnCount(); ++column) {
                cells.append(stats.item(row, column) != nullptr ? stats.item(row, column)->text() : QString());
            }
            rows.push_back(cells);
        }
        return rows;
    }
    // The difference a page's chart draws under its lines, if any.
    static std::optional<std::vector<float>> band_values(MetricPage& page) {
        const std::optional<ChartBand>& band = page.chart_->band_;
        if (!band) {
            return std::nullopt;
        }
        return std::vector<float>(band->series.values.begin(), band->series.values.end());
    }
    // Ticks or unticks a line's box in the table, as a click does.
    static void tick_row(GraphPanel& panel, int series_id, bool checked) {
        QTableWidget& stats = *panel.stats_table_;
        for (int row = 0; row < stats.rowCount(); ++row) {
            QTableWidgetItem* item = stats.item(row, 0);
            if (item != nullptr && item->data(Qt::UserRole).toInt() == series_id) {
                item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
                return;
            }
        }
        ADD_FAILURE() << "no row for line " << series_id;
    }
    static const std::map<int, SeriesEntry>& entries(const GraphPanel& panel) { return panel.entries_; }
    // A metric's page, shown (and so built).
    static MetricPage& page(GraphPanel& panel, const std::string& key) {
        const std::span<const MetricDefinition> all = metrics();
        const auto at =
            std::find_if(all.begin(), all.end(), [&key](const MetricDefinition& m) { return m.key == key; });
        panel.tabs_->setCurrentIndex(static_cast<int>(at - all.begin()));
        return *panel.pages_.at(key);
    }
    static QString readout(const MetricPage& page) { return page.hover_label_->text(); }
    static const QFont& readout_font(const MetricPage& page) { return page.readout_font_; }
    static int readout_height(const MetricPage& page) { return page.hover_label_->maximumHeight(); }
    // The crosshair's pixel column on a page's chart, if shown.
    static std::optional<int> cursor_x(MetricPage& page) { return page.chart_->cursor_pixel(); }
    // A page's chart zoomed to `span` seconds from `start`, and the seconds in view.
    static void zoom(MetricPage& page, double start, double span) { page.chart_->view_ = ChartWidget::View{start, span}; }
    static std::pair<double, double> in_view(MetricPage& page) { return page.chart_->x_range(); }
    // What a page's chart draws for a line.
    static const ChartSeries& drawn(MetricPage& page, int series_id) { return page.chart_->series_.at(series_id); }
    static void export_png(GraphPanel& panel) { panel.on_export_png(); }
    static void export_difference(GraphPanel& panel) { panel.on_export_difference(); }
    static QString status(const GraphPanel& panel) { return panel.status_label_->text(); }
    // What the exported image's table says: its headings, and each line's
    // label, colour and cells.
    static auto export_table(GraphPanel& panel) { return panel.export_table(); }

    // The run whose label is `label`.
    static int id_of(const GraphPanel& panel, const QString& label) {
        for (const auto& [id, entry] : panel.entries_) {
            if (entry.label == label) {
                return id;
            }
        }
        ADD_FAILURE() << "no line " << label.toStdString();
        return -1;
    }

    static std::shared_ptr<const ComparisonResult> shared(ComparisonResult result) {
        return std::make_shared<const ComparisonResult>(std::move(result));
    }

    // A result with VMAF's `values`, a frame each at 30 a second.
    static ComparisonResult vmaf_result(const std::string& name, const std::vector<float>& values) {
        ComparisonResult result;
        result.source = "source.mkv";
        result.distorted = name;
        result.fps = 30.0;
        FrameMetricResult vmaf{"vmaf", {}, {}, values, current_ffmpeg_provenance("vmaf", "9.0.1")};
        for (std::size_t i = 0; i < values.size(); ++i) {
            vmaf.frame.push_back(static_cast<int>(i));
            vmaf.time.push_back(static_cast<double>(i) / 30.0);
        }
        result.metric_results.add(std::move(vmaf));
        return result;
    }

    // `metric`'s `values` at `frames` (`fps` a second) added to `result`.
    static void add_frames(ComparisonResult& result, const std::string& metric, const std::vector<int>& frames,
                           const std::vector<float>& values, double fps = 30.0) {
        FrameMetricResult scores{metric, frames, {}, values, current_ffmpeg_provenance(metric, "9.0.1")};
        for (const int frame : frames) {
            scores.time.push_back(frame / fps);
        }
        result.metric_results.add(std::move(scores));
    }
};

namespace {

TEST_F(GraphPanelTest, ShowsTheRunsAsThePythonApp) {
    const nlohmann::json reference = test::load_reference("graph.json");
    ASSERT_FALSE(reference.at("pages").empty());
    VideoColors colors;
    GraphPanel panel(&colors);
    for (const nlohmann::json& run : reference.at("runs")) {
        panel.add_run({shared(test::result_from(run.at("result"))), QString::fromStdString(run.at("label")), {}, true});
    }
    for (const nlohmann::json& item : reference.at("pages")) {
        const std::string key = item.at("key");
        SCOPED_TRACE(key);
        MetricPage& shown = page(panel, key);
        shown.chart().resize(800, 400);
        const nlohmann::json& expected = item.at("table");
        QTableWidget& stats = table(panel);
        std::vector<std::string> headers;
        for (int column = 0; column < stats.columnCount(); ++column) {
            headers.push_back(stats.horizontalHeaderItem(column)->text().toStdString());
        }
        EXPECT_EQ(headers, expected.at("headers").get<std::vector<std::string>>());
        ASSERT_EQ(stats.rowCount(), static_cast<int>(expected.at("cells").size()));
        for (int row = 0; row < stats.rowCount(); ++row) {
            for (int column = 0; column < stats.columnCount(); ++column) {
                const nlohmann::json& cell =
                    expected.at("cells")[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)];
                const QTableWidgetItem* actual = stats.item(row, column);
                EXPECT_EQ(actual == nullptr ? nlohmann::json() : nlohmann::json(actual->text().toStdString()), cell)
                    << "row " << row << ", column " << column;
            }
        }
        EXPECT_EQ((std::vector<int>{frame_spin(panel).minimum(), frame_spin(panel).maximum()}),
                  item.at("frame_range").get<std::vector<int>>());
        for (const nlohmann::json& case_ : item.at("readouts")) {
            const std::vector<std::string> hidden = case_.at("hidden");
            panel.set_series_visible(id_of(panel, "b"), hidden.empty());
            if (case_.contains("frame")) {
                SCOPED_TRACE("frame " + std::to_string(case_.at("frame").get<int>()));
                shown.show_frame(case_.at("frame").get<int>(), entries(panel));
            } else {
                shown.on_hover(*test::reference_number(case_.at("x")), *test::reference_number(case_.at("y")),
                               entries(panel));
            }
            EXPECT_EQ(readout(shown).toStdString(), case_.at("text").get<std::string>());
        }
        panel.set_series_visible(id_of(panel, "b"), true);
    }
}

// The Difference box draws the first shown line minus the second, frame by
// frame, under them: where a page shows exactly two lines, on every page,
// those built later too; a frame missing in either is a gap, and XPSNR's
// infinity in both is no difference.
TEST_F(GraphPanelTest, DrawsTheDifferenceOfTwoShownLines) {
    constexpr float infinity = std::numeric_limits<float>::infinity();
    ComparisonResult a = vmaf_result("a.mkv", {90.0F, 80.0F, 70.0F, 60.0F});
    add_frames(a, "xpsnr", {0, 1, 2, 3}, {infinity, infinity, 40.0F, 41.0F});
    ComparisonResult b;
    b.source = "source.mkv";
    b.distorted = "b.mkv";
    b.fps = 30.0;
    add_frames(b, "vmaf", {0, 1, 3}, {85.0F, 82.0F, 50.0F});  // frame 2 not scored
    add_frames(b, "xpsnr", {0, 1, 3}, {infinity, 30.0F, 39.0F});
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(a), "a", {}, true},
                    {shared(b), "b", {}, true},
                    {shared(vmaf_result("c.mkv", {50.0F, 50.0F, 50.0F, 50.0F})), "c", {}, true}});
    MetricPage& vmaf = page(panel, "vmaf");
    QCheckBox& box = difference_box(panel);
    EXPECT_FALSE(box.isEnabled());  // three lines
    tick_row(panel, id_of(panel, "c"), false);
    ASSERT_TRUE(box.isEnabled());
    EXPECT_FALSE(band_values(vmaf));
    box.setChecked(true);
    ASSERT_TRUE(band_values(vmaf));
    const std::vector<float> vmaf_difference = *band_values(vmaf);
    ASSERT_EQ(vmaf_difference.size(), 4U);
    EXPECT_EQ(vmaf_difference[0], 5.0F);
    EXPECT_EQ(vmaf_difference[1], -2.0F);
    EXPECT_TRUE(std::isnan(vmaf_difference[2]));
    EXPECT_EQ(vmaf_difference[3], 10.0F);
    tick_row(panel, id_of(panel, "c"), true);
    EXPECT_FALSE(box.isEnabled());
    EXPECT_FALSE(band_values(vmaf));

    // XPSNR, built now, after the tick: c has none, so two lines show there.
    MetricPage& xpsnr = page(panel, "xpsnr");
    EXPECT_TRUE(box.isEnabled());
    ASSERT_TRUE(band_values(xpsnr));
    const std::vector<float> xpsnr_difference = *band_values(xpsnr);
    ASSERT_EQ(xpsnr_difference.size(), 4U);
    EXPECT_EQ(xpsnr_difference[0], 0.0F);          // identical frames in both
    EXPECT_TRUE(std::isnan(xpsnr_difference[1]));  // in one alone
    EXPECT_TRUE(std::isnan(xpsnr_difference[2]));  // not in b
    EXPECT_EQ(xpsnr_difference[3], 2.0F);
    box.setChecked(false);
    EXPECT_FALSE(band_values(xpsnr));
}

// Metric Graphs' Export CSV is gone (the right-click menus export each
// line's scores): with Difference on, the statistics' menu offers the
// difference the band shows, <first>_minus_<second>.csv after the two test
// videos rather than their lines' " (saved)" labels, of three runs shown the
// two with the page's metric; with it off, not.
TEST_F(GraphPanelTest, ExportsTheDifferenceTheBandShows) {
    ComparisonResult c;  // PSNR only: not on VMAF's page
    add_frames(c, "psnr", {0, 1, 2}, {40.0F, 40.0F, 40.0F});
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(vmaf_result("a.mkv", {90.0F, 80.0F, 70.0F})), "a (saved)", {}, true},
                    {shared(vmaf_result("b.mkv", {85.0F, 82.0F, 60.0F})), "b (saved)", {}, true},
                    {shared(c), "c", {}, true}});
    page(panel, "vmaf");
    EXPECT_TRUE(stats_menu(panel, "a (saved)").isEmpty());
    difference_box(panel).setChecked(true);
    EXPECT_EQ(stats_menu(panel, "a (saved)"), QStringList{"Export the difference as CSV..."});
    const test::Folder folder;
    QString suggested;
    panel.dialogs.save_file = [&](const QString&, const QString& name, const QString&) {
        suggested = name;
        return QString::fromStdString(platform::path_text(folder / name.toStdString()));
    };
    export_difference(panel);
    ASSERT_TRUE(panel.wait_until_file_writes_idle(std::chrono::seconds(10)));
    EXPECT_EQ(suggested, "a_minus_b.csv");
    std::ifstream file(folder / "a_minus_b.csv", std::ios::binary);
    std::vector<std::string> difference;
    for (std::string line; std::getline(file, line);) {
        difference.push_back(line);
    }
    ASSERT_EQ(difference.size(), 4U);
    EXPECT_TRUE(difference[0].starts_with("frame,time_s,vmaf,")) << difference[0];
    // frame, time_s, then VMAF's column; the other metrics' are blank
    EXPECT_TRUE(difference[1].starts_with("0,0.000000,5.0,,")) << difference[1];
    EXPECT_TRUE(difference[2].starts_with("1,0.033333,-2.0,,")) << difference[2];
    EXPECT_TRUE(difference[3].starts_with("2,0.066667,10.0,,")) << difference[3];

    // Off: nothing to export.
    difference_box(panel).setChecked(false);
    EXPECT_TRUE(stats_menu(panel, "a (saved)").isEmpty());
    suggested.clear();
    export_difference(panel);
    EXPECT_TRUE(suggested.isEmpty());
}

// Issue #19: an export's dialog starts in Settings' Export folder, read at
// each export, or in Documents where it is blank; never the working folder.
// A run file is imported under the name the Videos tab's button has, the
// counterpart of Export > Run file.
TEST_F(GraphPanelTest, RunFilesAreImported) {
    VideoColors colors;
    GraphPanel panel(&colors);
    QStringList texts;
    for (const QPushButton* button : panel.findChildren<QPushButton*>()) {
        texts << button->text();
    }
    EXPECT_TRUE(texts.contains("Import run file...")) << texts.join(", ").toStdString();
    EXPECT_FALSE(texts.contains("Add analysis results"));
    QString filter;
    panel.dialogs.open_file = [&filter](const QString&, const QString& asked) {
        filter = asked;
        return QString();
    };
    for (QPushButton* button : panel.findChildren<QPushButton*>()) {
        if (button->text() == "Import run file...") {
            button->click();
        }
    }
    EXPECT_EQ(filter.toStdString(), "Run files (*.metrics.json)");
}

TEST_F(GraphPanelTest, ExportsStartInTheExportFolder) {
    VideoColors colors;
    GraphPanel panel(&colors);
    std::string export_dir = "D:/exports";
    panel.dialogs = standard_dialogs(&panel, [&export_dir] { return export_dir; });
    EXPECT_EQ(path_of(panel.dialogs.export_folder()), path_of("D:/exports"));
    export_dir.clear();
    EXPECT_EQ(path_of(panel.dialogs.export_folder()), platform::home_directory() / "Documents");
    // A save dialog: a bare name in the folder, a full path as it is.
    EXPECT_EQ(path_of(save_dialog_start("a.csv", "D:/exports")), path_of("D:/exports/a.csv"));
    const QString full = text_of(platform::home_directory() / "elsewhere" / "log.zip");
    EXPECT_EQ(save_dialog_start(full, "D:/exports"), full);
}

// The statistics' right-click menu offers the Videos table's exports: a
// line's video's own, where the window can export it (a line of a saved run
// loaded has no row: none), and all the videos', each named after what
// Settings' Exports include; off where nothing is ticked, and all the
// videos' while there is no run to export.
TEST_F(GraphPanelTest, TheStatisticsMenuOffersTheVideosExports) {
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(vmaf_result("a.mkv", {90.0F, 80.0F})), "a", "row-a", true},
                    {shared(vmaf_result("b.mkv", {85.0F, 75.0F})), "b", "loaded-b", true}});
    // Before the window says what it can export: nothing to offer.
    EXPECT_TRUE(stats_menu(panel, "a").isEmpty());
    std::vector<std::string> items{"results", "graphs", "weak_segments"};
    bool run = true;
    panel.export_items = [&items] { return items; };
    panel.video_exportable = [](const std::string& identity) { return identity == "row-a"; };
    panel.videos_exportable = [&run] { return run; };
    EXPECT_EQ(stats_menu(panel, "a"), (QStringList{"Export this video's results, graphs and weak segments...",
                                                   "Export all videos' results, graphs and weak segments..."}));
    EXPECT_EQ(stats_menu(panel, "b"), QStringList{"Export all videos' results, graphs and weak segments..."});
    EXPECT_EQ(stats_menu(panel, "nowhere"), QStringList{"Export all videos' results, graphs and weak segments..."});
    items = {"results", "frames"};
    EXPECT_EQ(stats_menu(panel, "a"), (QStringList{"Export this video's results and per-frame scores...",
                                                   "Export all videos' results and per-frame scores..."}));
    items = {"results"};
    EXPECT_EQ(stats_menu(panel, "a"),
              (QStringList{"Export this video's results...", "Export all videos' results..."}));
    run = false;
    QStringList disabled;
    stats_menu(panel, "a", &disabled);
    EXPECT_EQ(disabled, QStringList{"Export all videos' results..."});
    items.clear();
    disabled.clear();
    EXPECT_EQ(stats_menu(panel, "a", &disabled),
              (QStringList{"Export: nothing is ticked in Settings > Storage > Exports include",
                           "Export: nothing is ticked in Settings > Storage > Exports include"}));
    EXPECT_EQ(disabled.size(), 2);
}

// An export that cannot be written says so, and not that it is complete.
TEST_F(GraphPanelTest, AFailedExportDoesNotSayItIsComplete) {
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(vmaf_result("a.mkv", {90.0F})), "a", {}, true},
                    {shared(vmaf_result("b.mkv", {80.0F})), "b", {}, true}});
    page(panel, "vmaf");
    difference_box(panel).setChecked(true);
    const test::Folder folder;
    std::ofstream(folder / "a file") << "x";
    const QString out = QString::fromStdString(platform::path_text(folder / "a file" / "a_minus_b.csv"));
    panel.dialogs.save_file = [&out](const QString&, const QString&, const QString&) { return out; };
    export_difference(panel);
    ASSERT_TRUE(panel.wait_until_file_writes_idle(std::chrono::seconds(10)));
    QCoreApplication::sendPostedEvents();  // what the writer's thread said
    EXPECT_TRUE(status(panel).startsWith("Could not")) << status(panel).toStdString();
}

// Export graph as PNG draws in the colours Settings asks for (export_colours), the
// space around the chart too; light when none are asked for.
TEST_F(GraphPanelTest, TheExportIsInTheColoursAskedFor) {
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(vmaf_result("a.mkv", {90.0F, 80.0F})), "a", {}, true}});
    page(panel, "vmaf");
    EXPECT_EQ(panel.render_export_image().toImage().pixelColor(1, 1), theme::chart_colours(false).background);
    panel.export_colours = [] { return theme::export_chart_colours("dark", "", ""); };
    EXPECT_EQ(panel.render_export_image().toImage().pixelColor(1, 1), theme::chart_colours(true).background);
}

// The readout pairs two lines by frame, as the Difference band does: every
// line at the frame the pointer picked, and the Δ row only for the same
// frame. (graph.json's Python app paired by time.)
TEST_F(GraphPanelTest, TheReadoutPairsTheLinesByFrame) {
    // 25 and 24 frames a second: at 0.8 s, the 24 fps run's nearest frame is
    // 19 (0.792 s), not 20.
    std::vector<int> frames(30);
    std::vector<float> frame_numbers(30);
    for (int i = 0; i < 30; ++i) {
        frames[static_cast<std::size_t>(i)] = i;
        frame_numbers[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    ComparisonResult at_25;
    at_25.fps = 25.0;
    add_frames(at_25, "vmaf", frames, std::vector<float>(30, 90.0F), 25.0);
    ComparisonResult at_24;
    at_24.fps = 24.0;
    add_frames(at_24, "vmaf", frames, frame_numbers, 24.0);
    {
        VideoColors colors;
        GraphPanel panel(&colors);
        panel.add_runs({{shared(at_25), "p25", "p25", true}, {shared(at_24), "p24", "p24", true}});
        MetricPage& vmaf = page(panel, "vmaf");
        vmaf.on_hover(0.8, 1000.0, entries(panel));  // every point at or below: the nearest in time, p25's frame 20
        const QString text = readout(vmaf);
        EXPECT_EQ(text.count(QStringLiteral("frame     20")), 2) << text.toStdString();
        EXPECT_TRUE(text.contains(QStringLiteral("Δ (p25 − p24) = +70.00"))) << text.toStdString();
    }

    // A shorter line first: the other's dip, at a frame it lacks, keeps its
    // own row, and there is no Δ of two different frames.
    std::vector<float> dip(10, 95.0F);
    dip[5] = 30.0F;
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(vmaf_result("short.mkv", {80.0F, 80.0F, 80.0F})), "short", {}, true},
                    {shared(vmaf_result("a.mkv", dip)), "a", {}, true}});
    MetricPage& vmaf = page(panel, "vmaf");
    vmaf.chart().resize(800, 400);
    vmaf.on_hover(5 / 30.0, 60.0, entries(panel));
    const QString text = readout(vmaf);
    EXPECT_TRUE(text.contains(QStringLiteral("frame      5   t=0:00:00.17   VMAF v0.6.1=30.00"))) << text.toStdString();
    EXPECT_FALSE(text.contains(QStringLiteral("Δ"))) << text.toStdString();
    // The crosshair on the dip picked, not on the first line's point.
    EXPECT_EQ(cursor_x(vmaf), vmaf.chart().pixel_for_time(5 / 30.0));

    // Three lines, one without a score there: no Δ of the other two.
    std::vector<float> unscored(10, 90.0F);
    unscored[5] = std::numeric_limits<float>::quiet_NaN();
    panel.add_run({shared(vmaf_result("c.mkv", unscored)), "c", {}, true});
    vmaf.on_hover(5 / 30.0, 60.0, entries(panel));
    EXPECT_FALSE(readout(vmaf).contains(QStringLiteral("Δ"))) << readout(vmaf).toStdString();
}

// Lines ticked off or on after a hover: the readout's text is of the lines
// shown, and fits the height the readout is given for them, whatever the
// order (it kept the hidden lines' rows, cut off at the bottom).
TEST_F(GraphPanelTest, TheReadoutFollowsTheLinesShown) {
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(vmaf_result("a.mkv", std::vector<float>(10, 90.0F))), "a", {}, true},
                    {shared(vmaf_result("b.mkv", std::vector<float>(10, 80.0F))), "b", {}, true},
                    {shared(vmaf_result("c.mkv", std::vector<float>(10, 70.0F))), "c", {}, true}});
    MetricPage& vmaf = page(panel, "vmaf");
    vmaf.chart().resize(800, 400);
    const auto lines_fit = [&vmaf] {
        // fit_hover_label's height: a line spacing a line, and 28 px of padding.
        const qsizetype lines = readout(vmaf).count(QLatin1Char('\n')) + 1;
        return lines * QFontMetrics(readout_font(vmaf)).lineSpacing() + 28 <= readout_height(vmaf);
    };
    vmaf.on_hover(5 / 30.0, 1000.0, entries(panel));
    EXPECT_TRUE(readout(vmaf).contains(QStringLiteral("[c]"))) << readout(vmaf).toStdString();
    for (const auto& [label, shown] : std::vector<std::pair<QString, bool>>{
             {"c", false}, {"b", false}, {"c", true}, {"b", true}, {"a", false}, {"c", false}}) {
        tick_row(panel, id_of(panel, label), shown);
        const QString text = readout(vmaf);
        EXPECT_EQ(text.contains("[" + label + "]"), shown) << text.toStdString();
        EXPECT_TRUE(lines_fit()) << text.toStdString();
    }
    // Go to frame's readout too: b alone, then a and b with their Δ, then b
    // alone again, without it.
    vmaf.show_frame(3, entries(panel));
    tick_row(panel, id_of(panel, "a"), true);
    EXPECT_TRUE(readout(vmaf).contains(QStringLiteral("Δ (a − b)"))) << readout(vmaf).toStdString();
    tick_row(panel, id_of(panel, "a"), false);
    EXPECT_FALSE(readout(vmaf).contains(QStringLiteral("Δ"))) << readout(vmaf).toStdString();
    EXPECT_TRUE(lines_fit()) << readout(vmaf).toStdString();
}

// XPSNR's infinity in the readout's Δ, as in the band: in both, no
// difference; in one alone, none to give.
TEST_F(GraphPanelTest, TheReadoutsDeltaOfAnIdenticalXpsnrFrame) {
    constexpr float infinity = std::numeric_limits<float>::infinity();
    ComparisonResult a = vmaf_result("a.mkv", {90.0F, 90.0F});
    add_frames(a, "xpsnr", {0, 1}, {infinity, infinity});
    ComparisonResult b = vmaf_result("b.mkv", {80.0F, 80.0F});
    add_frames(b, "xpsnr", {0, 1}, {infinity, 40.0F});
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(a), "a", {}, true}, {shared(b), "b", {}, true}});
    MetricPage& xpsnr = page(panel, "xpsnr");
    xpsnr.show_frame(0, entries(panel));
    EXPECT_TRUE(readout(xpsnr).contains(QStringLiteral("Δ (a − b) = +0.00"))) << readout(xpsnr).toStdString();
    xpsnr.show_frame(1, entries(panel));
    EXPECT_FALSE(readout(xpsnr).contains(QStringLiteral("Δ"))) << readout(xpsnr).toStdString();
}

TEST_F(GraphPanelTest, AnIdenticalXpsnrFrameIsDrawnAboveTheOthersAndKeptInTheScores) {
    // XPSNR scores a frame identical to its reference as infinity: drawn at
    // a stand-in above every real score (not as a gap, nor below good
    // frames); its statistics, readout and exports keep infinity.
    VideoColors colors;
    GraphPanel panel(&colors);
    ComparisonResult result = vmaf_result("a.mkv", {90, 90, 90, 90});
    result.metric_results.add(FrameMetricResult{"xpsnr", {0, 1, 2, 3}, {0.0, 0.1, 0.2, 0.3},
                                                {110.0F, INFINITY, NAN, INFINITY},
                                                current_ffmpeg_provenance("xpsnr", "9.0.1")});
    panel.add_run({shared(std::move(result)), "a", {}, true});
    MetricPage& xpsnr = page(panel, "xpsnr");
    const int id = id_of(panel, "a");
    const std::span<const float> plotted = drawn(xpsnr, id).values;
    EXPECT_EQ(plotted[0], 110.0F);
    EXPECT_EQ(plotted[1], static_cast<float>(xpsnr_infinity_plot_db));
    EXPECT_TRUE(std::isnan(plotted[2]));
    EXPECT_EQ(plotted[3], static_cast<float>(xpsnr_infinity_plot_db));
    const MetricCurve* curve = xpsnr.curve(id);
    ASSERT_NE(curve, nullptr);
    EXPECT_TRUE(std::isinf(curve->values[1]));
    EXPECT_TRUE(std::isinf(curve->stats.maximum));
}

TEST_F(GraphPanelTest, CvvdpsSecondsAreDrawnAtTheirMiddles) {
    // Each value is a whole second's: drawn at the second's middle, so that
    // the pointer anywhere over the second lands on it.
    VideoColors colors;
    GraphPanel panel(&colors);
    ComparisonResult result = vmaf_result("a.mkv", {90, 90, 90, 90, 90, 90});
    result.metric_results.add(SequenceMetricResult{"cvvdp", 8.75, current_ffmpeg_provenance("vmaf", "9.0.1"),
                                                   {0, 3, 6}, {0.0, 1.0, 2.0}, {8.5F, 9.0F, 8.0F}});
    panel.add_run({shared(std::move(result)), "a", {}, true});
    MetricPage& cvvdp = page(panel, "cvvdp");
    const std::span<const double> times = drawn(cvvdp, id_of(panel, "a")).times;
    EXPECT_EQ((std::vector<double>(times.begin(), times.end())), (std::vector<double>{0.5, 1.5, 2.5}));
}

TEST_F(GraphPanelTest, ARunHasOneLineAndOneTheUserRemovedStaysAwayUnlessAskedFor) {
    // Added again (a row's result replaced), a run's line is updated, not
    // doubled. Removed by the user, it stays away when the window brings the
    // graph up to date, and comes back when the user asks for the runs:
    // Show removed videos, enabled while a row's line is removed.
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.row_identities = [] { return std::vector<std::string>{"row a", "row b"}; };
    QPushButton* restore = nullptr;
    for (QPushButton* button : panel.findChildren<QPushButton*>()) {
        if (button->text() == "Show removed videos") {
            restore = button;
        }
    }
    ASSERT_NE(restore, nullptr);
    QSignalSpy asked(&panel, &GraphPanel::restore_removed_requested);
    panel.add_run({shared(vmaf_result("a.mkv", {80, 80})), "a", "row a", false});
    panel.add_run({shared(vmaf_result("b.mkv", {85, 85})), "b", "row b", false});
    panel.add_run({shared(vmaf_result("a.mkv", {99, 99})), "a", "row a", false});
    ASSERT_EQ(entries(panel).size(), 2u);
    EXPECT_EQ(entries(panel).at(id_of(panel, "a")).means.at("vmaf"), 99.0);

    EXPECT_FALSE(restore->isEnabled());
    panel.remove_runs({id_of(panel, "a")});
    EXPECT_TRUE(restore->isEnabled());
    panel.add_run({shared(vmaf_result("a.mkv", {99, 99})), "a", "row a", false});
    EXPECT_EQ(entries(panel).size(), 1u);
    EXPECT_TRUE(restore->isEnabled());
    restore->click();
    EXPECT_EQ(asked.count(), 1);
    panel.add_run({shared(vmaf_result("a.mkv", {99, 99})), "a", "row a", true});
    EXPECT_EQ(entries(panel).size(), 2u);
    EXPECT_FALSE(restore->isEnabled());
    // A line of no row (added from a file), removed: nothing to bring back.
    panel.add_run({shared(vmaf_result("c.mkv", {70, 70})), "c", "file c", false});
    panel.remove_runs({id_of(panel, "c")});
    EXPECT_FALSE(restore->isEnabled());
}

TEST_F(GraphPanelTest, AnUpdatedLineKeepsItsPlace) {
    // The table lists the lines in the order they were added, and a line set
    // again (a video's run going on) keeps its place on the page too: in the
    // readout, the difference's sign and the exported table. (The Python
    // app's page moved it to the end until release/v2.1 1408056.)
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_run({shared(vmaf_result("a.mkv", {90, 90, 90, 90})), "a", "row a", false});
    panel.add_run({shared(vmaf_result("b.mkv", {80, 80, 80, 80})), "b", "row b", false});
    MetricPage& vmaf = page(panel, "vmaf");
    panel.add_run({shared(vmaf_result("a.mkv", {91, 91, 91, 91})), "a", "row a", false});

    ASSERT_TRUE(vmaf.show_frame(1, entries(panel)));
    const QStringList lines = readout(vmaf).split('\n');
    ASSERT_EQ(lines.size(), 4) << readout(vmaf).toStdString();
    EXPECT_EQ((QStringList{lines[1].left(3), lines[2].left(3)}), (QStringList{"[a]", "[b]"}));
    EXPECT_EQ(lines[3].toStdString(), "Δ (a − b) = +11.00");
    QStringList exported;
    for (const auto& [label, color, cells] : export_table(panel).second) {
        exported.append(label);
    }
    EXPECT_EQ(exported, (QStringList{"a", "b"}));
}

TEST_F(GraphPanelTest, GoToFrameMovesAZoomedGraphToTheFrame) {
    // Issue #16: a frame out of a zoomed graph's view is brought into it.
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_run({shared(vmaf_result("a.mkv", std::vector<float>(3000, 90.0F))), "a", "row a", false});
    MetricPage& vmaf = page(panel, "vmaf");
    zoom(vmaf, 10.0, 10.0);
    frame_spin(panel).setValue(2400);  // 80 s at 30 fps
    const auto [start, end] = in_view(vmaf);
    EXPECT_DOUBLE_EQ(start, 75.0);
    EXPECT_DOUBLE_EQ(end, 85.0);
}

TEST_F(GraphPanelTest, TheExportedPngIsTheGraphWithItsKeyedTable) {
    // The file the user chose is written, a readable image: the chart and
    // the table, each row keyed by a square of its line's colour (the bare
    // chart says nothing once the file leaves the app).
    const test::Folder folder;
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_run({shared(vmaf_result("a.mkv", std::vector<float>(10, 95.0F))), "encode-a", {}, true});
    const std::filesystem::path out = folder / "graph.png";
    const QString out_text = QString::fromStdString(platform::path_text(out));
    QString suggested;
    panel.dialogs.save_file = [&out_text, &suggested](const QString&, const QString& name, const QString&) {
        suggested = name;
        return out_text;
    };
    export_png(panel);
    // Named after the one line's test video, and its metric.
    EXPECT_EQ(suggested.toStdString(), "encode-a_vmaf.png");
    const QImage image(out_text);
    ASSERT_FALSE(image.isNull());
    // The square: a block of the line's colour, which no line is thick
    // enough to make.
    constexpr int block = 8;
    const QRgb color = entries(panel).at(id_of(panel, "encode-a")).color.rgb();
    const auto solid_at = [&image, color](int left, int top) {
        for (int y = top; y < top + block; ++y) {
            for (int x = left; x < left + block; ++x) {
                if (image.pixel(x, y) != color) {
                    return false;
                }
            }
        }
        return true;
    };
    bool keyed = false;
    for (int y = 0; y + block <= image.height() && !keyed; ++y) {
        for (int x = 0; x + block <= image.width() && !keyed; ++x) {
            keyed = solid_at(x, y);
        }
    }
    EXPECT_TRUE(keyed) << "no square of the line's colour in the image";

    // With the lines of two test videos: after their reference.
    panel.add_run({shared(vmaf_result("b.mkv", std::vector<float>(10, 90.0F))), "encode-b", {}, true});
    export_png(panel);
    EXPECT_EQ(suggested.toStdString(), "source_vmaf.png");

    // A file that cannot be written says so.
    const QString unwritable = QString::fromStdString(platform::path_text(folder / "no such folder" / "graph.png"));
    panel.dialogs.save_file = [&unwritable](const QString&, const QString&, const QString&) { return unwritable; };
    QString alerted;
    panel.dialogs.alert = [&alerted](const QString& title, const QString&) { alerted = title; };
    export_png(panel);
    EXPECT_EQ(alerted.toStdString(), "Export failed");
}

TEST_F(GraphPanelTest, ALineWithNoScoresForTheMetricShowsDashesAndExports) {
    // b's VMAF scored no frame (every score NaN): no value in any of its
    // statistics, rather than zeros or the PSNR page's shares, and the graph
    // still exports. b comes first, as the exported table took its bands
    // from the first line and read past the end of b's. (The Python app's
    // table and export went wrong too until release/v2.1 c4ac19c.)
    VideoColors colors;
    GraphPanel panel(&colors);
    const auto run = [](const std::string& name, float vmaf, float psnr) {
        ComparisonResult result = vmaf_result(name, std::vector<float>(10, vmaf));
        FrameMetricResult scores{"psnr", {}, {}, std::vector<float>(10, psnr),
                                 current_ffmpeg_provenance("psnr", "9.0.1")};
        for (int i = 0; i < 10; ++i) {
            scores.frame.push_back(i);
            scores.time.push_back(i / 30.0);
        }
        result.metric_results.add(std::move(scores));
        return shared(std::move(result));
    };
    panel.add_run({run("b.mkv", NAN, 44.0F), "b", {}, true});
    panel.add_run({run("a.mkv", 90.0F, 40.0F), "a", {}, true});
    page(panel, "psnr");
    page(panel, "vmaf");

    QTableWidget& stats = table(panel);
    const int b = 0;  // added first
    ASSERT_EQ(stats.item(b, 0)->text(), "b");
    // After the means, before the ✕.
    for (int column = 1 + static_cast<int>(metrics().size()); column < stats.columnCount() - 1; ++column) {
        const QTableWidgetItem* cell = stats.item(b, column);
        ASSERT_NE(cell, nullptr) << column;
        EXPECT_EQ(cell->text().toStdString(), "—") << stats.horizontalHeaderItem(column)->text().toStdString();
    }
    const auto [headers, rows] = export_table(panel);
    ASSERT_EQ(rows.size(), 2u);
    for (const auto& [label, color, cells] : rows) {
        ASSERT_EQ(cells.size() + 1, headers.size()) << label.toStdString();
    }
    EXPECT_FALSE(panel.render_export_image().isNull());
}

// A run with VMAF's `vmaf` (a frame each at 30 a second) and CVVDP's
// seconds: frames 0, 3 and 6 begin them, scoring 8.5, 9 and 8.
std::shared_ptr<const ComparisonResult> range_run(const std::string& name, const std::vector<float>& vmaf) {
    ComparisonResult result;
    result.source = "source.mkv";
    result.distorted = name;
    result.fps = 30.0;
    FrameMetricResult scores{"vmaf", {}, {}, vmaf, current_ffmpeg_provenance("vmaf", "9.0.1")};
    for (std::size_t i = 0; i < vmaf.size(); ++i) {
        scores.frame.push_back(static_cast<int>(i));
        scores.time.push_back(static_cast<double>(i) / 30.0);
    }
    result.metric_results.add(std::move(scores));
    result.metric_results.add(SequenceMetricResult{"cvvdp", 8.75, current_ffmpeg_provenance("vmaf", "9.0.1"),
                                                   {0, 3, 6}, {0.0, 0.1, 0.2}, {8.5F, 9.0F, 8.0F}});
    return std::make_shared<const ComparisonResult>(std::move(result));
}

// Go to time and a range in times, to the millisecond, as complements of the
// frame boxes: a time is the frame shown then; a range of times the frames
// from the first at its start or after to the last at its end or before.
// The time boxes and the frame boxes show the same frames.
TEST_F(GraphPanelTest, GoToTimeAndARangeOfTimes) {
    EXPECT_EQ(FrameControls::parse_time("1:02.5"), 62.5);
    EXPECT_EQ(FrameControls::parse_time(" 0:01:02.500 "), 62.5);
    EXPECT_EQ(FrameControls::parse_time("62,25"), 62.25);
    EXPECT_EQ(FrameControls::parse_time("1:00:00"), 3600.0);
    EXPECT_EQ(FrameControls::parse_time("0.0405"), std::nullopt);  // past a millisecond
    EXPECT_EQ(FrameControls::parse_time("1:75"), std::nullopt);
    EXPECT_EQ(FrameControls::parse_time(""), std::nullopt);
    EXPECT_EQ(FrameControls::time_text(80.0333333), "0:01:20.033");
    EXPECT_EQ(FrameControls::time_text(5576.279), "1:32:56.279");

    VideoColors colors;
    GraphPanel panel(&colors);
    FrameControls& controls = frame_controls(panel);
    EXPECT_TRUE(controls.time_edit->isVisibleTo(&panel));
    EXPECT_TRUE(controls.range_start_time->isVisibleTo(&panel));

    // 30 fps: frame 2400 is at 80 s, 2401 at 80.0333 s.
    panel.add_run({shared(vmaf_result("a.mkv", std::vector<float>(3000, 90.0F))), "a", "row a", false});
    MetricPage& vmaf = page(panel, "vmaf");
    zoom(vmaf, 10.0, 10.0);
    type_time(*controls.time_edit, "1:20.017");  // frame 2400 is still shown
    EXPECT_EQ(frame_spin(panel).value(), 2400);
    const auto [start, end] = in_view(vmaf);
    EXPECT_DOUBLE_EQ(start, 75.0);
    EXPECT_DOUBLE_EQ(end, 85.0);
    type_time(*controls.time_edit, "0:01:20.033");  // 2401's time as shown
    EXPECT_EQ(frame_spin(panel).value(), 2401);
    frame_spin(panel).setValue(2700);
    EXPECT_EQ(controls.time_edit->text(), "0:01:30.000");
    type_time(*controls.time_edit, "nonsense");
    EXPECT_EQ(frame_spin(panel).value(), 2700);

    // A range of times: 0.1 s is frame 3, and frame 7 (0.2333 s) is the last
    // at 0.234 s or before.
    GraphPanel ranged(&colors);
    FrameControls& range_controls = frame_controls(ranged);
    ranged.add_run({range_run("a.mkv", {80, 82, 84, 86, 88, 90, 70, 60, 95, 99}), "a", {}, true});
    MetricPage& ranged_vmaf = page(ranged, "vmaf");
    type_time(*range_controls.range_start_time, "0:00:00.100");
    EXPECT_FALSE(ranged_vmaf.chart().highlight());  // one end: no range yet
    type_time(*range_controls.range_end_time, "0.234");
    EXPECT_EQ(range_start_spin(ranged).value(), 3);
    EXPECT_EQ(range_end_spin(ranged).value(), 7);
    EXPECT_TRUE(stats_hint(ranged).contains("frames 3 to 7")) << stats_hint(ranged).toStdString();
    ASSERT_TRUE(ranged_vmaf.chart().highlight());
    EXPECT_NEAR(ranged_vmaf.chart().highlight()->first, 2.5 / 30, 1e-9);
    // The times then shown are the frames': 0.234 s typed, frame 7's shown.
    EXPECT_EQ(range_controls.range_start_time->text(), "0:00:00.100");
    EXPECT_EQ(range_controls.range_end_time->text(), "0:00:00.233");

    // Frames chosen in their boxes: the time boxes show their times.
    range_start_spin(ranged).setValue(2);
    range_end_spin(ranged).setValue(4);
    EXPECT_EQ(range_controls.range_start_time->text(), "0:00:00.067");
    EXPECT_EQ(range_controls.range_end_time->text(), "0:00:00.133");
    // And those times typed back are the same frames.
    type_time(*range_controls.range_start_time, "0:00:00.067");
    type_time(*range_controls.range_end_time, "0:00:00.133");
    EXPECT_EQ(range_start_spin(ranged).value(), 2);
    EXPECT_EQ(range_end_spin(ranged).value(), 4);

    // No frame between the times: no range.
    clear_range_button(ranged).click();
    type_time(*range_controls.range_start_time, "0:00:00.040");
    type_time(*range_controls.range_end_time, "0:00:00.060");
    EXPECT_FALSE(ranged_vmaf.chart().highlight());
    EXPECT_EQ(range_start_spin(ranged).value(), range_start_spin(ranged).minimum());

    EXPECT_TRUE(clear_range_button(ranged).isEnabled());  // times typed to clear
    clear_range_button(ranged).click();
    EXPECT_TRUE(range_controls.range_start_time->text().isEmpty());
    EXPECT_TRUE(range_controls.range_end_time->text().isEmpty());
}

TEST_F(GraphPanelTest, AFrameRangesStatisticsAndShading) {
    // Issue #17: frames 7 to 3 (either way round) typed into the range's own
    // boxes give their statistics in the table and are shaded on the graph;
    // one end alone is no range yet, and Go to frame leaves the range alone.
    // A run with none of the frames has none, and CVVDP no score for part of
    // a video. Clear range brings back the whole videos' table exactly.
    VideoColors colors;
    GraphPanel panel(&colors);
    const std::vector<float> a_vmaf{80, 82, 84, 86, 88, 90, 70, 60, 95, 99};
    panel.add_run({range_run("a.mkv", a_vmaf), "a", {}, true});
    panel.add_run({range_run("b.mkv", {50, 51, 52}), "b", {}, true});
    MetricPage& vmaf = page(panel, "vmaf");
    const std::vector<QStringList> whole = cell_texts(panel);

    range_start_spin(panel).setValue(7);
    EXPECT_FALSE(vmaf.chart().highlight());
    EXPECT_EQ(cell_texts(panel), whole);
    EXPECT_TRUE(clear_range_button(panel).isEnabled());
    range_end_spin(panel).setValue(3);
    EXPECT_TRUE(stats_hint(panel).contains("frames 3 to 7")) << stats_hint(panel).toStdString();
    frame_spin(panel).setValue(1);
    EXPECT_EQ(range_start_spin(panel).value(), 7);
    EXPECT_TRUE(stats_hint(panel).contains("frames 3 to 7")) << stats_hint(panel).toStdString();

    const MetricDefinition& spec = *find_metric("vmaf");
    const std::vector<double> in_range(a_vmaf.begin() + 3, a_vmaf.begin() + 8);
    const ScoreStats stats = compute_stats(in_range, spec.thresholds, spec.aggregation, spec.direction);
    const auto text = [&spec](std::optional<double> value) { return QString::fromStdString(spec.format_value(value)); };
    const std::span<const MetricDefinition> all = metrics();
    const auto column_of = [&all](std::string_view key) {
        return 1 + static_cast<int>(std::find_if(all.begin(), all.end(), [key](const auto& m) { return m.key == key; }) -
                                    all.begin());
    };
    const int details = 1 + static_cast<int>(all.size());
    QTableWidget& cells = table(panel);
    ASSERT_EQ(cells.item(0, 0)->text(), "a");
    EXPECT_EQ(cells.item(0, column_of("vmaf"))->text(), text(aggregate_scores(in_range, spec.aggregation)));
    EXPECT_EQ(cells.item(0, details)->text(), text(stats.median));
    EXPECT_EQ(cells.item(0, details + 1)->text(), text(stats.stdev));
    EXPECT_EQ(cells.item(0, details + 2)->text(), text(stats.minimum));
    EXPECT_EQ(cells.item(0, details + 3)->text(), text(stats.maximum));
    EXPECT_EQ(cells.item(0, column_of("cvvdp"))->text(), "—");
    EXPECT_EQ(cells.item(1, column_of("vmaf"))->text(), "—");  // b's frames end at 2
    EXPECT_EQ(cells.item(1, details)->text(), "—");
    // Half a frame either side of frames 3 and 7.
    ASSERT_TRUE(vmaf.chart().highlight());
    EXPECT_NEAR(vmaf.chart().highlight()->first, 2.5 / 30, 1e-9);
    EXPECT_NEAR(vmaf.chart().highlight()->second, 7.5 / 30, 1e-9);

    clear_range_button(panel).click();
    EXPECT_FALSE(vmaf.chart().highlight());
    EXPECT_FALSE(clear_range_button(panel).isEnabled());
    EXPECT_EQ(range_start_spin(panel).value(), range_start_spin(panel).minimum());
    EXPECT_EQ(range_end_spin(panel).value(), range_end_spin(panel).minimum());
    EXPECT_EQ(cell_texts(panel), whole);
}

TEST_F(GraphPanelTest, AFrameRangeOnCvvdpsPageIsTheSecondsItIsIn) {
    // Frames 3 to 7 are in the seconds beginning at frames 3 and 6: their
    // statistics and shading; no whole-video score, in the table or the
    // exported image's.
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_run({range_run("a.mkv", {80, 82, 84, 86, 88, 90, 70, 60, 95, 99}), "a", {}, true});
    MetricPage& cvvdp = page(panel, "cvvdp");
    range_start_spin(panel).setValue(3);
    range_end_spin(panel).setValue(7);
    const MetricDefinition& spec = *find_metric("cvvdp");
    const ScoreStats seconds =
        compute_stats(std::vector<double>{9.0, 8.0}, spec.thresholds, spec.aggregation, spec.direction);
    const int details = 1 + static_cast<int>(metrics().size());
    EXPECT_EQ(table(panel).item(0, details)->text(), QString::fromStdString(spec.format_value(seconds.median)));
    EXPECT_TRUE(stats_hint(panel).contains("whole seconds")) << stats_hint(panel).toStdString();
    ASSERT_TRUE(cvvdp.chart().highlight());
    EXPECT_NEAR(cvvdp.chart().highlight()->first, 0.1, 1e-9);
    EXPECT_NEAR(cvvdp.chart().highlight()->second, 0.3, 1e-9);
    const auto [headers, rows] = export_table(panel);
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(std::get<2>(rows.front())[0], "—");
}

TEST_F(GraphPanelTest, AShiftDragAcrossTheGraphSelectsFrames) {
    // Shift+drag from frame 2 to frame 6 fills the range's two boxes and
    // shades the frames, leaving Go to frame as it was; a plain drag pans and
    // leaves the range as it is.
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_run({range_run("a.mkv", {80, 82, 84, 86, 88, 90, 70, 60, 95, 99}), "a", {}, true});
    MetricPage& vmaf = page(panel, "vmaf");
    ChartWidget& chart = vmaf.chart();
    chart.resize(800, 400);
    const int y = chart.plot_rect().center().y();
    const auto press_move_release = [&chart, y](double from, double to, Qt::KeyboardModifiers modifiers) {
        const QPointF start(chart.pixel_for_time(from), y);
        const QPointF end(chart.pixel_for_time(to), y);
        QMouseEvent press(QEvent::MouseButtonPress, start, start, Qt::LeftButton, Qt::LeftButton, modifiers);
        QMouseEvent move(QEvent::MouseMove, end, end, Qt::NoButton, Qt::LeftButton, modifiers);
        QMouseEvent release(QEvent::MouseButtonRelease, end, end, Qt::LeftButton, Qt::NoButton, modifiers);
        QCoreApplication::sendEvent(&chart, &press);
        QCoreApplication::sendEvent(&chart, &move);
        QCoreApplication::sendEvent(&chart, &release);
    };
    press_move_release(2 / 30.0, 6 / 30.0, Qt::ShiftModifier);
    EXPECT_EQ(range_start_spin(panel).value(), 2);
    EXPECT_EQ(range_end_spin(panel).value(), 6);
    EXPECT_EQ(frame_spin(panel).value(), 0);
    EXPECT_TRUE(stats_hint(panel).contains("frames 2 to 6")) << stats_hint(panel).toStdString();
    ASSERT_TRUE(chart.highlight());
    EXPECT_NEAR(chart.highlight()->first, 1.5 / 30, 1e-9);

    zoom(vmaf, 0.1, 0.1);
    press_move_release(0.12, 0.15, Qt::NoModifier);
    EXPECT_LT(in_view(vmaf).first, 0.1);  // panned
    EXPECT_EQ(range_end_spin(panel).value(), 6);
    EXPECT_TRUE(chart.highlight());
}

TEST_F(GraphPanelTest, WorstFramesListsTheDipsAndOpensThemInVideoCompare) {
    // Issue #24: the worst frame of each dip, within the range; a
    // double-click asks for that frame of that line in Video Compare; Weak
    // segments starts from the metric's first band of worse frames.
    VideoColors colors;
    GraphPanel panel(&colors);
    std::vector<float> values(90, 95.0F);
    values[9] = 60.0F;
    values[10] = 40.0F;
    values[70] = 50.0F;
    panel.add_run({shared(vmaf_result("a.mkv", values)), "a", "row:a", true});
    page(panel, "vmaf");
    worst_button(panel).click();
    QTableWidget& list = *worst_panel(panel).table();
    ASSERT_EQ(list.rowCount(), 2);
    EXPECT_EQ(list.item(0, 0)->text(), "10");
    EXPECT_EQ(list.item(1, 0)->text(), "70");
    EXPECT_EQ(list.item(0, 2)->text(), "40.00");

    std::pair<std::string, int> asked;
    WorstFrameList handed;
    QObject::connect(&panel, &GraphPanel::video_compare_requested,
                     [&](const std::string& identity, int frame, const WorstFrameList& stops) {
                         asked = {identity, frame};
                         handed = stops;
                     });
    emit list.cellDoubleClicked(1, 0);
    const std::pair<std::string, int> wanted{"row:a", 70};
    EXPECT_EQ(asked, wanted);
    // The list goes with it, for Video Compare's Previous and Next worst.
    EXPECT_EQ(handed.identity, "row:a");
    ASSERT_EQ(handed.stops.size(), 2U);
    EXPECT_EQ(handed.stops[0].frame, 10);
    EXPECT_EQ(handed.stops[0].score, "40.00");
    EXPECT_EQ(handed.stops[1].frame, 70);

    range_start_spin(panel).setValue(50);
    range_end_spin(panel).setValue(89);
    ASSERT_EQ(list.rowCount(), 1);
    EXPECT_EQ(list.item(0, 0)->text(), "70");

    range_start_spin(panel).setValue(0);
    QStringList suggested;
    panel.dialogs.save_file = [&suggested](const QString&, const QString& name, const QString&) {
        suggested.append(name);
        return QString();  // Cancel
    };
    worst_panel(panel).segments_button()->click();
    auto* dialog = panel.findChild<WeakSegmentsDialog*>();
    ASSERT_NE(dialog, nullptr);
    EXPECT_EQ(dialog->rules().conditions.front().text(), "vmaf<85");
    ASSERT_EQ(dialog->segments().size(), 2U);
    EXPECT_EQ(dialog->segments()[0].first_frame, 9);
    EXPECT_EQ(dialog->segments()[0].last_frame, 10);
    EXPECT_EQ(dialog->segments()[1].first_frame, 70);

    // The exports are named after the metrics they cover, so that one by
    // another metric does not take this one's name.
    for (QPushButton* button : dialog->findChildren<QPushButton*>()) {
        if (button->text().startsWith("Export")) {
            button->click();
        }
    }
    EXPECT_EQ(suggested, (QStringList{"a.vmaf.segments.json", "a.vmaf.segments.csv"}));
}

TEST_F(GraphPanelTest, WeakSegmentsTakesProfilesChangedInSettingsWhileOpen) {
    // Issue #135: the dialog is not modal. A profile made in Settings while
    // it is open is not replaced by Save as new profile under its name, and
    // Save to profile on one deleted there does not bring it back.
    VideoColors colors;
    GraphPanel panel(&colors);
    std::vector<float> values(90, 95.0F);
    values[40] = 60.0F;
    panel.add_run({shared(vmaf_result("a.mkv", values)), "a", "row:a", true});
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<70"};
    settings.weak_segment_profiles_json = R"([{"name": "Strict", "conditions": ["vmaf<92"]}])";
    const auto hand_back = [&] {
        panel.set_segment_profiles(segment_profiles(settings), segment_profile_in_use(settings).name);
    };
    hand_back();
    QObject::connect(&panel, &GraphPanel::weak_segment_rules_saved,
                     [&](const std::string& profile, const SegmentRules& rules) {
                         store_segment_profile(settings, profile, rules);
                         hand_back();
                     });
    QString alerted;
    panel.dialogs.ask_text = [](const QString&, const QString&, const QString&) -> std::optional<QString> {
        return QStringLiteral("film");
    };
    panel.dialogs.alert = [&alerted](const QString& title, const QString&) { alerted = title; };
    page(panel, "vmaf");
    worst_button(panel).click();
    worst_panel(panel).segments_button()->click();
    auto* dialog = panel.findChild<WeakSegmentsDialog*>();
    ASSERT_NE(dialog, nullptr);
    dialog->profile_combo()->setCurrentIndex(dialog->profile_combo()->findText("Strict"));

    // Meanwhile, in Settings: Film made, Strict deleted.
    store_segment_profiles(settings,
                           {segment_profiles(settings).front(), {"Film", SegmentRules{{{"vmaf", '<', 80.0}}}}});
    hand_back();
    EXPECT_EQ(dialog->profile_combo()->findText("Strict"), -1);
    EXPECT_EQ(dialog->profile_combo()->currentText(), "Default");  // the one in use
    EXPECT_EQ(condition_texts(dialog->rules().conditions), std::vector<std::string>{"vmaf<92"});  // kept

    dialog->save_new_button()->click();
    EXPECT_EQ(alerted, "Profile not saved");
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "Film")->rules.conditions),
              std::vector<std::string>{"vmaf<80"});
    ASSERT_TRUE(dialog->save_button()->isEnabled());
    dialog->save_button()->click();
    EXPECT_FALSE(segment_profile_named(settings, "Strict"));
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "")->rules.conditions),
              std::vector<std::string>{"vmaf<92"});
}

TEST_F(GraphPanelTest, WeakSegmentsStartsFromAProfileAndSavesOnlyWhenAsked) {
    // The profile in use (Settings > Weak segments) is where the dialog
    // starts; another can be picked. What is changed there holds for the
    // dialog alone until Save to profile (the profile picked: its conditions
    // on metrics this video has none of left as they were) or Save as new
    // profile.
    VideoColors colors;
    GraphPanel panel(&colors);
    std::vector<float> values(90, 95.0F);
    values[20] = 75.0F;
    values[40] = 60.0F;
    panel.add_run({shared(vmaf_result("a.mkv", values)), "a", "row:a", true});
    Settings settings;  // as the window keeps them, handing them back after each save
    settings.weak_segment_conditions = {"ssimulacra2<60", "vmaf<70"};
    settings.weak_segments_all = true;
    settings.weak_segment_merge_frames = 3;
    settings.weak_segment_profiles_json = R"([{"name": "Strict", "conditions": ["vmaf<92"]}])";
    const auto hand_back = [&] { panel.set_segment_profiles(segment_profiles(settings), segment_profile_in_use(settings).name); };
    hand_back();
    std::vector<std::pair<std::string, SegmentRules>> saved;
    QObject::connect(&panel, &GraphPanel::weak_segment_rules_saved,
                     [&](const std::string& profile, const SegmentRules& rules) {
                         saved.emplace_back(profile, rules);
                         store_segment_profile(settings, profile, rules);
                         hand_back();
                     });
    QStringList names{"Mine", "strict"};
    QString alerted;
    panel.dialogs.ask_text = [&names](const QString&, const QString&, const QString&) -> std::optional<QString> {
        return names.takeFirst();
    };
    panel.dialogs.alert = [&alerted](const QString& title, const QString&) { alerted = title; };
    page(panel, "vmaf");
    worst_button(panel).click();
    worst_panel(panel).segments_button()->click();
    auto* dialog = panel.findChild<WeakSegmentsDialog*>();
    ASSERT_NE(dialog, nullptr);
    EXPECT_EQ(dialog->profile_combo()->currentText(), "Default");
    const SegmentRules shown = dialog->rules();
    EXPECT_EQ(condition_texts(shown.conditions), std::vector<std::string>{"vmaf<70"});
    EXPECT_TRUE(shown.all);
    EXPECT_EQ(shown.merge_frames, 3);
    ASSERT_EQ(dialog->segments().size(), 1U);  // frame 40
    EXPECT_FALSE(dialog->save_button()->isEnabled());  // as saved

    QDoubleSpinBox* threshold = dialog->findChild<QDoubleSpinBox*>();
    ASSERT_NE(threshold, nullptr);
    threshold->setValue(80.0);
    EXPECT_EQ(dialog->segments().size(), 2U);  // frames 20 and 40, at once
    EXPECT_TRUE(saved.empty());  // not saved by changing it
    ASSERT_TRUE(dialog->save_button()->isEnabled());
    dialog->save_button()->click();
    ASSERT_EQ(saved.size(), 1U);
    EXPECT_EQ(saved[0].first, "");
    EXPECT_EQ(condition_texts(saved[0].second.conditions), (std::vector<std::string>{"vmaf<80", "ssimulacra2<60"}));
    EXPECT_TRUE(saved[0].second.ranges.empty());
    EXPECT_FALSE(dialog->save_button()->isEnabled());

    // Another profile picked: its rules, as saved.
    dialog->profile_combo()->setCurrentIndex(dialog->profile_combo()->findText("Strict"));
    EXPECT_EQ(condition_texts(dialog->rules().conditions), std::vector<std::string>{"vmaf<92"});
    EXPECT_FALSE(dialog->rules().all);
    EXPECT_FALSE(dialog->save_button()->isEnabled());
    // Saved as a new profile, which is then the one picked; a name taken
    // (any case) is refused.
    dialog->save_new_button()->click();
    ASSERT_EQ(saved.size(), 2U);
    EXPECT_EQ(saved[1].first, "Mine");
    EXPECT_EQ(dialog->profile_combo()->currentText(), "Mine");
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "Mine")->rules.conditions),
              std::vector<std::string>{"vmaf<92"});
    dialog->save_new_button()->click();
    EXPECT_EQ(saved.size(), 2U);
    EXPECT_EQ(alerted, "Profile not saved");

    // Changed and closed without saving: opened again from the profile in
    // use (still Default), as saved.
    threshold = dialog->findChild<QDoubleSpinBox*>();
    threshold->setValue(90.0);
    delete dialog;
    worst_panel(panel).segments_button()->click();
    dialog = panel.findChild<WeakSegmentsDialog*>();
    ASSERT_NE(dialog, nullptr);
    EXPECT_EQ(dialog->profile_combo()->currentText(), "Default");
    EXPECT_EQ(condition_texts(dialog->rules().conditions), std::vector<std::string>{"vmaf<80"});
    EXPECT_EQ(dialog->rules().merge_frames, 3);

    // Export all videos... asks the window for its report of every video, by
    // the rules shown: unsaved ones too.
    threshold = dialog->findChild<QDoubleSpinBox*>();
    threshold->setValue(85.0);
    std::vector<SegmentRules> reports;
    QObject::connect(&panel, &GraphPanel::segments_report_requested,
                     [&reports](const SegmentRules& rules) { reports.push_back(rules); });
    for (QPushButton* button : dialog->findChildren<QPushButton*>()) {
        if (button->text() == "Export all videos") {
            button->click();
        }
    }
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(condition_texts(reports[0].conditions), std::vector<std::string>{"vmaf<85"});
    EXPECT_EQ(reports[0].merge_frames, 3);
}

TEST_F(GraphPanelTest, WeakSegmentsLoadsAPresetWithoutSavingIt) {
    // From preset loads a built-in preset's rules in place of those shown,
    // as Settings does into a profile: the segments follow at once, nothing
    // is saved until Save to profile, which then keeps the preset's
    // conditions on metrics this video has none of too.
    VideoColors colors;
    GraphPanel panel(&colors);
    std::vector<float> values(90, 95.0F);
    for (int frame = 20; frame < 30; ++frame) {
        values[static_cast<std::size_t>(frame)] = 80.0F;  // ten frames below 85
    }
    values[50] = 80.0F;  // two, shorter than Grainy film's six
    values[51] = 80.0F;
    panel.add_run({shared(vmaf_result("a.mkv", values)), "a", "row:a", true});
    Settings settings;
    settings.weak_segment_conditions = {"ssimulacra2<60", "vmaf<70"};
    panel.set_segment_profiles(segment_profiles(settings), segment_profile_in_use(settings).name);
    std::vector<std::pair<std::string, SegmentRules>> saved;
    QObject::connect(&panel, &GraphPanel::weak_segment_rules_saved,
                     [&saved](const std::string& profile, const SegmentRules& rules) {
                         saved.emplace_back(profile, rules);
                     });
    page(panel, "vmaf");
    worst_button(panel).click();
    worst_panel(panel).segments_button()->click();
    auto* dialog = panel.findChild<WeakSegmentsDialog*>();
    ASSERT_NE(dialog, nullptr);
    EXPECT_TRUE(dialog->segments().empty());  // nothing below 70
    QMenu* menu = dialog->preset_button()->menu();
    ASSERT_NE(menu, nullptr);
    QAction* grainy = nullptr;
    for (QAction* action : menu->actions()) {
        if (action->text() == "Grainy film") {
            grainy = action;
        }
        // HDR's one condition is on CVVDP, which this video has no scores of.
        EXPECT_EQ(action->isEnabled(), action->text() != "HDR") << action->text().toStdString();
    }
    ASSERT_NE(grainy, nullptr);
    grainy->trigger();
    const SegmentRules shown = dialog->rules();
    EXPECT_EQ(condition_texts(shown.conditions), std::vector<std::string>{"vmaf<85"});
    EXPECT_EQ(shown.min_frames, 6);
    EXPECT_EQ(shown.merge_frames, 12);
    ASSERT_EQ(dialog->segments().size(), 1U);
    EXPECT_EQ(dialog->segments()[0].first_frame, 20);
    EXPECT_EQ(dialog->segments()[0].last_frame, 29);
    EXPECT_EQ(dialog->profile_combo()->currentText(), "Default");
    EXPECT_TRUE(saved.empty());  // not saved by loading it
    ASSERT_TRUE(dialog->save_button()->isEnabled());

    dialog->save_button()->click();
    ASSERT_EQ(saved.size(), 1U);
    EXPECT_EQ(saved[0].first, "");
    EXPECT_EQ(condition_texts(saved[0].second.conditions), (std::vector<std::string>{"vmaf<85", "ssimulacra2<55"}));
    EXPECT_EQ(saved[0].second.min_frames, 6);
    EXPECT_FALSE(dialog->save_button()->isEnabled());
}

TEST_F(GraphPanelTest, WeakSegmentsSavesTheHiddenConditionsSettingsHasNow) {
    // Save to profile keeps the profile's conditions on metrics this video
    // has none of as Settings has them now: one removed there while the
    // dialog is open is not put back. A preset loaded keeps its own.
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_run({shared(vmaf_result("a.mkv", std::vector<float>(90, 95.0F))), "a", "row:a", true});
    Settings settings;
    settings.weak_segment_conditions = {"ssimulacra2<60", "vmaf<70"};
    const auto hand_back = [&] {
        panel.set_segment_profiles(segment_profiles(settings), segment_profile_in_use(settings).name);
    };
    hand_back();
    std::vector<std::pair<std::string, SegmentRules>> saved;
    QObject::connect(&panel, &GraphPanel::weak_segment_rules_saved,
                     [&saved](const std::string& profile, const SegmentRules& rules) {
                         saved.emplace_back(profile, rules);
                     });
    page(panel, "vmaf");
    worst_button(panel).click();
    worst_panel(panel).segments_button()->click();
    auto* dialog = panel.findChild<WeakSegmentsDialog*>();
    ASSERT_NE(dialog, nullptr);
    EXPECT_FALSE(dialog->save_button()->isEnabled());  // as saved

    settings.weak_segment_conditions = {"vmaf<70"};  // in Settings: SSIMULACRA2's removed
    hand_back();
    EXPECT_FALSE(dialog->save_button()->isEnabled());  // still as saved: nothing to put back

    for (QAction* action : dialog->preset_button()->menu()->actions()) {
        if (action->text() == "Grainy film") {
            action->trigger();
        }
    }
    settings.weak_segment_conditions = {"vmaf<75"};  // changed in Settings again
    hand_back();
    ASSERT_TRUE(dialog->save_button()->isEnabled());
    dialog->save_button()->click();
    ASSERT_EQ(saved.size(), 1U);
    EXPECT_EQ(condition_texts(saved[0].second.conditions), (std::vector<std::string>{"vmaf<85", "ssimulacra2<55"}));
}

TEST_F(GraphPanelTest, WorstFramesKeepsTheChosenVideoWhileItIsRecalculated) {
    // Recalculating a video takes its line away until its new scores come;
    // the list shows another video meanwhile, then the chosen one again
    // (it stayed on the other video before).
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_run({shared(vmaf_result("a.mkv", {95, 40, 95})), "a", "row:a", true});
    panel.add_run({shared(vmaf_result("b.mkv", {95, 50, 95})), "b", "row:b", true});
    page(panel, "vmaf");
    worst_button(panel).click();
    WorstFramesPanel& worst = worst_panel(panel);
    QComboBox* videos = nullptr;
    for (QComboBox* combo : worst.findChildren<QComboBox*>()) {
        if (combo->count() > 0 && combo->itemText(0) == "a") {
            videos = combo;
        }
    }
    ASSERT_NE(videos, nullptr);
    videos->setCurrentIndex(1);
    ASSERT_EQ(worst.chosen_video()->identity, "row:b");

    panel.remove_by_identity("row:b");
    EXPECT_EQ(worst.chosen_video()->identity, "row:a");
    panel.add_run({shared(vmaf_result("b.mkv", {95, 55, 95})), "b", "row:b", false});
    EXPECT_EQ(worst.chosen_video()->identity, "row:b");
    EXPECT_EQ(worst.table()->item(0, 2)->text(), "55.00");
}
}  // namespace
}  // namespace vml::app
