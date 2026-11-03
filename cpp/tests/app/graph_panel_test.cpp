// Metric Graphs, held to the Python app's own panel (reference/graph.json):
// the table and the readouts for runs of every kind of score (but for the
// readouts' Δ lines, cut from graph.json where the Python app paired
// different frames or three lines were shown); and what the
// graph must keep: XPSNR's infinity in the scores while it is drawn at a
// stand-in, CVVDP's seconds at their middles, one line per run, a removed
// line kept away, an image to export.
#include "graph_panel.h"

#include <QCheckBox>
#include <QImage>
#include <QLabel>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

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
#include "reference.h"
#include "series_colors.h"
#include "test_videos.h"
#include "vml/platform/system.h"

namespace vml::app {

// The graph's parts, for its tests: the class GraphPanel, MetricPage and
// ChartWidget let in.
class GraphPanelTest : public ::testing::Test {
protected:
    static QTableWidget& table(GraphPanel& panel) { return *panel.stats_table_; }
    static QTabWidget& tabs(GraphPanel& panel) { return *panel.tabs_; }
    static QSpinBox& frame_spin(GraphPanel& panel) { return *panel.frame_spin_; }
    static QCheckBox& difference_box(GraphPanel& panel) { return *panel.difference_box_; }
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
    // The crosshair's pixel column on a page's chart, if shown.
    static std::optional<int> cursor_x(MetricPage& page) { return page.chart_->cursor_x_; }
    // What a page's chart draws for a line.
    static const ChartSeries& drawn(MetricPage& page, int series_id) { return page.chart_->series_.at(series_id); }
    static void export_png(GraphPanel& panel) { panel.on_export_png(); }
    static void export_csvs(GraphPanel& panel) { panel.on_export_csv(); }
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

// Export CSV with Difference on writes, beside each line's file, the
// difference the band shows, <first>_minus_<second>.csv: of three runs
// shown, the two with the page's metric.
TEST_F(GraphPanelTest, ExportsTheDifferenceTheBandShows) {
    ComparisonResult c;  // PSNR only: not on VMAF's page
    add_frames(c, "psnr", {0, 1, 2}, {40.0F, 40.0F, 40.0F});
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_runs({{shared(vmaf_result("a.mkv", {90.0F, 80.0F, 70.0F})), "a", {}, true},
                    {shared(vmaf_result("b.mkv", {85.0F, 82.0F, 60.0F})), "b", {}, true},
                    {shared(c), "c", {}, true}});
    page(panel, "vmaf");
    difference_box(panel).setChecked(true);
    const test::Folder folder;
    const QString out = QString::fromStdString(platform::path_text(folder / ""));
    panel.dialogs.choose_folder = [&out](const QString&, const QString&) { return out; };
    export_csvs(panel);
    ASSERT_TRUE(panel.wait_until_file_writes_idle(std::chrono::seconds(10)));
    const auto lines_of = [&folder](const std::string& name) {
        std::ifstream file(folder / name, std::ios::binary);
        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);) {
            lines.push_back(line);
        }
        return lines;
    };
    EXPECT_TRUE(std::filesystem::exists(folder / "c.csv"));
    const std::vector<std::string> difference = lines_of("a_minus_b.csv");
    ASSERT_EQ(difference.size(), 4U);
    EXPECT_EQ(difference[0], lines_of("a.csv")[0]);
    // frame, time_s, then VMAF's column; the other metrics' are blank
    EXPECT_TRUE(difference[1].starts_with("0,0.000000,5.0,,")) << difference[1];
    EXPECT_TRUE(difference[2].starts_with("1,0.033333,-2.0,,")) << difference[2];
    EXPECT_TRUE(difference[3].starts_with("2,0.066667,10.0,,")) << difference[3];

    // Off: the lines' files only.
    difference_box(panel).setChecked(false);
    export_csvs(panel);
    ASSERT_TRUE(panel.wait_until_file_writes_idle(std::chrono::seconds(10)));
    EXPECT_TRUE(std::filesystem::exists(folder / "a_2.csv"));
    EXPECT_FALSE(std::filesystem::exists(folder / "a_minus_b_2.csv"));
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
    // graph up to date, and comes back when the user asks for the runs.
    VideoColors colors;
    GraphPanel panel(&colors);
    panel.add_run({shared(vmaf_result("a.mkv", {80, 80})), "a", "row a", false});
    panel.add_run({shared(vmaf_result("b.mkv", {85, 85})), "b", "row b", false});
    panel.add_run({shared(vmaf_result("a.mkv", {99, 99})), "a", "row a", false});
    ASSERT_EQ(entries(panel).size(), 2u);
    EXPECT_EQ(entries(panel).at(id_of(panel, "a")).means.at("vmaf"), 99.0);

    panel.remove_runs({id_of(panel, "a")});
    panel.add_run({shared(vmaf_result("a.mkv", {99, 99})), "a", "row a", false});
    EXPECT_EQ(entries(panel).size(), 1u);
    panel.add_run({shared(vmaf_result("a.mkv", {99, 99})), "a", "row a", true});
    EXPECT_EQ(entries(panel).size(), 2u);
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
    panel.dialogs.save_file = [&out_text](const QString&, const QString&, const QString&) { return out_text; };
    export_png(panel);
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
}  // namespace
}  // namespace vml::app
