// Metric Graphs, held to the Python app's own panel (reference/graph.json):
// the table and the readouts for runs of every kind of score; and what the
// graph must keep: XPSNR's infinity in the scores while it is drawn at a
// stand-in, CVVDP's seconds at their middles, one line per run, a removed
// line kept away, an image to export.
#include "graph_panel.h"

#include <QImage>
#include <QLabel>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <memory>
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
    // What a page's chart draws for a line.
    static const ChartSeries& drawn(MetricPage& page, int series_id) { return page.chart_->series_.at(series_id); }
    static void export_png(GraphPanel& panel) { panel.on_export_png(); }
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
