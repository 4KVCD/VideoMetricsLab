// The Bitrate Viewer tab: the Python app's tests of the same behaviour
// (tests/test_bitrate_panel.py, test_series_colors.py), for what is ported
// (bitrate_panel.h).
#include "bitrate_panel.h"

#include <QApplication>
#include <QCheckBox>
#include <QDeadlineTimer>
#include <QImage>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QMenu>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QThread>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "chart.h"
#include "frame_controls.h"
#include "series_colors.h"
#include "test_videos.h"
#include "frame_types.h"
#include "vml/analysis_request.h"
#include "vml/platform/system.h"

namespace vml::app {

// The panel's parts, for its tests.
class BitratePanelTest : public ::testing::Test {
protected:
    // A file read: its details and packets, as the worker gives them.
    static void analyzed(BitratePanel& panel, const std::filesystem::path& path,
                         std::shared_ptr<const BitrateData> data,
                         std::shared_ptr<const std::vector<std::uint8_t>> frame_types = nullptr) {
        VideoInfo info;
        info.path = path;
        info.width = 1920;
        info.height = 1080;
        info.fps = 2.0;
        info.duration = 2.0;
        info.nb_frames = 4;
        info.codec_name = "h264";
        info.pix_fmt = "yuv420p";
        panel.on_analyzed({QString::fromStdString(platform::path_text(path)), info, std::move(data),
                          std::move(frame_types), file_identity(path)});
    }
    static void failed(BitratePanel& panel, const std::filesystem::path& path, const QString& error) {
        panel.on_failed(QString::fromStdString(platform::path_text(path)), error);
    }
    static void hover(BitratePanel& panel, double time) { panel.on_hovered(time); }
    static bool idle(const BitratePanel& panel) { return panel.worker_ == nullptr; }
    static const BitrateEntry& entry(const BitratePanel& panel, std::size_t row) {
        return panel.entries_.at(row).second;
    }
    static QColor line_color(const BitratePanel& panel, std::size_t row) {
        return panel.chart->series_.at(entry(panel, row).series).color;
    }
    static void zoom_to(BitratePanel& panel, std::pair<double, double> view) {
        panel.chart->view_ = ChartWidget::View{view.first, view.second - view.first};
    }
    static QString cell(const BitratePanel& panel, int row, int column) {
        return panel.table->item(row, column)->text();
    }
    static std::vector<float> band_values(const BitratePanel& panel) {
        const std::span<const float> values = panel.chart->band_->series.values;
        return {values.begin(), values.end()};
    }
    static QString band_label(const BitratePanel& panel) { return panel.chart->band_->label; }
    static std::size_t marks_drawn(const BitratePanel& panel, std::size_t row) {
        return panel.chart->series_.at(entry(panel, row).series).marks.size();
    }
    static QCheckBox* frame_types_box(const BitratePanel& panel) { return panel.frame_types_box; }
    static void mark_reference(BitratePanel& panel, std::size_t row) { panel.entries_.at(row).second.reference = true; }
    static QString export_title(const BitratePanel& panel) { return panel.export_title(); }
    static ExportTable export_table(const BitratePanel& panel) { return panel.export_table(); }
};

namespace {

enum Column { col_use, col_file, col_media, col_duration, col_frames, col_average, col_minimum, col_maximum, col_status };

// Four packets of half a second: 1,500 bytes in the first second, 2,500 in
// the second, a keyframe at the start of each.
std::shared_ptr<const BitrateData> two_seconds(const std::filesystem::path& path) {
    return std::make_shared<const BitrateData>(
        bitrate_data(path, {0.0, 0.5, 1.0, 1.5}, {0.5, 0.5, 0.5, 0.5}, {1000, 500, 2000, 500}, {true, false, true, false}));
}

std::unique_ptr<BitratePanel> make_panel(VideoColors& colors, std::string ffprobe = "ffprobe") {
    return std::make_unique<BitratePanel>([ffprobe] { return ffprobe; }, &colors);
}

}  // namespace

// A file is listed once however it is named, and works without a run; once
// read, its row has its figures, each graph its line, and the line's values
// show under the pointer; the graph is exported, and cannot be once no line
// is shown.
TEST_F(BitratePanelTest, AFileReadFillsItsRowAndEachGraph) {
    VideoColors colors;
    const auto panel = make_panel(colors);
    const test::Folder folder;
    const std::filesystem::path path = folder / "encode.mkv";
    panel->add_files({path, folder / "." / "encode.mkv"});
    ASSERT_EQ(panel->table->rowCount(), 1);
    EXPECT_EQ(cell(*panel, 0, col_file).toStdString(), "encode.mkv");
    EXPECT_EQ(cell(*panel, 0, col_status).toStdString(), "Ready to analyze");
    EXPECT_EQ(panel->table->editTriggers(), QAbstractItemView::NoEditTriggers);
    EXPECT_TRUE(panel->analyze_button->isEnabled());
    EXPECT_EQ(panel->analyze_button->text().toStdString(), "Calculate bitrate");
    EXPECT_FALSE(panel->export_button->isEnabled());
    // Export is in the footer under the graph, at its right end, as on
    // Metric Graphs (docs/ARCHITECTURE.md, "Exports"): Graph as PNG...
    EXPECT_EQ(panel->export_button->text().toStdString(), "Export");
    ASSERT_NE(panel->export_button->menu(), nullptr);
    EXPECT_EQ(panel->export_button->menu()->actions().front()->text().toStdString(), "Graph as PNG...");
    // (In a toolbar of its own: its place is taken in the panel's terms.)
    panel->resize(1200, 800);
    panel->layout()->activate();
    panel->export_button->parentWidget()->layout()->activate();
    const auto in_panel = [&panel](const QWidget* widget) {
        return QRect(widget->mapTo(panel.get(), QPoint(0, 0)), widget->size());
    };
    EXPECT_GT(in_panel(panel->export_button).top(), in_panel(panel->chart).bottom());
    EXPECT_GT(in_panel(panel->export_button).left(), in_panel(panel->frame_controls).right());
    EXPECT_EQ(panel->export_button->parentWidget()->property("vmlRole").toString().toStdString(), "toolbar");

    analyzed(*panel, path, two_seconds(path));
    EXPECT_EQ(cell(*panel, 0, col_duration).toStdString(), "0:00:02.00");
    EXPECT_EQ(cell(*panel, 0, col_frames).toStdString(), "4");
    EXPECT_EQ(cell(*panel, 0, col_average).toStdString(), "16 kb/s");
    EXPECT_EQ(cell(*panel, 0, col_minimum).toStdString(), "12 kb/s");
    EXPECT_EQ(cell(*panel, 0, col_maximum).toStdString(), "20 kb/s");
    EXPECT_EQ(cell(*panel, 0, col_status).toStdString(), "Complete");
    EXPECT_TRUE(panel->chart->has_data());
    EXPECT_EQ(panel->chart->y_axis_label().toStdString(), "Video bitrate (kb/s)");
    panel->frame_radio->click();
    EXPECT_EQ(panel->chart->y_axis_label().toStdString(), "Frame size (kbit)");
    hover(*panel, 0.6);  // nearest the frame at 0.5 s (4 kbit), not the next at 1 s
    EXPECT_EQ(panel->hover_label->text().toStdString(), "Time: 0:00:00.600   Frame 1\nencode.mkv: 4.00 kbit");
    panel->gop_radio->click();
    EXPECT_EQ(panel->chart->y_axis_label().toStdString(), "GOP bitrate (kb/s)");
    hover(*panel, 1.2);
    EXPECT_EQ(panel->hover_label->text().toStdString(), "Time: 0:00:01.200   Frame 2\nencode.mkv: 20.00 kb/s");

    const std::filesystem::path png = folder / "plot.png";
    QString suggested;
    panel->dialogs.save_file = [&png, &suggested](const QString&, const QString& name, const QString&) {
        suggested = name;
        return QString::fromStdString(platform::path_text(png));
    };
    panel->export_button->menu()->actions().front()->trigger();
    EXPECT_FALSE(QImage(QString::fromStdString(platform::path_text(png))).isNull());
    EXPECT_EQ(suggested.toStdString(), "encode_bitrate.png");  // named after the one video shown

    panel->table->item(0, col_use)->setCheckState(Qt::Unchecked);
    EXPECT_FALSE(panel->chart->has_data());
    EXPECT_FALSE(panel->export_button->isEnabled());
}

// Beside the time, each line's frame in their order, as videos differ in frame
// rate and start: on the frame graph the point whose value is shown, on a
// stepped graph the frame on screen, from the file's start or its first frame.
TEST_F(BitratePanelTest, TheFramesUnderThePointerAreBesideTheTime) {
    VideoColors colors;
    const auto panel = make_panel(colors);
    const test::Folder folder;
    const std::filesystem::path first = folder / "first.mkv";
    const std::filesystem::path late = folder / "late.mkv";  // 4 frames a second, from 10 s
    panel->add_files({first, late});
    analyzed(*panel, first, two_seconds(first));
    analyzed(*panel, late,
             std::make_shared<const BitrateData>(bitrate_data(late, {10.0, 10.25, 10.5, 10.75}, {0.25, 0.25, 0.25, 0.25},
                                                              {500, 500, 500, 500}, {true, false, true, false})));

    hover(*panel, 0.9);
    EXPECT_EQ(panel->hover_label->text().toStdString(),
              "Time: 0:00:00.900   Frame 1, 3\nfirst.mkv: 12.00 kb/s\nlate.mkv: 16.00 kb/s");
    panel->frame_radio->click();
    hover(*panel, 0.9);  // nearer the frames at 1 s and 0.75 s
    EXPECT_EQ(panel->hover_label->text().toStdString(),
              "Time: 0:00:00.900   Frame 2, 3\nfirst.mkv: 16.00 kbit\nlate.mkv: 4.00 kbit");

    panel->adjust_start_checkbox->setChecked(false);
    hover(*panel, 10.3);  // past the first video's end: its last frame, as its last value
    EXPECT_EQ(panel->hover_label->text().toStdString(),
              "Time: 0:00:10.300   Frame 3, 1\nfirst.mkv: 4.00 kbit\nlate.mkv: 4.00 kbit");
    panel->second_radio->click();
    hover(*panel, 10.6);
    EXPECT_EQ(panel->hover_label->text().toStdString(),
              "Time: 0:00:10.600   Frame 3, 2\nfirst.mkv: 20.00 kb/s\nlate.mkv: 16.00 kb/s");
}

// A frame range, in Metric Graphs' boxes: the table gives those frames'
// figures (a shorter video's, of those it has) and the graph shades them,
// until it is cleared. Go to frame shows each line at that frame, and says a
// shorter video lacks it.
TEST_F(BitratePanelTest, AFrameRangesFiguresAndGoToFrame) {
    VideoColors colors;
    const auto panel = make_panel(colors);
    const test::Folder folder;
    const std::filesystem::path first = folder / "first.mkv";
    const std::filesystem::path shorter = folder / "short.mkv";  // its two frames 1,000 bytes each
    panel->add_files({first, shorter});
    analyzed(*panel, first, two_seconds(first));
    analyzed(*panel, shorter,
             std::make_shared<const BitrateData>(
                 bitrate_data(shorter, {0.0, 0.5}, {0.5, 0.5}, {1000, 1000}, {true, false})));

    panel->frame_controls->range_start_spin->setValue(1);
    panel->frame_controls->range_end_spin->setValue(2);
    EXPECT_EQ(cell(*panel, 0, col_average).toStdString(), "20 kb/s");  // 2,500 bytes from 0.5 to 1.5 s
    EXPECT_EQ(cell(*panel, 0, col_minimum).toStdString(), "20 kb/s");
    EXPECT_EQ(cell(*panel, 1, col_average).toStdString(), "16 kb/s");  // its frame 1 alone
    EXPECT_EQ(panel->range_label->text().toStdString(), "Statistics of frames 1 to 2.");
    EXPECT_EQ(panel->chart->highlight(), (std::optional<std::pair<double, double>>{{0.5, 1.5}}));
    panel->frame_controls->clear_range_button->click();
    EXPECT_EQ(cell(*panel, 0, col_average).toStdString(), "16 kb/s");
    EXPECT_EQ(cell(*panel, 0, col_minimum).toStdString(), "12 kb/s");
    EXPECT_TRUE(panel->range_label->text().isEmpty());
    EXPECT_FALSE(panel->chart->highlight());

    panel->frame_controls->frame_spin->setValue(3);
    EXPECT_EQ(panel->hover_label->text().toStdString(),
              "Time: 0:00:01.500   Frame 3\nfirst.mkv: 20.00 kb/s\nshort.mkv  frame      3   not in this run");

    // The same in times, to the millisecond, as on Metric Graphs: the first
    // line's frames' times.
    FrameControls& controls = *panel->frame_controls;
    EXPECT_FALSE(controls.time_edit->isHidden());
    EXPECT_FALSE(controls.range_start_time->isHidden());
    EXPECT_EQ(controls.time_edit->text().toStdString(), "0:00:01.500");
    const auto type_time = [](QLineEdit& box, const QString& text) {
        box.setText(text);
        box.setModified(true);
        emit box.returnPressed();
        emit box.editingFinished();
    };
    type_time(*controls.time_edit, "0:00:00.999");  // frame 1 is still shown
    EXPECT_EQ(controls.frame_spin->value(), 1);
    EXPECT_TRUE(panel->hover_label->text().startsWith("Time: 0:00:00.500   Frame 1"));
    type_time(*controls.range_start_time, "0.5");
    type_time(*controls.range_end_time, "0:00:01.500");
    EXPECT_EQ(controls.selected_frames(), (std::optional<std::pair<int, int>>{{1, 3}}));
    EXPECT_EQ(panel->range_label->text().toStdString(), "Statistics of frames 1 to 3.");
    controls.range_start_spin->setValue(2);
    EXPECT_EQ(controls.range_start_time->text().toStdString(), "0:00:01.000");
}

// Issue #140: with a shorter video drawn first, a time past its end went to
// its last frame, and a range of such times was cleared: only the first
// line's times were looked at, while the frame boxes reach the longest
// line's last frame and show its times.
TEST_F(BitratePanelTest, TimesPastTheFirstVideosEndAreTheLongerOnes) {
    VideoColors colors;
    const auto panel = make_panel(colors);
    const std::filesystem::path shorter("short.mkv");
    const std::filesystem::path longer("long.mkv");
    panel->add_files({shorter, longer});
    analyzed(*panel, shorter,
             std::make_shared<const BitrateData>(
                 bitrate_data(shorter, {0.0, 0.5}, {0.5, 0.5}, {1000, 1000}, {true, false})));
    analyzed(*panel, longer, two_seconds(longer));
    FrameControls& controls = *panel->frame_controls;
    const auto type_time = [](QLineEdit& box, const QString& text) {
        box.setText(text);
        box.setModified(true);
        emit box.returnPressed();
        emit box.editingFinished();
    };
    controls.frame_spin->setValue(3);
    EXPECT_EQ(controls.time_edit->text().toStdString(), "0:00:01.500");
    controls.frame_spin->setValue(0);
    type_time(*controls.time_edit, "0:00:01.500");
    EXPECT_EQ(controls.frame_spin->value(), 3);
    type_time(*controls.time_edit, "0:00:01.200");  // within the longer one's frame 2
    EXPECT_EQ(controls.frame_spin->value(), 2);
    type_time(*controls.time_edit, "0:00:09");  // past both: the last frame there is
    EXPECT_EQ(controls.frame_spin->value(), 3);
    type_time(*controls.range_start_time, "0:00:01");
    type_time(*controls.range_end_time, "0:00:01.500");
    EXPECT_EQ(controls.selected_frames(), (std::optional<std::pair<int, int>>{{2, 3}}));
    type_time(*controls.range_start_time, "0:00:00.500");  // the shorter one's last frame, to the longer one's
    EXPECT_EQ(controls.selected_frames(), (std::optional<std::pair<int, int>>{{1, 3}}));
    // Its band runs from the shorter one's frame 1 to the end of the longer
    // one's frame 3 (it stopped at the shorter one's end, 1 s).
    EXPECT_EQ(panel->chart->highlight(), (std::optional<std::pair<double, double>>{{0.5, 2.0}}));
}

// A frame's type shows under the pointer in the frame graph; its colours only
// once ticked, and only there.
TEST_F(BitratePanelTest, FrameTypesInTheFrameGraph) {
    VideoColors colors;
    const auto panel = make_panel(colors);
    const test::Folder folder;
    const std::filesystem::path path = folder / "encode.mkv";
    panel->add_files({path});
    analyzed(*panel, path, two_seconds(path),
             std::make_shared<const std::vector<std::uint8_t>>(
                 std::vector<std::uint8_t>{frame_type_i, frame_type_b, frame_type_i, frame_type_unknown}));
    QCheckBox* const box = frame_types_box(*panel);
    EXPECT_FALSE(box->isEnabled());
    hover(*panel, 0.6);
    EXPECT_EQ(panel->hover_label->text().toStdString(), "Time: 0:00:00.600   Frame 1\nencode.mkv: 12.00 kb/s");

    panel->frame_radio->click();
    EXPECT_TRUE(box->isEnabled());
    EXPECT_EQ(marks_drawn(*panel, 0), 0u);
    hover(*panel, 0.6);
    EXPECT_EQ(panel->hover_label->text().toStdString(), "Time: 0:00:00.600   Frame 1\nencode.mkv: 4.00 kbit (B)");
    hover(*panel, 1.6);  // a type not known adds nothing
    EXPECT_EQ(panel->hover_label->text().toStdString(), "Time: 0:00:01.600   Frame 3\nencode.mkv: 4.00 kbit");

    box->click();
    EXPECT_EQ(marks_drawn(*panel, 0), 4u);
    panel->second_radio->click();
    EXPECT_FALSE(box->isEnabled());
    EXPECT_EQ(marks_drawn(*panel, 0), 0u);
}

// The exported image says what its lines are, as Metric Graphs' does: the
// graph and the frame range in its title, and each line drawn named (the
// reference marked) beside its colour, with the table's figures. It was the
// bare chart.
TEST_F(BitratePanelTest, TheExportedImageNamesItsGraphAndLines) {
    VideoColors colors;
    const auto panel = make_panel(colors);
    const test::Folder folder;
    const std::filesystem::path source = folder / "source.mkv";
    const std::filesystem::path encode = folder / "encode.mkv";
    panel->add_files({source, encode});
    analyzed(*panel, source, two_seconds(source));
    analyzed(*panel, encode, two_seconds(encode));
    mark_reference(*panel, 0);

    EXPECT_EQ(export_title(*panel).toStdString(), "Video bitrate vs time — Second based");
    {
        const auto [headers, rows] = export_table(*panel);
        ASSERT_EQ(headers.size(), 6u);
        ASSERT_EQ(rows.size(), 2u);
        EXPECT_EQ(std::get<0>(rows[0]).toStdString(), "source.mkv (the reference)");
        const auto& [label, color, cells] = rows[1];
        EXPECT_EQ(label.toStdString(), "encode.mkv");
        EXPECT_EQ(color, entry(*panel, 1).color);
        ASSERT_EQ(cells.size(), 5u);
        EXPECT_EQ(cells[0].toStdString(), "0:00:02.00");
        EXPECT_EQ(cells[1].toStdString(), "4");
        EXPECT_EQ(cells[2].toStdString(), "16 kb/s");
        EXPECT_EQ(cells[3].toStdString(), "12 kb/s");
        EXPECT_EQ(cells[4].toStdString(), "20 kb/s");
    }
    // Around the chart, not instead of it.
    EXPECT_GT(panel->render_export_image().height(), panel->chart->render_to_pixmap().height());
    // In the colours Settings asks for: a chosen background around the
    // chart and in it.
    panel->export_colours = [] { return theme::export_chart_colours("light", "#203040", "#e0e0e0"); };
    EXPECT_EQ(panel->render_export_image().toImage().pixelColor(1, 1), QColor(0x20, 0x30, 0x40));
    EXPECT_EQ(panel->chart->render_to_pixmap(panel->export_colours()).toImage().pixelColor(1, 1),
              QColor(0x20, 0x30, 0x40));
    panel->export_colours = nullptr;

    // A frame range: in the title, and the figures are of it; a line not
    // drawn is not listed.
    panel->gop_radio->click();
    panel->frame_controls->range_start_spin->setValue(1);
    panel->frame_controls->range_end_spin->setValue(2);
    panel->table->item(0, col_use)->setCheckState(Qt::Unchecked);
    EXPECT_EQ(export_title(*panel).toStdString(), "Video bitrate vs time — GOP based — frames 1 to 2");
    const auto rows = export_table(*panel).second;
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(std::get<0>(rows[0]).toStdString(), "encode.mkv");
    EXPECT_EQ(std::get<2>(rows[0])[2].toStdString(), "20 kb/s");  // 2,500 bytes from 0.5 to 1.5 s
}

// Why a file failed is on its own row: once a row above it is removed, the
// row that takes its place does not show it. (The Python app left the
// reason on the row.)
TEST_F(BitratePanelTest, AFailedFilesReasonStaysWithIt) {
    VideoColors colors;
    const auto panel = make_panel(colors);
    const test::Folder folder;
    panel->add_files({folder / "broken.mkv", folder / "fine.mkv"});
    failed(*panel, folder / "broken.mkv", "Invalid data found when processing input");
    EXPECT_EQ(cell(*panel, 0, col_status).toStdString(), "Failed");
    EXPECT_EQ(panel->table->item(0, col_status)->toolTip().toStdString(), "Invalid data found when processing input");
    EXPECT_EQ(panel->status_label->text().toStdString(),
              "Could not analyze broken.mkv: Invalid data found when processing input");

    panel->table->selectRow(0);
    panel->remove_button->click();
    ASSERT_EQ(panel->table->rowCount(), 1);
    EXPECT_EQ(cell(*panel, 0, col_file).toStdString(), "fine.mkv");
    EXPECT_EQ(panel->table->item(0, col_status)->toolTip().toStdString(), "");
}

// A video encoded again under the same name, as a finished run hands it over:
// the old figures go, and a read that then fails leaves none behind either.
TEST_F(BitratePanelTest, AFileChangedSinceItWasReadIsNotShownAsItWas) {
    VideoColors colors;
    const auto panel = make_panel(colors, "no-such-ffprobe");
    const test::Folder folder;
    const std::filesystem::path path = folder / "encode.mkv";
    std::ofstream(path) << "first encode";
    panel->add_files({path});
    analyzed(*panel, path, two_seconds(path));
    panel->add_files({path}, {}, true);  // the same file: read once
    EXPECT_TRUE(entry(*panel, 0).data);
    EXPECT_TRUE(idle(*panel));

    std::ofstream(path) << "the second encode, of another size";
    panel->add_files({path}, {}, true);
    EXPECT_FALSE(entry(*panel, 0).data);

    analyzed(*panel, path, two_seconds(path));
    failed(*panel, path, "gone");
    EXPECT_FALSE(entry(*panel, 0).data);
    EXPECT_EQ(cell(*panel, 0, col_average).toStdString(), "");
}

// #205: a graph export asks the tab to read only the files it wants read. A
// file read before and encoded again under its name since counted as read
// (state_of: complete), so the export drew the old encode's bitrate beside
// the new one's scores. It is wanted again; one that failed is not.
TEST_F(BitratePanelTest, AnExportWantsAFileReplacedSinceItWasReadReadAgain) {
    VideoColors colors;
    const auto panel = make_panel(colors, "no-such-ffprobe");
    const test::Folder folder;
    const std::filesystem::path path = folder / "encode.mkv";
    const std::filesystem::path broken = folder / "broken.mkv";
    EXPECT_TRUE(panel->wants_reading(path));  // not listed
    std::ofstream(path) << "first encode";
    std::ofstream(broken) << "not a video";
    panel->add_files({path, broken});
    EXPECT_TRUE(panel->wants_reading(path));  // listed, not read
    analyzed(*panel, path, two_seconds(path));
    failed(*panel, broken, "unreadable");
    EXPECT_FALSE(panel->wants_reading(path));
    EXPECT_FALSE(panel->wants_reading(broken));

    std::ofstream(path) << "the second encode, of another size";
    EXPECT_EQ(panel->state_of(path), BitrateState::complete);  // what the export went by
    EXPECT_TRUE(panel->wants_reading(path));
    VideoInfo info;
    info.path = path;
    panel->add_and_analyze({info});
    EXPECT_FALSE(entry(*panel, 0).data);  // the old encode's reading gone
}

// Issue #134: a saved comparison's graph export drew whatever the tab had
// read at its file's path, a newer encode under the same name say. The
// export's panel takes the comparison's saved bitrate for such a file, and
// draws nothing for one with none saved.
TEST_F(BitratePanelTest, AnExportDrawsTheSavedBitrateOfAFileReplacedSince) {
    VideoColors colors;
    const auto tab = make_panel(colors);
    const std::filesystem::path reference("ref.mkv");
    const std::filesystem::path encode("encode.mkv");
    const std::filesystem::path other("other.mkv");
    tab->add_files({reference, encode, other});
    analyzed(*tab, reference, two_seconds(reference));
    analyzed(*tab, encode, two_seconds(encode));  // the new encode's
    analyzed(*tab, other, two_seconds(other));
    SavedBitrate saved;
    saved.info.path = encode;
    saved.data = bitrate_data(encode, {0.0, 0.5}, {0.5, 0.5}, {7000, 3000}, {true, false});
    const auto exported = make_panel(colors);
    exported->adopt_read(*tab, {reference, encode, other}, {{encode, saved}, {other, std::nullopt}});
    ASSERT_EQ(cell(*exported, 1, col_file).toStdString(), "encode.mkv");
    EXPECT_TRUE(entry(*exported, 0).reference);
    EXPECT_EQ(entry(*exported, 0).data, entry(*tab, 0).data);
    EXPECT_EQ(entry(*exported, 1).data->sizes, saved.data.sizes);
    EXPECT_FALSE(entry(*exported, 1).reference);
    EXPECT_EQ(exported->table->rowCount(), 2);  // none for the one without saved bitrate
}

// Two comparisons of one file name exported together (an encode, and a new
// encode over it): the older's saved bitrate, under the export's key for it
// (graph_bundle.h's bitrate_key), is shown by its file's name, numbered as
// the second of it, never by the key; in a colour of its own, the newer's
// line beside it being another.
TEST_F(BitratePanelTest, AnExportsSavedBitrateIsShownByItsFilesName) {
    VideoColors colors;
    const auto tab = make_panel(colors);
    const std::filesystem::path reference("ref.mkv");
    const std::filesystem::path encode("encode.mkv");
    tab->add_files({reference, encode});
    analyzed(*tab, reference, two_seconds(reference));
    analyzed(*tab, encode, two_seconds(encode));  // the newer encode's
    std::filesystem::path key = encode;
    key += "|older";
    SavedBitrate saved;
    saved.info.path = encode;
    saved.data = bitrate_data(encode, {0.0, 0.5}, {0.5, 0.5}, {7000, 3000}, {true, false});
    const auto exported = make_panel(colors);
    exported->adopt_read(*tab, {reference, encode, key}, {{key, saved}});
    ASSERT_EQ(exported->table->rowCount(), 3);
    EXPECT_EQ(cell(*exported, 1, col_file).toStdString(), "encode.mkv");
    EXPECT_EQ(cell(*exported, 2, col_file).toStdString(), "encode.mkv (2)");
    EXPECT_EQ(entry(*exported, 2).path, encode);
    EXPECT_EQ(entry(*exported, 2).data->sizes, saved.data.sizes);
    EXPECT_EQ(entry(*exported, 2).color, colors.for_path(key));
    EXPECT_NE(entry(*exported, 2).color, entry(*exported, 1).color);
    // Alone with the reference (its own graph): its file's name.
    const auto alone = make_panel(colors);
    alone->adopt_read(*tab, {reference, key}, {{key, saved}});
    ASSERT_EQ(alone->table->rowCount(), 2);
    EXPECT_EQ(cell(*alone, 1, col_file).toStdString(), "encode.mkv");
}

// A video has the session's colour, whatever order the tabs met it in; a
// colour picked repaints its square and its line, the graph's zoom kept.
TEST_F(BitratePanelTest, ALinesColourIsTheSessionsForItsVideo) {
    VideoColors colors;
    const test::Folder folder;
    const std::filesystem::path first = folder / "first.mkv";
    const std::filesystem::path second = folder / "second.mkv";
    const QColor second_colour = colors.for_path(second);  // met first elsewhere (Metric Graphs)
    const auto panel = make_panel(colors);
    panel->add_files({first, second});
    analyzed(*panel, first, two_seconds(first));
    analyzed(*panel, second, two_seconds(second));
    EXPECT_EQ(entry(*panel, 1).color, second_colour);
    EXPECT_EQ(entry(*panel, 0).color, colors.for_path(first));
    EXPECT_NE(entry(*panel, 0).color, second_colour);
    EXPECT_EQ(line_color(*panel, 1), second_colour);

    zoom_to(*panel, {0.5, 1.5});
    colors.set_color(first, QColor("#123456"));
    EXPECT_EQ(entry(*panel, 0).color, QColor("#123456"));
    EXPECT_EQ(line_color(*panel, 0), QColor("#123456"));
    const QPixmap swatch = panel->table->item(0, col_use)->data(Qt::DecorationRole).value<QPixmap>();
    EXPECT_EQ(swatch.toImage().pixelColor(0, 0), QColor("#123456"));
    EXPECT_EQ(panel->chart->x_range(), (std::pair<double, double>{0.5, 1.5}));
    EXPECT_EQ(line_color(*panel, 1), second_colour);
}

// The graph's zoom stays while lines come and go: a file ticked off or on,
// read again or removed (the zoom reset at each, as the lines are drawn
// again). Another graph shows all of the time range again.
TEST_F(BitratePanelTest, ALineTickedReadOrRemovedKeepsTheZoom) {
    VideoColors colors;
    const test::Folder folder;
    const std::filesystem::path first = folder / "first.mkv";
    const std::filesystem::path second = folder / "second.mkv";
    const auto panel = make_panel(colors);
    panel->add_files({first, second});
    analyzed(*panel, first, two_seconds(first));
    analyzed(*panel, second, two_seconds(second));
    const std::pair<double, double> zoomed{0.5, 1.5};
    zoom_to(*panel, zoomed);

    panel->table->item(0, col_use)->setCheckState(Qt::Unchecked);
    EXPECT_EQ(panel->chart->x_range(), zoomed);
    panel->table->item(0, col_use)->setCheckState(Qt::Checked);
    EXPECT_EQ(panel->chart->x_range(), zoomed);
    analyzed(*panel, second, two_seconds(second));
    EXPECT_EQ(panel->chart->x_range(), zoomed);
    panel->table->selectRow(0);
    panel->remove_button->click();
    ASSERT_EQ(panel->table->rowCount(), 1);
    EXPECT_EQ(panel->chart->x_range(), zoomed);

    panel->frame_radio->click();
    EXPECT_NE(panel->chart->x_range(), zoomed);
}

// The Difference box is usable while exactly two lines are drawn; on, the
// band is the first line minus the second, in the lines' unit (Mb/s too),
// over time or frame by frame, with its value under the pointer. A third
// line takes the band away, and two lines with no time in common have none.
TEST_F(BitratePanelTest, DrawsTheDifferenceOfTwoLines) {
    VideoColors colors;
    const test::Folder folder;
    const std::filesystem::path first = folder / "first.mkv";
    const std::filesystem::path second = folder / "second.mkv";
    const std::filesystem::path third = folder / "third.mkv";
    const auto panel = make_panel(colors);
    panel->add_files({first, second, third});
    analyzed(*panel, first, two_seconds(first));  // 12 then 20 kb/s; frames of 8, 4, 16 and 4 kbit
    analyzed(*panel, second,
             std::make_shared<const BitrateData>(bitrate_data(second, {0.0, 0.5, 1.0, 1.5}, {0.5, 0.5, 0.5, 0.5},
                                                              {500, 500, 500, 500}, {true, false, true, false})));
    EXPECT_TRUE(panel->difference_box->isEnabled());
    EXPECT_FALSE(panel->chart->has_band());

    panel->difference_box->setChecked(true);  // 12 - 8 kb/s, then 20 - 8
    ASSERT_TRUE(panel->chart->has_band());
    EXPECT_EQ(band_values(*panel), (std::vector<float>{4, 4, 12, 12}));
    EXPECT_EQ(band_label(*panel).toStdString(), "first.mkv − second.mkv");
    hover(*panel, 1.2);
    EXPECT_EQ(panel->hover_label->text().toStdString(),
              "Time: 0:00:01.200   Frame 2\nfirst.mkv: 20.00 kb/s\nsecond.mkv: 8.00 kb/s\n"
              "Δ (first.mkv − second.mkv) = 12.00 kb/s");
    panel->frame_radio->click();
    EXPECT_EQ(band_values(*panel), (std::vector<float>{4, 0, 12, 0}));

    // 3,200 kb/s, 10 s into its file: the graph in Mb/s
    analyzed(*panel, third,
             std::make_shared<const BitrateData>(bitrate_data(third, {10.0, 10.5, 11.0, 11.5}, {0.5, 0.5, 0.5, 0.5},
                                                              {200'000, 200'000, 200'000, 200'000},
                                                              {true, false, true, false})));
    EXPECT_FALSE(panel->difference_box->isEnabled());
    EXPECT_FALSE(panel->chart->has_band());
    panel->table->item(0, col_use)->setCheckState(Qt::Unchecked);
    panel->second_radio->click();
    ASSERT_TRUE(panel->chart->has_band());
    const std::vector<float> mega = band_values(*panel);  // second minus third, both from their start
    ASSERT_EQ(mega.size(), 4U);
    EXPECT_FLOAT_EQ(mega[0], -3.192F);
    panel->adjust_start_checkbox->setChecked(false);  // 0-2 s in one file, 10-12 s in the other
    EXPECT_TRUE(panel->difference_box->isEnabled());
    EXPECT_FALSE(panel->chart->has_band());
}

}  // namespace vml::app
