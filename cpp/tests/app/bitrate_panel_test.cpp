// The Bitrate Viewer tab: the Python app's tests of the same behaviour
// (tests/test_bitrate_panel.py, test_series_colors.py), for what is ported
// (bitrate_panel.h).
#include "bitrate_panel.h"

#include <QApplication>
#include <QCheckBox>
#include <QDeadlineTimer>
#include <QImage>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QTableWidget>
#include <QThread>

#include <gtest/gtest.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "chart.h"
#include "series_colors.h"
#include "test_videos.h"
#include "vml/platform/system.h"

namespace vml::app {

// The panel's parts, for its tests.
class BitratePanelTest : public ::testing::Test {
protected:
    // A file read: its details and packets, as the worker gives them.
    static void analyzed(BitratePanel& panel, const std::filesystem::path& path,
                         std::shared_ptr<const BitrateData> data) {
        VideoInfo info;
        info.path = path;
        info.width = 1920;
        info.height = 1080;
        info.fps = 2.0;
        info.duration = 2.0;
        info.nb_frames = 4;
        info.codec_name = "h264";
        info.pix_fmt = "yuv420p";
        panel.on_analyzed({QString::fromStdString(platform::path_text(path)), info, std::move(data)});
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
    static void zoom_to(BitratePanel& panel, std::pair<double, double> view) { panel.chart->view_x_ = view; }
    static QString cell(const BitratePanel& panel, int row, int column) {
        return panel.table->item(row, column)->text();
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
    EXPECT_EQ(panel->hover_label->text().toStdString(), "Time: 0:00:00.600\nencode.mkv: 4.00 kbit");
    panel->gop_radio->click();
    EXPECT_EQ(panel->chart->y_axis_label().toStdString(), "GOP bitrate (kb/s)");
    hover(*panel, 1.2);
    EXPECT_EQ(panel->hover_label->text().toStdString(), "Time: 0:00:01.200\nencode.mkv: 20.00 kb/s");

    const std::filesystem::path png = folder / "plot.png";
    panel->dialogs.save_file = [&png](const QString&, const QString&, const QString&) {
        return QString::fromStdString(platform::path_text(png));
    };
    panel->export_button->click();
    EXPECT_FALSE(QImage(QString::fromStdString(platform::path_text(png))).isNull());

    panel->table->item(0, col_use)->setCheckState(Qt::Unchecked);
    EXPECT_FALSE(panel->chart->has_data());
    EXPECT_FALSE(panel->export_button->isEnabled());
}

// Calculate asks before reading again the files already read, and with No
// reads only the others; a file added meanwhile (a run finishing) waits its
// turn. Stop then ends ffprobe at once, marks the files not read as
// stopped, and keeps what was read. The stand-in ffprobe never answers:
// without Stop ending it, the scan would wait 60 s for it.
TEST_F(BitratePanelTest, StopEndsTheScanAndKeepsWhatWasRead) {
    const test::Folder folder;
    const std::filesystem::path log = folder / "log";
    std::filesystem::create_directory(log);
    const test::ScopedVariable tool("VML_TEST_TOOL", "hang");
    const test::ScopedVariable logged_to("VML_TEST_TOOL_LOG", platform::path_text(log));
    VideoColors colors;
    const auto panel = make_panel(colors, VML_TEST_CHILD);
    const std::vector<std::filesystem::path> paths{folder / "0.mkv", folder / "1.mkv", folder / "2.mkv"};
    panel->add_files({paths[0], paths[1]});
    analyzed(*panel, paths[0], two_seconds(paths[0]));
    const BitrateData* kept = entry(*panel, 0).data.get();

    std::vector<QString> asked;
    panel->dialogs.ask_no_default = [&asked](const QString& title, const QString&) {
        asked.push_back(title);
        return false;  // No: the files read keep their results
    };
    panel->analyze_button->click();
    EXPECT_EQ(asked, std::vector<QString>{"Recalculate bitrate?"});
    ASSERT_TRUE(wait_until([&log] { return !std::filesystem::is_empty(log); })) << "ffprobe never started";
    // The scan said the file had started before it started ffprobe, but the
    // window may not have taken the message yet: it is waiting, posted.
    QApplication::processEvents();
    panel->add_files({paths[2]}, {}, true);
    EXPECT_EQ(cell(*panel, 0, col_status).toStdString(), "Complete");
    EXPECT_EQ(cell(*panel, 1, col_status).toStdString(), "Reading video packets\xE2\x80\xA6");
    EXPECT_EQ(cell(*panel, 2, col_status).toStdString(), "Queued");
    EXPECT_EQ(panel->analyze_button->text().toStdString(), "Stop");
    ASSERT_TRUE(panel->analyze_button->isEnabled());

    panel->analyze_button->click();
    EXPECT_EQ(panel->analyze_button->text().toStdString(), "Stopping\xE2\x80\xA6");
    EXPECT_FALSE(panel->analyze_button->isEnabled());
    ASSERT_TRUE(wait_until([&panel] { return idle(*panel); })) << "Stop did not end ffprobe";
    EXPECT_EQ(entry(*panel, 0).data.get(), kept);
    EXPECT_EQ(cell(*panel, 1, col_status).toStdString(), "Stopped");
    EXPECT_EQ(cell(*panel, 2, col_status).toStdString(), "Stopped");
    EXPECT_TRUE(panel->status_label->text().contains("stopped")) << panel->status_label->text().toStdString();
    EXPECT_TRUE(panel->analyze_button->isEnabled());
    EXPECT_EQ(panel->analyze_button->text().toStdString(), "Calculate bitrate");
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

}  // namespace vml::app
