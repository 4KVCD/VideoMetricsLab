// Add/remove metrics' panel and Settings' Default metrics (MetricList): the
// two sections from the registry's one list, All and None, the count, the
// planes' switch apart from the metrics, and in the window, that the panel
// shows or hides columns without ticking what rows calculate.
#include <QApplication>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalSpy>
#include <QTableWidget>
#include <QToolButton>

#include <gtest/gtest.h>

#include <set>
#include <string>

#include "main_window.h"
#include "metric_list.h"
#include "vml/metrics.h"
#include "vml/settings.h"
#include "widgets.h"

namespace vml::app {
namespace {

using Keys = std::set<std::string>;

// All ticks a section's metrics and None unticks them, the other section's
// left alone; in the panel None keeps one metric shown, the section's first
// that was.
TEST(MetricList, AllAndNoneTickASectionKeepingOneInThePanel) {
    EXPECT_EQ(section_ticks({"vmaf"}, MetricSection::specialized, true, 1),
              (Keys{"vmaf", "vmaf_hdr", "psnr_yuv", "ssim_yuv", "ms_ssim", "ms_ssim_yuv", "psnr_hvs", "cambi",
                    "ciede2000"}));
    EXPECT_EQ(section_ticks({"vmaf", "psnr", "cambi"}, MetricSection::general_purpose, false, 1), Keys{"cambi"});
    EXPECT_EQ(section_ticks({"psnr", "ssim", "cvvdp"}, MetricSection::general_purpose, false, 1), Keys{"psnr"});
    EXPECT_EQ(section_ticks({"psnr", "ssim"}, MetricSection::general_purpose, false, 0), Keys{});
}

// The panel: the 17 metrics with their tooltips, Specialized folded and
// counted, the count of shown, the last shown kept, and the planes' switch
// apart from the metrics' ticks.
TEST(MetricList, PanelCountsTheMetricsShownAndKeepsThePlanesApart) {
    MetricList list(MetricList::Place::panel);
    const Keys general{"vmaf", "vmaf_neg", "vmaf_v1", "psnr", "ssim", "xpsnr", "ssimulacra2", "butteraugli", "cvvdp"};
    list.set_ticked(general);
    EXPECT_EQ(list.ticked(), general);
    EXPECT_FALSE(list.specialized_open());
    const auto label_with = [&list](const QString& text) {
        for (const QLabel* label : list.findChildren<QLabel*>()) {
            if (label->text() == text) {
                return true;
            }
        }
        for (const QToolButton* heading : list.findChildren<QToolButton*>()) {  // Specialized's, which folds
            if (heading->text() == text) {
                return true;
            }
        }
        return false;
    };
    EXPECT_TRUE(label_with("9 of 17 shown"));
    EXPECT_TRUE(label_with("Specialized (8) · 0 shown"));
    ASSERT_NE(list.box("cambi"), nullptr);
    EXPECT_EQ(list.box("psnr_yuv_y"), nullptr);  // a plane is no metric of the list
    EXPECT_TRUE(list.box("cambi")->toolTip().contains("24 is the most"));

    QSignalSpy ticks(&list, &MetricList::ticks_changed);
    QSignalSpy planes(&list, &MetricList::planes_toggled);
    list.set_specialized_open(true);
    list.box("cambi")->click();
    EXPECT_EQ(ticks.count(), 1);
    EXPECT_TRUE(list.ticked().contains("cambi"));
    EXPECT_TRUE(label_with("10 of 17 shown"));
    ASSERT_NE(list.planes_box(), nullptr);
    list.planes_box()->click();
    EXPECT_EQ(planes.count(), 1);
    EXPECT_EQ(ticks.count(), 1);  // the switch is no metric: the ticks and the count stay
    EXPECT_TRUE(label_with("10 of 17 shown"));

    list.set_ticked(Keys{"ssim"});
    EXPECT_FALSE(list.box("ssim")->isEnabled());  // the last shown stays
    EXPECT_EQ(list.box("ssim")->toolTip(), "At least one metric must stay shown.");
    EXPECT_TRUE(list.box("psnr")->isEnabled());
}

// Settings: Specialized open, any number ticked, no switch for the planes.
TEST(MetricList, SettingsListsAllSeventeenOpen) {
    MetricList list(MetricList::Place::settings);
    EXPECT_TRUE(list.specialized_open());
    EXPECT_EQ(list.planes_box(), nullptr);
    list.set_ticked(Keys{"psnr"});
    EXPECT_TRUE(list.box("psnr")->isEnabled());
    list.box("psnr")->click();
    EXPECT_TRUE(list.ticked().empty());
}

// A plane's tooltip says whose plane it is, then its metric's text, without
// its metric's bands.
TEST(MetricList, PlanesTipsAreTheirMetrics) {
    const QString tip = metric_tip("ssim_yuv_u");
    EXPECT_TRUE(tip.startsWith("SSIM YUV: U: the U plane alone, calculated with SSIM YUV (6:1:1).")) << tip.toStdString();
    EXPECT_TRUE(tip.contains("MSU's codec comparisons"));
    EXPECT_FALSE(tip.contains("The app's bands"));
    EXPECT_TRUE(metric_tip("ssim_yuv").contains("The app's bands"));
}

// In the window: showing a metric with the panel shows its column and saves
// it shown, and leaves its tick, what rows calculate, as it was; the planes'
// switch shows their columns and is saved apart. Their headers say what they
// measure.
class MetricsPanelInWindow : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        window_ = new MainWindow;
        window_->show();
        QApplication::processEvents();
    }
    static void TearDownTestSuite() {
        delete window_;
        window_ = nullptr;
    }
    static MetricList* open_panel() {
        for (QPushButton* button : window_->findChildren<QPushButton*>()) {
            if (button->text() == "Add/remove metrics") {
                button->click();
            }
        }
        auto* panel = window_->findChild<MetricsPanel*>();
        return panel == nullptr ? nullptr : panel->list();
    }
    static MainWindow* window_;
};

MainWindow* MetricsPanelInWindow::window_ = nullptr;

TEST_F(MetricsPanelInWindow, ShowingAMetricLeavesWhatRowsCalculate) {
    auto* table = window_->findChild<FillColumnTable*>();
    auto* header = window_->findChild<CheckableHeaderView*>();
    ASSERT_NE(table, nullptr);
    ASSERT_NE(header, nullptr);
    EXPECT_TRUE(table->isColumnHidden(col_cambi));
    EXPECT_TRUE(header->is_checked(col_cambi));  // ticked but hidden at first
    EXPECT_TRUE(header->is_checked(col_psnr_yuv));
    MetricList* list = open_panel();
    ASSERT_NE(list, nullptr);
    list->box("cambi")->click();
    list->box("psnr_yuv")->click();
    EXPECT_FALSE(table->isColumnHidden(col_cambi));
    EXPECT_TRUE(header->is_checked(col_cambi));
    EXPECT_TRUE(table->isColumnHidden(col_psnr_yuv_y));
    list->planes_box()->click();
    EXPECT_FALSE(table->isColumnHidden(col_psnr_yuv_y));
    EXPECT_TRUE(table->isColumnHidden(col_ssim_yuv_y));  // its metric hidden
    const Settings saved = Settings::load_from(app_data_folder() / "settings.json");
    EXPECT_TRUE(saved.show_yuv_planes);
    EXPECT_TRUE(saved.default_compute_cambi);
    EXPECT_EQ(std::count(saved.hidden_metrics.begin(), saved.hidden_metrics.end(), "cambi"), 0);
    EXPECT_EQ(std::count(saved.hidden_metrics.begin(), saved.hidden_metrics.end(), "ssim_yuv"), 1);
    EXPECT_TRUE(table->horizontalHeaderItem(col_psnr_yuv_y)->toolTip().contains("the Y plane alone"));
    list->window()->close();
}

}  // namespace
}  // namespace vml::app
