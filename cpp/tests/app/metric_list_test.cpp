// Add/remove metrics' panel and Settings' Default metrics (MetricList): the
// two sections from the registry's one list, All and None, the count, the
// planes' switch apart from the metrics, and in the window, that the panel
// shows or hides columns without ticking what rows calculate.
#include <QApplication>
#include <QCheckBox>
#include <QHeaderView>
#include <QLabel>
#include <QPointer>
#include <QMouseEvent>
#include <QPushButton>
#include <QSignalSpy>
#include <QTableView>
#include <QTableWidget>
#include <QTest>
#include <QToolButton>

#include <QScrollArea>
#include <QScrollBar>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "main_window.h"
#include "metric_list.h"
#include "metric_order.h"
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
              (Keys{"vmaf", "psnr_yuv", "psnr_yuv_mse", "ssim_yuv", "ms_ssim", "ms_ssim_yuv", "psnr_hvs", "cambi",
                    "ciede2000", "vmaf_yuv"}));
    EXPECT_EQ(section_ticks({"vmaf", "psnr", "cambi"}, MetricSection::general_purpose, false, 1), Keys{"cambi"});
    EXPECT_EQ(section_ticks({"psnr", "ssim", "cvvdp"}, MetricSection::general_purpose, false, 1), Keys{"psnr"});
    EXPECT_EQ(section_ticks({"psnr", "ssim"}, MetricSection::general_purpose, false, 0), Keys{});
}

// The panel: the 17 metrics with their tooltips, Specialized open at first
// (#251: folded, its metrics were found only by those who knew to unfold
// it) and, folded, counted; the count of shown, the last shown kept, and
// the planes' switch apart from the metrics' ticks.
TEST(MetricList, PanelCountsTheMetricsShownAndKeepsThePlanesApart) {
    MetricList list(MetricList::Place::panel);
    const Keys general{"vmaf", "vmaf_neg", "vmaf_v1", "vmaf_hdr", "psnr", "ssim", "xpsnr",
                       "ssimulacra2", "butteraugli", "cvvdp"};
    list.set_ticked(general);
    EXPECT_EQ(list.ticked(), general);
    EXPECT_TRUE(list.specialized_open());
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
    EXPECT_TRUE(label_with("10 of 19 shown"));
    EXPECT_TRUE(label_with("General purpose (10)"));
    EXPECT_TRUE(label_with("Specialized (9)"));
    list.set_specialized_open(false);  // folded still: how many of it are shown
    EXPECT_TRUE(label_with("Specialized (9) · 0 shown"));
    ASSERT_NE(list.box("cambi"), nullptr);
    EXPECT_EQ(list.box("psnr_yuv_y"), nullptr);  // a plane is no metric of the list
    EXPECT_TRUE(list.box("cambi")->toolTip().contains("24 is the most"));

    QSignalSpy ticks(&list, &MetricList::ticks_changed);
    QSignalSpy planes(&list, &MetricList::planes_toggled);
    list.set_specialized_open(true);
    list.box("cambi")->click();
    EXPECT_EQ(ticks.count(), 1);
    EXPECT_TRUE(list.ticked().contains("cambi"));
    EXPECT_TRUE(label_with("11 of 19 shown"));
    ASSERT_NE(list.planes_box(), nullptr);
    list.planes_box()->click();
    EXPECT_EQ(planes.count(), 1);
    EXPECT_EQ(ticks.count(), 1);  // the switch is no metric: the ticks and the count stay
    EXPECT_TRUE(label_with("11 of 19 shown"));

    list.set_ticked(Keys{"ssim"});
    EXPECT_FALSE(list.box("ssim")->isEnabled());  // the last shown stays
    EXPECT_EQ(list.box("ssim")->toolTip(), "At least one metric must stay shown.");
    EXPECT_TRUE(list.box("psnr")->isEnabled());
}

// The panel grows as Specialized opens and shrinks back as it folds, its
// content kept together under its heading: it kept its open height, the
// room spread as gaps between its lines (#237).
TEST(MetricList, PanelShrinksBackWhenSpecializedFolds) {
    auto* panel = new MetricsPanel(nullptr);
    const std::unique_ptr<MetricsPanel> owned(panel);
    // A layout's change reaches the window one level per pass of the event
    // loop, as it would with the app running.
    const auto settle = [] {
        for (int pass = 0; pass < 5; ++pass) {
            QApplication::processEvents();
        }
    };
    panel->list()->set_ticked(Keys{"vmaf"});
    panel->show_at(QPoint(0, 0));
    settle();
    const int open = panel->height();  // as it opens (#251)
    panel->list()->set_specialized_open(false);
    settle();
    const int folded = panel->height();
    EXPECT_LT(folded, open);
    panel->list()->set_specialized_open(true);
    settle();
    EXPECT_EQ(panel->height(), open);
    panel->list()->set_specialized_open(false);
    settle();
    EXPECT_EQ(panel->height(), folded);
    EXPECT_EQ(panel->list()->height(), panel->list()->sizeHint().height());
    panel->hide();
}

// Taller than its screen's room (Specialized open, #251, on a small screen or
// at a large scaling), the panel stays within it and its list scrolls, every
// metric still reachable; with room, nothing scrolls.
TEST(MetricList, PanelScrollsRatherThanRunPastTheScreen) {
    auto* panel = new MetricsPanel(nullptr);
    const std::unique_ptr<MetricsPanel> owned(panel);
    panel->list()->set_ticked(Keys{"vmaf"});
    // As wide as the list must be: its minimum width (72 characters) is more
    // than its size hint in the window's fonts, and a scroll area as narrow
    // as the hint cut the panel's right side off. Made so here.
    panel->list()->setMinimumWidth(panel->list()->sizeHint().width() + 200);
    panel->fit_height(0);  // no bound
    panel->show();
    QApplication::processEvents();
    const int open = panel->height();
    EXPECT_FALSE(panel->scroll()->verticalScrollBar()->isVisible());
    EXPECT_GE(panel->scroll()->viewport()->width(), panel->list()->minimumWidth());
    EXPECT_EQ(panel->list()->width(), panel->scroll()->viewport()->width());

    panel->fit_height(open / 2);
    QApplication::processEvents();
    EXPECT_LE(panel->height(), open / 2);
    QScrollBar* bar = panel->scroll()->verticalScrollBar();
    EXPECT_TRUE(bar->isVisible());
    EXPECT_GT(bar->maximum(), 0);
    bar->setValue(bar->maximum());  // the last metric in sight at the bottom
    QCheckBox* last = panel->list()->box("vmaf_yuv");
    ASSERT_NE(last, nullptr);
    const QPoint in_view = last->mapTo(panel->scroll()->viewport(), QPoint(0, last->height()));
    EXPECT_LE(in_view.y(), panel->scroll()->viewport()->height());

    panel->fit_height(open * 2);  // room again
    QApplication::processEvents();
    EXPECT_EQ(panel->height(), open);
    panel->hide();
}

// The button that opened the panel closes it: a press on it while the panel
// is open closes the panel and is not replayed to the button, whose click
// would open it again at once (#237). A press elsewhere outside it closes
// it and is still replayed, to reach what it lands on. (Qt's offscreen
// platform, the tests', replays nothing, so what is asked of Qt is checked:
// Qt::WA_NoMouseReplay.)
TEST(MetricList, ASecondClickOnTheButtonClosesThePanel) {
    QWidget window;
    window.resize(500, 300);
    auto* button = new QPushButton("Add/remove metrics", &window);
    button->setGeometry(300, 10, 160, 30);
    auto* other = new QPushButton("Other", &window);
    other->setGeometry(20, 10, 100, 30);
    window.show();
    QApplication::processEvents();
    // The panel's answer to a press on `target`: whether it closed, and
    // whether the press may be replayed.
    const auto press_on = [&](QWidget* target) {
        QPointer<MetricsPanel> panel = new MetricsPanel(&window);
        panel->set_anchor(button);
        panel->show_at(button->mapToGlobal(button->rect().bottomLeft()));
        QApplication::processEvents();
        QTest::mousePress(panel, Qt::LeftButton, {}, panel->mapFromGlobal(target->mapToGlobal(target->rect().center())));
        const bool closed = !panel->isVisible();
        const bool replayed = !panel->testAttribute(Qt::WA_NoMouseReplay);
        QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        EXPECT_TRUE(panel.isNull());  // deleted as it closed
        return std::pair(closed, replayed);
    };
    EXPECT_EQ(press_on(button), std::pair(true, false));
    EXPECT_EQ(press_on(other), std::pair(true, true));
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

// The Videos table's metric columns are in the registry's order, the one
// Metric Graphs, Video Compare and BD-rate list the metrics in: Metric
// Graphs' tabs had XPSNR after the YUV planes, the table after SSIM.
TEST(MetricList, TheTablesColumnsAreInTheRegistrysOrder) {
    std::vector<std::string> registry;
    for (const MetricDefinition& metric : metrics()) {
        if (std::any_of(metric_columns().begin(), metric_columns().end(),
                        [&metric](const MetricColumn& item) { return metric.key == item.key; })) {
            registry.emplace_back(metric.key);
        }
    }
    std::vector<std::string> table;
    for (const MetricColumn& item : metric_columns()) {
        table.emplace_back(item.key);
    }
    EXPECT_EQ(table, registry);
}

// A plane's tooltip says whose plane it is, then its metric's text, without
// its metric's bands.
TEST(MetricList, PlanesTipsAreTheirMetrics) {
    const QString tip = metric_tip("ssim_yuv_u");
    EXPECT_TRUE(tip.startsWith("SSIM U: the U plane alone, calculated with SSIM YUV (6:1:1).")) << tip.toStdString();
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
        QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);  // panels closed before: gone
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

// Add/remove metrics and Export open their panels on the press, as a
// button's menu opens, not a whole press later on the release. A click() (UI
// Automation's Invoke) or Space opens one panel. However it was opened, the
// button is down while its panel is open and up once it closes, as a Qt
// button's menu holds its button: drawn up at once, it flashed pressed for as
// long as the panel took to build (15-28 ms for Add/remove metrics).
TEST_F(MetricsPanelInWindow, APressOpensThePanelAtOnce) {
    QPushButton* metrics = nullptr;
    for (QPushButton* button : window_->findChildren<QPushButton*>()) {
        metrics = button->text() == "Add/remove metrics" ? button : metrics;
    }
    auto* exports = window_->findChild<QPushButton*>("tabExportButton");
    ASSERT_NE(metrics, nullptr);
    ASSERT_NE(exports, nullptr);
    const auto open_panels = [] {
        int open = 0;
        for (const PopupPanel* panel : window_->findChildren<PopupPanel*>()) {
            open += panel->isVisible() ? 1 : 0;
        }
        return open;
    };
    const auto close_panels = [] {
        for (PopupPanel* panel : window_->findChildren<PopupPanel*>()) {
            panel->close();
        }
        QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    };
    for (QPushButton* button : {metrics, exports}) {
        const std::string name = button->text().toStdString();
        QTest::mousePress(button, Qt::LeftButton, {}, button->rect().center());  // no release
        EXPECT_EQ(open_panels(), 1) << name;
        EXPECT_TRUE(button->isDown()) << name << ", pressed";
        close_panels();
        EXPECT_FALSE(button->isDown()) << name << ", closed";
        button->click();
        EXPECT_EQ(open_panels(), 1) << name;
        EXPECT_TRUE(button->isDown()) << name << ", click()";
        close_panels();
        EXPECT_FALSE(button->isDown()) << name << ", closed";
        QTest::keyClick(button, Qt::Key_Space);  // the keyboard's press and release
        EXPECT_EQ(open_panels(), 1) << name;
        EXPECT_TRUE(button->isDown()) << name << ", Space";
        close_panels();
        EXPECT_FALSE(button->isDown()) << name << ", closed";
    }
}

// The press on a panel's button that closes the panel, replayed to the
// button with its own time, opens nothing: once any popup had been closed
// with Esc, Qt replayed it whatever Qt::WA_NoMouseReplay said, and the second
// click reopened the panel. A new press, of its own time, still opens it.
TEST_F(MetricsPanelInWindow, TheClosingPressReplayedToTheButtonOpensNothing) {
    QPushButton* metrics = nullptr;
    for (QPushButton* button : window_->findChildren<QPushButton*>()) {
        metrics = button->text() == "Add/remove metrics" ? button : metrics;
    }
    auto* exports = window_->findChild<QPushButton*>("tabExportButton");
    ASSERT_NE(metrics, nullptr);
    ASSERT_NE(exports, nullptr);
    const auto open_panel = [] {
        for (PopupPanel* panel : window_->findChildren<PopupPanel*>()) {
            if (panel->isVisible()) {
                return panel;
            }
        }
        return static_cast<PopupPanel*>(nullptr);
    };
    const auto press = [](QWidget* target, const QPoint& global, quint64 time) {
        QMouseEvent event(QEvent::MouseButtonPress, target->mapFromGlobal(global), global, Qt::LeftButton,
                          Qt::LeftButton, Qt::NoModifier);
        event.setTimestamp(time);
        QApplication::sendEvent(target, &event);
    };
    const auto release = [](QWidget* target) {
        QTest::mouseRelease(target, Qt::LeftButton, {}, target->rect().center());
    };
    quint64 time = 1'000'000;
    for (QPushButton* button : {metrics, exports}) {
        const std::string name = button->text().toStdString();
        const QPoint on_button = button->mapToGlobal(button->rect().center());
        press(button, on_button, ++time);  // opens
        PopupPanel* panel = open_panel();
        ASSERT_NE(panel, nullptr) << name;
        press(panel, on_button, ++time);  // closes
        QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        EXPECT_EQ(open_panel(), nullptr) << name;
        press(button, on_button, time);  // that press, replayed
        EXPECT_EQ(open_panel(), nullptr) << name << ": reopened by the replayed press";
        EXPECT_FALSE(button->isDown()) << name;
        release(button);
        press(button, on_button, ++time);  // a new press
        EXPECT_NE(open_panel(), nullptr) << name << ": a new press opened nothing";
        if (PopupPanel* again = open_panel()) {
            again->close();
        }
        QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        release(button);
    }
}

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

// Each video's remove mark starts its row, as Metric Graphs' lines have it,
// frozen with the tick box and File name, and drawn there too (the frozen
// columns are a view of their own).
TEST_F(MetricsPanelInWindow, AVideosRemoveMarkStartsItsRow) {
    auto* table = window_->findChild<FillColumnTable*>();
    ASSERT_NE(table, nullptr);
    const QHeaderView* header = table->horizontalHeader();
    EXPECT_EQ(header->logicalIndex(0), col_remove);
    EXPECT_EQ(header->logicalIndex(1), col_graph);  // its line sent back to Metric Graphs
    EXPECT_EQ(header->logicalIndex(2), col_check);
    EXPECT_EQ(header->logicalIndex(3), col_path);
    EXPECT_NE(dynamic_cast<RemoveMarkDelegate*>(table->itemDelegateForColumn(col_remove)), nullptr);
    EXPECT_NE(dynamic_cast<RemoveMarkDelegate*>(table->itemDelegateForColumn(col_graph)), nullptr);
    EXPECT_TRUE(table->viewport()->testAttribute(Qt::WA_Hover));  // its hover drawn
    const auto* frozen = table->findChild<QTableView*>(QStringLiteral("frozenColumns"));
    ASSERT_NE(frozen, nullptr);
    EXPECT_EQ(frozen->itemDelegateForColumn(col_remove), table->itemDelegateForColumn(col_remove));
    for (const int column : {col_remove, col_graph, col_check, col_path}) {
        EXPECT_FALSE(frozen->horizontalHeader()->isSectionHidden(column)) << column;  // frozen
    }
    EXPECT_TRUE(frozen->horizontalHeader()->isSectionHidden(col_info));
}

// A metric's column dragged to another place puts the metric there in every
// list (shown_metrics) and saves the order; dropped among the video's
// columns, it comes first. A plane's column or a video's column goes back
// to its place. Default order, in Add/remove metrics, puts the registry's
// back.
TEST_F(MetricsPanelInWindow, ADraggedMetricColumnSetsTheOrderShown) {
    auto* table = window_->findChild<FillColumnTable*>();
    ASSERT_NE(table, nullptr);
    QHeaderView* header = table->horizontalHeader();
    const auto first_keys = [](std::size_t count) {
        std::vector<std::string> keys;
        for (const MetricDefinition* metric : shown_metrics()) {
            if (!metric->is_plane() && keys.size() < count) {
                keys.emplace_back(metric->key);
            }
        }
        return keys;
    };
    const auto saved_order = [] { return Settings::load_from(app_data_folder() / "settings.json").metric_order; };
    header->moveSection(header->visualIndex(col_cambi), header->visualIndex(col_psnr));
    EXPECT_EQ(first_keys(6), (std::vector<std::string>{"vmaf", "vmaf_neg", "vmaf_v1", "vmaf_hdr", "cambi", "psnr"}));
    EXPECT_EQ(header->visualIndex(col_cambi) + 1, header->visualIndex(col_psnr));
    const std::vector<std::string> saved = saved_order();
    ASSERT_GE(saved.size(), 6U);
    EXPECT_EQ(std::vector<std::string>(saved.begin(), saved.begin() + 6), first_keys(6));
    header->moveSection(header->visualIndex(col_ciede2000), header->visualIndex(col_info));
    EXPECT_EQ(first_keys(2), (std::vector<std::string>{"ciede2000", "vmaf"}));
    EXPECT_EQ(header->logicalIndex(4), col_info);  // the video's columns back in place
    const std::vector<std::string> before = saved_order();
    const int plane_at = header->visualIndex(col_psnr_yuv_y);
    header->moveSection(plane_at, header->visualIndex(col_vmaf));
    EXPECT_EQ(header->visualIndex(col_psnr_yuv_y), plane_at);
    EXPECT_EQ(header->visualIndex(col_psnr_yuv_y), header->visualIndex(col_psnr_yuv) + 1);  // after its metric
    header->moveSection(header->visualIndex(col_scaling), header->visualIndex(col_vmaf));
    EXPECT_EQ(header->logicalIndex(6), col_scaling);
    EXPECT_EQ(saved_order(), before);
    // With the mouse, the header's own drag: the columns put back from inside
    // its release. The metric after the first, dropped on the video's info
    // column, goes first; the info column stays where it was.
    const int dragged = header->logicalIndex(header->visualIndex(col_ciede2000) + 1);
    ASSERT_FALSE(header->isSectionHidden(dragged));
    ASSERT_FALSE(header->isSectionHidden(col_info));
    const auto centre = [header](int logical) {
        return QPointF(header->sectionViewportPosition(logical) + header->sectionSize(logical) / 2.0,
                       header->height() / 2.0);
    };
    const auto mouse = [header](QEvent::Type type, QPointF at, Qt::MouseButtons held) {
        QMouseEvent event(type, at, header->viewport()->mapToGlobal(at), Qt::LeftButton, held, Qt::NoModifier);
        QApplication::sendEvent(header->viewport(), &event);
    };
    mouse(QEvent::MouseButtonPress, centre(dragged), Qt::LeftButton);
    mouse(QEvent::MouseMove, centre(dragged) - QPointF(20, 0), Qt::LeftButton);
    mouse(QEvent::MouseMove, centre(col_info), Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, centre(col_info), Qt::NoButton);
    EXPECT_EQ(header->visualIndex(dragged) + 1, header->visualIndex(col_ciede2000));
    EXPECT_EQ(header->logicalIndex(4), col_info);
    EXPECT_EQ(saved_order().front(), first_keys(1).front());
    EXPECT_EQ(first_keys(2).back(), "ciede2000");
    MetricList* list = open_panel();
    ASSERT_NE(list, nullptr);
    auto* panel = qobject_cast<MetricsPanel*>(list->window());  // this one: an earlier test's may be left closed
    ASSERT_NE(panel, nullptr);
    ASSERT_TRUE(panel->default_order_button()->isEnabled());
    panel->default_order_button()->click();
    EXPECT_TRUE(saved_order().empty());
    EXPECT_EQ(shown_metrics(), ordered_metrics({}));
    EXPECT_FALSE(panel->default_order_button()->isEnabled());
    panel->close();
}

}  // namespace
}  // namespace vml::app
