// The tab bar's Export panel (export_panel.h, issue #224): its ticks are
// Settings' Exports include itself, it offers the open tab's selection
// before every video, and it says why it cannot export before the click.
#include "export_panel.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QPushButton>
#include <QRadioButton>
#include <QTest>

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/settings.h"

namespace vml::app {
namespace {

TEST(ExportPanel, ItsTicksAreSettingsExportsInclude) {
    Settings settings;
    settings.export_items = {"results", "graphs"};
    settings.export_formats = {"json"};
    settings.export_zip = true;
    ExportPanel panel(settings);
    int changed = 0;
    std::optional<bool> exported;
    ExportPanel::Videos videos{"The selected video on Videos: a", {}, "All scored videos (3)", {}};
    panel.videos = [&videos] { return videos; };
    panel.choices_changed = [&changed] { ++changed; };
    panel.export_videos = [&exported](bool selection) { exported = selection; };
    panel.refresh();

    EXPECT_TRUE(panel.item_box("results")->isChecked());
    EXPECT_TRUE(panel.item_box("graphs")->isChecked());
    EXPECT_FALSE(panel.item_box("frames")->isChecked());
    EXPECT_TRUE(panel.format_box("json")->isChecked());
    EXPECT_FALSE(panel.format_box("csv")->isChecked());
    EXPECT_TRUE(panel.form_combo()->currentData().toBool());
    EXPECT_EQ(changed, 0);  // shown, not changed

    // A tick here is Settings' own, saved as it is made.
    panel.item_box("worst_frames")->setChecked(true);
    panel.format_box("csv")->setChecked(true);
    panel.form_combo()->setCurrentIndex(1);
    EXPECT_EQ(settings.export_items, (std::vector<std::string>{"results", "graphs", "worst_frames"}));
    EXPECT_EQ(settings.export_formats, (std::vector<std::string>{"json", "csv"}));
    EXPECT_FALSE(settings.export_zip);
    EXPECT_EQ(changed, 3);

    // The tab's selection first, every video the other choice.
    EXPECT_TRUE(panel.selection_radio()->isChecked());
    EXPECT_EQ(panel.all_radio()->text().toStdString(), "All scored videos (3)");
    panel.export_button()->click();
    ASSERT_TRUE(exported.has_value());
    EXPECT_TRUE(*exported);

    // No selection on the tab: every video; none scored: greyed, saying why.
    videos = {{}, {}, "All scored videos (0)", "No test video has been scored against the reference yet."};
    panel.refresh();
    EXPECT_TRUE(panel.all_radio()->isChecked());
    EXPECT_FALSE(panel.export_button()->isEnabled());
    EXPECT_EQ(panel.export_button()->toolTip(), videos.all_refused);

    // Nothing ticked: greyed, saying so.
    videos.all_refused.clear();
    settings.export_items.clear();
    panel.refresh();
    EXPECT_FALSE(panel.export_button()->isEnabled());
    EXPECT_EQ(panel.export_button()->toolTip().toStdString(),
              "Export: nothing is ticked in Settings > Storage > Exports include");
}

// A second click on Export closes the panel, as Add/remove metrics' does
// (PopupPanel): the press that closes it is not replayed to the button,
// which would open it again at once. A press anywhere else (a tab) closes it
// and is replayed, so the tab is switched to.
TEST(ExportPanel, ASecondClickOnExportClosesThePanel) {
    QWidget window;
    window.resize(600, 300);
    auto* button = new QPushButton("Export", &window);
    button->setGeometry(480, 10, 100, 30);
    auto* other = new QPushButton("Videos", &window);
    other->setGeometry(20, 10, 100, 30);
    window.show();
    QApplication::processEvents();
    Settings settings;
    ExportPanel panel(settings, &window);
    panel.videos = [] { return ExportPanel::Videos{{}, {}, "All scored videos (1)", {}}; };
    // The panel's answer to a press on `target`: whether it closed, and
    // whether the press may be replayed.
    const auto press_on = [&](QWidget* target) {
        panel.open_under(button);
        QApplication::processEvents();
        QTest::mousePress(&panel, Qt::LeftButton, {}, panel.mapFromGlobal(target->mapToGlobal(target->rect().center())));
        return std::pair(!panel.isVisible(), !panel.testAttribute(Qt::WA_NoMouseReplay));
    };
    EXPECT_EQ(press_on(button), std::pair(true, false));
    EXPECT_EQ(press_on(other), std::pair(true, true));
}

}  // namespace
}  // namespace vml::app
