// The tab bar's Export panel (export_panel.h, issue #224): its ticks are
// Settings' Exports include itself, it offers the open tab's selection
// before every video, and it says why it cannot export before the click.
#include "export_panel.h"

#include <QCheckBox>
#include <QComboBox>
#include <QPushButton>
#include <QRadioButton>

#include <gtest/gtest.h>

#include <optional>
#include <string>
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

}  // namespace
}  // namespace vml::app
