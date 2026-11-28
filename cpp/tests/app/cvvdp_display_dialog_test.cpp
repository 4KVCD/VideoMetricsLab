// The CVVDP display dialog says why a name or a display can't be saved on a
// line of its own, and stays open to put it right (issue #239): no window
// over it.
#include <QLabel>
#include <QLineEdit>

#include <gtest/gtest.h>

#include "cvvdp_display_dialog.h"
#include "vml/cvvdp.h"

namespace vml::app {
namespace {

TEST(CvvdpDisplayDialog, ANameTakenIsSaidInTheDialog) {
    CvvdpDisplayDialog dialog(builtin_cvvdp_presets().front().settings.display, nullptr, std::nullopt, true,
                              {"My TV"});
    // Its two lines are there from the start, so nothing moves as it comes.
    EXPECT_GE(dialog.problem_line->minimumHeight(), dialog.problem_line->fontMetrics().lineSpacing() * 2);
    dialog.name_edit->setText("My TV");
    dialog.finish("save_new");
    EXPECT_TRUE(dialog.action.isEmpty());  // not closed
    EXPECT_EQ(dialog.result(), 0);
    EXPECT_TRUE(dialog.problem_line->text().contains("My TV"));

    dialog.name_edit->clear();
    dialog.name_edit->textEdited(QString());  // as typing does
    EXPECT_TRUE(dialog.problem_line->text().isEmpty());
    dialog.name_edit->setText("Den");
    dialog.finish("save_new");
    EXPECT_EQ(dialog.action, "save_new");
}

}  // namespace
}  // namespace vml::app
