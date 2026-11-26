// Lines that list every metric go on to more lines where they do not fit
// (widgets.h, WrappingLabel): Video Compare's frame line and the Videos tab's
// run lines were one line, cut at the window's edge.
#include <QApplication>
#include <QTest>
#include <QVBoxLayout>
#include <QWidget>

#include <gtest/gtest.h>

#include "widgets.h"

namespace vml::app {
namespace {

TEST(WrappingLabel, GrowsToItsTextAndKeepsItsHeightUntilCleared) {
    QWidget window;
    auto* layout = new QVBoxLayout(&window);
    auto* label = new WrappingLabel;
    layout->addWidget(label);
    layout->addStretch(1);
    window.resize(300, 300);
    window.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&window));
    const int one_line = label->height();

    QString scores;
    for (int metric = 0; metric < 20; ++metric) {
        scores += QStringLiteral("Metric%1 12.34 \u00B7 ").arg(metric);
    }
    label->set_text(scores);
    const int grown = label->height();
    EXPECT_GE(grown, one_line + 2 * label->fontMetrics().lineSpacing());
    // A shorter text: the height stays, so what is below does not move at
    // each frame or progress tick.
    label->set_text(QStringLiteral("Metric 12.34"));
    EXPECT_EQ(label->height(), grown);
    label->clear_text();
    EXPECT_EQ(label->height(), one_line);

    // unbroken() keeps a name and its score on one line, not broken at a
    // space or a hyphen; the text reads as it was given.
    const QString score = QStringLiteral("PSNR-HVS-M 90.00");
    const int margins = window.width() - label->width();
    window.resize(label->fontMetrics().horizontalAdvance(QStringLiteral("PSNR-HVS")) + margins, 300);
    QApplication::processEvents();
    label->set_text(score);
    EXPECT_GT(label->height(), one_line);  // free to break, it does
    label->clear_text();
    label->set_text(WrappingLabel::unbroken(score));
    EXPECT_EQ(label->height(), one_line);
    EXPECT_EQ(label->full_text(), score);
}

}  // namespace
}  // namespace vml::app
