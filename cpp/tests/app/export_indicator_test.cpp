// The indicator that says how an export went (export_indicator.h, issue
// #224), in place of the windows that said "Export complete": what it says,
// and a write's outcome reaching it from the writer's thread.
#include "export_indicator.h"

#include <QCoreApplication>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <vector>
#include <stdexcept>

#include "file_writes.h"

namespace vml::app {
namespace {

TEST(ExportIndicator, SaysWhatWasSaved) {
    EXPECT_EQ(saved_text({"Y:/out/a - results.json"}).toStdString(), "Saved a - results.json");
    EXPECT_EQ(saved_text({"Y:/out/a.json", "Y:/out/a.csv"}).toStdString(), "Saved 2 files");
    EXPECT_TRUE(saved_folder({"Y:/out/a.json", "Y:/out/a.csv"}).endsWith("out"));

    ExportIndicator indicator;
    indicator.show_outcome({{"Y:/out/a.json", "Y:/out/a.csv"}, {}, {}});
    EXPECT_FALSE(indicator.failed());
    EXPECT_TRUE(indicator.text().contains("Saved 2 files"));
    EXPECT_TRUE(indicator.text().contains("Open folder"));
}

// A failure says why, in short where the reason is long; Details has it all.
TEST(ExportIndicator, SaysWhyNothingWasSaved) {
    ExportIndicator indicator;
    QString shown;
    QString details;
    indicator.show_details = [&](const QString& error, const QString& all) {
        shown = error;
        details = all;
    };
    const QString long_reason = QString("Could not write ") + QString(200, 'x');
    indicator.show_outcome({{}, long_reason, "all of it"});
    EXPECT_TRUE(indicator.failed());
    EXPECT_TRUE(indicator.text().contains("Not saved:"));
    EXPECT_TRUE(indicator.text().contains("Could not write"));
    EXPECT_FALSE(indicator.text().contains(QString(200, 'x')));  // cut short
    EXPECT_TRUE(indicator.text().contains("Details"));
    // A Save dialog's Cancel says nothing, and leaves the failure shown.
    indicator.show_outcome({});
    EXPECT_TRUE(indicator.failed());
    // The next export's own outcome replaces it.
    indicator.show_saving();
    EXPECT_FALSE(indicator.failed());
    EXPECT_TRUE(indicator.text().contains("Saving..."));
}

// A note (what a click did, or why it did nothing) is shown as a save is,
// replacing a save's line; a failure stays over it until clicked.
TEST(ExportIndicator, ANoteSaysWhatAClickDidButNotOverAFailure) {
    ExportIndicator indicator;
    indicator.show_outcome({{"Y:/out/a.json"}, {}, {}});
    indicator.show_note("crf20.mkv is already in Metric Graphs.");
    EXPECT_EQ(indicator.text().toStdString(), "crf20.mkv is already in Metric Graphs.");
    EXPECT_FALSE(indicator.failed());
    // Hidden, it keeps its room, so nothing beside it moves as it goes: its
    // corner laid out again drew the buttons beside it over for a frame,
    // a flash where the line had faded.
    EXPECT_TRUE(indicator.sizePolicy().retainSizeWhenHidden());
    indicator.show_outcome({{}, "Could not write a.json", {}});
    indicator.show_note("a note");
    EXPECT_TRUE(indicator.failed());
    EXPECT_TRUE(indicator.text().contains("Not saved:"));
}

// A write queued with what to call when it is done calls it on the window's
// thread, with the error where it failed, and not write_failed then.
TEST(ExportIndicator, AWriteSaysHowItWent) {
    FileWrites writes;
    int failed_signals = 0;
    QObject::connect(&writes, &FileWrites::write_failed, [&failed_signals] { ++failed_signals; });
    std::vector<QString> errors;
    writes.submit("ok", [] {}, [&errors](const QString& error) { errors.push_back(error); });
    writes.submit("fails", [] { throw std::runtime_error("disk full"); },
                  [&errors](const QString& error) { errors.push_back(error); });
    ASSERT_TRUE(writes.wait_until_idle(std::chrono::seconds(10)));
    QCoreApplication::sendPostedEvents();
    ASSERT_EQ(errors.size(), 2U);
    EXPECT_TRUE(errors[0].isEmpty());
    EXPECT_EQ(errors[1].toStdString(), "disk full");
    EXPECT_EQ(failed_signals, 0);
}

}  // namespace
}  // namespace vml::app
