// A card's save button's one-click save (quick_save.h, issue #224): the formats
// in Settings' order, and a save that stops at the first file it cannot
// write, saying why.
#include "quick_save.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QContextMenuEvent>
#include <QCoreApplication>
#include <QMenu>
#include <QPixmap>
#include <QTest>
#include <QTimer>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "file_writes.h"
#include "test_videos.h"
#include "vml/platform/system.h"
#include "vml/settings.h"

namespace vml::app {
namespace {

TEST(QuickSave, WritesInSettingsOrderAndStopsAtAFailure) {
    Settings settings;
    settings.quick_save_formats = {"csv", "json"};
    EXPECT_EQ(quick_formats(settings), (std::vector<std::string>{"json", "csv"}));
    EXPECT_EQ(formats_text(quick_formats(settings)).toStdString(), "JSON and CSV");
    settings.quick_save_formats = {"csv"};
    EXPECT_EQ(formats_text(quick_formats(settings)).toStdString(), "CSV");

    std::vector<QString> asked;
    const auto file = [&asked](const QString& name, const QString& error) {
        return QuickFile{name, [&asked, error](const QString& path) {
                             asked.push_back(path);
                             return error;
                         }};
    };
    const test::Folder folder;
    const ExportOutcome outcome =
        save_quick(QString::fromStdString(platform::path_text(folder / "out")),
                   {file("a.json", {}), file("a.csv", "Could not write a.csv"), file("a.png", {})});
    ASSERT_EQ(asked.size(), 2U);  // not the PNG, after the CSV failed
    ASSERT_EQ(outcome.files.size(), 1);
    EXPECT_TRUE(outcome.files.front().endsWith("a.json"));
    EXPECT_EQ(outcome.error.toStdString(), "Could not write a.csv");
}

// A save button's files are made at the click where they read the card
// (data_files' text, image_file's picture), then written on the window's
// file writer (written_by), on its thread, so that the window answers
// meanwhile: a 2-hour film's bitrate data took about 300 ms to make and
// write. How it went is heard after, on this thread.
TEST(QuickSave, WritesOnTheFileWriterThenSaysHowItWent) {
    int texts = 0;
    int images = 0;
    std::vector<QuickFile> files = data_files("stats", {"json", "csv"}, [&texts](const std::string& format) {
        ++texts;
        return format;
    });
    files.push_back(image_file("graph.png", [&images] {
        ++images;
        QPixmap picture(4, 4);
        picture.fill(Qt::red);
        return picture;
    }));
    EXPECT_EQ(texts, 2);  // made now, not as they are written
    EXPECT_EQ(images, 1);
    std::thread::id written_on;
    files.push_back({"last.txt", [&written_on](const QString& path) {
                         written_on = std::this_thread::get_id();
                         return write_text_file(path, "x");
                     }});

    FileWrites writes;
    QuickSaves saves;
    saves.background = written_by(writes);
    const test::Folder folder;
    std::vector<ExportOutcome> heard;
    saves.save(QString::fromStdString(platform::path_text(folder / "out")), std::move(files),
               [&heard](const ExportOutcome& outcome) { heard.push_back(outcome); });
    ASSERT_TRUE(writes.wait_until_idle(std::chrono::seconds(10)));
    EXPECT_TRUE(heard.empty());  // not until this thread's events run
    QCoreApplication::sendPostedEvents();
    ASSERT_EQ(heard.size(), 1U);
    EXPECT_TRUE(heard.front().error.isEmpty());
    ASSERT_EQ(heard.front().files.size(), 4);
    EXPECT_TRUE(heard.front().files[2].endsWith("graph.png"));
    EXPECT_TRUE(std::filesystem::exists(folder / "out" / "graph.png"));
    EXPECT_NE(written_on, std::thread::id());
    EXPECT_NE(written_on, std::this_thread::get_id());

    // A save that throws says so, as its error, on the indicator.
    written_by(writes)([]() -> ExportOutcome { throw std::runtime_error("out of memory"); },
                       [&heard](const ExportOutcome& outcome) { heard.push_back(outcome); });
    ASSERT_TRUE(writes.wait_until_idle(std::chrono::seconds(10)));
    QCoreApplication::sendPostedEvents();
    ASSERT_EQ(heard.size(), 2U);
    EXPECT_EQ(heard.back().error.toStdString(), "out of memory");
}

// Issue #261: a save button says what it saves, as its words and its
// accessible name; a click on the words saves, and a press on its arrow opens
// Save as... instead, without saving; neither moves or resizes it.
TEST(QuickSave, SaveButtonsWordsSaveItsArrowOpensSaveAsAndItNarrowsToItsGlyph) {
    SaveButton button(QStringLiteral("Save graph"), SaveKind::picture);
    EXPECT_EQ(button.text().toStdString(), "Save graph");
    EXPECT_EQ(button.accessibleName().toStdString(), "Save graph");
    EXPECT_EQ(button.kind(), SaveKind::picture);
    int saved = 0;
    int menus = 0;
    button.save = [&saved] { ++saved; };
    button.save_as = [&menus](QMenu& menu) {
        ++menus;
        menu.addAction(QStringLiteral("Graph as PNG..."));
    };
    button.describe = [] { return QStringLiteral("Save this graph as PNG in the export folder.\nArrow: Save as..."); };
    button.resize(button.sizeHint());
    button.show();
    button.refresh();
    EXPECT_EQ(button.accessibleDescription().toStdString(), "Save this graph as PNG in the export folder.");
    const QRect before = button.geometry();
    QTest::mouseClick(&button, Qt::LeftButton, {}, QPoint(button.arrow_rect().left() / 2, button.height() / 2));
    EXPECT_EQ(saved, 1);
    EXPECT_EQ(menus, 0);
    // The menu opens on the press; closed here as soon as it shows.
    bool shown = false;
    QTimer::singleShot(0, [&shown] {
        if (QWidget* popup = QApplication::activePopupWidget()) {
            shown = true;
            popup->close();
        }
    });
    QTest::mousePress(&button, Qt::LeftButton, {}, button.arrow_rect().center());
    QTest::mouseRelease(&button, Qt::LeftButton, {}, button.arrow_rect().center());
    EXPECT_TRUE(shown);
    EXPECT_EQ(menus, 1);
    EXPECT_EQ(saved, 1);
    EXPECT_EQ(button.geometry(), before);

    // From the keyboard: Alt+Down, and the menu key as a context menu.
    const auto close_popup = [&shown] {
        shown = false;
        QTimer::singleShot(0, [&shown] {
            if (QWidget* popup = QApplication::activePopupWidget()) {
                shown = true;
                popup->close();
            }
        });
    };
    close_popup();
    QTest::keyClick(&button, Qt::Key_Down, Qt::AltModifier);
    EXPECT_TRUE(shown);
    EXPECT_EQ(menus, 2);
    close_popup();
    QContextMenuEvent from_keyboard(QContextMenuEvent::Keyboard, QPoint(2, 2), button.mapToGlobal(QPoint(2, 2)));
    QApplication::sendEvent(&button, &from_keyboard);
    EXPECT_TRUE(shown);
    EXPECT_EQ(menus, 3);
    EXPECT_EQ(saved, 1);

    // Words when there's room: down to its glyph and arrow, which still save
    // and open Save as... (a narrow window never grows for its words).
    EXPECT_TRUE(button.shows_words());
    EXPECT_LT(button.minimumSizeHint().width() + 40, button.sizeHint().width());
    button.resize(button.minimumSizeHint());
    EXPECT_FALSE(button.shows_words());
    EXPECT_EQ(button.accessibleName().toStdString(), "Save graph");
    QTest::mouseClick(&button, Qt::LeftButton, {}, QPoint(button.arrow_rect().left() / 2, button.height() / 2));
    EXPECT_EQ(saved, 2);
}

}  // namespace
}  // namespace vml::app
