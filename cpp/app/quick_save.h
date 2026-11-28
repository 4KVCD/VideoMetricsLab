// The quick saves (issue #224, docs/ARCHITECTURE.md, "Exports"): each
// card's save icon (SaveButton), which saves what the card shows at one click
// into the export folder, in the formats Settings' Quick saves names, and
// says how it went on the indicator (export_indicator.h).
#pragma once

#include <QIcon>
#include <QString>
#include <QStringList>
#include <QPixmap>
#include <QPushButton>
#include <QWidget>

#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "export_indicator.h"

class QMenu;

namespace vml {
struct Settings;
}

namespace vml::app {

class FileWrites;
struct QuickFile;

// Runs a quick save's `save` off the window's thread, then `done` with how
// it went, on the window's thread.
using QuickSaveRunner = std::function<void(std::function<ExportOutcome()> save, ExportReport done)>;

// What a tab's save icons need of the window: the export folder they save in
// (Settings' Export folder, or Documents where it is blank), the formats a
// quick save writes data in (quick_formats), and where its files are written
// (`background`: the window's file writer, file_writes.h). Unset: no folder,
// no format, written at once.
struct QuickSaves {
    std::function<QString()> folder;
    std::function<std::vector<std::string>()> formats;
    QuickSaveRunner background;

    QString export_folder() const { return folder ? folder() : QString(); }
    std::vector<std::string> data_formats() const { return formats ? formats() : std::vector<std::string>{}; }
    // Writes `files` in `into` (save_quick) by `background`, then says how
    // it went to `done` (report_to). A 2-hour film's bitrate data takes a few
    // hundred ms to make and write, more on a slow export folder: the window
    // answers meanwhile.
    void save(const QString& into, std::vector<QuickFile> files, const ExportReport& done) const;
};

// QuickSaves' background on `writes` (the window's file writer), after the
// writes queued before it. One that throws (out of memory, say) says so as
// its error.
QuickSaveRunner written_by(FileWrites& writes);

// The formats quick saves write data in, Settings' Quick saves
// (Settings::quick_save_formats), in Settings' order: JSON, then CSV.
std::vector<std::string> quick_formats(const Settings& settings);

// "JSON and CSV", "JSON" or "CSV": the formats as a tooltip names them.
QString formats_text(const std::vector<std::string>& formats);

// The refusal of a data save while Settings' Quick saves ticks no format.
QString no_format_text();
// The refusal of a save of a card that shows nothing yet.
QString nothing_shown_text();

// What a save icon's tooltip says it saves: "Save {what}, as {formats}, in the
// export folder.", or, for an image or images alone, "Save {what} in the
// export folder."; then where Save as... is. `formats` empty: images alone.
QString quick_save_tooltip(const QString& what, const std::vector<std::string>& formats);

// Marks `view` (a table, list or chart) as not data to save: a list of files
// to read, a list to pick from. Every other one is on a card with a save icon
// (the appearance tests check it: EveryCardThatShowsDataHasASaveIcon).
void not_exported(QWidget* view);
// Whether `view` is marked so.
bool is_not_exported(const QWidget* view);

// One file of a quick save: its name, and what writes it at the path given
// (the error, empty when written). `write` runs off the window's thread
// (QuickSaves::save): it must not touch a widget, so it holds plain data,
// taken when the save icon was clicked.
struct QuickFile {
    QString name;
    std::function<QString(const QString& path)> write;
};

// A data file for each format (quick_formats' keys) of a quick save, named
// `stem` with the format's extension, its text made by `text` for that
// format now: `text` may read the card.
std::vector<QuickFile> data_files(const QString& stem, const std::vector<std::string>& formats,
                                  const std::function<std::string(const std::string& format)>& text);
// An image's file, `name`, drawn by `image` now, on the window's thread.
QuickFile image_file(const QString& name, const std::function<QPixmap()>& image);

// Where the files named go in `folder`: each name as it is, or all numbered
// alike where any is taken, on disk or in `reserved` (unique_output_names):
// never over a file already there.
std::vector<QString> quick_save_paths(const QString& folder, const std::vector<QString>& names,
                                      std::set<std::filesystem::path>& reserved);

// Writes `files` in `folder` at their quick_save_paths, in order, stopping at
// the first that fails: the files written, or why not.
ExportOutcome save_quick(const QString& folder, const std::vector<QuickFile>& files);

// The tab bar's Export button's icon (issue #224's mockup): an arrow rising
// out of a tray, drawn in `button`'s text color (grayed when it is
// disabled) each time it is painted, as SaveButton draws its disk: it
// follows every theme in both interfaces, with no icon file to keep. The
// glyph is square, at the left of the icon's size: any more is room before
// the button's text.
QIcon export_icon(QWidget* button);

// A card's save icon, in its top-right corner: one click saves what the card
// shows (`save`); a right-click offers Save as... (`save_as` fills the menu).
// Greyed while `why_not` gives a reason, which its tooltip says; else its
// tooltip is `describe`'s. Looked at again as the pointer comes to it, and on
// refresh(). A push button, so that VML Modern's style sheet draws it as every
// other button (its "icon" role: narrower padding, a gap from the button
// beside it).
class SaveButton : public QPushButton {
    Q_OBJECT

public:
    explicit SaveButton(QWidget* parent = nullptr);

    std::function<QString()> why_not;
    std::function<QString()> describe;
    std::function<void()> save;
    std::function<void(QMenu&)> save_as;

    // Greyed or not, and its tooltip, as things are now.
    void refresh();
    // What it saves, or why it can't: its tooltip's first line, as plain
    // text (the window may show tooltips as rich text).
    QString summary() const;

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

protected:
    bool event(QEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
};

}  // namespace vml::app
