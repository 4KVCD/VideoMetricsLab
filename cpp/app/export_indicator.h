// How the window's exports say how they went (issue #224): the indicator
// at the tab bar's right end (ExportIndicator), in place of the
// windows that said "Export complete" or "Export failed" and of the status
// lines that said "Exported to ...". No export opens a window to say it
// worked.
#pragma once

#include <QString>
#include <QStringList>
#include <QWidget>

#include <functional>
#include <string>

class QLabel;
class QGraphicsOpacityEffect;
class QPropertyAnimation;
class QTimer;

namespace vml::app {

// How an export went: the files it wrote, or why it wrote none (`error`, a
// line; `details`, all of it, shown on asking).
struct ExportOutcome {
    QStringList files;
    QString error;
    QString details;
};

// Where an export says how it went: the indicator's (ExportIndicator::show_outcome).
using ExportReport = std::function<void(const ExportOutcome&)>;

// Says `outcome` to `report`, where set and where something was written or
// failed (not a Save dialog's Cancel).
void report_to(const ExportReport& report, const ExportOutcome& outcome);

// Writes `text` to `path` whole (QSaveFile): the error, empty when written.
QString write_text_file(const QString& path, const std::string& text);

// A file of the Save as... dialog's, written by `write` (the error, empty when
// written): its outcome, nothing written where `path` is empty (Cancel).
ExportOutcome save_as(const QString& path, const std::function<QString(const QString&)>& write);

// "Could not write {path}", the error a file that could not be written gives.
QString could_not_write(const QString& path);

// What the indicator says of a save that went through: "Saved 2 files", or the
// file's name where it is one.
QString saved_text(const QStringList& files);

// The folder the files were saved in, for Open folder: the first file's.
QString saved_folder(const QStringList& files);

// At the tab bar's right end: "Saving..." while an export is written,
// "Saved 2 files · Open folder" once it is (fading after a few seconds, with
// no window, sound or focus taken), and "Not saved: {why} · Details" in the
// error color where it failed, kept until clicked.
class ExportIndicator : public QWidget {
    Q_OBJECT

public:
    explicit ExportIndicator(QWidget* parent = nullptr);

    // An export is being written; `why` its tooltip (what it waits for).
    void show_saving(const QString& why = {});
    // How it went.
    void show_outcome(const ExportOutcome& outcome);
    // A line of what a click did, or why it did nothing, where the window
    // has no line beside it to say so (a row's mark on the Videos tab):
    // shown as a save that went through is, fading. It replaces what was
    // shown, but for a failure, which stays until clicked.
    void show_note(const QString& note);

    // For the tests: the text shown (rich text), and whether a failure is.
    QString text() const;
    bool failed() const { return failed_; }
    // Opens the folder, or shows the details (a link clicked): the whole
    // reason in a tooltip beside it, never a window.
    std::function<void(const QString& folder)> open_folder;
    std::function<void(const QString& error, const QString& details)> show_details;

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void changeEvent(QEvent* event) override;

private:
    void show_text();
    // A failure shown, clicked away.
    void dismiss_failure();
    void fade();

    QLabel* label_ = nullptr;
    QTimer* fade_timer_ = nullptr;
    QGraphicsOpacityEffect* opacity_ = nullptr;
    QPropertyAnimation* fading_ = nullptr;
    enum class State { idle, saving, saved, failed, note } state_ = State::idle;
    ExportOutcome outcome_;
    QString note_;
    QString saving_why_;
    bool dismiss_pending_ = false;  // a click to dismiss a failure, unless it was on a link
    bool failed_ = false;
};

}  // namespace vml::app
