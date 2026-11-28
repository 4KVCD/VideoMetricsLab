// FFmpeg's setup (issue #239): the one dialog for an ffmpeg the app can't
// use. It says what is wrong and offers to download FFmpeg (Windows) or to
// point at an ffmpeg.exe; what goes wrong on the way (a download that
// failed, a build still not usable, ffprobe not beside ffmpeg, the two in
// different folders) is said on its own line, the dialog staying open to
// try again, where each was a window of its own. It closes once FFmpeg is
// usable, or on Not now.
#pragma once

#include <QString>

#include "app_dialog.h"

class QLabel;
class QPushButton;

namespace vml::app {

class FfmpegSetupDialog : public AppDialog {
    Q_OBJECT

public:
    // `text`: what is wrong and where the tools were looked for.
    FfmpegSetupDialog(QWidget* parent, const QString& text);

    // What is wrong, again (looked at once more).
    void set_text(const QString& text);
    // What went wrong on the last try, in the failure colour; empty: none.
    void show_problem(const QString& problem);
    // A download is running: its buttons wait for it.
    void set_busy(bool busy);

    QLabel* problem_line() const { return problem_; }
    QPushButton* download_button() const { return download_; }  // none outside Windows
    QPushButton* locate_button() const { return locate_; }

signals:
    void download_requested();
    void locate_requested();

private:
    QLabel* text_ = nullptr;
    QLabel* problem_ = nullptr;
    QPushButton* download_ = nullptr;
    QPushButton* locate_ = nullptr;
};

}  // namespace vml::app
