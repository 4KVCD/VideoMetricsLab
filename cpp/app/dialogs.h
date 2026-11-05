// The dialogs that wait for the user (QMessageBox, QFileDialog), behind
// functions: the window's tests answer them in the user's place, as the test
// harness fails any dialog that opens.
#pragma once

#include <QString>
#include <QStringList>

#include <functional>
#include <optional>

class QWidget;

namespace vml::app {

// The answer to the film grain question at Calculate metrics.
struct FilmGrainAnswer {
    bool with = true;             // scored with the videos' film grain
    bool dont_ask_again = false;  // the answer made Settings' default
};

// The frame rate entered for a raw stream that stores none.
struct FrameRateAnswer {
    double fps = 0;
    bool for_the_others = false;  // also for the other videos without one
};

struct Dialogs {
    // A message with an OK button: information, and a failure (critical).
    std::function<void(const QString& title, const QString& text)> inform;
    std::function<void(const QString& title, const QString& text)> alert;
    // A question answered Yes or No: whether Yes.
    std::function<bool(const QString& title, const QString& text)> ask;
    // The same with No the default button, for a question whose No keeps
    // things as they are: Enter does not undo work (whether Yes).
    std::function<bool(const QString& title, const QString& text)> ask_no_default;
    // The file dialogs; an empty answer is Cancel.
    std::function<QString(const QString& title, const QString& filter)> open_file;
    std::function<QString(const QString& title, const QString& suggested, const QString& filter)> save_file;
    // A folder, the dialog opening at `start` (none: wherever it opens).
    std::function<QString(const QString& title, const QString& start)> choose_folder;
    // Whether the `videos` named, which have film grain and no choice for it
    // yet, are scored with it or without: nothing for Cancel.
    std::function<std::optional<FilmGrainAnswer>(const QStringList& videos)> ask_film_grain;
    // The frame rate of `video`, a raw stream that stores none: nothing for
    // Cancel. `reference_fps` (0: not known) is shown, and `others` videos more
    // have none.
    std::function<std::optional<FrameRateAnswer>(const QString& video, double reference_fps, int others)>
        ask_frame_rate;
};

// The real dialogs, over `parent`.
Dialogs standard_dialogs(QWidget* parent);

}  // namespace vml::app
