#include "dialogs.h"

#include <QCheckBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>

#include "app_dialog.h"
#include "i18n.h"
#include "theme.h"
#include "vml/ffprobe.h"
#include "vml/platform/system.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

// The videos named in a dialog, at most so many, then how many more.
QString named_videos(const QStringList& videos) {
    constexpr int shown = 8;
    QStringList lines;
    for (int i = 0; i < videos.size() && i < shown; ++i) {
        lines.append(QStringLiteral("• ") + videos[i]);
    }
    if (videos.size() > shown) {
        lines.append(ntr("and {count} more", "and {count} more", static_cast<long long>(videos.size() - shown)));
    }
    return lines.join('\n');
}

std::optional<FilmGrainAnswer> ask_film_grain(QWidget* parent, const QStringList& videos) {
    AppDialog dialog(parent, tr("Film grain"));
    auto* layout = new QVBoxLayout(dialog.body());
    layout->setContentsMargins(20, 18, 20, 16);
    auto* text = new QLabel(tr("These videos tell the player to add film grain. Calculate their metrics:"));
    text->setWordWrap(true);
    layout->addWidget(text);
    layout->addWidget(new QLabel(named_videos(videos)));
    auto* with = new QRadioButton(tr("With film grain (what viewers see)"));
    auto* without = new QRadioButton(tr("Without film grain (the coded pictures only)"));
    with->setChecked(true);
    layout->addWidget(with);
    layout->addWidget(without);
    auto* again = new QCheckBox(tr("Don't ask again (change it in Settings)"));
    layout->addWidget(again);
    add_dialog_buttons(dialog, layout, tr("Calculate metrics"));
    if (dialog.exec() != QDialog::Accepted) {
        return std::nullopt;
    }
    return FilmGrainAnswer{with->isChecked(), again->isChecked()};
}

// The rate in `text`, where it is one someone could mean.
std::optional<double> entered_frame_rate(const QString& text) {
    const std::optional<double> rate = typed_frame_rate(text.toStdString());
    if (rate && *rate >= lowest_typed_frame_rate && *rate <= highest_typed_frame_rate) {
        return rate;
    }
    return std::nullopt;
}

std::optional<FrameRateAnswer> ask_frame_rate(QWidget* parent, const QString& video, double reference_fps, int others) {
    AppDialog dialog(parent, tr("Enter the frame rate"));
    auto* layout = new QVBoxLayout(dialog.body());
    layout->setContentsMargins(20, 18, 20, 16);
    auto* text = new QLabel(tr("{video} is a raw video stream that does not store its frame rate. Enter it as a number "
                               "or a fraction, such as 24 or 24000/1001:",
                               {{"video", video}}));
    text->setTextFormat(Qt::PlainText);  // a file's name is not markup
    text->setWordWrap(true);
    layout->addWidget(text);
    auto* entry = new QLineEdit;
    layout->addWidget(entry);
    if (reference_fps > 0) {
        layout->addWidget(new QLabel(tr("The reference is {fps:.3f} fps.", {{"fps", reference_fps}})));
    }
    auto* for_the_others = new QCheckBox(ntr("Use it for {count} other video without a frame rate",
                                             "Use it for {count} other videos without a frame rate",
                                             static_cast<long long>(others)));
    for_the_others->setVisible(others > 0);
    layout->addWidget(for_the_others);
    QPushButton* ok = add_dialog_buttons(dialog, layout);
    ok->setEnabled(false);
    QObject::connect(entry, &QLineEdit::textChanged, ok,
                     [ok](const QString& typed) { ok->setEnabled(entered_frame_rate(typed).has_value()); });
    if (dialog.exec() != QDialog::Accepted) {
        return std::nullopt;
    }
    return FrameRateAnswer{*entered_frame_rate(entry->text()), for_the_others->isChecked()};
}

std::optional<CloseAnswer> ask_close_during_run(QWidget* parent) {
    Message message;
    message.kind = MessageKind::warning;
    message.title = tr("A run is in progress");
    message.text = tr("Closing the window stops the run, and the videos still being scored are left unfinished.");
    message.buttons = {{tr("Stop and close"), MessageButton::Role::danger},
                       {tr("Keep running"), MessageButton::Role::primary}};
    message.default_button = 1;  // Enter leaves the run going
    message.escape_button = 1;
    message.check = tr("Don't ask again (change it in Settings)");
    const MessageAnswer answer = show_message(parent, message);
    if (answer.button != 0) {
        return std::nullopt;
    }
    return CloseAnswer{answer.checked};
}

}  // namespace

QString save_dialog_start(const QString& suggested, const QString& folder) {
    const std::filesystem::path name = path_of(suggested);
    return name.is_relative() ? text_of(path_of(folder) / name) : suggested;
}

Dialogs standard_dialogs(QWidget* parent, std::function<std::string()> export_dir) {
    // Read at each export, so that a change in Settings applies at once.
    const auto export_folder = [export_dir = std::move(export_dir)] {
        const std::string chosen = export_dir ? export_dir() : std::string();
        return text_of(chosen.empty() ? platform::home_directory() / "Documents" : platform::path_from_text(chosen));
    };
    // A message with OK alone, of `kind`.
    const auto tell = [parent](MessageKind kind) {
        return [parent, kind](const QString& title, const QString& text) {
            Message message;
            message.kind = kind;
            message.title = title;
            message.text = text;
            show_message(parent, message);
        };
    };
    // A question answered Yes or No, Enter's `yes_default` or No.
    const auto question = [parent](bool yes_default) {
        return [parent, yes_default](const QString& title, const QString& text) {
            Message message;
            message.kind = MessageKind::question;
            message.title = title;
            message.text = text;
            message.buttons = {{tr("Yes"), yes_default ? MessageButton::Role::primary : MessageButton::Role::plain},
                               {tr("No"), MessageButton::Role::plain}};
            message.default_button = yes_default ? 0 : 1;
            message.escape_button = 1;
            return show_message(parent, message).button == 0;
        };
    };
    Dialogs dialogs;
    dialogs.inform = tell(MessageKind::information);
    dialogs.alert = tell(MessageKind::failure);
    dialogs.warn = tell(MessageKind::warning);
    dialogs.message = [parent](const Message& message) { return show_message(parent, message); };
    dialogs.ask = question(true);
    dialogs.ask_no_default = question(false);
    dialogs.open_file = [parent](const QString& title, const QString& filter) {
        return QFileDialog::getOpenFileName(parent, title, QString(), filter);
    };
    dialogs.open_file_at = [parent](const QString& title, const QString& start, const QString& filter) {
        return QFileDialog::getOpenFileName(parent, title, start, filter);
    };
    dialogs.open_files_at = [parent](const QString& title, const QString& start) {
        return QFileDialog::getOpenFileNames(parent, title, start);
    };
    dialogs.save_file = [parent, export_folder](const QString& title, const QString& suggested, const QString& filter) {
        return QFileDialog::getSaveFileName(parent, title, save_dialog_start(suggested, export_folder()), filter);
    };
    dialogs.choose_folder = [parent](const QString& title, const QString& start) {
        return QFileDialog::getExistingDirectory(parent, title, start);
    };
    dialogs.ask_film_grain = [parent](const QStringList& videos) { return ask_film_grain(parent, videos); };
    dialogs.ask_frame_rate = [parent](const QString& video, double reference_fps, int others) {
        return ask_frame_rate(parent, video, reference_fps, others);
    };
    dialogs.ask_close_during_run = [parent] { return ask_close_during_run(parent); };
    dialogs.export_folder = export_folder;
    dialogs.ask_text = [parent](const QString& title, const QString& label, const QString& text) {
        return ask_line(parent, title, label, text);
    };
    return dialogs;
}

}  // namespace vml::app
