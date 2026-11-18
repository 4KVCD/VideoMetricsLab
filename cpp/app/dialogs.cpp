#include "dialogs.h"

#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>

#include "i18n.h"
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
    QDialog dialog(parent);
    dialog.setWindowTitle(tr("Film grain"));
    auto* layout = new QVBoxLayout(&dialog);
    auto* text = new QLabel(tr("These videos tell the player to add film grain. Calculate their metrics:"), &dialog);
    text->setWordWrap(true);
    layout->addWidget(text);
    layout->addWidget(new QLabel(named_videos(videos), &dialog));
    auto* with = new QRadioButton(tr("With film grain (what viewers see)"), &dialog);
    auto* without = new QRadioButton(tr("Without film grain (the coded pictures only)"), &dialog);
    with->setChecked(true);
    layout->addWidget(with);
    layout->addWidget(without);
    auto* again = new QCheckBox(tr("Don't ask again (change it in Settings)"), &dialog);
    layout->addWidget(again);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Ok)->setText(tr("Calculate metrics"));
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
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
    QDialog dialog(parent);
    dialog.setWindowTitle(tr("Enter the frame rate"));
    auto* layout = new QVBoxLayout(&dialog);
    auto* text = new QLabel(tr("{video} is a raw video stream that does not store its frame rate. Enter it as a number "
                               "or a fraction, such as 24 or 24000/1001:",
                               {{"video", video}}),
                            &dialog);
    text->setTextFormat(Qt::PlainText);  // a file's name is not markup
    text->setWordWrap(true);
    layout->addWidget(text);
    auto* entry = new QLineEdit(&dialog);
    layout->addWidget(entry);
    if (reference_fps > 0) {
        layout->addWidget(new QLabel(tr("The reference is {fps:.3f} fps.", {{"fps", reference_fps}}), &dialog));
    }
    auto* for_the_others = new QCheckBox(ntr("Use it for {count} other video without a frame rate",
                                             "Use it for {count} other videos without a frame rate",
                                             static_cast<long long>(others)),
                                         &dialog);
    for_the_others->setVisible(others > 0);
    layout->addWidget(for_the_others);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    QPushButton* ok = buttons->button(QDialogButtonBox::Ok);
    ok->setEnabled(false);
    QObject::connect(entry, &QLineEdit::textChanged, ok,
                     [ok](const QString& typed) { ok->setEnabled(entered_frame_rate(typed).has_value()); });
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted) {
        return std::nullopt;
    }
    return FrameRateAnswer{*entered_frame_rate(entry->text()), for_the_others->isChecked()};
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
    return {
        [parent](const QString& title, const QString& text) { QMessageBox::information(parent, title, text); },
        [parent](const QString& title, const QString& text) { QMessageBox::critical(parent, title, text); },
        [parent](const QString& title, const QString& text) {
            return QMessageBox::question(parent, title, text) == QMessageBox::Yes;
        },
        [parent](const QString& title, const QString& text) {
            return QMessageBox::question(parent, title, text, QMessageBox::Yes | QMessageBox::No, QMessageBox::No) ==
                   QMessageBox::Yes;
        },
        [parent](const QString& title, const QString& filter) {
            return QFileDialog::getOpenFileName(parent, title, QString(), filter);
        },
        [parent, export_folder](const QString& title, const QString& suggested, const QString& filter) {
            return QFileDialog::getSaveFileName(parent, title, save_dialog_start(suggested, export_folder()), filter);
        },
        [parent](const QString& title, const QString& start) {
            return QFileDialog::getExistingDirectory(parent, title, start);
        },
        [parent](const QStringList& videos) { return ask_film_grain(parent, videos); },
        [parent](const QString& video, double reference_fps, int others) {
            return ask_frame_rate(parent, video, reference_fps, others);
        },
        export_folder,
        [parent](const QString& title, const QString& label, const QString& text) -> std::optional<QString> {
            bool ok = false;
            const QString answer = QInputDialog::getText(parent, title, label, QLineEdit::Normal, text, &ok);
            return ok ? std::optional(answer) : std::nullopt;
        },
    };
}

}  // namespace vml::app
