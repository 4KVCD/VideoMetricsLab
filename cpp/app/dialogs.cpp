#include "dialogs.h"

#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>

#include "i18n.h"

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

}  // namespace

Dialogs standard_dialogs(QWidget* parent) {
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
        [parent](const QString& title, const QString& suggested, const QString& filter) {
            return QFileDialog::getSaveFileName(parent, title, suggested, filter);
        },
        [parent](const QString& title, const QString& start) {
            return QFileDialog::getExistingDirectory(parent, title, start);
        },
        [parent](const QStringList& videos) { return ask_film_grain(parent, videos); },
    };
}

}  // namespace vml::app
