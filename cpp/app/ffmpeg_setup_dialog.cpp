#include "ffmpeg_setup_dialog.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "i18n.h"
#include "theme.h"
#include "window_helpers.h"

namespace vml::app {

FfmpegSetupDialog::FfmpegSetupDialog(QWidget* parent, const QString& text)
    : AppDialog(parent, app::tr("ffmpeg not usable")) {
    auto* layout = new QVBoxLayout(body());
    auto* top = new QHBoxLayout;
    auto* icon = new QLabel;
    icon->setPixmap(message_icon(MessageKind::warning, 32, devicePixelRatioF()));
    icon->setAlignment(Qt::AlignTop);
    top->addWidget(icon);
    text_ = new QLabel(text);
    text_->setTextFormat(Qt::PlainText);
    text_->setWordWrap(true);
    text_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    text_->setMinimumWidth(fontMetrics().averageCharWidth() * 60);
    top->addWidget(text_, 1);
    layout->addLayout(top);
    // What went wrong on the last try: here, not a window over this one.
    problem_ = new QLabel;
    problem_->setObjectName(QStringLiteral("ffmpegSetupProblem"));
    problem_->setTextFormat(Qt::PlainText);
    problem_->setWordWrap(true);
    problem_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    theme::style(problem_, "color: {failed}; font-weight: 600;");
    problem_->setVisible(false);
    layout->addWidget(problem_);
    auto* buttons = new QHBoxLayout;
    buttons->addStretch(1);
#ifdef _WIN32
    download_ = new QPushButton(app::tr("Download FFmpeg"));
    theme::set_role(download_, "primary");
    connect(download_, &QPushButton::clicked, this, &FfmpegSetupDialog::download_requested);
    buttons->addWidget(download_);
#endif
    locate_ = new QPushButton(app::with_program_names(app::tr("Locate ffmpeg.exe")));
    connect(locate_, &QPushButton::clicked, this, &FfmpegSetupDialog::locate_requested);
    buttons->addWidget(locate_);
    auto* later = new QPushButton(app::tr("Not now"));
    connect(later, &QPushButton::clicked, this, &QDialog::reject);
    buttons->addWidget(later);
    layout->addLayout(buttons);
    (download_ != nullptr ? download_ : locate_)->setDefault(true);
}

void FfmpegSetupDialog::set_text(const QString& text) { text_->setText(text); }

void FfmpegSetupDialog::show_problem(const QString& problem) {
    problem_->setText(problem);
    problem_->setVisible(!problem.isEmpty());
    adjustSize();
}

void FfmpegSetupDialog::set_busy(bool busy) {
    for (QPushButton* button : {download_, locate_}) {
        if (button != nullptr) {
            button->setEnabled(!busy);
        }
    }
}

}  // namespace vml::app
