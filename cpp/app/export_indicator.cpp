#include "export_indicator.h"

#include <QDesktopServices>
#include <QEvent>
#include <QGraphicsOpacityEffect>
#include <QHBoxLayout>
#include <QLabel>
#include <QPropertyAnimation>
#include <QSaveFile>
#include <QTimer>
#include <QToolTip>
#include <QUrl>

#include "i18n.h"
#include "theme.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

// How long "Saved" stays before it fades, and how long the fade takes.
constexpr int saved_shown_ms = 6000;
constexpr int fade_ms = 600;
// The most of a failure's reason the indicator shows.
constexpr int failure_shown_chars = 90;

}  // namespace

void report_to(const ExportReport& report, const ExportOutcome& outcome) {
    if (report && (!outcome.files.isEmpty() || !outcome.error.isEmpty())) {
        report(outcome);
    }
}

QString could_not_write(const QString& path) { return app::tr("Could not write {path}", {{"path", path}}); }

QString write_text_file(const QString& path, const std::string& text) {
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) ||
        file.write(text.data(), static_cast<qint64>(text.size())) != static_cast<qint64>(text.size()) ||
        !file.commit()) {
        return could_not_write(path);
    }
    return {};
}

ExportOutcome save_as(const QString& path, const std::function<QString(const QString&)>& write) {
    ExportOutcome outcome;
    if (path.isEmpty()) {
        return outcome;  // Cancel
    }
    outcome.error = write(path);
    outcome.details = outcome.error;
    if (outcome.error.isEmpty()) {
        outcome.files.append(path);
    }
    return outcome;
}

// ------------------------------------------------------------ the indicator

QString saved_text(const QStringList& files) {
    if (files.size() == 1) {
        return app::tr("Saved {name}", {{"name", name_of(path_of(files.front()))}});
    }
    return app::ntr("Saved {count} file", "Saved {count} files", files.size(), {{"count", files.size()}});
}

QString saved_folder(const QStringList& files) {
    return files.isEmpty() ? QString() : text_of(path_of(files.front()).parent_path());
}

ExportIndicator::ExportIndicator(QWidget* parent) : QWidget(parent) {
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(6, 0, 6, 0);
    label_ = new QLabel;
    label_->setTextFormat(Qt::RichText);
    label_->setTextInteractionFlags(Qt::LinksAccessibleByMouse | Qt::LinksAccessibleByKeyboard);
    label_->setFocusPolicy(Qt::NoFocus);  // never takes the focus from what the user is doing
    connect(label_, &QLabel::linkActivated, this, [this](const QString& link) {
        dismiss_pending_ = false;
        if (link == QStringLiteral("open")) {
            if (open_folder) {
                open_folder(saved_folder(outcome_.files));
            }
        } else if (link == QStringLiteral("details")) {
            // The whole reason, beside the indicator: no window. The failure
            // stays shown until the indicator itself is clicked.
            if (show_details) {
                show_details(outcome_.error, outcome_.details);
            }
        }
    });
    label_->installEventFilter(this);
    layout->addWidget(label_);
    opacity_ = new QGraphicsOpacityEffect(this);
    opacity_->setOpacity(1.0);
    setGraphicsEffect(opacity_);
    fading_ = new QPropertyAnimation(opacity_, "opacity", this);
    fading_->setDuration(fade_ms);
    fading_->setStartValue(1.0);
    fading_->setEndValue(0.0);
    connect(fading_, &QPropertyAnimation::finished, this, [this] {
        if (state_ == State::saved) {
            state_ = State::idle;
            hide();
        }
    });
    fade_timer_ = new QTimer(this);
    fade_timer_->setSingleShot(true);
    fade_timer_->setInterval(saved_shown_ms);
    connect(fade_timer_, &QTimer::timeout, this, [this] { fade(); });
    open_folder = [](const QString& folder) { QDesktopServices::openUrl(QUrl::fromLocalFile(folder)); };
    show_details = [this](const QString& error, const QString& details) {
        // Headed as the windows that said it were; it is in the log too
        // (MainWindow::report_export).
        QToolTip::showText(mapToGlobal(QPoint(0, height())),
                           app::tr("Export failed") + QStringLiteral("\n\n") + (details.isEmpty() ? error : details),
                           this);
    };
    hide();
}

void ExportIndicator::show_saving(const QString& why) {
    // A failure shown stays until clicked, or until the next export: what
    // is said is always of the latest.
    state_ = State::saving;
    failed_ = false;
    saving_why_ = why;
    fade_timer_->stop();
    fading_->stop();
    opacity_->setOpacity(1.0);
    show_text();
}

void ExportIndicator::show_outcome(const ExportOutcome& outcome) {
    if (outcome.error.isEmpty() && outcome.files.isEmpty()) {
        if (state_ == State::saving) {  // cancelled: nothing to say
            state_ = State::idle;
            hide();
        }
        return;
    }
    fading_->stop();
    opacity_->setOpacity(1.0);
    outcome_ = outcome;
    failed_ = !outcome.error.isEmpty();
    state_ = failed_ ? State::failed : State::saved;
    show_text();
    if (failed_) {
        fade_timer_->stop();
    } else {
        fade_timer_->start();
    }
}

QString ExportIndicator::text() const { return label_->text(); }

void ExportIndicator::show_text() {
    const auto link = [](const QString& target, const QString& text) {
        return QStringLiteral("<a href=\"%1\" style=\"color: %2;\">%3</a>")
            .arg(target, theme::color("link").name(), text.toHtmlEscaped());
    };
    const QString dot = QStringLiteral(" &middot; ");
    QString text;
    switch (state_) {
    case State::idle:
        hide();
        return;
    case State::saving:
        text = QStringLiteral("<span style=\"color: %1;\">%2</span>")
                   .arg(theme::color("muted").name(), app::tr("Saving...").toHtmlEscaped());
        break;
    case State::saved:
        text = QStringLiteral("<span style=\"color: %1;\">&#x2713;</span> %2")
                   .arg(theme::color("good").name(), saved_text(outcome_.files).toHtmlEscaped()) +
               dot + link(QStringLiteral("open"), app::tr("Open folder"));
        break;
    case State::failed: {
        // A long reason cut short, so that the tabs keep their room: all of
        // it is in Details and the tooltip.
        QString why = outcome_.error.simplified();
        if (why.size() > failure_shown_chars) {
            why = why.left(failure_shown_chars - 1).trimmed() + QChar(0x2026);
        }
        text = QStringLiteral("<span style=\"color: %1;\">&#x2715; <b>%2</b> %3</span>")
                   .arg(theme::color("failed").name(), app::tr("Not saved:").toHtmlEscaped(), why.toHtmlEscaped()) +
               dot + link(QStringLiteral("details"), app::tr("Details"));
        break;
    }
    }
    label_->setText(text);
    label_->setToolTip(state_ == State::failed  ? outcome_.details
                       : state_ == State::saved ? outcome_.files.join(QLatin1Char('\n'))
                                                : saving_why_);
    show();
}

void ExportIndicator::fade() {
    if (state_ == State::saved) {
        fading_->start();
    }
}

bool ExportIndicator::eventFilter(QObject* watched, QEvent* event) {
    // A click on a failure's text dismisses it, but not one on a link:
    // Details shows the reason, and the failure stays with it. Seen here
    // before the label takes the release, which then says whether a link was
    // clicked (linkActivated): so it is looked at once that is done.
    if (watched == label_ && event->type() == QEvent::MouseButtonRelease && state_ == State::failed) {
        dismiss_pending_ = true;
        QTimer::singleShot(0, this, [this] {
            if (dismiss_pending_) {
                dismiss_failure();
            }
        });
    }
    return QWidget::eventFilter(watched, event);
}

void ExportIndicator::dismiss_failure() {
    if (state_ == State::failed) {
        state_ = State::idle;
        failed_ = false;
        QToolTip::hideText();
        hide();
    }
}

void ExportIndicator::changeEvent(QEvent* event) {
    // Its colors are the theme's, set in its text: made again with them.
    if ((event->type() == QEvent::PaletteChange || event->type() == QEvent::StyleChange) && state_ != State::idle) {
        show_text();
    }
    QWidget::changeEvent(event);
}

}  // namespace vml::app
