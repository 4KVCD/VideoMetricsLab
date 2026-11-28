#include "update_offer.h"

#include <QAbstractButton>
#include <QDesktopServices>
#include <QHBoxLayout>
#include <QLabel>
#include <QNetworkAccessManager>
#include <QNetworkProxyFactory>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPushButton>
#include <QScrollArea>
#include <QUrl>
#include <QVBoxLayout>

#include <utility>

#include "i18n.h"
#include "popup_panel.h"
#include "theme.h"
#include "title_bar.h"
#include "vml/app_log.h"
#include "vml/python_text.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

const app_log::Logger logger("vml.main_window");  // the window's: the update check's lines are as they were

}  // namespace

UpdateOffer::UpdateOffer(Window window) : QObject(window.widget), window_(std::move(window)) {}

void UpdateOffer::check() {
    if (!window_.settings.check_for_updates) {
        return;
    }
    // Through the system's proxy, if it names one, as the Python app's
    // urllib asked.
    QNetworkProxyFactory::setUseSystemConfiguration(true);
    auto* network = new QNetworkAccessManager(this);
    QNetworkRequest request(QUrl(QString::fromUtf8(releases_url.data(), static_cast<qsizetype>(releases_url.size()))));
    request.setRawHeader("Accept", "application/vnd.github+json");
    // GitHub refuses requests without a User-Agent.
    request.setHeader(QNetworkRequest::UserAgentHeader, "VideoMetricsLab/" VML_VERSION);
    request.setTransferTimeout(update_check_timeout_seconds * 1000);
    QNetworkReply* reply = network->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, network] {
        const QString error = reply->error() == QNetworkReply::NoError ? QString() : reply->errorString();
        const QByteArray body = reply->readAll();
        reply->deleteLater();
        network->deleteLater();
        on_answer(body.toStdString(), error);
    });
}

void UpdateOffer::show_pending() {
    if (pending_) {
        show(*pending_);
    }
}

void UpdateOffer::on_answer(const std::string& body, const QString& error) {
    std::optional<Release> release;
    try {
        if (!error.isEmpty()) {
            throw UpdateCheckError(error.toStdString());
        }
        release = release_from_answer(body);
    } catch (const UpdateCheckError& failure) {
        logger.info(std::string("Update check: GitHub could not be asked (") + failure.what() + ")");
        return;
    }
    if (!is_newer(release->version, VML_VERSION)) {
        logger.info("Update check: VideoMetricsLab " VML_VERSION " is the latest release");
        return;
    }
    if (!offers_package_here(*release)) {
        logger.info("Update check: VideoMetricsLab " + release->version +
                    " is available, but not yet for this system (this is " VML_VERSION ")");
        return;
    }
    logger.info("Update check: VideoMetricsLab " + release->version + " is available (this is " VML_VERSION ")");
    offer(*release);
}

void UpdateOffer::offer(const Release& release) {
    // Nothing is shown for a release the user chose to skip.
    if (release.version == window_.settings.skipped_update_version) {
        return;
    }
    if (TitleButtons* badge = window_.badge()) {
        // VML Modern: a badge in the title bar, not a dialog taking the focus
        // at the start; a click on it shows the dialog.
        pending_ = release;
        badge->set_update(QString::fromStdString(release.version));
        // Opened on the badge's press, as Add/remove metrics' panel is on
        // its button's (popup_panel.h): connected once a badge.
        auto* button = qobject_cast<QAbstractButton*>(badge->update_button());
        if (button != nullptr && hooked_badge_ != button) {
            hooked_badge_ = button;
            PopupPanel::open_on_press(button, [this] { show_pending(); });
        }
        return;
    }
    show(release);
}

static QString update_headline(const Release& release) {
    return app::tr("{app} {version} is available. You have {current}.",
                   {{"app", "VideoMetricsLab"}, {"version", QString::fromStdString(release.version)},
                    {"current", VML_VERSION}});
}

QString UpdateOffer::keeps_text() const {
    return app::tr("Updating does not affect your saved results or settings: they are stored "
                   "separately from the app, in {cache_dir} and {user_data_dir}.",
                   {{"cache_dir", text_of(window_.cache_folder())},
                    {"user_data_dir", text_of(Settings::file().parent_path())}});
}

void UpdateOffer::download(const Release& release) const {
    // This system's package itself, where the release names it (the Python
    // app opened the page).
    QDesktopServices::openUrl(QUrl(QString::fromStdString(release.download_url.value_or(release.page_url))));
}

void UpdateOffer::skip(const Release& release) {
    window_.settings.skipped_update_version = release.version;
    window_.save_settings();
    pending_.reset();
    if (TitleButtons* badge = window_.badge()) {
        badge->set_update({});
    }
}

void UpdateOffer::show_panel(const Release& release, QWidget* badge) {
    // Made again each time: a release the badge offers is the one shown.
    delete panel_;
    auto* panel = new PopupPanel(window_.widget);
    panel->setAttribute(Qt::WA_DeleteOnClose);
    panel->setAccessibleName(app::tr("Update available"));
    auto* layout = new QVBoxLayout(panel);
    auto* heading = new QLabel(app::tr("Update available"));
    theme::set_role(heading, "title");
    layout->addWidget(heading);
    auto* headline = new QLabel(update_headline(release));
    headline->setTextFormat(Qt::PlainText);
    headline->setWordWrap(true);
    theme::style(headline, "font-weight: 600;");
    layout->addWidget(headline);
    auto* text = new QLabel(keeps_text());
    text->setTextFormat(Qt::PlainText);
    text->setWordWrap(true);
    theme::set_role(text, "hint");
    layout->addWidget(text);
    if (!python_strip(release.notes).empty()) {
        // What's new, scrolled where it is long, selectable to copy.
        auto* notes = new QLabel(QString::fromStdString(release.notes));
        notes->setObjectName(QStringLiteral("updateNotes"));
        notes->setTextFormat(Qt::PlainText);
        notes->setWordWrap(true);
        notes->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
        auto* scroll = new QScrollArea;
        scroll->setWidget(notes);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setMaximumHeight(panel->fontMetrics().height() * 14);
        layout->addWidget(scroll, 1);
    }
    auto* buttons = new QHBoxLayout;
    auto* download_button = new QPushButton(app::tr("Download"));
    theme::set_role(download_button, "primary");
    auto* skip_button = new QPushButton(app::tr("Skip this version"));
    auto* later_button = new QPushButton(app::tr("Later"));
    buttons->addStretch(1);
    for (QPushButton* button : {download_button, skip_button, later_button}) {
        buttons->addWidget(button);
    }
    layout->addLayout(buttons);
    connect(download_button, &QPushButton::clicked, panel, [this, panel, release] {
        panel->close();
        download(release);
    });
    connect(skip_button, &QPushButton::clicked, panel, [this, panel, release] {
        panel->close();
        skip(release);
    });
    connect(later_button, &QPushButton::clicked, panel, &QWidget::close);
    // Long lines wrapped, and wide enough that the text reads as lines.
    panel->setMinimumWidth(panel->fontMetrics().averageCharWidth() * 72);
    panel->setMaximumWidth(panel->fontMetrics().averageCharWidth() * 80);
    panel_ = panel;
    panel->open_under(badge);
    download_button->setFocus();
}

void UpdateOffer::show(const Release& release) {
    if (TitleButtons* badge = window_.badge(); badge != nullptr && badge->update_button()->isVisible()) {
        show_panel(release, badge->update_button());
        return;
    }
    if (box_ != nullptr) {
        box_->raise();
        return;
    }
    Message message;
    message.title = app::tr("Update available");
    message.headline = update_headline(release);
    message.text = keeps_text();
    if (!python_strip(release.notes).empty()) {
        message.details = QString::fromStdString(release.notes);
    }
    constexpr int download_answer = 0;
    constexpr int skip_answer = 1;
    message.buttons = {{app::tr("Download"), MessageButton::Role::primary},
                       {app::tr("Skip this version"), MessageButton::Role::plain},
                       {app::tr("Later"), MessageButton::Role::plain}};
    // Not waited for: the window goes on while it is open.
    AppDialog* box = message_dialog(window_.widget, message, [this, release](MessageAnswer answer) {
        if (answer.button == download_answer) {
            download(release);
        } else if (answer.button == skip_answer) {
            skip(release);
        }
    });
    box_ = box;
    box->open();
}

}  // namespace vml::app
