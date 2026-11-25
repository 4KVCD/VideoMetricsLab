#include "update_offer.h"

#include <QDesktopServices>
#include <QNetworkAccessManager>
#include <QNetworkProxyFactory>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>

#include <utility>

#include "i18n.h"
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
        return;
    }
    show(release);
}

void UpdateOffer::show(const Release& release) {
    if (box_ != nullptr) {
        box_->raise();
        return;
    }
    Message message;
    message.title = app::tr("Update available");
    message.headline = app::tr("{app} {version} is available. You have {current}.",
                               {{"app", "VideoMetricsLab"},
                                {"version", QString::fromStdString(release.version)},
                                {"current", VML_VERSION}});
    message.text = app::tr("Updating does not affect your saved results or settings: they are stored "
                           "separately from the app, in {cache_dir} and {user_data_dir}.",
                           {{"cache_dir", text_of(window_.cache_folder())},
                            {"user_data_dir", text_of(Settings::file().parent_path())}});
    if (!python_strip(release.notes).empty()) {
        message.details = QString::fromStdString(release.notes);
    }
    constexpr int download = 0;
    constexpr int skip = 1;
    message.buttons = {{app::tr("Download"), MessageButton::Role::primary},
                       {app::tr("Skip this version"), MessageButton::Role::plain},
                       {app::tr("Later"), MessageButton::Role::plain}};
    // Not waited for: the window goes on while it is open.
    AppDialog* box = message_dialog(window_.widget, message, [this, release](MessageAnswer answer) {
        if (answer.button == download) {
            // This system's package itself, where the release names it (the
            // Python app opened the page).
            QDesktopServices::openUrl(QUrl(QString::fromStdString(release.download_url.value_or(release.page_url))));
        } else if (answer.button == skip) {
            window_.settings.skipped_update_version = release.version;
            window_.save_settings();
            pending_.reset();
            if (TitleButtons* badge = window_.badge()) {
                badge->set_update({});
            }
        }
    });
    box_ = box;
    box->open();
}

}  // namespace vml::app
