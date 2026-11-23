#include "notifier.h"

#include <QByteArray>
#include <QDateTime>
#include <QLocale>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QTimer>
#include <QUrl>

#include <algorithm>
#include <cmath>
#include <optional>

#include "i18n.h"
#include "row_state.h"

namespace vml::app {

namespace {

ScoreStats sample_stats(double mean, double minimum, double worst_1) {
    ScoreStats stats;
    stats.count = 1440;
    stats.mean = mean;
    stats.median = mean;
    stats.minimum = minimum;
    stats.maximum = std::max(mean, 99.0);
    stats.worst_10_percent = (mean + worst_1) / 2;
    stats.worst_5_percent = (mean + 2 * worst_1) / 3;
    stats.worst_1_percent = worst_1;
    stats.worst_0_1_percent = (worst_1 + minimum) / 2;
    return stats;
}

}  // namespace

NotifiedRun sample_notified_run(const QString& outcome) {
    NotifiedRun run;
    run.outcome = RunOutcome::finished;
    run.outcome_text = outcome.toStdString();
    run.seconds = 2 * 3600 + 14 * 60 + 9;
    run.reference = "reference.mkv";
    run.finished_at = QLocale().toString(QDateTime::currentDateTime(), QLocale::ShortFormat).toStdString();
    run.app = "VideoMetricsLab " VML_VERSION;
    const QString scored = row_state_shown(RowState::complete);
    NotifiedVideo first{"encode-crf18.mkv", scored.toStdString(), true, 1440, {}};
    first.metrics = {{"vmaf", sample_stats(96.84, 88.12, 91.37)}, {"ssimulacra2", sample_stats(84.21, 70.55, 76.02)}};
    NotifiedVideo second{"encode-crf24.mkv", scored.toStdString(), true, 1440, {}};
    second.metrics = {{"vmaf", sample_stats(93.07, 79.40, 84.66)}, {"ssimulacra2", sample_stats(78.93, 61.20, 68.48)}};
    run.videos = {first, second};
    return run;
}

std::vector<NotificationPreset> notification_presets() {
    return {
        {app::tr("Status only"), "{app}: {outcome}", "{duration}", ""},
        {app::tr("Standard"), std::string(standard_notification_title), std::string(standard_notification_message),
         std::string(standard_notification_video_line)},
        {app::tr("Detailed"), "{app}: {outcome} ({scored_count}/{video_count})",
         app::tr("Reference: {reference}\nEnded {finished}, after {duration}\n\n{videos}").toStdString(),
         app::tr("{name}: {result}, {frames} frames\nMean: {scores}\nMinimum: {scores.min}\nWorst 1%: "
                 "{scores.worst_1}")
             .toStdString()},
    };
}

QStringList notification_problems(const NotificationPlan& plan) {
    QStringList lines;
    for (const std::string& service : plan.unreadable) {
        lines << app::tr("{service}'s keys were saved by another Windows user or on another PC: enter them again in "
                         "Settings > Notifications.",
                         {{"service", QString::fromStdString(service)}});
    }
    for (const std::string& service : plan.incomplete) {
        lines << app::tr("{service} has no keys yet: enter them in Settings > Notifications.",
                         {{"service", QString::fromStdString(service)}});
    }
    for (const std::string& service : plan.bad_address) {
        lines << (service == "Discord"
                      ? app::tr("The Discord webhook URL isn't a Discord webhook's: copy it from the channel's "
                                "settings, Integrations > Webhooks.")
                  : service == "Slack"
                      ? app::tr("The Slack webhook URL isn't an incoming webhook's: copy it from the app's Incoming "
                                "Webhooks page (https://hooks.slack.com/...).")
                      : app::tr("{service}'s address isn't one it takes: check it in Settings > Notifications.",
                                {{"service", QString::fromStdString(service)}}));
    }
    return lines;
}

Notifier::Notifier(QObject* parent) : QObject(parent), network_(new QNetworkAccessManager(this)) {}

void Notifier::send(const std::vector<NotificationRequest>& requests) {
    for (const NotificationRequest& item : requests) {
        ++pending_;
        post(item, false);
    }
}

void Notifier::finish(const QString& service, const QString& error) {
    --pending_;
    emit sent(service, error);
}

void Notifier::post(const NotificationRequest& item, bool again) {
    const QUrl url(QString::fromStdString(item.url));
    const QString service = QString::fromStdString(item.service);
    if (url.scheme() != "https" && !(item.plain_http && url.scheme() == "http")) {
        finish(service, item.plain_http ? app::tr("its address isn't an http or https URL")
                                           : app::tr("its address isn't an https URL"));
        return;
    }
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, QByteArray::fromStdString(item.content_type));
    request.setHeader(QNetworkRequest::UserAgentHeader, "VideoMetricsLab/" VML_VERSION);
    for (const auto& [name, value] : item.headers) {
        request.setRawHeader(QByteArray::fromStdString(name), QByteArray::fromStdString(value));
    }
    // A webhook's URL is its key: never followed elsewhere.
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::ManualRedirectPolicy);
    request.setTransferTimeout(20000);
    QNetworkReply* reply = network_->post(request, QByteArray::fromStdString(item.body));
    connect(reply, &QNetworkReply::finished, this, [this, reply, item, service, again] {
        reply->deleteLater();
        const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (status >= 200 && status < 300) {
            finish(service, QString());
            return;
        }
        const std::string body = reply->readAll().toStdString();
        // Discord asks to wait a moment when sent to too often: once, a short wait.
        if (const std::optional<double> wait = retry_after(status, body); wait && *wait <= 5.0 && !again) {
            QTimer::singleShot(static_cast<int>(std::lround(*wait * 1000)), this, [this, item] { post(item, true); });
            return;
        }
        // Qt's text can hold the URL, which for a webhook is its key, or its
        // path alone (the token in it); an answer could repeat a key.
        const std::string why = status == 0 ? reply->errorString().toStdString()
                                : body.empty() ? "HTTP " + std::to_string(status)
                                               : "HTTP " + std::to_string(status) + ": " + service_answer(body);
        finish(service, QString::fromStdString(without_secrets(why, item)));
    });
}

}  // namespace vml::app
