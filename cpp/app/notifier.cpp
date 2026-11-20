#include "notifier.h"

#include <QByteArray>
#include <QDateTime>
#include <QLocale>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>

#include <algorithm>

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

// What a service said about a message it refused: Pushover's "errors",
// Discord's "message", else the start of its answer.
QString service_answer(const QByteArray& body) {
    const QJsonObject answer = QJsonDocument::fromJson(body).object();
    if (const QJsonArray errors = answer.value("errors").toArray(); !errors.isEmpty()) {
        QStringList texts;
        for (const QJsonValue& error : errors) {
            texts << error.toString();
        }
        return texts.join("; ");
    }
    if (const QString message = answer.value("message").toString(); !message.isEmpty()) {
        return message;
    }
    return QString::fromUtf8(body.left(200)).simplified();
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
    if (plan.not_a_webhook) {
        lines << app::tr("The Discord webhook URL isn't a Discord webhook's: copy it from the channel's settings, "
                         "Integrations > Webhooks.");
    }
    return lines;
}

Notifier::Notifier(QObject* parent) : QObject(parent), network_(new QNetworkAccessManager(this)) {}

void Notifier::send(const std::vector<NotificationRequest>& requests) {
    for (const NotificationRequest& item : requests) {
        QNetworkRequest request(QUrl(QString::fromStdString(item.url)));
        request.setHeader(QNetworkRequest::ContentTypeHeader, QByteArray::fromStdString(item.content_type));
        request.setHeader(QNetworkRequest::UserAgentHeader, "VideoMetricsLab/" VML_VERSION);
        // A webhook's URL is its key: never followed elsewhere.
        request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::ManualRedirectPolicy);
        request.setTransferTimeout(20000);
        QNetworkReply* reply = network_->post(request, QByteArray::fromStdString(item.body));
        const QString service = QString::fromStdString(item.service);
        const QString url = QString::fromStdString(item.url);
        connect(reply, &QNetworkReply::finished, this, [this, reply, service, url] {
            reply->deleteLater();
            const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            if (status >= 200 && status < 300) {
                emit sent(service, QString());
                return;
            }
            const QString answer = service_answer(reply->readAll());
            if (status == 0) {
                // Qt's text can hold the URL, which for a webhook is its key.
                emit sent(service, reply->errorString().replace(url, service));
            } else {
                emit sent(service, answer.isEmpty() ? QStringLiteral("HTTP %1").arg(status)
                                                    : QStringLiteral("HTTP %1: %2").arg(status).arg(answer));
            }
        });
    }
}

}  // namespace vml::app
