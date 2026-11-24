// Sends notifications (notifications.h's requests) in the background, and
// says how each went: the window's run-end message and Settings' Send test.
// A failure stops nothing; the keys are never written anywhere.
#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include <vector>

#include "vml/notifications.h"

class QNetworkAccessManager;

namespace vml::app {

// A run to show a message's templates with, before there is one (Settings'
// preview and Send test): two encodes scored, its outcome `outcome`.
NotifiedRun sample_notified_run(const QString& outcome);

// The messages Settings > Notifications can start from, in the window's
// language: Status only (no file name or score), Standard (the default) and
// Detailed.
struct NotificationPreset {
    QString name;
    std::string title;
    std::string message;
    std::string video_line;
};
std::vector<NotificationPreset> notification_presets();

// What a plan could not send, in the window's language, one line each.
QStringList notification_problems(const NotificationPlan& plan);

class Notifier : public QObject {
    Q_OBJECT

public:
    explicit Notifier(QObject* parent = nullptr);

    // Sends each request; `sent` follows for each.
    void send(const std::vector<NotificationRequest>& requests);

    // The requests sent and not answered yet, a retry waiting for its turn
    // included: the window's close waits a moment for them.
    int pending() const { return pending_; }

signals:
    // How one request went: `error` empty when the service took it, else
    // why not (the service's own answer where it gave one), in the window's
    // language.
    void sent(const QString& service, const QString& error);

private:
    // Posts one request; `again` for the one retry a 429 that says how long
    // to wait gets.
    void post(const NotificationRequest& item, bool again);

    // Ends one of the pending requests, saying how it went.
    void finish(const QString& service, const QString& error);

    QNetworkAccessManager* network_;
    int pending_ = 0;
};

}  // namespace vml::app
