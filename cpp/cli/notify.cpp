#include "notify.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <utility>

#include "report.h"

namespace vml::cli {

namespace {

// What a service said about a message it refused: Pushover's "errors",
// Discord's "message", else the start of its answer on one line.
std::string service_answer(const std::string& body) {
    const nlohmann::json answer = nlohmann::json::parse(body, nullptr, false);
    if (answer.is_object()) {
        if (answer.contains("errors") && answer["errors"].is_array() && !answer["errors"].empty()) {
            std::string joined;
            for (const nlohmann::json& error : answer["errors"]) {
                if (error.is_string()) {
                    joined += (joined.empty() ? "" : "; ") + error.get<std::string>();
                }
            }
            if (!joined.empty()) {
                return joined;
            }
        }
        if (answer.contains("message") && answer["message"].is_string()) {
            return answer["message"].get<std::string>();
        }
    }
    std::string start = body.substr(0, 200);
    std::replace(start.begin(), start.end(), '\n', ' ');
    std::replace(start.begin(), start.end(), '\r', ' ');
    return start;
}

// `text` with each `secret` in it replaced by `name`.
std::string replaced(std::string text, const std::string& secret, const std::string& name) {
    if (secret.empty()) {
        return text;
    }
    for (std::size_t at = text.find(secret); at != std::string::npos; at = text.find(secret, at + name.size())) {
        text.replace(at, secret.size(), name);
    }
    return text;
}

// `text` with `url` (a webhook's URL, which is its key) replaced by `name`,
// and its path too: what curl's messages show of it when they show the
// request (`POST /api/webhooks/<id>/<token>`), with or without the query.
std::string without(std::string text, const std::string& url, const std::string& name) {
    text = replaced(std::move(text), url, name);
    const std::size_t scheme = url.find("://");
    const std::size_t path = url.find('/', scheme == std::string::npos ? 0 : scheme + 3);
    if (path == std::string::npos || path + 1 >= url.size()) {
        return text;  // no path: nothing more to hide
    }
    text = replaced(std::move(text), url.substr(path), name);
    return replaced(std::move(text), url.substr(path, url.find('?', path) - path), name);
}

}  // namespace

NotifiedVideo notified_video(const TestVideo& video, bool cancelled) {
    NotifiedVideo notified;
    notified.name = video.name();
    const std::string status = video_status(video, cancelled);
    const std::string why = !video.error.empty()         ? video.error
                            : !video.failures.empty() ? video.failures.front().first + ": " +
                                                            video.failures.front().second
                                                      : std::string();
    notified.result = status == "ok"          ? "Scored"
                      : status == "cancelled" ? "Cancelled"
                      : status == "partial"   ? "Partly failed" + (why.empty() ? "" : ": " + why)
                                              : "Failed" + (why.empty() ? "" : ": " + why);
    notified.scored = video.result && (status == "ok" || status == "partial");
    if (notified.scored) {
        notified.frames = video.result->compared_frame_count;
        notified.metrics = notified_metrics(*video.result);
    }
    return notified;
}

std::vector<NotifyLine> notify_run_end(const Settings& settings, const NotifiedRun& run, const HttpPost& post,
                                       const std::string& utc_timestamp) {
    if (!settings.pushover_enabled && !settings.discord_enabled) {
        return {{true, "No notification sent: no service is on in the app's Settings > Notifications"}};
    }
    const bool outcome_on = run.outcome == RunOutcome::finished                 ? settings.notify_finished
                            : run.outcome == RunOutcome::finished_with_failures ? settings.notify_with_failures
                                                                                : settings.notify_cancelled;
    if (!outcome_on) {
        return {{false, "No notification: Settings > Notifications sends none for a run that ends " +
                            run.outcome_text}};
    }
    if (run.seconds < settings.notify_min_minutes * 60.0) {
        return {{false, "No notification: the run was shorter than the " +
                            std::to_string(settings.notify_min_minutes) + " minutes Settings > Notifications asks for"}};
    }
    // A service on without its keys, or with keys that cannot be read here,
    // is said below (plan_notification).
    const NotificationPlan plan = plan_notification(settings, run, utc_timestamp);
    std::vector<NotifyLine> lines;
    for (const std::string& name : plan.unreadable) {
        lines.push_back({true, "No notification sent to " + name +
                                   ": its keys were saved by another Windows user or on another PC; enter them again "
                                   "in the app's Settings > Notifications"});
    }
    for (const std::string& name : plan.incomplete) {
        lines.push_back({true, "No notification sent to " + name +
                                   ": it has no keys yet; enter them in the app's Settings > Notifications"});
    }
    if (plan.not_a_webhook) {
        lines.push_back({true, "No notification sent to Discord: its webhook URL isn't a Discord webhook's"});
    }
    for (const NotificationRequest& request : plan.requests) {
        const platform::HttpResponse answer = post(request);
        if (answer.status >= 200 && answer.status < 300) {
            lines.push_back({false, "Notification sent to " + request.service});
            continue;
        }
        const std::string why =
            answer.status == 0 ? answer.error
                               : "HTTP " + std::to_string(answer.status) +
                                     (answer.body.empty() ? std::string() : ": " + service_answer(answer.body));
        lines.push_back({true, "The notification to " + request.service +
                                   " could not be sent: " + without(why, request.url, request.service)});
    }
    return lines;
}

platform::HttpResponse post_notification(const NotificationRequest& request) {
    return platform::http_post(request.url, request.content_type, request.body);
}

}  // namespace vml::cli
