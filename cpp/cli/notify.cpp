#include "notify.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include "report.h"

namespace vml::cli {

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
    if (std::none_of(notification_services().begin(), notification_services().end(),
                     [&settings](std::string_view service) { return service_enabled(settings, service); })) {
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
    for (const std::string& name : plan.bad_address) {
        lines.push_back({true, "No notification sent to " + name +
                                   ": its address isn't one it takes; check it in the app's Settings > Notifications"});
    }
    for (const NotificationRequest& request : plan.requests) {
        platform::HttpResponse answer = post(request);
        // Discord asks to wait a moment when sent to too often: once, a short wait.
        if (const std::optional<double> wait = retry_after(answer.status, answer.body); wait && *wait <= 5.0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(std::llround(*wait * 1000)));
            answer = post(request);
        }
        if (answer.status >= 200 && answer.status < 300) {
            lines.push_back({false, "Notification sent to " + request.service});
            continue;
        }
        const std::string why =
            answer.status == 0 ? answer.error
                               : "HTTP " + std::to_string(answer.status) +
                                     (answer.body.empty() ? std::string() : ": " + service_answer(answer.body));
        lines.push_back({true, "The notification to " + request.service +
                                   " could not be sent: " + without_secrets(why, request)});
    }
    return lines;
}

platform::HttpResponse post_notification(const NotificationRequest& request) {
    return platform::http_post({request.url, request.content_type, request.body, request.headers, request.plain_http});
}

}  // namespace vml::cli
