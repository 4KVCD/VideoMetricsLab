// --notify: the run's notification as it ends, as the app's Settings >
// Notifications say (notifications.h): to the services on there, with its
// message, for the outcomes and run lengths it is on for. What happened is
// said on the progress lines; a notification never changes the exit code.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "test_video.h"
#include "vml/notifications.h"
#include "vml/platform/http.h"
#include "vml/settings.h"

namespace vml::cli {

// A test video as the notification tells it: its name, how it ended (in the
// tables' words), and its metrics' statistics where it was scored.
NotifiedVideo notified_video(const TestVideo& video, bool cancelled);

// Sends a request; platform::http_post's, or a stand-in for the tests.
using HttpPost = std::function<platform::HttpResponse(const NotificationRequest&)>;

// What sending said, a line each: a problem (shown with -q too) or not.
struct NotifyLine {
    bool problem = false;
    std::string text;
};

// Sends `run`'s notification as `settings` say, through `post`, made at
// `utc_timestamp` (ISO 8601): whether it was sent to each service, or why
// not (nothing asked for this run, no service on, keys that cannot be read
// here, the service's own answer). Keys and webhook URLs are never in it.
std::vector<NotifyLine> notify_run_end(const Settings& settings, const NotifiedRun& run, const HttpPost& post,
                                       const std::string& utc_timestamp);

// platform::http_post for a request.
platform::HttpResponse post_notification(const NotificationRequest& request);

}  // namespace vml::cli
