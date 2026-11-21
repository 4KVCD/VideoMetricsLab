// --notify: the run's notification, as the app's Settings > Notifications
// say, and what is said about it; and curl's request (http_post elsewhere
// than on Windows), quoted as curl reads it.
#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

#include "notify.h"
#include "vml/platform/http.h"
#include "vml/results.h"

namespace vml::cli {
namespace {

NotifiedRun finished_run() {
    NotifiedRun run;
    run.outcome = RunOutcome::finished;
    run.outcome_text = "Finished";
    run.seconds = 600;
    run.reference = "reference.mkv";
    run.app = "VideoMetricsLab 2.7";
    run.videos.push_back({"encode.mkv", "Scored", true, 24, {}});
    return run;
}

std::vector<std::string> texts(const std::vector<NotifyLine>& lines) {
    std::vector<std::string> said;
    for (const NotifyLine& line : lines) {
        said.push_back((line.problem ? "! " : "") + line.text);
    }
    return said;
}

// Sent where Settings > Notifications ask for it, to each service on, and
// said; not sent, and said why, with no service on, for an outcome or a run
// length it is off for, or with keys another user saved; a service's
// refusal is said with its answer, and a webhook URL (its key), or its path
// with the token in it, never is.
TEST(CliNotify, SentAsTheSettingsSayAndSaid) {
    Settings settings;
    std::vector<NotificationRequest> posted;
    platform::HttpResponse answer{200, "{}", ""};
    const HttpPost post = [&](const NotificationRequest& request) {
        posted.push_back(request);
        return answer;
    };
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "2026-11-19T20:00:00Z")),
              std::vector<std::string>{"! No notification sent: no service is on in the app's Settings > Notifications"});
    settings.pushover_enabled = true;
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "")),
              std::vector<std::string>{"! No notification sent to Pushover: it has no keys yet; enter them in the "
                                       "app's Settings > Notifications"});
    EXPECT_TRUE(posted.empty());
    settings.pushover_user = "user";
    settings.pushover_token = "token";
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "2026-11-19T20:00:00Z")),
              std::vector<std::string>{"Notification sent to Pushover"});
    ASSERT_EQ(posted.size(), 1u);
    EXPECT_NE(posted[0].body.find("title=VideoMetricsLab%202.7%3A%20Finished"), std::string::npos);

    settings.notify_finished = false;
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "")),
              std::vector<std::string>{
                  "No notification: Settings > Notifications sends none for a run that ends Finished"});
    settings.notify_finished = true;
    settings.notify_min_minutes = 15;
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "")),
              std::vector<std::string>{
                  "No notification: the run was shorter than the 15 minutes Settings > Notifications asks for"});
    settings.notify_min_minutes = 0;

    answer = {400, R"({"errors":["application token is invalid"],"status":0})", ""};
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "")),
              std::vector<std::string>{
                  "! The notification to Pushover could not be sent: HTTP 400: application token is invalid"});

    settings.pushover_enabled = false;
    settings.discord_enabled = true;
    settings.discord_webhook = "https://discord.com/api/webhooks/1/secret";
    answer = {0, "", "Could not connect to https://discord.com/api/webhooks/1/secret"};
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "")),
              std::vector<std::string>{
                  "! The notification to Discord could not be sent: Could not connect to Discord"});
    // curl's request line (its --verbose) has the webhook's path, token and all.
    settings.discord_webhook = "https://discord.com/api/webhooks/1/secret?wait=true";
    answer = {0, "", "> POST /api/webhooks/1/secret?wait=true HTTP/2\n[:path: /api/webhooks/1/secret]"};
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "")),
              std::vector<std::string>{
                  "! The notification to Discord could not be sent: > POST Discord HTTP/2\n[:path: Discord]"});

    settings.discord_webhook = "dpapi:AAAA";  // another user's
    EXPECT_EQ(texts(notify_run_end(settings, finished_run(), post, "")),
              std::vector<std::string>{
                  "! No notification sent to Discord: its keys were saved by another Windows user or on another "
                  "PC; enter them again in the app's Settings > Notifications"});
}

// A test video as its notification tells it: scored, with its metrics'
// statistics; failed, with why; not reached in a cancelled run.
TEST(CliNotify, EachVideoAsTheTablesSayIt) {
    TestVideo scored;
    scored.path = "encode.mkv";
    scored.metrics = {"vmaf"};
    scored.result.emplace();
    scored.result->compared_frame_count = 3;
    FrameMetricResult vmaf;
    vmaf.key = "vmaf";
    vmaf.frame = {0, 1, 2};
    vmaf.time = {0.0, 0.04, 0.08};
    vmaf.values = {90.0F, 80.0F, 70.0F};
    scored.result->metric_results.add(vmaf);
    const NotifiedVideo told = notified_video(scored, false);
    EXPECT_EQ(told.name, "encode.mkv");
    EXPECT_EQ(told.result, "Scored");
    EXPECT_TRUE(told.scored);
    EXPECT_EQ(told.frames, 3);
    ASSERT_EQ(told.metrics.size(), 1u);
    EXPECT_DOUBLE_EQ(told.metrics[0].second.minimum, 70.0);

    TestVideo failed;
    failed.path = "broken.mkv";
    failed.metrics = {"vmaf"};
    failed.error = "the file ends early";
    EXPECT_EQ(notified_video(failed, false).result, "Failed: the file ends early");
    EXPECT_FALSE(notified_video(failed, false).scored);

    TestVideo waiting;
    waiting.path = "later.mkv";
    waiting.metrics = {"vmaf"};
    EXPECT_EQ(notified_video(waiting, true).result, "Cancelled");
}

// curl's request: every value quoted as curl reads it, so that a quote, a
// backslash or a line break in the message stays in the message.
TEST(CliNotify, CurlReadsTheRequestAsSent) {
    const std::string config = platform::curl_config({"https://api.pushover.net/1/messages.json",
                                                      "application/x-www-form-urlencoded",
                                                      "message=a \"b\" \\ c\nd",
                                                      {{"Access-Token", "k \"1\""}}},
                                                     std::chrono::seconds(20));
    EXPECT_NE(config.find("url = \"https://api.pushover.net/1/messages.json\"\n"), std::string::npos);
    EXPECT_NE(config.find("proto = \"=https\"\n"), std::string::npos);
    EXPECT_NE(config.find("header = \"Content-Type: application/x-www-form-urlencoded\"\n"), std::string::npos);
    EXPECT_NE(config.find("data-binary = \"message=a \\\"b\\\" \\\\ c\\nd\"\n"), std::string::npos);
    EXPECT_NE(config.find("max-time = 20\n"), std::string::npos);
    EXPECT_NE(config.find("write-out = \"\\n%{http_code}\"\n"), std::string::npos);
    EXPECT_NE(config.find("header = \"Access-Token: k \\\"1\\\"\"\n"), std::string::npos);
    // http:// only for a server of the user's own (Gotify, ntfy, a webhook).
    EXPECT_NE(platform::curl_config({"http://gotify.lan/message", "application/json", "{}", {}, true},
                                    std::chrono::seconds(20))
                  .find("proto = \"=https,http\"\n"),
              std::string::npos);
}

// curl reads only that request: -q, as its first argument (the only place
// curl takes it), keeps the user's ~/.curlrc (`insecure`, `location`,
// `verbose`) out of it.
TEST(CliNotify, CurlReadsNoCurlrc) {
    EXPECT_EQ(platform::curl_command(7), (std::vector<std::string>{"curl", "-q", "-K", "/dev/fd/7"}));
}

}  // namespace
}  // namespace vml::cli
