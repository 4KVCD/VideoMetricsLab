// Notifications when a run ends: the message its templates make, when one
// is wanted, the requests to Pushover and Discord, and the secrets' storage.
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "vml/notifications.h"
#include "vml/settings.h"

namespace vml {
namespace {

ScoreStats stats_of(double mean, double minimum, double worst_1) {
    ScoreStats stats;
    stats.count = 100;
    stats.mean = mean;
    stats.median = mean;
    stats.minimum = minimum;
    stats.maximum = 100.0;
    stats.worst_1_percent = worst_1;
    return stats;
}

NotifiedRun sample_run() {
    NotifiedRun run;
    run.outcome = RunOutcome::finished_with_failures;
    run.outcome_text = "Finished with failures";
    run.seconds = 3725.4;
    run.reference = "reference.mkv";
    run.app = "VideoMetricsLab 2.7";
    NotifiedVideo first{"encode-a.mkv", "Scored", true, 1440, {}};
    first.metrics = {{"vmaf", stats_of(95.214, 80.1, 88.2)}, {"psnr", stats_of(44.1, 38.0, 40.0)}};
    NotifiedVideo second{"encode-b.mkv", "Failed: the file ends early", false, 0, {}};
    run.videos = {first, second};
    return run;
}

// The Standard message, and fields of every kind: the run's, each video's,
// a metric's statistics, scores in the app's own format and units; a
// failed video's scores are its result; unknown fields and stray braces
// stay as typed; a single-line video line joins videos line by line.
// An error's text loses a webhook's URL and its path, with or without the
// query: the path holds the token (the window's notifier quoted Qt's text
// with only the whole URL taken out).
TEST(Notifications, AnErrorLosesTheWebhooksURLAndPath) {
    const std::string url = "https://discord.com/api/webhooks/1/secret?wait=true";
    EXPECT_EQ(without_secret_url("Error transferring " + url + " - server replied", url, "Discord"),
              "Error transferring Discord - server replied");
    EXPECT_EQ(without_secret_url("POST /api/webhooks/1/secret?wait=true failed", url, "Discord"),
              "POST Discord failed");
    EXPECT_EQ(without_secret_url("path /api/webhooks/1/secret: refused", url, "Discord"), "path Discord: refused");
    EXPECT_EQ(without_secret_url("Host discord.com not found", url, "Discord"), "Host discord.com not found");
    EXPECT_EQ(without_secret_url("timed out", "https://api.pushover.net", "Pushover"), "timed out");
}

TEST(Notifications, TemplatesFillInTheRun) {
    const NotifiedRun run = sample_run();
    EXPECT_EQ(render_notification(standard_notification_title, run, standard_notification_video_line),
              "VideoMetricsLab 2.7: Finished with failures");
    EXPECT_EQ(render_notification(standard_notification_message, run, standard_notification_video_line),
              "reference.mkv\n1:02:05\n\n"
              "encode-a.mkv\nVMAF v0.6.1 95.21 · PSNR 44.10 dB\n\n"
              "encode-b.mkv\nFailed: the file ends early");
    EXPECT_EQ(render_notification("{videos}", run, "{name}: {vmaf.min} / {vmaf.worst_1} / {ssim} ({frames})"),
              "encode-a.mkv: 80.10 / 88.20 / — (1440)\nencode-b.mkv: — / — / — (0)");
    EXPECT_EQ(render_notification("{scored_count} of {video_count}, {failed_count} failed", run, ""),
              "1 of 2, 1 failed");
    EXPECT_EQ(render_notification("{nope} {vmaf} {a{outcome}} }{", run, ""),
              "{nope} {vmaf} {aFinished with failures} }{");
    // A video line left empty: no videos in the message.
    EXPECT_EQ(render_notification("{outcome}{videos}", run, ""), "Finished with failures");
}

// A message only for the outcomes asked for, for runs at least as long as
// asked, and only with a service that has its keys.
TEST(Notifications, WantedOnlyAsAsked) {
    Settings settings;
    EXPECT_FALSE(notification_wanted(settings, RunOutcome::finished, 600));  // no service yet
    settings.discord_enabled = true;
    settings.discord_webhook = "https://discord.com/api/webhooks/1/abc";
    EXPECT_TRUE(notification_wanted(settings, RunOutcome::finished, 0));
    settings.notify_cancelled = false;
    EXPECT_FALSE(notification_wanted(settings, RunOutcome::cancelled, 600));
    EXPECT_TRUE(notification_wanted(settings, RunOutcome::finished_with_failures, 600));
    settings.notify_min_minutes = 10;
    EXPECT_FALSE(notification_wanted(settings, RunOutcome::finished, 599));
    EXPECT_TRUE(notification_wanted(settings, RunOutcome::finished, 600));
    settings.discord_enabled = false;
    settings.pushover_enabled = true;
    settings.pushover_user = "user";
    EXPECT_FALSE(notification_wanted(settings, RunOutcome::finished, 600));  // no token
    settings.pushover_token = "token";
    EXPECT_TRUE(notification_wanted(settings, RunOutcome::finished, 600));
}

// Pushover's form: encoded, priority kept in range, a title and message cut
// to its limits by characters, an empty message replaced.
TEST(Notifications, PushoverRequest) {
    const NotificationRequest request = pushover_request("u k", "t&k", 5, "Done", "a+b=c é", "fallback");
    EXPECT_EQ(request.url, "https://api.pushover.net/1/messages.json");
    EXPECT_EQ(request.content_type, "application/x-www-form-urlencoded");
    EXPECT_EQ(request.body, "token=t%26k&user=u%20k&message=a%2Bb%3Dc%20%C3%A9&priority=1&title=Done");
    EXPECT_NE(pushover_request("u", "t", 0, "", " ", "Finished").body.find("message=Finished"), std::string::npos);
    const std::string long_text(2000, 'x');
    const NotificationRequest cut = pushover_request("u", "t", 0, "", long_text, "");
    EXPECT_NE(cut.body.find("message=" + std::string(1023, 'x') + "%E2%80%A6&"), std::string::npos);
    EXPECT_EQ(cut_to_characters("éééé", 3), "éé…");
    EXPECT_EQ(cut_to_characters("ééé", 3), "ééé");
}

// Discord: only to a Discord webhook; an embed coloured by the outcome,
// mentioning no one.
TEST(Notifications, DiscordRequest) {
    EXPECT_TRUE(is_discord_webhook("https://discord.com/api/webhooks/123/abc"));
    EXPECT_TRUE(is_discord_webhook("https://discordapp.com/api/webhooks/123/abc"));
    EXPECT_FALSE(is_discord_webhook("http://discord.com/api/webhooks/123/abc"));
    EXPECT_FALSE(is_discord_webhook("https://discord.com.example.org/api/webhooks/123/abc"));
    EXPECT_FALSE(is_discord_webhook("https://example.org/api/webhooks/123/abc"));
    EXPECT_FALSE(is_discord_webhook("https://discord.com/api/webhooks/"));
    const NotificationRequest request =
        discord_request("https://discord.com/api/webhooks/1/a", "Title", "@everyone done", RunOutcome::cancelled,
                        "VideoMetricsLab 2.7", "2026-11-19T20:00:00Z", "Cancelled");
    EXPECT_EQ(request.content_type, "application/json");
    const nlohmann::json body = nlohmann::json::parse(request.body);
    const nlohmann::json& embed = body["embeds"][0];
    EXPECT_EQ(embed["title"], "Title");
    EXPECT_EQ(embed["description"], "@everyone done");
    EXPECT_EQ(embed["color"], 0x95A5A6);
    EXPECT_EQ(embed["footer"]["text"], "VideoMetricsLab 2.7");
    EXPECT_EQ(embed["timestamp"], "2026-11-19T20:00:00Z");
    EXPECT_TRUE(body["allowed_mentions"]["parse"].empty());
    const nlohmann::json empty = nlohmann::json::parse(
        discord_request("https://discord.com/api/webhooks/1/a", "", "", RunOutcome::finished, "", "", "Finished")
            .body);
    EXPECT_EQ(empty["embeds"][0]["title"], "Finished");
}

// A request to each service that is on, or to the one Send test names;
// keys that cannot be read back, a missing key or a URL that is not a
// Discord webhook's send nothing, and say so.
TEST(Notifications, PlanSendsToTheServicesOn) {
    Settings settings;
    settings.pushover_enabled = true;
    settings.pushover_user = "user";
    settings.pushover_token = "token";
    settings.discord_webhook = "https://discord.com/api/webhooks/1/a";  // off
    const NotifiedRun run = sample_run();
    NotificationPlan plan = plan_notification(settings, run, "");
    ASSERT_EQ(plan.requests.size(), 1u);
    EXPECT_EQ(plan.requests[0].service, "Pushover");
    EXPECT_NE(plan.requests[0].body.find("title=VideoMetricsLab%202.7%3A%20Finished%20with%20failures"),
              std::string::npos);
    plan = plan_notification(settings, run, "", "Discord");
    ASSERT_EQ(plan.requests.size(), 1u);
    EXPECT_EQ(plan.requests[0].service, "Discord");
    settings.discord_enabled = true;
    settings.discord_webhook = "https://example.org/hook";
    settings.pushover_token = "dpapi:AAAA";  // another user's
    plan = plan_notification(settings, run, "");
    EXPECT_TRUE(plan.requests.empty());
    EXPECT_TRUE(plan.not_a_webhook);
    EXPECT_EQ(plan.unreadable, std::vector<std::string>{"Pushover"});
    settings.pushover_token.clear();
    EXPECT_EQ(plan_notification(settings, run, "").incomplete, std::vector<std::string>{"Pushover"});
}

// A secret kept as it is, or encrypted where the system can, and read back
// either way; one that cannot be read back is nothing, not its stored text.
TEST(Notifications, SecretsStoredAsChosen) {
    EXPECT_EQ(stored_secret("abc", false), "abc");
    EXPECT_EQ(secret_of("abc"), "abc");
    EXPECT_EQ(stored_secret("", true), "");
    const std::string sealed = stored_secret("https://discord.com/api/webhooks/1/abc", true);
    if (secrets_can_be_encrypted()) {
        EXPECT_TRUE(secret_is_encrypted(sealed));
        EXPECT_EQ(sealed.find("discord"), std::string::npos);
        EXPECT_EQ(secret_of(sealed), "https://discord.com/api/webhooks/1/abc");
        EXPECT_EQ(secret_of("dpapi:AAAA"), std::nullopt);
    } else {
        EXPECT_EQ(sealed, "https://discord.com/api/webhooks/1/abc");
    }
}

}  // namespace
}  // namespace vml
