// Notifications when a run ends: the message its templates make, when one
// is wanted, the requests to each service, what an error may show, and the
// secrets' storage.
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <utility>
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

// The Standard message, and fields of every kind: the run's, each video's,
// a metric's statistics, scores in the app's own format and units; a
// failed video's scores are its result; unknown fields and stray braces
// stay as typed; a single-line video line joins videos line by line.
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
// keys that cannot be read back, a missing key or an address a service
// doesn't take send nothing, and say so.
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
    EXPECT_EQ(plan.bad_address, std::vector<std::string>{"Discord"});
    EXPECT_EQ(plan.unreadable, std::vector<std::string>{"Pushover"});
    settings.pushover_token.clear();
    EXPECT_EQ(plan_notification(settings, run, "").incomplete, std::vector<std::string>{"Pushover"});

    // The other services: each on its own, with what it needs, or not.
    Settings others;
    others.gotify_enabled = others.ntfy_enabled = others.pushbullet_enabled = others.slack_enabled =
        others.telegram_enabled = others.webhook_enabled = true;
    others.gotify_url = "http://192.168.1.5:8080";
    others.gotify_token = "gtoken";
    others.ntfy_topic = "vml_runs";
    others.pushbullet_token = "ptoken";
    others.slack_webhook = "https://hooks.slack.com/services/T1/B2/c3";
    others.telegram_token = "123:abc";
    others.telegram_chat = "-100200";
    others.webhook_url = "https://example.org/hook";
    plan = plan_notification(others, run, "");
    std::vector<std::string> sent;
    for (const NotificationRequest& request : plan.requests) {
        sent.push_back(request.service);
    }
    EXPECT_EQ(sent, (std::vector<std::string>{"Gotify", "ntfy", "Pushbullet", "Slack", "Telegram", "Webhook"}));
    others.telegram_chat.clear();
    others.ntfy_topic = "not a topic";
    others.slack_webhook = "https://example.org/services/T1";
    others.webhook_header = "no colon";
    plan = plan_notification(others, run, "");
    EXPECT_EQ(plan.incomplete, std::vector<std::string>{"Telegram"});
    EXPECT_EQ(plan.bad_address, (std::vector<std::string>{"ntfy", "Slack", "Webhook"}));
    others.webhook_header = " : no name";
    EXPECT_EQ(plan_notification(others, run, "", "Webhook").bad_address, std::vector<std::string>{"Webhook"});
    EXPECT_TRUE(notification_wanted(others, RunOutcome::finished, 0));
}

// Each service's request: where, its headers, and a body it takes, cut to
// its limits.
TEST(Notifications, EachServicesRequest) {
    using Headers = std::vector<std::pair<std::string, std::string>>;
    const NotificationRequest gotify = gotify_request("http://gotify.lan/", "tok", 12, "Title", "Body", "Done");
    EXPECT_EQ(gotify.url, "http://gotify.lan/message");
    EXPECT_TRUE(gotify.plain_http);
    EXPECT_EQ(gotify.headers, (Headers{{"X-Gotify-Key", "tok"}}));
    EXPECT_EQ(nlohmann::json::parse(gotify.body),
              (nlohmann::json{{"title", "Title"}, {"message", "Body"}, {"priority", 10}}));

    const NotificationRequest ntfy = ntfy_request("https://ntfy.sh", "vml_runs", "tk_1", 9, RunOutcome::cancelled,
                                                  "Title", std::string(5000, 'x'), "Cancelled");
    EXPECT_EQ(ntfy.url, "https://ntfy.sh/");  // JSON is published to the root
    EXPECT_EQ(ntfy.headers, (Headers{{"Authorization", "Bearer tk_1"}}));
    const nlohmann::json ntfy_body = nlohmann::json::parse(ntfy.body);
    EXPECT_EQ(ntfy_body["topic"], "vml_runs");
    EXPECT_EQ(ntfy_body["priority"], 5);
    EXPECT_EQ(ntfy_body["tags"], nlohmann::json::array({"no_entry_sign"}));
    EXPECT_EQ(ntfy_body["message"].get<std::string>().size(), 4096u);  // 4093 bytes and "…"
    EXPECT_TRUE(ntfy_request("https://ntfy.sh", "t", "", 3, RunOutcome::finished, "", "m", "").headers.empty());

    const NotificationRequest pushbullet = pushbullet_request("ptok", "dev1", "", "Body", "Finished");
    EXPECT_EQ(pushbullet.url, "https://api.pushbullet.com/v2/pushes");
    EXPECT_EQ(pushbullet.headers, (Headers{{"Access-Token", "ptok"}}));
    EXPECT_EQ(nlohmann::json::parse(pushbullet.body),
              (nlohmann::json{{"type", "note"}, {"title", "Finished"}, {"body", "Body"}, {"device_iden", "dev1"}}));

    // Slack: plain text, so "<!channel>" or "*" are not markup; 3000 a section.
    const std::string long_text = "<!channel> " + std::string(6000, 'y');
    const nlohmann::json slack =
        nlohmann::json::parse(slack_request("https://hooks.slack.com/services/a", "Done", long_text, "").body);
    EXPECT_EQ(slack["text"], "Done");
    ASSERT_EQ(slack["blocks"].size(), 4u);  // the header, then 3000 + 3000 + 11 characters
    EXPECT_EQ(slack["blocks"][0]["text"], (nlohmann::json{{"type", "plain_text"}, {"text", "Done"}}));
    EXPECT_EQ(slack["blocks"][1]["text"]["type"], "plain_text");
    EXPECT_EQ(slack["blocks"][1]["text"]["text"].get<std::string>().size(), 3000u);
    EXPECT_EQ(slack["blocks"][3]["text"]["text"], std::string(11, 'y'));
    EXPECT_TRUE(is_slack_webhook("https://hooks.slack.com/services/T1/B2/c3"));
    EXPECT_FALSE(is_slack_webhook("https://hooks.slack.com.example.org/x"));

    // Telegram: the title bold, the rest as text, <, > and & as entities.
    const NotificationRequest telegram = telegram_request("123:abc", "-100200", true, "A & B", "x < y", "");
    EXPECT_EQ(telegram.url, "https://api.telegram.org/bot123:abc/sendMessage");
    const nlohmann::json telegram_body = nlohmann::json::parse(telegram.body);
    EXPECT_EQ(telegram_body["text"], "<b>A &amp; B</b>\nx &lt; y");
    EXPECT_EQ(telegram_body["parse_mode"], "HTML");
    EXPECT_EQ(telegram_body["chat_id"], "-100200");
    EXPECT_EQ(telegram_body["disable_notification"], true);

    // The webhook: the run as JSON, and its header, on one line.
    const NotifiedRun run = sample_run();
    const NotificationRequest webhook = webhook_request("http://ha.lan:8123/api/webhook/vml", " X-Key : s3 \r\n",
                                                        run, "Title", "Body", "2026-11-21T12:00:00Z");
    EXPECT_TRUE(webhook.plain_http);
    EXPECT_EQ(webhook.headers, (Headers{{"X-Key", "s3 "}}));
    const nlohmann::json hook = nlohmann::json::parse(webhook.body);
    EXPECT_EQ(hook["event"], "run_ended");
    EXPECT_EQ(hook["outcome"], "finished_with_failures");
    EXPECT_EQ(hook["duration_seconds"], 3725);
    EXPECT_EQ(hook["finished_at"], "2026-11-21T12:00:00Z");
    EXPECT_EQ(hook["videos"][0]["metrics"]["vmaf"]["mean"], 95.214);
    EXPECT_EQ(hook["videos"][1]["scored"], false);
    EXPECT_TRUE(hook["videos"][1]["metrics"].empty());
    EXPECT_TRUE(is_server_url("http://192.168.1.5:8080"));
    EXPECT_FALSE(is_server_url("ftp://example.org"));
    EXPECT_FALSE(is_server_url("https://"));
    EXPECT_TRUE(is_ntfy_topic("vml_runs-2"));
    EXPECT_FALSE(is_ntfy_topic(std::string(65, 'a')));
}

// Discord: each test video a field where the message lists them and they
// fit, the mention notifying only whom it names, silence without one, and
// the User-Agent Discord asks for.
TEST(Notifications, DiscordFieldsMentionsAndSilence) {
    Settings settings;
    settings.discord_enabled = true;
    settings.discord_webhook = "https://discord.com/api/webhooks/1/a";
    settings.discord_mention = "<@&42>";
    NotifiedRun run = sample_run();  // finished with failures
    NotificationPlan plan = plan_notification(settings, run, "");
    ASSERT_EQ(plan.requests.size(), 1u);
    nlohmann::json body = nlohmann::json::parse(plan.requests[0].body);
    const nlohmann::json& embed = body["embeds"][0];
    EXPECT_EQ(embed["description"], "reference.mkv\n1:02:05");  // the videos are fields
    ASSERT_EQ(embed["fields"].size(), 2u);
    EXPECT_EQ(embed["fields"][0]["name"], "encode-a.mkv");  // not again as the value's first line
    EXPECT_EQ(embed["fields"][0]["value"], "VMAF v0.6.1 95.21 · PSNR 44.10 dB");
    EXPECT_EQ(embed["fields"][1]["value"], "Failed: the file ends early");
    EXPECT_EQ(body["content"], "<@&42>");
    EXPECT_EQ(body["allowed_mentions"], (nlohmann::json{{"parse", nlohmann::json::array()}, {"roles", {"42"}}}));
    EXPECT_FALSE(body.contains("flags"));
    ASSERT_EQ(plan.requests[0].headers.size(), 1u);
    EXPECT_EQ(plan.requests[0].headers[0].first, "User-Agent");
    EXPECT_EQ(plan.requests[0].headers[0].second, "DiscordBot (https://github.com/4KVCD/VideoMetricsLab, 2.7)");

    // A run that finished: no mention for "problems", so silent.
    run.outcome = RunOutcome::finished;
    settings.discord_silent = true;
    body = nlohmann::json::parse(plan_notification(settings, run, "").requests[0].body);
    EXPECT_FALSE(body.contains("content"));
    EXPECT_EQ(body["flags"], 4096);
    // More videos than Discord takes as fields: the message as text.
    run.videos.resize(30, run.videos[0]);
    body = nlohmann::json::parse(plan_notification(settings, run, "").requests[0].body);
    EXPECT_FALSE(body["embeds"][0].contains("fields"));
    EXPECT_NE(body["embeds"][0]["description"].get<std::string>().find("encode-a.mkv"), std::string::npos);
    // A message that lists no videos: no fields either.
    run.videos.resize(1);
    settings.notify_message = "{duration}";
    body = nlohmann::json::parse(plan_notification(settings, run, "").requests[0].body);
    EXPECT_FALSE(body["embeds"][0].contains("fields"));
    EXPECT_EQ(body["embeds"][0]["description"], "1:02:05");
}

// What an error or an answer shows loses every key: a header's token, and
// a token in the URL's path; each service's own words are found in its
// answer; a 429 says how long to wait.
TEST(Notifications, ErrorsLoseEveryKey) {
    const NotificationRequest gotify = gotify_request("https://g.lan", "sekrit", 5, "t", "m", "");
    EXPECT_EQ(without_secrets("refused: key sekrit", gotify), "refused: key Gotify");
    const NotificationRequest telegram = telegram_request("123:abc", "1", false, "t", "m", "");
    EXPECT_EQ(without_secrets("POST /bot123:abc/sendMessage failed", telegram), "POST Telegram failed");
    // A header value too short to be a key, and to be told from other words.
    const NotificationRequest webhook = webhook_request("https://h.lan/x", "X-Key: 1", sample_run(), "t", "m", "");
    EXPECT_EQ(without_secrets("HTTP 401", webhook), "HTTP 401");
    EXPECT_EQ(service_answer(R"({"ok":false,"error_code":400,"description":"Bad Request: chat not found"})"),
              "Bad Request: chat not found");
    EXPECT_EQ(service_answer(R"({"error":{"type":"invalid_request","message":"Access token is missing"}})"),
              "Access token is missing");
    EXPECT_EQ(service_answer(R"({"error":"Unauthorized","errorCode":401,"errorDescription":"bad token"})"),
              "bad token");
    EXPECT_EQ(service_answer(R"({"errors":["user key is invalid"],"status":0})"), "user key is invalid");
    EXPECT_EQ(service_answer("invalid_payload"), "invalid_payload");
    EXPECT_EQ(retry_after(429, R"({"message":"You are being rate limited.","retry_after":0.5,"global":false})"),
              0.5);
    EXPECT_EQ(retry_after(400, R"({"retry_after":0.5})"), std::nullopt);
    EXPECT_EQ(retry_after(429, "slow down"), std::nullopt);
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
