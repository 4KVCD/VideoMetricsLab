// Notifications when a run ends (Settings > Notifications): the message,
// made from templates of fields, and the requests that send it to each
// service (Pushover, Discord, Gotify, ntfy, Pushbullet, Slack, Telegram, a
// webhook). The window sends them (notifier.h), the command line too
// (cli/notify.h); everything here is without the network, so that it can be
// tested.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/stats.h"

namespace vml {

struct ComparisonResult;
struct Settings;

// How a run ended.
enum class RunOutcome { finished, finished_with_failures, cancelled };

// A test video of a run, as its message tells it.
struct NotifiedVideo {
    std::string name;    // its file name, without the folder
    std::string result;  // how it ended, as the window says it ("Scored", "Failed: ...")
    bool scored = false;
    long long frames = 0;  // frames compared
    // Each metric scored, by key in the app's order, with its statistics.
    std::vector<std::pair<std::string, ScoreStats>> metrics;
};

// A run, as its message tells it. The words are the window's, in its
// language: the fields only put them in place.
struct NotifiedRun {
    RunOutcome outcome = RunOutcome::finished;
    std::string outcome_text;  // "Finished", in the window's language
    double seconds = 0.0;      // its length, pauses left out
    std::string reference;     // the reference's file name
    std::string finished_at;   // the local time it ended, as the window shows times
    std::string app;           // "VideoMetricsLab 2.7"
    std::vector<NotifiedVideo> videos;
};

// The Standard message, also the settings' default: how the run ended, its
// length, and each test video's name with its metrics' mean scores. Words
// come from fields only, so it suits every language.
inline constexpr std::string_view standard_notification_title = "{app}: {outcome}";
inline constexpr std::string_view standard_notification_message = "{reference}\n{duration}\n\n{videos}";
inline constexpr std::string_view standard_notification_video_line = "{name}\n{scores}";

// Each metric's statistics, in the order `order` gives (ordered_metrics:
// Settings::metric_order, empty for the standard order), from a video's
// result: a frame metric's of its frames; a whole-video score's (CVVDP) mean
// is the score, the rest from its timeline where it has one.
std::vector<std::pair<std::string, ScoreStats>> notified_metrics(const ComparisonResult& result,
                                                                 const std::vector<std::string>& order = {});

// The statistics fields have after a metric's key ("{vmaf.min}"); the key
// alone ("{vmaf}") is the mean.
const std::vector<std::string_view>& notification_statistics();

// A field's value: `run`'s, or with `video`, that video's (and the run's
// fields too). Nothing for a field there is none of, which is left as it
// was typed.
//
// Run: outcome, duration (h:mm:ss), reference, finished, app, video_count,
// scored_count, failed_count, videos (each video's line, `video_line` filled
// in, one after another; a blank line between lines of several lines).
// Video: name, result, frames, scores (each metric's mean: "VMAF v0.6.1
// 95.21 · PSNR 44.10 dB"; for a video not scored, its result),
// scores.STAT (each metric's STAT), KEY and KEY.STAT (a metric's; a dash
// where the video has none of it).
std::optional<std::string> notification_field(std::string_view field, const NotifiedRun& run,
                                              const NotifiedVideo* video, std::string_view video_line);

// `text` with each {field} filled in (notification_field); an unknown
// field, or a brace that opens none, is left as it is.
std::string render_notification(std::string_view text, const NotifiedRun& run, std::string_view video_line,
                                const NotifiedVideo* video = nullptr);

// The services a message can go to, by name as the window and the log show
// them, in the window's order.
const std::vector<std::string_view>& notification_services();

// Whether `service` is on in `settings`.
bool service_enabled(const Settings& settings, std::string_view service);

// Whether `settings` ask for a message for a run that ended so, after
// `seconds`, with a service to send it to.
bool notification_wanted(const Settings& settings, RunOutcome outcome, double seconds);

// A request to send: where, and what.
struct NotificationRequest {
    std::string service;  // as notification_services names it, for the log
    std::string url;
    std::string content_type;
    std::string body;
    // Headers besides Content-Type, by name and value (a token sent in one).
    std::vector<std::pair<std::string, std::string>> headers{};
    // Whether an http:// URL is sent to: a server of the user's own, on
    // their network. Otherwise https only.
    bool plain_http = false;
    // The keys in it that are not in its URL (a header's token), taken out
    // of any error text shown with without_secrets.
    std::vector<std::string> secrets{};
};

// `text` (an error's, or what a service answered) with the request's keys
// taken out: its URL and its path (without_secret_url) and its secrets of 6
// characters or more (a shorter one would be found inside other words),
// each replaced by the service's name.
std::string without_secrets(std::string text, const NotificationRequest& request);

// What a service said about a message it refused, from its answer's body:
// Pushover's "errors", Telegram's "description", Gotify's and ntfy's
// "error", Pushbullet's error's "message", Discord's "message", else the
// start of the answer on one line.
std::string service_answer(std::string_view body);

// The seconds a service asks to wait before sending again, for a 429
// answer that says so (Discord's "retry_after"); nothing otherwise.
std::optional<double> retry_after(int status, std::string_view body);

// `text` cut to at most `limit` characters (code points), "…" ending it
// where it was cut.
std::string cut_to_characters(std::string_view text, std::size_t limit);

// The Pushover message: `user` key, application `token`, `priority` (-2 to
// 1). Pushover takes a title of up to 250 characters and a message of up to
// 1024, which must not be empty: longer ones are cut, an empty message is
// `title`, or the outcome.
NotificationRequest pushover_request(std::string_view user, std::string_view token, int priority,
                                     std::string_view title, std::string_view message, std::string_view fallback);

// `text` (an error's, curl's or Qt's) with `url`, a webhook's URL and so
// its key, replaced by `name`, and its path too, with or without its query:
// what an error shows of it when it shows the request (`POST
// /api/webhooks/<id>/<token>`) rather than the whole URL.
std::string without_secret_url(std::string text, const std::string& url, const std::string& name);

// Whether `url` is a Discord webhook's (https, on discord.com or
// discordapp.com, under /api/webhooks/): nothing is sent anywhere else.
bool is_discord_webhook(std::string_view url);

// Whether `url` is an http or https address with a host and nothing that
// could not be in one (a space, a line break): a server of the user's own
// (Gotify, ntfy, a webhook).
bool is_server_url(std::string_view url);

// Whether `url` is a Slack incoming webhook's (https, hooks.slack.com).
bool is_slack_webhook(std::string_view url);

// Whether `topic` is an ntfy topic's name: 1 to 64 of A-Z, a-z, 0-9, - and _.
bool is_ntfy_topic(std::string_view topic);

// Discord's own parts of a message, beyond its text.
struct DiscordOptions {
    // Whom it mentions ("<@&role ID>", "<@user ID>", "@here", "@everyone"),
    // where it mentions anyone: the mention is the message's content, and
    // only that mention notifies.
    std::string mention;
    // Sent without a push or a sound, where it mentions no one.
    bool silent = false;
    // Each test video as a field of its own: its name, then its line
    // (`video_line` filled in for it), with `message_without_videos` as the
    // embed's text, where the message lists the videos and they fit
    // Discord's limits (25 fields, 1024 characters a field, 6000 in all);
    // otherwise `message` as it is.
    const NotifiedRun* run = nullptr;
    std::string_view video_line;
    std::string_view message_without_videos;
};

// The Discord message to webhook `url`: an embed with `title` (up to 256
// characters) and `message` (up to 4096), coloured by `outcome`, `app` in
// its footer and `utc_timestamp` (ISO 8601) as its time, with `options`.
// Mentions in its text (@everyone, a file named so) notify no one.
NotificationRequest discord_request(std::string_view url, std::string_view title, std::string_view message,
                                    RunOutcome outcome, std::string_view app, std::string_view utc_timestamp,
                                    std::string_view fallback, const DiscordOptions& options = {});

// The Gotify message to the server at `server` (its address, with any path
// it is under), with the application's `token` and `priority` (0 to 10).
NotificationRequest gotify_request(std::string_view server, std::string_view token, int priority,
                                   std::string_view title, std::string_view message, std::string_view fallback);

// The ntfy message to `topic` on the server at `server`, with an access
// `token` where it has one, `priority` (1 to 5) and a tag (an emoji in the
// apps) for `outcome`. ntfy takes up to 4096 bytes: a longer message is cut.
NotificationRequest ntfy_request(std::string_view server, std::string_view topic, std::string_view token,
                                 int priority, RunOutcome outcome, std::string_view title, std::string_view message,
                                 std::string_view fallback);

// The Pushbullet note, with the access `token`, to the device `device` (its
// iden), or to every device for "".
NotificationRequest pushbullet_request(std::string_view token, std::string_view device, std::string_view title,
                                       std::string_view message, std::string_view fallback);

// The Slack message to incoming webhook `url`: `title` as a header (up to
// 150 characters) and `message` as plain text sections (3000 characters
// each), so nothing in it is read as Slack's markup.
NotificationRequest slack_request(std::string_view url, std::string_view title, std::string_view message,
                                  std::string_view fallback);

// The Telegram message from the bot with `token` to chat `chat` (an ID, or
// a public channel's @name): `title` in bold, then `message`, up to 4096
// characters, without a sound where `silent`.
NotificationRequest telegram_request(std::string_view token, std::string_view chat, bool silent,
                                     std::string_view title, std::string_view message, std::string_view fallback);

// The webhook message to `url`: the run as JSON, for a program to read
// (docs/USER_GUIDE.md, Notifications): the rendered `title` and `message`,
// how it ended (a key and the window's words), when (`utc_timestamp`), how
// long, the reference, and each test video's result and its metrics'
// statistics. `header` ("Name: value", or "") is sent with it.
NotificationRequest webhook_request(std::string_view url, std::string_view header, const NotifiedRun& run,
                                    std::string_view title, std::string_view message, std::string_view utc_timestamp);

// What to send for a run, by `settings`: a request to each service that is
// on (or only to `only`, a service's name, as Send test asks, on or not),
// and the services that could not be sent to, and why.
struct NotificationPlan {
    std::vector<NotificationRequest> requests;
    std::vector<std::string> unreadable;  // services whose keys cannot be read back here
    std::vector<std::string> incomplete;  // services missing a key, an address or a chat
    std::vector<std::string> bad_address; // services whose address (URL, topic, header) is not one they take
};
NotificationPlan plan_notification(const Settings& settings, const NotifiedRun& run, std::string_view utc_timestamp,
                                   std::string_view only = {});

// A secret (a key, a webhook URL) as settings.json keeps it: encrypted for
// this PC's user where `encrypted` and the system can ("dpapi:" and
// base64), else as it is.
std::string stored_secret(std::string_view secret, bool encrypted);

// A stored secret's own text; nothing where it is encrypted and cannot be
// read back here (another user's, another PC's).
std::optional<std::string> secret_of(std::string_view stored);

// Whether a stored secret is encrypted.
bool secret_is_encrypted(std::string_view stored);

// Whether this system can encrypt secrets for its user (Windows).
bool secrets_can_be_encrypted();

}  // namespace vml
