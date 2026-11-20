// Notifications when a run ends (Settings > Notifications): the message,
// made from templates of fields, and the requests that send it to Pushover
// and Discord. The window sends them (notifier.h); everything here is
// without the network, so that it can be tested.
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

// Each metric's statistics, in the app's order, from a video's result: a
// frame metric's of its frames; a whole-video score's (CVVDP) mean is the
// score, the rest from its timeline where it has one.
std::vector<std::pair<std::string, ScoreStats>> notified_metrics(const ComparisonResult& result);

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

// Whether `settings` ask for a message for a run that ended so, after
// `seconds`, with a service to send it to.
bool notification_wanted(const Settings& settings, RunOutcome outcome, double seconds);

// A request to send: where, and what.
struct NotificationRequest {
    std::string service;  // "Pushover" or "Discord", for the log
    std::string url;
    std::string content_type;
    std::string body;
};

// `text` cut to at most `limit` characters (code points), "…" ending it
// where it was cut.
std::string cut_to_characters(std::string_view text, std::size_t limit);

// The Pushover message: `user` key, application `token`, `priority` (-2 to
// 1). Pushover takes a title of up to 250 characters and a message of up to
// 1024, which must not be empty: longer ones are cut, an empty message is
// `title`, or the outcome.
NotificationRequest pushover_request(std::string_view user, std::string_view token, int priority,
                                     std::string_view title, std::string_view message, std::string_view fallback);

// Whether `url` is a Discord webhook's (https, on discord.com or
// discordapp.com, under /api/webhooks/): nothing is sent anywhere else.
bool is_discord_webhook(std::string_view url);

// The Discord message to webhook `url`: an embed with `title` (up to 256
// characters) and `message` (up to 4096), coloured by `outcome`, `app` in
// its footer and `utc_timestamp` (ISO 8601) as its time. Mentions in it
// (@everyone, a file named so) notify no one.
NotificationRequest discord_request(std::string_view url, std::string_view title, std::string_view message,
                                    RunOutcome outcome, std::string_view app, std::string_view utc_timestamp,
                                    std::string_view fallback);

// What to send for a run, by `settings`: a request to each service that is
// on (or only to `only`, "Pushover" or "Discord", as Send test asks, on or
// not), and the services that could not be sent to, and why.
struct NotificationPlan {
    std::vector<NotificationRequest> requests;
    std::vector<std::string> unreadable;  // services whose keys cannot be read back here
    std::vector<std::string> incomplete;  // services missing a key
    bool not_a_webhook = false;           // Discord's URL is not a Discord webhook's
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
