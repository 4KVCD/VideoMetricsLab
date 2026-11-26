#include "vml/notifications.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/results.h"
#include "vml/settings.h"

namespace vml {

namespace {

constexpr std::string_view encrypted_prefix = "dpapi:";

// What statistic each field suffix is.
std::optional<double> statistic(const ScoreStats& stats, std::string_view name) {
    if (stats.count == 0) {
        return std::nullopt;
    }
    if (name == "mean") {
        return stats.mean;
    }
    if (name == "min") {
        return stats.minimum;
    }
    if (name == "max") {
        return stats.maximum;
    }
    if (name == "median") {
        return stats.median;
    }
    if (name == "worst_10") {
        return stats.worst_10_percent;
    }
    if (name == "worst_5") {
        return stats.worst_5_percent;
    }
    if (name == "worst_1") {
        return stats.worst_1_percent;
    }
    if (name == "worst_0_1") {
        return stats.worst_0_1_percent;
    }
    return std::nullopt;
}

bool is_statistic(std::string_view name) {
    const auto& names = notification_statistics();
    return std::find(names.begin(), names.end(), name) != names.end();
}

// A metric's score as the app shows it, with its unit.
std::string shown(const MetricDefinition& metric, std::optional<double> value) {
    std::string text = metric.format_value(value);
    if (value && std::isfinite(*value)) {
        text += metric.value_suffix;
    }
    return text;
}

// Each metric's `name` statistic: "VMAF v0.6.1 95.21 · PSNR 44.10 dB".
std::string scores_line(const NotifiedVideo& video, std::string_view name) {
    std::string line;
    for (const auto& [key, stats] : video.metrics) {
        const MetricDefinition* metric = find_metric(key);
        if (metric == nullptr) {
            continue;
        }
        if (!line.empty()) {
            line += " · ";
        }
        line += std::string(metric->short_label) + " " + shown(*metric, statistic(stats, name));
    }
    return line;
}

std::string duration_text(double seconds) {
    const long long total = std::max(0LL, std::llround(seconds));
    char text[32];
    std::snprintf(text, sizeof text, "%lld:%02lld:%02lld", total / 3600, total / 60 % 60, total % 60);
    return text;
}

std::string videos_text(const NotifiedRun& run, std::string_view video_line) {
    std::vector<std::string> lines;
    bool several_lines = false;
    for (const NotifiedVideo& video : run.videos) {
        std::string line = render_notification(video_line, run, video_line, &video);
        if (line.empty()) {
            continue;
        }
        several_lines = several_lines || line.find('\n') != std::string::npos;
        lines.push_back(std::move(line));
    }
    std::string text;
    for (const std::string& line : lines) {
        if (!text.empty()) {
            text += several_lines ? "\n\n" : "\n";
        }
        text += line;
    }
    return text;
}

std::string percent_encoded(std::string_view text) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
            byte == '-' || byte == '_' || byte == '.' || byte == '~') {
            out += c;
        } else {
            out += '%';
            out += hex[byte >> 4];
            out += hex[byte & 15];
        }
    }
    return out;
}

std::string base64(std::string_view bytes) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::size_t at = 0;
    for (; at + 2 < bytes.size(); at += 3) {
        const std::uint32_t n = (static_cast<unsigned char>(bytes[at]) << 16) |
                                (static_cast<unsigned char>(bytes[at + 1]) << 8) |
                                static_cast<unsigned char>(bytes[at + 2]);
        out += alphabet[n >> 18];
        out += alphabet[(n >> 12) & 63];
        out += alphabet[(n >> 6) & 63];
        out += alphabet[n & 63];
    }
    if (at < bytes.size()) {
        std::uint32_t n = static_cast<unsigned char>(bytes[at]) << 16;
        if (at + 1 < bytes.size()) {
            n |= static_cast<unsigned char>(bytes[at + 1]) << 8;
        }
        out += alphabet[n >> 18];
        out += alphabet[(n >> 12) & 63];
        out += at + 1 < bytes.size() ? alphabet[(n >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

std::optional<std::string> from_base64(std::string_view text) {
    std::string out;
    std::uint32_t bits = 0;
    int count = 0;
    for (const char c : text) {
        int value = -1;
        if (c >= 'A' && c <= 'Z') {
            value = c - 'A';
        } else if (c >= 'a' && c <= 'z') {
            value = c - 'a' + 26;
        } else if (c >= '0' && c <= '9') {
            value = c - '0' + 52;
        } else if (c == '+') {
            value = 62;
        } else if (c == '/') {
            value = 63;
        } else if (c == '=') {
            break;
        } else {
            return std::nullopt;
        }
        bits = (bits << 6) | static_cast<std::uint32_t>(value);
        count += 6;
        if (count >= 8) {
            count -= 8;
            out += static_cast<char>((bits >> count) & 0xFF);
        }
    }
    return out;
}

std::string_view text_or(std::string_view text, std::string_view fallback) {
    return text.find_first_not_of(" \t\r\n") == std::string_view::npos ? fallback : text;
}

}  // namespace

std::vector<std::pair<std::string, ScoreStats>> notified_metrics(const ComparisonResult& result,
                                                                 const std::vector<std::string>& order) {
    std::vector<std::pair<std::string, ScoreStats>> metrics;
    for (const MetricDefinition* ordered : ordered_metrics(order)) {
        const MetricDefinition& metric = *ordered;
        const std::string key(metric.key);
        if (const FrameMetricResult* frames = result.metric_results.frame(key)) {
            const std::vector<double> values(frames->values.begin(), frames->values.end());
            metrics.emplace_back(key, compute_stats(values, {}, metric.aggregation, metric.direction));
        } else if (const SequenceMetricResult* sequence = result.metric_results.sequence(key)) {
            ScoreStats stats;
            if (sequence->has_timeline()) {
                const std::vector<double> values(sequence->values.begin(), sequence->values.end());
                stats = compute_stats(values, {}, metric.aggregation, metric.direction);
            }
            if (stats.count == 0) {
                stats.count = 1;
                stats.minimum = stats.maximum = stats.median = sequence->score;
                stats.worst_10_percent = stats.worst_5_percent = stats.worst_1_percent =
                    stats.worst_0_1_percent = sequence->score;
            }
            stats.mean = sequence->score;
            metrics.emplace_back(key, stats);
        }
    }
    return metrics;
}

const std::vector<std::string_view>& notification_statistics() {
    static const std::vector<std::string_view> names{"mean",     "min",     "max",     "median",
                                                     "worst_10", "worst_5", "worst_1", "worst_0_1"};
    return names;
}

std::optional<std::string> notification_field(std::string_view field, const NotifiedRun& run,
                                              const NotifiedVideo* video, std::string_view video_line) {
    if (video != nullptr) {
        if (field == "name") {
            return video->name;
        }
        if (field == "result") {
            return video->result;
        }
        if (field == "frames") {
            return std::to_string(video->frames);
        }
        const std::size_t dot = field.find('.');
        const std::string_view head = field.substr(0, dot);
        const std::string_view stat = dot == std::string_view::npos ? "mean" : field.substr(dot + 1);
        if (head == "scores" && is_statistic(stat)) {
            return video->scored ? scores_line(*video, stat) : video->result;
        }
        if (const MetricDefinition* metric = find_metric(head); metric != nullptr && is_statistic(stat)) {
            for (const auto& [key, stats] : video->metrics) {
                if (key == head) {
                    return shown(*metric, statistic(stats, stat));
                }
            }
            return metric->format_value(std::nullopt);
        }
    }
    if (field == "outcome") {
        return run.outcome_text;
    }
    if (field == "duration") {
        return duration_text(run.seconds);
    }
    if (field == "reference") {
        return run.reference;
    }
    if (field == "finished") {
        return run.finished_at;
    }
    if (field == "app") {
        return run.app;
    }
    if (field == "video_count") {
        return std::to_string(run.videos.size());
    }
    if (field == "scored_count" || field == "failed_count") {
        const bool scored = field == "scored_count";
        return std::to_string(std::count_if(run.videos.begin(), run.videos.end(), [scored](const NotifiedVideo& v) {
            return v.scored == scored;
        }));
    }
    if (field == "videos" && video == nullptr) {
        return videos_text(run, video_line);
    }
    return std::nullopt;
}

std::string render_notification(std::string_view text, const NotifiedRun& run, std::string_view video_line,
                                const NotifiedVideo* video) {
    std::string out;
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t open = text.find('{', at);
        if (open == std::string_view::npos) {
            out += text.substr(at);
            break;
        }
        out += text.substr(at, open - at);
        const std::size_t close = text.find('}', open + 1);
        const std::size_t next_open = text.find('{', open + 1);
        if (close == std::string_view::npos || next_open < close) {
            out += '{';
            at = open + 1;
            continue;
        }
        const std::string_view field = text.substr(open + 1, close - open - 1);
        if (const std::optional<std::string> value = notification_field(field, run, video, video_line)) {
            out += *value;
        } else {
            out += text.substr(open, close - open + 1);
        }
        at = close + 1;
    }
    return out;
}

const std::vector<std::string_view>& notification_services() {
    static const std::vector<std::string_view> names{"Pushover",   "Discord", "Gotify",   "ntfy",
                                                     "Pushbullet", "Slack",   "Telegram", "Webhook"};
    return names;
}

bool service_enabled(const Settings& settings, std::string_view service) {
    return service == "Pushover"     ? settings.pushover_enabled
           : service == "Discord"    ? settings.discord_enabled
           : service == "Gotify"     ? settings.gotify_enabled
           : service == "ntfy"       ? settings.ntfy_enabled
           : service == "Pushbullet" ? settings.pushbullet_enabled
           : service == "Slack"      ? settings.slack_enabled
           : service == "Telegram"   ? settings.telegram_enabled
           : service == "Webhook"    ? settings.webhook_enabled
                                     : false;
}

namespace {

// Whether `service`, on, has what it needs to be sent to (its keys may still
// be another user's: plan_notification tells).
bool service_set_up(const Settings& s, std::string_view service) {
    return service == "Pushover"     ? !s.pushover_user.empty() && !s.pushover_token.empty()
           : service == "Discord"    ? !s.discord_webhook.empty()
           : service == "Gotify"     ? !s.gotify_url.empty() && !s.gotify_token.empty()
           : service == "ntfy"       ? !s.ntfy_url.empty() && !s.ntfy_topic.empty()
           : service == "Pushbullet" ? !s.pushbullet_token.empty()
           : service == "Slack"      ? !s.slack_webhook.empty()
           : service == "Telegram"   ? !s.telegram_token.empty() && !s.telegram_chat.empty()
           : service == "Webhook"    ? !s.webhook_url.empty()
                                     : false;
}

}  // namespace

bool notification_wanted(const Settings& settings, RunOutcome outcome, double seconds) {
    const bool service =
        std::any_of(notification_services().begin(), notification_services().end(), [&settings](auto name) {
            return service_enabled(settings, name) && service_set_up(settings, name);
        });
    const bool wanted = outcome == RunOutcome::finished                 ? settings.notify_finished
                        : outcome == RunOutcome::finished_with_failures ? settings.notify_with_failures
                                                                        : settings.notify_cancelled;
    return service && wanted && seconds >= settings.notify_min_minutes * 60.0;
}

std::string cut_to_characters(std::string_view text, std::size_t limit) {
    // Code points: the bytes that do not continue one (10xxxxxx).
    std::size_t characters = 0;
    for (std::size_t at = 0; at < text.size(); ++at) {
        if ((static_cast<unsigned char>(text[at]) & 0xC0) != 0x80) {
            ++characters;
        }
    }
    if (characters <= limit || limit == 0) {
        return characters <= limit ? std::string(text) : std::string();
    }
    std::size_t kept = 0;
    std::size_t at = 0;
    for (; at < text.size(); ++at) {
        if ((static_cast<unsigned char>(text[at]) & 0xC0) != 0x80) {
            if (kept == limit - 1) {
                break;
            }
            ++kept;
        }
    }
    return std::string(text.substr(0, at)) + "…";
}

NotificationRequest pushover_request(std::string_view user, std::string_view token, int priority,
                                     std::string_view title, std::string_view message, std::string_view fallback) {
    const std::string cut_title = cut_to_characters(title, 250);
    const std::string cut_message = cut_to_characters(text_or(message, text_or(cut_title, fallback)), 1024);
    std::string body = "token=" + percent_encoded(token) + "&user=" + percent_encoded(user) +
                       "&message=" + percent_encoded(cut_message) +
                       "&priority=" + std::to_string(std::clamp(priority, -2, 1));
    if (!cut_title.empty()) {
        body += "&title=" + percent_encoded(cut_title);
    }
    return {"Pushover", "https://api.pushover.net/1/messages.json", "application/x-www-form-urlencoded", body};
}

namespace {

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

}  // namespace

std::string without_secret_url(std::string text, const std::string& url, const std::string& name) {
    text = replaced(std::move(text), url, name);
    const std::size_t scheme = url.find("://");
    const std::size_t path = url.find('/', scheme == std::string::npos ? 0 : scheme + 3);
    if (path == std::string::npos || path + 1 >= url.size()) {
        return text;  // no path: nothing more to hide
    }
    text = replaced(std::move(text), url.substr(path), name);
    // The path without its query, where it is more than "/": every "/" in
    // an error ("Cannot POST /") would be the service's name.
    const std::string bare = url.substr(path, url.find('?', path) - path);
    return bare.size() > 1 ? replaced(std::move(text), bare, name) : text;
}

std::string without_secrets(std::string text, const NotificationRequest& request) {
    text = without_secret_url(std::move(text), request.url, request.service);
    for (const std::string& secret : request.secrets) {
        // A short one is left: as text it would be found in other words
        // ("401" holds a header value of "1"), and it is no key worth keeping.
        if (secret.size() >= 6) {
            text = replaced(std::move(text), secret, request.service);
        }
    }
    return text;
}

std::string service_answer(std::string_view body) {
    const nlohmann::json answer = nlohmann::json::parse(body, nullptr, false);
    if (answer.is_object()) {
        const auto text_at = [&answer](const char* key) -> std::string {
            const auto found = answer.find(key);
            return found != answer.end() && found->is_string() ? found->get<std::string>() : std::string();
        };
        if (const auto errors = answer.find("errors"); errors != answer.end() && errors->is_array()) {
            std::string joined;
            for (const nlohmann::json& error : *errors) {
                if (error.is_string()) {
                    joined += (joined.empty() ? "" : "; ") + error.get<std::string>();
                }
            }
            if (!joined.empty()) {
                return joined;  // Pushover
            }
        }
        if (const auto error = answer.find("error"); error != answer.end() && error->is_object()) {
            if (const auto message = error->find("message"); message != error->end() && message->is_string()) {
                return message->get<std::string>();  // Pushbullet
            }
        }
        for (const char* key : {"description", "errorDescription", "error", "message"}) {
            if (std::string text = text_at(key); !text.empty()) {
                return text;  // Telegram; Gotify; ntfy (and Gotify's short form); Discord
            }
        }
    }
    std::string start(body.substr(0, 200));  // Slack's "invalid_payload", a proxy's page
    std::replace(start.begin(), start.end(), '\n', ' ');
    std::replace(start.begin(), start.end(), '\r', ' ');
    return start;
}

std::optional<double> retry_after(int status, std::string_view body) {
    if (status != 429) {
        return std::nullopt;
    }
    const nlohmann::json answer = nlohmann::json::parse(body, nullptr, false);
    if (!answer.is_object() || !answer.contains("retry_after") || !answer["retry_after"].is_number()) {
        return std::nullopt;
    }
    const double seconds = answer["retry_after"].get<double>();
    return std::isfinite(seconds) && seconds >= 0 ? std::optional(seconds) : std::nullopt;
}

bool is_discord_webhook(std::string_view url) {
    for (const std::string_view host :
         {"discord.com", "discordapp.com", "ptb.discord.com", "canary.discord.com"}) {
        const std::string start = "https://" + std::string(host) + "/api/webhooks/";
        if (url.size() > start.size() && url.substr(0, start.size()) == start) {
            return url.find_first_of(" \t\r\n") == std::string_view::npos;
        }
    }
    return false;
}

bool is_server_url(std::string_view url) {
    std::size_t host = 0;
    if (url.substr(0, 8) == "https://") {
        host = 8;
    } else if (url.substr(0, 7) == "http://") {
        host = 7;
    } else {
        return false;
    }
    const std::size_t end = url.find_first_of("/?#", host);
    return end != host && host < url.size() &&
           url.find_first_of(" \t\r\n\"<>\\") == std::string_view::npos;
}

bool is_slack_webhook(std::string_view url) {
    constexpr std::string_view start = "https://hooks.slack.com/";
    return url.size() > start.size() && url.substr(0, start.size()) == start &&
           url.find_first_of(" \t\r\n") == std::string_view::npos;
}

bool is_ntfy_topic(std::string_view topic) {
    return !topic.empty() && topic.size() <= 64 && std::all_of(topic.begin(), topic.end(), [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
    });
}

namespace {

// The JSON as sent: text that is not UTF-8 replaced, not refused.
std::string dumped(const nlohmann::json& body) {
    return body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// `text` cut to at most `limit` bytes, at a character's start, "…" (3
// bytes) ending it where it was cut.
std::string cut_to_bytes(std::string_view text, std::size_t limit) {
    if (text.size() <= limit) {
        return std::string(text);
    }
    std::size_t at = limit >= 3 ? limit - 3 : 0;
    while (at > 0 && (static_cast<unsigned char>(text[at]) & 0xC0) == 0x80) {
        --at;
    }
    return std::string(text.substr(0, at)) + "…";
}

// `text` with no line breaks: what a header's name or value can hold.
std::string one_line(std::string_view text) {
    std::string line(text);
    line.erase(std::remove_if(line.begin(), line.end(), [](char c) { return c == '\r' || c == '\n'; }), line.end());
    return line;
}

// `server` with no slash at its end, then `path` ("/message").
std::string on_server(std::string_view server, std::string_view path) {
    std::string url(server);
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    return url + std::string(path);
}

std::string_view outcome_key(RunOutcome outcome) {
    return outcome == RunOutcome::finished                 ? "finished"
           : outcome == RunOutcome::finished_with_failures ? "finished_with_failures"
                                                           : "cancelled";
}

// `text` for Telegram's HTML: <, > and & as entities.
std::string html_escaped(std::string_view text) {
    std::string out;
    for (const char c : text) {
        out += c == '<' ? "&lt;" : c == '>' ? "&gt;" : c == '&' ? "&amp;" : std::string(1, c);
    }
    return out;
}

// The first `count` characters (code points) of `text`.
std::string first_characters(std::string_view text, std::size_t count) {
    std::size_t at = 0;
    for (std::size_t seen = 0; at < text.size(); ++at) {
        if ((static_cast<unsigned char>(text[at]) & 0xC0) != 0x80 && seen++ == count) {
            break;
        }
    }
    return std::string(text.substr(0, at));
}

// The characters (code points) in `text`.
std::size_t characters_in(std::string_view text) {
    return static_cast<std::size_t>(std::count_if(text.begin(), text.end(), [](char c) {
        return (static_cast<unsigned char>(c) & 0xC0) != 0x80;
    }));
}

// The mentions a Discord message may make: its mention's role or user, or
// @here and @everyone; none for anything else.
nlohmann::json allowed_mentions(std::string_view mention) {
    nlohmann::json allowed{{"parse", nlohmann::json::array()}};
    const auto digits = [](std::string_view id) {
        return !id.empty() && std::all_of(id.begin(), id.end(), [](char c) { return c >= '0' && c <= '9'; });
    };
    if (mention == "@here" || mention == "@everyone") {
        allowed["parse"] = nlohmann::json::array({"everyone"});
    } else if (mention.size() > 4 && mention.substr(0, 3) == "<@&" && mention.back() == '>' &&
               digits(mention.substr(3, mention.size() - 4))) {
        allowed["roles"] = nlohmann::json::array({std::string(mention.substr(3, mention.size() - 4))});
    } else if (mention.size() > 3 && mention.substr(0, 2) == "<@" && mention.back() == '>') {
        std::string_view id = mention.substr(2, mention.size() - 3);
        if (!id.empty() && id.front() == '!') {
            id.remove_prefix(1);
        }
        if (digits(id)) {
            allowed["users"] = nlohmann::json::array({std::string(id)});
        }
    }
    return allowed;
}

// The embed's fields: each video's name and its line, where they fit.
std::optional<nlohmann::json> video_fields(const DiscordOptions& options, std::size_t used) {
    if (options.run == nullptr || options.run->videos.empty() || options.run->videos.size() > 25) {
        return std::nullopt;
    }
    nlohmann::json fields = nlohmann::json::array();
    std::size_t total = used;
    for (const NotifiedVideo& video : options.run->videos) {
        std::string line = render_notification(options.video_line, *options.run, options.video_line, &video);
        // The name is the field's: not again as the line's first.
        if (line == video.name) {
            line.clear();
        } else if (line.size() > video.name.size() && line.compare(0, video.name.size(), video.name) == 0 &&
                   line[video.name.size()] == '\n') {
            line.erase(0, video.name.size() + 1);
        }
        const std::string name = cut_to_characters(video.name, 256);
        if (text_or(line, "").empty()) {
            line = video.result.empty() ? std::string("-") : video.result;
        }
        if (characters_in(line) > 1024) {
            return std::nullopt;
        }
        total += characters_in(name) + characters_in(line);
        if (total > 6000) {
            return std::nullopt;
        }
        fields.push_back({{"name", name}, {"value", line}, {"inline", false}});
    }
    return fields;
}

}  // namespace

NotificationRequest discord_request(std::string_view url, std::string_view title, std::string_view message,
                                    RunOutcome outcome, std::string_view app, std::string_view utc_timestamp,
                                    std::string_view fallback, const DiscordOptions& options) {
    const int colour = outcome == RunOutcome::finished                 ? 0x2ECC71
                       : outcome == RunOutcome::finished_with_failures ? 0xE67E22
                                                                       : 0x95A5A6;
    nlohmann::json embed{{"color", colour}};
    const std::string cut_title = cut_to_characters(title, 256);
    const std::string footer = cut_to_characters(app, 2048);
    std::size_t used = characters_in(cut_title) + characters_in(footer);
    // A field a video, where the message lists them and they fit.
    std::optional<nlohmann::json> fields;
    std::string text(message);
    if (options.run != nullptr && options.message_without_videos != message) {
        const std::string without = cut_to_characters(options.message_without_videos, 4096);
        if ((fields = video_fields(options, used + characters_in(without)))) {
            text = without;
        }
    }
    const std::string cut_message = cut_to_characters(text, 4096);
    if (!cut_title.empty()) {
        embed["title"] = cut_title;
    }
    if (text_or(cut_message, "").empty() && cut_title.empty() && !fields) {
        embed["title"] = cut_to_characters(fallback, 256);
    } else if (!text_or(cut_message, "").empty()) {
        embed["description"] = cut_message;
    }
    if (fields) {
        embed["fields"] = *fields;
    }
    if (!footer.empty()) {
        embed["footer"] = {{"text", footer}};
    }
    if (!utc_timestamp.empty()) {
        embed["timestamp"] = std::string(utc_timestamp);
    }
    nlohmann::json body{{"embeds", nlohmann::json::array({embed})},
                        {"allowed_mentions", allowed_mentions(options.mention)}};
    if (!options.mention.empty()) {
        body["content"] = cut_to_characters(options.mention, 2000);
    } else if (options.silent) {
        body["flags"] = 1 << 12;  // SUPPRESS_NOTIFICATIONS
    }
    // Discord asks that requests say what they come from, in a bot's form.
    const std::size_t space = app.rfind(' ');
    const std::string version = space == std::string_view::npos ? std::string("1") : std::string(app.substr(space + 1));
    NotificationRequest request{"Discord", std::string(url), "application/json", dumped(body)};
    request.headers = {{"User-Agent", "DiscordBot (https://github.com/4KVCD/VideoMetricsLab, " + version + ")"}};
    return request;
}

NotificationRequest gotify_request(std::string_view server, std::string_view token, int priority,
                                   std::string_view title, std::string_view message, std::string_view fallback) {
    const std::string cut_title = cut_to_characters(title, 250);
    const nlohmann::json body{{"title", std::string(text_or(cut_title, fallback))},
                              {"message", std::string(text_or(message, text_or(cut_title, fallback)))},
                              {"priority", std::clamp(priority, 0, 10)}};
    NotificationRequest request{"Gotify", on_server(server, "/message"), "application/json", dumped(body)};
    request.headers = {{"X-Gotify-Key", one_line(token)}};
    request.plain_http = true;
    request.secrets = {std::string(token)};
    return request;
}

NotificationRequest ntfy_request(std::string_view server, std::string_view topic, std::string_view token,
                                 int priority, RunOutcome outcome, std::string_view title, std::string_view message,
                                 std::string_view fallback) {
    const std::string_view tag = outcome == RunOutcome::finished                 ? "white_check_mark"
                                 : outcome == RunOutcome::finished_with_failures ? "warning"
                                                                                 : "no_entry_sign";
    nlohmann::json body{{"topic", std::string(topic)},
                        {"message", cut_to_bytes(text_or(message, text_or(title, fallback)), 4096)},
                        {"priority", std::clamp(priority, 1, 5)},
                        {"tags", nlohmann::json::array({std::string(tag)})}};
    if (const std::string cut_title = cut_to_characters(title, 250); !text_or(cut_title, "").empty()) {
        body["title"] = cut_title;
    }
    // JSON is published to the server's root, the topic in the body.
    NotificationRequest request{"ntfy", on_server(server, "/"), "application/json", dumped(body)};
    if (!token.empty()) {
        request.headers = {{"Authorization", "Bearer " + one_line(token)}};
        request.secrets.emplace_back(token);
    }
    request.plain_http = true;  // the topic is in the body only, which no error repeats
    return request;
}

NotificationRequest pushbullet_request(std::string_view token, std::string_view device, std::string_view title,
                                       std::string_view message, std::string_view fallback) {
    const std::string cut_title = cut_to_characters(title, 250);
    nlohmann::json body{{"type", "note"},
                        {"title", std::string(text_or(cut_title, fallback))},
                        {"body", std::string(text_or(message, text_or(cut_title, fallback)))}};
    if (!device.empty()) {
        body["device_iden"] = std::string(device);
    }
    NotificationRequest request{"Pushbullet", "https://api.pushbullet.com/v2/pushes", "application/json",
                                dumped(body)};
    request.headers = {{"Access-Token", one_line(token)}};
    request.secrets = {std::string(token)};
    return request;
}

NotificationRequest slack_request(std::string_view url, std::string_view title, std::string_view message,
                                  std::string_view fallback) {
    const std::string heading = cut_to_characters(text_or(title, fallback), 150);
    nlohmann::json blocks = nlohmann::json::array();
    blocks.push_back({{"type", "header"}, {"text", {{"type", "plain_text"}, {"text", heading}}}});
    // Plain text, so that nothing in it is read as Slack's markup; a section
    // takes 3000 characters, so a long message takes several (10 at most).
    std::string rest(text_or(message, ""));
    for (int section = 0; !rest.empty() && section < 10; ++section) {
        // The last one takes "…" where more was left; the others are cut
        // at their 3000th character, the next going on from there.
        const std::string part = section == 9 ? cut_to_characters(rest, 3000) : first_characters(rest, 3000);
        rest.erase(0, section == 9 ? rest.size() : part.size());
        blocks.push_back({{"type", "section"}, {"text", {{"type", "plain_text"}, {"text", part}}}});
    }
    // "text" is what the push and a screen reader show.
    const nlohmann::json body{{"text", heading}, {"blocks", blocks}};
    return {"Slack", std::string(url), "application/json", dumped(body)};
}

NotificationRequest telegram_request(std::string_view token, std::string_view chat, bool silent,
                                     std::string_view title, std::string_view message, std::string_view fallback) {
    // 4096 characters once its HTML is read: the title, a line break, the message.
    const std::string cut_title = cut_to_characters(text_or(title, ""), 256);
    const std::size_t room = 4096 - characters_in(cut_title) - 1;
    const std::string cut_message = cut_to_characters(text_or(message, ""), room);
    std::string text;
    if (!cut_title.empty()) {
        text = "<b>" + html_escaped(cut_title) + "</b>";
    }
    if (!cut_message.empty()) {
        text += (text.empty() ? "" : "\n") + html_escaped(cut_message);
    }
    if (text.empty()) {
        text = html_escaped(cut_to_characters(fallback, 4096));
    }
    nlohmann::json body{{"chat_id", std::string(chat)}, {"text", text}, {"parse_mode", "HTML"},
                        {"link_preview_options", {{"is_disabled", true}}}};
    if (silent) {
        body["disable_notification"] = true;
    }
    NotificationRequest request{"Telegram", "https://api.telegram.org/bot" + std::string(token) + "/sendMessage",
                                "application/json", dumped(body)};
    request.secrets = {std::string(token)};  // in the URL's path too: without_secret_url takes that
    return request;
}

NotificationRequest webhook_request(std::string_view url, std::string_view header, const NotifiedRun& run,
                                    std::string_view title, std::string_view message, std::string_view utc_timestamp) {
    nlohmann::json videos = nlohmann::json::array();
    for (const NotifiedVideo& video : run.videos) {
        nlohmann::json metrics = nlohmann::json::object();
        for (const auto& [key, stats] : video.metrics) {
            nlohmann::json values = nlohmann::json::object();
            for (const std::string_view name : notification_statistics()) {
                const std::optional<double> value = statistic(stats, name);
                values[std::string(name)] =
                    value && std::isfinite(*value) ? nlohmann::json(*value) : nlohmann::json(nullptr);
            }
            metrics[key] = values;
        }
        videos.push_back({{"name", video.name},
                          {"result", video.result},
                          {"scored", video.scored},
                          {"frames", video.frames},
                          {"metrics", metrics}});
    }
    const nlohmann::json body{{"event", "run_ended"},
                              {"app", run.app},
                              {"outcome", std::string(outcome_key(run.outcome))},
                              {"outcome_text", run.outcome_text},
                              {"title", std::string(title)},
                              {"message", std::string(message)},
                              {"finished_at", std::string(utc_timestamp)},
                              {"duration_seconds", std::llround(std::max(0.0, run.seconds))},
                              {"reference", run.reference},
                              {"videos", videos}};
    NotificationRequest request{"Webhook", std::string(url), "application/json", dumped(body)};
    request.plain_http = true;
    if (const std::size_t colon = header.find(':'); colon != std::string_view::npos) {
        std::string name = one_line(header.substr(0, colon));
        std::string value = one_line(header.substr(colon + 1));
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        request.headers = {{name, value}};
        if (!value.empty()) {
            request.secrets = {value};
        }
    }
    return request;
}

NotificationPlan plan_notification(const Settings& settings, const NotifiedRun& run, std::string_view utc_timestamp,
                                   std::string_view only) {
    NotificationPlan plan;
    const std::string title = render_notification(settings.notify_title, run, settings.notify_video_line);
    const std::string message = render_notification(settings.notify_message, run, settings.notify_video_line);
    const auto wanted = [&only, &settings](std::string_view service) {
        return only.empty() ? service_enabled(settings, service) : only == service;
    };
    // Each secret read back; nothing where one cannot be (another user's).
    const auto read = [](std::initializer_list<const std::string*> stored) {
        std::vector<std::string> secrets;
        for (const std::string* one : stored) {
            std::optional<std::string> secret = secret_of(*one);
            if (!secret) {
                return std::optional<std::vector<std::string>>();
            }
            secrets.push_back(std::move(*secret));
        }
        return std::optional(secrets);
    };
    const std::string& fallback = run.outcome_text;
    if (wanted("Pushover")) {
        if (const auto keys = read({&settings.pushover_user, &settings.pushover_token}); !keys) {
            plan.unreadable.emplace_back("Pushover");
        } else if ((*keys)[0].empty() || (*keys)[1].empty()) {
            plan.incomplete.emplace_back("Pushover");
        } else {
            plan.requests.push_back(
                pushover_request((*keys)[0], (*keys)[1], settings.pushover_priority, title, message, fallback));
        }
    }
    if (wanted("Discord")) {
        if (const auto keys = read({&settings.discord_webhook}); !keys) {
            plan.unreadable.emplace_back("Discord");
        } else if ((*keys)[0].empty()) {
            plan.incomplete.emplace_back("Discord");
        } else if (!is_discord_webhook((*keys)[0])) {
            plan.bad_address.emplace_back("Discord");
        } else {
            DiscordOptions options;
            const bool problem = run.outcome != RunOutcome::finished;
            if (settings.discord_mention_when == "always" ||
                (settings.discord_mention_when == "problems" && problem)) {
                options.mention = settings.discord_mention;
            }
            options.silent = settings.discord_silent;
            // The videos as fields: the message without its {videos}.
            std::string without(settings.notify_message);
            for (std::size_t at = without.find("{videos}"); at != std::string::npos; at = without.find("{videos}")) {
                without.erase(at, 8);
            }
            std::string shown = render_notification(without, run, settings.notify_video_line);
            while (shown.find("\n\n\n") != std::string::npos) {
                shown.replace(shown.find("\n\n\n"), 3, "\n\n");
            }
            shown.erase(shown.find_last_not_of(" \t\r\n") + 1);
            options.run = &run;
            options.video_line = settings.notify_video_line;
            options.message_without_videos = shown;
            if (without == settings.notify_message) {
                options.run = nullptr;  // the message lists no videos: none as fields either
            }
            plan.requests.push_back(discord_request((*keys)[0], title, message, run.outcome, run.app, utc_timestamp,
                                                    fallback, options));
        }
    }
    if (wanted("Gotify")) {
        if (const auto keys = read({&settings.gotify_token}); !keys) {
            plan.unreadable.emplace_back("Gotify");
        } else if ((*keys)[0].empty() || settings.gotify_url.empty()) {
            plan.incomplete.emplace_back("Gotify");
        } else if (!is_server_url(settings.gotify_url)) {
            plan.bad_address.emplace_back("Gotify");
        } else {
            plan.requests.push_back(
                gotify_request(settings.gotify_url, (*keys)[0], settings.gotify_priority, title, message, fallback));
        }
    }
    if (wanted("ntfy")) {
        if (const auto keys = read({&settings.ntfy_topic, &settings.ntfy_token}); !keys) {
            plan.unreadable.emplace_back("ntfy");
        } else if ((*keys)[0].empty() || settings.ntfy_url.empty()) {
            plan.incomplete.emplace_back("ntfy");
        } else if (!is_server_url(settings.ntfy_url) || !is_ntfy_topic((*keys)[0])) {
            plan.bad_address.emplace_back("ntfy");
        } else {
            plan.requests.push_back(ntfy_request(settings.ntfy_url, (*keys)[0], (*keys)[1], settings.ntfy_priority,
                                                 run.outcome, title, message, fallback));
        }
    }
    if (wanted("Pushbullet")) {
        if (const auto keys = read({&settings.pushbullet_token}); !keys) {
            plan.unreadable.emplace_back("Pushbullet");
        } else if ((*keys)[0].empty()) {
            plan.incomplete.emplace_back("Pushbullet");
        } else {
            plan.requests.push_back(
                pushbullet_request((*keys)[0], settings.pushbullet_device, title, message, fallback));
        }
    }
    if (wanted("Slack")) {
        if (const auto keys = read({&settings.slack_webhook}); !keys) {
            plan.unreadable.emplace_back("Slack");
        } else if ((*keys)[0].empty()) {
            plan.incomplete.emplace_back("Slack");
        } else if (!is_slack_webhook((*keys)[0])) {
            plan.bad_address.emplace_back("Slack");
        } else {
            plan.requests.push_back(slack_request((*keys)[0], title, message, fallback));
        }
    }
    if (wanted("Telegram")) {
        if (const auto keys = read({&settings.telegram_token}); !keys) {
            plan.unreadable.emplace_back("Telegram");
        } else if ((*keys)[0].empty() || settings.telegram_chat.empty()) {
            plan.incomplete.emplace_back("Telegram");
        } else if ((*keys)[0].find_first_of("/?# \t\r\n") != std::string::npos) {
            plan.bad_address.emplace_back("Telegram");  // a token is "123456:ABC...": nothing that changes the URL
        } else {
            plan.requests.push_back(telegram_request((*keys)[0], settings.telegram_chat, settings.telegram_silent,
                                                     title, message, fallback));
        }
    }
    if (wanted("Webhook")) {
        if (const auto keys = read({&settings.webhook_url, &settings.webhook_header}); !keys) {
            plan.unreadable.emplace_back("Webhook");
        } else if ((*keys)[0].empty()) {
            plan.incomplete.emplace_back("Webhook");
        } else if (!is_server_url((*keys)[0]) ||
                   (!(*keys)[1].empty() && ((*keys)[1].find(':') == std::string::npos ||
                                            (*keys)[1].find_first_not_of(" \t") == (*keys)[1].find(':')))) {
            plan.bad_address.emplace_back("Webhook");
        } else {
            plan.requests.push_back(webhook_request((*keys)[0], (*keys)[1], run, title, message, utc_timestamp));
        }
    }
    return plan;
}

std::string stored_secret(std::string_view secret, bool encrypted) {
    if (encrypted && !secret.empty()) {
        if (const std::optional<std::string> sealed = platform::protect_for_user(secret)) {
            return std::string(encrypted_prefix) + base64(*sealed);
        }
    }
    return std::string(secret);
}

std::optional<std::string> secret_of(std::string_view stored) {
    if (!secret_is_encrypted(stored)) {
        return std::string(stored);
    }
    const std::optional<std::string> sealed = from_base64(stored.substr(encrypted_prefix.size()));
    return sealed ? platform::reveal_for_user(*sealed) : std::nullopt;
}

bool secret_is_encrypted(std::string_view stored) {
    return stored.substr(0, encrypted_prefix.size()) == encrypted_prefix;
}

bool secrets_can_be_encrypted() { return platform::protect_for_user("check").has_value(); }

}  // namespace vml
