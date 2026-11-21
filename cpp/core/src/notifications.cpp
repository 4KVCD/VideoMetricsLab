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

std::vector<std::pair<std::string, ScoreStats>> notified_metrics(const ComparisonResult& result) {
    std::vector<std::pair<std::string, ScoreStats>> metrics;
    for (const MetricDefinition& metric : vml::metrics()) {
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

bool notification_wanted(const Settings& settings, RunOutcome outcome, double seconds) {
    const bool service = (settings.pushover_enabled && !settings.pushover_user.empty() &&
                          !settings.pushover_token.empty()) ||
                         (settings.discord_enabled && !settings.discord_webhook.empty());
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
    return replaced(std::move(text), url.substr(path, url.find('?', path) - path), name);
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

NotificationRequest discord_request(std::string_view url, std::string_view title, std::string_view message,
                                    RunOutcome outcome, std::string_view app, std::string_view utc_timestamp,
                                    std::string_view fallback) {
    const int colour = outcome == RunOutcome::finished                 ? 0x2ECC71
                       : outcome == RunOutcome::finished_with_failures ? 0xE67E22
                                                                       : 0x95A5A6;
    nlohmann::json embed{{"color", colour}};
    const std::string cut_title = cut_to_characters(title, 256);
    const std::string cut_message = cut_to_characters(message, 4096);
    if (!cut_title.empty()) {
        embed["title"] = cut_title;
    }
    if (text_or(cut_message, "").empty() && cut_title.empty()) {
        embed["title"] = cut_to_characters(fallback, 256);
    } else if (!text_or(cut_message, "").empty()) {
        embed["description"] = cut_message;
    }
    if (!app.empty()) {
        embed["footer"] = {{"text", std::string(app)}};
    }
    if (!utc_timestamp.empty()) {
        embed["timestamp"] = std::string(utc_timestamp);
    }
    const nlohmann::json body{
        {"embeds", nlohmann::json::array({embed})},
        {"allowed_mentions", {{"parse", nlohmann::json::array()}}},
    };
    return {"Discord", std::string(url), "application/json",
            body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace)};
}

NotificationPlan plan_notification(const Settings& settings, const NotifiedRun& run, std::string_view utc_timestamp,
                                   std::string_view only) {
    NotificationPlan plan;
    const std::string title = render_notification(settings.notify_title, run, settings.notify_video_line);
    const std::string message = render_notification(settings.notify_message, run, settings.notify_video_line);
    const auto wanted = [&only](std::string_view service, bool enabled) {
        return only.empty() ? enabled : only == service;
    };
    if (wanted("Pushover", settings.pushover_enabled)) {
        const std::optional<std::string> user = secret_of(settings.pushover_user);
        const std::optional<std::string> token = secret_of(settings.pushover_token);
        if (!user || !token) {
            plan.unreadable.emplace_back("Pushover");
        } else if (user->empty() || token->empty()) {
            plan.incomplete.emplace_back("Pushover");
        } else {
            plan.requests.push_back(
                pushover_request(*user, *token, settings.pushover_priority, title, message, run.outcome_text));
        }
    }
    if (wanted("Discord", settings.discord_enabled)) {
        const std::optional<std::string> url = secret_of(settings.discord_webhook);
        if (!url) {
            plan.unreadable.emplace_back("Discord");
        } else if (url->empty()) {
            plan.incomplete.emplace_back("Discord");
        } else if (!is_discord_webhook(*url)) {
            plan.not_a_webhook = true;
        } else {
            plan.requests.push_back(
                discord_request(*url, title, message, run.outcome, run.app, utc_timestamp, run.outcome_text));
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
