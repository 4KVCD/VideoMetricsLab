#include "vml/settings.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>

#include "vml/app_log.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"

namespace vml {

namespace {

using Json = nlohmann::ordered_json;

// Where a setting is kept in Settings, by its type.
using Field = std::variant<bool Settings::*, int Settings::*, std::string Settings::*,
                           std::vector<std::string> Settings::*, std::vector<int> Settings::*>;

// Every setting by its key in settings.json (the Python dataclass's field
// names), except the CVVDP presets, which are read apart.
const std::unordered_map<std::string, Field>& fields() {
    static const std::unordered_map<std::string, Field> by_key{
        {"ffmpeg_dir", &Settings::ffmpeg_dir},
        {"cache_dir", &Settings::cache_dir},
        {"export_dir", &Settings::export_dir},
        {"auto_export", &Settings::auto_export},
        {"auto_export_segments", &Settings::auto_export_segments},
        {"export_items", &Settings::export_items},
        {"export_formats", &Settings::export_formats},
        {"export_zip", &Settings::export_zip},
        {"quick_save_formats", &Settings::quick_save_formats},
        {"after_run_export", &Settings::after_run_export},
        {"after_run_place", &Settings::after_run_place},
        {"default_gpu_decode", &Settings::default_gpu_decode},
        {"default_compute_psnr", &Settings::default_compute_psnr},
        {"default_compute_ssim", &Settings::default_compute_ssim},
        {"default_compute_xpsnr", &Settings::default_compute_xpsnr},
        {"default_compute_psnr_yuv", &Settings::default_compute_psnr_yuv},
        {"default_compute_psnr_yuv_mse", &Settings::default_compute_psnr_yuv_mse},
        {"default_compute_vmaf_yuv", &Settings::default_compute_vmaf_yuv},
        {"default_compute_ssim_yuv", &Settings::default_compute_ssim_yuv},
        {"default_compute_ms_ssim", &Settings::default_compute_ms_ssim},
        {"default_compute_ms_ssim_yuv", &Settings::default_compute_ms_ssim_yuv},
        {"default_compute_psnr_hvs", &Settings::default_compute_psnr_hvs},
        {"default_compute_cambi", &Settings::default_compute_cambi},
        {"default_compute_ciede2000", &Settings::default_compute_ciede2000},
        {"default_compute_vmaf", &Settings::default_compute_vmaf},
        {"default_compute_vmaf_neg", &Settings::default_compute_vmaf_neg},
        {"default_compute_vmaf_v1", &Settings::default_compute_vmaf_v1},
        {"default_compute_vmaf_hdr", &Settings::default_compute_vmaf_hdr},
        {"default_compute_ssimulacra2", &Settings::default_compute_ssimulacra2},
        {"default_compute_butteraugli", &Settings::default_compute_butteraugli},
        {"default_compute_cvvdp", &Settings::default_compute_cvvdp},
        {"default_vmaf_on_gpu", &Settings::default_vmaf_on_gpu},
        {"default_vship_build", &Settings::default_vship_build},
        {"film_grain_default", &Settings::film_grain_default},
        {"graph_metric", &Settings::graph_metric},
        {"graph_worst_frames", &Settings::graph_worst_frames},
        {"weak_segment_conditions", &Settings::weak_segment_conditions},
        {"weak_segments_all", &Settings::weak_segments_all},
        {"weak_segment_min_frames", &Settings::weak_segment_min_frames},
        {"weak_segment_merge_frames", &Settings::weak_segment_merge_frames},
        {"weak_segment_profile", &Settings::weak_segment_profile},
        {"parallel_jobs", &Settings::parallel_jobs},
        {"gpu_metrics_together", &Settings::gpu_metrics_together},
        {"gpu", &Settings::gpu},
        {"vvc_decode_on_gpu", &Settings::vvc_decode_on_gpu},
        {"recognise_test_remuxes", &Settings::recognise_test_remuxes},
        {"use_cache", &Settings::use_cache},
        {"check_frame_counts", &Settings::check_frame_counts},
        {"check_for_updates", &Settings::check_for_updates},
        {"confirm_close_during_run", &Settings::confirm_close_during_run},
        {"skipped_update_version", &Settings::skipped_update_version},
        {"language", &Settings::language},
        {"theme", &Settings::theme},
        {"interface", &Settings::interface_style},
        {"accent_colour", &Settings::accent_colour},
        {"light_look", &Settings::light_look},
        {"dark_look", &Settings::dark_look},
        {"video_names", &Settings::video_names},
        {"video_name_start", &Settings::video_name_start},
        {"video_name_end", &Settings::video_name_end},
        {"remember_window_size", &Settings::remember_window_size},
        {"window_width", &Settings::window_width},
        {"window_height", &Settings::window_height},
        {"window_geometry", &Settings::window_geometry},
        {"last_reference_dir", &Settings::last_reference_dir},
        {"last_test_dir", &Settings::last_test_dir},
        {"remember_videos", &Settings::remember_videos},
        {"remembered_reference", &Settings::remembered_reference},
        {"remembered_tests", &Settings::remembered_tests},
        {"highlight_best_worst", &Settings::highlight_best_worst},
        {"compare_loop", &Settings::compare_loop},
        {"frame_preview_color_mode", &Settings::frame_preview_color_mode},
        {"compare_decoded_videos", &Settings::compare_decoded_videos},
        {"compare_loop_before_ms", &Settings::compare_loop_before_ms},
        {"compare_loop_after_ms", &Settings::compare_loop_after_ms},
        {"export_graph_theme", &Settings::export_graph_theme},
        {"export_graph_background", &Settings::export_graph_background},
        {"export_graph_text", &Settings::export_graph_text},
        {"hidden_metrics", &Settings::hidden_metrics},
        {"listed_metrics", &Settings::listed_metrics},
        {"show_yuv_planes", &Settings::show_yuv_planes},
        {"cvvdp_default_preset", &Settings::cvvdp_default_preset},
        {"notify_finished", &Settings::notify_finished},
        {"notify_with_failures", &Settings::notify_with_failures},
        {"notify_cancelled", &Settings::notify_cancelled},
        {"notify_min_minutes", &Settings::notify_min_minutes},
        {"notify_title", &Settings::notify_title},
        {"notify_message", &Settings::notify_message},
        {"notify_video_line", &Settings::notify_video_line},
        {"notify_secrets_encrypted", &Settings::notify_secrets_encrypted},
        {"pushover_enabled", &Settings::pushover_enabled},
        {"pushover_user", &Settings::pushover_user},
        {"pushover_token", &Settings::pushover_token},
        {"pushover_priority", &Settings::pushover_priority},
        {"discord_enabled", &Settings::discord_enabled},
        {"discord_webhook", &Settings::discord_webhook},
        {"discord_mention", &Settings::discord_mention},
        {"discord_mention_when", &Settings::discord_mention_when},
        {"discord_silent", &Settings::discord_silent},
        {"gotify_enabled", &Settings::gotify_enabled},
        {"gotify_url", &Settings::gotify_url},
        {"gotify_token", &Settings::gotify_token},
        {"gotify_priority", &Settings::gotify_priority},
        {"ntfy_enabled", &Settings::ntfy_enabled},
        {"ntfy_url", &Settings::ntfy_url},
        {"ntfy_topic", &Settings::ntfy_topic},
        {"ntfy_token", &Settings::ntfy_token},
        {"ntfy_priority", &Settings::ntfy_priority},
        {"pushbullet_enabled", &Settings::pushbullet_enabled},
        {"pushbullet_token", &Settings::pushbullet_token},
        {"pushbullet_device", &Settings::pushbullet_device},
        {"slack_enabled", &Settings::slack_enabled},
        {"slack_webhook", &Settings::slack_webhook},
        {"telegram_enabled", &Settings::telegram_enabled},
        {"telegram_token", &Settings::telegram_token},
        {"telegram_chat", &Settings::telegram_chat},
        {"telegram_silent", &Settings::telegram_silent},
        {"webhook_enabled", &Settings::webhook_enabled},
        {"webhook_url", &Settings::webhook_url},
        {"webhook_header", &Settings::webhook_header},
        {"saved_results_column_widths", &Settings::saved_results_column_widths},
        {"saved_results_hidden_columns", &Settings::saved_results_hidden_columns},
    };
    return by_key;
}

// Sets a setting from its value in the file when the value is of its type
// (as Python's _fits checks it: a bool is not an int); otherwise leaves the
// default, and returns the type, as Python names it, for the log.
std::optional<std::string> set_if_it_fits(Settings& settings, const Field& field, const Json& value) {
    if (const auto* flag = std::get_if<bool Settings::*>(&field)) {
        if (!value.is_boolean()) {
            return "<class 'bool'>";
        }
        settings.*(*flag) = value.get<bool>();
    } else if (const auto* number = std::get_if<int Settings::*>(&field)) {
        if (!value.is_number_integer()) {
            return "<class 'int'>";
        }
        settings.*(*number) = value.get<int>();
    } else if (const auto* text = std::get_if<std::string Settings::*>(&field)) {
        if (!value.is_string()) {
            return "<class 'str'>";
        }
        settings.*(*text) = value.get<std::string>();
    } else if (const auto* list = std::get_if<std::vector<std::string> Settings::*>(&field)) {
        if (!value.is_array() || !std::all_of(value.begin(), value.end(), [](const Json& v) { return v.is_string(); })) {
            return "list[str]";
        }
        settings.*(*list) = value.get<std::vector<std::string>>();
    } else if (const auto* numbers = std::get_if<std::vector<int> Settings::*>(&field)) {
        if (!value.is_array() ||
            !std::all_of(value.begin(), value.end(), [](const Json& v) { return v.is_number_integer(); })) {
            return "list[int]";
        }
        settings.*(*numbers) = value.get<std::vector<int>>();
    }
    return std::nullopt;
}

// A setting of secret_settings: never written to the log.
bool holds_a_key(const std::string& name) {
    const auto& secrets = secret_settings();
    return std::any_of(secrets.begin(), secrets.end(), [&name](const auto& secret) { return secret.first == name; });
}

// A setting the file holds in a type of its own, in the log: a key's by
// its type alone (a key typed in as a number, or a list of webhook URLs).
void log_unfit(const std::string& name, const Json& value, const std::string& kind) {
    const std::string held = holds_a_key(name) ? std::string("a JSON ") + value.type_name() : python_repr_of(value);
    app_log::write(app_log::Level::warning, "vml.settings",
                   "Settings: " + name + " is " + held + ", not a " + kind + "; using its default");
}

}  // namespace

const std::vector<std::pair<std::string_view, std::string Settings::*>>& secret_settings() {
    static const std::vector<std::pair<std::string_view, std::string Settings::*>> secrets{
        {"pushover_user", &Settings::pushover_user},       {"pushover_token", &Settings::pushover_token},
        {"discord_webhook", &Settings::discord_webhook},   {"gotify_token", &Settings::gotify_token},
        {"ntfy_topic", &Settings::ntfy_topic},             {"ntfy_token", &Settings::ntfy_token},
        {"pushbullet_token", &Settings::pushbullet_token}, {"slack_webhook", &Settings::slack_webhook},
        {"telegram_token", &Settings::telegram_token},     {"webhook_url", &Settings::webhook_url},
        {"webhook_header", &Settings::webhook_header}};
    return secrets;
}

const std::vector<std::string>& export_item_keys() {
    static const std::vector<std::string> items{"results", "frames", "graphs", "worst_frames", "weak_segments"};
    return items;
}

const std::vector<std::string>& export_format_keys() {
    static const std::vector<std::string> formats{"json", "csv"};
    return formats;
}

std::vector<std::string> exported_items(const Settings& settings) {
    const bool written = !settings.export_formats.empty();
    std::vector<std::string> items;
    for (const std::string& item : settings.export_items) {
        if (written || item == "graphs") {  // the rest are files in a format
            items.push_back(item);
        }
    }
    return items;
}

namespace {

// The exports after each run as a file from before export_items chose
// them: results (JSON) and graphs in a .zip, weak segments as files of their
// own, each next to the test video or in the export folder; both on, all in
// the .zip, where the results and graphs went. Neither: the defaults.
void carry_over_auto_export(Settings& settings) {
    const auto chosen = [](const std::string& place) { return place == "beside" || place == "folder"; };
    const bool results = chosen(settings.auto_export);
    const bool segments = chosen(settings.auto_export_segments);
    if (!results && !segments) {
        return;
    }
    settings.export_items.clear();
    if (results) {
        settings.export_items = {"results", "graphs"};
    }
    if (segments) {
        settings.export_items.push_back("weak_segments");
    }
    settings.after_run_export = true;
    settings.after_run_place = results ? settings.auto_export : settings.auto_export_segments;
    // Results were JSON alone, weak segments JSON and CSV: with weak
    // segments, both (the results as CSV too, which there was no choice of).
    settings.export_formats = segments ? std::vector<std::string>{"json", "csv"} : std::vector<std::string>{"json"};
    settings.export_zip = results;
}

// Exports as the first v2.7 builds chose them, results by format
// ("results_json", "results_csv") and weak segments always as JSON and CSV:
// Results in the formats its boxes ticked, or, with weak segments alone,
// both formats as they were written.
void carry_over_result_formats(Settings& settings) {
    std::vector<std::string> formats;
    std::vector<std::string> items;
    bool results = false;
    for (const std::string& item : settings.export_items) {
        if (item == "results_json" || item == "results_csv") {
            formats.push_back(item == "results_json" ? "json" : "csv");
            if (!results) {
                items.push_back("results");
                results = true;
            }
        } else {
            items.push_back(item);
        }
    }
    if (results) {
        std::sort(formats.begin(), formats.end(), std::greater<>());  // json, then csv
        settings.export_formats = formats;
    } else if (std::find(items.begin(), items.end(), "weak_segments") != items.end()) {
        settings.export_formats = export_format_keys();
    }
    settings.export_items = items;
}

// The metrics hidden at first that the window did not have when these
// settings were saved (not in listed_metrics): hidden, and ticked so that
// showing one in Add/remove metrics calculates it. A file without
// listed_metrics is v2.6's or older, which had none of them; one that
// lists a metric keeps its ticks and what it shows.
void hide_new_metrics(Settings& settings) {
    const auto has = [](const std::vector<std::string>& keys, std::string_view key) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    };
    for (const std::string_view key : metrics_hidden_at_first) {
        if (has(settings.listed_metrics, key)) {
            continue;
        }
        if (!has(settings.hidden_metrics, key)) {
            settings.hidden_metrics.emplace_back(key);
        }
        if (bool* compute = settings.default_compute(key)) {
            *compute = true;
        }
    }
}

// Settings > Notifications' keys are kept as plain text where they cannot
// be encrypted (secrets_can_be_encrypted: on Linux and macOS), so the
// settings folder is the user's alone (0700) and so is each file written in
// it (0600), whatever the umask: other users of the system cannot read the
// keys. The folder is closed first, so a file is never open to others while
// it is being written. On Windows the profile folder's ACL already keeps
// them to the user; there std::filesystem::permissions only sets or clears
// the read-only attribute, and both leave it off.
void keep_folder_private(const std::filesystem::path& folder) {
    std::error_code ignored;
    std::filesystem::permissions(folder, std::filesystem::perms::group_all | std::filesystem::perms::others_all,
                                 std::filesystem::perm_options::remove, ignored);
}

void keep_file_private(const std::filesystem::path& file) {
    std::error_code ignored;
    std::filesystem::permissions(file, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ignored);
}

// Where settings.json stops being JSON, by its line and column, for the
// log. Not nlohmann's message: its "last read: '...'" quotes the text there,
// and in a file cut off in a key or a webhook URL (plain text on Linux and
// macOS) that is part of the key.
std::string where_unreadable(const std::string& text, const Json::exception& error) {
    const auto* parse = dynamic_cast<const Json::parse_error*>(&error);
    if (parse == nullptr || parse->byte == 0) {
        return "it isn't JSON";
    }
    // byte counts from 1, as nlohmann's own "at line L, column C" does.
    const std::size_t end = std::min<std::size_t>(parse->byte - 1, text.size());
    const auto before = text.begin() + static_cast<std::ptrdiff_t>(end);
    const auto lines = static_cast<std::size_t>(std::count(text.begin(), before, '\n'));
    const std::size_t line_start = lines == 0 ? 0 : text.rfind('\n', end - 1) + 1;
    return "it isn't JSON at line " + std::to_string(lines + 1) + ", column " + std::to_string(end - line_start + 1);
}

}  // namespace

int default_parallel_jobs(unsigned cores) { return cores > parallel_by_default_above_cores ? 2 : 1; }

std::filesystem::path app_data_folder() {
    const std::optional<std::string> chosen = platform::environment_variable(app_data_variable);
    if (chosen && !chosen->empty()) {
        std::error_code ignored;
        const std::filesystem::path folder = platform::path_from_text(*chosen);
        const std::filesystem::path whole = std::filesystem::absolute(folder, ignored);
        return whole.empty() ? folder : whole;
    }
    return platform::home_directory() / ".videometricslab";
}

std::filesystem::path Settings::file() { return app_data_folder() / "settings.json"; }

Settings Settings::defaults() {
    Settings settings;
    settings.parallel_jobs = default_parallel_jobs(platform::logical_cores());
    return settings;
}

Settings Settings::load_checked(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(platform::path_text(path) + ": it could not be read (no such file, or no access)");
    }
    std::stringstream contents;
    contents << stream.rdbuf();
    const std::string text = python_json_readable(contents.str());
    try {
        if (!Json::parse(text).is_object()) {
            throw std::runtime_error(platform::path_text(path) + ": it is not a settings file (no JSON object)");
        }
    } catch (const Json::exception& error) {
        throw std::runtime_error(platform::path_text(path) + ": " + where_unreadable(text, error));
    }
    return load_from(path);
}

Settings Settings::load() {
    const std::filesystem::path path = file();
    std::error_code ignored;
    std::filesystem::create_directories(path.parent_path(), ignored);
    keep_folder_private(path.parent_path());
    return load_from(path);
}

Settings Settings::load_from(const std::filesystem::path& path) {
    Settings settings = defaults();
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return settings;
    }
    std::stringstream contents;
    contents << stream.rdbuf();
    const std::string text = python_json_readable(contents.str());
    Json data;
    try {
        data = Json::parse(text);
    } catch (const Json::exception& error) {
        // The defaults replace it at the next save: a copy for the user to look at.
        std::filesystem::path copy = path;
        copy += ".unreadable";
        std::error_code ignored;
        keep_folder_private(path.parent_path());
        std::filesystem::copy_file(path, copy, std::filesystem::copy_options::overwrite_existing, ignored);
        keep_file_private(copy);
        app_log::Logger("vml.settings")
            .warning("The settings could not be read, and the defaults are used (copy kept as " +
                     platform::path_text(copy) + "): " + where_unreadable(text, error));
        return settings;
    }
    if (!data.is_object()) {
        return settings;
    }
    for (auto item = data.begin(); item != data.end(); ++item) {
        if (item.key() == "weak_segment_profiles") {
            // A list of objects; kept as JSON, as the CVVDP presets are.
            if (item.value().is_array() &&
                std::all_of(item.value().begin(), item.value().end(), [](const Json& v) { return v.is_object(); })) {
                settings.weak_segment_profiles_json = item.value().dump();
            } else {
                log_unfit(item.key(), item.value(), "list[dict]");
            }
            continue;
        }
        if (item.key() == "cvvdp_presets") {
            // A list of objects; kept as JSON for the CVVDP presets to read.
            if (item.value().is_array() &&
                std::all_of(item.value().begin(), item.value().end(), [](const Json& v) { return v.is_object(); })) {
                settings.cvvdp_presets_json = item.value().dump();
            } else {
                log_unfit(item.key(), item.value(), "list[dict]");
            }
            continue;
        }
        const auto found = fields().find(item.key());
        if (found != fields().end()) {
            if (const std::optional<std::string> kind = set_if_it_fits(settings, found->second, item.value())) {
                log_unfit(item.key(), item.value(), *kind);
            }
        }
    }
    if (settings.theme == "mocha") {  // a theme before there were looks
        settings.theme = "dark";
        settings.dark_look = "mocha";
    }
    if (!data.contains("export_items")) {
        carry_over_auto_export(settings);
    } else if (!data.contains("export_formats")) {
        carry_over_result_formats(settings);
    }
    hide_new_metrics(settings);
    return settings;
}

std::optional<std::filesystem::path> Settings::ffmpeg_dir_path() const {
    const std::size_t start = ffmpeg_dir.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t end = ffmpeg_dir.find_last_not_of(" \t\r\n");
    return platform::path_from_text(ffmpeg_dir.substr(start, end - start + 1));
}

namespace {

// Where a metric's default_compute_<key> setting is kept, or nullptr for a
// key with none.
bool Settings::* default_compute_field(std::string_view key) {
    const auto found = fields().find("default_compute_" + std::string(key));
    if (found == fields().end()) {
        return nullptr;
    }
    const auto* flag = std::get_if<bool Settings::*>(&found->second);
    return flag != nullptr ? *flag : nullptr;
}

}  // namespace

bool* Settings::default_compute(std::string_view key) {
    bool Settings::* const field = default_compute_field(key);
    return field != nullptr ? &(this->*field) : nullptr;
}

const bool* Settings::default_compute(std::string_view key) const {
    bool Settings::* const field = default_compute_field(key);
    return field != nullptr ? &(this->*field) : nullptr;
}

std::optional<std::filesystem::path> Settings::cache_dir_path() const {
    const std::size_t start = cache_dir.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t end = cache_dir.find_last_not_of(" \t\r\n");
    return platform::path_from_text(cache_dir.substr(start, end - start + 1));
}

std::string Settings::text() const {
    // In the Python dataclass's order, which json.dumps(asdict(...)) keeps.
    Json data = Json::object();
    data["ffmpeg_dir"] = ffmpeg_dir;
    data["cache_dir"] = cache_dir;
    data["export_dir"] = export_dir;
    data["export_items"] = export_items;
    data["export_formats"] = export_formats;
    data["export_zip"] = export_zip;
    data["quick_save_formats"] = quick_save_formats;
    data["after_run_export"] = after_run_export;
    data["after_run_place"] = after_run_place;
    data["default_gpu_decode"] = default_gpu_decode;
    data["default_compute_psnr"] = default_compute_psnr;
    data["default_compute_ssim"] = default_compute_ssim;
    data["default_compute_xpsnr"] = default_compute_xpsnr;
    data["default_compute_psnr_yuv"] = default_compute_psnr_yuv;
    data["default_compute_psnr_yuv_mse"] = default_compute_psnr_yuv_mse;
    data["default_compute_vmaf_yuv"] = default_compute_vmaf_yuv;
    data["default_compute_ssim_yuv"] = default_compute_ssim_yuv;
    data["default_compute_ms_ssim"] = default_compute_ms_ssim;
    data["default_compute_ms_ssim_yuv"] = default_compute_ms_ssim_yuv;
    data["default_compute_psnr_hvs"] = default_compute_psnr_hvs;
    data["default_compute_cambi"] = default_compute_cambi;
    data["default_compute_ciede2000"] = default_compute_ciede2000;
    data["default_compute_vmaf"] = default_compute_vmaf;
    data["default_compute_vmaf_neg"] = default_compute_vmaf_neg;
    data["default_compute_vmaf_v1"] = default_compute_vmaf_v1;
    data["default_compute_vmaf_hdr"] = default_compute_vmaf_hdr;
    data["default_compute_ssimulacra2"] = default_compute_ssimulacra2;
    data["default_compute_butteraugli"] = default_compute_butteraugli;
    data["default_compute_cvvdp"] = default_compute_cvvdp;
    data["default_vmaf_on_gpu"] = default_vmaf_on_gpu;
    data["default_vship_build"] = default_vship_build;
    data["film_grain_default"] = film_grain_default;
    data["graph_metric"] = graph_metric;
    data["graph_worst_frames"] = graph_worst_frames;
    data["weak_segment_conditions"] = weak_segment_conditions;
    data["weak_segments_all"] = weak_segments_all;
    data["weak_segment_min_frames"] = weak_segment_min_frames;
    data["weak_segment_merge_frames"] = weak_segment_merge_frames;
    Json profiles = Json::parse(weak_segment_profiles_json, nullptr, false);
    data["weak_segment_profiles"] = profiles.is_array() ? profiles : Json::array();
    data["weak_segment_profile"] = weak_segment_profile;
    data["parallel_jobs"] = parallel_jobs;
    data["gpu_metrics_together"] = gpu_metrics_together;
    data["gpu"] = gpu;
    data["vvc_decode_on_gpu"] = vvc_decode_on_gpu;
    data["recognise_test_remuxes"] = recognise_test_remuxes;
    data["use_cache"] = use_cache;
    data["check_frame_counts"] = check_frame_counts;
    data["check_for_updates"] = check_for_updates;
    data["confirm_close_during_run"] = confirm_close_during_run;
    data["skipped_update_version"] = skipped_update_version;
    data["language"] = language;
    data["theme"] = theme;
    data["interface"] = interface_style;
    data["accent_colour"] = accent_colour;
    data["light_look"] = light_look;
    data["dark_look"] = dark_look;
    data["video_names"] = video_names;
    data["video_name_start"] = video_name_start;
    data["video_name_end"] = video_name_end;
    data["remember_window_size"] = remember_window_size;
    data["window_width"] = window_width;
    data["window_height"] = window_height;
    data["window_geometry"] = window_geometry;
    data["last_reference_dir"] = last_reference_dir;
    data["last_test_dir"] = last_test_dir;
    data["remember_videos"] = remember_videos;
    data["remembered_reference"] = remembered_reference;
    data["remembered_tests"] = remembered_tests;
    data["highlight_best_worst"] = highlight_best_worst;
    data["frame_preview_color_mode"] = frame_preview_color_mode;
    data["compare_decoded_videos"] = compare_decoded_videos;
    data["compare_loop"] = compare_loop;
    data["compare_loop_before_ms"] = compare_loop_before_ms;
    data["compare_loop_after_ms"] = compare_loop_after_ms;
    data["export_graph_theme"] = export_graph_theme;
    data["export_graph_background"] = export_graph_background;
    data["export_graph_text"] = export_graph_text;
    data["hidden_metrics"] = hidden_metrics;
    data["show_yuv_planes"] = show_yuv_planes;
    // The metrics this window has, and those an earlier or later version's
    // file listed: a metric added later is then hidden at first there too.
    std::vector<std::string> listed = listed_metrics;
    for (const MetricDefinition& metric : metrics()) {
        // A plane is listed with its metric's switch, not as a metric.
        if (!metric.is_plane() && std::find(listed.begin(), listed.end(), metric.key) == listed.end()) {
            listed.emplace_back(metric.key);
        }
    }
    data["listed_metrics"] = listed;
    Json presets = Json::parse(cvvdp_presets_json, nullptr, false);
    data["cvvdp_presets"] = presets.is_array() ? presets : Json::array();
    data["cvvdp_default_preset"] = cvvdp_default_preset;
    data["notify_finished"] = notify_finished;
    data["notify_with_failures"] = notify_with_failures;
    data["notify_cancelled"] = notify_cancelled;
    data["notify_min_minutes"] = notify_min_minutes;
    data["notify_title"] = notify_title;
    data["notify_message"] = notify_message;
    data["notify_video_line"] = notify_video_line;
    data["notify_secrets_encrypted"] = notify_secrets_encrypted;
    data["pushover_enabled"] = pushover_enabled;
    data["pushover_user"] = pushover_user;
    data["pushover_token"] = pushover_token;
    data["pushover_priority"] = pushover_priority;
    data["discord_enabled"] = discord_enabled;
    data["discord_webhook"] = discord_webhook;
    data["discord_mention"] = discord_mention;
    data["discord_mention_when"] = discord_mention_when;
    data["discord_silent"] = discord_silent;
    data["gotify_enabled"] = gotify_enabled;
    data["gotify_url"] = gotify_url;
    data["gotify_token"] = gotify_token;
    data["gotify_priority"] = gotify_priority;
    data["ntfy_enabled"] = ntfy_enabled;
    data["ntfy_url"] = ntfy_url;
    data["ntfy_topic"] = ntfy_topic;
    data["ntfy_token"] = ntfy_token;
    data["ntfy_priority"] = ntfy_priority;
    data["pushbullet_enabled"] = pushbullet_enabled;
    data["pushbullet_token"] = pushbullet_token;
    data["pushbullet_device"] = pushbullet_device;
    data["slack_enabled"] = slack_enabled;
    data["slack_webhook"] = slack_webhook;
    data["telegram_enabled"] = telegram_enabled;
    data["telegram_token"] = telegram_token;
    data["telegram_chat"] = telegram_chat;
    data["telegram_silent"] = telegram_silent;
    data["webhook_enabled"] = webhook_enabled;
    data["webhook_url"] = webhook_url;
    data["webhook_header"] = webhook_header;
    data["saved_results_column_widths"] = saved_results_column_widths;
    data["saved_results_hidden_columns"] = saved_results_hidden_columns;
    return python_json_dumps(data, 2);
}

std::optional<std::string> Settings::save() const { return save_to(file()); }

std::optional<std::string> Settings::save_to(const std::filesystem::path& path) const {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    keep_folder_private(path.parent_path());
    std::random_device random;
    const std::filesystem::path temporary =
        path.parent_path() /
        platform::path_from_text(platform::path_text(path.filename()) + "." + std::to_string(random()) + ".tmp");
    {
        std::ofstream stream(temporary, std::ios::binary);
        keep_file_private(temporary);  // before anything is in it
        const std::string contents = text();
        stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        stream.close();
        if (!stream) {
            std::filesystem::remove(temporary, error);
            return "Could not save settings: could not write " + platform::path_text(temporary);
        }
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return "Could not save settings: " + error.message();
    }
    return std::nullopt;
}

std::vector<std::string> Settings::default_extra_features() const {
    std::vector<std::string> features;
    if (default_compute_psnr) {
        features.emplace_back("name=psnr");
    }
    if (default_compute_ssim) {
        features.emplace_back("name=float_ssim");
    }
    return features;
}

}  // namespace vml
