// The app's preferences, saved in settings.json between sessions. A port of
// vmaf_app/core/settings.py: the same file, read the same way, so both apps
// share it.
#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/notifications.h"

namespace vml {

// What an export can include (Settings::export_items), in the order Settings
// shows them: results, per-frame scores ("frames"), graphs (every metric's,
// and the bitrate with the reference's), worst frames (every metric's) and
// weak segments. A new kind of export is added here.
const std::vector<std::string>& export_item_keys();
// The formats the files of an export are written in (Settings::export_formats),
// in Settings' order: "json" and "csv". Graphs are images whatever is ticked.
const std::vector<std::string>& export_format_keys();

struct Settings;
// What an export saves files of: Settings::export_items, less all but the
// graphs while no format is ticked.
std::vector<std::string> exported_items(const Settings& settings);

// The metrics Add/remove metrics lists unticked until the user ticks them
// (Settings::hidden_metrics): the ones codec comparisons such as MSU's and
// Xiph's report, which most users do not need.
inline constexpr std::array<std::string_view, 5> metrics_hidden_at_first{"ms_ssim", "ms_ssim_yuv", "psnr_hvs",
                                                                        "cambi", "ciede2000"};

// Above this many logical cores, a new install scores two videos at once.
constexpr unsigned parallel_by_default_above_cores = 12;

// The parallel-jobs setting a PC with `cores` logical cores starts with.
int default_parallel_jobs(unsigned cores);

// The environment variable that names the folder of the app's own files in
// place of the user's home/.videometricslab: a profile of its own, for the
// window and the command line alike.
inline constexpr const char* app_data_variable = "VML_HOME";

// The folder of the app's own files: settings.json, the saved scores
// (results_cache), the logs and the GPU checks. VML_HOME's folder where it
// is set (not empty), made absolute; else .videometricslab in the user's home.
std::filesystem::path app_data_folder();

struct Settings {
    // The folder holding ffmpeg and ffprobe; empty means "search the PATH".
    std::string ffmpeg_dir;
    // Where finished runs are saved; empty means the per-user default.
    std::string cache_dir;
    // The folder CSV and PNG exports go to; empty means "ask each time".
    std::string export_dir;
    // What an export includes (Settings > Storage > Exports include), the
    // right-click menus' and the one after each run alike: export_items' keys,
    // in any order; and whether as a .zip file or loose files.
    // Per-frame scores are left out until asked for: a long video's are big.
    std::vector<std::string> export_items = {"results", "graphs", "worst_frames", "weak_segments"};
    // The formats results and weak segments are written in, each ticked one:
    // export_format_keys' keys.
    std::vector<std::string> export_formats = {"json", "csv"};
    bool export_zip = true;
    // Whether each video is exported after its run, and where: "beside" next
    // to the test video, "folder" in export_dir (next to the test video while
    // that is empty).
    bool after_run_export = false;
    std::string after_run_place = "beside";
    // Before export_items, read only to carry their choice over (never
    // saved): after each run, results and graphs as a .zip ("" not,
    // "beside", "folder"), and weak segments as JSON and CSV, likewise.
    std::string auto_export;
    std::string auto_export_segments;

    // What a newly added video starts with (each row's own settings are
    // edited in the window's Options panel).
    bool default_gpu_decode = true;
    bool default_compute_psnr = true;
    bool default_compute_ssim = true;
    bool default_compute_xpsnr = true;
    // Off by default: the 6:1:1 PSNR of JVET's reports and YUV-SSIM of
    // MSU's, for comparing with them.
    bool default_compute_psnr_yuv = false;
    bool default_compute_ssim_yuv = false;
    // Ticked, but hidden at first (metrics_hidden_at_first): calculated
    // once shown in Add/remove metrics.
    bool default_compute_ms_ssim = true;
    bool default_compute_ms_ssim_yuv = true;
    bool default_compute_psnr_hvs = true;
    bool default_compute_cambi = true;
    bool default_compute_ciede2000 = true;
    bool default_compute_vmaf = true;
    bool default_compute_vmaf_neg = true;
    bool default_compute_vmaf_v1 = true;
    // Off by default: it scores PQ (HDR10) videos only.
    bool default_compute_vmaf_hdr = false;
    bool default_compute_ssimulacra2 = true;
    bool default_compute_butteraugli = true;
    bool default_compute_cvvdp = true;
    // Whether VMAF v0.6.1, NEG and v1 run on the GPU.
    bool default_vmaf_on_gpu = true;
    // The Vship build new videos use for SSIMULACRA2, Butteraugli and CVVDP
    // (vship_build_value: "vship-fast" or "vship").
    std::string default_vship_build = "vship-fast";
    // How a video with film grain it can be scored with or without is scored
    // until its own choice is made: "with", "without", or "" to ask at
    // Calculate metrics (VideoInfo::with_film_grain).
    std::string film_grain_default;
    std::string graph_metric = "vmaf";
    // How many dips the Worst Frames tab's Worst frames list shows (10, 25 or 50).
    int graph_worst_frames = 10;
    // Metric Graphs' Weak segments, as last set there (saved_segment_rules,
    // weak_frames.h): its conditions as the command line's --segments writes
    // them ("vmaf<90"; none: the metric shown, from its first band of worse
    // frames), whether every one must hold rather than any, the shortest
    // segment and the gap joined across, in frames.
    std::vector<std::string> weak_segment_conditions;
    bool weak_segments_all = false;
    int weak_segment_min_frames = 1;
    int weak_segment_merge_frames = 0;
    // Further weak segment profiles (segment_profiles.h), as JSON text of
    // their list ([{"name", "conditions", "all", "min_frames",
    // "merge_frames"}]); the rules above are the Default profile. The one in
    // use: a name of theirs, or "" (or a name none has) for Default.
    std::string weak_segment_profiles_json = "[]";
    std::string weak_segment_profile;

    // How many videos to score at once (1 or 2).
    int parallel_jobs = 1;
    // Calculate a video's GPU metrics in one Vship pass; otherwise one at a
    // time.
    bool gpu_metrics_together = false;
    // The GPU the app scores on (gpu_choice.h), by the name DirectX gives it,
    // with its place where two GPUs have that name ("... #2"); empty for
    // automatic, a discrete GPU first. Applies from the app's next start.
    std::string gpu;
    // Decode a VVC video on the GPU (vvdec-vulkan, its pictures handed over to the GPU metrics) where the GPU
    // metrics can take them; otherwise, and by default, the app's VVdeC on the CPU. The scores are the same.
    bool vvc_decode_on_gpu = false;
    // Recognise a test video remuxed after it was scored (raw, MKV, MP4) by
    // its stream (metric_cache.h's StreamFingerprinter): each test video is
    // read once more after its scores are saved, which takes as long as
    // reading the file. Off by default.
    bool recognise_test_remuxes = false;
    // Use a saved result instead of calculating again.
    bool use_cache = true;
    // When a run of two videos scored fewer frame pairs than their lengths
    // hold, their frames are counted to tell whether they were paired out of
    // step (frame_pairing.h): the counts their files state, always; with
    // this, a file that states none is read through to count them. Off by
    // default: minutes for a film on a network drive.
    bool check_frame_counts = false;

    bool check_for_updates = true;
    // Whether closing the window during a run asks first (the run stops):
    // off once "Don't ask again" was ticked, back on in Settings > General.
    bool confirm_close_during_run = true;
    std::string skipped_update_version;
    std::string language;
    std::string theme;
    // The window's interface: "modern" (VML Modern: cards, grouped
    // toolbars, one accent button per tab; the default, also for a settings
    // file from before there was a choice) or "classic" (the system's own
    // controls); anything else is classic.
    std::string interface_style = "modern";
    // The accent colour, "#rrggbb", or "" for the system's.
    std::string accent_colour;
    // VML Modern's look for light ("light", "dim", "paper" or "desert") and
    // for dark ("dark", "mocha", "joker", "nitrate" or "phosphor"); theme picks
    // which is shown, "" (the
    // system's) either. A theme of "mocha", from before there were looks, is
    // read as "dark" with the dark look "mocha".
    std::string light_look = "light";
    std::string dark_look = "dark";
    // How the window shows a test video's name (video_names.h): "differs"
    // (what tells the videos shown apart, their shared start and end left
    // out), "ends" (its first video_name_start and last video_name_end
    // characters) or "full". Tooltips, exports and file names keep the
    // whole name.
    std::string video_names = "differs";
    int video_name_start = 10;
    int video_name_end = 16;
    bool remember_window_size = true;
    // The size the window was closed at; these defaults stand for none saved
    // (the window then opens at its default for the screen).
    int window_width = 1280;
    int window_height = 800;
    // Where and how the window was last closed (Qt's saveGeometry, as
    // base64): its screen, place, size and whether maximized. Empty until
    // first saved; the size above is then used, as it was before 2.6.
    std::string window_geometry;
    std::string last_reference_dir;
    std::string last_test_dir;
    bool remember_videos = false;
    std::string remembered_reference;
    std::vector<std::string> remembered_tests;
    bool highlight_best_worst = false;
    std::string frame_preview_color_mode = "display_aware";
    int compare_decoded_videos = 3;
    // Video Compare's loop: whether it is on (Loop ticked), and how long it
    // plays before and after its frame (ms).
    bool compare_loop = false;
    int compare_loop_before_ms = 1000;
    int compare_loop_after_ms = 1000;
    // Exported graphs' colours (Metric Graphs' and the Bitrate tab's Export
    // PNG): the chart colours of "light", "dark" or "window" (the theme in
    // use), and a background and text colour ("#rrggbb") in place of
    // theirs, none when empty.
    std::string export_graph_theme = "light";
    std::string export_graph_background;
    std::string export_graph_text;
    // The metrics left out of the test-video table (and not calculated).
    std::vector<std::string> hidden_metrics{metrics_hidden_at_first.begin(), metrics_hidden_at_first.end()};
    // Add/remove metrics' "Y, U and V of the YUV metrics": the planes of
    // PSNR YUV, SSIM YUV, MS-SSIM YUV and PSNR-HVS-M shown and calculated
    // with their metric (MetricDefinition::plane_of).
    bool show_yuv_planes = false;
    // Every metric the window had when the settings were saved: one added
    // since, if hidden at first, is hidden when they are read (load_from).
    std::vector<std::string> listed_metrics;
    // The CVVDP presets the user saved, as JSON text of their list
    // ([{"name": ..., "settings": ...}]), and the preset new videos start
    // with: "", or a name no preset has, is Automatic (automatic_cvvdp_preset:
    // the SDR or HDR display, as the reference is).
    std::string cvvdp_presets_json = "[]";
    std::string cvvdp_default_preset;
    // Notifications when a run ends (Settings > Notifications,
    // notifications.h): for which outcomes, for runs of at least so many
    // minutes, and the message's templates (title, message, each test
    // video's line).
    bool notify_finished = true;
    bool notify_with_failures = true;
    bool notify_cancelled = true;
    int notify_min_minutes = 0;
    std::string notify_title = std::string(standard_notification_title);
    std::string notify_message = std::string(standard_notification_message);
    std::string notify_video_line = std::string(standard_notification_video_line);
    // Whether the services' keys are stored encrypted for this PC's user
    // (stored_secret), or as they are.
    bool notify_secrets_encrypted = true;
    // Pushover: its user key and application token (as stored_secret keeps
    // them), and the messages' priority (-2 to 1).
    bool pushover_enabled = false;
    std::string pushover_user;
    std::string pushover_token;
    int pushover_priority = 0;
    // Discord: a channel's webhook URL (as stored_secret keeps it); whom a
    // message mentions ("<@&role ID>", "<@user ID>", "@here", "@everyone";
    // "" no one) and when ("problems": a run with failures or cancelled;
    // "always"); and whether a message that mentions no one is sent without
    // a push or a sound.
    bool discord_enabled = false;
    std::string discord_webhook;
    std::string discord_mention;
    std::string discord_mention_when = "problems";
    bool discord_silent = false;
    // Gotify: its server's address, an application's token (as stored_secret
    // keeps it), and the messages' priority (0 to 10).
    bool gotify_enabled = false;
    std::string gotify_url;
    std::string gotify_token;
    int gotify_priority = 5;
    // ntfy: its server's address, the topic (whose name is its password, so
    // kept as a secret), an access token where the topic needs one, and the
    // messages' priority (1 to 5).
    bool ntfy_enabled = false;
    std::string ntfy_url = "https://ntfy.sh";
    std::string ntfy_topic;
    std::string ntfy_token;
    int ntfy_priority = 3;
    // Pushbullet: an access token, and a device's iden ("" every device).
    bool pushbullet_enabled = false;
    std::string pushbullet_token;
    std::string pushbullet_device;
    // Slack: an incoming webhook's URL.
    bool slack_enabled = false;
    std::string slack_webhook;
    // Telegram: a bot's token, the chat's ID (or a public channel's
    // @name), and whether messages are sent without a sound.
    bool telegram_enabled = false;
    std::string telegram_token;
    std::string telegram_chat;
    bool telegram_silent = false;
    // A webhook of the user's own (Home Assistant, n8n, a script): its URL
    // and a header sent with it ("Authorization: Bearer ...", or ""). It
    // gets the run as JSON (webhook_request).
    bool webhook_enabled = false;
    std::string webhook_url;
    std::string webhook_header;
    // The Saved Results table's columns (saved_results_panel.h): their
    // widths in pixels, one a column in its order (Test video, Reference,
    // Saved, Metrics, Size), or none for widths fitted to the tab; and the
    // columns hidden by their key ("reference", "saved", "metrics", "size";
    // the test video's is always shown).
    std::vector<int> saved_results_column_widths;
    std::vector<std::string> saved_results_hidden_columns;

    // The saved settings: defaults for a missing or unreadable file, and
    // for each value of the wrong type (that setting alone); unknown keys
    // are ignored. Reading makes the settings folder if it is missing, as
    // the Python app does.
    static Settings load();
    // load() from a given file, for tests.
    static Settings load_from(const std::filesystem::path& file);
    // load_from() for a file the user named (the command line's --settings),
    // which must be there and be JSON: throws std::runtime_error saying why
    // not, where load_from() would take the defaults. Writes nothing.
    static Settings load_checked(const std::filesystem::path& file);
    // Every setting at its default, as a new profile has them.
    static Settings defaults();
    // Where settings.json is: in app_data_folder().
    static std::filesystem::path file();

    // Saves the settings as the Python app does (json.dumps(asdict(self),
    // indent=2)): every setting, in its order. The file is written whole
    // beside the old one, then put in its place, so a crash or a full disk
    // part-way through never leaves a cut-off file. Returns nothing on
    // success, or a message to show the user.
    std::optional<std::string> save() const;
    // save() to a given file, for tests.
    std::optional<std::string> save_to(const std::filesystem::path& file) const;
    // The text save() writes.
    std::string text() const;

    // The folders, or nothing when the setting is empty.
    std::optional<std::filesystem::path> ffmpeg_dir_path() const;
    std::optional<std::filesystem::path> cache_dir_path() const;
    // The libvmaf features these defaults ask for: PSNR and SSIM.
    std::vector<std::string> default_extra_features() const;
    // Whether new videos have a metric ("vmaf", "psnr", ...) ticked: its
    // default_compute_<key> setting, or nullptr for a key with none.
    bool* default_compute(std::string_view key);
    const bool* default_compute(std::string_view key) const;
};

// The settings that hold a notification service's key, or a URL or header
// with one in it: kept as stored_secret keeps them, never written to the
// log, entered in Settings > Notifications' key fields. Each one's name in
// settings.json and its member. The one list of them.
const std::vector<std::pair<std::string_view, std::string Settings::*>>& secret_settings();

}  // namespace vml
