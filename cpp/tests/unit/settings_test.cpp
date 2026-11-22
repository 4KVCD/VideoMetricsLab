#include "vml/settings.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/app_log.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

using Json = nlohmann::json;

// Each setting by its key in settings.json, as JSON, to hold to the Python
// app's loaded settings.
const std::map<std::string, std::function<Json(const Settings&)>>& settings_by_key() {
    static const std::map<std::string, std::function<Json(const Settings&)>> by_key{
        {"ffmpeg_dir", [](const Settings& s) { return Json(s.ffmpeg_dir); }},
        {"cache_dir", [](const Settings& s) { return Json(s.cache_dir); }},
        {"export_dir", [](const Settings& s) { return Json(s.export_dir); }},
        {"auto_export", [](const Settings& s) { return Json(s.auto_export); }},
        {"auto_export_segments", [](const Settings& s) { return Json(s.auto_export_segments); }},
        {"export_items", [](const Settings& s) { return Json(s.export_items); }},
        {"export_formats", [](const Settings& s) { return Json(s.export_formats); }},
        {"export_zip", [](const Settings& s) { return Json(s.export_zip); }},
        {"after_run_export", [](const Settings& s) { return Json(s.after_run_export); }},
        {"after_run_place", [](const Settings& s) { return Json(s.after_run_place); }},
        {"default_gpu_decode", [](const Settings& s) { return Json(s.default_gpu_decode); }},
        {"default_compute_psnr", [](const Settings& s) { return Json(s.default_compute_psnr); }},
        {"default_compute_ssim", [](const Settings& s) { return Json(s.default_compute_ssim); }},
        {"default_compute_xpsnr", [](const Settings& s) { return Json(s.default_compute_xpsnr); }},
        {"default_compute_vmaf", [](const Settings& s) { return Json(s.default_compute_vmaf); }},
        {"default_compute_vmaf_neg", [](const Settings& s) { return Json(s.default_compute_vmaf_neg); }},
        {"default_compute_vmaf_v1", [](const Settings& s) { return Json(s.default_compute_vmaf_v1); }},
        {"default_compute_ssimulacra2", [](const Settings& s) { return Json(s.default_compute_ssimulacra2); }},
        {"default_compute_butteraugli", [](const Settings& s) { return Json(s.default_compute_butteraugli); }},
        {"default_compute_cvvdp", [](const Settings& s) { return Json(s.default_compute_cvvdp); }},
        {"default_vmaf_on_gpu", [](const Settings& s) { return Json(s.default_vmaf_on_gpu); }},
        {"default_vship_build", [](const Settings& s) { return Json(s.default_vship_build); }},
        {"film_grain_default", [](const Settings& s) { return Json(s.film_grain_default); }},
        {"graph_metric", [](const Settings& s) { return Json(s.graph_metric); }},
        {"weak_segment_conditions", [](const Settings& s) { return Json(s.weak_segment_conditions); }},
        {"weak_segments_all", [](const Settings& s) { return Json(s.weak_segments_all); }},
        {"weak_segment_min_frames", [](const Settings& s) { return Json(s.weak_segment_min_frames); }},
        {"weak_segment_merge_frames", [](const Settings& s) { return Json(s.weak_segment_merge_frames); }},
        {"weak_segment_profiles", [](const Settings& s) { return Json(s.weak_segment_profiles_json); }},
        {"weak_segment_profile", [](const Settings& s) { return Json(s.weak_segment_profile); }},
        {"parallel_jobs", [](const Settings& s) { return Json(s.parallel_jobs); }},
        {"gpu_metrics_together", [](const Settings& s) { return Json(s.gpu_metrics_together); }},
        {"vvc_decode_on_gpu", [](const Settings& s) { return Json(s.vvc_decode_on_gpu); }},
        {"use_cache", [](const Settings& s) { return Json(s.use_cache); }},
        {"check_frame_counts", [](const Settings& s) { return Json(s.check_frame_counts); }},
        {"check_for_updates", [](const Settings& s) { return Json(s.check_for_updates); }},
        {"skipped_update_version", [](const Settings& s) { return Json(s.skipped_update_version); }},
        {"language", [](const Settings& s) { return Json(s.language); }},
        {"theme", [](const Settings& s) { return Json(s.theme); }},
        {"interface", [](const Settings& s) { return Json(s.interface_style); }},
        {"accent_colour", [](const Settings& s) { return Json(s.accent_colour); }},
        {"video_names", [](const Settings& s) { return Json(s.video_names); }},
        {"video_name_start", [](const Settings& s) { return Json(s.video_name_start); }},
        {"video_name_end", [](const Settings& s) { return Json(s.video_name_end); }},
        {"remember_window_size", [](const Settings& s) { return Json(s.remember_window_size); }},
        {"window_width", [](const Settings& s) { return Json(s.window_width); }},
        {"window_height", [](const Settings& s) { return Json(s.window_height); }},
        {"last_reference_dir", [](const Settings& s) { return Json(s.last_reference_dir); }},
        {"last_test_dir", [](const Settings& s) { return Json(s.last_test_dir); }},
        {"remember_videos", [](const Settings& s) { return Json(s.remember_videos); }},
        {"remembered_reference", [](const Settings& s) { return Json(s.remembered_reference); }},
        {"remembered_tests", [](const Settings& s) { return Json(s.remembered_tests); }},
        {"highlight_best_worst", [](const Settings& s) { return Json(s.highlight_best_worst); }},
        {"frame_preview_color_mode", [](const Settings& s) { return Json(s.frame_preview_color_mode); }},
        {"compare_decoded_videos", [](const Settings& s) { return Json(s.compare_decoded_videos); }},
        {"hidden_metrics", [](const Settings& s) { return Json(s.hidden_metrics); }},
        {"cvvdp_presets", [](const Settings& s) { return Json(s.cvvdp_presets_json); }},
        {"cvvdp_default_preset", [](const Settings& s) { return Json(s.cvvdp_default_preset); }},
        {"saved_results_column_widths", [](const Settings& s) { return Json(s.saved_results_column_widths); }},
        {"saved_results_hidden_columns", [](const Settings& s) { return Json(s.saved_results_hidden_columns); }},
    };
    return by_key;
}

// settings.json is read as the Python app reads it: every setting it has
// (none missing here), each value of the wrong type left at its default
// while the others load, unknown keys ignored, and defaults for a file that
// is missing, not JSON or not an object.
TEST(Settings, LoadsSettingsJsonAsThePythonAppDoes) {
    const nlohmann::json reference = test::load_reference("settings.json");
    const std::filesystem::path file = std::filesystem::temp_directory_path() / "vml_settings_test.json";
    ASSERT_FALSE(reference["cases"].empty());
    for (const nlohmann::json& item : reference["cases"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        std::filesystem::remove(file);
        if (!item["file"].is_null()) {
            std::ofstream(file, std::ios::binary) << item["file"].get<std::string>();
        }
        const Settings settings = Settings::load_from(file);
        for (const auto& [key, value] : item["expected"].items()) {
            const auto found = settings_by_key().find(key);
            if (found == settings_by_key().end()) {
                ADD_FAILURE() << "the C++ Settings has no " << key;
                continue;
            }
            const Json expected =
                key == "parallel_jobs" && value.is_null()
                    ? Json(default_parallel_jobs(platform::logical_cores()))  // the PC's default
                    : value;
            EXPECT_EQ(found->second(settings), expected) << key;
        }
    }
    std::filesystem::remove(file);
}

// What exports include, from a settings.json saved before they were chosen
// one by one (when only after each run exported): results (JSON) and graphs
// in one .zip, weak segments as JSON and CSV files of their own, where each
// was going; both, all in the .zip; neither, the defaults (everything, as
// JSON and CSV, in a .zip, not after each run). A file that has the new
// choice keeps it.
TEST(Settings, ExportsCarryOverTheOldAfterRunChoices) {
    const test::Folder folder;
    const auto load = [&folder](const std::string& text) {
        std::ofstream(folder / "settings.json", std::ios::binary) << text;
        return Settings::load_from(folder / "settings.json");
    };
    using Items = std::vector<std::string>;
    Settings settings = load(R"({"auto_export": "folder"})");
    EXPECT_EQ(settings.export_items, (Items{"results", "graphs"}));
    EXPECT_EQ(settings.export_formats, Items{"json"});
    EXPECT_TRUE(settings.export_zip);
    EXPECT_TRUE(settings.after_run_export);
    EXPECT_EQ(settings.after_run_place, "folder");
    settings = load(R"({"auto_export_segments": "beside"})");
    EXPECT_EQ(settings.export_items, (Items{"weak_segments"}));
    EXPECT_EQ(settings.export_formats, (Items{"json", "csv"}));
    EXPECT_FALSE(settings.export_zip);
    EXPECT_TRUE(settings.after_run_export);
    EXPECT_EQ(settings.after_run_place, "beside");
    settings = load(R"({"auto_export": "beside", "auto_export_segments": "folder"})");
    EXPECT_EQ(settings.export_items, (Items{"results", "graphs", "weak_segments"}));
    EXPECT_EQ(settings.export_formats, (Items{"json", "csv"}));
    EXPECT_TRUE(settings.export_zip);
    EXPECT_EQ(settings.after_run_place, "beside");
    settings = load(R"({"auto_export": ""})");
    EXPECT_EQ(settings.export_items,
              (Items{"results", "graphs", "worst_frames", "weak_segments"}));  // per-frame: asked for
    EXPECT_EQ(settings.export_formats, export_format_keys());
    EXPECT_TRUE(settings.export_zip);
    EXPECT_FALSE(settings.after_run_export);
    settings = load(R"({"auto_export": "folder", "export_items": ["results"], "export_formats": ["csv"],
                         "export_zip": false})");
    EXPECT_EQ(settings.export_items, (Items{"results"}));
    EXPECT_EQ(settings.export_formats, Items{"csv"});
    EXPECT_FALSE(settings.export_zip);
    EXPECT_FALSE(settings.after_run_export);
    // Saved: the new choice, never the old keys.
    ASSERT_EQ(settings.save_to(folder / "saved.json"), std::nullopt);
    const std::string saved = platform::read_file(folder / "saved.json").value_or("");
    EXPECT_EQ(saved.find("auto_export"), std::string::npos);
    EXPECT_NE(saved.find("\"export_items\": [\n    \"results\"\n  ]"), std::string::npos);
    EXPECT_NE(saved.find("\"export_formats\": [\n    \"csv\"\n  ]"), std::string::npos);
}

// VML Modern is the default interface, for a settings file from before
// there was a choice too; one that picked VML Classic keeps it.
TEST(Settings, ModernIsTheDefaultInterface) {
    const test::Folder folder;
    const auto load = [&folder](const std::string& text) {
        std::ofstream(folder / "settings.json", std::ios::binary) << text;
        return Settings::load_from(folder / "settings.json");
    };
    EXPECT_EQ(Settings{}.interface_style, "modern");
    EXPECT_EQ(load(R"({"theme": "dark"})").interface_style, "modern");
    EXPECT_EQ(load(R"({"interface": "classic"})").interface_style, "classic");
}

// Exports as the first v2.7 builds saved them, results by format: Results in
// those formats; weak segments alone, in both, as they were written.
TEST(Settings, ExportsCarryOverResultsByFormat) {
    const test::Folder folder;
    const auto load = [&folder](const std::string& text) {
        std::ofstream(folder / "settings.json", std::ios::binary) << text;
        return Settings::load_from(folder / "settings.json");
    };
    using Items = std::vector<std::string>;
    Settings settings = load(R"({"export_items": ["results_json", "graphs", "weak_segments"]})");
    EXPECT_EQ(settings.export_items, (Items{"results", "graphs", "weak_segments"}));
    EXPECT_EQ(settings.export_formats, Items{"json"});
    settings = load(R"({"export_items": ["results_json", "results_csv"]})");
    EXPECT_EQ(settings.export_items, Items{"results"});
    EXPECT_EQ(settings.export_formats, (Items{"json", "csv"}));
    settings = load(R"({"export_items": ["results_csv", "graphs"]})");
    EXPECT_EQ(settings.export_items, (Items{"results", "graphs"}));
    EXPECT_EQ(settings.export_formats, Items{"csv"});
    settings = load(R"({"export_items": ["graphs", "weak_segments"]})");
    EXPECT_EQ(settings.export_items, (Items{"graphs", "weak_segments"}));
    EXPECT_EQ(settings.export_formats, (Items{"json", "csv"}));
    // Already chosen apart: as saved.
    settings = load(R"({"export_items": ["results"], "export_formats": ["csv"]})");
    EXPECT_EQ(settings.export_items, Items{"results"});
    EXPECT_EQ(settings.export_formats, Items{"csv"});
}

// The settings read are saved as the Python app saves them, to the byte:
// every setting in its order, indented as json.dumps(indent=2) indents,
// text past ASCII escaped, and the CVVDP presets' keys in their order.
TEST(Settings, SavesSettingsJsonAsThePythonAppDoes) {
    const nlohmann::json reference = test::load_reference("settings.json");
    const test::Folder folder;
    int saved = 0;
    for (const nlohmann::json& item : reference["cases"]) {
        if (item["saved"].is_null()) {
            continue;
        }
        SCOPED_TRACE(item["name"].get<std::string>());
        std::ofstream(folder / "read.json", std::ios::binary) << item["file"].get<std::string>();
        const Settings settings = Settings::load_from(folder / "read.json");
        ASSERT_EQ(settings.save_to(folder / "settings.json"), std::nullopt);
        EXPECT_EQ(platform::read_file(folder / "settings.json"), item["saved"].get<std::string>());
        ++saved;
    }
    EXPECT_GT(saved, 0);
}

// The settings can hold Settings > Notifications' keys as plain text (on
// Linux and macOS, where they cannot be encrypted), so they are the user's
// alone, whatever the umask: the folder 0700, the file and the copy kept of
// one that could not be read 0600. On Windows (where the profile folder's
// ACL does this) neither is left read-only, which would stop the next save.
TEST(Settings, KeptFromOtherUsers) {
    const test::Folder folder;
    const std::filesystem::path profile = folder / "profile";
    const std::filesystem::path file = profile / "settings.json";
    ASSERT_EQ(Settings().save_to(file), std::nullopt);
    std::filesystem::remove(file);
    std::ofstream(file, std::ios::binary) << R"({"discord_webhook": "https://discord.com/api/webh)";
    Settings::load_from(file);
    const std::filesystem::path copy = profile / "settings.json.unreadable";
    ASSERT_TRUE(std::filesystem::exists(copy));
    ASSERT_EQ(Settings().save_to(file), std::nullopt);
    ASSERT_EQ(Settings().save_to(file), std::nullopt);  // replaces it
    using std::filesystem::perms;
    const auto permissions = [](const std::filesystem::path& path) {
        return std::filesystem::status(path).permissions() & perms::all;
    };
#ifdef _WIN32
    EXPECT_NE(permissions(file) & perms::owner_write, perms::none);
    EXPECT_NE(permissions(copy) & perms::owner_write, perms::none);
#else
    EXPECT_EQ(permissions(profile), perms::owner_all);
    EXPECT_EQ(permissions(file), perms::owner_read | perms::owner_write);
    EXPECT_EQ(permissions(copy), perms::owner_read | perms::owner_write);
#endif
}

// A settings file cut off in a key is logged by where it stops being JSON,
// never by the text there (nlohmann's "last read: '...'"), part of the key.
TEST(Settings, UnreadableFileLoggedWithoutItsText) {
    const test::Folder folder;
    std::ofstream(folder / "settings.json", std::ios::binary)
        << "{\n  \"pushover_enabled\": true,\n  \"pushover_token\": \"made-up-token-1234";
    std::vector<std::string> logged;
    app_log::forward_lines([&logged](int, std::string_view, std::string_view text) { logged.emplace_back(text); });
    Settings::load_from(folder / "settings.json");
    app_log::forward_lines({});
    ASSERT_EQ(logged.size(), 1u);
    EXPECT_EQ(logged[0].find("made-up"), std::string::npos) << logged[0];
    EXPECT_NE(logged[0].find("it isn't JSON at line 3, column "), std::string::npos) << logged[0];
}

}  // namespace
}  // namespace vml
