#include "vml/settings.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>

#include "reference.h"
#include "test_videos.h"
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
        {"parallel_jobs", [](const Settings& s) { return Json(s.parallel_jobs); }},
        {"gpu_metrics_together", [](const Settings& s) { return Json(s.gpu_metrics_together); }},
        {"use_cache", [](const Settings& s) { return Json(s.use_cache); }},
        {"check_for_updates", [](const Settings& s) { return Json(s.check_for_updates); }},
        {"skipped_update_version", [](const Settings& s) { return Json(s.skipped_update_version); }},
        {"language", [](const Settings& s) { return Json(s.language); }},
        {"theme", [](const Settings& s) { return Json(s.theme); }},
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

}  // namespace
}  // namespace vml
