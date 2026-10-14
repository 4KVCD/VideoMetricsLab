#include "vml/settings.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <variant>

#include "vml/app_log.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"

namespace vml {

namespace {

using Json = nlohmann::ordered_json;

// Where a setting is kept in Settings, by its type.
using Field = std::variant<bool Settings::*, int Settings::*, std::string Settings::*,
                           std::vector<std::string> Settings::*>;

// Every setting by its key in settings.json (the Python dataclass's field
// names), except the CVVDP presets, which are read apart.
const std::unordered_map<std::string, Field>& fields() {
    static const std::unordered_map<std::string, Field> by_key{
        {"ffmpeg_dir", &Settings::ffmpeg_dir},
        {"cache_dir", &Settings::cache_dir},
        {"export_dir", &Settings::export_dir},
        {"default_gpu_decode", &Settings::default_gpu_decode},
        {"default_compute_psnr", &Settings::default_compute_psnr},
        {"default_compute_ssim", &Settings::default_compute_ssim},
        {"default_compute_xpsnr", &Settings::default_compute_xpsnr},
        {"default_compute_vmaf", &Settings::default_compute_vmaf},
        {"default_compute_vmaf_neg", &Settings::default_compute_vmaf_neg},
        {"default_compute_vmaf_v1", &Settings::default_compute_vmaf_v1},
        {"default_compute_ssimulacra2", &Settings::default_compute_ssimulacra2},
        {"default_compute_butteraugli", &Settings::default_compute_butteraugli},
        {"default_compute_cvvdp", &Settings::default_compute_cvvdp},
        {"default_ssimulacra2_backend", &Settings::default_ssimulacra2_backend},
        {"default_butteraugli_backend", &Settings::default_butteraugli_backend},
        {"default_vmaf_on_gpu", &Settings::default_vmaf_on_gpu},
        {"graph_metric", &Settings::graph_metric},
        {"parallel_jobs", &Settings::parallel_jobs},
        {"gpu_metrics_together", &Settings::gpu_metrics_together},
        {"gpu_backend", &Settings::gpu_backend},
        {"use_cache", &Settings::use_cache},
        {"check_for_updates", &Settings::check_for_updates},
        {"skipped_update_version", &Settings::skipped_update_version},
        {"language", &Settings::language},
        {"theme", &Settings::theme},
        {"remember_window_size", &Settings::remember_window_size},
        {"window_width", &Settings::window_width},
        {"window_height", &Settings::window_height},
        {"last_reference_dir", &Settings::last_reference_dir},
        {"last_test_dir", &Settings::last_test_dir},
        {"remember_videos", &Settings::remember_videos},
        {"remembered_reference", &Settings::remembered_reference},
        {"remembered_tests", &Settings::remembered_tests},
        {"highlight_best_worst", &Settings::highlight_best_worst},
        {"frame_preview_color_mode", &Settings::frame_preview_color_mode},
        {"compare_decoded_videos", &Settings::compare_decoded_videos},
        {"hidden_metrics", &Settings::hidden_metrics},
        {"cvvdp_default_preset", &Settings::cvvdp_default_preset},
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
    }
    return std::nullopt;
}

// A setting the file holds in a type of its own, in the log.
void log_unfit(const std::string& name, const Json& value, const std::string& kind) {
    app_log::write(app_log::Level::warning, "vml.settings",
                   "Settings: " + name + " is " + python_repr_of(value) + ", not a " + kind + "; using its default");
}

}  // namespace

int default_parallel_jobs(unsigned cores) { return cores > parallel_by_default_above_cores ? 2 : 1; }

std::filesystem::path Settings::file() { return platform::home_directory() / ".videometricslab" / "settings.json"; }

Settings Settings::load() {
    const std::filesystem::path path = file();
    std::error_code ignored;
    std::filesystem::create_directories(path.parent_path(), ignored);
    return load_from(path);
}

Settings Settings::load_from(const std::filesystem::path& path) {
    Settings settings;
    settings.parallel_jobs = default_parallel_jobs(platform::logical_cores());
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return settings;
    }
    std::stringstream contents;
    contents << stream.rdbuf();
    Json data;
    try {
        data = Json::parse(contents.str());
    } catch (const Json::exception&) {
        return settings;
    }
    if (!data.is_object()) {
        return settings;
    }
    for (auto item = data.begin(); item != data.end(); ++item) {
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
    data["default_gpu_decode"] = default_gpu_decode;
    data["default_compute_psnr"] = default_compute_psnr;
    data["default_compute_ssim"] = default_compute_ssim;
    data["default_compute_xpsnr"] = default_compute_xpsnr;
    data["default_compute_vmaf"] = default_compute_vmaf;
    data["default_compute_vmaf_neg"] = default_compute_vmaf_neg;
    data["default_compute_vmaf_v1"] = default_compute_vmaf_v1;
    data["default_compute_ssimulacra2"] = default_compute_ssimulacra2;
    data["default_compute_butteraugli"] = default_compute_butteraugli;
    data["default_compute_cvvdp"] = default_compute_cvvdp;
    data["default_ssimulacra2_backend"] = default_ssimulacra2_backend;
    data["default_butteraugli_backend"] = default_butteraugli_backend;
    data["default_vmaf_on_gpu"] = default_vmaf_on_gpu;
    data["graph_metric"] = graph_metric;
    data["parallel_jobs"] = parallel_jobs;
    data["gpu_metrics_together"] = gpu_metrics_together;
    data["gpu_backend"] = gpu_backend;
    data["use_cache"] = use_cache;
    data["check_for_updates"] = check_for_updates;
    data["skipped_update_version"] = skipped_update_version;
    data["language"] = language;
    data["theme"] = theme;
    data["remember_window_size"] = remember_window_size;
    data["window_width"] = window_width;
    data["window_height"] = window_height;
    data["last_reference_dir"] = last_reference_dir;
    data["last_test_dir"] = last_test_dir;
    data["remember_videos"] = remember_videos;
    data["remembered_reference"] = remembered_reference;
    data["remembered_tests"] = remembered_tests;
    data["highlight_best_worst"] = highlight_best_worst;
    data["frame_preview_color_mode"] = frame_preview_color_mode;
    data["compare_decoded_videos"] = compare_decoded_videos;
    data["hidden_metrics"] = hidden_metrics;
    Json presets = Json::parse(cvvdp_presets_json, nullptr, false);
    data["cvvdp_presets"] = presets.is_array() ? presets : Json::array();
    data["cvvdp_default_preset"] = cvvdp_default_preset;
    return python_json_dumps(data, 2);
}

std::optional<std::string> Settings::save() const { return save_to(file()); }

std::optional<std::string> Settings::save_to(const std::filesystem::path& path) const {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    std::random_device random;
    const std::filesystem::path temporary =
        path.parent_path() /
        platform::path_from_text(platform::path_text(path.filename()) + "." + std::to_string(random()) + ".tmp");
    {
        std::ofstream stream(temporary, std::ios::binary);
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
