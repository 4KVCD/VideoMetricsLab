#include "vml/model_select.h"

#include <array>
#include <stdexcept>
#include <utility>

#include "vml/platform/system.h"

namespace vml {

namespace {

// A comparison at or above this size counts as 4K.
constexpr int uhd_width_threshold = 3840;
constexpr int uhd_height_threshold = 2160;

// The bundled models by their stable ids (saved in options and results; a
// path is never saved, since it differs between installs).
constexpr std::array<std::pair<std::string_view, std::string_view>, 8> builtin_model_files{{
    {"vmaf_v1_3d0h", "vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json"},
    {"vmaf_v1_5d0h", "vmaf_v1.0.16/vmaf_v1.0.16_5d0h.json"},
    {"vmaf_v1_1d5h_2160", "vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json"},
    {"vmaf_v1_3d0h_2160", "vmaf_v1.0.16/vmaf_v1.0.16_3d0h_2160.json"},
    {"vmaf_v1_hfr_3d0h", "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h.json"},
    {"vmaf_v1_hfr_5d0h", "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_5d0h.json"},
    {"vmaf_v1_hfr_1d5h_2160", "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json"},
    {"vmaf_v1_hfr_3d0h_2160", "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h_2160.json"},
}};

constexpr std::string_view builtin_prefix = "__builtin:";

const std::string_view* model_file(std::string_view model_id) {
    for (const auto& [id, file] : builtin_model_files) {
        if (id == model_id) {
            return &file;
        }
    }
    return nullptr;
}

bool is_uhd(int width, int height) { return width >= uhd_width_threshold || height >= uhd_height_threshold; }

}  // namespace

std::filesystem::path resource_folder() {
    if (const std::optional<std::string> named = platform::environment_variable("VML_RESOURCES")) {
        return platform::path_from_text(*named).make_preferred();
    }
    return platform::path_from_text(VML_DEFAULT_RESOURCES).make_preferred();
}

bool is_builtin_model_choice(std::string_view choice) {
    return choice.starts_with(builtin_prefix) && model_file(choice.substr(builtin_prefix.size())) != nullptr;
}

std::filesystem::path builtin_model_path(std::string_view model_id) {
    const std::string_view* file = model_file(model_id);
    if (file == nullptr) {
        throw std::invalid_argument("Unknown built-in VMAF model: " + std::string(model_id));
    }
    // Separators as the system writes them (backslashes on Windows), as the
    // Python app's resolved path has them.
    const std::filesystem::path path = (resource_folder() / "models" / platform::path_from_text(*file)).make_preferred();
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        throw std::runtime_error("Built-in VMAF model is missing: " + platform::path_text(path));
    }
    return path;
}

std::string builtin_choice(std::string_view model_id) {
    if (model_file(model_id) == nullptr) {
        throw std::invalid_argument("Unknown built-in VMAF model: " + std::string(model_id));
    }
    return std::string(builtin_prefix) + std::string(model_id);
}

std::string model_for_resolution(int width, int height) {
    return std::string(is_uhd(width, height) ? uhd_model : default_model);
}

std::string resolve_model(const VmafOptions& options, int width, int height) {
    if (options.model_choice == custom_model_choice) {
        if (options.custom_model_path.empty()) {
            throw std::invalid_argument("No custom model file selected.");
        }
        return "path=" + options.custom_model_path;
    }
    if (is_builtin_model_choice(options.model_choice)) {
        const std::string_view id = std::string_view(options.model_choice).substr(builtin_prefix.size());
        return "path=" + platform::path_text(builtin_model_path(id));
    }
    if (options.model_choice == auto_model_choice) {
        return model_for_resolution(width, height);
    }
    return options.model_choice;
}

std::string v1_model_for_resolution(int width, int height) {
    // Below 4K, the 1080p model at three screen heights; at 4K, the 4K model
    // at one and a half: the conditions v0.6.1's two models were made for.
    return builtin_choice(is_uhd(width, height) ? "vmaf_v1_1d5h_2160" : "vmaf_v1_3d0h");
}

std::string resolve_v1_model(const VmafOptions& options, int width, int height) {
    std::string choice = options.model_choice_v1;
    if (choice == auto_model_choice) {
        choice = v1_model_for_resolution(width, height);
    }
    if (!is_builtin_model_choice(choice)) {
        throw std::invalid_argument("Not a VMAF v1 model: " + choice);
    }
    return "path=" + platform::path_text(builtin_model_path(std::string_view(choice).substr(builtin_prefix.size())));
}

}  // namespace vml
