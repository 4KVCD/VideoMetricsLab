#include "vml/model_select.h"

#include <array>
#include <stdexcept>
#include <utility>

#include "vml/platform/system.h"
#include "vml/translatable.h"

namespace vml {

namespace {

// A comparison at or above this size counts as 4K.
constexpr int uhd_width_threshold = 3840;
constexpr int uhd_height_threshold = 2160;

// A bundled model: its stable id (saved in options and results; a path is
// never saved, since it differs between installs), the metric it belongs to
// ("vmaf_v1" or "vmaf_hdr"), its file, and the name the window shows it by.
struct BuiltinModel {
    std::string_view id;
    std::string_view metric;
    std::string_view file;
    const char* name;
};

// The bundled models, in the order of the window's VMAF v1 and HDR VMAF
// dropdowns.
constexpr std::array<BuiltinModel, 16> builtin_models{{
    {"vmaf_v1_3d0h", "vmaf_v1", "vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json", N_("VMAF v1 (1080p / 3H)")},
    {"vmaf_v1_5d0h", "vmaf_v1", "vmaf_v1.0.16/vmaf_v1.0.16_5d0h.json", N_("VMAF v1 (1080p phone / 5H)")},
    {"vmaf_v1_1d5h_2160", "vmaf_v1", "vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json", N_("VMAF v1 (4K / 1.5H)")},
    {"vmaf_v1_3d0h_2160", "vmaf_v1", "vmaf_v1.0.16/vmaf_v1.0.16_3d0h_2160.json", N_("VMAF v1 (4K / 3H, up to 110)")},
    {"vmaf_v1_hfr_3d0h", "vmaf_v1", "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h.json", N_("VMAF v1 HFR (1080p / 3H)")},
    {"vmaf_v1_hfr_5d0h", "vmaf_v1", "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_5d0h.json",
     N_("VMAF v1 HFR (1080p phone / 5H)")},
    {"vmaf_v1_hfr_1d5h_2160", "vmaf_v1", "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json",
     N_("VMAF v1 HFR (4K / 1.5H)")},
    {"vmaf_v1_hfr_3d0h_2160", "vmaf_v1", "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h_2160.json",
     N_("VMAF v1 HFR (4K / 3H, up to 110)")},
    {"hdr_vmaf_3d0h", "vmaf_hdr", "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json", N_("HDR VMAF (1080p / 3H)")},
    {"hdr_vmaf_5d0h", "vmaf_hdr", "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json", N_("HDR VMAF (1080p phone / 5H)")},
    {"hdr_vmaf_1d5h_2160", "vmaf_hdr", "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json", N_("HDR VMAF (4K / 1.5H)")},
    {"hdr_vmaf_3d0h_2160", "vmaf_hdr", "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json",
     N_("HDR VMAF (4K / 3H, up to 110)")},
    {"hdr_vmaf_hfr_3d0h", "vmaf_hdr", "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json",
     N_("HDR VMAF HFR (1080p / 3H)")},
    {"hdr_vmaf_hfr_5d0h", "vmaf_hdr", "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json",
     N_("HDR VMAF HFR (1080p phone / 5H)")},
    {"hdr_vmaf_hfr_1d5h_2160", "vmaf_hdr", "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json",
     N_("HDR VMAF HFR (4K / 1.5H)")},
    {"hdr_vmaf_hfr_3d0h_2160", "vmaf_hdr", "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json",
     N_("HDR VMAF HFR (4K / 3H, up to 110)")},
}};

constexpr std::string_view builtin_prefix = "__builtin:";

const BuiltinModel* builtin_model(std::string_view model_id) {
    for (const BuiltinModel& model : builtin_models) {
        if (model.id == model_id) {
            return &model;
        }
    }
    return nullptr;
}

const std::string_view* model_file(std::string_view model_id) {
    const BuiltinModel* model = builtin_model(model_id);
    return model != nullptr ? &model->file : nullptr;
}

bool is_uhd(int width, int height) { return width >= uhd_width_threshold || height >= uhd_height_threshold; }

// What a model= value that names a file starts with.
constexpr std::string_view model_file_prefix = "path=";

}  // namespace

std::optional<std::filesystem::path> packaged_resource_folder() {
#ifdef __APPLE__
    static const std::filesystem::path packaged =
        platform::executable_path().parent_path().parent_path() / "Resources" / "resources";
#else
    static const std::filesystem::path packaged = platform::executable_path().parent_path() / "resources";
#endif
    std::error_code error;
    if (std::filesystem::is_directory(packaged / "models", error)) {
        return packaged;
    }
    return std::nullopt;
}

std::filesystem::path resource_folder() {
    // An empty one names no folder (a script that "removed" it with
    // PowerShell 7's SetEnvironmentVariable($name, $null) left it so).
    if (const std::optional<std::string> named = platform::environment_variable("VML_RESOURCES");
        named && !named->empty()) {
        return platform::path_from_text(*named).make_preferred();
    }
    // A packaged build's; else the repository's, for a build run from its
    // folder.
    if (const std::optional<std::filesystem::path> packaged = packaged_resource_folder()) {
        return *packaged;
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

namespace {

// Auto, then each bundled model of `metric` (its word its id).
std::vector<ModelChoice> choices_of(std::string_view metric) {
    std::vector<ModelChoice> made{
        {N_("Auto (analysis resolution)"), std::string(auto_model_choice), std::string(auto_word)}};
    for (const BuiltinModel& model : builtin_models) {
        if (model.metric == metric) {
            made.push_back({model.name, builtin_choice(model.id), std::string(model.id)});
        }
    }
    return made;
}

// The model= value ("path=<file>") of `choice`, a bundled model of `metric`,
// or of `automatic` when the choice is Auto. Throws std::invalid_argument
// for a choice that is not one of that metric's models ("Not a VMAF v1
// model: ...", `kind` naming them).
std::string resolve_builtin(const std::string& choice, std::string_view metric, std::string_view kind,
                            const std::string& automatic) {
    const std::string& chosen = choice == auto_model_choice ? automatic : choice;
    const BuiltinModel* model =
        is_builtin_model_choice(chosen) ? builtin_model(std::string_view(chosen).substr(builtin_prefix.size())) : nullptr;
    if (model == nullptr || model->metric != metric) {
        throw std::invalid_argument("Not " + std::string(kind) + " model: " + chosen);
    }
    return model_option_for_file(platform::path_text(builtin_model_path(model->id)));
}

}  // namespace

const std::vector<ModelChoice>& v1_model_choices() {
    static const std::vector<ModelChoice> choices = choices_of("vmaf_v1");
    return choices;
}

const std::vector<ModelChoice>& hdr_model_choices() {
    static const std::vector<ModelChoice> choices = choices_of("vmaf_hdr");
    return choices;
}

std::string model_for_resolution(int width, int height) {
    return std::string(is_uhd(width, height) ? uhd_model : default_model);
}

std::string model_option_for_file(std::string_view file) { return std::string(model_file_prefix) + std::string(file); }

std::optional<std::string_view> model_file_of(std::string_view model) {
    if (!model.starts_with(model_file_prefix)) {
        return std::nullopt;
    }
    return model.substr(model_file_prefix.size());
}

std::string model_shown(std::string_view model) {
    const std::optional<std::string_view> file = model_file_of(model);
    if (!file) {
        return std::string(model);
    }
    // Either separator: a path saved on Windows has backslashes, and one
    // read on another system is still that path.
    const std::string_view name = file->substr(file->find_last_of("/\\") + 1);
    for (const BuiltinModel& builtin : builtin_models) {
        const std::string_view bundled = builtin.file.substr(builtin.file.find_last_of('/') + 1);
        if (name == bundled && name.ends_with(".json")) {
            return "version=" + std::string(name.substr(0, name.size() - std::string_view(".json").size()));
        }
    }
    return std::string(model);
}

std::string resolve_model(const VmafOptions& options, int width, int height) {
    if (options.model_choice == custom_model_choice) {
        if (options.custom_model_path.empty()) {
            throw std::invalid_argument("No custom model file selected.");
        }
        return model_option_for_file(options.custom_model_path);
    }
    if (is_builtin_model_choice(options.model_choice)) {
        const std::string_view id = std::string_view(options.model_choice).substr(builtin_prefix.size());
        return model_option_for_file(platform::path_text(builtin_model_path(id)));
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
    return resolve_builtin(options.model_choice_v1, "vmaf_v1", "a VMAF v1", v1_model_for_resolution(width, height));
}

std::string hdr_model_for_resolution(int width, int height) {
    // As VMAF v1's Auto: at 4K the 4K model at one and a half screen
    // heights, below it the 1080p model at three.
    return builtin_choice(is_uhd(width, height) ? "hdr_vmaf_1d5h_2160" : "hdr_vmaf_3d0h");
}

std::string resolve_hdr_model(const VmafOptions& options, int width, int height) {
    return resolve_builtin(options.model_choice_hdr, "vmaf_hdr", "an HDR VMAF", hdr_model_for_resolution(width, height));
}

}  // namespace vml
