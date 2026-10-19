// Turning a comparison's VMAF model choice into the FFmpeg model= value it
// means, choosing the 4K model for "Auto" at 4K; and the VMAF model files
// the app ships. A port of vmaf_app/core/model_select.py and
// builtin_models.py.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "vml/options.h"

namespace vml {

inline constexpr std::string_view default_model = "version=vmaf_v0.6.1";
inline constexpr std::string_view uhd_model = "version=vmaf_4k_v0.6.1";

// A packaged build's resources folder, when the program runs from one: the
// "resources" folder beside its programs (scripts/build_release.ps1), or on
// macOS the app bundle's Contents/Resources/resources, the programs being in
// Contents/MacOS (scripts/build_release_macos.sh).
std::optional<std::filesystem::path> packaged_resource_folder();

// The folder the app's own files are in (the VMAF models, the bundled tools):
// the folder the VML_RESOURCES environment variable names; else a packaged
// build's (packaged_resource_folder); else the one this build was made to
// use (the repository's resources folder, in a development build).
std::filesystem::path resource_folder();

// Whether a model choice names a bundled model ("__builtin:<id>").
bool is_builtin_model_choice(std::string_view choice);
// The bundled model's file. Throws std::invalid_argument for an unknown id,
// and std::runtime_error when the file is missing (an installation error,
// not a reason to fall back to another model).
std::filesystem::path builtin_model_path(std::string_view model_id);
// The model choice that names a bundled model.
std::string builtin_choice(std::string_view model_id);

// The model "Auto" picks for frames compared at this size.
std::string model_for_resolution(int width, int height);

// The model= value that names a model file: "path=<file>". (One that names
// a version of a model libvmaf has built in is "version=<name>".)
std::string model_option_for_file(std::string_view file);
// The file a model= value names, or nothing for a value that names none.
std::optional<std::string_view> model_file_of(std::string_view model);

// The FFmpeg model= value for the options' model choice, at the size the
// two videos are compared at. Throws std::invalid_argument for a custom
// model with no file.
std::string resolve_model(const VmafOptions& options, int width, int height);

// The bundled VMAF v1 model "Auto" picks for frames compared at this size,
// and the model= value of the options' VMAF v1 choice ("path=<file>").
std::string v1_model_for_resolution(int width, int height);
std::string resolve_v1_model(const VmafOptions& options, int width, int height);

}  // namespace vml
