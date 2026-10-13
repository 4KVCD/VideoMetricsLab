#include "vml/libvmaf.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <functional>
#include <optional>

#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// libvmaf's handles, released however a run ends.
struct ContextAndModel {
    const Libvmaf& library;
    VmafContext* context = nullptr;
    VmafModel* model = nullptr;

    ~ContextAndModel() {
        if (model != nullptr) {
            library.model_destroy(model);
        }
        if (context != nullptr) {
            library.close(context);
        }
    }
};

// A model's scores for frames, predicted by libvmaf from `features`, the
// model loaded by `load` (which gets the model's handle to set and its
// configuration, and returns libvmaf's answer). `model_name` is for
// messages.
std::vector<double> predict(const std::function<int(VmafModel**, VmafModelConfig*)>& load,
                            const std::string& model_name, const std::vector<unsigned>& frames,
                            const std::vector<FeatureValues>& features) {
    for (const auto& [name, values] : features) {
        if (values.size() != frames.size()) {
            throw std::invalid_argument(name + " has " + std::to_string(values.size()) + " values for " +
                                        std::to_string(frames.size()) + " frames");
        }
    }
    const Libvmaf& library = libvmaf();
    ContextAndModel handles{library};
    const VmafConfiguration configuration{VMAF_LOG_LEVEL_ERROR, 0, 1, 0, 0};
    // When vmaf_init fails it frees the context it made but leaves the
    // pointer to it set, so the context is only kept once it has started.
    VmafContext* context = nullptr;
    check_libvmaf(library.init(&context, configuration), "Starting libvmaf");
    handles.context = context;
    // Loading a model can fail after making it, and leaves it set; it is
    // then destroyed like a loaded one.
    VmafModelConfig model_config{"vmaf", 0};
    check_libvmaf(load(&handles.model, &model_config), "Loading the " + model_name + " model");
    // These calls are made for every frame, so what failed is only put into
    // words when one does.
    const auto check_frame = [](int error, const char* what, unsigned frame) {
        if (error != 0) {
            check_libvmaf(error, what + std::to_string(frame));
        }
    };
    for (const auto& [name, values] : features) {
        for (std::size_t i = 0; i < frames.size(); ++i) {
            check_frame(library.import_feature_score(handles.context, name.c_str(), values[i], frames[i]),
                        "Giving libvmaf frame ", frames[i]);
        }
    }
    std::vector<double> scores;
    scores.reserve(frames.size());
    for (const unsigned frame : frames) {
        double score = 0.0;
        check_frame(library.score_at_index(handles.context, handles.model, &score, frame), "Predicting frame ",
                    frame);
        scores.push_back(to_6_decimals(score));
    }
    return scores;
}

}  // namespace

void check_libvmaf(int error, const std::string& what) {
    if (error != 0) {
        throw LibvmafError(what + " failed (libvmaf error " + std::to_string(error) + ")");
    }
}

double to_6_decimals(double value) { return python_float(format_fixed(value, 6)).value_or(value); }

Libvmaf::Libvmaf(const std::filesystem::path& path)
    : library(path),
      version(library.function<decltype(version)>("vmaf_version")),
      init(library.function<decltype(init)>("vmaf_init")),
      close(library.function<decltype(close)>("vmaf_close")),
      model_load(library.function<decltype(model_load)>("vmaf_model_load")),
      model_destroy(library.function<decltype(model_destroy)>("vmaf_model_destroy")),
      import_feature_score(library.function<decltype(import_feature_score)>("vmaf_import_feature_score")),
      score_at_index(library.function<decltype(score_at_index)>("vmaf_score_at_index")),
      use_feature(library.function<decltype(use_feature)>("vmaf_use_feature")),
      feature_dictionary_set(library.function<decltype(feature_dictionary_set)>("vmaf_feature_dictionary_set")),
      feature_dictionary_free(library.function<decltype(feature_dictionary_free)>("vmaf_feature_dictionary_free")),
      preallocate_pictures(library.function<decltype(preallocate_pictures)>("vmaf_preallocate_pictures")),
      fetch_preallocated_picture(
          library.function<decltype(fetch_preallocated_picture)>("vmaf_fetch_preallocated_picture")),
      picture_unref(library.function<decltype(picture_unref)>("vmaf_picture_unref")),
      read_pictures(library.function<decltype(read_pictures)>("vmaf_read_pictures")),
      feature_score_at_index(library.function<decltype(feature_score_at_index)>("vmaf_feature_score_at_index")),
      model_load_from_path(library.function<decltype(model_load_from_path)>("vmaf_model_load_from_path")),
      use_features_from_model(library.function<decltype(use_features_from_model)>("vmaf_use_features_from_model")),
      write_output(library.function<decltype(write_output)>("vmaf_write_output")),
      picture_alloc(library.function<decltype(picture_alloc)>("vmaf_picture_alloc")) {
    // VmafPicture gained its colour field with vmaf_picture_convert (Netflix/vmaf
    // 0497a0f2). A library without that function lays its pictures out
    // without the field, as the app's headers do not.
    if (library.find("vmaf_picture_convert") == nullptr) {
        throw LibvmafError(platform::path_text(path) +
                           " is older than the libvmaf-fast the app was built with (it has no vmaf_picture_convert)");
    }
    cuda_state_init = reinterpret_cast<decltype(cuda_state_init)>(library.find("vmaf_cuda_state_init"));
    cuda_import_state = reinterpret_cast<decltype(cuda_import_state)>(library.find("vmaf_cuda_import_state"));
    cuda_preallocate_pictures =
        reinterpret_cast<decltype(cuda_preallocate_pictures)>(library.find("vmaf_cuda_preallocate_pictures"));
    cuda_fetch_preallocated_picture = reinterpret_cast<decltype(cuda_fetch_preallocated_picture)>(
        library.find("vmaf_cuda_fetch_preallocated_picture"));
}

std::filesystem::path libvmaf_folder() { return resource_folder() / "tools" / "libvmaf"; }

const Libvmaf& libvmaf() {
    try {
        static const Libvmaf loaded(libvmaf_folder() / platform::shared_library_name("libvmaf"));
        return loaded;
    } catch (const platform::LibraryError& error) {
        throw LibvmafError(error.what());
    }
}

std::string libvmaf_fast_version() {
    static const std::string version = [] {
        try {
            std::ifstream record(libvmaf_folder() / "libvmaf-fast.json", std::ios::binary);
            const nlohmann::json data = nlohmann::json::parse(record);
            const nlohmann::json& named = data.at("version");
            return named.is_string() ? named.get<std::string>() : named.dump();
        } catch (const std::exception&) {
            return std::string("unknown");
        }
    }();
    return version;
}

std::vector<double> predict_scores(const std::string& version, const std::vector<unsigned>& frames,
                                   const std::vector<FeatureValues>& features) {
    const Libvmaf& library = libvmaf();
    return predict(
        [&](VmafModel** model, VmafModelConfig* config) { return library.model_load(model, config, version.c_str()); },
        version, frames, features);
}

std::vector<double> predict_model_file_scores(const std::filesystem::path& model, const std::vector<unsigned>& frames,
                                              const std::vector<FeatureValues>& features) {
    const Libvmaf& library = libvmaf();
    const std::string path = libvmaf_path(model);
    return predict(
        [&](VmafModel** loaded, VmafModelConfig* config) {
            return library.model_load_from_path(loaded, config, path.c_str());
        },
        platform::path_text(model.filename()), frames, features);
}

std::string libvmaf_path(const std::filesystem::path& file) {
    if (std::optional<std::string> path = platform::fopen_path(file)) {
        return *path;
    }
    throw LibvmafError("libvmaf cannot open " + platform::path_text(file) +
                       ": its name has letters outside this PC's code page");
}

}  // namespace vml
