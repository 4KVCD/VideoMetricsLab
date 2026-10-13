// The app's own libvmaf: libvmaf-fast (github.com/4KVCD/libvmaf-fast), a
// fork of libvmaf shipped in vmaf_app/tools/libvmaf. The app calls it in its
// own process: to have libvmaf predict VMAF from the features the Vulkan
// engine measures, for VMAF and NEG with CUDA, and for PSNR, SSIM and XPSNR.
// A port of vmaf_app/core/libvmaf.py.
//
// Its functions are declared by its own public headers, at the commit the
// bundled library was built from (third_party/libvmaf-fast), so that the
// structures the app shares with it are laid out as it lays them out.
#pragma once

#include <libvmaf/feature.h>
#include <libvmaf/libvmaf.h>

#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vml/platform/library.h"

namespace vml {

// A call into libvmaf failed. The caller then calculates the scores another
// way, with FFmpeg.
class LibvmafError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Throws LibvmafError if a libvmaf call returned an error, saying what failed.
void check_libvmaf(int error, const std::string& what);

// The bundled library and the functions of it the app calls.
struct Libvmaf {
    explicit Libvmaf(const std::filesystem::path& path);

    platform::DynamicLibrary library;
    decltype(&::vmaf_version) version;
    decltype(&::vmaf_init) init;
    decltype(&::vmaf_close) close;
    decltype(&::vmaf_model_load) model_load;
    decltype(&::vmaf_model_destroy) model_destroy;
    decltype(&::vmaf_import_feature_score) import_feature_score;
    decltype(&::vmaf_score_at_index) score_at_index;
    // Its own feature extractors, which the app runs for PSNR, SSIM and
    // XPSNR, on pictures from a pool libvmaf allocates.
    decltype(&::vmaf_use_feature) use_feature;
    decltype(&::vmaf_feature_dictionary_set) feature_dictionary_set;
    decltype(&::vmaf_feature_dictionary_free) feature_dictionary_free;
    decltype(&::vmaf_preallocate_pictures) preallocate_pictures;
    decltype(&::vmaf_fetch_preallocated_picture) fetch_preallocated_picture;
    decltype(&::vmaf_picture_unref) picture_unref;
    decltype(&::vmaf_read_pictures) read_pictures;
    decltype(&::vmaf_feature_score_at_index) feature_score_at_index;
};

// Where the bundled library is, with its build record.
std::filesystem::path libvmaf_folder();

// The bundled library, loaded the first time it is asked for. Throws
// LibvmafError when it cannot be loaded, or when it is not the libvmaf-fast
// whose headers the app was built with (its pictures would be laid out
// otherwise, and libvmaf would write past the app's).
const Libvmaf& libvmaf();

// The libvmaf-fast release the bundled library is, as the script that
// installed it recorded ("v1"; "v1-2-g211f80e0" for a local build two
// commits after it), or "unknown" without a record.
std::string libvmaf_fast_version();

// A feature's values for every frame, under the name libvmaf's own
// extractor gives it. The values stay where they are, not copied.
using FeatureValues = std::pair<std::string, std::span<const double>>;

// A libvmaf model's scores for frames, predicted by libvmaf itself from
// `features`, each with a value for every frame. `version` names a model
// libvmaf has built in, such as "vmaf_v0.6.1". Each score is kept to 6
// decimals, as the Python app keeps it. Throws LibvmafError.
std::vector<double> predict_scores(const std::string& version, const std::vector<unsigned>& frames,
                                   const std::vector<FeatureValues>& features);

}  // namespace vml
