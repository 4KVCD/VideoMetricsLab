#include "vml/vmaf_v1_gpu.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <set>

#include "vml/app_log.h"
#include "vml/gpu_frames.h"
#include "vml/isolated.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/temp_files.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.vmaf_v1_gpu");

// A feature's options, as its model gives them, in the model's order.
using Options = nlohmann::ordered_json;
using ModelOptions = std::map<std::string, Options>;

// The features libvmaf's CPU extractors calculate when the GPU does not:
// each with its extractor, and the start of the name libvmaf gives its
// score.
struct CpuExtractor {
    const char* feature;
    const char* extractor;
    const char* prefix;
};
constexpr std::array<CpuExtractor, 2> cpu_extractors{{{v1_cambi, "cambi", "cambi"},
                                                      {v1_speed, "speed_chroma", "speed_chroma_uv"}}};
// The features the GPU always calculates, with the start of their names.
constexpr std::array<std::pair<const char*, const char*>, 2> gpu_features{
    {{v1_adm3, "integer_adm3"}, {v1_motion3, "integer_motion3"}}};

// What vv_v1_cambi and vv_v1_speed answer for options they do not
// calculate: libvmaf's CPU code calculates that feature.
constexpr int unsupported_options = -4;

// The flag of a context that reads the decoders' pictures where they left
// them (vv_pictures).
constexpr int pictures_flag = 1 << 2;

// The models the probe holds the GPU to libvmaf's CPU values with, and the
// bits of the frames it gives each.
constexpr std::array<std::pair<const char*, int>, 2> probe_models{{
    {"vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json", 8},
    {"vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json", 10},
}};

// libvmaf's handles, released however the work ends.
struct LibvmafHandles {
    const Libvmaf& library;
    VmafContext* context = nullptr;
    VmafModel* model = nullptr;

    explicit LibvmafHandles(const Libvmaf& from) : library(from) {}
    ~LibvmafHandles() {
        if (model != nullptr) {
            library.model_destroy(model);
        }
        if (context != nullptr) {
            library.close(context);
        }
    }
    LibvmafHandles(const LibvmafHandles&) = delete;
    LibvmafHandles& operator=(const LibvmafHandles&) = delete;

    // Starts libvmaf with `threads` threads (0: on the calling thread) and
    // every `n_subsample`th frame.
    void start(int threads, int n_subsample) {
        const VmafConfiguration configuration{VMAF_LOG_LEVEL_ERROR, static_cast<unsigned>(std::max(0, threads)),
                                              static_cast<unsigned>(std::max(1, n_subsample)), 0, 0};
        // When vmaf_init fails it frees the context it made but leaves the
        // pointer to it set, so the context is only kept once it has started.
        VmafContext* made = nullptr;
        check_libvmaf(library.init(&made, configuration), "Starting libvmaf");
        context = made;
    }

    // Loads the model in `file` (and destroys one that fails half-made).
    void load_model(const std::filesystem::path& file) {
        VmafModelConfig config{"vmaf", 0};
        const std::string path = libvmaf_path(file);
        check_libvmaf(library.model_load_from_path(&model, &config, path.c_str()), "Loading the model");
    }
};

// An option's value as libvmaf's option parser reads it (the Python app's
// _text): Python's str() of it, with true and false as libvmaf takes them.
std::string option_text(const Options& value) {
    if (value.is_boolean()) {
        return value.get<bool>() ? "true" : "false";
    }
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_number_float()) {
        return python_repr(value.get<double>());
    }
    return value.dump();
}

// A number among the options, as Python's float() reads it.
double option_number(const Options& value) {
    if (value.is_boolean()) {
        return value.get<bool>() ? 1.0 : 0.0;
    }
    if (value.is_number()) {
        return value.get<double>();
    }
    if (value.is_string()) {
        if (const std::optional<double> number = python_float(value.get<std::string>())) {
            return *number;
        }
    }
    throw VmafV1Error("the model's option " + value.dump() + " is not a number");
}

// An option as Python's bool() takes it.
bool option_truthy(const Options& value) {
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number()) {
        return value.get<double>() != 0.0;
    }
    if (value.is_string() || value.is_array() || value.is_object()) {
        return !value.empty();
    }
    return false;
}

// An option's value, or `fallback` when the model does not set it.
Options option_or(const Options& options, const char* key, const Options& fallback) {
    return options.contains(key) ? options.at(key) : fallback;
}

// A model's four features and their options. Throws VmafV1Error unless they
// are VMAF v1's.
ModelOptions model_options(const std::filesystem::path& model) {
    const std::string name = platform::path_text(model.filename());
    ModelOptions options;
    try {
        std::ifstream file(model, std::ios::binary);
        if (!file) {
            throw std::runtime_error("it cannot be read");
        }
        const nlohmann::ordered_json data = nlohmann::ordered_json::parse(file);
        const nlohmann::ordered_json& names = data.at("model_dict").at("feature_names");
        const nlohmann::ordered_json& settings = data.at("model_dict").at("feature_opts_dicts");
        if (names.size() != settings.size()) {
            throw std::runtime_error("it has " + std::to_string(names.size()) + " features and " +
                                     std::to_string(settings.size()) + " sets of options");
        }
        for (std::size_t i = 0; i < names.size(); ++i) {
            options[names[i].get<std::string>()] = settings[i];
        }
    } catch (const std::exception& error) {
        throw VmafV1Error(name + " is not a VMAF model file: " + error.what());
    }
    const std::set<std::string> wanted{v1_adm3, v1_motion3, v1_cambi, v1_speed};
    std::set<std::string> found;
    for (const auto& [feature, settings] : options) {
        found.insert(feature);
    }
    if (found != wanted) {
        throw VmafV1Error(name + " is not a VMAF v1 model");
    }
    return options;
}

// Whether SpEED has no block to score at this size, with SpEED's options:
// SpEED works on the chroma planes, scaled by the model's speed_prescale and
// then down by 16, in blocks of 5. On a video too small for one block,
// libvmaf reports the problem and then reads past its buffers, on the CPU
// as well as here, so such a video is refused.
bool speed_too_small(const Options& speed, int width, int height) {
    const double prescale = option_number(option_or(speed, "speed_prescale", 1.0));
    const auto blocks = [prescale](int size) {
        return static_cast<long long>(static_cast<double>(size / 2) * prescale + 0.5) >> 4;
    };
    return std::min(blocks(width), blocks(height)) < 5;
}

bool environment_set(const char* name) { return !platform::environment_variable(name).value_or("").empty(); }

// vv_v1_cambi's nine numbers for the model's CAMBI options, or nothing when
// the model sets an option the GPU does not support (full_ref, heatmaps,
// the enc_* and src_* sizes, another max_log_contrast or EOTF): libvmaf
// then calculates CAMBI.
std::optional<std::array<double, 9>> cambi_gpu_options(const Options& options) {
    // cambi.c's defaults.
    static const Options defaults{{"cambi_high_res_speedup", 0},  {"cambi_vis_lum_threshold", 0.0},
                                  {"cambi_max_val", 1000.0},      {"topk", 0.6},
                                  {"cambi_topk", 0.6},            {"window_size", 65},
                                  {"tvi_threshold", 0.019},       {"max_log_contrast", 2},
                                  {"eotf", "bt1886"},             {"cambi_eotf", "bt1886"}};
    Options o = defaults;
    for (const auto& [key, value] : options.items()) {
        if (!defaults.contains(key)) {
            return std::nullopt;
        }
        o[key] = value;
    }
    // As cambi.c chooses: topk if it is not the default, else cambi_topk;
    // and cambi_eotf if it is not the default, else eotf.
    const double topk =
        option_number(o["topk"]) != 0.6 ? option_number(o["topk"]) : option_number(o["cambi_topk"]);
    const std::string eotf =
        option_text(o["cambi_eotf"]) != "bt1886" ? option_text(o["cambi_eotf"]) : option_text(o["eotf"]);
    if ((eotf != "bt1886" && eotf != "pq") || static_cast<long long>(option_number(o["max_log_contrast"])) != 2) {
        return std::nullopt;
    }
    const double test_flags = (environment_set("VMAF_FAST_TEST_CAMBI_POOL_ON_CPU") ? 1.0 : 0.0) +
                              (environment_set("VMAF_FAST_TEST_CAMBI_BRUTE_FORCE") ? 2.0 : 0.0);
    return std::array<double, 9>{option_number(o["cambi_high_res_speedup"]),
                                 option_number(o["cambi_vis_lum_threshold"]),
                                 option_number(o["cambi_max_val"]),
                                 topk,
                                 option_number(o["window_size"]),
                                 option_number(o["tvi_threshold"]),
                                 option_number(o["max_log_contrast"]),
                                 eotf == "bt1886" ? 0.0 : 1.0,
                                 test_flags};
}

// vv_v1_speed's seven numbers for the model's SpEED options, or nothing when
// the model sets an option the GPU does not support: libvmaf then
// calculates SpEED.
std::optional<std::array<double, 7>> speed_gpu_options(const Options& options) {
    // speed.c's defaults, and the short names libvmaf also accepts.
    static const Options defaults{{"speed_kernelscale", 1.0}, {"speed_prescale", 1.0},
                                  {"speed_prescale_method", "nearest"}, {"speed_sigma_nn", 0.29},
                                  {"speed_nn_floor", 0.0}, {"speed_weight_var_mode", 0},
                                  {"speed_max_val", 1000.0}};
    static const std::map<std::string, std::string, std::less<>> aliases{
        {"ks", "speed_kernelscale"}, {"ps", "speed_prescale"}, {"psm", "speed_prescale_method"},
        {"snn", "speed_sigma_nn"},   {"nnf", "speed_nn_floor"}, {"wvm", "speed_weight_var_mode"},
        {"mxv", "speed_max_val"}};
    Options named = Options::object();
    for (const auto& [key, value] : options.items()) {
        const auto alias = aliases.find(key);
        named[alias == aliases.end() ? key : alias->second] = value;
    }
    if (named.size() != options.size()) {
        return std::nullopt;  // a setting given under both of its names
    }
    Options o = defaults;
    for (const auto& [key, value] : named.items()) {
        if (!defaults.contains(key)) {
            return std::nullopt;
        }
        o[key] = value;
    }
    // As libvmaf parses the option's text.
    const auto number = [&o](const char* key) {
        const std::optional<double> parsed = python_float(option_text(o[key]));
        if (!parsed) {
            throw VmafV1Error(std::string("the model's ") + key + " is not a number");
        }
        return *parsed;
    };
    return std::array<double, 7>{number("speed_kernelscale"),
                                 number("speed_prescale"),
                                 option_text(o["speed_prescale_method"]) == "bilinear" ? 1.0 : 0.0,
                                 number("speed_sigma_nn"),
                                 number("speed_nn_floor"),
                                 std::trunc(number("speed_weight_var_mode")),
                                 number("speed_max_val")};
}

// Starts libvmaf's CPU extractors, with the model's options, for the
// features the GPU does not calculate (`on_gpu`).
void use_cpu_features(const Libvmaf& library, VmafContext* context, const ModelOptions& options,
                      const std::set<std::string>& on_gpu) {
    for (const CpuExtractor& wanted : cpu_extractors) {
        if (on_gpu.contains(wanted.feature)) {
            continue;
        }
        VmafFeatureDictionary* dictionary = nullptr;
        for (const auto& [key, value] : options.at(wanted.feature).items()) {
            if (library.feature_dictionary_set(&dictionary, key.c_str(), option_text(value).c_str()) != 0) {
                library.feature_dictionary_free(&dictionary);
                throw LibvmafError("Setting " + key + " failed");
            }
        }
        // libvmaf takes the dictionary.
        check_libvmaf(library.use_feature(context, wanted.extractor, dictionary),
                      std::string("Starting ") + wanted.extractor);
    }
}

// Scores three small grey frame pairs with a context.
void score_probe_pictures(const Libvmaf& library, VmafContext* context) {
    for (unsigned index = 0; index < 3; ++index) {
        std::array<VmafPicture, 2> pictures{};
        for (std::size_t made = 0; made < pictures.size(); ++made) {
            VmafPicture& picture = pictures[made];
            if (const int error = library.picture_alloc(&picture, VMAF_PIX_FMT_YUV420P, 8, v1_probe_width,
                                                        v1_probe_height);
                error != 0) {
                for (std::size_t given = 0; given < made; ++given) {
                    library.picture_unref(&pictures[given]);
                }
                check_libvmaf(error, "Allocating a picture");
            }
            for (int plane = 0; plane < 3; ++plane) {
                std::memset(picture.data[plane], static_cast<int>(100 + index),
                            static_cast<std::size_t>(picture.stride[plane]) * picture.h[plane]);
            }
        }
        check_libvmaf(library.read_pictures(context, &pictures[0], &pictures[1], index), "Scoring a probe frame");
    }
    check_libvmaf(library.read_pictures(context, nullptr, nullptr, 0), "Finishing the probe");
}

// The names of the features a context has scores for. libvmaf has no call
// that lists them, so they are read from its JSON output.
std::vector<std::string> output_names(const Libvmaf& library, VmafContext* context) {
    // Made first, so that its short name can be found (libvmaf_path).
    const TempFile listing = TempFile::make<LibvmafError>(std::filesystem::temp_directory_path(), "vml-vmaf-v1-",
                                                          ".json", "A file for libvmaf's feature list could not be made.");
    const std::filesystem::path& file = listing.path();
    check_libvmaf(library.write_output(context, libvmaf_path(file).c_str(), VMAF_OUTPUT_FORMAT_JSON),
                  "Listing features");
    std::ifstream written(file, std::ios::binary);
    const nlohmann::json output = nlohmann::json::parse(written);
    std::vector<std::string> names;
    for (const auto& [name, value] : output.at("frames").back().at("metrics").items()) {
        names.push_back(name);
    }
    return names;
}

// For a model: {feature: the name the model's prediction reads it under},
// and {CAMBI and SpEED: the name libvmaf's extractor gives it, with the
// model's options}.
struct FeatureNames {
    std::map<std::string, std::string> model;
    std::map<std::string, std::string> cpu;
};

// libvmaf builds both names from the options, so they are found by running
// it on three small frames, once per model.
const FeatureNames& feature_names(const std::filesystem::path& model) {
    static std::mutex mutex;
    static std::map<std::string, FeatureNames> found;
    std::error_code error;
    const std::string key =
        platform::path_text(std::filesystem::weakly_canonical(std::filesystem::absolute(model, error), error));
    {
        const std::lock_guard lock(mutex);
        if (const auto known = found.find(key); known != found.end()) {
            return known->second;
        }
    }
    const Libvmaf& library = libvmaf();
    const ModelOptions options = model_options(model);
    FeatureNames names;
    for (const bool with_model : {true, false}) {
        std::vector<std::string> listed;
        {
            LibvmafHandles handles(library);
            handles.start(0, 1);
            if (with_model) {
                handles.load_model(model);
                check_libvmaf(library.use_features_from_model(handles.context, handles.model), "Setting up the model");
            } else {
                use_cpu_features(library, handles.context, options, {});
            }
            score_probe_pictures(library, handles.context);
            listed = output_names(library, handles.context);
        }
        std::vector<std::pair<const char*, const char*>> wanted;
        if (with_model) {
            wanted.assign(gpu_features.begin(), gpu_features.end());
        }
        for (const CpuExtractor& extractor : cpu_extractors) {
            wanted.emplace_back(extractor.feature, extractor.prefix);
        }
        std::map<std::string, std::string>& named = with_model ? names.model : names.cpu;
        for (const auto& [feature, prefix] : wanted) {
            const std::string start(prefix);
            std::vector<std::string> matches;
            for (const std::string& name : listed) {
                if (name == start || name.starts_with(start + "_")) {
                    matches.push_back(name);
                }
            }
            if (matches.size() != 1) {
                std::string shown;
                for (const std::string& name : matches.empty() ? listed : matches) {
                    shown += (shown.empty() ? "" : ", ") + name;
                }
                throw VmafV1Error(std::string("libvmaf names ") + feature +
                                  " in a way this build does not know: " + shown);
            }
            named[feature] = matches.front();
        }
    }
    const std::lock_guard lock(mutex);
    return found.emplace(key, std::move(names)).first->second;
}

// A frame's planes in a packed 4:2:0 frame: where each starts, its rows and
// the bytes of each row.
struct PlaneLayout {
    std::size_t offset = 0;
    std::size_t rows = 0;
    std::size_t row_bytes = 0;
};

std::array<PlaneLayout, 3> packed_layout(int width, int height, int bit_depth) {
    const std::size_t sample = bit_depth <= 8 ? 1 : 2;
    const auto columns = static_cast<std::size_t>(width), rows = static_cast<std::size_t>(height);
    const std::size_t chroma_columns = (columns + 1) / 2, chroma_rows = (rows + 1) / 2;
    std::array<PlaneLayout, 3> planes{};
    planes[0] = {0, rows, columns * sample};
    std::size_t offset = columns * rows * sample;
    for (std::size_t plane = 1; plane < 3; ++plane) {
        planes[plane] = {offset, chroma_rows, chroma_columns * sample};
        offset += chroma_columns * chroma_rows * sample;
    }
    return planes;
}

// Copies the given planes of a packed frame into a libvmaf picture.
// libvmaf rounds an odd size's chroma planes down, and FFmpeg up.
void fill_picture(VmafPicture& picture, std::span<const std::uint8_t> frame, const std::array<PlaneLayout, 3>& layout,
                  std::size_t sample, std::initializer_list<int> planes) {
    for (const int plane : planes) {
        const PlaneLayout& from = layout[static_cast<std::size_t>(plane)];
        const std::size_t rows = std::min<std::size_t>(from.rows, picture.h[plane]);
        const std::size_t width = std::min<std::size_t>(from.row_bytes, picture.w[plane] * sample);
        auto* target = static_cast<std::uint8_t*>(picture.data[plane]);
        const auto stride = static_cast<std::size_t>(picture.stride[plane]);
        for (std::size_t row = 0; row < rows; ++row) {
            std::memcpy(target + row * stride, frame.data() + from.offset + row * from.row_bytes, width);
        }
    }
}

}  // namespace

std::string v1_library_build() {
    return "libvmaf-fast " + libvmaf_fast_version() + " (Vulkan: ADM3, motion3, CAMBI, SpEED)";
}

// ------------------------------------------------------------------ V1Scorer

struct V1Scorer::State {
    const Libvmaf& library = libvmaf();
    const VulkanEngine& engine = vulkan_engine();
    std::filesystem::path model;
    std::string name;
    unsigned step = 1;
    unsigned count = 0;
    std::size_t sample = 1;
    std::ptrdiff_t luma_stride = 0;
    std::array<PlaneLayout, 3> layout{};
    const FeatureNames* names = nullptr;
    VulkanContext* gpu = nullptr;
    VmafContext* cpu = nullptr;  // libvmaf's extractors, for the features the GPU does not calculate
    std::optional<SharedSlots> shared;
    // The features the GPU calculates beyond ADM3 and motion3.
    std::set<std::string> on_gpu;
    // Where a decoder handing frames over puts its chroma planes in each
    // shared buffer: from `offset`, `spacing` bytes apart, rows `stride`
    // bytes apart. Nothing: they go through the engine's host memory.
    struct SharedChroma {
        std::uint64_t offset = 0;
        std::uint32_t spacing = 0;
        std::uint32_t stride = 0;
    };
    std::optional<SharedChroma> shared_chroma;
    // Whether the GPU, rather than this thread, waits for the decoder's
    // copies into the shared buffers (through the decoder's timeline
    // semaphore, imported): this thread then goes on to the next pair at once.
    bool after = false;
    // Whether the engine reads the decoders' pictures where they are
    // (vv_pictures).
    bool pictures = false;
    // The decoders' slots given back through release_later that the engine
    // may still read: the last frame that reads each, its stream and slot.
    struct Held {
        long long last = 0;
        GpuFrameStream* stream = nullptr;
        int slot = 0;
    };
    std::deque<Held> held;
    // libvmaf's pictures page-locked for the decoder that downloads into
    // them, or nullptr for those it could not lock.
    std::map<void*, GpuFrameStream*> pinned;

    bool cpu_features() const { return cpu != nullptr; }

    void check(int error, const std::string& what) const { check_vulkan(engine, error, what); }

    // CAMBI on the GPU, where the engine and the model's options allow.
    void start_cambi_on_gpu(const ModelOptions& options) {
        if (engine.v1_cambi == nullptr || engine.v1_cambi_scores == nullptr) {
            return;
        }
        if (const std::optional<std::array<double, 9>> cambi = cambi_gpu_options(options.at(v1_cambi))) {
            const int code = engine.v1_cambi(gpu, cambi->data());
            if (code == 0) {
                on_gpu.insert(v1_cambi);
            } else if (code != unsupported_options) {
                check(code, "Starting CAMBI");
            }
        }
    }
    // SpEED on the GPU, where the engine and the model's options allow. Not
    // on arm64 (Apple silicon): libvmaf's SpEED there is its arm64 code (a
    // NEON covariance kernel, the filter and the decimation in one, its
    // multiplies and adds fused), which the engine, a port of its x86 code,
    // does not give bit for bit; libvmaf calculates it.
    void start_speed_on_gpu(const ModelOptions& options) {
#ifdef __aarch64__
        static_cast<void>(options);
        return;
#endif
        if (engine.v1_speed == nullptr || engine.submit_v1 == nullptr || engine.chroma_staging == nullptr ||
            engine.v1_speed_scores == nullptr) {
            return;
        }
        if (const std::optional<std::array<double, 7>> speed = speed_gpu_options(options.at(v1_speed))) {
            const int code = engine.v1_speed(gpu, speed->data());
            if (code == 0) {
                on_gpu.insert(v1_speed);
            } else if (code != unsupported_options) {
                check(code, "Starting SpEED");
            }
        }
    }
    // What the decoder's hand-over gives besides the luma: the chroma in
    // place for SpEED, and its timeline, to start after its copies.
    void connect_handover(GpuFrameStream& stream) {
        if (on_gpu.contains(v1_speed) && engine.shared_chroma != nullptr && stream.copies_planes()) {
            SharedChroma chroma;
            if (engine.shared_chroma(gpu, &chroma.offset, &chroma.spacing, &chroma.stride) == 0) {
                shared_chroma = chroma;
            }
        }
        if (engine.import_timeline != nullptr && engine.commit_after != nullptr) {
            if (void* const timeline = stream.timeline(); timeline != nullptr) {
                after = engine.import_timeline(gpu, timeline) == 0;
                platform::close_native_handle(timeline);
            }
        }
    }
    // libvmaf's CPU extractors, which run only for the features the GPU
    // does not calculate, on `threads` workers (else as many as the cores
    // allow, or three for SpEED alone), with a pool of pictures for `width`
    // x `height` x `bit_depth`.
    void start_cpu_features(const ModelOptions& options, int width, int height, int bit_depth,
                            std::optional<int> threads) {
        int workers = threads ? *threads : std::max(2, std::min(16, static_cast<int>(platform::logical_cores()) - 2));
        // SpEED's chroma alone (arm64, start_speed_on_gpu) keeps up with the
        // GPU on three: an M1 Pro scored 4K 10-bit and 1080p as fast on two
        // as on eight, and the pool of pictures for eight was 380 MB more at
        // 4K (two pictures for each worker and two more pairs).
        if (!threads && on_gpu.contains(v1_cambi) && !on_gpu.contains(v1_speed)) {
            workers = std::min(workers, 3);
        }
        LibvmafHandles handles(library);
        handles.start(workers, 1);
        use_cpu_features(library, handles.context, options, on_gpu);
        const VmafPictureConfiguration pool{
            {static_cast<unsigned>(width), static_cast<unsigned>(height), static_cast<unsigned>(bit_depth),
             VMAF_PIX_FMT_YUV420P},
            static_cast<unsigned>(2 * (workers + 2))};
        check_libvmaf(library.preallocate_pictures(handles.context, pool), "Allocating pictures");
        cpu = std::exchange(handles.context, nullptr);
    }

    ~State() {
        shared.reset();  // the decoder's imports go back before the context that allocated them
        if (gpu != nullptr) {
            engine.destroy(gpu);  // waits for its frames
        }
        release_all();
        // Locked memory is unlocked before vmaf_close frees the pictures it
        // is in: CUDA must let go of memory before it is freed. No download
        // into it is under way (each returns once its copy is done).
        for (const auto& [address, stream] : pinned) {
            if (stream != nullptr) {
                stream->unpin(address);
            }
        }
        if (cpu != nullptr) {
            library.close(cpu);  // waits for the extractors
        }
    }

    // Takes two pictures from libvmaf's pool, fills them with `fill` and
    // gives them to libvmaf as frame `index`.
    void to_libvmaf(const std::function<void(VmafPicture&, VmafPicture&)>& fill, unsigned index) {
        VmafPicture reference{};
        VmafPicture distorted{};
        check_libvmaf(library.fetch_preallocated_picture(cpu, &reference), "Taking a picture");
        if (const int error = library.fetch_preallocated_picture(cpu, &distorted); error != 0) {
            library.picture_unref(&reference);  // a picture never given back would keep vmaf_close waiting
            check_libvmaf(error, "Taking a picture");
        }
        try {
            fill(reference, distorted);
        } catch (...) {
            library.picture_unref(&reference);
            library.picture_unref(&distorted);
            throw;
        }
        // libvmaf takes both pictures, even when it fails.
        check_libvmaf(library.read_pictures(cpu, &reference, &distorted, index),
                      "Scoring frame " + std::to_string(index));
    }

    // Page-locks a picture from libvmaf's pool the first time it is used, so
    // that the decoder can write into it by itself. The pool's pictures are
    // used again and again, so each is locked once.
    void pin(VmafPicture& picture, GpuFrameStream& stream) {
        void* const address = picture.data[0];
        if (!pinned.contains(address)) {
            const std::uint64_t size = static_cast<std::uint64_t>(picture.stride[0]) * picture.h[0] +
                                       2 * static_cast<std::uint64_t>(picture.stride[1]) * picture.h[1];
            pinned[address] = stream.pin(address, size) ? &stream : nullptr;
        }
    }

    // Gives libvmaf's extractors a pair the decoders hold, as frame `count`:
    // the whole test frame (CAMBI reads its luma) and the source's chroma
    // (SpEED reads both videos' chroma; nothing here reads the source's
    // luma), downloaded straight into libvmaf's pictures.
    void to_libvmaf_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
        to_libvmaf(
            [&](VmafPicture& reference_picture, VmafPicture& distorted_picture) {
                pin(reference_picture, test);
                pin(distorted_picture, test);
                const std::array<std::int64_t, 3> pitches{distorted_picture.stride[0], distorted_picture.stride[1],
                                                          distorted_picture.stride[2]};
                test.download_planes(test_slot,
                                     {distorted_picture.data[0], distorted_picture.data[1], distorted_picture.data[2]},
                                     pitches);
                reference.download_planes(reference_slot,
                                          {nullptr, reference_picture.data[1], reference_picture.data[2]}, pitches);
            },
            count);
    }

    // Gives back the slots held by release_later that the engine is done
    // with. The engine has finished every frame up to one round of shared
    // slots back, because starting a frame in a shared slot waits for the
    // frame that used the slot before.
    void release_done() {
        const long long done = static_cast<long long>(count) - static_cast<long long>(shared->count());
        while (!held.empty() && held.front().last <= done) {
            const Held given = held.front();
            held.pop_front();
            given.stream->release(given.slot);
        }
    }

    // Gives back every slot release_later holds, once the engine is done.
    void release_all() {
        while (!held.empty()) {
            const Held given = held.front();
            held.pop_front();
            given.stream->release(given.slot);
        }
    }
};

// The ADM and motion options of a VMAF v1 model as the GPU takes them, in
// its order; throws for one it does not calculate.
std::array<double, 9> adm_motion_values(const Options& adm, const Options& motion) {
    const std::array<double, 9> values{
        option_number(option_or(adm, "adm_norm_view_dist", 3.0)),
        option_number(option_or(adm, "adm_ref_display_height", 1080)),
        option_number(option_or(adm, "adm_csf_mode", 0)),
        option_number(option_or(adm, "adm_noise_weight", 0.03125)),
        option_number(option_or(adm, "adm_dlm_weight", 1.0)),
        option_number(option_or(adm, "adm_min_val", 0.0)),
        option_number(option_or(motion, "motion_max_val", 10000.0)),
        option_truthy(option_or(motion, "motion_five_frame_window", false)) ? 1.0 : 0.0,
        option_truthy(option_or(motion, "motion_moving_average", false)) ? 1.0 : 0.0};
    static const std::set<std::string> adm_known{"adm_norm_view_dist", "adm_ref_display_height", "adm_csf_mode",
                                                 "adm_noise_weight",   "adm_dlm_weight",         "adm_min_val",
                                                 "adm_enhn_gain_limit"};
    static const std::set<std::string> motion_known{"motion_max_val", "motion_five_frame_window",
                                                    "motion_moving_average"};
    std::set<std::string> unsupported;
    for (const auto& [key, value] : adm.items()) {
        if (!adm_known.contains(key)) {
            unsupported.insert(key);
        }
    }
    for (const auto& [key, value] : motion.items()) {
        if (!motion_known.contains(key)) {
            unsupported.insert(key);
        }
    }
    if (!unsupported.empty() || option_number(option_or(adm, "adm_enhn_gain_limit", 100.0)) != 1.0) {
        std::string listed;
        for (const std::string& key : unsupported) {
            listed += (listed.empty() ? "" : ", ") + key;
        }
        throw VmafV1Error("the model has ADM or motion options the GPU does not calculate: " + listed);
    }
    return values;
}

V1Scorer::V1Scorer(int width, int height, int bit_depth, const std::filesystem::path& model, int n_subsample,
                   std::optional<int> device, std::optional<int> threads, std::string name, GpuFrameStream* shared,
                   bool gpu_cambi, bool gpu_speed, bool pictures)
    : state_(std::make_unique<State>()) {
    State& s = *state_;
    const VulkanEngine& engine = s.engine;
    if (engine.create_v1 == nullptr || engine.features_v1 == nullptr) {
        throw VmafV1Error("the bundled Vulkan engine has no VMAF v1");
    }
    s.model = model;
    s.name = std::move(name);
    s.step = static_cast<unsigned>(std::max(1, n_subsample));
    const ModelOptions options = model_options(model);
    if (speed_too_small(options.at(v1_speed), width, height)) {
        throw VmafV1Error(std::to_string(width) + "x" + std::to_string(height) +
                          " is too small for this VMAF v1 model's SpEED feature");
    }
    s.names = &feature_names(model);
    if (!device) {
        const std::optional<VulkanDevice> chosen = best_vulkan_device(vulkan_devices());
        if (!chosen) {
            throw VmafV1Error("no GPU that Vulkan can calculate VMAF on");
        }
        device = chosen->index;
    }
    const std::array<double, 9> values = adm_motion_values(options.at(v1_adm3), options.at(v1_motion3));
    int flags = (vulkan_frames_in_flight << 8) | (shared != nullptr ? vulkan_shared_flag : 0);
    if (pictures && shared != nullptr && engine.pictures != nullptr && engine.pictures_mode != nullptr) {
        flags |= pictures_flag;
    }
    s.check(engine.create_v1(&s.gpu, *device, width, height, bit_depth, flags, values.data()), "Starting Vulkan");
    if (shared != nullptr) {
        if ((width & 1) != 0 || (height & 1) != 0) {
            throw VmafV1Error("an odd size is not handed over on the GPU");
        }
        s.shared.emplace(engine, s.gpu, *shared);
    }
    if (gpu_cambi) {
        s.start_cambi_on_gpu(options);
    }
    if (gpu_speed) {
        s.start_speed_on_gpu(options);
    }
    if (s.shared) {
        s.connect_handover(*shared);
    }
    s.pictures = (flags & pictures_flag) != 0 && engine.pictures_mode(s.gpu) == 1;
    if (!s.on_gpu.contains(v1_cambi) || !s.on_gpu.contains(v1_speed)) {
        s.start_cpu_features(options, width, height, bit_depth, threads);
    }
    s.sample = bit_depth <= 8 ? 1 : 2;
    s.luma_stride = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(width) * s.sample);
    s.layout = packed_layout(width, height, bit_depth);
    frame_bytes_ = s.layout[2].offset + s.layout[2].rows * s.layout[2].row_bytes;
}

V1Scorer::~V1Scorer() = default;

void V1Scorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    State& s = *state_;
    if (std::min(reference.size(), distorted.size()) < frame_bytes_) {
        throw VmafV1Error("frame " + std::to_string(s.count) + " is cut short");
    }
    const bool score = s.count % s.step == 0;
    int code = 0;
    if (s.on_gpu.contains(v1_speed)) {  // the GPU needs the chroma planes too
        std::array<const std::uint8_t*, 6> planes{};
        std::array<std::ptrdiff_t, 6> strides{};
        for (std::size_t plane = 0; plane < 3; ++plane) {
            planes[plane] = reference.data() + s.layout[plane].offset;
            planes[3 + plane] = distorted.data() + s.layout[plane].offset;
            strides[plane] = strides[3 + plane] = static_cast<std::ptrdiff_t>(s.layout[plane].row_bytes);
        }
        code = s.engine.submit_v1(s.gpu, planes.data(), strides.data(), score ? 1 : 0);
    } else {
        code = s.engine.submit(s.gpu, reference.data(), s.luma_stride, distorted.data(), s.luma_stride, score ? 1 : 0);
    }
    if (code < 0) {
        s.check(code, "Scoring frame " + std::to_string(s.count));
    }
    if (score && s.cpu_features()) {
        // The whole test frame (CAMBI reads its luma) and the source's chroma
        // (SpEED reads both videos' chroma; nothing here reads the source's
        // luma).
        s.to_libvmaf(
            [&](VmafPicture& reference_picture, VmafPicture& distorted_picture) {
                fill_picture(reference_picture, reference, s.layout, s.sample, {1, 2});
                fill_picture(distorted_picture, distorted, s.layout, s.sample, {0, 1, 2});
            },
            s.count);
    }
    ++s.count;
}

void V1Scorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
    State& s = *state_;
    if (!s.shared) {
        throw std::logic_error("a decoder's pictures for a scorer that shares no memory with it");
    }
    const bool score = s.count % s.step == 0;
    const std::string what = "Scoring frame " + std::to_string(s.count);
    const SharedSlots::Next next = s.shared->next();
    if (s.pictures) {
        // The engine reads both pictures where the decoders left them.
        s.release_done();
        const GpuFrameStream::Texture ref = reference.slot_texture(reference_slot);
        const GpuFrameStream::Texture dis = test.slot_texture(test_slot);
        s.check(s.engine.pictures(s.gpu, ref.handle, ref.width, ref.height, ref.x, ref.y, dis.handle, dis.width,
                                  dis.height, dis.x, dis.y),
                what);
        s.check(s.engine.commit(s.gpu, score ? 1 : 0), what);
        if (score && s.cpu_features()) {
            s.to_libvmaf_decoded(reference, reference_slot, test, test_slot);
        }
        ++s.count;
        return;
    }
    // The copies into the shared buffers. When the GPU waits for them
    // (after), this thread does not: each copy gives the value the decoder's
    // timeline reaches once it is done, else 0.
    const auto copy_luma = [&s, &next](GpuFrameStream& stream, int slot, std::uint64_t address) -> std::uint64_t {
        if (s.after) {
            return stream.copy_luma_async(slot, address, next.pitch);
        }
        stream.copy_luma(slot, address, next.pitch);
        return 0;
    };
    const auto copy_planes = [&s](GpuFrameStream& stream, int slot, const std::array<std::uint64_t, 3>& addresses,
                                  const std::array<std::int64_t, 3>& pitches) -> std::uint64_t {
        if (s.after) {
            return stream.copy_planes_async(slot, addresses, pitches);
        }
        stream.copy_planes(slot, addresses, pitches);
        return 0;
    };
    std::uint64_t copied = 0;
    const bool chroma_copied = score && s.on_gpu.contains(v1_speed) && s.shared_chroma;
    if (chroma_copied) {
        // Each picture's chroma planes go into the slot's shared buffer along
        // with its luma: one copy on the GPU per picture.
        const State::SharedChroma& chroma = *s.shared_chroma;
        const auto at = [&](std::uint64_t plane) { return next.base + chroma.offset + plane * chroma.spacing; };
        const std::array<std::int64_t, 3> pitches{next.pitch, chroma.stride, chroma.stride};
        const std::uint64_t reference_copied =
            copy_planes(reference, reference_slot, {next.reference, at(0), at(1)}, pitches);
        const std::uint64_t test_copied = copy_planes(test, test_slot, {next.distorted, at(2), at(3)}, pitches);
        copied = std::max(reference_copied, test_copied);
    } else {
        copied = copy_luma(reference, reference_slot, next.reference);
        if (score) {
            copied = std::max(copied, copy_luma(test, test_slot, next.distorted));
        }
    }
    if (copied != 0) {
        s.check(s.engine.commit_after(s.gpu, copied), what);
    }
    if (score && s.on_gpu.contains(v1_speed) && !chroma_copied) {
        // The chroma planes are downloaded into the engine's memory for them.
        std::array<std::uint8_t*, 4> chroma{};
        std::uint32_t stride = 0;
        s.check(s.engine.chroma_staging(s.gpu, chroma.data(), &stride), what);
        const std::array<std::int64_t, 3> pitches{0, stride, stride};
        reference.download_planes(reference_slot, {nullptr, chroma[0], chroma[1]}, pitches);
        test.download_planes(test_slot, {nullptr, chroma[2], chroma[3]}, pitches);
    }
    s.check(s.engine.commit(s.gpu, score ? 1 : 0), what);
    if (score && s.cpu_features()) {
        s.to_libvmaf_decoded(reference, reference_slot, test, test_slot);
    }
    ++s.count;
}

bool V1Scorer::pictures() const { return state_->pictures; }

void V1Scorer::release_later(GpuFrameStream& stream, int slot, bool reference) {
    State& s = *state_;
    s.held.push_back({static_cast<long long>(s.count) - 1 + (reference ? 1 : 0), &stream, slot});
}

V1Features V1Scorer::features() {
    State& s = *state_;
    s.check(s.engine.flush(s.gpu), "Scoring");
    V1Features result;
    for (unsigned frame = 0; frame < s.count; frame += s.step) {
        result.frames.push_back(frame);
    }
    constexpr double missing = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> rows(static_cast<std::size_t>(s.count) * 6);
    if (s.count != 0) {
        if (s.cpu_features()) {
            check_libvmaf(s.library.read_pictures(s.cpu, nullptr, nullptr, 0), "Finishing");
        }
        s.check(s.engine.features_v1(s.gpu, rows.data(), s.count), "Reading the features");
    }
    // A column of `values`, `width` values per frame, at `column`, for the
    // frames scored.
    const auto pick = [&result](const std::vector<double>& values, std::size_t width, std::size_t column) {
        std::vector<double> picked;
        picked.reserve(result.frames.size());
        for (const unsigned frame : result.frames) {
            picked.push_back(values[frame * width + column]);
        }
        return picked;
    };
    result.values[v1_adm3] = pick(rows, 6, 0);
    result.values[v1_motion3] = pick(rows, 6, 3);
    if (s.on_gpu.contains(v1_speed)) {
        std::vector<double> speed(static_cast<std::size_t>(s.count) * 3, missing);
        if (s.count != 0) {
            s.check(s.engine.v1_speed_scores(s.gpu, speed.data(), s.count), "Reading SpEED");
        }
        result.values[v1_speed] = pick(speed, 3, 2);  // speed_chroma_uv_score
    }
    if (s.on_gpu.contains(v1_cambi)) {
        std::vector<double> cambi(s.count, missing);
        if (s.count != 0) {
            s.check(s.engine.v1_cambi_scores(s.gpu, cambi.data(), s.count), "Reading CAMBI");
        }
        result.values[v1_cambi] = pick(cambi, 1, 0);
    }
    for (const auto& [feature, name] : s.names->cpu) {
        if (s.on_gpu.contains(feature)) {
            continue;
        }
        std::vector<double>& column = result.values[feature];
        for (const unsigned frame : result.frames) {
            double value = 0;
            check_libvmaf(s.library.feature_score_at_index(s.cpu, name.c_str(), &value, frame),
                          "Reading frame " + std::to_string(frame));
            column.push_back(value);
        }
    }
    return result;
}

PairScores V1Scorer::finish() {
    State& s = *state_;
    PairScores result;
    if (s.count == 0) {
        s.release_all();
        result.by_metric.emplace_back(s.name, std::vector<double>());
        return result;
    }
    const V1Features measured = features();
    s.release_all();  // the engine has finished every frame
    result.by_metric.emplace_back(s.name, predict_v1(s.model, measured));
    result.frames = measured.frames;
    return result;
}

std::vector<double> predict_v1(const std::filesystem::path& model, const V1Features& features) {
    std::vector<FeatureValues> given;
    for (const auto& [feature, name] : feature_names(model).model) {
        const auto values = features.values.find(feature);
        if (values == features.values.end()) {
            throw VmafV1Error("no values of " + feature);
        }
        given.emplace_back(name, values->second);
    }
    return predict_model_file_scores(model, features.frames, given);
}

V1Reference v1_cpu_reference(const std::filesystem::path& model, int width, int height, int bit_depth,
                             const std::vector<std::vector<std::uint8_t>>& reference,
                             const std::vector<std::vector<std::uint8_t>>& distorted, int n_subsample, int threads) {
    if (reference.size() != distorted.size()) {
        throw std::invalid_argument("as many reference frames as distorted ones are needed");
    }
    const Libvmaf& library = libvmaf();
    const FeatureNames& names = feature_names(model);
    LibvmafHandles handles(library);
    handles.start(threads, n_subsample);
    handles.load_model(model);
    check_libvmaf(library.use_features_from_model(handles.context, handles.model), "Setting up the model");
    const VmafPictureConfiguration pool{
        {static_cast<unsigned>(width), static_cast<unsigned>(height), static_cast<unsigned>(bit_depth),
         VMAF_PIX_FMT_YUV420P},
        static_cast<unsigned>(2 * (std::max(1, threads) + 2))};
    check_libvmaf(library.preallocate_pictures(handles.context, pool), "Allocating pictures");
    const std::array<PlaneLayout, 3> layout = packed_layout(width, height, bit_depth);
    const std::size_t sample = bit_depth <= 8 ? 1 : 2;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        std::array<VmafPicture, 2> pictures{};
        const std::array<const std::vector<std::uint8_t>*, 2> frames{&reference[index], &distorted[index]};
        std::size_t taken = 0;
        try {
            for (; taken < 2; ++taken) {
                check_libvmaf(library.fetch_preallocated_picture(handles.context, &pictures[taken]),
                              "Taking a picture");
            }
            for (std::size_t i = 0; i < 2; ++i) {
                fill_picture(pictures[i], *frames[i], layout, sample, {0, 1, 2});
            }
        } catch (...) {
            // A picture taken and never handed over keeps vmaf_close waiting for it.
            for (std::size_t given = 0; given < taken; ++given) {
                library.picture_unref(&pictures[given]);
            }
            throw;
        }
        check_libvmaf(library.read_pictures(handles.context, &pictures[0], &pictures[1], static_cast<unsigned>(index)),
                      "Scoring frame " + std::to_string(index));
    }
    check_libvmaf(library.read_pictures(handles.context, nullptr, nullptr, 0), "Finishing");
    V1Reference result;
    for (std::size_t frame = 0; frame < reference.size(); frame += static_cast<std::size_t>(std::max(1, n_subsample))) {
        result.features.frames.push_back(static_cast<unsigned>(frame));
    }
    for (const auto& [feature, name] : names.model) {
        std::vector<double>& column = result.features.values[feature];
        for (const unsigned frame : result.features.frames) {
            double value = 0;
            check_libvmaf(library.feature_score_at_index(handles.context, name.c_str(), &value, frame),
                          "Reading frame " + std::to_string(frame));
            column.push_back(value);
        }
    }
    for (const unsigned frame : result.features.frames) {
        double value = 0;
        check_libvmaf(library.score_at_index(handles.context, handles.model, &value, frame),
                      "Predicting frame " + std::to_string(frame));
        result.scores.push_back(to_6_decimals(value));
    }
    return result;
}

VulkanProbeFrames v1_probe_frames(int bits, int count) {
    constexpr int width = v1_probe_width, height = v1_probe_height;
    const std::int64_t peak = (std::int64_t{1} << bits) - 1;
    const bool wide = bits != 8;
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const auto at = [](int x, int y) {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
    };
    // A sample as the frames hold it: scaled to `bits`, little-endian.
    const auto append = [peak, wide](std::vector<std::uint8_t>& frame, std::int64_t value) {
        const std::int64_t sample = value * peak / 255;
        frame.push_back(static_cast<std::uint8_t>(sample & 0xFF));
        if (wide) {
            frame.push_back(static_cast<std::uint8_t>(sample >> 8));
        }
    };
    std::vector<std::int64_t> noise(pixels), reference(pixels), distorted(pixels);
    VulkanProbeFrames frames;
    for (int index = 0; index < count; ++index) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // numpy's 64-bit product wraps around, and its >> keeps the sign.
                const auto mixed = static_cast<std::uint64_t>(std::int64_t{x} * 2654435761LL +
                                                              std::int64_t{y} * 40503LL + std::int64_t{index} * 69069LL);
                const auto product = static_cast<std::int64_t>(mixed * 1103515245ULL);
                noise[at(x, y)] = (product >> 16) & 0xFF;
                reference[at(x, y)] = (((x * 3 + y * 2 + index * 7) % 256) + noise[at(x, y)]) / 2;
            }
        }
        // Blurred along each row (the first and last columns kept), then noisy.
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::int64_t value = reference[at(x, y)];
                if (x > 0 && x < width - 1) {
                    value = (reference[at(x - 1, y)] + 2 * reference[at(x, y)] + reference[at(x + 1, y)]) / 4;
                }
                distorted[at(x, y)] = std::clamp<std::int64_t>(value + noise[at(x, y)] % 9 - 4, 0, 255);
            }
        }
        for (const bool test_frame : {false, true}) {
            const std::int64_t shift = test_frame ? 3 : 0;
            const std::vector<std::int64_t>& luma = test_frame ? distorted : reference;
            std::vector<std::uint8_t> frame;
            frame.reserve((pixels + pixels / 2) * (wide ? 2 : 1));
            for (const std::int64_t value : luma) {
                append(frame, value);
            }
            for (const bool v_plane : {false, true}) {
                for (int cy = 0; cy < height / 2; ++cy) {
                    for (int cx = 0; cx < width / 2; ++cx) {
                        append(frame, v_plane ? (cx + cy * 2 + index * 5 + 2 * shift) % 256
                                              : (cx * 2 + cy + index * 3 + shift) % 256);
                    }
                }
            }
            (test_frame ? frames.distorted : frames.reference).push_back(std::move(frame));
        }
    }
    return frames;
}

// --------------------------------------------------------------- MultiScorer

MultiScorer::MultiScorer(int width, int height, int bit_depth, const GpuModels& models, int n_subsample,
                         const std::string& backend, std::optional<int> device, GpuFrameStream* shared,
                         bool pictures) {
    GpuModels vmaf_models;
    std::optional<std::string> v1_model;
    for (const auto& [key, model] : models) {
        if (key == "vmaf_v1") {
            v1_model = model;
        } else {
            vmaf_models.emplace_back(key, model);
        }
    }
    const bool vulkan = backend == "vulkan";
    if (!vmaf_models.empty()) {
        if (vulkan) {
            vulkan_ =
                std::make_unique<VulkanScorer>(width, height, bit_depth, vmaf_models, n_subsample, device, shared);
        } else {
            cuda_ = std::make_unique<CudaScorer>(width, height, bit_depth, vmaf_models, n_subsample, shared != nullptr);
        }
    }
    if (v1_model) {
        const std::string_view file = model_file_of(*v1_model).value_or(*v1_model);
        v1_ = std::make_unique<V1Scorer>(width, height, bit_depth, platform::path_from_text(file), n_subsample,
                                         vulkan ? device : std::nullopt, std::nullopt, "vmaf_v1", shared, true, true,
                                         pictures);
    }
    if (!vulkan_ && !cuda_ && !v1_) {
        throw std::invalid_argument("no model to score");
    }
}

MultiScorer::~MultiScorer() = default;

std::size_t MultiScorer::frame_bytes() const {
    return vulkan_ ? vulkan_->frame_bytes() : cuda_ ? cuda_->frame_bytes() : v1_->frame_bytes();
}

void MultiScorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    for (PairScorer* scorer : {static_cast<PairScorer*>(vulkan_.get()), static_cast<PairScorer*>(cuda_.get()),
                               static_cast<PairScorer*>(v1_.get())}) {
        if (scorer != nullptr) {
            scorer->add(reference, distorted);
        }
    }
}

void MultiScorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
    // VMAF v0.6.1 and NEG read only the luma planes, copied on the GPU.
    const auto copy_reference = [&](std::uint64_t address, std::int64_t pitch) {
        reference.copy_luma(reference_slot, address, pitch);
    };
    const auto copy_test = [&](std::uint64_t address, std::int64_t pitch) { test.copy_luma(test_slot, address, pitch); };
    if (vulkan_) {
        vulkan_->add_shared(copy_reference, copy_test);
    }
    if (cuda_) {
        cuda_->add_decoded({.copy = copy_reference, .write = {}}, {.copy = copy_test, .write = {}});
    }
    if (v1_) {
        v1_->add_decoded(reference, reference_slot, test, test_slot);
    }
}

bool MultiScorer::pictures() const { return v1_ && v1_->pictures(); }

void MultiScorer::release_later(GpuFrameStream& stream, int slot, bool reference) {
    v1_->release_later(stream, slot, reference);
}

PairScores MultiScorer::finish() {
    PairScores result;
    bool first = true;
    for (PairScorer* scorer : {static_cast<PairScorer*>(vulkan_.get()), static_cast<PairScorer*>(cuda_.get()),
                               static_cast<PairScorer*>(v1_.get())}) {
        if (scorer == nullptr) {
            continue;
        }
        PairScores scored = scorer->finish();
        if (!first && scored.frames != result.frames) {
            throw VmafV1Error("the GPU's scorers scored different frames");
        }
        result.frames = std::move(scored.frames);
        for (auto& metric : scored.by_metric) {
            result.by_metric.push_back(std::move(metric));
        }
        first = false;
    }
    return result;
}

PairScores score_decoded_v1(const VideoInfo& source, const VideoInfo& distorted,
                            const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                            const GpuPlan& plan, int n_subsample, const std::optional<std::string>& duration_limit,
                            long long total_frames, const std::string& scale_algorithm,
                            const std::pair<std::string, std::string>& decoders, const std::string& ffmpeg,
                            const std::string& ffprobe,
                            const std::function<void(long long, long long, double)>& on_progress,
                            const CancelToken& cancel) {
    const auto& [source_backend, test_backend] = decoders;
    for (const std::string& library : {source_backend, test_backend}) {
        if (library.empty() || !decoder_available(library)) {
            throw GpuDecodeUnavailableError("the " + library + " decoder library is not bundled");
        }
    }
    require_same_range(source, distorted);
    // Each video's plan, the test video's first: whole frames, at the size
    // and depth compared at.
    std::vector<DecodePlan> plans;
    for (const bool test_video : {true, false}) {
        const VideoInfo& info = test_video ? distorted : source;
        DecodePlan decoded = plan_decode(info, test_video ? distorted_crop : source_crop, 6, false,
                                         std::pair(plan.width, plan.height), scale_algorithm);
        widen_to(decoded, info, plan.bit_depth);
        const auto [supported, refusal] = decoder_supports(0, decoded, test_video ? test_backend : source_backend);
        if (!supported) {
            throw GpuDecodeUnavailableError(refusal);
        }
        plans.push_back(decoded);
    }
    const std::optional<int> device = plan.device >= 0 ? std::optional(plan.device) : std::nullopt;
    // When the engine reads pictures where the decoder left them (V1Scorer's
    // pictures mode), they stay the decoder's until the GPU is done with them:
    // two pairs being matched up, the frames in flight (3), and the previous
    // source picture for motion. That takes six of a decoder's slots, and is
    // faster (the Python app's, 4K HEVC: 98.2 -> 106.1 fps). It is used for
    // two H.264 videos too. Windows' H.264 decoder gives layers of a texture
    // array, which are copied into textures of their own anyway
    // (native/mf_frames.cpp), so the speed is the same (1080p: 251.0 -> 251.3
    // fps), but it uses less GPU memory, because the engine's buffers for the
    // planes are not made (1080p: 342 -> 291 MB committed).
    const bool v1 = std::any_of(plan.models.begin(), plan.models.end(),
                                [](const auto& model) { return model.first == "vmaf_v1"; });
    const int pool = plan.backend == "vulkan" && v1 && source_backend == test_backend &&
                             hands_over_textures(test_backend)
                         ? 6
                         : 4;
    GpuFrameStream test(distorted, plans[0], 0, pool, test_backend, ffmpeg, ffprobe, true);
    GpuFrameStream reference(source, plans[1], 0, pool, source_backend, ffmpeg, ffprobe, true);
    // Declared after the streams so that it goes first: it holds on to the
    // decoders' memory.
    std::optional<MultiScorer> scorer;
    // The decoder's pictures stay on the GPU and reach the scorers without a
    // copy through the CPU. This works with NVIDIA's decoder, and with AMD's
    // when it hands over to Vulkan (not to libvmaf's CUDA code), when the
    // same decoder decodes both videos.
    bool on_gpu = false;
    if (test.handover() && reference.handover() && source_backend == test_backend &&
        (test_backend == "nvidia" || plan.backend == "vulkan")) {
        try {
            scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample, plan.backend, device,
                           &test, pool > 4 && test.gives_textures() && reference.gives_textures());
            on_gpu = true;
        } catch (const LibvmafError& error) {
            logger.info(std::string("VMAF v1 with the GPU: the frames go through system memory (") + error.what() +
                        ")");
        }
    }
    if (!scorer) {
        scorer.emplace(plan.width, plan.height, plan.bit_depth, plan.models, n_subsample, plan.backend, device);
    }
    if (std::any_of(plans.begin(), plans.end(),
                    [&](const DecodePlan& decoded) { return decoded.frame_bytes() != scorer->frame_bytes(); })) {
        throw VmafV1Error("the decoders' frames are not the size the scorers take");
    }
    std::array<std::vector<std::uint8_t>, 2> buffers;
    if (!on_gpu) {
        buffers[0].resize(scorer->frame_bytes());
        buffers[1].resize(scorer->frame_bytes());
    }
    test.start();
    reference.start();
    SlotReleases releases;
    if (on_gpu && scorer->pictures()) {  // given back once the GPU has read them
        releases.test = [&](int slot) { scorer->release_later(test, slot, false); };
        releases.reference = [&](int slot) { scorer->release_later(reference, slot, true); };
    }
    score_frame_pairs(
        test, reference, distorted, duration_limit, total_frames,
        [&](int test_slot, int reference_slot) {
            if (on_gpu) {
                scorer->add_decoded(reference, reference_slot, test, test_slot);
            } else {
                reference.download(reference_slot, buffers[0].data());
                test.download(test_slot, buffers[1].data());
                scorer->add(buffers[0], buffers[1]);
            }
        },
        on_progress, cancel, releases);
    return scorer->finish();
}

// ------------------------------------------------------------- availability

V1Probe probe_v1() {
    try {
        const VulkanProbe vulkan = probe_vulkan();
        if (!vulkan.available) {
            return {false, vulkan.text};
        }
        for (const auto& [model, bits] : probe_models) {
            const std::filesystem::path path = resource_folder() / "models" / platform::path_from_text(model);
            std::error_code error;
            if (!std::filesystem::is_regular_file(path, error)) {
                return {false, "the VMAF v1 models are not bundled"};
            }
            const VulkanProbeFrames frames = v1_probe_frames(bits);
            const V1Reference expected =
                v1_cpu_reference(path, v1_probe_width, v1_probe_height, bits, frames.reference, frames.distorted);
            V1Features got;
            {
                V1Scorer scorer(v1_probe_width, v1_probe_height, bits, path, 1, vulkan.device, 2);
                for (std::size_t i = 0; i < frames.reference.size(); ++i) {
                    scorer.add(frames.reference[i], frames.distorted[i]);
                }
                got = scorer.features();
            }
            for (const char* feature : {v1_adm3, v1_motion3, v1_cambi, v1_speed}) {
                const std::vector<double>& mine = got.values.at(feature);
                const std::vector<double>& theirs = expected.features.values.at(feature);
                // Bit for bit: the same doubles, NaN for NaN.
                if (mine.size() != theirs.size() ||
                    (!mine.empty() && std::memcmp(mine.data(), theirs.data(), mine.size() * sizeof(double)) != 0)) {
                    return {false, "the GPU calculates VMAF v1 wrongly (" + std::to_string(bits) + "-bit self-test)"};
                }
            }
        }
        return {true, vulkan.text};
    } catch (const LibvmafError& error) {
        return {false, error.what()};
    } catch (const std::filesystem::filesystem_error& error) {
        return {false, error.what()};
    }
}

namespace {

// The probe's result, kept for the process, and the lock held while it runs.
std::mutex v1_probe_mutex;
std::optional<V1Probe> v1_probed;

}  // namespace

bool v1_probe_ready() {
    const std::unique_lock lock(v1_probe_mutex, std::try_to_lock);
    return lock.owns_lock() && v1_probed.has_value();
}

V1Probe v1_available() {
    std::optional<V1Probe>& probed = v1_probed;
    const std::lock_guard lock(v1_probe_mutex);
    if (!probed) {
        try {
            if (!isolation_program()) {
                probed = probe_v1();
            } else {
                const nlohmann::json result = run_isolated("vmaf-v1-probe", "the GPU VMAF v1 probe",
                                                           nlohmann::json::object(), {}, CancelToken());
                probed = V1Probe{result.at("available").get<bool>(), result.at("text").get<std::string>()};
            }
        } catch (const IsolatedCrashError& error) {
            probed = V1Probe{false, error.what()};
        } catch (const std::exception& error) {
            logger.warning(std::string("The GPU VMAF v1 probe failed: ") + error.what());
            probed = V1Probe{false, std::string("the GPU probe failed: ") + error.what()};
        }
        logger.info(std::string("VMAF v1 with the GPU") + (probed->available ? "" : " unavailable") + ": " +
                    probed->text);
    }
    return *probed;
}

bool v1_on_gpu(const std::string& model, std::optional<std::pair<int, int>> size) {
    Options speed = Options::object();
    if (!model.empty()) {
        const std::string_view file = model_file_of(model).value_or(model);
        try {
            speed = model_options(platform::path_from_text(file)).at(v1_speed);
        } catch (const VmafV1Error&) {
            return false;
        }
    }
    if (size && speed_too_small(speed, size->first, size->second)) {
        return false;
    }
    return v1_available().available;
}

}  // namespace vml
