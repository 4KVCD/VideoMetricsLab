#include "vml/libvmaf_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

#include "vml/app_log.h"
#include "vml/chroma_vmaf.h"
#include "vml/gpu_cpu_metrics.h"
#include "vml/gpu_frames.h"
#include "vml/host_frames.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/python_text.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.libvmaf_cpu");

// Copies `rows` rows of `row_bytes` from `from` to `to`, where they are written packed.
void copy_strided_rows(Plane from, std::size_t row_bytes, std::size_t rows, std::uint8_t* to) {
    for (std::size_t row = 0; row < rows; ++row) {
        std::copy_n(from.data + row * from.stride, row_bytes, to + row * row_bytes);
    }
}

using Value = CpuFeature::Value;

// `scorer`'s `error` after it started, as GpuMetricsFailed: score_plan then
// calculates its metrics in libvmaf instead.
GpuMetricsFailed gpu_failed(const GpuMetricsScorer& scorer, const std::exception& error) {
    return GpuMetricsFailed(metric_labels(scorer.metrics(), " and ") + " on the GPU failed: " + error.what());
}

// libvmaf's CAMBI score without options; with them it is named from them
// (cambi_feature).
constexpr std::string_view cambi_default_feature = "Cambi_feature_cambi_score";

// A plane of a metric of the three planes (MetricDefinition::plane_of) is
// the feature its metric's extractor gives for that plane, read in the same
// pass (extractors_for starts the extractor for its metric).
constexpr std::array<CpuFeature, 25> cpu_features{{
    {"psnr", "psnr", "psnr_y", 6},
    {"ssim", "float_ssim", "float_ssim", 6},
    {"xpsnr", "xpsnr", "xpsnr_y", 4},
    {"psnr_yuv", "psnr", "psnr_y", 6, true, Value::planes_611, "psnr_cb", "psnr_cr"},
    {"psnr_yuv_mse", "psnr", "psnr_y", 6, true, Value::planes_611_mse, "psnr_cb", "psnr_cr"},
    {"psnr_yuv_y", "psnr", "psnr_y", 6, true},
    {"psnr_yuv_u", "psnr", "psnr_cb", 6, true},
    {"psnr_yuv_v", "psnr", "psnr_cr", 6, true},
    {"ssim_yuv", "float_ssim", "float_ssim", 6, true, Value::planes_611, "float_ssim_cb", "float_ssim_cr"},
    {"ssim_yuv_y", "float_ssim", "float_ssim", 6, true},
    {"ssim_yuv_u", "float_ssim", "float_ssim_cb", 6, true},
    {"ssim_yuv_v", "float_ssim", "float_ssim_cr", 6, true},
    {"cambi", "cambi", cambi_default_feature, 6},
    {"ms_ssim", "float_ms_ssim", "float_ms_ssim", 6},
    {"ms_ssim_yuv", "float_ms_ssim", "float_ms_ssim", 6, true, Value::planes_611, "float_ms_ssim_cb",
     "float_ms_ssim_cr"},
    {"ms_ssim_yuv_y", "float_ms_ssim", "float_ms_ssim", 6, true},
    {"ms_ssim_yuv_u", "float_ms_ssim", "float_ms_ssim_cb", 6, true},
    {"ms_ssim_yuv_v", "float_ms_ssim", "float_ms_ssim_cr", 6, true},
    {"psnr_hvs", "psnr_hvs", "psnr_hvs", 6, true},
    {"psnr_hvs_y", "psnr_hvs", "psnr_hvs_y", 6, true},
    {"psnr_hvs_u", "psnr_hvs", "psnr_hvs_cb", 6, true},
    {"psnr_hvs_v", "psnr_hvs", "psnr_hvs_cr", 6, true},
    {"ciede2000", "ciede", "ciede2000", 6, true, Value::delta_e},
    {"vmaf_yuv_u", "", "", 6, true, Value::chroma_vmaf},
    {"vmaf_yuv_v", "", "", 6, true, Value::chroma_vmaf},
}};

bool is_chroma_vmaf(std::string_view key) { return key == vmaf_u_key || key == vmaf_v_key; }

// The PSNR of the planes' squared errors weighted (6 Y + Cb + Cr) / 8, from
// the planes' PSNR in dB. 10^(-PSNR / 10) is a plane's MSE / peak^2, and the
// three planes share one peak, so it cancels. A plane at libvmaf's ceiling
// (6 x bits + 12 dB) counts as the ceiling's error: the result never exceeds
// it.
double psnr_of_weighted_mse(double y, double cb, double cr) {
    const auto error = [](double psnr) { return std::pow(10.0, -psnr / 10.0); };
    return -10.0 * std::log10((6 * error(y) + error(cb) + error(cr)) / 8);
}

// `feature`'s value of a frame from what its features read as calculated:
// `value` (its feature) and `cb` and `cr` (a 6:1:1 metric's chroma ones).
double combined(const CpuFeature& feature, double value, double cb, double cr) {
    switch (feature.value) {
    case CpuFeature::Value::planes_611:
        return (6 * value + cb + cr) / 8;
    case CpuFeature::Value::planes_611_mse:
        return psnr_of_weighted_mse(value, cb, cr);
    case CpuFeature::Value::delta_e:
        return std::pow(10.0, (45.0 - value) / 20.0);  // +infinity (identical) gives 0
    default:
        return value;
    }
}

// The limits cpu_metrics_unfit checks, from libvmaf's extractors.
constexpr int ms_ssim_smallest_plane = 176;  // float_ms_ssim: 11 x 2^4, its window at the 5th scale
constexpr int cambi_smallest_side = 216;     // cambi.c CAMBI_MIN_WIDTH_HEIGHT, for one of the two sides
constexpr int cambi_largest_width_plus_height = 6092;  // its window then stays under its 4226-entry tables
constexpr int cambi_option_smallest = 144;   // the range of its enc_width and enc_height
constexpr int cambi_option_largest = 7680;
constexpr int psnr_hvs_smallest_side = 16;   // two 8x8 blocks of 4:2:0 chroma
constexpr int psnr_hvs_deepest = 12;
constexpr int vmaf_smallest_plane = 32;      // VMAF's smallest side (the registry's), for each chroma plane

// The most memory CpuScorer's pictures may take (unless two pairs alone take
// more, at 8K). On 4K 10-bit video from NVIDIA's decoder, PSNR + SSIM run
// at 49 fps with 2 pairs, 93 with 3, 129 with 4, 159 with 5, and 172 with 6
// (this limit) or 8, where the decoders' feed levels off. XPSNR's pairs take
// longer, and it keeps a pair from the previous frames, so it gets more:
// 135 fps with 7 pairs, 154 with 8, 169 with 9 (its limit) and 171 with 10.
constexpr std::size_t picture_memory = std::size_t{200} << 20;
constexpr std::size_t xpsnr_picture_memory = std::size_t{270} << 20;
// PSNR-HVS-M takes about 3.5 times as long as PSNR per pair: alone on the
// CPU (Core Ultra 9 285K, software-decoded video), 4K 10-bit runs at 115 fps
// with 6 pairs, 181 with 9, 200 with 11, 223 with 13 (12 threads and this
// limit: 13 pairs of 4K 10-bit pictures with chroma) and 225 with 25; 1080p
// 8-bit at 662 fps with 9 pairs, 834 with 13 and 830 with 25.
constexpr int medium_threads = 12;
constexpr std::size_t medium_picture_memory = std::size_t{420} << 20;
// MS-SSIM, CAMBI and CIEDE2000 take far longer per pair than the decoders
// take to feed one, so they get a pair per thread: on 4K 10-bit video beside
// GPU VMAF (RTX 5090, Core Ultra 9 285K, 24 threads), the five new metrics
// run at 6.0 fps with 6 pairs, 10.8 with 12 and 15.1 with 24 (this limit:
// 24 pairs of 4K 10-bit pictures with chroma), for 850 MB more at the peak.
constexpr std::size_t heavy_picture_memory = std::size_t{800} << 20;
// The most threads a pair's MS-SSIM is shared by (cpu_threads): one frame
// thread scales 7.1 / 13.2 / 19.1 fps with 0 / 1 / 2 helpers (4K 10-bit).
constexpr int ms_ssim_threads_per_pair = 3;

bool has(const std::vector<std::string>& metrics, std::string_view key) {
    return std::find(metrics.begin(), metrics.end(), key) != metrics.end();
}

// An extractor of libvmaf's and the options it is started with.
using ExtractorOptions = std::vector<std::pair<std::string, std::string>>;
struct Extractor {
    std::string name;
    ExtractorOptions options;
};

// The size CAMBI scores the test video at (its enc_width and enc_height):
// its own, where it was scaled up to the compared size on at least one side
// and not down on the other, and libvmaf takes that size; otherwise nothing,
// the compared size (cambi.c itself scores a video scaled down at the
// compared size).
std::optional<std::pair<int, int>> cambi_encode_size(const CambiInput& cambi, int width, int height) {
    const auto in_range = [](int side) { return side >= cambi_option_smallest && side <= cambi_option_largest; };
    if (cambi.width <= 0 || cambi.height <= 0 || cambi.width > width || cambi.height > height ||
        (cambi.width == width && cambi.height == height) || !in_range(cambi.width) || !in_range(cambi.height)) {
        return std::nullopt;
    }
    return std::pair(cambi.width, cambi.height);
}

// CAMBI's options for its test video, in libvmaf's own order of their names
// (the order it names its score by).
ExtractorOptions cambi_options(const CambiInput& cambi, int width, int height, int bit_depth) {
    ExtractorOptions options;
    if (cambi.bit_depth > 0 && cambi.bit_depth < bit_depth) {
        options.emplace_back("enc_bitdepth", std::to_string(cambi.bit_depth));
    }
    if (const std::optional<std::pair<int, int>> size = cambi_encode_size(cambi, width, height)) {
        options.emplace_back("enc_height", std::to_string(size->second));
        options.emplace_back("enc_width", std::to_string(size->first));
    }
    if (cambi.pq) {
        options.emplace_back("eotf", "pq");
    }
    return options;
}

// The name libvmaf gives CAMBI's score with these options
// (feature_name.c's vmaf_feature_name_from_options): its alias "cambi", then
// each option's own short name and value, in the order of the options' names.
std::string cambi_feature(const ExtractorOptions& options) {
    if (options.empty()) {
        return std::string(cambi_default_feature);
    }
    static const std::map<std::string, std::string, std::less<>> short_names{
        {"enc_bitdepth", "encbd"}, {"enc_height", "ench"}, {"enc_width", "encw"}, {"eotf", "eotf"}};
    std::string name = "cambi";
    for (const auto& [option, value] : options) {
        name += "_" + short_names.at(option) + "_" + value;
    }
    return name;
}

// The extractors that score `metrics`, each started once; a plane of a
// metric (MetricDefinition::plane_of) starts its metric's. PSNR and SSIM read
// the chroma planes only for their 6:1:1 metric. With XPSNR, the luma's PSNR
// comes from the squared errors it sums anyway (its option "psnr"), not a
// pass of its own over both pictures (4K PSNR + SSIM + XPSNR 6% faster with
// Intel's decoder, 11% with NVIDIA's); PSNR then scores the chroma alone, for
// PSNR YUV. A run that has PSNR's or SSIM's Y saved asks for the U and V
// planes alone (JobRun's make_611 makes the 6:1:1 of them), and PSNR and
// SSIM then score the chroma alone.
std::vector<Extractor> extractors_for(const std::vector<std::string>& requested, int frame_rate,
                                      const ExtractorOptions& cambi) {
    std::vector<std::string> metrics;
    for (const std::string& metric : requested) {
        if (cpu_feature(metric) == nullptr) {
            throw std::invalid_argument(metric + " is not scored with libvmaf's CPU extractors");
        }
        metrics.emplace_back(scored_by(metric));
    }
    const auto asked = [&requested](std::initializer_list<std::string_view> keys) {
        return std::any_of(keys.begin(), keys.end(), [&requested](std::string_view key) {
            return std::find(requested.begin(), requested.end(), key) != requested.end();
        });
    };
    const bool psnr_yuv = has(metrics, "psnr_yuv") || has(metrics, "psnr_yuv_mse");  // the planes' PSNR
    const bool ssim_yuv = has(metrics, "ssim_yuv");
    const bool psnr_luma = asked({"psnr", "psnr_yuv", "psnr_yuv_mse", "psnr_yuv_y"});
    const bool ssim_luma = asked({"ssim", "ssim_yuv", "ssim_yuv_y"});
    const bool xpsnr = has(metrics, "xpsnr");
    std::vector<Extractor> extractors;
    if (psnr_yuv || (psnr_luma && !xpsnr)) {
        // Set both ways, as libvmaf's PSNR reads the chroma of a 4:2:0
        // picture by default.
        extractors.push_back({"psnr", {{"enable_chroma", psnr_yuv ? "true" : "false"},
                                       {"enable_luma", psnr_luma && !xpsnr ? "true" : "false"}}});
    }
    if (ssim_yuv || ssim_luma) {
        extractors.push_back({"float_ssim", {}});
        if (ssim_yuv) {
            extractors.back().options.emplace_back("enable_chroma", "true");
        }
        if (!ssim_luma) {
            extractors.back().options.emplace_back("enable_luma", "false");  // libvmaf-fast's
        }
    }
    if (xpsnr) {
        // Weighted by the source's activity (the extractor's default), as
        // FFmpeg's filter weights by its first input, the source in the app's
        // commands.
        extractors.push_back({"xpsnr", {{"frame_rate", std::to_string(frame_rate)}}});
        if (psnr_luma) {
            extractors.back().options.emplace_back("psnr", "true");
        }
    }
    if (has(metrics, "cambi")) {
        extractors.push_back({"cambi", cambi});
    }
    // MS-SSIM's planes share one set of decimated buffers, as SSIM's do.
    const bool ms_ssim_yuv = has(metrics, "ms_ssim_yuv");
    if (ms_ssim_yuv || has(metrics, "ms_ssim")) {
        extractors.push_back({"float_ms_ssim", {}});
        if (ms_ssim_yuv) {
            extractors.back().options.emplace_back("enable_chroma", "true");
        }
    }
    if (has(metrics, "psnr_hvs")) {
        extractors.push_back({"psnr_hvs", {}});
    }
    if (has(metrics, "ciede2000")) {
        extractors.push_back({"ciede", {}});
    }
    return extractors;
}

// A context of libvmaf's, closed however the check ends.
struct Context {
    const Libvmaf& library;
    VmafContext* context = nullptr;
    ~Context() {
        if (context != nullptr) {
            library.close(context);
        }
    }
};

bool scores_xpsnr_and_psnr() {
    const Libvmaf& library = libvmaf();
    Context made{library};
    // When vmaf_init fails it frees the context it made but leaves the
    // pointer to it set, so the context is only kept once it has started.
    VmafContext* context = nullptr;
    check_libvmaf(library.init(&context, VmafConfiguration{VMAF_LOG_LEVEL_ERROR, 0, 1, 0, 0}), "Starting libvmaf");
    made.context = context;
    VmafFeatureDictionary* options = nullptr;
    check_libvmaf(library.feature_dictionary_set(&options, "psnr", "true"), "Setting an option");
    if (library.use_feature(context, "xpsnr", options) != 0) {
        return false;
    }
    check_libvmaf(library.preallocate_pictures(context, VmafPictureConfiguration{{64, 64, 8, VMAF_PIX_FMT_YUV400P}, 2}),
                  "Allocating pictures");
    VmafPicture reference{};
    VmafPicture distorted{};
    check_libvmaf(library.fetch_preallocated_picture(context, &reference), "Taking a picture");
    if (const int error = library.fetch_preallocated_picture(context, &distorted); error != 0) {
        library.picture_unref(&reference);  // a picture never given back would keep vmaf_close waiting for it
        check_libvmaf(error, "Taking a picture");
    }
    // libvmaf takes both pictures, even when it fails.
    check_libvmaf(library.read_pictures(context, &reference, &distorted, 0), "Scoring");
    check_libvmaf(library.read_pictures(context, nullptr, nullptr, 0), "Finishing");
    double value = 0;
    return library.feature_score_at_index(context, "xpsnr_y", &value, 0) == 0 &&
           library.feature_score_at_index(context, "psnr_y", &value, 0) == 0;
}

}  // namespace

const CpuFeature* cpu_feature(std::string_view key) {
    const auto found = std::find_if(cpu_features.begin(), cpu_features.end(),
                                    [&](const CpuFeature& feature) { return feature.key == key; });
    return found == cpu_features.end() ? nullptr : &*found;
}

bool keeps_exact(std::string_view key) {
    const std::string_view metric = scored_by(key);
    return key == "psnr" || key == "ssim" || (key != metric && (metric == "psnr_yuv" || metric == "ssim_yuv"));
}

bool made_of_exact_planes(std::string_view key) {
    return key == "psnr_yuv" || key == "psnr_yuv_mse" || key == "ssim_yuv";
}

double yuv_score(std::string_view key, double y, double u, double v) {
    const CpuFeature* feature = cpu_feature(key);
    if (feature == nullptr || !made_of_exact_planes(key)) {
        throw std::invalid_argument(std::string(key) + " is not made of its planes' scores");
    }
    return python_round(combined(*feature, y, u, v), feature->decimals);
}

bool cpu_metrics_read_chroma(const std::vector<std::string>& metrics) {
    return std::any_of(metrics.begin(), metrics.end(), [](const std::string& metric) {
        const CpuFeature* feature = cpu_feature(metric);
        return feature != nullptr && feature->chroma;
    });
}

CpuMetricsCost cpu_metrics_cost(const std::vector<std::string>& metrics) {
    CpuMetricsCost cost = CpuMetricsCost::light;
    for (const std::string& key : metrics) {
        const std::string_view metric = scored_by(key);
        if (metric == "ms_ssim" || metric == "ms_ssim_yuv" || metric == "cambi" || metric == "ciede2000") {
            return CpuMetricsCost::heavy;
        }
        if (metric == "psnr_hvs") {
            cost = CpuMetricsCost::medium;
        }
    }
    return cost;
}

std::map<std::string, std::string> cpu_metrics_unfit(const std::vector<std::string>& metrics, int width,
                                                     int height, int bit_depth, const CambiInput& cambi) {
    const std::string size = std::to_string(width) + "x" + std::to_string(height);
    const auto square = [](int side) { return std::to_string(side) + "x" + std::to_string(side); };
    std::map<std::string, std::string> unfit;
    for (const std::string& key : metrics) {
        const std::string_view metric = scored_by(key);  // a plane is unfit where its metric is
        if ((metric == "ms_ssim" && std::min(width, height) < ms_ssim_smallest_plane) ||
            (metric == "ms_ssim_yuv" && std::min(width, height) < 2 * ms_ssim_smallest_plane)) {
            const int side = metric == "ms_ssim" ? ms_ssim_smallest_plane : 2 * ms_ssim_smallest_plane;
            unfit[key] = (metric == "ms_ssim" ? "MS-SSIM" : "MS-SSIM YUV") + std::string(" needs pictures of at least ") +
                         square(side) + "; these are " + size + ".";
        } else if (metric == "cambi") {
            const std::pair<int, int> scored = cambi_encode_size(cambi, width, height).value_or(std::pair(width, height));
            if (std::max(width, height) < cambi_smallest_side ||
                std::max(scored.first, scored.second) < cambi_smallest_side) {
                unfit[key] = "CAMBI needs a video at least " + std::to_string(cambi_smallest_side) +
                             " pixels wide or high.";
            } else if (width + height > cambi_largest_width_plus_height) {
                unfit[key] = "CAMBI is calculated for pictures up to 3840x2160 (width plus height at most " +
                             std::to_string(cambi_largest_width_plus_height) + "); these are " + size + ".";
            }
        } else if (metric == "psnr_hvs" && std::min(width, height) < psnr_hvs_smallest_side) {
            unfit[key] = "PSNR-HVS-M needs pictures of at least " + square(psnr_hvs_smallest_side) +
                         "; these are " + size + ".";
        } else if (metric == "psnr_hvs" && bit_depth > psnr_hvs_deepest) {
            unfit[key] = "PSNR-HVS-M is calculated up to " + std::to_string(psnr_hvs_deepest) +
                         " bits; these videos are compared at " + std::to_string(bit_depth) + ".";
        } else if (is_chroma_vmaf(key) && std::min(width / 2, height / 2) < vmaf_smallest_plane) {
            unfit[key] = "VMAF YUV needs pictures of at least " + square(2 * vmaf_smallest_plane) +
                         " (chroma planes of " + square(vmaf_smallest_plane) + "); these are " + size + ".";
        } else if (metric == "ciede2000" && bit_depth != 8 && bit_depth != 10 && bit_depth != 12 && bit_depth != 16) {
            unfit[key] = "CIEDE2000 is calculated at 8, 10, 12 or 16 bits; these videos are compared at " +
                         std::to_string(bit_depth) + ".";
        }
    }
    return unfit;
}

std::string cpu_library_build() { return "libvmaf-fast " + libvmaf_fast_version() + " (CPU)"; }

CpuScorer::CpuScorer(int width, int height, int bit_depth, std::vector<std::string> metrics, int n_subsample,
                     int threads, int frame_rate, const CambiInput& cambi, std::optional<int> gpu,
                     const std::string& vmaf_model, bool vmaf_on_vulkan,
                     std::optional<ChromaFromDecoders> chroma_from_decoders, GpuFrameStream* shared_decoder)
    : library_(libvmaf()), metrics_(std::move(metrics)), step_(static_cast<unsigned>(std::max(1, n_subsample))) {
    threads = std::max(0, threads);
    // VMAF YUV's U and V first: whether the pictures need their planes.
    std::vector<std::string> chroma_vmaf;
    std::vector<std::string> others;
    std::partition_copy(metrics_.begin(), metrics_.end(), std::back_inserter(chroma_vmaf), std::back_inserter(others),
                        [](const std::string& key) { return is_chroma_vmaf(key); });
    if (!chroma_vmaf.empty()) {
        chroma_vmaf_ = std::make_unique<ChromaVmaf>(width, height, bit_depth, vmaf_model, chroma_vmaf, n_subsample,
                                                    threads, vmaf_on_vulkan ? gpu : std::nullopt,
                                                    chroma_from_decoders);
    }
    // The GPU's CIEDE2000 reads the chroma too.
    chroma_ = cpu_metrics_read_chroma(others) || (chroma_vmaf_ && !chroma_vmaf_->takes_decoded());
    odd_ = width % 2 != 0 || height % 2 != 0;
    sample_ = bytes_per_sample(bit_depth);
    packed_ = PackedFrame(width, height, bit_depth);
    frame_bytes_ = packed_.bytes();
    decoded_bytes_ = chroma_ ? frame_bytes_ : static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * sample_;
    const ExtractorOptions cambi_with = cambi_options(cambi, width, height, bit_depth);
    start_gpu(width, height, bit_depth, gpu, cambi_with, shared_decoder);
    std::erase_if(libvmaf_metrics_, [](const std::string& key) { return is_chroma_vmaf(key); });
    const CpuThreads split = cpu_threads(width, height, bit_depth, threads, libvmaf_metrics_, chroma_);
    const int pairs = split.pairs;
    const VmafConfiguration configuration{VMAF_LOG_LEVEL_ERROR, split.frame_threads, step_, 0, 0};
    VmafContext* context = nullptr;
    check_libvmaf(library_.init(&context, configuration), "Starting libvmaf");
    context_ = context;
    try {
        check_libvmaf(library_.set_helper_threads(split.helpers), "Starting libvmaf's helper threads");
        cambi_feature_ = cambi_feature(cambi_with);
        for (const Extractor& extractor : extractors_for(libvmaf_metrics_, frame_rate, cambi_with)) {
            VmafFeatureDictionary* options = nullptr;
            for (const auto& [key, value] : extractor.options) {
                if (library_.feature_dictionary_set(&options, key.c_str(), value.c_str()) != 0) {
                    library_.feature_dictionary_free(&options);
                    throw LibvmafError("Setting " + extractor.name + "'s " + key + " failed");
                }
            }
            // libvmaf frees the options (if it refuses them early, a few bytes leak).
            check_libvmaf(library_.use_feature(context_, extractor.name.c_str(), options),
                          "Starting " + extractor.name);
        }
        const VmafPictureConfiguration pool{
            {static_cast<unsigned>(width), static_cast<unsigned>(height), static_cast<unsigned>(bit_depth),
             chroma_ ? VMAF_PIX_FMT_YUV420P : VMAF_PIX_FMT_YUV400P},
            static_cast<unsigned>(2 * pairs)};
        check_libvmaf(library_.preallocate_pictures(context_, pool), "Allocating pictures");
    } catch (...) {
        release();
        throw;
    }
}

CpuScorer::~CpuScorer() { release(); }

std::vector<std::string> CpuScorer::gpu_metrics() const {
    std::vector<std::string> on_gpu = gpu_ ? gpu_->metrics() : std::vector<std::string>{};
    if (chroma_vmaf_) {
        const std::vector<std::string> chroma = chroma_vmaf_->keys_on_gpu();
        on_gpu.insert(on_gpu.end(), chroma.begin(), chroma.end());
    }
    return on_gpu;
}

void CpuScorer::start_gpu(int width, int height, int bit_depth, std::optional<int> gpu,
                          const ExtractorOptions& cambi_with, GpuFrameStream* shared_decoder) {
    libvmaf_metrics_ = metrics_;
    const std::vector<std::string> wanted = gpu ? gpu_metrics_of(metrics_, odd_, cambi_with) : std::vector<std::string>{};
    if (wanted.empty()) {
        return;
    }
    // The decoders' pictures on the GPU only where nothing but the GPU reads the pairs: no picture is downloaded.
    const bool gpu_alone = !chroma_ && !chroma_vmaf_ && wanted.size() == metrics_.size();
    try {
        // It takes the decoders' pictures only where it starts every metric: none is left to libvmaf, which would
        // read the pictures.
        gpu_ = std::make_unique<GpuMetricsScorer>(width, height, bit_depth, *gpu, wanted, cambi_with,
                                                  gpu_alone ? shared_decoder : nullptr);
    } catch (const LibvmafError& error) {  // VmafVulkanError among them
        logger.warning(metric_labels(wanted, " and ") + " are calculated on the CPU: " + error.what());
        return;
    }
    for (const std::string& key : gpu_->metrics()) {
        std::erase(libvmaf_metrics_, key);
    }
    logger.info(metric_labels(gpu_->metrics(), " and ") + " on the GPU");
}

void CpuScorer::release() {
    // The page-locked memory is unregistered first, because vmaf_close frees
    // the pictures, and memory must be unregistered from CUDA before it is
    // freed. No copy into it can be under way (a download returns once its
    // copy is done), and an extractor still reading it is not affected.
    for (const auto& [address, stream] : pinned_) {
        if (stream != nullptr) {
            stream->unpin(address);
        }
    }
    pinned_.clear();
    if (context_ != nullptr) {
        library_.close(context_);
        context_ = nullptr;
    }
    // The helper threads end with their scratch (a scorer in the app's own
    // process; an isolated one's process ends anyway).
    library_.set_helper_threads(0);
}

VmafPicture CpuScorer::take_picture() {
    VmafPicture picture{};
    check_libvmaf(library_.fetch_preallocated_picture(context_, &picture), "Taking a picture");
    return picture;
}

void CpuScorer::give_back(VmafPicture& reference, VmafPicture& distorted) {
    // Pictures taken from the pool must be given back: vmaf_close would wait
    // for them.
    library_.picture_unref(&reference);
    library_.picture_unref(&distorted);
}

void CpuScorer::score(VmafPicture& reference, VmafPicture& distorted) {
    // Every pair: VMAF's motion reads the frame before. From the decoders' memory, add_decoded gave them.
    if (chroma_vmaf_ && !chroma_vmaf_->takes_decoded()) {
        try {
            chroma_vmaf_->add(reference, distorted);  // copies the planes before it returns
        } catch (const std::exception& error) {
            give_back(reference, distorted);
            if (const std::vector<std::string> on_gpu = chroma_vmaf_->keys_on_gpu(); !on_gpu.empty()) {
                throw GpuMetricsFailed(metric_labels(on_gpu, " and ") + " on the GPU failed: " + error.what());
            }
            throw;
        } catch (...) {
            give_back(reference, distorted);
            throw;
        }
    }
    if (gpu_ && count_ % step_ == 0) {  // only the pairs scored go to the GPU
        try {
            gpu_->add(reference, distorted);  // copies the planes before it returns
        } catch (const std::exception& error) {
            give_back(reference, distorted);
            throw gpu_failed(*gpu_, error);
        } catch (...) {
            give_back(reference, distorted);
            throw;
        }
    }
    if (libvmaf_metrics_.empty()) {
        give_back(reference, distorted);
    } else {
        // libvmaf takes both pictures, even when it fails (pull request 1652).
        check_libvmaf(library_.read_pictures(context_, &reference, &distorted, count_),
                      "Scoring frame " + std::to_string(count_));
    }
    ++count_;
}

void CpuScorer::fill(VmafPicture& picture, const FramePlanes& frame) const {
    if (chroma_) {
        fill_picture(picture, frame, sample_, {0, 1, 2});
    } else {
        fill_picture(picture, frame, sample_, {0});
    }
}

bool CpuScorer::laid_out_packed(const VmafPicture& picture) const {
    const auto row_bytes = [&](int plane) { return static_cast<std::ptrdiff_t>(picture.w[plane] * sample_); };
    if (picture.stride[0] != row_bytes(0)) {
        return false;
    }
    if (!chroma_) {
        return true;
    }
    // libvmaf's planes follow one another (vmaf_picture_alloc); a packed
    // frame's chroma rows are half the width rounded up, libvmaf's down.
    const auto* y = static_cast<const std::uint8_t*>(picture.data[0]);
    const auto* u = static_cast<const std::uint8_t*>(picture.data[1]);
    const auto* v = static_cast<const std::uint8_t*>(picture.data[2]);
    return !odd_ && picture.stride[1] == row_bytes(1) && picture.stride[2] == row_bytes(2) &&
           u == y + picture.stride[0] * picture.h[0] && v == u + picture.stride[1] * picture.h[1];
}

void CpuScorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    if (std::min(reference.size(), distorted.size()) < frame_bytes_) {
        throw LibvmafError("frame " + std::to_string(count_) + " is shorter than a picture");
    }
    add_planes(packed_.planes(reference.data()), packed_.planes(distorted.data()));
}

void CpuScorer::add_planes(const FramePlanes& reference, const FramePlanes& distorted) {
    if (chroma_vmaf_ && chroma_vmaf_->takes_decoded()) {
        throw std::logic_error("VMAF YUV's planes are copied from the decoders: the pairs go to add_decoded");
    }
    if (luma_to_gpu_alone()) {
        // The engine copies the two luma planes from where they are, as it would from libvmaf's picture.
        if (count_ % step_ == 0) {
            try {
                gpu_->add_luma(reference[0], distorted[0]);
            } catch (const std::exception& error) {
                throw gpu_failed(*gpu_, error);
            }
        }
        ++count_;
        return;
    }
    VmafPicture reference_picture = take_picture();
    VmafPicture distorted_picture{};
    try {
        distorted_picture = take_picture();
    } catch (...) {
        library_.picture_unref(&reference_picture);
        throw;
    }
    try {
        // The two at the same time, the source's on the side thread.
        side_.both([&] { fill(reference_picture, reference); }, [&] { fill(distorted_picture, distorted); });
    } catch (...) {
        give_back(reference_picture, distorted_picture);
        throw;
    }
    score(reference_picture, distorted_picture);
}

void CpuScorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
    add_decoded(reference, reference_slot, test, test_slot, nullptr);
}

void CpuScorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot,
                            const std::function<void(Plane, Plane)>& also) {
    if (gpu_ && gpu_->takes_shared()) {
        if (also) {
            throw std::logic_error("the luma planes for another scorer from a GPU scorer that downloads none");
        }
        add_decoded_luma(reference, reference_slot, test, test_slot);
        return;
    }
    if (chroma_vmaf_ && chroma_vmaf_->takes_decoded()) {
        try {
            chroma_vmaf_->add_decoded(reference, reference_slot, test, test_slot);
        } catch (const GpuDecodeFailedError&) {
            throw;  // the decoder's, which the run answers as it does any (not a GPU metric's)
        } catch (const std::exception& error) {
            throw GpuMetricsFailed(metric_labels(chroma_vmaf_->keys_on_gpu(), " and ") +
                                   " on the GPU failed: " + error.what());
        }
        if (libvmaf_metrics_.empty() && !gpu_ && !also) {
            ++count_;  // nothing else reads the pair: no picture downloaded
            return;
        }
    }
    if (!also && luma_to_gpu_alone() && !gpu_decoder(reference.backend()).cuda_copies &&
        !gpu_decoder(test.backend()).cuda_copies) {
        add_decoded_luma(reference, reference_slot, test, test_slot);
        return;
    }
    VmafPicture reference_picture = take_picture();
    VmafPicture distorted_picture{};
    try {
        distorted_picture = take_picture();
    } catch (...) {
        library_.picture_unref(&reference_picture);
        throw;
    }
    try {
        if (!also && reads_only_test_picture()) {
            // At 4K the source's download was about half of the time this
            // thread spent on a pair, for a picture nothing reads.
            download(distorted_picture, test, test_slot, 1);
        } else {
            download_both(reference_picture, reference, reference_slot, distorted_picture, test, test_slot);
        }
        if (also) {
            const auto plane = [](const VmafPicture& picture) {
                return Plane{static_cast<const std::uint8_t*>(picture.data[0]),
                             static_cast<std::ptrdiff_t>(picture.stride[0])};
            };
            also(plane(reference_picture), plane(distorted_picture));
        }
    } catch (...) {
        give_back(reference_picture, distorted_picture);
        throw;
    }
    score(reference_picture, distorted_picture);
}

bool CpuScorer::luma_to_gpu_alone() const {
    return gpu_ && gpu_->takes_luma_alone() && libvmaf_metrics_.empty() && !chroma_vmaf_ && !chroma_;
}

void CpuScorer::add_decoded_luma(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test,
                                 int test_slot) {
    if (count_ % step_ != 0) {
        ++count_;  // a pair not scored: nothing reads it
        return;
    }
    try {
        if (gpu_->takes_shared()) {
            // The decoders copy them on the GPU into the engine's memory.
            gpu_->add_shared(reference, reference_slot, test, test_slot);
        } else if (reference.gives_planes() && test.gives_planes()) {
            // The engine copies both from where the software decoder holds them.
            gpu_->add_luma(reference.slot_planes(reference_slot)[0], test.slot_planes(test_slot)[0]);
        } else {
            // Each luma plane written once, into the engine's memory, by the decoders in download_both's order.
            const Plane packed = packed_.luma(nullptr);
            const auto row_bytes = static_cast<std::size_t>(packed.stride);
            const auto write = [&](GpuFrameStream& stream, int slot, std::uint8_t* plane) {
                if (stream.gives_planes()) {
                    copy_strided_rows(stream.slot_planes(slot)[0], row_bytes, decoded_bytes_ / row_bytes, plane);
                } else {
                    stream.download(slot, plane);
                }
            };
            gpu_->add_decoded([&](std::uint8_t* reference_plane, std::uint8_t* test_plane) {
                take_both(
                    reference, test, [&] { write(reference, reference_slot, reference_plane); },
                    [&] { write(test, test_slot, test_plane); });
            });
        }
    } catch (const GpuDecodeFailedError&) {
        throw;  // the decoder's, which the run answers as it does any (not a GPU metric's)
    } catch (const std::exception& error) {
        throw gpu_failed(*gpu_, error);
    }
    ++count_;
}

bool CpuScorer::reads_only_test_picture() const {
    // VMAF YUV's chroma scorer reads the source's planes from the picture
    // unless it took them from the decoders itself.
    const bool chroma_vmaf_reads = chroma_vmaf_ && !chroma_vmaf_->takes_decoded();
    return gpu_ && !gpu_->reads_reference() && libvmaf_metrics_.empty() && !chroma_vmaf_reads;
}

void CpuScorer::download_both(VmafPicture& reference_picture, GpuFrameStream& reference, int reference_slot,
                              VmafPicture& distorted_picture, GpuFrameStream& test, int test_slot) {
    take_both(
        reference, test, [&] { download(reference_picture, reference, reference_slot, 0); },
        [&] { download(distorted_picture, test, test_slot, 1); });
}

void CpuScorer::take_both(const GpuFrameStream& reference, const GpuFrameStream& test,
                          const std::function<void()>& take_reference, const std::function<void()>& take_test) {
    if (downloads_at_once(reference.backend(), test.backend())) {
        // The two decoders write their pictures at the same time, the
        // source's on the side thread: one after the other they took about
        // 3 ms each per 4K 10-bit pair, on the one thread that feeds the
        // scorers.
        side_.both(take_reference, take_test);
    } else if (reference.gives_planes() || test.gives_planes()) {
        // A picture copied from where the software decoder holds it calls
        // only on that decoder, so it is copied on the side thread beside the
        // other decoder's download, which stays on this thread, as it always
        // was (Intel's and AMD's are not known to download from another).
        if (reference.gives_planes()) {
            side_.both(take_reference, take_test);
        } else {
            side_.both(take_test, take_reference);
        }
    } else {
        take_reference();
        take_test();
    }
}

void CpuScorer::download(VmafPicture& picture, GpuFrameStream& stream, int slot, std::size_t which) {
    void* const address = picture.data[0];
    // NVIDIA's decoder writes an odd size's chroma rows and columns rounded
    // up, one more than libvmaf's planes hold.
    if (gpu_decoder(stream.backend()).cuda_copies && !(chroma_ && odd_)) {
        {
            const std::lock_guard lock(pinned_mutex_);  // the two videos' downloads run at once
            if (!pinned_.contains(address)) {
                // The picture's planes, one after the other (vmaf_picture_alloc).
                const std::uint64_t bytes = static_cast<std::uint64_t>(picture.stride[0]) * picture.h[0] +
                                            (chroma_ ? 2 * static_cast<std::uint64_t>(picture.stride[1]) * picture.h[1]
                                                     : 0);
                pinned_[address] = stream.pin(address, bytes) ? &stream : nullptr;
            }
        }
        if (chroma_) {
            stream.download_planes(slot, {address, picture.data[1], picture.data[2]},
                                   {picture.stride[0], picture.stride[1], picture.stride[2]});
        } else {
            stream.download_planes(slot, {address, nullptr, nullptr}, {picture.stride[0], 0, 0});
        }
    } else if (stream.gives_planes()) {
        fill(picture, stream.slot_planes(slot));  // one copy, whatever the picture's row spacings
    } else if (laid_out_packed(picture)) {
        stream.download(slot, address);
    } else {
        std::vector<std::uint8_t>& staging = staging_[which];
        staging.resize(decoded_bytes_);
        stream.download(slot, staging.data());
        fill(picture, packed_.planes(staging.data()));
    }
}

PairScores CpuScorer::finish() {
    PairScores result;
    if (count_ == 0) {
        for (const std::string& metric : metrics_) {
            result.by_metric.emplace_back(metric, std::vector<double>());
        }
        return result;
    }
    if (!libvmaf_metrics_.empty()) {
        check_libvmaf(library_.read_pictures(context_, nullptr, nullptr, 0), "Finishing");
    }
    // The GPU's values, one for each pair scored (every step_-th frame).
    std::map<std::string, std::vector<double>> on_gpu;
    if (gpu_) {
        try {
            on_gpu = gpu_->finish();
        } catch (const std::exception& error) {
            throw gpu_failed(*gpu_, error);
        }
    }
    // VMAF YUV's U and V, wherever they were scored: one value for each pair scored, as the GPU's.
    std::map<std::string, std::vector<double>> chroma_vmaf;
    if (chroma_vmaf_) {
        const std::vector<std::string> chroma_on_gpu = chroma_vmaf_->keys_on_gpu();
        try {
            chroma_vmaf = chroma_vmaf_->finish();
        } catch (const std::exception& error) {
            if (chroma_on_gpu.empty()) {
                throw;
            }
            throw GpuMetricsFailed(metric_labels(chroma_on_gpu, " and ") + " on the GPU failed: " + error.what());
        }
        result.features_on_gpu.insert(result.features_on_gpu.end(), chroma_on_gpu.begin(), chroma_on_gpu.end());
    }
    for (unsigned frame = 0; frame < count_; frame += step_) {
        result.frames.push_back(frame);
    }
    for (const std::string& metric : metrics_) {
        const CpuFeature& feature = *cpu_feature(metric);
        if (feature.value == Value::chroma_vmaf) {
            std::vector<double>& column = chroma_vmaf.at(metric);
            if (column.size() != result.frames.size()) {
                throw LibvmafError(metric + " has " + std::to_string(column.size()) + " scores for " +
                                   std::to_string(result.frames.size()) + " frames");
            }
            result.by_metric.emplace_back(metric, std::move(column));  // to 6 decimals already
            continue;
        }
        const bool metric_on_gpu = gpu_ && has(gpu_->metrics(), metric);
        if (metric_on_gpu) {
            result.features_on_gpu.push_back(metric);
        }
        const auto read = [&](std::string_view feature_name, unsigned frame) {
            if (metric_on_gpu) {
                // The GPU's columns by libvmaf's feature names, CAMBI's as "cambi" (libvmaf's name comes from its
                // options: cambi_feature).
                return on_gpu.at(metric == "cambi" ? metric : std::string(feature_name)).at(frame / step_);
            }
            const std::string name(feature_name);
            double value = 0;
            check_libvmaf(library_.feature_score_at_index(context_, name.c_str(), &value, frame),
                          "Reading " + name + " for frame " + std::to_string(frame));
            return value;
        };
        std::vector<double> column;
        column.reserve(result.frames.size());
        const bool exact = keeps_exact(metric) && !metric_on_gpu;
        std::vector<double> exact_column;
        exact_column.reserve(exact ? result.frames.size() : 0);
        const bool of_planes = feature.value == Value::planes_611 || feature.value == Value::planes_611_mse;
        const std::string_view name = metric == "cambi" ? std::string_view(cambi_feature_) : feature.feature;
        for (const unsigned frame : result.frames) {
            // at full precision, then rounded once
            const double value = combined(feature, read(name, frame), of_planes ? read(feature.cb, frame) : 0.0,
                                          of_planes ? read(feature.cr, frame) : 0.0);
            column.push_back(python_round(value, feature.decimals));
            if (exact) {
                exact_column.push_back(value);
            }
        }
        result.by_metric.emplace_back(metric, std::move(column));
        if (exact) {
            result.exact.emplace(metric, std::move(exact_column));
        }
    }
    return result;
}

bool cpu_scores_xpsnr() {
    static const bool answer = [] {
        try {
            return scores_xpsnr_and_psnr();
        } catch (const LibvmafError&) {
            return false;
        }
    }();
    return answer;
}

int cpu_pairs(int width, int height, int bit_depth, int threads, bool xpsnr, bool chroma, CpuMetricsCost cost) {
    // A picture's bytes as vmaf_picture_alloc lays it out: rows of 32
    // samples, and with chroma two planes half as wide and tall.
    const auto plane = [&](int w, int h) {
        return static_cast<std::size_t>((w + 31) & ~31) * static_cast<std::size_t>(h) * bytes_per_sample(bit_depth);
    };
    const std::size_t picture = plane(width, height) + (chroma ? 2 * plane(width / 2, height / 2) : 0);
    threads = std::max(1, threads);
    std::size_t luma_memory = xpsnr ? xpsnr_picture_memory : picture_memory;
    int busy = std::min(8, threads);
    if (cost == CpuMetricsCost::medium) {
        luma_memory = medium_picture_memory;
        busy = std::min(medium_threads, threads);
    } else if (cost == CpuMetricsCost::heavy) {
        luma_memory = heavy_picture_memory;
        busy = threads;
    }
    const std::size_t memory = chroma ? luma_memory * 3 / 2 : luma_memory;
    const auto fit = static_cast<long long>(memory / (2 * picture));
    const long long wanted = busy + 1;
    return static_cast<int>(std::max(2LL, std::min(wanted, fit))) + (xpsnr ? 1 : 0);
}

CpuThreads cpu_threads(int width, int height, int bit_depth, int threads, const std::vector<std::string>& metrics,
                       bool chroma) {
    threads = std::max(0, threads);
    // With libvmaf calculating nothing, each pair is given back as soon as the
    // GPU has copied it: two pairs are all the decoders need.
    if (metrics.empty()) {
        return {2, static_cast<unsigned>(std::min(threads, 2)), 0};
    }
    const auto is_ms_ssim = [](const std::string& key) {
        const std::string_view metric = scored_by(key);
        return metric == "ms_ssim" || metric == "ms_ssim_yuv";
    };
    // MS-SSIM alone: each pair gets up to three threads (its frame thread and
    // two helpers), the rest go to more pairs. Each frame thread keeps its own
    // MS-SSIM buffers and a pair of pictures, so a third of the frame threads
    // take about half the memory, at a cost in speed: 4K 10-bit on a Core
    // Ultra 9 285K (24 threads), 24 frame threads 72 fps and 1678 MB at the
    // peak, 8 frame threads + 16 helpers 58-61 fps and 834 MB (1080p 8-bit:
    // 369-376 fps and 458 MB, 362-368 fps and 310 MB). With any other metric,
    // fewer frame threads would slow that one, which is not split.
    const bool ms_ssim_alone = std::all_of(metrics.begin(), metrics.end(), is_ms_ssim);
    const int busy = ms_ssim_alone ? (threads + ms_ssim_threads_per_pair - 1) / ms_ssim_threads_per_pair : threads;
    const int pairs = cpu_pairs(width, height, bit_depth, busy, has(metrics, "xpsnr"), chroma,
                                cpu_metrics_cost(metrics));
    // No more frame threads than pairs: a pair is scored by one thread, and
    // each thread that ever scores one keeps its own copy of every
    // extractor's buffers (MS-SSIM's and CAMBI's: about 100 MB each at 4K).
    const int frame_threads = std::min(busy, pairs);
    // The threads left over share each pair's MS-SSIM (with other metrics,
    // only those the pairs' memory limit leaves idle, at 8K).
    const bool ms_ssim = std::any_of(metrics.begin(), metrics.end(), is_ms_ssim);
    const int helpers = ms_ssim ? threads - frame_threads : 0;
    return {pairs, static_cast<unsigned>(frame_threads), static_cast<unsigned>(helpers)};
}

void fill_picture(VmafPicture& picture, const FramePlanes& frame, std::size_t sample,
                  std::initializer_list<int> planes) {
    for (const int plane : planes) {
        const Plane& from = frame[static_cast<std::size_t>(plane)];
        const std::size_t row_bytes = picture.w[plane] * sample;
        auto* target = static_cast<std::uint8_t*>(picture.data[plane]);
        const auto stride = static_cast<std::size_t>(picture.stride[plane]);
        for (std::size_t row = 0; row < picture.h[plane]; ++row) {
            std::memcpy(target + row * stride, from.data + static_cast<std::ptrdiff_t>(row) * from.stride, row_bytes);
        }
    }
}

PairScores score_decoded_cpu(const VideoInfo& source, const VideoInfo& distorted,
                             const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                             int width, int height, int bit_depth, const std::vector<std::string>& metrics,
                             int n_subsample, const FrameRange& range, long long total_frames,
                             const std::pair<std::string, std::string>& decoders, int threads, int frame_rate,
                             const CambiInput& cambi, std::optional<int> gpu, const std::string& vmaf_model,
                             bool vmaf_on_vulkan, const std::string& scale_algorithm, const std::string& ffmpeg,
                             const std::string& ffprobe, bool want_vvdec,
                             const std::function<void()>& on_vvdec_opening,
                             const GpuMetricsReport& on_gpu_metrics,
                             const std::function<void(long long, long long, double)>& on_progress,
                             const CancelToken& cancel) {
    const auto& [source_backend, test_backend] = decoders;
    require_same_range(source, distorted);
    const std::pair size(width, height);
    std::vector<DecodePlan> plans = plan_pair(source, distorted, source_crop, distorted_crop, size, bit_depth,
                                              !cpu_metrics_read_chroma(metrics), scale_algorithm, decoders);
    require_frames_of_size(plans, size, bit_depth);
    // A VVC video decoded by vvdec-vulkan where wanted, on the GPU it picks (this pass has none of its own)
    const bool test_vvdec = vvdec_chosen(plans[0], want_vvdec);
    const bool reference_vvdec = vvdec_chosen(plans[1], want_vvdec);
    if ((test_vvdec || reference_vvdec) && on_vvdec_opening) {
        on_vvdec_opening();
    }
    plan_pair_seek(plans, distorted, source, range.decode_start, ffmpeg, ffprobe, cancel);
    GpuFrameStream test = open_pass_stream(distorted, plans[0], 4, test_backend, test_vvdec, -1, ffmpeg, ffprobe);
    GpuFrameStream reference =
        open_pass_stream(source, plans[1], 4, source_backend, reference_vvdec, -1, ffmpeg, ffprobe);
    // Where the two decoders share GPU memory, the GPU's metrics may take their pictures on the GPU (CpuScorer's
    // `shared_decoder`): vmaf_v1_gpu.cpp's may_share, whose backend term always holds here, as the GPU's metrics
    // are the Vulkan engine's. Only NVIDIA's decoders hand their pictures over in this pass (open_pass_stream
    // asks no other to), so AMD's and Intel's pairs go through system memory, as do vvdec-vulkan's (HostPairs).
    const bool may_share = gpu && !test_vvdec && !reference_vvdec && test.shares_gpu_memory_with(reference);
    // Declared after the streams so that it goes first: it gives back the
    // memory their decoders page-locked, and the GPU memory they imported.
    CpuScorer scorer(width, height, bit_depth, metrics, n_subsample, threads, frame_rate, cambi, gpu, vmaf_model,
                     vmaf_on_vulkan, std::nullopt, may_share ? &test : nullptr);
    if (on_gpu_metrics) {
        on_gpu_metrics(scorer.metric_keys(), scorer.gpu_metrics());
    }
    // vvdec-vulkan's GPU copies its pictures only into a buffer of its stream's, from which they are copied into the
    // scorer's pictures. Last, so that it goes first: its side thread's jobs use the scorer.
    std::optional<HostPairs> pairs;
    if (test_vvdec || reference_vvdec) {
        pairs.emplace(reference, plans[1], test, plans[0]);
    }
    test.start();
    reference.start();
    score_frame_pairs(
        test, reference, range, total_frames,
        [&](int test_slot, int reference_slot) {
            if (pairs) {
                pairs->score(reference_slot, test_slot, scorer, nullptr);
            } else {
                scorer.add_decoded(reference, reference_slot, test, test_slot);
            }
        },
        on_progress, cancel);
    return scorer.finish();
}

}  // namespace vml
