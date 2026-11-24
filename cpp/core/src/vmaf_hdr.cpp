#include "vml/vmaf_hdr.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <future>
#include <stdexcept>
#include <thread>
#include <utility>

#include "vml/app_log.h"
#include "vml/colour.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/host_frames.h"
#include "vml/libvmaf_cpu.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/threads.h"
#include "vml/vmaf_v1_gpu.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.vmaf_hdr");

// libvmaf's names for the colours a model's conversion target can say, which
// are FFmpeg's names for the same tags.
template <typename Value>
using ColourNames = std::initializer_list<std::pair<std::string_view, Value>>;

constexpr ColourNames<VmafColorRange> range_names{{"limited", VMAF_COLOR_RANGE_LIMITED},
                                                  {"full", VMAF_COLOR_RANGE_FULL}};
constexpr ColourNames<VmafColorPrimaries> primaries_names{{"bt709", VMAF_COLOR_PRIMARIES_BT709},
                                                          {"bt2020", VMAF_COLOR_PRIMARIES_BT2020},
                                                          {"smpte432", VMAF_COLOR_PRIMARIES_SMPTE432}};
constexpr ColourNames<VmafColorTransferCharacteristic> transfer_names{{"bt709", VMAF_COLOR_TRC_BT709},
                                                                      {"smpte2084", VMAF_COLOR_TRC_SMPTE2084}};
constexpr ColourNames<VmafColorMatrixCoefficients> matrix_names{{"bt709", VMAF_COLOR_MATRIX_BT709},
                                                                {"bt2020nc", VMAF_COLOR_MATRIX_BT2020_NCL},
                                                                {"ictcp", VMAF_COLOR_MATRIX_ICTCP}};

// The value `names` gives `name`, or nothing.
template <typename Value>
std::optional<Value> named(ColourNames<Value> names, std::string_view name) {
    for (const auto& [listed, value] : names) {
        if (listed == name) {
            return value;
        }
    }
    return std::nullopt;
}

// The value of a model's colour field, or LibvmafError.
template <typename Value>
Value model_colour(ColourNames<Value> names, const nlohmann::json& colourspace, const char* field) {
    const std::string name = colourspace.at(field).get<std::string>();
    if (const std::optional<Value> value = named(names, name)) {
        return *value;
    }
    throw LibvmafError(std::string("the model's ") + field + " is not one the app reads: " + name);
}

// A video's tag as HDR VMAF reads it: `untagged` for no tag, its value from
// `names`, or nothing for one HDR VMAF does not convert from.
template <typename Value>
std::optional<Value> tag_value(ColourNames<Value> names, const std::string& tag, Value untagged) {
    if (is_untagged_tag(tag)) {
        return untagged;
    }
    return named(names, ascii_lower(tag));
}

// The colours of one video of a pair whose range is `full_range`, or why
// HDR VMAF cannot score it ("the source" or "the test video": `which`).
std::variant<VmafColor, std::string> video_colours(const VideoInfo& video, bool full_range, const char* which) {
    if (ascii_lower(video.color_transfer) != "smpte2084") {
        const std::string said = is_untagged_tag(video.color_transfer) ? "has no transfer tag"
                                                                        : "is " + video.color_transfer;
        return std::string("HDR VMAF scores PQ (HDR10) video only: ") + which + "'s transfer " + said + ".";
    }
    const std::optional<VmafColorMatrixCoefficients> matrix =
        tag_value(matrix_names, video.color_space, VMAF_COLOR_MATRIX_BT2020_NCL);
    if (!matrix || *matrix == VMAF_COLOR_MATRIX_ICTCP) {
        return std::string("HDR VMAF converts no video with the colour matrix ") + video.color_space + " (" + which +
               ").";
    }
    const std::optional<VmafColorPrimaries> primaries =
        tag_value(primaries_names, video.color_primaries, VMAF_COLOR_PRIMARIES_BT2020);
    if (!primaries) {
        return std::string("HDR VMAF converts no video with the primaries ") + video.color_primaries + " (" + which +
               ").";
    }
    return VmafColor{full_range ? VMAF_COLOR_RANGE_FULL : VMAF_COLOR_RANGE_LIMITED, *primaries,
                     VMAF_COLOR_TRC_SMPTE2084, *matrix};
}

bool colour_equal(const VmafColor& a, const VmafColor& b) {
    return a.range == b.range && a.primaries == b.primaries && a.trc == b.trc && a.matrix == b.matrix;
}

// The size a picture of `width` x `height` scales to for a display whose
// short side is `display` (conversion_policy.c's scaled_size).
std::pair<unsigned, unsigned> scaled_size(unsigned width, unsigned height, unsigned display) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument("HDR VMAF cannot scale an empty picture.");
    }
    const bool portrait = height > width;
    const unsigned short_side = portrait ? width : height;
    const unsigned long_side = portrait ? height : width;
    if (short_side == display) {  // already at the display size: kept exactly
        return {width, height};
    }
    // The long side at the picture's aspect ratio, rounded to the nearest even.
    const unsigned long long scaled =
        (static_cast<unsigned long long>(long_side) * display + short_side) / (2ULL * short_side) * 2;
    if (scaled > 4ULL * display) {
        throw std::invalid_argument("HDR VMAF's model would scale " + std::to_string(width) + "x" +
                                    std::to_string(height) + " to a long side of " + std::to_string(scaled) +
                                    ", over its limit of " + std::to_string(4 * display) + ".");
    }
    const auto scaled_side = static_cast<unsigned>(scaled);
    return portrait ? std::pair(display, scaled_side) : std::pair(scaled_side, display);
}

// The filter libvmaf scales a picture with (conversion_policy.c's
// resample_filter).
VmafResampleFilter resample_filter(unsigned width, unsigned height, unsigned to_width, unsigned to_height) {
    if (width == to_width && height == to_height) {
        return VMAF_RESAMPLE_DEFAULT;
    }
    return static_cast<unsigned long long>(width) * height < static_cast<unsigned long long>(to_width) * to_height
               ? VMAF_RESAMPLE_BICUBIC
               : VMAF_RESAMPLE_LANCZOS;
}

// Two scaled sides may differ by this much from the rounding to even
// (854x480 and 1280x720 scale to 1922 and 1920 at 1080).
constexpr unsigned aspect_tolerance = 2;

bool within_tolerance(unsigned a, unsigned b) { return (a > b ? a - b : b - a) <= aspect_tolerance; }

// One of HdrScorer's threads and the pair of pictures it converts: the
// pictures as the videos give them, kept from one frame to the next (they
// never go to libvmaf, unless one already is what the model asks for), and
// the converted ones, until they are given to libvmaf.
struct Converter {
    explicit Converter(const Libvmaf& from) : library(from) {}
    ~Converter() {
        if (busy) {
            thread.wait_quietly();
        }
        for (VmafPicture* picture : {&given[0], &given[1], &converted[0], &converted[1]}) {
            if (picture->priv != nullptr) {
                library.picture_unref(picture);
            }
        }
        for (VmafPictureConvertContext* context : contexts) {
            if (context != nullptr) {
                library.picture_convert_context_close(context);
            }
        }
    }
    Converter(const Converter&) = delete;
    Converter& operator=(const Converter&) = delete;

    const Libvmaf& library;
    // The source's and the test video's, each built from its first picture,
    // as libvmaf builds its own (convert_context_get).
    std::array<VmafPictureConvertContext*, 2> contexts{};
    std::array<VmafPicture, 2> given{};
    std::array<VmafPicture, 2> converted{};
    bool busy = false;  // converting, or converted and not yet given to libvmaf
    SideThread thread;  // last, so that it goes first
};

// A libvmaf picture given back however the work ends.
struct OwnedPicture {
    explicit OwnedPicture(const Libvmaf& from) : library(from) {}
    ~OwnedPicture() {
        if (picture.priv != nullptr) {
            library.picture_unref(&picture);
        }
    }
    OwnedPicture(const OwnedPicture&) = delete;
    OwnedPicture& operator=(const OwnedPicture&) = delete;

    const Libvmaf& library;
    VmafPicture picture{};
};

// libvmaf's conversion of `input` to `target`.
std::unique_ptr<OwnedPicture> libvmaf_conversion(const Libvmaf& library, const VmafPicture& input,
                                                 const VmafPictureConvertTarget& target) {
    VmafPictureConvertContext* context = nullptr;
    check_libvmaf(library.picture_convert_context_init(&context, &input, &target),
                  "Readying the conversion of the pictures for HDR VMAF");
    auto converted = std::make_unique<OwnedPicture>(library);
    const int error = library.picture_convert(context, &converted->picture, &input);
    library.picture_convert_context_close(context);
    check_libvmaf(error, "Converting a picture for HDR VMAF");
    return converted;
}

// A plan of vmaf_picture_convert_gpu_plan's, freed by libvmaf.
using ConversionPlan = std::unique_ptr<std::uint32_t, decltype(&::vmaf_picture_convert_gpu_plan_free)>;

// Fills a picture of 16-bit samples with every luma code its bit depth has,
// in order, row after row, and the chroma codes in two scrambled orders
// (each also every code, the multipliers being odd), so that the codes meet
// in many combinations: the check of the GPU's conversion.
void fill_every_code(VmafPicture& picture) {
    const unsigned codes = 1u << picture.bpc;
    constexpr std::array<unsigned, 3> multipliers{1, 37, 101};
    for (std::size_t plane = 0; plane < 3; ++plane) {
        for (unsigned y = 0; y < picture.h[plane]; ++y) {
            auto* const row = reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(picture.data[plane]) +
                                                               static_cast<std::ptrdiff_t>(y) * picture.stride[plane]);
            for (unsigned x = 0; x < picture.w[plane]; ++x) {
                const unsigned index = y * picture.w[plane] + x;
                row[x] = static_cast<std::uint16_t>(index * multipliers[plane] % codes);
            }
        }
    }
}

}  // namespace

HdrModelTarget hdr_model_target(const std::filesystem::path& file) {
    try {
        std::ifstream stream(file, std::ios::binary);
        const nlohmann::json data = nlohmann::json::parse(stream);
        const nlohmann::json& target = data.at("model_dict").at("conversion_target");
        const nlohmann::json& colourspace = target.at("colorspace");
        HdrModelTarget model;
        model.colour = {model_colour(range_names, colourspace, "range"),
                        model_colour(primaries_names, colourspace, "primaries"),
                        model_colour(transfer_names, colourspace, "trc"),
                        model_colour(matrix_names, colourspace, "matrix")};
        if (target.at("pixel_format").get<std::string>() != "420") {
            throw LibvmafError("the model's pixel format is not 4:2:0");
        }
        model.pixel_format = VMAF_PIX_FMT_YUV420P;
        model.bit_depth = target.at("bit_depth").get<unsigned>();
        model.display = target.at("resolution").at("display").get<unsigned>();
        if (model.display != 1080 && model.display != 2160) {
            throw LibvmafError("the model's display size is not 1080 or 2160 lines");
        }
        return model;
    } catch (const nlohmann::json::exception& error) {
        throw LibvmafError(platform::path_text(file.filename()) + " is not an HDR VMAF model: " + error.what());
    } catch (const LibvmafError& error) {
        throw LibvmafError(platform::path_text(file.filename()) + ": " + error.what());
    }
}

std::variant<HdrColours, std::string> hdr_colours(VideoInfo source, VideoInfo distorted) {
    // with_shared_colours would guess a pair's matrix that neither video tags
    // from its height, as SDR video's (BT.709, or BT.601 below 650 lines);
    // a PQ video's is BT.2020's. A pair that is not PQ is refused below.
    if (is_untagged_tag(source.color_space) && is_untagged_tag(distorted.color_space)) {
        source.color_space = distorted.color_space = "bt2020nc";
    }
    const auto [shared_source, shared_distorted] = with_shared_colours(source, distorted);
    const bool full_range = shared_source.full_range();
    std::variant<VmafColor, std::string> source_colour = video_colours(shared_source, full_range, "the source");
    if (auto* refusal = std::get_if<std::string>(&source_colour)) {
        return std::move(*refusal);
    }
    std::variant<VmafColor, std::string> distorted_colour =
        video_colours(shared_distorted, full_range, "the test video");
    if (auto* refusal = std::get_if<std::string>(&distorted_colour)) {
        return std::move(*refusal);
    }
    return HdrColours{std::get<VmafColor>(source_colour), std::get<VmafColor>(distorted_colour)};
}

std::array<VmafPictureConvertTarget, 2> hdr_targets(unsigned source_width, unsigned source_height,
                                                    unsigned distorted_width, unsigned distorted_height,
                                                    const HdrModelTarget& model) {
    const auto [width, height] = scaled_size(source_width, source_height, model.display);
    const auto [distorted_to_width, distorted_to_height] =
        scaled_size(distorted_width, distorted_height, model.display);
    // A landscape and a portrait picture never match, however close their
    // scaled sizes; a square matches both.
    const bool mixed_orientation = (source_width > source_height && distorted_height > distorted_width) ||
                                   (source_height > source_width && distorted_width > distorted_height);
    if (mixed_orientation || !within_tolerance(width, distorted_to_width) ||
        !within_tolerance(height, distorted_to_height)) {
        throw std::invalid_argument("HDR VMAF cannot scale the source (" + std::to_string(source_width) + "x" +
                                    std::to_string(source_height) + ") and the test video (" +
                                    std::to_string(distorted_width) + "x" + std::to_string(distorted_height) +
                                    ") to one size: their shapes differ.");
    }
    VmafPictureConvertTarget source_target{};
    source_target.pix_fmt = model.pixel_format;
    source_target.bpc = model.bit_depth;
    source_target.w = width;
    source_target.h = height;
    source_target.color = model.colour;
    VmafPictureConvertTarget distorted_target = source_target;
    source_target.resample_filter = resample_filter(source_width, source_height, width, height);
    distorted_target.resample_filter = resample_filter(distorted_width, distorted_height, width, height);
    return {source_target, distorted_target};
}

bool picture_matches(const VmafPicture& picture, const VmafPictureConvertTarget& target) {
    return colour_equal(picture.color, target.color) &&
           (target.pix_fmt == VMAF_PIX_FMT_UNKNOWN || picture.pix_fmt == target.pix_fmt) &&
           (target.bpc == 0 || picture.bpc == target.bpc) && (target.w == 0 || picture.w[0] == target.w) &&
           (target.h == 0 || picture.h[0] == target.h);
}

int hdr_conversion_threads(unsigned cores, bool features_on_gpu) {
    if (features_on_gpu) {  // the conversion is then all the CPU does
        return std::clamp(static_cast<int>(cores * 2 / 3), 1, 16);
    }
    return std::clamp(static_cast<int>(cores / 3), 1, 12);
}

// ---------------------------------------------------------------- HdrScorer

struct HdrScorer::State {
    State(const Libvmaf& from) : library(from), handles(from) {}

    // Converts the pair `converter` holds, on its thread.
    void convert(Converter& converter) const;

    // Gives the features' scorer (the GPU's, else libvmaf) the pair
    // `converter` converted, as frame `fed`.
    void feed(Converter& converter);

    // The features' scorer on the GPU, or nullptr (the log says why). With
    // `share`, one that takes the pairs from the decoders (V1Scorer's `shared`).
    std::unique_ptr<V1Scorer> make_gpu_scorer(GpuFrameStream* share = nullptr) const;
    // libvmaf's own extractors for the features (`handles`).
    void start_cpu_features();
    // The conversion threads.
    void start_converters();
    // Has `gpu` convert the pictures, if it and libvmaf can (the log says
    // why not): gpu_converts. The first pair is checked (check_gpu_conversion).
    void start_gpu_conversion();
    // The first sample, as text, where the GPU's conversion of `given`
    // (side `side`) differs from libvmaf's (`cpu`), or an empty text.
    std::string conversion_difference(std::size_t side, const VmafPicture& given, const VmafPicture& cpu) const;
    // Holds the GPU's conversion to libvmaf's on the first pair and on a
    // picture with every luma code. On a difference (a driver that rounds
    // otherwise), the pictures are converted on the CPU from here on, with
    // a new scorer for the features, and the log says why.
    void check_gpu_conversion(const FramePlanes& reference, const FramePlanes& distorted);

    const Libvmaf& library;
    std::filesystem::path model_file;
    int n_subsample = 1;
    int threads = 1;
    std::optional<int> device;  // the features' GPU, if any (-1 the best)
    // The first pair of add_decoded, downloaded for check_gpu_conversion
    // (both videos' pictures, packed), and every pair of it where the GPU
    // does not take them from the decoders.
    std::vector<std::uint8_t> downloaded;
    // The features on the GPU (VMAF v1's engine, with the HDR model's
    // options), or nullptr: libvmaf's own extractors (`handles`).
    std::unique_ptr<V1Scorer> gpu;
    // Whether `gpu` converts the pictures too: the pairs then go to it as the
    // videos give them, and there are no converters.
    bool gpu_converts = false;
    LibvmafHandles handles;
    std::array<VmafPictureConvertTarget, 2> targets{};
    HdrColours colours;
    int width = 0;
    int height = 0;
    unsigned bit_depth = 8;
    PackedFrame packed;
    unsigned step = 1;
    unsigned given = 0;  // the pairs given so far
    unsigned fed = 0;    // the pairs given to libvmaf so far
    // Pair number n is converted by converters[n % size]: each has one pair
    // at a time, and they come back in their order.
    std::vector<std::unique_ptr<Converter>> converters;
};

void HdrScorer::State::convert(Converter& converter) const {
    for (std::size_t side = 0; side < 2; ++side) {
        VmafPicture& given_picture = converter.given[side];
        const VmafPictureConvertTarget& target = targets[side];
        if (picture_matches(given_picture, target)) {
            // libvmaf takes it as it is: it goes to libvmaf, and the next
            // frame gets a new one.
            converter.converted[side] = given_picture;
            given_picture = VmafPicture{};
            continue;
        }
        VmafPictureConvertContext*& context = converter.contexts[side];
        if (context == nullptr) {
            check_libvmaf(library.picture_convert_context_init(&context, &given_picture, &target),
                          "Readying the conversion of the pictures for HDR VMAF");
        }
        check_libvmaf(library.picture_convert(context, &converter.converted[side], &given_picture),
                      "Converting a picture for HDR VMAF");
    }
}

void HdrScorer::State::feed(Converter& converter) {
    if (gpu) {
        // The GPU's scorer copies the planes in, and the pictures are then
        // given back (the converter's destructor gives them back if it
        // throws): each conversion makes new ones.
        const auto planes = [](const VmafPicture& picture) {
            FramePlanes planes;
            for (std::size_t plane = 0; plane < planes.size(); ++plane) {
                planes[plane] = {static_cast<const std::uint8_t*>(picture.data[plane]), picture.stride[plane]};
            }
            return planes;
        };
        ++fed;
        gpu->add_planes(planes(converter.converted[0]), planes(converter.converted[1]));
        for (VmafPicture& picture : converter.converted) {
            library.picture_unref(&picture);
            picture = VmafPicture{};
        }
        return;
    }
    // libvmaf owns both pictures from here, whatever it answers.
    VmafPicture reference = std::exchange(converter.converted[0], VmafPicture{});
    VmafPicture distorted = std::exchange(converter.converted[1], VmafPicture{});
    const unsigned frame = fed++;
    const int error = library.read_pictures(handles.context, &reference, &distorted, frame);
    if (error != 0) {
        check_libvmaf(error, "Giving libvmaf frame " + std::to_string(frame) + " for HDR VMAF");
    }
}

HdrScorer::HdrScorer(int width, int height, int bit_depth, const std::filesystem::path& model_file,
                     const HdrColours& colours, int n_subsample, int threads, std::optional<int> gpu,
                     GpuFrameStream* shared)
    : state_(std::make_unique<State>(libvmaf())) {
    State& s = *state_;
    if (s.library.picture_convert == nullptr) {
        throw LibvmafError("the bundled libvmaf cannot convert pictures for HDR VMAF");
    }
    const HdrModelTarget model = hdr_model_target(model_file);
    s.targets = hdr_targets(static_cast<unsigned>(width), static_cast<unsigned>(height),
                            static_cast<unsigned>(width), static_cast<unsigned>(height), model);
    s.colours = colours;
    s.width = width;
    s.height = height;
    s.bit_depth = static_cast<unsigned>(bit_depth);
    s.packed = PackedFrame(width, height, bit_depth);
    s.step = static_cast<unsigned>(std::max(1, n_subsample));
    s.model_file = model_file;
    s.n_subsample = n_subsample;
    s.threads = std::max(1, threads);
    s.device = gpu;
    // The decoders copy the pictures into the GPU's scorer only where it
    // converts them (10- and 12-bit, with libvmaf's plan), into its buffers
    // for the converted pictures: so not where the model shrinks them.
    const bool share = shared != nullptr && s.bit_depth > 8 && s.library.picture_convert_gpu_plan != nullptr &&
                       static_cast<unsigned>(width) <= s.targets[0].w &&
                       static_cast<unsigned>(height) <= s.targets[0].h;
    s.gpu = s.make_gpu_scorer(share ? shared : nullptr);
    if (s.gpu) {
        s.start_gpu_conversion();
        if (s.gpu && s.gpu->shares_decoders() && !s.gpu_converts) {
            // It would score the decoders' pictures unconverted: one that
            // takes them in system memory (its GPU memory given back first).
            s.gpu.reset();
            s.gpu = s.make_gpu_scorer();
            if (!s.gpu) {
                s.start_cpu_features();
            }
        }
    } else {
        s.start_cpu_features();
    }
    if (!s.gpu_converts) {
        s.start_converters();
    } else if (s.gpu->shares_decoders()) {
        logger.info("HDR VMAF's pictures copied into the GPU's memory by the decoders");
    }
}

std::unique_ptr<V1Scorer> HdrScorer::State::make_gpu_scorer(GpuFrameStream* share) const {
    if (!device) {
        return nullptr;
    }
    // The converted pictures' features on the GPU: VMAF v1's engine takes
    // the HDR model's options (CAMBI and SpEED on the GPU too), and gives
    // libvmaf's values; every frame goes to it (motion reads the one
    // before), and it scores every n_subsample-th, as libvmaf.
    const VmafPictureConvertTarget& converted = targets[0];
    try {
        auto scorer = std::make_unique<V1Scorer>(
            static_cast<int>(converted.w), static_cast<int>(converted.h), static_cast<int>(converted.bpc),
            model_file, n_subsample, *device >= 0 ? device : std::nullopt, threads, "vmaf_hdr", share, true, true,
            false, nullptr, share != nullptr);
        logger.info("HDR VMAF's features on the GPU (Vulkan)");
        return scorer;
    } catch (const LibvmafError& error) {
        if (share != nullptr) {  // where the decoders and Vulkan cannot share memory: from system memory
            logger.info(std::string("HDR VMAF's pictures read in system memory: ") + error.what());
            return make_gpu_scorer();
        }
        logger.info(std::string("HDR VMAF's features on the CPU: ") + error.what());
        return nullptr;
    }
}

void HdrScorer::State::start_cpu_features() {
    handles.start(threads, n_subsample);
    handles.load_model(model_file);
    check_libvmaf(library.use_features_from_model(handles.context, handles.model), "Readying HDR VMAF's features");
}

void HdrScorer::State::start_converters() {
    const int count = hdr_conversion_threads(std::thread::hardware_concurrency(), gpu != nullptr);
    for (int i = 0; i < count; ++i) {
        converters.push_back(std::make_unique<Converter>(library));
    }
}

void HdrScorer::State::start_gpu_conversion() {
    std::string why;
    if (bit_depth <= 8) {
        why = "the GPU converts 10- and 12-bit pictures";
    } else if (library.picture_convert_gpu_plan == nullptr || library.picture_convert_gpu_plan_free == nullptr) {
        why = "the bundled libvmaf has no conversion plan";
    } else if (gpu->can_convert_on_gpu(why)) {
        // Each video's plan, from a picture as the videos give them (only its
        // format, size and colours are read).
        const std::array<VmafColor, 2> given_colours{colours.source, colours.distorted};
        std::array<ConversionPlan, 2> plans{ConversionPlan(nullptr, library.picture_convert_gpu_plan_free),
                                            ConversionPlan(nullptr, library.picture_convert_gpu_plan_free)};
        std::array<std::span<const std::uint32_t>, 2> words;
        for (std::size_t side = 0; side < 2 && why.empty(); ++side) {
            OwnedPicture input(library);
            check_libvmaf(library.picture_alloc(&input.picture, VMAF_PIX_FMT_YUV420P, bit_depth,
                                                static_cast<unsigned>(width), static_cast<unsigned>(height)),
                          "Making a picture for HDR VMAF");
            input.picture.color = given_colours[side];
            std::uint32_t* plan = nullptr;
            std::size_t count = 0;
            if (const int error = library.picture_convert_gpu_plan(&input.picture, &targets[side], &plan, &count);
                error != 0) {
                why = "libvmaf has no plan of this conversion for the GPU (" + std::to_string(error) + ")";
            } else {
                plans[side].reset(plan);
                words[side] = {plan, count};
            }
        }
        if (why.empty()) {
            if (gpu->convert_on_gpu(words, why)) {
                gpu_converts = true;
                logger.info("HDR VMAF's pictures converted on the GPU (to be checked against libvmaf's on the "
                            "first frame)");
                return;
            }
            // The engine refused: a scorer that is not half set up (the old
            // one's GPU memory given back first).
            gpu.reset();
            gpu = make_gpu_scorer();
            if (!gpu) {
                start_cpu_features();
            }
        }
    }
    logger.info("HDR VMAF's pictures converted on the CPU: " + why);
}

std::string HdrScorer::State::conversion_difference(std::size_t side, const VmafPicture& input,
                                                   const VmafPicture& cpu) const {
    std::array<std::vector<std::uint16_t>, 3> out;
    std::array<std::uint16_t*, 3> out_planes{};
    std::array<std::ptrdiff_t, 3> out_strides{};
    std::array<const std::uint16_t*, 3> planes{};
    std::array<std::ptrdiff_t, 3> strides{};
    for (std::size_t plane = 0; plane < 3; ++plane) {
        out[plane].resize(static_cast<std::size_t>(cpu.w[plane]) * cpu.h[plane]);
        out_planes[plane] = out[plane].data();
        out_strides[plane] = static_cast<std::ptrdiff_t>(cpu.w[plane]) * 2;
        planes[plane] = static_cast<const std::uint16_t*>(input.data[plane]);
        strides[plane] = input.stride[plane];
    }
    gpu->test_convert(static_cast<int>(side), planes, strides, out_planes, out_strides);
    for (std::size_t plane = 0; plane < 3; ++plane) {
        for (unsigned y = 0; y < cpu.h[plane]; ++y) {
            const auto* const row = reinterpret_cast<const std::uint16_t*>(
                static_cast<const std::uint8_t*>(cpu.data[plane]) + static_cast<std::ptrdiff_t>(y) * cpu.stride[plane]);
            const std::uint16_t* const gpu_row = out[plane].data() + static_cast<std::size_t>(y) * cpu.w[plane];
            for (unsigned x = 0; x < cpu.w[plane]; ++x) {
                if (row[x] != gpu_row[x]) {
                    return "plane " + std::to_string(plane) + " sample (" + std::to_string(x) + ", " +
                           std::to_string(y) + "): libvmaf " + std::to_string(row[x]) + ", the GPU " +
                           std::to_string(gpu_row[x]);
                }
            }
        }
    }
    return {};
}

void HdrScorer::State::check_gpu_conversion(const FramePlanes& reference, const FramePlanes& distorted) {
    const std::size_t sample = bytes_per_sample(static_cast<int>(bit_depth));
    const std::array<const FramePlanes*, 2> frames{&reference, &distorted};
    const std::array<VmafColor, 2> given_colours{colours.source, colours.distorted};
    const std::array<const char*, 2> sides{"the source's", "the test video's"};
    // Each side's first frame and picture of every code. libvmaf converts
    // the four at once, on threads of their own (each conversion runs on one
    // thread), and the GPU converts each, here, as libvmaf finishes it.
    struct Case {
        std::size_t side = 0;
        bool made = false;  // the picture of every code, else the first frame
        std::unique_ptr<OwnedPicture> input;
        std::future<std::unique_ptr<OwnedPicture>> cpu;  // last, so that it waits before `input` goes
    };
    std::string difference;
    try {
        std::vector<Case> cases;
        cases.reserve(4);
        for (std::size_t side = 0; side < 2; ++side) {
            for (const bool made : {false, true}) {
                auto input = std::make_unique<OwnedPicture>(library);
                VmafPicture& picture = input->picture;
                check_libvmaf(library.picture_alloc(&picture, VMAF_PIX_FMT_YUV420P, bit_depth,
                                                    static_cast<unsigned>(width), static_cast<unsigned>(height)),
                              "Making a picture for HDR VMAF");
                picture.color = given_colours[side];
                if (made) {
                    fill_every_code(picture);
                } else {
                    fill_picture(picture, *frames[side], sample, {0, 1, 2});
                }
                std::future<std::unique_ptr<OwnedPicture>> cpu =
                    std::async(std::launch::async,
                               [this, side, &picture] { return libvmaf_conversion(library, picture, targets[side]); });
                cases.push_back({side, made, std::move(input), std::move(cpu)});
            }
        }
        for (Case& checked : cases) {
            const std::unique_ptr<OwnedPicture> cpu = checked.cpu.get();
            if (difference.empty()) {
                if (std::string at = conversion_difference(checked.side, checked.input->picture, cpu->picture);
                    !at.empty()) {
                    difference = std::string(sides[checked.side]) +
                                 (checked.made ? " picture of every code" : " first frame") + " differs at " + at;
                }
            }
        }
    } catch (const std::exception& error) {
        difference = error.what();
    }
    if (difference.empty()) {
        logger.info("HDR VMAF's pictures converted on the GPU: the first frame and a picture of every code came out "
                    "as libvmaf's");
        return;
    }
    logger.warning("HDR VMAF's pictures converted on the CPU: the GPU's conversion is not libvmaf's (" + difference +
                   ")");
    gpu_converts = false;
    gpu.reset();  // its GPU memory given back before the new scorer takes its own
    gpu = make_gpu_scorer();
    if (!gpu) {
        start_cpu_features();
    }
    start_converters();
}

HdrScorer::~HdrScorer() = default;

std::size_t HdrScorer::frame_bytes() const { return state_->packed.bytes(); }

void HdrScorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    add_planes(state_->packed.planes(reference.data()), state_->packed.planes(distorted.data()));
}

void HdrScorer::add_planes(const FramePlanes& reference, const FramePlanes& distorted) {
    State& s = *state_;
    if (s.gpu_converts && s.given == 0) {
        s.check_gpu_conversion(reference, distorted);
    }
    if (s.gpu_converts) {  // the GPU copies the pictures in as the videos give them
        ++s.given;
        ++s.fed;
        s.gpu->add_planes(reference, distorted);
        return;
    }
    Converter& converter = *s.converters[s.given % s.converters.size()];
    if (converter.busy) {  // its pair is the oldest not yet given to libvmaf
        converter.busy = false;
        converter.thread.wait();
        s.feed(converter);
    }
    const std::size_t sample = bytes_per_sample(static_cast<int>(s.bit_depth));
    const std::array<const FramePlanes*, 2> frames{&reference, &distorted};
    const std::array<VmafColor, 2> colours{s.colours.source, s.colours.distorted};
    for (std::size_t side = 0; side < 2; ++side) {
        VmafPicture& picture = converter.given[side];
        if (picture.priv == nullptr) {
            check_libvmaf(s.library.picture_alloc(&picture, VMAF_PIX_FMT_YUV420P, s.bit_depth,
                                                  static_cast<unsigned>(s.width), static_cast<unsigned>(s.height)),
                          "Making a picture for HDR VMAF");
            picture.color = colours[side];
        }
        fill_picture(picture, *frames[side], sample, {0, 1, 2});
    }
    ++s.given;
    converter.busy = true;
    converter.thread.run([&s, &converter] { s.convert(converter); });
}

bool HdrScorer::takes_decoded() const {
    const State& s = *state_;
    return s.gpu_converts && s.gpu->shares_decoders();
}

void HdrScorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
    State& s = *state_;
    if (!takes_decoded() || s.given == 0) {
        // Downloaded: to check the GPU's conversion on the first pair, or
        // where the GPU does not take the pairs from the decoders.
        const std::size_t bytes = s.packed.bytes();
        s.downloaded.resize(2 * bytes);
        reference.download(reference_slot, s.downloaded.data());
        test.download(test_slot, s.downloaded.data() + bytes);
        const FramePlanes reference_planes = s.packed.planes(s.downloaded.data());
        const FramePlanes test_planes = s.packed.planes(s.downloaded.data() + bytes);
        if (takes_decoded()) {
            s.check_gpu_conversion(reference_planes, test_planes);
        }
        if (!takes_decoded()) {  // the check turned it off, or it was never on
            add_planes(reference_planes, test_planes);
            return;
        }
    }
    ++s.given;
    ++s.fed;
    s.gpu->add_decoded(reference, reference_slot, test, test_slot);
}

PairScores HdrScorer::finish() {
    State& s = *state_;
    while (s.fed < s.given) {
        Converter& converter = *s.converters[s.fed % s.converters.size()];
        converter.busy = false;
        converter.thread.wait();
        s.feed(converter);
    }
    if (s.gpu) {
        PairScores scores = s.gpu->finish();  // under "vmaf_hdr", each to 6 decimals as below
        scores.features_on_gpu.emplace_back("vmaf_hdr");
        return scores;
    }
    check_libvmaf(s.library.read_pictures(s.handles.context, nullptr, nullptr, 0), "Finishing HDR VMAF");
    PairScores scores;
    std::vector<double> values;
    for (unsigned frame = 0; frame < s.given; frame += s.step) {
        double score = 0.0;
        const int error = s.library.score_at_index(s.handles.context, s.handles.model, &score, frame);
        if (error != 0) {
            check_libvmaf(error, "Predicting HDR VMAF of frame " + std::to_string(frame));
        }
        scores.frames.push_back(frame);
        values.push_back(to_6_decimals(score));
    }
    scores.by_metric.emplace_back("vmaf_hdr", std::move(values));
    return scores;
}

// ------------------------------------------------------- decoded in the app

PairScores score_decoded_hdr(const VideoInfo& source, const VideoInfo& distorted,
                             const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                             const GpuPlan& plan, int n_subsample, const FrameRange& range, long long total_frames,
                             const std::string& scale_algorithm, const std::pair<std::string, std::string>& decoders,
                             const std::string& ffmpeg, const std::string& ffprobe, bool want_vvdec,
                             const std::function<void()>& on_vvdec_opening,
                             const std::function<void(long long, long long, double)>& on_progress,
                             const CancelToken& cancel) {
    const auto& [source_backend, test_backend] = decoders;
    for (const std::string& library : {source_backend, test_backend}) {
        if (library.empty() || !decoder_available(library)) {
            throw GpuDecodeUnavailableError("the " + library + " decoder library is not bundled");
        }
    }
    require_same_range(source, distorted);
    std::vector<DecodePlan> plans = plan_pair(source, distorted, source_crop, distorted_crop,
                                              std::pair(plan.width, plan.height), plan.bit_depth, false,
                                              scale_algorithm, decoders);
    // A VVC video decoded by vvdec-vulkan where wanted, on the GPU it picks (this pass has none of its own)
    const bool test_vvdec = vvdec_chosen(plans[0], want_vvdec);
    const bool reference_vvdec = vvdec_chosen(plans[1], want_vvdec);
    if ((test_vvdec || reference_vvdec) && on_vvdec_opening) {
        on_vvdec_opening();
    }
    plan_pair_seek(plans, distorted, source, range.decode_start, ffmpeg, ffprobe, cancel);
    // AMD's decoder hands its pictures over on the GPU where nothing beside
    // reads them in system memory (pass_may_hand_over); NVIDIA's always does.
    const bool hand_over = pass_may_hand_over(plan);
    GpuFrameStream test =
        open_pass_stream(distorted, plans[0], 4, test_backend, test_vvdec, -1, ffmpeg, ffprobe, hand_over);
    GpuFrameStream reference =
        open_pass_stream(source, plans[1], 4, source_backend, reference_vvdec, -1, ffmpeg, ffprobe, hand_over);
    // Declared after the streams so that they go first. The decoders copy
    // the pictures into the GPU's memory where they share it with Vulkan
    // (HdrScorer's `shared`); vvdec-vulkan's do not (it shares none).
    const bool may_share = !test_vvdec && !reference_vvdec && test.shares_gpu_memory_with(reference);
    const std::unique_ptr<HdrScorer> scorer = make_hdr_scorer(plan, n_subsample, may_share ? &test : nullptr);
    std::optional<BesideScorer> beside;  // the CPU's metrics, from the same frames
    start_beside(beside, plan, n_subsample);
    if (std::any_of(plans.begin(), plans.end(),
                    [&](const DecodePlan& decoded) { return decoded.frame_bytes() != scorer->frame_bytes(); })) {
        throw LibvmafError("the decoders' frames are not the size HDR VMAF's scorer takes");
    }
    if (beside && beside->frame_bytes() != scorer->frame_bytes()) {
        throw LibvmafError("the CPU's metrics take frames of another size than HDR VMAF");
    }
    // Where HDR VMAF's GPU does not take the pictures from the decoders, they
    // go through system memory, the software decoder's read where it holds
    // them. Last, so that it goes first: its side thread's jobs use the
    // scorers.
    std::optional<HostPairs> pairs;
    if (!scorer->takes_decoded()) {
        pairs.emplace(reference, plans[1], test, plans[0]);
    }
    test.start();
    reference.start();
    score_frame_pairs(
        test, reference, range, total_frames,
        [&](int test_slot, int reference_slot) {
            if (pairs) {
                pairs->score(reference_slot, test_slot, *scorer, beside ? &*beside : nullptr);
                return;
            }
            // The CPU's metrics read a download of the planes they need.
            if (beside) {
                beside->add_decoded(reference, reference_slot, test, test_slot);
            }
            scorer->add_decoded(reference, reference_slot, test, test_slot);
        },
        on_progress, cancel);
    PairScores scores = scorer->finish();
    add_beside(scores, beside);
    return scores;
}

}  // namespace vml
