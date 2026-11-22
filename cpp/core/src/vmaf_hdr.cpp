#include "vml/vmaf_hdr.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
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

    const Libvmaf& library;
    // The features on the GPU (VMAF v1's engine, with the HDR model's
    // options), or nullptr: libvmaf's own extractors (`handles`).
    std::unique_ptr<V1Scorer> gpu;
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
                     const HdrColours& colours, int n_subsample, int threads, std::optional<int> gpu)
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
    if (gpu) {
        // The converted pictures' features on the GPU: VMAF v1's engine
        // takes the HDR model's options (CAMBI and SpEED on the GPU too),
        // and gives libvmaf's values; every frame goes to it (motion reads
        // the one before), and it scores every n_subsample-th, as libvmaf.
        const VmafPictureConvertTarget& converted = s.targets[0];
        try {
            s.gpu = std::make_unique<V1Scorer>(
                static_cast<int>(converted.w), static_cast<int>(converted.h), static_cast<int>(converted.bpc),
                model_file, n_subsample, *gpu >= 0 ? gpu : std::nullopt, std::max(1, threads), "vmaf_hdr");
            logger.info("HDR VMAF's features on the GPU (Vulkan)");
        } catch (const LibvmafError& error) {
            logger.info(std::string("HDR VMAF's features on the CPU: ") + error.what());
        }
    }
    if (!s.gpu) {
        s.handles.start(std::max(1, threads), n_subsample);
        s.handles.load_model(model_file);
        check_libvmaf(s.library.use_features_from_model(s.handles.context, s.handles.model),
                      "Readying HDR VMAF's features");
    }
    const int converters = hdr_conversion_threads(std::thread::hardware_concurrency(), s.gpu != nullptr);
    for (int i = 0; i < converters; ++i) {
        s.converters.push_back(std::make_unique<Converter>(s.library));
    }
}

HdrScorer::~HdrScorer() = default;

std::size_t HdrScorer::frame_bytes() const { return state_->packed.bytes(); }

void HdrScorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    add_planes(state_->packed.planes(reference.data()), state_->packed.planes(distorted.data()));
}

void HdrScorer::add_planes(const FramePlanes& reference, const FramePlanes& distorted) {
    State& s = *state_;
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
    GpuFrameStream test = open_pass_stream(distorted, plans[0], 4, test_backend, test_vvdec, -1, ffmpeg, ffprobe);
    GpuFrameStream reference =
        open_pass_stream(source, plans[1], 4, source_backend, reference_vvdec, -1, ffmpeg, ffprobe);
    // Declared after the streams so that they go first.
    const std::unique_ptr<HdrScorer> scorer = make_hdr_scorer(plan, n_subsample);
    std::optional<BesideScorer> beside;  // the CPU's metrics, from the same frames
    start_beside(beside, plan, n_subsample);
    if (std::any_of(plans.begin(), plans.end(),
                    [&](const DecodePlan& decoded) { return decoded.frame_bytes() != scorer->frame_bytes(); })) {
        throw LibvmafError("the decoders' frames are not the size HDR VMAF's scorer takes");
    }
    if (beside && beside->frame_bytes() != scorer->frame_bytes()) {
        throw LibvmafError("the CPU's metrics take frames of another size than HDR VMAF");
    }
    // The pictures in system memory, the software decoder's read where it
    // holds them. Last, so that it goes first: its side thread's jobs use the
    // scorers.
    HostPairs pairs(reference, plans[1], test, plans[0]);
    test.start();
    reference.start();
    score_frame_pairs(
        test, reference, range, total_frames,
        [&](int test_slot, int reference_slot) {
            pairs.score(reference_slot, test_slot, *scorer, beside ? &*beside : nullptr);
        },
        on_progress, cancel);
    PairScores scores = scorer->finish();
    add_beside(scores, beside);
    return scores;
}

}  // namespace vml
