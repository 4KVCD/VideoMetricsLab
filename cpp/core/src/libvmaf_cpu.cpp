#include "vml/libvmaf_cpu.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

#include "vml/gpu_frames.h"
#include "vml/number_format.h"
#include "vml/python_text.h"

namespace vml {

namespace {

constexpr std::array<CpuFeature, 3> cpu_features{{
    {"psnr", "psnr", "psnr_y", 6},
    {"ssim", "float_ssim", "float_ssim", 6},
    {"xpsnr", "xpsnr", "xpsnr_y", 4},
}};

// The most memory CpuScorer's pictures may take (unless two pairs alone take
// more, at 8K). On 4K 10-bit video from NVIDIA's decoder, PSNR + SSIM run
// at 49 fps with 2 pairs, 93 with 3, 129 with 4, 159 with 5, and 172 with 6
// (this limit) or 8, where the decoders' feed levels off. XPSNR's pairs take
// longer, and it keeps a pair from the previous frames, so it gets more:
// 135 fps with 7 pairs, 154 with 8, 169 with 9 (its limit) and 171 with 10.
constexpr std::size_t picture_memory = std::size_t{200} << 20;
constexpr std::size_t xpsnr_picture_memory = std::size_t{270} << 20;

// A number kept to `decimals`, as Python's float(f"{value:.6f}") keeps it.
double rounded(double value, int decimals) { return python_float(format_fixed(value, decimals)).value_or(value); }

bool has(const std::vector<std::string>& metrics, std::string_view key) {
    return std::find(metrics.begin(), metrics.end(), key) != metrics.end();
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

std::string cpu_library_build() { return "libvmaf-fast " + libvmaf_fast_version() + " (CPU)"; }

CpuScorer::CpuScorer(int width, int height, int bit_depth, std::vector<std::string> metrics, int n_subsample,
                     int threads, int frame_rate)
    : library_(libvmaf()), metrics_(std::move(metrics)), step_(static_cast<unsigned>(std::max(1, n_subsample))) {
    const std::size_t sample = bit_depth <= 8 ? 1 : 2;
    rows_ = static_cast<std::size_t>(height);
    row_bytes_ = static_cast<std::size_t>(width) * sample;
    luma_bytes_ = rows_ * row_bytes_;
    frame_bytes_ = luma_bytes_ + 2 * ((static_cast<std::size_t>(width) + 1) / 2) * ((rows_ + 1) / 2) * sample;
    threads = std::max(0, threads);
    const VmafConfiguration configuration{VMAF_LOG_LEVEL_ERROR, static_cast<unsigned>(threads), step_, 0, 0};
    VmafContext* context = nullptr;
    check_libvmaf(library_.init(&context, configuration), "Starting libvmaf");
    context_ = context;
    try {
        // With XPSNR, PSNR comes from the squared errors it sums anyway (its
        // option "psnr"), not a pass of its own over both pictures: 4K PSNR +
        // SSIM + XPSNR 6% faster with Intel's decoder, 11% with NVIDIA's.
        const bool fused = has(metrics_, "psnr") && has(metrics_, "xpsnr");
        for (const std::string& metric : metrics_) {
            if (fused && metric == "psnr") {
                continue;
            }
            const CpuFeature* feature = cpu_feature(metric);
            if (feature == nullptr) {
                throw std::invalid_argument(metric + " is not scored with libvmaf's CPU extractors");
            }
            const std::string extractor(feature->extractor);
            VmafFeatureDictionary* options = nullptr;
            if (metric == "xpsnr") {
                // Weighted by the source's activity (the extractor's
                // default), as FFmpeg's filter weights by its first input,
                // the source in the app's commands.
                std::vector<std::pair<std::string, std::string>> settings{{"frame_rate", std::to_string(frame_rate)}};
                if (fused) {
                    settings.emplace_back("psnr", "true");
                }
                for (const auto& [key, value] : settings) {
                    if (library_.feature_dictionary_set(&options, key.c_str(), value.c_str()) != 0) {
                        library_.feature_dictionary_free(&options);
                        throw LibvmafError("Setting " + extractor + "'s " + key + " failed");
                    }
                }
            }
            // libvmaf frees the options (if it refuses them early, a few bytes leak).
            check_libvmaf(library_.use_feature(context_, extractor.c_str(), options), "Starting " + extractor);
        }
        const int pairs = cpu_pairs(width, height, bit_depth, threads, has(metrics_, "xpsnr"));
        const VmafPictureConfiguration pool{
            {static_cast<unsigned>(width), static_cast<unsigned>(height), static_cast<unsigned>(bit_depth),
             VMAF_PIX_FMT_YUV400P},
            static_cast<unsigned>(2 * pairs)};
        check_libvmaf(library_.preallocate_pictures(context_, pool), "Allocating pictures");
    } catch (...) {
        release();
        throw;
    }
}

CpuScorer::~CpuScorer() { release(); }

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
    // libvmaf takes both pictures, even when it fails (pull request 1652).
    check_libvmaf(library_.read_pictures(context_, &reference, &distorted, count_),
                  "Scoring frame " + std::to_string(count_));
    ++count_;
}

void CpuScorer::fill(VmafPicture& picture, Plane luma) const {
    auto* target = static_cast<std::uint8_t*>(picture.data[0]);
    const auto stride = static_cast<std::size_t>(picture.stride[0]);
    for (std::size_t row = 0; row < rows_; ++row) {
        std::memcpy(target + row * stride, luma.data + static_cast<std::ptrdiff_t>(row) * luma.stride, row_bytes_);
    }
}

void CpuScorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    if (std::min(reference.size(), distorted.size()) < frame_bytes_) {
        throw LibvmafError("frame " + std::to_string(count_) + " is shorter than a picture");
    }
    const auto packed = [this](std::span<const std::uint8_t> frame) {
        return FramePlanes{Plane{frame.data(), static_cast<std::ptrdiff_t>(row_bytes_)}};
    };
    add_planes(packed(reference), packed(distorted));
}

void CpuScorer::add_planes(const FramePlanes& reference, const FramePlanes& distorted) {
    VmafPicture reference_picture = take_picture();
    VmafPicture distorted_picture{};
    try {
        distorted_picture = take_picture();
    } catch (...) {
        library_.picture_unref(&reference_picture);
        throw;
    }
    fill(reference_picture, reference[0]);
    fill(distorted_picture, distorted[0]);
    score(reference_picture, distorted_picture);
}

void CpuScorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
    add_decoded(reference, reference_slot, test, test_slot, nullptr);
}

void CpuScorer::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot,
                            const std::function<void(Plane, Plane)>& also) {
    VmafPicture reference_picture = take_picture();
    VmafPicture distorted_picture{};
    try {
        distorted_picture = take_picture();
    } catch (...) {
        library_.picture_unref(&reference_picture);
        throw;
    }
    try {
        download_both(reference_picture, reference, reference_slot, distorted_picture, test, test_slot);
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

void CpuScorer::download_both(VmafPicture& reference_picture, GpuFrameStream& reference, int reference_slot,
                              VmafPicture& distorted_picture, GpuFrameStream& test, int test_slot) {
    if (!downloads_at_once(reference.backend(), test.backend())) {
        download(reference_picture, reference, reference_slot, 0);
        download(distorted_picture, test, test_slot, 1);
        return;
    }
    // The two decoders write their pictures at the same time, the source's
    // on the side thread: one after the other they took about 3 ms each per
    // 4K 10-bit pair, on the one thread that feeds the scorers.
    side_.both([&] { download(reference_picture, reference, reference_slot, 0); },
               [&] { download(distorted_picture, test, test_slot, 1); });
}

void CpuScorer::copy_packed(Plane plane, std::uint8_t* target) const { pack_plane(plane, rows_, row_bytes_, target); }

void CpuScorer::download(VmafPicture& picture, GpuFrameStream& stream, int slot, std::size_t which) {
    void* const address = picture.data[0];
    const auto pitch = static_cast<std::size_t>(picture.stride[0]);
    if (stream.backend() == "nvidia") {
        {
            const std::lock_guard lock(pinned_mutex_);  // the two videos' downloads run at once
            if (!pinned_.contains(address)) {
                pinned_[address] = stream.pin(address, pitch * picture.h[0]) ? &stream : nullptr;
            }
        }
        stream.download_planes(slot, {address, nullptr, nullptr}, {static_cast<std::int64_t>(pitch), 0, 0});
    } else if (pitch == row_bytes_) {
        stream.download(slot, address);
    } else {
        std::vector<std::uint8_t>& staging = staging_[which];
        staging.resize(luma_bytes_);
        stream.download(slot, staging.data());
        fill(picture, Plane{staging.data(), static_cast<std::ptrdiff_t>(row_bytes_)});
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
    check_libvmaf(library_.read_pictures(context_, nullptr, nullptr, 0), "Finishing");
    for (unsigned frame = 0; frame < count_; frame += step_) {
        result.frames.push_back(frame);
    }
    for (const std::string& metric : metrics_) {
        const CpuFeature& feature = *cpu_feature(metric);
        const std::string name(feature.feature);
        std::vector<double> column;
        column.reserve(result.frames.size());
        for (const unsigned frame : result.frames) {
            double value = 0;
            check_libvmaf(library_.feature_score_at_index(context_, name.c_str(), &value, frame),
                          "Reading " + metric + " for frame " + std::to_string(frame));
            column.push_back(rounded(value, feature.decimals));
        }
        result.by_metric.emplace_back(metric, std::move(column));
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

int cpu_pairs(int width, int height, int bit_depth, int threads, bool xpsnr) {
    const std::size_t picture = static_cast<std::size_t>((width + 31) & ~31) * static_cast<std::size_t>(height) *
                                (bit_depth <= 8 ? 1 : 2);
    const std::size_t memory = xpsnr ? xpsnr_picture_memory : picture_memory;
    const auto fit = static_cast<long long>(memory / (2 * picture));
    const long long wanted = std::min(8, std::max(1, threads)) + 1;
    return static_cast<int>(std::max(2LL, std::min(wanted, fit))) + (xpsnr ? 1 : 0);
}

PairScores score_decoded_cpu(const VideoInfo& source, const VideoInfo& distorted,
                             const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                             int width, int height, int bit_depth, const std::vector<std::string>& metrics,
                             int n_subsample, const std::optional<std::string>& duration_limit, long long total_frames,
                             const std::pair<std::string, std::string>& decoders, int threads, int frame_rate,
                             const std::string& scale_algorithm, const std::string& ffmpeg,
                             const std::string& ffprobe,
                             const std::function<void(long long, long long, double)>& on_progress,
                             const CancelToken& cancel) {
    const auto& [source_backend, test_backend] = decoders;
    require_same_range(source, distorted);
    // Each video's plan, the test video's first: its luma alone, at the size
    // and depth compared at, 10-bit samples in the low bits as FFmpeg's
    // yuv420p10le has them.
    std::vector<DecodePlan> plans;
    for (const bool test_video : {true, false}) {
        const VideoInfo& info = test_video ? distorted : source;
        DecodePlan plan = plan_decode(info, test_video ? distorted_crop : source_crop, 6, true,
                                      std::pair(width, height), scale_algorithm);
        widen_to(plan, info, bit_depth);
        const auto [supported, refusal] = decoder_supports(0, plan, test_video ? test_backend : source_backend);
        if (!supported) {
            throw GpuDecodeUnavailableError(refusal);
        }
        plans.push_back(plan);
    }
    const std::size_t luma_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                                   (bit_depth > 8 ? 2 : 1);
    if (std::any_of(plans.begin(), plans.end(),
                    [&](const DecodePlan& plan) { return plan.frame_bytes() != luma_bytes; })) {
        throw GpuDecodeUnavailableError("the decoders' frames are not the size compared at");
    }
    GpuFrameStream test(distorted, plans[0], 0, 4, test_backend, ffmpeg, ffprobe);
    GpuFrameStream reference(source, plans[1], 0, 4, source_backend, ffmpeg, ffprobe);
    // Declared after the streams so that it goes first: it gives back the
    // memory their decoders page-locked.
    CpuScorer scorer(width, height, bit_depth, metrics, n_subsample, threads, frame_rate);
    test.start();
    reference.start();
    score_frame_pairs(
        test, reference, distorted, duration_limit, total_frames,
        [&](int test_slot, int reference_slot) { scorer.add_decoded(reference, reference_slot, test, test_slot); },
        on_progress, cancel);
    return scorer.finish();
}

}  // namespace vml
