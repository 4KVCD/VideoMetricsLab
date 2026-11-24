#include "vml/libvmaf_cpu.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

#include "vml/gpu_frames.h"
#include "vml/host_frames.h"
#include "vml/number_format.h"
#include "vml/python_text.h"

namespace vml {

namespace {

constexpr std::array<CpuFeature, 5> cpu_features{{
    {"psnr", "psnr", "psnr_y", 6},
    {"ssim", "float_ssim", "float_ssim", 6},
    {"xpsnr", "xpsnr", "xpsnr_y", 4},
    {"psnr_yuv", "psnr", "psnr_y", 6, "psnr_cb", "psnr_cr"},
    {"ssim_yuv", "float_ssim", "float_ssim", 6, "float_ssim_cb", "float_ssim_cr"},
}};

// The most memory CpuScorer's pictures may take (unless two pairs alone take
// more, at 8K). On 4K 10-bit video from NVIDIA's decoder, PSNR + SSIM run
// at 49 fps with 2 pairs, 93 with 3, 129 with 4, 159 with 5, and 172 with 6
// (this limit) or 8, where the decoders' feed levels off. XPSNR's pairs take
// longer, and it keeps a pair from the previous frames, so it gets more:
// 135 fps with 7 pairs, 154 with 8, 169 with 9 (its limit) and 171 with 10.
constexpr std::size_t picture_memory = std::size_t{200} << 20;
constexpr std::size_t xpsnr_picture_memory = std::size_t{270} << 20;

bool has(const std::vector<std::string>& metrics, std::string_view key) {
    return std::find(metrics.begin(), metrics.end(), key) != metrics.end();
}

// An extractor of libvmaf's and the options it is started with.
struct Extractor {
    std::string name;
    std::vector<std::pair<std::string, std::string>> options;
};

// The extractors that score `metrics`, each started once. PSNR and SSIM read
// the chroma planes only for their 6:1:1 metric. With XPSNR, the luma's PSNR
// comes from the squared errors it sums anyway (its option "psnr"), not a
// pass of its own over both pictures (4K PSNR + SSIM + XPSNR 6% faster with
// Intel's decoder, 11% with NVIDIA's); PSNR then scores the chroma alone, for
// PSNR YUV.
std::vector<Extractor> extractors_for(const std::vector<std::string>& metrics, int frame_rate) {
    for (const std::string& metric : metrics) {
        if (cpu_feature(metric) == nullptr) {
            throw std::invalid_argument(metric + " is not scored with libvmaf's CPU extractors");
        }
    }
    const bool psnr_yuv = has(metrics, "psnr_yuv");
    const bool ssim_yuv = has(metrics, "ssim_yuv");
    const bool psnr = psnr_yuv || has(metrics, "psnr");
    const bool xpsnr = has(metrics, "xpsnr");
    std::vector<Extractor> extractors;
    if (psnr && (!xpsnr || psnr_yuv)) {
        // Set both ways, as libvmaf's PSNR reads the chroma of a 4:2:0
        // picture by default.
        extractors.push_back({"psnr", {{"enable_chroma", psnr_yuv ? "true" : "false"},
                                       {"enable_luma", xpsnr ? "false" : "true"}}});
    }
    if (ssim_yuv || has(metrics, "ssim")) {
        extractors.push_back({"float_ssim", {}});
        if (ssim_yuv) {
            extractors.back().options.emplace_back("enable_chroma", "true");
        }
    }
    if (xpsnr) {
        // Weighted by the source's activity (the extractor's default), as
        // FFmpeg's filter weights by its first input, the source in the app's
        // commands.
        extractors.push_back({"xpsnr", {{"frame_rate", std::to_string(frame_rate)}}});
        if (psnr) {
            extractors.back().options.emplace_back("psnr", "true");
        }
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

bool cpu_metrics_read_chroma(const std::vector<std::string>& metrics) {
    return std::any_of(metrics.begin(), metrics.end(), [](const std::string& metric) {
        const CpuFeature* feature = cpu_feature(metric);
        return feature != nullptr && !feature->cb.empty();
    });
}

std::string cpu_library_build() { return "libvmaf-fast " + libvmaf_fast_version() + " (CPU)"; }

CpuScorer::CpuScorer(int width, int height, int bit_depth, std::vector<std::string> metrics, int n_subsample,
                     int threads, int frame_rate)
    : library_(libvmaf()), metrics_(std::move(metrics)), step_(static_cast<unsigned>(std::max(1, n_subsample))) {
    chroma_ = cpu_metrics_read_chroma(metrics_);
    odd_ = width % 2 != 0 || height % 2 != 0;
    sample_ = bytes_per_sample(bit_depth);
    packed_ = PackedFrame(width, height, bit_depth);
    frame_bytes_ = packed_.bytes();
    decoded_bytes_ = chroma_ ? frame_bytes_ : static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * sample_;
    threads = std::max(0, threads);
    const VmafConfiguration configuration{VMAF_LOG_LEVEL_ERROR, static_cast<unsigned>(threads), step_, 0, 0};
    VmafContext* context = nullptr;
    check_libvmaf(library_.init(&context, configuration), "Starting libvmaf");
    context_ = context;
    try {
        for (const Extractor& extractor : extractors_for(metrics_, frame_rate)) {
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
        const int pairs = cpu_pairs(width, height, bit_depth, threads, has(metrics_, "xpsnr"), chroma_);
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
    const auto take_reference = [&] { download(reference_picture, reference, reference_slot, 0); };
    const auto take_test = [&] { download(distorted_picture, test, test_slot, 1); };
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
    check_libvmaf(library_.read_pictures(context_, nullptr, nullptr, 0), "Finishing");
    for (unsigned frame = 0; frame < count_; frame += step_) {
        result.frames.push_back(frame);
    }
    for (const std::string& metric : metrics_) {
        const CpuFeature& feature = *cpu_feature(metric);
        const auto read = [&](std::string_view feature_name, unsigned frame) {
            const std::string name(feature_name);
            double value = 0;
            check_libvmaf(library_.feature_score_at_index(context_, name.c_str(), &value, frame),
                          "Reading " + name + " for frame " + std::to_string(frame));
            return value;
        };
        std::vector<double> column;
        column.reserve(result.frames.size());
        for (const unsigned frame : result.frames) {
            double value = read(feature.feature, frame);
            if (!feature.cb.empty()) {
                // 6:1:1, at full precision, then rounded once
                value = (6 * value + read(feature.cb, frame) + read(feature.cr, frame)) / 8;
            }
            column.push_back(python_round(value, feature.decimals));
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

int cpu_pairs(int width, int height, int bit_depth, int threads, bool xpsnr, bool chroma) {
    // A picture's bytes as vmaf_picture_alloc lays it out: rows of 32
    // samples, and with chroma two planes half as wide and tall.
    const auto plane = [&](int w, int h) {
        return static_cast<std::size_t>((w + 31) & ~31) * static_cast<std::size_t>(h) * bytes_per_sample(bit_depth);
    };
    const std::size_t picture = plane(width, height) + (chroma ? 2 * plane(width / 2, height / 2) : 0);
    const std::size_t luma_memory = xpsnr ? xpsnr_picture_memory : picture_memory;
    const std::size_t memory = chroma ? luma_memory * 3 / 2 : luma_memory;
    const auto fit = static_cast<long long>(memory / (2 * picture));
    const long long wanted = std::min(8, std::max(1, threads)) + 1;
    return static_cast<int>(std::max(2LL, std::min(wanted, fit))) + (xpsnr ? 1 : 0);
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
                             const std::string& scale_algorithm, const std::string& ffmpeg,
                             const std::string& ffprobe, bool want_vvdec,
                             const std::function<void()>& on_vvdec_opening,
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
    // Declared after the streams so that it goes first: it gives back the
    // memory their decoders page-locked.
    CpuScorer scorer(width, height, bit_depth, metrics, n_subsample, threads, frame_rate);
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
