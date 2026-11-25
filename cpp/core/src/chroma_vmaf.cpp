#include "vml/chroma_vmaf.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "vml/app_log.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/model_select.h"
#include "vml/pair_scorer.h"
#include "vml/platform/system.h"
#include "vml/vmaf_vulkan.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.chroma_vmaf");

constexpr std::string_view version_prefix = "version=";

// The plane of a 4:2:0 picture (1 U, 2 V) that `key` scores.
int plane_of_key(const std::string& key) {
    if (key == vmaf_u_key) {
        return 1;
    }
    if (key == vmaf_v_key) {
        return 2;
    }
    throw std::invalid_argument(key + " is not a chroma plane's VMAF");
}

}  // namespace

// One plane's VMAF: on the GPU (gpu) or in libvmaf (cpu).
struct ChromaVmaf::Plane {
    Plane(const Libvmaf& from, std::string scored, int which) : library(from), key(std::move(scored)), index(which) {}

    const Libvmaf& library;
    std::string key;
    int index = 1;  // the picture's plane: 1 U, 2 V
    std::unique_ptr<VulkanScorer> gpu;
    std::unique_ptr<LibvmafHandles> cpu;
    unsigned count = 0;
    unsigned step = 1;
    std::size_t row_bytes = 0;  // a chroma row's samples, in bytes

    void add(const VmafPicture& reference, const VmafPicture& distorted) {
        if (gpu) {
            FramePlanes from_reference{};
            FramePlanes from_distorted{};
            from_reference[0] = {static_cast<const std::uint8_t*>(reference.data[index]), reference.stride[index]};
            from_distorted[0] = {static_cast<const std::uint8_t*>(distorted.data[index]), distorted.stride[index]};
            gpu->add_planes(from_reference, from_distorted);  // copies them before it returns
        } else {
            VmafPicture to_reference = take();
            VmafPicture to_distorted{};
            if (const int error = library.fetch_preallocated_picture(cpu->context, &to_distorted); error != 0) {
                library.picture_unref(&to_reference);  // one never given back would keep vmaf_close waiting
                check_libvmaf(error, "Taking a picture");
            }
            copy(reference, to_reference);
            copy(distorted, to_distorted);
            // libvmaf takes both pictures, even when it fails.
            check_libvmaf(library.read_pictures(cpu->context, &to_reference, &to_distorted, count),
                          "Scoring frame " + std::to_string(count));
        }
        ++count;
    }

    // The pair from the decoders' slots, which copy this plane on the GPU into the scorer's memory.
    void add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
        gpu->add_shared(copy_from(reference, reference_slot), copy_from(test, test_slot));
        ++count;
    }

    // add_shared's copy of this plane of `decoder`'s picture in `slot`: without waiting where the engine waits
    // for the decoder's copies on the GPU (as luma_copy for the luma).
    std::function<std::uint64_t(std::uint64_t, std::int64_t)> copy_from(GpuFrameStream& decoder, int slot) const {
        const bool waits_on = gpu->waits_on(decoder);
        const int which = index;
        return [&decoder, slot, waits_on, which](std::uint64_t address, std::int64_t pitch) -> std::uint64_t {
            std::array<std::uint64_t, 3> addresses{};  // 0: the planes not copied
            std::array<std::int64_t, 3> pitches{};
            addresses[which] = address;
            pitches[which] = pitch;
            if (waits_on) {
                return decoder.copy_planes_async(slot, addresses, pitches);
            }
            decoder.copy_planes(slot, addresses, pitches);
            return 0;
        };
    }

    VmafPicture take() const {
        VmafPicture picture{};
        check_libvmaf(library.fetch_preallocated_picture(cpu->context, &picture), "Taking a picture");
        return picture;
    }

    // This plane of `from` into the luma of `to`, a picture of its size.
    void copy(const VmafPicture& from, VmafPicture& to) const {
        const auto* source = static_cast<const std::uint8_t*>(from.data[index]);
        auto* target = static_cast<std::uint8_t*>(to.data[0]);
        for (unsigned row = 0; row < to.h[0]; ++row) {
            std::memcpy(target + static_cast<std::ptrdiff_t>(row) * to.stride[0],
                        source + static_cast<std::ptrdiff_t>(row) * from.stride[index], row_bytes);
        }
    }

    std::vector<double> finish() {
        if (count == 0) {
            return {};
        }
        if (gpu) {
            PairScores scored = gpu->finish();  // to 6 decimals already (predict_scores)
            return std::move(scored.by_metric.at(0).second);
        }
        check_libvmaf(library.read_pictures(cpu->context, nullptr, nullptr, 0), "Finishing");
        std::vector<double> scores;
        for (unsigned frame = 0; frame < count; frame += step) {
            double score = 0;
            check_libvmaf(library.score_at_index(cpu->context, cpu->model, &score, frame),
                          "Reading " + key + " for frame " + std::to_string(frame));
            scores.push_back(to_6_decimals(score));
        }
        return scores;
    }
};

bool chroma_vmaf_fits_gpu(int width, int height, int bit_depth, const std::string& model) {
    const int chroma_width = width / 2;  // as libvmaf's 4:2:0 pictures hold them
    const int chroma_height = height / 2;
    // The engine's two built-in models (none for a model file), at the depths and sizes it scores VMAF at.
    return !gpu_models(true, false, model).empty() && bit_depth <= 10 && chroma_width % 2 == 0 &&
           chroma_height % 2 == 0;
}

ChromaVmaf::ChromaVmaf(int width, int height, int bit_depth, const std::string& model,
                       const std::vector<std::string>& keys, int n_subsample, int threads, std::optional<int> gpu,
                       std::optional<ChromaFromDecoders> from_decoders) {
    const Libvmaf& library = libvmaf();
    const int chroma_width = width / 2;  // as libvmaf's 4:2:0 pictures hold them
    const int chroma_height = height / 2;
    const unsigned step = static_cast<unsigned>(std::max(1, n_subsample));
    const GpuModels built_in = gpu_models(true, false, model);
    const bool gpu_fits = gpu && chroma_vmaf_fits_gpu(width, height, bit_depth, model);
    const int cpu_threads = keys.empty() ? 0 : (std::max(0, threads) + static_cast<int>(keys.size()) - 1) /
                                                   static_cast<int>(keys.size());
    const auto models_of = [&](const std::string& key) {
        return std::vector<std::pair<std::string, std::string>>{{key, built_in.front().second}};
    };
    for (const std::string& key : keys) {
        auto plane = std::make_unique<Plane>(library, key, plane_of_key(key));
        plane->step = step;
        plane->row_bytes = static_cast<std::size_t>(chroma_width) * bytes_per_sample(bit_depth);
        planes_.push_back(std::move(plane));
    }
    // From the decoders' memory: all planes or none, so that the CPU's pictures need no chroma. An odd size's
    // chroma rows the decoders round up, one sample more than the scorers' planes take.
    if (gpu_fits && from_decoders && width % 2 == 0 && height % 2 == 0 && from_decoders->test->copies_planes()) {
        try {
            for (const std::unique_ptr<Plane>& plane : planes_) {
                plane->gpu = std::make_unique<VulkanScorer>(chroma_width, chroma_height, bit_depth,
                                                            models_of(plane->key), n_subsample, from_decoders->device,
                                                            from_decoders->test);
            }
            takes_decoded_ = true;
            logger.info("VMAF YUV's planes are copied on the GPU from the decoders' memory");
        } catch (const LibvmafError& error) {  // VmafVulkanError among them
            logger.info(std::string("VMAF YUV's planes go through system memory: ") + error.what());
            for (const std::unique_ptr<Plane>& plane : planes_) {
                plane->gpu.reset();
            }
        }
    }
    for (const std::unique_ptr<Plane>& plane : planes_) {
        const std::string& key = plane->key;
        if (gpu_fits && !plane->gpu) {
            try {
                plane->gpu = std::make_unique<VulkanScorer>(chroma_width, chroma_height, bit_depth, models_of(key),
                                                            n_subsample,
                                                            *gpu >= 0 ? std::optional(*gpu) : std::nullopt);
            } catch (const LibvmafError& error) {  // VmafVulkanError among them
                logger.warning(key + " is calculated on the CPU: " + error.what());
            }
        }
        if (!plane->gpu) {
            plane->cpu = std::make_unique<LibvmafHandles>(library);
            plane->cpu->start(cpu_threads, n_subsample);
            if (const std::optional<std::string_view> file = model_file_of(model)) {
                plane->cpu->load_model(platform::path_from_text(std::string(*file)));
            } else if (model.starts_with(version_prefix)) {
                plane->cpu->load_builtin(model.substr(version_prefix.size()));
            } else {
                throw LibvmafError("VMAF YUV cannot read the model " + model);
            }
            check_libvmaf(library.use_features_from_model(plane->cpu->context, plane->cpu->model),
                          "Starting the model's features");
            // A pair for each thread and one being filled: a plane of a 4K
            // picture is 4 MB at most.
            const auto pictures = static_cast<unsigned>(2 * (std::max(1, cpu_threads) + 1));
            check_libvmaf(library.preallocate_pictures(
                              plane->cpu->context,
                              VmafPictureConfiguration{{static_cast<unsigned>(chroma_width),
                                                        static_cast<unsigned>(chroma_height),
                                                        static_cast<unsigned>(bit_depth), VMAF_PIX_FMT_YUV400P},
                                                       pictures}),
                          "Allocating pictures");
        }
    }
}

ChromaVmaf::~ChromaVmaf() = default;

std::vector<std::string> ChromaVmaf::keys_on_gpu() const {
    std::vector<std::string> keys;
    for (const std::unique_ptr<Plane>& plane : planes_) {
        if (plane->gpu) {
            keys.push_back(plane->key);
        }
    }
    return keys;
}

void ChromaVmaf::add(const VmafPicture& reference, const VmafPicture& distorted) {
    for (const std::unique_ptr<Plane>& plane : planes_) {
        plane->add(reference, distorted);
    }
}

void ChromaVmaf::add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot) {
    if (!takes_decoded_) {
        throw std::logic_error("VMAF YUV's planes are given in system memory, not from the decoders");
    }
    for (const std::unique_ptr<Plane>& plane : planes_) {
        plane->add_decoded(reference, reference_slot, test, test_slot);
    }
}

std::map<std::string, std::vector<double>> ChromaVmaf::finish() {
    std::map<std::string, std::vector<double>> scores;
    for (const std::unique_ptr<Plane>& plane : planes_) {
        scores.emplace(plane->key, plane->finish());
    }
    return scores;
}

}  // namespace vml
