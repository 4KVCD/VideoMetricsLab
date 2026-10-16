#include "vml/vmaf_cuda.h"

#include <libvmaf/libvmaf_cuda.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "vml/platform/library.h"
#include "vml/platform/system.h"
#include "vml/vmaf_vulkan.h"

namespace vml {

namespace {

// CUDA's driver API, as far as the uploads use it: part of NVIDIA's driver,
// which libvmaf loads too. Its types as cuda.h declares them.
using CudaResult = int;
using CudaDevice = int;
using CudaContext = struct CUctx_st*;
using CudaStream = struct CUstream_st*;
using CudaDevicePointer = unsigned long long;

// CUDA_MEMCPY2D: a copy of a rectangle of memory, with CUDA's names.
struct CudaCopy2D {
    std::size_t srcXInBytes = 0;
    std::size_t srcY = 0;
    unsigned srcMemoryType = 0;
    const void* srcHost = nullptr;
    CudaDevicePointer srcDevice = 0;
    void* srcArray = nullptr;
    std::size_t srcPitch = 0;
    std::size_t dstXInBytes = 0;
    std::size_t dstY = 0;
    unsigned dstMemoryType = 0;
    void* dstHost = nullptr;
    CudaDevicePointer dstDevice = 0;
    void* dstArray = nullptr;
    std::size_t dstPitch = 0;
    std::size_t WidthInBytes = 0;
    std::size_t Height = 0;
};

constexpr unsigned memory_host = 1, memory_device = 2;  // CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_DEVICE
constexpr unsigned host_alloc_portable = 1;              // CU_MEMHOSTALLOC_PORTABLE
constexpr unsigned stream_non_blocking = 1;              // CU_STREAM_NON_BLOCKING

// The GPU libvmaf scores on: CUDA's first, which in CUDA's default order is
// the fastest. NVIDIA's decoder and FFmpeg's -hwaccel cuda take the same one
// (the Python app's gpu_frames.CUDA_DEVICE).
constexpr int cuda_device = 0;

// The driver's library and the functions of it the uploads call.
struct CudaDriver {
    explicit CudaDriver(platform::DynamicLibrary loaded)
        : library(std::move(loaded)),
          init(library.function<decltype(init)>("cuInit")),
          device_get(library.function<decltype(device_get)>("cuDeviceGet")),
          primary_retain(library.function<decltype(primary_retain)>("cuDevicePrimaryCtxRetain")),
          primary_release(library.function<decltype(primary_release)>("cuDevicePrimaryCtxRelease_v2")),
          push_current(library.function<decltype(push_current)>("cuCtxPushCurrent_v2")),
          pop_current(library.function<decltype(pop_current)>("cuCtxPopCurrent_v2")),
          host_alloc(library.function<decltype(host_alloc)>("cuMemHostAlloc")),
          free_host(library.function<decltype(free_host)>("cuMemFreeHost")),
          stream_create(library.function<decltype(stream_create)>("cuStreamCreate")),
          stream_destroy(library.function<decltype(stream_destroy)>("cuStreamDestroy_v2")),
          copy_2d_async(library.function<decltype(copy_2d_async)>("cuMemcpy2DAsync_v2")),
          stream_synchronize(library.function<decltype(stream_synchronize)>("cuStreamSynchronize")) {}

    platform::DynamicLibrary library;
    CudaResult (*init)(unsigned flags);
    CudaResult (*device_get)(CudaDevice* device, int ordinal);
    CudaResult (*primary_retain)(CudaContext* context, CudaDevice device);
    CudaResult (*primary_release)(CudaDevice device);
    CudaResult (*push_current)(CudaContext context);
    CudaResult (*pop_current)(CudaContext* context);
    CudaResult (*host_alloc)(void** pointer, std::size_t bytes, unsigned flags);
    CudaResult (*free_host)(void* pointer);
    CudaResult (*stream_create)(CudaStream* stream, unsigned flags);
    CudaResult (*stream_destroy)(CudaStream stream);
    CudaResult (*copy_2d_async)(const CudaCopy2D* copy, CudaStream stream);
    CudaResult (*stream_synchronize)(CudaStream stream);
};

// The driver, loaded the first time it is asked for. Throws LibvmafError
// when it cannot be (no NVIDIA driver).
const CudaDriver& cuda_driver() {
    try {
#ifdef _WIN32
        static const CudaDriver loaded(platform::DynamicLibrary::system("nvcuda.dll"));
#else
        static const CudaDriver loaded(platform::DynamicLibrary::system("libcuda.so.1"));
#endif
        return loaded;
    } catch (const platform::LibraryError& error) {
        throw LibvmafError(error.what());
    }
}

// Throws LibvmafError if a CUDA call returned an error.
void check_cuda(CudaResult error, const std::string& what) {
    if (error != 0) {
        throw LibvmafError(what + " failed (CUDA error " + std::to_string(error) + ")");
    }
}

// The GPU address of a libvmaf picture's luma plane.
std::uint64_t luma_address(const VmafPicture& picture) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(picture.data[0]));
}

}  // namespace

// Copies frames' luma from system memory into libvmaf's pictures on the GPU,
// through one page-locked buffer per side, allocated once, on a CUDA stream
// of its own (the Python app's _HostUpload).
//
// This is faster than libvmaf's own handling of frames in system memory (a
// pool of CPU pictures, each uploaded from pageable memory when it is read):
// about 330 4K 10-bit pairs a second on an RTX 5090, against about 180.
// libvmaf's page-locked pictures (HOST_PINNED) are no
// faster, because each fetch allocates, locks and zeroes a new one, which
// takes 9 ms per 4K pair.
//
// It works in libvmaf's CUDA context, the primary context of its GPU, which
// is made current around each use (Current) because the frames arrive on
// the caller's thread.
class CudaUpload {
public:
    CudaUpload(int width, int height, int sample)
        : cu_(cuda_driver()),
          row_bytes_(static_cast<std::size_t>(width) * static_cast<std::size_t>(sample)),
          rows_(static_cast<std::size_t>(height)) {
        check_cuda(cu_.init(0), "Starting CUDA");
        check_cuda(cu_.device_get(&device_, cuda_device), "Finding the GPU");
        check_cuda(cu_.primary_retain(&context_, device_), "Taking the GPU");
        try {
            const Current current(*this);
            // Non-blocking: waited for by itself, not with libvmaf's own work
            // (which a wait for the whole context would wait for too).
            check_cuda(cu_.stream_create(&stream_, stream_non_blocking), "Starting the upload");
            for (void*& buffer : staging_) {
                check_cuda(cu_.host_alloc(&buffer, luma_bytes(), host_alloc_portable),
                           "Allocating the upload's memory");
            }
        } catch (...) {
            close();
            throw;
        }
    }

    ~CudaUpload() { close(); }
    CudaUpload(const CudaUpload&) = delete;
    CudaUpload& operator=(const CudaUpload&) = delete;

    // libvmaf's CUDA context, current on this thread while this lives.
    class Current {
    public:
        explicit Current(CudaUpload& upload) : cu_(upload.cu_) {
            check_cuda(cu_.push_current(upload.context_), "Taking the GPU");
        }
        ~Current() {
            CudaContext popped = nullptr;
            cu_.pop_current(&popped);
        }
        Current(const Current&) = delete;
        Current& operator=(const Current&) = delete;

    private:
        const CudaDriver& cu_;
    };

    // The bytes of a luma plane, its rows packed.
    std::size_t luma_bytes() const { return row_bytes_ * rows_; }

    // The page-locked memory of `side` (0 the source, 1 the test video),
    // to write a luma plane into, its rows packed, for send_staged.
    std::uint8_t* staging(int side) { return static_cast<std::uint8_t*>(staging_.at(static_cast<std::size_t>(side))); }

    // Starts copying `luma` (luma_bytes(), rows packed) to the GPU picture
    // at `address`, whose rows are `pitch` bytes apart. Call wait() before
    // sending on the same side again. Only with Current alive.
    void send(int side, const std::uint8_t* luma, std::uint64_t address, std::int64_t pitch) {
        std::memcpy(staging(side), luma, luma_bytes());
        send_staged(side, address, pitch);
    }

    // As send, for a plane already written into staging(side).
    void send_staged(int side, std::uint64_t address, std::int64_t pitch) {
        CudaCopy2D copy;
        copy.srcMemoryType = memory_host;
        copy.srcHost = staging(side);
        copy.srcPitch = row_bytes_;
        copy.dstMemoryType = memory_device;
        copy.dstDevice = address;
        copy.dstPitch = static_cast<std::size_t>(pitch);
        copy.WidthInBytes = row_bytes_;
        copy.Height = rows_;
        check_cuda(cu_.copy_2d_async(&copy, stream_), "Uploading a frame");
    }

    // Waits until the copies started have finished.
    void wait() { check_cuda(cu_.stream_synchronize(stream_), "Uploading a frame"); }

    // wait() where an error is already on its way: the copies' memory is
    // not given back before they have ended.
    void wait_quietly() { cu_.stream_synchronize(stream_); }

private:
    // Frees the buffers and the stream, and releases the GPU's context.
    void close() {
        if (context_ == nullptr) {
            return;
        }
        if (cu_.push_current(context_) == 0) {
            if (stream_ != nullptr) {
                cu_.stream_synchronize(stream_);
                cu_.stream_destroy(stream_);
            }
            for (void* buffer : staging_) {
                if (buffer != nullptr) {
                    cu_.free_host(buffer);
                }
            }
            CudaContext popped = nullptr;
            cu_.pop_current(&popped);
        }
        stream_ = nullptr;
        staging_ = {};
        cu_.primary_release(device_);
        context_ = nullptr;
    }

    const CudaDriver& cu_;
    std::size_t row_bytes_;
    std::size_t rows_;
    CudaDevice device_ = 0;
    CudaContext context_ = nullptr;
    CudaStream stream_ = nullptr;
    std::array<void*, 2> staging_{};
};

std::string cuda_library_build() { return "libvmaf-fast " + libvmaf_fast_version() + " (CUDA)"; }

CudaScorer::CudaScorer(int width, int height, int bit_depth,
                       const std::vector<std::pair<std::string, std::string>>& models, int n_subsample,
                       bool on_device)
    : library_(libvmaf()), step_(static_cast<unsigned>(std::max(1, n_subsample))) {
    if (library_.cuda_state_init == nullptr || library_.cuda_import_state == nullptr ||
        library_.cuda_preallocate_pictures == nullptr || library_.cuda_fetch_preallocated_picture == nullptr) {
        throw LibvmafError("the bundled libvmaf has no CUDA code");
    }
    const VmafConfiguration configuration{VMAF_LOG_LEVEL_ERROR, 0, step_, 0, 0};
    // When vmaf_init fails it frees the context it made but leaves the
    // pointer to it set, so the context is only kept once it has started.
    VmafContext* context = nullptr;
    check_libvmaf(library_.init(&context, configuration), "Starting libvmaf");
    context_ = context;
    try {
        // libvmaf keeps a copy of the state; the state itself, a few bytes,
        // is never freed, as in libvmaf's own vmaf tool.
        VmafCudaState* state = nullptr;
        check_libvmaf(library_.cuda_state_init(&state, VmafCudaConfiguration{nullptr}), "Starting CUDA");
        check_libvmaf(library_.cuda_import_state(context_, state), "Starting CUDA");
        for (const auto& [name, version] : models) {
            // Loading a model can fail after making it, and leaves it set; it
            // is then destroyed like a loaded one.
            VmafModel* model = nullptr;
            VmafModelConfig config{name.c_str(), 0};
            const int error = library_.model_load(&model, &config, version.c_str());
            if (model != nullptr) {
                models_.emplace_back(name, model);
            }
            check_libvmaf(error, "Loading the " + version + " model");
            check_libvmaf(library_.use_features_from_model(context_, model), "Setting up " + version);
        }
        // libvmaf's pictures in GPU memory, filled by NVIDIA's decoder, or
        // by the upload from frames in system memory.
        VmafCudaPictureConfiguration pictures{};
        pictures.pic_params.w = static_cast<unsigned>(width);
        pictures.pic_params.h = static_cast<unsigned>(height);
        pictures.pic_params.bpc = static_cast<unsigned>(bit_depth);
        pictures.pic_params.pix_fmt = VMAF_PIX_FMT_YUV420P;
        pictures.pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE;
        check_libvmaf(library_.cuda_preallocate_pictures(context_, pictures), "Allocating pictures");
        if (!on_device) {
            upload_ = std::make_unique<CudaUpload>(width, height, bit_depth <= 8 ? 1 : 2);
        }
    } catch (...) {
        close();
        throw;
    }
    // A frame as FFmpeg writes it: the luma plane, then the chroma planes,
    // whose size FFmpeg rounds up for an odd width or height.
    const auto w = static_cast<std::size_t>(width), h = static_cast<std::size_t>(height);
    frame_bytes_ = (w * h + 2 * ((w + 1) / 2) * ((h + 1) / 2)) * (bit_depth <= 8 ? 1 : 2);
}

CudaScorer::~CudaScorer() { close(); }

void CudaScorer::close() {
    upload_.reset();
    for (const auto& [name, model] : models_) {
        library_.model_destroy(model);
    }
    models_.clear();
    if (context_ != nullptr) {
        library_.close(context_);
        context_ = nullptr;
    }
}

void CudaScorer::add_pictures(const std::function<void(VmafPicture&, VmafPicture&)>& fill) {
    VmafPicture reference{};
    VmafPicture distorted{};
    check_libvmaf(library_.cuda_fetch_preallocated_picture(context_, &reference), "Taking a picture");
    try {
        check_libvmaf(library_.cuda_fetch_preallocated_picture(context_, &distorted), "Taking a picture");
    } catch (...) {
        library_.picture_unref(&reference);
        throw;
    }
    try {
        fill(reference, distorted);
    } catch (...) {
        // Pictures taken from the pool must be given back: vmaf_close would
        // wait for them forever.
        library_.picture_unref(&reference);
        library_.picture_unref(&distorted);
        throw;
    }
    // libvmaf takes both pictures, even when it fails (pull request 1652).
    const int error = library_.read_pictures(context_, &reference, &distorted, count_);
    if (error != 0) {
        check_libvmaf(error, "Scoring frame " + std::to_string(count_));
    }
    ++count_;
}

void CudaScorer::add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) {
    if (!upload_) {
        throw LibvmafError("this scorer takes pictures in GPU memory only");
    }
    CudaUpload& upload = *upload_;
    for (const std::span<const std::uint8_t> frame : {reference, distorted}) {
        if (frame.size() < upload.luma_bytes()) {
            throw LibvmafError("a frame of " + std::to_string(frame.size()) + " bytes, where its luma alone is " +
                               std::to_string(upload.luma_bytes()));
        }
    }
    const CudaUpload::Current current(upload);
    add_pictures([&](VmafPicture& source, VmafPicture& test) {
        // Both copies run at the same time, and the wait covers both. When
        // one cannot start, the other's is waited for before its picture
        // goes back to the pool.
        try {
            upload.send(0, reference.data(), luma_address(source), source.stride[0]);
            upload.send(1, distorted.data(), luma_address(test), test.stride[0]);
        } catch (...) {
            upload.wait_quietly();
            throw;
        }
        upload.wait();
    });
}

void CudaScorer::add_decoded(const Decoded& reference, const Decoded& distorted) {
    if (reference.copy && distorted.copy) {
        // Both on the GPU: no upload, and NVIDIA's decoder makes the
        // context current itself.
        add_pictures([&](VmafPicture& source, VmafPicture& test) {
            reference.copy(luma_address(source), source.stride[0]);
            distorted.copy(luma_address(test), test.stride[0]);
        });
        return;
    }
    if (!upload_) {
        throw LibvmafError("this scorer takes pictures in GPU memory only");
    }
    CudaUpload& upload = *upload_;
    // A picture copied on the GPU, or written into the side's page-locked
    // memory and uploaded from it.
    const auto give = [&upload](const Decoded& side, int index, VmafPicture& picture) {
        if (side.copy) {
            side.copy(luma_address(picture), picture.stride[0]);
        } else {
            side.write(upload.staging(index));
            upload.send_staged(index, luma_address(picture), picture.stride[0]);
        }
    };
    const CudaUpload::Current current(upload);
    add_pictures([&](VmafPicture& source, VmafPicture& test) {
        try {
            give(reference, 0, source);
            give(distorted, 1, test);
        } catch (...) {
            upload.wait_quietly();
            throw;
        }
        upload.wait();
    });
}

PairScores CudaScorer::finish() {
    PairScores result;
    if (count_ == 0) {
        for (const auto& [name, model] : models_) {
            result.by_metric.emplace_back(name, std::vector<double>());
        }
        return result;
    }
    check_libvmaf(library_.read_pictures(context_, nullptr, nullptr, 0), "Finishing");
    for (unsigned frame = 0; frame < count_; frame += step_) {
        result.frames.push_back(frame);
    }
    for (const auto& [name, model] : models_) {
        std::vector<double> scores;
        scores.reserve(result.frames.size());
        for (const unsigned frame : result.frames) {
            double value = 0.0;
            const int error = library_.score_at_index(context_, model, &value, frame);
            if (error != 0) {
                check_libvmaf(error, "Reading " + name + " for frame " + std::to_string(frame));
            }
            scores.push_back(to_6_decimals(value));
        }
        result.by_metric.emplace_back(name, std::move(scores));
    }
    return result;
}

CudaProbe probe_cuda() {
    std::error_code missing;
    if (!std::filesystem::is_regular_file(libvmaf_folder() / platform::shared_library_name("libvmaf"), missing)) {
        return {false, "libvmaf with CUDA is not bundled"};
    }
    try {
        const char* version = libvmaf().version();
        // Small frames with flat, smooth and textured areas, the test video's
        // blurred, noisy and sharpened in turn: the Vulkan probe's own.
        constexpr int width = 256, height = 144;
        const VulkanProbeFrames frames = vulkan_probe_frames(8, 3, width, height);
        CudaScorer scorer(width, height, 8, {{"vmaf", "vmaf_v0.6.1"}});
        for (std::size_t i = 0; i < frames.reference.size(); ++i) {
            scorer.add(frames.reference[i], frames.distorted[i]);
        }
        const PairScores scored = scorer.finish();
        const std::vector<double>& vmaf = scored.by_metric.at(0).second;
        if (vmaf.size() != frames.reference.size() ||
            !std::all_of(vmaf.begin(), vmaf.end(), [](double score) { return std::isfinite(score); })) {
            return {false, "libvmaf's GPU test scored nothing"};
        }
        return {true, "libvmaf " + std::string(version != nullptr ? version : "")};
    } catch (const LibvmafError& error) {
        return {false, error.what()};
    }
}

}  // namespace vml
