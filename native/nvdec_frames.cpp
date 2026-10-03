// nvdec_frames: decodes a video on an NVIDIA GPU (NVDEC) and hands back its
// pictures in GPU memory, cropped and laid out as the GPU metrics read them,
// without a copy through system memory or a CPU conversion.
//
// The app feeds it the compressed packets (FFmpeg copies the stream out of
// its container: vmaf_app/core/nvdec_frames.py) with their timestamps. NVIDIA's
// parser (nvcuvid) finds the pictures in them, the decoder decodes them, and
// each picture, in display order, is converted by a small kernel from the
// decoder's NV12/P016 surface into a slot of a pool in GPU memory: the luma,
// then the U and V planes, each packed (pitch = width). From there a slot is
// copied, by the GPU's copy engines, into a page-locked buffer of Vship's
// (nvf_download) or into a libvmaf picture on the GPU (nvf_copy_luma).
//
// The conversions are exact: the samples are moved, never computed, except
// for a right shift by 6 that turns P016's 10-bit samples (in the top bits)
// into the low-bit layout of yuv420p10le, which is what FFmpeg's own
// conversion does (checked frame by frame against FFmpeg).
//
// Everything runs in the device's primary CUDA context: Vship (CUDA runtime)
// and libvmaf use that one too, so their memory and this pool are one address
// space. nvcuda.dll and nvcuvid.dll come with the NVIDIA driver and are
// loaded at run time; this library links neither. The kernels are PTX for
// sm_30 at PTX ISA 6.0, which every driver since CUDA 9 compiles for whatever
// GPU it runs on (the CUDA toolkit's compiler no longer targets GPUs older
// than Turing; NVDEC goes back much further).
//
// Threads: nvf_push (and nvf_finish) from one thread, which also runs the
// parser's callbacks and so the conversions; nvf_pop / nvf_download /
// nvf_copy_luma / nvf_release from another; nvf_abort from any.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "ffnvcodec/dynlink_cuda.h"
#include "ffnvcodec/dynlink_nvcuvid.h"
#include "gpu_frames.h"

namespace {

// ------------------------------------------------------------------ driver

struct Driver {
    HMODULE cuda = nullptr, cuvid = nullptr;
    tcuInit *cuInit;
    tcuDeviceGetCount *cuDeviceGetCount;
    tcuDeviceGet *cuDeviceGet;
    tcuDevicePrimaryCtxRetain *cuDevicePrimaryCtxRetain;
    tcuDevicePrimaryCtxRelease *cuDevicePrimaryCtxRelease;
    tcuDevicePrimaryCtxSetFlags *cuDevicePrimaryCtxSetFlags;
    tcuCtxPushCurrent_v2 *cuCtxPushCurrent;
    tcuCtxPopCurrent_v2 *cuCtxPopCurrent;
    tcuMemAlloc_v2 *cuMemAlloc;
    tcuMemFree_v2 *cuMemFree;
    tcuMemcpy2DAsync_v2 *cuMemcpy2DAsync;
    tcuMemcpyDtoHAsync_v2 *cuMemcpyDtoHAsync;
    tcuStreamCreate *cuStreamCreate;
    tcuStreamDestroy_v2 *cuStreamDestroy;
    tcuEventCreate *cuEventCreate;
    tcuEventDestroy_v2 *cuEventDestroy;
    tcuEventRecord *cuEventRecord;
    tcuEventSynchronize *cuEventSynchronize;
    tcuModuleLoadData *cuModuleLoadData;
    tcuModuleUnload *cuModuleUnload;
    tcuModuleGetFunction *cuModuleGetFunction;
    tcuLaunchKernel *cuLaunchKernel;
    tcuGetErrorName *cuGetErrorName;
    tcuvidGetDecoderCaps *cuvidGetDecoderCaps;
    tcuvidCreateDecoder *cuvidCreateDecoder;
    tcuvidDestroyDecoder *cuvidDestroyDecoder;
    tcuvidDecodePicture *cuvidDecodePicture;
    tcuvidGetDecodeStatus *cuvidGetDecodeStatus;
    tcuvidMapVideoFrame64 *cuvidMapVideoFrame;
    tcuvidUnmapVideoFrame64 *cuvidUnmapVideoFrame;
    tcuvidCtxLockCreate *cuvidCtxLockCreate;
    tcuvidCtxLockDestroy *cuvidCtxLockDestroy;
    tcuvidCreateVideoParser *cuvidCreateVideoParser;
    tcuvidParseVideoData *cuvidParseVideoData;
    tcuvidDestroyVideoParser *cuvidDestroyVideoParser;
};

Driver g_driver;
std::mutex g_driver_lock;
bool g_driver_loaded = false;
std::string g_driver_error;

template <typename T>
bool load_symbol(HMODULE module, T *&target, const char *name) {
    target = reinterpret_cast<T *>(reinterpret_cast<void *>(GetProcAddress(module, name)));
    if (!target) g_driver_error = std::string("the NVIDIA driver has no ") + name;
    return target != nullptr;
}

bool load_driver() {
    std::lock_guard<std::mutex> lock(g_driver_lock);
    if (g_driver_loaded) return true;
    if (!g_driver_error.empty()) return false;
    Driver &d = g_driver;
    d.cuda = LoadLibraryW(L"nvcuda.dll");
    if (!d.cuda) { g_driver_error = "nvcuda.dll (the NVIDIA driver's CUDA library) is not installed"; return false; }
    d.cuvid = LoadLibraryW(L"nvcuvid.dll");
    if (!d.cuvid) { g_driver_error = "nvcuvid.dll (the NVIDIA driver's video decoder library) is not installed"; return false; }
    bool ok = load_symbol(d.cuda, d.cuInit, "cuInit")
        && load_symbol(d.cuda, d.cuDeviceGetCount, "cuDeviceGetCount")
        && load_symbol(d.cuda, d.cuDeviceGet, "cuDeviceGet")
        && load_symbol(d.cuda, d.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain")
        && load_symbol(d.cuda, d.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2")
        && load_symbol(d.cuda, d.cuDevicePrimaryCtxSetFlags, "cuDevicePrimaryCtxSetFlags_v2")
        && load_symbol(d.cuda, d.cuCtxPushCurrent, "cuCtxPushCurrent_v2")
        && load_symbol(d.cuda, d.cuCtxPopCurrent, "cuCtxPopCurrent_v2")
        && load_symbol(d.cuda, d.cuMemAlloc, "cuMemAlloc_v2")
        && load_symbol(d.cuda, d.cuMemFree, "cuMemFree_v2")
        && load_symbol(d.cuda, d.cuMemcpy2DAsync, "cuMemcpy2DAsync_v2")
        && load_symbol(d.cuda, d.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2")
        && load_symbol(d.cuda, d.cuStreamCreate, "cuStreamCreate")
        && load_symbol(d.cuda, d.cuStreamDestroy, "cuStreamDestroy_v2")
        && load_symbol(d.cuda, d.cuEventCreate, "cuEventCreate")
        && load_symbol(d.cuda, d.cuEventDestroy, "cuEventDestroy_v2")
        && load_symbol(d.cuda, d.cuEventRecord, "cuEventRecord")
        && load_symbol(d.cuda, d.cuEventSynchronize, "cuEventSynchronize")
        && load_symbol(d.cuda, d.cuModuleLoadData, "cuModuleLoadData")
        && load_symbol(d.cuda, d.cuModuleUnload, "cuModuleUnload")
        && load_symbol(d.cuda, d.cuModuleGetFunction, "cuModuleGetFunction")
        && load_symbol(d.cuda, d.cuLaunchKernel, "cuLaunchKernel")
        && load_symbol(d.cuda, d.cuGetErrorName, "cuGetErrorName")
        && load_symbol(d.cuvid, d.cuvidGetDecoderCaps, "cuvidGetDecoderCaps")
        && load_symbol(d.cuvid, d.cuvidCreateDecoder, "cuvidCreateDecoder")
        && load_symbol(d.cuvid, d.cuvidDestroyDecoder, "cuvidDestroyDecoder")
        && load_symbol(d.cuvid, d.cuvidDecodePicture, "cuvidDecodePicture")
        && load_symbol(d.cuvid, d.cuvidGetDecodeStatus, "cuvidGetDecodeStatus")
        && load_symbol(d.cuvid, d.cuvidMapVideoFrame, "cuvidMapVideoFrame64")
        && load_symbol(d.cuvid, d.cuvidUnmapVideoFrame, "cuvidUnmapVideoFrame64")
        && load_symbol(d.cuvid, d.cuvidCtxLockCreate, "cuvidCtxLockCreate")
        && load_symbol(d.cuvid, d.cuvidCtxLockDestroy, "cuvidCtxLockDestroy")
        && load_symbol(d.cuvid, d.cuvidCreateVideoParser, "cuvidCreateVideoParser")
        && load_symbol(d.cuvid, d.cuvidParseVideoData, "cuvidParseVideoData")
        && load_symbol(d.cuvid, d.cuvidDestroyVideoParser, "cuvidDestroyVideoParser");
    if (!ok) return false;
    CUresult result = d.cuInit(0);
    if (result != CUDA_SUCCESS) {
        g_driver_error = "CUDA could not start (cuInit error " + std::to_string(static_cast<int>(result)) + ")";
        return false;
    }
    g_driver_loaded = true;
    return true;
}

std::string cuda_error(CUresult result) {
    const char *name = nullptr;
    if (g_driver.cuGetErrorName) g_driver.cuGetErrorName(result, &name);
    return name ? std::string(name) : "CUDA error " + std::to_string(static_cast<int>(result));
}

// ----------------------------------------------------------------- kernels

// One thread per output sample: dst[y][x] = src[y * src_pitch + x * src_step]
// (bytes), and for 16-bit samples shifted right by `shift`. U and V are two
// launches over the interleaved chroma plane, src_step 2 samples apart.
const char kKernels[] = R"PTX(
.version 6.0
.target sm_30
.address_size 64

.visible .entry plane8(
    .param .u64 p_src, .param .u32 p_src_pitch, .param .u32 p_src_step,
    .param .u64 p_dst, .param .u32 p_dst_pitch,
    .param .u32 p_width, .param .u32 p_height)
{
    .reg .pred %p<3>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %ctaid.y;
    mov.u32 %r6, %ntid.y;
    mov.u32 %r7, %tid.y;
    mad.lo.s32 %r8, %r5, %r6, %r7;
    ld.param.u32 %r9, [p_width];
    ld.param.u32 %r10, [p_height];
    setp.ge.u32 %p1, %r4, %r9;
    setp.ge.u32 %p2, %r8, %r10;
    or.pred %p1, %p1, %p2;
    @%p1 bra PLANE8_DONE;
    ld.param.u64 %rd1, [p_src];
    ld.param.u32 %r11, [p_src_pitch];
    ld.param.u32 %r12, [p_src_step];
    ld.param.u64 %rd2, [p_dst];
    ld.param.u32 %r13, [p_dst_pitch];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mul.wide.u32 %rd3, %r8, %r11;
    mul.wide.u32 %rd4, %r4, %r12;
    add.s64 %rd5, %rd1, %rd3;
    add.s64 %rd5, %rd5, %rd4;
    ld.global.u8 %rs1, [%rd5];
    mul.wide.u32 %rd6, %r8, %r13;
    cvt.u64.u32 %rd7, %r4;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd8, %rd8, %rd7;
    st.global.u8 [%rd8], %rs1;
PLANE8_DONE:
    ret;
}

.visible .entry plane16(
    .param .u64 p_src, .param .u32 p_src_pitch, .param .u32 p_src_step,
    .param .u64 p_dst, .param .u32 p_dst_pitch,
    .param .u32 p_width, .param .u32 p_height, .param .u32 p_shift)
{
    .reg .pred %p<3>;
    .reg .b16 %rs<3>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %ctaid.y;
    mov.u32 %r6, %ntid.y;
    mov.u32 %r7, %tid.y;
    mad.lo.s32 %r8, %r5, %r6, %r7;
    ld.param.u32 %r9, [p_width];
    ld.param.u32 %r10, [p_height];
    setp.ge.u32 %p1, %r4, %r9;
    setp.ge.u32 %p2, %r8, %r10;
    or.pred %p1, %p1, %p2;
    @%p1 bra PLANE16_DONE;
    ld.param.u64 %rd1, [p_src];
    ld.param.u32 %r11, [p_src_pitch];
    ld.param.u32 %r12, [p_src_step];
    ld.param.u64 %rd2, [p_dst];
    ld.param.u32 %r13, [p_dst_pitch];
    ld.param.u32 %r14, [p_shift];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mul.wide.u32 %rd3, %r8, %r11;
    mul.wide.u32 %rd4, %r4, %r12;
    add.s64 %rd5, %rd1, %rd3;
    add.s64 %rd5, %rd5, %rd4;
    ld.global.u16 %rs1, [%rd5];
    shr.u16 %rs2, %rs1, %r14;
    mul.wide.u32 %rd6, %r8, %r13;
    mul.wide.u32 %rd7, %r4, 2;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd8, %rd8, %rd7;
    st.global.u16 [%rd8], %rs2;
PLANE16_DONE:
    ret;
}
)PTX";

constexpr unsigned kBlockX = 32, kBlockY = 8;

// ----------------------------------------------------------------- decoder

struct Ready {
    int slot;
    long long pts;
};

struct Decoder {
    Params params{};
    Info info{};
    std::vector<unsigned char> extradata;
    CUdevice device = 0;
    CUcontext context = nullptr;
    bool retained = false;
    CUstream decode_stream = nullptr, output_stream = nullptr;
    CUevent decode_event = nullptr, output_event = nullptr;
    CUmodule module = nullptr;
    CUfunction plane8 = nullptr, plane16 = nullptr;
    CUvideoctxlock lock = nullptr;
    CUvideoparser parser = nullptr;
    CUvideodecoder decoder = nullptr;
    CUdeviceptr pool = 0;
    size_t frame_bytes = 0, luma_bytes = 0, chroma_bytes = 0;
    int bytes_per_sample = 1;
    int surface_height = 0;

    std::mutex mutex;
    std::condition_variable changed;
    std::deque<int> free_slots;
    std::deque<Ready> ready;
    bool ended = false, aborted = false, failed = false;
    std::string error;

    void fail(const std::string &text) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!failed) error = text;
        failed = true;
        changed.notify_all();
    }

    bool check(CUresult result, const char *what) {
        if (result == CUDA_SUCCESS) return true;
        fail(std::string(what) + " failed: " + cuda_error(result));
        return false;
    }

    bool stopped() {
        std::lock_guard<std::mutex> guard(mutex);
        return failed || aborted;
    }
};

// Pushes the decoder's context on the calling thread for one scope.
struct ContextScope {
    bool pushed;
    explicit ContextScope(Decoder *d) : pushed(g_driver.cuCtxPushCurrent(d->context) == CUDA_SUCCESS) {}
    ~ContextScope() {
        if (pushed) {
            CUcontext ignored;
            g_driver.cuCtxPopCurrent(&ignored);
        }
    }
};

int CUDAAPI on_sequence(void *user, CUVIDEOFORMAT *format) {
    Decoder *d = static_cast<Decoder *>(user);
    const Driver &cu = g_driver;
    int depth = format->bit_depth_luma_minus8 + 8;
    if (d->decoder) {
        // A new sequence header: the same format carries on; any change of
        // size or depth mid-stream is the FFmpeg path's to handle.
        if (static_cast<int>(format->coded_width) != d->info.coded_width
            || static_cast<int>(format->coded_height) != d->info.coded_height
            || depth != d->info.bit_depth || format->chroma_format != d->info.chroma_format) {
            d->fail("the video's format changes partway through");
            return 0;
        }
        return d->info.decode_surfaces;
    }
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        d->info.coded_width = format->coded_width;
        d->info.coded_height = format->coded_height;
        d->info.display_left = format->display_area.left;
        d->info.display_top = format->display_area.top;
        d->info.display_right = format->display_area.right;
        d->info.display_bottom = format->display_area.bottom;
        d->info.bit_depth = depth;
        d->info.chroma_format = format->chroma_format;
        d->info.progressive = format->progressive_sequence;
    }

    char text[256];
    if (format->chroma_format != cudaVideoChromaFormat_420) {
        d->fail("the video is not 4:2:0");
        return 0;
    }
    if (depth != d->params.bit_depth || format->bit_depth_chroma_minus8 != format->bit_depth_luma_minus8) {
        snprintf(text, sizeof text, "the video decodes at %d bits, not %d", depth, d->params.bit_depth);
        d->fail(text);
        return 0;
    }
    if (!format->progressive_sequence) {
        d->fail("the video is interlaced");
        return 0;
    }
    // The pictures FFmpeg hands on are the display area. Only its right and
    // bottom edges are cut here; a display area offset from the top left is
    // left to the FFmpeg path, as are odd coded sizes (NVDEC scales to an
    // even target size).
    if (format->display_area.left != 0 || format->display_area.top != 0
        || format->display_area.right != d->params.width || format->display_area.bottom != d->params.height
        || (format->coded_width & 1) || (format->coded_height & 1)) {
        snprintf(text, sizeof text, "the decoder's picture (%ux%u, showing %d,%d-%d,%d) is not the %dx%d video",
                 format->coded_width, format->coded_height, format->display_area.left, format->display_area.top,
                 format->display_area.right, format->display_area.bottom, d->params.width, d->params.height);
        d->fail(text);
        return 0;
    }
    if (d->params.crop_x + d->params.crop_w > d->params.width || d->params.crop_y + d->params.crop_h > d->params.height) {
        d->fail("the crop is outside the picture");
        return 0;
    }

    CUVIDDECODECAPS caps{};
    caps.eCodecType = format->codec;
    caps.eChromaFormat = format->chroma_format;
    caps.nBitDepthMinus8 = format->bit_depth_luma_minus8;
    if (!d->check(cu.cuvidGetDecoderCaps(&caps), "Asking the GPU's decoder what it supports")) return 0;
    cudaVideoSurfaceFormat surface = depth > 8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    if (!caps.bIsSupported || !(caps.nOutputFormatMask & (1 << surface))
        || format->coded_width > caps.nMaxWidth || format->coded_height > caps.nMaxHeight
        || (format->coded_width >> 4) * (format->coded_height >> 4) > caps.nMaxMBCount
        || format->coded_width < caps.nMinWidth || format->coded_height < caps.nMinHeight) {
        snprintf(text, sizeof text, "this GPU's decoder cannot decode this video (%ux%u, %d-bit)",
                 format->coded_width, format->coded_height, depth);
        d->fail(text);
        return 0;
    }

    // Enough surfaces for the stream's references, the parser's display
    // delay and the picture being converted.
    int surfaces = format->min_num_decode_surfaces + 4 + 1;
    if (surfaces < 8) surfaces = 8;
    CUVIDDECODECREATEINFO create{};
    create.ulWidth = format->coded_width;
    create.ulHeight = format->coded_height;
    create.ulNumDecodeSurfaces = surfaces;
    create.CodecType = format->codec;
    create.ChromaFormat = format->chroma_format;
    create.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    create.bitDepthMinus8 = format->bit_depth_luma_minus8;
    create.ulMaxWidth = format->coded_width;
    create.ulMaxHeight = format->coded_height;
    // The whole coded picture, at its own size: no scaling.
    create.display_area.left = 0;
    create.display_area.top = 0;
    create.display_area.right = static_cast<short>(format->coded_width);
    create.display_area.bottom = static_cast<short>(format->coded_height);
    create.OutputFormat = surface;
    create.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    create.ulTargetWidth = format->coded_width;
    create.ulTargetHeight = format->coded_height;
    create.ulNumOutputSurfaces = 2;
    create.vidLock = d->lock;
    if (!d->check(cu.cuvidCreateDecoder(&d->decoder, &create), "Starting the GPU's decoder")) {
        d->decoder = nullptr;
        return 0;
    }
    d->surface_height = (static_cast<int>(format->coded_height) + 1) & ~1;
    std::lock_guard<std::mutex> guard(d->mutex);
    d->info.decode_surfaces = surfaces;
    return surfaces;
}

int CUDAAPI on_decode(void *user, CUVIDPICPARAMS *picture) {
    Decoder *d = static_cast<Decoder *>(user);
    if (!d->decoder || d->stopped()) return 0;
    if (!d->check(g_driver.cuvidDecodePicture(d->decoder, picture), "Decoding a picture")) return 0;
    std::lock_guard<std::mutex> guard(d->mutex);
    d->info.decoded++;
    return 1;
}

bool launch(Decoder *d, CUfunction kernel, CUdeviceptr src, unsigned src_pitch, unsigned src_step,
            CUdeviceptr dst, unsigned dst_pitch, unsigned width, unsigned height, unsigned shift) {
    void *args[] = {&src, &src_pitch, &src_step, &dst, &dst_pitch, &width, &height, &shift};
    return d->check(g_driver.cuLaunchKernel(kernel, (width + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY, 1,
                                            kBlockX, kBlockY, 1, 0, d->decode_stream, args, nullptr),
                    "Converting a picture");
}

int CUDAAPI on_display(void *user, CUVIDPARSERDISPINFO *display) {
    Decoder *d = static_cast<Decoder *>(user);
    const Driver &cu = g_driver;
    if (!d->decoder) return 0;
    int slot;
    {
        std::unique_lock<std::mutex> guard(d->mutex);
        d->changed.wait(guard, [d] { return !d->free_slots.empty() || d->aborted || d->failed; });
        if (d->aborted || d->failed) return 0;
        slot = d->free_slots.front();
        d->free_slots.pop_front();
    }
    auto give_back = [d, slot] {
        std::lock_guard<std::mutex> guard(d->mutex);
        d->free_slots.push_front(slot);
    };

    CUVIDGETDECODESTATUS status{};
    if (cu.cuvidGetDecodeStatus(d->decoder, display->picture_index, &status) == CUDA_SUCCESS
        && (status.decodeStatus == cuvidDecodeStatus_Error || status.decodeStatus == cuvidDecodeStatus_Error_Concealed)) {
        // FFmpeg would conceal the damage its own way: the pictures would
        // not be the ones its decoder makes.
        give_back();
        d->fail("the GPU's decoder found an error in the video");
        return 0;
    }

    CUVIDPROCPARAMS process{};
    process.progressive_frame = display->progressive_frame;
    process.second_field = display->repeat_first_field + 1;
    process.top_field_first = display->top_field_first;
    process.unpaired_field = display->repeat_first_field < 0;
    process.output_stream = d->decode_stream;
    unsigned long long surface = 0;
    unsigned pitch = 0;
    if (!d->check(cu.cuvidMapVideoFrame(d->decoder, display->picture_index, &surface, &pitch, &process),
                  "Reading a decoded picture")) {
        give_back();
        return 0;
    }

    const Params &p = d->params;
    const unsigned bps = d->bytes_per_sample;
    CUdeviceptr dst = d->pool + static_cast<size_t>(slot) * d->frame_bytes;
    CUdeviceptr luma = surface + static_cast<size_t>(p.crop_y) * pitch + static_cast<size_t>(p.crop_x) * bps;
    bool ok;
    if (bps == 1) {
        ok = launch(d, d->plane8, luma, pitch, 1, dst, p.crop_w, p.crop_w, p.crop_h, 0);
    } else {
        ok = launch(d, d->plane16, luma, pitch, 2, dst, p.crop_w * 2, p.crop_w, p.crop_h, p.shift);
    }
    if (ok && !p.luma_only) {
        unsigned chroma_w = (p.crop_w + 1) / 2, chroma_h = (p.crop_h + 1) / 2;
        CUdeviceptr chroma = surface + static_cast<size_t>(pitch) * d->surface_height
                             + static_cast<size_t>(p.crop_y / 2) * pitch + static_cast<size_t>(p.crop_x / 2) * 2 * bps;
        CUdeviceptr u = dst + d->luma_bytes, v = u + d->chroma_bytes;
        CUfunction kernel = bps == 1 ? d->plane8 : d->plane16;
        ok = launch(d, kernel, chroma, pitch, 2 * bps, u, chroma_w * bps, chroma_w, chroma_h, p.shift)
             && launch(d, kernel, chroma + bps, pitch, 2 * bps, v, chroma_w * bps, chroma_w, chroma_h, p.shift);
    }
    // The surface may only be unmapped once the kernels have read it.
    ok = ok && d->check(cu.cuEventRecord(d->decode_event, d->decode_stream), "Converting a picture")
         && d->check(cu.cuEventSynchronize(d->decode_event), "Converting a picture");
    CUresult unmapped = cu.cuvidUnmapVideoFrame(d->decoder, surface);
    if (!ok || !d->check(unmapped, "Releasing a decoded picture")) {
        give_back();
        return 0;
    }
    std::lock_guard<std::mutex> guard(d->mutex);
    d->ready.push_back({slot, display->timestamp});
    d->info.displayed++;
    d->changed.notify_all();
    return 1;
}

int CUDAAPI on_operating_point(void *, CUVIDOPERATINGPOINTINFO *) {
    return 0;  // operating point 0, its own layers only: what FFmpeg decodes by default
}

void destroy(Decoder *d) {
    const Driver &cu = g_driver;
    if (d->context) {
        ContextScope scope(d);
        if (d->parser) cu.cuvidDestroyVideoParser(d->parser);
        if (d->decoder) cu.cuvidDestroyDecoder(d->decoder);
        if (d->lock) cu.cuvidCtxLockDestroy(d->lock);
        if (d->pool) cu.cuMemFree(d->pool);
        if (d->module) cu.cuModuleUnload(d->module);
        if (d->decode_event) cu.cuEventDestroy(d->decode_event);
        if (d->output_event) cu.cuEventDestroy(d->output_event);
        if (d->decode_stream) cu.cuStreamDestroy(d->decode_stream);
        if (d->output_stream) cu.cuStreamDestroy(d->output_stream);
    }
    if (d->retained) cu.cuDevicePrimaryCtxRelease(d->device);
    delete d;
}

void copy_text(char *out, int size, const std::string &text) {
    if (out && size > 0) snprintf(out, size, "%s", text.c_str());
}

}  // namespace

// --------------------------------------------------------------------- API

// A decoder for one video, or null with the reason in `error`.
NVF_API void *nvf_open(const Params *params, char *error, int error_size) {
    if (!load_driver()) {
        copy_text(error, error_size, g_driver_error);
        return nullptr;
    }
    const Driver &cu = g_driver;
    Params p = *params;
    if (!params_valid(p)) {
        copy_text(error, error_size, "invalid decoder parameters");
        return nullptr;
    }
    Decoder *d = new Decoder();
    d->params = p;
    if (p.extradata && p.extradata_size > 0) d->extradata.assign(p.extradata, p.extradata + p.extradata_size);
    d->params.extradata = nullptr;
    d->bytes_per_sample = p.bit_depth > 8 ? 2 : 1;
    d->luma_bytes = static_cast<size_t>(p.crop_w) * p.crop_h * d->bytes_per_sample;
    d->chroma_bytes = p.luma_only ? 0 : static_cast<size_t>((p.crop_w + 1) / 2) * ((p.crop_h + 1) / 2) * d->bytes_per_sample;
    d->frame_bytes = d->luma_bytes + 2 * d->chroma_bytes;
    d->info.frame_bytes = static_cast<long long>(d->frame_bytes);

    auto give_up = [&](const std::string &text) {
        copy_text(error, error_size, text);
        destroy(d);
        return nullptr;
    };
    CUresult result = cu.cuDeviceGet(&d->device, p.device);
    if (result != CUDA_SUCCESS) return give_up("no CUDA device " + std::to_string(p.device) + ": " + cuda_error(result));
    // Waits for the GPU then sleep instead of spinning a CPU core -- also the
    // driver's own, inside the decoder calls. Refused once the context is
    // running with other flags (Vship may have started it): the events
    // below are blocking-sync either way.
    cu.cuDevicePrimaryCtxSetFlags(d->device, CU_CTX_SCHED_BLOCKING_SYNC);
    result = cu.cuDevicePrimaryCtxRetain(&d->context, d->device);
    if (result != CUDA_SUCCESS) return give_up("CUDA could not start on the GPU: " + cuda_error(result));
    d->retained = true;
    const char *step = "Starting CUDA";
    {
        ContextScope scope(d);
        result = scope.pushed ? cu.cuStreamCreate(&d->decode_stream, CU_STREAM_NON_BLOCKING) : static_cast<CUresult>(201);  // CUDA_ERROR_INVALID_CONTEXT
        if (result == CUDA_SUCCESS) result = cu.cuStreamCreate(&d->output_stream, CU_STREAM_NON_BLOCKING);
        if (result == CUDA_SUCCESS) result = cu.cuEventCreate(&d->decode_event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING);
        if (result == CUDA_SUCCESS) result = cu.cuEventCreate(&d->output_event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING);
        if (result == CUDA_SUCCESS) {
            step = "Loading the GPU conversion kernels";
            result = cu.cuModuleLoadData(&d->module, kKernels);
        }
        if (result == CUDA_SUCCESS) result = cu.cuModuleGetFunction(&d->plane8, d->module, "plane8");
        if (result == CUDA_SUCCESS) result = cu.cuModuleGetFunction(&d->plane16, d->module, "plane16");
        if (result == CUDA_SUCCESS) {
            step = "Allocating GPU memory for the decoded pictures";
            result = cu.cuMemAlloc(&d->pool, d->frame_bytes * p.pool);
        }
        if (result == CUDA_SUCCESS) {
            step = "Starting the GPU's decoder";
            result = cu.cuvidCtxLockCreate(&d->lock, d->context);
        }
    }
    if (result != CUDA_SUCCESS) return give_up(std::string(step) + " failed: " + cuda_error(result));
    for (int slot = 0; slot < p.pool; slot++) d->free_slots.push_back(slot);

    CUVIDEOFORMATEX extension{};
    CUVIDPARSERPARAMS parser{};
    parser.CodecType = static_cast<cudaVideoCodec>(p.codec);
    parser.ulMaxNumDecodeSurfaces = 1;  // the sequence callback sets the real number
    parser.ulClockRate = 0;
    parser.ulMaxDisplayDelay = 4;       // decode ahead of display, as FFmpeg's cuvid decoder does
    parser.pUserData = d;
    parser.pfnSequenceCallback = on_sequence;
    parser.pfnDecodePicture = on_decode;
    parser.pfnDisplayPicture = on_display;
    parser.pfnGetOperatingPoint = on_operating_point;
    if (!d->extradata.empty()) {
        if (d->extradata.size() > sizeof extension.raw_seqhdr_data) return give_up("the video's sequence header is too long");
        memcpy(extension.raw_seqhdr_data, d->extradata.data(), d->extradata.size());
        extension.format.seqhdr_data_length = static_cast<unsigned>(d->extradata.size());
        parser.pExtVideoInfo = &extension;
    }
    {
        ContextScope scope(d);
        result = cu.cuvidCreateVideoParser(&d->parser, &parser);
    }
    if (result != CUDA_SUCCESS) return give_up("Starting the GPU's video parser failed: " + cuda_error(result));
    return d;
}

// Decodes one packet (a whole picture); its pictures reach nvf_pop in
// display order. Blocks while the pool is full. 0, or NVF_ERROR / NVF_ABORTED.
NVF_API int nvf_push(void *handle, const unsigned char *data, int size, long long pts) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (d->stopped()) return d->aborted ? NVF_ABORTED : NVF_ERROR;
    CUVIDSOURCEDATAPACKET packet{};
    packet.flags = CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE;
    packet.payload_size = static_cast<tcu_ulong>(size);
    packet.payload = data;
    packet.timestamp = pts;
    CUresult result;
    {
        ContextScope scope(d);
        result = g_driver.cuvidParseVideoData(d->parser, &packet);
    }
    if (result != CUDA_SUCCESS) d->check(result, "Parsing the video");
    std::lock_guard<std::mutex> guard(d->mutex);
    return d->aborted ? NVF_ABORTED : d->failed ? NVF_ERROR : 0;
}

// The end of the video: the pictures still held for display come out, then
// nvf_pop reports the end.
NVF_API int nvf_finish(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (!d->stopped()) {
        CUVIDSOURCEDATAPACKET packet{};
        packet.flags = CUVID_PKT_ENDOFSTREAM;
        CUresult result;
        {
            ContextScope scope(d);
            result = g_driver.cuvidParseVideoData(d->parser, &packet);
        }
        if (result != CUDA_SUCCESS) d->check(result, "Finishing the video");
    }
    std::lock_guard<std::mutex> guard(d->mutex);
    d->ended = true;
    d->changed.notify_all();
    return d->aborted ? NVF_ABORTED : d->failed ? NVF_ERROR : 0;
}

// The next picture: NVF_FRAME with its slot and timestamp, NVF_END,
// NVF_ERROR, NVF_ABORTED, or NVF_TIMEOUT after `timeout_ms` (< 0: no limit).
NVF_API int nvf_pop(void *handle, int timeout_ms, int *slot, long long *pts) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::unique_lock<std::mutex> guard(d->mutex);
    auto settled = [d] { return !d->ready.empty() || d->ended || d->aborted || d->failed; };
    if (timeout_ms < 0) {
        d->changed.wait(guard, settled);
    } else if (!d->changed.wait_for(guard, std::chrono::milliseconds(timeout_ms), settled)) {
        return NVF_TIMEOUT;
    }
    if (d->aborted) return NVF_ABORTED;
    if (d->failed) return NVF_ERROR;
    if (d->ready.empty()) return NVF_END;
    *slot = d->ready.front().slot;
    *pts = d->ready.front().pts;
    d->ready.pop_front();
    return NVF_FRAME;
}

// Copies a slot (frame_bytes, planes packed) into page-locked host memory,
// returning once it is there. 0 or NVF_ERROR.
NVF_API int nvf_download(void *handle, int slot, void *host) {
    Decoder *d = static_cast<Decoder *>(handle);
    const Driver &cu = g_driver;
    ContextScope scope(d);
    CUdeviceptr src = d->pool + static_cast<size_t>(slot) * d->frame_bytes;
    bool ok = d->check(cu.cuMemcpyDtoHAsync(host, src, d->frame_bytes, d->output_stream), "Copying a picture from the GPU")
              && d->check(cu.cuEventRecord(d->output_event, d->output_stream), "Copying a picture from the GPU")
              && d->check(cu.cuEventSynchronize(d->output_event), "Copying a picture from the GPU");
    return ok ? 0 : NVF_ERROR;
}

// Copies a slot's luma plane into a pitched plane in GPU memory (a libvmaf
// picture), returning once it is there. 0 or NVF_ERROR.
NVF_API int nvf_copy_luma(void *handle, int slot, unsigned long long dst, long long dst_pitch) {
    Decoder *d = static_cast<Decoder *>(handle);
    const Driver &cu = g_driver;
    ContextScope scope(d);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = d->pool + static_cast<size_t>(slot) * d->frame_bytes;
    copy.srcPitch = static_cast<size_t>(d->params.crop_w) * d->bytes_per_sample;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = static_cast<CUdeviceptr>(dst);
    copy.dstPitch = static_cast<size_t>(dst_pitch);
    copy.WidthInBytes = copy.srcPitch;
    copy.Height = d->params.crop_h;
    bool ok = d->check(cu.cuMemcpy2DAsync(&copy, d->output_stream), "Copying a picture on the GPU")
              && d->check(cu.cuEventRecord(d->output_event, d->output_stream), "Copying a picture on the GPU")
              && d->check(cu.cuEventSynchronize(d->output_event), "Copying a picture on the GPU");
    return ok ? 0 : NVF_ERROR;
}

NVF_API unsigned long long nvf_slot_pointer(void *handle, int slot) {
    Decoder *d = static_cast<Decoder *>(handle);
    return d->pool + static_cast<size_t>(slot) * d->frame_bytes;
}

NVF_API void nvf_release(void *handle, int slot) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    d->free_slots.push_back(slot);
    d->changed.notify_all();
}

// Ends every wait, now and later: nvf_push and nvf_pop return NVF_ABORTED.
NVF_API void nvf_abort(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    d->aborted = true;
    d->changed.notify_all();
}

// Only once no other thread is in a call on this decoder.
NVF_API void nvf_close(void *handle) {
    if (handle) destroy(static_cast<Decoder *>(handle));
}

NVF_API int nvf_error(void *handle, char *out, int size) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    copy_text(out, size, d->error);
    return static_cast<int>(d->error.size());
}

NVF_API void nvf_info(void *handle, Info *out) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    *out = d->info;
}

// Whether device `device`'s decoder can decode `codec` 4:2:0 at `bit_depth`
// and this size: 1, or 0 with the reason in `error`.
NVF_API int nvf_supports(int device, int codec, int bit_depth, int width, int height, char *error, int error_size) {
    if (!load_driver()) {
        copy_text(error, error_size, g_driver_error);
        return 0;
    }
    const Driver &cu = g_driver;
    CUdevice dev;
    CUcontext context;
    CUresult result = cu.cuDeviceGet(&dev, device);
    if (result == CUDA_SUCCESS) result = cu.cuDevicePrimaryCtxRetain(&context, dev);
    if (result != CUDA_SUCCESS) {
        copy_text(error, error_size, "CUDA could not start on the GPU: " + cuda_error(result));
        return 0;
    }
    CUVIDDECODECAPS caps{};
    caps.eCodecType = static_cast<cudaVideoCodec>(codec);
    caps.eChromaFormat = cudaVideoChromaFormat_420;
    caps.nBitDepthMinus8 = bit_depth - 8;
    result = cu.cuCtxPushCurrent(context);
    if (result == CUDA_SUCCESS) {
        result = cu.cuvidGetDecoderCaps(&caps);
        CUcontext ignored;
        cu.cuCtxPopCurrent(&ignored);
    }
    cu.cuDevicePrimaryCtxRelease(dev);
    if (result != CUDA_SUCCESS) {
        copy_text(error, error_size, "Asking the GPU's decoder what it supports failed: " + cuda_error(result));
        return 0;
    }
    cudaVideoSurfaceFormat surface = bit_depth > 8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    if (!caps.bIsSupported || !(caps.nOutputFormatMask & (1 << surface))) {
        copy_text(error, error_size, "this GPU's decoder does not decode this codec at this bit depth");
        return 0;
    }
    if (static_cast<unsigned>(width) > caps.nMaxWidth || static_cast<unsigned>(height) > caps.nMaxHeight
        || width < caps.nMinWidth || height < caps.nMinHeight) {
        copy_text(error, error_size, "this GPU's decoder does not decode pictures of this size");
        return 0;
    }
    return 1;
}
