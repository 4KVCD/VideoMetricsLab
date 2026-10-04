// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// vmaf_app/tools/vmaf_vulkan/licenses/LICENSE.libvmaf.txt.
//
// VMAF's features on any GPU with Vulkan: VIF, ADM and motion, the three
// that VMAF v0.6.1 and VMAF NEG are predicted from.
//
// A port of libvmaf's CUDA feature extractors (libvmaf/src/feature/cuda:
// integer_vif_cuda.c, integer_adm_cuda.c, integer_motion_cuda.c and their
// kernels, at the commit and with the pull requests that
// scripts/build_libvmaf_cuda.ps1 pins). The compute shaders in shaders/ give
// the integer sums the CUDA kernels give; the functions below marked "as
// libvmaf" turn them into feature scores with libvmaf's own expressions, in
// the same types and order, so a feature is the same double as CUDA's.
//
// What differs from the CUDA code is the structure, not the numbers:
//   - VMAF and VMAF NEG come from one pass over the frames. Their features
//     differ only in the enhancement gain limit (100 and 1): motion, VIF's
//     filters, ADM's wavelet transform and denominator are shared, and only
//     the limited parts are calculated per limit. libvmaf runs VIF and ADM
//     twice.
//   - No float or double on the GPU (see shaders/common.slang).
//   - Sums are kept as pairs of 32-bit words (64-bit atomics are optional).
//
// The library scores features only. The caller predicts VMAF from them with
// libvmaf (vmaf_import_feature_score), which also keeps the model code one.
//
// Built by scripts/build_vmaf_vulkan.ps1 with MSVC and the static C runtime,
// as the bundled libvmaf is: powf(), pow() and log2() below then are the
// same functions as in libvmaf's own conversion of the sums.
#define _USE_MATH_DEFINES
#define VK_NO_PROTOTYPES
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "shaders_spv.h"

#define VV_EXPORT extern "C" __declspec(dllexport)

// ------------------------------------------------------------------ Vulkan

#define VK_INSTANCE_FUNCTIONS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr)

#define VK_DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCreateFence) X(vkDestroyFence) \
    X(vkWaitForFences) X(vkResetFences) X(vkQueueSubmit) X(vkDeviceWaitIdle)

namespace {

struct InstanceApi {
    HMODULE library = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define X(name) PFN_##name name = nullptr;
    VK_INSTANCE_FUNCTIONS(X)
#undef X
};

struct DeviceApi {
#define X(name) PFN_##name name = nullptr;
    VK_DEVICE_FUNCTIONS(X)
#undef X
};

thread_local std::string g_error;

int fail(int code, const std::string &message)
{
    g_error = message;
    return code;
}

// One instance per process, made on first use and kept: vulkan-1.dll is
// loaded by the first call, never inside DllMain.
InstanceApi *instance_api()
{
    static InstanceApi api;
    static bool tried = false;
    if (tried)
        return api.instance ? &api : nullptr;
    tried = true;
    api.library = LoadLibraryW(L"vulkan-1.dll");
    if (!api.library) {
        g_error = "vulkan-1.dll was not found (no Vulkan driver)";
        return nullptr;
    }
    api.vkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)GetProcAddress(api.library, "vkGetInstanceProcAddr");
    auto create = (PFN_vkCreateInstance)api.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create) {
        g_error = "vulkan-1.dll has no vkCreateInstance";
        return nullptr;
    }
    VkApplicationInfo application = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    application.pApplicationName = "VideoMetricsLab VMAF";
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    info.pApplicationInfo = &application;
    VkInstance instance;
    VkResult result = create(&info, nullptr, &instance);
    if (result != VK_SUCCESS) {
        g_error = "vkCreateInstance failed (" + std::to_string(result) + ")";
        return nullptr;
    }
#define X(name) api.name = (PFN_##name)api.vkGetInstanceProcAddr(instance, #name);
    VK_INSTANCE_FUNCTIONS(X)
#undef X
    api.instance = instance;
    return &api;
}

std::vector<VkPhysicalDevice> physical_devices(InstanceApi *api)
{
    uint32_t count = 0;
    api->vkEnumeratePhysicalDevices(api->instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    if (count)
        api->vkEnumeratePhysicalDevices(api->instance, &count, devices.data());
    devices.resize(count);
    return devices;
}

// --------------------------------------------------------------- features

enum { kScales = 4, kVifSums = 9 };
// The sums' slots (64 bits each) in the accumulator buffer.
enum {
    kSlotSad = 0,
    kSlotVif = 1,                               // [scale][kVifSums]
    kSlotCsfDen = kSlotVif + kScales * kVifSums, // [scale][band]
    kSlotCm = kSlotCsfDen + kScales * 3,         // [limit][scale][band]
    kSlots = kSlotCm + 2 * kScales * 3
};

// What vv_features returns per frame.
enum {
    kVifScale0 = 0,      // 0..3: VIF per scale, gain limit 100 (VMAF)
    kAdm2 = 4,
    kMotion2 = 5,
    kVifScale0Neg = 6,   // 6..9: VIF per scale, gain limit 1 (VMAF NEG)
    kAdm2Neg = 10,
    kMotion = 11,
    kFeatures = 12
};

struct FrameSums {
    bool scored = false;
    uint64_t slots[kSlots] = {};
};

// As libvmaf: write_scores() of integer_vif_cuda.c.
void vif_scores(const uint64_t *slots, int limit, double out[kScales])
{
    for (int scale = 0; scale < kScales; ++scale) {
        const int64_t *sums = (const int64_t *)&slots[kSlotVif + scale * kVifSums];
        const int64_t x = sums[0], num_x = sums[1], den_log = sums[2], num_non_log = sums[3],
                      den_non_log = sums[4], x2 = sums[5 + 2 * limit], num_log = sums[6 + 2 * limit];
        float num = num_log / 2048.0 + x2 + (den_non_log - ((num_non_log) / 16384.0) / (65025.0));
        float den = den_log / 2048.0 - (x + (num_x * 17)) + den_non_log;
        out[scale] = num / den;
    }
}

struct dwt_model_params {
    float a;
    float k;
    float f0;
    float g[4];
};

const dwt_model_params dwt_7_9_YCbCr_threshold[3] = {
    { 0.495f, 0.466f, 0.401f, { 1.501f, 1.0f, 0.534f, 1.0f } },
    { 1.633f, 0.353f, 0.209f, { 1.520f, 1.0f, 0.502f, 1.0f } },
    { 0.944f, 0.521f, 0.404f, { 1.868f, 1.0f, 0.516f, 1.0f } }
};

const float dwt_7_9_basis_function_amplitudes[6][4] = {
    { 0.62171, 0.67234, 0.72709, 0.67234 },
    { 0.34537, 0.41317, 0.49428, 0.41317 },
    { 0.18004, 0.22727, 0.28688, 0.22727 },
    { 0.091401, 0.11792, 0.15214, 0.11792 },
    { 0.045943, 0.059758, 0.077727, 0.059758 },
    { 0.023013, 0.030018, 0.039156, 0.030018 }
};

#define ADM_BORDER_FACTOR (0.1)
const double kAdmNormViewDist = 3.0;    // DEFAULT_ADM_NORM_VIEW_DIST
const int kAdmRefDisplayHeight = 1080;  // DEFAULT_ADM_REF_DISPLAY_HEIGHT

// As libvmaf: dwt_quant_step() of integer_adm_cuda.c.
float dwt_quant_step(const dwt_model_params *params, int lambda, int theta,
                     double adm_norm_view_dist, int adm_ref_display_height)
{
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;
    float temp = log10(pow(2.0, lambda + 1) * params->f0 * params->g[theta] / r);
    float Q = 2.0 * params->a * pow(10.0, params->k * temp * temp) /
              dwt_7_9_basis_function_amplitudes[lambda][theta];
    return Q;
}

// As libvmaf: the i_rfactor table of integer_compute_adm_cuda(), for the
// default viewing distance and display height (the only ones VMAF's models
// use).
void adm_rfactors(uint32_t i_rfactor[12])
{
    const double pow2_32 = pow(2, 32);
    for (unsigned scale = 0; scale < 4; ++scale) {
        float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, kAdmNormViewDist, kAdmRefDisplayHeight);
        float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, kAdmNormViewDist, kAdmRefDisplayHeight);
        float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };
        if (scale == 0) {
            i_rfactor[0] = 36453;
            i_rfactor[1] = 36453;
            i_rfactor[2] = 49417;
        } else {
            i_rfactor[scale * 3] = (uint32_t)(rfactor[0] * pow2_32);
            i_rfactor[scale * 3 + 1] = (uint32_t)(rfactor[1] * pow2_32);
            i_rfactor[scale * 3 + 2] = (uint32_t)(rfactor[2] * pow2_32);
        }
    }
}

// As libvmaf: conclude_adm_cm() of integer_adm_cuda.c.
void conclude_adm_cm(const int64_t *accum, int h, int w, int scale, float *result)
{
    int left = w * ADM_BORDER_FACTOR - 0.5;
    int top = h * ADM_BORDER_FACTOR - 0.5;
    int right = w - left;
    int bottom = h - top;
    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));

    const uint32_t shift_xcub[3] = { (uint32_t)ceil(log2(w) - 4), (uint32_t)ceil(log2(w) - 4),
                                     (uint32_t)ceil(log2(w) - 3) };
    int constant_offset[3] = { 52, 52, 57 };

    uint32_t shift_cub = (uint32_t)ceil(log2(w));
    float final_shift[3] = { powf(2, (45 - shift_cub - shift_inner_accum)),
                             powf(2, (39 - shift_cub - shift_inner_accum)),
                             powf(2, (36 - shift_cub - shift_inner_accum)) };
    float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    float f_accum;
    *result = 0;
    for (int i = 0; i < 3; ++i) {
        if (scale == 0) {
            f_accum = (float)(accum[i] / pow(2, (constant_offset[i] - shift_xcub[i] - shift_inner_accum)));
        } else {
            f_accum = (float)(accum[i] / final_shift[scale - 1]);
        }
        *result += powf(f_accum, 1.0f / 3.0f) + powf_add;
    }
}

// As libvmaf: conclude_adm_csf_den() of integer_adm_cuda.c.
void conclude_adm_csf_den(const uint64_t *accum, int h, int w, int scale, float *result,
                          float adm_norm_view_dist, float adm_ref_display_height)
{
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;
    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist,
                                         adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist,
                                         adm_ref_display_height);
    const float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };
    const uint32_t accum_convert_float[4] = { 18, 32, 27, 23 };

    int32_t shift_accum;
    double shift_csf;
    if (scale == 0) {
        shift_accum = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
        shift_accum = shift_accum > 0 ? shift_accum : 0;
        shift_csf = pow(2, (accum_convert_float[scale] - shift_accum));
    } else {
        shift_accum = (int32_t)ceil(log2(bottom - top));
        const uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
        shift_csf = pow(2, (accum_convert_float[scale] - shift_accum - shift_cub));
    }
    const float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    *result = 0;
    for (int i = 0; i < 3; ++i) {
        const double csf = (double)(accum[i] / shift_csf) * pow(rfactor[i], 3);
        *result += powf(csf, 1.0f / 3.0f) + powf_add;
    }
}

// As libvmaf: write_scores() of integer_adm_cuda.c.
double adm_score(const uint64_t *slots, int limit, unsigned w, unsigned h)
{
    double num = 0;
    double den = 0;
    const int64_t *adm_cm = (const int64_t *)&slots[kSlotCm + limit * kScales * 3];
    const uint64_t *adm_csf = &slots[kSlotCsfDen];
    float num_scale;
    float den_scale;
    for (unsigned scale = 0; scale < 4; ++scale) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        conclude_adm_cm(&adm_cm[scale * 3], h, w, scale, &num_scale);
        conclude_adm_csf_den(&adm_csf[scale * 3], h, w, scale, &den_scale, kAdmNormViewDist,
                             kAdmRefDisplayHeight);
        num += num_scale;
        den += den_scale;
    }
    const double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);
    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;
    if (den == 0.0)
        return 1.0f;
    return num / den;
}

// As libvmaf: normalize_and_scale_sad() of integer_motion_cuda.c.
double normalize_and_scale_sad(uint64_t sad, unsigned w, unsigned h)
{
    return (float)(sad / 256.) / (w * h);
}

float float_from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

// log_generate() of vif_statistics.cuh, (uint16_t)roundf(log2f(i) * 2048),
// as the GPU calculates it: CUDA's log2f() is the polynomial below (read from
// the kernel's PTX), which is not the host's log2f() to the last bit.
uint32_t cuda_log_generate(uint32_t i)
{
    const float value = (float)i;
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    const uint32_t exponent = (bits - 0x3F3504F3u) & 0xFF800000u;
    const float m = float_from_bits(bits - exponent);
    const float e = fmaf((float)(int32_t)exponent, float_from_bits(0x34000000u), 0.0f);
    const float f = m + -1.0f;
    float p = fmaf(f, float_from_bits(0x3DC6B27Fu), float_from_bits(0xBE2C7F30u));
    static const uint32_t coefficients[8] = { 0x3E2FCF2Au, 0xBE374E43u, 0x3E520BF4u, 0xBE763C8Bu,
                                              0x3E93BF99u, 0xBEB8AA49u, 0x3EF6384Au, 0xBF38AA3Bu };
    for (uint32_t coefficient : coefficients)
        p = fmaf(p, f, float_from_bits(coefficient));
    const float square = f * (f * p);
    const float log2 = e + fmaf(f, float_from_bits(0x3FB8AA3Bu), square);
    return (uint32_t)(log2 * 2048.0f + 0.5f) & 0xFFFFu;
}

// ------------------------------------------------------------------ engine

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *mapped = nullptr;
};

struct Pipeline {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t bindings = 0;
};

struct Pass {
    int shader;
    VkDescriptorSet set;
    uint32_t constants[32];
    uint32_t constantBytes;
    uint32_t groups[3];
};

struct Slot {
    Buffer staging, result;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool busy = false;
    bool scored = false;
    unsigned index = 0;
};

enum { kPushBytes = 128, kMaxBindings = 6 };

} // namespace

struct vv_context {
    InstanceApi *api = nullptr;
    DeviceApi vk;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::string deviceName;

    int w = 0, h = 0, bpc = 8;
    bool nativeDouble = false;
    int skip = 0;  // timing tests: 1 = no motion, 2 = no VIF, 4 = no ADM
    int passLimit = 0;  // timing tests: only the first N scored passes
    uint32_t strideBytes = 0, planeBytes = 0;

    Pipeline pipelines[kShaderCount];
    std::vector<Buffer *> buffers;
    Buffer picRef, picDis, blur[2], vifTmp, rdRef[2], rdDis[2], logTable, divTable;
    Buffer bandsRef[2], bandsDis[2], admR, admA, admF, acc;

    std::vector<Pass> motion[2];  // by frame parity
    std::vector<Pass> scored;     // VIF and ADM

    std::vector<Slot> slots;
    unsigned nextSlot = 0;
    unsigned frames = 0;
    std::vector<FrameSums> sums;
    bool failed = false;

    ~vv_context();
    int init(int deviceIndex, int width, int height, int bitDepth, int flags);
    int create_buffer(Buffer &buffer, VkDeviceSize size, bool hostVisible);
    int create_pipeline(int shader, uint32_t bindings);
    int add_pass(std::vector<Pass> &list, int shader, std::initializer_list<Buffer *> bound,
                 const void *constants, uint32_t constantBytes, uint32_t gx, uint32_t gy);
    int upload(Buffer &target, const void *data, size_t bytes);
    int build_passes();
    int submit(const uint8_t *ref, ptrdiff_t refStride, const uint8_t *dis, ptrdiff_t disStride, bool score);
    int staging(uint8_t **ref, uint8_t **dis);
    int commit(bool score);
    Slot *pending = nullptr;
    int collect(Slot &slot);
    int flush();
};

vv_context::~vv_context()
{
    if (!device)
        return;
    vk.vkDeviceWaitIdle(device);
    for (Slot &slot : slots) {
        if (slot.fence)
            vk.vkDestroyFence(device, slot.fence, nullptr);
    }
    for (Buffer *buffer : buffers) {
        if (buffer->buffer)
            vk.vkDestroyBuffer(device, buffer->buffer, nullptr);
        if (buffer->memory)
            vk.vkFreeMemory(device, buffer->memory, nullptr);
    }
    for (Pipeline &pipeline : pipelines) {
        if (pipeline.pipeline) vk.vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        if (pipeline.layout) vk.vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
        if (pipeline.setLayout) vk.vkDestroyDescriptorSetLayout(device, pipeline.setLayout, nullptr);
        if (pipeline.module) vk.vkDestroyShaderModule(device, pipeline.module, nullptr);
    }
    if (descriptorPool) vk.vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (commandPool) vk.vkDestroyCommandPool(device, commandPool, nullptr);
    vk.vkDestroyDevice(device, nullptr);
}

int vv_context::create_buffer(Buffer &buffer, VkDeviceSize size, bool hostVisible)
{
    buffers.push_back(&buffer);
    buffer.size = (size + 3) & ~VkDeviceSize(3);
    VkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    info.size = buffer.size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk.vkCreateBuffer(device, &info, nullptr, &buffer.buffer) != VK_SUCCESS)
        return fail(-1, "vkCreateBuffer failed");
    VkMemoryRequirements requirements;
    vk.vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
    // Host memory: cached if there is such a type (results are read back).
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags wanted[3] = {
        hostVisible ? visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        hostVisible ? visible : VkMemoryPropertyFlags(0), VkMemoryPropertyFlags(0)
    };
    int type = -1;
    for (int attempt = 0; attempt < (hostVisible ? 2 : 3) && type < 0; ++attempt) {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memoryProperties.memoryTypes[i].propertyFlags & wanted[attempt]) == wanted[attempt]) {
                type = (int)i;
                break;
            }
        }
    }
    if (type < 0)
        return fail(-1, "no suitable GPU memory type");
    VkMemoryAllocateInfo allocate = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = (uint32_t)type;
    if (vk.vkAllocateMemory(device, &allocate, nullptr, &buffer.memory) != VK_SUCCESS)
        return fail(-2, "out of GPU memory (" + std::to_string(requirements.size >> 20) + " MB buffer)");
    if (vk.vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0) != VK_SUCCESS)
        return fail(-1, "vkBindBufferMemory failed");
    if (hostVisible && vk.vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped) != VK_SUCCESS)
        return fail(-1, "vkMapMemory failed");
    return 0;
}

int vv_context::create_pipeline(int shader, uint32_t bindings)
{
    Pipeline &pipeline = pipelines[shader];
    if (pipeline.pipeline)
        return 0;
    pipeline.bindings = bindings;
    VkShaderModuleCreateInfo module = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    module.codeSize = kShaders[shader].bytes;
    module.pCode = kShaders[shader].code;
    if (vk.vkCreateShaderModule(device, &module, nullptr, &pipeline.module) != VK_SUCCESS)
        return fail(-1, std::string("vkCreateShaderModule failed for ") + kShaders[shader].name);
    VkDescriptorSetLayoutBinding layoutBindings[kMaxBindings] = {};
    for (uint32_t i = 0; i < bindings; ++i) {
        layoutBindings[i].binding = i;
        layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[i].descriptorCount = 1;
        layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo setLayout = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    setLayout.bindingCount = bindings;
    setLayout.pBindings = layoutBindings;
    if (vk.vkCreateDescriptorSetLayout(device, &setLayout, nullptr, &pipeline.setLayout) != VK_SUCCESS)
        return fail(-1, "vkCreateDescriptorSetLayout failed");
    VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes };
    VkPipelineLayoutCreateInfo layout = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &pipeline.setLayout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &range;
    if (vk.vkCreatePipelineLayout(device, &layout, nullptr, &pipeline.layout) != VK_SUCCESS)
        return fail(-1, "vkCreatePipelineLayout failed");
    VkComputePipelineCreateInfo info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = pipeline.module;
    info.stage.pName = "main";
    info.layout = pipeline.layout;
    if (vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline.pipeline) != VK_SUCCESS)
        return fail(-1, std::string("the GPU driver could not compile the shader ") + kShaders[shader].name);
    return 0;
}

int vv_context::add_pass(std::vector<Pass> &list, int shader, std::initializer_list<Buffer *> bound,
                         const void *constants, uint32_t constantBytes, uint32_t gx, uint32_t gy)
{
    if (int error = create_pipeline(shader, (uint32_t)bound.size()))
        return error;
    Pass pass = {};
    pass.shader = shader;
    VkDescriptorSetAllocateInfo allocate = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocate.descriptorPool = descriptorPool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &pipelines[shader].setLayout;
    if (vk.vkAllocateDescriptorSets(device, &allocate, &pass.set) != VK_SUCCESS)
        return fail(-1, "vkAllocateDescriptorSets failed");
    VkDescriptorBufferInfo infos[kMaxBindings];
    VkWriteDescriptorSet writes[kMaxBindings];
    uint32_t count = 0;
    for (Buffer *buffer : bound) {
        infos[count] = { buffer->buffer, 0, VK_WHOLE_SIZE };
        writes[count] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[count].dstSet = pass.set;
        writes[count].dstBinding = count;
        writes[count].descriptorCount = 1;
        writes[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[count].pBufferInfo = &infos[count];
        ++count;
    }
    vk.vkUpdateDescriptorSets(device, count, writes, 0, nullptr);
    memcpy(pass.constants, constants, constantBytes);
    pass.constantBytes = constantBytes;
    pass.groups[0] = gx;
    pass.groups[1] = gy;
    pass.groups[2] = 1;
    list.push_back(pass);
    return 0;
}

namespace {

void barrier(const DeviceApi &vk, VkCommandBuffer commands, VkPipelineStageFlags from, VkPipelineStageFlags to)
{
    VkMemoryBarrier memory = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT;
    vk.vkCmdPipelineBarrier(commands, from, to, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

const VkPipelineStageFlags kCompute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
const VkPipelineStageFlags kTransfer = VK_PIPELINE_STAGE_TRANSFER_BIT;

uint32_t groups(int count, int size) { return (uint32_t)((count + size - 1) / size); }

int ceil_log2(int value) { return (int)ceil(log2((double)value)); }

} // namespace

// Copies `data` into a device buffer through slot 0's staging buffer.
int vv_context::upload(Buffer &target, const void *data, size_t bytes)
{
    Buffer staging;
    if (int error = create_buffer(staging, bytes, true))
        return error;
    memcpy(staging.mapped, data, bytes);
    VkCommandBuffer commands = slots[0].commands;
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(commands, &begin);
    VkBufferCopy region = { 0, 0, bytes };
    vk.vkCmdCopyBuffer(commands, staging.buffer, target.buffer, 1, &region);
    vk.vkEndCommandBuffer(commands);
    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    VkResult result = vk.vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (result == VK_SUCCESS)
        result = vk.vkDeviceWaitIdle(device);
    vk.vkDestroyBuffer(device, staging.buffer, nullptr);
    vk.vkFreeMemory(device, staging.memory, nullptr);
    buffers.pop_back();
    return result == VK_SUCCESS ? 0 : fail(-1, "uploading a table to the GPU failed");
}

int vv_context::build_passes()
{
    const bool deep = bpc > 8;
    const int strideWords = (int)(strideBytes / 4);
    int error = 0;

    // Motion (integer_motion_cuda.c: calculate_motion_score).
    for (int parity = 0; parity < 2 && !error && !(skip & 1); ++parity) {
        const int32_t constants[] = { w, h, strideWords, bpc, 1 << (bpc - 1), kSlotSad };
        error = add_pass(motion[parity], deep ? kShader_motion_16 : kShader_motion_8,
                         { &picRef, &blur[parity], &blur[1 - parity], &acc }, constants, sizeof constants,
                         groups(w, 16), groups(h, 16));
    }

    // VIF (integer_vif_cuda.c: filter1d_8, filter1d_16).
    if (!(skip & 2)) {
        const uint64_t epsilon = 0x3EDB7CDFD9D7BDBBull;  // 65536 * 1.0e-10
        int sw = w, sh = h, sourceStride = strideWords;
        for (int scale = 0; scale < kScales && !error; ++scale) {
            if (scale > 0) {
                sw /= 2;
                sh /= 2;
            }
            uint32_t shiftVP = 16, addVP = 32768, shiftSq = 16, addSq = 32768;
            if (scale == 0) {
                shiftVP = (uint32_t)bpc;
                addVP = 1u << (bpc - 1);
                shiftSq = (uint32_t)(bpc - 8) * 2;
                addSq = bpc == 8 ? 0 : 1u << (shiftSq - 1);
            }
            // Scale s writes the next scale's images to pair s % 2.
            Buffer *inRef = scale == 0 ? &picRef : &rdRef[(scale - 1) % 2];
            Buffer *inDis = scale == 0 ? &picDis : &rdDis[(scale - 1) % 2];
            const int nextStride = (sw + 1) / 2;
            const uint32_t vertical[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)sourceStride,
                                          shiftVP, addVP, shiftSq, addSq };
            const int verticalShader = scale == 0 ? (deep ? kShader_vif_vert_0_16 : kShader_vif_vert_0_8)
                                                  : kShader_vif_vert_1 + (scale - 1);
            error = add_pass(scored, verticalShader, { inRef, inDis, &vifTmp }, vertical, sizeof vertical,
                             groups(sw, 16), groups(sh, 8));
            if (error)
                break;
            const uint32_t horizontal[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)nextStride,
                                            (uint32_t)(kSlotVif + scale * kVifSums), 100, 1,
                                            (uint32_t)epsilon, (uint32_t)(epsilon >> 32) };
            const int horizontalShader = (nativeDouble ? kShader_vif_hori_native_0 : kShader_vif_hori_0) + scale;
            error = add_pass(scored, horizontalShader,
                             { &vifTmp, &rdRef[scale % 2], &rdDis[scale % 2], &acc, &logTable }, horizontal,
                             sizeof horizontal, groups(sw, 64), (uint32_t)sh);
            sourceStride = nextStride;
        }
    }

    // ADM (integer_adm_cuda.c: integer_compute_adm_cuda).
    uint32_t i_rfactor[12];
    adm_rfactors(i_rfactor);
    int inW = w, inH = h, inStride = strideWords;
    for (int scale = 0; scale < kScales && !error && !(skip & 4); ++scale) {
        const int set = scale % 2;
        Buffer *inRef = scale == 0 ? &picRef : &bandsRef[1 - set];
        Buffer *inDis = scale == 0 ? &picDis : &bandsDis[1 - set];
        const int bw = (inW + 1) / 2, bh = (inH + 1) / 2;
        const int bandStride = set == 0 ? (w + 1) / 2 : ((w + 1) / 2 + 1) / 2;
        const int outStride = (w + 1) / 2;  // of admR, admA, admF

        {   // dwt2_8_device / adm_dwt2_16_device / adm_dwt2_s123_combined_device
            static const int kV[4][2] = { { 0, 0 }, { 0, 0 }, { 16, 32768 }, { 16, 32768 } };
            static const int kH[4][2] = { { 16, 32768 }, { 15, 16384 }, { 16, 32768 }, { 15, 16384 } };
            const int32_t constants[] = { inW, inH, inStride, bandStride,
                                          scale == 0 ? bpc : kV[scale][0], scale == 0 ? 1 << (bpc - 1) : kV[scale][1],
                                          kH[scale][0], kH[scale][1] };
            const int shader = scale == 0 ? (deep ? kShader_adm_dwt_0_16 : kShader_adm_dwt_0_8) : kShader_adm_dwt;
            error = add_pass(scored, shader, { inRef, inDis, &bandsRef[set], &bandsDis[set] }, constants,
                             sizeof constants, groups(bw, 16), groups(bh, 8));
            if (error)
                break;
        }
        {   // adm_csf_den_scale_device / adm_csf_den_s123_device
            int left = bw * (float)(ADM_BORDER_FACTOR) - 0.5f;
            int top = bh * (float)(ADM_BORDER_FACTOR) - 0.5f;
            int right = bw - left;
            int bottom = bh - top;
            uint32_t shiftSq = 0, addSq = 0, shiftCub = 0, addCub = 0, shiftAccum, addAccum;
            if (scale == 0) {
                const int32_t shift_area = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
                shiftAccum = shift_area > 0 ? (uint32_t)shift_area : 0;
                addAccum = shiftAccum > 0 ? 1u << (shiftAccum - 1) : 0;
            } else {
                static const uint32_t shift_sq[3] = { 31, 30, 31 };
                shiftSq = shift_sq[scale - 1];
                addSq = 1u << shiftSq;
                shiftCub = (uint32_t)ceil_log2(right - left);
                addCub = shiftCub ? 1u << (shiftCub - 1) : 0;
                shiftAccum = (uint32_t)ceil_log2(bottom - top);
                addAccum = shiftAccum ? 1u << (shiftAccum - 1) : 0;
            }
            const uint32_t constants[] = { (uint32_t)top, (uint32_t)left, (uint32_t)right, (uint32_t)bandStride,
                                           shiftSq, addSq, shiftCub, addCub, shiftAccum, addAccum,
                                           (uint32_t)(kSlotCsfDen + scale * 3) };
            error = add_pass(scored, scale == 0 ? kShader_adm_csf_den_0 : kShader_adm_csf_den,
                             { &bandsRef[set], &acc }, constants, sizeof constants,
                             groups(right - left, 1024), (uint32_t)(bottom - top));
            if (error)
                break;
        }
        // The enhancement gain limit: 100 (VMAF), then 1 (VMAF NEG).
        for (int limit = 0; limit < 2 && !error; ++limit) {
            {   // adm_decouple_device / adm_decouple_s123_device, adm_csf_device / i4_adm_csf_device
                int left = bw * (float)(ADM_BORDER_FACTOR) - 0.5f - 1;
                int top = bh * (float)(ADM_BORDER_FACTOR) - 0.5f - 1;
                int right = bw - left + 2;
                int bottom = bh - top + 2;
                if (left < 0) left = 0;
                if (right > bw) right = bw;
                if (top < 0) top = 0;
                if (bottom > bh) bottom = bh;
                const uint32_t constants[] = { (uint32_t)top, (uint32_t)bottom, (uint32_t)left, (uint32_t)right,
                                               (uint32_t)bandStride, (uint32_t)outStride,
                                               limit == 0 ? 100u : 1u,
                                               i_rfactor[scale * 3], i_rfactor[scale * 3 + 1], i_rfactor[scale * 3 + 2] };
                error = add_pass(scored, scale == 0 ? kShader_adm_decouple_0 : kShader_adm_decouple,
                                 { &bandsRef[set], &bandsDis[set], &admR, &admA, &admF, &divTable }, constants,
                                 sizeof constants, groups(right - left, 16), groups(bottom - top, 8));
                if (error)
                    break;
            }
            {   // adm_cm_device / i4_adm_cm_device
                int left = bw * (float)(ADM_BORDER_FACTOR) - 0.5f;
                int top = bh * (float)(ADM_BORDER_FACTOR) - 0.5f;
                int right = bw - left;
                int bottom = bh - top;
                int start_col, end_col, start_row, end_row;
                if (scale == 0) {
                    start_col = std::max(0, left);
                    end_col = std::min(right, bw);
                    start_row = std::max(0, top);
                    end_row = std::min(bottom, bh);
                } else {
                    start_col = (left > 1) ? left : ((left <= 0) ? 0 : 1);
                    end_col = (right < (bw - 1)) ? right : ((right > (bw - 1)) ? bw : bw - 1);
                    start_row = (top > 1) ? top : ((top <= 0) ? 0 : 1);
                    end_row = (bottom < (bh - 1)) ? bottom : ((bottom > (bh - 1)) ? bh : bh - 1);
                }
                struct {
                    int32_t w, h, startRow, startCol, endCol, stride;
                    uint32_t rfactor[3];
                    int32_t shiftSub[3], shiftSq[3], addSq[3], shiftCub[3], addCub[3];
                    int32_t shiftInner, addInner;
                    uint32_t slot;
                } constants = {};
                constants.w = bw;
                constants.h = bh;
                constants.startRow = start_row;
                constants.startCol = start_col;
                constants.endCol = end_col;
                constants.stride = outStride;
                static const int shift_sub[3] = { 10, 10, 12 }, fixed_shift[3] = { 4, 4, 3 };
                static const int shift_xsq[3] = { 29, 29, 30 };
                for (int band = 0; band < 3; ++band) {
                    constants.rfactor[band] = i_rfactor[scale * 3 + band];
                    constants.shiftSub[band] = scale == 0 ? shift_sub[band] : 0;
                    constants.shiftSq[band] = scale == 0 ? shift_xsq[band] : 30;
                    constants.addSq[band] = 1 << (constants.shiftSq[band] - 1);
                    const double shift = scale == 0 ? ceil(log2((double)bw) - fixed_shift[band]) : ceil(log2((double)bw));
                    constants.shiftCub[band] = shift > 0 ? (int32_t)shift : 0;
                    constants.addCub[band] = constants.shiftCub[band] ? 1 << (constants.shiftCub[band] - 1) : 0;
                }
                constants.shiftInner = ceil_log2(bh);
                constants.addInner = constants.shiftInner ? 1 << (constants.shiftInner - 1) : 0;
                constants.slot = (uint32_t)(kSlotCm + limit * kScales * 3 + scale * 3);
                error = add_pass(scored, scale == 0 ? kShader_adm_cm_0 : kShader_adm_cm,
                                 { &admR, &admA, &admF, &acc }, &constants, sizeof constants, 1,
                                 (uint32_t)std::max(0, end_row - start_row));
            }
        }
        inW = bw;
        inH = bh;
        inStride = bandStride;
    }
    return error;
}

int vv_context::init(int deviceIndex, int width, int height, int bitDepth, int flags)
{
    api = instance_api();
    if (!api)
        return -1;
    if (width < 32 || height < 32 || width > 16384 || height > 16384)
        return fail(-3, "unsupported picture size");
    if (bitDepth < 8 || bitDepth > 16)
        return fail(-3, "unsupported bit depth");
    w = width;
    h = height;
    bpc = bitDepth;

    std::vector<VkPhysicalDevice> devices = physical_devices(api);
    if (deviceIndex < 0 || deviceIndex >= (int)devices.size())
        return fail(-3, "no such GPU");
    physical = devices[(size_t)deviceIndex];
    VkPhysicalDeviceProperties properties;
    api->vkGetPhysicalDeviceProperties(physical, &properties);
    deviceName = properties.deviceName;
    VkPhysicalDeviceFeatures features;
    api->vkGetPhysicalDeviceFeatures(physical, &features);
    if (!features.shaderInt64)
        return fail(-4, deviceName + " has no 64-bit integers in shaders");
    nativeDouble = (flags & 1) && features.shaderFloat64;
    skip = (flags >> 16) & 7;
    passLimit = (flags >> 20) & 0xFF;
    api->vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);

    uint32_t familyCount = 0;
    api->vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    api->vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    int family = -1;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = (int)i;
            break;
        }
    }
    if (family < 0)
        return fail(-4, deviceName + " has no compute queue");
    queueFamily = (uint32_t)family;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enabled = {};
    enabled.shaderInt64 = VK_TRUE;
    enabled.shaderFloat64 = nativeDouble ? VK_TRUE : VK_FALSE;
    VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabled;
    VkResult result = api->vkCreateDevice(physical, &deviceInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        device = VK_NULL_HANDLE;
        return fail(-1, "vkCreateDevice failed (" + std::to_string(result) + ")");
    }
#define X(name) vk.name = (PFN_##name)api->vkGetDeviceProcAddr(device, #name);
    VK_DEVICE_FUNCTIONS(X)
#undef X
    vk.vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    if (vk.vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        return fail(-1, "vkCreateCommandPool failed");
    VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 * kMaxBindings };
    VkDescriptorPoolCreateInfo descriptorInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptorInfo.maxSets = 64;
    descriptorInfo.poolSizeCount = 1;
    descriptorInfo.pPoolSizes = &poolSize;
    if (vk.vkCreateDescriptorPool(device, &descriptorInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        return fail(-1, "vkCreateDescriptorPool failed");

    const uint32_t sampleBytes = bpc > 8 ? 2 : 1;
    strideBytes = ((uint32_t)w * sampleBytes + 3) & ~3u;
    planeBytes = strideBytes * (uint32_t)h;
    const VkDeviceSize pixels = (VkDeviceSize)w * h;
    const int w1 = (w + 1) / 2, h1 = (h + 1) / 2, w2 = (w1 + 1) / 2, h2 = (h1 + 1) / 2;
    const int rw1 = (w / 2 + 1) / 2, rh1 = (h / 2 + 1) / 2;
    struct { Buffer *buffer; VkDeviceSize bytes; } sized[] = {
        { &picRef, planeBytes }, { &picDis, planeBytes },
        { &blur[0], pixels * 4 }, { &blur[1], pixels * 4 }, { &vifTmp, pixels * 32 },
        { &rdRef[0], (VkDeviceSize)w1 * h1 * 4 }, { &rdDis[0], (VkDeviceSize)w1 * h1 * 4 },
        { &rdRef[1], (VkDeviceSize)rw1 * rh1 * 4 }, { &rdDis[1], (VkDeviceSize)rw1 * rh1 * 4 },
        { &logTable, 32768 * 4 }, { &divTable, 65536 * 4 },
        { &bandsRef[0], (VkDeviceSize)w1 * h1 * 16 }, { &bandsDis[0], (VkDeviceSize)w1 * h1 * 16 },
        { &bandsRef[1], (VkDeviceSize)w2 * h2 * 16 }, { &bandsDis[1], (VkDeviceSize)w2 * h2 * 16 },
        { &admR, (VkDeviceSize)w1 * h1 * 16 }, { &admA, (VkDeviceSize)w1 * h1 * 16 },
        { &admF, (VkDeviceSize)w1 * h1 * 16 }, { &acc, kSlots * 8 },
    };
    for (auto &entry : sized) {
        if (int error = create_buffer(*entry.buffer, entry.bytes, false))
            return error;
    }

    const int depth = std::clamp((flags >> 8) & 0xFF, 1, 16);
    slots.resize((size_t)(((flags >> 8) & 0xFF) ? depth : 3));
    std::vector<VkCommandBuffer> commands(slots.size());
    VkCommandBufferAllocateInfo allocate = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocate.commandPool = commandPool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = (uint32_t)slots.size();
    if (vk.vkAllocateCommandBuffers(device, &allocate, commands.data()) != VK_SUCCESS)
        return fail(-1, "vkAllocateCommandBuffers failed");
    for (size_t i = 0; i < slots.size(); ++i) {
        Slot &slot = slots[i];
        slot.commands = commands[i];
        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vk.vkCreateFence(device, &fenceInfo, nullptr, &slot.fence) != VK_SUCCESS)
            return fail(-1, "vkCreateFence failed");
        if (int error = create_buffer(slot.staging, (VkDeviceSize)planeBytes * 2, true))
            return error;
        if (int error = create_buffer(slot.result, kSlots * 8, true))
            return error;
    }

    // The tables, and zeros where a first frame reads before anything wrote:
    // the previous blur (as the CUDA code's memset) and the band images.
    std::vector<uint32_t> table(65536);
    for (uint32_t i = 0; i < 32768; ++i)
        table[i] = cuda_log_generate(32768 + i);
    if (int error = upload(logTable, table.data(), 32768 * 4))
        return error;
    const float div_Q_factor = 1073741824;  // 2^30
    for (int i = -32768; i < 32768; ++i)
        table[(size_t)(i + 32768)] = i == 0 ? 0 : (uint32_t)(int32_t)(div_Q_factor / float(i));
    if (int error = upload(divTable, table.data(), 65536 * 4))
        return error;
    {
        VkCommandBuffer cb = slots[0].commands;
        VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk.vkBeginCommandBuffer(cb, &begin);
        for (Buffer *buffer : { &blur[0], &blur[1], &vifTmp, &rdRef[0], &rdDis[0], &rdRef[1], &rdDis[1],
                                &bandsRef[0], &bandsDis[0], &bandsRef[1], &bandsDis[1], &admR, &admA, &admF })
            vk.vkCmdFillBuffer(cb, buffer->buffer, 0, VK_WHOLE_SIZE, 0);
        vk.vkEndCommandBuffer(cb);
        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cb;
        if (vk.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
            vk.vkDeviceWaitIdle(device) != VK_SUCCESS)
            return fail(-1, "clearing the GPU buffers failed");
    }
    return build_passes();
}

int vv_context::collect(Slot &slot)
{
    if (!slot.busy)
        return 0;
    VkResult result = vk.vkWaitForFences(device, 1, &slot.fence, VK_TRUE, 60ull * 1000 * 1000 * 1000);
    slot.busy = false;
    if (result != VK_SUCCESS) {
        failed = true;
        return fail(-5, result == VK_TIMEOUT ? "the GPU did not finish a frame in 60 seconds"
                                             : "the GPU failed (" + std::to_string(result) + ")");
    }
    vk.vkResetFences(device, 1, &slot.fence);
    FrameSums &frame = sums[slot.index];
    frame.scored = slot.scored;
    const uint32_t *words = (const uint32_t *)slot.result.mapped;
    for (int i = 0; i < kSlots; ++i)
        frame.slots[i] = (uint64_t)words[2 * i] | ((uint64_t)words[2 * i + 1] << 32);
    return 0;
}

// The next frame pair's two luma planes in the memory the GPU copies them
// from, for a caller that can write them there itself (a decoder): rows of
// strideBytes, the reference at *ref and the distorted at *dis. commit()
// then scores what was written.
int vv_context::staging(uint8_t **ref, uint8_t **dis)
{
    if (failed)
        return fail(-5, "the GPU failed on an earlier frame");
    Slot &slot = slots[nextSlot];
    if (int error = collect(slot))
        return error;
    pending = &slot;
    *ref = (uint8_t *)slot.staging.mapped;
    *dis = (uint8_t *)slot.staging.mapped + planeBytes;
    return 0;
}

int vv_context::submit(const uint8_t *ref, ptrdiff_t refStride, const uint8_t *dis, ptrdiff_t disStride, bool score)
{
    uint8_t *targets[2];
    if (int error = staging(&targets[0], &targets[1]))
        return error;
    const size_t rowBytes = (size_t)w * (bpc > 8 ? 2 : 1);
    const uint8_t *planes[2] = { ref, score ? dis : nullptr };
    const ptrdiff_t strides[2] = { refStride, disStride };
    for (int plane = 0; plane < 2; ++plane) {
        if (!planes[plane])
            continue;
        uint8_t *to = targets[plane];
        if ((size_t)strides[plane] == strideBytes) {
            memcpy(to, planes[plane], planeBytes - (strideBytes - rowBytes));
        } else {
            for (int y = 0; y < h; ++y)
                memcpy(to + (size_t)y * strideBytes, planes[plane] + (ptrdiff_t)y * strides[plane], rowBytes);
        }
    }
    return commit(score);
}

int vv_context::commit(bool score)
{
    if (!pending)
        return fail(-3, "no frame was started");
    Slot &slot = *pending;
    pending = nullptr;
    nextSlot = (nextSlot + 1) % (unsigned)slots.size();

    const unsigned index = frames++;
    sums.emplace_back();
    slot.index = index;
    slot.scored = score;

    VkCommandBuffer cb = slot.commands;
    vk.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(cb, &begin);
    barrier(vk, cb, kCompute | kTransfer, kTransfer);
    VkBufferCopy copy = { 0, 0, planeBytes };
    vk.vkCmdCopyBuffer(cb, slot.staging.buffer, picRef.buffer, 1, &copy);
    if (score) {
        copy.srcOffset = planeBytes;
        vk.vkCmdCopyBuffer(cb, slot.staging.buffer, picDis.buffer, 1, &copy);
    }
    vk.vkCmdFillBuffer(cb, acc.buffer, 0, VK_WHOLE_SIZE, 0);
    barrier(vk, cb, kTransfer, kCompute);
    const std::vector<Pass> *lists[2] = { &motion[index % 2], score ? &scored : nullptr };
    for (const std::vector<Pass> *list : lists) {
        if (!list)
            continue;
        int done = 0;
        for (const Pass &pass : *list) {
            if (list == &scored && passLimit && done++ >= passLimit)
                break;
            if (!pass.groups[0] || !pass.groups[1])
                continue;
            const Pipeline &pipeline = pipelines[pass.shader];
            vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &pass.set, 0, nullptr);
            vk.vkCmdPushConstants(cb, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pass.constantBytes, pass.constants);
            vk.vkCmdDispatch(cb, pass.groups[0], pass.groups[1], pass.groups[2]);
            barrier(vk, cb, kCompute, kCompute | kTransfer);
        }
    }
    VkBufferCopy back = { 0, 0, kSlots * 8 };
    vk.vkCmdCopyBuffer(cb, acc.buffer, slot.result.buffer, 1, &back);
    vk.vkEndCommandBuffer(cb);
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cb;
    VkResult result = vk.vkQueueSubmit(queue, 1, &submitInfo, slot.fence);
    if (result != VK_SUCCESS) {
        failed = true;
        return fail(-5, "vkQueueSubmit failed (" + std::to_string(result) + ")");
    }
    slot.busy = true;
    return 0;
}

int vv_context::flush()
{
    int error = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        // In submission order, so the first failure is the one reported.
        Slot &slot = slots[(nextSlot + i) % slots.size()];
        if (int slotError = collect(slot))
            error = error ? error : slotError;
    }
    return error;
}

// --------------------------------------------------------------------- API

// The GPUs Vulkan lists; the index is what vv_create takes. Returns how many
// there are, or a negative error. `flags`: 1 = usable (64-bit integers),
// 2 = has double.
VV_EXPORT int vv_device(int index, char *name, int nameBytes, uint32_t *vendor, uint32_t *type, uint32_t *flags)
{
    InstanceApi *api = instance_api();
    if (!api)
        return -1;
    std::vector<VkPhysicalDevice> devices = physical_devices(api);
    if (index >= 0 && index < (int)devices.size()) {
        VkPhysicalDeviceProperties properties;
        api->vkGetPhysicalDeviceProperties(devices[(size_t)index], &properties);
        VkPhysicalDeviceFeatures features;
        api->vkGetPhysicalDeviceFeatures(devices[(size_t)index], &features);
        if (name && nameBytes > 0) {
            strncpy(name, properties.deviceName, (size_t)nameBytes - 1);
            name[nameBytes - 1] = 0;
        }
        if (vendor) *vendor = properties.vendorID;
        if (type) *type = (uint32_t)properties.deviceType;
        if (flags) *flags = (features.shaderInt64 ? 1u : 0u) | (features.shaderFloat64 ? 2u : 0u);
    }
    return (int)devices.size();
}

// `flags`: 1 = use the GPU's double where it has one (the scores are the
// same; see shaders/vif_hori.slang); bits 8-15 = frames in flight (0: 3).
// Bits 16-18 leave out motion, VIF or ADM, and bits 20-27 all but the first
// N passes of VIF and ADM, to time the others.
VV_EXPORT int vv_create(vv_context **out, int device, int width, int height, int bitDepth, int flags)
{
    vv_context *context = new vv_context();
    int error = context->init(device, width, height, bitDepth, flags);
    if (error) {
        delete context;
        *out = nullptr;
        return error;
    }
    *out = context;
    return 0;
}

VV_EXPORT void vv_destroy(vv_context *context) { delete context; }

VV_EXPORT const char *vv_error() { return g_error.c_str(); }

// The next frame pair's luma planes (16-bit little-endian samples above 8
// bits). `score` 0: only motion is calculated for the frame (libvmaf's
// n_subsample), and `distorted` may be null.
VV_EXPORT int vv_submit(vv_context *context, const uint8_t *reference, ptrdiff_t referenceStride,
                        const uint8_t *distorted, ptrdiff_t distortedStride, int score)
{
    return context->submit(reference, referenceStride, distorted, distortedStride, score != 0);
}

// The memory to write the next pair's luma planes to (rows *stride bytes
// apart), and vv_commit to score them: vv_submit without its copy.
VV_EXPORT int vv_staging(vv_context *context, uint8_t **reference, uint8_t **distorted, uint32_t *stride)
{
    *stride = context->strideBytes;
    return context->staging(reference, distorted);
}

VV_EXPORT int vv_commit(vv_context *context, int score) { return context->commit(score != 0); }

// Waits for every submitted frame.
VV_EXPORT int vv_flush(vv_context *context) { return context->flush(); }

// A frame's features (kFeatures doubles), after vv_flush. Returns 1 when
// the frame was scored, 0 when only its motion was (the others are 0 then).
VV_EXPORT int vv_features(vv_context *context, unsigned index, double *out)
{
    if (index >= context->sums.size())
        return fail(-3, "no such frame");
    const std::vector<FrameSums> &sums = context->sums;
    const unsigned w = (unsigned)context->w, h = (unsigned)context->h;
    for (int i = 0; i < kFeatures; ++i)
        out[i] = 0.0;
    // As libvmaf (integer_motion_cuda.c): the first frame's motion is 0, and
    // motion2 is the lesser of a frame's motion and the next frame's.
    auto motion = [&](unsigned frame) {
        return frame == 0 ? 0.0 : normalize_and_scale_sad(sums[frame].slots[kSlotSad], w, h);
    };
    const double own = motion(index);
    out[kMotion] = own;
    if (index == 0) {
        out[kMotion2] = 0.0;
    } else if (index + 1 < sums.size()) {
        const double next = motion(index + 1);
        out[kMotion2] = own < next ? own : next;
    } else {
        out[kMotion2] = own;
    }
    const FrameSums &frame = sums[index];
    if (!frame.scored)
        return 0;
    vif_scores(frame.slots, 0, &out[kVifScale0]);
    vif_scores(frame.slots, 1, &out[kVifScale0Neg]);
    out[kAdm2] = adm_score(frame.slots, 0, w, h);
    out[kAdm2Neg] = adm_score(frame.slots, 1, w, h);
    return 1;
}

// A frame's raw sums (kSlots 64-bit values), for tests.
VV_EXPORT int vv_sums(vv_context *context, unsigned index, uint64_t *out, int count)
{
    if (index >= context->sums.size())
        return fail(-3, "no such frame");
    memcpy(out, context->sums[index].slots, sizeof(uint64_t) * (size_t)std::min(count, (int)kSlots));
    return kSlots;
}

VV_EXPORT const char *vv_device_name(vv_context *context) { return context->deviceName.c_str(); }

// A device buffer's first `bytes` bytes, after vv_flush; for tests. `which`:
// 0 admR, 1 admA, 2 admF, 3 bandsRef[0], 4 bandsDis[0], 5 bandsRef[1], 6 bandsDis[1], 7 vifTmp.
VV_EXPORT int vv_read_buffer(vv_context *context, int which, void *out, uint64_t bytes)
{
    Buffer *all[] = { &context->admR, &context->admA, &context->admF, &context->bandsRef[0], &context->bandsDis[0],
                      &context->bandsRef[1], &context->bandsDis[1], &context->vifTmp };
    if (which < 0 || which >= (int)(sizeof all / sizeof all[0]))
        return fail(-3, "no such buffer");
    Buffer *source = all[which];
    bytes = std::min<uint64_t>(bytes, source->size);
    Buffer staging;
    if (int error = context->create_buffer(staging, bytes, true))
        return error;
    const DeviceApi &vk = context->vk;
    VkCommandBuffer cb = context->slots[0].commands;
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkResetCommandBuffer(cb, 0);
    vk.vkBeginCommandBuffer(cb, &begin);
    barrier(vk, cb, kCompute | kTransfer, kTransfer);
    VkBufferCopy region = { 0, 0, bytes };
    vk.vkCmdCopyBuffer(cb, source->buffer, staging.buffer, 1, &region);
    vk.vkEndCommandBuffer(cb);
    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    VkResult result = vk.vkQueueSubmit(context->queue, 1, &submit, VK_NULL_HANDLE);
    if (result == VK_SUCCESS)
        result = vk.vkDeviceWaitIdle(context->device);
    if (result == VK_SUCCESS)
        memcpy(out, staging.mapped, (size_t)bytes);
    vk.vkDestroyBuffer(context->device, staging.buffer, nullptr);
    vk.vkFreeMemory(context->device, staging.memory, nullptr);
    context->buffers.pop_back();
    return result == VK_SUCCESS ? 0 : fail(-1, "reading a GPU buffer failed");
}
