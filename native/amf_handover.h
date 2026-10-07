// amf_frames' picture hand-over (Params.handover): AMD's counterpart of
// nvdec_frames' pictures that never leave the GPU, for the GPU metrics'
// Vulkan VMAF (vmaf_vulkan.dll) and libvmaf's CPU features.
//
// AMF decodes on Direct3D 11 as without it. Where Vulkan can import AMF's
// own textures (which Direct3D 11 shares), a slot holds its picture's texture
// and every copy out of it reads the texture where it is (Slots::hold,
// direct_copy, amf_direct.slang): into Vulkan VMAF's memory at once, and into
// the slot buffer for the other copies, which go on from there as below. On
// a Radeon 780M that decoded a 4K pair at 113 pictures a second instead of
// 99 (AMF's own speed), and scored VMAF v1 at 75 frames a second instead of
// 60, every frame's score the same.
//
// Otherwise each picture's crop is copied
// by Direct3D 11 into a texture shared with a Vulkan device made here on the
// same GPU (one for the whole process, so that every decoder in it writes
// into the same memory), and by Vulkan out of it into a slot buffer -- the
// luma, then the chroma through a scratch buffer and amf_handover.slang,
// planar as nvdec_frames' pool holds it -- and from
// there by the GPU into the buffers Vulkan VMAF exports (nvf_import_vulkan:
// an opaque Win32 handle from the same GPU and driver) and libvmaf's
// pictures in system memory, made known to the GPU (nvf_pin,
// VK_EXT_external_memory_host): nvf_copy_luma, nvf_download_planes. The CPU
// copies nothing; a destination that is not pinned is reached through a
// host-visible buffer and one memcpy (nvf_download, and
// nvf_download_planes' fallback).
//
// The decoding thread waits for both copies of each picture. On a Radeon
// 780M, which shares its memory with the CPU, that scored VMAF v1 (whose
// pace is libvmaf's CPU features') 4-11% faster than through system memory
// with 9-31% less CPU time, a 4K or a 1080p pair. Not waiting (Vulkan
// waiting for Direct3D 11's fence on the GPU, the copies in the queue's
// order; or only while the caller waited for pictures) let a 4K pair decode
// at 89 pictures a second instead of 62, but VMAF v1 then took up to 20%
// more CPU time and was no faster. VMAF v0.6.1, which the GPU scores alone,
// was 3-9% slower than through system memory either way there: the copies
// are the GPU's work, which the system memory way leaves to the CPU.
//
// AMF's own decoder on Vulkan, which would need no Direct3D 11 copy, is not
// used: on a Radeon 780M (driver 32.0.31041.1004) it decodes HEVC's last
// B-pictures before a closed GOP's IDR wrong -- on a Vulkan device AMF makes
// itself too, the pictures it reads back itself included (a UHD Blu-ray's,
// and x265's with keyint=12:no-open-gop=1) -- where its Direct3D 11 decoder,
// and FFmpeg's Vulkan and D3D11VA decoding on the same driver, are right.
//
// Vulkan is loaded at run time (vulkan-1.dll, with the graphics driver).

#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "amf_handover_spv.h"
#include "gpu_frames.h"

// Vulkan's structures are made as {sType}, the rest zero.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

namespace handover {

#define HANDOVER_INSTANCE_FUNCTIONS(X)                                                                                \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceQueueFamilyProperties)       \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceProperties2)          \
    X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr)

#define HANDOVER_DEVICE_FUNCTIONS(X)                                                                                  \
    X(vkGetDeviceQueue) X(vkQueueSubmit) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements)        \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateCommandPool)                    \
    X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer)                 \
    X(vkResetCommandBuffer) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)                    \
    X(vkCmdCopyImageToBuffer) X(vkCmdCopyBuffer) X(vkCmdPipelineBarrier) X(vkCmdBindPipeline)                        \
    X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateComputePipelines) X(vkCreateDescriptorPool)    \
    X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)                                  \
    X(vkGetMemoryHostPointerPropertiesEXT) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements)      \
    X(vkBindImageMemory) X(vkGetMemoryWin32HandlePropertiesKHR) X(vkCreateImageView) X(vkDestroyImageView)

// The process's Vulkan device for AMF's decoders, made once (shared()).
struct Device {
    std::mutex mutex;  // making it; and every vkQueueSubmit on `queue`
    bool tried = false, ready = false;  // ready: made whole
    std::string error;  // why there is none
    HMODULE library = nullptr;
    PFN_vkGetInstanceProcAddr getInstanceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    // The copies into other memory (nvf_copy_luma, nvf_download_planes): on
    // a transfer-only queue (the GPU's copy engine) where there is one, which
    // works beside the compute queues -- on the compute queue they waited
    // behind Vulkan VMAF's frame, as long as it took (10 ms a 4K copy).
    uint32_t copy_family = 0;
    VkQueue copy_queue = VK_NULL_HANDLE;
    bool copy_compute = true;  // whether the copy queue also computes (the stages it may wait on)
    VkPhysicalDeviceMemoryProperties memory{};
    uint8_t luid[VK_LUID_SIZE] = {};  // the GPU's: Direct3D 11's adapter
    uint8_t device_uuid[VK_UUID_SIZE] = {}, driver_uuid[VK_UUID_SIZE] = {};  // what an exporter must match
    VkDeviceSize host_alignment = 4096;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipelines[2] = {};  // 8-bit, 16-bit samples
    // amf_direct.slang (Slots::direct_plane): set 0 a picture's two plane
    // views, set 1 the buffer it writes; no pipelines if they could not be made.
    VkDescriptorSetLayout direct_layouts[2] = {};
    VkPipelineLayout direct_layout = VK_NULL_HANDLE;
    VkPipeline direct_pipelines[2] = {};  // 8-bit, 16-bit samples
    // A timeline semaphore the copies into other APIs' memory signal when
    // they are not waited for (Slots::copy_luma's `signal`), which Vulkan
    // VMAF imports (nvf_timeline) and waits for on the GPU; none if the GPU
    // cannot share one. `timeline_value`: the last value signalled (under `mutex`).
    VkSemaphore timeline = VK_NULL_HANDLE;
    uint64_t timeline_value = 0;
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR = nullptr;
    PFN_vkGetSemaphoreCounterValue vkGetSemaphoreCounterValue = nullptr;
    PFN_vkCreateSemaphore vkCreateSemaphore = nullptr;
    PFN_vkDestroySemaphore vkDestroySemaphore = nullptr;
#define X(name) PFN_##name name = nullptr;
    HANDOVER_INSTANCE_FUNCTIONS(X)
    HANDOVER_DEVICE_FUNCTIONS(X)
#undef X
};

inline Device g_device;

// Memory another API exported (nvf_import_vulkan), known by the address handed back:
// imports are told apart by bits 40 and up, so that offsets into one can be
// added to it, as to a CUDA device pointer.
struct Import {
    unsigned long long base;
    VkDeviceSize size;
    VkBuffer buffer;
    VkDeviceMemory memory;
};

// System memory the GPU writes into (nvf_pin): pages around what was asked for.
struct Pin {
    uintptr_t start;
    size_t size;
    const void *asked;
    VkBuffer buffer;
    VkDeviceMemory memory;
};

inline std::mutex g_registry;  // imports and pins: every decoder of the process uses them
inline std::vector<Import> g_imports;
inline std::vector<Pin> g_pins;
inline unsigned long long g_next_import = 1;

inline int memory_type(uint32_t bits, VkMemoryPropertyFlags wanted) {
    for (uint32_t i = 0; i < g_device.memory.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g_device.memory.memoryTypes[i].propertyFlags & wanted) == wanted) return static_cast<int>(i);
    return -1;
}

inline bool has_extension(const std::vector<VkExtensionProperties> &listed, const char *name) {
    for (const VkExtensionProperties &extension : listed)
        if (!strcmp(extension.extensionName, name)) return true;
    return false;
}

// amf_direct.slang's pipelines, for Slots::direct_plane; none if any part
// cannot be made (the hand-over then copies by Direct3D 11, as before).
inline void make_timeline(Device &g);

inline void make_direct(Device &g) {
    VkDescriptorSetLayoutBinding views[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        views[i].binding = i;
        views[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        views[i].descriptorCount = 1;
        views[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutBinding outputs[2] = {};  // the buffer, as samples and as 16 bytes at a time
    for (uint32_t i = 0; i < 2; ++i) {
        outputs[i].binding = i;
        outputs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        outputs[i].descriptorCount = 1;
        outputs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 2;
    set_info.pBindings = views;
    if (g.vkCreateDescriptorSetLayout(g.device, &set_info, nullptr, &g.direct_layouts[0]) != VK_SUCCESS) return;
    set_info.bindingCount = 2;
    set_info.pBindings = outputs;
    if (g.vkCreateDescriptorSetLayout(g.device, &set_info, nullptr, &g.direct_layouts[1]) != VK_SUCCESS) return;
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 9 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 2;
    layout_info.pSetLayouts = g.direct_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (g.vkCreatePipelineLayout(g.device, &layout_info, nullptr, &g.direct_layout) != VK_SUCCESS) return;
    const uint32_t *code[2] = {kDirectSpirv8, kDirectSpirv16};
    const size_t bytes[2] = {sizeof kDirectSpirv8, sizeof kDirectSpirv16};
    VkPipeline made[2] = {};
    for (int i = 0; i < 2; ++i) {
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = bytes[i];
        module_info.pCode = code[i];
        VkShaderModule module = VK_NULL_HANDLE;
        if (g.vkCreateShaderModule(g.device, &module_info, nullptr, &module) != VK_SUCCESS) return;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = g.direct_layout;
        const VkResult compiled = g.vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &made[i]);
        g.vkDestroyShaderModule(g.device, module, nullptr);
        if (compiled != VK_SUCCESS) return;
    }
    g.direct_pipelines[0] = made[0];
    g.direct_pipelines[1] = made[1];
}

// The device's timeline semaphore, exportable as an opaque Win32 handle; none
// if it cannot be made (every copy is then waited for).
inline void make_timeline(Device &g) {
    g.vkCreateSemaphore = reinterpret_cast<PFN_vkCreateSemaphore>(g.vkGetDeviceProcAddr(g.device, "vkCreateSemaphore"));
    g.vkDestroySemaphore = reinterpret_cast<PFN_vkDestroySemaphore>(g.vkGetDeviceProcAddr(g.device, "vkDestroySemaphore"));
    g.vkGetSemaphoreWin32HandleKHR = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        g.vkGetDeviceProcAddr(g.device, "vkGetSemaphoreWin32HandleKHR"));
    g.vkGetSemaphoreCounterValue = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
        g.vkGetDeviceProcAddr(g.device, "vkGetSemaphoreCounterValue"));
    if (!g.vkCreateSemaphore || !g.vkGetSemaphoreWin32HandleKHR || !g.vkGetSemaphoreCounterValue) return;
    VkExportSemaphoreCreateInfo exported{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exported.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.pNext = &exported;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    info.pNext = &type;
    if (g.vkCreateSemaphore(g.device, &info, nullptr, &g.timeline) != VK_SUCCESS) g.timeline = VK_NULL_HANDLE;
}

// Whether the timeline semaphore has reached `value` (0: nothing to wait for).
inline bool timeline_reached(uint64_t value) {
    if (!value) return true;
    uint64_t now = 0;
    return g_device.vkGetSemaphoreCounterValue(g_device.device, g_device.timeline, &now) == VK_SUCCESS && now >= value;
}

// The device, made on first use on the GPU whose LUID (DXGI's adapter LUID)
// is `luid` -- the Direct3D 11 device AMF decodes on -- or null with
// g_device.error saying why.
inline Device *shared(const uint8_t *luid) {
    Device &g = g_device;
    std::lock_guard<std::mutex> lock(g.mutex);
    if (g.ready) {
        if (!memcmp(luid, g.luid, VK_LUID_SIZE)) return &g;
        g.error = "the hand-over's Vulkan device is on another GPU";
        return nullptr;
    }
    if (g.tried) return nullptr;  // and failed: g.error says why
    g.tried = true;
    g.library = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!g.library) {
        g.error = "there is no Vulkan driver (vulkan-1.dll)";
        return nullptr;
    }
    g.getInstanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        reinterpret_cast<void *>(GetProcAddress(g.library, "vkGetInstanceProcAddr")));
    auto create_instance = g.getInstanceProc
        ? reinterpret_cast<PFN_vkCreateInstance>(g.getInstanceProc(nullptr, "vkCreateInstance")) : nullptr;
    if (!create_instance) {
        g.error = "Vulkan's loader has no vkCreateInstance";
        return nullptr;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "VideoMetricsLab amf_frames";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    VkResult result = create_instance(&instance_info, nullptr, &g.instance);
    if (result != VK_SUCCESS) {
        g.error = "vkCreateInstance failed (" + std::to_string(result) + ")";
        return nullptr;
    }
#define X(name) g.name = reinterpret_cast<PFN_##name>(g.getInstanceProc(g.instance, #name));
    HANDOVER_INSTANCE_FUNCTIONS(X)
#undef X
    uint32_t count = 0;
    g.vkEnumeratePhysicalDevices(g.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> physicals(count);
    if (count) g.vkEnumeratePhysicalDevices(g.instance, &count, physicals.data());
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    ids.pNext = &host;
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &ids;
    for (VkPhysicalDevice physical : physicals) {
        g.vkGetPhysicalDeviceProperties2(physical, &properties2);
        if (properties2.properties.apiVersion >= VK_API_VERSION_1_3 && ids.deviceLUIDValid
            && !memcmp(ids.deviceLUID, luid, VK_LUID_SIZE)) {
            g.physical = physical;
            break;
        }
    }
    if (!g.physical) {
        g.error = "Vulkan 1.3 does not list the GPU AMD's decoder runs on";
        return nullptr;
    }
    g.vkGetPhysicalDeviceMemoryProperties(g.physical, &g.memory);
    if (host.minImportedHostPointerAlignment) g.host_alignment = host.minImportedHostPointerAlignment;
    memcpy(g.luid, luid, VK_LUID_SIZE);
    memcpy(g.device_uuid, ids.deviceUUID, VK_UUID_SIZE);
    memcpy(g.driver_uuid, ids.driverUUID, VK_UUID_SIZE);

    g.vkEnumerateDeviceExtensionProperties(g.physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> listed(count);
    if (count) g.vkEnumerateDeviceExtensionProperties(g.physical, nullptr, &count, listed.data());
    const char *extensions[3] = {VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
                                 VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    for (int i = 0; i < 2; ++i) {
        if (!has_extension(listed, extensions[i])) {
            g.error = std::string("the GPU's Vulkan driver has no ") + extensions[i];
            return nullptr;
        }
    }
    const bool share_semaphores = has_extension(listed, extensions[2]);
    // The shader's 8- and 16-bit buffers.
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features have11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    have11.pNext = &have12;
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    have.pNext = &have11;
    g.vkGetPhysicalDeviceFeatures2(g.physical, &have);
    if (!have11.storageBuffer16BitAccess || !have11.uniformAndStorageBuffer16BitAccess || !have12.storageBuffer8BitAccess
        || !have12.uniformAndStorageBuffer8BitAccess || !have12.shaderInt8 || !have.features.shaderInt16) {
        g.error = "the GPU's Vulkan has no 8- and 16-bit buffers";
        return nullptr;
    }
    VkPhysicalDeviceVulkan12Features want12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    want12.storageBuffer8BitAccess = want12.uniformAndStorageBuffer8BitAccess = want12.shaderInt8 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features want11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    want11.pNext = &want12;
    want11.storageBuffer16BitAccess = want11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    const bool timelines = share_semaphores && have12.timelineSemaphore;  // (else every copy is waited for)
    want12.timelineSemaphore = timelines ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceFeatures2 want{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    want.pNext = &want11;
    want.features.shaderInt16 = VK_TRUE;

    // One queue, of the first family that computes.
    uint32_t families_count = 0;
    g.vkGetPhysicalDeviceQueueFamilyProperties(g.physical, &families_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(families_count);
    g.vkGetPhysicalDeviceQueueFamilyProperties(g.physical, &families_count, families.data());
    int family = -1;
    for (uint32_t i = 0; i < families_count && family < 0; ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) family = static_cast<int>(i);
    if (family < 0) {
        g.error = "the GPU's Vulkan has no compute queue";
        return nullptr;
    }
    g.family = static_cast<uint32_t>(family);
    g.copy_family = g.family;
    for (uint32_t i = 0; i < families_count; ++i) {
        const VkQueueFlags flags = families[i].queueFlags;
        if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
            g.copy_family = i;
            g.copy_compute = false;
            break;
        }
    }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queues[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO},
                                         {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    queues[0].queueFamilyIndex = g.family;
    queues[0].queueCount = 1;
    queues[0].pQueuePriorities = &priority;
    queues[1].queueFamilyIndex = g.copy_family;
    queues[1].queueCount = 1;
    queues[1].pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &want;
    device_info.queueCreateInfoCount = g.copy_family != g.family ? 2 : 1;
    device_info.pQueueCreateInfos = queues;
    device_info.enabledExtensionCount = timelines ? 3 : 2;
    device_info.ppEnabledExtensionNames = extensions;
    result = g.vkCreateDevice(g.physical, &device_info, nullptr, &g.device);
    if (result != VK_SUCCESS) {
        g.device = VK_NULL_HANDLE;
        g.error = "vkCreateDevice failed (" + std::to_string(result) + ")";
        return nullptr;
    }
#define X(name) g.name = reinterpret_cast<PFN_##name>(g.vkGetDeviceProcAddr(g.device, #name));
    HANDOVER_DEVICE_FUNCTIONS(X)
#undef X
    g.vkGetDeviceQueue(g.device, g.family, 0, &g.queue);
    g.vkGetDeviceQueue(g.device, g.copy_family, 0, &g.copy_queue);
    if (timelines) make_timeline(g);

    VkDescriptorSetLayoutBinding bindings[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 2;
    set_info.pBindings = bindings;
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 6 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &g.set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (g.vkCreateDescriptorSetLayout(g.device, &set_info, nullptr, &g.set_layout) != VK_SUCCESS
        || g.vkCreatePipelineLayout(g.device, &layout_info, nullptr, &g.pipeline_layout) != VK_SUCCESS) {
        g.error = "making the hand-over's shader layout failed";
        return nullptr;
    }
    const uint32_t *code[2] = {kHandoverSpirv8, kHandoverSpirv16};
    const size_t bytes[2] = {sizeof kHandoverSpirv8, sizeof kHandoverSpirv16};
    for (int i = 0; i < 2; ++i) {
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = bytes[i];
        module_info.pCode = code[i];
        VkShaderModule module = VK_NULL_HANDLE;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = g.pipeline_layout;
        if (g.vkCreateShaderModule(g.device, &module_info, nullptr, &module) != VK_SUCCESS) {
            g.error = "the hand-over's shader did not load";
            return nullptr;
        }
        pipeline_info.stage.module = module;
        const VkResult compiled =
            g.vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &g.pipelines[i]);
        g.vkDestroyShaderModule(g.device, module, nullptr);
        if (compiled != VK_SUCCESS) {
            g.error = "the hand-over's shader did not compile";
            return nullptr;
        }
    }
    make_direct(g);  // (without it, the pictures go through Direct3D 11's copies)
    g.ready = true;
    return &g;
}

// A buffer of `size` bytes: device-local, or host-visible and mapped.
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    VkDeviceSize size = 0;
};

// A buffer both queues use (written by one, read by the other): shared by
// their families, where they are two.
inline void both_families(VkBufferCreateInfo &info, uint32_t families[2]) {
    Device &g = g_device;
    families[0] = g.family;
    families[1] = g.copy_family;
    if (g.family == g.copy_family) return;
    info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    info.queueFamilyIndexCount = 2;
    info.pQueueFamilyIndices = families;
}

inline bool make_buffer(Buffer &out, VkDeviceSize size, bool host_visible) {
    Device &g = g_device;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = std::max<VkDeviceSize>(4, (size + 3) & ~VkDeviceSize(3));
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    uint32_t families[2];
    both_families(info, families);
    if (g.vkCreateBuffer(g.device, &info, nullptr, &out.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements;
    g.vkGetBufferMemoryRequirements(g.device, out.buffer, &requirements);
    int type = host_visible
        ? memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                       | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        : memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type < 0 && host_visible)
        type = memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type < 0) return false;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = static_cast<uint32_t>(type);
    if (g.vkAllocateMemory(g.device, &allocate, nullptr, &out.memory) != VK_SUCCESS
        || g.vkBindBufferMemory(g.device, out.buffer, out.memory, 0) != VK_SUCCESS)
        return false;
    if (host_visible && g.vkMapMemory(g.device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped) != VK_SUCCESS) return false;
    out.size = size;
    return true;
}

inline void free_buffer(Buffer &buffer) {
    Device &g = g_device;
    if (buffer.buffer) g.vkDestroyBuffer(g.device, buffer.buffer, nullptr);
    if (buffer.memory) g.vkFreeMemory(g.device, buffer.memory, nullptr);  // unmaps
    buffer = Buffer{};
}

// Commands recorded and run one batch at a time, waited for.
struct Commands {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    // A batch that failed or never finished: its buffer and fence may still be
    // the GPU's, and are never used again (each later run fails).
    bool broken = false;
    bool pending = false;  // submitted, not yet waited for
    bool copies_only = false;  // on the copy queue (copy_queue), which may not compute

    // `copies`: for the copy queue.
    bool make(bool copies = false) {
        Device &g = g_device;
        copies_only = copies;
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = copies ? g.copy_family : g.family;
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (g.vkCreateCommandPool(g.device, &pool_info, nullptr, &pool) != VK_SUCCESS) return false;
        allocate.commandPool = pool;
        return g.vkAllocateCommandBuffers(g.device, &allocate, &buffer) == VK_SUCCESS
               && g.vkCreateFence(g.device, &fence_info, nullptr, &fence) == VK_SUCCESS;
    }

    // Begins a batch (once the last one is done), after everything the queue
    // ran before has been written.
    void begin() {
        Device &g = g_device;
        if (!finish()) return;
        g.vkResetCommandBuffer(buffer, 0);
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        g.vkBeginCommandBuffer(buffer, &info);
        barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, stages());
    }

    // The stages its commands run in: a transfer-only queue has no compute stage.
    VkPipelineStageFlags stages() const {
        return copies_only && !g_device.copy_compute ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                                      : VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    void barrier(VkPipelineStageFlags from, VkPipelineStageFlags to) {
        if (broken) return;
        VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        memory.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        g_device.vkCmdPipelineBarrier(buffer, from, to, 0, 1, &memory, 0, nullptr, 0, nullptr);
    }

    // Ends the batch and submits it; `signal`: also signalling the device's
    // timeline semaphore, with the value given back.
    bool submit(uint64_t *signal = nullptr) {
        Device &g = g_device;
        if (broken) return false;
        barrier(stages(), VK_PIPELINE_STAGE_HOST_BIT);
        g.vkEndCommandBuffer(buffer);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &buffer;
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        uint64_t value = 0;
        if (signal) {
            timeline.signalSemaphoreValueCount = 1;
            timeline.pSignalSemaphoreValues = &value;
            submit.pNext = &timeline;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &g.timeline;
        }
        VkResult result;
        {
            std::lock_guard<std::mutex> lock(g.mutex);  // (the values in the order of the submissions)
            if (signal) value = ++g.timeline_value;
            result = g.vkQueueSubmit(copies_only ? g.copy_queue : g.queue, 1, &submit, fence);
            if (signal && result != VK_SUCCESS) --g.timeline_value;
        }
        if (signal) *signal = value;
        if (result != VK_SUCCESS) {
            broken = true;
            return false;
        }
        pending = true;
        return true;
    }

    // Waits for the batch submitted last, if any; the host then sees what it wrote.
    bool finish() {
        Device &g = g_device;
        if (broken) return false;
        if (!pending) return true;
        if (g.vkWaitForFences(g.device, 1, &fence, VK_TRUE, 60ull * 1000 * 1000 * 1000) != VK_SUCCESS) {
            broken = true;
            return false;
        }
        g.vkResetFences(g.device, 1, &fence);
        pending = false;
        return true;
    }

    // Runs the batch and waits for it.
    bool run() { return submit() && finish(); }

    void free() {
        Device &g = g_device;
        if (pending && !broken) g.vkWaitForFences(g.device, 1, &fence, VK_TRUE, 60ull * 1000 * 1000 * 1000);
        if (fence) g.vkDestroyFence(g.device, fence, nullptr);
        if (pool) g.vkDestroyCommandPool(g.device, pool, nullptr);  // and its buffer
        *this = Commands{};
    }
};

// ------------------------------------------------------- imports and pins

// Another Vulkan device's export -- Vulkan VMAF's (vv_export): an opaque
// Win32 handle of a dedicated allocation of `bytes`, from memory type
// `memory_type` of a device whose deviceUUID and driverUUID are these
// (vv_shared_device) -- as a buffer the decoders copy into; *address is what
// nvf_copy_luma is given. 0, or a negative error: -4 when the exporter is
// another GPU or driver, which Vulkan does not import from.
inline int import_memory(void *handle, unsigned long long bytes, uint32_t memory_type, const uint8_t *device_uuid,
                         const uint8_t *driver_uuid, unsigned long long *address, void **memory) {
    Device &g = g_device;
    if (!g.device) return -1;
    if (memcmp(device_uuid, g.device_uuid, VK_UUID_SIZE) || memcmp(driver_uuid, g.driver_uuid, VK_UUID_SIZE)) return -4;
    Import imported{};
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.pNext = &external;
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (g.vkCreateBuffer(g.device, &info, nullptr, &imported.buffer) != VK_SUCCESS) return -2;
    VkMemoryRequirements requirements;
    g.vkGetBufferMemoryRequirements(g.device, imported.buffer, &requirements);
    VkMemoryDedicatedAllocateInfo exclusive{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    exclusive.buffer = imported.buffer;
    VkImportMemoryWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    import.pNext = &exclusive;
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    import.handle = handle;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.pNext = &import;
    allocate.allocationSize = bytes;
    allocate.memoryTypeIndex = memory_type;
    if (memory_type >= g.memory.memoryTypeCount || !(requirements.memoryTypeBits & (1u << memory_type))
        || g.vkAllocateMemory(g.device, &allocate, nullptr, &imported.memory) != VK_SUCCESS
        || g.vkBindBufferMemory(g.device, imported.buffer, imported.memory, 0) != VK_SUCCESS) {
        if (imported.memory) g.vkFreeMemory(g.device, imported.memory, nullptr);
        g.vkDestroyBuffer(g.device, imported.buffer, nullptr);
        return -3;
    }
    imported.size = bytes;
    std::lock_guard<std::mutex> lock(g_registry);
    imported.base = (g_next_import++) << 40;
    g_imports.push_back(imported);
    *address = imported.base;
    *memory = reinterpret_cast<void *>(static_cast<uintptr_t>(imported.base));
    return 0;
}

inline void unimport(void *memory) {
    const unsigned long long base = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(memory));
    std::lock_guard<std::mutex> lock(g_registry);
    for (size_t i = 0; i < g_imports.size(); ++i) {
        if (g_imports[i].base != base) continue;
        g_device.vkDestroyBuffer(g_device.device, g_imports[i].buffer, nullptr);
        g_device.vkFreeMemory(g_device.device, g_imports[i].memory, nullptr);
        g_imports.erase(g_imports.begin() + static_cast<std::ptrdiff_t>(i));
        return;
    }
}

// The import holding [address, address + bytes): its buffer and the offset.
inline bool find_import(unsigned long long address, unsigned long long bytes, VkBuffer &buffer, VkDeviceSize &offset) {
    std::lock_guard<std::mutex> lock(g_registry);
    for (const Import &imported : g_imports) {
        if (address >= imported.base && address + bytes <= imported.base + imported.size) {
            buffer = imported.buffer;
            offset = address - imported.base;
            return true;
        }
    }
    return false;
}

// System memory at `host` made known to the GPU, which then writes into it
// itself (VK_EXT_external_memory_host): the whole pages around it, which must
// be committed. 0, or a negative error (it is then copied into by the CPU).
inline int pin(const void *host, unsigned long long bytes) {
    Device &g = g_device;
    if (!g.device || !host || !bytes) return -1;
    const uintptr_t align = static_cast<uintptr_t>(g.host_alignment);
    const uintptr_t start = reinterpret_cast<uintptr_t>(host) & ~(align - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(host) + bytes + align - 1) & ~(align - 1);
    for (uintptr_t at = start; at < end;) {  // every page of it committed and writable
        MEMORY_BASIC_INFORMATION region;
        if (!VirtualQuery(reinterpret_cast<const void *>(at), &region, sizeof region) || region.State != MEM_COMMIT
            || !(region.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE)))
            return -2;
        at = reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize;
    }
    Pin pinned{start, end - start, host, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkMemoryHostPointerPropertiesEXT properties{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (g.vkGetMemoryHostPointerPropertiesEXT(g.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                              reinterpret_cast<void *>(start), &properties) != VK_SUCCESS)
        return -3;
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.pNext = &external;
    info.size = pinned.size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;  // (direct_copy's output too)
    if (g.vkCreateBuffer(g.device, &info, nullptr, &pinned.buffer) != VK_SUCCESS) return -4;
    VkMemoryRequirements requirements;
    g.vkGetBufferMemoryRequirements(g.device, pinned.buffer, &requirements);
    int type = memory_type(requirements.memoryTypeBits & properties.memoryTypeBits, 0);
    VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import.pHostPointer = reinterpret_cast<void *>(start);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.pNext = &import;
    allocate.allocationSize = pinned.size;
    allocate.memoryTypeIndex = static_cast<uint32_t>(type);
    if (type < 0 || g.vkAllocateMemory(g.device, &allocate, nullptr, &pinned.memory) != VK_SUCCESS
        || g.vkBindBufferMemory(g.device, pinned.buffer, pinned.memory, 0) != VK_SUCCESS) {
        if (pinned.memory) g.vkFreeMemory(g.device, pinned.memory, nullptr);
        g.vkDestroyBuffer(g.device, pinned.buffer, nullptr);
        return -5;
    }
    std::lock_guard<std::mutex> lock(g_registry);
    g_pins.push_back(pinned);
    return 0;
}

inline int unpin(const void *host) {
    std::lock_guard<std::mutex> lock(g_registry);
    for (size_t i = 0; i < g_pins.size(); ++i) {
        if (g_pins[i].asked != host) continue;
        g_device.vkDestroyBuffer(g_device.device, g_pins[i].buffer, nullptr);
        g_device.vkFreeMemory(g_device.device, g_pins[i].memory, nullptr);
        g_pins.erase(g_pins.begin() + static_cast<std::ptrdiff_t>(i));
        return 0;
    }
    return -1;
}

// The pin holding [host, host + bytes): its buffer and the offset.
inline bool find_pin(const void *host, size_t bytes, VkBuffer &buffer, VkDeviceSize &offset) {
    const uintptr_t at = reinterpret_cast<uintptr_t>(host);
    std::lock_guard<std::mutex> lock(g_registry);
    for (const Pin &pinned : g_pins) {
        if (at >= pinned.start && at + bytes <= pinned.start + pinned.size) {
            buffer = pinned.buffer;
            offset = at - pinned.start;
            return true;
        }
    }
    return false;
}

// ------------------------------------------------------------ a decoder's

// A plane of a slot's picture: where it starts in the slot, and its size.
struct Plane {
    VkDeviceSize offset;
    size_t row_bytes, rows;
};

// The picture as Direct3D 11 copies it in: its shared texture, and the
// commands that copy it on into a slot.
struct Picture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Commands convert;
};

// One of AMF's own pictures (a texture of its pool, shared by Direct3D 11 as
// a KMT handle), imported by Vulkan, with views of its two planes.
struct Source {
    HANDLE handle = nullptr;  // its KMT handle: which texture it is
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView views[2] = {};  // luma, chroma (UINT: the samples' bits)
    VkDescriptorSet set = VK_NULL_HANDLE;
};

// One decoder's slots and commands.
struct Slots {
    Params params{};
    size_t sample = 1;
    // Straight from AMF's pictures (Params.handover where Vulkan imports
    // them): a slot holds its picture's texture (held[slot]), and each copy
    // out of it reads the texture itself (direct_plane, amf_direct.slang) --
    // into Vulkan VMAF's memory, pinned memory or the staging buffer at once.
    // AMF's surface stays the slot's until the slot is given back.
    struct Held {
        int source = -1;  // in `sources`; -1: the slot buffer holds the picture (as without)
        int x = 0, y = 0;  // the crop's first luma sample in the texture
        uint64_t copied = 0;  // the timeline's value once its last copy not waited for is done (0: none)
    };
    // Added to by the decoding thread (source_of) while the copies read the
    // ones before (direct_copy): in place, never moved -- a vector's growing
    // moved them under a copy reading one, which then read freed memory.
    static constexpr uint32_t kMaxSources = 64, kMaxOutputs = 64;
    std::array<Source, kMaxSources> sources{};
    std::atomic<size_t> source_count{0};
    std::vector<Held> held;  // per slot
    std::vector<std::pair<VkBuffer, VkDescriptorSet>> outputs;  // the buffers direct_plane has written, and their sets
    VkDescriptorPool direct_descriptors = VK_NULL_HANDLE;
    Commands direct;  // direct_plane's (on the queue that computes; one caller at a time: `copy_lock`)
    // The copies not waited for (Device::timeline), taken in turn; `active`:
    // the batch direct_plane records into.
    static constexpr int kRing = 8;
    Commands ring[kRing];
    int next_ring = 0;
    Commands *active = &direct;
    std::vector<Buffer> slots;  // the pictures, planar (frame_bytes each): those not held (amf_frames.cpp's copies)
    Buffer chroma;              // the crop's U and V, interleaved, on their way into a slot
    // The descriptor pool `direct_descriptors`'s sets are made under this: by
    // the decoding thread (source_of) and the copies (output_set) -- Vulkan
    // has a pool's allocations made one at a time.
    std::mutex descriptor_lock;
    Buffer staging;             // host-visible, for destinations that are not pinned (made when first needed)
    // The decoding thread's: two, so that Direct3D 11 copies a picture into
    // one while the other's goes on into its slot.
    Picture pictures[2];
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets;  // per slot: the slot and `chroma`
    Commands copying;  // nvf_copy_luma / nvf_download*'s (one caller at a time: `copy_lock`)
    std::mutex copy_lock;
    std::string error;

    Plane plane(int index) const {
        const size_t w = static_cast<size_t>(params.crop_w), h = static_cast<size_t>(params.crop_h);
        const size_t cw = (w + 1) / 2, ch = (h + 1) / 2;
        if (index == 0) return {0, w * sample, h};
        return {static_cast<VkDeviceSize>(w * h * sample + (index - 1) * cw * ch * sample), cw * sample, ch};
    }

    // Direct3D 11's shared texture `texture` (an NT handle), `format` and the
    // crop's size, as pictures[index]; in the general layout from then on.
    bool import_picture(int index, HANDLE texture, VkFormat format, std::string &why) {
        Device &g = g_device;
        Picture &picture = pictures[index];
        if (!picture.convert.make()) {
            why = "making the hand-over's commands failed";
            return false;
        }
        VkExternalMemoryImageCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.pNext = &external;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {static_cast<uint32_t>(params.crop_w), static_cast<uint32_t>(params.crop_h), 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (g.vkCreateImage(g.device, &info, nullptr, &picture.image) != VK_SUCCESS) {
            why = "Vulkan cannot take Direct3D 11's pictures";
            return false;
        }
        VkMemoryWin32HandlePropertiesKHR properties{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
        g.vkGetMemoryWin32HandlePropertiesKHR(g.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT, texture,
                                              &properties);
        VkMemoryRequirements requirements;
        g.vkGetImageMemoryRequirements(g.device, picture.image, &requirements);
        const int type = memory_type(requirements.memoryTypeBits & properties.memoryTypeBits, 0);
        VkMemoryDedicatedAllocateInfo exclusive{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        exclusive.image = picture.image;
        VkImportMemoryWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
        import.pNext = &exclusive;
        import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
        import.handle = texture;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.pNext = &import;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = static_cast<uint32_t>(type);
        if (type < 0 || g.vkAllocateMemory(g.device, &allocate, nullptr, &picture.memory) != VK_SUCCESS
            || g.vkBindImageMemory(g.device, picture.image, picture.memory, 0) != VK_SUCCESS) {
            why = "Vulkan cannot take Direct3D 11's pictures";
            return false;
        }
        // Into the general layout once, before Direct3D 11 writes into it.
        picture.convert.begin();
        image_barrier(picture, VK_IMAGE_LAYOUT_UNDEFINED, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
        if (!picture.convert.run()) {
            why = "Vulkan cannot take Direct3D 11's pictures";
            return false;
        }
        return true;
    }

    // `picture` from `from` (layout, queue family) to the general layout on
    // `to`: Direct3D 11 writes it as the external queue family.
    void image_barrier(Picture &picture, VkImageLayout layout, uint32_t from, uint32_t to) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = layout;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = from;
        barrier.dstQueueFamilyIndex = to;
        barrier.image = picture.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g_device.vkCmdPipelineBarrier(picture.convert.buffer,
                                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // The slots and buffers, and `textures` (Direct3D 11's two shared textures
    // of the crop, in `format`) imported.
    // `textures`: amf_frames.cpp's, which the crop is copied into by Direct3D
    // 11 where a picture is not held, imported, with a buffer for each slot
    // that picture goes on into; null: every picture is held (mf_frames.cpp's),
    // no slot buffers -- at 4K, 6 x 25 MB of GPU memory a decoder: a held
    // picture is copied from its texture straight where it goes (direct_copy).
    bool make(const Params &p, const HANDLE textures[2], VkFormat format, std::string &why) {
        Device &g = g_device;
        params = p;
        sample = p.bit_depth > 8 ? 2 : 1;
        slots.resize(static_cast<size_t>(p.pool));
        if (!copying.make(true)) {
            why = "making the hand-over's buffers failed";
            return false;
        }
        if (!textures) return true;
        for (Buffer &slot : slots) {
            if (!make_buffer(slot, frame_bytes(p), false)) {
                why = "out of GPU memory for the hand-over's pictures";
                return false;
            }
        }
        const size_t cw = static_cast<size_t>((p.crop_w + 1) / 2), ch = static_cast<size_t>((p.crop_h + 1) / 2);
        if (!make_buffer(chroma, cw * ch * 2 * sample, false)) {
            why = "making the hand-over's buffers failed";
            return false;
        }
        if (!import_picture(0, textures[0], format, why) || !import_picture(1, textures[1], format, why)) return false;
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(2 * p.pool)};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = static_cast<uint32_t>(p.pool);
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &size;
        if (g.vkCreateDescriptorPool(g.device, &pool_info, nullptr, &descriptors) != VK_SUCCESS) {
            why = "making the hand-over's descriptors failed";
            return false;
        }
        sets.resize(slots.size());
        std::vector<VkDescriptorSetLayout> layouts(slots.size(), g.set_layout);
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = descriptors;
        allocate.descriptorSetCount = static_cast<uint32_t>(sets.size());
        allocate.pSetLayouts = layouts.data();
        if (g.vkAllocateDescriptorSets(g.device, &allocate, sets.data()) != VK_SUCCESS) {
            why = "making the hand-over's descriptors failed";
            return false;
        }
        for (size_t i = 0; i < slots.size(); ++i) {
            VkDescriptorBufferInfo infos[2] = {{slots[i].buffer, 0, VK_WHOLE_SIZE}, {chroma.buffer, 0, VK_WHOLE_SIZE}};
            VkWriteDescriptorSet writes[2] = {};
            for (uint32_t b = 0; b < 2; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = sets[i];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &infos[b];
            }
            g.vkUpdateDescriptorSets(g.device, 2, writes, 0, nullptr);
        }
        return true;
    }

    // The picture Direct3D 11 has copied into pictures[index] (and finished
    // copying) into `slot`, once it is there.
    bool convert(int slot, int index) {
        Device &g = g_device;
        Picture &picture = pictures[index];
        Commands &c = picture.convert;
        c.begin();
        if (c.broken) return false;
        image_barrier(picture, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_EXTERNAL, g.family);  // from Direct3D 11
        VkBufferImageCopy luma{};
        luma.bufferRowLength = static_cast<uint32_t>(params.crop_w);
        luma.imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
        luma.imageExtent = {static_cast<uint32_t>(params.crop_w), static_cast<uint32_t>(params.crop_h), 1};
        g.vkCmdCopyImageToBuffer(c.buffer, picture.image, VK_IMAGE_LAYOUT_GENERAL, slots[slot].buffer, 1, &luma);
        const uint32_t cw = static_cast<uint32_t>((params.crop_w + 1) / 2), ch = static_cast<uint32_t>((params.crop_h + 1) / 2);
        if (!params.luma_only) {
            VkBufferImageCopy uv{};
            uv.bufferRowLength = cw;
            uv.imageSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1};
            uv.imageExtent = {cw, ch, 1};
            g.vkCmdCopyImageToBuffer(c.buffer, picture.image, VK_IMAGE_LAYOUT_GENERAL, chroma.buffer, 1, &uv);
        }
        const uint32_t luma_count = static_cast<uint32_t>(params.crop_w) * static_cast<uint32_t>(params.crop_h);
        const uint32_t first = params.shift ? 0 : luma_count;
        const uint32_t end = luma_count + (params.luma_only ? 0 : cw * ch);
        if (first < end) {  // the 10-bit samples shifted, the chroma made planar
            c.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            const uint32_t groups = std::min<uint32_t>((end - first + 255) / 256, 65535);
            const uint32_t constants[6] = {first, luma_count, cw * ch, static_cast<uint32_t>(params.luma_only),
                                           static_cast<uint32_t>(params.shift), groups * 256};
            g.vkCmdBindPipeline(c.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipelines[sample == 2 ? 1 : 0]);
            g.vkCmdBindDescriptorSets(c.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeline_layout, 0, 1, &sets[slot], 0,
                                      nullptr);
            g.vkCmdPushConstants(c.buffer, g.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof constants, constants);
            g.vkCmdDispatch(c.buffer, groups, 1, 1);
        }
        image_barrier(picture, VK_IMAGE_LAYOUT_GENERAL, g.family, VK_QUEUE_FAMILY_EXTERNAL);  // back to Direct3D 11
        return c.run();
    }

    // Whether AMF's pictures can be read where they are (the device made
    // amf_direct.slang's pipelines); making what that takes the first time.
    bool can_hold() {
        Device &g = g_device;
        if (!g.direct_pipelines[0] || !g.direct_pipelines[1]) return false;
        if (direct_descriptors) return true;
        held.assign(slots.size(), Held{});
        VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * kMaxSources},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kMaxOutputs}};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = kMaxSources + kMaxOutputs;
        pool_info.poolSizeCount = 2;
        pool_info.pPoolSizes = sizes;
        if (!direct.make()) return false;
        for (Commands &c : ring)
            if (!c.make()) return false;
        return g.vkCreateDescriptorPool(g.device, &pool_info, nullptr, &direct_descriptors) == VK_SUCCESS;
    }

    // AMF's texture `handle` (KMT), `width` x `height` in `format`, as one of
    // `sources`: its index (imported the first time), or -1 and why not.
    int source_of(HANDLE handle, uint32_t width, uint32_t height, VkFormat format, std::string &why) {
        const size_t count = source_count.load(std::memory_order_relaxed);  // (this thread's own)
        for (size_t i = 0; i < count; ++i)
            if (sources[i].handle == handle) return static_cast<int>(i);
        if (count >= kMaxSources) {
            why = "AMD's decoder has more pictures than the hand-over takes";
            return -1;
        }
        Device &g = g_device;
        Source source;
        source.handle = handle;
        VkExternalMemoryImageCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.pNext = &external;
        info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;  // its planes viewed as UINT
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {width, height, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto discard = [&](const char *text) {
            for (VkImageView view : source.views)
                if (view) g.vkDestroyImageView(g.device, view, nullptr);
            if (source.image) g.vkDestroyImage(g.device, source.image, nullptr);
            if (source.memory) g.vkFreeMemory(g.device, source.memory, nullptr);
            why = text;
            return -1;
        };
        if (g.vkCreateImage(g.device, &info, nullptr, &source.image) != VK_SUCCESS)
            return discard("Vulkan cannot take AMD's decoded pictures");
        VkMemoryWin32HandlePropertiesKHR properties{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
        g.vkGetMemoryWin32HandlePropertiesKHR(g.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT, handle,
                                              &properties);
        VkMemoryRequirements requirements;
        g.vkGetImageMemoryRequirements(g.device, source.image, &requirements);
        const int type = memory_type(requirements.memoryTypeBits & properties.memoryTypeBits, 0);
        VkMemoryDedicatedAllocateInfo exclusive{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        exclusive.image = source.image;
        VkImportMemoryWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
        import.pNext = &exclusive;
        import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
        import.handle = handle;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.pNext = &import;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = static_cast<uint32_t>(type);
        if (type < 0 || g.vkAllocateMemory(g.device, &allocate, nullptr, &source.memory) != VK_SUCCESS
            || g.vkBindImageMemory(g.device, source.image, source.memory, 0) != VK_SUCCESS)
            return discard("Vulkan cannot take AMD's decoded pictures");
        const bool wide = sample == 2;
        for (int plane = 0; plane < 2; ++plane) {
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = source.image;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = plane == 0 ? (wide ? VK_FORMAT_R16_UINT : VK_FORMAT_R8_UINT)
                                     : (wide ? VK_FORMAT_R16G16_UINT : VK_FORMAT_R8G8_UINT);
            view.subresourceRange = {static_cast<VkImageAspectFlags>(plane == 0 ? VK_IMAGE_ASPECT_PLANE_0_BIT
                                                                                : VK_IMAGE_ASPECT_PLANE_1_BIT),
                                     0, 1, 0, 1};
            if (g.vkCreateImageView(g.device, &view, nullptr, &source.views[plane]) != VK_SUCCESS)
                return discard("Vulkan cannot view AMD's decoded pictures' planes");
        }
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = direct_descriptors;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &g.direct_layouts[0];
        std::lock_guard<std::mutex> pool_lock(descriptor_lock);
        if (g.vkAllocateDescriptorSets(g.device, &set_info, &source.set) != VK_SUCCESS)
            return discard("making the hand-over's descriptors failed");
        VkDescriptorImageInfo images[2] = {{VK_NULL_HANDLE, source.views[0], VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, source.views[1], VK_IMAGE_LAYOUT_GENERAL}};
        VkWriteDescriptorSet writes[2] = {};
        for (uint32_t b = 0; b < 2; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = source.set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[b].pImageInfo = &images[b];
        }
        g.vkUpdateDescriptorSets(g.device, 2, writes, 0, nullptr);
        sources[count] = source;
        source_count.store(count + 1, std::memory_order_release);
        return static_cast<int>(count);
    }

    // `slot` holds AMF's picture in sources[source], the crop's first luma
    // sample at (x, y) (the decoding thread, before the slot is queued).
    void hold(int slot, int source, int x, int y) { held[static_cast<size_t>(slot)] = {source, x, y}; }

    // The slot's picture is in its buffer again (the slot given back).
    void unhold(int slot) {
        if (static_cast<size_t>(slot) < held.size()) held[static_cast<size_t>(slot)] = Held{};
    }

    // The timeline's value once the slot's last copy not waited for is done (0: none).
    uint64_t copy_pending(int slot) const {
        return static_cast<size_t>(slot) < held.size() ? held[static_cast<size_t>(slot)].copied : 0;
    }

    // Whether copies need not be waited for: the device has a timeline semaphore.
    bool can_signal() const { return g_device.timeline != VK_NULL_HANDLE && direct_descriptors; }

    // Every copy not waited for, done.
    void finish_copies() {
        std::lock_guard<std::mutex> lock(copy_lock);
        for (Commands &c : ring) c.finish();
    }

    // Whether a picture not held is in its slot buffer, which a decoder that
    // holds every picture (make without textures) has none of; else why not.
    bool slot_buffer_ready(int slot) {
        if (static_cast<size_t>(slot) < slots.size() && slots[static_cast<size_t>(slot)].buffer) return true;
        error = "the picture is neither held nor in a slot buffer";
        return false;
    }

    bool holds(int slot) const {
        return static_cast<size_t>(slot) < held.size() && held[static_cast<size_t>(slot)].source >= 0;
    }

    // The descriptor set of `buffer` as amf_direct.slang's output; null if none can be made.
    VkDescriptorSet output_set(VkBuffer buffer) {
        for (const auto &known : outputs)
            if (known.first == buffer) return known.second;
        Device &g = g_device;
        if (outputs.size() >= kMaxOutputs) return VK_NULL_HANDLE;
        std::lock_guard<std::mutex> pool_lock(descriptor_lock);
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = direct_descriptors;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &g.direct_layouts[1];
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (g.vkAllocateDescriptorSets(g.device, &set_info, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkDescriptorBufferInfo info{buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2] = {};
        for (uint32_t b = 0; b < 2; ++b) {  // as samples, and as 16 bytes at a time
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &info;
        }
        g.vkUpdateDescriptorSets(g.device, 2, writes, 0, nullptr);
        outputs.emplace_back(buffer, set);
        return set;
    }

    // The held picture's texture taken from Direct3D 11 (the external queue
    // family) before `direct`'s reads, or given back after them.
    void source_barrier(const Source &source, bool acquire) {
        Device &g = g_device;
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = acquire ? 0 : VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = acquire ? VK_ACCESS_SHADER_READ_BIT : 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : g.family;
        barrier.dstQueueFamilyIndex = acquire ? g.family : VK_QUEUE_FAMILY_EXTERNAL;
        barrier.image = source.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g.vkCmdPipelineBarrier(active->buffer,
                               acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               acquire ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                               nullptr, 0, nullptr, 1, &barrier);
    }

    // Plane `index` (0 Y, 1 U, 2 V) of the held picture of `slot` into
    // `buffer` at `offset` bytes, rows `pitch` bytes apart, recorded into
    // `direct` (begun, the texture acquired). False if `buffer` cannot be written.
    bool direct_plane(int slot, int index, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize pitch) {
        Device &g = g_device;
        const Held &h = held[static_cast<size_t>(slot)];
        const VkDescriptorSet output = output_set(buffer);
        if (!output || offset % sample || pitch % sample) return false;
        const Plane p = plane(index);
        const uint32_t width = static_cast<uint32_t>(p.row_bytes / sample), height = static_cast<uint32_t>(p.rows);
        const bool wide = offset % 16 == 0 && pitch % 16 == 0;  // rows starting on 16 bytes: written 16 bytes at a time
        const uint32_t constants[9] = {static_cast<uint32_t>(index == 0 ? h.x : h.x / 2),
                                       static_cast<uint32_t>(index == 0 ? h.y : h.y / 2),
                                       width, height,
                                       static_cast<uint32_t>(offset / sample), static_cast<uint32_t>(pitch / sample),
                                       static_cast<uint32_t>(params.shift), static_cast<uint32_t>(index),
                                       wide ? 1u : 0u};
        const uint32_t across = wide ? (width + static_cast<uint32_t>(16 / sample) - 1) / static_cast<uint32_t>(16 / sample)
                                     : width;  // threads across a row
        const VkDescriptorSet sets_[2] = {sources[static_cast<size_t>(h.source)].set, output};
        g.vkCmdBindPipeline(active->buffer, VK_PIPELINE_BIND_POINT_COMPUTE, g.direct_pipelines[sample == 2 ? 1 : 0]);
        g.vkCmdBindDescriptorSets(active->buffer, VK_PIPELINE_BIND_POINT_COMPUTE, g.direct_layout, 0, 2, sets_, 0, nullptr);
        g.vkCmdPushConstants(active->buffer, g.direct_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof constants, constants);
        g.vkCmdDispatch(active->buffer, (across + 15) / 16, (height + 15) / 16, 1);
        return true;
    }

    // The held picture of `slot`, each plane into its (buffer, offset, pitch)
    // (a null buffer: not that plane), on the queue that computes: waited
    // for, or (`signal`) not, the timeline's value once it is done given back.
    bool direct_copy(int slot, const VkBuffer *buffers, const VkDeviceSize *offsets, const VkDeviceSize *pitches,
                     uint64_t *signal = nullptr) {
        active = signal ? &ring[next_ring++ % kRing] : &direct;
        Commands &c = *active;
        if (c.broken) {
            error = "copying a picture on the GPU failed";
            return false;
        }
        const Source &source = sources[static_cast<size_t>(held[static_cast<size_t>(slot)].source)];
        c.begin();  // (a batch of the ring: once its last one is done)
        source_barrier(source, true);
        for (int i = 0; i < (params.luma_only ? 1 : 3); ++i) {
            if (!buffers[i]) continue;
            if (!direct_plane(slot, i, buffers[i], offsets[i], pitches[i])) {
                error = "a plane's destination cannot be written by the hand-over's shader";
                source_barrier(source, false);
                c.run();
                return false;
            }
        }
        source_barrier(source, false);
        if (signal) {
            if (!c.submit(signal)) {
                error = "copying a picture on the GPU failed";
                return false;
            }
            held[static_cast<size_t>(slot)].copied = *signal;
            return true;
        }
        if (!c.run()) {
            error = "copying a picture on the GPU failed";
            return false;
        }
        return true;
    }

    // The held picture of `slot` into its slot buffer, for the copies that
    // read the slot buffer (nvf_download, nvf_download_planes); then held no
    // longer. Under `copy_lock`.
    // A held picture's planes into the staging buffer, packed (frame_bytes),
    // straight from its texture.
    bool held_to_staging(int slot) {
        if (!ensure_staging()) {
            error = "making the hand-over's staging buffer failed";
            return false;
        }
        VkBuffer buffers[3] = {staging.buffer, staging.buffer, staging.buffer};
        VkDeviceSize offsets[3], pitches[3];
        for (int i = 0; i < 3; ++i) {
            offsets[i] = plane(i).offset;
            pitches[i] = plane(i).row_bytes;
        }
        return direct_copy(slot, buffers, offsets, pitches);
    }

    // Copies rows of `plane` of `slot` to `buffer` at `offset`, `pitch` bytes apart.
    void copy_rows(Commands &c, int slot, const Plane &p, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize pitch) {
        if (pitch == p.row_bytes) {
            VkBufferCopy whole{p.offset, offset, p.row_bytes * p.rows};
            g_device.vkCmdCopyBuffer(c.buffer, slots[slot].buffer, buffer, 1, &whole);
            return;
        }
        std::vector<VkBufferCopy> rows(p.rows);
        for (size_t r = 0; r < p.rows; ++r) rows[r] = {p.offset + r * p.row_bytes, offset + r * pitch, p.row_bytes};
        g_device.vkCmdCopyBuffer(c.buffer, slots[slot].buffer, buffer, static_cast<uint32_t>(rows.size()), rows.data());
    }

    bool ensure_staging() {
        return staging.buffer || make_buffer(staging, frame_bytes(params), true);
    }

    // The slot's luma into imported memory at `address`, rows `pitch` apart.
    bool copy_luma(int slot, unsigned long long address, long long pitch, uint64_t *signal = nullptr) {
        const Plane p = plane(0);
        VkBuffer buffer;
        VkDeviceSize offset;
        if (pitch < static_cast<long long>(p.row_bytes)
            || !find_import(address, static_cast<unsigned long long>(pitch) * (p.rows - 1) + p.row_bytes, buffer, offset)) {
            error = "the luma's destination is not memory the decoder imported";
            return false;
        }
        std::lock_guard<std::mutex> lock(copy_lock);
        if (holds(slot)) {  // straight from AMF's picture
            const VkBuffer buffers[3] = {buffer, VK_NULL_HANDLE, VK_NULL_HANDLE};
            const VkDeviceSize offsets[3] = {offset, 0, 0}, pitches[3] = {static_cast<VkDeviceSize>(pitch), 0, 0};
            return direct_copy(slot, buffers, offsets, pitches, signal && can_signal() ? signal : nullptr);
        }
        if (signal) *signal = 0;  // (waited for)
        if (!slot_buffer_ready(slot)) return false;
        if (copying.broken) {
            error = "copying a picture on the GPU failed";
            return false;
        }
        copying.begin();
        copy_rows(copying, slot, p, buffer, offset, static_cast<VkDeviceSize>(pitch));
        if (!copying.run()) {
            error = "copying a picture on the GPU failed";
            return false;
        }
        return true;
    }

    // The slot's planes each into imported memory (nvf_import_vulkan) at its
    // own address (0: not that plane), rows `pitches` apart, on the copy queue.
    bool copy_planes(int slot, const unsigned long long *addresses, const long long *pitches,
                     uint64_t *signal = nullptr) {
        std::lock_guard<std::mutex> lock(copy_lock);
        VkBuffer buffers[3] = {};
        VkDeviceSize offsets[3] = {}, steps[3] = {};
        for (int i = 0; i < (params.luma_only ? 1 : 3); ++i) {
            if (!addresses[i]) continue;
            const Plane p = plane(i);
            if (pitches[i] < static_cast<long long>(p.row_bytes)
                || !find_import(addresses[i], static_cast<unsigned long long>(pitches[i]) * (p.rows - 1) + p.row_bytes,
                                buffers[i], offsets[i])) {
                error = "a plane's destination is not memory the decoder imported";
                return false;
            }
            steps[i] = static_cast<VkDeviceSize>(pitches[i]);
        }
        if (holds(slot))  // straight from AMF's picture
            return direct_copy(slot, buffers, offsets, steps, signal && can_signal() ? signal : nullptr);
        if (signal) *signal = 0;  // (waited for)
        if (!slot_buffer_ready(slot)) return false;
        if (copying.broken) {
            error = "copying a picture on the GPU failed";
            return false;
        }
        copying.begin();
        for (int i = 0; i < (params.luma_only ? 1 : 3); ++i) {
            if (buffers[i]) copy_rows(copying, slot, plane(i), buffers[i], offsets[i], steps[i]);
        }
        if (!copying.run()) {
            error = "copying a picture on the GPU failed";
            return false;
        }
        return true;
    }

    // The slot's planes each to its own address (null: not that plane), rows
    // `pitches` apart: by the GPU where the memory is pinned, else through
    // the staging buffer and memcpy.
    bool download_planes(int slot, void *const *planes, const long long *pitches) {
        std::lock_guard<std::mutex> lock(copy_lock);
        if (holds(slot)) {  // straight from its texture: into pinned memory, else the staging buffer
            VkBuffer buffers[3] = {};
            VkDeviceSize offsets[3] = {}, steps[3] = {};
            bool staged[3] = {};
            for (int i = 0; i < (params.luma_only ? 1 : 3); ++i) {
                if (!planes[i]) continue;
                const Plane p = plane(i);
                if (pitches[i] < static_cast<long long>(p.row_bytes)) {
                    error = "a plane's rows are shorter than the picture's";
                    return false;
                }
                const size_t span = static_cast<size_t>(pitches[i]) * (p.rows - 1) + p.row_bytes;
                VkBuffer buffer;
                VkDeviceSize offset;
                if (find_pin(planes[i], span, buffer, offset) && offset % sample == 0 && pitches[i] % sample == 0) {
                    buffers[i] = buffer;
                    offsets[i] = offset;
                    steps[i] = static_cast<VkDeviceSize>(pitches[i]);
                } else {
                    if (!ensure_staging()) {
                        error = "making the hand-over's staging buffer failed";
                        return false;
                    }
                    buffers[i] = staging.buffer;
                    offsets[i] = p.offset;
                    steps[i] = p.row_bytes;
                    staged[i] = true;
                }
            }
            if (!direct_copy(slot, buffers, offsets, steps)) return false;
            for (int i = 0; i < 3; ++i) {
                if (!staged[i]) continue;
                const Plane p = plane(i);
                const uint8_t *from = static_cast<const uint8_t *>(staging.mapped) + p.offset;
                uint8_t *to = static_cast<uint8_t *>(planes[i]);
                for (size_t r = 0; r < p.rows; ++r)
                    memcpy(to + r * static_cast<size_t>(pitches[i]), from + r * p.row_bytes, p.row_bytes);
            }
            return true;
        }
        if (!slot_buffer_ready(slot)) return false;
        if (copying.broken) {
            error = "copying a picture from the GPU failed";
            return false;
        }
        copying.begin();
        bool staged[3] = {};
        for (int i = 0; i < (params.luma_only ? 1 : 3); ++i) {
            if (!planes[i]) continue;
            const Plane p = plane(i);
            const size_t span = static_cast<size_t>(pitches[i]) * (p.rows - 1) + p.row_bytes;
            VkBuffer buffer;
            VkDeviceSize offset;
            if (pitches[i] < static_cast<long long>(p.row_bytes)) {
                error = "a plane's rows are shorter than the picture's";
                return false;
            }
            if (find_pin(planes[i], span, buffer, offset)) {
                copy_rows(copying, slot, p, buffer, offset, static_cast<VkDeviceSize>(pitches[i]));
            } else {
                if (!ensure_staging()) {
                    error = "making the hand-over's staging buffer failed";
                    return false;
                }
                VkBufferCopy whole{p.offset, p.offset, p.row_bytes * p.rows};
                g_device.vkCmdCopyBuffer(copying.buffer, slots[slot].buffer, staging.buffer, 1, &whole);
                staged[i] = true;
            }
        }
        if (!copying.run()) {
            error = "copying a picture from the GPU failed";
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            if (!staged[i]) continue;
            const Plane p = plane(i);
            const uint8_t *from = static_cast<const uint8_t *>(staging.mapped) + p.offset;
            uint8_t *to = static_cast<uint8_t *>(planes[i]);
            for (size_t r = 0; r < p.rows; ++r)
                memcpy(to + r * static_cast<size_t>(pitches[i]), from + r * p.row_bytes, p.row_bytes);
        }
        return true;
    }

    // The slot's picture, planes packed (frame_bytes), into `host`.
    bool download(int slot, void *host) {
        std::lock_guard<std::mutex> lock(copy_lock);
        if (holds(slot)) {  // straight from its texture
            if (!held_to_staging(slot)) return false;
            memcpy(host, staging.mapped, frame_bytes(params));
            return true;
        }
        if (!slot_buffer_ready(slot)) return false;
        if (!ensure_staging() || copying.broken) {
            error = copying.broken ? "copying a picture from the GPU failed" : "making the hand-over's staging buffer failed";
            return false;
        }
        copying.begin();
        VkBufferCopy whole{0, 0, frame_bytes(params)};
        g_device.vkCmdCopyBuffer(copying.buffer, slots[slot].buffer, staging.buffer, 1, &whole);
        if (!copying.run()) {
            error = "copying a picture from the GPU failed";
            return false;
        }
        memcpy(host, staging.mapped, frame_bytes(params));
        return true;
    }

    void free() {
        for (Picture &picture : pictures) picture.convert.free();  // once its last copy is done
        copying.free();
        direct.free();
        for (Commands &c : ring) c.free();
        for (size_t i = 0; i < source_count; ++i) {
            Source &source = sources[i];
            for (VkImageView view : source.views)
                if (view) g_device.vkDestroyImageView(g_device.device, view, nullptr);
            if (source.image) g_device.vkDestroyImage(g_device.device, source.image, nullptr);
            if (source.memory) g_device.vkFreeMemory(g_device.device, source.memory, nullptr);
            source = Source{};
        }
        source_count = 0;
        outputs.clear();
        if (direct_descriptors) g_device.vkDestroyDescriptorPool(g_device.device, direct_descriptors, nullptr);
        direct_descriptors = VK_NULL_HANDLE;
        for (Buffer &slot : slots) free_buffer(slot);
        free_buffer(chroma);
        free_buffer(staging);
        for (Picture &picture : pictures) {
            if (picture.image) g_device.vkDestroyImage(g_device.device, picture.image, nullptr);
            if (picture.memory) g_device.vkFreeMemory(g_device.device, picture.memory, nullptr);
            picture.image = VK_NULL_HANDLE;
            picture.memory = VK_NULL_HANDLE;
        }
        if (descriptors) g_device.vkDestroyDescriptorPool(g_device.device, descriptors, nullptr);
        descriptors = VK_NULL_HANDLE;
    }
};

}  // namespace handover

#pragma GCC diagnostic pop
