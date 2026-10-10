// Video Compare's frames, drawn and presented with Vulkan.
//
// This is the Vulkan counterpart of native/d3d11_tonemap.cpp, which Windows
// draws with. This one is kept as the start of the presenter for macOS and
// Linux, which have no Direct3D; on Windows it draws only when the app is
// asked for it (VIDEOMETRICSLAB_PRESENTER=vulkan). It has the same calls
// (vmaf_present_*), the same pacing, and a shader
// (native/vulkan_presenter.slang) that draws a frame with the same arithmetic:
// NV12 or P010 turned into RGB, and HDR video mapped to SDR or, for an HDR
// display, converted to PQ in BT.2020. NV12 and P010 are the only frames
// playback hands over (gstreamer_playback._output_format).
//
// The frames come from GStreamer's Direct3D 11 decoders, so each one is handed
// from Direct3D 11 to Vulkan on the GPU:
//   - Vulkan cannot read the decoders' textures as they are: they are not
//     shared, and an NV12 or P010 texture shared from Direct3D 11 reads back
//     wrong in Vulkan (its chroma plane on an RTX 5090 with driver 616.92; the
//     whole frame at 4K on a Core Ultra 9 285K's iGPU). Shared textures of one
//     plane read back right on both, at every size tried (64x48 to 8K).
//   - So a small Direct3D 11 compute shader copies each frame's two planes
//     (the part of them the drawing samples) into two such textures (R8 and
//     R8G8, or R16 and R16G16 for P010), which Vulkan imports once and
//     samples. There are two sets of them, used in turn, so that Direct3D 11
//     can copy a frame while Vulkan may still be drawing the one before.
//   - A fence shared by the two APIs orders the work on the GPU: Vulkan's
//     drawing waits for Direct3D 11's copy. Before Direct3D 11 copies into a
//     set again, the CPU checks that Vulkan has drawn the frame it held, which
//     by then it has.
//
// Pacing is the Direct3D 11 presenter's. The caller waits for the swapchain
// holding no lock (vmaf_present_wait, which acquires the next image and waits
// until the display has let go of it), hands the frame over holding
// GStreamer's device lock (vmaf_present_texture), and lets go of the lock
// before Vulkan draws and presents the frame (vmaf_present_flip): the present
// has nothing to wait for, and the decoders, which need that lock for each
// frame, never wait for Vulkan or the display. With three images, at most two
// frames wait for the display, as with the Direct3D 11 presenter's maximum
// frame latency of 2.
//
// Vulkan is loaded at run time (vulkan-1.dll, installed with the graphics
// driver), and only the driver of the GPU the video is decoded on. Without it,
// or without what the hand-over needs, vmaf_present_create returns null,
// vmaf_present_problem says why, and the app uses the Direct3D 11 presenter.
// Besides that presenter's calls, there are vmaf_present_start (the swapchain,
// made on the presenter's thread), vmaf_present_flip and vmaf_present_problem.
#define VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan_presenter_spv.h"

using Microsoft::WRL::ComPtr;

// Vulkan's structures are made as {sType}, the rest zero.
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

// The colour spaces the caller asks for, numbered as DXGI numbers them
// (locked_presentation.py): sRGB, and HDR10 (PQ in BT.2020).
constexpr int kHdr10Space = 12;

// Frames whose drawing the GPU may not have finished yet, at most.
constexpr int kFramesInFlight = 3;
// Sets of textures shared with Vulkan, used in turn.
constexpr int kHandoverSlots = 2;
// Source textures whose shader views are kept: the decoders' pools hand the
// same textures round.
constexpr int kKeptSourceViews = 16;
// The longest the CPU waits for the GPU before taking it for hung or lost.
constexpr uint64_t kGpuTimeoutNs = 2'000'000'000;

// The shader's constants (DrawConstants in native/vulkan_presenter.slang).
struct DrawConstants {
    float source[4];        // the part of the frame drawn, in texture coordinates
    float to_rgb[3][4];     // rows of (Y, U, V) coefficients, the offset in w
    float luma[4];          // HDR: the source primaries' luminance weights; w: SDR white (nits)
    float to_output[3][4];  // HDR: linear source RGB to linear BT.709 or BT.2020, by rows
    float shade;            // 0: as it comes; 1: HDR mapped to SDR; 2: HDR kept, as PQ in BT.2020
    float kind;             // HDR: 1 for PQ, 2 for HLG
    float peak;             // HDR: the light mapped to SDR white, and HLG's display peak (nits)
};
static_assert(sizeof(DrawConstants) == 140, "the layout of DrawConstants in native/vulkan_presenter.slang");

// ---------------------------------------------------------------------------
// What went wrong

// Why the last call that failed did, for the log and the self-test
// (vmaf_present_problem).
static std::mutex problem_lock;
static char problem_text[256] = "";

static void set_problem(const char* format, ...) {
    std::lock_guard<std::mutex> hold(problem_lock);
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(problem_text, sizeof problem_text, format, arguments);
    va_end(arguments);
}

// ---------------------------------------------------------------------------
// Vulkan: the instance, and a device shared by the presenters

#define INSTANCE_FUNCTIONS(X)                                                                                         \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2)                   \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties)                                \
    X(vkGetPhysicalDeviceImageFormatProperties2) X(vkGetPhysicalDeviceExternalSemaphoreProperties)                    \
    X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkCreateWin32SurfaceKHR)       \
    X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceWin32PresentationSupportKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR)  \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR)

#define DEVICE_FUNCTIONS(X)                                                                                           \
    X(vkDestroyDevice) X(vkDeviceWaitIdle) X(vkGetDeviceQueue) X(vkQueueSubmit) X(vkQueueWaitIdle)                    \
    X(vkQueuePresentKHR) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR)                  \
    X(vkAcquireNextImageKHR) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory)   \
    X(vkFreeMemory) X(vkBindImageMemory) X(vkGetMemoryWin32HandlePropertiesKHR) X(vkCreateImageView)                  \
    X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler) X(vkCreateDescriptorSetLayout)                       \
    X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets)  \
    X(vkUpdateDescriptorSets) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateShaderModule)            \
    X(vkDestroyShaderModule) X(vkCreateRenderPass) X(vkDestroyRenderPass) X(vkCreateGraphicsPipelines)                \
    X(vkDestroyPipeline) X(vkCreateFramebuffer) X(vkDestroyFramebuffer) X(vkCreateCommandPool)                        \
    X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkResetCommandBuffer) X(vkBeginCommandBuffer)               \
    X(vkEndCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBindPipeline)  \
    X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdDraw)              \
    X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkImportSemaphoreWin32HandleKHR) X(vkWaitSemaphores)                 \
    X(vkGetSemaphoreCounterValue) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)

#define DECLARE_FUNCTION(name) PFN_##name name = nullptr;

// The process's Vulkan instance: made the first time it is needed, and kept.
struct VulkanInstance {
    VkInstance handle = VK_NULL_HANDLE;
    // Whether surfaces may offer colour spaces besides sRGB, HDR10 among them
    // (VK_EXT_swapchain_colorspace).
    bool colour_spaces = false;
    INSTANCE_FUNCTIONS(DECLARE_FUNCTION)
};

static bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    for (const VkExtensionProperties& extension : extensions)
        if (strcmp(extension.extensionName, name) == 0)
            return true;
    return false;
}

// The file name of each GPU maker's Vulkan driver manifest, by PCI vendor ID.
// The loader is told to load only the driver of the GPU the video is decoded
// on. With every driver loaded (Intel's beside NVIDIA's, on a PC with both),
// the window's process took some 200 MB more memory.
static const wchar_t* driver_manifest(UINT vendor) {
    switch (vendor) {
    case 0x10DE:
        return L"nv-vk64.json";  // NVIDIA
    case 0x8086:
        return L"igvk64.json";  // Intel
    case 0x1002:
        return L"amd-vulkan64.json";  // AMD
    default:
        return nullptr;
    }
}

// An environment variable set to a value for as long as this lives, then put
// back as it was.
class TemporaryEnvironment {
  public:
    TemporaryEnvironment(const wchar_t* name, const wchar_t* value) : name_(name) {
        const DWORD length = GetEnvironmentVariableW(name, nullptr, 0);
        if (length) {
            previous_.resize(length);
            GetEnvironmentVariableW(name, previous_.data(), length);
            previous_.resize(length - 1);  // without its terminating null
            had_value_ = true;
        }
        SetEnvironmentVariableW(name, value);
    }
    ~TemporaryEnvironment() { SetEnvironmentVariableW(name_, had_value_ ? previous_.c_str() : nullptr); }
    TemporaryEnvironment(const TemporaryEnvironment&) = delete;
    TemporaryEnvironment& operator=(const TemporaryEnvironment&) = delete;

  private:
    const wchar_t* name_;
    std::wstring previous_;
    bool had_value_ = false;
};

// A Vulkan instance for drawing into windows, with the loader's drivers
// limited to `driver` (a manifest's file name, as VK_LOADER_DRIVERS_SELECT
// takes it) unless it is null. Null if none could be made, the problem noted.
// `colour_spaces`: whether surfaces offer HDR10 and the like.
static VkInstance create_instance(PFN_vkGetInstanceProcAddr get, const wchar_t* driver, bool* colour_spaces) {
    auto enumerate_extensions = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        get(nullptr, "vkEnumerateInstanceExtensionProperties"));
    auto create = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
    if (!enumerate_extensions || !create) {
        set_problem("the Vulkan loader is incomplete");
        return VK_NULL_HANDLE;
    }
    // The loader reads it as it looks for drivers, in both calls below.
    std::unique_ptr<TemporaryEnvironment> one_driver;
    if (driver)
        one_driver = std::make_unique<TemporaryEnvironment>(L"VK_LOADER_DRIVERS_SELECT", driver);
    uint32_t count = 0;
    enumerate_extensions(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    enumerate_extensions(nullptr, &count, available.data());
    if (!has_extension(available, VK_KHR_SURFACE_EXTENSION_NAME) ||
        !has_extension(available, VK_KHR_WIN32_SURFACE_EXTENSION_NAME)) {
        set_problem("Vulkan cannot draw into windows here");
        return nullptr;
    }
    std::vector<const char*> extensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    *colour_spaces = has_extension(available, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    if (*colour_spaces)
        extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    VkApplicationInfo application = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "VideoMetricsLab";
    application.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &application;
    info.enabledExtensionCount = uint32_t(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    VkInstance handle = VK_NULL_HANDLE;
    const VkResult result = create(&info, nullptr, &handle);
    if (result != VK_SUCCESS) {
        set_problem("the Vulkan instance could not be made (VkResult %d)", int(result));
        return VK_NULL_HANDLE;
    }
    return handle;
}

// The process's Vulkan instance, for a GPU of the maker `vendor` (a PCI
// vendor ID): its driver alone, where its manifest's name is known and the
// driver is found under it, else every driver.
static VulkanInstance* make_instance(UINT vendor) {
    HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
    if (!loader) {
        set_problem("Vulkan is not installed (no vulkan-1.dll)");
        return nullptr;
    }
    auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        reinterpret_cast<void*>(GetProcAddress(loader, "vkGetInstanceProcAddr")));
    auto enumerate_version =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(get(nullptr, "vkEnumerateInstanceVersion"));
    uint32_t version = VK_API_VERSION_1_0;
    if (enumerate_version)
        enumerate_version(&version);
    if (version < VK_API_VERSION_1_2) {
        set_problem("Vulkan 1.2 or newer is needed");
        return nullptr;
    }
    bool colour_spaces = false;
    VkInstance handle = VK_NULL_HANDLE;
    if (const wchar_t* driver = driver_manifest(vendor)) {
        handle = create_instance(get, driver, &colour_spaces);
        // A driver under another name: the instance has no device, and is made again with every driver.
        uint32_t devices = 0;
        if (handle)
            reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(get(handle, "vkEnumeratePhysicalDevices"))(
                handle, &devices, nullptr);
        if (handle && devices == 0) {
            reinterpret_cast<PFN_vkDestroyInstance>(get(handle, "vkDestroyInstance"))(handle, nullptr);
            handle = VK_NULL_HANDLE;
        }
    }
    if (!handle)
        handle = create_instance(get, nullptr, &colour_spaces);
    if (!handle)
        return nullptr;
    auto instance = new VulkanInstance;
    instance->handle = handle;
    instance->colour_spaces = colour_spaces;
#define LOAD_INSTANCE_FUNCTION(name) instance->name = reinterpret_cast<PFN_##name>(get(handle, #name));
    INSTANCE_FUNCTIONS(LOAD_INSTANCE_FUNCTION)
    return instance;
}

// The instance for GPUs of the maker `vendor`, made the first time one is
// asked for and kept, as is a failure to make it. There is one in practice:
// the presenters are all on the GPU GStreamer decodes on
// (gstreamer_playback.d3d11_device).
static VulkanInstance* vulkan_instance(UINT vendor) {
    static std::mutex lock;
    static auto* made = new std::vector<std::pair<UINT, VulkanInstance*>>();  // never freed, as kept_device
    std::lock_guard<std::mutex> hold(lock);
    for (const auto& [maker, instance] : *made)
        if (maker == vendor)
            return instance;
    VulkanInstance* instance = make_instance(vendor);
    made->emplace_back(vendor, instance);
    return instance;
}

// The render pass and pipeline that draw into images of one format.
struct Drawing {
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

// A Vulkan device on the GPU of GStreamer's Direct3D 11 device, with what
// every presenter on it shares. Made once and kept (making one takes some
// 40 ms), though a lost one is replaced.
struct VulkanDevice {
    VulkanInstance* instance = nullptr;
    LUID adapter = {};
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice handle = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    // vkQueueSubmit, vkQueuePresentKHR and vkQueueWaitIdle need the queue to themselves.
    std::mutex queue_lock;
    VkPhysicalDeviceMemoryProperties memory = {};
    // Bilinear and clamped at the edges, as the Direct3D 11 presenter samples.
    VkSampler sampler = VK_NULL_HANDLE;
    // A frame's two planes (bindings 0 and 1), read with that sampler.
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule vertex_shader = VK_NULL_HANDLE;
    VkShaderModule fragment_shader = VK_NULL_HANDLE;
    // One drawing per format of window image met so far.
    std::mutex drawings_lock;
    std::vector<Drawing> drawings;
    std::atomic<bool> lost{false};
    DEVICE_FUNCTIONS(DECLARE_FUNCTION)

    ~VulkanDevice() {
        if (!handle)
            return;
        vkDeviceWaitIdle(handle);
        for (const Drawing& drawing : drawings) {
            vkDestroyPipeline(handle, drawing.pipeline, nullptr);
            vkDestroyRenderPass(handle, drawing.render_pass, nullptr);
        }
        vkDestroyShaderModule(handle, vertex_shader, nullptr);
        vkDestroyShaderModule(handle, fragment_shader, nullptr);
        vkDestroyPipelineLayout(handle, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(handle, set_layout, nullptr);
        vkDestroySampler(handle, sampler, nullptr);
        vkDestroyDevice(handle, nullptr);
    }
};

// An HRESULT for a Vulkan call's result, the problem noted. A lost device is
// marked so, for the next presenter to make a new one.
static HRESULT vulkan_failure(VulkanDevice& device, VkResult result, const char* doing) {
    if (result == VK_SUCCESS)
        return S_OK;
    set_problem("Vulkan failed %s (VkResult %d)", doing, int(result));
    if (result == VK_ERROR_DEVICE_LOST) {
        device.lost = true;
        return DXGI_ERROR_DEVICE_REMOVED;
    }
    return result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY ? E_OUTOFMEMORY : E_FAIL;
}

// Whether a Direct3D 11 texture of `format` can be imported and sampled.
static bool can_import_texture(const VulkanInstance& instance, VkPhysicalDevice physical, VkFormat format) {
    VkPhysicalDeviceExternalImageFormatInfo external = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
    external.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    VkPhysicalDeviceImageFormatInfo2 info = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &external};
    info.format = format;
    info.type = VK_IMAGE_TYPE_2D;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    VkExternalImageFormatProperties external_properties = {VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 properties = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &external_properties};
    return instance.vkGetPhysicalDeviceImageFormatProperties2(physical, &info, &properties) == VK_SUCCESS &&
           (external_properties.externalMemoryProperties.externalMemoryFeatures &
            VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
}

// Whether a Direct3D 11 fence can be imported as a timeline semaphore.
static bool can_import_fence(const VulkanInstance& instance, VkPhysicalDevice physical) {
    VkSemaphoreTypeCreateInfo type = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkPhysicalDeviceExternalSemaphoreInfo info = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO, &type};
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE_BIT;
    VkExternalSemaphoreProperties properties = {VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    instance.vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &info, &properties);
    return properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
}

// The sampler, layouts and shaders every presenter on the device uses.
static bool make_shared_objects(VulkanDevice& device) {
    VkSamplerCreateInfo sampler = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = VK_LOD_CLAMP_NONE;
    if (device.vkCreateSampler(device.handle, &sampler, nullptr, &device.sampler) != VK_SUCCESS)
        return false;
    VkDescriptorSetLayoutBinding planes[2] = {};
    for (uint32_t binding = 0; binding < 2; binding++) {
        planes[binding].binding = binding;
        planes[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        planes[binding].descriptorCount = 1;
        planes[binding].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        planes[binding].pImmutableSamplers = &device.sampler;
    }
    VkDescriptorSetLayoutCreateInfo set_layout = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_layout.bindingCount = 2;
    set_layout.pBindings = planes;
    if (device.vkCreateDescriptorSetLayout(device.handle, &set_layout, nullptr, &device.set_layout) != VK_SUCCESS)
        return false;
    VkPushConstantRange constants = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                     sizeof(DrawConstants)};
    VkPipelineLayoutCreateInfo pipeline_layout = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout.setLayoutCount = 1;
    pipeline_layout.pSetLayouts = &device.set_layout;
    pipeline_layout.pushConstantRangeCount = 1;
    pipeline_layout.pPushConstantRanges = &constants;
    if (device.vkCreatePipelineLayout(device.handle, &pipeline_layout, nullptr, &device.pipeline_layout) != VK_SUCCESS)
        return false;
    VkShaderModuleCreateInfo vertex = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    vertex.codeSize = sizeof kVertexSpirv;
    vertex.pCode = kVertexSpirv;
    VkShaderModuleCreateInfo fragment = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    fragment.codeSize = sizeof kFragmentSpirv;
    fragment.pCode = kFragmentSpirv;
    return device.vkCreateShaderModule(device.handle, &vertex, nullptr, &device.vertex_shader) == VK_SUCCESS &&
           device.vkCreateShaderModule(device.handle, &fragment, nullptr, &device.fragment_shader) == VK_SUCCESS;
}

// A device on the GPU whose Direct3D 11 adapter is `adapter`, if it can do
// everything the presenter needs; null (the problem noted) if not.
static std::shared_ptr<VulkanDevice> make_device(VulkanInstance* instance, LUID adapter) {
    uint32_t count = 0;
    instance->vkEnumeratePhysicalDevices(instance->handle, &count, nullptr);
    std::vector<VkPhysicalDevice> candidates(count);
    instance->vkEnumeratePhysicalDevices(instance->handle, &count, candidates.data());
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties = {};
    for (VkPhysicalDevice candidate : candidates) {
        VkPhysicalDeviceIDProperties ids = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 candidate_properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &ids};
        instance->vkGetPhysicalDeviceProperties2(candidate, &candidate_properties);
        if (ids.deviceLUIDValid && memcmp(ids.deviceLUID, &adapter, VK_LUID_SIZE) == 0) {
            physical = candidate;
            properties = candidate_properties.properties;
        }
    }
    if (!physical) {
        set_problem("Vulkan has no device on the GPU the video is decoded on");
        return nullptr;
    }
    if (properties.apiVersion < VK_API_VERSION_1_2) {
        set_problem("the GPU's Vulkan driver is older than Vulkan 1.2");
        return nullptr;
    }
    if (properties.limits.maxPushConstantsSize < sizeof(DrawConstants)) {
        set_problem("the GPU's Vulkan driver takes too few shader constants");
        return nullptr;
    }
    instance->vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    instance->vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, available.data());
    const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    for (const char* name : extensions)
        if (!has_extension(available, name)) {
            set_problem("the GPU's Vulkan driver lacks %s", name);
            return nullptr;
        }
    VkPhysicalDeviceVulkan12Features supported12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 supported = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &supported12};
    instance->vkGetPhysicalDeviceFeatures2(physical, &supported);
    if (!supported12.timelineSemaphore) {
        set_problem("the GPU's Vulkan driver lacks timeline semaphores");
        return nullptr;
    }
    for (VkFormat format : {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R16_UNORM, VK_FORMAT_R16G16_UNORM})
        if (!can_import_texture(*instance, physical, format)) {
            set_problem("the GPU's Vulkan driver cannot import Direct3D 11 textures (format %d)", int(format));
            return nullptr;
        }
    if (!can_import_fence(*instance, physical)) {
        set_problem("the GPU's Vulkan driver cannot import Direct3D 11 fences");
        return nullptr;
    }
    // A queue that draws and can present to windows.
    instance->vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    instance->vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    uint32_t family = 0;
    while (family < count && !((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                               instance->vkGetPhysicalDeviceWin32PresentationSupportKHR(physical, family)))
        family++;
    if (family == count) {
        set_problem("no Vulkan queue on the GPU draws into windows");
        return nullptr;
    }
    const float priority = 1;
    VkDeviceQueueCreateInfo queue = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan12Features features12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features12};
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.enabledExtensionCount = uint32_t(std::size(extensions));
    info.ppEnabledExtensionNames = extensions;
    auto device = std::make_shared<VulkanDevice>();
    device->instance = instance;
    device->adapter = adapter;
    device->physical = physical;
    device->queue_family = family;
    const VkResult result = instance->vkCreateDevice(physical, &info, nullptr, &device->handle);
    if (result != VK_SUCCESS) {
        device->handle = VK_NULL_HANDLE;
        set_problem("the Vulkan device could not be made (VkResult %d)", int(result));
        return nullptr;
    }
#define LOAD_DEVICE_FUNCTION(name) \
    device->name = reinterpret_cast<PFN_##name>(instance->vkGetDeviceProcAddr(device->handle, #name));
    DEVICE_FUNCTIONS(LOAD_DEVICE_FUNCTION)
    device->vkGetDeviceQueue(device->handle, family, 0, &device->queue);
    instance->vkGetPhysicalDeviceMemoryProperties(physical, &device->memory);
    if (!make_shared_objects(*device)) {
        set_problem("the Vulkan presenter's shaders could not be set up");
        return nullptr;
    }
    return device;
}

// The device last made, kept for the process: never released, since at the
// process's exit the driver may be unloaded before this library.
static std::mutex kept_device_lock;
static auto* kept_device = new std::shared_ptr<VulkanDevice>();

static std::shared_ptr<VulkanDevice> device_for(VulkanInstance* instance, LUID adapter) {
    std::lock_guard<std::mutex> hold(kept_device_lock);
    std::shared_ptr<VulkanDevice>& kept = *kept_device;
    if (kept && !kept->lost && memcmp(&kept->adapter, &adapter, sizeof adapter) == 0)
        return kept;
    std::shared_ptr<VulkanDevice> made = make_device(instance, adapter);
    if (made)
        kept = made;
    return made;
}

// The render pass and pipeline that draw into images of `format`, made the
// first time a window needs them, and kept with the device.
static bool drawing_for(VulkanDevice& device, VkFormat format, Drawing* drawing) {
    std::lock_guard<std::mutex> hold(device.drawings_lock);
    for (const Drawing& made : device.drawings)
        if (made.format == format) {
            *drawing = made;
            return true;
        }
    Drawing made;
    made.format = format;
    // The window's image: cleared, drawn into, then presented.
    VkAttachmentDescription image = {};
    image.format = format;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    image.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    image.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    image.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference colour = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colour;
    // The clear waits for the image, which the submission waits for at the colour output stage.
    VkSubpassDependency acquired = {};
    acquired.srcSubpass = VK_SUBPASS_EXTERNAL;
    acquired.dstSubpass = 0;
    acquired.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    acquired.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    acquired.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo pass = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 1;
    pass.pAttachments = &image;
    pass.subpassCount = 1;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = 1;
    pass.pDependencies = &acquired;
    if (device.vkCreateRenderPass(device.handle, &pass, nullptr, &made.render_pass) != VK_SUCCESS)
        return false;

    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = device.vertex_shader;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = device.fragment_shader;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertices = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    VkPipelineViewportStateCreateInfo viewport = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterization = {
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo multisample = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState written = {};
    written.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &written;
    // Set for each frame: where it goes in the window.
    const VkDynamicState per_frame[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = per_frame;
    VkGraphicsPipelineCreateInfo pipeline = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertices;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &rasterization;
    pipeline.pMultisampleState = &multisample;
    pipeline.pColorBlendState = &blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = device.pipeline_layout;
    pipeline.renderPass = made.render_pass;
    if (device.vkCreateGraphicsPipelines(device.handle, VK_NULL_HANDLE, 1, &pipeline, nullptr, &made.pipeline) !=
        VK_SUCCESS) {
        device.vkDestroyRenderPass(device.handle, made.render_pass, nullptr);
        return false;
    }
    device.drawings.push_back(made);
    *drawing = made;
    return true;
}

// ---------------------------------------------------------------------------
// Direct3D 11's part of the hand-over

// A frame's two planes copied, code for code, into the textures shared with
// Vulkan: a value read from a UNORM texture as a float and written to one of
// the same depth is the same code. Only the part of the frame the drawing
// samples is copied (`region`, in chroma texels: left, top, right, bottom),
// by a thread per luma texel of it counted from its corner; the threads of
// its first chroma-sized quarter copy a chroma texel too. The dispatch's size
// comes from a buffer (DispatchIndirect), so one recorded command list fits
// every region. Of the shapes tried, groups of 16x16 threads were the fastest
// on both an RTX 5090 and a Core Ultra 9 285K's iGPU: a whole 4K P010 frame,
// read from video memory, took 0.026 and 0.82 ms (a thread per chroma texel
// and its 2x2 luma texels: 0.025 and 1.19 ms).
static const char copy_planes_source[] = R"(
Texture2D<float> luma_in : register(t0);
Texture2D<float2> chroma_in : register(t1);
RWTexture2D<unorm float> luma_out : register(u0);
RWTexture2D<unorm float2> chroma_out : register(u1);
cbuffer Region : register(b0) {
    uint4 region;
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 size = region.zw - region.xy;
    if (all(id.xy < size * 2)) {
        uint2 luma = region.xy * 2 + id.xy;
        luma_out[luma] = luma_in[luma];
    }
    if (all(id.xy < size)) {
        uint2 chroma = region.xy + id.xy;
        chroma_out[chroma] = chroma_in[chroma];
    }
}
)";

// The thread groups of the copy shader that cover `region`.
static void copy_groups_for(const UINT region[4], UINT groups[3]) {
    groups[0] = ((region[2] - region[0]) * 2 + 15) / 16;
    groups[1] = ((region[3] - region[1]) * 2 + 15) / 16;
    groups[2] = 1;
}

// The copy shader's bytecode, compiled once a process and never released (as
// d3d11_tonemap.cpp's shaders: at the process's exit the compiler's library
// may be unloaded before this one).
static std::mutex copy_shader_lock;
static ID3DBlob* copy_shader_code = nullptr;

static HRESULT copy_shader(ID3DBlob** code) {
    std::lock_guard<std::mutex> hold(copy_shader_lock);
    if (!copy_shader_code) {
        ComPtr<ID3DBlob> errors;
        const HRESULT hr = D3DCompile(copy_planes_source, sizeof copy_planes_source - 1, nullptr, nullptr, nullptr,
                                      "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &copy_shader_code, &errors);
        if (FAILED(hr)) {
            set_problem("the frame hand-over's shader did not compile (0x%08x)", unsigned(hr));
            return hr;
        }
    }
    *code = copy_shader_code;
    return S_OK;
}

// Whether Direct3D 11 compute shaders can write textures of `format`.
static bool can_write(ID3D11Device* device, DXGI_FORMAT format) {
    D3D11_FEATURE_DATA_FORMAT_SUPPORT2 support = {format, 0};
    return SUCCEEDED(device->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &support, sizeof support)) &&
           (support.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE);
}

// The part of a frame of `width` x `height` that drawing `source` samples
// (x, y, width, height in luma pixels; all of it if null), in chroma texels:
// left, top, right, bottom. The bilinear filter reaches up to a texel past
// the part drawn, so a chroma texel more is taken all round.
static void sampled_region(UINT width, UINT height, const int* source, UINT region[4]) {
    const int chroma_width = int(width / 2), chroma_height = int(height / 2);
    int left = 0, top = 0, right = chroma_width, bottom = chroma_height;
    if (source) {
        left = std::clamp(source[0] / 2 - 1, 0, chroma_width);
        top = std::clamp(source[1] / 2 - 1, 0, chroma_height);
        right = std::clamp((source[0] + source[2] + 1) / 2 + 1, left, chroma_width);
        bottom = std::clamp((source[1] + source[3] + 1) / 2 + 1, top, chroma_height);
    }
    region[0] = UINT(left);
    region[1] = UINT(top);
    region[2] = UINT(right);
    region[3] = UINT(bottom);
}

// ---------------------------------------------------------------------------
// The presenter

// A texture Direct3D 11 writes and Vulkan samples: one plane of a frame.
struct SharedPlane {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11UnorderedAccessView> writer;  // Direct3D 11's view of it, for the copy
    VkImage image = VK_NULL_HANDLE;            // the same texture in Vulkan
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;         // Vulkan's view of it, for the drawing
};

// One frame's way from Direct3D 11 to Vulkan: its two planes, and the
// descriptor set the drawing reads them through.
struct HandoverSlot {
    SharedPlane luma, chroma;
    VkDescriptorSet descriptors = VK_NULL_HANDLE;
    // The number of the last frame drawn from it: Vulkan's drawn timeline
    // reaches it when Vulkan has finished with the slot.
    uint64_t last_frame = 0;
};

// A source texture's shader views, kept, and the Direct3D 11 command lists
// that copy it into each hand-over slot, recorded the first time and run
// again for every frame after (the part of the frame they copy comes from
// buffers): recording one each frame cost some 14 us with the device lock
// held.
struct SourceViews {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> luma, chroma;
    ComPtr<ID3D11CommandList> copy[kHandoverSlots];
};

// A window image's drawing of a slot, recorded once and submitted again for
// every frame that draws the same way: re-recorded only when the slot is
// made again or the drawing's constants, place or background change.
struct RecordedDrawing {
    VkCommandBuffer commands = VK_NULL_HANDLE;
    bool recorded = false;
    uint64_t slots_made = 0;  // Presenter::slots_made as it was recorded
    DrawConstants constants = {};
    VkViewport viewport = {};
    float clear[4] = {};
};

// The window's swapchain and what drawing into its images needs.
struct Swapchain {
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    VkColorSpaceKHR colour_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D extent = {};
    Drawing drawing;
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkFramebuffer> framebuffers;
    // One per image: signalled when the drawing into it is done; its present waits for it.
    std::vector<VkSemaphore> ready_to_present;
};

// A frame of drawing until the GPU has done it: what its submission waits on.
struct FrameInFlight {
    VkFence drawn = VK_NULL_HANDLE;         // signalled when the frame's drawing is done
    VkSemaphore acquired = VK_NULL_HANDLE;  // signalled when the image it is drawn into is free
};

// A frame handed over by vmaf_present_texture, to be drawn and presented by
// vmaf_present_flip once the caller has let go of the device lock.
struct PendingDraw {
    bool waiting = false;
    int slot = 0;
    DrawConstants constants = {};
    float source[4] = {};  // the part of the frame drawn, in its pixels
    bool fitted = true;    // fitted into the window; else drawn at `target`
    int target[4] = {};
    float clear[4] = {};
    VkColorSpaceKHR colour_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
};

struct Presenter {
    VulkanInstance* instance = nullptr;
    std::shared_ptr<VulkanDevice> vulkan;

    // Direct3D 11: GStreamer's device, and the hand-over's work on it.
    ComPtr<ID3D11Device> d3d;
    ComPtr<ID3D11DeviceContext> immediate;
    ComPtr<ID3D11DeviceContext4> immediate4;
    ComPtr<ID3D11DeviceContext> deferred;
    ComPtr<ID3D11ComputeShader> copy_planes;
    // The part of the frame the copy shader copies (its constant buffer), its
    // dispatch's thread groups (DispatchIndirect's arguments), and the region
    // they were set for.
    ComPtr<ID3D11Buffer> copy_region, copy_groups;
    UINT copy_region_now[4] = {};
    SourceViews source_views[kKeptSourceViews];
    int next_source_view = 0;
    ComPtr<ID3D11Texture2D> source_copy;  // frames a shader cannot read, copied first

    // The hand-over. Frames are numbered from 1 as they are handed over.
    // Direct3D 11 signals the copied fence with a frame's number once it has
    // copied it, and Vulkan's drawing of that frame waits for it.
    ComPtr<ID3D11Fence> copied_fence;
    VkSemaphore copied = VK_NULL_HANDLE;       // the same fence in Vulkan
    VkSemaphore drawn = VK_NULL_HANDLE;        // Vulkan's own timeline: a frame's number once it is drawn
    uint64_t frames_handed_over = 0;
    uint64_t last_frame_submitted = 0;         // the last frame whose drawing was given to the GPU
    HandoverSlot slots[kHandoverSlots];
    uint64_t slots_made = 0;                   // how many times the slots were made
    UINT slot_width = 0, slot_height = 0;
    bool slot_deep = false;                    // R16 planes for P010, else R8 for NV12
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    // The window and its swapchain.
    HWND window = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    bool hdr_display = false;                  // 10-bit images, and HDR10 where the display takes it
    bool hdr10_offered = false;                // the surface offers HDR10, as of the last swapchain made
    Swapchain swapchain;
    bool swapchain_stale = false;              // to be made again before the next image is acquired

    // Drawing.
    VkCommandPool command_pool = VK_NULL_HANDLE;
    // By window image and slot (image * kHandoverSlots + slot); kept when the
    // swapchain is made again, and recorded again.
    std::vector<RecordedDrawing> recorded;
    FrameInFlight frames[kFramesInFlight];
    uint64_t frames_drawn = 0;                 // frames whose drawing was submitted, of any slot
    VkFence acquired_free = VK_NULL_HANDLE;    // signalled when the image last acquired is free
    int acquired = -1;                         // the image the next frame is drawn into, once acquired
    bool image_free = false;                   // and the display has let go of it
    PendingDraw pending;
};

static FrameInFlight& next_frame(Presenter* p) {
    return p->frames[p->frames_drawn % kFramesInFlight];
}

// Waits until the GPU has finished everything this presenter gave it: its
// frames' drawing, and its presents' waits. False if the GPU did not finish in
// time, when what it may still use must not be freed.
static bool wait_idle(Presenter* p) {
    VulkanDevice& vk = *p->vulkan;
    for (FrameInFlight& frame : p->frames)
        if (frame.drawn && vk.vkWaitForFences(vk.handle, 1, &frame.drawn, VK_TRUE, kGpuTimeoutNs) != VK_SUCCESS)
            return false;
    std::lock_guard<std::mutex> hold(vk.queue_lock);
    return vk.vkQueueWaitIdle(vk.queue) == VK_SUCCESS;
}

// Waits until Vulkan has drawn frame `number`.
static HRESULT wait_drawn(Presenter* p, uint64_t number) {
    VulkanDevice& vk = *p->vulkan;
    // Usually drawn long since: reading the counter (0.02 us) costs far less
    // than a wait, which took 14 us a frame in playback on an RTX 5090 with
    // nothing to wait for.
    uint64_t drawn = 0;
    if (vk.vkGetSemaphoreCounterValue(vk.handle, p->drawn, &drawn) == VK_SUCCESS && drawn >= number)
        return S_OK;
    VkSemaphoreWaitInfo wait = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &p->drawn;
    wait.pValues = &number;
    return vulkan_failure(vk, vk.vkWaitSemaphores(vk.handle, &wait, kGpuTimeoutNs), "waiting for a frame's drawing");
}

static void free_swapchain(Presenter* p) {
    VulkanDevice& vk = *p->vulkan;
    Swapchain& swapchain = p->swapchain;
    for (VkFramebuffer framebuffer : swapchain.framebuffers)
        vk.vkDestroyFramebuffer(vk.handle, framebuffer, nullptr);
    for (VkImageView view : swapchain.views)
        vk.vkDestroyImageView(vk.handle, view, nullptr);
    for (VkSemaphore semaphore : swapchain.ready_to_present)
        vk.vkDestroySemaphore(vk.handle, semaphore, nullptr);
    if (swapchain.handle)
        vk.vkDestroySwapchainKHR(vk.handle, swapchain.handle, nullptr);
    swapchain = Swapchain{};
    // The recorded drawings were of its framebuffers.
    for (RecordedDrawing& drawing : p->recorded)
        drawing.recorded = false;
}

// Makes the window's swapchain, or makes it again: at the window's size, and
// in `colour_space` if the display takes it (else sRGB). S_FALSE if the
// window has no area to draw in (a minimized window).
static HRESULT make_swapchain(Presenter* p, VkColorSpaceKHR colour_space) {
    VulkanDevice& vk = *p->vulkan;
    VulkanInstance& instance = *p->instance;
    VkSurfaceCapabilitiesKHR capabilities;
    VkResult result = instance.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical, p->surface, &capabilities);
    if (result != VK_SUCCESS)
        return vulkan_failure(vk, result, "reading the video window's surface");
    VkExtent2D extent = capabilities.currentExtent;
    if (extent.width == UINT32_MAX) {  // the surface takes its size from the swapchain
        RECT client = {};
        GetClientRect(p->window, &client);
        extent.width = std::clamp(uint32_t(std::max<LONG>(client.right, 0)), capabilities.minImageExtent.width,
                                  capabilities.maxImageExtent.width);
        extent.height = std::clamp(uint32_t(std::max<LONG>(client.bottom, 0)), capabilities.minImageExtent.height,
                                   capabilities.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0)
        return S_FALSE;

    uint32_t count = 0;
    instance.vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical, p->surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> offered(count);
    instance.vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical, p->surface, &count, offered.data());
    auto first_offered = [&](std::initializer_list<VkFormat> formats, VkColorSpaceKHR space) {
        for (VkFormat format : formats)
            for (const VkSurfaceFormatKHR& surface_format : offered)
                if (surface_format.format == format && surface_format.colorSpace == space)
                    return format;
        return VK_FORMAT_UNDEFINED;
    };
    // An HDR display gets 10-bit images, as the Direct3D 11 presenter's
    // R10G10B10A2 buffers, for HDR10 and sRGB alike; others get 8-bit ones.
    const std::initializer_list<VkFormat> ten_bit = {VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                                     VK_FORMAT_A2R10G10B10_UNORM_PACK32};
    const std::initializer_list<VkFormat> eight_bit = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    const VkFormat hdr10 = p->hdr_display ? first_offered(ten_bit, VK_COLOR_SPACE_HDR10_ST2084_EXT) : VK_FORMAT_UNDEFINED;
    p->hdr10_offered = hdr10 != VK_FORMAT_UNDEFINED;
    VkFormat format = VK_FORMAT_UNDEFINED;
    if (colour_space == VK_COLOR_SPACE_HDR10_ST2084_EXT && p->hdr10_offered) {
        format = hdr10;
    } else {
        colour_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        if (p->hdr_display)
            format = first_offered(ten_bit, colour_space);
        if (format == VK_FORMAT_UNDEFINED)
            format = first_offered(eight_bit, colour_space);
    }
    if (format == VK_FORMAT_UNDEFINED) {
        set_problem("the video window takes no image format the presenter draws");
        return E_FAIL;
    }
    Drawing drawing;
    if (!drawing_for(vk, format, &drawing)) {
        set_problem("the Vulkan presenter's pipeline could not be made");
        return E_FAIL;
    }

    // The old swapchain's images must be out of use before it goes.
    if (p->swapchain.handle && !wait_idle(p))
        return vulkan_failure(vk, VK_TIMEOUT, "finishing the frames before resizing");
    VkSwapchainCreateInfoKHR info = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = p->surface;
    // Three images: one on screen, and at most two waiting for the display.
    info.minImageCount = std::max(3u, capabilities.minImageCount);
    if (capabilities.maxImageCount)
        info.minImageCount = std::min(info.minImageCount, capabilities.maxImageCount);
    info.imageFormat = format;
    info.imageColorSpace = colour_space;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = capabilities.currentTransform;
    // Opaque, where the surface allows it; else the first way it takes.
    for (VkCompositeAlphaFlagBitsKHR alpha :
         {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
          VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR})
        if (capabilities.supportedCompositeAlpha & alpha) {
            info.compositeAlpha = alpha;
            break;
        }
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // at the display's refresh, as Present(1, 0)
    info.clipped = VK_TRUE;
    info.oldSwapchain = p->swapchain.handle;
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    result = vk.vkCreateSwapchainKHR(vk.handle, &info, nullptr, &handle);
    free_swapchain(p);  // retired, whether or not the new one was made
    p->acquired = -1;
    p->image_free = false;
    if (result != VK_SUCCESS)
        return vulkan_failure(vk, result, "making the video window's swapchain");

    Swapchain& swapchain = p->swapchain;
    swapchain.handle = handle;
    swapchain.colour_space = colour_space;
    swapchain.extent = extent;
    swapchain.drawing = drawing;
    vk.vkGetSwapchainImagesKHR(vk.handle, handle, &count, nullptr);
    swapchain.images.resize(count);
    vk.vkGetSwapchainImagesKHR(vk.handle, handle, &count, swapchain.images.data());
    if (p->recorded.size() < size_t(count) * kHandoverSlots)
        p->recorded.resize(size_t(count) * kHandoverSlots);
    for (VkImage image : swapchain.images) {
        VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        swapchain.views.push_back(VK_NULL_HANDLE);
        result = vk.vkCreateImageView(vk.handle, &view, nullptr, &swapchain.views.back());
        if (result != VK_SUCCESS)
            return vulkan_failure(vk, result, "making the window's image views");
        VkFramebufferCreateInfo framebuffer = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = drawing.render_pass;
        framebuffer.attachmentCount = 1;
        framebuffer.pAttachments = &swapchain.views.back();
        framebuffer.width = extent.width;
        framebuffer.height = extent.height;
        framebuffer.layers = 1;
        swapchain.framebuffers.push_back(VK_NULL_HANDLE);
        result = vk.vkCreateFramebuffer(vk.handle, &framebuffer, nullptr, &swapchain.framebuffers.back());
        if (result != VK_SUCCESS)
            return vulkan_failure(vk, result, "making the window's framebuffers");
        VkSemaphoreCreateInfo semaphore = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        swapchain.ready_to_present.push_back(VK_NULL_HANDLE);
        result = vk.vkCreateSemaphore(vk.handle, &semaphore, nullptr, &swapchain.ready_to_present.back());
        if (result != VK_SUCCESS)
            return vulkan_failure(vk, result, "making the window's semaphores");
    }
    p->swapchain_stale = false;
    return S_OK;
}

// Gives back an image acquired but not drawn into: an empty submission waits
// for its acquire semaphore, which can then be used again. The image itself
// goes with its swapchain.
static HRESULT give_back_acquired(Presenter* p) {
    if (p->acquired < 0)
        return S_OK;
    VulkanDevice& vk = *p->vulkan;
    FrameInFlight& frame = next_frame(p);
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &frame.acquired;
    submit.pWaitDstStageMask = &stage;
    vk.vkResetFences(vk.handle, 1, &frame.drawn);
    VkResult result;
    {
        std::lock_guard<std::mutex> hold(vk.queue_lock);
        result = vk.vkQueueSubmit(vk.queue, 1, &submit, frame.drawn);
    }
    if (result == VK_SUCCESS)
        result = vk.vkWaitForFences(vk.handle, 1, &frame.drawn, VK_TRUE, kGpuTimeoutNs);
    p->acquired = -1;
    p->image_free = false;
    return vulkan_failure(vk, result, "giving back a window image");
}

static void free_plane(Presenter* p, SharedPlane& plane) {
    VulkanDevice& vk = *p->vulkan;
    vk.vkDestroyImageView(vk.handle, plane.view, nullptr);
    vk.vkDestroyImage(vk.handle, plane.image, nullptr);
    vk.vkFreeMemory(vk.handle, plane.memory, nullptr);
    plane = SharedPlane{};
}

static void free_slot(Presenter* p, HandoverSlot& slot) {
    free_plane(p, slot.luma);
    free_plane(p, slot.chroma);
    slot.last_frame = 0;
}

// Makes one shared plane: a Direct3D 11 texture that its compute shaders can
// write, shared with Vulkan and imported there.
static HRESULT make_shared_plane(Presenter* p, UINT width, UINT height, DXGI_FORMAT format, VkFormat vulkan_format,
                                 SharedPlane& plane) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
    HRESULT hr = p->d3d->CreateTexture2D(&desc, nullptr, &plane.texture);
    if (SUCCEEDED(hr))
        hr = p->d3d->CreateUnorderedAccessView(plane.texture.Get(), nullptr, &plane.writer);
    ComPtr<IDXGIResource1> resource;
    if (SUCCEEDED(hr))
        hr = plane.texture.As(&resource);
    HANDLE handle = nullptr;
    if (SUCCEEDED(hr))
        hr = resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr,
                                          &handle);
    if (FAILED(hr)) {
        set_problem("a texture shared with Vulkan could not be made (0x%08x)", unsigned(hr));
        return hr;
    }

    VulkanDevice& vk = *p->vulkan;
    VkExternalMemoryImageCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    VkImageCreateInfo image = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &external};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = vulkan_format;
    image.extent = {width, height, 1};
    image.mipLevels = image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkResult result = vk.vkCreateImage(vk.handle, &image, nullptr, &plane.image);
    VkMemoryWin32HandlePropertiesKHR handle_properties = {VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
    if (result == VK_SUCCESS)
        result = vk.vkGetMemoryWin32HandlePropertiesKHR(vk.handle, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
                                                        handle, &handle_properties);
    if (result == VK_SUCCESS) {
        VkMemoryRequirements requirements;
        vk.vkGetImageMemoryRequirements(vk.handle, plane.image, &requirements);
        const uint32_t allowed = requirements.memoryTypeBits & handle_properties.memoryTypeBits;
        uint32_t type = 0;
        while (type < vk.memory.memoryTypeCount && !(allowed & (1u << type)))
            type++;
        VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated.image = plane.image;
        VkImportMemoryWin32HandleInfoKHR import = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR, &dedicated};
        import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
        import.handle = handle;
        VkMemoryAllocateInfo allocate = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = type;
        result = type < vk.memory.memoryTypeCount ? vk.vkAllocateMemory(vk.handle, &allocate, nullptr, &plane.memory)
                                                  : VK_ERROR_FEATURE_NOT_PRESENT;
    }
    // Importing does not take the handle over (Win32 handles stay the caller's).
    CloseHandle(handle);
    if (result == VK_SUCCESS)
        result = vk.vkBindImageMemory(vk.handle, plane.image, plane.memory, 0);
    if (result == VK_SUCCESS) {
        VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = plane.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = vulkan_format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        result = vk.vkCreateImageView(vk.handle, &view, nullptr, &plane.view);
    }
    return vulkan_failure(vk, result, "importing a texture shared by Direct3D 11");
}

// Makes the hand-over slots for frames of `width` x `height` (NV12, or P010 if
// `deep`), unless they are made already.
static HRESULT ensure_slots(Presenter* p, UINT width, UINT height, bool deep) {
    if (p->slots[0].luma.image && width == p->slot_width && height == p->slot_height && deep == p->slot_deep)
        return S_OK;
    // Vulkan must have finished with the slots before they go.
    if (p->last_frame_submitted) {
        const HRESULT hr = wait_drawn(p, p->last_frame_submitted);
        if (FAILED(hr))
            return hr;
    }
    for (HandoverSlot& slot : p->slots)
        free_slot(p, slot);
    p->slot_width = p->slot_height = 0;
    for (SourceViews& views : p->source_views)
        for (ComPtr<ID3D11CommandList>& copy : views.copy)
            copy.Reset();  // they wrote into the old slots' textures
    const DXGI_FORMAT luma = deep ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
    const DXGI_FORMAT chroma = deep ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R8G8_UNORM;
    const VkFormat vulkan_luma = deep ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
    const VkFormat vulkan_chroma = deep ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM;
    for (HandoverSlot& slot : p->slots) {
        HRESULT hr = make_shared_plane(p, width, height, luma, vulkan_luma, slot.luma);
        if (SUCCEEDED(hr))
            hr = make_shared_plane(p, width / 2, height / 2, chroma, vulkan_chroma, slot.chroma);
        if (FAILED(hr)) {
            for (HandoverSlot& made : p->slots)
                free_slot(p, made);
            return hr;
        }
        // The drawing reads the two planes through the slot's descriptor set.
        VkDescriptorImageInfo planes[2] = {};
        planes[0].imageView = slot.luma.view;
        planes[1].imageView = slot.chroma.view;
        VkWriteDescriptorSet writes[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
                                          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        for (uint32_t binding = 0; binding < 2; binding++) {
            // Shared textures stay in the general layout, the one Direct3D 11 knows.
            planes[binding].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            writes[binding].dstSet = slot.descriptors;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[binding].pImageInfo = &planes[binding];
        }
        p->vulkan->vkUpdateDescriptorSets(p->vulkan->handle, 2, writes, 0, nullptr);
    }
    p->slots_made++;  // the drawings recorded from the old slots go too
    p->slot_width = width;
    p->slot_height = height;
    p->slot_deep = deep;
    return S_OK;
}

static SourceViews* source_views(Presenter* p, ID3D11Texture2D* texture, DXGI_FORMAT format) {
    for (SourceViews& views : p->source_views)
        if (views.texture.Get() == texture)
            return &views;
    SourceViews& views = p->source_views[p->next_source_view];
    p->next_source_view = (p->next_source_view + 1) % kKeptSourceViews;
    views = SourceViews{};
    views.texture = texture;
    const bool deep = format == DXGI_FORMAT_P010;
    D3D11_SHADER_RESOURCE_VIEW_DESC desc = {};
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipLevels = 1;
    desc.Format = deep ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
    HRESULT hr = p->d3d->CreateShaderResourceView(texture, &desc, &views.luma);
    desc.Format = deep ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R8G8_UNORM;
    if (SUCCEEDED(hr))
        hr = p->d3d->CreateShaderResourceView(texture, &desc, &views.chroma);
    if (FAILED(hr)) {
        views = SourceViews{};
        return nullptr;
    }
    return &views;
}

// The texture a shader can read `frame`'s `subresource` from, and its
// description. It is the frame's own texture, or a copy in a texture of the
// presenter's for a slice of a decoder's texture array or a texture bound for
// decoding only, as d3d11h264dec's are (as d3d11_tonemap.cpp's
// present_source makes). Null if the copy could not be made.
static ID3D11Texture2D* readable_source(Presenter* p, ID3D11Texture2D* frame, UINT subresource,
                                        D3D11_TEXTURE2D_DESC& desc) {
    frame->GetDesc(&desc);
    if (desc.ArraySize == 1 && subresource == 0 && (desc.BindFlags & D3D11_BIND_SHADER_RESOURCE))
        return frame;
    D3D11_TEXTURE2D_DESC own = desc;
    own.MipLevels = own.ArraySize = 1;
    own.SampleDesc.Count = 1;
    own.SampleDesc.Quality = 0;
    own.Usage = D3D11_USAGE_DEFAULT;
    own.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    own.CPUAccessFlags = own.MiscFlags = 0;
    D3D11_TEXTURE2D_DESC have = {};
    if (p->source_copy)
        p->source_copy->GetDesc(&have);
    if (!p->source_copy || have.Width != own.Width || have.Height != own.Height || have.Format != own.Format) {
        p->source_copy.Reset();
        if (FAILED(p->d3d->CreateTexture2D(&own, nullptr, &p->source_copy)))
            return nullptr;
    }
    p->immediate->CopySubresourceRegion(p->source_copy.Get(), 0, 0, 0, 0, frame, subresource, nullptr);
    desc = own;
    return p->source_copy.Get();
}

// Direct3D 11's part of drawing a frame. Its planes are copied into the next
// hand-over slot, only the part that drawing `source` samples (x, y, width,
// height in the frame's pixels; all of it if null), and the copied fence is
// signalled with the frame's number. Gives the slot's index, and the
// description of the texture the frame was read from (`desc`). Called with
// GStreamer's device lock held.
static HRESULT hand_over(Presenter* p, ID3D11Resource* frame, UINT subresource, const int* source,
                         D3D11_TEXTURE2D_DESC& desc, int* handed) {
    ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = frame->QueryInterface(IID_PPV_ARGS(&texture));
    if (FAILED(hr))
        return hr;
    texture->GetDesc(&desc);
    if (desc.Format != DXGI_FORMAT_NV12 && desc.Format != DXGI_FORMAT_P010) {
        set_problem("frames of DXGI format %d are not drawn (only NV12 and P010)", int(desc.Format));
        return DXGI_ERROR_UNSUPPORTED;
    }
    ID3D11Texture2D* readable = readable_source(p, texture.Get(), subresource, desc);
    if (!readable)
        return E_OUTOFMEMORY;
    SourceViews* views = source_views(p, readable, desc.Format);
    if (!views)
        return E_FAIL;
    hr = ensure_slots(p, desc.Width, desc.Height, desc.Format == DXGI_FORMAT_P010);
    if (FAILED(hr))
        return hr;
    const uint64_t number = p->frames_handed_over + 1;
    const int index = int(number % kHandoverSlots);
    HandoverSlot& slot = p->slots[index];
    // Vulkan has drawn the frame the slot held, two frames ago, long since;
    // this makes sure before the slot is written again.
    if (slot.last_frame) {
        hr = wait_drawn(p, slot.last_frame);
        if (FAILED(hr))
            return hr;
    }
    UINT region[4];
    sampled_region(desc.Width, desc.Height, source, region);
    if (memcmp(region, p->copy_region_now, sizeof region) != 0) {
        UINT groups[3];
        copy_groups_for(region, groups);
        p->immediate->UpdateSubresource(p->copy_region.Get(), 0, nullptr, region, 0, 0);
        p->immediate->UpdateSubresource(p->copy_groups.Get(), 0, nullptr, groups, 0, 0);
        memcpy(p->copy_region_now, region, sizeof region);
    }
    ComPtr<ID3D11CommandList>& copy = views->copy[index];
    if (!copy) {
        ID3D11DeviceContext* context = p->deferred.Get();
        context->CSSetShader(p->copy_planes.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, p->copy_region.GetAddressOf());
        ID3D11ShaderResourceView* inputs[2] = {views->luma.Get(), views->chroma.Get()};
        context->CSSetShaderResources(0, 2, inputs);
        ID3D11UnorderedAccessView* outputs[2] = {slot.luma.writer.Get(), slot.chroma.writer.Get()};
        context->CSSetUnorderedAccessViews(0, 2, outputs, nullptr);
        context->DispatchIndirect(p->copy_groups.Get(), 0);
        hr = context->FinishCommandList(FALSE, &copy);
        if (FAILED(hr))
            return hr;
    }
    // Run on GStreamer's context, its own state kept as it was.
    p->immediate->ExecuteCommandList(copy.Get(), TRUE);
    hr = p->immediate4->Signal(p->copied_fence.Get(), number);
    if (FAILED(hr))
        return hr;
    // Sent to the GPU now, rather than whenever Direct3D 11 would: Vulkan's
    // drawing waits for the fence.
    p->immediate->Flush();
    p->frames_handed_over = number;
    *handed = index;
    return S_OK;
}

// The shader's constants for a frame read from a texture of `texture_width` x
// `texture_height`. `source`: the part drawn (x, y, width, height in the
// texture's pixels). `to_rgb`: 12 floats, rows of Y, U, V coefficients and an
// offset. `shading`: null for the colours as they come, or 16 floats: 1 to map
// HDR to SDR or 2 to keep it HDR (PQ in BT.2020); the transfer, 1 PQ or 2 HLG;
// the source primaries' luminance weights (3); linear source RGB to linear
// BT.709 or BT.2020 by rows (9); the light mapped to SDR white and HLG's
// display peak (nits); SDR white (nits). The same as d3d11_tonemap.cpp's
// present_constants takes.
static DrawConstants draw_constants(UINT texture_width, UINT texture_height, const float source[4],
                                    const float* to_rgb, const float* shading) {
    DrawConstants c = {};
    c.source[0] = source[0] / texture_width;
    c.source[1] = source[1] / texture_height;
    c.source[2] = source[2] / texture_width;
    c.source[3] = source[3] / texture_height;
    for (int row = 0; row < 3; row++)
        for (int i = 0; i < 4; i++)
            c.to_rgb[row][i] = to_rgb[4 * row + i];
    if (shading) {
        c.shade = shading[0];
        c.kind = shading[1];
        for (int i = 0; i < 3; i++)
            c.luma[i] = shading[2 + i];
        for (int row = 0; row < 3; row++)
            for (int i = 0; i < 3; i++)
                c.to_output[row][i] = shading[5 + 3 * row + i];
        c.peak = shading[14];
        c.luma[3] = shading[15];
    }
    return c;
}

// Where the frame goes in an image of `image` size: `target` (x, y, width,
// height in the image's pixels), or, without one, `source`'s size fitted into
// the image and letterboxed, on whole pixels, as the Direct3D 11 presenter
// places it.
static VkViewport placed(const float source[4], const int* target, VkExtent2D image) {
    float t[4];
    if (target) {
        for (int i = 0; i < 4; i++)
            t[i] = float(target[i]);
    } else {
        const UINT width = image.width, height = image.height;
        const float scale = std::min(width / source[2], height / source[3]);
        t[2] = std::round(source[2] * scale);
        t[3] = std::round(source[3] * scale);
        t[0] = std::floor((width - t[2]) / 2);
        t[1] = std::floor((height - t[3]) / 2);
    }
    // Vulkan needs a viewport with an area.
    return VkViewport{t[0], t[1], std::max(t[2], 1.0f), std::max(t[3], 1.0f), 0, 1};
}

// Records the drawing of the frame in `slot` into `framebuffer` (of `extent`):
// cleared to `clear`, then the frame drawn in `viewport`. The slot's textures
// are taken over from Direct3D 11 for it, and given back after.
static void record_drawing(VulkanDevice& vk, VkCommandBuffer commands, const HandoverSlot& slot,
                           const Drawing& drawing, VkFramebuffer framebuffer, VkExtent2D extent,
                           const DrawConstants& constants, const VkViewport& viewport, const float* clear) {
    VkImageMemoryBarrier planes[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER},
                                      {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
    const VkImage images[2] = {slot.luma.image, slot.chroma.image};
    for (int i = 0; i < 2; i++) {
        planes[i].oldLayout = planes[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        planes[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        planes[i].dstQueueFamilyIndex = vk.queue_family;
        planes[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        planes[i].image = images[i];
        planes[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                            nullptr, 0, nullptr, 2, planes);
    VkClearValue clear_value = {};
    for (int i = 0; i < 4; i++)
        clear_value.color.float32[i] = clear[i];
    VkRenderPassBeginInfo pass = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = drawing.render_pass;
    pass.framebuffer = framebuffer;
    pass.renderArea.extent = extent;
    pass.clearValueCount = 1;
    pass.pClearValues = &clear_value;
    vk.vkCmdBeginRenderPass(commands, &pass, VK_SUBPASS_CONTENTS_INLINE);
    vk.vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, drawing.pipeline);
    vk.vkCmdSetViewport(commands, 0, 1, &viewport);
    const VkRect2D whole = {{0, 0}, extent};
    vk.vkCmdSetScissor(commands, 0, 1, &whole);
    vk.vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline_layout, 0, 1, &slot.descriptors,
                               0, nullptr);
    vk.vkCmdPushConstants(commands, vk.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                          sizeof constants, &constants);
    vk.vkCmdDraw(commands, 4, 1, 0, 0);
    vk.vkCmdEndRenderPass(commands);
    for (int i = 0; i < 2; i++) {
        planes[i].srcQueueFamilyIndex = vk.queue_family;
        planes[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        planes[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        planes[i].dstAccessMask = 0;
    }
    vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                            0, nullptr, 0, nullptr, 2, planes);
}

// Vulkan's part of drawing a frame: the frame handed over drawn into the
// image acquired, then presented. The image's drawing of the slot is recorded
// once and submitted again while nothing about it changes: recording it each
// frame cost some 20 us.
static HRESULT draw_and_present(Presenter* p, const PendingDraw& draw) {
    VulkanDevice& vk = *p->vulkan;
    FrameInFlight& frame = next_frame(p);
    const uint32_t image = uint32_t(p->acquired);
    const uint64_t number = p->frames_handed_over;
    HandoverSlot& slot = p->slots[draw.slot];
    const VkViewport viewport = placed(draw.source, draw.fitted ? nullptr : draw.target, p->swapchain.extent);
    RecordedDrawing& drawing = p->recorded[image * kHandoverSlots + draw.slot];
    VkResult result = VK_SUCCESS;
    if (!drawing.commands) {
        VkCommandBufferAllocateInfo allocate = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = p->command_pool;
        allocate.commandBufferCount = 1;
        result = vk.vkAllocateCommandBuffers(vk.handle, &allocate, &drawing.commands);
        if (result != VK_SUCCESS)
            return vulkan_failure(vk, result, "making a command buffer");
    }
    if (!drawing.recorded || drawing.slots_made != p->slots_made ||
        memcmp(&drawing.constants, &draw.constants, sizeof draw.constants) != 0 ||
        memcmp(&drawing.viewport, &viewport, sizeof viewport) != 0 ||
        memcmp(drawing.clear, draw.clear, sizeof draw.clear) != 0) {
        // Its last submission is done: the image it drew into was given back by the display.
        vk.vkResetCommandBuffer(drawing.commands, 0);
        VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vk.vkBeginCommandBuffer(drawing.commands, &begin);
        record_drawing(vk, drawing.commands, slot, p->swapchain.drawing, p->swapchain.framebuffers[image],
                       p->swapchain.extent, draw.constants, viewport, draw.clear);
        result = vk.vkEndCommandBuffer(drawing.commands);
        if (result != VK_SUCCESS) {
            drawing.recorded = false;
            return vulkan_failure(vk, result, "recording a frame's drawing");
        }
        drawing.recorded = true;
        drawing.slots_made = p->slots_made;
        drawing.constants = draw.constants;
        drawing.viewport = viewport;
        std::copy(draw.clear, draw.clear + 4, drawing.clear);
    }

    // It waits for the image (at the colour output) and for Direct3D 11's copy
    // (at the fragment shader), and signals the image ready to present and
    // the frame drawn.
    const VkSemaphore waits[2] = {frame.acquired, p->copied};
    const VkPipelineStageFlags stages[2] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
    const uint64_t wait_values[2] = {0, number};
    const VkSemaphore signals[2] = {p->swapchain.ready_to_present[image], p->drawn};
    const uint64_t signal_values[2] = {0, number};
    VkTimelineSemaphoreSubmitInfo values = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    values.waitSemaphoreValueCount = 2;
    values.pWaitSemaphoreValues = wait_values;
    values.signalSemaphoreValueCount = 2;
    values.pSignalSemaphoreValues = signal_values;
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &values};
    submit.waitSemaphoreCount = 2;
    submit.pWaitSemaphores = waits;
    submit.pWaitDstStageMask = stages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &drawing.commands;
    submit.signalSemaphoreCount = 2;
    submit.pSignalSemaphores = signals;
    VkPresentInfoKHR present = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &p->swapchain.ready_to_present[image];
    present.swapchainCount = 1;
    present.pSwapchains = &p->swapchain.handle;
    present.pImageIndices = &image;
    vk.vkResetFences(vk.handle, 1, &frame.drawn);
    {
        std::lock_guard<std::mutex> hold(vk.queue_lock);
        result = vk.vkQueueSubmit(vk.queue, 1, &submit, frame.drawn);
        if (result != VK_SUCCESS)
            return vulkan_failure(vk, result, "submitting a frame's drawing");
        result = vk.vkQueuePresentKHR(vk.queue, &present);
    }
    slot.last_frame = number;
    p->last_frame_submitted = number;
    p->frames_drawn++;
    p->acquired = -1;
    p->image_free = false;
    if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
        p->swapchain_stale = true;  // presented all the same; made again before the next frame
        return S_OK;
    }
    return vulkan_failure(vk, result, "presenting a frame");
}

// Frees what the presenter made, except its window, which goes with the view.
// Whatever the GPU might still be using is left, if the GPU does not finish.
static void destroy_presenter(Presenter* p) {
    if (p->vulkan) {
        VulkanDevice& vk = *p->vulkan;
        if (p->swapchain.handle)
            give_back_acquired(p);
        if (wait_idle(p)) {
            free_swapchain(p);
            if (p->surface)
                p->instance->vkDestroySurfaceKHR(p->instance->handle, p->surface, nullptr);
            for (HandoverSlot& slot : p->slots)
                free_slot(p, slot);
            vk.vkDestroyDescriptorPool(vk.handle, p->descriptor_pool, nullptr);
            vk.vkDestroyCommandPool(vk.handle, p->command_pool, nullptr);
            for (FrameInFlight& frame : p->frames) {
                vk.vkDestroyFence(vk.handle, frame.drawn, nullptr);
                vk.vkDestroySemaphore(vk.handle, frame.acquired, nullptr);
            }
            vk.vkDestroyFence(vk.handle, p->acquired_free, nullptr);
            vk.vkDestroySemaphore(vk.handle, p->copied, nullptr);
            vk.vkDestroySemaphore(vk.handle, p->drawn, nullptr);
        }
    }
    delete p;
}

// The presenter's Direct3D 11 side: its contexts, the copy shader, and the
// fence it shares with Vulkan (returned in `fence_handle`).
static HRESULT prepare_direct3d(Presenter* p, HANDLE* fence_handle) {
    p->d3d->GetImmediateContext(&p->immediate);
    HRESULT hr = p->immediate.As(&p->immediate4);
    ComPtr<ID3D11Device5> d3d5;
    if (SUCCEEDED(hr))
        hr = p->d3d.As(&d3d5);
    if (FAILED(hr)) {
        set_problem("Direct3D 11 fences are not available (Windows 10 1703 or newer is needed)");
        return hr;
    }
    hr = p->d3d->CreateDeferredContext(0, &p->deferred);
    ID3DBlob* code = nullptr;
    if (SUCCEEDED(hr))
        hr = copy_shader(&code);
    if (SUCCEEDED(hr))
        hr = p->d3d->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &p->copy_planes);
    // The copy's region and thread groups: none, as copy_region_now says,
    // until the first frame's are set.
    const UINT no_groups[3] = {0, 0, 1};
    D3D11_BUFFER_DESC buffer = {};
    buffer.ByteWidth = sizeof p->copy_region_now;
    buffer.Usage = D3D11_USAGE_DEFAULT;
    buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial = {p->copy_region_now, 0, 0};
    if (SUCCEEDED(hr))
        hr = p->d3d->CreateBuffer(&buffer, &initial, &p->copy_region);
    buffer.ByteWidth = sizeof no_groups;
    buffer.BindFlags = 0;
    buffer.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
    initial.pSysMem = no_groups;
    if (SUCCEEDED(hr))
        hr = p->d3d->CreateBuffer(&buffer, &initial, &p->copy_groups);
    if (FAILED(hr))
        return hr;
    for (DXGI_FORMAT format :
         {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM})
        if (!can_write(p->d3d.Get(), format)) {
            set_problem("Direct3D 11 cannot write the shared textures (format %d)", int(format));
            return E_FAIL;
        }
    hr = d3d5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&p->copied_fence));
    if (SUCCEEDED(hr))
        hr = p->copied_fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, fence_handle);
    if (FAILED(hr))
        set_problem("a Direct3D 11 fence could not be shared (0x%08x)", unsigned(hr));
    return hr;
}

// The presenter's Vulkan side: its semaphores, fences, command buffers and
// descriptor sets.
static HRESULT prepare_vulkan(Presenter* p, HANDLE fence_handle) {
    VulkanDevice& vk = *p->vulkan;
    VkSemaphoreTypeCreateInfo timeline = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo timeline_semaphore = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline};
    VkResult result = vk.vkCreateSemaphore(vk.handle, &timeline_semaphore, nullptr, &p->copied);
    if (result == VK_SUCCESS) {
        VkImportSemaphoreWin32HandleInfoKHR import = {VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
        import.semaphore = p->copied;
        import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE_BIT;
        import.handle = fence_handle;
        result = vk.vkImportSemaphoreWin32HandleKHR(vk.handle, &import);
    }
    if (result == VK_SUCCESS)
        result = vk.vkCreateSemaphore(vk.handle, &timeline_semaphore, nullptr, &p->drawn);
    VkCommandPoolCreateInfo pool = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = vk.queue_family;
    if (result == VK_SUCCESS)
        result = vk.vkCreateCommandPool(vk.handle, &pool, nullptr, &p->command_pool);
    VkSemaphoreCreateInfo binary_semaphore = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo signalled = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    signalled.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (FrameInFlight& frame : p->frames) {
        if (result == VK_SUCCESS)
            result = vk.vkCreateFence(vk.handle, &signalled, nullptr, &frame.drawn);
        if (result == VK_SUCCESS)
            result = vk.vkCreateSemaphore(vk.handle, &binary_semaphore, nullptr, &frame.acquired);
    }
    VkFenceCreateInfo unsignalled = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (result == VK_SUCCESS)
        result = vk.vkCreateFence(vk.handle, &unsignalled, nullptr, &p->acquired_free);
    VkDescriptorPoolSize planes = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * kHandoverSlots};
    VkDescriptorPoolCreateInfo descriptor_pool = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_pool.maxSets = kHandoverSlots;
    descriptor_pool.poolSizeCount = 1;
    descriptor_pool.pPoolSizes = &planes;
    if (result == VK_SUCCESS)
        result = vk.vkCreateDescriptorPool(vk.handle, &descriptor_pool, nullptr, &p->descriptor_pool);
    for (HandoverSlot& slot : p->slots) {
        VkDescriptorSetAllocateInfo set = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set.descriptorPool = p->descriptor_pool;
        set.descriptorSetCount = 1;
        set.pSetLayouts = &vk.set_layout;
        if (result == VK_SUCCESS)
            result = vk.vkAllocateDescriptorSets(vk.handle, &set, &slot.descriptors);
    }
    return vulkan_failure(vk, result, "setting up the presenter");
}

// A presenter on GStreamer's Direct3D 11 `device`, without its window yet.
// Null, the problem noted, if Vulkan cannot present its frames.
static Presenter* make_presenter(ID3D11Device* device, bool hdr_display) {
    ComPtr<IDXGIDevice> dxgi;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC adapter_desc;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgi))) || FAILED(dxgi->GetAdapter(&adapter)) ||
        FAILED(adapter->GetDesc(&adapter_desc))) {
        set_problem("the GPU of GStreamer's Direct3D 11 device is unknown");
        return nullptr;
    }
    VulkanInstance* instance = vulkan_instance(adapter_desc.VendorId);
    if (!instance)
        return nullptr;
    std::shared_ptr<VulkanDevice> vulkan = device_for(instance, adapter_desc.AdapterLuid);
    if (!vulkan)
        return nullptr;
    auto p = new (std::nothrow) Presenter;
    if (!p)
        return nullptr;
    p->instance = instance;
    p->vulkan = vulkan;
    p->d3d = device;
    p->hdr_display = hdr_display;
    HANDLE fence_handle = nullptr;
    HRESULT hr = prepare_direct3d(p, &fence_handle);
    if (SUCCEEDED(hr))
        hr = prepare_vulkan(p, fence_handle);
    // Importing does not take the handle over.
    if (fence_handle)
        CloseHandle(fence_handle);
    if (FAILED(hr)) {
        destroy_presenter(p);
        return nullptr;
    }
    return p;
}

static LRESULT CALLBACK window_procedure(HWND window, UINT message, WPARAM w, LPARAM l) {
    if (message == WM_ERASEBKGND)
        return 1;  // the frames cover it
    if (message == WM_PAINT) {
        ValidateRect(window, nullptr);
        return 0;
    }
    return DefWindowProcW(window, message, w, l);
}

// The presenter's window, a child covering `parent`, and its swapchain. On
// the thread `parent` belongs to.
static HRESULT attach_window(Presenter* p, HWND parent) {
    static ATOM window_class = 0;
    HINSTANCE module = GetModuleHandleW(nullptr);
    if (!window_class) {
        WNDCLASSEXW description = {sizeof description};
        description.lpfnWndProc = window_procedure;
        description.hInstance = module;
        description.lpszClassName = L"VideoMetricsLabVulkanVideo";
        window_class = RegisterClassExW(&description);
        if (!window_class) {
            set_problem("the video window's class could not be registered");
            return E_FAIL;
        }
    }
    RECT client = {};
    GetClientRect(parent, &client);
    // Disabled: the mouse is the view's (Windows' hit testing skips it).
    p->window = CreateWindowExW(0, L"VideoMetricsLabVulkanVideo", L"",
                                WS_CHILD | WS_VISIBLE | WS_DISABLED | WS_CLIPSIBLINGS, 0, 0,
                                std::max<LONG>(client.right, 1), std::max<LONG>(client.bottom, 1), parent, nullptr,
                                module, nullptr);
    if (!p->window) {
        set_problem("the video window could not be made");
        return E_FAIL;
    }
    VkWin32SurfaceCreateInfoKHR surface = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surface.hinstance = module;
    surface.hwnd = p->window;
    VkResult result = p->instance->vkCreateWin32SurfaceKHR(p->instance->handle, &surface, nullptr, &p->surface);
    if (result != VK_SUCCESS)
        return vulkan_failure(*p->vulkan, result, "making the video window's surface");
    VkBool32 supported = VK_FALSE;
    p->instance->vkGetPhysicalDeviceSurfaceSupportKHR(p->vulkan->physical, p->vulkan->queue_family, p->surface,
                                                      &supported);
    uint32_t formats = 0;
    p->instance->vkGetPhysicalDeviceSurfaceFormatsKHR(p->vulkan->physical, p->surface, &formats, nullptr);
    if (!supported || formats == 0) {
        set_problem("Vulkan cannot present to the video window");
        return E_FAIL;
    }
    // The swapchain itself is made on the presenter's thread (vmaf_present_start,
    // or the first wait), not on the window's: on an RTX 5090 the first one
    // takes some 85 ms, NVIDIA making a Direct3D 12 device for it.
    return S_OK;
}

// ---------------------------------------------------------------------------
// The calls, as d3d11_tonemap.dll's

// Any thread, any time: the Vulkan device on the first GPU (the one GStreamer
// decodes on: gstreamer_playback.d3d11_device), its pipelines for the usual
// window formats, and the copy shader, made ahead of the first presenter. 0,
// or an error (vmaf_present_problem says why).
extern "C" __declspec(dllexport) int vmaf_present_prepare() {
    ID3DBlob* code = nullptr;
    HRESULT hr = copy_shader(&code);
    if (FAILED(hr))
        return hr;
    ComPtr<IDXGIFactory1> factory;
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 desc;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumAdapters1(0, &adapter)) ||
        FAILED(adapter->GetDesc1(&desc))) {
        set_problem("no GPU was found");
        return E_FAIL;
    }
    VulkanInstance* instance = vulkan_instance(desc.VendorId);
    if (!instance)
        return E_FAIL;
    std::shared_ptr<VulkanDevice> device = device_for(instance, desc.AdapterLuid);
    if (!device)
        return E_FAIL;
    Drawing drawing;
    if (!drawing_for(*device, VK_FORMAT_B8G8R8A8_UNORM, &drawing) ||
        !drawing_for(*device, VK_FORMAT_A2B10G10R10_UNORM_PACK32, &drawing)) {
        set_problem("the Vulkan presenter's pipelines could not be made");
        return E_FAIL;
    }
    return S_OK;
}

// Why the last call that failed did: a sentence for the log.
extern "C" __declspec(dllexport) const char* vmaf_present_problem() {
    static thread_local char copy[sizeof problem_text];
    std::lock_guard<std::mutex> hold(problem_lock);
    memcpy(copy, problem_text, sizeof copy);
    return copy;
}

// On the thread the window `parent` belongs to, with the device lock held.
// `hdr_display`: 10-bit images, for PQ BT.2020 frames as well as SDR ones.
// Null if Vulkan cannot present the frames (vmaf_present_problem says why).
extern "C" __declspec(dllexport) void* vmaf_present_create(ID3D11Device* device, HWND parent, int hdr_display) {
    if (!device || !parent) {
        set_problem("no device or window was given");
        return nullptr;
    }
    Presenter* p = make_presenter(device, hdr_display != 0);
    if (!p)
        return nullptr;
    if (FAILED(attach_window(p, parent))) {
        HWND window = p->window;
        destroy_presenter(p);
        if (window)
            DestroyWindow(window);  // made on this thread, which owns it
        return nullptr;
    }
    return p;
}

// The window follows the view's size (on its thread); the swapchain follows
// the window before the next frame.
extern "C" __declspec(dllexport) void vmaf_present_follow(void* opaque, HWND parent) {
    auto p = static_cast<Presenter*>(opaque);
    if (!p || !p->window)
        return;
    RECT client = {}, own = {};
    GetClientRect(parent, &client);
    GetClientRect(p->window, &own);
    if (client.right != own.right || client.bottom != own.bottom)
        SetWindowPos(p->window, nullptr, 0, 0, client.right, client.bottom, SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOMOVE);
}

// On the presenter's thread, before its first frame: the swapchain made while
// the decoders start, rather than when the first frame is due. 0, or an error
// (vmaf_present_problem says why). Holding no lock.
extern "C" __declspec(dllexport) int vmaf_present_start(void* opaque) {
    auto p = static_cast<Presenter*>(opaque);
    if (!p)
        return E_INVALIDARG;
    if (p->swapchain.handle)
        return S_OK;
    const HRESULT hr = make_swapchain(p, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
    return FAILED(hr) ? hr : S_OK;  // without an area yet, it is made at a later wait
}

// Until the swapchain takes a frame without waiting: an image acquired, and
// free of the display. 0, 1 at `timeout_ms`, or -1 if Vulkan failed
// (vmaf_present_problem says why). Holding no lock.
extern "C" __declspec(dllexport) int vmaf_present_wait(void* opaque, unsigned timeout_ms) {
    auto p = static_cast<Presenter*>(opaque);
    if (!p || p->vulkan->lost)
        return -1;
    VulkanDevice& vk = *p->vulkan;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto remaining_ns = [&]() -> uint64_t {
        const auto left = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - std::chrono::steady_clock::now());
        return left.count() > 0 ? uint64_t(left.count()) : 0;
    };
    // A window with no area to draw in (minimized) takes no frames: asked again shortly.
    auto not_now = [&] {
        Sleep(std::min(timeout_ms, 20u));
        return 1;
    };
    for (int attempt = 0; p->acquired < 0; attempt++) {
        if (attempt == 2)
            return not_now();
        RECT client = {};
        GetClientRect(p->window, &client);
        if (client.right <= 0 || client.bottom <= 0)
            return not_now();
        if (!p->swapchain.handle || p->swapchain_stale || uint32_t(client.right) != p->swapchain.extent.width ||
            uint32_t(client.bottom) != p->swapchain.extent.height) {
            const HRESULT hr = make_swapchain(p, p->swapchain.colour_space);
            if (hr == S_FALSE)
                return not_now();
            if (FAILED(hr))
                return -1;
        }
        // The frame drawn three frames ago, which waited on this acquire semaphore, is done.
        FrameInFlight& frame = next_frame(p);
        VkResult result = vk.vkWaitForFences(vk.handle, 1, &frame.drawn, VK_TRUE, remaining_ns());
        if (result == VK_TIMEOUT)
            return 1;
        if (result != VK_SUCCESS) {
            vulkan_failure(vk, result, "waiting for a frame's drawing");
            return -1;
        }
        vk.vkResetFences(vk.handle, 1, &p->acquired_free);
        uint32_t image = 0;
        result = vk.vkAcquireNextImageKHR(vk.handle, p->swapchain.handle, remaining_ns(), frame.acquired,
                                          p->acquired_free, &image);
        if (result == VK_TIMEOUT || result == VK_NOT_READY)
            return 1;
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            p->swapchain_stale = true;  // made again, then tried once more
            continue;
        }
        if (result == VK_SUBOPTIMAL_KHR)
            p->swapchain_stale = true;  // drawn into all the same, then made again
        else if (result != VK_SUCCESS) {
            vulkan_failure(vk, result, "acquiring a window image");
            return -1;
        }
        p->acquired = int(image);
    }
    if (!p->image_free) {
        // Acquired images may still be on screen until the display lets go of them.
        const VkResult result = vk.vkWaitForFences(vk.handle, 1, &p->acquired_free, VK_TRUE, remaining_ns());
        if (result == VK_TIMEOUT)
            return 1;
        if (result != VK_SUCCESS) {
            vulkan_failure(vk, result, "waiting for a window image");
            return -1;
        }
        p->image_free = true;
    }
    return 0;
}

// After vmaf_present_texture, holding no lock: the frame it handed over drawn
// into the image acquired, and presented. Vulkan's part of a frame needs no
// Direct3D 11, so the decoders, which need the device lock for each frame, do
// not wait for it, nor for the swapchain to be made again in another colour
// space, which takes some 4-5 ms. Returns 0, or an error (vmaf_present_problem
// says why).
extern "C" __declspec(dllexport) int vmaf_present_flip(void* opaque) {
    auto p = static_cast<Presenter*>(opaque);
    if (!p)
        return E_INVALIDARG;
    if (!p->pending.waiting)
        return S_OK;
    const PendingDraw draw = p->pending;
    p->pending.waiting = false;
    if (draw.colour_space != p->swapchain.colour_space) {
        // Between frames: the image acquired goes back with the old swapchain.
        HRESULT hr = give_back_acquired(p);
        if (SUCCEEDED(hr))
            hr = make_swapchain(p, draw.colour_space);
        if (hr == S_FALSE)
            return S_OK;  // no area to draw in for now: this frame is not shown
        if (FAILED(hr))
            return hr;
        const int waited = vmaf_present_wait(p, 1000);
        if (waited != 0)
            return waited > 0 ? S_OK : E_FAIL;
    }
    return draw_and_present(p, draw);
}

// With the device lock held: the frame handed over to Vulkan, which draws and
// presents it at vmaf_present_flip. `frame`: the texture, `subresource` the
// frame's in it. `source`: the frame's part shown (x, y, width, height in its
// pixels), or null for all of it; `target`: where in the window (device
// pixels), or null to fit it, letterboxed. `to_rgb` and `shading`: as
// draw_constants takes them. `space`: the colour space asked for (DXGI's
// number: 0 sRGB, 12 HDR10); `clear`: the colour beside the frame. A frame of
// a minimized window is not shown.
extern "C" __declspec(dllexport) int vmaf_present_texture(void* opaque, ID3D11Resource* frame, unsigned subresource,
                                                          const int* source, const int* target, const float* to_rgb,
                                                          const float* shading, int space, const float* clear) {
    auto p = static_cast<Presenter*>(opaque);
    if (!p || !frame || !to_rgb || !clear)
        return E_INVALIDARG;
    if (p->vulkan->lost)
        return DXGI_ERROR_DEVICE_REMOVED;
    // A frame handed over and never flipped is presented first.
    if (p->pending.waiting) {
        const HRESULT hr = vmaf_present_flip(p);
        if (FAILED(hr))
            return hr;
    }
    // An image to draw into: vmaf_present_wait has one ready, unless not asked first.
    if (!p->image_free) {
        const int waited = vmaf_present_wait(p, 100);
        if (waited != 0)
            return waited > 0 ? S_OK : E_FAIL;
    }
    D3D11_TEXTURE2D_DESC desc;
    int slot = 0;
    const HRESULT hr = hand_over(p, frame, subresource, source, desc, &slot);
    if (FAILED(hr))
        return hr;
    PendingDraw& draw = p->pending;
    draw.waiting = true;
    draw.slot = slot;
    draw.source[0] = draw.source[1] = 0;
    draw.source[2] = float(desc.Width);
    draw.source[3] = float(desc.Height);
    if (source)
        for (int i = 0; i < 4; i++)
            draw.source[i] = float(source[i]);
    draw.constants = draw_constants(desc.Width, desc.Height, draw.source, to_rgb, shading);
    draw.fitted = target == nullptr;
    if (target)
        std::copy(target, target + 4, draw.target);
    std::copy(clear, clear + 4, draw.clear);
    // HDR10 where asked for and the display takes it, else sRGB.
    draw.colour_space = space == kHdr10Space && p->hdr10_offered ? VK_COLOR_SPACE_HDR10_ST2084_EXT
                                                                 : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    return S_OK;
}

// With the device lock held, on any thread: the window goes with the view.
extern "C" __declspec(dllexport) void vmaf_present_destroy(void* opaque) {
    auto p = static_cast<Presenter*>(opaque);
    if (p)
        destroy_presenter(p);
}
