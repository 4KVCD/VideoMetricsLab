#include "vml/memory_fit.h"

#include <algorithm>
#include <array>
#include <new>
#include <string_view>

#include "vml/app_log.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// The metrics that keep memory on the GPU, as the first key of their group.
constexpr std::array<std::string_view, 8> gpu_metrics{"vmaf",    "vmaf_neg",    "vmaf_v1", "vmaf_hdr",
                                                      "ms_ssim", "ms_ssim_yuv", "cambi",   "ciede2000"};

// The group `key` belongs to (memory_groups), by its metric's key: a plane's
// metric's, VMAF's for VMAF YUV's planes (made with VMAF's U and V from VMAF,
// JobRun::merged), and PSNR YUV's for PSNR YUV MSE (from the planes PSNR YUV
// reads, in its pass).
std::string group_name(const std::string& key) {
    const MetricDefinition& metric = metric_definition(key);
    const std::string_view name = metric.is_plane() ? metric.plane_of : metric.key;
    return name == "vmaf_yuv" ? "vmaf" : name == "psnr_yuv_mse" ? "psnr_yuv" : std::string(name);
}

const app_log::Logger logger("vml.memory_fit");

constexpr std::uint64_t megabyte = 1ull << 20;

std::string megabytes(std::uint64_t bytes) { return std::to_string(bytes / megabyte) + " MB"; }

}  // namespace

std::vector<std::vector<std::string>> memory_groups(const std::vector<std::string>& keys) {
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> groups;
    for (const std::string& key : keys) {
        const std::string name = group_name(key);
        const auto found = std::find(names.begin(), names.end(), name);
        if (found == names.end()) {
            names.push_back(name);
            groups.push_back({key});
        } else {
            groups[static_cast<std::size_t>(found - names.begin())].push_back(key);
        }
    }
    return groups;
}

bool group_uses_gpu_memory(const std::vector<std::string>& group) {
    return !group.empty() &&
           std::find(gpu_metrics.begin(), gpu_metrics.end(), group_name(group.front())) != gpu_metrics.end();
}

std::size_t memory_split_groups(const std::vector<std::string>& keys, bool gpu) {
    const std::vector<std::vector<std::string>> groups = memory_groups(keys);
    return gpu ? static_cast<std::size_t>(std::count_if(groups.begin(), groups.end(), group_uses_gpu_memory))
               : groups.size();
}

std::optional<std::pair<std::vector<std::string>, std::vector<std::string>>> halve_metrics(
    const std::vector<std::string>& keys, bool gpu) {
    const std::size_t splittable = memory_split_groups(keys, gpu);
    if (splittable < 2) {
        return std::nullopt;
    }
    const std::size_t first_count = (splittable + 1) / 2;
    std::pair<std::vector<std::string>, std::vector<std::string>> halves;
    std::size_t placed = 0;  // splittable groups placed so far
    for (const std::vector<std::string>& group : memory_groups(keys)) {
        const bool splits = !gpu || group_uses_gpu_memory(group);
        std::vector<std::string>& half = !splits || placed < first_count ? halves.first : halves.second;
        half.insert(half.end(), group.begin(), group.end());
        placed += splits ? 1 : 0;
    }
    return halves;
}

MemoryWatch::MemoryWatch(const std::vector<std::string>& keys, GpuReader gpu_memory, HeadroomReader headroom)
    : watch_gpu_(gpu_memory && memory_split_groups(keys, true) >= 2),
      watch_ram_(memory_split_groups(keys, false) >= 2),
      gpu_memory_(std::move(gpu_memory)),
      headroom_(std::move(headroom)) {}

bool MemoryWatch::check(long long done) {
    const std::lock_guard lock(mutex_);
    if (!active() || shortage_ || checked_ >= checks) {
        return false;
    }
    if (!first_done_) {
        first_done_ = done;  // the pass's first report: its first pair may not be scored yet
        return false;
    }
    if (done <= *first_done_) {
        return false;
    }
    ++checked_;
    shortage_ = look();
    return shortage_.has_value();
}

std::optional<MemoryShortage> MemoryWatch::shortage() const {
    const std::lock_guard lock(mutex_);
    return shortage_;
}

std::optional<MemoryShortage> MemoryWatch::look() {
    if (watch_gpu_) {
        if (std::optional<GpuMemory> memory = gpu_memory_()) {
            if (const std::optional<std::uint64_t> cap = memory_test_cap("VML_GPU_MEMORY_BUDGET_MB")) {
                memory->budget = std::min(memory->budget, *cap);
            }
            if (checked_ == 1) {
                logger.info("The pass's GPU memory: " + megabytes(memory->used) + " used of " +
                            megabytes(memory->budget) + ", " + megabytes(memory->spilled) + " in the PC's memory");
            }
            if (memory->spilled > 0 || memory->used > memory->budget) {
                return MemoryShortage{true, "the GPU's memory is full (" + megabytes(memory->used) + " used of " +
                                                megabytes(memory->budget) +
                                                (memory->spilled > 0 ? ", " + megabytes(memory->spilled) +
                                                                           " in the PC's memory instead"
                                                                     : std::string()) +
                                                ")"};
            }
        }
    }
    if (watch_ram_ && headroom_) {
        std::optional<unsigned long long> left = headroom_();
        if (const std::optional<std::uint64_t> cap = memory_test_cap("VML_MEMORY_HEADROOM_MB"); left && cap) {
            left = std::min<unsigned long long>(*left, *cap);
        }
        if (left && *left < headroom_floor) {
            return MemoryShortage{false, "the PC's memory is nearly full (" + megabytes(*left) + " left)"};
        }
    }
    return std::nullopt;
}

std::optional<bool> memory_refused(const std::exception& error) {
    const std::string_view text = error.what();
    if (text.find("out of GPU memory") != std::string_view::npos) {
        return true;
    }
    if (dynamic_cast<const std::bad_alloc*>(&error) != nullptr ||
        text.find("(libvmaf error -12)") != std::string_view::npos ||
        text.find("bad allocation") != std::string_view::npos ||
        text.find("std::bad_alloc") != std::string_view::npos) {
        return false;
    }
    return std::nullopt;
}

std::optional<std::uint64_t> memory_test_cap(const char* variable) {
    const std::string text(python_strip(platform::environment_variable(variable).value_or("")));
    if (text.empty()) {
        return std::nullopt;
    }
    try {
        return static_cast<std::uint64_t>(std::stoull(text)) * megabyte;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

}  // namespace vml
