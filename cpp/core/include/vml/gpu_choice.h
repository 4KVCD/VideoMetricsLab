// The GPU the app scores on, where a PC has more than one: Settings' GPU, or
// the command line's --gpu, by its name. Left automatic, each part picks as it
// always has (a discrete GPU first). Chosen, VMAF and VMAF v1 (Vulkan), and
// SSIMULACRA2, Butteraugli and CVVDP (Vship) run on that GPU, and automatic
// decoding tries its maker's decoder first.
//
// The choice travels in an environment variable, VML_GPU, which the app and
// the command line set as they start: the processes they start for their GPU
// work (isolated.h) inherit it, and a GPU check's saved answer is kept for
// each choice (gpu_checks.h).
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vml {

inline constexpr const char* gpu_variable = "VML_GPU";

// The GPU chosen in this process (VML_GPU), or nothing for automatic.
std::optional<std::string> chosen_gpu();

// Sets VML_GPU for this process and those it starts: `name`, or automatic for
// nothing or an empty name.
void choose_gpu(const std::optional<std::string>& name);

// Which of `names` (GPUs as a driver or DirectX names them) `chosen` names:
// the first that is `chosen` itself, ignoring case, else the first that has it
// in its name ("radeon" for "AMD Radeon(TM) Graphics"). Nothing when none does.
std::optional<std::size_t> chosen_among(const std::vector<std::string>& names, std::string_view chosen);

}  // namespace vml
