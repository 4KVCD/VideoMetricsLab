// The GPU the app scores on, where a PC has more than one: Settings' GPU, or
// the command line's --gpu, by its name. Left automatic, each part picks as it
// always has (a discrete GPU first). Chosen, VMAF and VMAF v1 (Vulkan), and
// SSIMULACRA2, Butteraugli and CVVDP (Vship) run on that GPU, and automatic
// decoding tries its maker's decoder first.
//
// The choice travels in an environment variable, VML_GPU, which the app and
// the command line set as they start (resolve_chosen_gpu): the processes they
// start for their GPU work (isolated.h) inherit it, and a GPU check's saved
// answer is kept for each choice (gpu_checks.h).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/platform/system.h"

namespace vml {

inline constexpr const char* gpu_variable = "VML_GPU";

// The GPU chosen in this process (VML_GPU), or nothing for automatic.
std::optional<std::string> chosen_gpu();

// Sets VML_GPU for this process and those it starts: `name`, or automatic for
// nothing or an empty name.
void choose_gpu(const std::optional<std::string>& name);

// The text that chooses each of `names` (GPUs as a driver or DirectX names
// them, in its order), as Settings saves it: its name, or, where more than one
// GPU has that name (two of one card), the name and its place among them,
// from 1: "NVIDIA GeForce RTX 3070 #2".
std::vector<std::string> gpu_choice_texts(const std::vector<std::string>& names);

// Which of `names` (GPUs as a driver or DirectX names them) `chosen` can
// name, ignoring case: the first that is `chosen` itself; else, for a name
// and a place ("... #2", gpu_choice_texts), that one of the GPUs of that
// name, or none where there are fewer; else every one that has it in its name
// ("radeon" for "AMD Radeon(TM) Graphics"). None for an empty `chosen`.
std::vector<std::size_t> gpus_named(const std::vector<std::string>& names, std::string_view chosen);

// The one of `names` that `chosen` names (gpus_named), or nothing: also when a
// part of a name is in more than one, as each API lists the GPUs in an order
// of its own (DirectX's, Vulkan's, Vship's discrete-first), so that the first
// of each would not be the same GPU.
std::optional<std::size_t> chosen_among(const std::vector<std::string>& names, std::string_view chosen);

// Makes the GPU chosen in this process (VML_GPU) the choice text
// (gpu_choice_texts) of the one DirectX lists that it names (chosen_among),
// once, as the app and the
// command line start: a part of a name ("radeon") is then matched in every
// API's list as that GPU's name. Left as it is where it names no GPU
// DirectX lists, or more than one (each API then finds none: the CPU), and
// where the system lists none (a Mac).
void resolve_chosen_gpu();

// CUDA's number for the GPU `chosen` names (gpu_choice_texts' text), for
// NVIDIA's decoder and libvmaf's CUDA code, which otherwise take CUDA's
// first GPU. 0 where nothing is chosen; where the GPU chosen is not one of
// the NVIDIA GPUs of `adapters` (DirectX's list: NVIDIA's decoder is then
// only tried after the chosen GPU's maker's, as before, and CUDA VMAF is not
// used, gpu_vmaf.cpp); and where it is the only NVIDIA GPU there. Else the
// one of `cuda_gpus()` (CUDA's GPUs in its order, asked only then, as it
// starts CUDA) with the chosen adapter's LUID: the very card DirectX lists,
// two cards of one name included, whose order CUDA (fastest first) and
// DirectX (the main display's first) need not share. Where the LUIDs are
// not known, the one `chosen` names (chosen_among: "... #2" is the second
// of that name in CUDA's order). Nothing where none is (CUDA_VISIBLE_DEVICES
// hides it): nothing of NVIDIA's then runs on another GPU instead. Where the
// system lists no GPUs (macOS), the one of `cuda_gpus()` it names, else 0.
// One of CUDA's GPUs: its name, and its LUID as GpuAdapter's (0 where CUDA
// does not say it: other systems).
struct CudaGpu {
    std::string name;
    std::uint64_t luid = 0;
};
std::optional<int> chosen_cuda_device_among(const std::optional<std::string>& chosen,
                                            const std::vector<platform::GpuAdapter>& adapters,
                                            const std::function<std::vector<CudaGpu>()>& cuda_gpus);

}  // namespace vml
