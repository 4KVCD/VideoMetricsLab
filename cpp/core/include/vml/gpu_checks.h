// The GPU checks' answers, kept between starts: each check (whether VMAF,
// VMAF v1 and Vship's metrics can run on this PC's GPU, and on which) runs in
// a process of its own and takes 0.5 to 2 s, so an answer that the GPU works
// is saved and used at later starts, until the app, its GPU libraries, a GPU,
// a driver or the environment that steers them changes. Only passes are
// saved: a GPU that failed a check is checked again at every start.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include "vml/cancel.h"

namespace vml {

// gpu_checks.json, beside the settings file.
std::filesystem::path gpu_checks_file();

// Runs GPU check `task` ("vulkan-probe", "cuda-probe", "vmaf-v1-probe",
// "vship-probe" or "nvdec-probe") with `request` in a process of its own, as run_isolated
// does, unless an answer saved at an earlier start on this PC as it is now
// answers it; a pass is saved. `on_message` is given what the check's process
// sends as it goes (run_isolated's). Throws what run_isolated throws, and
// Cancelled before anything when `cancel` is set.
nlohmann::json run_gpu_check(std::string_view task, std::string_view what, const nlohmann::json& request,
                             const CancelToken& cancel,
                             const std::function<void(const nlohmann::json&)>& on_message = {});

// The key a check's answer is saved under: `pc` (what the answers depend on in
// this process: the app, its GPU files, the GPUs, their drivers, the
// environment), the check `task` and its `request`, but for the GPUs the
// request skips (Vship's probe run again without a GPU whose check crashed).
// The first request, which crashed, is then answered at the next start by the
// answer the retry found, rather than crashing a check process at every start.
nlohmann::json gpu_check_key(nlohmann::json pc, std::string_view task, nlohmann::json request);

// Deletes every saved answer, so that the checks run again at the next start:
// after a GPU fault in a run (the saved answer may be wrong now), and for
// the checks a user asks for (the window's self-test, vml-cli devices).
void forget_gpu_checks();

// Whether `answer` of check `task` says the GPU works: the answers saved.
bool gpu_check_passed(std::string_view task, const nlohmann::json& answer);

// The answer saved in `file` under `key`, or nothing (also when the file is
// missing, unreadable or damaged).
std::optional<nlohmann::json> saved_gpu_check(const std::filesystem::path& file, const nlohmann::json& key);

// Saves `answer` in `file` under `key`, beside the last few others (each GPU
// of a PC, a driver switched back to). Written whole, so a crash or a full
// disk leaves the file as it was; a failure to write is ignored, as the check
// then only runs again.
void save_gpu_check(const std::filesystem::path& file, const nlohmann::json& key, const nlohmann::json& answer);

// The stamps (path, size, time written) of the Vulkan manifests and of the
// libraries they name, which the Vulkan loader loads.
nlohmann::json vulkan_manifest_stamps(const std::vector<std::filesystem::path>& manifests);

}  // namespace vml
