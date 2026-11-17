#include "vml/gpu_checks.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "vml/app_log.h"
#include "vml/build_info.h"
#include "vml/isolated.h"
#include "vml/libvmaf.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"
#include "vml/settings.h"
#include "vml/temp_files.h"
#include "vml/vmaf_vulkan.h"
#include "vml/vship.h"

namespace vml {
namespace {

// How many answers are kept: enough for each check on each GPU of a PC and a
// driver switched back to, while the file stays a few KB.
constexpr std::size_t kept_answers = 16;

// Saves and deletions of this process, one at a time: the checks run on
// threads of their own, and each save reads the file, adds to it and writes
// it back.
std::mutex file_mutex;

// A file's path, size and time written, or null when it cannot be read.
nlohmann::json stamp(const std::filesystem::path& file) {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(file, error);
    const std::optional<long long> written = platform::modified_ns(file);
    if (error || !written) {
        return nullptr;
    }
    return {platform::path_text(file), size, *written};
}

// The values of "library_path" anywhere in a manifest: a driver's ("ICD"),
// a layer's ("layer"), each of a list's ("layers").
void add_library_paths(const nlohmann::json& value, std::vector<std::string>& paths) {
    if (value.is_object()) {
        for (const auto& [name, item] : value.items()) {
            if (name == "library_path" && item.is_string()) {
                paths.push_back(item.get<std::string>());
            } else {
                add_library_paths(item, paths);
            }
        }
    } else if (value.is_array()) {
        for (const nlohmann::json& item : value) {
            add_library_paths(item, paths);
        }
    }
}

// The answers `file` holds, oldest first; none when it is missing,
// unreadable or damaged.
nlohmann::json saved_answers(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        return nlohmann::json::array();
    }
    const nlohmann::json content = nlohmann::json::parse(in, nullptr, false);
    if (!content.is_object() || !content.contains("answers") || !content["answers"].is_array()) {
        return nlohmann::json::array();
    }
    return content["answers"];
}

// The environment variables that steer which GPUs and drivers the checks
// use: the Vulkan loader's (VK_), the app's, the GPU makers' device lists,
// and VMAF v1's test switches.
nlohmann::json steering_environment() {
    nlohmann::json values = nlohmann::json::array();
    for (const char* prefix : {"VK_", "VMAF_FAST_TEST_"}) {
        for (std::string& variable : platform::environment_starting_with(prefix)) {
            values.push_back(std::move(variable));
        }
    }
    for (const char* name : {"VML_GPU_VENDOR", "VML_VMAF_VULKAN_DEVICE", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                             "ROCR_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR"}) {
        if (const std::optional<std::string> value = platform::environment_variable(name)) {
            values.push_back(std::string(name) + "=" + *value);
        }
    }
    return values;
}

// The stamps of the files the checks run: this program (the checks' code),
// the GPU libraries they load, the VMAF v1 models they score with.
nlohmann::json check_files() {
    nlohmann::json files = nlohmann::json::array();
    if (const std::optional<std::filesystem::path> program = isolation_program()) {
        files.push_back(stamp(*program));
    }
    files.push_back(stamp(vulkan_engine_path()));
    files.push_back(stamp(libvmaf_folder() / platform::shared_library_name("libvmaf")));
    files.push_back(stamp(vship_library_path(VshipBuild::fast)));
    files.push_back(stamp(vship_library_path(VshipBuild::stock)));
    std::error_code error;
    for (const auto& model : std::filesystem::directory_iterator(resource_folder() / "models", error)) {
        files.push_back(stamp(model.path()));
    }
    return files;
}

// What the checks' answers depend on in this process, other than the check
// itself: the app, its GPU files, the GPUs and their drivers, the
// environment. Nothing where the GPUs and drivers cannot be told apart. Read
// once: it holds no Vulkan or CUDA, only what the system lists, and up to
// five checks ask for it.
const std::optional<nlohmann::json>& this_pc() {
    static const std::optional<nlohmann::json> pc = []() -> std::optional<nlohmann::json> {
        const std::optional<platform::GpuDriverFacts> facts = platform::gpu_driver_facts();
        if (!facts) {
            return std::nullopt;
        }
        nlohmann::json driver_files = nlohmann::json::array();
        for (const std::filesystem::path& file : facts->driver_files) {
            driver_files.push_back(stamp(file));
        }
        std::vector<std::filesystem::path> manifests = facts->vulkan_manifests;
        std::sort(manifests.begin(), manifests.end());
        return nlohmann::json{{"app", std::string(VML_VERSION) + " " + build_commit()},
                              {"files", check_files()},
                              {"gpus", facts->gpus},
                              {"vulkan", vulkan_manifest_stamps(manifests)},
                              {"driver_files", driver_files},
                              {"environment", steering_environment()}};
    }();
    return pc;
}

}  // namespace

std::filesystem::path gpu_checks_file() { return Settings::file().parent_path() / "gpu_checks.json"; }

nlohmann::json vulkan_manifest_stamps(const std::vector<std::filesystem::path>& manifests) {
    nlohmann::json stamps = nlohmann::json::array();
    for (const std::filesystem::path& manifest : manifests) {
        nlohmann::json entry = {{"manifest", stamp(manifest)}};
        std::ifstream in(manifest, std::ios::binary);
        std::vector<std::string> libraries;
        add_library_paths(nlohmann::json::parse(in, nullptr, false), libraries);
        nlohmann::json library_stamps = nlohmann::json::array();
        for (const std::string& library : libraries) {
            // A path with a folder is relative to the manifest's; a bare
            // name is found by the system's search, and only its name is
            // known.
            const std::filesystem::path path = platform::path_from_text(library);
            library_stamps.push_back(path.has_parent_path() ? stamp(manifest.parent_path() / path)
                                                            : nlohmann::json(library));
        }
        entry["libraries"] = std::move(library_stamps);
        stamps.push_back(std::move(entry));
    }
    return stamps;
}

bool gpu_check_passed(std::string_view task, const nlohmann::json& answer) {
    if (!answer.is_object()) {
        return false;
    }
    if (task == "vship-probe") {
        return answer.contains("device") && answer["device"].is_object();
    }
    return answer.value("available", false) && !answer.value("failed", false);
}

std::optional<nlohmann::json> saved_gpu_check(const std::filesystem::path& file, const nlohmann::json& key) {
    // Not while this process writes the file: on Windows, a file held open
    // cannot be replaced, and the other check's pass would not be saved.
    const std::lock_guard lock(file_mutex);
    for (const nlohmann::json& entry : saved_answers(file)) {
        if (entry.is_object() && entry.value("key", nlohmann::json()) == key && entry.contains("answer")) {
            return entry["answer"];
        }
    }
    return std::nullopt;
}

void save_gpu_check(const std::filesystem::path& file, const nlohmann::json& key, const nlohmann::json& answer) {
    const std::lock_guard lock(file_mutex);
    nlohmann::json answers = saved_answers(file);
    answers.erase(std::remove_if(answers.begin(), answers.end(),
                                 [&key](const nlohmann::json& entry) {
                                     return !entry.is_object() || entry.value("key", nlohmann::json()) == key;
                                 }),
                  answers.end());
    answers.push_back({{"key", key}, {"answer", answer}});
    if (answers.size() > kept_answers) {
        answers.erase(answers.begin(), answers.end() - static_cast<std::ptrdiff_t>(kept_answers));
    }
    const nlohmann::json content = {{"answers", answers}};
    std::error_code ignored;
    std::filesystem::create_directories(file.parent_path(), ignored);
    try {
        write_file_whole(file, [&](std::ostream& out) { out << content.dump() << '\n'; });
    } catch (const std::runtime_error&) {
        // Not saved: the check runs again next start.
    }
}

nlohmann::json run_gpu_check(std::string_view task, std::string_view what, const nlohmann::json& request,
                             const CancelToken& cancel) {
    cancel.throw_if_cancelled();
    std::optional<nlohmann::json> key;
    if (const std::optional<nlohmann::json>& pc = this_pc()) {
        key = *pc;
        (*key)["check"] = {{"task", task}, {"request", request}};
        if (std::optional<nlohmann::json> saved = saved_gpu_check(gpu_checks_file(), *key)) {
            app_log::write(app_log::Level::info, "vml.gpu_checks",
                           std::string(what) + ": the answer of an earlier start, on this GPU and driver");
            return std::move(*saved);
        }
    }
    nlohmann::json answer = run_isolated(task, what, request, {}, cancel);
    if (key && gpu_check_passed(task, answer)) {
        save_gpu_check(gpu_checks_file(), *key, answer);
    }
    return answer;
}

void forget_gpu_checks() {
    const std::lock_guard lock(file_mutex);
    std::error_code error;
    if (std::filesystem::remove(gpu_checks_file(), error)) {
        app_log::write(app_log::Level::info, "vml.gpu_checks", "The GPU checks will run again at the next start");
    }
}

}  // namespace vml
