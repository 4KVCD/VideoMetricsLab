// What the app asks of the system it runs on: folders, the PATH, the
// environment, the CPU and the GPUs.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vml::platform {

// The user's home folder, as Python's Path.home() finds it: USERPROFILE on
// Windows, HOME elsewhere.
std::filesystem::path home_directory();

// An environment variable's value, or nothing when it is not set.
std::optional<std::string> environment_variable(std::string_view name);

// A program's file name on this system: "ffmpeg.exe" on Windows, "ffmpeg"
// elsewhere.
std::string executable_name(std::string_view name);

// The program `name` as Python's shutil.which finds it: in the current
// folder and then each folder on the PATH on Windows (with PATHEXT's
// extensions), on the PATH elsewhere. Nothing when it is not found.
std::optional<std::filesystem::path> find_on_path(std::string_view name);

// How many logical cores the CPU has (Python's os.cpu_count()), at least 1.
unsigned logical_cores();

// The PCI vendor IDs of the hardware GPUs DirectX lists (Windows only; empty
// elsewhere): 0x10DE NVIDIA, 0x1002 and 0x1022 AMD, 0x8086 Intel.
std::vector<unsigned> gpu_vendor_ids();

// A path as UTF-8 text, and back; the app keeps paths as UTF-8.
std::string path_text(const std::filesystem::path& path);
std::filesystem::path path_from_text(std::string_view utf8);

}  // namespace vml::platform
