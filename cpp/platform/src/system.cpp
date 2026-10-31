// What every system's part of system.h shares.
#include "vml/platform/system.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <system_error>
#include <utility>

namespace vml::platform {

std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (!file.is_open() || error) {
        return std::nullopt;
    }
    std::string data(static_cast<std::size_t>(size), '\0');
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    data.resize(static_cast<std::size_t>(file.gcount()));  // a file cut short since its size was read
    return data;
}

DisplayHdrInfo display_hdr_from_edr(double potential_headroom, std::string device_name) {
    DisplayHdrInfo info;
    info.device_name = std::move(device_name);
    if (!std::isfinite(potential_headroom) || potential_headroom <= 0) {
        return info;
    }
    info.hdr_enabled = potential_headroom > 1.0;
    if (*info.hdr_enabled) {
        info.hdr_supported = true;
    }
    info.edr_headroom = std::max(1.0, potential_headroom);
    return info;
}

}  // namespace vml::platform
