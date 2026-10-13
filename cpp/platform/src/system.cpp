// What every system's part of system.h shares.
#include "vml/platform/system.h"

#include <fstream>
#include <system_error>

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

}  // namespace vml::platform
