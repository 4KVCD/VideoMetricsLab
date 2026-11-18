// Writing a .zip file of files made in memory (the window's graph exports).
#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace vml {

// `files` (each its name in the archive and its bytes, in this order),
// deflated, as one .zip file at `destination`, replacing it. Throws
// std::runtime_error, saying what failed, when it cannot.
void write_zip(const std::filesystem::path& destination, const std::vector<std::pair<std::string, std::string>>& files);

}  // namespace vml
