#pragma once
// write_file_whole's parts, for its tests.

#include <filesystem>
#include <functional>
#include <ostream>

namespace vml {

// write_file_whole, with `wait` called before each new try of a rename
// refused while the file is open elsewhere (`attempt` from 0): it sleeps.
void write_file_whole(const std::filesystem::path& path, const std::function<void(std::ostream&)>& write,
                      const std::function<void(int attempt)>& wait);

}  // namespace vml
