// What a child's output reader keeps of what it read ahead of a line's end, on
// every system (process_windows.cpp, process_posix.cpp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <string>

namespace vml::platform {

// The first whole line of `unread`, without its line end, or nothing while it
// holds no line end.
inline std::optional<std::string> take_line(std::string& unread) {
    const std::size_t end = unread.find('\n');
    if (end == std::string::npos) {
        return std::nullopt;
    }
    std::string line = unread.substr(0, end);
    unread.erase(0, end + 1);
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

// Up to `bytes` of `unread`, moved to `into`; how many.
inline std::size_t take_bytes(std::string& unread, void* into, std::size_t bytes) {
    const std::size_t count = std::min(bytes, unread.size());
    std::memcpy(into, unread.data(), count);
    unread.erase(0, count);
    return count;
}

}  // namespace vml::platform
