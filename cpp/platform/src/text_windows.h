// Text between the app's UTF-8 and Windows' UTF-16.
#pragma once

// <windows.h> without what the app does not use, and without its min and
// max macros (MinGW's C++ headers may have set these already).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <string>
#include <string_view>

namespace vml::platform {

// UTF-8 as UTF-16. A byte that is not UTF-8 becomes U+FFFD.
inline std::wstring widen(std::string_view utf8) {
    if (utf8.empty()) {
        return {};
    }
    const int length = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), length);
    return wide;
}

// UTF-16 as UTF-8.
inline std::string narrow(std::wstring_view wide) {
    if (wide.empty()) {
        return {};
    }
    const int length =
        WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
    std::string utf8(static_cast<std::size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), utf8.data(), length, nullptr,
                        nullptr);
    return utf8;
}

}  // namespace vml::platform
