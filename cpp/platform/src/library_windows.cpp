// DynamicLibrary on Windows: LoadLibraryExW.
#include "vml/platform/library.h"

#include "text_windows.h"  // first: it sets <windows.h> up

#include <utility>

#include "vml/platform/system.h"

namespace vml::platform {

namespace {

// Windows' own words for an error code.
std::string message_for(DWORD code) {
    wchar_t* text = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0,
        reinterpret_cast<wchar_t*>(&text), 0, nullptr);
    std::wstring said = length > 0 ? std::wstring(text, length) : std::wstring();
    LocalFree(text);
    while (!said.empty() && (said.back() == L'\n' || said.back() == L'\r' || said.back() == L' ' || said.back() == L'.')) {
        said.pop_back();
    }
    return narrow(said) + " (Windows error " + std::to_string(code) + ")";
}

}  // namespace

DynamicLibrary::DynamicLibrary(const std::filesystem::path& path) : path_(path), path_text_(path_text(path)) {
    // The libraries it needs are looked for in its own folder, then in the
    // system's: not in the current folder or on the PATH, where another
    // program's copy could be found first.
    const std::filesystem::path absolute = std::filesystem::absolute(path);
    handle_ = LoadLibraryExW(absolute.c_str(), nullptr,
                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (handle_ == nullptr) {
        throw LibraryError(path_text_ + " could not be loaded: " + message_for(GetLastError()));
    }
}

DynamicLibrary::~DynamicLibrary() {
    if (handle_ != nullptr) {
        FreeLibrary(static_cast<HMODULE>(handle_));
    }
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : path_(std::move(other.path_)), path_text_(std::move(other.path_text_)),
      handle_(std::exchange(other.handle_, nullptr)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            FreeLibrary(static_cast<HMODULE>(handle_));
        }
        path_ = std::move(other.path_);
        path_text_ = std::move(other.path_text_);
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

DynamicLibrary::Function DynamicLibrary::find(const char* name) const {
    return reinterpret_cast<Function>(GetProcAddress(static_cast<HMODULE>(handle_), name));
}

}  // namespace vml::platform
