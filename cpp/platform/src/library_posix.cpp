// DynamicLibrary on Linux and macOS: dlopen.
#include "vml/platform/library.h"

#include <dlfcn.h>

#include <cstring>
#include <utility>

#include "vml/platform/system.h"

namespace vml::platform {

DynamicLibrary::DynamicLibrary(const std::filesystem::path& path) : path_(path), path_text_(path_text(path)) {
    handle_ = dlopen(std::filesystem::absolute(path).c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
        const char* why = dlerror();
        throw LibraryError(path_text_ + " could not be loaded: " + (why != nullptr ? why : "unknown error"));
    }
}

DynamicLibrary DynamicLibrary::system(const std::string& name) {
    DynamicLibrary library;
    library.path_ = path_from_text(name);
    library.path_text_ = name;
    // A name without a folder: the system's loader looks for it in its usual
    // places.
    library.handle_ = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library.handle_ == nullptr) {
        const char* why = dlerror();
        throw LibraryError(name + " could not be loaded: " + (why != nullptr ? why : "unknown error"));
    }
    return library;
}

DynamicLibrary::~DynamicLibrary() {
    if (handle_ != nullptr) {
        dlclose(handle_);
    }
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : path_(std::move(other.path_)), path_text_(std::move(other.path_text_)),
      handle_(std::exchange(other.handle_, nullptr)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            dlclose(handle_);
        }
        path_ = std::move(other.path_);
        path_text_ = std::move(other.path_text_);
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

DynamicLibrary::Function DynamicLibrary::find(const char* name) const {
    void* const address = dlsym(handle_, name);
    // POSIX makes an object pointer from dlsym usable as a function pointer;
    // copying it, rather than casting it, keeps ISO C++'s pedantic checks quiet.
    Function function = nullptr;
    std::memcpy(&function, &address, sizeof function);
    return function;
}

}  // namespace vml::platform
