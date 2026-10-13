// Loading a shared library (a DLL on Windows) while the app runs, and
// finding its functions: how the app uses the libraries it ships beside
// itself (libvmaf-fast, its Vulkan engine, Vship, the GPU decoders), as the
// Python app does through ctypes.
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace vml::platform {

// A library could not be loaded, or has no function of a name.
class LibraryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class DynamicLibrary {
public:
    // Any function's address, before it is given its own type.
    using Function = void (*)();

    // Loads the library at `path`. On Windows the libraries it needs are
    // looked for in its own folder, then in the system's, as Python's ctypes
    // looks for them; on Linux and macOS, the system's loader finds them by
    // its usual rules. Throws LibraryError when it cannot be loaded, saying
    // why.
    explicit DynamicLibrary(const std::filesystem::path& path);

    // Loads a library the system has, by its file name: from Windows' system
    // folder (System32), or wherever the system's loader finds it on Linux
    // and macOS. For a driver's library, such as NVIDIA's CUDA driver
    // (nvcuda.dll, libcuda.so.1). Throws LibraryError as the constructor does.
    static DynamicLibrary system(const std::string& name);

    ~DynamicLibrary();
    DynamicLibrary(DynamicLibrary&& other) noexcept;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    // The exported function `name`, or nullptr when the library has none.
    Function find(const char* name) const;

    // The exported function `name` as a pointer of type F, such as
    // int (*)(int, int). Throws LibraryError when the library has none.
    template <typename F>
    F function(const char* name) const {
        const Function found = find(name);
        if (found == nullptr) {
            throw LibraryError(path_text_ + " has no function " + name);
        }
        return reinterpret_cast<F>(found);
    }

    const std::filesystem::path& path() const { return path_; }

    // The system's own handle of the library (an HMODULE on Windows), for a
    // library that is handed another one to use.
    void* native_handle() const { return handle_; }

private:
    DynamicLibrary() = default;

    std::filesystem::path path_;
    std::string path_text_;  // for messages
    void* handle_ = nullptr;
};

}  // namespace vml::platform
