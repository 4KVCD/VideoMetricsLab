// The system on Windows.
#include "vml/platform/system.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <dxgi.h>
#include <windows.h>

#include <algorithm>
#include <cwctype>
#include <thread>

#include "text_windows.h"

#pragma comment(lib, "dxgi.lib")

namespace vml::platform {

std::optional<std::string> environment_variable(std::string_view name) {
    const std::wstring wide_name = widen(name);
    const DWORD length = GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
    if (length == 0) {
        return std::nullopt;
    }
    std::wstring value(length, L'\0');
    const DWORD written = GetEnvironmentVariableW(wide_name.c_str(), value.data(), length);
    value.resize(written);
    return narrow(value);
}

std::filesystem::path home_directory() {
    if (const std::optional<std::string> profile = environment_variable("USERPROFILE")) {
        return path_from_text(*profile);
    }
    const std::string drive = environment_variable("HOMEDRIVE").value_or("");
    return path_from_text(drive + environment_variable("HOMEPATH").value_or(""));
}

std::string executable_name(std::string_view name) { return std::string(name) + ".exe"; }

std::optional<std::filesystem::path> find_on_path(std::string_view name) {
    // The extensions to try, from PATHEXT; none when the name has one of them.
    std::vector<std::wstring> extensions;
    std::wstring wanted = widen(name);
    std::wstring lowered = wanted;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](wchar_t c) { return std::towlower(c); });
    const std::wstring pathext = widen(environment_variable("PATHEXT").value_or(".COM;.EXE;.BAT;.CMD"));
    bool has_extension = false;
    for (std::size_t start = 0; start <= pathext.size();) {
        const std::size_t end = std::min(pathext.find(L';', start), pathext.size());
        std::wstring extension = pathext.substr(start, end - start);
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](wchar_t c) { return std::towlower(c); });
        if (!extension.empty()) {
            extensions.push_back(extension);
            if (lowered.size() > extension.size() && lowered.ends_with(extension)) {
                has_extension = true;
            }
        }
        start = end + 1;
    }
    if (has_extension) {
        extensions.assign(1, L"");
    }
    std::vector<std::filesystem::path> folders{std::filesystem::current_path()};
    const std::wstring path = widen(environment_variable("PATH").value_or(""));
    for (std::size_t start = 0; start <= path.size();) {
        const std::size_t end = std::min(path.find(L';', start), path.size());
        if (end > start) {
            folders.emplace_back(path.substr(start, end - start));
        }
        start = end + 1;
    }
    for (const std::filesystem::path& folder : folders) {
        for (const std::wstring& extension : extensions) {
            std::filesystem::path candidate = folder / (wanted + extension);
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error)) {
                return candidate;
            }
        }
    }
    return std::nullopt;
}

unsigned logical_cores() { return std::max(1u, std::thread::hardware_concurrency()); }

std::vector<unsigned> gpu_vendor_ids() {
    std::vector<unsigned> vendors;
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
        return vendors;
    }
    IDXGIAdapter1* adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description)) && !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            vendors.push_back(description.VendorId);
        }
        adapter->Release();
    }
    factory->Release();
    return vendors;
}

std::string path_text(const std::filesystem::path& path) { return narrow(path.native()); }

std::filesystem::path path_from_text(std::string_view utf8) { return std::filesystem::path(widen(utf8)); }

}  // namespace vml::platform
