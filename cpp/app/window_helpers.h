// Small helpers the window's two files share (main_window.cpp,
// main_window_run.cpp): paths and text as the window shows them.
#pragma once

#include <QString>

#include <filesystem>
#include <string_view>

#include "vml/platform/system.h"
#include "vml/vship.h"

namespace vml::app {

inline QString qt_text(std::string_view text) {
    return QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()));
}

inline QString text_of(const std::filesystem::path& path) { return QString::fromStdString(platform::path_text(path)); }

inline QString name_of(const std::filesystem::path& path) { return text_of(path.filename()); }

// Whether two paths name the same file as Python's Path compares them: on
// Windows, whatever the case of their letters.
inline bool same_path(const std::filesystem::path& a, const std::filesystem::path& b) {
#ifdef _WIN32
    return QString::compare(text_of(std::filesystem::path(a).make_preferred()),
                            text_of(std::filesystem::path(b).make_preferred()), Qt::CaseInsensitive) == 0;
#else
    return a == b;
#endif
}

// Whether two paths name the same file once resolved (the Python app's
// Path.resolve() ==): a saved result records a path as it was given,
// relative or absolute, with either slash, in any case on Windows.
inline bool same_file(const std::filesystem::path& a, const std::filesystem::path& b) {
    const auto resolved = [](const std::filesystem::path& path) {
        std::error_code error;
        const std::filesystem::path absolute = std::filesystem::absolute(path, error);
        const std::filesystem::path canonical = std::filesystem::weakly_canonical(error ? path : absolute, error);
        return error ? absolute : canonical;
    };
    return same_path(resolved(a), resolved(b));
}

// Whether a GPU Vship can use is here (CVVDP runs on it); the probe is kept
// after the first time.
inline bool vship_available() { return detect_vship_device().first.has_value(); }

}  // namespace vml::app
