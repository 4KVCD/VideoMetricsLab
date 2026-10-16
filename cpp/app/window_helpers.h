// Small helpers the window and its panels share: paths and text as the
// window shows them, and what the GPU check found (vship_available).
#pragma once

#include <QString>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "vml/platform/system.h"
#include "vml/vship.h"

namespace vml::app {

inline QString qt_text(std::string_view text) {
    return QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()));
}

inline QString text_of(const std::filesystem::path& path) { return QString::fromStdString(platform::path_text(path)); }

inline QString name_of(const std::filesystem::path& path) { return text_of(path.filename()); }

// A path as the window takes it from a dialog, a drop or the reference box:
// with the system's separators, as Python's Path gives it.
inline std::filesystem::path path_of(const QString& text) {
    return platform::path_from_text(text.toStdString()).make_preferred();
}

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

// Whether a GPU Vship can use is here (SSIMULACRA2 and Butteraugli set to the
// GPU, and CVVDP, run on it), as the last GPU check to finish found. The
// window never waits for a check: one starts with the app
// and takes seconds, and a run checks again after a failed one. Until the
// first has finished, a GPU is taken to be there, and the rows are brought up
// to date when it finishes (MainWindow::refresh_metric_cells); a run waits
// for it, off the window's thread (MainWindow::wait_for_gpu_checks).
inline bool vship_available() {
    const auto known = known_vship_device();
    return !known || known->first.has_value();
}

}  // namespace vml::app
