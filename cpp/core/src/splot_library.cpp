#include "vml/splot_library.h"

#include <cstring>
#include <filesystem>
#include <mutex>

#include "vml/gpu_frames.h"

namespace vml {

namespace {

void load(SplotLibrary& splot) {
    const std::filesystem::path path = splot_library_path();
    std::error_code ignored;
    if (!std::filesystem::is_regular_file(path, ignored)) {
        splot.problem = "splot is not bundled";
        return;
    }
    try {
        const platform::DynamicLibrary& loaded = splot.library.emplace(path);
        if (const char* missing = splot.calls.bind([&](const char* name) { return loaded.find(name); })) {
            splot.problem = std::string("splot has no ") + missing;
        } else if (std::strcmp(splot.calls.version(), kSplotVersion) != 0) {
            splot.problem = "splot is not the version the app was made for";
        }
    } catch (const platform::LibraryError& error) {
        splot.problem = std::string("splot could not be loaded: ") + error.what();
    }
    if (!splot.problem.empty()) {
        splot.calls = {};
        splot.library.reset();
    }
}

}  // namespace

const SplotLibrary& splot_library() {
    static std::once_flag loaded;
    static SplotLibrary splot;
    std::call_once(loaded, [] { load(splot); });
    return splot;
}

SplotSettings splot_settings(const SplotCalls& calls, bool apply_grain) {
    SplotSettings settings{};
    calls.default_settings(&settings);
    settings.n_threads = 0;
    settings.max_frame_delay = 0;
    settings.skip_film_grain = apply_grain ? 0 : 1;
    return settings;
}

}  // namespace vml
