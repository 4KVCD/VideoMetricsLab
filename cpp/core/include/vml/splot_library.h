// splot, the AV2 decoder (native/av2/splot_api.h), loaded once for the life
// of the process from splot_library_path(): for probing AV2 videos
// (av2_probe.cpp) and Video Compare's playback. The GPU metrics' software
// decoder is handed the library's handle instead (gpu_frames.cpp).
#pragma once

#include <optional>
#include <string>

#include "av2/splot_api.h"
#include "vml/platform/library.h"

namespace vml {

struct SplotLibrary {
    std::optional<platform::DynamicLibrary> library;  // nothing where it cannot be used
    SplotCalls calls;
    std::string problem;  // why it cannot be used: "splot is not bundled", ...
};

const SplotLibrary& splot_library();

// The settings the app decodes with: as many threads as the CPU has, the
// frames in flight splot's choice, the film grain added where `apply_grain`
// (grain_decoding.h).
SplotSettings splot_settings(const SplotCalls& calls, bool apply_grain);

}  // namespace vml
