#include "native_grain.h"

#include <d3d11.h>

#include <mutex>
#include <stdexcept>
#include <string>

#include "grain_params.h"
#include "vml/model_select.h"
#include "vml/platform/library.h"

namespace vml::app {

namespace {

// The library's functions, loaded once for the life of the process; or why
// they could not be.
struct GrainLibrary {
    std::optional<platform::DynamicLibrary> library;
    QString problem;
    int (*prepare)() = nullptr;
    void* (*create)(ID3D11Device*) = nullptr;
    const char* (*error)(void*) = nullptr;
    int (*apply_planes)(void*, ID3D11Texture2D* const*, const unsigned*, int, const GrainParams*) = nullptr;
    int (*pack)(void*, ID3D11Texture2D* const*, int, int, int, int, ID3D11Texture2D*) = nullptr;
    void (*destroy)(void*) = nullptr;
};

const GrainLibrary& grain_library() {
    static std::once_flag loaded;
    static GrainLibrary functions;
    std::call_once(loaded, [] {
        // Beside the presenter, where the Python app has it too.
        const std::filesystem::path path = resource_folder() / "native" / "d3d11_film_grain.dll";
        std::error_code ignored;
        if (!std::filesystem::is_regular_file(path, ignored)) {
            functions.problem = QStringLiteral("d3d11_film_grain.dll is not bundled");
            return;
        }
        try {
            const platform::DynamicLibrary& library = functions.library.emplace(path);
            functions.prepare = library.function<decltype(functions.prepare)>("vmaf_grain_prepare");
            functions.create = library.function<decltype(functions.create)>("vmaf_grain_create");
            functions.error = library.function<decltype(functions.error)>("vmaf_grain_error");
            functions.apply_planes = library.function<decltype(functions.apply_planes)>("vmaf_grain_apply_planes");
            functions.pack = library.function<decltype(functions.pack)>("vmaf_grain_pack");
            functions.destroy = library.function<decltype(functions.destroy)>("vmaf_grain_destroy");
        } catch (const platform::LibraryError& error) {
            // A library without these functions: an older build.
            functions.problem = QStringLiteral("d3d11_film_grain.dll could not be used: %1")
                                    .arg(QString::fromUtf8(error.what()));
            functions.library.reset();
            return;
        }
        // Its grain patterns built and its shaders compiled once, now, rather
        // than on the first frame with grain.
        if (functions.prepare() != 0) {
            functions.problem = QStringLiteral("d3d11_film_grain.dll could not compile its shaders");
            functions.library.reset();
        }
    });
    return functions;
}

QString library_error(void* drawer) {
    const char* text = grain_library().error(drawer);
    return QString::fromUtf8(text != nullptr ? text : "");
}

}  // namespace

std::optional<QString> NativeGrain::unavailable() {
    const GrainLibrary& functions = grain_library();
    if (!functions.library) {
        return functions.problem;
    }
    return std::nullopt;
}

NativeGrain::NativeGrain(ID3D11Device* device) {
    if (const std::optional<QString> problem = unavailable()) {
        throw std::runtime_error(problem->toStdString());
    }
    drawer_ = grain_library().create(device);
    if (drawer_ == nullptr) {
        throw std::runtime_error("the film grain drawer could not be made on this GPU");
    }
}

NativeGrain::~NativeGrain() { grain_library().destroy(drawer_); }

std::optional<QString> NativeGrain::draw(ID3D11Texture2D* const planes[3], int count, const GrainParams& params,
                                         std::pair<int, int> chroma_shift, ID3D11Texture2D* target) {
    const GrainLibrary& functions = grain_library();
    const unsigned subresources[3] = {0, 0, 0};
    if (functions.apply_planes(drawer_, planes, subresources, count, &params) != 0) {
        return QStringLiteral("drawing the grain failed: %1").arg(library_error(drawer_));
    }
    if (functions.pack(drawer_, planes, count, params.depth, chroma_shift.first, chroma_shift.second, target) != 0) {
        return QStringLiteral("copying the grained frame failed: %1").arg(library_error(drawer_));
    }
    return std::nullopt;
}

}  // namespace vml::app
