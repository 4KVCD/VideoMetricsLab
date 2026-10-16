#include "self_test.h"

#include <QImageReader>
#include <QMessageBox>
#include <QSslSocket>
#include <QStyle>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <tuple>

#include "vml/cancel.h"
#include "vml/ffmpeg_tools.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/isolated.h"
#include "vml/model_select.h"
#include "vml/perceptual_cpu.h"
#include "vml/platform/system.h"
#include "vml/settings.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vship.h"
#ifdef VML_NATIVE_PLAYBACK
#include "native_dav1d.h"
#include "native_device.h"
#include "native_grain.h"
#include "native_presentation.h"
#endif

namespace vml::app {

std::string SelfTestReport::text() const {
    std::string joined;
    for (const std::string& line : lines_) {
        if (!joined.empty()) {
            joined += "\n";
        }
        joined += line;
    }
    return joined;
}

namespace {

// Whether the program runs from a packaged build (its resources beside it)
// or from the build folder.
bool packaged() {
    std::error_code error;
    return std::filesystem::is_directory(platform::executable_path().parent_path() / "resources" / "models", error);
}

std::string join(const std::vector<std::string>& items, const char* separator = ", ") {
    std::string joined;
    for (const std::string& item : items) {
        joined += (joined.empty() ? "" : separator) + item;
    }
    return joined;
}

}  // namespace

SelfTestReport self_test(const QApplication& app) {
    SelfTestReport report(std::string("VideoMetricsLab ") + VML_APP_VERSION + " self-test (C++, Qt " + qVersion() + ")");
    report.note(std::string("packaged build: ") + (packaged() ? "yes" : "no, running from the build folder"));
    const Settings settings = Settings::load();

    const ToolsStatus tools = check_tools(settings);
    if (tools.ok()) {
        report.ok("ffmpeg " + format_version(tools.ffmpeg.version) + " and ffprobe");
    } else {
        for (const std::string& problem : tools.problems()) {
            report.fail(problem);
        }
    }

#ifdef VML_NATIVE_PLAYBACK
    // Native playback: FFmpeg's libraries, the GPU's decoders and the
    // presenter (where the Python app checked GStreamer and its elements).
    if (const std::optional<std::string> problem = load_ffmpeg_libraries()) {
        report.warn("Native playback unavailable, videos play through ffmpeg.exe: " + *problem);
    } else {
        report.ok("FFmpeg's libraries for native playback (tools/ffmpeg)");
        try {
            const std::vector<std::string> decoders = NativeDevice::shared()->gpu_decoders();
            if (decoders.empty()) {
                report.warn("no Direct3D 11 GPU decoders; video decodes in software");
            } else {
                report.ok("GPU decoders on this machine (Direct3D 11): " + join(decoders));
            }
        } catch (const std::exception& error) {
            report.warn(std::string("no Direct3D 11 device, videos play through ffmpeg.exe: ") + error.what());
        }
        if (const std::optional<QString> no_grain = NativeGrain::unavailable()) {
            report.warn("Film grain for VVC videos unavailable (" + no_grain->toStdString() +
                        "); only FFmpeg's is drawn: on 8-bit video, and AV1's");
        } else {
            report.ok("Film grain for VVC videos in Video Compare (d3d11_film_grain.dll; AV1's by FFmpeg)");
        }
        if (const std::optional<QString> no_dav1d = Dav1dVideo::unavailable()) {
            report.warn("Video Compare cannot decode AV1 in software (" + no_dav1d->toStdString() +
                        "): on a GPU without an AV1 decoder, AV1 videos play through ffmpeg.exe");
        } else {
            report.ok("AV1 in software in Video Compare, where the GPU has no AV1 decoder (dav1d)");
        }
    }
    if (const std::optional<QString> problem = NativePresentation::unavailable()) {
        report.warn("Native playback's presenter absent or out of date; videos play through FFmpeg");
    } else {
        report.ok("Native playback's presenter and HDR shader (d3d11_tonemap.dll)");
    }
#else
    report.warn("Native playback is not built on this system; videos play through ffmpeg.exe");
#endif

    // Which Qt platform plugin, style and image formats loaded: the packaged
    // build ships a trimmed Qt, and if a piece were missing Qt would quietly
    // fall back to a plain style or refuse an image format.
    std::vector<std::string> formats;
    for (const QByteArray& format : QImageReader::supportedImageFormats()) {
        formats.push_back(format.toStdString());
    }
    std::sort(formats.begin(), formats.end());
    report.ok(std::string("Qt ") + qVersion() + " on '" + QGuiApplication::platformName().toStdString() +
              "', style '" + app.style()->name().toStdString() + "', images: " + join(formats));
    // The update check asks GitHub over HTTPS, through a TLS plugin of Qt's
    // (tls/): without one it fails without a word but a line in the log.
    if (const QString backend = QSslSocket::activeBackend(); !backend.isEmpty() && QSslSocket::supportsSsl()) {
        report.ok("HTTPS for the update check (Qt's TLS backend '" + backend.toStdString() + "')");
    } else {
        report.warn("no TLS backend in Qt's plugins: the update check cannot ask GitHub");
    }

    for (const auto& [backend, maker, library] :
         {std::tuple("nvidia", "NVIDIA", "nvdec_frames"), std::tuple("intel", "Intel", "vpl_frames"),
          std::tuple("amd", "AMD", "amf_frames")}) {
        const std::string name = platform::shared_library_name(library);
        if (decoder_available(backend)) {
            report.ok(std::string(maker) + " frame decoder for the GPU metrics (" + name + ")");
        } else {
            report.warn(std::string(maker) + " frame decoder absent (" + name +
                        "); FFmpeg decodes the GPU metrics' videos");
        }
    }
    // The software decoder uses the app's FFmpeg libraries (tools/ffmpeg) and
    // dav1d beside them, so the codecs it reports say whether those load too.
    const std::string software = platform::shared_library_name("software_frames");
    const auto [codecs, why] = software_codecs();
    if (!codecs.empty()) {
        report.ok("software frame decoder (" + software + "): " + join(codecs));
    } else {
        report.warn("software frame decoder absent (" + software + ": " + why +
                    "); FFmpeg's pipes carry the videos it decodes in software");
    }

    for (const char* metric : {"ssimulacra2", "butteraugli"}) {
        if (const std::optional<std::filesystem::path> tool = find_metric_executable(metric)) {
            report.ok(std::string(metric) + " (" + platform::path_text(*tool) + ")");
        } else {
            report.warn(std::string(metric) + " tool absent");
        }
    }

    // The GPU libraries run in processes of their own (isolated.h): each is
    // this program started again.
    try {
        const CancelToken cancel;
        const nlohmann::json child = run_isolated("process-id", "a child process", nlohmann::json::object(),
                                                  [](const nlohmann::json&) {}, cancel);
        report.ok("GPU libraries run in a process of their own (started process " + child.dump() + ")");
    } catch (const std::exception& error) {
        report.fail(std::string("no process for the GPU libraries could be started: ") + error.what());
    }

    set_vship_backend(settings.gpu_backend);
    const auto [vship_device, vship_reason] = detect_vship_device();
    if (vship_device) {
        report.ok("Vship " + vship_device->version + " GPU metrics (" + backend_label(vship_device->backend) + ": " +
                  vship_device->name + ")");
    } else {
        report.warn("Vship GPU metrics unavailable; CPU fallback is enabled (" + vship_reason + ")");
    }

    set_gpu_backend(settings.gpu_backend);
    const GpuVmafBackend vmaf = gpu_vmaf_backend();
    if (!vmaf.backend.empty()) {
        report.ok("VMAF on the GPU (" + vmaf.text + ")");
    } else {
        report.warn("VMAF on the GPU unavailable; FFmpeg's libvmaf is used (" + vmaf.text + ")");
    }
    const V1Probe v1 = v1_available();
    if (v1.available) {
        report.ok("VMAF v1 with the GPU (" + v1.text + ")");
    } else {
        report.warn("VMAF v1 with the GPU unavailable; FFmpeg's libvmaf is used (" + v1.text + ")");
    }
    return report;
}

int run_self_test(const QApplication& app, bool quiet) {
    const SelfTestReport report = self_test(app);
    // Both printed and saved: a packaged build has no console to print to,
    // and an automated check has no one to close a dialog.
    std::fputs((report.text() + "\n").c_str(), stdout);
    std::optional<std::filesystem::path> destination = Settings::file().parent_path() / "self-test.txt";
    std::error_code error;
    std::filesystem::create_directories(destination->parent_path(), error);
    std::ofstream file(*destination, std::ios::binary);
    file << report.text();
    // Closed before checking: what is still buffered is written at close.
    file.close();
    if (!file) {
        destination.reset();
    }
    if (!quiet) {
        QMessageBox box;
        box.setWindowTitle(QStringLiteral("Self-test"));
        box.setText(QString::fromStdString(report.text()) +
                    (destination ? QStringLiteral("\n\nSaved to %1").arg(QString::fromStdString(platform::path_text(*destination)))
                                 : QString()));
        box.exec();
    }
    return report.failed() ? 1 : 0;
}

}  // namespace vml::app
