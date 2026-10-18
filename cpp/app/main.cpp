// VideoMetricsLab's window, in C++: the Python app's vmaf_app/main.py.
//
// This program is also the process of its own that native GPU code runs in
// (isolated.h), as the Python app's packaged program is: started with
// isolated_argument, it runs that task and makes no window. Started with
// --self-test, it checks what it finds at run time (self_test.h).
#include <QApplication>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d9.h>
#endif

#include "i18n.h"
#include "main_window.h"
#include "self_test.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/gpu_vmaf.h"
#include "vml/isolated.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/settings.h"
#include "vml/temp_files.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vship.h"

namespace {

// The system the language and the theme follow when Settings leaves them so.
#ifdef __APPLE__
const std::string system_name = "macOS";
#else
const std::string system_name = "Windows";
#endif

// Shows the window in the `chosen` language, or the system's (Windows' or
// macOS's) display language when none is chosen, with English where there
// is no translation. Qt's own dialogs and buttons follow, in Qt's
// translations where it has them, and a right-to-left language mirrors the
// window.
void apply_language(const QString& chosen) {
    const QString windows = vml::app::windows_language();
    const QString code = vml::app::set_language(chosen.isEmpty() ? windows : chosen);
    vml::app_log::write(vml::app_log::Level::info, "vml.main",
                        "Language: " + code.toStdString() + " (" +
                            (chosen.isEmpty() ? "as " + system_name : std::string("chosen in Settings")) + "; " +
                            system_name + ": " + windows.toStdString() + ")");
    vml::app::install_qt_translations(code);
    if (vml::app::right_to_left(code)) {
        QApplication::setLayoutDirection(Qt::RightToLeft);
    }
}

QtMessageHandler qt_default_handler = nullptr;

// Qt's own messages, written to the app's log at the matching level, and
// passed on to where Qt writes them otherwise.
void log_qt_message(QtMsgType type, const QMessageLogContext& context, const QString& message) {
    using vml::app_log::Level;
    Level level = Level::warning;
    switch (type) {
        case QtDebugMsg: level = Level::debug; break;
        case QtInfoMsg: level = Level::info; break;
        case QtWarningMsg: level = Level::warning; break;
        case QtCriticalMsg: level = Level::error; break;
        case QtFatalMsg: level = Level::critical; break;
    }
    vml::app_log::write(level, "vml.qt", message.toStdString());
    if (qt_default_handler != nullptr) {
        qt_default_handler(type, context, message);
    }
}

// An exception nothing caught ends the program: said in the log first, and
// the abort that follows is reported with its stack (report_crashes_to). On
// Windows this is the main thread's handler only, as Microsoft's run-time
// keeps one for each thread: an exception that escapes another thread is
// reported as the abort alone.
void log_uncaught_exception() {
    std::string what = "no exception";
    if (const std::exception_ptr error = std::current_exception()) {
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& exception) {
            what = exception.what();
        } catch (...) {
            what = "an exception of no standard type";
        }
    }
    vml::app_log::write(vml::app_log::Level::critical, "vml", "Uncaught exception: " + what);
    std::abort();
}

// Starts this session's log, headed with what the app runs on (the Python
// app's start_session_log).
void start_session_log() {
    if (!vml::app_log::start_logging()) {
        return;
    }
    // A crash ends the program before anything could reach the log: where it
    // happened goes to a file of its own beside it.
    vml::platform::report_crashes_to(vml::app_log::log_dir() / vml::app_log::crash_file_name);
    const vml::app_log::Logger log("vml.main");
    log.info("==== VideoMetricsLab" + std::string(vml::app_log::session_start));
    for (const std::string& line : vml::app_log::environment_lines(VML_APP_VERSION)) {
        log.info(line);
    }
    log.info(std::string("Qt ") + qVersion());
    qt_default_handler = qInstallMessageHandler(log_qt_message);
    std::set_terminate(log_uncaught_exception);
}

#ifdef _WIN32
// Direct3D 9, started on a thread of its own while the window is built. Qt
// starts it when it makes the window, to ask which display a GPU is on
// (QWindowsWindow::forcedScreenForGLWindow), though the window is drawn
// without the GPU; starting it loads the display driver and checks its
// signature, about half of the window thread's work as the window first
// shows (63 of 120 ms). Kept until the window has shown, so that the driver
// is still loaded for Qt.
class Direct3DWarmup {
public:
    Direct3DWarmup()
        : thread_([this] {
              using Create = IDirect3D9*(WINAPI*)(UINT);
              // From System32, where Qt loads it (QSystemLibrary): one beside the
              // app would be another library and warm nothing.
              HMODULE library = LoadLibraryExW(L"d3d9.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
              const auto create =
                  library != nullptr ? reinterpret_cast<Create>(GetProcAddress(library, "Direct3DCreate9")) : nullptr;
              if (create != nullptr) {
                  direct3d_ = create(D3D_SDK_VERSION);
                  if (direct3d_ != nullptr) {
                      D3DADAPTER_IDENTIFIER9 identifier{};
                      direct3d_->GetAdapterIdentifier(0, 0, &identifier);
                  }
              }
          }) {}
    ~Direct3DWarmup() {
        thread_.join();
        if (direct3d_ != nullptr) {
            direct3d_->Release();
        }
    }
    Direct3DWarmup(const Direct3DWarmup&) = delete;
    Direct3DWarmup& operator=(const Direct3DWarmup&) = delete;

private:
    IDirect3D9* direct3d_ = nullptr;
    std::thread thread_;  // last: it starts once the rest is set up
};
#endif

}  // namespace

int main(int argc, char** argv) {
    using namespace vml;
    const std::vector<std::string> words = platform::utf8_arguments(argc, argv);
    if (words.size() == 3 && words[0] == isolated_argument) {
        return run_isolated_task(words[1], platform::path_from_text(words[2]));
    }
    set_isolation_program(platform::executable_path());
#ifdef _WIN32
    auto direct3d_warmup = std::make_unique<Direct3DWarmup>();
#endif

    QApplication app(argc, argv);
    QApplication::setApplicationName("VideoMetricsLab");
#ifdef __APPLE__
    // Qt's Fusion style, as the window looks on Windows: macOS's own style
    // draws its controls larger, which squeezed the Videos tab's table to one
    // row and hid the metric headings' tick boxes. (Fusion is built into Qt
    // Widgets: no plugin.)
    QApplication::setStyle(QStringLiteral("Fusion"));
#endif
    if (std::find(words.begin(), words.end(), "--self-test") != words.end()) {
        return app::run_self_test(app, std::find(words.begin(), words.end(), "--quiet") != words.end());
    }
    start_session_log();
    // What runs ended outright (End task, a crash) left in the temporary
    // folder, removed on a thread of its own so that a large folder does not
    // hold up the start. Qt waits for it as the program ends.
    (void)QtConcurrent::run([] {
        std::error_code error;
        const std::filesystem::path folder = std::filesystem::temp_directory_path(error);
        if (!error) {
            remove_leftovers(folder);
        }
    });
    const Settings settings = Settings::load();
    apply_language(QString::fromStdString(settings.language));
    // Before the window is built, which reads it to colour itself.
    app::theme::apply_theme(QString::fromStdString(settings.theme));
    app_log::write(app_log::Level::info, "vml.main",
                   "Theme: " + (settings.theme.empty() ? "as " + system_name : settings.theme));
    set_gpu_backend(settings.gpu_backend);
    set_vship_backend(settings.gpu_backend);
    // The first probes load Vship and start the GPU's driver, which takes
    // seconds; started now, they are usually done before a run is asked for.
    // The window never waits for them (window_helpers.h), but a run does
    // (MainWindow::wait_for_gpu_checks). VMAF v1's (1.75 s on its own) too,
    // when it is calculated by default on the GPU.
    (void)QtConcurrent::run([] { detect_vship_device(); });
    (void)QtConcurrent::run([] { gpu_vmaf_backend(); });
    if (settings.default_compute_vmaf_v1 && settings.default_vmaf_on_gpu) {
        (void)QtConcurrent::run([] { v1_available(); });
    }

    app::MainWindow window;
    window.show();
#ifdef _WIN32
    direct3d_warmup.reset();
#endif
    window.check_for_updates();  // once, now, and at no other time
    return QApplication::exec();
}
