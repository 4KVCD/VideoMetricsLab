// VideoMetricsLab's window, in C++: the Python app's vmaf_app/main.py, as
// far as main_window.h says the window has been ported.
//
// This program is also the process of its own that native GPU code runs in
// (isolated.h), as the Python app's packaged program is: started with
// isolated_argument, it runs that task and makes no window.
#include <QApplication>
#include <QLibraryInfo>
#include <QTranslator>
#include <QtConcurrent/QtConcurrentRun>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "i18n.h"
#include "main_window.h"
#include "theme.h"
#include "vml/isolated.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/settings.h"
#include "vml/temp_files.h"
#include "vml/vship.h"

namespace {

// Shows the window in the `chosen` language, or Windows' display language
// when none is chosen, with English where there is no translation. Qt's own
// dialogs and buttons follow, in Qt's translations where it has them, and a
// right-to-left language mirrors the window.
void apply_language(QApplication& app, const QString& chosen) {
    const QString code = vml::app::set_language(chosen.isEmpty() ? vml::app::windows_language() : chosen);
    if (code == "en") {
        return;
    }
    auto* translator = new QTranslator(&app);
    if (translator->load("qt_" + code, QLibraryInfo::path(QLibraryInfo::TranslationsPath))) {
        QApplication::installTranslator(translator);
    }
    if (vml::app::right_to_left(code)) {
        QApplication::setLayoutDirection(Qt::RightToLeft);
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace vml;
    const std::vector<std::string> words = platform::utf8_arguments(argc, argv);
    if (words.size() == 3 && words[0] == isolated_argument) {
        return run_isolated_task(words[1], platform::path_from_text(words[2]));
    }
    set_isolation_program(platform::executable_path());

    QApplication app(argc, argv);
    QApplication::setApplicationName("VideoMetricsLab");
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
    apply_language(app, QString::fromStdString(settings.language));
    // Before the window is built, which reads it to colour itself.
    app::theme::apply_theme(QString::fromStdString(settings.theme));
    set_vship_backend(settings.gpu_backend);
    // The first probe loads Vship and starts the GPU's driver, which would
    // freeze the window for as long if it came when the first video is
    // added; started now, it is done by then.
    (void)QtConcurrent::run([] { detect_vship_device(); });

    app::MainWindow window;
    window.show();
    return QApplication::exec();
}
