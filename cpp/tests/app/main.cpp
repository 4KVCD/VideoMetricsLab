// The window's tests run with no screen (Qt's offscreen platform), as the
// Python app's do; with VML_TEST_ON_SCREEN=1, on the display, for the tests
// of what only shows there (native playback, which is off offscreen). A dialog that opens would wait for a click that never
// comes: it fails the test instead, and is closed so the tests go on. No
// test keeps the PC awake: a run's request to is a stand-in's.
#include <QApplication>
#include <QTimer>
#include <QWidget>

#include <gtest/gtest.h>

#include <clocale>

#include "main_window.h"
#include "vml/isolated.h"
#include "vml/platform/system.h"

int main(int argc, char** argv) {
    if (qgetenv("VML_TEST_ON_SCREEN") != "1") {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    // Native GPU code (the GPU's probe for CVVDP, VMAF on the GPU) runs where
    // it does in the app: in a process of its own.
    vml::set_isolation_program(vml::platform::path_from_text(VML_CLI));
    vml::app::set_keep_awake([](bool /*awake*/) { return false; });
    QApplication app(argc, argv);
#ifndef _WIN32
    std::setlocale(LC_NUMERIC, "C");  // as the app does (app/main.cpp)
#endif
    ::testing::InitGoogleTest(&argc, argv);
    QTimer dialogs;
    QObject::connect(&dialogs, &QTimer::timeout, [] {
        if (QWidget* dialog = QApplication::activeModalWidget()) {
            ADD_FAILURE() << "A dialog opened: " << dialog->windowTitle().toStdString();
            dialog->close();
        }
    });
    dialogs.start(100);
    return RUN_ALL_TESTS();
}
