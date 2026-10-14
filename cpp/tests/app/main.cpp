// The window's tests run with no screen (Qt's offscreen platform), as the
// Python app's do. A dialog that opens would wait for a click that never
// comes: it fails the test instead, and is closed so the tests go on.
#include <QApplication>
#include <QTimer>
#include <QWidget>

#include <gtest/gtest.h>

#include "vml/isolated.h"
#include "vml/platform/system.h"

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    // Native GPU code (the GPU's probe for CVVDP) runs where it does in the
    // app: in a process of its own.
    vml::set_isolation_program(vml::platform::path_from_text(VML_CLI));
    QApplication app(argc, argv);
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
