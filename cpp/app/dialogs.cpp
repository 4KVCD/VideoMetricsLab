#include "dialogs.h"

#include <QFileDialog>
#include <QMessageBox>

namespace vml::app {

Dialogs standard_dialogs(QWidget* parent) {
    return {
        [parent](const QString& title, const QString& text) { QMessageBox::information(parent, title, text); },
        [parent](const QString& title, const QString& text) { QMessageBox::critical(parent, title, text); },
        [parent](const QString& title, const QString& text) {
            return QMessageBox::question(parent, title, text) == QMessageBox::Yes;
        },
        [parent](const QString& title, const QString& filter) {
            return QFileDialog::getOpenFileName(parent, title, QString(), filter);
        },
        [parent](const QString& title, const QString& suggested, const QString& filter) {
            return QFileDialog::getSaveFileName(parent, title, suggested, filter);
        },
        [parent](const QString& title) { return QFileDialog::getExistingDirectory(parent, title); },
    };
}

}  // namespace vml::app
