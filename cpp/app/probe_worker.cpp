#include "probe_worker.h"

#include <exception>
#include <utility>

#include "vml/ffprobe.h"
#include "vml/platform/system.h"

namespace vml::app {

ProbeWorker::ProbeWorker(QStringList paths, std::string ffprobe, QObject* parent)
    : QThread(parent), paths_(std::move(paths)), ffprobe_(std::move(ffprobe)) {
    qRegisterMetaType<vml::app::Probed>();
}

void ProbeWorker::run() {
    for (const QString& path : paths_) {
        if (cancelled_) {
            break;
        }
        Probed result{path, std::nullopt, QString()};
        try {
            result.info = probe_video(platform::path_from_text(path.toStdString()), ffprobe_);
        } catch (const std::exception& error) {
            // A background boundary: even an unexpected failure becomes the
            // row's error rather than ending the thread before it says so.
            result.error = QString::fromStdString(error.what());
        }
        emit probed(result);
    }
    emit finished_all();
}

}  // namespace vml::app
