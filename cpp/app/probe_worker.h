// Reads newly added videos off the window's thread (the Python app's
// vmaf_app/ui/probe_worker.py, its media half): ffprobe is a process to
// start, about 70 ms for a small file and more for a large one on a slow
// disk, so eight videos read on the window's thread would freeze it for a
// second or more.
#pragma once

#include <QMetaType>
#include <QString>
#include <QStringList>
#include <QThread>

#include <atomic>
#include <optional>

#include "vml/video.h"

namespace vml::app {

// One video read: its information, or why it could not be read.
struct Probed {
    QString path;
    std::optional<VideoInfo> info;
    QString error;
};

// Reads each video in turn, and says what it found for each, in the order
// they were given.
class ProbeWorker : public QThread {
    Q_OBJECT

public:
    // `ffprobe`: the program to read them with (find_tool's answer).
    ProbeWorker(QStringList paths, std::string ffprobe, QObject* parent = nullptr);

    // Stops after the video being read.
    void cancel() { cancelled_ = true; }

signals:
    void probed(const vml::app::Probed& probed);
    void finished_all();

protected:
    void run() override;

private:
    QStringList paths_;
    std::string ffprobe_;
    std::atomic<bool> cancelled_ = false;
};

}  // namespace vml::app

Q_DECLARE_METATYPE(vml::app::Probed)
