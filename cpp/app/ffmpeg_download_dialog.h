// Downloading FFmpeg when the app cannot find it (Windows; as FastFlix
// does): asks GitHub for FFmpeg-Builds' latest release, downloads its
// stable Windows build with a progress dialog, checks its SHA-256 and
// unpacks ffmpeg.exe and ffprobe.exe into a folder of the app's own
// (ffmpeg_download.h). The window then checks them and uses the folder.
#pragma once

#include <QFile>
#include <QObject>
#include <QString>

#include <filesystem>
#include <string>

#include "app_dialog.h"
#include "vml/cancel.h"
#include "vml/ffmpeg_download.h"
#include "vml/sha256.h"

class QNetworkAccessManager;
class QNetworkReply;
class QThread;
class QWidget;

namespace vml::app {

// The folder downloaded FFmpeg goes into: beside the app's settings, never
// another program's.
std::filesystem::path downloaded_ffmpeg_folder();

class FfmpegDownload : public QObject {
    Q_OBJECT

public:
    // Into `folder`, with its progress dialog over `parent`.
    FfmpegDownload(std::filesystem::path folder, QWidget* parent);
    ~FfmpegDownload() override;
    void start();

signals:
    // ffmpeg.exe and ffprobe.exe are in `folder` (not yet checked).
    void downloaded(const QString& folder);
    // Why it failed, for the user, with what else they can do.
    void failed(const QString& message);
    void cancelled();

private:
    void on_release(QNetworkReply* reply);
    void on_checksums(QNetworkReply* reply);
    void download_build();
    void on_build_data();
    void on_build_finished();
    void unpack();
    void on_unpacked();
    void cancel();
    void fail(const QString& message);
    // The *.part files a download leaves while it runs, or after a crash.
    void remove_parts() const;
    QNetworkReply* get(const std::string& url, int timeout_seconds);
    // Why a reply failed, for the user; empty if it did not.
    static QString reply_problem(QNetworkReply* reply);

    std::filesystem::path folder_;
    QWidget* parent_;
    QNetworkAccessManager* network_ = nullptr;
    ProgressDialog* progress_ = nullptr;
    QNetworkReply* reply_ = nullptr;
    FfmpegBuild build_;
    QFile zip_;
    Sha256 hash_;
    QThread* unpacker_ = nullptr;
    std::string unpack_error_;
    CancelToken cancel_;
    bool done_ = false;
};

}  // namespace vml::app
