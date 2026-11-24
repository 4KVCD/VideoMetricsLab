#include "ffmpeg_download_dialog.h"

#include <QNetworkAccessManager>
#include <QNetworkProxyFactory>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QThread>
#include <QUrl>
#include <QWidget>

#include <system_error>

#include "i18n.h"
#include "window_helpers.h"
#include "vml/ffmpeg_tools.h"
#include "vml/settings.h"

namespace vml::app {

namespace {

constexpr int release_timeout_seconds = 15;
// Without a byte for this long, a download is given up (Qt's transfer
// timeout counts from the last data received).
constexpr int download_timeout_seconds = 60;
// Room the download needs: its zip, and ffmpeg.exe and ffprobe.exe
// unpacked from it, each about as big as the zip.
constexpr long long space_factor = 3;
constexpr long long mb = 1024 * 1024;

QString manual_ways() {
    return app::tr("\n\nYou can also install FFmpeg yourself (winget install Gyan.FFmpeg) and click \"Locate "
                   "ffmpeg.exe...\".");
}

}  // namespace

std::filesystem::path downloaded_ffmpeg_folder() {
    return Settings::file().parent_path() / "ffmpeg";
}

FfmpegDownload::FfmpegDownload(std::filesystem::path folder, QWidget* parent)
    : QObject(parent), folder_(std::move(folder)), parent_(parent) {}

FfmpegDownload::~FfmpegDownload() {
    // A Cancel or the window closing while it unpacks: on_unpacked, which
    // would tidy up, never runs once this is gone.
    cancel_.cancel();
    if (unpacker_ != nullptr) {
        unpacker_->wait();
        delete unpacker_;
        remove_parts();
    }
}

void FfmpegDownload::start() {
    remove_parts();  // a crashed download's
    progress_ = new ProgressDialog(parent_, app::tr("Download FFmpeg"), app::tr("Looking for the latest FFmpeg build..."));
    progress_->setWindowModality(Qt::WindowModal);
    connect(progress_, &QDialog::rejected, this, [this] { cancel(); });  // Cancel, Esc or its close button
    progress_->show();
    // Through the system's proxy, if it names one, as the update check.
    QNetworkProxyFactory::setUseSystemConfiguration(true);
    network_ = new QNetworkAccessManager(this);
    reply_ = get(std::string(ffmpeg_builds_release_url), release_timeout_seconds);
    connect(reply_, &QNetworkReply::finished, this, [this, reply = reply_] { on_release(reply); });
}

QNetworkReply* FfmpegDownload::get(const std::string& url, int timeout_seconds) {
    QNetworkRequest request(QUrl(QString::fromStdString(url)));
    if (url.starts_with("https://api.github.com/")) {
        request.setRawHeader("Accept", "application/vnd.github+json");
    }
    // GitHub refuses requests without a User-Agent; its files are served
    // from another host it redirects to.
    request.setHeader(QNetworkRequest::UserAgentHeader, "VideoMetricsLab/" VML_VERSION);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    request.setTransferTimeout(timeout_seconds * 1000);
    return network_->get(request);
}

QString FfmpegDownload::reply_problem(QNetworkReply* reply) {
    if (reply->error() == QNetworkReply::NoError) {
        return {};
    }
    const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    if (status == 403 || status == 429) {
        // GitHub's limit on requests without an account, per network.
        return app::tr("GitHub refused for now: too many requests from this network. Try again in an hour.");
    }
    return app::tr("GitHub could not be reached: {error}", {{"error", reply->errorString()}});
}

void FfmpegDownload::on_release(QNetworkReply* reply) {
    reply->deleteLater();
    reply_ = nullptr;
    if (done_) {
        return;
    }
    if (const QString problem = reply_problem(reply); !problem.isEmpty()) {
        fail(problem);
        return;
    }
    const std::optional<FfmpegBuild> build = choose_ffmpeg_build(reply->readAll().toStdString());
    if (!build) {
        fail(app::tr("GitHub lists no Windows build of FFmpeg {version} or newer.",
                     {{"version", QString::fromStdString(format_version(minimum_ffmpeg_version))}}));
        return;
    }
    build_ = *build;
    std::error_code error;
    std::filesystem::create_directories(folder_, error);
    const std::filesystem::space_info space = std::filesystem::space(folder_, error);
    const long long needed = build_.size * space_factor;
    if (!error && static_cast<long long>(space.available) < needed) {
        fail(app::tr("Not enough free space in {folder}: {needed} MB are needed, {free} MB are free.",
                     {{"folder", text_of(folder_)},
                      {"needed", needed / mb},
                      {"free", static_cast<long long>(space.available) / mb}}));
        return;
    }
    if (!build_.sha256.empty()) {
        download_build();
    } else if (build_.checksums_url.empty()) {
        fail(app::tr("The release gives no checksum to check the download against."));
    } else {
        reply_ = get(build_.checksums_url, release_timeout_seconds);
        connect(reply_, &QNetworkReply::finished, this, [this, reply = reply_] { on_checksums(reply); });
    }
}

void FfmpegDownload::on_checksums(QNetworkReply* reply) {
    reply->deleteLater();
    reply_ = nullptr;
    if (done_) {
        return;
    }
    if (const QString problem = reply_problem(reply); !problem.isEmpty()) {
        fail(problem);
        return;
    }
    const std::optional<std::string> sha256 = checksum_from_list(reply->readAll().toStdString(), build_.name);
    if (!sha256) {
        fail(app::tr("The release gives no checksum to check the download against."));
        return;
    }
    build_.sha256 = *sha256;
    download_build();
}

void FfmpegDownload::download_build() {
    zip_.setFileName(text_of(folder_ / "download.zip.part"));
    if (!zip_.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        fail(app::tr("Could not write {path}", {{"path", zip_.fileName()}}));
        return;
    }
    progress_->set_label(app::tr("Downloading FFmpeg {version} ({size} MB)...",
                                    {{"version", QString::fromStdString(build_.version)}, {"size", build_.size / mb}}));
    progress_->set_range(0, static_cast<int>(build_.size / 1024));  // KB: an int holds them
    reply_ = get(build_.url, download_timeout_seconds);
    // Written as it comes, never the whole file in memory.
    connect(reply_, &QNetworkReply::readyRead, this, [this] { on_build_data(); });
    connect(reply_, &QNetworkReply::downloadProgress, this, [this](qint64 received, qint64 /*total*/) {
        if (progress_ != nullptr) {
            progress_->set_value(static_cast<int>(received / 1024));
        }
    });
    connect(reply_, &QNetworkReply::finished, this, [this] { on_build_finished(); });
}

void FfmpegDownload::on_build_data() {
    if (done_ || reply_ == nullptr) {
        return;
    }
    const QByteArray data = reply_->readAll();
    hash_.update(data.constData(), static_cast<std::size_t>(data.size()));
    if (zip_.write(data) != data.size()) {
        fail(app::tr("Could not write {path}", {{"path", zip_.fileName()}}));
    }
}

void FfmpegDownload::on_build_finished() {
    on_build_data();  // what came with the end
    QNetworkReply* reply = reply_;
    reply_ = nullptr;
    reply->deleteLater();
    if (done_) {
        return;
    }
    zip_.close();
    if (const QString problem = reply_problem(reply); !problem.isEmpty()) {
        fail(problem);
        return;
    }
    if (hash_.hexdigest() != build_.sha256) {
        fail(app::tr("The download does not match its checksum: it was damaged, or the build changed while it "
                     "downloaded. Try again."));
        return;
    }
    unpack();
}

void FfmpegDownload::unpack() {
    progress_->set_label(app::tr("Unpacking FFmpeg..."));
    progress_->set_range(0, 0);
    // A few seconds of decompressing, off the window's thread.
    unpacker_ = QThread::create([this, zip = folder_ / "download.zip.part"] {
        try {
            unpack_ffmpeg(zip, folder_, cancel_);
        } catch (const Cancelled&) {
            // cancel() has said so
        } catch (const std::exception& error) {
            unpack_error_ = error.what();
        }
    });
    connect(unpacker_, &QThread::finished, this, [this] { on_unpacked(); });
    unpacker_->start();
}

void FfmpegDownload::on_unpacked() {
    unpacker_->deleteLater();
    unpacker_ = nullptr;
    remove_parts();  // the zip, and what a failure left
    if (done_) {
        return;
    }
    if (!unpack_error_.empty()) {
        fail(QString::fromStdString(unpack_error_));
        return;
    }
    done_ = true;
    delete progress_;
    progress_ = nullptr;
    emit downloaded(text_of(folder_));
}

void FfmpegDownload::cancel() {
    if (done_) {
        return;
    }
    done_ = true;
    cancel_.cancel();
    if (reply_ != nullptr) {
        reply_->abort();
    }
    zip_.close();
    if (unpacker_ == nullptr) {
        remove_parts();  // else when it has stopped
    }
    progress_->deleteLater();
    progress_ = nullptr;
    emit cancelled();
}

void FfmpegDownload::fail(const QString& message) {
    if (done_) {
        return;
    }
    done_ = true;
    if (reply_ != nullptr) {
        reply_->abort();
    }
    zip_.close();
    if (unpacker_ == nullptr) {
        remove_parts();
    }
    if (progress_ != nullptr) {
        progress_->disconnect(this);  // its closing is not a Cancel
        progress_->deleteLater();
        progress_ = nullptr;
    }
    emit failed(message + manual_ways());
}

void FfmpegDownload::remove_parts() const {
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(folder_, error)) {
        if (entry.path().extension() == ".part") {
            std::filesystem::remove(entry.path(), error);
        }
    }
}

}  // namespace vml::app
