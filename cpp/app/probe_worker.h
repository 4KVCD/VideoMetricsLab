// Reads newly added videos, and looks up their saved scores, off the window's
// thread (the Python app's vmaf_app/ui/probe_worker.py). Both are slow
// enough to be felt: ffprobe is a process to start, about 70 ms for a small
// file and more for a large one on a slow disk, then for an H.264, HEVC, VVC
// or AV1 video FFmpeg reads its first frames for film grain (50-400 ms), and
// a feature-length video's saved scores are several MB to read. Eight videos
// done on the window's thread would freeze it for a second or more.
#pragma once

#include <QMetaType>
#include <QString>
#include <QStringList>
#include <QThread>

#include <atomic>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/analysis_request.h"
#include "vml/cancel.h"
#include "vml/cvvdp.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml::app {

// One video read: its information, or why it could not be read.
struct Probed {
    QString path;
    std::optional<VideoInfo> info;
    QString error;
};

// What to look up for a row in the saved scores: the file its comparison is
// filed under (RowData::identity_path), its request, and every other score
// it can show (displayable_metric_specs).
struct SavedScoresQuery {
    std::filesystem::path identity;
    AnalysisRequest request;
    std::vector<MetricRequestSpec> supplemental;
};

// The saved scores found for a row, with the result's label and the token
// of the request they were found for (metric_cache.h's lookup_token).
struct SavedScoresFound {
    QString path;
    ComparisonResult result;
    QString label;
    std::string token;
};

// CVVDP's scores saved for a row's comparison with other displays, and the
// CVVDP parameters of the request they were looked up for.
struct OtherDisplaysFound {
    QString path;
    std::vector<std::pair<CvvdpSettings, double>> scores;
    nlohmann::json parameters;
};

// Reads each video in turn, or looks up each one's saved scores, and says
// what it found for each, in the order they were given.
class ProbeWorker : public QThread {
    Q_OBJECT

public:
    // Reads the videos with `ffprobe` (find_tool's answer), and the film
    // grain their streams describe with `ffmpeg` (detect_film_grain); with
    // no `ffmpeg`, the film grain is left unread.
    ProbeWorker(QStringList paths, std::string ffprobe, std::string ffmpeg, QObject* parent = nullptr);
    // Looks up the saved scores of the rows `paths` (as `queries` asks for
    // each) under `base`, compared with the reference `source`.
    ProbeWorker(QStringList paths, std::filesystem::path base, std::filesystem::path source,
                std::map<QString, SavedScoresQuery> queries, QObject* parent = nullptr);

    // Stops, ending the ffprobe or FFmpeg reading a video; a lookup under way
    // is let finish.
    void cancel() {
        cancelled_ = true;
        cancel_.cancel();
    }

signals:
    void probed(const vml::app::Probed& probed);
    // Only for a row that has saved scores: a miss is the usual case.
    void saved_found(const vml::app::SavedScoresFound& found);
    // For every row whose lookup includes CVVDP, so that none found clears
    // what was found before.
    void other_displays_found(const vml::app::OtherDisplaysFound& found);
    void finished_all();

protected:
    void run() override;

private:
    void look_up(const QString& path);

    QStringList paths_;
    std::string ffprobe_;
    std::string ffmpeg_;
    // The lookup's, when it looks up saved scores.
    std::filesystem::path base_;
    std::filesystem::path source_;
    std::map<QString, SavedScoresQuery> queries_;
    bool looks_up_ = false;
    std::atomic<bool> cancelled_ = false;
    CancelToken cancel_;
};

}  // namespace vml::app

Q_DECLARE_METATYPE(vml::app::Probed)
Q_DECLARE_METATYPE(vml::app::SavedScoresFound)
Q_DECLARE_METATYPE(vml::app::OtherDisplaysFound)
