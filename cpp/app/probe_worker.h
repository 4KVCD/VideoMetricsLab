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
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
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
// it can show (displayable_metric_specs), and the metrics never to take from
// another recipe's saved scores (RowData::forgotten_metrics).
struct SavedScoresQuery {
    std::filesystem::path identity;
    AnalysisRequest request;
    std::vector<MetricRequestSpec> supplemental;
    std::set<std::string> not_from_other_recipes;
};

// A video's rate, from a stretch of its packets (sampled_video_rate): the
// details it was read with, of which only the rate is to be taken
// (VideoInfo::take_bit_rate).
struct RateEstimated {
    VideoInfo estimate;
};

// The saved scores found for a row, with the token of the request they were
// found for (metric_cache.h's lookup_token).
struct SavedScoresFound {
    QString path;
    ComparisonResult result;
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
    // Which videos' first frames are read (scan_stream_start): every one's,
    // for a video added; only H.264's, for a saved run's details read again
    // (take_read_details), which take only whether it is monochrome.
    enum class StreamScan { every_codec, h264_only };

    // Reads the videos with `ffprobe` (find_tool's answer), and their first
    // frames (their film grain, whether H.264 is monochrome) with `ffmpeg`, as
    // `scan` says; with no `ffmpeg`, those are left unread.
    ProbeWorker(QStringList paths, std::string ffprobe, std::string ffmpeg, StreamScan scan,
                QObject* parent = nullptr);
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

    // Marked by the window as it takes lane_probed, which comes after every
    // video's probed: from then on the worker only samples rates, which
    // nothing waits for. The window's thread only.
    void set_lane_taken() { lane_taken_ = true; }
    bool lane_taken() const { return lane_taken_; }

signals:
    void probed(const vml::app::Probed& probed);
    // Only for a row that has saved scores: a miss is the usual case.
    void saved_found(const vml::app::SavedScoresFound& found);
    // For every row whose lookup includes CVVDP, so that none found clears
    // what was found before.
    void other_displays_found(const vml::app::OtherDisplaysFound& found);
    // Every video has been read: what waits for the videos (the saved scores'
    // lookup, "Reading..." taken down) need not wait for their rates.
    void lane_probed();
    // After lane_probed, for each video that states no rate of its own (only
    // with StreamScan::every_codec; none for a file changed since it was read).
    void rate_estimated(const vml::app::RateEstimated& estimated);
    void finished_all();

protected:
    void run() override;

private:
    // A video whose rate is sampled once every video has been read, with the
    // size and time its file had when it was read.
    struct RateToSample {
        std::filesystem::path file;
        std::pair<std::uintmax_t, std::optional<long long>> stamp;
    };

    void look_up(const QString& path);
    void estimate_rates(const std::vector<RateToSample>& videos);

    QStringList paths_;
    std::string ffprobe_;
    std::string ffmpeg_;
    StreamScan scan_ = StreamScan::every_codec;
    // The lookup's, when it looks up saved scores.
    std::filesystem::path base_;
    std::filesystem::path source_;
    std::map<QString, SavedScoresQuery> queries_;
    bool looks_up_ = false;
    std::atomic<bool> cancelled_ = false;
    bool lane_taken_ = false;
    CancelToken cancel_;
};

}  // namespace vml::app

Q_DECLARE_METATYPE(vml::app::Probed)
Q_DECLARE_METATYPE(vml::app::RateEstimated)
Q_DECLARE_METATYPE(vml::app::SavedScoresFound)
Q_DECLARE_METATYPE(vml::app::OtherDisplaysFound)
