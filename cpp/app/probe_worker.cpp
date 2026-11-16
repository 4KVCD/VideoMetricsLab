#include "probe_worker.h"

#include <exception>
#include <utility>

#include "vml/crop_detect.h"
#include "vml/ffprobe.h"
#include "vml/film_grain_detect.h"
#include "vml/metric_cache.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml::app {

namespace {

void register_types() {
    qRegisterMetaType<vml::app::Probed>();
    qRegisterMetaType<vml::app::SavedScoresFound>();
    qRegisterMetaType<vml::app::OtherDisplaysFound>();
}

}  // namespace

ProbeWorker::ProbeWorker(QStringList paths, std::string ffprobe, std::string ffmpeg, StreamScan scan,
                         QObject* parent)
    : QThread(parent),
      paths_(std::move(paths)),
      ffprobe_(std::move(ffprobe)),
      ffmpeg_(std::move(ffmpeg)),
      scan_(scan) {
    register_types();
}

ProbeWorker::ProbeWorker(QStringList paths, std::filesystem::path base, std::filesystem::path source,
                         std::map<QString, SavedScoresQuery> queries, std::string ffmpeg, QObject* parent)
    : QThread(parent),
      paths_(std::move(paths)),
      ffmpeg_(std::move(ffmpeg)),
      base_(std::move(base)),
      source_(std::move(source)),
      queries_(std::move(queries)),
      looks_up_(true) {
    register_types();
}

void ProbeWorker::run() {
    if (looks_up_) {
        // Every row's scores first, read as saved where they wait for black
        // bars (5 to 10 s to find for a 4K video); then the bars, and those
        // rows again.
        std::vector<std::pair<QString, std::vector<std::string>>> waiting;
        for (const QString& path : paths_) {
            if (cancelled_) {
                break;
            }
            if (std::optional<std::vector<std::string>> sent = look_up(path)) {
                waiting.emplace_back(path, std::move(*sent));
            }
        }
        for (const auto& [path, sent] : waiting) {
            if (cancelled_) {
                break;
            }
            look_up_again(path, sent);
        }
        emit finished_all();
        return;
    }
    for (const QString& path : paths_) {
        if (cancelled_) {
            break;
        }
        Probed result{path, std::nullopt, QString()};
        try {
            result.info = probe_video(platform::path_from_text(path.toStdString()), ffprobe_, cancel_);
            if (result.info->is_raw_stream()) {
                result.info = raw_stream_as_found(*std::move(result.info), ffprobe_, cancel_);
            }
            if (!ffmpeg_.empty() &&
                (scan_ == StreamScan::every_codec || ascii_lower(result.info->codec_name) == "h264")) {
                const StreamStart start = scan_stream_start(*result.info, ffmpeg_, cancel_);
                result.info->film_grain = start.film_grain;
                result.info->monochrome_h264 = start.monochrome_h264;
            }
        } catch (const Cancelled&) {
            break;  // ended by cancel(): nothing to say about this video
        } catch (const std::exception& error) {
            // A background boundary: even an unexpected failure becomes the
            // row's error rather than ending the thread before it says so.
            result.error = QString::fromStdString(error.what());
        }
        emit probed(result);
    }
    emit finished_all();
}

std::optional<std::vector<std::string>> ProbeWorker::look_up(const QString& path) {
    // Where it waits, looked up again whatever it finds now (CVVDP's other
    // displays too), and whatever fails after its scores were sent
    std::optional<std::vector<std::string>> sent;
    try {
        const SavedScoresQuery& query = queries_.at(path);
        const AnalysisRequest& request = query.request;
        // The token first: a file replaced during a long read gives another,
        // and the window refuses the scores found under the old one.
        const std::string token = lookup_token(source_, query.identity, request);
        // Whether a saved comparison it reads waits for black bars: asked, not found yet
        const bool waits = find_black_bars_of_older_scores(base_, source_, query.identity, request.recipe,
                                                           [](const VideoInfo&) {});
        if (waits) {
            sent.emplace();
        }
        if (auto found = load_result(base_, source_, query.identity, request.recipe, request.metrics,
                                     request.perceptual_backends, query.supplemental, query.not_from_other_recipes,
                                     waits ? ScoresWaitingForBars::read : ScoresWaitingForBars::not_read)) {
            if (sent) {
                *sent = found->first.metric_results.keys();
            }
            emit saved_found({path, std::move(found->first), QString::fromStdString(found->second), token, {}, waits});
        }
        if (auto others = other_cvvdp_scores(base_, source_, query.identity, request.recipe, request.metrics,
                                             query.supplemental, query.not_from_other_recipes)) {
            emit other_displays_found({path, std::move(others->scores), std::move(others->parameters)});
        }
    } catch (const std::exception&) {
        // A damaged saved file must not end the lookup, leaving the videos
        // after this one without their saved scores.
    }
    return sent;
}

void ProbeWorker::look_up_again(const QString& path, const std::vector<std::string>& first) {
    try {
        const SavedScoresQuery& query = queries_.at(path);
        const AnalysisRequest& request = query.request;
        const std::string token = lookup_token(source_, query.identity, request);
        // Black bars an older rule found, found again where the rule keeps no
        // answer now, so that saved scores whose boxes come out the same stay:
        // once per video, taking what a run takes to find them (which a run
        // waiting for the lookup then has).
        find_black_bars_of_older_scores(base_, source_, query.identity, request.recipe, [this](const VideoInfo& video) {
            detect_crop(ffmpeg_, video, cancel_, std::nullopt, "");
        });
        SavedScoresFound again{path, {}, {}, token, first};
        if (auto found = load_result(base_, source_, query.identity, request.recipe, request.metrics,
                                     request.perceptual_backends, query.supplemental, query.not_from_other_recipes)) {
            again.result = std::move(found->first);
            again.label = QString::fromStdString(found->second);
        }
        emit saved_found(again);
        if (auto others = other_cvvdp_scores(base_, source_, query.identity, request.recipe, request.metrics,
                                             query.supplemental, query.not_from_other_recipes)) {
            emit other_displays_found({path, std::move(others->scores), std::move(others->parameters)});
        }
    } catch (const std::exception&) {
        // Cancelled, or a damaged saved file: the window drops the scores sent
        // first, as the lookup ends without this answer
        // (MainWindow::drop_unchecked_saved_scores)
    }
}

}  // namespace vml::app
