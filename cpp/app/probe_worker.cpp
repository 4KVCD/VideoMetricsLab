#include "probe_worker.h"

#include <exception>
#include <utility>

#include "vml/ffprobe.h"
#include "vml/metric_cache.h"
#include "vml/platform/system.h"

namespace vml::app {

namespace {

void register_types() {
    qRegisterMetaType<vml::app::Probed>();
    qRegisterMetaType<vml::app::SavedScoresFound>();
    qRegisterMetaType<vml::app::OtherDisplaysFound>();
}

}  // namespace

ProbeWorker::ProbeWorker(QStringList paths, std::string ffprobe, QObject* parent)
    : QThread(parent), paths_(std::move(paths)), ffprobe_(std::move(ffprobe)) {
    register_types();
}

ProbeWorker::ProbeWorker(QStringList paths, std::filesystem::path base, std::filesystem::path source,
                         std::map<QString, SavedScoresQuery> queries, QObject* parent)
    : QThread(parent),
      paths_(std::move(paths)),
      base_(std::move(base)),
      source_(std::move(source)),
      queries_(std::move(queries)),
      looks_up_(true) {
    register_types();
}

void ProbeWorker::run() {
    for (const QString& path : paths_) {
        if (cancelled_) {
            break;
        }
        if (looks_up_) {
            look_up(path);
            continue;
        }
        Probed result{path, std::nullopt, QString()};
        try {
            result.info = probe_video(platform::path_from_text(path.toStdString()), ffprobe_, cancel_);
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

void ProbeWorker::look_up(const QString& path) {
    try {
        const SavedScoresQuery& query = queries_.at(path);
        const AnalysisRequest& request = query.request;
        // The token first: a file replaced during a long read gives another,
        // and the window refuses the scores found under the old one.
        const std::string token = lookup_token(source_, query.identity, request);
        if (auto found = load_result(base_, source_, query.identity, request.recipe, request.metrics,
                                     request.perceptual_backends, query.supplemental)) {
            emit saved_found({path, std::move(found->first), QString::fromStdString(found->second), token});
        }
        if (auto others = other_cvvdp_scores(base_, source_, query.identity, request.recipe, request.metrics,
                                             query.supplemental)) {
            emit other_displays_found({path, std::move(others->scores), std::move(others->parameters)});
        }
    } catch (const std::exception&) {
        // A damaged saved file must not end the lookup, leaving the videos
        // after this one without their saved scores.
    }
}

}  // namespace vml::app
