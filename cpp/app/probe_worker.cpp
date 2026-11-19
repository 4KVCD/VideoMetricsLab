#include "probe_worker.h"

#include <exception>
#include <filesystem>
#include <system_error>
#include <utility>

#include "vml/ffprobe.h"
#include "vml/film_grain_detect.h"
#include "vml/metric_cache.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml::app {

namespace {

// A file's size and time written, which change when it is replaced.
std::pair<std::uintmax_t, std::optional<long long>> file_stamp(const std::filesystem::path& file) {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(file, error);
    return {error ? 0 : size, platform::modified_ns(file)};
}

void register_types() {
    qRegisterMetaType<vml::app::Probed>();
    qRegisterMetaType<vml::app::RateEstimated>();
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
                         std::map<QString, SavedScoresQuery> queries, QObject* parent, WholeStreams whole_streams)
    : QThread(parent),
      paths_(std::move(paths)),
      base_(std::move(base)),
      source_(std::move(source)),
      queries_(std::move(queries)),
      looks_up_(true) {
    whole_streams_ = whole_streams == WholeStreams::yes;
    register_types();
}

void ProbeWorker::run() {
    std::vector<RateToSample> rates_to_sample;
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
            const std::filesystem::path file = platform::path_from_text(path.toStdString());
            const auto stamp = file_stamp(file);
            // The rate's sample waits until every video has been read: a
            // row's details, and the saved scores' lookup, do not wait for it.
            result.info = probe_video(file, ffprobe_, cancel_, RateSample::later);
            if (result.info->is_raw_stream()) {
                result.info = raw_stream_as_found(*std::move(result.info), ffprobe_, cancel_);
            }
            if (scan_ == StreamScan::every_codec && needs_rate_sample(*result.info)) {
                rates_to_sample.push_back({file, stamp});
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
    if (!looks_up_) {
        emit lane_probed();
        estimate_rates(rates_to_sample);
    }
    emit finished_all();
}

void ProbeWorker::estimate_rates(const std::vector<RateToSample>& videos) {
    for (const RateToSample& video : videos) {
        if (cancelled_) {
            return;
        }
        try {
            std::optional<VideoInfo> estimate = sampled_video_rate(video.file, ffprobe_, cancel_);
            if (estimate && file_stamp(video.file) == video.stamp) {
                emit rate_estimated({*std::move(estimate)});
            }
        } catch (const Cancelled&) {
            return;
        } catch (const std::exception&) {
            // A background boundary: the video keeps the rate it was read with.
        }
    }
}

void ProbeWorker::look_up(const QString& path) {
    try {
        const SavedScoresQuery& query = queries_.at(path);
        const AnalysisRequest& request = query.request;
        // The token first: a file replaced during a long read gives another,
        // and the window refuses the scores found under the old one.
        const std::string token = lookup_token(source_, query.identity, request);
        // A remux's test stream read to know it by stops with the lookup; a
        // whole one is read only where this lookup is for that.
        const StreamReads stream_reads(cancel_, whole_streams_,
                                       [this, &path](const std::filesystem::path&) { emit stream_check_started(path); });
        take_whole_stream_wanted();  // none of an earlier row's
        if (auto found = load_result(base_, source_, query.identity, request.recipe, request.metrics,
                                     request.perceptual_backends, query.supplemental, query.not_from_other_recipes)) {
            emit saved_found({path, std::move(found->first), token});
        }
        if (auto others = other_cvvdp_scores(base_, source_, query.identity, request.recipe, request.metrics,
                                             query.supplemental, query.not_from_other_recipes)) {
            emit other_displays_found({path, std::move(others->scores), std::move(others->parameters)});
        }
        if (take_whole_stream_wanted() && !cancelled_) {
            emit stream_check_wanted(path);
        }
    } catch (const std::exception&) {
        // A damaged saved file must not end the lookup, leaving the videos
        // after this one without their saved scores.
    }
}

}  // namespace vml::app
