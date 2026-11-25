#include "results_export.h"

#include <QAction>
#include <QMenu>

#include <algorithm>
#include <fstream>
#include <set>

#include "dialogs.h"
#include "i18n.h"
#include "window_helpers.h"
#include "vml/build_info.h"
#include "vml/score_summary.h"
#include "vml/export_csv.h"
#include "vml/frame_pairing.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/results_summary.h"
#include "vml/temp_files.h"
#include "vml/weak_frames.h"

namespace vml::app {

namespace {

using Json = nlohmann::ordered_json;

// The metrics any of `results` has scores of, in the app's order.
std::vector<std::string> scored_keys(const std::vector<const ComparisonResult*>& results) {
    std::vector<std::string> keys;
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        if (std::any_of(results.begin(), results.end(), [&key](const ComparisonResult* result) {
                return score_samples(result->metric_results, key).has_value();
            })) {
            keys.push_back(key);
        }
    }
    return keys;
}

// A run's entry, as the command line's --json writes a video's.
Json run_entry(const ExportedRun& run) {
    const ComparisonResult& result = *run.result;
    Json entry{{"path", platform::path_text(result.distorted)},
               {"reference", platform::path_text(result.source)},
               {"label", run.label.toStdString()},
               {"status", run.failures.empty() ? "ok" : "partial"}};
    add_media(entry, result.distorted_info);
    if (!run.failures.empty()) {
        Json failed = Json::object();
        for (const auto& [key, why] : run.failures) {
            failed[key] = why.toStdString();
        }
        entry["failed_metrics"] = failed;
    }
    entry["compared_frames"] = result.compared_frame_count;
    entry["frame_sync"] = frame_pairing_json(result);
    entry["vmaf_model"] = result.model;
    entry["reference_crop"] = summary_crop_json(result.source_crop);
    entry["test_crop"] = summary_crop_json(result.distorted_crop);
    Json scored = Json::object();  // in the app's order
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        if (std::optional<Json> summary = metric_summary(result, key)) {
            scored[key] = std::move(*summary);
        }
    }
    entry["metrics"] = scored;
    // Another row's file scored another way: not a point of its own on a
    // BD-rate curve (read_rate_file), the file's own entry being that.
    entry["variant_of"] = run.variant_of ? Json(platform::path_text(*run.variant_of)) : Json(nullptr);
    if (!run.cvvdp_display.empty()) {
        entry["cvvdp_display"] = run.cvvdp_display;
    }
    const bool has_cvvdp = result.metric_results.sequence("cvvdp") != nullptr;
    if (has_cvvdp || !run.other_displays.empty()) {
        // Every display CVVDP was scored on, the first (metrics.cvvdp's) too:
        // a display that is no preset by its parameters alone (name "").
        Json displays = Json::array();
        if (has_cvvdp) {
            displays.push_back(cvvdp_display_entry(run.cvvdp_display, &result));
        }
        for (const ExportedDisplay& other : run.other_displays) {
            Json display = cvvdp_display_entry(other.name, other.result.get());
            // Its own run's times: the first display's are the video's
            Json record;
            add_video_record(record, other.facts.get(), nullptr, other.name);
            display["timings"] = record["timings"];
            displays.push_back(std::move(display));
        }
        entry["cvvdp_displays"] = displays;
    }
    add_video_record(entry, run.facts.get(), run.result.get(), run.cvvdp_display);
    return entry;
}

// The window's runs the exported videos came from, as one: the first's
// start, the last's end (none while one is still running), their times
// added up, and the latest finished one's machine (`machine` without one).
RunRecord runs_record(const std::vector<ExportedRun>& runs, const MachineFacts& machine) {
    std::set<const RunRecord*> records;
    for (const ExportedRun& run : runs) {
        if (run.record) {
            records.insert(run.record.get());
        }
    }
    RunRecord all{"gui", std::nullopt, "", "", std::nullopt, std::nullopt, static_cast<int>(records.size()),
                  machine};
    if (records.empty()) {
        return all;
    }
    const bool running = std::any_of(records.begin(), records.end(),
                                     [](const RunRecord* record) { return record->finished_utc.empty(); });
    double wall = 0.0;
    double paused = 0.0;
    std::string latest;  // the latest finished run's end
    for (const RunRecord* record : records) {
        // utc_text's times sort as text
        if (all.started_utc.empty() || record->started_utc < all.started_utc) {
            all.started_utc = record->started_utc;
        }
        if (!record->finished_utc.empty() && record->finished_utc > latest) {
            latest = record->finished_utc;
            all.machine = record->machine;
        }
        wall += record->wall_seconds.value_or(0.0);
        paused += record->paused_seconds.value_or(0.0);
    }
    if (!running) {
        all.finished_utc = latest;
        all.wall_seconds = wall;
        all.paused_seconds = paused;
    }
    return all;
}

}  // namespace

Json export_document(const std::optional<VideoInfo>& reference, const std::vector<ExportedRun>& runs,
                     const MachineFacts& machine) {
    Json source = nullptr;
    if (reference) {
        source = Json{{"path", platform::path_text(reference->path)}};
        add_media(source, *reference);
    }
    Json document{{"app", "VideoMetricsLab"}, {"url", project_url}, {"version", VML_VERSION}, {"commit", build_commit()}, {"format", results_format},
                  {"reference", source}, {"videos", Json::array()}};
    add_run_record(document, runs_record(runs, machine));
    for (const ExportedRun& run : runs) {
        document["videos"].push_back(run_entry(run));
    }
    return document;
}

QString export_contents(const std::vector<std::string>& items) {
    const auto has = [&items](std::string_view key) { return std::find(items.begin(), items.end(), key) != items.end(); };
    QStringList kinds;
    if (has("results")) {
        kinds << app::tr("results");
    }
    if (has("frames")) {
        kinds << app::tr("per-frame scores");
    }
    if (has("score_summary")) {
        kinds << app::tr("score summary");
    }
    if (has("graphs")) {
        kinds << app::tr("graphs");
    }
    if (has("worst_frames")) {
        kinds << app::tr("worst frames");
    }
    if (has("weak_segments")) {
        kinds << app::tr("weak segments");
    }
    if (kinds.size() <= 1) {
        return kinds.value(0);
    }
    const QString last = kinds.takeLast();
    return app::tr("{items} and {last}", {{"items", kinds.join(app::tr(", "))}, {"last", last}});
}

QString export_action_text(ExportScope scope, const std::vector<std::string>& items) {
    const QString contents = export_contents(items);
    if (contents.isEmpty()) {
        return app::tr("Export: nothing is ticked in Settings > Storage > Exports include");
    }
    switch (scope) {
    case ExportScope::this_video:
        return app::tr("Export this video's {contents}...", {{"contents", contents}});
    case ExportScope::all_videos:
        return app::tr("Export all videos' {contents}...", {{"contents", contents}});
    case ExportScope::selected_videos:
        return app::tr("Export the selected videos' {contents}...", {{"contents", contents}});
    }
    return contents;
}

QString export_item_title(const std::string& key) {
    if (key == "results") {
        return app::tr("Results");
    }
    if (key == "frames") {
        return app::tr("Per-frame scores");
    }
    if (key == "score_summary") {
        return app::tr("Score summary");
    }
    if (key == "graphs") {
        return app::tr("Graphs");
    }
    if (key == "worst_frames") {
        return app::tr("Worst frames");
    }
    if (key == "weak_segments") {
        return app::tr("Weak segments");
    }
    return QString::fromStdString(key);
}

QString nothing_scored_text() {
    return app::tr("No test video has been scored against the reference yet.");
}

QString no_conditions_text() {
    return app::tr("The profile picked has no conditions: set them in Settings > Weak segments, or pick a preset.");
}

QString one_reference_text() {
    return app::tr("Graphs and weak segments are exported for test videos of one reference: select comparisons that "
                   "share it.");
}

bool settled_for_export(const std::vector<std::string>& requested, const std::vector<std::string>& kept,
                        const std::vector<std::string>& failed) {
    const auto in = [](const std::vector<std::string>& keys, const std::string& key) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    };
    bool any = false;
    for (const std::string& key : requested) {
        if (in(kept, key)) {
            any = true;
        } else if (!in(failed, key)) {
            return false;
        }
    }
    return any;
}

std::vector<std::filesystem::path> write_loose_export(const std::filesystem::path& folder,
                                                      const std::vector<BundleFile>& files) {
    std::vector<std::string> names;
    for (const BundleFile& file : files) {
        names.push_back(file.first);
    }
    // Looked at as it is written: the writes run one after another, so an
    // export written before this one is on disk by now.
    std::set<std::filesystem::path> reserved;
    const std::vector<std::filesystem::path> paths = unique_output_names(folder, names, reserved);
    for (std::size_t i = 0; i < files.size(); ++i) {
        std::filesystem::create_directories(paths[i].parent_path());
        write_file_whole(paths[i], [&bytes = files[i].second](std::ostream& out) { out << bytes; });
    }
    return paths;
}

std::vector<std::string> shared_stems(const QString& name) {
    return {file_safe(name + QStringLiteral(" - results")).toStdString(),
            file_safe(name + QStringLiteral(" - weak segments")).toStdString()};
}

std::vector<BundleFile> in_video_folders(std::vector<BundleFile> files, const std::vector<QString>& names,
                                         const std::vector<std::string>& shared) {
    std::vector<std::string> prefixes;
    for (const QString& name : names) {
        prefixes.push_back(file_safe(name).toStdString());
    }
    std::sort(prefixes.begin(), prefixes.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    for (BundleFile& file : files) {
        if (std::any_of(shared.begin(), shared.end(),
                        [&file](const std::string& stem) { return file.first.starts_with(stem + "."); })) {
            continue;
        }
        for (const std::string& prefix : prefixes) {
            if (file.first.starts_with(prefix + " - ")) {
                file.first = prefix + "/" + file.first;
                break;
            }
        }
    }
    return files;
}

std::string text_lines(const std::string& text) {
#ifdef _WIN32
    std::string crlf;
    crlf.reserve(text.size() + text.size() / 32);
    for (const char c : text) {
        crlf += c == '\n' ? std::string("\r\n") : std::string(1, c);
    }
    return crlf;
#else
    return text;
#endif
}

std::vector<BundleFile> made_files(std::vector<BundleFile> files, const std::vector<FrameFiles>& frames) {
    for (const FrameFiles& video : frames) {
        if (video.worst_dips > 0) {
            WorstFramesReports reports =
                worst_frames_reports(video.result->metric_results, video.worst_dips, video.json, video.csv);
            if (video.json) {
                files.at(video.json_at).second = python_json_dumps(reports.json) + "\n";
            }
            if (video.csv) {
                files.at(video.csv_at).second = text_lines(reports.csv);
            }
            continue;
        }
        PerFrameTexts texts = per_frame_texts(*video.result, video.csv, video.json);
        if (video.json) {
            files.emplace_back(video.stem + ".json", std::move(texts.json));
        }
        if (video.csv) {
            files.emplace_back(video.stem + ".csv", std::move(texts.csv));
        }
    }
    return files;
}

std::vector<BundleFile> export_files(const Settings& settings, const std::vector<BundleVideo>& videos,
                                     const std::filesystem::path& reference,
                                     const std::function<Json()>& document, const SegmentRules& rules,
                                     bool several, const QString& name, std::vector<FrameFiles>& frames,
                                     bool& segments_skipped) {
    const std::vector<std::string> items = exported_items(settings);
    const auto wanted = [&items](std::string_view key) {
        return std::find(items.begin(), items.end(), key) != items.end();
    };
    // Results and weak segments in each format ticked.
    const std::vector<std::string>& formats = settings.export_formats;
    const bool json = std::find(formats.begin(), formats.end(), "json") != formats.end();
    const bool csv = std::find(formats.begin(), formats.end(), "csv") != formats.end();
    std::vector<BundleFile> files;
    frames.clear();
    segments_skipped = false;
    if (wanted("results")) {
        const Json made = document();
        const std::string stem = file_safe(name + QStringLiteral(" - results")).toStdString();
        if (json) {
            files.push_back({stem + ".json", python_json_dumps(made) + "\n"});
        }
        if (csv) {
            files.push_back({stem + ".csv", results_csv(made)});
        }
    }
    if (wanted("score_summary") && (json || csv) && !videos.empty()) {
        // Every video's scores in one file of each format, as the command
        // line's --score-summary writes them.
        std::vector<ScoreSummaryEntry> entries;
        for (const BundleVideo& video : videos) {
            entries.push_back({video.label.toStdString(), video.path, reference,
                               score_summary(*video.result, settings.cvvdp_presets_json)});
        }
        const std::string stem = file_safe(name + QStringLiteral(" - score summary")).toStdString();
        if (json) {
            files.push_back(
                {stem + ".json", python_json_dumps(score_summary_json(entries, settings.saved_details_columns)) + "\n"});
        }
        if (csv) {
            files.push_back({stem + ".csv", score_summary_csv(entries, settings.saved_details_columns)});
        }
    }
    if (wanted("frames") && (json || csv)) {
        // A file for each video, named as its graphs are.
        const std::vector<QString> names = bundle_names(videos);
        for (std::size_t i = 0; i < videos.size(); ++i) {
            frames.push_back(
                {videos[i].result, file_safe(names[i] + QStringLiteral(" - frames")).toStdString(), json, csv});
        }
    }
    if (wanted("worst_frames") && (json || csv)) {
        // Each video's, every metric's dips, as many as Worst Frames lists.
        const std::vector<QString> names = bundle_names(videos);
        const auto dips = static_cast<std::size_t>(std::max(settings.graph_worst_frames, 1));
        // Listed here, made as they are written (made_files): issue #140's
        // per-frame scores, for the same reason.
        for (std::size_t i = 0; i < videos.size(); ++i) {
            const std::string stem = file_safe(names[i] + QStringLiteral(" - worst frames")).toStdString();
            FrameFiles worst{videos[i].result, stem, json, csv, dips};
            if (json) {
                worst.json_at = files.size();
                files.push_back({stem + ".json", std::string()});
            }
            if (csv) {
                worst.csv_at = files.size();
                files.push_back({stem + ".csv", std::string()});
            }
            frames.push_back(std::move(worst));
        }
    }
    if (!wanted("weak_segments") || videos.empty()) {
        return files;
    }
    if (rules.conditions.empty()) {
        segments_skipped = true;
    } else if (several) {
        // One report of every video, as Export weak segments report saves it.
        std::vector<SegmentsReportVideo> report;
        std::vector<const ComparisonResult*> results;
        for (const BundleVideo& video : videos) {
            report.push_back({video.label.toStdString(), platform::path_text(video.path),
                              platform::path_text(reference), video.result.get()});
            results.push_back(video.result.get());
        }
        // None of the profile's conditions on a metric any of them has: one
        // starting condition for them all, as the command line's report
        // takes it (#221's review: the several videos' report had none).
        SegmentRules used = rules;
        const std::vector<std::string> keys = scored_keys(results);
        if (std::none_of(rules.conditions.begin(), rules.conditions.end(), [&keys](const WeakCondition& condition) {
                return std::find(keys.begin(), keys.end(), condition.key) != keys.end();
            })) {
            if (const MetricDefinition* fallback = starting_metric(keys, settings.graph_metric)) {
                used.conditions = {default_weak_condition(*fallback)};
            }
        }
        const std::string stem =
            (file_safe(name + QStringLiteral(" - weak segments")) + "." + qt_text(segments_file_suffix(used)))
                .toStdString();
        if (json) {
            files.push_back({stem + ".json", python_json_dumps(segments_report_json(report, used)) + "\n"});
        }
        if (csv) {
            files.push_back({stem + ".csv", text_lines(segments_report_csv(report, used))});
        }
    } else {
        // By the conditions on the video's metrics, as the Weak segments
        // dialog and the command line take them; the rest are said.
        const BundleVideo& video = videos.front();
        std::vector<WeakCondition> left_out;
        SegmentRules used = scored_rules(video.result->metric_results, rules, &left_out);
        if (used.conditions.empty()) {
            // None of the profile's on this video's metrics: one on the metric
            // Metric Graphs shows where it is scored, else on its first, as
            // the Worst Frames tab and the command line's --segments saved
            // start (default_weak_condition).
            if (const MetricDefinition* fallback =
                    starting_metric(scored_keys({video.result.get()}), settings.graph_metric)) {
                used.conditions = {default_weak_condition(*fallback)};
            }
        }
        const std::vector<WeakSegment> segments = weak_segments(*video.result, used);
        // Named after the metrics used; after the profile's where it has none.
        const std::string stem =
            suggested_file_name(video.label,
                                "." + qt_text(segments_file_suffix(used.conditions.empty() ? rules : used)))
                .toStdString();
        if (json) {
            files.push_back(
                {stem + ".json",
                 python_json_dumps(segments_file_json(*video.result, segments, used, left_out)) + "\n"});
        }
        if (csv) {
            files.push_back({stem + ".csv", text_lines(segments_csv(segments, used))});
        }
    }
    return files;
}

}  // namespace vml::app
