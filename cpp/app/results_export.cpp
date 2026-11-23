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
    Json document{{"app", "VideoMetricsLab"}, {"version", VML_VERSION}, {"commit", build_commit()}, {"format", results_format},
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
        for (std::size_t i = 0; i < videos.size(); ++i) {
            const std::string stem = file_safe(names[i] + QStringLiteral(" - worst frames")).toStdString();
            const MetricResultSet& results = videos[i].result->metric_results;
            if (json) {
                files.push_back({stem + ".json", python_json_dumps(worst_frames_report_json(results, dips)) + "\n"});
            }
            if (csv) {
                files.push_back({stem + ".csv", text_lines(worst_frames_report_csv(results, dips))});
            }
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
        for (const BundleVideo& video : videos) {
            report.push_back({video.label.toStdString(), platform::path_text(video.path),
                              platform::path_text(reference), video.result.get()});
        }
        const std::string stem =
            (file_safe(name + QStringLiteral(" - weak segments")) + "." + qt_text(segments_file_suffix(rules)))
                .toStdString();
        if (json) {
            files.push_back({stem + ".json", python_json_dumps(segments_report_json(report, rules)) + "\n"});
        }
        if (csv) {
            files.push_back({stem + ".csv", text_lines(segments_report_csv(report, rules))});
        }
    } else {
        // By the conditions on the video's metrics, as the Weak segments
        // dialog and the command line take them; the rest are said.
        const BundleVideo& video = videos.front();
        std::vector<WeakCondition> left_out;
        const SegmentRules used = scored_rules(video.result->metric_results, rules, &left_out);
        const std::vector<WeakSegment> segments = weak_segments(*video.result, used);
        // Named after the metrics used; after the profile's where it has none.
        const std::string stem =
            suggested_file_name(video.label,
                                "." + qt_text(segments_file_suffix(used.conditions.empty() ? rules : used)))
                .toStdString();
        if (json) {
            files.push_back({stem + ".json", python_json_dumps(segments_file_json(segments, used, left_out)) + "\n"});
        }
        if (csv) {
            files.push_back({stem + ".csv", text_lines(segments_csv(segments, used))});
        }
    }
    return files;
}

}  // namespace vml::app
