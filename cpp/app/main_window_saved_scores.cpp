// The window's saved scores (the Python app's MainWindow, its "persistent
// result cache" part): each row's scores looked up off the window's thread
// when it is added or the reference changes, a run's saved as it finishes,
// and Recalculate, which forgets a row's.
#include <QItemSelectionModel>
#include <QMenu>

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "file_writes.h"
#include "formatting.h"
#include "graph_panel.h"
#include "i18n.h"
#include "main_window.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/metric_cache.h"

// The window's text is app::tr's, as in main_window.cpp.
namespace vml::app {

std::filesystem::path MainWindow::cache_folder() const {
    return settings_.cache_dir_path().value_or(default_cache_dir());
}

AnalysisRequest MainWindow::analysis_request_of(const RowData& row_data) const {
    // The row's request as runs save it and lookups ask for it: one place,
    // so that a lookup never asks with other settings than a run saves under.
    return analysis_request(row_data.options, requested_metrics(row_data), row_data.metric_backends, row_data.cvvdp);
}

void MainWindow::start_cache_lookup(const std::vector<std::filesystem::path>& paths) {
    // Looks for the saved scores of the rows `paths`. A newer lookup replaces
    // one under way, and the window refuses the older one's answers (by its
    // number): which saved scores apply depends on the reference and on each
    // row's settings, as they are now. None as the window closes: a reading
    // started then would hold up the closing.
    if (paths.empty() || !source_info_ || !settings_.use_cache || closing_) {
        return;
    }
    // The rows the replaced lookup had not answered for are asked about
    // again. Asked only about `paths`, editing one row while the saved scores
    // load would leave the others empty, and calculated again in full.
    std::vector<std::filesystem::path> asked = cancel_cache_lookup();
    asked.insert(asked.end(), paths.begin(), paths.end());
    const int generation = ++cache_generation_;
    QStringList texts;
    std::map<QString, SavedScoresQuery> queries;
    cache_lookup_paths_.clear();
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        const QString text = text_of(row_data->path);
        const bool wanted = std::any_of(asked.begin(), asked.end(), [&](const std::filesystem::path& path) {
            return same_path(path, row_data->path);
        });
        if (!wanted || queries.contains(text)) {
            continue;  // rows removed meanwhile are not asked about
        }
        texts.append(text);
        queries.emplace(text, SavedScoresQuery{row_data->identity_path(), analysis_request_of(*row_data),
                                               displayable_metric_specs(row_data->options, row_data->cvvdp)});
        cache_lookup_paths_.push_back(row_data->path);
    }
    if (texts.isEmpty()) {
        return;
    }
    auto* worker = new ProbeWorker(texts, cache_folder(), source_info_->path, std::move(queries), this);
    cache_worker_ = worker;
    probe_workers_.push_back(worker);
    connect(worker, &ProbeWorker::saved_found, this,
            [this, generation](const SavedScoresFound& found) { on_saved_found(generation, found); });
    connect(worker, &ProbeWorker::other_displays_found, this,
            [this, generation](const OtherDisplaysFound& found) { on_other_displays_found(generation, found); });
    connect(worker, &ProbeWorker::finished_all, this, [this, worker] { on_cache_lookup_finished(worker); });
    worker->start();
}

void MainWindow::reload_saved_scores(const std::vector<int>& rows) {
    // Looks the rows' saved scores up again after a change to what they ask
    // for (the Python app's _reload_cached_for_rows): scores saved since the
    // last lookup come back, and the rows of a lookup whose answers the
    // change made the window refuse are asked about again.
    std::vector<std::filesystem::path> paths;
    for (const int row : rows) {
        paths.push_back(rows_[static_cast<std::size_t>(row)]->path);
    }
    start_cache_lookup(paths);
}

std::vector<std::filesystem::path> MainWindow::cancel_cache_lookup() {
    // Stops the lookup under way, if any; returns the rows it was asked
    // about, whose answers will now be refused. A lookup is under way until
    // its last signal has been handled, not until its thread ends: answers it
    // sent may still wait in the queue.
    if (cache_worker_ == nullptr) {
        return {};
    }
    cache_worker_->cancel();  // retired as it ends, like every reader
    cache_worker_ = nullptr;
    return std::exchange(cache_lookup_paths_, {});
}

void MainWindow::on_cache_lookup_finished(ProbeWorker* worker) {
    if (worker == cache_worker_) {
        cache_worker_ = nullptr;
        cache_lookup_paths_.clear();
    }
    on_probe_finished(worker);
}

void MainWindow::on_saved_found(int generation, const SavedScoresFound& found) {
    // Shows saved scores on their row, if they are still the answer for it:
    // the same lookup, and the token of the row's request as it is now (a
    // file replaced, or the row's settings changed, while they were read).
    if (generation != cache_generation_ || !source_info_) {
        return;
    }
    const std::optional<int> row = row_index_of_path(path_of(found.path));
    if (!row) {
        return;
    }
    RowData& row_data = *rows_[static_cast<std::size_t>(*row)];
    if (lookup_token(source_info_->path, row_data.identity_path(), analysis_request_of(row_data)) != found.token) {
        return;
    }
    const std::shared_ptr<CompletedRun> previous = row_data.completed_run;
    auto run = std::make_shared<CompletedRun>();
    run->result = found.result;
    run->label = found.label;
    if (previous) {
        // The row's line updated in place: its colour, and whether it is
        // hidden or removed, stay. Replaced under a new identity, they were
        // reset whenever a display switched back to had a saved CVVDP score.
        run->graph_identity = previous->graph_identity;
    }
    if (row_data.completed_run) {
        // The saved scores replace what the row shows when they hold
        // something it does not, ticked or not, as saved scores are shown
        // whatever is ticked. Those the row shows and they lack stay beside
        // them: a row showing a GPU SSIMULACRA2, then set to the CPU, never
        // gets it back from the saved scores, and every later answer, such
        // as a display switched to with a saved CVVDP, would be refused.
        const MetricResultSet& shown = row_data.completed_run->result.metric_results;
        const std::vector<std::string>& saved = found.result.metric_results.keys();
        if (std::all_of(saved.begin(), saved.end(), [&shown](const std::string& key) { return shown.has(key); })) {
            return;
        }
        for (const std::string& key : shown.keys()) {
            if (run->result.has_metric(key)) {
                continue;
            }
            if (const FrameMetricResult* frame = shown.frame(key)) {
                run->result.metric_results.add(*frame);
            } else if (const SequenceMetricResult* sequence = shown.sequence(key)) {
                run->result.metric_results.add(*sequence);
            }
        }
    }
    row_data.completed_run = std::move(run);
    row_data.analysis_status = has_requested_results(row_data) ? std::optional(RowState::cached) : std::nullopt;
    // The scores and crops come from the saved result, the video's details
    // from reading it now: an older result lacks tags a newer read has.
    if (row_data.options.resample_test) {
        set_resample_row_info(*row, *row_data.options.resample_test);
    } else {
        set_row_info(*row, row_data.video_info ? *row_data.video_info : found.result.distorted_info);
    }
    set_row_metrics(*row);
    // The graph kept in step as scores arrive, so that the tab shows them all
    // when opened.
    const std::shared_ptr<CompletedRun>& shown = row_data.completed_run;
    graph_panel_->add_run({std::shared_ptr<const ComparisonResult>(shown, &shown->result), shown->label,
                           shown->graph_identity, previous == nullptr});
    sync_frame_compare();
}

void MainWindow::on_other_displays_found(int generation, const OtherDisplaysFound& found) {
    // Keeps CVVDP's scores saved for other displays, for the CVVDP cell.
    if (generation != cache_generation_) {
        return;
    }
    const std::optional<int> row = row_index_of_path(path_of(found.path));
    if (!row) {
        return;
    }
    RowData& row_data = *rows_[static_cast<std::size_t>(*row)];
    if (nlohmann::json(row_data.cvvdp.parameters()) != found.parameters) {
        return;  // the row's display changed meanwhile: its own lookup follows
    }
    row_data.cvvdp_elsewhere = found.scores;
    set_row_metrics(*row);
}

void MainWindow::show_table_menu(const QPoint& position) {
    // The table's right-click menu. None during a run: its entry forgets
    // saved scores and puts rows back to calculate, which would fight the
    // run using them.
    if (run_active_) {
        return;
    }
    std::set<int> selected;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        selected.insert(index.row());
    }
    if (selected.empty()) {
        return;
    }
    QMenu menu(this);
    const QAction* recalculate =
        menu.addAction(app::tr("Recalculate selected metrics (ignore cached/previous results)"));
    if (menu.exec(distorted_table_->viewport()->mapToGlobal(position)) == recalculate) {
        recompute_rows(std::vector<int>(selected.begin(), selected.end()));
    }
}

void MainWindow::recompute_rows(const std::vector<int>& rows) {
    // Forgets the saved scores of the rows' ticked metrics, so that the next
    // run calculates them again. Those of unticked metrics are neither
    // calculated again nor forgotten, and stay on show.
    const std::filesystem::path base = cache_folder();  // now, not as each deletion runs
    // A lookup under way must not bring back the scores just forgotten; the
    // other rows it was asked about are asked about again below.
    const std::vector<std::filesystem::path> interrupted = cancel_cache_lookup();
    ++cache_generation_;
    for (const int row : rows) {
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        row_data.analysis_status.reset();
        drop_metric_results(row, requested_metrics(row_data));
        row_data.status_detail.clear();
        if (source_info_) {
            file_writes_->submit(QStringLiteral("clear cached result for %1").arg(name_of(row_data.path)),
                                 [base, source = source_info_->path, identity = row_data.identity_path(),
                                  request = analysis_request_of(row_data)] {
                                     clear_metrics(base, source, identity, request.recipe, request.metrics);
                                 });
        }
        // The Scaling cell for the row's settings again, not the cleared
        // result's crops. Not a resolution test's: its Info cell describes
        // the test, not a video.
        if (row_data.video_info && !row_data.options.resample_test) {
            set_row_info(row, *row_data.video_info);
        }
    }
    show_status(ntr("Cleared {count} result -- make sure it is checked, then click Calculate metrics to recompute.",
                    "Cleared {count} results -- make sure they're checked, then click Calculate metrics to recompute.",
                    static_cast<long long>(rows.size())));
    sync_frame_compare();
    std::vector<std::filesystem::path> others;
    for (const std::filesystem::path& path : interrupted) {
        if (std::none_of(rows.begin(), rows.end(),
                         [&](int row) { return same_path(rows_[static_cast<std::size_t>(row)]->path, path); })) {
            others.push_back(path);
        }
    }
    start_cache_lookup(others);
}

void MainWindow::drop_metric_results(int row, const std::vector<std::string>& keys) {
    // Takes some of the row's scores away, keeping the rest on show.
    RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    const std::shared_ptr<CompletedRun> run = row_data.completed_run;
    const auto dropped = [&keys](const std::string& key) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    };
    if (!run || std::none_of(keys.begin(), keys.end(), [&run](const std::string& key) {
            return run->result.has_metric(key);
        })) {
        set_row_metrics(row);
        return;
    }
    MetricResultSet kept;
    for (const std::string& key : run->result.metric_results.keys()) {
        if (dropped(key)) {
            continue;
        }
        if (const FrameMetricResult* frame = run->result.metric_results.frame(key)) {
            kept.add(*frame);
        } else if (const SequenceMetricResult* sequence = run->result.metric_results.sequence(key)) {
            kept.add(*sequence);
        }
    }
    row_data.analysis_status.reset();
    if (kept.empty()) {
        if (!graph_panel_->remove_by_identity(run->graph_identity)) {
            graph_panel_->remove_by_path(row_data.path);
        }
        row_data.completed_run.reset();
        row_data.status_detail.clear();
    } else {
        // A copy, not the result edited: the graph holds the old one.
        auto rest = std::make_shared<CompletedRun>();
        rest->result = run->result;
        rest->label = run->label;
        rest->result.metric_results = std::move(kept);
        // The same line, updated: under a new identity it took a new colour
        // and came back even if hidden or removed from the graph.
        rest->graph_identity = run->graph_identity;
        graph_panel_->add_run({std::shared_ptr<const ComparisonResult>(rest, &rest->result), rest->label,
                               rest->graph_identity, false});
        row_data.completed_run = std::move(rest);
    }
    set_row_metrics(row);
    sync_frame_compare();
}

QString MainWindow::cvvdp_elsewhere_note(const RowData& row_data) const {
    // For an empty CVVDP cell: the scores saved for other displays. A CVVDP
    // score holds for the display it was made for only, so a video whose
    // display differs from its earlier run's shows none: this says that one
    // exists, and which display shows it again.
    std::vector<std::pair<CvvdpSettings, double>> others;
    for (const auto& [settings, score] : row_data.cvvdp_elsewhere) {
        if (!settings.same_as(row_data.cvvdp)) {
            others.emplace_back(settings, score);
        }
    }
    if (others.empty()) {
        return {};
    }
    std::stable_sort(others.begin(), others.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    QStringList lines;
    for (const auto& [settings, score] : others) {
        const std::optional<CvvdpPreset> preset = matching_cvvdp_preset(settings, settings_.cvvdp_presets_json);
        const QString name =
            preset ? preset_label(*preset)
                   : app::tr("Custom ({display})", {{"display", display_description(settings.display)}});
        lines.append(app::tr("• {name}{scaled}: {score:.3f} JOD",
                             {{"name", name},
                              {"scaled", settings.resize_to_display ? app::tr(", video scaled to fill it") : QString()},
                              {"score", score}}));
    }
    return app::tr("\n\nNo CVVDP score for this video's display yet. Saved for other displays (choose one in CVVDP "
                   "display to show it; they are not scores for this display):\n") +
           lines.join("\n");
}

}  // namespace vml::app
