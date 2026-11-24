// The window's saved scores (the Python app's MainWindow, its "persistent
// result cache" part): each row's scores looked up off the window's thread
// when it is added or the reference changes, a run's saved as it finishes,
// and the table's right-click menu: Recalculate, which forgets a row's, and
// Export results.
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
#include "stream_records.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/append.h"
#include "vml/file_fingerprint.h"
#include "vml/film_grain_scoring.h"
#include "vml/metric_cache.h"

// The window's text is app::tr's, as in main_window.cpp.
namespace vml::app {

std::filesystem::path MainWindow::cache_folder() const {
    return settings_.cache_dir_path().value_or(default_cache_dir());
}

AnalysisRequest MainWindow::analysis_request_of(const RowData& row_data) const {
    // The row's request as runs save it and lookups ask for it: one place,
    // so that a lookup never asks with other settings than a run saves under.
    // Its videos as the run's result gives them (save_job_result): a
    // resampled reference's test video is the reference itself.
    const VideoInfo source = source_info_.value_or(VideoInfo{});
    const VideoInfo distorted = row_data.options.resample_test ? source : row_data.video_info.value_or(VideoInfo{});
    return analysis_request(row_data.options, requested_metrics(row_data), automatic_perceptual_backends,
                            row_data.cvvdp, source, distorted);
}

std::shared_ptr<const RunFacts> MainWindow::saved_facts(const RowData& row_data, const ComparisonResult& shown) const {
    // The row's request as it is now (its saved scores answered it when
    // they were shown, and its ticks may have changed since), with the
    // videos as the run that saved them read them: their black bars were
    // kept under those (kept_crop_answer). A score the row shows was saved,
    // by the run that made it if not before.
    auto facts = std::make_shared<RunFacts>();
    facts->settings = VideoSettings{shown.source_info,
                                    shown.distorted_info,
                                    row_data.options,
                                    requested_metrics(row_data),
                                    {automatic_perceptual_backends.begin(), automatic_perceptual_backends.end()},
                                    row_data.cvvdp};
    for (const std::string& key : facts->settings.metrics) {
        facts->metrics.emplace_back(key, shown.has_metric(key) ? MetricOutcome{metric_saved, ""}
                                                               : MetricOutcome{metric_not_run, "no saved score", "no_saved_score"});
    }
    return facts;
}

bool MainWindow::start_cache_lookup(const std::vector<std::filesystem::path>& paths) {
    // Looks for the saved scores of the rows `paths`. A newer lookup replaces
    // one under way, and the window refuses the older one's answers (by its
    // number): which saved scores apply depends on the reference and on each
    // row's settings, as they are now. None as the window closes: a reading
    // started then would hold up the closing.
    if (paths.empty() || !source_info_ || !settings_.use_cache || closing_ || source_info_->is_raw_stream()) {
        return false;  // a raw stream's scores are not saved (save_job_result)
    }
    // The rows the replaced lookup had not answered for are asked about
    // again. Asked only about `paths`, editing one row while the saved scores
    // load would leave the others empty, and calculated again in full.
    std::vector<std::filesystem::path> asked = cancel_cache_lookup();
    append(asked, paths);
    // A remux check under way, or due, is for the lookup this one replaces:
    // its rows are asked about again.
    append(asked, stream_check_paths_);
    append(asked, stream_checking_paths_);
    cancel_stream_check();
    sync_remux_setting();
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
        const AnalysisRequest request = analysis_request_of(*row_data);
        const auto forgotten = row_data->forgotten_metrics.find(request.recipe.identity().dump());
        queries.emplace(text, SavedScoresQuery{row_data->identity_path(), request,
                                               displayable_metric_specs(row_data->options, row_data->cvvdp),
                                               forgotten == row_data->forgotten_metrics.end() ? std::set<std::string>()
                                                                                              : forgotten->second});
        cache_lookup_paths_.push_back(row_data->path);
    }
    if (texts.isEmpty()) {
        return false;
    }
    auto* worker = new ProbeWorker(texts, cache_folder(), source_info_->path, std::move(queries), this);
    cache_worker_ = worker;
    probe_workers_.push_back(worker);
    connect(worker, &ProbeWorker::saved_found, this,
            [this, generation](const SavedScoresFound& found) { on_saved_found(generation, found); });
    connect(worker, &ProbeWorker::other_displays_found, this,
            [this, generation](const OtherDisplaysFound& found) { on_other_displays_found(generation, found); });
    connect(worker, &ProbeWorker::stream_check_wanted, this, [this, generation](const QString& path) {
        if (generation == cache_generation_) {
            stream_check_paths_.push_back(path_of(path));
        }
    });
    connect(worker, &ProbeWorker::finished_all, this, [this, worker] { on_cache_lookup_finished(worker); });
    worker->start();
    return true;
}

void MainWindow::start_stream_check() {
    // The rows whose test stream began as a saved comparison's: read whole,
    // after the other rows' lookups, by a lookup of its own that nothing
    // waits for (a run cancels it: the row is calculated instead).
    if (stream_check_paths_.empty() || stream_check_worker_ != nullptr || !source_info_ || run_active_ ||
        closing_ || !settings_.use_cache || !recognise_remuxes_->load()) {
        return;
    }
    QStringList texts;
    std::map<QString, SavedScoresQuery> queries;
    stream_checking_paths_.clear();
    for (const std::filesystem::path& path : std::exchange(stream_check_paths_, {})) {
        const std::optional<int> row = row_index_of_path(path);
        if (!row) {
            continue;  // removed meanwhile
        }
        const RowData& row_data = *rows_[static_cast<std::size_t>(*row)];
        const QString text = text_of(row_data.path);
        if (queries.contains(text)) {
            continue;
        }
        const AnalysisRequest request = analysis_request_of(row_data);
        const auto forgotten = row_data.forgotten_metrics.find(request.recipe.identity().dump());
        queries.emplace(text, SavedScoresQuery{row_data.identity_path(), request,
                                               displayable_metric_specs(row_data.options, row_data.cvvdp),
                                               forgotten == row_data.forgotten_metrics.end() ? std::set<std::string>()
                                                                                              : forgotten->second});
        texts.append(text);
        stream_checking_paths_.push_back(row_data.path);
    }
    if (texts.isEmpty()) {
        return;
    }
    const int generation = cache_generation_;  // the lookup it completes: refused once another replaces it
    auto* worker = new ProbeWorker(texts, cache_folder(), source_info_->path, std::move(queries), this,
                                   ProbeWorker::WholeStreams::yes);
    stream_check_worker_ = worker;
    probe_workers_.push_back(worker);
    connect(worker, &ProbeWorker::saved_found, this,
            [this, generation](const SavedScoresFound& found) { on_saved_found(generation, found); });
    connect(worker, &ProbeWorker::other_displays_found, this,
            [this, generation](const OtherDisplaysFound& found) { on_other_displays_found(generation, found); });
    connect(worker, &ProbeWorker::stream_check_started, this, [this](const QString& path) {
        show_stream_status(app::tr("Checking whether {name} is a remux of a video already scored: reading its "
                                   "stream\xE2\x80\xA6",
                                   {{"name", name_of(path_of(path))}}));
    });
    connect(worker, &ProbeWorker::finished_all, this, [this, worker] { on_stream_check_finished(worker); });
    worker->start();
}

void MainWindow::cancel_stream_check() {
    if (stream_check_worker_ != nullptr) {
        stream_check_worker_->cancel();  // retired as it ends, like every reader
        stream_check_worker_ = nullptr;
        end_stream_status();
    }
    stream_check_paths_.clear();
    stream_checking_paths_.clear();
}

void MainWindow::on_stream_check_finished(ProbeWorker* worker) {
    on_probe_finished(worker);
    if (worker == stream_check_worker_) {  // not one cancelled, whose status went with it
        stream_check_worker_ = nullptr;
        stream_checking_paths_.clear();
        end_stream_status();
    }
}

void MainWindow::show_stream_status(const QString& text) {
    if (run_active_) {
        return;  // the run's own status says what matters then
    }
    stream_status_ = text;
    show_status(text);
}

void MainWindow::end_stream_status() {
    if (!stream_status_.isEmpty() && status_label_->text() == stream_status_ && !run_active_) {
        show_status(app::tr("Ready."));
    }
    stream_status_.clear();
}

void MainWindow::use_stream_fingerprinter(const std::string& ffmpeg, const std::string& ffprobe) {
    stream_reader_ = stream_fingerprinter_using(ffmpeg, ffprobe);
    sync_remux_setting();
}

void MainWindow::sync_remux_setting() {
    recognise_remuxes_->store(settings_.recognise_test_remuxes);
    // Off: no fingerprinter, so test videos are known by their files only,
    // and nothing is remembered of a file looked up meanwhile (a reader
    // that answered nothing would have its nothing kept until a restart).
    set_stream_fingerprinter(settings_.recognise_test_remuxes ? stream_reader_ : StreamFingerprinter());
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
    // A replaced lookup ends too (cancelled). A run waiting for saved scores
    // starts once none is under way: it waits for every lookup that replaced
    // the one it asked for, and not for one a Recalculate cancelled with none
    // after it (each lookup's end comes, cancelled or not).
    const bool current = worker == cache_worker_;
    if (current) {
        cache_worker_ = nullptr;
        cache_lookup_paths_.clear();
    }
    on_probe_finished(worker);
    if (run_waits_for_saved_scores_ && cache_worker_ == nullptr) {
        stop_waiting_for_saved_scores();
        on_run_clicked();  // the rows' saved scores are in: the run asked for starts
    }
    if (current && cache_worker_ == nullptr) {
        start_stream_check();  // after the others, and not before a run
    }
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
    if (row_data.video_info && row_data.video_info->frame_rate_unknown()) {
        return;  // a raw stream not yet given its rate: its saved scores are known by it (test_rate)
    }
    const AnalysisRequest request = analysis_request_of(row_data);
    if (lookup_token(source_info_->path, row_data.identity_path(), request) != found.token) {
        return;
    }
    // A comparison saved by an older version has its files recorded now, so
    // that it is found when they move: off the window's thread and after the
    // lookup, whose numbers a 4K reference's seconds of decoding would hold up.
    // And its test stream, which no older version recorded, for a remux of it
    // to keep these scores (with Settings' "Recognize test videos remuxed"):
    // read as after a run's save, apart from the writes; queued only where
    // none is recorded, so a video whose stream is says nothing.
    sync_remux_setting();
    // Recorded with the test video's rate, which a remux must have to be
    // found: a row is looked up as it is added, before its video is read, and
    // its request has no rate then; the saved comparison has its video's.
    ComparisonRecipe recorded = request.recipe;
    if (!(recorded.test_rate > 0.0)) {
        recorded.test_rate = found.result.distorted_info.fps;
    }
    file_writes_->submit(QStringLiteral("identity %1").arg(text_of(row_data.path.stem())),
                         [base = cache_folder(), source = source_info_->path, distorted = row_data.identity_path(),
                          recipe = std::move(recorded), records = stream_records_,
                          recognise = settings_.recognise_test_remuxes] {
                             record_saved_files(base, source, distorted, recipe);
                             if (recognise && test_stream_wanted(base, distorted)) {
                                 records->add(base, distorted);
                             }
                         });
    const std::shared_ptr<CompletedRun> previous = row_data.completed_run;
    auto run = std::make_shared<CompletedRun>();
    run->result = found.result;
    // Named after the row, as a run calculated now is: the label saved with
    // the scores is that of the file they were saved for, which a companion
    // row (a film grain twin, another display's CVVDP, the other scale
    // direction) shares with the video's own row.
    run->label = text_of(row_data.path.stem());
    // They answer the row's request as it is now (its token): its videos'
    // film grain as read and chosen now, which a result saved before either
    // was known lacks, and their bitrates as read now, which an older
    // version may have taken from the whole file (the BD-rate tab reads
    // them). A resolution test's test video is the reference.
    take_film_grain(run->result.source_info, *source_info_);
    run->result.source_info.take_bit_rate(*source_info_);
    if (row_data.options.resample_test) {
        take_film_grain(run->result.distorted_info, *source_info_);
        run->result.distorted_info.take_bit_rate(*source_info_);
    } else if (row_data.video_info) {
        take_film_grain(run->result.distorted_info, *row_data.video_info);
        run->result.distorted_info.take_bit_rate(*row_data.video_info);
    }
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
            run->result.metric_results.copy_from(shown, key);
        }
    }
    for (const std::string& key : run->result.metric_results.keys()) {
        row_data.metric_failures.erase(key);  // a score now answers it
    }
    run->from_saved_scores = true;
    row_data.completed_run = std::move(run);
    refresh_length_note();  // scored already: compared
    row_data.analysis_status = has_requested_results(row_data) ? std::optional(RowState::cached) : std::nullopt;
    row_data.status_detail =
        scores_detail(row_data.completed_run->result) +
        app::tr("\nLoaded from a previous run (matching files and calculation settings) -- right-click to "
                "recompute.");
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
    if (row_data.video_info && row_data.video_info->frame_rate_unknown()) {
        return;  // as on_saved_found
    }
    if (nlohmann::json(row_data.cvvdp.parameters()) != found.parameters) {
        return;  // the row's display changed meanwhile: its own lookup follows
    }
    row_data.cvvdp_elsewhere = found.scores;
    set_row_metrics(*row);
}

void MainWindow::show_table_menu(const QPoint& position) {
    // The table's right-click menu: Recalculate for the selected rows, but
    // not during a run (it forgets saved scores and puts rows back to
    // calculate, which would fight the run using them); and the exports, at
    // any time.
    std::set<int> selected;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        selected.insert(index.row());
    }
    QMenu menu(this);
    const QAction* recalculate = nullptr;
    if (!run_active_ && !selected.empty()) {
        recalculate = menu.addAction(app::tr("Recalculate selected metrics (ignore cached/previous results)"));
        menu.addSeparator();
    }
    // The rows selected, where the row clicked is one of several selected,
    // as the Export button takes them; else the video clicked, once all its
    // metrics are scored; then all the videos'. Each of what Settings'
    // Exports include.
    const int clicked = distorted_table_->rowAt(position.y());
    const bool scored = clicked >= 0 && scored_video(*rows_[static_cast<std::size_t>(clicked)]).has_value();
    const bool in_selection = selected.size() > 1 && selected.contains(clicked);
    const std::vector<std::string> items = exported_items(settings_);
    const bool anything = !export_contents(items).isEmpty();
    QAction* this_video = nullptr;
    QAction* selected_videos = nullptr;
    if (in_selection) {
        // Usable where one of them has every ticked metric scored, or failed
        // on its run, as the export takes only those (export_rows).
        const bool any_scored = std::any_of(selected.begin(), selected.end(), [this](int row) {
            return scored_video(*rows_[static_cast<std::size_t>(row)]).has_value();
        });
        selected_videos = menu.addAction(export_action_text(ExportScope::selected_videos, items));
        selected_videos->setEnabled(anything && any_scored);
    } else if (scored) {
        this_video = menu.addAction(export_action_text(ExportScope::this_video, items));
        this_video->setEnabled(anything);
    }
    QAction* all_videos = menu.addAction(export_action_text(ExportScope::all_videos, items));
    all_videos->setEnabled(anything && !exported_runs().empty());
    // Then the Export button's quick grabs, of the same rows.
    menu.addSeparator();
    SavedResultsPanel::add_quick_grabs(menu, !export_menu_rows().empty(),
                                       [this](SavedResultsPanel::Export kind) { export_rows_as(kind); },
                                       videos_refused());
    const QAction* chosen = menu.exec(distorted_table_->viewport()->mapToGlobal(position));
    if (chosen != nullptr && chosen == recalculate) {
        recompute_rows(std::vector<int>(selected.begin(), selected.end()));
    } else if (chosen != nullptr && chosen == this_video) {
        export_this_video(clicked);
    } else if (chosen != nullptr && chosen == selected_videos) {
        export_rows();
    } else if (chosen != nullptr && chosen == all_videos) {
        export_all_videos();
    }
}

void MainWindow::recompute_rows(const std::vector<int>& rows, bool forget_saved) {
    // Forgets the saved scores of the rows' ticked metrics, so that the next
    // run calculates them again. Those of unticked metrics are neither
    // calculated again nor forgotten, and stay on show. Without
    // `forget_saved` the saved files are left alone: already gone.
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
        if (forget_saved && source_info_) {
            file_writes_->submit(QStringLiteral("clear cached result for %1").arg(name_of(row_data.path)),
                                 [base, source = source_info_->path, identity = row_data.identity_path(),
                                  request = analysis_request_of(row_data)] {
                                     clear_metrics(base, source, identity, request.recipe, request.metrics);
                                 });
            // Auto's saved scores (Source only) or the whole videos' (a
            // range) would answer them again (load_result).
            const ComparisonRecipe recipe = analysis_request_of(row_data).recipe;
            if (recipe.crop_mode == CropMode::source_only || recipe.range_start > 0 || recipe.range_end > 0) {
                const std::vector<std::string> keys = requested_metrics(row_data);
                row_data.forgotten_metrics[recipe.identity().dump()].insert(keys.begin(), keys.end());
            }
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

void MainWindow::on_saved_results_cleared() {
    // Settings' Clear saved results has deleted every saved score: the rows
    // showing scores have their ticked metrics calculated again by the next
    // run, as its question said, rather than skipped for the scores on show.
    // Not during a run (Settings is locked then; a clear asked for just
    // before it started): the run's rows are being calculated afresh.
    for (std::size_t row = 0; row < rows_.size(); ++row) {
        if (!rows_[row]->cvvdp_elsewhere.empty()) {
            rows_[row]->cvvdp_elsewhere.clear();  // the other displays' saved scores went too
            set_row_metrics(static_cast<int>(row));
        }
    }
    if (run_active_) {
        return;
    }
    std::vector<int> scored;
    for (std::size_t row = 0; row < rows_.size(); ++row) {
        if (rows_[row]->completed_run) {
            scored.push_back(static_cast<int>(row));
        }
    }
    if (!scored.empty()) {
        recompute_rows(scored, false);
    }
}

void MainWindow::drop_metric_results(int row, const std::vector<std::string>& keys) {
    // Takes some of the row's scores away, keeping the rest on show.
    RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    for (const std::string& key : keys) {
        row_data.metric_failures.erase(key);
    }
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
        kept.copy_from(run->result.metric_results, key);
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
        rest->facts = run->facts;
        rest->record = run->record;
        rest->from_saved_scores = run->from_saved_scores;
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
