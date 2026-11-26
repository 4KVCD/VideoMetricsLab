// The window's metrics: which are ticked, shown and available for each
// row, Add/remove metrics' panel, and the saved scores a row can reuse, with
// the notes its cells carry.
#include "main_window.h"

#include <QTableWidgetItem>

#include <algorithm>
#include <numeric>
#include <utility>

#include "formatting.h"
#include "i18n.h"
#include "metric_list.h"
#include "settings_panel.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/score_identity.h"
#include "vml/vship.h"
#include "vship_names.h"
#include "widgets.h"
#include "window_helpers.h"

namespace vml::app {

// ------------------------------------------------------------- the metrics

bool MainWindow::default_metric_ticked(int column) const {
    // Whether new rows start with this metric ticked. SSIMULACRA2,
    // Butteraugli and CVVDP are not in VmafOptions: reading only the options,
    // their headings would show unticked even when the metric is on.
    const MetricColumn* metric = metric_column_of(column);
    return metric_enabled(default_options_, column) || default_extra_metric_keys_.contains(metric->key);
}

bool MainWindow::metric_enabled(const VmafOptions& options, int column) {
    const char* key = metric_column_of(column)->key;
    return metric_definition(key).ffmpeg_binding.has_value() && options.metric_enabled(key);
}

void MainWindow::set_metric_option(VmafOptions& options, int column, bool checked) {
    options.set_metric_enabled(metric_column_of(column)->key, checked);
}

std::vector<std::string> MainWindow::selected_metrics(const RowData& row_data) {
    // The row's tick boxes, metrics whose column is hidden included, in the
    // registry's order. A plane has no tick box of its own: it comes with
    // its metric's, which the registry lists before it.
    const std::vector<std::string> ffmpeg = row_data.options.requested_metrics();
    std::vector<std::string> selected;
    const auto has = [&selected](std::string_view key) {
        return std::find(selected.begin(), selected.end(), key) != selected.end();
    };
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        const bool ticked = metric.is_plane() ? has(metric.plane_of)
                                              : std::find(ffmpeg.begin(), ffmpeg.end(), key) != ffmpeg.end() ||
                                                    row_data.extra_metric_keys.contains(key);
        if (ticked) {
            selected.push_back(key);
        }
    }
    return selected;
}

std::vector<std::string> MainWindow::requested_metrics(const RowData& row_data) const {
    // What a run of this row calculates: its ticks, less hidden metrics and
    // metrics the row's kind of comparison cannot produce. Hiding a metric
    // leaves the row's tick untouched, so showing it again restores it.
    std::vector<std::string> requested;
    for (const std::string& key : selected_metrics(row_data)) {
        if (metric_shown(key) && !metric_unavailable_reason(row_data, key)) {
            requested.push_back(key);
        }
    }
    return with_metric_inputs(std::move(requested));  // VMAF with VMAF YUV, which is made from its score
}

std::optional<QString> MainWindow::metric_unavailable_reason(const RowData& row_data, const std::string& key) {
    // CVVDP needs every frame, as it judges each one together with the frames
    // before it, and a supported GPU, as the app has no CPU version of it.
    if (key == "cvvdp") {
        if (row_data.options.n_subsample > 1) {
            return app::tr("CVVDP judges each frame together with the frames before it, so it needs every frame: it is "
                           "not available while libvmaf frame subsample is above 1.");
        }
        if (!vship_available(row_data.options.vship_build)) {
            if (row_data.options.vship_build != VshipBuild::fast && vship_available(VshipBuild::fast)) {
                return app::tr("CVVDP is calculated on the GPU only, and no GPU that {build} can use was found; with "
                               "Vship compute set to GPU it can be calculated.",
                               {{"build", vship_short_name(row_data.options.vship_build)}});
            }
            return app::tr("CVVDP is calculated on the GPU only, and no GPU that Vship can use was found.");
        }
    }
    return std::nullopt;
}

bool MainWindow::row_metric_enabled(const RowData& row_data, int column) {
    const std::vector<std::string> selected = selected_metrics(row_data);
    return std::find(selected.begin(), selected.end(), metric_column_of(column)->key) != selected.end();
}

bool MainWindow::metric_shown(std::string_view key) const {
    const MetricDefinition& metric = metric_definition(key);
    if (metric.is_plane()) {
        return settings_.show_yuv_planes && !hidden_metrics_.contains(std::string(metric.plane_of));
    }
    return !hidden_metrics_.contains(std::string(key));
}

long long MainWindow::listed_metric_count() {
    return std::count_if(metric_columns().begin(), metric_columns().end(),
                         [](const MetricColumn& item) { return !metric_definition(item.key).is_plane(); });
}

void MainWindow::apply_metric_visibility() {
    long long shown = 0;
    for (const MetricColumn& item : metric_columns()) {
        const bool visible = metric_shown(item.key);
        distorted_table_->setColumnHidden(item.column, !visible);
        shown += visible && !metric_definition(item.key).is_plane() ? 1 : 0;
    }
    metrics_count_label_->setText(shown_count_text(shown, listed_metric_count()));
}

void MainWindow::show_metrics_panel(const QPoint& top_left, std::optional<QPoint> top_right, QWidget* anchor) {
    if (run_active_) {
        return;
    }
    auto* panel = new MetricsPanel(this);
    panel->set_anchor(anchor);
    MetricList* list = panel->list();
    std::set<std::string> shown;
    for (const MetricColumn& item : metric_columns()) {
        if (!metric_definition(item.key).is_plane() && !hidden_metrics_.contains(item.key)) {
            shown.insert(item.key);
        }
    }
    list->set_ticked(shown);
    list->set_planes_shown(settings_.show_yuv_planes);
    connect(list, &MetricList::ticks_changed, this, [this, list] { on_metrics_shown_changed(list->ticked()); });
    connect(list, &MetricList::planes_toggled, this, [this](bool shown_planes) { on_yuv_planes_toggled(shown_planes); });
    panel->show_at(top_left, top_right);
}

void MainWindow::on_metrics_shown_changed(const std::set<std::string>& shown) {
    // Hides or shows the metrics' columns as the panel's ticks say, keeping
    // at least one shown, and saves the choice. The rows' own ticks stay as
    // they are: they say what a row calculates of what is shown.
    if (shown.empty() || run_active_) {
        return;
    }
    bool more_shown = false;
    std::set<std::string> hidden;
    settings_.hidden_metrics.clear();
    for (const MetricColumn& item : metric_columns()) {
        if (metric_definition(item.key).is_plane()) {
            continue;
        }
        if (shown.contains(item.key)) {
            more_shown = more_shown || hidden_metrics_.contains(item.key);
        } else {
            hidden.insert(item.key);
            settings_.hidden_metrics.emplace_back(item.key);
        }
    }
    hidden_metrics_ = std::move(hidden);
    save_settings();
    apply_metric_visibility();
    check_ticks_soon();  // a hidden metric is not calculated: the ticks a run takes changed
    refresh_shown_metrics(more_shown);
}

void MainWindow::on_yuv_planes_toggled(bool shown) {
    // Shows or hides the planes' columns, and saves the choice.
    settings_.show_yuv_planes = shown;
    save_settings();
    apply_metric_visibility();
    refresh_shown_metrics(shown);
    check_ticks_soon();  // the planes a run asks for changed with them
}

void MainWindow::refresh_shown_metrics(bool more_shown) {
    // Every row's cells after columns were shown or hidden: what a run asks
    // for has changed with them.
    std::vector<int> every_row;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        set_row_metrics(row);
        every_row.push_back(row);
    }
    if (more_shown) {
        // Scores saved for the metrics while they were hidden come back.
        reload_saved_scores(every_row);
    }
}

void MainWindow::apply_metric_selection(const std::vector<int>& rows, int column, bool checked, bool set_default) {
    // Turns one metric on or off for `rows`. `set_default` carries the choice
    // to rows added later: true for the header shortcut, a statement about
    // the whole table; false for a tick in one row's own cell.
    if (syncing_panel_ || run_active_) {
        return;
    }
    const std::string key = metric_column_of(column)->key;
    const bool ffmpeg = metric_definition(key).ffmpeg_binding.has_value();
    for (const int row : rows) {
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        if (!ffmpeg) {
            if (checked) {
                row_data.extra_metric_keys.insert(key);
            } else {
                row_data.extra_metric_keys.erase(key);
            }
        } else {
            set_metric_option(row_data.options, column, checked);
        }
        row_data.analysis_status.reset();
        set_row_metrics(row);
    }
    if (set_default) {
        if (!ffmpeg) {
            if (checked) {
                default_extra_metric_keys_.insert(key);
            } else {
                default_extra_metric_keys_.erase(key);
            }
        } else {
            set_metric_option(default_options_, column, checked);
        }
        // Saved as the default too, the same setting as Settings > Default
        // metrics.
        *settings_.default_compute(key) = checked;
        settings_panel_->show_default_metric(key, checked);
        save_settings();
    }
    reload_saved_scores(rows);
    on_table_selection_changed();
}

void MainWindow::on_table_item_changed(QTableWidgetItem* item) {
    // A metric tick box in the table was clicked. It applies to every
    // selected row when the clicked row is one of them, so ticking PSNR
    // across a selection is one click; otherwise to the clicked row alone.
    if (syncing_table_) {
        return;
    }
    const int column = item->column();
    if (metric_column_of(column) == nullptr) {
        return;
    }
    const int row = item->row();
    if (row >= static_cast<int>(rows_.size())) {
        return;  // a row still being built: its boxes are filled in after
    }
    if (!(item->flags() & Qt::ItemIsUserCheckable)) {
        return;  // a cell showing a score or live progress, not a tick box
    }
    const bool checked = item->checkState() == Qt::Checked;
    if (checked == row_metric_enabled(*rows_[static_cast<std::size_t>(row)], column)) {
        // itemChanged also fires for text, colour and font edits: only a box
        // that now disagrees with the row it stands for is a click.
        return;
    }
    if (run_active_) {
        set_row_metrics(row);  // nothing may change mid-run
        return;
    }
    std::set<int> selected;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        selected.insert(index.row());
    }
    const std::vector<int> rows = selected.contains(row) ? std::vector<int>(selected.begin(), selected.end())
                                                         : std::vector<int>{row};
    apply_metric_selection(rows, column, checked, false);
}

void MainWindow::on_metric_column_toggled(int column, bool checked) {
    // The header's shortcuts apply to every row. Not during a run, which owns
    // them: the box the click flipped goes back.
    if (run_active_) {
        metric_header_->set_checked(column, !checked);
        return;
    }
    std::vector<int> rows(rows_.size());
    std::iota(rows.begin(), rows.end(), 0);
    apply_metric_selection(rows, column, checked, true);
}

// -------------------------------------------------------------- the scores

std::vector<std::string> MainWindow::reusable_metric_keys(const RowData& row_data) const {
    // The requested metrics the row already has and a run can keep
    // (metric_cache.h's reusable_keys, the command line's rule too).
    if (!row_data.completed_run) {
        return {};
    }
    std::vector<ColourTags> infos;
    for (const std::optional<VideoInfo>* info : {&source_info_, &row_data.video_info}) {
        if (*info) {
            infos.push_back(colour_tags_of(**info));
        }
    }
    const ComparisonResult& result = row_data.completed_run->result;
    return reusable_keys(result,
                         analysis_request(row_data.options, requested_metrics(row_data), automatic_perceptual_backends,
                                          row_data.cvvdp, result.source_info, result.distorted_info)
                             .metrics,
                         automatic_perceptual_backends,
                         [build = row_data.options.vship_build](const std::string&) { return vship_available(build); },
                         infos);
}

MetricResultSet MainWindow::reusable_results(const RowData& row_data) const {
    // Those metrics' scores, for a run to keep.
    MetricResultSet kept;
    for (const std::string& key : reusable_metric_keys(row_data)) {
        const MetricResultSet& scores = row_data.completed_run->result.metric_results;
        if (const SequenceMetricResult* sequence = scores.sequence(key)) {
            kept.add(*sequence);
        } else {
            kept.add(*scores.frame(key));
        }
    }
    return kept;
}

bool MainWindow::has_requested_results(const RowData& row_data) const {
    // Whether the row already has a result for every metric it asks for.
    const std::vector<std::string> requested = requested_metrics(row_data);
    if (requested.empty()) {
        return false;
    }
    const std::vector<std::string> kept = reusable_metric_keys(row_data);
    return std::all_of(requested.begin(), requested.end(), [&kept](const std::string& key) {
        return std::find(kept.begin(), kept.end(), key) != kept.end();
    });
}

bool MainWindow::backend_matches(const RowData& row_data, const std::string& key, const MetricProvenance& provenance) {
    // Whether a score was made where its metric is calculated now
    // (metric_cache.h's made_as_chosen): SSIMULACRA2 and Butteraugli on the
    // GPU where the row's Vship build can use one. The other metrics' always
    // are.
    const auto choice = automatic_perceptual_backends.find(key);
    const VshipBuild build = row_data.options.vship_build;
    return made_as_chosen(key,
                          choice == automatic_perceptual_backends.end() ? std::nullopt : std::optional(choice->second),
                          provenance.compute_backend, [build](const std::string&) { return vship_available(build); });
}

std::optional<VshipBuild> MainWindow::other_vship_build(const RowData& row_data, const MetricProvenance& provenance) {
    // The Vship build that made a GPU score of Vship's, where its scores are not
    // those of the one that calculates for the row (vship_build_in_use): Vship
    // 5.1.2's after Vship-fast failed. Their scores differ slightly; stock Vship's
    // Vulkan, HIP and CUDA share theirs.
    const std::string& id = provenance.implementation_compatibility_id;
    const std::optional<VshipBuild> made = score_identity::is_vship51_id(id)      ? std::optional(VshipBuild::stock)
                                           : score_identity::is_vship_fast_id(id) ? std::optional(VshipBuild::fast)
                                                                                  : std::nullopt;
    const bool in_use_is_stock = vship_build_in_use(row_data.options.vship_build) != VshipBuild::fast;
    return made && (*made == VshipBuild::stock) != in_use_is_stock ? made : std::nullopt;
}

QString MainWindow::other_scale_note() {
    // The tooltip's word on a SSIMULACRA2 or Butteraugli score calculated on
    // the CPU where the GPU can calculate it (before a GPU was found, or after
    // it failed), the rare case marked "(CPU)" in the cell.
    return app::tr("\n\nCalculated on the CPU, which gives different numbers from the GPU, so the next run calculates "
                   "it again on this PC's GPU.");
}

std::optional<double> MainWindow::score_of(CompletedRun& run, int column) {
    // A metric column's score: CVVDP's one for the whole video, the others'
    // mean of their frames. Worked out once a run, not at each refresh of
    // its row: for a two-hour video that takes 0.6 ms a metric each time.
    if (const auto known = run.scores.find(column); known != run.scores.end()) {
        return known->second;
    }
    const std::string key = metric_column_of(column)->key;
    std::optional<double> score;
    if (const SequenceMetricResult* sequence = run.result.metric_results.sequence(key)) {
        score = sequence->score;
    } else if (const FrameMetricResult* frame = run.result.metric_results.frame(key)) {
        score = frame->aggregate();
    }
    run.scores.emplace(column, score);
    return score;
}

QString MainWindow::identical_frames_note(CompletedRun& run, int column) {
    // Says when frames were identical to the reference and scored infinity:
    // without it, the mean silently describes fewer frames than the run
    // measured. Counted once a run.
    auto counted = run.identical.find(column);
    if (counted == run.identical.end()) {
        std::pair<long long, long long> found{0, 0};
        if (const FrameMetricResult* frame = run.result.metric_results.frame(metric_column_of(column)->key)) {
            found = {std::count_if(frame->values.begin(), frame->values.end(),
                                   [](float value) { return std::isinf(value) && value > 0; }),
                     static_cast<long long>(frame->values.size())};
        }
        counted = run.identical.emplace(column, found).first;
    }
    const auto [identical, count] = counted->second;
    if (identical == 0) {
        return {};
    }
    return app::tr("\n\n{identical} of {count} frames were identical to the reference and scored infinity; they "
                   "contribute zero distortion and are included in the frame count for the XPSNR sequence average.",
                   {{"identical", identical}, {"count", count}});
}

QString MainWindow::cvvdp_note(const RowData& row_data) const {
    // A CVVDP score's tooltip: what the number means, and which display it
    // is for.
    const std::optional<CvvdpPreset> preset =
        matching_cvvdp_preset(row_data.cvvdp, settings_.cvvdp_presets_json, row_data.cvvdp_preset);
    return app::tr("CVVDP of the whole video, in JOD (just-objectionable differences): 10 means no visible "
                   "difference, and one JOD lower means 75% of viewers would pick the reference as better.\n\nFor "
                   "the display {preset}: {display}",
                   {{"preset", preset ? preset_label(*preset) : app::tr("(custom)")},
                    {"display", display_description(row_data.cvvdp.display)}}) +
           (row_data.cvvdp.resize_to_display ? app::tr(", video scaled to fill it") : QString()) +
           app::tr(".\n\nThe JOD of each second is plotted in Metric Graphs.");
}

}  // namespace vml::app
