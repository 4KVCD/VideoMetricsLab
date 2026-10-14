// The window's Options panel part (the Python app's MainWindow, its
// "per-video settings panel"): the selected rows' settings shown in the
// panel, and each control the user edits applied to them.
#include <QComboBox>
#include <QFileDialog>
#include <QLabel>

#include <set>
#include <string>
#include <vector>

#include "i18n.h"
#include "main_window.h"
#include "options_panel.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/metrics.h"

// The window's text is app::tr's, as in main_window.cpp.
namespace vml::app {

void MainWindow::on_table_selection_changed() {
    // Shows the selected rows' options in the Options panel.
    std::set<int> rows;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        rows.insert(index.row());
    }
    panel_target_rows_.assign(rows.begin(), rows.end());
    if (rows.empty()) {
        options_panel_->setEnabled(false);
        options_panel_->target_label->setText(
            app::tr("Select one or more videos in the table to view or edit their settings."));
        syncing_panel_ = true;
        options_panel_->show_backends(default_options_.vmaf_on_gpu, default_metric_backends_);
        syncing_panel_ = false;
        return;
    }
    // A run owns these settings until it finishes. This is reached when
    // background work finishes as well as when the user clicks a row. Don't
    // enable the panel during a run: a change made there would be attached to
    // a result calculated with the previous settings.
    options_panel_->setEnabled(!run_active_);
    const RowData& first = *rows_[static_cast<std::size_t>(*rows.begin())];
    if (rows.size() == 1) {
        options_panel_->target_label->setText(app::tr("Editing settings for: {name}", {{"name", name_of(first.path)}}));
    } else {
        options_panel_->target_label->setText(
            ntr("Editing settings for {count} selected video -- editing anything below applies to all of them.",
                "Editing settings for {count} selected videos -- editing anything below applies to all of them.",
                static_cast<long long>(rows.size())));
    }
    write_panel_options(first.options);
}

void MainWindow::write_panel_options(const VmafOptions& options) {
    // Shows `options` in the panel, with the first selected row's metrics in
    // the headings, without the handlers reading their own changes back as
    // edits.
    syncing_panel_ = true;
    const bool any_selected = !panel_target_rows_.empty();
    const RowData* first = any_selected ? rows_[static_cast<std::size_t>(panel_target_rows_.front())].get() : nullptr;
    std::vector<VmafOptions> selected;
    for (const int row : panel_target_rows_) {
        selected.push_back(rows_[static_cast<std::size_t>(row)]->options);
    }
    if (selected.empty()) {
        selected.push_back(options);
    }
    options_panel_->show_options(options, first != nullptr ? first->metric_backends : default_metric_backends_,
                                 selected);
    const std::vector<std::string> requested = first != nullptr ? requested_metrics(*first) : std::vector<std::string>();
    for (const MetricColumn& metric : metric_columns()) {
        metric_header_->set_checked(metric.column, first != nullptr ? std::find(requested.begin(), requested.end(),
                                                                                metric.key) != requested.end()
                                                                    : default_metric_ticked(metric.column));
    }
    syncing_panel_ = false;
}

VmafOptions MainWindow::read_panel_options() const {
    // The options the panel shows, with the metrics ticked in the table's
    // column headings.
    std::vector<std::string> extra_features;
    for (const MetricDefinition& metric : metrics()) {
        if (metric.kind != MetricKind::frame || !metric.ffmpeg_binding ||
            metric.ffmpeg_binding->libvmaf_feature.empty()) {
            continue;
        }
        for (const MetricColumn& column : metric_columns()) {
            if (column.key == metric.key && metric_header_->is_checked(column.column)) {
                extra_features.emplace_back(metric.ffmpeg_binding->libvmaf_feature);
            }
        }
    }
    return options_panel_->read_options(extra_features, metric_header_->is_checked(col_vmaf),
                                        metric_header_->is_checked(col_vmaf_neg),
                                        metric_header_->is_checked(col_vmaf_v1), metric_header_->is_checked(col_xpsnr));
}

void MainWindow::on_panel_field_edited(const QString& field) {
    // Applies one edited control to the selected rows. Only that field:
    // copying the whole panel over a mixed multi-row selection would copy the
    // first selected row's other values over all the others.
    if (syncing_panel_) {
        return;
    }
    const VmafOptions panel = read_panel_options();
    // Copies the field from the panel into `target`; whether it changed.
    const auto apply = [&panel, &field](VmafOptions& target) {
        const VmafOptions before = target;
        if (field == "gpu") {
            target.gpu_decode = panel.gpu_decode;
            target.gpu_vendor = panel.gpu_vendor;
        } else if (field == "model") {
            target.model = panel.model;
            target.model_choice = panel.model_choice;
            target.custom_model_path = panel.custom_model_path;
        } else if (field == "model_v1") {
            target.model_v1 = panel.model_v1;
            target.model_choice_v1 = panel.model_choice_v1;
        } else if (field == "crop_mode") {
            target.crop_mode = panel.crop_mode;
        } else if (field == "duration_limit") {
            target.duration_limit = panel.duration_limit;
        } else if (field == "scale_algorithm") {
            target.scale_algorithm = panel.scale_algorithm;
        } else if (field == "scale_direction") {
            target.scale_direction = panel.scale_direction;
        } else if (field == "n_subsample") {
            target.n_subsample = panel.n_subsample;
        } else if (field == "n_threads") {
            target.n_threads = panel.n_threads;
        } else if (field == "vmaf_on_gpu") {
            target.vmaf_on_gpu = panel.vmaf_on_gpu;
        }
        return !(target == before);
    };
    apply(default_options_);
    if (field == "vmaf_on_gpu") {
        // The last choice is what the next session starts with, as for
        // SSIMULACRA2's and Butteraugli's (on_metric_backend_changed).
        settings_.default_vmaf_on_gpu = panel.vmaf_on_gpu;
        save_settings();
    }
    // Which GPU, how many threads, and how frames are scaled change how a
    // comparison is made, not what it is (the recipe's identity): its scores
    // stay.
    const bool execution_only =
        field == "gpu" || field == "n_threads" || field == "vmaf_on_gpu" || field == "scale_algorithm";
    std::vector<int> applied;
    for (const int row : panel_target_rows_) {
        if (apply(rows_[static_cast<std::size_t>(row)]->options)) {
            applied.push_back(row);
        }
    }
    const std::vector<int> changed = execution_only ? std::vector<int>() : applied;
    invalidate_completed_results(changed);
    for (const int row : applied) {
        set_row_black_bars(row);
    }
    reload_saved_scores(changed);
}

void MainWindow::on_metric_backend_changed(const QString& metric_key) {
    // Sets where SSIMULACRA2 or Butteraugli is calculated for the selected
    // rows. Where, never what: no score is taken away.
    if (syncing_panel_) {
        return;
    }
    const std::string key = metric_key.toStdString();
    const std::string backend = options_panel_->metric_backend(key);
    default_metric_backends_[key] = backend;
    for (const int row : panel_target_rows_) {
        rows_[static_cast<std::size_t>(row)]->metric_backends[key] = backend;
    }
    // The last choice is what the next session starts with, too.
    (key == "ssimulacra2" ? settings_.default_ssimulacra2_backend : settings_.default_butteraugli_backend) = backend;
    save_settings();
}

void MainWindow::on_model_changed(int index) {
    // Sets the selected rows' VMAF model, asking for the file of a custom one.
    if (syncing_panel_) {
        return;
    }
    if (model_choices()[static_cast<std::size_t>(index)].choice == custom_model_choice) {
        const QString path = QFileDialog::getOpenFileName(this, app::tr("Select VMAF model file (.json)"));
        if (path.isEmpty()) {
            // Back to Auto, which comes here again and is applied below.
            options_panel_->model_combo->setCurrentIndex(0);
            return;
        }
        options_panel_->custom_model_path = path.toStdString();
    }
    on_panel_field_edited("model");
}

void MainWindow::on_scale_direction_combo_changed(int index) {
    // Sets which video the selected rows resize. "Test both" (index 2), which
    // adds a row for the other direction, is not ported to C++ yet, and its
    // entry cannot be chosen.
    if (syncing_panel_ || index == 2) {
        return;
    }
    on_panel_field_edited("scale_direction");
}

}  // namespace vml::app
