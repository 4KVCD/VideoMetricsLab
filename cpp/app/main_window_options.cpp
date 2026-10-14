// The window's Options panel part (the Python app's MainWindow, its
// "per-video settings panel"): the selected rows' settings shown in the
// panel, and each control the user edits applied to them.
#include <QComboBox>
#include <QFileDialog>
#include <QLabel>
#include <QMessageBox>

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cvvdp_display_dialog.h"
#include "i18n.h"
#include "main_window.h"
#include "options_panel.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/cvvdp.h"
#include "vml/metrics.h"

// The window's text is app::tr's, as in main_window.cpp.
namespace vml::app {

namespace {

// A made-up path that stands for a second run of the same test video with
// the other scale direction (the Python app's
// synthetic_scale_direction_variant_path). Both directions can then be rows
// of their own, side by side, without one taking the other's saved scores.
std::filesystem::path synthetic_scale_direction_variant_path(const std::filesystem::path& distorted,
                                                             ScaleDirection direction) {
    std::filesystem::path name = distorted.stem();
    name += direction == ScaleDirection::distorted_to_source ? " [upscale-distorted-to-source]"
                                                            : " [downscale-source-to-distorted]";
    name += distorted.extension();
    return distorted.parent_path() / name;
}

}  // namespace

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
    std::vector<CvvdpSettings> displays;
    for (const int row : panel_target_rows_) {
        displays.push_back(rows_[static_cast<std::size_t>(row)]->cvvdp);
    }
    if (displays.empty()) {
        displays.push_back(default_cvvdp_);
    }
    options_panel_->show_cvvdp(displays, first != nullptr ? first->cvvdp_preset : settings_.cvvdp_default_preset,
                               settings_.cvvdp_presets_json);
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
    // Sets which video the selected rows resize, or adds the other direction
    // for "Test both".
    if (syncing_panel_) {
        return;
    }
    if (index == 2) {
        // "Test both" is not a value a row can have: it is an action that adds
        // a row for the other direction and leaves this row's direction as it
        // is, so the dropdown goes back to that direction.
        add_opposite_scale_direction_rows(panel_target_rows_);
        if (!panel_target_rows_.empty()) {
            write_panel_options(rows_[static_cast<std::size_t>(panel_target_rows_.front())]->options);
        }
        return;
    }
    on_panel_field_edited("scale_direction");
}

void MainWindow::add_opposite_scale_direction_rows(const std::vector<int>& rows) {
    // For each of `rows` whose resolution differs from the reference's, adds
    // a second row for the same test video with the other scale direction, so
    // that "scale the source down" and "scale the test video up" can be run
    // and compared side by side, rather than one picked and the other run
    // again later.
    if (!source_info_) {
        QMessageBox::warning(this, app::tr("No reference"), app::tr("Please select a reference video first."));
        return;
    }
    int added = 0;
    // By value: adding rows does not change these, but the selection can.
    for (const int row : std::vector<int>(rows)) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        if (row_data.options.resample_test) {
            continue;  // a resolution test scales the reference itself, in no direction
        }
        if (row_data.scale_direction_pinned) {
            // Already a companion row: the row it was made from is the other
            // direction, so the pair is complete. Without this, "Test both" on
            // a companion adds a needless third row named "x [upscale-...]
            // [downscale-...]".
            continue;
        }
        const std::optional<VideoInfo>& info = row_data.video_info;
        if (!info || (info->width == source_info_->width && info->height == source_info_->height)) {
            continue;  // not read yet, or the same size, so both directions are alike
        }
        const ScaleDirection opposite = row_data.options.scale_direction == ScaleDirection::source_to_distorted
                                            ? ScaleDirection::distorted_to_source
                                            : ScaleDirection::source_to_distorted;
        const std::filesystem::path path = synthetic_scale_direction_variant_path(row_data.path, opposite);
        if (row_index_of_path(path)) {
            continue;  // already added
        }
        const std::shared_ptr<RowData> original = rows_[static_cast<std::size_t>(row)];
        const int new_row = add_table_row(path);
        RowData& companion = *rows_[static_cast<std::size_t>(new_row)];
        companion.options = original->options;
        companion.options.scale_direction = opposite;
        // The same comparison in the other direction: the same metric ticks,
        // GPU/CPU choices and CVVDP display. Don't take the defaults: a video
        // set to CPU SSIMULACRA2 with CVVDP ticked would get a companion
        // without CVVDP and with SSIMULACRA2 on the GPU.
        companion.extra_metric_keys = original->extra_metric_keys;
        companion.metric_backends = original->metric_backends;
        companion.cvvdp = original->cvvdp;
        companion.cvvdp_preset = original->cvvdp_preset;
        companion.scale_direction_pinned = true;
        // The companion decodes the same file as the row it came from; only
        // the scale direction differs, which the saved scores tell apart
        // through the options.
        companion.media_path = original->identity_path();
        companion.video_info = original->video_info;
        set_row_info(new_row, *companion.video_info);
        // Drawn again with the copied choices: the row was drawn with the
        // defaults when added.
        set_row_metrics(new_row);
        start_cache_lookup({path});
        ++added;
    }
    if (added == 0) {
        inform_(app::tr("Nothing to add"),
                app::tr("Selected row(s) either already have a matching-opposite row, don't have a resolution "
                        "mismatch against the reference, or aren't a normal comparison (e.g. a resolution test)."));
        return;
    }
    show_status(ntr("Added {count} row testing the opposite scaling direction -- check it and click Calculate "
                    "metrics.",
                    "Added {count} rows testing the opposite scaling direction -- check them and click Calculate "
                    "metrics.",
                    added));
}

// --------------------------------------------------------------- CVVDP's display

void MainWindow::apply_cvvdp(const std::function<CvvdpSettings(const CvvdpSettings&)>& change,
                             const std::optional<std::vector<int>>& rows, const std::optional<std::string>& preset_name) {
    // Gives the selected rows (or `rows`) new CVVDP settings, `change(old)`,
    // and `preset_name` as the preset they were given. Only CVVDP's score
    // depends on them: a row keeps every other score, loses a CVVDP score
    // made for the old settings, and gets back one saved for the new ones,
    // if there is one.
    if (syncing_panel_ || run_active_) {
        return;
    }
    std::vector<int> changed;
    for (const int row : rows ? *rows : panel_target_rows_) {
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        const CvvdpSettings settings = change(row_data.cvvdp);
        if (preset_name && *preset_name != row_data.cvvdp_preset) {
            row_data.cvvdp_preset = *preset_name;
            set_row_metrics(row);  // the score's tooltip names the preset
        }
        if (settings.same_as(row_data.cvvdp)) {
            continue;
        }
        row_data.cvvdp = settings;
        drop_metric_results(row, {"cvvdp"});
        changed.push_back(row);
    }
    reload_saved_scores(changed);
    if (!panel_target_rows_.empty()) {
        write_panel_options(rows_[static_cast<std::size_t>(panel_target_rows_.front())]->options);
    }
}

void MainWindow::on_cvvdp_preset_chosen(int index) {
    // Gives the selected rows the display of the CVVDP preset chosen. The
    // display only: "Scale the video to fill the display" is the video's own
    // setting, which choosing a built-in preset must not switch off.
    if (syncing_panel_) {
        return;
    }
    const QVariant name = options_panel_->cvvdp_preset_combo->itemData(index);
    if (!name.isValid()) {
        return;  // "Mixed" or "Custom", which are no preset
    }
    if (const std::optional<CvvdpPreset> preset =
            cvvdp_preset_named(name.toString().toStdString(), settings_.cvvdp_presets_json)) {
        apply_cvvdp(
            [&preset](const CvvdpSettings& old) {
                CvvdpSettings settings = old;
                settings.display = preset->settings.display;
                return settings;
            },
            std::nullopt, preset->name);
    }
}

void MainWindow::on_cvvdp_resize_toggled(bool checked) {
    // Sets "Scale the video to fill the display" on the selected rows.
    apply_cvvdp([checked](const CvvdpSettings& old) {
        CvvdpSettings settings = old;
        settings.resize_to_display = checked;
        return settings;
    });
}

std::set<std::string> MainWindow::cvvdp_preset_names() const {
    // The names of every CVVDP preset, built-in and saved.
    std::set<std::string> names;
    for (const CvvdpPreset& preset : cvvdp_presets(settings_.cvvdp_presets_json)) {
        names.insert(preset.name);
    }
    return names;
}

void MainWindow::on_cvvdp_add_preset() {
    // A new preset of the user's own, starting from the first selected row's
    // display. Like every preset they save, it becomes the default for new
    // videos: someone who tunes CVVDP will keep using it.
    if (panel_target_rows_.empty() || run_active_) {
        return;
    }
    const CvvdpSettings settings = rows_[static_cast<std::size_t>(panel_target_rows_.front())]->cvvdp;
    CvvdpDisplayDialog dialog(settings.display, this, std::nullopt, true, cvvdp_preset_names());
    if (run_cvvdp_dialog_(dialog) != QDialog::Accepted) {
        return;
    }
    save_cvvdp_preset(dialog, settings, std::nullopt, false);
}

void MainWindow::on_cvvdp_edit_display() {
    // Edits the first selected row's display. Saving can rename and update
    // the user's preset it came from, or keep it as a new one; "Apply without
    // saving" only changes the selected rows.
    if (panel_target_rows_.empty() || run_active_) {
        return;
    }
    const RowData& first = *rows_[static_cast<std::size_t>(panel_target_rows_.front())];
    const CvvdpSettings settings = first.cvvdp;
    const std::optional<CvvdpPreset> match =
        matching_cvvdp_preset(settings, settings_.cvvdp_presets_json, first.cvvdp_preset);
    const std::optional<std::string> own = match && !match->builtin ? std::optional(match->name) : std::nullopt;
    CvvdpDisplayDialog dialog(settings.display, this, own, false, cvvdp_preset_names());
    if (run_cvvdp_dialog_(dialog) != QDialog::Accepted) {
        return;
    }
    if (dialog.action == "apply") {
        const CvvdpDisplay display = dialog.display();
        apply_cvvdp([&display](const CvvdpSettings& old) {
            CvvdpSettings changed = old;
            changed.display = display;
            return changed;
        });
        return;
    }
    save_cvvdp_preset(dialog, settings, dialog.action == "save" ? own : std::nullopt);
}

void MainWindow::save_cvvdp_preset(const CvvdpDisplayDialog& dialog, const CvvdpSettings& settings,
                                   const std::optional<std::string>& replacing, bool apply_to_rows) {
    // Saves the dialog's display as a user preset, and from the display
    // editor gives it to the selected rows. With `replacing`, the new preset
    // takes its place, and so renames it. A new preset becomes the default for
    // new videos, and a renamed default stays the default under its new name.
    //
    // "Add preset..." does not change the rows (apply_to_rows false): making
    // a preset is not choosing it, and applying it would drop the selected
    // video's CVVDP score.
    const std::string name = dialog.preset_name();
    const CvvdpDisplay saved = dialog.display();  // a preset is a display, not the resize choice
    std::string presets = settings_.cvvdp_presets_json;
    // The other videos using the preset being replaced, found before it
    // changes. Don't update only the selected ones: the rest would keep the
    // old values and show as "Custom", unasked.
    std::vector<int> others;
    if (replacing && apply_to_rows) {
        for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
            const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
            const std::optional<CvvdpPreset> match =
                matching_cvvdp_preset(row_data.cvvdp, presets, row_data.cvvdp_preset);
            if (std::find(panel_target_rows_.begin(), panel_target_rows_.end(), row) == panel_target_rows_.end() &&
                match && match->name == *replacing) {
                others.push_back(row);
            }
        }
    }
    if (replacing) {
        presets = without_user_preset(presets, *replacing);
    }
    try {
        settings_.cvvdp_presets_json = with_user_preset(presets, name, saved);
    } catch (const std::invalid_argument& error) {
        QMessageBox::warning(this, app::tr("CVVDP preset"), tr_message(QString::fromStdString(error.what())));
        return;
    }
    if (!replacing || settings_.cvvdp_default_preset == *replacing) {
        settings_.cvvdp_default_preset = name;
    }
    default_cvvdp_ = cvvdp_from_settings();
    const std::optional<std::string> error = settings_.save();
    const auto with_saved_display = [&saved](const CvvdpSettings& old) {
        CvvdpSettings changed = old;
        changed.display = saved;
        return changed;
    };
    if (apply_to_rows) {
        apply_cvvdp(with_saved_display, std::nullopt, name);
    }
    if (!others.empty()) {
        if (saved.identity() == settings.display.identity()) {
            // Renamed only: the videos keep their values, and the name.
            apply_cvvdp([](const CvvdpSettings& old) { return old; }, others, name);
        } else if (QMessageBox::question(
                       this, app::tr("CVVDP preset"),
                       ntr("{count} other video uses your preset \"{name}\". Update it to the saved values too?\n\n"
                           "A CVVDP score made for the old values is cleared; one already saved for the new values "
                           "is shown instead. Choose No to keep the old values (shown as Custom).",
                           "{count} other videos use your preset \"{name}\". Update them to the saved values too?\n\n"
                           "A CVVDP score made for the old values is cleared; one already saved for the new values "
                           "is shown instead. Choose No to keep the old values (shown as Custom).",
                           static_cast<long long>(others.size()), {{"name", QString::fromStdString(*replacing)}})) ==
                   QMessageBox::Yes) {
            apply_cvvdp(with_saved_display, others, name);
        }
    }
    // apply_cvvdp shows the panel again only when a row changed; a rename
    // changes no row but does change the dropdown.
    if (!panel_target_rows_.empty()) {
        write_panel_options(rows_[static_cast<std::size_t>(panel_target_rows_.front())]->options);
    }
    const QString quoted = QString::fromStdString(name);
    QString done;
    if (replacing) {
        done = name == *replacing ? app::tr("Saved your CVVDP preset \"{name}\".", {{"name", quoted}})
                                  : app::tr("Renamed your CVVDP preset \"{old}\" to \"{name}\" and saved it.",
                                            {{"old", QString::fromStdString(*replacing)}, {"name", quoted}});
    } else if (apply_to_rows) {
        done = app::tr("Saved the CVVDP preset \"{name}\". Videos added from now on use it (change that in Settings).",
                       {{"name", quoted}});
    } else {
        done = app::tr("Saved the CVVDP preset \"{name}\". Videos added from now on use it (change that in "
                       "Settings); choose it in the CVVDP display list to use it for the selected videos.",
                       {{"name", quoted}});
    }
    show_status(error ? tr_message(QString::fromStdString(*error)) : done);
}

void MainWindow::on_cvvdp_delete_preset() {
    // Deletes the user's chosen CVVDP preset, after asking.
    const QVariant chosen = options_panel_->cvvdp_preset_combo->currentData();
    const std::string name = chosen.isValid() ? chosen.toString().toStdString() : std::string();
    const std::optional<CvvdpPreset> preset =
        name.empty() ? std::nullopt : cvvdp_preset_named(name, settings_.cvvdp_presets_json);
    if (!preset || preset->builtin) {
        return;
    }
    const QString quoted = QString::fromStdString(name);
    if (QMessageBox::question(
            this, app::tr("Delete CVVDP preset"),
            app::tr("Delete your preset \"{name}\"?\n\nVideos using it keep their settings.", {{"name", quoted}})) !=
        QMessageBox::Yes) {
        return;
    }
    settings_.cvvdp_presets_json = without_user_preset(settings_.cvvdp_presets_json, name);
    if (settings_.cvvdp_default_preset == name) {
        settings_.cvvdp_default_preset.clear();
    }
    default_cvvdp_ = cvvdp_from_settings();
    const std::optional<std::string> error = settings_.save();
    if (!panel_target_rows_.empty()) {
        write_panel_options(rows_[static_cast<std::size_t>(panel_target_rows_.front())]->options);
    }
    show_status(error ? tr_message(QString::fromStdString(*error))
                      : app::tr("Deleted the CVVDP preset \"{name}\".", {{"name", quoted}}));
}

}  // namespace vml::app
