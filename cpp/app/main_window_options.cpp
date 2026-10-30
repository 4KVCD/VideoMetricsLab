// The window's Options panel part (the Python app's MainWindow, its
// "per-video settings panel"): the selected rows' settings shown in the
// panel, and each control the user edits applied to them.
#include <QComboBox>
#include <QLabel>
#include <QMessageBox>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cvvdp_display_dialog.h"
#include "i18n.h"
#include "main_window.h"
#include "options_panel.h"
#include "settings_panel.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/cvvdp.h"
#include "vml/film_grain_scoring.h"
#include "vml/isolated.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/vship.h"

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
        options_panel_->show_vmaf_backend(default_options_.vmaf_on_gpu);
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
    options_panel_->show_options(options, selected);
    const std::vector<std::string> requested = first != nullptr ? requested_metrics(*first) : std::vector<std::string>();
    for (const MetricColumn& metric : metric_columns()) {
        metric_header_->set_checked(metric.column, first != nullptr ? std::find(requested.begin(), requested.end(),
                                                                                metric.key) != requested.end()
                                                                    : default_metric_ticked(metric.column));
    }
    std::vector<CvvdpSettings> displays;
    bool automatic = !panel_target_rows_.empty();
    for (const int row : panel_target_rows_) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        displays.push_back(row_data.cvvdp);
        automatic = automatic && row_data.cvvdp_automatic;
    }
    std::string preferred = first != nullptr ? first->cvvdp_preset : std::string();
    if (displays.empty()) {
        // No video selected: what a new one starts with.
        const CvvdpPreset preset = new_row_cvvdp_preset();
        CvvdpSettings settings;
        settings.display = preset.settings.display;
        displays.push_back(settings);
        preferred = preset.name;
        automatic = cvvdp_default_automatic();
    }
    options_panel_->show_cvvdp(displays, preferred, settings_.cvvdp_presets_json, automatic);
    show_panel_film_grain();
    syncing_panel_ = false;
}

void MainWindow::show_panel_film_grain() {
    // The selected rows' videos' film grain choice.
    const bool syncing = std::exchange(syncing_panel_, true);
    std::vector<FilmGrainVideo> videos;
    for (const int row : panel_target_rows_) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        videos.push_back({row_data.video_info ? &*row_data.video_info : nullptr,
                          row_data.options.resample_test.has_value()});
    }
    options_panel_->show_film_grain(videos);
    syncing_panel_ = syncing;
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
        } else if (field == "vship_build") {
            target.vship_build = panel.vship_build;
        }
        return !(target == before);
    };
    apply(default_options_);
    if (field == "vmaf_on_gpu") {
        // The last choice is what the next session starts with.
        settings_.default_vmaf_on_gpu = panel.vmaf_on_gpu;
        save_settings();
    }
    if (field == "vship_build" && !vship_probe_ready(panel.vship_build)) {
        // Its GPU check, for the rows to say what it can calculate: a build
        // other than the default is checked only when a row chooses it.
        (void)QtConcurrent::run([build = panel.vship_build] { detect_vship_device(build, gpu_probes_token()); });
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
    if (field == "model" || field == "model_v1" || field == "vship_build") {
        // A model decides its own VMAF column's score only, and the Vship
        // build Vship's metrics' scores: the rest stay.
        const std::vector<std::string> decided = field == "model"      ? std::vector<std::string>{"vmaf"}
                                                 : field == "model_v1" ? std::vector<std::string>{"vmaf_v1"}
                                                 : std::vector<std::string>{"ssimulacra2", "butteraugli", "cvvdp"};
        std::vector<int> without_scores;
        for (const int row : changed) {
            if (rows_[static_cast<std::size_t>(row)]->completed_run) {
                drop_metric_results(row, decided);
            } else {
                without_scores.push_back(row);
            }
        }
        invalidate_completed_results(without_scores);
    } else {
        invalidate_completed_results(changed);
    }
    for (const int row : applied) {
        set_row_black_bars(row);
    }
    reload_saved_scores(changed);
}

void MainWindow::on_model_changed(int index) {
    // Sets the selected rows' VMAF model, asking for the file of a custom one.
    if (syncing_panel_) {
        return;
    }
    if (model_choices()[static_cast<std::size_t>(index)].choice == custom_model_choice) {
        const QString path = dialogs_.open_file(app::tr("Select VMAF model file (.json)"), QString());
        if (path.isEmpty()) {
            // Cancelled: the model stays the one the rows have, and so do
            // their scores.
            const VmafOptions& current = panel_target_rows_.empty()
                                             ? default_options_
                                             : rows_[static_cast<std::size_t>(panel_target_rows_.front())]->options;
            const std::vector<ModelChoice>& choices = model_choices();
            const auto found = std::find_if(choices.begin(), choices.end(), [&current](const ModelChoice& item) {
                return item.choice == current.model_choice;
            });
            syncing_panel_ = true;
            options_panel_->model_combo->setCurrentIndex(
                found == choices.end() ? 0 : static_cast<int>(found - choices.begin()));
            syncing_panel_ = false;
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
        // The same comparison in the other direction: the same metric ticks
        // and CVVDP display. Don't take the defaults: where they leave CVVDP
        // out, a video with CVVDP ticked would get a companion without it.
        companion.extra_metric_keys = original->extra_metric_keys;
        companion.cvvdp = original->cvvdp;
        companion.cvvdp_preset = original->cvvdp_preset;
        companion.cvvdp_automatic = original->cvvdp_automatic;
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
        dialogs_.inform(
            app::tr("Nothing to add"),
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
                             const std::optional<std::vector<int>>& rows, const std::optional<std::string>& preset_name,
                             std::optional<bool> automatic) {
    // Gives the selected rows (or `rows`) new CVVDP settings, `change(old)`,
    // and `preset_name` as the preset they were given; with `automatic`,
    // whether their display follows the reference from now on. Only CVVDP's
    // score depends on them: a row keeps every other score, loses a CVVDP
    // score made for the old settings, and gets back one saved for the new
    // ones, if there is one.
    if (syncing_panel_ || run_active_) {
        return;
    }
    std::vector<int> changed;
    for (const int row : rows ? *rows : panel_target_rows_) {
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        if (automatic) {
            row_data.cvvdp_automatic = *automatic;
        }
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
    // "Automatic" (an empty name, which no preset has): the display follows
    // the reference again.
    const bool automatic = name.toString().isEmpty();
    const std::optional<CvvdpPreset> preset =
        automatic ? std::optional(automatic_cvvdp_preset(source_info_ ? &*source_info_ : nullptr))
                  : cvvdp_preset_named(name.toString().toStdString(), settings_.cvvdp_presets_json);
    if (preset) {
        apply_cvvdp(
            [&preset](const CvvdpSettings& old) {
                CvvdpSettings settings = old;
                settings.display = preset->settings.display;
                return settings;
            },
            std::nullopt, preset->name, automatic);
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
        apply_cvvdp(
            [&display](const CvvdpSettings& old) {
                CvvdpSettings changed = old;
                changed.display = display;
                return changed;
            },
            std::nullopt, std::nullopt, false);
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
    const std::optional<std::string> error = settings_.save();
    settings_panel_->fill_cvvdp_default_combo();
    const auto with_saved_display = [&saved](const CvvdpSettings& old) {
        CvvdpSettings changed = old;
        changed.display = saved;
        return changed;
    };
    if (apply_to_rows) {
        apply_cvvdp(with_saved_display, std::nullopt, name, false);
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
        settings_.cvvdp_default_preset.clear();  // Automatic
    }
    const std::optional<std::string> error = settings_.save();
    settings_panel_->fill_cvvdp_default_combo();
    if (!panel_target_rows_.empty()) {
        write_panel_options(rows_[static_cast<std::size_t>(panel_target_rows_.front())]->options);
    }
    show_status(error ? tr_message(QString::fromStdString(*error))
                      : app::tr("Deleted the CVVDP preset \"{name}\".", {{"name", quoted}}));
}

// ------------------------------------------------------------- film grain

std::vector<std::filesystem::path> MainWindow::choose_film_grain(const std::vector<int>& rows, bool reference,
                                                                 bool with) {
    // Scores the videos of `rows` (a resolution test's is the reference) and,
    // with `reference`, the reference with their film grain or without. A
    // row whose saved scores are now filed otherwise (its lookup token: "with"
    // and no choice are the same scores) loses the scores it shows; returns
    // those rows, to be looked up again.
    std::vector<int> affected = rows;
    if (reference) {
        affected.resize(rows_.size());
        std::iota(affected.begin(), affected.end(), 0);  // every row is compared with it
    }
    const auto token = [this](int row) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        return source_info_ ? lookup_token(source_info_->path, row_data.identity_path(), analysis_request_of(row_data))
                            : std::string();
    };
    std::vector<std::string> before;
    for (const int row : affected) {
        before.push_back(token(row));
    }
    if (reference && source_info_) {
        source_info_->with_film_grain = with;
        for (const std::shared_ptr<RowData>& row_data : rows_) {
            if (row_data->options.resample_test && row_data->video_info) {
                take_film_grain(*row_data->video_info, *source_info_);  // its copy of the reference
            }
        }
        show_source_label();
    }
    for (const int row : rows) {
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        if (!row_data.options.resample_test && row_data.video_info && film_grain_choosable(*row_data.video_info)) {
            row_data.video_info->with_film_grain = with;
            set_row_info(row, row_data.video_info);  // its Media info tag
        }
    }
    std::vector<int> changed;
    std::vector<std::filesystem::path> paths;
    for (std::size_t i = 0; i < affected.size(); ++i) {
        if (token(affected[i]) != before[i]) {
            changed.push_back(affected[i]);
            paths.push_back(rows_[static_cast<std::size_t>(affected[i])]->path);
        }
    }
    invalidate_completed_results(changed);
    sync_frame_compare();  // the viewers show the videos as they are scored
    return paths;
}

std::vector<std::filesystem::path> MainWindow::apply_film_grain_default() {
    // Gives the videos with a film grain choice to make and none made yet
    // (the reference too) Settings' default, unless it is to ask. Returns the
    // rows whose saved scores are to be looked up again (choose_film_grain).
    const std::optional<bool> with = film_grain_default_choice(settings_.film_grain_default);
    if (!with) {
        return {};
    }
    const auto unchosen = [](const VideoInfo& info) { return film_grain_choosable(info) && !info.with_film_grain; };
    std::vector<int> rows;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        if (!row_data.options.resample_test && row_data.video_info && unchosen(*row_data.video_info)) {
            rows.push_back(row);
        }
    }
    const bool reference = source_info_ && unchosen(*source_info_);
    if (rows.empty() && !reference) {
        return {};
    }
    return choose_film_grain(rows, reference, *with);
}

void MainWindow::on_film_grain_chosen(bool with) {
    // The Options panel's Film grain, for the selected rows.
    if (syncing_panel_ || run_active_) {
        return;
    }
    start_cache_lookup(choose_film_grain(panel_target_rows_, false, with));
    show_panel_film_grain();
}

}  // namespace vml::app
