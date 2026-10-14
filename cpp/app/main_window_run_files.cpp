// The window's saved runs: rows' results saved to files of their own, and a
// saved run loaded into the table (the Python app's _on_save_selected and
// _on_load_saved_run).
#include "main_window.h"

#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidgetItem>

#include <set>

#include "file_writes.h"
#include "formatting.h"
#include "i18n.h"
#include "vml/export_csv.h"
#include "vml/model_select.h"
#include "vml/run_file.h"
#include "vml/time_format.h"
#include "widgets.h"
#include "window_helpers.h"

// The window's text is app::tr's (i18n.h), always written so: see
// main_window.cpp.
namespace vml::app {

namespace {

// The file dialogs' filter for saved runs.
QString run_file_filter() {
    return app::tr("Analysis results (*{suffix})",
                   {{"suffix", QString::fromUtf8(result_suffix.data(), static_cast<int>(result_suffix.size()))}});
}

// A CVVDP score's display, as its provenance records it; nothing for a
// record CVVDP cannot read.
std::optional<CvvdpSettings> cvvdp_display_of(const SequenceMetricResult& cvvdp) {
    try {
        nlohmann::json parameters = nlohmann::json::object();
        for (const auto& [name, text] : cvvdp.provenance.parameters) {
            parameters[name] = nlohmann::json::parse(text);
        }
        return CvvdpSettings::from_parameters(parameters);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }
}

}  // namespace

void MainWindow::on_load_saved_run() {
    // Asks for a saved run and loads it into the table.
    const QString chosen = open_file_(app::tr("Load analysis results"), run_file_filter());
    if (!chosen.isEmpty()) {
        load_saved_run(path_of(chosen));
    }
}

void MainWindow::load_saved_run(const std::filesystem::path& path) {
    // A saved run in the table: a row of its own, unticked, showing its
    // scores with the settings that made them.
    std::pair<ComparisonResult, std::string> loaded;
    try {
        loaded = load_run(path);
    } catch (const RunFileError& error) {
        alert_(app::tr("Failed to load run"), tr_message(QString::fromStdString(error.what())));
        return;
    }
    const ComparisonResult& result = loaded.first;
    // A result belongs to a pair of reference and test video: under another
    // reference it would show a comparison never made. And one row per
    // video: with two rows for a path, every lookup by path (saved scores,
    // reads, a run's result) finds only the first.
    if (row_index_of_path(result.distorted) &&
        !ask_(app::tr("Already added"), app::tr("{name} is already in the list. Show the saved run's scores in its "
                                                "place?",
                                                {{"name", name_of(result.distorted)}}))) {
        return;
    }
    if (!source_info_) {
        // Nothing to contradict: the run's own reference is taken, so that
        // the window and the result agree about what was compared.
        adopt_source_from_run(result);
    } else if (!same_file(source_info_->path, result.source) && !offer_to_switch_source(result)) {
        return;
    }
    if (const std::optional<int> existing = row_index_of_path(result.distorted)) {
        distorted_table_->removeRow(*existing);
        rows_.erase(rows_.begin() + *existing);
    }

    auto run = std::make_shared<CompletedRun>();
    run->result = result;
    run->label = QString::fromStdString(loaded.second);
    const int row = add_table_row(result.distorted);
    RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    row_data.video_info = result.distorted_info;
    row_data.completed_run = run;
    row_data.analysis_status.reset();
    // The metric columns follow the row's options: they are set from the
    // scores the run holds, or its PSNR, SSIM and XPSNR would show N/A.
    VmafOptions& options = row_data.options;
    const FrameScores frames = frame_scores_from_results(result.metric_results);
    const auto has = [&frames](const char* key) { return frames.metrics.contains(key); };
    options.extra_features.clear();
    if (has("psnr")) {
        options.extra_features.emplace_back("name=psnr");
    }
    if (has("ssim")) {
        options.extra_features.emplace_back("name=float_ssim");
    }
    options.compute_xpsnr = has("xpsnr");
    options.compute_vmaf = has("vmaf");
    options.compute_vmaf_neg = has("vmaf_neg");
    options.compute_vmaf_v1 = has("vmaf_v1");
    if (result.model_choice_v1 && !result.model_choice_v1->empty()) {
        options.model_choice_v1 = *result.model_choice_v1;
    }
    options.model = result.model;
    // The model chosen too: without it, a later lookup or calculation would
    // put the row back to Auto. An older file has none, and its two forms
    // are told apart by the model.
    if (result.model_choice && !result.model_choice->empty()) {
        options.model_choice = *result.model_choice;
        if (options.model_choice == custom_model_choice && result.model.starts_with("path=")) {
            options.custom_model_path = result.model.substr(std::string_view("path=").size());
        }
    } else if (result.model.starts_with("version=")) {
        options.model_choice = result.model;
    } else if (result.model.starts_with("path=")) {
        options.model_choice = std::string(custom_model_choice);
        options.custom_model_path = result.model.substr(std::string_view("path=").size());
    }
    if (!options.model_choice.empty() && is_builtin_model_choice(options.model_choice)) {
        // A VMAF v1 model as the VMAF model, from before VMAF v1 had its own
        // column: it is VMAF v1's choice now, and VMAF v0.6.1 is on Auto.
        options.model_choice_v1 = options.model_choice;
        options.model_choice = std::string(auto_model_choice);
    }
    options.scale_direction = result.scale_direction;
    options.scale_algorithm = result.scale_algorithm;
    options.resample_test = result.resample_target;
    // A CVVDP score means something only with the display it was made for:
    // the row takes that display, so that the score and the row agree.
    if (const SequenceMetricResult* cvvdp = result.metric_results.sequence("cvvdp")) {
        row_data.extra_metric_keys.insert("cvvdp");
        if (const std::optional<CvvdpSettings> display = cvvdp_display_of(*cvvdp)) {
            row_data.cvvdp = *display;
        }
    }
    if (result.resample_target) {
        row_data.media_path = result.source_info.path;  // what the test decodes
    }
    distorted_table_->item(row, col_check)->setCheckState(Qt::Unchecked);
    if (result.resample_target) {
        // As one added or found in the saved scores: what the test scales the
        // reference to, not the reference's own details and bitrate.
        set_resample_row_info(row, *result.resample_target);
    } else {
        set_row_info(row, result.distorted_info);
    }
    set_row_metrics(row);
    row_data.status_detail = app::tr("Loaded from saved run");
    refresh_row_state(row);
}

void MainWindow::adopt_source_from_run(const ComparisonResult& result) {
    // Makes the reference a loaded run was measured against the window's, as
    // choosing it would: the rows are checked against it again.
    source_info_ = result.source_info;
    source_edit_->setText(text_of(result.source));
    const VideoInfo& info = result.source_info;
    source_info_label_->setText(app::tr("{media}, {bitrate}  ({duration})  [from saved run]",
                                        {{"media", media_info_string(info)},
                                         {"bitrate", bitrate_string(info)},
                                         {"duration", QString::fromStdString(format_hms(info.duration, 1))}}));
    check_rows_against_new_source(result.source);
}

bool MainWindow::offer_to_switch_source(const ComparisonResult& result) {
    // Asks before showing a run measured against another reference, and
    // whether to go ahead: yes makes the run's reference the window's, the
    // only way the table and the result mean the same thing.
    if (!ask_(app::tr("Different reference video"),
              app::tr("This saved run was measured against:\n    {source}\n\nbut the selected reference is:\n    "
                      "{path}\n\nScores from two different references cannot be compared. Switch the reference to "
                      "the one this run used?",
                      {{"source", text_of(result.source)}, {"path", text_of(source_info_->path)}}))) {
        return false;
    }
    adopt_source_from_run(result);
    return true;
}

std::vector<std::shared_ptr<CompletedRun>> MainWindow::selected_runs() const {
    // The results of the selected rows that have one, top to bottom.
    std::set<int> rows;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        rows.insert(index.row());
    }
    std::vector<std::shared_ptr<CompletedRun>> runs;
    for (const int row : rows) {
        if (const std::shared_ptr<CompletedRun>& run = rows_[static_cast<std::size_t>(row)]->completed_run) {
            runs.push_back(run);
        }
    }
    return runs;
}

void MainWindow::on_save_selected() {
    // Saves the selected rows' results to files: one where the user says,
    // several into a folder, each named after its run.
    const std::vector<std::shared_ptr<CompletedRun>> runs = selected_runs();
    if (runs.empty()) {
        inform_(app::tr("Nothing selected"), app::tr("Select one or more completed rows to save."));
        return;
    }
    const QString suffix = QString::fromUtf8(result_suffix.data(), static_cast<int>(result_suffix.size()));
    if (runs.size() == 1) {
        const QString chosen = save_file_(app::tr("Save analysis results"), runs.front()->label + suffix,
                                          run_file_filter());
        if (!chosen.isEmpty()) {
            submit_save(runs.front()->result, path_of(chosen), runs.front()->label);
        }
        return;
    }
    const QString folder = choose_folder_(app::tr("Choose folder to save runs into"));
    if (folder.isEmpty()) {
        return;
    }
    std::set<std::filesystem::path> reserved;
    for (const std::shared_ptr<CompletedRun>& run : runs) {
        submit_save(run->result,
                    unique_output_path(path_of(folder), run->label.toStdString(), suffix.toStdString(), reserved),
                    run->label);
    }
}

void MainWindow::submit_save(const ComparisonResult& result, const std::filesystem::path& path, const QString& label) {
    // Saves a result off the window's thread: a long video's takes a while.
    // Only saving waits for it, so the rest of the window stays usable.
    save_button_->setEnabled(false);
    file_writes_->submit(QStringLiteral("save %1").arg(name_of(path)),
                         [result, path, label = label.toStdString()] { save_run(result, path, label); });
}

}  // namespace vml::app
