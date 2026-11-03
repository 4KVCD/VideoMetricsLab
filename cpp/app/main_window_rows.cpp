// The window's test videos: their rows and cells (status, black bars,
// sizes, scores), adding and reading them, resample tests, and the rows'
// order, sorting, removal and best/worst highlighting.
#include "main_window.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFont>
#include <QFormLayout>
#include <QHeaderView>
#include <QItemSelection>
#include <QItemSelectionModel>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <tuple>
#include <utility>
#include <variant>

#include "formatting.h"
#include "graph_panel.h"
#include "i18n.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/film_grain_scoring.h"
#include "vml/geometry.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/python_text.h"
#include "vml/score_identity.h"
#include "vml/time_format.h"
#include "vml/vship.h"
#include "vship_names.h"
#include "widgets.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

// A metric cell's score, as compared by Highlight best/worst results:
// nothing where it shows none to compare.
constexpr int score_role = Qt::UserRole;

// "1920x1080".
QString size_text(int width, int height) { return QStringLiteral("%1x%2").arg(width).arg(height); }
// A file name split into its runs of digits and of other text, for sorting
// "clip 9" before "clip 10" (the Python app's re.split(r"(\d+)", name)): the
// text casefolded, the digits compared as numbers.
std::vector<QString> natural_parts(const QString& name) {
    std::vector<QString> parts{QString()};
    bool digits = false;
    for (const QChar c : name) {
        if (c.isDigit() != digits) {
            digits = c.isDigit();
            parts.emplace_back();
        }
        parts.back() += c;
    }
    for (std::size_t i = 0; i < parts.size(); i += 2) {
        parts[i] = parts[i].toCaseFolded();
    }
    return parts;
}

// A run of digits as the number Python's int() reads in it, of any length
// and in any script's digits: its digits' values, without leading zeros.
std::vector<int> digit_values(const QString& digits) {
    std::vector<int> values;
    for (const QChar c : digits) {
        if (!values.empty() || c.digitValue() != 0) {
            values.push_back(c.digitValue());
        }
    }
    return values;
}

// Python's comparison of the parts' lists: the text parts (even places) as
// text, the digit parts (odd places) as whole numbers.
bool natural_less(const std::vector<QString>& a, const std::vector<QString>& b) {
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (i % 2 == 0) {
            if (const int order = QString::compare(a[i], b[i]); order != 0) {
                return order < 0;
            }
            continue;
        }
        const std::vector<int> x = digit_values(a[i]);
        const std::vector<int> y = digit_values(b[i]);
        if (x.size() != y.size()) {
            return x.size() < y.size();
        }
        if (x != y) {
            return x < y;
        }
    }
    return a.size() < b.size();
}

// What a column sorts a row by: one kind for every row of a column.
using SortKey = std::variant<std::vector<QString>, std::tuple<long long, double, double>, long long, int, QString,
                             double>;

bool sort_less(const SortKey& a, const SortKey& b) {
    if (const auto* x = std::get_if<std::vector<QString>>(&a)) {
        return natural_less(*x, std::get<std::vector<QString>>(b));
    }
    if (const auto* x = std::get_if<QString>(&a)) {
        return QString::compare(*x, std::get<QString>(b)) < 0;
    }
    return a < b;
}

}  // namespace

// --------------------------------------------------------- the test videos

int MainWindow::add_table_row(const std::filesystem::path& path) {
    // Rows are added at the end, so the table is not in the order a heading's
    // sort arrow says any more.
    clear_sort_indicator();
    const int row = distorted_table_->rowCount();
    distorted_table_->insertRow(row);

    auto* check_item = new QTableWidgetItem;
    check_item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    check_item->setCheckState(Qt::Checked);
    distorted_table_->setItem(row, col_check, check_item);

    auto* path_item = new QTableWidgetItem(name_of(path));
    path_item->setToolTip(text_of(path));  // the full path on hover
    distorted_table_->setItem(row, col_path, path_item);
    distorted_table_->setItem(row, col_info, new QTableWidgetItem(app::tr("Probing...")));
    auto* bars_item = new QTableWidgetItem;
    bars_item->setTextAlignment(Qt::AlignCenter);
    distorted_table_->setItem(row, col_black_bars, bars_item);
    distorted_table_->setItem(row, col_scaling, new QTableWidgetItem(QString()));
    distorted_table_->setItem(row, col_bitrate, new QTableWidgetItem(QString()));
    for (const MetricColumn& item : metric_columns()) {
        auto* cell = new QTableWidgetItem(QString());
        cell->setTextAlignment(Qt::AlignCenter);
        // Coloured by the best/worst highlighting alone from here on.
        cell->setBackground(QColor(0, 0, 0, 0));
        distorted_table_->setItem(row, item.column, cell);
    }

    // New rows start with a copy of the defaults, so that several similar
    // files need not each be set up from scratch. Each row has its own copy,
    // so editing one row never changes another.
    auto row_data = std::make_shared<RowData>();
    row_data->path = path;
    row_data->options = default_options_;
    row_data->extra_metric_keys = default_extra_metric_keys_;
    // A preset is a display; scaling to fill it is each row's own choice and
    // starts off.
    const CvvdpPreset preset = new_row_cvvdp_preset();
    row_data->cvvdp.display = preset.settings.display;
    row_data->cvvdp_preset = preset.name;
    row_data->cvvdp_automatic = cvvdp_default_automatic();
    rows_.push_back(row_data);
    set_row_metrics(row);
    return row;
}

void MainWindow::refresh_metric_cells() {
    // Every row's metric cells again, after a GPU check: which metrics this
    // PC can calculate, and which saved scores answer the rows, depend on it.
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        set_row_metrics(row);
    }
    schedule_best_worst();
}

void MainWindow::on_vship_checked() {
    // A Vship probe finished: the rows say what can be calculated now. Where
    // Vship-fast turns out to use no GPU and 5.1.2 one, the rows set to
    // Vship-fast look their saved scores up again, as 5.1.2's
    // (vship_build_in_use): a lookup made before the probe asked for
    // Vship-fast's. Not during a run, whose rows take its scores; the next
    // probe to finish tries again.
    refresh_metric_cells();
    const bool vship51_in_use = vship_build_in_use(VshipBuild::fast) == VshipBuild::stock;
    if (vship51_in_use && !vship51_in_use_) {
        if (run_active_) {
            return;
        }
        std::vector<int> rows;
        for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
            if (rows_[static_cast<std::size_t>(row)]->options.vship_build == VshipBuild::fast) {
                rows.push_back(row);
            }
        }
        reload_saved_scores(rows);
    }
    vship51_in_use_ = vship51_in_use;
}

void MainWindow::set_row_metrics(int row) {
    // Each metric cell shows its score, or a tick box for calculating it. A
    // metric that has been measured shows only the number: there is no
    // choice left to make about it, and a tick box beside a score invites
    // unticking it as though that would undo the measurement. A metric
    // without a score shows a tick box in the column its result will go in,
    // so choosing what to calculate is one click on the row. Scores stay when
    // the ticks change: ticking another metric changes what is asked for,
    // not what was measured.
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    CompletedRun* run = row_data.completed_run.get();
    // setCheckState/setData below emit itemChanged, which is also how a real
    // click arrives: without this the window would read its own repaint back
    // as the user asking for a change.
    syncing_table_ = true;
    for (const MetricColumn& metric : metric_columns()) {
        QTableWidgetItem* item = distorted_table_->item(row, metric.column);
        if (item == nullptr) {
            continue;
        }
        const std::optional<double> score = run != nullptr ? score_of(*run, metric.column) : std::nullopt;
        if (score) {
            show_score_cell(*item, row_data, *run, metric, *score);
        } else if (const std::optional<QString> unavailable = metric_unavailable_reason(row_data, metric.key)) {
            show_unavailable_cell(*item, *unavailable);
        } else {
            show_tick_cell(*item, row_data, metric, row_metric_enabled(row_data, metric.column));
        }
    }
    syncing_table_ = false;
    schedule_best_worst();
    refresh_row_state(row);
    set_row_black_bars(row);
    set_row_scaling(row);
}

// A measured metric's cell: its score (with the other implementation's
// name when it is on another scale), and no tick box.
void MainWindow::show_score_cell(QTableWidgetItem& item, const RowData& row_data, CompletedRun& run,
                                 const MetricColumn& metric, double score) {
    // Measured: the score replaces the tick box. Taking the check
    // state's value away is what removes the box: Qt draws one for
    // any item that has the role set at all.
    item.setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    item.setData(Qt::CheckStateRole, QVariant());
    const MetricDefinition& definition = metric_definition(metric.key);
    const FrameMetricResult* frame_metric = run.result.metric_results.frame(metric.key);
    const SequenceMetricResult* sequence_metric = run.result.metric_results.sequence(metric.key);
    QString text = qt_text(definition.format_value(score));
    const bool other_scale =
        frame_metric != nullptr && !backend_matches(row_data, metric.key, frame_metric->provenance);
    const MetricProvenance* provenance = frame_metric != nullptr      ? &frame_metric->provenance
                                         : sequence_metric != nullptr ? &sequence_metric->provenance
                                                                      : nullptr;
    const std::optional<VshipBuild> other_build =
        provenance != nullptr ? other_vship_build(row_data, *provenance) : std::nullopt;
    if (other_scale) {
        // Visible without hovering: a score from the other
        // implementation is on a different scale from the rest.
        text += " (" + QString::fromStdString(frame_metric->provenance.compute_backend).toUpper() + ")";
    } else if (other_build) {
        text += vship_build_mark(*other_build);
    }
    item.setText(text);
    // Compared with the others' unless on another scale or by another build.
    item.setData(score_role, other_scale || other_build ? QVariant() : QVariant(score));
    const QString build_note =
        other_build ? other_vship_build_note(*other_build, vship_build_in_use(row_data.options.vship_build))
                    : QString();
    item.setToolTip(definition.kind == MetricKind::sequence
                        ? cvvdp_note(row_data) + build_note
                        : app::tr("Mean of calculated frame scores.") + identical_frames_note(run, metric.column) +
                              (other_scale ? other_scale_note() : build_note));
    QFont font;
    font.setBold(true);
    item.setFont(font);
    theme::foreground(&item, std::nullopt);
}

// The cell of a metric the row cannot have (`reason`): "n/a", and no tick box.
void MainWindow::show_unavailable_cell(QTableWidgetItem& item, const QString& reason) {
    // No tick box: there is nothing to choose. The row's own selection
    // is left alone for when it applies again.
    item.setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    item.setData(Qt::CheckStateRole, QVariant());
    item.setText("n/a");
    item.setToolTip(reason);
    item.setForeground(theme::color("faint"));
    item.setData(score_role, QVariant());
    item.setFont(QFont());
}

// A metric's tick box, ticked when `enabled`, with "Failed" when it failed
// on the last run, and why.
void MainWindow::show_tick_cell(QTableWidgetItem& item, const RowData& row_data, const MetricColumn& metric,
                                bool enabled) {
    item.setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    item.setCheckState(enabled ? Qt::Checked : Qt::Unchecked);
    const bool failed = enabled && (row_data.analysis_status == RowState::failed ||
                                     row_data.analysis_status == RowState::partly_failed);
    item.setText(failed ? app::tr("Failed") : QString());
    QString tooltip;
    if (failed) {
        QString reason = row_data.metric_failures.contains(metric.key) ? row_data.metric_failures.at(metric.key)
                                                                       : QString();
        if (reason.isEmpty() && row_data.analysis_status == RowState::failed) {
            reason = row_data.status_detail.section("\n\n", 0, 0);  // the whole video failed: its reason
        }
        tooltip = (reason.isEmpty()
                       ? app::tr("This metric failed on the last run.")
                       : app::tr("Failed on the last run: {reason}", {{"reason", tr_message(reason)}})) +
                  app::tr("\n\nLeft ticked, it is calculated again on the next run; untick to skip it.\nThe full "
                          "details are in the log (Settings > Storage > Log files).");
    } else {
        tooltip = enabled ? app::tr("Ticked: calculated on the next run. Untick to skip it.")
                          : app::tr("Not selected. Tick to calculate this metric.");
    }
    if (std::string_view(metric.key) == "cvvdp") {
        tooltip += cvvdp_elsewhere_note(row_data);
    }
    item.setToolTip(tooltip);
    theme::foreground(&item, failed ? std::optional(theme::color("failed")) : std::nullopt);
    item.setData(score_role, QVariant());
    item.setFont(QFont());
}

RowState MainWindow::row_state(const RowData& row_data) const {
    // How the row's analysis stands, in one phrase. Above all what the
    // metric cells cannot say: which stage a live job is at, and why a
    // finished job is not shown.
    if (row_data.analysis_status) {
        return *row_data.analysis_status;
    }
    if (requested_metrics(row_data).empty()) {
        return RowState::no_metrics;
    }
    if (has_requested_results(row_data)) {
        return RowState::complete;
    }
    return row_data.completed_run ? RowState::partial : RowState::not_calculated;
}

void MainWindow::refresh_row_state(int row) {
    // The row's state on its file name: the full path, then the state and
    // any detail, with a colour for the two cases nothing else on the row
    // shows.
    QTableWidgetItem* item = distorted_table_->item(row, col_path);
    if (item == nullptr) {
        return;
    }
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    const RowState state = row_state(row_data);
    QStringList lines{text_of(row_data.path), row_state_shown(state)};
    if (!row_data.status_detail.isEmpty()) {
        lines.append(tr_message(row_data.status_detail));
    }
    item->setToolTip(lines.join("\n\n"));
    if (state == RowState::failed) {
        item->setForeground(theme::color("failed"));
    } else if (finished_not_shown(state)) {
        // Done and saved, but on purpose not shown: without a mark the row
        // would look untouched.
        item->setForeground(theme::color("stale"));
    } else {
        theme::foreground(item, std::nullopt);
    }
}

QString MainWindow::crop_detail(const QString& label, const VideoInfo& info, const std::optional<CropBox>& crop) {
    // One video's crop, spelled out for the Black bars cell's tooltip.
    if (!crop) {
        return app::tr("{label}: not checked for black bars; no crop was applied.", {{"label", label}});
    }
    // In the pixels it was compared at: a video whose size changes is scaled
    // to one size first (VideoInfo::scaled_to).
    const auto [width, height] = info.frame_size();
    if (crop->is_noop(width, height)) {
        return app::tr("{label}: no black bars. Compared in full at {size}.",
                       {{"label", label}, {"size", size_text(width, height)}});
    }
    QStringList sides;
    for (const auto& [side, pixels] : {std::pair{N_("top {pixels} px"), std::max(0, crop->y)},
                                       std::pair{N_("bottom {pixels} px"), std::max(0, height - crop->y - crop->h)},
                                       std::pair{N_("left {pixels} px"), std::max(0, crop->x)},
                                       std::pair{N_("right {pixels} px"), std::max(0, width - crop->x - crop->w)}}) {
        if (pixels != 0) {
            sides.append(app::tr(side, {{"pixels", pixels}}));
        }
    }
    return app::tr("{label}: black bars cropped off -- {sides}.\nCompared at {cropped} instead of {full}.",
                   {{"label", label},
                    {"sides", sides.join(", ")},
                    {"cropped", size_text(crop->w, crop->h)},
                    {"full", size_text(width, height)}});
}

void MainWindow::set_row_black_bars(int row, bool probe_failed) {
    // Whether the test video has black bars, and nothing more; the pixel
    // counts, and what was cropped off the reference, are in the tooltip.
    // Auto crop is worked out during a run, and can differ between rows, as
    // each has its own range: the cell says it is pending until a result
    // gives the crops the run used.
    QTableWidgetItem* item = distorted_table_->item(row, col_black_bars);
    if (item == nullptr) {
        return;
    }
    const auto show = [item](const QString& text, const QString& tooltip, bool muted) {
        item->setText(text);
        item->setToolTip(tooltip);
        theme::foreground(item, muted ? std::optional(theme::color("dimmed")) : std::nullopt);
    };
    if (probe_failed) {
        show(app::tr("Unknown"), app::tr("Black bars could not be checked because the video could not be read."), true);
        return;
    }
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    if (!row_data.completed_run) {
        if (row_data.options.crop_mode == CropMode::none) {
            show(app::tr("Off"),
                 app::tr("Black-bar detection is disabled for this row; the full frames will be compared."), true);
        } else {
            show(app::tr("Pending"), app::tr("Black bars will be detected when this row is run."), true);
        }
        return;
    }
    const ComparisonResult& result = row_data.completed_run->result;
    // A resolution test has no test video of its own: both of its videos come
    // from the reference, so the reference's bars are the answer.
    const bool resample = row_data.options.resample_test.has_value();
    const std::optional<CropBox>& subject_crop = resample ? result.source_crop : result.distorted_crop;
    const VideoInfo& subject_info = resample ? result.source_info : result.distorted_info;
    if (!subject_crop && !result.source_crop) {
        show(app::tr("Off"), app::tr("Black-bar detection was disabled for this run; no crop was applied."), true);
        return;
    }
    if (!subject_crop) {
        // Source only: the reference's bars cut, the test video compared at its full frame
        show(app::tr("Source only"),
             crop_detail(app::tr("Reference"), result.source_info, result.source_crop) + "\n\n" +
                 app::tr("The test video is compared at its full frame."),
             false);
        return;
    }
    QString tooltip = crop_detail(app::tr("Reference"), result.source_info, result.source_crop);
    if (!resample) {
        tooltip = crop_detail(app::tr("Test video"), result.distorted_info, result.distorted_crop) + "\n\n" + tooltip;
    }
    const auto [width, height] = subject_info.frame_size();  // as compared (crop_detail)
    show(subject_crop->is_noop(width, height) ? app::tr("No") : app::tr("Yes"), tooltip, false);
}

std::pair<QString, QString> MainWindow::resize_mismatch(int row) const {
    // (short tag, full explanation) saying which of the two videos is
    // resized to match the other, when their sizes differ. The two are
    // compared after their black bars come off, as the run compares them: a
    // 1920x1080 letterboxed reference and a 1920x804 encode of the same film
    // are the same picture, not a resize. After a run, it uses the crops and
    // direction the run used, which stay right if the row's settings are
    // edited afterwards; before one, the row's own setting.
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    const CompletedRun* run = row_data.completed_run.get();
    if (run == nullptr && (!source_info_ || !row_data.video_info)) {
        return {};
    }
    const VideoInfo& source = run != nullptr ? run->result.source_info : *source_info_;
    const VideoInfo& distorted = run != nullptr ? run->result.distorted_info : *row_data.video_info;
    const Size reference = run != nullptr ? content_size(source, run->result.source_crop)
                                          : Size{source.width, source.height};
    const Size test = run != nullptr ? content_size(distorted, run->result.distorted_crop)
                                     : Size{distorted.width, distorted.height};
    if (reference == test) {
        return {QString(), app::tr("Reference and test video are compared at the same {size} -- no scaling needed.",
                              {{"size", size_text(reference.first, reference.second)}})};
    }
    if (run == nullptr && row_data.options.crop_mode != CropMode::none &&
        (reference.first == test.first || reference.second == test.second)) {
        // One dimension already matches and only the other differs: the shape
        // of black bars. Whether any scaling is needed is known once they are
        // detected, when the row runs.
        return {app::tr("Pending"),
                app::tr("The reference is {reference} and the test video {test}, a difference in one dimension only -- "
                        "the shape of black bars on one of them.\n\nBars are detected when this row runs, and are "
                        "removed before the two are compared, so whether any scaling is needed is known then.",
                        {{"reference", size_text(source.width, source.height)},
                         {"test", size_text(distorted.width, distorted.height)}})};
    }
    // A "Test both" companion row always goes its own way, set as it is
    // made; a saved result from before results kept the direction says the
    // default, whatever made it.
    const ScaleDirection direction = run != nullptr && !row_data.scale_direction_pinned
                                         ? run->result.scale_direction
                                         : row_data.options.scale_direction;
    const bool cropped_either =
        reference != Size{source.width, source.height} || test != Size{distorted.width, distorted.height};
    const QString cropped = cropped_either ? app::tr(" (after black bars)") : QString();
    if (direction == ScaleDirection::distorted_to_source) {
        return {app::tr("Test upscaled to source"),
                app::tr("Test video upscaled {from_size} -> {to_size}{cropped} to match the reference.",
                        {{"from_size", size_text(test.first, test.second)},
                         {"to_size", size_text(reference.first, reference.second)},
                         {"cropped", cropped}})};
    }
    return {app::tr("\u2193 source"),
            app::tr("Reference downscaled {from_size} -> {to_size}{cropped} to match the test video.",
                    {{"from_size", size_text(reference.first, reference.second)},
                     {"to_size", size_text(test.first, test.second)},
                     {"cropped", cropped}})};
}

void MainWindow::fit_columns(std::initializer_list<int> columns) {
    // Fitted once, when the window is next idle: fitting measures every
    // row's cell, and fitting for each row as it is drawn would take 90,000
    // measurements a column for 300 rows.
    if (columns_to_fit_.empty()) {
        QTimer::singleShot(0, this, [this] { fit_pending_columns(); });
    }
    columns_to_fit_.insert(columns.begin(), columns.end());
}

void MainWindow::fit_pending_columns() {
    const std::set<int> columns = std::exchange(columns_to_fit_, {});
    // Fitted as if no column were sorted: once one is, Qt makes room for the
    // sort arrow in every heading, so a column fitted after sorting would
    // come out wider (Video bitrate 112 px rather than 84), and File name
    // narrower.
    QHeaderView* header = distorted_table_->horizontalHeader();
    const bool sorted_shown = header->isSortIndicatorShown();
    header->setSortIndicatorShown(false);
    for (const int column : columns) {
        distorted_table_->resizeColumnToContents(column);
    }
    header->setSortIndicatorShown(sorted_shown);
}

void MainWindow::set_row_scaling(int row) {
    QTableWidgetItem* item = distorted_table_->item(row, col_scaling);
    if (item == nullptr) {
        return;
    }
    const auto [tag, explanation] = resize_mismatch(row);
    item->setText(tag);
    item->setToolTip(explanation);
    fit_columns({col_scaling});
}

void MainWindow::set_row_info(int row, const std::optional<VideoInfo>& info, const QString& error) {
    QTableWidgetItem* item = distorted_table_->item(row, col_info);
    QTableWidgetItem* scaling_item = distorted_table_->item(row, col_scaling);
    if (!error.isEmpty() || !info) {
        set_row_status(row, RowState::failed, error);
        item->setText(app::tr("Probe failed"));
        item->setToolTip(error);
        item->setForeground(theme::color("failed"));
        distorted_table_->item(row, col_bitrate)->setText(QString());
        scaling_item->setText(QString());
        scaling_item->setToolTip(QString());
        set_row_black_bars(row, true);
    } else {
        item->setText(media_info_string(*info));
        // Back to normal text: the placeholder shown while probing greys the
        // cell, and leaving it grey makes a probed row look disabled.
        theme::foreground(item, std::nullopt);
        const QString grain = film_grain_note(*info);
        item->setToolTip(QString::fromStdString(format_hms(info->duration, 1)) +
                         (grain.isEmpty() ? QString() : "\n" + grain));
        distorted_table_->item(row, col_bitrate)->setText(bitrate_string(*info));
        distorted_table_->item(row, col_bitrate)->setToolTip(bitrate_note(*info));
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        row_data.video_info = info;
        set_row_scaling(row);
        if (row_data.analysis_status == RowState::reading) {
            row_data.analysis_status.reset();
            set_row_metrics(row);
        }
        set_row_black_bars(row);
    }
    // Snug to what is in them, while staying draggable in between updates.
    fit_columns({col_info, col_bitrate, col_scaling});
}

void MainWindow::set_row_status(int row, RowState state, const QString& detail) {
    RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    row_data.analysis_status = state;
    row_data.status_detail = detail;
    refresh_row_state(row);
}

void MainWindow::on_add_distorted() {
    const QStringList paths =
        QFileDialog::getOpenFileNames(this, app::tr("Select test videos"), video_dialog_dir("test"));
    if (paths.isEmpty()) {
        return;
    }
    remember_video_dir("test", path_of(paths.front()));
    std::vector<std::filesystem::path> chosen;
    for (const QString& path : paths) {
        chosen.push_back(path_of(path));
    }
    add_test_videos(chosen);
}

void MainWindow::on_files_dropped_on_table(const QStringList& paths) {
    if (run_active_ || paths.isEmpty()) {
        return;
    }
    remember_video_dir("test", path_of(paths.front()));
    std::vector<std::filesystem::path> dropped;
    for (const QString& path : paths) {
        dropped.push_back(path_of(path));
    }
    add_test_videos(dropped);
}

void MainWindow::add_test_videos(const std::vector<std::filesystem::path>& paths) {
    // A row for each video not already in the table, in order. The rows
    // appear at once; reading each file and looking up its saved scores
    // happen on workers, as for a handful of long videos they take a couple
    // of seconds.
    std::vector<std::filesystem::path> added;
    for (const std::filesystem::path& path : paths) {
        const auto same = [&path](const std::filesystem::path& other) { return same_path(path, other); };
        if (std::any_of(rows_.begin(), rows_.end(), [&same](const auto& row) { return same(row->path); }) ||
            std::any_of(added.begin(), added.end(), same)) {
            continue;
        }
        added.push_back(path);
    }
    if (added.empty()) {
        return;
    }
    for (const std::filesystem::path& path : added) {
        const int row = add_table_row(path);
        set_row_status(row, RowState::reading);
    }
    start_media_probe(added);
    start_cache_lookup(added);
}

void MainWindow::on_add_resample_test() {
    // Adds a resolution test for the width the user picks.
    if (!source_info_) {
        QMessageBox::warning(this, app::tr("No reference"), app::tr("Please select a reference video first."));
        return;
    }
    std::vector<ResampleTarget> targets;
    for (const ResampleTarget& target : resample_target_choices()) {
        if (target.width < source_info_->width) {
            targets.push_back(target);
        }
    }
    if (targets.empty()) {
        dialogs_.inform(
            app::tr("No smaller resolution available"),
            app::tr("The smallest resolution test is {smallest} pixels wide, which is not below this "
                    "reference's {width}-pixel width.",
                    {{"smallest", resample_target_choices().back().width}, {"width", source_info_->width}}));
        return;
    }
    // The size, and the scaling algorithm of both steps (the Options' one
    // first, which made every test before tests had one of their own).
    QDialog dialog(this);
    dialog.setWindowTitle(app::tr("Add resolution test"));
    auto* form = new QFormLayout(&dialog);
    auto* sizes = new QComboBox;
    for (const ResampleTarget& target : targets) {
        sizes->addItem(QString::fromStdString(target.label));
    }
    sizes->setCurrentIndex(std::min(1, static_cast<int>(targets.size()) - 1));
    auto* scalers = new QComboBox;
    for (const std::string_view scaler : resample_scalers) {
        scalers->addItem(QString::fromUtf8(scaler.data(), static_cast<qsizetype>(scaler.size())));
    }
    scalers->setCurrentText(QString::fromStdString(default_options_.scale_algorithm));
    form->addRow(app::tr("Downscale to (then scale back up):"), sizes);
    form->addRow(app::tr("Scaling algorithm:"), scalers);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    form->addRow(buttons);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }
    ResampleTarget target = targets[static_cast<std::size_t>(sizes->currentIndex())];
    target.scaler = scalers->currentText().toStdString();
    add_resample_test(target);
}

void MainWindow::add_resample_test(const ResampleTarget& target) {
    // A row for a resolution test of the reference at `target`.
    const std::filesystem::path path = synthetic_resample_distorted_path(source_info_->path, target);
    if (row_index_of_path(path)) {
        dialogs_.inform(app::tr("Already added"),
                        app::tr("A {label} resolution test for this reference is already in the list.",
                                {{"label", QString::fromStdString(target.label + " " + target.scaler)}}));
        return;
    }
    const int row = add_table_row(path);
    RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    // A resolution test decodes only the reference; its test video is made in
    // the filter graph. So the reference is the file the saved scores follow,
    // and the target is part of the options.
    row_data.media_path = source_info_->path;
    row_data.options.resample_test = target;
    row_data.video_info = *source_info_;
    set_resample_row_info(row, target);
    // Drawn again now it is a resolution test: drawn when added, its
    // SSIMULACRA2, Butteraugli and CVVDP cells showed tick boxes, not n/a.
    set_row_metrics(row);
    start_cache_lookup({path});
}

void MainWindow::set_resample_row_info(int row, const ResampleTarget& target) {
    // Shows what a resolution test scales the reference down to and back up
    // to, and with which scaling algorithm (named as FFmpeg names it, as the
    // Options' list shows it, in every language).
    const VideoInfo& info = *source_info_;
    const int down_h = resample_down_height(target.width, {info.width, info.height});
    QTableWidgetItem* item = distorted_table_->item(row, col_info);
    item->setText(app::tr("Downscale to {down}, upscale back to {back}",
                          {{"down", size_text(target.width, down_h)}, {"back", size_text(info.width, info.height)}}) +
                  " (" + QString::fromStdString(target.scaler) + ")");
    item->setToolTip(QString::fromStdString(format_hms(info.duration, 1)));
    distorted_table_->item(row, col_bitrate)->setText("N/A");
    set_row_black_bars(row);
    fit_columns({col_info, col_bitrate});
}

void MainWindow::start_media_probe(const std::vector<std::filesystem::path>& paths) {
    // Cancels nothing, on purpose: a probe answers a question about one file,
    // and the answer stays true whatever the user does next.
    if (paths.empty()) {
        return;
    }
    QStringList texts;
    for (const std::filesystem::path& path : paths) {
        texts.append(text_of(path));
    }
    auto* worker = new ProbeWorker(texts, find_tool("ffprobe", settings_), find_tool("ffmpeg", settings_), this);
    probe_workers_.push_back(worker);
    connect(worker, &ProbeWorker::probed, this, [this](const Probed& probed) { on_probed(probed); });
    connect(worker, &ProbeWorker::finished_all, this, [this, worker] { on_probe_finished(worker); });
    show_reading(ntr("Reading {count} video...", "Reading {count} videos...", static_cast<long long>(paths.size())));
    worker->start();
}

void MainWindow::on_probed(const Probed& probed) {
    const std::optional<int> row = row_index_of_path(path_of(probed.path));
    if (!row) {
        return;  // removed while the probe was in flight
    }
    // What the row's saved scores are filed under can depend on what reading
    // the video finds (its film grain), and the lookup started with the probe:
    // an answer to the request without it is dropped (a later one is refused
    // by its token), and the lookup made again when the lane ends, once for
    // all its videos.
    const RowData& row_data = *rows_[static_cast<std::size_t>(*row)];
    const auto token = [&] {
        return lookup_token(source_info_->path, row_data.identity_path(), analysis_request_of(row_data));
    };
    const std::optional<std::string> asked = source_info_ ? std::optional(token()) : std::nullopt;
    std::optional<VideoInfo> info = probed.info;
    const std::optional<bool> with = film_grain_default_choice(settings_.film_grain_default);
    if (info && with && film_grain_choosable(*info)) {
        info->with_film_grain = *with;  // Settings' default, until the video is chosen for
    }
    set_row_info(*row, info, info ? QString() : probed.error);
    if (asked && token() != *asked) {
        invalidate_completed_results({*row});
        probed_lookups_.push_back(row_data.path);
    }
}

void MainWindow::on_probe_finished(ProbeWorker* worker) {
    retire(worker);
    if (!probed_lookups_.empty()) {
        start_cache_lookup(std::exchange(probed_lookups_, {}));
    }
    if (std::any_of(probe_workers_.begin(), probe_workers_.end(), [](ProbeWorker* w) { return w->isRunning(); })) {
        return;  // another lane is still reading
    }
    if (!run_active_) {
        show_ready();
    }
    on_table_selection_changed();
}

void MainWindow::apply_row_order(const std::vector<int>& order) {
    // The table's cells and the rows move together, and the same videos stay
    // selected. Every other piece of state finds its row by identity or by
    // path, never by position.
    std::vector<int> sorted_order = order;
    std::sort(sorted_order.begin(), sorted_order.end());
    std::vector<int> identity(order.size());
    std::iota(identity.begin(), identity.end(), 0);
    if (sorted_order != identity || order == identity || order.size() != rows_.size()) {
        return;
    }
    QTableWidget* table = distorted_table_;
    std::set<const RowData*> selected;
    for (const QModelIndex& index : table->selectionModel()->selectedRows()) {
        selected.insert(rows_[static_cast<std::size_t>(index.row())].get());
    }
    const QModelIndex current = table->currentIndex();
    const RowData* current_row = current.isValid() ? rows_[static_cast<std::size_t>(current.row())].get() : nullptr;
    const int columns = table->columnCount();
    table->blockSignals(true);
    table->selectionModel()->blockSignals(true);
    syncing_table_ = true;
    std::vector<std::vector<QTableWidgetItem*>> cells(order.size());
    for (int row = 0; row < static_cast<int>(order.size()); ++row) {
        for (int column = 0; column < columns; ++column) {
            cells[static_cast<std::size_t>(row)].push_back(table->takeItem(row, column));
        }
    }
    std::vector<std::shared_ptr<RowData>> reordered;
    for (int position = 0; position < static_cast<int>(order.size()); ++position) {
        const int old = order[static_cast<std::size_t>(position)];
        for (int column = 0; column < columns; ++column) {
            if (QTableWidgetItem* item = cells[static_cast<std::size_t>(old)][static_cast<std::size_t>(column)]) {
                table->setItem(position, column, item);
            }
        }
        reordered.push_back(rows_[static_cast<std::size_t>(old)]);
    }
    rows_ = std::move(reordered);
    table->clearSelection();
    QItemSelection selection;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        if (selected.contains(rows_[static_cast<std::size_t>(row)].get())) {
            selection.select(table->model()->index(row, 0), table->model()->index(row, columns - 1));
        }
    }
    table->selectionModel()->select(selection, QItemSelectionModel::Select);
    if (current_row != nullptr) {
        const auto found = std::find_if(rows_.begin(), rows_.end(),
                                        [current_row](const auto& row) { return row.get() == current_row; });
        table->selectionModel()->setCurrentIndex(
            table->model()->index(static_cast<int>(found - rows_.begin()), current.column()),
            QItemSelectionModel::NoUpdate);
    }
    syncing_table_ = false;
    table->selectionModel()->blockSignals(false);
    table->blockSignals(false);
    table->viewport()->update();
    on_table_selection_changed();
    sync_frame_compare();  // Video Compare and Metric Graphs list the videos in the table's order
}

void MainWindow::on_rows_dragged(const QList<int>& rows, int target) {
    // Rows dragged within the table go, in their order, before `target`.
    if (run_active_ || rows.isEmpty()) {
        return;
    }
    const std::set<int> moved(rows.begin(), rows.end());
    std::vector<int> rest;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        if (!moved.contains(row)) {
            rest.push_back(row);
        }
    }
    const int at =
        target - static_cast<int>(std::count_if(rows.begin(), rows.end(), [target](int row) { return row < target; }));
    std::vector<int> order(rest.begin(), rest.begin() + at);
    order.insert(order.end(), moved.begin(), moved.end());
    order.insert(order.end(), rest.begin() + at, rest.end());
    std::vector<int> identity(order.size());
    std::iota(identity.begin(), identity.end(), 0);
    if (order != identity) {
        clear_sort_indicator();
    }
    apply_row_order(order);
}

void MainWindow::on_header_clicked(int column) {
    // Sorts the rows by the column whose heading was clicked; the same
    // heading again reverses it. A row without a value goes last whichever
    // way.
    if (run_active_ || rows_.empty()) {
        return;
    }
    const Qt::SortOrder order = column == sort_column_ && sort_order_ == Qt::AscendingOrder ? Qt::DescendingOrder
                                                                                              : Qt::AscendingOrder;
    std::vector<std::optional<SortKey>> keys;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        const std::optional<VideoInfo>& info = row_data.video_info;
        std::optional<SortKey> key;
        if (column == col_path) {
            key = natural_parts(name_of(row_data.path));
        } else if (column == col_info) {
            if (info) {
                key = std::tuple{static_cast<long long>(info->width) * info->height, info->fps, info->duration};
            }
        } else if (column == col_bitrate) {
            if (info && info->bit_rate != 0) {
                key = info->bit_rate;
            }
        } else if (column == col_check) {
            const QTableWidgetItem* item = distorted_table_->item(row, col_check);
            key = item != nullptr && item->checkState() == Qt::Checked ? 0 : 1;
        } else if (column == col_scaling) {
            const QTableWidgetItem* item = distorted_table_->item(row, col_scaling);
            if (item != nullptr && !item->text().isEmpty()) {
                key = item->text().toCaseFolded();
            }
        } else if (metric_column_of(column) != nullptr && row_data.completed_run) {
            // NaN would scramble the sort.
            if (const std::optional<double> score = score_of(*row_data.completed_run, column);
                score && !std::isnan(*score)) {
                key = *score;
            }
        }
        keys.push_back(std::move(key));
    }
    std::vector<int> present;
    std::vector<int> missing;
    for (int row = 0; row < static_cast<int>(keys.size()); ++row) {
        (keys[static_cast<std::size_t>(row)] ? present : missing).push_back(row);
    }
    if (present.empty()) {
        return;  // nothing in that column to sort by
    }
    std::stable_sort(present.begin(), present.end(), [&keys, order](int a, int b) {
        const SortKey& x = *keys[static_cast<std::size_t>(a)];
        const SortKey& y = *keys[static_cast<std::size_t>(b)];
        return order == Qt::DescendingOrder ? sort_less(y, x) : sort_less(x, y);
    });
    sort_column_ = column;
    sort_order_ = order;
    QHeaderView* header = distorted_table_->horizontalHeader();
    header->setSortIndicator(column, order);
    header->setSortIndicatorShown(true);
    present.insert(present.end(), missing.begin(), missing.end());
    apply_row_order(present);
}

void MainWindow::clear_sort_indicator() {
    if (!sort_column_) {
        return;
    }
    sort_column_.reset();
    distorted_table_->horizontalHeader()->setSortIndicatorShown(false);
}

std::optional<int> MainWindow::row_index_of_path(const std::filesystem::path& path) const {
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        if (same_path(rows_[static_cast<std::size_t>(row)]->path, path)) {
            return row;
        }
    }
    return std::nullopt;
}

void MainWindow::on_remove_distorted() {
    std::set<int, std::greater<>> rows;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        rows.insert(index.row());
    }
    // Their lines leave the graph too: once its row is gone, nothing else
    // could remove a line.
    std::vector<std::string> lines;
    for (const int row : rows) {
        if (const std::shared_ptr<CompletedRun>& run = rows_[static_cast<std::size_t>(row)]->completed_run) {
            lines.push_back(run->graph_identity);
        }
    }
    graph_panel_->remove_by_identities(lines);
    for (const int row : rows) {
        distorted_table_->removeRow(row);
        rows_.erase(rows_.begin() + row);
    }
    on_table_selection_changed();
    sync_frame_compare();
}

void MainWindow::on_remove_all_distorted() {
    if (rows_.empty()) {
        return;
    }
    // Confirmed, because it can discard a lot of completed work from the
    // table at once. The saved scores themselves are kept.
    const auto answer = QMessageBox::question(
        this, app::tr("Remove all videos"),
        ntr("Remove {count} video from the list?\n\nSaved results are kept -- re-adding a video shows its scores "
            "again.",
            "Remove all {count} videos from the list?\n\nSaved results are kept -- re-adding a video shows its "
            "scores again.",
            static_cast<long long>(rows_.size())),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return;
    }
    std::vector<std::string> lines;
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        if (row_data->completed_run) {
            lines.push_back(row_data->completed_run->graph_identity);
        }
    }
    graph_panel_->remove_by_identities(lines);
    distorted_table_->setRowCount(0);
    rows_.clear();
    on_table_selection_changed();
    sync_frame_compare();
}

void MainWindow::on_highlight_toggled(bool on) {
    settings_.highlight_best_worst = on;
    save_settings();
    apply_best_worst();
}

void MainWindow::schedule_best_worst() { best_worst_timer_->start(); }

void MainWindow::apply_best_worst() {
    // Colours each metric's best score green and its worst red, among the
    // test videos' (MetricDefinition::best_and_worst), while Highlight
    // best/worst results is ticked: only where two or more videos have a
    // score to compare. A score from the other implementation than its row
    // asks for, marked "(CPU)" or "(GPU)", is on another scale and is not
    // compared (set_row_metrics gives it no score_role). Unticked, no cell
    // has a colour.
    const bool on = highlight_check_->isChecked();
    const QColor none(0, 0, 0, 0);
    // setData emits itemChanged, as a click's edit does.
    const bool syncing = std::exchange(syncing_table_, true);
    for (const MetricColumn& column : metric_columns()) {
        std::vector<QTableWidgetItem*> items;
        std::vector<std::optional<double>> scores;
        for (int row = 0; row < distorted_table_->rowCount(); ++row) {
            if (QTableWidgetItem* item = distorted_table_->item(row, column.column)) {
                const QVariant score = item->data(score_role);
                items.push_back(item);
                scores.push_back(score.isValid() ? std::optional(score.toDouble()) : std::nullopt);
            }
        }
        const std::vector<ScoreMark> marks = on ? metric_definition(column.key).best_and_worst(scores)
                                                : std::vector<ScoreMark>(items.size(), ScoreMark::none);
        for (std::size_t index = 0; index < items.size(); ++index) {
            items[index]->setData(Qt::BackgroundRole, marks[index] == ScoreMark::best    ? theme::color("best")
                                                      : marks[index] == ScoreMark::worst ? theme::color("worst")
                                                                                         : none);
        }
    }
    syncing_table_ = syncing;
}

}  // namespace vml::app
