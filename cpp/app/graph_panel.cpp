#include "graph_panel.h"

#include <QAbstractItemView>
#include <QColor>
#include <QFont>
#include <QFontMetrics>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QScrollBar>
#include <QSpinBox>
#include <QTabBar>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <tuple>

#include "chart.h"
#include "file_writes.h"
#include "i18n.h"
#include "series_colors.h"
#include "theme.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/export_csv.h"
#include "vml/number_format.h"
#include "vml/run_file.h"
#include "vml/time_format.h"

// The tab's text is app::tr's (i18n.h), as in main_window.cpp.
namespace vml::app {

namespace {

// The statistics' cells: a monospaced face, so that digits share a width.
constexpr const char* number_font_family = "Consolas";

// How many lines the table at the top shows before it scrolls: any more
// would crowd out the graph it describes.
constexpr int visible_series_rows = 4;

// How many steps either side of the pointer the readout looks for a point to
// snap to.
constexpr double hover_search_steps = 5;

// Layout of the exported PNG (render_export_image).
constexpr int export_margin = 16;
constexpr int export_gap = 8;
constexpr int export_swatch = 12;
constexpr int export_row_padding = 6;

// Stands in for a value when sizing a readout column: digits at their
// widest, so that a column's width never depends on the value under the
// pointer.
constexpr double widest_sample = -88.88;

// The metric notes under the graph.
const char* const xpsnr_infinity_note =
    N_("XPSNR: ∞ is plotted at {db:g} dB; stored scores and statistics retain infinity.");
const char* const cvvdp_note =
    N_("CVVDP: the curve is the JOD of each second, plotted at the middle of the second; the CVVDP column is the "
       "whole video's JOD, which is not the mean of its seconds. Median to Max describe the seconds.");

// The table's columns: the series, then one mean per metric, then the
// selected metric's detail, then the remove button.
int first_mean_column() { return 1; }
int mean_column_end() { return 1 + static_cast<int>(metrics().size()); }
bool is_mean_column(int column) { return column >= first_mean_column() && column < mean_column_end(); }

bool worst_is_high(const MetricDefinition& metric) { return metric.direction == MetricDirection::lower_is_better; }

QString text_of_std(const std::string& text) { return QString::fromStdString(text); }

// A value as Python's format(value, "g") writes it.
QString g_text(double value) {
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%g", value);
    return QString::fromLatin1(text.data());
}

// How long a text is as Python counts it: in characters, not UTF-16 units.
int length_of(const QString& text) { return static_cast<int>(text.toUcs4().size()); }

// `text` filled with spaces on the right to `width` characters, as Python's
// f"{text:<{width}}".
QString pad_right(const QString& text, int width) {
    const int missing = width - length_of(text);
    return missing > 0 ? text + QString(missing, ' ') : text;
}

QString trimmed_right(QString text) {
    while (!text.isEmpty() && text.back().isSpace()) {
        text.chop(1);
    }
    return text;
}

// The frame metrics, in the app's order.
std::vector<const MetricDefinition*> frame_metrics() {
    std::vector<const MetricDefinition*> found;
    for (const MetricDefinition& metric : metrics()) {
        if (metric.kind == MetricKind::frame) {
            found.push_back(&metric);
        }
    }
    return found;
}

// The video a result's line takes its colour from: the recorded test video,
// but the real file for a resolution test (its reference) and for a scale
// direction companion (the test video it compares the other way).
std::filesystem::path color_path(const ComparisonResult& result) {
    if (result.resample_target) {
        return result.source;
    }
    const std::filesystem::path& physical = result.distorted_info.path;
    if (!physical.empty() &&
        result.distorted == synthetic_scale_direction_variant_path(physical, result.scale_direction)) {
        return physical;
    }
    return result.distorted;
}

// How many frames score +inf (identical to the reference), by metric, where
// a metric has any.
std::map<std::string, long long> identical_frame_counts(const ComparisonResult& result) {
    std::map<std::string, long long> counts;
    for (const MetricDefinition& metric : metrics()) {
        if (const FrameMetricResult* frame = result.metric_results.frame(std::string(metric.key))) {
            const auto count = std::count_if(frame->values.begin(), frame->values.end(),
                                             [](float value) { return std::isinf(value) && value > 0; });
            if (count > 0) {
                counts[std::string(metric.key)] = count;
            }
        }
    }
    return counts;
}

// Each metric's figure for the whole run: the mean of its frames by the
// metric's own way of averaging, or a whole-video score (CVVDP).
std::map<std::string, std::optional<double>> metric_means(const ComparisonResult& result) {
    std::map<std::string, std::optional<double>> means;
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        if (const SequenceMetricResult* sequence = result.metric_results.sequence(key)) {
            means[key] = sequence->score;
        } else if (const FrameMetricResult* frame = result.metric_results.frame(key);
                   frame != nullptr && !frame->values.empty()) {
            means[key] = frame->aggregate();
        } else {
            means[key] = std::nullopt;
        }
    }
    return means;
}

// The value recorded for exactly `frame`, or nothing; never a neighbour's.
std::optional<double> value_at_frame(std::span<const int> frames, std::span<const float> values, int frame) {
    const auto found = std::lower_bound(frames.begin(), frames.end(), frame);
    if (found == frames.end() || *found != frame) {
        return std::nullopt;
    }
    const double value = values[static_cast<std::size_t>(found - frames.begin())];
    return std::isnan(value) ? std::nullopt : std::optional(value);
}

// The index of the time nearest `x`.
std::size_t nearest_index_by_time(std::span<const double> times, double x) {
    const auto index = static_cast<std::size_t>(std::lower_bound(times.begin(), times.end(), x) - times.begin());
    if (index == 0) {
        return 0;
    }
    if (index >= times.size()) {
        return times.size() - 1;
    }
    return times[index] - x < x - times[index - 1] ? index : index - 1;
}

// The typical time between a line's points.
double step_of(std::span<const double> times) {
    return times.size() > 1 ? (times.back() - times.front()) / static_cast<double>(times.size() - 1) : 1.0;
}

// The index of the second containing time `x` on a per-second line.
std::size_t second_at(const MetricCurve& curve, double x) {
    const auto after = std::upper_bound(curve.starts.begin(), curve.starts.end(), x) - curve.starts.begin();
    return static_cast<std::size_t>(std::clamp<std::ptrdiff_t>(after - 1, 0,
                                                              static_cast<std::ptrdiff_t>(curve.starts.size()) - 1));
}

// When point `index` begins: its frame's time, or its second's start.
double start_time(const MetricCurve& curve, std::size_t index) {
    return curve.starts.empty() ? curve.times[index] : curve.starts[index];
}

// The median of `values`, as numpy's: the middle one, or the mean of the two.
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

ScoreStats stats_of(std::span<const float> values, const MetricDefinition& metric) {
    const std::vector<double> scores(values.begin(), values.end());
    return compute_stats(scores, metric.thresholds, metric.aggregation, metric.direction);
}

// The statistics as the table and the exported image show them, at the
// metric's precision: mean, median, standard deviation, minimum, maximum,
// then the worst frames' tail.
std::vector<QString> summary(const ScoreStats& stats, const MetricDefinition& metric) {
    std::vector<QString> cells;
    for (const double value : {stats.mean, stats.median, stats.stdev, stats.minimum, stats.maximum,
                               stats.worst_10_percent, stats.worst_5_percent, stats.worst_1_percent,
                               stats.worst_0_1_percent}) {
        cells.push_back(text_of_std(metric.format_value(value)));
    }
    return cells;
}

QString threshold_label(const Threshold& threshold) {
    return QString(QChar::fromLatin1(threshold.comparison)) + " " + g_text(threshold.value);
}

QString percentage_text(double percentage) { return text_of_std(format_fixed(percentage, 1)) + "%"; }

// A line's statistics as the table and the exported image show them: the
// summary, then the share of its frames in each of the metric's score
// bands. A line with no scored frame (every score NaN) has no value in any
// of them: its statistics would read as zeros, and it has no frames to
// share between the bands.
std::vector<QString> statistics_cells(const ScoreStats& stats, const MetricDefinition& metric) {
    std::vector<QString> cells = summary(stats, metric);
    if (stats.count == 0) {
        return std::vector<QString>(cells.size() + metric.thresholds.size(),
                                    text_of_std(metric.format_value(std::nullopt)));
    }
    for (const ThresholdCount& count : stats.thresholds) {
        cells.push_back(percentage_text(count.percentage));
    }
    return cells;
}

// The worst frames' tail shares, as the table heads them.
constexpr std::array<const char*, 4> tail_shares{"10%", "5%", "1%", "0.1%"};

}  // namespace

// ================================================================ a page

MetricPage::MetricPage(const MetricDefinition& metric, QWidget* parent)
    : QWidget(parent), metric_(&metric), sign_(worst_is_high(metric) ? -1.0F : 1.0F) {
    placeholder_ = worst_is_high(metric)
                       ? app::tr("Hover to inspect a point (locks onto the worst nearby score at or below your cursor, "
                                 "so dips are easy to land on).\nScroll to zoom, drag to pan, double-click to reset.")
                       : app::tr("Hover to inspect a point (locks onto the lowest nearby score at or below your "
                                 "cursor, so dips are easy to land on).\nScroll to zoom, drag to pan, double-click to "
                                 "reset.");
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    // A lower-is-better metric is drawn upside down (0 at the top), so that a
    // better encode sits higher on every page, and its bad frames hang down
    // as dips, as VMAF's do.
    const QString axis = qt_text(metric.axis_label);
    chart_ = new ChartWidget(worst_is_high(metric) ? app::tr("{axis_label} (lower is better)", {{"axis_label", axis}})
                                                   : axis,
                             metric.fixed_y_max, worst_is_high(metric));
    layout->addWidget(chart_, 1);

    no_data_label_ = new QLabel(app::tr("No {label} data among the currently visible series -- tick the {label} column "
                                        "header before running to see it here.",
                                        {{"label", qt_text(metric.label)}}));
    no_data_label_->setAlignment(Qt::AlignCenter);
    theme::style(no_data_label_, "color: {faint}; font-style: italic; padding: 12px;");
    no_data_label_->setVisible(false);
    layout->addWidget(no_data_label_);

    // The readout, rewritten at each move of the pointer (ReadoutLabel). It
    // is monospaced, so that its columns line up; the hint shown until the
    // first hover keeps the window's font, as the rest of the tab (in the
    // readout's, it looked out of place). Its height is fixed and its width
    // capped (fit_hover_label).
    hover_label_ = new ReadoutLabel(placeholder_);
    readout_font_ = QFont(QStringLiteral("Consolas"));
    readout_font_.setStyleHint(QFont::Monospace);
    fit_readout_margins();
    fit_hover_label();
    auto* hover_row = new QHBoxLayout;
    hover_row->setContentsMargins(0, 0, 0, 0);
    hover_row->addWidget(hover_label_);
    hover_row->addStretch(1);
    layout->addLayout(hover_row);

    connect(chart_, &ChartWidget::left, this, [this] { chart_->set_cursor_time(std::nullopt); });
    // GraphPanel takes the chart's hovered: the readout needs its runs.
}

void MetricPage::set_curve(int series_id, const SeriesEntry& entry) {
    remove_curves({series_id});
    if (metric_->kind == MetricKind::sequence) {
        set_timeline_curve(series_id, entry);
        return;
    }
    // A metric is calculated for a whole run or not at all, so its scores
    // are there or not, frame for frame.
    const FrameMetricResult* result = entry.result->metric_results.frame(std::string(metric_->key));
    if (result == nullptr || result->values.empty()) {
        update_no_data_label();
        return;
    }
    ChartSeries series{result->time, result->values, entry.result, entry.color, entry.visible};
    if (metric_->key == "xpsnr" &&
        std::any_of(result->values.begin(), result->values.end(), [](float v) { return std::isinf(v) && v > 0; })) {
        // Drawn at a stand-in; the result's own scores, which the statistics,
        // the readout and exports read, keep infinity.
        auto drawn = std::make_shared<std::pair<std::shared_ptr<const ComparisonResult>, std::vector<float>>>(
            entry.result, result->values);
        for (float& value : drawn->second) {
            if (std::isinf(value) && value > 0) {
                value = static_cast<float>(xpsnr_infinity_plot_db);
            }
        }
        series.values = drawn->second;
        series.owner = drawn;
    }
    chart_->set_series(series_id, std::move(series));

    MetricCurve curve;
    curve.stats = stats_of(result->values, *metric_);
    curve.values = result->values;
    curve.frames = result->frame;
    curve.times = result->time;
    curve.owner = entry.result;
    curve.label = entry.label;
    curve.visible = entry.visible;
    for (const MetricDefinition* metric : frame_metrics()) {
        if (metric == metric_) {
            continue;
        }
        const FrameMetricResult* other = entry.result->metric_results.frame(std::string(metric->key));
        if (other == nullptr || other->values.empty() ||
            std::all_of(other->values.begin(), other->values.end(), [](float v) { return std::isnan(v); })) {
            continue;
        }
        curve.others.push_back({metric, other->frame, other->values, other->frame == result->frame});
    }
    curves_[series_id] = std::move(curve);
    update_no_data_label();
    fit_hover_label();
}

void MetricPage::set_timeline_curve(int series_id, const SeriesEntry& entry) {
    // A whole-video metric's timeline: CVVDP's JOD for each second, drawn at
    // the middle of its second, so that the pointer anywhere over the second
    // lands on it (at its first frame, half a second in would pick the
    // next). The statistics are of the seconds; the whole video's score is
    // in the table's CVVDP column.
    const SequenceMetricResult* result = entry.result->metric_results.sequence(std::string(metric_->key));
    if (result == nullptr || !result->has_timeline()) {
        update_no_data_label();
        return;
    }
    const std::vector<double>& starts = result->time;
    std::vector<double> lengths;
    for (std::size_t i = 1; i < starts.size(); ++i) {
        lengths.push_back(starts[i] - starts[i - 1]);
    }
    const double typical = lengths.empty() ? 1.0 : median(lengths);
    auto middles = std::make_shared<std::pair<std::shared_ptr<const ComparisonResult>, std::vector<double>>>(
        entry.result, std::vector<double>());
    for (std::size_t i = 0; i < starts.size(); ++i) {
        middles->second.push_back(starts[i] + (i < lengths.size() ? lengths[i] : typical) / 2);
    }
    chart_->set_series(series_id, {middles->second, result->values, middles, entry.color, entry.visible});

    MetricCurve curve;
    curve.stats = stats_of(result->values, *metric_);
    curve.values = result->values;
    curve.frames = result->frame;
    curve.times = middles->second;
    curve.owner = middles;
    curve.label = entry.label;
    curve.visible = entry.visible;
    curve.starts = starts;
    curves_[series_id] = std::move(curve);
    update_no_data_label();
    fit_hover_label();
}

void MetricPage::remove_curves(const std::vector<int>& series_ids) {
    // The readout sized again once, however many go.
    std::vector<int> removed;
    for (const int id : series_ids) {
        if (curves_.erase(id) > 0) {
            removed.push_back(id);
        }
    }
    if (removed.empty()) {
        return;
    }
    chart_->remove_series(removed);
    update_no_data_label();
    fit_hover_label();
}

void MetricPage::renumber(const std::map<int, int>& renumbered) {
    std::map<int, MetricCurve> curves;
    for (auto& [id, curve] : curves_) {
        const auto found = renumbered.find(id);
        curves.emplace(found == renumbered.end() ? id : found->second, std::move(curve));
    }
    curves_ = std::move(curves);
    chart_->renumber(renumbered);
}

void MetricPage::set_visible(int series_id, bool visible) {
    const auto found = curves_.find(series_id);
    if (found == curves_.end()) {
        return;
    }
    found->second.visible = visible;
    chart_->set_series_visible(series_id, visible);
    fit_hover_label();
}

const MetricCurve* MetricPage::curve(int series_id) const {
    const auto found = curves_.find(series_id);
    return found == curves_.end() ? nullptr : &found->second;
}

void MetricPage::update_no_data_label() { no_data_label_->setVisible(curves_.empty()); }

void MetricPage::fit_hover_label() {
    // The readout sized to the widest and tallest text it can show, from the
    // lines now: fixed, it cut off long names; sized at each hover, it laid
    // the tab out again each time.
    refresh_readout_layout();
    const QFontMetrics metrics(readout_font_);
    const std::vector<QString> labels = visible_labels();
    QStringList lines{app::tr("Time: {time}", {{"time", QStringLiteral("0:00:00.00")}})};
    std::map<std::string, std::optional<double>> widest;
    for (const auto& [metric, width] : columns_) {
        widest[std::string(metric->key)] = widest_sample;
    }
    for (const QString& label : labels) {
        lines.append(readout_row(readout_prefix(label, 8'888'888, 0.0), widest_sample, widest));
    }
    if (labels.size() == 2) {
        lines.append(QStringLiteral("Δ (%1 − %2) = -88.88").arg(labels[0], labels[1]));
    }
    // The label shows the placeholder, in the window's font, until the
    // pointer first comes over the chart, then the readout, never resized
    // between them: sized to the readout alone, a short name left the
    // placeholder cut off.
    const QStringList placeholder_lines = placeholder_.split('\n');
    const QFontMetrics hint(font());
    constexpr int padding = 28;  // the margins' 6 px each side, and some room
    int width = 0;
    for (const QString& line : lines) {
        width = std::max(width, metrics.horizontalAdvance(line));
    }
    for (const QString& line : placeholder_lines) {
        width = std::max(width, hint.horizontalAdvance(line));
    }
    // The taller of the two texts, not both: they are never shown together.
    const int height = std::max(static_cast<int>(lines.size()) * metrics.lineSpacing(),
                                static_cast<int>(placeholder_lines.size()) * hint.lineSpacing()) +
                       padding;
    // A maximum, not a fixed width: the label takes no more room than its
    // text needs, and can still shrink with the window.
    hover_label_->setMaximumWidth(width + padding);
    hover_label_->setFixedHeight(height);
}

void MetricPage::fit_readout_margins() {
    // The room around the readout's text, as the style sheet's 6 px of
    // padding it once had laid it out: QLabel took that padding for a frame,
    // so it indented the text by half an x where it starts and at the top,
    // and made room for a whole x. In the label's font, the window's or the
    // readout's.
    const int x = hover_label_->fontMetrics().horizontalAdvance(QLatin1Char('x'));
    hover_label_->set_indent(x / 2, x - x / 2);
    hover_label_->setContentsMargins(6, 6, 6, 6);
}

std::vector<QString> MetricPage::visible_labels() const {
    std::vector<QString> labels;
    for (const auto& [id, curve] : curves_) {
        if (curve.visible) {
            labels.push_back(curve.label);
        }
    }
    return labels;
}

QString MetricPage::readout_prefix(const QString& label, int frame, double time) const {
    // The fields before a value, aligned across the lines: the series' name
    // padded to the longest, so that the columns after it do not zig-zag.
    const QString series = pad_right("[" + label + "]", series_width_);
    const QString time_text = text_of_std(format_hms(time, 2));
    if (metric_->kind == MetricKind::sequence) {
        return app::tr("{series}  second from frame {frame:>6}   t={time}   ",
                       {{"series", series}, {"frame", frame}, {"time", time_text}});
    }
    return app::tr("{series}  frame {frame:>6}   t={time}   ",
                   {{"series", series}, {"frame", frame}, {"time", time_text}});
}

std::vector<const MetricDefinition*> MetricPage::extra_columns() const {
    // The other metrics a shown line has, in the app's order: the columns
    // after the "|". A metric no shown line has gets no column.
    std::set<const MetricDefinition*> present;
    for (const auto& [id, curve] : curves_) {
        if (curve.visible) {
            for (const MetricCurve::Other& other : curve.others) {
                present.insert(other.metric);
            }
        }
    }
    std::vector<const MetricDefinition*> columns;
    for (const MetricDefinition* metric : frame_metrics()) {
        if (present.contains(metric)) {
            columns.push_back(metric);
        }
    }
    return columns;
}

void MetricPage::refresh_readout_layout() {
    const auto field_width = [](const MetricDefinition& metric) {
        return length_of(qt_text(metric.label) + "=" + text_of_std(metric.format_value(widest_sample)));
    };
    series_width_ = 0;
    for (const QString& label : visible_labels()) {
        series_width_ = std::max(series_width_, length_of("[" + label + "]"));
    }
    main_width_ = std::max(field_width(*metric_), length_of("no " + qt_text(metric_->label)));
    columns_.clear();
    for (const MetricDefinition* metric : extra_columns()) {
        columns_.emplace_back(metric, field_width(*metric));
    }
}

QString MetricPage::readout_row(const QString& prefix, std::optional<double> value,
                                const std::map<std::string, std::optional<double>>& others) const {
    // One line's readout: the shared fields, this page's metric, then "|"
    // and the other metrics, each padded to a width set by the metric, not
    // by the value, so that the columns line up. A line without a column's
    // value leaves it blank rather than shifting the rest along.
    const QString label = qt_text(metric_->label);
    const QString main = value ? label + "=" + text_of_std(metric_->format_value(*value)) : "no " + label;
    if (columns_.empty()) {
        return prefix + main;
    }
    QStringList fields;
    for (const auto& [metric, width] : columns_) {
        const auto found = others.find(std::string(metric->key));
        const QString text = found == others.end() || !found->second
                                 ? QString()
                                 : qt_text(metric->label) + "=" + text_of_std(metric->format_value(*found->second));
        fields.append(pad_right(text, width));
    }
    return trimmed_right(prefix + pad_right(main, main_width_) + "  |  " + fields.join(QStringLiteral("  ")));
}

std::map<std::string, std::optional<double>> MetricPage::other_values(const MetricCurve& curve, int frame,
                                                                      std::size_t index) {
    // The other metrics at this point: by index where they share the line's
    // frame numbers, by frame number where they do not.
    std::map<std::string, std::optional<double>> values;
    for (const MetricCurve::Other& other : curve.others) {
        const std::string key(other.metric->key);
        if (other.same_frames) {
            const double value = other.values[index];
            values[key] = std::isnan(value) ? std::nullopt : std::optional(value);  // NaN: not scored here
        } else {
            values[key] = value_at_frame(other.frames, other.values, frame);
        }
    }
    return values;
}

QString MetricPage::readout_missing_frame(const QString& label, int frame) const {
    return app::tr("{series}  frame {frame:>6}   not in this run",
                   {{"series", pad_right("[" + label + "]", series_width_)}, {"frame", frame}});
}

void MetricPage::set_hover_text(const QString& text) {
    // Dragging across one frame's pixels reports the same each time: not
    // drawn again.
    if (text != hover_text_) {
        hover_text_ = text;
        if (!showing_readout_) {  // the hint gives way, once
            showing_readout_ = true;
            hover_label_->setFont(readout_font_);
            fit_readout_margins();
        }
        hover_label_->set_text(text);
    }
}

std::size_t MetricPage::find_hover_index(const MetricCurve& curve, double x, double y, double half_window) const {
    // The frame to report for the line at the pointer. Not merely the one
    // nearest in time, which makes a narrow dip hard to land on: of the
    // points within `half_window`, the one nearest in time at or below the
    // pointer's height, else the lowest (mirrored for a lower-is-better
    // metric: the worst frame either way).
    const std::span<const double> times = curve.times;
    const auto lo =
        static_cast<std::size_t>(std::lower_bound(times.begin(), times.end(), x - half_window) - times.begin());
    const auto hi =
        static_cast<std::size_t>(std::upper_bound(times.begin(), times.end(), x + half_window) - times.begin());
    if (lo >= hi) {
        return nearest_index_by_time(times, x);
    }
    // Compared as 32-bit floats, as numpy compares the scores with the
    // pointer. NaN is never at or below it.
    const auto at_or_below = static_cast<float>(y * sign_);
    std::optional<std::size_t> nearest;
    double nearest_distance = 0;
    std::optional<std::size_t> lowest;
    float lowest_value = 0;
    for (std::size_t i = lo; i < hi; ++i) {
        const float value = curve.values[i] * sign_;
        if (value <= at_or_below) {
            const double distance = std::abs(times[i] - x);
            if (!nearest || distance < nearest_distance) {
                nearest = i;
                nearest_distance = distance;
            }
        }
        if (std::isfinite(value) && (!lowest || value < lowest_value)) {
            lowest = i;
            lowest_value = value;
        }
    }
    if (nearest) {
        return *nearest;
    }
    // Every point near the pointer missing this metric: the nearest frame in
    // time, whose readout says it has no value.
    return lowest ? *lowest : nearest_index_by_time(times, x);
}

double MetricPage::find_shared_hover_time(const std::vector<const MetricCurve*>& shown, double x, double y,
                                          double x_per_pixel) const {
    // find_hover_index for several lines: one moment for all of them, by the
    // same rule over their points together, so that every line's readout is
    // of the same moment.
    const auto at_or_below = static_cast<float>(y * sign_);
    std::optional<double> best_time;
    double best_distance = 0;
    std::optional<double> fallback_time;
    float fallback_value = 0;
    for (const MetricCurve* curve : shown) {
        const std::span<const double> times = curve->times;
        const double half_window = std::max(step_of(times), x_per_pixel) * hover_search_steps;
        const auto lo = std::lower_bound(times.begin(), times.end(), x - half_window) - times.begin();
        const auto hi = std::upper_bound(times.begin(), times.end(), x + half_window) - times.begin();
        if (lo >= hi) {
            continue;
        }
        std::optional<std::ptrdiff_t> lowest;
        float lowest_value = 0;
        std::optional<std::ptrdiff_t> nearest;
        double nearest_distance = 0;
        for (auto i = lo; i < hi; ++i) {
            const float value = curve->values[static_cast<std::size_t>(i)] * sign_;
            if (std::isfinite(value) && (!lowest || value < lowest_value)) {
                lowest = i;
                lowest_value = value;
            }
            if (value <= at_or_below) {
                const double distance = std::abs(times[static_cast<std::size_t>(i)] - x);
                if (!nearest || distance < nearest_distance) {
                    nearest = i;
                    nearest_distance = distance;
                }
            }
        }
        if (lowest && (!fallback_time || lowest_value < fallback_value)) {
            fallback_time = times[static_cast<std::size_t>(*lowest)];
            fallback_value = lowest_value;
        }
        if (nearest && (!best_time || nearest_distance < best_distance)) {
            best_time = times[static_cast<std::size_t>(*nearest)];
            best_distance = nearest_distance;
        }
    }
    if (best_time) {
        return *best_time;
    }
    return fallback_time ? *fallback_time : x;  // nothing near in any line: the pointer's own time
}

void MetricPage::on_hover(double x, double y, const std::map<int, SeriesEntry>& entries) {
    const double x_per_pixel = chart_->seconds_per_pixel();
    std::vector<std::pair<const SeriesEntry*, const MetricCurve*>> shown;
    for (const auto& [id, curve] : curves_) {
        const auto entry = entries.find(id);
        if (curve.visible && entry != entries.end()) {
            shown.emplace_back(&entry->second, &curve);
        }
    }
    const QString time_line = app::tr("Time: {time}", {{"time", text_of_std(format_hms(x, 2))}});
    if (shown.empty()) {
        chart_->set_cursor_time(x);
        set_hover_text(time_line);
        return;
    }
    std::vector<std::tuple<const SeriesEntry*, const MetricCurve*, std::size_t>> picks;
    if (metric_->kind == MetricKind::sequence) {
        // A per-second line: the second under the pointer. Snapping to dips
        // would look five whole seconds either side.
        for (const auto& [entry, curve] : shown) {
            picks.emplace_back(entry, curve, second_at(*curve, x));
        }
    } else if (shown.size() > 1) {
        // Every line at the same moment: each snapped to its own dip would
        // compare different frames.
        std::vector<const MetricCurve*> curves;
        for (const auto& [entry, curve] : shown) {
            curves.push_back(curve);
        }
        const double target = find_shared_hover_time(curves, x, y, x_per_pixel);
        for (const auto& [entry, curve] : shown) {
            picks.emplace_back(entry, curve, nearest_index_by_time(curve->times, target));
        }
    } else {
        const auto& [entry, curve] = shown.front();
        const double half_window = std::max(step_of(curve->times), x_per_pixel) * hover_search_steps;
        picks.emplace_back(entry, curve, find_hover_index(*curve, x, y, half_window));
    }

    QStringList lines{time_line};
    std::vector<std::pair<QString, double>> found;
    for (const auto& [entry, curve, index] : picks) {
        const double value = curve->values[index];
        const int frame = curve->frames[index];
        const QString prefix = readout_prefix(entry->label, frame, start_time(*curve, index));
        const auto others = other_values(*curve, frame, index);
        if (std::isnan(value)) {
            // The run has the metric but not this frame (subsampled, or a
            // metric that failed on some frames).
            lines.append(readout_row(prefix, std::nullopt, others));
            continue;
        }
        lines.append(readout_row(prefix, value, others));
        found.emplace_back(entry->label, value);
    }
    if (found.size() == 2) {
        const double delta = found[0].second - found[1].second;
        if (!std::isnan(delta)) {
            lines.append(QStringLiteral("Δ (%1 − %2) = %3")
                             .arg(found[0].first, found[1].first, text_of_std(metric_->format_delta(delta))));
        }
    }
    const auto& [entry, curve, index] = picks.front();
    chart_->set_cursor_time(curve->times[index]);
    set_hover_text(lines.join('\n'));
}

bool MetricPage::show_frame(int frame, const std::map<int, SeriesEntry>& entries) {
    // Every shown line at exactly `frame`: unlike hovering, which snaps to a
    // dip nearby, this is the frame asked for, so that two runs can be
    // compared at one moment.
    std::vector<std::pair<const SeriesEntry*, const MetricCurve*>> shown;
    for (const auto& [id, curve] : curves_) {
        const auto entry = entries.find(id);
        if (curve.visible && entry != entries.end()) {
            shown.emplace_back(&entry->second, &curve);
        }
    }
    if (shown.empty()) {
        set_hover_text(app::tr("Frame {frame}: no visible series.", {{"frame", frame}}));
        return false;
    }
    QStringList lines{app::tr("Frame {frame}", {{"frame", frame}})};
    std::vector<std::pair<QString, double>> found;
    std::optional<double> cursor_time;
    for (const auto& [entry, curve] : shown) {
        std::size_t index = 0;
        if (!curve->starts.empty()) {
            // A per-second line: the second the frame is in.
            const auto after = std::upper_bound(curve->frames.begin(), curve->frames.end(), frame);
            if (after == curve->frames.begin()) {
                lines.append(readout_missing_frame(entry->label, frame));
                continue;
            }
            index = static_cast<std::size_t>(after - curve->frames.begin()) - 1;
        } else {
            // A run can be shorter, or subsampled: a frame it lacks is said
            // to be missing, not shown with a neighbour's score.
            const auto at = std::lower_bound(curve->frames.begin(), curve->frames.end(), frame);
            if (at == curve->frames.end() || *at != frame) {
                lines.append(readout_missing_frame(entry->label, frame));
                continue;
            }
            index = static_cast<std::size_t>(at - curve->frames.begin());
        }
        const double value = curve->values[index];
        const QString prefix = readout_prefix(entry->label, curve->frames[index], start_time(*curve, index));
        const auto others = other_values(*curve, frame, index);
        if (std::isnan(value)) {
            lines.append(readout_row(prefix, std::nullopt, others));
            continue;
        }
        lines.append(readout_row(prefix, value, others));
        found.emplace_back(entry->label, value);
        if (!cursor_time) {
            cursor_time = curve->times[index];
        }
    }
    if (found.size() == 2) {
        const double delta = found[0].second - found[1].second;
        if (!std::isnan(delta)) {
            lines.append(QStringLiteral("Δ (%1 − %2) = %3")
                             .arg(found[0].first, found[1].first, text_of_std(metric_->format_delta(delta))));
        }
    }
    if (cursor_time) {
        chart_->set_cursor_time(*cursor_time);
    }
    set_hover_text(lines.join('\n'));
    return !found.empty();
}

std::pair<int, int> MetricPage::frame_range() const {
    std::optional<int> lo;
    std::optional<int> hi;
    for (const auto& [id, curve] : curves_) {
        if (!curve.visible || curve.frames.empty()) {
            continue;
        }
        lo = lo ? std::min(*lo, curve.frames.front()) : curve.frames.front();
        hi = hi ? std::max(*hi, curve.frames.back()) : curve.frames.back();
    }
    return {lo.value_or(0), hi.value_or(0)};
}

// ================================================================ the tab

GraphPanel::GraphPanel(VideoColors* colors, QWidget* parent) : QWidget(parent), colors_(colors) {
    dialogs = standard_dialogs(this);
    connect(colors_, &VideoColors::changed, this,
            [this](const QString& key, const QColor& color) { on_video_color_changed(key, color); });
    // Exports of a feature-length run take seconds each: written off the
    // window's thread, so that the window does not freeze.
    file_writes_ = new FileWrites(this);
    connect(file_writes_, &FileWrites::write_failed, this,
            [this](const QString& description, const QString& error) { on_file_write_failed(description, error); });
    connect(file_writes_, &FileWrites::became_idle, this, [this] { on_file_writes_idle(); });

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);

    // At the top, one table that is both the list of lines and their
    // statistics, shared by every metric's page: the first column has each
    // line's colour, its tick box and its name. Four lines high at most,
    // scrolling beyond, so that a long list leaves room for the graph.
    auto* top = new QGroupBox(app::tr("Series and statistics"));
    auto* top_layout = new QVBoxLayout(top);
    // Without it, nothing said the metric columns could be clicked, or that
    // the detail to their right was of the one clicked.
    stats_hint_ = new QLabel;
    theme::style(stats_hint_, "color: {muted};");
    top_layout->addWidget(stats_hint_);
    stats_table_ = new QTableWidget;
    auto* swatches = new ColorSwatchDelegate(stats_table_);
    connect(swatches, &ColorSwatchDelegate::clicked, this, [this](int row) { choose_video_color(row); });
    stats_table_->setItemDelegateForColumn(0, swatches);
    stats_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    connect(stats_table_, &QTableWidget::itemChanged, this,
            [this](QTableWidgetItem* item) { on_stats_item_changed(item); });
    connect(stats_table_, &QTableWidget::cellClicked, this,
            [this](int row, int column) { on_stats_cell_clicked(row, column); });
    stats_table_->horizontalHeader()->setSectionsClickable(true);
    connect(stats_table_->horizontalHeader(), &QHeaderView::sectionClicked, this,
            [this](int column) { on_stats_header_clicked(column); });
    top_layout->addWidget(stats_table_, 1);
    root->addWidget(top);

    // Below, a tab per metric, each with its full-width chart. Each chart
    // keeps a pixmap of its drawing, its main cost, so pages are built when
    // first shown: the first metric's now, the others when chosen.
    tabs_ = new QuietTabWidget;
    // The metric tabs' own look: the native one can make the selected tab
    // hard to tell from the others.
    tabs_->tabBar()->setStyleSheet(QStringLiteral(
        "QTabBar::tab { padding: 7px 16px; margin-right: 3px; background: palette(button); "
        "color: palette(button-text); border: 1px solid palette(mid); border-bottom: 3px solid transparent; }"
        "QTabBar::tab:selected { background: palette(highlight); color: palette(highlighted-text); "
        "border-bottom: 3px solid palette(highlighted-text); font-weight: bold; }"));
    const std::span<const MetricDefinition> all = metrics();
    tabs_->addTab(build_page(all.front()), qt_text(all.front().label));
    for (std::size_t i = 1; i < all.size(); ++i) {
        tabs_->addTab(new QWidget, qt_text(all[i].label));  // built when first shown
    }
    connect(tabs_, &QTabWidget::currentChanged, this, [this](int index) { on_tab_changed(index); });
    root->addWidget(tabs_, 1);
    root->addWidget(build_action_bar());

    // Exports finish later: said here, not in a dialog taking the focus.
    status_label_ = new QLabel;
    status_label_->hide();
    theme::style(status_label_, "color: {muted};");
    root->addWidget(status_label_);
    metric_hint_ = new QLabel(app::tr("Calculate metrics or load analysis results to view graphs."));
    root->addWidget(metric_hint_);

    setup_stats_table();
    cap_panel_heights();
}

GraphPanel::~GraphPanel() = default;

MetricPage* GraphPanel::build_page(const MetricDefinition& metric) {
    auto* page = new MetricPage(metric);
    connect(&page->chart(), &ChartWidget::hovered, this,
            [this, page](double x, double y) { page->on_hover(x, y, entries_); });
    pages_[std::string(metric.key)] = page;
    return page;
}

void GraphPanel::on_tab_changed(int index) {
    // Builds a page when first shown, and brings the table and the hint up
    // to date for it.
    if (index < 0) {
        return;
    }
    const MetricDefinition& metric = metrics()[static_cast<std::size_t>(index)];
    const std::string key(metric.key);
    if (!pages_.contains(key)) {
        MetricPage* page = build_page(metric);
        const bool blocked = tabs_->blockSignals(true);
        QWidget* placeholder = tabs_->widget(index);
        tabs_->removeTab(index);
        delete placeholder;
        tabs_->insertTab(index, page, qt_text(metric.label));
        tabs_->setCurrentIndex(index);
        tabs_->blockSignals(blocked);
        // The runs already shown: lines are added only to the pages there
        // are.
        for (const auto& [id, entry] : entries_) {
            page->set_curve(id, entry);
        }
    }
    if (!selecting_available_metric_) {
        preferred_metric_ = key;
        emit metric_changed(QString::fromStdString(key));
    }
    update_metric_hint();
    refresh_stats_table();
    refresh_frame_range();
}

void GraphPanel::set_preferred_metric(const std::string& key) {
    preferred_metric_ = find_metric(key) != nullptr ? key : "vmaf";
    select_available_metric();
}

void GraphPanel::update_metric_hint() {
    // A note under the page: why the metric has no data, or how to read it.
    const MetricDefinition& metric = current_metric();
    const auto has = [this](std::string_view key) {
        return std::any_of(entries_.begin(), entries_.end(),
                           [key](const auto& item) { return item.second.result->has_metric(std::string(key)); });
    };
    const bool available = has(metric.key);
    QString hint = available ? QString()
                             : app::tr("{label} was not calculated. Tick it in the {label} column in Videos, or load "
                                       "results containing it.",
                                       {{"label", qt_text(metric.label)}});
    if (available && metric.key == "xpsnr") {
        hint = app::tr(xpsnr_infinity_note, {{"db", xpsnr_infinity_plot_db}});
    }
    if (available && metric.kind == MetricKind::sequence) {
        hint = app::tr(cvvdp_note);
    }
    metric_hint_->setText(hint);
    metric_hint_->setVisible(!hint.isEmpty());
    const std::span<const MetricDefinition> all = metrics();
    for (std::size_t i = 0; i < all.size(); ++i) {
        tabs_->setTabToolTip(static_cast<int>(i), has(all[i].key) ? QString() : app::tr("Not calculated"));
    }
}

void GraphPanel::select_available_metric() {
    // The preferred metric if a run has it, else the first that has data.
    std::vector<std::string> available;
    for (const MetricDefinition& metric : metrics()) {
        if (std::any_of(entries_.begin(), entries_.end(), [&metric](const auto& item) {
                return item.second.result->has_metric(std::string(metric.key));
            })) {
            available.emplace_back(metric.key);
        }
    }
    const std::string key = std::find(available.begin(), available.end(), preferred_metric_) != available.end()
                                ? preferred_metric_
                                : (available.empty() ? preferred_metric_ : available.front());
    const std::span<const MetricDefinition> all = metrics();
    const auto at = std::find_if(all.begin(), all.end(), [&key](const MetricDefinition& m) { return m.key == key; });
    selecting_available_metric_ = true;
    tabs_->setCurrentIndex(static_cast<int>(at - all.begin()));
    selecting_available_metric_ = false;
    update_metric_hint();
}

QWidget* GraphPanel::build_action_bar() {
    // The buttons under the graph: add saved results, export, go to a frame.
    auto* bar = new QWidget;
    auto* layout = new QHBoxLayout(bar);
    auto* add_button = new QPushButton(app::tr("Add analysis results..."));
    connect(add_button, &QPushButton::clicked, this, [this] { on_add_saved_run(); });
    layout->addWidget(add_button);
    auto* png_button = new QPushButton(app::tr("Export graph as PNG"));
    connect(png_button, &QPushButton::clicked, this, [this] { on_export_png(); });
    layout->addWidget(png_button);
    export_csv_button_ = new QPushButton(app::tr("Export CSV..."));
    connect(export_csv_button_, &QPushButton::clicked, this, [this] { on_export_csv(); });
    layout->addWidget(export_csv_button_);
    layout->addSpacing(16);
    layout->addWidget(new QLabel(app::tr("Go to frame:")));
    frame_spin_ = new QSpinBox;
    frame_spin_->setRange(0, 0);
    frame_spin_->setKeyboardTracking(false);  // on Enter, not at each digit typed
    frame_spin_->setToolTip(app::tr("Reports every visible series at this exact frame, so two encodes can be compared "
                                    "at one moment."));
    connect(frame_spin_, &QSpinBox::valueChanged, this, [this](int frame) { on_frame_requested(frame); });
    layout->addWidget(frame_spin_);
    auto* go_button = new QPushButton(app::tr("Go"));
    connect(go_button, &QPushButton::clicked, this, [this] { on_frame_requested(frame_spin_->value()); });
    layout->addWidget(go_button);
    layout->addStretch(1);
    return bar;
}

void GraphPanel::on_frame_requested(int frame) {
    pages_.at(std::string(current_metric().key))->show_frame(frame, entries_);
}

void GraphPanel::refresh_frame_range() {
    // The Go to frame box bounded by what is drawn, so that it cannot ask
    // for a frame no line has.
    const auto page = pages_.find(std::string(current_metric().key));
    if (page == pages_.end()) {
        return;
    }
    const auto [lo, hi] = page->second->frame_range();
    const bool blocked = frame_spin_->blockSignals(true);
    frame_spin_->setRange(lo, std::max(lo, hi));
    frame_spin_->blockSignals(blocked);
}

void GraphPanel::cap_panel_heights() {
    // The table held to four rows, measured from its own font and frame, so
    // that it fits at any text size or display scale.
    const int row_height = stats_table_->verticalHeader()->defaultSectionSize();
    const int header_height = stats_table_->horizontalHeader()->sizeHint().height();
    int chrome = 2 * stats_table_->frameWidth();
    if (const QScrollBar* scroll_bar = stats_table_->horizontalScrollBar();
        scroll_bar != nullptr && scroll_bar->isVisible()) {
        chrome += scroll_bar->height();
    }
    stats_table_->setMaximumHeight(header_height + visible_series_rows * row_height + chrome + 2);
}

const MetricDefinition& GraphPanel::current_metric() const {
    const int index = tabs_->currentIndex();
    return metrics()[static_cast<std::size_t>(index >= 0 ? index : 0)];
}

void GraphPanel::setup_stats_table() {
    // The columns: the series, every metric's mean (the comparison being
    // made, so always there), then the selected metric's statistics in full.
    const MetricDefinition& metric = current_metric();
    QStringList headers{app::tr("Series")};
    for (const MetricDefinition& each : metrics()) {
        headers.append(qt_text(each.label));
    }
    for (const char* label : {N_("Median"), N_("StDev"), N_("Min"), N_("Max")}) {
        headers.append(app::tr(label));
    }
    for (const char* share : tail_shares) {
        headers.append(worst_is_high(metric) ? app::tr("{share} High", {{"share", share}})
                                             : app::tr("{share} Low", {{"share", share}}));
    }
    for (const Threshold& threshold : metric.thresholds) {
        headers.append(threshold_label(threshold));
    }
    headers.append(QString());  // the remove button
    stats_table_->setColumnCount(static_cast<int>(headers.size()));
    stats_table_->setHorizontalHeaderLabels(headers);
    stats_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    stats_table_->verticalHeader()->setVisible(false);
    const std::span<const MetricDefinition> all = metrics();
    for (int column = first_mean_column(); column < mean_column_end(); ++column) {
        QTableWidgetItem* head = stats_table_->horizontalHeaderItem(column);
        if (head == nullptr) {
            continue;
        }
        const MetricDefinition& spec = all[static_cast<std::size_t>(column - first_mean_column())];
        const bool selected = spec.key == metric.key;
        // The axis label carries the unit ("PSNR (dB)"), which the heading
        // leaves off to keep the column narrow.
        const QString axis = qt_text(spec.axis_label);
        head->setToolTip((spec.kind == MetricKind::sequence
                              ? app::tr("{axis_label} of the whole video (not the mean of its seconds).\n",
                                        {{"axis_label", axis}})
                              : app::tr("Mean {axis_label} over all scored frames.\n", {{"axis_label", axis}})) +
                         (selected ? app::tr("Showing its detailed statistics.")
                                   : app::tr("Click this column to show detailed {label} statistics.",
                                             {{"label", qt_text(spec.label)}})));
        // Link-coloured: the columns that can be clicked look it.
        head->setForeground(theme::color(selected ? "link_selected" : "link"));
        QFont font;
        font.setBold(selected);
        head->setFont(font);
    }
    stats_hint_->setText(app::tr("Click a metric column for its full statistics — showing {label}.",
                                 {{"label", qt_text(metric.label)}}));
}

QTableWidgetItem* GraphPanel::series_name_item(const SeriesEntry& entry) const {
    // The first cell: colour square, tick box and name in one, which makes
    // the table the list of lines too.
    auto* item = new QTableWidgetItem(entry.label);
    item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable | Qt::ItemIsSelectable);
    item->setCheckState(entry.visible ? Qt::Checked : Qt::Unchecked);
    // The colour square is the legend.
    QPixmap swatch(12, 12);
    swatch.fill(entry.color);
    item->setData(Qt::DecorationRole, swatch);
    QFont font = item->font();
    font.setBold(true);
    item->setFont(font);
    item->setForeground(theme::readable(entry.color));
    item->setToolTip(app::tr("Click the color square to change this video's color in both graphs."));
    return item;
}

void GraphPanel::choose_video_color(int row) {
    if (const std::optional<int> id = series_id_at_row(row); id && entries_.contains(*id)) {
        colors_->choose(color_path(*entries_.at(*id).result), this);
    }
}

void GraphPanel::on_video_color_changed(const QString& path_key, const QColor& color) {
    // Every run of the video in its new colour, its statistics untouched.
    populating_stats_ = true;
    int row = 0;
    for (auto& [id, entry] : entries_) {
        if (VideoColors::path_key(color_path(*entry.result)) == path_key) {
            entry.color = color;
            for (const auto& [key, page] : pages_) {
                page->chart().set_series_color(id, color);
            }
            QTableWidgetItem* item = series_name_item(entry);
            item->setData(Qt::UserRole, id);
            stats_table_->setItem(row, 0, item);
        }
        ++row;
    }
    populating_stats_ = false;
}

std::optional<int> GraphPanel::series_id_at_row(int row) const {
    const QTableWidgetItem* item = stats_table_->item(row, 0);
    const QVariant id = item == nullptr ? QVariant() : item->data(Qt::UserRole);
    return id.isValid() ? std::optional(id.toInt()) : std::nullopt;
}

void GraphPanel::on_stats_cell_clicked(int row, int column) {
    // The last column's ✕ removes the line; a metric's mean shows its
    // statistics.
    if (is_mean_column(column)) {
        show_metric_detail(column - first_mean_column());
        return;
    }
    if (column != stats_table_->columnCount() - 1) {
        return;
    }
    if (const std::optional<int> id = series_id_at_row(row)) {
        remove_runs({*id});
    }
}

void GraphPanel::on_stats_header_clicked(int column) {
    if (is_mean_column(column)) {
        show_metric_detail(column - first_mean_column());
    }
}

void GraphPanel::show_metric_detail(int index) {
    // The metric the table details is the graph's own, rather than one of
    // its own: they are two views of one metric, and must not disagree.
    if (index >= 0 && index < static_cast<int>(metrics().size()) && index != tabs_->currentIndex()) {
        tabs_->setCurrentIndex(index);
    }
}

void GraphPanel::on_stats_item_changed(QTableWidgetItem* item) {
    // A line's tick box in the table shows or hides it.
    if (populating_stats_ || item->column() != 0) {
        return;
    }
    const QVariant id = item->data(Qt::UserRole);
    if (id.isValid()) {
        set_visible(id.toInt(), item->checkState() == Qt::Checked);
    }
}

void GraphPanel::add_run(const Run& run) {
    if (add_series(run)) {
        series_changed();
    }
}

void GraphPanel::add_runs(const std::vector<Run>& runs) {
    bool changed = false;
    for (const Run& run : runs) {
        changed = add_series(run) || changed;
    }
    if (changed) {
        series_changed();
    }
}

bool GraphPanel::add_series(const Run& run) {
    // Adds or updates a line; whether anything changed. A line showing this
    // very result under this label is left alone: the window replaces a
    // result to change it, never edits it, and drawing every line again at
    // each visit to the tab took 0.7 s with six two-hour videos.
    std::string identity = run.identity;
    if (identity.empty()) {
        // Without a row's identity, one line per test video.
        std::error_code error;
        const std::filesystem::path absolute = std::filesystem::absolute(run.result->distorted, error);
        identity = "path:" + platform::path_text(std::filesystem::weakly_canonical(absolute, error));
    }
    if (suppressed_identities_.contains(identity)) {
        if (!run.restore) {
            return false;
        }
        suppressed_identities_.erase(identity);
    }
    const QString label = run.label.isEmpty() ? text_of(run.result->distorted.stem()) : run.label;
    // A resolution test's or a direction companion's made-up path: the
    // colour of the video it is.
    const std::filesystem::path path = color_path(*run.result);
    const QColor color = colors_->for_path(path);
    for (auto& [id, existing] : entries_) {
        if (existing.identity != identity) {
            continue;
        }
        if (existing.result == run.result && existing.label == label) {
            return false;
        }
        if (VideoColors::path_key(color_path(*existing.result)) != VideoColors::path_key(path)) {
            existing.color = color;
        }
        existing.result = run.result;
        existing.label = label;
        existing.means = metric_means(*run.result);
        existing.identical = identical_frame_counts(*run.result);
        for (const auto& [key, page] : pages_) {
            page->set_curve(id, existing);
        }
        return true;
    }
    const int id = next_id_++;
    SeriesEntry entry{run.result, label, color, true, identity, metric_means(*run.result),
                      identical_frame_counts(*run.result)};
    for (const auto& [key, page] : pages_) {
        page->set_curve(id, entry);
    }
    entries_.emplace(id, std::move(entry));
    return true;
}

void GraphPanel::series_changed() {
    apply_order();
    select_available_metric();
    refresh_stats_table();
    refresh_frame_range();
}

void GraphPanel::set_order(std::vector<std::string> identities) {
    order_ = std::move(identities);
    if (apply_order()) {
        refresh_stats_table();
    }
}

bool GraphPanel::apply_order() {
    // The lines given new ids in the order wanted, as every page and the
    // table go by id; whether any moved.
    std::map<std::string, std::size_t> position;
    for (std::size_t index = 0; index < order_.size(); ++index) {
        position.emplace(order_[index], index);
    }
    std::vector<int> wanted;
    for (const auto& [id, entry] : entries_) {
        wanted.push_back(id);
    }
    const auto place = [&](int id) {
        const auto found = position.find(entries_.at(id).identity);
        return std::pair(found == position.end() ? order_.size() : found->second, id);
    };
    if (std::is_sorted(wanted.begin(), wanted.end(), [&](int a, int b) { return place(a) < place(b); })) {
        return false;
    }
    std::stable_sort(wanted.begin(), wanted.end(), [&](int a, int b) { return place(a) < place(b); });
    std::map<int, int> renumbered;
    std::map<int, SeriesEntry> entries;
    for (const int old : wanted) {
        renumbered.emplace(old, next_id_);
        entries.emplace(next_id_++, std::move(entries_.at(old)));
    }
    entries_ = std::move(entries);
    for (const auto& [key, page] : pages_) {
        page->renumber(renumbered);
    }
    return true;
}

bool GraphPanel::remove_by_path(const std::filesystem::path& distorted) {
    // A test video's line, when its row goes: left, it would show a
    // comparison just discarded, with no row to remove it from.
    for (const auto& [id, entry] : entries_) {
        if (same_path(entry.result->distorted, distorted)) {
            remove_runs({id}, false);
            return true;
        }
    }
    return false;
}

bool GraphPanel::remove_by_identity(const std::string& identity) {
    for (const auto& [id, entry] : entries_) {
        if (entry.identity == identity) {
            remove_runs({id}, false);
            return true;
        }
    }
    return false;
}

void GraphPanel::remove_by_identities(const std::vector<std::string>& identities,
                                      const std::vector<std::filesystem::path>& paths) {
    // remove_by_identity for each, or for one not found remove_by_path for
    // the path beside it (a line added by path). The table, the frame range
    // and the pages brought up to date once: one at a time, three hundred
    // lines took 12 s.
    std::map<std::string, int> by_identity;
    for (const auto& [id, entry] : entries_) {
        by_identity.emplace(entry.identity, id);
    }
    std::vector<int> chosen;
    for (std::size_t i = 0; i < identities.size(); ++i) {
        std::optional<int> id;
        if (const auto found = by_identity.find(identities[i]); found != by_identity.end()) {
            id = found->second;
        } else if (i < paths.size()) {
            for (const auto& [candidate, entry] : entries_) {
                if (std::find(chosen.begin(), chosen.end(), candidate) == chosen.end() &&
                    same_path(entry.result->distorted, paths[i])) {
                    id = candidate;
                    break;
                }
            }
        }
        if (id && std::find(chosen.begin(), chosen.end(), *id) == chosen.end()) {
            chosen.push_back(*id);
        }
    }
    remove_runs(chosen, false);
}

void GraphPanel::remove_runs(const std::vector<int>& series_ids, bool suppress) {
    std::vector<int> removed;
    for (const int id : series_ids) {
        const auto found = entries_.find(id);
        if (found == entries_.end()) {
            continue;
        }
        if (suppress) {
            suppressed_identities_.insert(found->second.identity);
        }
        entries_.erase(found);
        removed.push_back(id);
    }
    if (removed.empty()) {
        return;
    }
    for (const auto& [key, page] : pages_) {
        page->remove_curves(removed);
    }
    select_available_metric();
    refresh_stats_table();
    refresh_frame_range();
}

void GraphPanel::set_series_visible(int series_id, bool visible) {
    set_visible(series_id, visible);
    refresh_stats_table();
    refresh_frame_range();
}

void GraphPanel::set_visible(int series_id, bool visible) {
    const auto found = entries_.find(series_id);
    if (found == entries_.end() || found->second.visible == visible) {
        return;
    }
    found->second.visible = visible;
    for (const auto& [key, page] : pages_) {
        page->set_visible(series_id, visible);
    }
}

void GraphPanel::refresh_theme() { refresh_stats_table(); }

bool GraphPanel::wait_until_file_writes_idle(std::chrono::milliseconds timeout) {
    return file_writes_->wait_until_idle(timeout);
}

void GraphPanel::refresh_stats_table() {
    // A row per line (shown or not: the table is the list of lines, and a
    // hidden one needs its row to be ticked again), every metric's mean, and
    // the selected metric's statistics.
    populating_stats_ = true;
    setup_stats_table();
    const MetricDefinition& metric = current_metric();
    const MetricPage* page = pages_.at(std::string(metric.key));
    const std::span<const MetricDefinition> all = metrics();
    const auto number_item = [](const QString& text) {
        // Right-aligned in a monospaced face: near-equal numbers (SSIM's
        // 0.9938 against 0.9699) differ in digits that line up.
        auto* item = new QTableWidgetItem(text);
        item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        item->setFont(QFont(QString::fromLatin1(number_font_family), QFont().pointSize()));
        return item;
    };
    stats_table_->setRowCount(static_cast<int>(entries_.size()));
    int row = 0;
    for (const auto& [id, entry] : entries_) {
        QTableWidgetItem* name = series_name_item(entry);
        name->setData(Qt::UserRole, id);
        stats_table_->setItem(row, 0, name);
        // Every metric's mean, its page built or not: a page not yet shown
        // must not decide what the comparison says.
        for (int column = first_mean_column(); column < mean_column_end(); ++column) {
            const MetricDefinition& spec = all[static_cast<std::size_t>(column - first_mean_column())];
            const std::string key(spec.key);
            const auto mean = entry.means.find(key);
            QTableWidgetItem* item = number_item(
                text_of_std(spec.format_value(mean == entry.means.end() ? std::nullopt : mean->second)));
            if (const auto identical = entry.identical.find(key); identical != entry.identical.end()) {
                item->setToolTip(app::tr("{identical} frames identical to the reference are included in the XPSNR "
                                         "sequence average: zero distortion, with their frames included in the count.",
                                         {{"identical", identical->second}}));
            }
            const bool selected = spec.key == metric.key;
            item->setBackground(theme::color(selected ? "mean_selected" : "mean_tint"));
            if (selected) {
                QFont font = item->font();
                font.setBold(true);
                item->setFont(font);
            } else {
                item->setToolTip(app::tr("Click this column to show detailed {label} statistics.",
                                         {{"label", qt_text(spec.label)}}));
            }
            stats_table_->setItem(row, column, item);
        }
        const int detail_start = mean_column_end();
        std::vector<QString> cells;
        if (const MetricCurve* curve = page->curve(id)) {
            // At the metric's own precision (SSIM's whole range is 0-1); the
            // mean has its own column.
            const std::vector<QString> stats = statistics_cells(curve->stats, metric);
            cells.assign(stats.begin() + 1, stats.end());
        } else {
            cells.assign(static_cast<std::size_t>(stats_table_->columnCount() - detail_start - 1), QString());
        }
        for (std::size_t i = 0; i < cells.size(); ++i) {
            stats_table_->setItem(row, detail_start + static_cast<int>(i), number_item(cells[i]));
        }
        // A plain item, handled by cellClicked: a button in the cell was
        // drawn over the first column before the widths settled.
        auto* remove = new QTableWidgetItem(QStringLiteral("✕"));
        remove->setFont(symbol_font(stats_table_->font()));
        remove->setFlags(Qt::ItemIsEnabled);
        remove->setTextAlignment(Qt::AlignCenter);
        remove->setToolTip(app::tr("Remove {label} from the graph", {{"label", entry.label}}));
        stats_table_->setItem(row, stats_table_->columnCount() - 1, remove);
        ++row;
    }
    populating_stats_ = false;
    cap_panel_heights();
}

// -------------------------------------------------------------- exports

std::pair<std::vector<QString>, std::vector<std::tuple<QString, QColor, std::vector<QString>>>>
GraphPanel::export_table() {
    // The exported image's table: its headings, and (label, colour, cells)
    // for each line drawn on the chart.
    const MetricDefinition& metric = current_metric();
    const MetricPage* page = pages_.at(std::string(metric.key));
    std::vector<std::pair<const SeriesEntry*, const MetricCurve*>> shown;
    for (const auto& [id, curve] : page->curves()) {
        if (const auto entry = entries_.find(id); curve.visible && entry != entries_.end()) {
            shown.emplace_back(&entry->second, &curve);
        }
    }
    if (shown.empty()) {
        return {};
    }
    // The statistics' own names, in English as in the Python app's image.
    std::vector<QString> headers{app::tr("Series"), "Mean", "Median", "StDev", "Min", "Max"};
    for (const char* share : tail_shares) {
        headers.push_back(QString::fromLatin1(share) + (worst_is_high(metric) ? " High" : " Low"));
    }
    // The metric's bands, not the first line's: a line with no scored frame
    // has none.
    for (const Threshold& threshold : metric.thresholds) {
        headers.push_back(threshold_label(threshold));
    }
    const bool per_video = metric.kind == MetricKind::sequence;
    if (per_video) {
        // The first statistic would be the mean of the seconds, which is not
        // CVVDP's score: the whole video's JOD takes its place.
        headers[1] = app::tr("Whole video");
    }
    std::vector<std::tuple<QString, QColor, std::vector<QString>>> rows;
    for (const auto& [entry, curve] : shown) {
        std::vector<QString> cells = statistics_cells(curve->stats, metric);
        if (per_video) {
            const auto overall = entry->means.find(std::string(metric.key));
            cells[0] = text_of_std(metric.format_value(overall == entry->means.end() ? std::nullopt : overall->second));
        }
        rows.emplace_back(entry->label, entry->color, std::move(cells));
    }
    return {std::move(headers), std::move(rows)};
}

QPixmap GraphPanel::render_export_image() {
    // The bare chart is a set of unlabelled lines, which says nothing once
    // the file leaves the app: the title, a legend keyed by the lines'
    // colours, and the statistics go around it.
    const MetricDefinition& metric = current_metric();
    const QPixmap chart = pages_.at(std::string(metric.key))->chart().render_to_pixmap();
    const auto [headers, rows] = export_table();

    QFont title_font(font());
    title_font.setBold(true);
    title_font.setPointSize(std::max(10, title_font.pointSize() + 3));
    const QFontMetrics title_metrics(title_font);
    QString title = app::tr("{label} vs time", {{"label", qt_text(metric.label)}});
    if (metric.key == "xpsnr") {
        title += app::tr(" — ∞ plotted at {db:g} dB (display only)", {{"db", xpsnr_infinity_plot_db}});
    }
    QFont cell_font(QStringLiteral("Consolas"));
    cell_font.setStyleHint(QFont::Monospace);
    const QFontMetrics cell_metrics(cell_font);
    const int row_height = cell_metrics.lineSpacing() + export_row_padding;

    // The first column carries the colour square too.
    std::vector<int> widths;
    for (std::size_t column = 0; column < headers.size(); ++column) {
        int width = cell_metrics.horizontalAdvance(headers[column]);
        for (const auto& [label, color, cells] : rows) {
            width = std::max(width, cell_metrics.horizontalAdvance(column == 0 ? label : cells[column - 1]));
        }
        if (column == 0) {
            width += export_swatch + export_gap;
        }
        widths.push_back(width + 2 * export_gap);
    }
    int table_width = 0;
    for (const int width : widths) {
        table_width += width;
    }
    const int table_height = rows.empty() ? 0 : static_cast<int>(rows.size() + 1) * row_height;
    const int content_width = std::max(chart.width(), table_width);
    const int height = export_margin + title_metrics.height() + export_gap + chart.height() +
                       (rows.empty() ? 0 : export_gap + table_height) + export_margin;

    QPixmap image(content_width + 2 * export_margin, height);
    image.fill(Qt::white);
    QPainter painter(&image);
    painter.setPen(QColor(QStringLiteral("#111111")));
    painter.setFont(title_font);
    painter.drawText(export_margin, export_margin + title_metrics.ascent(), title);
    int y = export_margin + title_metrics.height() + export_gap;
    painter.drawPixmap(export_margin, y, chart);
    y += chart.height() + export_gap;
    if (!rows.empty()) {
        QFont header_font(cell_font);
        header_font.setBold(true);
        int baseline = y + cell_metrics.ascent() + export_row_padding / 2;
        painter.setFont(header_font);
        painter.setPen(QColor(QStringLiteral("#111111")));
        int x = export_margin;
        for (std::size_t column = 0; column < headers.size(); ++column) {
            painter.drawText(x + export_gap, baseline, headers[column]);
            x += widths[column];
        }
        // The plain cell font again, not the headings' bold.
        painter.setFont(cell_font);
        int index = 1;
        for (const auto& [label, color, cells] : rows) {
            baseline = y + index * row_height + cell_metrics.ascent() + export_row_padding / 2;
            x = export_margin;
            // The square ties the row to its line on the chart above.
            painter.fillRect(x + export_gap, baseline - export_swatch, export_swatch, export_swatch, color);
            painter.setPen(QColor(QStringLiteral("#111111")));
            painter.drawText(x + export_gap + export_swatch + export_gap, baseline, label);
            x += widths[0];
            for (std::size_t column = 0; column < cells.size(); ++column) {
                painter.drawText(x + export_gap, baseline, cells[column]);
                x += widths[column + 1];
            }
            ++index;
        }
    }
    painter.end();
    return image;
}

void GraphPanel::on_add_saved_run() {
    // A saved run's line.
    const QString chosen = dialogs.open_file(
        app::tr("Open analysis results"),
        app::tr("Analysis results (*{suffix})",
                {{"suffix", QString::fromUtf8(result_suffix.data(), static_cast<int>(result_suffix.size()))}}));
    if (chosen.isEmpty()) {
        return;
    }
    const std::filesystem::path path = path_of(chosen);
    try {
        auto [result, label] = load_run(path);
        std::error_code error;
        const std::filesystem::path resolved =
            std::filesystem::weakly_canonical(std::filesystem::absolute(path, error), error);
        add_run({std::make_shared<const ComparisonResult>(std::move(result)), QString::fromStdString(label),
                 "saved-file:" + platform::path_text(resolved), true});
    } catch (const RunFileError& error) {
        dialogs.alert(app::tr("Failed to load run"), QString::fromStdString(error.what()));
    }
}

void GraphPanel::on_export_png() {
    const QString suggested = QString::fromStdString(std::string(current_metric().key)) + "_comparison.png";
    const QString path = dialogs.save_file(app::tr("Export graph"), suggested, app::tr("PNG image (*.png)"));
    if (!path.isEmpty()) {
        render_export_image().save(path);
    }
}

void GraphPanel::on_export_csv() {
    // Each line's scores as CSV, off the window's thread: a feature-length
    // run's is bigger than its saved run, one per line. Only this button
    // waits for them.
    if (entries_.empty()) {
        dialogs.inform(app::tr("No data"), app::tr("There are no series to export."));
        return;
    }
    const QString folder = dialogs.choose_folder(app::tr("Choose export folder"), {});
    if (folder.isEmpty()) {
        return;
    }
    export_csv_button_->setEnabled(false);
    std::set<std::filesystem::path> reserved;
    for (const auto& [id, entry] : entries_) {
        const std::filesystem::path out =
            unique_output_path(path_of(folder), entry.label.toStdString(), ".csv", reserved);
        file_writes_->submit(QStringLiteral("export %1").arg(name_of(out)),
                             [result = entry.result, out] { export_csv(*result, out); });
    }
    export_destination_ = folder;
    status_label_->show();
    status_label_->setText(ntr("Exporting {count} CSV file to {directory}...",
                               "Exporting {count} CSV files to {directory}...",
                               static_cast<long long>(entries_.size()), {{"directory", folder}}));
}

void GraphPanel::on_file_writes_idle() {
    // The export is done: said, and the button usable again.
    if (!file_writes_->idle()) {
        return;
    }
    export_csv_button_->setEnabled(true);
    if (export_destination_) {
        status_label_->setText(app::tr("Export complete: {path}", {{"path", *export_destination_}}));
        status_label_->show();
        export_destination_.reset();
    }
}

void GraphPanel::on_file_write_failed(const QString& description, const QString& error) {
    status_label_->setText(
        app::tr("Could not {description}: {error}", {{"description", description}, {"error", error}}));
    status_label_->show();
}

}  // namespace vml::app
