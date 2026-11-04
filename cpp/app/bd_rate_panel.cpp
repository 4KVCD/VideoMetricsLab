#include "bd_rate_panel.h"

#include <QAbstractItemView>
#include <QClipboard>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFileInfo>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QSaveFile>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>
#include <set>

#include "i18n.h"
#include "rate_curve_chart.h"
#include "theme.h"
#include "vml/append.h"
#include "vml/bd_rate.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

// Points on a drawn curve: enough for it to look smooth at any width.
constexpr int curve_samples = 96;
// The results table shows this many rows before it scrolls.
constexpr int visible_result_rows = 20;

const char* const add_videos_tooltip =
    N_("Add videos scored in this session. Each one's sequence is its reference's file name, and its bitrate is "
       "the video's own.");
const char* const import_tooltip =
    N_("Add encodes from a CSV file with a row per encode: columns sequence, kbps, and a column per metric key "
       "(vmaf, psnr, ssim...), as the command line's bdrate reads.");

// The metric keys, in the registry's order: the results' columns.
std::vector<std::string> all_metric_keys() {
    std::vector<std::string> keys;
    for (const MetricDefinition& metric : metrics()) {
        keys.emplace_back(metric.key);
    }
    return keys;
}

// A BD-rate as a cell shows it: "−26.6%", "+0.4%", "n/a".
QString percent_cell_text(const std::optional<double>& value) {
    if (!value) {
        return app::tr("n/a");
    }
    QString text = qt_text(bd_rate_percent_text(value));
    if (text.startsWith(QLatin1Char('-'))) {
        text[0] = QChar(0x2212);  // the minus sign, as the window shows negatives
    }
    return text;
}

// A results cell: its text, its colour (green for bitrate saved, red for
// bitrate lost, faint for none) and why it has none.
QTableWidgetItem* percent_item(const std::optional<double>& value, const QString& why) {
    auto* item = new QTableWidgetItem(percent_cell_text(value));
    item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    if (!value) {
        theme::foreground(item, theme::color("faint"));
        item->setToolTip(why);
    } else if (*value <= -0.05) {  // shown as -0.1% or less
        theme::foreground(item, theme::color("good"));
    } else if (*value >= 0.05) {
        theme::foreground(item, theme::color("failed"));
    }
    return item;
}

QTableWidgetItem* text_item(const QString& text) {
    auto* item = new QTableWidgetItem(text);
    item->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    return item;
}

// A table sized to its rows, up to `rows` of them: the tab scrolls, not the
// table, for a usual number of sequences.
void fit_height(QTableWidget* table, int rows) {
    int height = table->horizontalHeader()->height() + 2 * table->frameWidth();
    for (int row = 0; row < std::min(rows, table->rowCount()); ++row) {
        height += table->rowHeight(row);
    }
    table->setFixedHeight(height);
}

// The real dialog for BdRatePanel::choose_videos: a tick list.
std::vector<int> choose_from_list(QWidget* parent, const QStringList& items) {
    QDialog dialog(parent);
    dialog.setWindowTitle(app::tr("Add scored videos"));
    auto* layout = new QVBoxLayout(&dialog);
    layout->addWidget(new QLabel(app::tr("Tick the videos to add.")));
    auto* list = new QListWidget;
    for (const QString& text : items) {
        auto* item = new QListWidgetItem(text, list);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Unchecked);
    }
    list->setMinimumWidth(560);
    layout->addWidget(list);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
    std::vector<int> chosen;
    if (dialog.exec() == QDialog::Accepted) {
        for (int row = 0; row < list->count(); ++row) {
            if (list->item(row)->checkState() == Qt::Checked) {
                chosen.push_back(row);
            }
        }
    }
    return chosen;
}

// A side's one-encode table for a scored video.
RateTable table_of(RateTable::Encode encode) {
    RateTable table;
    table.columns = {"sequence", "kbps"};
    for (const std::string& key : all_metric_keys()) {
        const auto score = encode.scores.find(key);
        if (score != encode.scores.end() && score->second) {
            table.metric_keys.push_back(key);
            table.columns.push_back(key);
        }
    }
    table.encodes.push_back(std::move(encode));
    return table;
}

}  // namespace

BdRatePanel::BdRatePanel(std::function<std::vector<GraphPanel::Run>()> runs, QWidget* parent)
    : QWidget(parent), runs_(std::move(runs)) {
    dialogs = standard_dialogs(this);
    choose_videos = [this](const QStringList& items) { return choose_from_list(this, items); };
    copy_text = [](const QString& text) { QGuiApplication::clipboard()->setText(text); };

    auto* root = new QVBoxLayout(this);
    auto* sides = new QHBoxLayout;
    sides->addWidget(build_side(anchor, app::tr("Anchor")), 1);
    sides->addWidget(build_side(test, app::tr("Test")), 1);
    root->addLayout(sides);
    add_options_row(root);
    add_results(root);
    add_chart(root);
    root->addStretch(1);
    recalculate();
}

// A side's box: its name, its sequences, and its buttons.
QGroupBox* BdRatePanel::build_side(Side& side, const QString& title) {
    side.box = new QGroupBox(title);
    auto* layout = new QVBoxLayout(side.box);
    auto* name_row = new QHBoxLayout;
    name_row->addWidget(new QLabel(app::tr("Name:")));
    side.name_edit = new QLineEdit;
    side.name_edit->setPlaceholderText(title);
    connect(side.name_edit, &QLineEdit::textChanged, this, [this] { on_name_changed(); });
    name_row->addWidget(side.name_edit, 1);
    side.count_label = new QLabel;
    name_row->addWidget(side.count_label);
    layout->addLayout(name_row);

    side.sequences = new QTableWidget(0, 3);
    side.sequences->setHorizontalHeaderLabels({app::tr("Sequence"), app::tr("Encodes"), app::tr("Metrics scored")});
    side.sequences->setEditTriggers(QAbstractItemView::NoEditTriggers);
    side.sequences->setSelectionMode(QAbstractItemView::NoSelection);
    side.sequences->verticalHeader()->setVisible(false);
    side.sequences->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    side.sequences->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    side.sequences->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    side.sequences->setMaximumHeight(160);
    layout->addWidget(side.sequences);

    auto* buttons = new QHBoxLayout;
    side.add_videos_button = new QPushButton(app::tr("Add videos\xE2\x80\xA6"));
    side.add_videos_button->setToolTip(app::tr(add_videos_tooltip));
    side.import_button = new QPushButton(app::tr("Import CSV\xE2\x80\xA6"));
    side.import_button->setToolTip(app::tr(import_tooltip));
    side.clear_button = new QPushButton(app::tr("Clear"));
    connect(side.add_videos_button, &QPushButton::clicked, this, [this, &side] { on_add_videos(side); });
    connect(side.import_button, &QPushButton::clicked, this, [this, &side] { on_import(side); });
    connect(side.clear_button, &QPushButton::clicked, this, [this, &side] { on_clear(side); });
    buttons->addWidget(side.add_videos_button);
    buttons->addWidget(side.import_button);
    buttons->addWidget(side.clear_button);
    buttons->addStretch(1);
    layout->addLayout(buttons);
    refresh_side_table(side);
    return side.box;
}

// Group by, the warnings, Export CSV and Copy as markdown.
void BdRatePanel::add_options_row(QVBoxLayout* root) {
    auto* row = new QHBoxLayout;
    row->addWidget(new QLabel(app::tr("Group by:")));
    group_combo = new QComboBox;
    group_combo->setToolTip(app::tr("A mean and median for each value of this column of the anchor's CSV files."));
    connect(group_combo, &QComboBox::currentIndexChanged, this, [this] { recalculate(); });
    row->addWidget(group_combo);
    warnings_button = new QPushButton;
    connect(warnings_button, &QPushButton::clicked, this, [this] { on_show_warnings(); });
    row->addWidget(warnings_button);
    row->addStretch(1);
    export_button = new QPushButton(app::tr("Export CSV\xE2\x80\xA6"));
    connect(export_button, &QPushButton::clicked, this, [this] { on_export(); });
    copy_button = new QPushButton(app::tr("Copy as markdown"));
    connect(copy_button, &QPushButton::clicked, this, [this] { on_copy(); });
    row->addWidget(export_button);
    row->addWidget(copy_button);
    root->addLayout(row);
    refresh_group_choices();
}

// The heading and the table: a column per metric, a row per summary and
// sequence.
void BdRatePanel::add_results(QVBoxLayout* root) {
    heading_label = new QLabel;
    heading_label->setWordWrap(true);
    root->addWidget(heading_label);
    results = new QTableWidget(0, 1 + static_cast<int>(metrics().size()));
    QStringList labels{QString()};
    for (const MetricDefinition& metric : metrics()) {
        labels.append(qt_text(metric.short_label));
    }
    results->setHorizontalHeaderLabels(labels);
    results->setEditTriggers(QAbstractItemView::NoEditTriggers);
    results->setSelectionBehavior(QAbstractItemView::SelectRows);
    results->setSelectionMode(QAbstractItemView::SingleSelection);
    results->verticalHeader()->setVisible(false);
    results->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    for (int column = 1; column < results->columnCount(); ++column) {
        results->horizontalHeader()->setSectionResizeMode(column, QHeaderView::ResizeToContents);
    }
    connect(results, &QTableWidget::currentCellChanged, this, [this](int row) { on_results_row(row); });
    root->addWidget(results);
}

// The chosen sequence's name, the metric to draw, and the chart.
void BdRatePanel::add_chart(QVBoxLayout* root) {
    auto* row = new QHBoxLayout;
    sequence_label = new QLabel;
    QFont bold = sequence_label->font();
    bold.setBold(true);
    sequence_label->setFont(bold);
    row->addWidget(sequence_label);
    metric_combo = new QComboBox;
    for (const MetricDefinition& metric : metrics()) {
        metric_combo->addItem(qt_text(metric.short_label), qt_text(metric.key));
    }
    connect(metric_combo, &QComboBox::currentIndexChanged, this, [this] { refresh_chart(); });
    row->addWidget(metric_combo);
    row->addStretch(1);
    root->addLayout(row);
    chart = new RateCurveChart;
    chart->setMinimumHeight(300);
    root->addWidget(chart);
}

void BdRatePanel::refresh_theme() {
    fill_results();  // its cells' colours are set one by one
    chart->update();
}

// ------------------------------------------------------------- the encodes

void BdRatePanel::on_add_videos(Side& side) {
    const std::vector<GraphPanel::Run> runs = runs_();
    if (runs.empty()) {
        dialogs.inform(app::tr("No scored videos"),
                       app::tr("Score videos on the Videos tab first, or import a CSV file."));
        return;
    }
    QStringList items;
    for (const GraphPanel::Run& run : runs) {
        const ComparisonResult& result = *run.result;
        const QString name = run.label.isEmpty() ? qt_text(platform::path_text(result.distorted.filename())) : run.label;
        const long long bits_per_second = result.distorted_info.bit_rate;
        const QString rate = bits_per_second > 0
                                 ? app::tr("{rate:.0f} kb/s", {{"rate", static_cast<double>(bits_per_second) / 1000}})
                                 : app::tr("bitrate unknown");
        items.append(app::tr("{video} (reference {reference}, {rate})",
                             {{"video", name},
                              {"reference", qt_text(platform::path_text(result.source.filename()))},
                              {"rate", rate}}));
    }
    QStringList unknown;
    for (const int index : choose_videos(items)) {
        const GraphPanel::Run& run = runs[static_cast<std::size_t>(index)];
        const std::optional<RateTable::Encode> encode = rate_encode(*run.result);
        const QString name = qt_text(platform::path_text(run.result->distorted.filename()));
        if (!encode) {
            unknown.append(name);
            continue;
        }
        const QString key = QStringLiteral("video:") +
                            (run.identity.empty() ? qt_text(platform::path_text(run.result->distorted))
                                                  : qt_text(run.identity));
        std::erase_if(side.sources, [&key](const auto& source) { return source.first == key; });
        side.sources.emplace_back(key, table_of(*encode));
        side.whole_file_rates.erase(key);
        if (run.result->distorted_info.bit_rate_whole_file) {
            side.whole_file_rates[key] = name;
        }
    }
    if (!unknown.isEmpty()) {
        dialogs.inform(app::tr("Bitrate unknown"),
                       app::tr("These videos were not added, as their bitrate is not known:\n{videos}",
                               {{"videos", unknown.join(QLatin1Char('\n'))}}));
    }
    side_changed(side);
}

void BdRatePanel::on_import(Side& side) {
    const QString path = dialogs.open_file(app::tr("Import encodes"), app::tr("CSV file (*.csv)"));
    if (!path.isEmpty()) {
        import_csv(side, path);
    }
}

void BdRatePanel::import_csv(Side& side, const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        dialogs.alert(app::tr("Import failed"), app::tr("Could not read {path}", {{"path", path}}));
        return;
    }
    const QByteArray bytes = file.readAll();
    RateTable table;
    try {
        table = read_rate_table(std::string_view(bytes.constData(), static_cast<std::size_t>(bytes.size())),
                                QFileInfo(path).fileName().toStdString());
    } catch (const std::exception& error) {
        dialogs.alert(app::tr("Import failed"), qt_text(error.what()));
        return;
    }
    const QString key = QStringLiteral("csv:") + QFileInfo(path).absoluteFilePath();
    std::erase_if(side.sources, [&key](const auto& source) { return source.first == key; });
    side.sources.emplace_back(key, std::move(table));
    if (side.name_edit->text().trimmed().isEmpty()) {
        side.name_edit->setText(QFileInfo(path).completeBaseName());
    }
    side_changed(side);
}

void BdRatePanel::on_clear(Side& side) {
    side.sources.clear();
    side.whole_file_rates.clear();
    side.name_edit->clear();  // it named what was cleared
    side_changed(side);
}

void BdRatePanel::side_changed(Side& side) {
    refresh_side_table(side);
    refresh_group_choices();
    recalculate();
}

RateTable BdRatePanel::merged(const Side& side) {
    RateTable all;
    std::set<std::string> metric_keys;
    for (const auto& [_, table] : side.sources) {
        metric_keys.insert(table.metric_keys.begin(), table.metric_keys.end());
        for (const std::string& column : table.columns) {
            if (std::ranges::count(all.columns, column) == 0) {
                all.columns.push_back(column);
            }
        }
        append(all.encodes, table.encodes);
    }
    for (const std::string& key : all_metric_keys()) {
        if (metric_keys.contains(key)) {
            all.metric_keys.push_back(key);
        }
    }
    return all;
}

// A row per sequence: its encodes, and the metrics scored at every one.
void BdRatePanel::refresh_side_table(Side& side) {
    std::map<std::string, std::vector<const RateTable::Encode*>> by_name;
    const RateTable all = merged(side);
    for (const RateTable::Encode& encode : all.encodes) {
        by_name[encode.sequence].push_back(&encode);
    }
    side.sequences->setRowCount(static_cast<int>(by_name.size()));
    int row = 0;
    for (const auto& [name, encodes] : by_name) {
        int scored = 0;
        for (const std::string& key : all_metric_keys()) {
            scored += std::ranges::all_of(encodes, [&key](const RateTable::Encode* encode) {
                const auto score = encode->scores.find(key);
                return score != encode->scores.end() && score->second.has_value();
            });
        }
        side.sequences->setItem(row, 0, text_item(qt_text(name)));
        side.sequences->setItem(row, 1, text_item(QString::number(encodes.size())));
        side.sequences->setItem(row, 2, text_item(QString::number(scored)));
        ++row;
    }
    side.count_label->setText(app::tr("Sequences: {sequences}, encodes: {encodes}",
                                      {{"sequences", static_cast<long long>(by_name.size())},
                                       {"encodes", static_cast<long long>(all.encodes.size())}}));
    side.clear_button->setEnabled(!side.sources.empty());
}

// None, then the anchor's text columns; the choice kept while it is there.
void BdRatePanel::refresh_group_choices() {
    const QString chosen = group_combo->currentData().toString();
    const QSignalBlocker blocker(group_combo);
    group_combo->clear();
    group_combo->addItem(app::tr("None"), QString());
    for (const std::string& column : group_columns(merged(anchor))) {
        group_combo->addItem(qt_text(column), qt_text(column));
    }
    group_combo->setCurrentIndex(std::max(0, group_combo->findData(chosen)));
}

QString BdRatePanel::side_name(const Side& side) const {
    const QString name = side.name_edit->text().trimmed();
    return name.isEmpty() ? side.name_edit->placeholderText() : name;
}

// ------------------------------------------------------------- the results

void BdRatePanel::recalculate() {
    report_.reset();
    warnings_.clear();
    const RateTable anchor_table = merged(anchor);
    const RateTable test_table = merged(test);
    if (anchor_table.encodes.empty() || test_table.encodes.empty()) {
        heading_label->setText(app::tr("Add the anchor's encodes and the test's: scored videos, or CSV files in the "
                                       "command line's bdrate format."));
    } else {
        BdRateOptions options;
        options.metrics = all_metric_keys();
        if (const QString column = group_combo->currentData().toString(); !column.isEmpty()) {
            options.group_by = column.toStdString();
        }
        try {
            BdRateReport report = bd_rate_report(anchor_table, test_table, options);
            for (const std::string& warning : report.warnings) {
                warnings_.append(qt_text(warning));
            }
            if (report.sequences.empty()) {
                heading_label->setText(app::tr("the anchor and the test have no sequence in common"));
            } else {
                report_ = std::move(report);
                show_heading();
            }
        } catch (const std::exception& error) {
            heading_label->setText(qt_text(error.what()));
        }
    }
    for (const Side* side : {&anchor, &test}) {
        for (const auto& [_, video] : side->whole_file_rates) {
            warnings_.append(app::tr("{video}: its bitrate is the whole file's, soundtrack included",
                                     {{"video", video}}));
        }
    }
    if (report_) {
        const bool still_there = chosen_sequence_ && std::ranges::any_of(report_->sequences, [this](const auto& one) {
                                     return one.name == *chosen_sequence_;
                                 });
        if (!still_there) {
            chosen_sequence_ = report_->sequences.front().name;
        }
    } else {
        chosen_sequence_.reset();
    }
    warnings_button->setText(app::tr("Warnings: {count}", {{"count", static_cast<long long>(warnings_.size())}}));
    warnings_button->setVisible(!warnings_.isEmpty());
    export_button->setEnabled(report_.has_value());
    copy_button->setEnabled(report_.has_value());
    fill_results();
    refresh_chart();
}

// The names are in the heading, the chart's legend and the copied table.
void BdRatePanel::on_name_changed() {
    if (report_) {
        show_heading();
        refresh_chart();
    }
}

void BdRatePanel::show_heading() {
    heading_label->setText(
        app::tr("{test} against {anchor}. Sequences in both: {count}. Negative: less bitrate for the same score.",
                {{"test", side_name(test)},
                 {"anchor", side_name(anchor)},
                 {"count", static_cast<long long>(report_->sequences.size())}}));
}

void BdRatePanel::fill_results() {
    filling_ = true;
    sequence_rows_.clear();
    results->setRowCount(0);
    if (!report_) {
        fit_height(results, 0);
        filling_ = false;
        return;
    }
    const BdRateReport& report = *report_;
    const auto add_row = [&](const QString& name, const BdRateRow& row, const BdRateSequence* sequence) {
        const int index = results->rowCount();
        results->insertRow(index);
        QTableWidgetItem* name_item = text_item(name);
        results->setItem(index, 0, name_item);
        for (std::size_t column = 0; column < report.metrics.size(); ++column) {
            const std::string& key = report.metrics[column];
            QString why = app::tr("No sequence has a BD-rate for this metric.");
            if (sequence != nullptr) {
                const auto reason = sequence->why.find(key);
                why = reason == sequence->why.end() ? QString() : app::tr(qt_text(reason->second));
            }
            results->setItem(index, static_cast<int>(column) + 1, percent_item(row.percent.at(key), why));
        }
        if (sequence != nullptr) {
            sequence_rows_[index] = sequence->name;
            name_item->setToolTip(app::tr("Encodes: {anchor} in the anchor, {test} in the test",
                                          {{"anchor", static_cast<long long>(sequence->anchor_encodes)},
                                           {"test", static_cast<long long>(sequence->test_encodes)}}));
            return;
        }
        // A summary row: bold, on the tint Metric Graphs gives its means
        for (int column = 0; column < results->columnCount(); ++column) {
            QTableWidgetItem* item = results->item(index, column);
            QFont font = item->font();
            font.setBold(true);
            item->setFont(font);
            item->setBackground(theme::color("mean_tint"));
        }
    };
    add_row(app::tr("mean"), report.all.mean, nullptr);
    add_row(app::tr("median"), report.all.median, nullptr);
    for (const BdRateSummary& group : report.groups) {
        const QString name = group.group.empty()
                                 ? app::tr("(no {column})", {{"column", qt_text(*report.group_by)}})
                                 : qt_text(group.group);
        add_row(app::tr("{group} mean", {{"group", name}}), group.mean, nullptr);
        add_row(app::tr("{group} median", {{"group", name}}), group.median, nullptr);
    }
    for (const BdRateSequence& sequence : report.sequences) {
        add_row(qt_text(sequence.name), sequence.row, &sequence);
    }
    for (const auto& [row, name] : sequence_rows_) {
        if (chosen_sequence_ && name == *chosen_sequence_) {
            results->setCurrentCell(row, 0);
        }
    }
    fit_height(results, visible_result_rows);
    filling_ = false;
}

void BdRatePanel::on_results_row(int row) {
    if (filling_) {
        return;
    }
    const auto sequence = sequence_rows_.find(row);
    if (sequence == sequence_rows_.end()) {
        return;  // a summary row: the chart keeps its sequence
    }
    chosen_sequence_ = sequence->second;
    refresh_chart();
}

// The chosen sequence's curves for the chosen metric.
void BdRatePanel::refresh_chart() {
    if (!report_ || !chosen_sequence_) {
        sequence_label->clear();
        chart->set_plot(std::nullopt, app::tr("The rate curves of the sequence chosen in the table are drawn here."));
        return;
    }
    const std::string key = metric_combo->currentData().toString().toStdString();
    const MetricDefinition& metric = *find_metric(key);
    const bool lower_is_better = metric.direction == MetricDirection::lower_is_better;
    const auto curve_of = [&](const Side& side) {
        RateCurveChart::Curve curve{side_name(side), {}, {}};
        std::size_t encodes = 0;
        for (const RateTable::Encode& encode : merged(side).encodes) {
            if (encode.sequence != *chosen_sequence_) {
                continue;
            }
            ++encodes;
            // A dot where the encode has a score, and a rate and score that
            // can be drawn (a CSV file may say 0 or inf)
            const auto score = encode.scores.find(key);
            if (score != encode.scores.end() && score->second && std::isfinite(*score->second) &&
                std::isfinite(encode.kbps) && encode.kbps > 0) {
                curve.points.push_back({encode.kbps, *score->second});
            }
        }
        // The curve the BD-rate integrates, only where it took every encode
        if (curve.points.size() == encodes) {
            curve.line = pchip_curve(curve.points, lower_is_better, curve_samples);
        }
        return curve;
    };
    const auto sequence = std::ranges::find(report_->sequences, *chosen_sequence_, &BdRateSequence::name);
    const std::optional<double> percent = sequence->row.percent.at(key);
    QString label = qt_text(metric.short_label) + QStringLiteral(": ") + percent_cell_text(percent);
    if (const auto why = sequence->why.find(key); why != sequence->why.end()) {
        label += QStringLiteral(" (") + app::tr(qt_text(why->second)) + QStringLiteral(")");
    }
    sequence_label->setText(qt_text(*chosen_sequence_));
    chart->set_plot(RateCurveChart::Plot{curve_of(anchor), curve_of(test), qt_text(metric.short_label),
                                         lower_is_better, label});
}

// ------------------------------------------------------------- the buttons

void BdRatePanel::on_export() {
    if (!report_) {
        return;
    }
    const QString path =
        dialogs.save_file(app::tr("Export BD-rates"), QStringLiteral("bd_rate.csv"), app::tr("CSV file (*.csv)"));
    if (path.isEmpty()) {
        return;
    }
    const std::string text = bd_rate_csv(*report_);
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) ||
        file.write(text.data(), static_cast<qint64>(text.size())) != static_cast<qint64>(text.size()) ||
        !file.commit()) {
        dialogs.alert(app::tr("Export failed"), app::tr("Could not write {path}", {{"path", path}}));
    }
}

void BdRatePanel::on_copy() {
    if (report_) {
        copy_text(qt_text(bd_rate_table(*report_, side_name(anchor).toStdString(), side_name(test).toStdString())));
    }
}

void BdRatePanel::on_show_warnings() { dialogs.inform(app::tr("BD-rate warnings"), warnings_.join(QLatin1Char('\n'))); }

}  // namespace vml::app
