#include "vml/bd_rate_report.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

#include "vml/bd_rate.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/results_summary.h"
#include "vml/translatable.h"

namespace vml {

namespace {

// Fewer encodes than this make a rough BD-rate: the curves' ends are guessed
// from three points or fewer.
constexpr std::size_t enough_encodes = 4;

// ------------------------------------------------------------- the CSV files

struct CsvRow {
    std::size_t line = 0;  // where it starts in the file, from 1
    std::vector<std::string> fields;
};

// The rows of a CSV file's text: fields between commas; a field in double
// quotes may hold commas, line ends and "" for a quote. Blank lines, and
// rows of empty fields (",,,", as a spreadsheet leaves after its data), are
// skipped. Throws std::runtime_error for a quote not closed.
std::vector<CsvRow> csv_rows(std::string_view text) {
    if (text.starts_with("\xEF\xBB\xBF")) {
        text.remove_prefix(3);
    }
    std::vector<CsvRow> rows;
    CsvRow row{1, {}};
    std::string field;
    std::size_t line = 1;
    bool quoted = false;
    const auto end_row = [&] {
        row.fields.push_back(std::move(field));
        field.clear();
        const bool blank = std::ranges::all_of(row.fields, [](const std::string& one) {
            return python_strip(one).empty();
        });
        if (!blank) {
            rows.push_back(std::move(row));
        }
        row = CsvRow{line, {}};
    };
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char c = text[index];
        if (quoted) {
            if (c == '"' && index + 1 < text.size() && text[index + 1] == '"') {
                field += '"';
                ++index;
            } else if (c == '"') {
                quoted = false;
            } else {
                line += c == '\n' ? 1 : 0;
                field += c;
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            row.fields.push_back(std::move(field));
            field.clear();
        } else if (c == '\r' || c == '\n') {
            if (c == '\r' && index + 1 < text.size() && text[index + 1] == '\n') {
                ++index;
            }
            ++line;
            end_row();
        } else {
            field += c;
        }
    }
    if (quoted) {
        throw std::runtime_error("line " + std::to_string(row.line) + ": a quote is not closed");
    }
    if (!field.empty() || !row.fields.empty()) {
        end_row();
    }
    return rows;
}

// A cell's number, or why it is not one.
double number_in(const std::string& cell, const std::string& column, std::size_t line) {
    const std::optional<double> value = python_float(cell);
    if (!value) {
        throw std::runtime_error("line " + std::to_string(line) + ": " + column + " is not a number: " +
                                 python_string_repr(cell));
    }
    return *value;
}

// The column names of the header row, in lower case (a column without a
// name, as a spreadsheet's empty columns, is "" and left out of the
// table); why not, for a column named twice or the sequence or kbps column
// missing.
std::vector<std::string> header_of(const CsvRow& header) {
    std::vector<std::string> columns;
    for (const std::string& field : header.fields) {
        std::string name = ascii_lower(python_strip(field));
        if (!name.empty() && std::find(columns.begin(), columns.end(), name) != columns.end()) {
            throw std::runtime_error("the column " + python_string_repr(name) + " is there twice");
        }
        columns.push_back(std::move(name));
    }
    for (const char* needed : {"sequence", "kbps"}) {
        if (std::find(columns.begin(), columns.end(), needed) == columns.end()) {
            throw std::runtime_error(std::string("no ") + needed + " column");
        }
    }
    return columns;
}

// One encode from its row.
RateTable::Encode encode_of(const CsvRow& row, const std::vector<std::string>& columns) {
    if (row.fields.size() != columns.size()) {
        throw std::runtime_error("line " + std::to_string(row.line) + " has " + std::to_string(row.fields.size()) +
                                 " fields, the header " + std::to_string(columns.size()));
    }
    RateTable::Encode encode;
    for (std::size_t index = 0; index < columns.size(); ++index) {
        const std::string& column = columns[index];
        const std::string cell(python_strip(row.fields[index]));
        if (column.empty()) {
            continue;  // a column without a name
        }
        if (column == "sequence") {
            encode.sequence = cell;
        } else if (column == "kbps") {
            encode.kbps = number_in(cell, column, row.line);
        } else if (find_metric(column) != nullptr) {
            encode.scores[column] = cell.empty() ? std::nullopt : std::optional(number_in(cell, column, row.line));
        } else {
            encode.text[column] = cell;
        }
    }
    if (encode.sequence.empty()) {
        throw std::runtime_error("line " + std::to_string(row.line) + ": no sequence");
    }
    return encode;
}

// ------------------------------------------------------------- the BD-rates

// The rate points of `encodes` for a metric; none when one is not scored.
std::optional<std::vector<RatePoint>> points_of(const std::vector<const RateTable::Encode*>& encodes,
                                                const std::string& metric) {
    std::vector<RatePoint> points;
    for (const RateTable::Encode* encode : encodes) {
        const auto score = encode->scores.find(metric);
        if (score == encode->scores.end() || !score->second) {
            return std::nullopt;
        }
        points.push_back({encode->kbps, *score->second});
    }
    return points;
}

// Whether any of a table's encodes has a score of `metric`.
bool has_scores(const RateTable& table, const std::string& metric) {
    return std::ranges::any_of(table.encodes, [&metric](const RateTable::Encode& encode) {
        const auto score = encode.scores.find(metric);
        return score != encode.scores.end() && score->second.has_value();
    });
}

// Each sequence's encodes, by name.
std::map<std::string, std::vector<const RateTable::Encode*>> by_sequence(const RateTable& table) {
    std::map<std::string, std::vector<const RateTable::Encode*>> sequences;
    for (const RateTable::Encode& encode : table.encodes) {
        sequences[encode.sequence].push_back(&encode);
    }
    return sequences;
}

// The columns: those asked for, or every metric either table has a column
// for, in the registry's order.
std::vector<std::string> shown_metrics(const RateTable& anchor, const RateTable& test, const BdRateOptions& options) {
    if (options.metrics) {
        return *options.metrics;
    }
    std::vector<std::string> keys;
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        for (const RateTable* table : {&anchor, &test}) {
            if (std::ranges::count(table->metric_keys, key) > 0) {
                keys.push_back(key);
                break;
            }
        }
    }
    return keys;
}

// A sequence's BD-rates, with a warning for each metric that has none.
BdRateSequence sequence_rates(const std::string& name, const std::vector<const RateTable::Encode*>& anchor,
                              const std::vector<const RateTable::Encode*>& test, const BdRateReport& report,
                              const std::set<std::string>& unscored, std::vector<std::string>& warnings) {
    BdRateSequence sequence{name, {}, anchor.size(), test.size(), {}, {}};
    if (anchor.size() < enough_encodes || test.size() < enough_encodes) {
        warnings.push_back(name + ": " + std::to_string(anchor.size()) + " encodes in the anchor and " +
                           std::to_string(test.size()) + " in the test; fewer than " +
                           std::to_string(enough_encodes) + " make a rough BD-rate");
    }
    for (const std::string& metric : report.metrics) {
        const std::optional<std::vector<RatePoint>> anchor_points = points_of(anchor, metric);
        const std::optional<std::vector<RatePoint>> test_points = points_of(test, metric);
        std::string why;
        if (!anchor_points || !test_points) {
            why = N_("not scored at every encode");
        } else {
            const bool lower_is_better = find_metric(metric)->direction == MetricDirection::lower_is_better;
            const BdRateResult result = bd_rate(*anchor_points, *test_points, lower_is_better);
            sequence.row.percent[metric] = result.percent;
            why = bd_rate_gap_text(result.gap);
        }
        if (!why.empty()) {
            sequence.row.percent[metric] = std::nullopt;
            sequence.why[metric] = why;
            if (!unscored.contains(metric)) {  // said once for the whole table
                warnings.push_back(name + ": " + metric + ": " + why + ", n/a");
            }
        }
    }
    return sequence;
}

// The mean and median of `sequences`, each metric's over those that have a
// BD-rate for it.
BdRateSummary summary_of(const std::string& group, const std::vector<const BdRateSequence*>& sequences,
                         const BdRateReport& report) {
    BdRateSummary summary{group, sequences.size(), {}, {}};
    for (const std::string& metric : report.metrics) {
        std::vector<std::optional<double>> values;
        for (const BdRateSequence* sequence : sequences) {
            values.push_back(sequence->row.percent.at(metric));
        }
        summary.mean.percent[metric] = mean_of(values);
        summary.median.percent[metric] = median_of(values);
    }
    return summary;
}

// Grouping: each sequence's group from its first encode in the anchor's
// table, a warning where the test's says another; then each group's summary.
void add_groups(BdRateReport& report, const std::map<std::string, std::vector<const RateTable::Encode*>>& anchor,
                const std::map<std::string, std::vector<const RateTable::Encode*>>& test) {
    const std::string& column = *report.group_by;
    std::map<std::string, std::vector<const BdRateSequence*>> groups;
    for (BdRateSequence& sequence : report.sequences) {
        // An encode without the column (a scored video beside a CSV file's
        // rows, in the window) is in no group: an empty cell.
        const auto& ours = anchor.at(sequence.name).front()->text;
        const auto cell = ours.find(column);
        sequence.group = cell == ours.end() ? std::string() : cell->second;
        const auto& their_text = test.at(sequence.name).front()->text;
        const auto theirs = their_text.find(column);
        if (theirs != their_text.end() && theirs->second != sequence.group) {
            report.warnings.push_back(sequence.name + ": " + column + " is " + python_string_repr(sequence.group) +
                                      " in the anchor and " + python_string_repr(theirs->second) +
                                      " in the test; the anchor's is used");
        }
        groups[sequence.group].push_back(&sequence);
    }
    for (const auto& [name, members] : groups) {
        report.groups.push_back(summary_of(name, members, report));
    }
}

// A row of the markdown table.
std::string table_row(const std::string& name, const BdRateRow& row, const BdRateReport& report) {
    std::string line = "| " + name + " |";
    for (const std::string& metric : report.metrics) {
        line += " " + bd_rate_percent_text(row.percent.at(metric)) + " |";
    }
    return line + "\n";
}

// A row of the CSV file: the name, then the group and counts (where `sequence`
// gives them), then the BD-rates.
std::string csv_row(const std::string& name, const BdRateRow& row, const BdRateSequence* sequence,
                    const BdRateReport& report) {
    std::string line = csv_cell(name);
    if (report.group_by) {
        line += "," + (sequence != nullptr ? csv_cell(sequence->group) : std::string());
    }
    line += "," + (sequence != nullptr ? std::to_string(sequence->anchor_encodes) : std::string());
    line += "," + (sequence != nullptr ? std::to_string(sequence->test_encodes) : std::string());
    const auto number = [](const std::optional<double>& value) { return value ? python_repr(*value) : std::string(); };
    for (const std::string& metric : report.metrics) {
        line += "," + number(row.percent.at(metric));
    }
    return line + "\r\n";
}

}  // namespace

RateTable read_rate_table(std::string_view text, const std::string& name) {
    try {
        const std::vector<CsvRow> rows = csv_rows(text);
        if (rows.empty()) {
            throw std::runtime_error("it is empty");
        }
        RateTable table;
        const std::vector<std::string> header = header_of(rows.front());
        table.columns = header;
        std::erase(table.columns, std::string());  // the unnamed ones, which encode_of skips too
        for (const MetricDefinition& metric : metrics()) {
            if (std::ranges::count(table.columns, metric.key) > 0) {
                table.metric_keys.emplace_back(metric.key);
            }
        }
        for (std::size_t index = 1; index < rows.size(); ++index) {
            table.encodes.push_back(encode_of(rows[index], header));
        }
        return table;
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(name + ": " + error.what());
    }
}

long long rate_of_scores(const ComparisonResult& result) {
    if (result.resample_target) {
        return 0;
    }
    for (const std::string& key : result.metric_results.keys()) {
        const FrameMetricResult* frames = result.metric_results.frame(key);
        const MetricProvenance& provenance =
            frames != nullptr ? frames->provenance : result.metric_results.sequence(key)->provenance;
        if (provenance.parameters.contains("range_frames")) {
            return 0;
        }
    }
    return result.distorted_info.bit_rate;
}

std::optional<RateTable::Encode> rate_encode(const ComparisonResult& result) {
    const long long bits_per_second = rate_of_scores(result);
    if (bits_per_second <= 0) {
        return std::nullopt;
    }
    RateTable::Encode encode;
    encode.sequence = platform::path_text(result.source.stem());
    encode.reference = platform::path_text(result.source);
    encode.kbps = static_cast<double>(bits_per_second) / 1000;
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        if (const FrameMetricResult* frames = result.metric_results.frame(key)) {
            encode.scores[key] = frames->aggregate();
        } else if (const SequenceMetricResult* whole = result.metric_results.sequence(key)) {
            encode.scores[key] = whole->score;
        }
    }
    return encode;
}

std::vector<std::string> group_columns(const RateTable& table) {
    std::vector<std::string> columns;
    for (const std::string& column : table.columns) {
        if (column != "sequence" && column != "kbps" && find_metric(column) == nullptr) {
            columns.push_back(column);
        }
    }
    return columns;
}

namespace {

// A sequence is known by its reference's file name: encodes of two files of
// one name (the same clip in two folders, or two clips) count as one sequence.
void warn_of_references_sharing_a_name(const RateTable& anchor, const RateTable& test,
                                       std::vector<std::string>& warnings) {
    std::map<std::string, std::set<std::string>> references;
    for (const RateTable* table : {&anchor, &test}) {
        for (const RateTable::Encode& encode : table->encodes) {
            if (!encode.reference.empty()) {
                references[encode.sequence].insert(encode.reference);
            }
        }
    }
    for (const auto& [sequence, paths] : references) {
        if (paths.size() > 1) {
            std::string listed;
            for (const std::string& path : paths) {
                listed += (listed.empty() ? "" : ", ") + path;
            }
            warnings.push_back(sequence + ": its encodes have different references, " + listed);
        }
    }
}

}  // namespace

BdRateReport bd_rate_report(const RateTable& anchor, const RateTable& test, const BdRateOptions& options) {
    BdRateReport report;
    report.metrics = shown_metrics(anchor, test, options);
    report.group_by = options.group_by ? std::optional(ascii_lower(python_strip(*options.group_by))) : std::nullopt;
    if (report.group_by && std::ranges::count(group_columns(anchor), *report.group_by) == 0) {
        throw std::runtime_error("argument --group-by: the anchor's file has no column " +
                                 python_string_repr(*report.group_by) + " other than sequence, kbps and the scores");
    }
    // A metric a table has no scores of: said once, not for every sequence
    std::set<std::string> unscored;
    for (const std::string& metric : report.metrics) {
        for (const auto& [table, whose] : {std::pair{&anchor, "anchor"}, std::pair{&test, "test"}}) {
            if (!has_scores(*table, metric)) {
                unscored.insert(metric);
                report.warnings.push_back(metric + ": the " + whose + " has no scores of it, n/a");
            }
        }
    }
    warn_of_references_sharing_a_name(anchor, test, report.warnings);
    const auto anchor_sequences = by_sequence(anchor);
    const auto test_sequences = by_sequence(test);
    for (const auto& [name, encodes] : anchor_sequences) {
        const auto theirs = test_sequences.find(name);
        if (theirs == test_sequences.end()) {
            report.warnings.push_back(name + ": only in the anchor, left out");
            continue;
        }
        report.sequences.push_back(
            sequence_rates(name, encodes, theirs->second, report, unscored, report.warnings));
    }
    for (const auto& [name, encodes] : test_sequences) {
        if (!anchor_sequences.contains(name)) {
            report.warnings.push_back(name + ": only in the test, left out");
        }
    }
    if (report.sequences.empty()) {
        return report;  // nothing to sum up; the warnings name every sequence
    }
    std::vector<const BdRateSequence*> all;
    for (const BdRateSequence& sequence : report.sequences) {
        all.push_back(&sequence);
    }
    report.all = summary_of({}, all, report);
    if (report.group_by) {
        add_groups(report, anchor_sequences, test_sequences);
    }
    return report;
}

std::string bd_rate_percent_text(const std::optional<double>& value) {
    if (!value) {
        return "n/a";
    }
    return (std::signbit(*value) ? "" : "+") + format_fixed(*value, 1) + "%";
}

std::string bd_rate_table(const BdRateReport& report, const std::string& anchor_name, const std::string& test_name) {
    std::string text = "BD-rate of " + test_name + " against " + anchor_name + ", " +
                       std::to_string(report.sequences.size()) +
                       (report.sequences.size() == 1 ? " sequence" : " sequences") +
                       " (- is less bitrate for the same score)\n\n";
    text += "| |";
    std::string rule = "|---|";
    for (const std::string& metric : report.metrics) {
        text += " " + std::string(find_metric(metric)->short_label) + " |";
        rule += "---|";
    }
    text += "\n" + rule + "\n";
    text += table_row("mean", report.all.mean, report);
    text += table_row("median", report.all.median, report);
    for (const BdRateSummary& group : report.groups) {
        text += table_row(group.group + " mean", group.mean, report);
        text += table_row(group.group + " median", group.median, report);
    }
    for (const BdRateSequence& sequence : report.sequences) {
        text += table_row(sequence.name, sequence.row, report);
    }
    return text;
}

std::string bd_rate_csv(const BdRateReport& report) {
    std::string text = "\xEF\xBB\xBFrow";
    if (report.group_by) {
        text += "," + csv_cell(*report.group_by);
    }
    text += ",anchor_encodes,test_encodes";
    for (const std::string& metric : report.metrics) {
        text += "," + metric;
    }
    text += "\r\n";
    text += csv_row("mean", report.all.mean, nullptr, report);
    text += csv_row("median", report.all.median, nullptr, report);
    for (const BdRateSummary& group : report.groups) {
        text += csv_row(group.group + " mean", group.mean, nullptr, report);
        text += csv_row(group.group + " median", group.median, nullptr, report);
    }
    for (const BdRateSequence& sequence : report.sequences) {
        text += csv_row(sequence.name, sequence.row, &sequence, report);
    }
    return text;
}

}  // namespace vml
