// BD-rates of a test encoder's encodes against an anchor's (vml/bd_rate.h),
// for each sequence the two have and each metric; their mean and median over
// the sequences (and over each group of them). The command line's bdrate and
// the window's BD-rate tab.
#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/results.h"

namespace vml {

// One encoder's encodes: read from a CSV file with a row per encode, or
// gathered from scored videos.
struct RateTable {
    struct Encode {
        std::string sequence;
        double kbps = 0;
        std::map<std::string, std::optional<double>> scores;  // by metric key; none for an empty cell
        std::map<std::string, std::string> text;              // the other columns' cells, by column name
        std::string reference;  // the reference's path, for a scored video; a CSV file's rows have none
        // The test video's file name where its bitrate is the whole file's,
        // soundtrack included (VideoInfo::bit_rate_whole_file; a results
        // document's "bit_rate_of": "file"), which makes its BD-rate look
        // worse than it is: said as a warning. Empty otherwise.
        std::string whole_file_video{};
        bool operator==(const Encode&) const = default;
    };
    std::vector<std::string> metric_keys;  // its metric columns, in the registry's order
    std::vector<std::string> columns;      // every column's name, in lower case
    std::vector<Encode> encodes;
};

// The table in `text`, a CSV file's contents (RFC 4180, a UTF-8 BOM
// skipped): a header row of column names, in any case, with sequence and
// kbps among them, then a row per encode. Throws std::runtime_error saying
// what is wrong, `name` naming the file.
RateTable read_rate_table(std::string_view text, const std::string& name);

// The table in a file of either form, by its text: JSON (its first
// character, after a UTF-8 BOM and white space, a brace) or else CSV
// (read_rate_table). A JSON file is
// - a results document, as Export results and the command line's --json
//   write it, or the command line's --batch {"batch": [...]}: an encode for
//   each video with a bitrate of the whole file, the sequence its
//   reference's file name without its extension, each metric's score;
// - or a run file (.metrics.json): the encode of its one video, as
//   rate_encode makes it.
// A video of a range, failed or cancelled, or with no bitrate is left out.
// Throws std::runtime_error, `name` naming the file, when none is left.
// `stat` picks a JSON file's score: "score" (the one the app shows), or
// another of its metric summary's statistics ("worst_1_percent",
// metric_summary's names), null where a metric has none of it; a CSV file
// has only its scores, and `stat` other than "score" throws for one.
RateTable read_rate_file(std::string_view text, const std::string& name, const std::string& stat = "score");

// `table` as a CSV file read_rate_table reads back as it is: its columns in
// their order, numbers in full, an empty cell for a score not had. CR LF
// line ends, as the app's other CSV files, but no byte order mark: it is
// for programs (compare --scores-csv, read by bdrate and scripts).
std::string rate_table_csv(const RateTable& table);

// A scored comparison as an encode of its test video: the sequence is the
// reference's file name without its extension, the rate the test video's
// bitrate (VideoInfo::bit_rate, in kb/s), and each metric's score the one the
// app shows (a frame metric's aggregate, CVVDP's score). None where the
// test video's bitrate is not known.
std::optional<RateTable::Encode> rate_encode(const ComparisonResult& result);

// The warning for each encode of `table` whose bitrate is the whole file's
// (Encode::whole_file_video), as the command line gives it.
std::vector<std::string> whole_file_rate_warnings(const RateTable& table);

// The bits per second that go with `result`'s scores: its test video's bitrate,
// which is of the whole file, or 0 for scores of only a part of it (a range) or
// of no file of its own (a resolution test).
long long rate_of_scores(const ComparisonResult& result);

// The text columns a group can be taken from: the table's columns other than
// sequence, kbps and the metrics', in its order.
std::vector<std::string> group_columns(const RateTable& table);

// A row of results: each metric's BD-rate in % (none where there is none).
struct BdRateRow {
    std::map<std::string, std::optional<double>> percent;
};

struct BdRateSequence {
    std::string name;
    std::string group;  // its group column's cell in the anchor's encodes
    std::size_t anchor_encodes = 0;
    std::size_t test_encodes = 0;
    BdRateRow row;
    std::map<std::string, std::string> why;  // each metric with no BD-rate: why
};

// The mean and median of some sequences: all of them, or a group's.
struct BdRateSummary {
    std::string group;  // empty for all the sequences
    std::size_t sequences = 0;
    BdRateRow mean;
    BdRateRow median;
};

struct BdRateReport {
    std::vector<std::string> metrics;  // the columns, in the registry's order
    std::optional<std::string> group_by;
    std::vector<BdRateSequence> sequences;  // in name order
    BdRateSummary all;
    std::vector<BdRateSummary> groups;  // in name order
    std::vector<std::string> warnings;
};

// What a report shows.
struct BdRateOptions {
    // The metrics (keys, in the registry's order); none: every metric either
    // table has a column for.
    std::optional<std::vector<std::string>> metrics;
    // A text column of the anchor's (group_columns) to group the sequences by.
    std::optional<std::string> group_by;
    // Whether a metric neither table has scores of is warned of, for each
    // side. The BD-rate tab asks for every metric and shows those as n/a,
    // which says it: a warning per side for each would bury the others.
    bool warn_of_metrics_neither_has = true;
};

// The BD-rates of `test` against `anchor` for the sequences both have; with
// none in both, a report without sequences, whose warnings name those left
// out. Throws std::runtime_error for a group column the anchor does not
// have, the message naming the command line's option.
BdRateReport bd_rate_report(const RateTable& anchor, const RateTable& test, const BdRateOptions& options);

// A BD-rate as the table shows it: "+1.2%", "-0.0%" (the sign as Python's
// "{:+.1f}" gives it), or "n/a".
std::string bd_rate_percent_text(const std::optional<double>& value);

// The report as the command line prints it: a line naming the encoders, then
// a markdown table with a column per metric, and rows for the mean, the
// median, each group's, and each sequence.
std::string bd_rate_table(const BdRateReport& report, const std::string& anchor_name, const std::string& test_name);

// The report as a CSV file, for a spreadsheet: the table's rows, with their
// name, a sequence's group and encode counts, and each BD-rate in % in full
// (empty for none), by metric key. A UTF-8 byte order mark and CR LF line
// ends, as the app's other CSV files.
std::string bd_rate_csv(const BdRateReport& report);

}  // namespace vml
