#include "bdrate.h"

#include <exception>
#include <filesystem>
#include <stdexcept>

#include "codes.h"
#include "common.h"
#include "compare.h"
#include "vml/build_info.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

// What the JSON is: its own format, not compare's results (results_format).
constexpr const char* bdrate_format = "bdrate-1";

// A number in the JSON, in full, or null for none.
Json number_or_null(const std::optional<double>& value) { return value ? Json(*value) : Json(nullptr); }

Json row_json(const BdRateRow& row, const BdRateReport& report) {
    Json percent = Json::object();
    for (const std::string& metric : report.metrics) {
        percent[metric] = number_or_null(row.percent.at(metric));
    }
    return Json{{"bd_rate", percent}};
}

// The file a table comes from, read; nothing, with the reason in `why`, when
// it cannot be.
std::optional<RateTable> read_table_file(const std::string& text_path, std::string& why) {
    const std::filesystem::path path = python_path(text_path);
    try {
        return read_rate_file(read_whole_file(path), platform::path_text(path));
    } catch (const std::exception& error) {
        why += (why.empty() ? "" : "\n") + std::string(error.what());
        return std::nullopt;
    }
}

// The files could not give BD-rates (exit_input): `why` printed, and with
// --json the document of the error, as the results would be written.
int refuse_input(const BdrateArguments& arguments, const std::string& why, const char* code) {
    print_error(why);
    if (arguments.json_path) {
        const Json document{{"app", "VideoMetricsLab"}, {"version", VML_VERSION}, {"commit", build_commit()},
                            {"format", bdrate_format},  {"anchor", arguments.anchor}, {"test", arguments.test},
                            {"error", why},             {"error_code", code}};
        if (*arguments.json_path == "-") {
            print_results(python_json_dumps(document) + "\n");
        } else if (const std::optional<std::string> failure =
                       write_json_file(python_path(*arguments.json_path), document)) {
            print_error("The JSON file could not be written: " + *failure);
        }
    }
    return exit_input;
}

}  // namespace

BdRateOptions bd_rate_options(const BdrateArguments& arguments) {
    return {arguments.metrics, arguments.group_by};
}

Json bd_rate_json(const BdRateReport& report, const std::string& anchor_path, const std::string& test_path) {
    Json groups = Json::array();
    for (const BdRateSummary& group : report.groups) {
        groups.push_back(Json{{"group", group.group},
                              {"sequences", group.sequences},
                              {"mean", row_json(group.mean, report)},
                              {"median", row_json(group.median, report)}});
    }
    Json sequences = Json::array();
    for (const BdRateSequence& sequence : report.sequences) {
        Json entry = row_json(sequence.row, report);
        Json described{{"sequence", sequence.name}};
        if (report.group_by) {
            described["group"] = sequence.group;
        }
        described["anchor_encodes"] = sequence.anchor_encodes;
        described["test_encodes"] = sequence.test_encodes;
        described.update(entry);
        described["why"] = sequence.why;
        sequences.push_back(described);
    }
    return Json{{"app", "VideoMetricsLab"},
                {"version", VML_VERSION},
                {"commit", build_commit()},
                {"format", bdrate_format},
                {"method", "pchip"},
                {"anchor", anchor_path},
                {"test", test_path},
                {"metrics", report.metrics},
                {"group_by", report.group_by ? Json(*report.group_by) : Json(nullptr)},
                {"mean", row_json(report.all.mean, report)},
                {"median", row_json(report.all.median, report)},
                {"groups", groups},
                {"sequences", sequences},
                {"warnings", report.warnings}};
}

int bdrate(const BdrateArguments& arguments) {
    std::string unread;
    const std::optional<RateTable> anchor = read_table_file(arguments.anchor, unread);
    const std::optional<RateTable> test = read_table_file(arguments.test, unread);
    if (!anchor || !test) {
        return refuse_input(arguments, unread, codes::rates_unreadable);
    }
    BdRateReport report;
    try {
        report = bd_rate_report(*anchor, *test, bd_rate_options(arguments));
    } catch (const std::runtime_error& error) {
        print_error(error.what());  // --group-by naming no column of the anchor's
        return exit_usage;
    }
    // As the BD-rate tab warns of them
    for (const RateTable* table : {&*anchor, &*test}) {
        for (std::string& warning : whole_file_rate_warnings(*table)) {
            report.warnings.push_back(std::move(warning));
        }
    }
    if (!arguments.quiet) {
        for (const std::string& warning : report.warnings) {
            print_error("warning: " + warning);
        }
    }
    if (report.metrics.empty()) {
        return refuse_input(arguments, "neither the anchor nor the test has a column of scores of a metric",
                            codes::no_common_metric);
    }
    if (report.sequences.empty()) {
        return refuse_input(arguments, "the anchor and the test have no sequence in common",
                            codes::no_common_sequence);
    }
    const Json document = bd_rate_json(report, arguments.anchor, arguments.test);
    if (arguments.json_path && *arguments.json_path == "-") {
        return print_results(python_json_dumps(document) + "\n") ? exit_ok : exit_failed;
    }
    const auto name_of = [](const std::string& path) { return platform::path_text(python_path(path).filename()); };
    if (!print_results(bd_rate_table(report, name_of(arguments.anchor), name_of(arguments.test)))) {
        return exit_failed;
    }
    if (arguments.json_path) {
        if (const std::optional<std::string> failure = write_json_file(python_path(*arguments.json_path), document)) {
            print_error("The JSON file could not be written: " + *failure);
            return exit_failed;
        }
    }
    return exit_ok;
}

}  // namespace vml::cli
