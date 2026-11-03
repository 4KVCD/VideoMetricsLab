#include "bdrate.h"

#include <cerrno>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>

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

// A file's text. Throws std::runtime_error saying why it could not be read,
// as the Python app's OSError.
std::string read_file(const std::filesystem::path& path) {
    errno = 0;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        const int code = errno != 0 ? errno : ENOENT;
        throw std::runtime_error(python_os_error(std::error_code(code, std::generic_category()), path));
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

// A number in the JSON, in full, or null for none.
Json number_or_null(const std::optional<double>& value) { return value ? Json(*value) : Json(nullptr); }

Json row_json(const BdRateRow& row, const BdRateReport& report) {
    Json percent = Json::object();
    for (const std::string& metric : report.metrics) {
        percent[metric] = number_or_null(row.percent.at(metric));
    }
    return Json{{"bd_rate", percent}};
}

// The file a table comes from, read; nothing, with the reason printed, when
// it cannot be.
std::optional<RateTable> read_table_file(const std::string& text_path) {
    const std::filesystem::path path = python_path(text_path);
    try {
        return read_rate_table(read_file(path), platform::path_text(path));
    } catch (const std::exception& error) {
        print_error(error.what());
        return std::nullopt;
    }
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
    const std::optional<RateTable> anchor = read_table_file(arguments.anchor);
    const std::optional<RateTable> test = read_table_file(arguments.test);
    if (!anchor || !test) {
        return exit_usage;
    }
    BdRateReport report;
    try {
        report = bd_rate_report(*anchor, *test, bd_rate_options(arguments));
    } catch (const std::runtime_error& error) {
        print_error(error.what());
        return exit_usage;
    }
    if (!arguments.quiet) {
        for (const std::string& warning : report.warnings) {
            print_error("warning: " + warning);
        }
    }
    if (report.sequences.empty()) {
        print_error("the anchor and the test have no sequence in common");
        return exit_usage;
    }
    const Json document = bd_rate_json(report, arguments.anchor, arguments.test);
    if (arguments.json_path && *arguments.json_path == "-") {
        platform::write_text(platform::Stream::output, python_json_dumps(document) + "\n");
        return exit_ok;
    }
    const auto name_of = [](const std::string& path) { return platform::path_text(python_path(path).filename()); };
    platform::write_text(platform::Stream::output,
                         bd_rate_table(report, name_of(arguments.anchor), name_of(arguments.test)));
    if (arguments.json_path) {
        if (const std::optional<std::string> failure = write_json_file(python_path(*arguments.json_path), document)) {
            print_error("The JSON file could not be written: " + *failure);
            return exit_failed;
        }
    }
    return exit_ok;
}

}  // namespace vml::cli
