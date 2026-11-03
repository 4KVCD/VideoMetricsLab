// The bdrate command: the BD-rates (vml/bd_rate_report.h) of a test encoder's
// encodes against an anchor's, read from two CSV files with a row per
// encode, for each sequence and metric; their mean and median over the
// sequences (and over each group of them, with --group-by); and W, a
// weighted sum of the metrics' BD-rates. As a markdown table or JSON.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "arguments.h"
#include "vml/bd_rate_report.h"

namespace vml::cli {

// Runs the bdrate command; returns the exit code: 0 when the files were
// read (cells with no BD-rate are n/a, with a warning), 1 when the JSON file
// could not be written, 2 for a wrong command or files that cannot be read.
int bdrate(const BdrateArguments& arguments);

// The command's options as the core's report takes them.
BdRateOptions bd_rate_options(const BdrateArguments& arguments);

// The report as the command writes it in JSON, its numbers in full.
nlohmann::ordered_json bd_rate_json(const BdRateReport& report, const std::string& anchor_path,
                                    const std::string& test_path);

}  // namespace vml::cli
