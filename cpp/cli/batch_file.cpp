#include "batch_file.h"

#include <algorithm>
#include <stdexcept>

#include "compare.h"
#include "vml/csv_rows.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml::cli {

namespace {

// The header's column names in lower case; why not, for a column named
// twice or the reference or test column missing.
std::vector<std::string> columns_of(const CsvRow& header) {
    std::vector<std::string> columns;
    for (const std::string& field : header.fields) {
        std::string name = ascii_lower(python_strip(field));
        if (!name.empty() && std::ranges::count(columns, name) > 0) {
            throw std::runtime_error("the column " + python_string_repr(name) + " is there twice");
        }
        columns.push_back(std::move(name));
    }
    for (const char* needed : {"reference", "test"}) {
        if (std::ranges::count(columns, std::string(needed)) == 0) {
            throw std::runtime_error(std::string("no ") + needed + " column");
        }
    }
    return columns;
}

// A path cell as a path: relative to the list's folder where it is relative.
std::filesystem::path path_in(const std::string& cell, const std::filesystem::path& folder) {
    const std::filesystem::path path = python_path(cell);
    return path.is_absolute() ? path : folder / path;
}

BatchRow row_of(const CsvRow& row, const std::vector<std::string>& columns, const std::filesystem::path& folder) {
    if (row.fields.size() != columns.size()) {
        throw std::runtime_error("line " + std::to_string(row.line) + " has " + std::to_string(row.fields.size()) +
                                 " fields, the header " + std::to_string(columns.size()));
    }
    BatchRow batch_row;
    batch_row.line = row.line;
    for (std::size_t index = 0; index < columns.size(); ++index) {
        const std::string& column = columns[index];
        std::string cell(python_strip(row.fields[index]));
        if (column.empty()) {
            continue;  // a column without a name, as a spreadsheet's empty ones
        }
        if ((column == "reference" || column == "test") && cell.empty()) {
            throw std::runtime_error("line " + std::to_string(row.line) + ": no " + column);
        }
        if (column == "reference") {
            batch_row.reference = path_in(cell, folder);
        } else if (column == "test") {
            // A pattern too: relative to the list's folder, as a path is.
            batch_row.test = platform::path_text(path_in(cell, folder));
        } else if (column == "sequence") {
            batch_row.sequence = std::move(cell);
        } else {
            batch_row.text[column] = std::move(cell);
        }
    }
    return batch_row;
}

}  // namespace

std::vector<BatchGroup> read_batch_file(std::string_view text, const std::filesystem::path& folder,
                                        const std::string& name) {
    try {
        const std::vector<CsvRow> rows = csv_rows(text);
        if (rows.size() < 2) {
            throw std::runtime_error("it has no rows of videos");
        }
        const std::vector<std::string> columns = columns_of(rows.front());
        std::vector<BatchGroup> groups;
        std::vector<std::string> group_files;  // each group's reference, resolved()
        for (std::size_t index = 1; index < rows.size(); ++index) {
            BatchRow row = row_of(rows[index], columns, folder);
            const std::string file = resolved(row.reference);
            const auto found = std::ranges::find(group_files, file);
            if (found == group_files.end()) {
                group_files.push_back(file);
                groups.push_back({row.reference, {}});
                groups.back().rows.push_back(std::move(row));
            } else {
                groups[static_cast<std::size_t>(found - group_files.begin())].rows.push_back(std::move(row));
            }
        }
        return groups;
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(name + ": " + error.what());
    }
}

}  // namespace vml::cli
