// The rows of a CSV file (RFC 4180), as the bdrate command's tables and the
// compare command's --batch list are read.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace vml {

struct CsvRow {
    std::size_t line = 0;  // where it starts in the file, from 1
    std::vector<std::string> fields;
};

// The rows of a CSV file's text (a UTF-8 BOM skipped): fields between
// commas; a field in double quotes may hold commas, line ends and "" for a
// quote. Blank lines, and rows of empty fields (",,,", as a spreadsheet
// leaves after its data), are skipped. Throws std::runtime_error for a
// quote not closed.
std::vector<CsvRow> csv_rows(std::string_view text);

}  // namespace vml
