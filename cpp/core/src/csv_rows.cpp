#include "vml/csv_rows.h"

#include <algorithm>
#include <stdexcept>

#include "vml/python_text.h"

namespace vml {

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

}  // namespace vml
