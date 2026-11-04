#include "common.h"

#include <algorithm>
#include <cerrno>
#include <fstream>

#include "vml/export_csv.h"
#include "vml/geometry.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"

namespace vml::cli {

namespace {

std::string left_justified(const std::string& text, std::size_t width) {
    const std::size_t length = python_length(text);
    return length >= width ? text : text + std::string(width - length, ' ');
}

std::string right_justified(const std::string& text, std::size_t width) {
    const std::size_t length = python_length(text);
    return length >= width ? text : std::string(width - length, ' ') + text;
}

}  // namespace

std::string name_of(const std::filesystem::path& path) { return platform::path_text(path.filename()); }

void print_error(const std::string& text) { platform::write_text(platform::Stream::error, text + "\n"); }

bool print_results(const std::string& text) {
    if (platform::write_text(platform::Stream::output, text)) {
        return true;
    }
    print_error("The results could not be written to the standard output.");
    return false;
}

std::string text_file(std::string text) {
#ifdef _WIN32
    std::string crlf;
    crlf.reserve(text.size() + text.size() / 16);
    for (const char c : text) {
        if (c == '\n') {
            crlf += '\r';
        }
        crlf += c;
    }
    return crlf;
#else
    return text;
#endif
}

std::optional<std::string> write_json_file(const std::filesystem::path& path, const nlohmann::ordered_json& document) {
    if (path.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            return python_os_error(error, path.parent_path());
        }
    }
    errno = 0;
    std::ofstream file(path, std::ios::binary);
    const int opened = errno;
    file << text_file(python_json_dumps(document) + "\n");
    file.close();
    if (!file) {
        const int code = opened != 0 ? opened : (errno != 0 ? errno : EIO);
        return python_os_error(std::error_code(code, std::generic_category()), path);
    }
    return std::nullopt;
}

std::string with_thousands(long long number) {
    std::string digits = std::to_string(number < 0 ? -number : number);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) {
        digits.insert(static_cast<std::size_t>(at), ",");
    }
    return (number < 0 ? "-" : "") + digits;
}

void add_table(std::vector<std::string>& lines, const std::vector<std::vector<std::string>>& rows,
               bool last_column_free) {
    std::vector<std::size_t> widths(rows.front().size(), 0);
    for (const auto& row : rows) {
        for (std::size_t column = 0; column < row.size(); ++column) {
            widths[column] = std::max(widths[column], python_length(row[column]));
        }
    }
    for (const auto& row : rows) {
        std::string line = "  " + left_justified(row[0], widths[0]);
        const std::size_t justified_end = last_column_free ? row.size() - 1 : row.size();
        for (std::size_t column = 1; column < justified_end; ++column) {
            line += "  " + right_justified(row[column], widths[column]);
        }
        if (last_column_free) {
            line += "  " + row.back();
        }
        lines.emplace_back(python_rstrip(line));
    }
}

bool is_raw_yuv(const std::filesystem::path& path) { return ascii_lower(platform::path_text(path.extension())) == ".yuv"; }

std::optional<std::string> name_taken_note(const std::filesystem::path& written, const std::string& label,
                                           const std::string& suffix) {
    const std::string wanted = output_stem(label) + suffix;
    const std::string name = platform::path_text(written.filename());
    if (name == wanted) {
        return std::nullopt;
    }
    return "note: " + wanted + " is there already, so this is written as " + name;
}

std::string raw_stream_needs_fps(const std::string& which, const VideoInfo& info) {
    return which + " is a raw " + codec_label(info.codec_name) +
           " stream, which records no frame rate: give its frame rate with --fps, such as --fps 24000/1001.";
}

}  // namespace vml::cli
