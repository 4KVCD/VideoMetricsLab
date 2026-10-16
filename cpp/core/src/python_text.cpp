#include "vml/python_text.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "vml/platform/system.h"

namespace vml {

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

std::string_view strip(std::string_view text) {
    const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
    while (!text.empty() && is_space(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && is_space(text.back())) {
        text.remove_suffix(1);
    }
    return text;
}

// `text` without its underscores, or nothing if one is not between two
// digits (Python's rule for numbers written with underscores).
std::optional<std::string> without_underscores(std::string_view text) {
    std::string kept;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '_') {
            if (i == 0 || i + 1 == text.size() || !is_digit(text[i - 1]) || !is_digit(text[i + 1])) {
                return std::nullopt;
            }
            continue;
        }
        kept += text[i];
    }
    return kept;
}

bool equals_ignoring_case(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

// Whether `text` is a decimal number as Python's float() reads one:
// digits with an optional point (or a point and digits), then an optional
// exponent; the sign already taken off.
bool is_decimal_number(std::string_view text) {
    std::size_t i = 0;
    std::size_t digits = 0;
    while (i < text.size() && is_digit(text[i])) {
        ++i;
        ++digits;
    }
    if (i < text.size() && text[i] == '.') {
        ++i;
        while (i < text.size() && is_digit(text[i])) {
            ++i;
            ++digits;
        }
    }
    if (digits == 0) {
        return false;
    }
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
            ++i;
        }
        const std::size_t exponent_start = i;
        while (i < text.size() && is_digit(text[i])) {
            ++i;
        }
        if (i == exponent_start) {
            return false;
        }
    }
    return i == text.size();
}

}  // namespace

std::optional<double> python_float(std::string_view text) {
    const std::optional<std::string> cleaned = without_underscores(strip(text));
    if (!cleaned || cleaned->empty()) {
        return std::nullopt;
    }
    std::string_view number = *cleaned;
    double sign = 1.0;
    if (number.front() == '+' || number.front() == '-') {
        sign = number.front() == '-' ? -1.0 : 1.0;
        number.remove_prefix(1);
    }
    if (equals_ignoring_case(number, "inf") || equals_ignoring_case(number, "infinity")) {
        return sign * std::numeric_limits<double>::infinity();
    }
    if (equals_ignoring_case(number, "nan")) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!is_decimal_number(number)) {
        return std::nullopt;
    }
    // strtod reads the decimal text to the nearest double, as Python does.
    const std::string terminated(number);
    return sign * std::strtod(terminated.c_str(), nullptr);
}

std::optional<long long> python_int(std::string_view text) {
    const std::optional<std::string> cleaned = without_underscores(strip(text));
    if (!cleaned || cleaned->empty()) {
        return std::nullopt;
    }
    std::string_view number = *cleaned;
    bool negative = false;
    if (number.front() == '+' || number.front() == '-') {
        negative = number.front() == '-';
        number.remove_prefix(1);
    }
    if (number.empty()) {
        return std::nullopt;
    }
    // The lowest number is one further from 0 than the highest: FFmpeg writes
    // it for a missing timestamp (AV_NOPTS_VALUE).
    unsigned long long value = 0;
    const unsigned long long limit =
        static_cast<unsigned long long>(std::numeric_limits<long long>::max()) + (negative ? 1 : 0);
    for (const char c : number) {
        if (!is_digit(c)) {
            return std::nullopt;
        }
        const auto digit = static_cast<unsigned long long>(c - '0');
        if (value > (limit - digit) / 10) {
            return std::nullopt;  // Python would give a bigger number; none of ours is
        }
        value = value * 10 + digit;
    }
    // Negated as an unsigned number, which wraps to the lowest one exactly.
    return static_cast<long long>(negative ? 0 - value : value);
}

std::string replace_invalid_utf8(std::string_view text) {
    constexpr std::string_view replacement = "\xEF\xBF\xBD";  // U+FFFD
    std::string result;
    result.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<unsigned char>(text[i]);
        if (lead < 0x80) {
            result += text[i++];
            continue;
        }
        // The length of the sequence the lead byte starts, and the range its
        // second byte must be in (which rules out overlong forms, surrogates
        // and numbers past U+10FFFF).
        std::size_t length = 0;
        unsigned char low = 0x80;
        unsigned char high = 0xBF;
        if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            low = lead == 0xE0 ? 0xA0 : 0x80;
            high = lead == 0xED ? 0x9F : 0xBF;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            low = lead == 0xF0 ? 0x90 : 0x80;
            high = lead == 0xF4 ? 0x8F : 0xBF;
        }
        if (length == 0) {  // not a lead byte at all
            result += replacement;
            ++i;
            continue;
        }
        // The longest well-formed beginning of the sequence; a sequence cut
        // short becomes one U+FFFD, as Python decodes it.
        std::size_t good = 1;
        while (good < length && i + good < text.size()) {
            const auto byte = static_cast<unsigned char>(text[i + good]);
            const unsigned char from = good == 1 ? low : 0x80;
            const unsigned char to = good == 1 ? high : 0xBF;
            if (byte < from || byte > to) {
                break;
            }
            ++good;
        }
        if (good == length) {
            result.append(text.substr(i, length));
        } else {
            result += replacement;
        }
        i += good;
    }
    return result;
}

std::string ascii_lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return out;
}

std::string ascii_upper(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 'a' + 'A');
        }
    }
    return out;
}

namespace {

// The whitespace Python's str.strip() takes off ASCII text.
bool is_python_space(char c) { return c == ' ' || (c >= '\t' && c <= '\r') || (c >= '\x1c' && c <= '\x1f'); }

// Whether `c` starts a UTF-8 character: any byte but a continuation byte.
bool starts_character(char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }

}  // namespace

std::string_view python_strip(std::string_view text) {
    while (!text.empty() && is_python_space(text.front())) {
        text.remove_prefix(1);
    }
    return python_rstrip(text);
}

std::string_view python_rstrip(std::string_view text) {
    while (!text.empty() && is_python_space(text.back())) {
        text.remove_suffix(1);
    }
    return text;
}

std::string first_characters(std::string_view text, std::size_t count) {
    std::size_t seen = 0;
    for (std::size_t at = 0; at < text.size(); ++at) {
        if (starts_character(text[at]) && seen++ == count) {
            return std::string(text.substr(0, at));
        }
    }
    return std::string(text);
}

std::string last_characters(std::string_view text, std::size_t count) {
    std::size_t start = text.size();
    for (std::size_t kept = 0; start > 0 && kept < count; ++kept) {
        do {
            --start;
        } while (start > 0 && !starts_character(text[start]));
    }
    return std::string(text.substr(start));
}

std::string python_string_repr(std::string_view text) {
    const bool double_quotes = text.find('\'') != std::string_view::npos && text.find('"') == std::string_view::npos;
    const char quote = double_quotes ? '"' : '\'';
    std::string quoted(1, quote);
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == quote || c == '\\') {
            quoted += '\\';
            quoted += c;
        } else if (c == '\t') {
            quoted += "\\t";
        } else if (c == '\n') {
            quoted += "\\n";
        } else if (c == '\r') {
            quoted += "\\r";
        } else if (byte < 0x20 || byte == 0x7F) {
            constexpr char digits[] = "0123456789abcdef";
            quoted += "\\x";
            quoted += digits[byte >> 4];
            quoted += digits[byte & 0xF];
        } else {
            quoted += c;
        }
    }
    return quoted + quote;
}

std::string python_os_error(const std::error_code& error, const std::filesystem::path& path) {
    std::string prefix = "[Errno " + std::to_string(error.value()) + "] ";
    std::string reason;
#ifdef _WIN32
    if (error.category() == std::system_category()) {
        prefix = "[WinError " + std::to_string(error.value()) + "] ";
        reason = error.message();
    } else {
        // The C library's own words, as Python's strerror() (the standard
        // library's generic category words them in lower case).
        char text[256] = {};
        strerror_s(text, sizeof text, error.value());
        reason = text;
    }
#else
    reason = error.message();
#endif
    // Windows' text ends in a full stop, which Python leaves out.
    while (!reason.empty() && (reason.back() == '.' || std::isspace(static_cast<unsigned char>(reason.back())))) {
        reason.pop_back();
    }
    return prefix + reason + ": " + python_string_repr(platform::path_text(path));
}

std::vector<std::string_view> python_splitlines(std::string_view text) {
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        // How many bytes the line break at i takes, if there is one.
        std::size_t length = 0;
        const auto byte = static_cast<unsigned char>(text[i]);
        if (byte == '\r') {
            length = i + 1 < text.size() && text[i + 1] == '\n' ? 2 : 1;
        } else if (byte == '\n' || byte == '\v' || byte == '\f' || (byte >= 0x1C && byte <= 0x1E)) {
            length = 1;
        } else if (byte == 0xC2 && i + 1 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x85) {
            length = 2;  // U+0085
        } else if (byte == 0xE2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x80 &&
                   (static_cast<unsigned char>(text[i + 2]) == 0xA8 ||
                    static_cast<unsigned char>(text[i + 2]) == 0xA9)) {
            length = 3;  // U+2028, U+2029
        }
        if (length == 0) {
            ++i;
            continue;
        }
        lines.push_back(text.substr(start, i - start));
        i += length;
        start = i;
    }
    if (start < text.size()) {
        lines.push_back(text.substr(start));
    }
    return lines;
}

std::string python_list2cmdline(const std::vector<std::string>& arguments) {
    std::string line;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string& argument = arguments[i];
        if (i > 0) {
            line += ' ';
        }
        const bool quoted = argument.empty() || argument.find_first_of(" \t") != std::string::npos;
        if (quoted) {
            line += '"';
        }
        std::size_t backslashes = 0;  // waiting to be doubled, if a quote follows them
        for (const char c : argument) {
            if (c == '\\') {
                ++backslashes;
                continue;
            }
            if (c == '"') {
                line.append(backslashes * 2, '\\');
                line += "\\\"";
            } else {
                line.append(backslashes, '\\');
                line += c;
            }
            backslashes = 0;
        }
        line.append(backslashes, '\\');
        if (quoted) {
            line.append(backslashes, '\\');  // doubled before the closing quote
            line += '"';
        }
    }
    return line;
}

std::size_t python_length(std::string_view text) {
    return static_cast<std::size_t>(std::count_if(text.begin(), text.end(), starts_character));
}

}  // namespace vml
