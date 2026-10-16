// Reading text as the Python app reads it, so that what ffprobe, FFmpeg's
// logs and settings.json say is taken the same way by both apps.
#pragma once

#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace vml {

// The number Python's float(text) gives, or nothing where it would raise
// ValueError: surrounding whitespace allowed, "inf", "infinity" and "nan"
// in any case, underscores between digits.
std::optional<double> python_float(std::string_view text);

// The number Python's int(text) gives, or nothing where it would raise
// ValueError (or the number would not fit in 64 bits): a whole number in
// base 10, surrounding whitespace allowed, underscores between digits.
std::optional<long long> python_int(std::string_view text);

// `text` with every byte that is not part of well-formed UTF-8 replaced by
// U+FFFD, as Python decodes with errors="replace".
std::string replace_invalid_utf8(std::string_view text);

// ASCII text in lower case, as Python's casefold() makes the ASCII names
// the app compares: FFmpeg's codec and format names, an environment
// variable's value.
std::string ascii_lower(std::string_view text);

// ASCII text in upper case, as Python's upper() makes codec names and
// headings. Bytes past ASCII are kept as they are, so UTF-8 stays intact.
std::string ascii_upper(std::string_view text);

// `text` without the whitespace around it, as Python's str.strip() takes it
// off: the ASCII whitespace, which for str.strip() includes the separators
// 0x1C to 0x1F (float() and int() do not take those off). Python also takes
// Unicode's other spaces off; nothing here needs that.
std::string_view python_strip(std::string_view text);

// `text` without the whitespace at its end, as Python's str.rstrip() takes
// it off (the same whitespace as python_strip).
std::string_view python_rstrip(std::string_view text);

// The first `count` characters of UTF-8 text, as Python's text[:count].
std::string first_characters(std::string_view text, std::size_t count);

// The last `count` characters of UTF-8 text, as Python's text[-count:]
// keeps the end of a program's error output for a message.
std::string last_characters(std::string_view text, std::size_t count);

// `parts` with `separator` between them, as Python's separator.join(parts).
// `parts` is any range of strings or string views.
template <typename Parts>
std::string join(const Parts& parts, std::string_view separator) {
    std::string joined;
    bool first = true;
    for (const auto& part : parts) {
        if (!first) {
            joined += separator;
        }
        joined += part;
        first = false;
    }
    return joined;
}

// The same for a list written in place: join({"a", "b"}, ",").
inline std::string join(std::initializer_list<std::string_view> parts, std::string_view separator) {
    return join<std::initializer_list<std::string_view>>(parts, separator);
}

// Python's repr() of a string, as messages quote names and values: in
// single quotes, or double quotes for text with a single quote and no
// double one, ASCII control characters escaped. Characters past ASCII are
// kept, as repr() keeps the printable ones.
std::string python_string_repr(std::string_view text);

// The system's reason a file or folder could not be made, as Python's
// OSError words it: "[Errno 13] Permission denied: 'C:\\out\\a.json'" for a C
// library code (errno), "[WinError 123] The filename, directory name, or
// volume label syntax is incorrect: 'C:\\out'" for a Windows one.
std::string python_os_error(const std::error_code& error, const std::filesystem::path& path);

// `text`'s lines, as Python's str.splitlines() cuts well-formed UTF-8 text:
// at \n, \r, \r\n, the separators \v, \f and 0x1C to 0x1E, and U+0085,
// U+2028 and U+2029; no empty line after a last line break.
std::vector<std::string_view> python_splitlines(std::string_view text);

// How many characters Python's len() counts in well-formed UTF-8 text.
std::size_t python_length(std::string_view text);

// A command as one line, ready to copy, as Python's
// subprocess.list2cmdline writes it (the Microsoft C run-time's rules): an
// empty argument, or one with a space or a tab, in double quotes; a double
// quote escaped, with the backslashes before it doubled.
std::string python_list2cmdline(const std::vector<std::string>& arguments);

}  // namespace vml
