// Reading text as the Python app reads it, so that what ffprobe, FFmpeg's
// logs and settings.json say is taken the same way by both apps.
#pragma once

#include <optional>
#include <string>
#include <string_view>

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

// `text` without the whitespace around it, as Python's str.strip() takes it
// off: the ASCII whitespace, which for str.strip() includes the separators
// 0x1C to 0x1F (float() and int() do not take those off). Python also takes
// Unicode's other spaces off; nothing here needs that.
std::string_view python_strip(std::string_view text);

}  // namespace vml
