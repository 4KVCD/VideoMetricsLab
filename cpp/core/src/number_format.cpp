#include "vml/number_format.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace vml {

namespace {

// Text for NaN and the infinities, as Python writes them in both repr() and
// "{:.2f}". (A C++ library may write "-nan(ind)" or the like instead.)
const char* non_finite_text(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    return value > 0 ? "inf" : "-inf";
}

// A finite number's shortest round-trip digits and its decimal exponent:
// 94.5 is digits "945", exponent 1 (9.45e+01).
struct ShortestDigits {
    bool negative;
    std::string digits;
    int exponent;
};

ShortestDigits shortest_digits(double value) {
    char buffer[32];  // the longest is "-1.2345678901234567e-308"
    const auto [end, error] = std::to_chars(buffer, buffer + sizeof buffer, value, std::chars_format::scientific);
    if (error != std::errc()) {
        throw std::runtime_error("a number could not be written as text");
    }
    // The text is "[-]d[.ddd]e(+|-)xx".
    std::string_view text(buffer, static_cast<std::size_t>(end - buffer));
    ShortestDigits result{false, {}, 0};
    if (text.front() == '-') {
        result.negative = true;
        text.remove_prefix(1);
    }
    const std::size_t e = text.find('e');
    for (const char c : text.substr(0, e)) {
        if (c != '.') {
            result.digits += c;
        }
    }
    const std::string_view exponent = text.substr(e + 1);
    for (const char c : exponent.substr(1)) {  // the digits after the sign
        result.exponent = result.exponent * 10 + (c - '0');
    }
    if (exponent.front() == '-') {
        result.exponent = -result.exponent;
    }
    return result;
}

}  // namespace

std::string python_repr(double value) {
    if (!std::isfinite(value)) {
        return non_finite_text(value);
    }
    const ShortestDigits number = shortest_digits(value);
    const std::string& digits = number.digits;
    std::string text = number.negative ? "-" : "";

    // Python's rule (format_float_short in its pystrtod.c): the decimal point
    // sits `point` places after the start of the digits, and exponent form is
    // used when that is four or more places before the first digit (1.23e-05,
    // not 0.0000123) or more than 16 after its start (1e+16, not
    // 10000000000000000.0).
    const int point = number.exponent + 1;
    if (-4 < point && point <= 16) {
        if (point <= 0) {  // 0.000123
            text += "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
        } else if (static_cast<std::size_t>(point) >= digits.size()) {  // 90.0, 1000.0
            text += digits + std::string(static_cast<std::size_t>(point) - digits.size(), '0') + ".0";
        } else {  // 94.5
            text += digits.substr(0, static_cast<std::size_t>(point)) + "." +
                    digits.substr(static_cast<std::size_t>(point));
        }
        return text;
    }
    // 1.5e-05, 1e+16: at least two exponent digits, always with its sign.
    text += digits.front();
    if (digits.size() > 1) {
        text += "." + digits.substr(1);
    }
    const int magnitude = std::abs(number.exponent);
    text += number.exponent < 0 ? "e-" : "e+";
    if (magnitude < 10) {
        text += '0';
    }
    text += std::to_string(magnitude);
    return text;
}

std::string format_fixed(double value, int decimals) {
    if (!std::isfinite(value)) {
        return non_finite_text(value);
    }
    // to_chars rounds from the number's exact binary value, as Python does.
    char buffer[400];  // DBL_MAX has 309 digits before the point
    const auto [end, error] =
        std::to_chars(buffer, buffer + sizeof buffer, value, std::chars_format::fixed, decimals);
    if (error != std::errc()) {
        throw std::runtime_error("a number could not be written as text");
    }
    return std::string(buffer, end);
}

}  // namespace vml
