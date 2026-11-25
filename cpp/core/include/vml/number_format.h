// Numbers as text, written as the Python app (v2.0 and before) wrote them,
// so that the app's CSV and JSON files and its tables read as they did.
#pragma once

#include <cstdint>
#include <string>

namespace vml {

// `value` as Python's repr() writes it: the shortest text that reads back as
// the same number, with ".0" on a whole number ("90.0"), exponent form below
// 0.0001 and from 1e16 ("1e-05", "1e+16"), and "nan", "inf" or "-inf".
std::string python_repr(double value);

// `value` with `decimals` digits after the point, rounded as Python's
// "{:.2f}" rounds it (to the nearest, from the number's exact binary value).
std::string format_fixed(double value, int decimals);

// Python's round(value, decimals): the decimal format_fixed gives, read back.
double python_round(double value, int decimals);

// `value` as Python's "{:g}" writes it: six significant digits, no
// trailing zeros, exponent form below 0.0001 and from 1e6 ("1e+06"); with
// `digits`, as "{:.12g}" (12) does, exponent form from 1e12.
std::string format_general(double value, int digits = 6);

// A file size as the app shows it, 1,024 bytes to a KB: whole KB, at least
// 1, under 1 MB ("12 KB"); else MB to one decimal ("3.4 MB").
struct ShownSize {
    double value;
    bool megabytes;
};
ShownSize shown_size(std::uintmax_t bytes);
// shown_size as text, for the command line (the window translates its own).
std::string file_size_text(std::uintmax_t bytes);

}  // namespace vml
