// Numbers as text, written as the Python app writes them, so that the C++
// app's CSV and JSON files and its tables read the same as the Python app's.
#pragma once

#include <string>

namespace vml {

// `value` as Python's repr() writes it: the shortest text that reads back as
// the same number, with ".0" on a whole number ("90.0"), exponent form below
// 0.0001 and from 1e16 ("1e-05", "1e+16"), and "nan", "inf" or "-inf".
std::string python_repr(double value);

// `value` with `decimals` digits after the point, rounded as Python's
// "{:.2f}" rounds it (to the nearest, from the number's exact binary value).
std::string format_fixed(double value, int decimals);

// `value` as Python's "{:g}" writes it: six significant digits, no
// trailing zeros, exponent form below 0.0001 and from 1e6 ("1e+06"); with
// `digits`, as "{:.12g}" (12) does, exponent form from 1e12.
std::string format_general(double value, int digits = 6);

}  // namespace vml
