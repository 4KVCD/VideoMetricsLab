// Reading the Python app's answers (tests/reference, made by
// make_reference.py) and comparing numbers with them.
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vml::test {

// The JSON file of that name in tests/reference.
nlohmann::json load_reference(std::string_view file_name);

// A number written by float.hex() ("0x1.7ap+6", "inf", "nan"); null is none.
std::optional<double> reference_number(const nlohmann::json& text);
std::vector<double> reference_numbers(const nlohmann::json& list);

// Whether `actual` is the Python app's `expected`: the same number, NaN for
// NaN. On Windows, where the Python app runs, that means exactly; elsewhere,
// to 12 significant digits.
bool same_number(double expected, double actual);

}  // namespace vml::test
