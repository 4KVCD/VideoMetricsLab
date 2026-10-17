// Reading the Python app's answers (tests/reference, made by its own code
// when the app was ported) and comparing numbers with them.
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/results.h"

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

// A result as the reference files describe it, its numbers written by
// float.hex().
ComparisonResult result_from(const nlohmann::json& data);

// A provenance as provenance_to_dict writes it.
MetricProvenance provenance_from(const nlohmann::json& data);

// Scores written by float.hex(), as the 32-bit floats results keep them.
std::vector<float> floats_from(const nlohmann::json& list);

}  // namespace vml::test
