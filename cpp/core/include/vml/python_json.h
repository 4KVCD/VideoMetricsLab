// JSON written as the Python app writes it, so that scripts reading the
// command line's --json, and files compared byte for byte, see no
// difference between the two apps.
#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace vml {

// `value` as Python's json.dumps(value, indent=2) writes it: keys in their
// order, two spaces of indent, ", " and ": " separators, every character
// past ASCII as \uXXXX (surrogate pairs past U+FFFF), floats as Python's
// repr() writes them, and NaN and infinities as Python writes them (NaN,
// Infinity, -Infinity), although JSON has none.
std::string python_json_dumps(const nlohmann::ordered_json& value);

}  // namespace vml
