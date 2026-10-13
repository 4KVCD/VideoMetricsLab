// JSON written as the Python app writes it, so that scripts reading the
// command line's --json, and files compared byte for byte, see no
// difference between the two apps.
#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace vml {

// `value` as Python's json.dumps(value, indent=indent) writes it: keys in
// their order, `indent` spaces a level, every character past ASCII as
// \uXXXX (surrogate pairs past U+FFFF), floats as Python's repr() writes
// them, and NaN and infinities as Python writes them (NaN, Infinity,
// -Infinity), although JSON has none.
std::string python_json_dumps(const nlohmann::ordered_json& value, int indent = 2);

// `value` as the Python app's saved scores write it (metric_cache's
// _canonical): json.dumps(value, sort_keys=True, separators=(",", ":"),
// allow_nan=False). A sorted json keeps its keys in Python's order. Throws
// std::invalid_argument for NaN or an infinity, which Python refuses too.
std::string python_json_compact(const nlohmann::json& value);

}  // namespace vml
