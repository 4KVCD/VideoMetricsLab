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

// Python's bool() of a value read from JSON: false for null, false, zero,
// and an empty text, list or object.
template <typename Json>
bool python_truthy(const Json& value) {
    switch (value.type()) {
        case Json::value_t::null:
            return false;
        case Json::value_t::boolean:
            return value.template get<bool>();
        case Json::value_t::number_integer:
        case Json::value_t::number_unsigned:
        case Json::value_t::number_float:
            return value.template get<double>() != 0.0;  // NaN is true, as in Python
        case Json::value_t::string:
            return !value.template get_ref<const std::string&>().empty();
        case Json::value_t::array:
        case Json::value_t::object:
            return !value.empty();
        default:
            return true;
    }
}

}  // namespace vml
