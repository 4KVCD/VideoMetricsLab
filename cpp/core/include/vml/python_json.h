// JSON written as the Python app writes it, so that scripts reading the
// command line's --json, and files compared byte for byte, see no
// difference between the two apps.
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>

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

// `value` as the Python app's saved runs write it (run_io.save_run):
// json.dumps(value, allow_nan=False), on one line with ", " between items
// and ": " after keys, keys in their order. Throws std::invalid_argument for
// NaN or an infinity.
std::string python_json_line(const nlohmann::ordered_json& value);

// Python's str() of a value read from JSON: a text itself; anything else as
// JSON writes it.
std::string python_str(const nlohmann::json& value);

// Python's repr() of a value read from JSON: None, True, 3.0, 'text',
// [1, 2], {'a': 1}.
std::string python_repr_of(const nlohmann::json& value);

// data[name] as Python reads it from a dict: throws std::invalid_argument
// for a missing key, saying the key in quotes as Python's KeyError does.
const nlohmann::json& python_item(const nlohmann::json& data, std::string_view name);

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
