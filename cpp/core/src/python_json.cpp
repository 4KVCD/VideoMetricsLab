#include "vml/python_json.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

#include "vml/number_format.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// \uXXXX for a code point up to U+FFFF, in lower-case hex as Python writes
// it.
void append_escape(std::string& out, unsigned code) {
    constexpr char digits[] = "0123456789abcdef";
    out += "\\u";
    for (int shift = 12; shift >= 0; shift -= 4) {
        out += digits[(code >> shift) & 0xF];
    }
}

// The length of the well-formed UTF-8 sequence at `at`, or 0 if there is
// none there: Python's decoder rejects the same (overlong forms,
// surrogates, past U+10FFFF).
std::size_t sequence_length(const std::string& text, std::size_t at) {
    const auto byte_at = [&text](std::size_t i) { return static_cast<unsigned char>(text[i]); };
    const unsigned char lead = byte_at(at);
    const std::size_t length = lead >= 0xC2 && lead <= 0xDF ? 2 : lead >= 0xE0 && lead <= 0xEF ? 3
                               : lead >= 0xF0 && lead <= 0xF4                     ? 4
                                                                                  : 0;
    if (length == 0 || at + length > text.size()) {
        return 0;
    }
    for (std::size_t k = 1; k < length; ++k) {
        if ((byte_at(at + k) & 0xC0) != 0x80) {
            return 0;
        }
    }
    // The second byte's range after these leads.
    const unsigned char second = byte_at(at + 1);
    if ((lead == 0xE0 && second < 0xA0) || (lead == 0xED && second > 0x9F) || (lead == 0xF0 && second < 0x90) ||
        (lead == 0xF4 && second > 0x8F)) {
        return 0;
    }
    return length;
}

// A string in quotes, escaped as Python's ensure_ascii does: every character
// outside the printable ASCII ones as \uXXXX.
void write_string(std::string& out, const std::string& text) {
    out += '"';
    for (std::size_t i = 0; i < text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        if (byte < 0x80) {
            switch (byte) {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                case '\b':
                    out += "\\b";
                    break;
                case '\f':
                    out += "\\f";
                    break;
                default:
                    if (byte < 0x20 || byte == 0x7F) {
                        append_escape(out, byte);
                    } else {
                        out += static_cast<char>(byte);
                    }
            }
            ++i;
            continue;
        }
        const std::size_t length = sequence_length(text, i);
        if (length == 0) {
            // A byte that is not UTF-8 (a POSIX file name can hold any): as
            // Python shows a byte it could not decode, \udcXX.
            append_escape(out, 0xDC00 + byte);
            ++i;
            continue;
        }
        // A character past ASCII: its code point, as \uXXXX or a surrogate
        // pair.
        unsigned code = byte & (length == 4 ? 0x07 : length == 3 ? 0x0F : 0x1F);
        for (std::size_t k = 1; k < length; ++k) {
            code = (code << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
        }
        if (code > 0xFFFF) {
            code -= 0x10000;
            append_escape(out, 0xD800 + (code >> 10));
            append_escape(out, 0xDC00 + (code & 0x3FF));
        } else {
            append_escape(out, code);
        }
        i += length;
    }
    out += '"';
}

void write_value(std::string& out, const nlohmann::ordered_json& value, int indent, int depth) {
    const std::string inner(static_cast<std::size_t>(indent * (depth + 1)), ' ');
    const std::string outer(static_cast<std::size_t>(indent * depth), ' ');
    switch (value.type()) {
        case nlohmann::ordered_json::value_t::object: {
            if (value.empty()) {
                out += "{}";
                return;
            }
            out += "{\n";
            bool first = true;
            for (auto item = value.begin(); item != value.end(); ++item) {
                out += first ? "" : ",\n";
                first = false;
                out += inner;
                write_string(out, item.key());
                out += ": ";
                write_value(out, item.value(), indent, depth + 1);
            }
            out += "\n" + outer + "}";
            return;
        }
        case nlohmann::ordered_json::value_t::array: {
            if (value.empty()) {
                out += "[]";
                return;
            }
            out += "[\n";
            bool first = true;
            for (const auto& item : value) {
                out += first ? "" : ",\n";
                first = false;
                out += inner;
                write_value(out, item, indent, depth + 1);
            }
            out += "\n" + outer + "]";
            return;
        }
        case nlohmann::ordered_json::value_t::string:
            write_string(out, value.get_ref<const std::string&>());
            return;
        case nlohmann::ordered_json::value_t::boolean:
            out += value.get<bool>() ? "true" : "false";
            return;
        case nlohmann::ordered_json::value_t::number_integer:
            out += std::to_string(value.get<long long>());
            return;
        case nlohmann::ordered_json::value_t::number_unsigned:
            out += std::to_string(value.get<unsigned long long>());
            return;
        case nlohmann::ordered_json::value_t::number_float: {
            const double number = value.get<double>();
            if (std::isnan(number)) {
                out += "NaN";
            } else if (std::isinf(number)) {
                out += number > 0 ? "Infinity" : "-Infinity";
            } else {
                out += python_repr(number);
            }
            return;
        }
        default:
            out += "null";
    }
}

// `value` on one line, with `item_separator` between items and
// `key_separator` after keys, refusing NaN and infinities (allow_nan=False).
// A sorted json's keys come in the order Python's sort_keys gives (UTF-8's
// byte order is the code points' order); an ordered one's in their order.
template <typename Json>
void write_flat(std::string& out, const Json& value, std::string_view item_separator, std::string_view key_separator) {
    switch (value.type()) {
        case Json::value_t::object: {
            out += '{';
            bool first = true;
            for (auto item = value.begin(); item != value.end(); ++item) {
                out += first ? std::string_view() : item_separator;
                first = false;
                write_string(out, item.key());
                out += key_separator;
                write_flat(out, item.value(), item_separator, key_separator);
            }
            out += '}';
            return;
        }
        case Json::value_t::array: {
            out += '[';
            bool first = true;
            for (const auto& item : value) {
                out += first ? std::string_view() : item_separator;
                first = false;
                write_flat(out, item, item_separator, key_separator);
            }
            out += ']';
            return;
        }
        case Json::value_t::string:
            write_string(out, value.template get_ref<const std::string&>());
            return;
        case Json::value_t::boolean:
            out += value.template get<bool>() ? "true" : "false";
            return;
        case Json::value_t::number_integer:
            out += std::to_string(value.template get<long long>());
            return;
        case Json::value_t::number_unsigned:
            out += std::to_string(value.template get<unsigned long long>());
            return;
        case Json::value_t::number_float: {
            const double number = value.template get<double>();
            if (!std::isfinite(number)) {
                throw std::invalid_argument("Out of range float values are not JSON compliant");
            }
            out += python_repr(number);
            return;
        }
        default:
            out += "null";
    }
}

}  // namespace

std::string python_json_dumps(const nlohmann::ordered_json& value, int indent) {
    std::string out;
    write_value(out, value, indent, 0);
    return out;
}

std::string python_json_compact(const nlohmann::json& value) {
    std::string out;
    write_flat(out, value, ",", ":");
    return out;
}

std::string python_json_line(const nlohmann::ordered_json& value) {
    std::string out;
    write_flat(out, value, ", ", ": ");
    return out;
}

std::string python_str(const nlohmann::json& value) {
    return value.is_string() ? value.get<std::string>() : value.dump();
}

std::string python_repr_of(const nlohmann::json& value) {
    switch (value.type()) {
        case nlohmann::json::value_t::null:
            return "None";
        case nlohmann::json::value_t::boolean:
            return value.get<bool>() ? "True" : "False";
        case nlohmann::json::value_t::number_integer:
            return std::to_string(value.get<long long>());
        case nlohmann::json::value_t::number_unsigned:
            return std::to_string(value.get<unsigned long long>());
        case nlohmann::json::value_t::number_float:
            return python_repr(value.get<double>());
        case nlohmann::json::value_t::string:
            return python_string_repr(value.get_ref<const std::string&>());
        case nlohmann::json::value_t::array: {
            std::string out = "[";
            for (std::size_t i = 0; i < value.size(); ++i) {
                out += (i == 0 ? "" : ", ") + python_repr_of(value[i]);
            }
            return out + "]";
        }
        case nlohmann::json::value_t::object: {
            std::string out = "{";
            bool first = true;
            for (auto item = value.begin(); item != value.end(); ++item) {
                out += (first ? "" : ", ") + python_string_repr(item.key()) + ": " + python_repr_of(item.value());
                first = false;
            }
            return out + "}";
        }
        default:
            return value.dump();
    }
}

const nlohmann::json& python_item(const nlohmann::json& data, std::string_view name) {
    const auto found = data.is_object() ? data.find(name) : data.end();
    if (found == data.end()) {
        throw std::invalid_argument(python_string_repr(name));
    }
    return *found;
}

}  // namespace vml
