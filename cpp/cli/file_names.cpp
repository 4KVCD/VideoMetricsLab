#include "file_names.h"

#include <array>
#include <cstdio>
#include <string_view>
#include <vector>

namespace vml::cli {

namespace {

constexpr std::array<std::string_view, 6> field_names = {"test", "reference", "version", "date", "time", "sequence"};

bool is_field(std::string_view name) {
    for (const std::string_view field : field_names) {
        if (field == name) {
            return true;
        }
    }
    return false;
}

// The text of field `name` in `fields` (none where it has none).
std::optional<std::string> field_text(std::string_view name, const NameFields& fields) {
    if (name == "test") {
        return fields.test;
    }
    if (name == "reference") {
        return fields.reference;
    }
    if (name == "version") {
        return fields.version;
    }
    if (name == "date") {
        return fields.date;
    }
    if (name == "time") {
        return fields.time;
    }
    if (name == "sequence") {
        return fields.sequence;
    }
    return std::nullopt;
}

// One piece of a name: text as it is, or a field.
struct Piece {
    std::string text;
    bool field = false;
};

// `text` cut into pieces: "{name}" a field, "{{" and "}}" a brace; a brace
// with no partner, or "{...}" that is no field, left as text, with `wrong`
// saying so (the first such).
std::vector<Piece> pieces_of(const std::string& text, std::optional<std::string>& wrong) {
    std::vector<Piece> pieces;
    std::string plain;
    for (std::size_t at = 0; at < text.size(); ++at) {
        const char c = text[at];
        if ((c == '{' || c == '}') && at + 1 < text.size() && text[at + 1] == c) {
            plain += c;
            ++at;
            continue;
        }
        if (c == '}') {
            if (!wrong) {
                wrong = "a } with no { before it (write }} for a brace)";
            }
            plain += c;
            continue;
        }
        if (c != '{') {
            plain += c;
            continue;
        }
        const std::size_t end = text.find('}', at + 1);
        const std::string name = end == std::string::npos ? "" : text.substr(at + 1, end - at - 1);
        if (end == std::string::npos || !is_field(name)) {
            if (!wrong) {
                wrong = end == std::string::npos
                            ? "a { with no } after it (write {{ for a brace)"
                            : "{" + name + "} is not a field: use {test}, {reference}, {version}, {date}, {time} or "
                                           "{sequence}";
            }
            plain += c;
            continue;
        }
        if (!plain.empty()) {
            pieces.push_back({std::move(plain), false});
            plain.clear();
        }
        pieces.push_back({name, true});
        at = end;
    }
    if (!plain.empty()) {
        pieces.push_back({std::move(plain), false});
    }
    return pieces;
}

std::string digits(int value, int width) {
    char text[16];
    std::snprintf(text, sizeof text, "%0*d", width, value);
    return text;
}

}  // namespace

NameFields run_fields(const std::tm& started) {
    NameFields fields;
    fields.version = VML_VERSION;
    fields.date = digits(started.tm_year + 1900, 4) + digits(started.tm_mon + 1, 2) + digits(started.tm_mday, 2);
    fields.time = digits(started.tm_hour, 2) + digits(started.tm_min, 2) + digits(started.tm_sec, 2);
    return fields;
}

std::optional<std::string> name_template_error(const std::string& name, bool batch) {
    std::optional<std::string> wrong;
    const std::vector<Piece> pieces = pieces_of(name, wrong);
    if (wrong) {
        return wrong;
    }
    for (const Piece& piece : pieces) {
        if (!piece.field && piece.text.find_first_of("/\\") != std::string::npos) {
            return "it names files, not folders: no / or \\ (the files go in --csv's, --segments-dir's and "
                   "--run-dir's folders)";
        }
        if (piece.field && piece.text == "sequence" && !batch) {
            return "{sequence} is a --batch row's sequence: only with --batch";
        }
    }
    return std::nullopt;
}

std::optional<std::string> video_field_in(const std::string& text) {
    std::optional<std::string> wrong;
    for (const Piece& piece : pieces_of(text, wrong)) {
        if (piece.field && (piece.text == "test" || piece.text == "reference" || piece.text == "sequence")) {
            return "{" + piece.text + "}";
        }
    }
    return std::nullopt;
}

std::string expand_names(const std::string& text, const NameFields& fields) {
    std::optional<std::string> wrong;
    std::string expanded;
    for (const Piece& piece : pieces_of(text, wrong)) {
        if (!piece.field) {
            expanded += piece.text;
            continue;
        }
        const std::optional<std::string> value = field_text(piece.text, fields);
        expanded += value ? *value : "{" + piece.text + "}";
    }
    return expanded;
}

}  // namespace vml::cli
