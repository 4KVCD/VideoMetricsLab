#include "vml/update_check.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>

#include "vml/python_json.h"
#include "vml/python_text.h"

namespace vml {

std::optional<std::vector<long long>> version_tuple(std::string_view text) {
    // As Python's re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip()).
    text = python_strip(text);
    if (text.starts_with('v')) {
        text.remove_prefix(1);
    }
    std::vector<long long> parts;
    while (true) {
        long long part = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), part);
        if (error != std::errc() || end == text.data() || *text.data() == '-' || *text.data() == '+') {
            return std::nullopt;  // no digits here (or more than a version holds)
        }
        parts.push_back(part);
        text.remove_prefix(static_cast<std::size_t>(end - text.data()));
        if (text.empty()) {
            break;
        }
        if (text.front() != '.') {
            return std::nullopt;
        }
        text.remove_prefix(1);
    }
    while (parts.size() > 1 && parts.back() == 0) {
        parts.pop_back();
    }
    return parts;
}

bool is_newer(std::string_view candidate, std::string_view current) {
    const std::optional<std::vector<long long>> newer = version_tuple(candidate);
    const std::optional<std::vector<long long>> older = version_tuple(current);
    return newer && older && *newer > *older;  // as Python compares tuples
}

Release release_from_answer(std::string_view body) {
    const nlohmann::json data = nlohmann::json::parse(body.begin(), body.end(), nullptr, false);
    if (data.is_discarded()) {
        throw UpdateCheckError("GitHub's answer is not JSON");
    }
    // Each field as Python reads it: str(data.get(name) or default).
    const auto field = [&data](const char* name, const std::string& otherwise) {
        if (!data.is_object() || !data.contains(name) || !python_truthy(data.at(name))) {
            return otherwise;
        }
        return python_str(data.at(name));
    };
    const std::string tag = field("tag_name", "");
    if (!version_tuple(tag)) {
        throw UpdateCheckError("GitHub named no release version (tag " + python_string_repr(tag) + ")");
    }
    std::string_view version = python_strip(tag);
    if (version.starts_with('v')) {
        version.remove_prefix(1);
    }
    return Release{std::string(version), field("html_url", "https://github.com/4KVCD/VideoMetricsLab/releases/latest"),
                   field("body", "")};
}

}  // namespace vml
