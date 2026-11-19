#include "vml/segment_profiles.h"

#include <algorithm>

#include <nlohmann/json.hpp>

#include "vml/python_text.h"

namespace vml {

namespace {

using Json = nlohmann::ordered_json;

constexpr std::string_view default_name = "Default";

bool same_name(std::string_view a, std::string_view b) {
    return ascii_lower(python_strip(a)) == ascii_lower(python_strip(b));
}

SegmentRules rules_of(const Json& profile) {
    std::vector<std::string> conditions;
    if (const auto found = profile.find("conditions"); found != profile.end() && found->is_array()) {
        for (const Json& condition : *found) {
            if (condition.is_string()) {
                conditions.push_back(condition.get<std::string>());
            }
        }
    }
    const auto boolean = [&profile](const char* key) {
        const auto found = profile.find(key);
        return found != profile.end() && found->is_boolean() && found->get<bool>();
    };
    const auto number = [&profile](const char* key, int otherwise) {
        const auto found = profile.find(key);
        return found != profile.end() && found->is_number_integer() ? found->get<int>() : otherwise;
    };
    return saved_segment_rules(conditions, boolean("all"), number("min_frames", 1), number("merge_frames", 0));
}

}  // namespace

std::vector<SegmentProfile> segment_profiles(const Settings& settings) {
    std::vector<SegmentProfile> profiles{
        {"", saved_segment_rules(settings.weak_segment_conditions, settings.weak_segments_all,
                                 settings.weak_segment_min_frames, settings.weak_segment_merge_frames)}};
    const Json saved = Json::parse(settings.weak_segment_profiles_json, nullptr, false);
    if (!saved.is_array()) {
        return profiles;
    }
    for (const Json& profile : saved) {
        const auto name = profile.find("name");
        if (!profile.is_object() || name == profile.end() || !name->is_string()) {
            continue;
        }
        const std::string text = std::string(python_strip(name->get<std::string>()));
        if (text.empty() || same_name(text, default_name) ||
            std::any_of(profiles.begin(), profiles.end(),
                        [&text](const SegmentProfile& other) { return same_name(other.name, text); })) {
            continue;
        }
        profiles.push_back({text, rules_of(profile)});
    }
    return profiles;
}

std::optional<SegmentProfile> segment_profile_named(const Settings& settings, std::string_view name) {
    const std::vector<SegmentProfile> profiles = segment_profiles(settings);
    if (python_strip(name).empty() || same_name(name, default_name)) {
        return profiles.front();
    }
    const auto found = std::find_if(profiles.begin() + 1, profiles.end(),
                                    [name](const SegmentProfile& profile) { return same_name(profile.name, name); });
    return found != profiles.end() ? std::optional(*found) : std::nullopt;
}

SegmentProfile segment_profile_in_use(const Settings& settings) {
    return segment_profile_named(settings, settings.weak_segment_profile).value_or(segment_profiles(settings).front());
}

void store_segment_profiles(Settings& settings, const std::vector<SegmentProfile>& profiles) {
    Json list = Json::array();
    for (const SegmentProfile& profile : profiles) {
        if (profile.name.empty()) {
            settings.weak_segment_conditions = condition_texts(profile.rules.conditions);
            settings.weak_segments_all = profile.rules.all;
            settings.weak_segment_min_frames = profile.rules.min_frames;
            settings.weak_segment_merge_frames = profile.rules.merge_frames;
            continue;
        }
        list.push_back(Json{{"name", profile.name},
                            {"conditions", condition_texts(profile.rules.conditions)},
                            {"all", profile.rules.all},
                            {"min_frames", profile.rules.min_frames},
                            {"merge_frames", profile.rules.merge_frames}});
    }
    settings.weak_segment_profiles_json = list.dump();
}

void store_segment_profile(Settings& settings, const std::string& name, const SegmentRules& rules) {
    std::vector<SegmentProfile> profiles = segment_profiles(settings);
    const auto found = std::find_if(profiles.begin(), profiles.end(), [&name](const SegmentProfile& profile) {
        return name.empty() ? profile.name.empty() : !profile.name.empty() && same_name(profile.name, name);
    });
    SegmentRules kept = rules;
    kept.ranges.clear();
    if (found != profiles.end()) {
        found->rules = kept;
    } else {
        profiles.push_back({name, kept});
    }
    store_segment_profiles(settings, profiles);
}

bool segment_profile_name_free(const std::vector<SegmentProfile>& profiles, std::string_view name) {
    return !python_strip(name).empty() && name.find(',') == std::string_view::npos &&
           !same_name(name, default_name) &&
           std::none_of(profiles.begin(), profiles.end(),
                        [name](const SegmentProfile& profile) { return same_name(profile.name, name); });
}

}  // namespace vml
