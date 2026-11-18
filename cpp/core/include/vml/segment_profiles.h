// Weak segment profiles: named sets of weak segment rules (weak_frames.h),
// for different kinds of video. Settings keeps them: the Default profile in
// weak_segment_* (the rules saved before there were profiles), the others in
// weak_segment_profiles, and the one in use in weak_segment_profile, which
// the weak segments report, the exports after each run and the command
// line's --segments saved follow.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/settings.h"
#include "vml/weak_frames.h"

namespace vml {

struct SegmentProfile {
    std::string name;  // "" for Default
    SegmentRules rules;
};

// Every profile: Default first, then the others in their saved order. One
// saved without a name, or with a name another has (Default's too, in any
// case), is left out; its rules are read as saved_segment_rules reads them.
std::vector<SegmentProfile> segment_profiles(const Settings& settings);

// The profile named `name` (any case; "" or "Default" is Default), or none.
std::optional<SegmentProfile> segment_profile_named(const Settings& settings, std::string_view name);

// The profile in use (weak_segment_profile): Default where it names none.
SegmentProfile segment_profile_in_use(const Settings& settings);

// Keeps `profiles` in `settings`: the one named "" in weak_segment_*, the
// others in weak_segment_profiles, in their order (without ranges).
void store_segment_profiles(Settings& settings, const std::vector<SegmentProfile>& profiles);

// Keeps `rules` as profile `name`'s ("" for Default), adding it after the
// others if there is none of that name.
void store_segment_profile(Settings& settings, const std::string& name, const SegmentRules& rules);

// Whether `name` can name a new profile beside `profiles`: not empty or
// only spaces, without a comma (the command line's --segments
// "profile:NAME,vmaf<80" ends the name at one), not Default's, and no other
// profile's (any case). A profile saved with a comma before this is still
// read, and can be renamed.
bool segment_profile_name_free(const std::vector<SegmentProfile>& profiles, std::string_view name);

}  // namespace vml
