// How the window names the two Vship builds: "GPU", the app's own build
// (Vship-fast), and "GPU - Vship", stock Vship. Logs, the self-test and the
// command line's `devices` name the library builds (vship_build_label).
#pragma once

#include <QString>

#include "i18n.h"
#include "vml/vship.h"

namespace vml::app {

// A Vship build as the window's choices name it.
inline QString vship_choice_label(VshipBuild build) {
    return build == VshipBuild::stock ? QStringLiteral("GPU - Vship") : QStringLiteral("GPU");
}

// The mark in a cell whose score `made` calculated, not the row's build:
// square brackets, as parentheses already say where a score was calculated
// ("(GPU)", "(CPU)").
inline QString vship_build_mark(VshipBuild made) {
    return made == VshipBuild::stock ? QStringLiteral(" [Vship]") : QStringLiteral(" [GPU]");
}

// The tooltip's word on such a score: made by `made`, where the row's video
// now calculates with `in_use` (vship_build_in_use: its choice, or the other
// build where the chosen one cannot use this GPU).
inline QString other_vship_build_note(VshipBuild made, VshipBuild in_use) {
    return app::tr("\n\nCalculated with {build}, not with this video's {chosen}; their scores differ slightly, so the "
                   "next run calculates it again with {chosen}.",
                   {{"build", vship_choice_label(made)}, {"chosen", vship_choice_label(in_use)}});
}

}  // namespace vml::app
