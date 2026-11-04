// How the window names the Vship builds: "GPU", the app's own build
// (Vship-fast), and "Vship Vulkan", "Vship HIP" and "Vship CUDA", stock Vship's.
// Logs, the self-test and the command line's `devices` name the library builds
// (vship_build_label).
#pragma once

#include <QAbstractItemView>
#include <QComboBox>
#include <QString>

#include <algorithm>
#include <vector>

#include "i18n.h"
#include "vml/vship.h"

namespace vml::app {

// A Vship build's name in the window's sentences.
inline QString vship_short_name(VshipBuild build) {
    switch (build) {
    case VshipBuild::stock:
        return QStringLiteral("Vship Vulkan");
    case VshipBuild::hip:
        return QStringLiteral("Vship HIP");
    case VshipBuild::cuda:
        return QStringLiteral("Vship CUDA");
    default:
        return QStringLiteral("GPU");
    }
}

// A Vship build as the window's choices name it.
inline QString vship_choice_label(VshipBuild build) {
    switch (build) {
    case VshipBuild::stock:
        return app::tr("Vship Vulkan (preferred)");
    case VshipBuild::hip:
        return app::tr("Vship HIP (use only if Vulkan has problems)");
    case VshipBuild::cuda:
        return app::tr("Vship CUDA (use only if Vulkan has problems)");
    default:
        return app::tr("GPU (fastest and most accurate)");
    }
}

// A choice of `builds`, each entry's data its value (vship_build_value). Its text
// is long for the panels it sits in: the closed box takes the width it is given,
// and the list that opens shows the whole text.
inline void add_vship_choices(QComboBox& combo, const std::vector<VshipBuild>& builds) {
    for (const VshipBuild build : builds) {
        combo.addItem(vship_choice_label(build), QString::fromUtf8(vship_build_value(build)));
    }
    combo.setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    combo.setMinimumContentsLength(12);
    combo.view()->setMinimumWidth(combo.view()->sizeHintForColumn(0) + 16);
}

// Shows `build` in such a choice, or its first entry where it is not listed.
inline void show_vship_choice(QComboBox& combo, VshipBuild build) {
    combo.setCurrentIndex(std::max(0, combo.findData(QString::fromUtf8(vship_build_value(build)))));
}

// The build such a choice shows.
inline VshipBuild shown_vship_choice(const QComboBox& combo) {
    return vship_build_from(combo.currentData().toString().toStdString());
}

// The mark in a cell whose score `made` calculated, not the row's build:
// square brackets, as parentheses already say where a score was calculated
// ("(GPU)", "(CPU)"). Stock Vship's builds share their scores, so a score says
// only that it is stock Vship's.
inline QString vship_build_mark(VshipBuild made) {
    return made == VshipBuild::fast ? QStringLiteral(" [GPU]") : QStringLiteral(" [Vship]");
}

// The tooltip's word on such a score: made by `made`, where the row's video
// now calculates with `in_use` (vship_build_in_use: its choice, or the other
// build where the chosen one cannot use this GPU).
inline QString other_vship_build_note(VshipBuild made, VshipBuild in_use) {
    return app::tr("\n\nCalculated with {build}, not with this video's {chosen}; their scores differ slightly, so the "
                   "next run calculates it again with {chosen}.",
                   {{"build", vship_short_name(made)}, {"chosen", vship_short_name(in_use)}});
}

}  // namespace vml::app
