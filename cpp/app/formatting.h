// How probe results and scores are rendered as text and colour in the table
// (the Python app's vmaf_app/ui/formatting.py): data in, something to show
// out, so the wording and thresholds are in one place.
#pragma once

#include <QString>

#include "vml/cvvdp.h"
#include "vml/video.h"

namespace vml::app {

// Shown wherever a value genuinely doesn't exist, as opposed to being 0.
inline const QString not_computed = QStringLiteral("N/A");

// The one-line summary shown in the Media info column: "1920x1080, 23.98fps,
// hevc".
QString media_info_string(const VideoInfo& info);

// Bitrate at a human scale: Mb/s once it is over a megabit, else kb/s; "≈"
// when it is the whole file's, the soundtrack included (bitrate_note).
QString bitrate_string(const VideoInfo& info);

// The Bitrate cell's tooltip: what the figure is when it is not the video's
// own.
QString bitrate_note(const VideoInfo& info);

// A CVVDP preset's name as shown: a built-in one's translated, the user's own
// as they wrote it. The name itself stays the identifier.
QString preset_label(const CvvdpPreset& preset);

// The CVVDP display choice that follows the reference (automatic_cvvdp_preset),
// as the Settings and Options lists show it.
QString automatic_cvvdp_label();

// A display in one line, for tooltips, in the window's language (the Python
// app's display_description): '30" 3840x2160 SDR, 200 nits, 250 lux, 0.75 m
// (2.0 x screen height)'. CvvdpDisplay::describe() gives it in English, for
// the log.
QString display_description(const CvvdpDisplay& display);

}  // namespace vml::app
