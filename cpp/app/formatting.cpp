#include "formatting.h"

#include <cmath>
#include <format>

#include <QStringList>

#include "i18n.h"
#include "vml/film_grain_detect.h"
#include "vml/film_grain_scoring.h"
#include "vml/number_format.h"

namespace vml::app {

QString media_info_string(const VideoInfo& info) {
    QString text = QString::fromStdString(std::format("{}x{}, {:.2f}fps, ", info.width, info.height, info.fps)) +
                   QString::fromStdString(info.codec_name);
    const FilmGrainScoring grain = effective_film_grain(info);
    if (grain.scoring != GrainScoring::none) {
        text += ", " + (grain.scoring == GrainScoring::with ? tr("film grain") : tr("film grain left out"));
    }
    return text;
}

QString film_grain_note(const VideoInfo& info) {
    const FilmGrainScoring grain = effective_film_grain(info);
    if (grain.scoring == GrainScoring::none) {
        return {};
    }
    QStringList kinds;
    for (const std::string& name : film_grain_names(grain.kinds)) {
        kinds.append(name == "h274"           ? QStringLiteral("H.274")
                     : name == "h274 model 1" ? tr("H.274 model 1")
                     : name == "afgs1"        ? QStringLiteral("AFGS1")
                     : name == "av1"          ? QStringLiteral("AV1")
                                              : QString::fromStdString(name));
    }
    const QString listed = kinds.join(", ");
    if (!grain.why_without.empty()) {
        return tr("Film grain ({kinds}) left out: {why}",
                  {{"kinds", listed}, {"why", tr_message(QString::fromStdString(grain.why_without))}});
    }
    return grain.scoring == GrainScoring::with
               ? tr("Film grain ({kinds}): the scores are made with it, as viewers see the video", {{"kinds", listed}})
               : tr("Film grain ({kinds}): the scores are made without it, from the coded pictures",
                    {{"kinds", listed}});
}

QString bitrate_string(const VideoInfo& info) {
    if (info.bit_rate == 0) {
        return not_computed;
    }
    const QString approximate = info.bit_rate_whole_file ? QStringLiteral("≈") : QString();
    if (info.bit_rate >= 1'000'000) {
        return approximate + tr("{rate:.1f} Mb/s", {{"rate", static_cast<double>(info.bit_rate) / 1'000'000.0}});
    }
    // Python's floor division: bitrates are never negative.
    return approximate + tr("{rate} kb/s", {{"rate", info.bit_rate / 1000}});
}

QString bitrate_note(const VideoInfo& info) {
    if (info.bit_rate != 0 && info.bit_rate_whole_file) {
        return tr("The whole file's bitrate, soundtrack included: the file does not record the video's own.");
    }
    return {};
}

QString preset_label(const CvvdpPreset& preset) {
    const QString name = QString::fromStdString(preset.name);
    return preset.builtin ? tr(name) : name;
}

QString automatic_cvvdp_label() { return tr("Automatic (SDR or HDR, as the reference is)"); }

QString display_description(const CvvdpDisplay& display) {
    // The resolution as CvvdpDisplay::describe() writes it: a whole number
    // as one, a hand-edited fraction ("3840.5") as it is.
    const auto resolution = [](double value) {
        return value == std::floor(value) ? std::to_string(static_cast<long long>(value)) : python_repr(value);
    };
    return tr("{diagonal:g}\" {width}x{height} {range}, {nits:g} nits, {lux:g} lux, {distance:.2f} m "
              "({heights:.1f} x screen height)",
              {{"diagonal", display.diagonal_inches},
               {"width", resolution(display.width)},
               {"height", resolution(display.height)},
               {"range", display.hdr ? "HDR" : "SDR"},
               {"nits", display.peak_luminance},
               {"lux", display.ambient_lux},
               {"distance", display.viewing_distance_m},
               {"heights", display.distance_in_heights()}});
}

}  // namespace vml::app
