#include "formatting.h"

#include <cmath>
#include <format>

#include "i18n.h"
#include "vml/number_format.h"

namespace vml::app {

QString media_info_string(const VideoInfo& info) {
    return QString::fromStdString(std::format("{}x{}, {:.2f}fps, ", info.width, info.height, info.fps)) +
           QString::fromStdString(info.codec_name);
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

QColor vmaf_band_colour(double mean) {
    if (mean >= 95) {
        return {198, 239, 206};  // green
    }
    if (mean >= 90) {
        return {255, 235, 156};  // amber
    }
    return {255, 199, 206};  // pink
}

QString preset_label(const CvvdpPreset& preset) {
    const QString name = QString::fromStdString(preset.name);
    return preset.builtin ? tr(name) : name;
}

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
