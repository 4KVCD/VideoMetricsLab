#include "formatting.h"

#include <format>

#include "i18n.h"

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
        return approximate +
               QString::fromStdString(std::format("{:.1f} Mb/s", static_cast<double>(info.bit_rate) / 1'000'000.0));
    }
    // Python's floor division: bitrates are never negative.
    return approximate + QString::fromStdString(std::format("{} kb/s", info.bit_rate / 1000));
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

}  // namespace vml::app
