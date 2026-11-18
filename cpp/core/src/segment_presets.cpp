#include "vml/segment_presets.h"

#include <algorithm>

#include "vml/python_text.h"
#include "vml/translatable.h"

namespace vml {

namespace {

SegmentRules rules(std::vector<WeakCondition> conditions, int min_frames, int merge_frames) {
    SegmentRules made;
    made.conditions = std::move(conditions);
    made.min_frames = min_frames;
    made.merge_frames = merge_frames;
    return made;
}

}  // namespace

// The thresholds are starting points, by each metric's own scale: VMAF's
// 0-100 (90+ is a good encode), SSIMULACRA2's (70+ high quality, 50 medium,
// 30 low) and CVVDP's JOD (10 is no visible difference). Any condition makes
// a frame weak. The shortest segment leaves out a frame or two of flicker
// where a kind of video has them; the gap joins dips that read as one.
const std::vector<SegmentPreset>& segment_presets() {
    static const std::vector<SegmentPreset> presets{
        {"animation", N_("Animation"),
         N_("Flat colour and sharp lines, which encode well and score high: high VMAF and SSIMULACRA2 thresholds "
            "catch banding and damaged lines."),
         rules({{"vmaf", '<', 93.0}, {"ssimulacra2", '<', 75.0}}, 3, 6)},
        {"cgi", N_("CGI"),
         N_("Clean rendered detail and smooth gradients: high VMAF and SSIMULACRA2 thresholds catch lost detail and "
            "banding."),
         rules({{"vmaf", '<', 92.0}, {"ssimulacra2", '<', 70.0}}, 3, 6)},
        {"digital", N_("Grain-free digital"),
         N_("Clean live action from digital cameras: the usual thresholds of a high-quality encode."),
         rules({{"vmaf", '<', 90.0}, {"ssimulacra2", '<', 65.0}}, 3, 6)},
        {"grainy-film", N_("Grainy film"),
         N_("Film grain lowers the scores of frames whose look is kept: lower thresholds, and only stretches of six "
            "frames or more, so that grain alone is not weak."),
         rules({{"vmaf", '<', 85.0}, {"ssimulacra2", '<', 55.0}}, 6, 12)},
        {"hdr", N_("HDR"),
         N_("CVVDP, which models HDR displays (VMAF's models are for SDR video): CVVDP must be scored."),
         rules({{"cvvdp", '<', 9.0}}, 1, 24)},
        {"preview", N_("Low-bitrate preview"),
         N_("Only severe breakdowns lasting twelve frames or more, for encodes not meant to look like the source."),
         rules({{"vmaf", '<', 70.0}, {"ssimulacra2", '<', 40.0}}, 12, 24)},
    };
    return presets;
}

const SegmentPreset* segment_preset(std::string_view name) {
    const std::string wanted = ascii_lower(python_strip(name));
    const auto& presets = segment_presets();
    const auto found = std::find_if(presets.begin(), presets.end(), [&wanted](const SegmentPreset& preset) {
        return preset.key == wanted || ascii_lower(preset.name) == wanted;
    });
    return found != presets.end() ? &*found : nullptr;
}

}  // namespace vml
