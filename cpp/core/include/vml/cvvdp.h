// ColorVideoVDP (CVVDP) settings: the display it models, the presets, and
// how a display is described to Vship. A port of vmaf_app/core/cvvdp.py.
//
// CVVDP predicts how visible the differences between two videos are to a
// person watching a particular display from a particular distance in a
// particular room, so its score depends on that display as much as on the
// videos. A display is described by its physical properties, never by a
// name, so that renaming a preset changes nothing. The built-in presets are
// the official ColorVideoVDP display models that fit comparing video
// encodes (pycvvdp's display_models.json, v0.5.7), with the official tool's
// own values where a model leaves a property out (contrast 1000:1,
// reflectivity 0.005, exposure 1).
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vml {

// A display as CVVDP models it. The resolution is kept as a number, as the
// Python app keeps a hand-edited one ("width": 3840.5), and made whole
// where it is used.
struct CvvdpDisplay {
    double width = 3840;
    double height = 2160;
    double diagonal_inches = 30.0;
    double viewing_distance_m = 0.7472;
    double peak_luminance = 200.0;  // cd/m^2 (nits)
    double contrast = 1000.0;       // peak : black, 1000 for 1000:1
    double ambient_lux = 250.0;     // light falling on the screen
    double reflectivity = 0.005;    // the share of ambient light the screen reflects
    double exposure = 1.0;
    bool hdr = false;

    // Throws std::invalid_argument for a display CVVDP cannot model.
    void validate() const;

    // What a score depends on: the physical properties, rounded so that a
    // value typed or read back as 0.7472 is the same display.
    nlohmann::ordered_json identity() const;

    // The viewing distance as a multiple of the screen's height, which the
    // diagonal and the aspect ratio give.
    double distance_in_heights() const;
};

// Everything that changes a CVVDP score besides the two videos.
struct CvvdpSettings {
    CvvdpDisplay display;
    // Whether the compared frames are scaled up to fill the display, aspect
    // kept (Vship's resizeToDisplay). Off is the official tool's default:
    // frames are shown pixel for pixel, so a 1080p picture covers a quarter
    // of a 4K display.
    bool resize_to_display = false;

    // The settings as a request holds them (the Python app's
    // spec_parameters): the display's identity, and whether frames fill it.
    nlohmann::ordered_json parameters() const;

    // The settings a request's parameters say (from_spec_parameters): the
    // identity's rounded values, so that what is scored is exactly what a
    // saved score says it was scored with. Throws std::invalid_argument for
    // a display CVVDP cannot model, and nlohmann's errors for parameters of
    // another shape.
    static CvvdpSettings from_parameters(const nlohmann::json& parameters);

    // Whether two settings give the same scores.
    bool same_as(const CvvdpSettings& other) const { return parameters() == other.parameters(); }
};

// The display as Vship 5.1 reads it: the JSON text of a display_models.json
// entry under vship_model_key (Vship takes text that starts with "{" as the
// config itself), with every property given, so that none falls back to a
// Vship default.
inline constexpr const char* vship_model_key = "videometricslab_display";
std::string vship_display_json(const CvvdpDisplay& display);

// A named display. Only its display is the preset: whether a video is scaled
// to fill the display is each video's own setting.
struct CvvdpPreset {
    std::string name;
    CvvdpSettings settings;
    bool builtin = false;
};

// The built-in presets, the first of them the default for every video.
const std::vector<CvvdpPreset>& builtin_cvvdp_presets();

// The built-in presets, then the user's, from Settings' cvvdp_presets (the
// JSON text of its list). A saved preset CVVDP cannot model, or written in a
// form the app does not write, is skipped rather than stopping the run.
std::vector<CvvdpPreset> cvvdp_presets(const std::string& user_presets_json);

// The preset with this name, built-in or the user's, or nothing.
std::optional<CvvdpPreset> cvvdp_preset_named(const std::string& name, const std::string& user_presets_json);

// The preset whose display `settings` model, or nothing (the Python app's
// matching_preset): `preferred`, the preset the video was given, while its
// display still matches, so that of two presets with the same display the
// one chosen is named; else the last with that display, so that a user's
// preset saved from a built-in one is named as the user named it.
std::optional<CvvdpPreset> matching_cvvdp_preset(const CvvdpSettings& settings, const std::string& user_presets_json,
                                                 const std::string& preferred = "");

// The size Vship calculates CVVDP at for frames of `size`: their own size,
// or with resize_to_display, the larger of their own size and their size
// fitted to the display, aspect kept. The larger, because it is not known
// whether Vship shrinks a frame larger than the display, and the size
// decides whether Intel's GPU hangs (vship.h's size_refusal).
std::pair<int, int> cvvdp_computed_size(std::pair<int, int> size, const CvvdpSettings& settings);

// One second of video as CVVDP scored it: its first frame, how many frames
// it has, and their JOD.
struct CvvdpWindow {
    long long first = 0;
    long long frames = 0;
    double jod = 0.0;
};

// The whole video's JOD from its seconds' (the Python app's
// pool_cvvdp_windows). Vship pools a second's per-frame quality q as
// ColorVideoVDP does, JOD(sqrt(sum of q^2 / n)), except that a single frame
// is JOD(q * IMAGE_INT); turning each second's JOD back into its sum
// rebuilds the whole video's.
double pool_cvvdp_windows(const std::vector<CvvdpWindow>& windows);

}  // namespace vml
