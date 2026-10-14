#include "vml/cvvdp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vml/number_format.h"
#include "vml/python_json.h"
#include "vml/python_text.h"

namespace vml {

namespace {

using Json = nlohmann::ordered_json;

// Python's round(value, digits): the decimal text {:.Nf} gives, read back.
double rounded(double value, int digits) { return python_float(format_fixed(value, digits)).value(); }

// Python's int() of a resolution: the whole part.
long long whole(double value) { return static_cast<long long>(value); }

// A built-in preset made from an official display model.
CvvdpPreset official(std::string name, std::string description, double width, double height, double diagonal_inches,
                     double viewing_distance_m, double peak_luminance, double contrast, double ambient_lux,
                     bool hdr = false) {
    CvvdpDisplay display;
    display.width = width;
    display.height = height;
    display.diagonal_inches = diagonal_inches;
    display.viewing_distance_m = viewing_distance_m;
    display.peak_luminance = peak_luminance;
    display.contrast = contrast;
    display.ambient_lux = ambient_lux;
    display.hdr = hdr;
    return {std::move(name), {display, false}, true, std::move(description)};
}

// Python's bool() of a value read from JSON.
bool truthy(const nlohmann::json& value) {
    switch (value.type()) {
        case nlohmann::json::value_t::null:
            return false;
        case nlohmann::json::value_t::boolean:
            return value.get<bool>();
        case nlohmann::json::value_t::number_integer:
        case nlohmann::json::value_t::number_unsigned:
        case nlohmann::json::value_t::number_float:
            return value.get<double>() != 0.0;
        case nlohmann::json::value_t::string:
        case nlohmann::json::value_t::array:
        case nlohmann::json::value_t::object:
            return !value.empty();
        default:
            return true;
    }
}

// A saved preset's name, as Python's str() makes it of a value that is not
// text; nothing for a list or an object.
std::optional<std::string> name_of(const nlohmann::json& value) {
    switch (value.type()) {
        case nlohmann::json::value_t::string:
            return value.get<std::string>();
        case nlohmann::json::value_t::number_integer:
            return std::to_string(value.get<long long>());
        case nlohmann::json::value_t::number_unsigned:
            return std::to_string(value.get<unsigned long long>());
        case nlohmann::json::value_t::number_float:
            return python_repr(value.get<double>());
        case nlohmann::json::value_t::boolean:
            return value.get<bool>() ? "True" : "False";
        case nlohmann::json::value_t::null:
            return "None";
        default:
            return std::nullopt;
    }
}

// The settings a saved preset holds (the Python app's
// CvvdpSettings.from_dict): the display properties it names, the defaults
// for the rest. Throws std::invalid_argument for one CVVDP cannot model, or
// written in a form the app does not write (the Python app also takes a
// display written as a list of pairs).
CvvdpSettings saved_settings(const nlohmann::json& data) {
    if (!data.is_object()) {
        throw std::invalid_argument("A preset's settings must be an object.");
    }
    CvvdpSettings settings;
    if (const auto display = data.find("display"); display != data.end()) {
        if (!display->is_object()) {
            throw std::invalid_argument("A preset's display must be an object.");
        }
        CvvdpDisplay& into = settings.display;
        const std::pair<const char*, double*> numbers[] = {
            {"width", &into.width},
            {"height", &into.height},
            {"diagonal_inches", &into.diagonal_inches},
            {"viewing_distance_m", &into.viewing_distance_m},
            {"peak_luminance", &into.peak_luminance},
            {"contrast", &into.contrast},
            {"ambient_lux", &into.ambient_lux},
            {"reflectivity", &into.reflectivity},
            {"exposure", &into.exposure},
        };
        for (const auto& [name, value] : numbers) {
            if (const auto found = display->find(name); found != display->end()) {
                if (!found->is_number()) {  // true and false are not numbers here
                    throw std::invalid_argument("Every display value must be a finite number.");
                }
                *value = found->get<double>();
            }
        }
        if (const auto hdr = display->find("hdr"); hdr != display->end()) {
            into.hdr = truthy(*hdr);
        }
    }
    settings.display.validate();
    if (const auto resize = data.find("resize_to_display"); resize != data.end()) {
        settings.resize_to_display = truthy(*resize);
    }
    return settings;
}

// CVVDP's constants (Vship's and ColorVideoVDP's jod_a, jod_exp and
// image_int, from Vship 5.1.1's src/HIP/cvvdp/parameters.hpp).
constexpr double jod_a = 0.0439569391310215;
constexpr double jod_exp = 0.9302042722702026;
constexpr double image_int = 0.577918291091919;

double jod_linear_a() {
    static const double value = jod_a * std::pow(0.1, jod_exp - 1.0);
    return value;
}

// CVVDP's JOD for a pooled quality q.
double jod_from_quality(double q) { return 10.0 - (q > 0.1 ? jod_a * std::pow(q, jod_exp) : jod_linear_a() * q); }

// The pooled quality q that gives a JOD (jod_from_quality's inverse).
double quality_from_jod(double jod) {
    const double gap = std::max(0.0, 10.0 - jod);
    const double linear = gap / jod_linear_a();
    return linear <= 0.1 ? linear : std::pow(gap / jod_a, 1.0 / jod_exp);
}

}  // namespace

void CvvdpDisplay::validate() const {
    const double numbers[] = {width,        height,      diagonal_inches, viewing_distance_m, peak_luminance,
                              contrast,     ambient_lux, reflectivity,    exposure};
    if (!std::all_of(std::begin(numbers), std::end(numbers), [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("Every display value must be a finite number.");
    }
    if (width < 16 || height < 16) {
        throw std::invalid_argument("The display resolution must be at least 16 x 16.");
    }
    const std::pair<const char*, double> positive[] = {{"screen size", diagonal_inches},
                                                       {"viewing distance", viewing_distance_m},
                                                       {"peak brightness", peak_luminance},
                                                       {"contrast", contrast},
                                                       {"exposure", exposure}};
    for (const auto& [label, value] : positive) {
        if (!(value > 0)) {
            throw std::invalid_argument(std::string("The display's ") + label + " must be greater than zero.");
        }
    }
    if (ambient_lux < 0 || !(reflectivity >= 0 && reflectivity < 1)) {
        throw std::invalid_argument("Ambient light cannot be negative and reflectivity must be between 0 and 1.");
    }
}

nlohmann::ordered_json CvvdpDisplay::identity() const {
    return Json{{"resolution", Json::array({whole(width), whole(height)})},
                {"diagonal_inches", rounded(diagonal_inches, 4)},
                {"viewing_distance_m", rounded(viewing_distance_m, 4)},
                {"peak_luminance", rounded(peak_luminance, 4)},
                {"contrast", rounded(contrast, 4)},
                {"ambient_lux", rounded(ambient_lux, 4)},
                {"reflectivity", rounded(reflectivity, 6)},
                {"exposure", rounded(exposure, 4)},
                {"hdr", hdr}};
}

double CvvdpDisplay::distance_in_heights() const {
    const double aspect = width / height;
    const double height_m = diagonal_inches * 0.0254 / std::sqrt(1 + aspect * aspect);
    return viewing_distance_m / height_m;
}

nlohmann::ordered_json CvvdpSettings::parameters() const {
    return Json{{"display", display.identity()}, {"resize_to_display", resize_to_display}};
}

CvvdpSettings CvvdpSettings::from_parameters(const nlohmann::json& parameters) {
    const nlohmann::json& identity = parameters.at("display");
    CvvdpSettings settings;
    CvvdpDisplay& display = settings.display;
    display.width = identity.at("resolution").at(0).get<double>();
    display.height = identity.at("resolution").at(1).get<double>();
    display.diagonal_inches = identity.at("diagonal_inches").get<double>();
    display.viewing_distance_m = identity.at("viewing_distance_m").get<double>();
    display.peak_luminance = identity.at("peak_luminance").get<double>();
    display.contrast = identity.at("contrast").get<double>();
    display.ambient_lux = identity.at("ambient_lux").get<double>();
    display.reflectivity = identity.at("reflectivity").get<double>();
    display.exposure = identity.at("exposure").get<double>();
    display.hdr = identity.at("hdr").get<bool>();
    display.validate();
    settings.resize_to_display = parameters.at("resize_to_display").get<bool>();
    return settings;
}

std::string vship_display_json(const CvvdpDisplay& display) {
    const Json model{{"name", "VideoMetricsLab display"},
                     {"colorspace", display.hdr ? "HDR" : "SDR"},
                     {"resolution", Json::array({whole(display.width), whole(display.height)})},
                     {"viewing_distance_meters", display.viewing_distance_m},
                     {"diagonal_size_inches", display.diagonal_inches},
                     {"max_luminance", display.peak_luminance},
                     {"contrast", display.contrast},
                     {"E_ambient", display.ambient_lux},
                     {"k_refl", display.reflectivity},
                     {"exposure", display.exposure},
                     {"source", "VideoMetricsLab"}};
    return python_json_dumps(Json{{vship_model_key, model}}, 1);
}

const std::vector<CvvdpPreset>& builtin_cvvdp_presets() {
    // The names are identifiers, saved in settings; the window shows them
    // translated, and the descriptions too.
    static const std::vector<CvvdpPreset> presets{
        official("30-inch 4K monitor, office", "Official 'standard_4k' -- the official tool's default display.",
                 3840, 2160, 30, 0.7472, 200, 1000, 250),
        official("24-inch 1080p monitor, office", "Official 'standard_fhd'.", 1920, 1080, 24, 0.6, 200, 1000, 250),
        official("30-inch 4K HDR monitor, dim room", "Official 'standard_hdr_pq' (HLG uses the same display).", 3840,
                 2160, 30, 0.7472, 1500, 1'000'000, 10, true),
        official("30-inch 4K HDR monitor, dark room", "Official 'standard_hdr_dark'.", 3840, 2160, 30, 0.7472, 1500,
                 1'000'000, 0, true),
        official("65-inch 4K HDR TV, 1000 nits, living room", "Official '65inch_hdr_pq_1Knit'.", 3840, 2160, 65,
                 1.98, 1000, 1'000'000, 5, true),
        official("65-inch 4K HDR TV, 2000 nits, living room", "Official '65inch_hdr_pq_2Knit'.", 3840, 2160, 65,
                 1.98, 2000, 1'000'000, 5, true),
        official("65-inch 4K HDR TV, 4000 nits, living room", "Official '65inch_hdr_pq_4knit'.", 3840, 2160, 65,
                 1.98, 4000, 1'000'000, 5, true),
        official("6-inch phone",
                 "Official 'standard_phone' (contrast unspecified there: 1000:1, the official default).", 2400, 1080,
                 6, 0.4, 500, 1000, 250),
    };
    return presets;
}

std::vector<CvvdpPreset> cvvdp_presets(const std::string& user_presets_json) {
    std::vector<CvvdpPreset> found = builtin_cvvdp_presets();
    const nlohmann::json saved = nlohmann::json::parse(user_presets_json, nullptr, false);
    if (!saved.is_array()) {
        return found;
    }
    for (const nlohmann::json& data : saved) {
        if (!data.is_object() || !data.contains("name") || !data.contains("settings")) {
            continue;
        }
        const std::optional<std::string> name = name_of(data.at("name"));
        if (!name) {
            continue;
        }
        try {
            found.push_back({*name, saved_settings(data.at("settings")), false, {}});
        } catch (const std::invalid_argument&) {
            continue;  // skipped, as the Python app skips it
        }
    }
    return found;
}

std::optional<CvvdpPreset> cvvdp_preset_named(const std::string& name, const std::string& user_presets_json) {
    for (CvvdpPreset& preset : cvvdp_presets(user_presets_json)) {
        if (preset.name == name) {
            return std::move(preset);
        }
    }
    return std::nullopt;
}

std::string with_user_preset(const std::string& user_presets_json, const std::string& name,
                             const CvvdpDisplay& display) {
    const std::string kept_name(python_strip(name));
    if (kept_name.empty()) {
        throw std::invalid_argument("A preset needs a name.");
    }
    // Refused, so that a built-in preset always means what it says.
    for (const CvvdpPreset& preset : builtin_cvvdp_presets()) {
        if (preset.name == kept_name) {
            throw std::invalid_argument("\"" + kept_name + "\" is a built-in preset; choose another name.");
        }
    }
    display.validate();
    Json presets = Json::parse(without_user_preset(user_presets_json, kept_name));
    // As the Python app saves a display (asdict): the resolution as the
    // whole numbers its editor gives, the rest as numbers with a fraction.
    const auto resolution = [](double value) {
        return value == std::floor(value) ? Json(static_cast<long long>(value)) : Json(value);
    };
    presets.push_back({{"name", kept_name},
                       {"settings",
                        {{"display",
                          {{"width", resolution(display.width)},
                           {"height", resolution(display.height)},
                           {"diagonal_inches", display.diagonal_inches},
                           {"viewing_distance_m", display.viewing_distance_m},
                           {"peak_luminance", display.peak_luminance},
                           {"contrast", display.contrast},
                           {"ambient_lux", display.ambient_lux},
                           {"reflectivity", display.reflectivity},
                           {"exposure", display.exposure},
                           {"hdr", display.hdr}}},
                         {"resize_to_display", false}}}});
    return presets.dump();
}

std::string without_user_preset(const std::string& user_presets_json, const std::string& name) {
    const Json saved = Json::parse(user_presets_json, nullptr, false);
    Json kept = Json::array();
    if (saved.is_array()) {
        for (const Json& data : saved) {
            const bool named = data.is_object() && data.contains("name") && data.at("name") == name;
            if (!named) {
                kept.push_back(data);
            }
        }
    }
    return kept.dump();
}

std::optional<CvvdpPreset> matching_cvvdp_preset(const CvvdpSettings& settings, const std::string& user_presets_json,
                                                 const std::string& preferred) {
    std::vector<CvvdpPreset> candidates = cvvdp_presets(user_presets_json);
    const Json identity = settings.display.identity();
    const auto matches = [&identity](const CvvdpPreset& preset) {
        return preset.settings.display.identity() == identity;
    };
    if (!preferred.empty()) {
        for (CvvdpPreset& preset : candidates) {
            if (preset.name == preferred && matches(preset)) {
                return std::move(preset);
            }
        }
    }
    for (auto preset = candidates.rbegin(); preset != candidates.rend(); ++preset) {
        if (matches(*preset)) {
            return std::move(*preset);
        }
    }
    return std::nullopt;
}

std::pair<int, int> cvvdp_computed_size(std::pair<int, int> size, const CvvdpSettings& settings) {
    if (!settings.resize_to_display) {
        return size;
    }
    const double scale = std::min(settings.display.width / size.first, settings.display.height / size.second);
    // Python's round(): halves to even, as nearbyint rounds by default.
    const std::pair<int, int> fitted{static_cast<int>(std::nearbyint(size.first * scale)),
                                     static_cast<int>(std::nearbyint(size.second * scale))};
    return static_cast<long long>(fitted.first) * fitted.second > static_cast<long long>(size.first) * size.second
               ? fitted
               : size;
}

double pool_cvvdp_windows(const std::vector<CvvdpWindow>& windows) {
    long long total = 0;
    for (const CvvdpWindow& window : windows) {
        total += window.frames;
    }
    if (total == 1) {
        return windows.front().jod;
    }
    double squares = 0.0;
    for (const CvvdpWindow& window : windows) {
        const double q = quality_from_jod(window.jod);
        squares += window.frames == 1 ? std::pow(q / image_int, 2.0) : static_cast<double>(window.frames) * q * q;
    }
    return jod_from_quality(std::sqrt(squares / static_cast<double>(total)));
}

}  // namespace vml
