#include "vml/av2_probe.h"

#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <string>

#include "dav2d/dav2d_api.h"
#include "vml/av2_stream.h"
#include "vml/colour.h"
#include "vml/ffprobe.h"
#include "vml/film_grain_detect.h"
#include "vml/gpu_frames.h"
#include "vml/isolated.h"
#include "vml/platform/library.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

// ffprobe's names of H.273's colour primaries and transfer characteristics
// (the matrix's: colour.h's ffmpeg_matrix_name).
const std::map<int, std::string_view> primaries_names{
    {1, "bt709"},  {4, "bt470m"},    {5, "bt470bg"},  {6, "smpte170m"}, {7, "smpte240m"}, {8, "film"},
    {9, "bt2020"}, {10, "smpte428"}, {11, "smpte431"}, {12, "smpte432"}, {22, "jedec-p22"}};
const std::map<int, std::string_view> transfer_names{
    {1, "bt709"},          {4, "gamma22"},       {5, "gamma28"},   {6, "smpte170m"},    {7, "smpte240m"},
    {8, "linear"},         {9, "log100"},        {10, "log316"},   {11, "iec61966-2-4"}, {12, "bt1361e"},
    {13, "iec61966-2-1"},  {14, "bt2020-10"},    {15, "bt2020-12"}, {16, "smpte2084"},  {17, "smpte428"},
    {18, "arib-std-b67"}};

std::string name_of(const std::map<int, std::string_view>& names, int code) {
    const auto found = names.find(code);
    return found == names.end() ? std::string() : std::string(found->second);
}

// dav2d's functions, as the probe calls them.
struct Dav2dCalls {
    unsigned (*version_api)() = nullptr;
    void (*default_settings)(Dav2dSettings*) = nullptr;
    int (*open)(void**, const Dav2dSettings*) = nullptr;
    int (*send_data)(void*, Dav2dData*) = nullptr;
    int (*get_picture)(void*, Dav2dPicture*) = nullptr;
    void (*close)(void**) = nullptr;
    uint8_t* (*data_create)(Dav2dData*, size_t) = nullptr;
    void (*data_unref)(Dav2dData*) = nullptr;
    void (*picture_unref)(Dav2dPicture*) = nullptr;

    explicit Dav2dCalls(const platform::DynamicLibrary& library)
        : version_api(library.function<unsigned (*)()>("dav2d_version_api")),
          default_settings(library.function<void (*)(Dav2dSettings*)>("dav2d_default_settings")),
          open(library.function<int (*)(void**, const Dav2dSettings*)>("dav2d_open")),
          send_data(library.function<int (*)(void*, Dav2dData*)>("dav2d_send_data")),
          get_picture(library.function<int (*)(void*, Dav2dPicture*)>("dav2d_get_picture")),
          close(library.function<void (*)(void**)>("dav2d_close")),
          data_create(library.function<uint8_t* (*)(Dav2dData*, size_t)>("dav2d_data_create")),
          data_unref(library.function<void (*)(Dav2dData*)>("dav2d_data_unref")),
          picture_unref(library.function<void (*)(Dav2dPicture*)>("dav2d_picture_unref")) {}
};

// The first picture dav2d decodes from the stream's first units: its size,
// depth, layout, colour and whether it carries a film grain model.
nlohmann::json first_picture(const std::filesystem::path& path) {
    const platform::DynamicLibrary library(dav2d_library_path());
    const Dav2dCalls dav2d(library);
    if (dav2d.version_api() >> 16 != kDav2dApiMajor) {
        throw std::runtime_error("dav2d is not the version the app was made for");
    }
    Dav2dSettings settings{};
    dav2d.default_settings(&settings);
    settings.n_threads = 1;
    settings.max_frame_delay = 1;
    settings.all_layers = 0;
    settings.logger_callback = nullptr;
    av2::Reader reader(path);  // before dav2d starts: it throws for a file it cannot read
    void* context = nullptr;
    if (dav2d.open(&context, &settings) < 0) {
        throw std::runtime_error("dav2d could not start");
    }
    Dav2dPicture picture{};
    bool got = false;
    // A picture comes out within a few units; a stream that gives none in 64
    // is not decoded.
    for (int units = 0; units < 64 && !got; units++) {
        std::optional<av2::TemporalUnit> unit = reader.next();
        Dav2dData data{};
        if (unit) {
            std::uint8_t* buffer = dav2d.data_create(&data, unit->data.size());
            if (buffer == nullptr) {
                break;
            }
            std::copy(unit->data.begin(), unit->data.end(), buffer);
        }
        if (!unit) {
            // The end: dav2d gives the pictures it holds once told so.
            dav2d.send_data(context, nullptr);
            got = dav2d.get_picture(context, &picture) == 0;
            break;
        }
        do {
            const int sent = dav2d.send_data(context, &data);
            const int result = dav2d.get_picture(context, &picture);
            if (result == 0) {
                got = true;
                break;
            }
            if ((sent < 0 && sent != kDav2dAgain) || (result < 0 && result != kDav2dAgain)) {
                break;
            }
        } while (data.sz > 0);
        if (data.sz > 0) {
            dav2d.data_unref(&data);
        }
    }
    nlohmann::json result;
    if (got) {
        result = {{"width", picture.p.w},
                  {"height", picture.p.h},
                  {"bit_depth", picture.p.bpc},
                  {"layout", picture.p.layout},
                  {"film_grain", dav2d_adds_grain(picture.film_grain)}};
        if (picture.content != nullptr && picture.content->color_description_present) {
            int codes[3];
            dav2d_colour_codes(*picture.content, codes);
            result["colour"] = {codes[0], codes[1], codes[2], picture.content->color.range};
        }
        dav2d.picture_unref(&picture);
    }
    dav2d.close(&context);
    if (!got) {
        throw std::runtime_error("dav2d decoded no picture of it");
    }
    return result;
}

}  // namespace

bool is_av2_file(const std::filesystem::path& path) { return av2::container_of(path).has_value(); }

nlohmann::json av2_probe_task(const nlohmann::json& request) {
    const std::filesystem::path path = platform::path_from_text(request.at("path").get<std::string>());
    av2::Reader reader(path, av2::Payloads::skipped);
    nlohmann::json result = first_picture(path);
    // Each unit decodes one picture, shown once. An IVF frame's or a Matroska
    // block's time is that of the first picture shown from it on, which is
    // its first unit's own where no picture decoded before is still waiting
    // to be shown: always at a closed-loop keyframe, and at the end of a
    // stream as avmenc ends it.
    // So many pictures after the first frame, in so many ticks, give the
    // rate: up to the last keyframe after the first, else up to the last frame.
    long long units = 0, first_pts = 0, last_start = 0, last_pts = 0, key_start = 0, key_pts = 0;
    while (std::optional<av2::TemporalUnit> unit = reader.next()) {
        if (unit->index == 0) {
            if (units == 0) {
                first_pts = unit->container_pts;
            }
            last_start = unit->number;
            last_pts = unit->container_pts;
            if (unit->kind.closed_key && unit->number > 0) {
                key_start = unit->number;
                key_pts = unit->container_pts;
            }
        }
        units++;
    }
    const Rational base = reader.time_base();
    result["container"] = reader.container() == av2::Container::ivf        ? "ivf"
                          : reader.container() == av2::Container::matroska ? "matroska"
                                                                           : "obu";
    result["frames"] = units;
    result["time_base"] = {base.numerator, base.denominator};
    result["rate_pictures"] = key_start > 0 ? key_start : last_start;
    result["rate_ticks"] = (key_start > 0 ? key_pts : last_pts) - first_pts;
    return result;
}

VideoInfo av2_info_from(const std::filesystem::path& path, const nlohmann::json& result) {
    VideoInfo info;
    info.path = path;
    info.codec_name = "av2";
    info.format_name = result.at("container").get<std::string>();
    info.width = result.at("width").get<int>();
    info.height = result.at("height").get<int>();
    const int depth = result.at("bit_depth").get<int>();
    const int layout = result.at("layout").get<int>();
    const std::string format = layout == kDav2dI400 ? "gray" : layout == kDav2dI422 ? "yuv422p"
                                                       : layout == kDav2dI444 ? "yuv444p"
                                                                              : "yuv420p";
    info.pix_fmt = depth > 8 ? format + std::to_string(depth) + "le" : format;
    info.nb_frames = result.at("frames").get<long long>();
    info.film_grain = result.at("film_grain").get<bool>() ? film_grain_av2 : 0u;
    if (const auto colour = result.find("colour"); colour != result.end()) {
        info.color_primaries = name_of(primaries_names, colour->at(0).get<int>());
        info.color_transfer = name_of(transfer_names, colour->at(1).get<int>());
        info.color_space = std::string(ffmpeg_matrix_name(colour->at(2).get<int>()).value_or(""));
        info.color_range = colour->at(3).get<int>() ? "pc" : "tv";
    }
    // An IVF or Matroska file is timed: so many pictures in so many ticks of
    // its time base. One picture alone says nothing: a tick each.
    if (info.format_name != "obu") {
        const Rational base{result.at("time_base").at(0).get<long long>(),
                            result.at("time_base").at(1).get<long long>()};
        const long long pictures = result.at("rate_pictures").get<long long>();
        const long long ticks = result.at("rate_ticks").get<long long>();
        const double seconds = static_cast<double>(ticks > 0 ? ticks : 1) * static_cast<double>(base.numerator) /
                               static_cast<double>(base.denominator);
        info.fps = static_cast<double>(pictures > 0 ? pictures : 1) / seconds;
        info.nominal_fps = info.average_fps = info.fps;
        info.duration = static_cast<double>(info.nb_frames) / info.fps;
    }
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (!error && info.duration > 0) {
        info.bit_rate = static_cast<long long>(std::llround(static_cast<double>(bytes) * 8.0 / info.duration));
        info.bit_rate_whole_file = true;
    }
    return info;
}

VideoInfo probe_av2(const std::filesystem::path& path, const CancelToken& cancel) {
    const nlohmann::json request{{"path", platform::path_text(path)}};
    nlohmann::json result;
    try {
        result = isolation_program() ? run_isolated("av2-probe", "dav2d", request, {}, cancel)
                                     : av2_probe_task(request);  // the tests: no program to run it in
    } catch (const Cancelled&) {
        throw;
    } catch (const std::exception& error) {
        throw ProbeError("Could not read the AV2 video " + platform::path_text(path.filename()) + ": " + error.what());
    }
    return av2_info_from(path, result);
}

}  // namespace vml
