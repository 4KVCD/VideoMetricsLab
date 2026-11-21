#include "vml/av2_probe.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "av2/splot_api.h"
#include "vml/av2_stream.h"
#include "vml/colour.h"
#include "vml/ffprobe.h"
#include "vml/film_grain_detect.h"
#include "vml/gpu_frames.h"
#include "vml/isolated.h"
#include "vml/platform/system.h"
#include "vml/splot_library.h"

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

// The first picture splot decodes from the stream's first units: its size,
// depth, layout, colour and whether it carries a film grain model.
nlohmann::json first_picture(const std::filesystem::path& path) {
    // Missing (not built for this system), it is named as the software
    // decoder's refusal names it, not by the loader's error and paths. Looked
    // for at each probe: the library is loaded once, where it was first found.
    if (std::error_code ignored; !std::filesystem::is_regular_file(splot_library_path(), ignored)) {
        throw std::runtime_error("splot is not bundled");
    }
    const SplotLibrary& splot = splot_library();
    if (!splot.library) {
        throw std::runtime_error(splot.problem);
    }
    const SplotCalls& api = splot.calls;
    // One thread, one picture in flight: the first picture is all it decodes.
    SplotSettings settings = splot_settings(api, true);
    settings.n_threads = 1;
    settings.max_frame_delay = 1;
    av2::Reader reader(path);  // before splot starts: it throws for a file it cannot read
    void* context = nullptr;
    if (api.open(&context, &settings) != kSplotOk) {
        throw std::runtime_error("splot could not start");
    }
    SplotPicture picture{};
    int result = kSplotAgain;
    // A picture comes out within a few units; a stream that gives none in 64
    // is not decoded.
    for (int units = 0; units < 64 && result == kSplotAgain; units++) {
        const std::optional<av2::TemporalUnit> unit = reader.next();
        if (!unit) {
            // The end: splot gives the pictures it holds, waiting for them.
            api.drain(context);
            result = api.get_picture(context, &picture);
            break;
        }
        const SplotDataProps props{unit->number, 0, 0};
        int sent = api.send_data(context, unit->data.data(), unit->data.size(), &props);
        if (sent == kSplotAgain) {
            // Its queue is full: it waits for a picture, then takes the unit.
            result = api.get_picture(context, &picture);
            if (result != kSplotAgain) {
                break;
            }
            sent = api.send_data(context, unit->data.data(), unit->data.size(), &props);
        }
        if (sent != kSplotOk) {
            result = sent;
            break;
        }
        result = api.get_picture(context, &picture);  // EAGAIN at once where it has none yet
    }
    nlohmann::json json;
    const bool got = result == kSplotOk;
    if (got) {
        json = {{"width", picture.width},
                {"height", picture.height},
                {"bit_depth", picture.bit_depth},
                {"layout", picture.layout},
                {"film_grain", picture.has_film_grain != 0}};
        if (splot_says_colour(picture.color)) {
            json["colour"] = {picture.color.primaries, picture.color.transfer, picture.color.matrix,
                              picture.color.full_range};
        }
        api.picture_unref(&picture);
    }
    api.close(&context);
    if (!got) {
        throw std::runtime_error("splot decoded no picture of it");
    }
    return json;
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
    const std::string format = layout == kSplotI400 ? "gray" : layout == kSplotI422 ? "yuv422p"
                                                       : layout == kSplotI444 ? "yuv444p"
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
        result = isolation_program() ? run_isolated("av2-probe", "splot", request, {}, cancel)
                                     : av2_probe_task(request);  // the tests: no program to run it in
    } catch (const Cancelled&) {
        throw;
    } catch (const std::exception& error) {
        throw ProbeError("Could not read the AV2 video " + platform::path_text(path.filename()) + ": " + error.what());
    }
    return av2_info_from(path, result);
}

}  // namespace vml
