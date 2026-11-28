#include "vml/metric_cache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <future>
#include <utility>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <tuple>
#include <variant>

#include "metric_cache_parts.h"
#include "vml/app_log.h"
#include "vml/append.h"
#include "vml/bitrate_store.h"
#include "vml/build_info.h"
#include "vml/cancel.h"
#include "vml/colour.h"
#include "vml/crop_detect.h"
#include "vml/file_fingerprint.h"
#include "vml/film_grain_detect.h"
#include "vml/film_grain_scoring.h"
#include "vml/geometry.h"
#include "vml/hardware_decoding.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/npz.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/temp_files.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/result_json.h"
#include "vml/saved_index.h"
#include "vml/score_identity.h"
#include "vml/settings.h"
#include "vml/sha256.h"
#include "vml/vship.h"

namespace vml {

namespace {

using namespace cache_parts;

// The id of a score saved before scores recorded their provenance.
constexpr const char* unversioned_compatibility = "unversioned";

// What a score's metadata carries when its pair has a video of a codec the
// GPUs decode to other pictures than FFmpeg's (gpu_decodes_exactly), which
// is decoded in software now: the version of that list. Such a pair's scores
// saved without it, or with an older one, may have been made from a GPU's
// pictures, and are calculated again. Raise it when a codec joins the list.
constexpr const char* exact_decoding = "exact_decoding";
constexpr int exact_decoding_version = 1;

// What a score's metadata carries when its pair has H.264 in a lossless 4:4:4
// profile, or of no known profile (hardware_decoding.h): such a video was
// decoded to other pictures by Intel's GPUs before 550f10e, and by NVIDIA's
// before the RTX 50 series before 9f983c8, which a score does not record.
// Such a pair's scores saved without it are calculated again.
constexpr const char* exact_lossless_h264 = "exact_lossless_h264";
constexpr int exact_lossless_h264_version = 1;

// What a score's metadata carries when its pair has a video scored with its
// AV1 film grain, added as dav1d adds it, or an AV1 video whose grain is not
// known, decoded as dav1d decodes it: every route on every system does so
// from 2.3 on, and the score shares Windows' identity (film_grain_identity).
// A Mac's score made before may hold VideoToolbox's grain, which no score
// records: outside Windows, such a pair's scores count only with it. Windows writes it and asks for
// nothing (every route there added it alike). Raise it when a decoder that
// adds AV1's grain otherwise joins.
constexpr const char* av1_grain_dav1d = "av1_grain_dav1d";
constexpr int av1_grain_dav1d_version = 1;

// The marks a pair's saved scores carry, and must carry to be taken.
struct DecodingMarks {
    bool inexact_codec = false;  // exact_decoding
    bool lossless_h264 = false;  // exact_lossless_h264
    bool av1_grain = false;      // av1_grain_dav1d: scored with its AV1 grain
    bool av1_grain_unknown = false;  // an AV1 video whose grain is not known
};

// The two rules of the av1_grain_dav1d mark, for a video as a run reads it
// and as a saved comparison's context.json gives it (grain_facts).
bool adds_av1_grain(const VideoInfo& info) {
    const FilmGrainScoring grain = effective_film_grain(info);
    return grain.scoring == GrainScoring::with && (grain.kinds & film_grain_av1) != 0;
}

bool av1_grain_unknown(const VideoInfo& info) { return ascii_lower(info.codec_name) == "av1" && !info.film_grain; }

void add_grain_marks(DecodingMarks& marks, const VideoInfo& source, const VideoInfo& distorted) {
    marks.av1_grain = adds_av1_grain(source) || adds_av1_grain(distorted);
    marks.av1_grain_unknown = av1_grain_unknown(source) || av1_grain_unknown(distorted);
}

// What the mark's rules read of a video in a context.json: its codec and its
// film grain, as video_info_to_dict wrote them. Grain in another shape is not
// known; a video with no codec named may be an AV1 one.
VideoInfo grain_facts(const nlohmann::json& info) {
    VideoInfo video;
    video.codec_name = "av1";
    if (!info.is_object()) {
        return video;
    }
    if (const nlohmann::json codec = info.value("codec_name", nlohmann::json()); codec.is_string()) {
        video.codec_name = codec.get<std::string>();
    }
    if (const nlohmann::json names = info.value("film_grain", nlohmann::json()); names.is_array() &&
        std::all_of(names.begin(), names.end(), [](const nlohmann::json& name) { return name.is_string(); })) {
        video.film_grain = film_grain_from_names(names.get<std::vector<std::string>>());
    }
    if (const nlohmann::json with = info.value("with_film_grain", nlohmann::json()); with.is_boolean()) {
        video.with_film_grain = with.get<bool>();
    }
    return video;
}

DecodingMarks decoding_marks(const std::string& source_codec, const std::string& source_profile,
                             const std::string& distorted_codec, const std::string& distorted_profile) {
    const auto lossless_h264 = [](const std::string& codec, const std::string& profile) {
        VideoInfo video;
        video.codec_name = codec;
        video.profile = profile;
        return h264_444_profile(video) || h264_profile_unknown(video);
    };
    return {!gpu_decodes_exactly(source_codec) || !gpu_decodes_exactly(distorted_codec),
            lossless_h264(source_codec, source_profile) || lossless_h264(distorted_codec, distorted_profile)};
}

// The RGB pixel formats Vship scored wrongly without a transfer tag before
// app version 1.2.2, and the transfer tags that count as none.
const std::regex& old_vship_rgb() {
    static const std::regex pattern(
        "gbrp(?:(?:9|10|12|14|16)(?:le|be)?)?|rgb24|bgr24|rgba|bgra|argb|abgr|rgb0|bgr0|rgb48(?:le|be)|rgba64(?:le|be)");
    return pattern;
}

std::string sha256_hex(const std::string& text) {
    Sha256 hash;
    hash.update(text.data(), text.size());
    return hash.hexdigest();
}

// A provenance read back (provenance_from_dict); nothing for one of another
// shape.
std::optional<MetricProvenance> provenance_from(const nlohmann::json& data) {
    if (!data.is_object()) {
        return std::nullopt;
    }
    try {
        return provenance_from_dict(data);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

// A video as the rules read it from its tags.
VideoInfo tagged_video(const ColourTags& tags) {
    VideoInfo info;
    info.pix_fmt = tags.pix_fmt;
    info.color_transfer = tags.color_transfer;
    info.color_primaries = tags.color_primaries;
    info.color_space = tags.color_space;
    info.height = tags.height;
    return info;
}

// Whether a pair's video is now read with the other's colours, as a score of
// the mapping before colours_lent was not; not where the tags are not known.
bool colours_read_otherwise(const std::vector<ColourTags>& infos) {
    return infos.size() == 2 && colours_lent(tagged_video(infos[0]), tagged_video(infos[1]));
}

// Whether the pair's videos are read with another colour matrix now than
// each video's own tag or guess gave (matrix_shared in colour.h); not where
// the two videos' tags are not known.
bool matrix_read_otherwise(const std::vector<ColourTags>& infos) {
    if (infos.size() != 2) {
        return false;
    }
    try {
        return matrix_shared(tagged_video(infos[0]), tagged_video(infos[1]));
    } catch (const std::exception&) {
        return false;
    }
}

// Whether a CPU SSIMULACRA2 or Butteraugli score predates the colour
// handling the CPU's tools have now: one made before 8-bit videos were
// converted to RGB precisely, for a pair with an 8-bit video (or a format
// not known: FFmpeg's 8-bit 4:2:0 and 4:2:2 ones changed); one made before
// the pair was read with one colour matrix only where that changes what is
// read.
bool stale_cpu_score(const MetricProvenance& provenance, const std::vector<ColourTags>& infos) {
    if (provenance.compute_backend != "cpu" ||
        (provenance.implementation != "ssimulacra2" && provenance.implementation != "butteraugli")) {
        return false;
    }
    const nlohmann::json mapping = parameter(provenance, "color_tags");
    if (mapping == nlohmann::json(score_identity::cpu_colour_tags)) {
        return false;
    }
    if (colours_read_otherwise(infos)) {
        return true;
    }
    if (mapping == nlohmann::json(score_identity::cpu_colour_tags_unlent)) {
        return false;
    }
    const bool eight_bit = infos.size() != 2 || std::any_of(infos.begin(), infos.end(), [](const ColourTags& info) {
                               return bit_depth(info.pix_fmt) <= 8;
                           });
    if (eight_bit) {
        return true;
    }
    if (mapping == nlohmann::json(score_identity::cpu_colour_tags_8_bit_rgb)) {
        return false;
    }
    return mapping != nlohmann::json(score_identity::cpu_colour_tags_unshared_matrix) || matrix_read_otherwise(infos);
}

// Whether a video is gray tagged full range, which Vship 5.1 was sent
// squeezed into the limited range.
bool full_range_gray(const ColourTags& info) {
    static const std::regex gray("gray(?:(?:9|10|12|14|16)(?:le|be)?)?");
    const std::string range = ascii_lower(python_strip(info.color_range));
    return std::regex_match(ascii_lower(python_strip(info.pix_fmt)), gray) &&
           is_full_range_name(range);
}

// Whether a saved XPSNR is FFmpeg's xpsnr filter's of a pair compared at more
// than 8 bits, which it scores wrongly where FFmpeg pads a picture's rows
// (ffmpeg_scores_xpsnr in ffmpeg_run.cpp); which of them it padded is not
// known, so none is used.
bool unsure_ffmpeg_xpsnr(const MetricProvenance& provenance, const std::vector<ColourTags>& infos) {
    if (provenance.implementation != "ffmpeg/xpsnr" || infos.empty()) {
        return false;
    }
    std::vector<std::string> pix_fmts;
    for (const ColourTags& info : infos) {
        pix_fmts.push_back(info.pix_fmt);
    }
    return bit_depth(analysis_pix_fmt(pix_fmts)) > 8;
}

// Whether a saved score is known to be wrong and must be calculated again:
// FFmpeg's XPSNR above 8 bits (unsure_ffmpeg_xpsnr), a stale CPU score, or a
// Vship 5.1 GPU score of a pair a later fix reads otherwise, for a score
// that predates the fix. That is a pair with a full-range gray video
// (full_range_gray) for a score of a mapping before "/4", which came after
// that fix (the mappings before cannot tell); one whose videos are read with
// one colour matrix where that changes what Vship reads
// (matrix_read_otherwise), one with a video without chroma siting read with
// the other's where that changes what it reads (chroma_location_shared), or,
// from app version 1.2 or 1.2.1, one with an RGB video without a transfer
// tag. Vship-fast's scores are stale only for a pair with a video now read
// with the other's colours.
bool stale_score(const MetricProvenance& provenance, const std::vector<ColourTags>& infos) {
    if (unsure_ffmpeg_xpsnr(provenance, infos) || stale_cpu_score(provenance, infos)) {
        return true;
    }
    if (provenance.compute_backend == "gpu" &&
        score_identity::is_vship_fast_id(provenance.implementation_compatibility_id)) {
        return parameter(provenance, "color_tags") != nlohmann::json(score_identity::vship_colour_tags) &&
               colours_read_otherwise(infos);
    }
    if (provenance.compute_backend != "gpu" ||
        !score_identity::is_vship51_id(provenance.implementation_compatibility_id)) {
        return false;
    }
    const nlohmann::json mapping = parameter(provenance, "color_tags");
    if (mapping == nlohmann::json(score_identity::vship_colour_tags)) {
        return false;
    }
    if (std::any_of(infos.begin(), infos.end(), full_range_gray)) {
        return true;
    }
    if (colours_read_otherwise(infos)) {
        return true;
    }
    if (mapping == nlohmann::json(score_identity::vship_colour_tags_unlent)) {
        return false;
    }
    if (matrix_read_otherwise(infos)) {
        return true;
    }
    if (mapping == nlohmann::json(score_identity::vship_colour_tags_unshared_matrix)) {
        return false;
    }
    if (infos.size() == 2 && chroma_location_shared(infos[0].chroma_location, infos[1].chroma_location)) {
        return true;
    }
    if (mapping == nlohmann::json(score_identity::vship_colour_tags_unshared_siting)) {
        return false;
    }
    return std::any_of(infos.begin(), infos.end(), [](const ColourTags& info) {
        const std::string transfer = ascii_lower(python_strip(info.color_transfer));
        return std::regex_match(ascii_lower(python_strip(info.pix_fmt)), old_vship_rgb()) &&
               is_untagged_tag(transfer);
    });
}

// Whether a saved perceptual score's compatibility id answers such a
// request: the request's Vship build's on the GPU, or libjxl's on the CPU,
// whose id may have a version inside.
bool auto_perceptual_compatible(const MetricRequestSpec& spec, const std::string& value) {
    const std::string& key = spec.key;
    const bool vship51 = spec.implementation_compatibility_id.ends_with(score_identity::vship51_gpu_or_cpu_suffix);
    return (vship51 ? score_identity::is_vship51_gpu(key, value) : value == score_identity::vship_gpu(key)) ||
           value == score_identity::libjxl_cpu(key) ||
           (value.starts_with(key + "-libjxl-") && value.ends_with("-cpu-v1"));
}

bool compatible(const MetricRequestSpec& spec, const MetricProvenance& provenance) {
    const std::string& compatibility = provenance.implementation_compatibility_id;
    if (is_auto_perceptual(spec)) {
        return auto_perceptual_compatible(spec, compatibility);
    }
    if (compatibility == unversioned_compatibility) {
        return spec.implementation_compatibility_id == score_identity::libvmaf;
    }
    return compatibility == spec.implementation_compatibility_id;
}

// Writes `bytes` to `path` through a temporary file that is then renamed
// over it, so that a crash never leaves half a file.
void write_atomically(const std::filesystem::path& path, const std::string& bytes) {
    std::filesystem::create_directories(path.parent_path());
    write_file_whole(path, [&bytes](std::ostream& file) {
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    });
}

// The metadata saved with a score: the cache format, the kind of score, the
// request it answers and the score's provenance (with the version that made
// it, which the results name; no lookup reads it), and the pair's decoding
// marks: the AV1 grain mark as the score itself says where it was kept from
// an earlier result (av1_grain_dav1d), as the pair says for one made now.
std::string metadata(const std::string& key, const char* kind, const MetricRequestSpec& spec,
                     const MetricProvenance& provenance, const DecodingMarks& marks,
                     const std::optional<bool>& av1_grain_mark) {
    nlohmann::json data{{"format_version", metric_cache_format_version},
                        {"kind", kind},
                        {"key", key},
                        {"request", spec.identity()},
                        {"provenance", provenance_to_dict(provenance)}};
    if (marks.inexact_codec) {
        data[exact_decoding] = exact_decoding_version;
    }
    if (marks.lossless_h264) {
        data[exact_lossless_h264] = exact_lossless_h264_version;
    }
    // A score this run made: as its pair; one it kept: as it was saved.
    if (av1_grain_mark.value_or(marks.av1_grain || marks.av1_grain_unknown)) {
        data[av1_grain_dav1d] = av1_grain_dav1d_version;
    }
    return python_json_compact(data);
}

// The request whose file holds `spec`'s scores: a YUV metric's Y plane is
// saved as the plain metric it equals (saved_as), one file for both.
MetricRequestSpec saved_spec(const MetricRequestSpec& spec) {
    MetricRequestSpec saved = spec;
    saved.key = std::string(saved_as(spec.key));
    return saved;
}

// The file named after `spec` itself, where a YUV metric's Y plane was
// saved before it was filed as its plain metric (metric_path): still read.
std::filesystem::path own_metric_path(const std::filesystem::path& directory, const MetricRequestSpec& spec) {
    return directory / platform::path_from_text(spec.key + "_" + metric_identity_hash(spec) + ".npz");
}

// The request a perceptual score is saved under: the implementation that
// made it, so that a CPU score never passes for a Vship GPU one, nor one
// Vship build's for the other's. A score the run's other build made (Vship
// 5.1.2 standing in for Vship-fast) is so kept for that build's requests.
MetricRequestSpec stored_spec(const MetricRequestSpec& spec, const MetricProvenance& provenance) {
    MetricRequestSpec stored = saved_spec(spec);
    if (spec.backend_id == "perceptual") {
        stored.implementation_compatibility_id = provenance.implementation_compatibility_id;
    }
    return stored;
}

// Whether a score of `key` is saved with its scores as calculated (the
// "exact" array): PSNR's and SSIM's, the Y that PSNR YUV and SSIM YUV take
// from the save to calculate only their U and V (JobRun's make_611).
bool saves_exact(const std::string& key) { return key == "psnr" || key == "ssim"; }

void store_metric(const std::filesystem::path& directory, const FrameMetricResult& result,
                  const MetricRequestSpec& spec, const DecodingMarks& marks) {
    const MetricRequestSpec stored = stored_spec(spec, result.provenance);
    const std::string described =
        metadata(stored.key, "frame", stored, result.provenance, marks, result.av1_grain_dav1d);
    std::vector<std::pair<std::string, NpyArray>> arrays{{"metadata", npy_text(described)},
                                                         {"frame", npy_int32(result.frame)},
                                                         {"time", npy_float64(result.time)},
                                                         {"values", npy_float32(result.values)}};
    if (saves_exact(stored.key) && !result.exact.empty() && result.exact.size() == result.values.size()) {
        arrays.emplace_back("exact", npy_float64(result.exact));
    }
    write_atomically(metric_path(directory, stored), npz_bytes(arrays));
}

void store_metric(const std::filesystem::path& directory, const SequenceMetricResult& result,
                  const MetricRequestSpec& spec, const DecodingMarks& marks) {
    const MetricRequestSpec stored = stored_spec(spec, result.provenance);
    std::vector<std::pair<std::string, NpyArray>> arrays{
        {"metadata",
         npy_text(metadata(stored.key, "sequence", stored, result.provenance, marks, result.av1_grain_dav1d))},
        {"score", npy_float64_scalar(result.score)}};
    if (result.has_timeline()) {
        arrays.emplace_back("frame", npy_int32(result.frame));
        arrays.emplace_back("time", npy_float64(result.time));
        arrays.emplace_back("values", npy_float32(result.values));
    }
    write_atomically(metric_path(directory, stored), npz_bytes(arrays));
}

nlohmann::json optional_text(const std::optional<std::string>& text) {
    return text ? nlohmann::json(*text) : nlohmann::json(nullptr);
}

// The comparison's context.json: its files, crops, frame rate and recipe.
// With frames_short (frame_pairing.h) only where the result has one, which
// the Python app did not write.
nlohmann::json context_json(const ComparisonResult& result, const std::string& label, const ComparisonRecipe& recipe) {
    nlohmann::json context = {{"format_version", metric_cache_format_version},
            {"label", label},
            {"source", python_path_text(result.source)},
            {"distorted", python_path_text(result.distorted)},
            {"fps", result.fps},
            {"source_crop", nlohmann::json(crop_to_dict(result.source_crop))},
            {"distorted_crop", nlohmann::json(crop_to_dict(result.distorted_crop))},
            {"source_info", nlohmann::json(video_info_to_dict(result.source_info))},
            {"distorted_info", nlohmann::json(video_info_to_dict(result.distorted_info))},
            {"scale_direction", scale_direction_value(result.scale_direction)},
            {"scale_algorithm", result.scale_algorithm},
            {"resample_target", nlohmann::json(resample_target_to_dict(result.resample_target))},
            {"compared_frame_count", result.compared_frame_count},
            {"model", result.model},
            {"model_choice", optional_text(result.model_choice)},
            {"model_v1", result.model_v1},
            {"model_choice_v1", optional_text(result.model_choice_v1)},
            {"recipe", recipe.identity()},
            {"black_bars_rule", black_bars_rule_version},
            {"saved_by", build_stamp()}};
    if (result.model_choice_hdr) {
        // Only with HDR VMAF, so every other comparison's context stays as it was.
        context["model_hdr"] = result.model_hdr;
        context["model_choice_hdr"] = *result.model_choice_hdr;
    }
    if (result.frames_short) {
        context["frames_short"] = frames_short_json(result.frames_short);
    }
    return context;
}

// The two videos' colour tags as a comparison's context.json has them.
std::vector<ColourTags> context_colour_tags(const nlohmann::json& context) {
    std::vector<ColourTags> infos;
    if (!context.is_object()) {
        return infos;
    }
    for (const char* key : {"source_info", "distorted_info"}) {
        const nlohmann::json info = context.value(key, nlohmann::json());
        if (info.is_object()) {
            const auto text = [&info](const char* name) {
                const nlohmann::json value = info.value(name, nlohmann::json());
                return value.is_null() ? std::string() : python_str(value);
            };
            const nlohmann::json height = info.value("height", nlohmann::json());
            infos.push_back({text("pix_fmt"), text("color_transfer"), text("chroma_location"), text("color_space"),
                             text("color_range"), height.is_number_integer() ? height.get<int>() : 0,
                             text("color_primaries")});
        }
    }
    return infos;
}

// The decoding marks of the pair a comparison's context.json gives, whose
// saved scores count only with them. A context saved before 2.3 has no
// profile: its H.264 videos count as maybe lossless.
DecodingMarks decoding_marks(const nlohmann::json& context) {
    if (!context.is_object()) {
        DecodingMarks marks;
        marks.av1_grain_unknown = true;  // nothing says the videos are not AV1 ones
        return marks;
    }
    const auto field = [&context](const char* video, const char* name) {
        const nlohmann::json info = context.value(video, nlohmann::json());
        const nlohmann::json value = info.is_object() ? info.value(name, nlohmann::json()) : nlohmann::json();
        return value.is_string() ? value.get<std::string>() : std::string();
    };
    const std::string source_codec = field("source_info", "codec_name");
    const std::string distorted_codec = field("distorted_info", "codec_name");
    DecodingMarks marks = decoding_marks(source_codec, field("source_info", "profile"), distorted_codec,
                                         field("distorted_info", "profile"));
    // A video with no codec named is no video of either kind (as before).
    marks.inexact_codec = (!source_codec.empty() && !gpu_decodes_exactly(source_codec)) ||
                          (!distorted_codec.empty() && !gpu_decodes_exactly(distorted_codec));
    add_grain_marks(marks, grain_facts(context.value("source_info", nlohmann::json())),
                    grain_facts(context.value("distorted_info", nlohmann::json())));
    return marks;
}

// Whether a saved score was made from the pictures its pair is decoded to
// now: one with each of the pair's decoding marks.
bool decoded_as_now(const nlohmann::json& metadata, const DecodingMarks& marks) {
#ifndef _WIN32
    if ((marks.av1_grain || marks.av1_grain_unknown) &&
        metadata.value(av1_grain_dav1d, nlohmann::json()) != av1_grain_dav1d_version) {
        return false;
    }
#endif
    return (!marks.inexact_codec || metadata.value(exact_decoding, nlohmann::json()) == exact_decoding_version) &&
           (!marks.lossless_h264 ||
            metadata.value(exact_lossless_h264, nlohmann::json()) == exact_lossless_h264_version);
}

// A saved score's metadata, or nothing if the file cannot be read.
std::optional<std::pair<nlohmann::json, std::map<std::string, NpyArray>>> read_score(const std::filesystem::path& path) {
    std::optional<std::map<std::string, NpyArray>> arrays = read_npz(path);
    if (!arrays || !arrays->contains("metadata")) {
        return std::nullopt;
    }
    const std::optional<std::string> text = npy_text_value(arrays->at("metadata"));
    if (!text) {
        return std::nullopt;
    }
    nlohmann::json metadata = nlohmann::json::parse(python_json_readable(*text), nullptr, false);
    if (!metadata.is_object()) {
        return std::nullopt;
    }
    return std::pair{std::move(metadata), std::move(*arrays)};
}

// `object` without the keys `names`.
nlohmann::json without(nlohmann::json object, std::initializer_list<const char*> names) {
    if (object.is_object()) {
        for (const char* name : names) {
            object.erase(name);
        }
    }
    return object;
}

// A frame metric with only every `step`-th frame (sampled_every): the
// frames a subsampled request covers.
FrameMetricResult sampled_every(FrameMetricResult result, int step) {
    if (step <= 1 || std::none_of(result.frame.begin(), result.frame.end(), [step](int f) { return f % step != 0; })) {
        return result;
    }
    FrameMetricResult kept{result.key, {}, {}, {}, result.provenance, result.av1_grain_dav1d};
    const bool exact = result.exact.size() == result.frame.size();
    for (std::size_t i = 0; i < result.frame.size(); ++i) {
        if (result.frame[i] % step == 0) {
            kept.frame.push_back(result.frame[i]);
            kept.time.push_back(result.time[i]);
            kept.values.push_back(result.values[i]);
            if (exact) {
                kept.exact.push_back(result.exact[i]);
            }
        }
    }
    return kept;
}

// A saved timeline's arrays, or nothing if one is missing or their lengths
// differ.
std::optional<std::tuple<std::vector<int>, std::vector<double>, std::vector<float>>> timeline(
    const std::map<std::string, NpyArray>& arrays) {
    if (!arrays.contains("frame") || !arrays.contains("time") || !arrays.contains("values")) {
        return std::nullopt;
    }
    const auto frames = npy_numbers(arrays.at("frame"));
    const auto times = npy_numbers(arrays.at("time"));
    const auto values = npy_numbers(arrays.at("values"));
    if (!frames || !times || !values || frames->size() != times->size() || frames->size() != values->size()) {
        return std::nullopt;
    }
    std::tuple<std::vector<int>, std::vector<double>, std::vector<float>> converted;
    for (const double frame : *frames) {
        std::get<0>(converted).push_back(static_cast<int>(frame));
    }
    std::get<1>(converted) = *times;
    for (const double value : *values) {
        std::get<2>(converted).push_back(static_cast<float>(value));
    }
    return converted;
}

// The score saved in `path` if it was saved for exactly `spec`
// (_load_metric_file), from the pictures the pair is decoded to now
// (the pair's decoding marks); nothing otherwise, or if the file cannot be
// read.
std::optional<LoadedMetric> load_metric_file(const std::filesystem::path& path, const MetricRequestSpec& spec,
                                             const DecodingMarks& marks) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        return std::nullopt;
    }
    try {
        const auto score = read_score(path);
        if (!score) {
            return std::nullopt;
        }
        const auto& [metadata, arrays] = *score;
        if (!current_format(metadata) || !decoded_as_now(metadata, marks) ||
            metadata.value("key", nlohmann::json()) != spec.key ||
            metadata.value("request", nlohmann::json()) != spec.identity()) {
            return std::nullopt;
        }
        const std::optional<MetricProvenance> provenance = provenance_from(metadata.at("provenance"));
        if (!provenance) {
            return std::nullopt;
        }
        const nlohmann::json kind = metadata.at("kind");
        const bool marked = metadata.value(av1_grain_dav1d, nlohmann::json()) == av1_grain_dav1d_version;
        if (kind == "frame") {
            auto frames = timeline(arrays);
            if (!frames) {
                return std::nullopt;
            }
            auto& [frame, time, values] = *frames;
            FrameMetricResult result{spec.key, std::move(frame), std::move(time), std::move(values), *provenance, marked};
            if (arrays.contains("exact")) {
                // A score saved before it had them, or cut short: none, the rounded scores stand alone.
                if (std::optional<std::vector<double>> exact = npy_numbers(arrays.at("exact"));
                    exact && exact->size() == result.values.size()) {
                    result.exact = std::move(*exact);
                }
            }
            return sampled_every(std::move(result), spec.coverage ? spec.coverage->step : 1);
        }
        if (kind == "sequence") {
            std::optional<std::vector<double>> score_value;
            if (arrays.contains("score")) {
                score_value = npy_numbers(arrays.at("score"));
            }
            if (!score_value || score_value->size() != 1) {
                return std::nullopt;
            }
            SequenceMetricResult result{spec.key, score_value->front(), *provenance, {}, {}, {}, marked};
            if (arrays.contains("values")) {
                auto frames = timeline(arrays);
                if (!frames) {
                    return std::nullopt;
                }
                std::tie(result.frame, result.time, result.values) = std::move(*frames);
            }
            return result;
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }
    return std::nullopt;
}

// The comparison's score files of one metric, by name: <key>_<hash>.npz.
std::vector<std::filesystem::path> score_files(const std::filesystem::path& directory, const std::string& key) {
    std::vector<std::filesystem::path> paths;
    std::error_code error;
    std::filesystem::directory_iterator entry(directory, error);
    for (; !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        const std::string name = platform::path_text(entry->path().filename());
        if (name.starts_with(key + "_") && name.ends_with(".npz")) {
            paths.push_back(entry->path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// The saved scores that answer a perceptual request which does not fix
// where the metric is calculated, each with the request carrying its
// implementation: the request's Vship build's first, then the CPU's.
std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> auto_perceptual_candidates(
    const std::filesystem::path& directory, const MetricRequestSpec& spec) {
    std::vector<std::pair<int, std::pair<std::filesystem::path, MetricRequestSpec>>> ranked;
    for (const std::filesystem::path& path : score_files(directory, spec.key)) {
        try {
            const auto score = read_score(path);
            if (!score) {
                continue;
            }
            const nlohmann::json& metadata = score->first;
            const nlohmann::json request = metadata.value("request", nlohmann::json::object());
            const nlohmann::json compatibility = request.value("implementation_compatibility_id", nlohmann::json());
            const nlohmann::json provenance = metadata.value("provenance", nlohmann::json::object());
            const std::string value = compatibility.is_string() ? compatibility.get<std::string>() : std::string();
            if (!current_format(metadata) || metadata.value("key", nlohmann::json()) != spec.key ||
                !auto_perceptual_compatible(spec, value) || !provenance.is_object() ||
                provenance.value("implementation_compatibility_id", nlohmann::json()) != compatibility) {
                continue;
            }
            MetricRequestSpec concrete = spec;
            concrete.implementation_compatibility_id = value;
            if (request != concrete.identity()) {
                continue;
            }
            const int rank = is_vship_id(value) ? 0 : 1;
            ranked.push_back({rank, {path, concrete}});
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> found;
    for (auto& [rank, candidate] : ranked) {
        found.push_back(std::move(candidate));
    }
    return found;
}

// The size this comparison's frames are compared at, worked out from the
// sizes and black bars in its context.json, as the run works it out. Auto
// picks VMAF's model by it.
std::optional<Size> compared_size(const std::filesystem::path& directory) {
    try {
        const std::optional<nlohmann::json> context = read_json(directory / "context.json");
        if (!context || !context->is_object()) {
            return std::nullopt;
        }
        const VideoInfo source = video_info_from_dict(context->at("source_info"));
        if (python_truthy(context->value("resample_target", nlohmann::json()))) {
            // A resolution test compares at the source's own (cropped) size:
            // the downscale is undone before the comparison.
            return resample_analysis_dimensions(source,
                                                crop_from_dict(context->value("source_crop", nlohmann::json())));
        }
        const VideoInfo distorted = video_info_from_dict(context->at("distorted_info"));
        return compared_dimensions(source, distorted,
                                   scale_direction_from(context->at("scale_direction").get<std::string>()),
                                   crop_from_dict(context->value("source_crop", nlohmann::json())),
                                   crop_from_dict(context->value("distorted_crop", nlohmann::json())));
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

// An object's member, or {} when it is missing, null or not an object (the
// Python app's `x.get(name) or {}`).
nlohmann::json member_object(const nlohmann::json& object, const char* name) {
    const nlohmann::json value = object.is_object() ? object.value(name, nlohmann::json()) : nlohmann::json();
    return value.is_object() ? value : nlohmann::json::object();
}

// The model a saved VMAF score was calculated with, or nothing when that is
// not certain (_vmaf_model_ran): an explicitly chosen (or bundled) model is
// the model; for Auto, the model the score's provenance records, or the one
// a score migrated from the first cache format was saved under.
std::optional<std::string> vmaf_model_ran(const nlohmann::json& request, const nlohmann::json& provenance) {
    const nlohmann::json parameters = member_object(request, "parameters");
    const nlohmann::json choice = parameters.value("model_choice", nlohmann::json());
    if (choice.is_string() && choice != "" && choice != std::string(auto_model_choice) &&
        choice != std::string(custom_model_choice)) {
        return choice.get<std::string>();
    }
    if (choice != std::string(auto_model_choice)) {
        return std::nullopt;
    }
    const nlohmann::json ran = member_object(provenance, "parameters").value("model", nlohmann::json());
    if (ran.is_string() && ran.get<std::string>().starts_with("version=")) {
        return ran.get<std::string>();
    }
    const nlohmann::json model = parameters.value("model", nlohmann::json());
    if (provenance.is_object() && provenance.value("implementation", nlohmann::json()) == "legacy-v1-cache" &&
        model.is_string() && model.get<std::string>().starts_with("version=")) {
        return model.get<std::string>();
    }
    return std::nullopt;
}

// Saved VMAF scores of the same comparison under another request that were
// calculated with the same model, best match first: for a request on Auto,
// any other Auto score, or a score of the model Auto picks here; for an
// explicit model, an Auto score that certainly ran it.
std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> vmaf_equivalents(const std::filesystem::path& directory,
                                                                                  const MetricRequestSpec& spec) {
    const nlohmann::json wanted = spec.identity();
    const nlohmann::json& parameters = wanted["parameters"];
    const nlohmann::json choice = parameters.value("model_choice", nlohmann::json());
    const nlohmann::json rest = without(wanted, {"parameters"});
    const std::filesystem::path exact = metric_path(directory, spec);
    std::optional<std::optional<std::string>> auto_model;  // worked out once, only if needed
    std::vector<std::pair<int, std::pair<std::filesystem::path, MetricRequestSpec>>> ranked;
    for (const std::filesystem::path& path : score_files(directory, "vmaf")) {
        if (path == exact) {
            continue;
        }
        try {
            const auto score = read_score(path);
            if (!score) {
                continue;
            }
            const nlohmann::json& metadata = score->first;
            const nlohmann::json request = member_object(metadata, "request");
            const nlohmann::json stored = member_object(request, "parameters");
            if (!current_format(metadata) || metadata.value("key", nlohmann::json()) != "vmaf" ||
                without(request, {"parameters"}) != rest ||
                stored.value("custom_model", nlohmann::json("")) != parameters.value("custom_model", nlohmann::json(""))) {
                continue;
            }
            int rank = 1;
            if (stored.value("model_choice", nlohmann::json()) != choice) {
                const std::optional<std::string> ran = vmaf_model_ran(request, member_object(metadata, "provenance"));
                if (!ran) {
                    continue;
                }
                if (choice == std::string(auto_model_choice)) {
                    if (!auto_model) {
                        auto_model.emplace();
                        if (const std::optional<Size> size = compared_size(directory)) {
                            *auto_model = model_for_resolution(size->first, size->second);
                        }
                    }
                    if (*auto_model != *ran) {
                        continue;
                    }
                } else if (!(choice.is_string() && choice.get<std::string>().starts_with("version=") &&
                             stored.value("model_choice", nlohmann::json()) == std::string(auto_model_choice) &&
                             *ran == choice.get<std::string>())) {
                    continue;
                }
                rank = 2;
            }
            MetricRequestSpec concrete = spec;
            concrete.parameters = stored;
            ranked.push_back({rank, {path, concrete}});
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> found;
    for (auto& [rank, candidate] : ranked) {
        found.push_back(std::move(candidate));
    }
    return found;
}

bool is_v1_choice(const nlohmann::json& choice) {
    return choice.is_string() && !choice.get<std::string>().empty() && is_builtin_model_choice(choice.get<std::string>());
}

// VMAF v1 scores saved before VMAF v1 had a column of its own: under the key
// "vmaf", with a bundled v1 model as the model choice. They answer a VMAF v1
// request for that model, or one on Auto when it is the model Auto picks.
std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> old_v1_scores(const std::filesystem::path& directory,
                                                                               const MetricRequestSpec& spec) {
    const nlohmann::json wanted = spec.identity();
    const nlohmann::json choice = wanted["parameters"].value("model_choice", nlohmann::json());
    const nlohmann::json rest = without(wanted, {"key", "parameters"});
    std::optional<std::optional<std::string>> auto_model;
    std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> found;
    for (const std::filesystem::path& path : score_files(directory, "vmaf")) {
        try {
            const auto score = read_score(path);
            if (!score) {
                continue;
            }
            const nlohmann::json& metadata = score->first;
            const nlohmann::json request = member_object(metadata, "request");
            const nlohmann::json stored = member_object(request, "parameters");
            const nlohmann::json stored_choice = stored.value("model_choice", nlohmann::json());
            if (!current_format(metadata) || metadata.value("key", nlohmann::json()) != "vmaf" ||
                !is_v1_choice(stored_choice) || without(request, {"key", "parameters"}) != rest) {
                continue;
            }
            if (choice == std::string(auto_model_choice)) {
                if (!auto_model) {
                    auto_model.emplace();
                    if (const std::optional<Size> size = compared_size(directory)) {
                        *auto_model = v1_model_for_resolution(size->first, size->second);
                    }
                }
                if (!*auto_model || stored_choice != **auto_model) {
                    continue;
                }
            } else if (stored_choice != choice) {
                continue;
            }
            MetricRequestSpec concrete = spec;
            concrete.key = "vmaf";
            concrete.parameters = stored;
            found.emplace_back(path, concrete);
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    return found;
}

void merge_legacy(const std::filesystem::path& legacy, const std::filesystem::path& directory) {
    std::error_code error;
    std::filesystem::directory_iterator item(legacy, error);
    for (; !error && item != std::filesystem::directory_iterator(); item.increment(error)) {
        const std::filesystem::path target = directory / item->path().filename();
        std::error_code ignored;
        if (std::filesystem::exists(target, ignored)) {
            if (item->is_regular_file(ignored)) {
                std::filesystem::remove(item->path(), ignored);
            }
        } else {
            std::filesystem::rename(item->path(), target, ignored);
        }
    }
    std::error_code ignored;
    std::filesystem::remove(legacy, ignored);  // only if it is empty now
}

}  // namespace

std::optional<SavedScore> read_saved_score(const std::filesystem::path& file) {
    try {
        const auto score = read_score(file);
        if (!score) {
            return std::nullopt;
        }
        const auto& [metadata, arrays] = *score;
        const nlohmann::json key = metadata.value("key", nlohmann::json());
        if (!current_format(metadata) || !key.is_string()) {
            return std::nullopt;
        }
        const std::optional<MetricProvenance> provenance = provenance_from(metadata.at("provenance"));
        if (!provenance) {
            return std::nullopt;
        }
        // The checks load_metric makes of any score, whatever is asked.
        // Issue #133: without them, the tab showed, opened and exported
        // scores the Videos tab calculates again.
        const nlohmann::json context = read_json(file.parent_path() / "context.json").value_or(nlohmann::json());
        const bool outdated = !decoded_as_now(metadata, decoding_marks(context)) ||
                              stale_score(*provenance, context_colour_tags(context));
        const bool marked = metadata.value(av1_grain_dav1d, nlohmann::json()) == av1_grain_dav1d_version;
        const nlohmann::json kind = metadata.at("kind");
        if (kind == "frame") {
            auto frames = timeline(arrays);
            if (!frames) {
                return std::nullopt;
            }
            auto& [frame, time, values] = *frames;
            return SavedScore{FrameMetricResult{key.get<std::string>(), std::move(frame), std::move(time),
                                                std::move(values), *provenance, marked},
                              outdated};
        }
        if (kind == "sequence" && arrays.contains("score")) {
            const std::optional<std::vector<double>> value = npy_numbers(arrays.at("score"));
            if (!value || value->size() != 1) {
                return std::nullopt;
            }
            SequenceMetricResult result{key.get<std::string>(), value->front(), *provenance, {}, {}, {}, marked};
            if (arrays.contains("values")) {
                auto frames = timeline(arrays);
                if (!frames) {
                    return std::nullopt;
                }
                std::tie(result.frame, result.time, result.values) = std::move(*frames);
            }
            return SavedScore{std::move(result), outdated};
        }
    } catch (const nlohmann::json::exception&) {
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

namespace cache_parts {

std::string hash_of(const std::string& source_identity, const std::string& distorted_identity,
                    const nlohmann::json& recipe_identity) {
    return sha256_hex(python_json_compact(
        {{"source", source_identity}, {"distorted", distorted_identity}, {"recipe", recipe_identity}}));
}

std::optional<nlohmann::json> parameter(const MetricProvenance& provenance, const std::string& name) {
    const auto found = provenance.parameters.find(name);
    if (found == provenance.parameters.end()) {
        return std::nullopt;
    }
    return nlohmann::json::parse(python_json_readable(found->second), nullptr, false);
}

nlohmann::json frames_short_json(const std::optional<FramesShort>& short_of) {
    if (!short_of) {
        return nullptr;
    }
    const auto count = [](const std::optional<long long>& value) {
        return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
    };
    return {{"scored_pairs", short_of->scored_pairs},
            {"length_frames", short_of->length_frames},
            {"source_frames", count(short_of->source_frames)},
            {"test_frames", count(short_of->test_frames)},
            {"source_length", count(short_of->source_length)},
            {"test_length", count(short_of->test_length)}};
}

std::optional<FramesShort> frames_short_from_json(const nlohmann::json& saved) {
    if (!saved.is_object() || !saved.value("scored_pairs", nlohmann::json()).is_number_integer() ||
        !saved.value("length_frames", nlohmann::json()).is_number_integer()) {
        return std::nullopt;
    }
    const auto count = [&saved](const char* key) -> std::optional<long long> {
        const nlohmann::json value = saved.value(key, nlohmann::json());
        return value.is_number_integer() ? std::optional(value.get<long long>()) : std::nullopt;
    };
    return FramesShort{saved["scored_pairs"].get<long long>(), saved["length_frames"].get<long long>(),
                       count("source_frames"), count("test_frames"), count("source_length"),
                       count("test_length")};
}

void loaded_model(const MetricResultSet& results, const std::vector<MetricRequestSpec>& specs, const std::string& key,
                  std::string& model, std::optional<std::string>& choice) {
    const FrameMetricResult* metric = results.frame(key);
    const auto spec =
        std::find_if(specs.begin(), specs.end(), [&key](const MetricRequestSpec& asked) { return asked.key == key; });
    if (metric == nullptr || spec == specs.end()) {
        return;
    }
    if (const std::optional<nlohmann::json> ran = parameter(metric->provenance, "model");
        ran && ran->is_string() && !ran->get<std::string>().empty()) {
        model = ran->get<std::string>();
    }
    if (const auto chosen = spec->parameters.find("model_choice"); chosen != spec->parameters.end()) {
        if (chosen->is_string()) {
            choice = chosen->get<std::string>();
        } else {
            choice.reset();
        }
    }
}

const MetricProvenance& provenance_of(const LoadedMetric& metric) {
    return std::visit([](const auto& result) -> const MetricProvenance& { return result.provenance; }, metric);
}

bool is_vship_id(std::string_view id) { return id.find("-vship-") != std::string_view::npos; }

bool is_auto_perceptual(const MetricRequestSpec& spec) {
    const std::string& id = spec.implementation_compatibility_id;
    return spec.backend_id == "perceptual" &&
           (id.ends_with(score_identity::gpu_or_cpu_suffix) || id.ends_with(score_identity::vship51_gpu_or_cpu_suffix));
}

std::optional<nlohmann::json> read_json(const std::filesystem::path& path) {
    const std::optional<std::string> text = platform::read_file(path);
    if (!text) {
        return std::nullopt;
    }
    nlohmann::json data = nlohmann::json::parse(python_json_readable(*text), nullptr, false);
    if (data.is_discarded()) {
        return std::nullopt;
    }
    return data;
}

bool current_format(const nlohmann::json& metadata) {
    const nlohmann::json version = metadata.value("format_version", nlohmann::json());
    return version.is_number() && version.get<double>() == metric_cache_format_version;
}

void with_profiles_read_now(nlohmann::json& context, const ComparisonRecipe& recipe) {
    if (!context.is_object()) {
        return;
    }
    using VideoProfile = std::pair<const char*, const std::string*>;  // the context's field, the profile read now
    for (const auto& [video, profile] :
         {VideoProfile{"source_info", &recipe.source_profile}, VideoProfile{"distorted_info", &recipe.distorted_profile}}) {
        auto info = context.find(video);
        if (profile->empty() || info == context.end() || !info->is_object()) {
            continue;
        }
        const auto saved = info->find("profile");
        if (saved == info->end() || !saved->is_string() || saved->get<std::string>().empty()) {
            (*info)["profile"] = *profile;
        }
    }
}

std::vector<std::pair<nlohmann::json, double>> other_parameters(const std::filesystem::path& directory,
                                                                const MetricRequestSpec& spec, bool ranged,
                                                                const ComparisonRecipe& recipe) {
    const nlohmann::json wanted = spec.identity();
    const nlohmann::json rest = without(wanted, {"parameters"});
    const std::vector<std::filesystem::path> paths = score_files(directory, spec.key);
    nlohmann::json context =
        paths.empty() ? nlohmann::json() : read_json(directory / "context.json").value_or(nlohmann::json());
    with_profiles_read_now(context, recipe);
    const std::vector<ColourTags> infos = context_colour_tags(context);
    const DecodingMarks marks = decoding_marks(context);
    std::vector<std::pair<nlohmann::json, double>> found;
    for (const std::filesystem::path& path : paths) {
        try {
            const auto score = read_score(path);
            if (!score) {
                continue;
            }
            const auto& [metadata, arrays] = *score;
            const nlohmann::json request = metadata.value("request", nlohmann::json::object());
            const nlohmann::json request_rest = without(request, {"parameters"});
            if (!current_format(metadata) || !decoded_as_now(metadata, marks) ||
                metadata.value("key", nlohmann::json()) != spec.key ||
                metadata.value("kind", nlohmann::json()) != "sequence" ||
                request_rest != rest ||
                request.value("parameters", nlohmann::json()) == wanted.at("parameters")) {
                continue;
            }
            const nlohmann::json made = metadata.value("provenance", nlohmann::json());
            if (made.is_object()) {
                const std::optional<MetricProvenance> provenance = provenance_from(made);
                if (!provenance || stale_score(*provenance, infos) ||
                    (ranged && !provenance->parameters.contains("range_frames"))) {
                    continue;
                }
            } else if (ranged) {
                continue;
            }
            const nlohmann::json parameters = request.at("parameters");
            const std::optional<std::vector<double>> value =
                arrays.contains("score") ? npy_numbers(arrays.at("score")) : std::nullopt;
            if (!parameters.is_object() || !value || value->size() != 1) {
                continue;
            }
            if (std::isfinite(value->front())) {
                found.emplace_back(parameters, value->front());
            }
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    return found;
}

std::optional<LoadedMetric> load_metric(const std::filesystem::path& directory, const MetricRequestSpec& spec,
                                        const std::string& backend, const nlohmann::json& context) {
    const std::vector<ColourTags> infos = context_colour_tags(context);
    const DecodingMarks marks = decoding_marks(context);
    const auto stale = [&infos](const LoadedMetric& metric) { return stale_score(provenance_of(metric), infos); };
    if (saved_as(spec.key) != spec.key) {
        // A YUV metric's Y: the plain metric's score, from its file, or from
        // the Y's own file of before (own_metric_path).
        std::optional<LoadedMetric> result = load_metric(directory, saved_spec(spec), backend, context);
        if (!result) {
            result = load_metric_file(own_metric_path(directory, spec), spec, marks);
            return result && !stale(*result) ? result : std::nullopt;
        }
        std::visit([&spec](auto& metric) { metric.key = spec.key; }, *result);
        return result;
    }
    if (is_auto_perceptual(spec)) {
        for (const auto& [path, concrete] : auto_perceptual_candidates(directory, spec)) {
            if (backend == "cpu" && is_vship_id(concrete.implementation_compatibility_id)) {
                continue;
            }
            std::optional<LoadedMetric> result = load_metric_file(path, concrete, marks);
            if (result && !stale(*result)) {
                return result;
            }
        }
        return std::nullopt;
    }
    std::optional<LoadedMetric> result = load_metric_file(metric_path(directory, spec), spec, marks);
    if (result && stale(*result)) {
        return std::nullopt;
    }
    if (!result && spec.key == "vmaf" && spec.backend_id == "ffmpeg") {
        for (const auto& [path, concrete] : vmaf_equivalents(directory, spec)) {
            result = load_metric_file(path, concrete, marks);
            if (result) {
                break;
            }
        }
    }
    if (!result && spec.key == "vmaf_v1" && spec.backend_id == "ffmpeg") {
        for (const auto& [path, concrete] : old_v1_scores(directory, spec)) {
            std::optional<LoadedMetric> old = load_metric_file(path, concrete, marks);
            if (old && std::holds_alternative<FrameMetricResult>(*old)) {
                FrameMetricResult v1 = std::get<FrameMetricResult>(std::move(*old));
                v1.key = "vmaf_v1";
                return v1;
            }
        }
    }
    return result;
}

}  // namespace cache_parts

std::filesystem::path default_cache_dir() {
    return app_data_folder() / "results_cache";
}

std::string recipe_hash(const std::filesystem::path& source, const std::filesystem::path& distorted,
                        const ComparisonRecipe& recipe) {
    return hash_of(file_identity(source), file_identity(distorted), recipe.identity());
}

std::string metric_identity_hash(const MetricRequestSpec& spec) {
    return sha256_hex(python_json_compact(spec.identity()));
}

namespace {

// What the saved scores know of files, kept for the process: their sampled
// hashes, video fingerprints and stream fingerprints by path identity (each
// read once), and each base's comparisons as their identity.json and
// context.json record them (each read again when either file changes:
// another process, the command line or another window, saves comparisons
// too).
struct RecordedComparison {
    std::filesystem::path folder;
    nlohmann::json recipe;
    std::string source_path, source_sampled, source_video;
    std::string distorted_path, distorted_sampled;
    std::string distorted_rate, distorted_head, distorted_stream;
    // Its folder's time when it was read (as_read).
    std::filesystem::file_time_type folder_time{};
};

// A comparison folder as last read: its time as its folder's listing gave
// it, and what its identity.json and context.json recorded (nothing where
// either was missing or unreadable).
struct ReadComparison {
    std::filesystem::file_time_type folder_time;
    std::shared_ptr<const RecordedComparison> recorded;
};

// The saved comparisons recorded under a base (recorded_comparisons).
using RecordedList = std::vector<std::shared_ptr<const RecordedComparison>>;

// A base's comparison folders as last listed and read, kept for the process
// (recorded_comparisons).
struct Listing {
    // Whether its folder was listed, and the folder's time then (none where
    // there was no folder).
    bool listed = false;
    std::optional<std::filesystem::file_time_type> folder_time;
    // By folder, and those with a record.
    std::map<std::filesystem::path, ReadComparison> folders;
    std::shared_ptr<const RecordedList> list = std::make_shared<const RecordedList>();
    // The folders this process changed since (recorded_changed), each by the
    // number of its latest change; and the number of the latest change to
    // the whole folder (recorded_forgotten), 0 for none since it was listed.
    std::map<std::filesystem::path, std::uint64_t> changed;
    std::uint64_t forgotten = 0;
};

struct KnownFiles {
    std::mutex mutex;
    VideoFingerprinter fingerprinter;
    StreamFingerprinter stream_fingerprinter;
    std::map<std::string, std::optional<std::string>> sampled;
    std::map<std::string, std::optional<std::string>> video;
    // The video fingerprints being made (video_of), by file: the answer, or
    // none where the one making it was stopped.
    std::map<std::string, std::shared_future<std::optional<std::optional<std::string>>>> video_pending;
    std::map<std::pair<std::string, int>, std::optional<std::string>> stream;
    // By base. A Listing's folders and list are changed only by the thread
    // that holds `listing` as well as `mutex`: that thread reads them with
    // `listing` alone.
    std::map<std::filesystem::path, Listing> recorded;
    std::mutex listing;
    // Numbers this process's changes to the comparisons' files
    // (recorded_changed, recorded_forgotten).
    std::uint64_t changes = 0;
};

KnownFiles& known_files() {
    static KnownFiles known;
    return known;
}

// This process wrote the identity.json or context.json of the comparison in
// `folder`, made it or deleted it: it is read again (or dropped) at the next
// lookup (recorded_comparisons), whatever the file system did to the times.
void recorded_changed(const std::filesystem::path& folder) {
    KnownFiles& known = known_files();
    const std::lock_guard lock(known.mutex);
    ++known.changes;
    for (auto& [base, listing] : known.recorded) {
        // A listing being made has it read again too, after.
        listing.changed[base / format_folder / folder.filename()] = known.changes;
    }
}

// This process deleted comparisons under `base` (Settings' Clear): its
// folder is listed again at the next lookup.
void recorded_forgotten(const std::filesystem::path& base) {
    KnownFiles& known = known_files();
    const std::lock_guard lock(known.mutex);
    ++known.changes;
    if (const auto found = known.recorded.find(base); found != known.recorded.end()) {
        found->second.forgotten = known.changes;
    }
}

std::optional<std::string> sampled_of(const std::filesystem::path& file) {
    const std::string identity = file_identity(file);
    KnownFiles& known = known_files();
    {
        const std::lock_guard lock(known.mutex);
        if (const auto found = known.sampled.find(identity); found != known.sampled.end()) {
            return found->second;
        }
    }
    std::optional<std::string> hash = sampled_file_hash(file);
    const std::lock_guard lock(known.mutex);
    known.sampled[identity] = hash;
    return hash;
}

// The video fingerprint of `file`, made once per process: a caller asking
// while another makes it waits for that answer (without the mutex). One
// stopped by `cancel` is not kept; a caller that waited for it then makes it
// itself, unless its own `cancel` stopped too.
std::optional<std::string> video_of(const std::filesystem::path& file, const CancelToken& cancel = CancelToken()) {
    const std::string identity = file_identity(file);
    KnownFiles& known = known_files();
    while (true) {
        VideoFingerprinter fingerprinter;
        std::shared_future<std::optional<std::optional<std::string>>> pending;
        std::promise<std::optional<std::optional<std::string>>> answer;
        {
            const std::lock_guard lock(known.mutex);
            if (const auto found = known.video.find(identity); found != known.video.end()) {
                return found->second;
            }
            // One recorded for this file before (another comparison of the
            // same reference): no decoding.
            for (const auto& [base, listing] : known.recorded) {
                for (const auto& [folder, read] : listing.folders) {
                    if (read.recorded && read.recorded->source_path == identity &&
                        !read.recorded->source_video.empty()) {
                        known.video[identity] = read.recorded->source_video;
                        return read.recorded->source_video;
                    }
                }
            }
            if (const auto making = known.video_pending.find(identity); making != known.video_pending.end()) {
                pending = making->second;
            } else {
                fingerprinter = known.fingerprinter;
                if (!fingerprinter) {
                    return std::nullopt;
                }
                known.video_pending.emplace(identity, answer.get_future().share());
            }
        }
        if (pending.valid()) {
            if (const std::optional<std::optional<std::string>> made = pending.get()) {
                return *made;
            }
            if (cancel.cancelled()) {
                return std::nullopt;
            }
            continue;  // the one making it was stopped: made here
        }
        // Ends the wait of those waiting for it, whatever happens here.
        std::optional<std::optional<std::string>> made;
        struct Answers {
            KnownFiles& known;
            const std::string& identity;
            std::promise<std::optional<std::optional<std::string>>>& answer;
            std::optional<std::optional<std::string>>& made;
            ~Answers() {
                {
                    const std::lock_guard lock(known.mutex);
                    known.video_pending.erase(identity);
                    if (made) {
                        known.video[identity] = *made;
                    }
                }
                answer.set_value(made);
            }
        } const answers{known, identity, answer, made};
        std::optional<std::string> fingerprint = fingerprinter(file, cancel);
        if (cancel.cancelled()) {
            return std::nullopt;  // not kept: the next caller makes it
        }
        made = fingerprint;
        return fingerprint;
    }
}

// What a test video's stream fingerprint is made of: its first
// `head_packets` packets, a cheap first check, or all of them.
constexpr int head_packets = 100;

// How the lookups on this thread read streams (StreamReads): what stops
// them (none: nothing), whether they may read whole streams, and whom to
// tell first; and whether one would have read a whole stream.
struct ThreadStreamReads {
    std::optional<CancelToken> cancel;
    bool whole = false;
    std::function<void(const std::filesystem::path&)> announce;
    bool whole_wanted = false;
};
thread_local ThreadStreamReads lookup_reads;

// The stream fingerprint of `file`'s first `packets` packets (0: all), read
// once (a read cancelled is not kept). Not `read`: only one read before.
// Nothing without a stream fingerprinter (the setting off), not even one
// read before, and nothing kept for when one is set.
std::optional<std::string> stream_of(const std::filesystem::path& file, int packets, bool read = true,
                                     const CancelToken& cancel = lookup_reads.cancel.value_or(CancelToken())) {
    const std::pair<std::string, int> key{file_identity(file), packets};
    KnownFiles& known = known_files();
    StreamFingerprinter fingerprinter;
    {
        const std::lock_guard lock(known.mutex);
        fingerprinter = known.stream_fingerprinter;
        if (!fingerprinter) {
            return std::nullopt;
        }
        if (const auto found = known.stream.find(key); found != known.stream.end()) {
            return found->second;
        }
    }
    if (!read || cancel.cancelled()) {
        return std::nullopt;
    }
    std::optional<std::string> fingerprint = fingerprinter(file, packets, cancel);
    if (cancel.cancelled()) {
        return std::nullopt;
    }
    const std::lock_guard lock(known.mutex);
    known.stream[key] = fingerprint;
    return fingerprint;
}

// A test video's frame rate as identity.json records it, to the hundredth:
// a remux to MP4 can state 24.003 for 24 (an average over its rounded
// duration). Empty for none.
std::string rate_text(double fps) {
    if (!(fps > 0.0)) {
        return {};
    }
    char text[32];
    std::snprintf(text, sizeof text, "%.2f", fps);
    return text;
}

std::string text_at(const nlohmann::json& data, const char* side, const char* field) {
    const nlohmann::json value = data.value(side, nlohmann::json::object()).value(field, nlohmann::json());
    return value.is_string() ? value.get<std::string>() : std::string();
}

// Whether identity.json `data` recorded its test stream in the form
// stream_fingerprint makes now: one recorded in another (or before forms
// were recorded: 1) is as if none were, read again and never matched.
bool stream_of_this_format(const nlohmann::json& data) {
    const nlohmann::json format =
        data.value("distorted", nlohmann::json::object()).value("stream_format", nlohmann::json());
    return format.is_number_integer() && format.get<int>() == stream_fingerprint_format;
}

// What the comparison in `folder` (whose time is `folder_time`) recorded in
// its identity.json and context.json: nothing where either is missing or
// cannot be read (saved before identity.json was recorded: nothing to know
// it by).
ReadComparison read_comparison(const std::filesystem::path& folder, std::filesystem::file_time_type folder_time) {
    ReadComparison read{folder_time, nullptr};
    const std::optional<nlohmann::json> identity = read_json(folder / "identity.json");
    const std::optional<nlohmann::json> context = read_json(folder / "context.json");
    if (identity && identity->is_object() && context && context->is_object()) {
        RecordedComparison recorded{folder,
                                    context->value("recipe", nlohmann::json()),
                                    text_at(*identity, "source", "path"),
                                    text_at(*identity, "source", "sampled"),
                                    text_at(*identity, "source", "video"),
                                    text_at(*identity, "distorted", "path"),
                                    text_at(*identity, "distorted", "sampled"),
                                    text_at(*identity, "distorted", "rate"),
                                    text_at(*identity, "distorted", "stream_head"),
                                    text_at(*identity, "distorted", "stream")};
        if (!stream_of_this_format(*identity)) {
            recorded.distorted_head.clear();
            recorded.distorted_stream.clear();
        }
        recorded.folder_time = folder_time;
        read.recorded = std::make_shared<const RecordedComparison>(std::move(recorded));
    }
    return read;
}

// Every comparison under `base` with an identity.json. Issue #136: the
// whole folder was listed at every lookup or save whose own folder is
// missing, which with thousands of comparisons, or a cache on a network
// share, held up every new row and every first save.
//
// The listing is kept for the process. The folder is listed again (a
// comparison read again only where its folder's time changed) where its own
// time changed since: comparisons saved or deleted by another process (the
// command line, another window), on a file system that changes it (NTFS,
// network shares); and after Settings' Clear. What this process saves or
// deletes is read again on its own (recorded_changed). A comparison another
// process deleted or saved again is read again when a lookup would take it
// (as_read), whatever the file system did to the folder's time; one it
// saved, on a file system that leaves the folder's time (FAT, a RAM drive),
// is found once that time changes, or in the next session.
std::shared_ptr<const RecordedList> recorded_comparisons(const std::filesystem::path& base) {
    KnownFiles& known = known_files();
    const std::lock_guard one_listing(known.listing);  // one lists, the others take what it found
    const std::filesystem::path root = base / format_folder;
    std::optional<std::filesystem::file_time_type> folder_time;
    std::error_code error;
    if (const auto time = std::filesystem::last_write_time(root, error); !error) {
        folder_time = time;
    }
    Listing* listing = nullptr;
    std::uint64_t changes = 0;
    bool whole = false;
    std::set<std::filesystem::path> changed;
    {
        const std::lock_guard lock(known.mutex);
        listing = &known.recorded[base];
        changes = known.changes;
        whole = !listing->listed || listing->forgotten != 0 || listing->folder_time != folder_time;
        if (!whole && listing->changed.empty()) {
            return listing->list;
        }
        for (const auto& [folder, number] : listing->changed) {
            changed.insert(folder);
        }
    }
    // Read without `mutex`: only this thread changes the listing's folders.
    std::map<std::filesystem::path, ReadComparison> now;
    bool listed_whole = false;  // the folder listed to its end
    std::vector<std::pair<std::filesystem::path, std::optional<ReadComparison>>> updates;
    if (whole) {
        std::filesystem::directory_iterator folder(root, error);
        for (; !error && folder != std::filesystem::directory_iterator(); folder.increment(error)) {
            std::error_code unread;
            const std::filesystem::file_time_type time = folder->last_write_time(unread);
            if (unread || !folder->is_directory(unread)) {
                continue;
            }
            if (const auto known_read = listing->folders.find(folder->path());
                known_read != listing->folders.end() && known_read->second.folder_time == time &&
                !changed.contains(folder->path())) {
                now.insert(*known_read);
                continue;
            }
            now.emplace(folder->path(), read_comparison(folder->path(), time));
        }
        listed_whole = !error;
    } else {
        for (const std::filesystem::path& folder : changed) {
            std::error_code unread;
            const std::filesystem::file_time_type time = std::filesystem::last_write_time(folder, unread);
            if (unread || !std::filesystem::is_directory(folder, unread)) {
                updates.emplace_back(folder, std::nullopt);  // deleted
            } else {
                updates.emplace_back(folder, read_comparison(folder, time));
            }
        }
    }
    const std::lock_guard lock(known.mutex);
    if (whole) {
        listing->folders = std::move(now);
        // A listing that could not be made, or stopped part-way (a share
        // gone, access refused), serves this lookup only: kept, it would
        // stand for the rest of the session while the folder's time stays.
        listing->listed = listed_whole;
        listing->folder_time = folder_time;
        if (listing->forgotten <= changes) {
            listing->forgotten = 0;  // else cleared meanwhile: listed again next time
        }
    } else {
        for (auto& [folder, read] : updates) {
            if (read) {
                listing->folders[folder] = std::move(*read);
            } else {
                listing->folders.erase(folder);
            }
        }
    }
    // A folder changed again while it was being read is read again next time.
    std::erase_if(listing->changed, [changes](const auto& folder) { return folder.second <= changes; });
    auto list = std::make_shared<RecordedList>();
    list->reserve(listing->folders.size());
    for (const auto& [path, read] : listing->folders) {
        if (read.recorded) {
            list->push_back(read.recorded);
        }
    }
    listing->list = std::move(list);
    return listing->list;
}

// Whether the comparison `recorded`, of a kept listing, is still as it was
// read: its identity.json there, its folder's time the same. Where it is
// not (deleted or saved again by another process), it is read again at the
// next lookup (recorded_comparisons).
bool as_read(const RecordedComparison& recorded) {
    std::error_code error;
    const std::filesystem::file_time_type time = std::filesystem::last_write_time(recorded.folder, error);
    if (!error && time == recorded.folder_time &&
        std::filesystem::is_regular_file(recorded.folder / "identity.json", error)) {
        return true;
    }
    recorded_changed(recorded.folder);
    return false;
}

// Those of `candidates` saved with the reference `source`, cheapest first:
// by its path; with none, by its sampled hash; with none, by its stream
// (decoded only where a candidate recorded one).
std::vector<RecordedComparison> with_same_reference(const std::vector<RecordedComparison>& candidates,
                                                    const std::filesystem::path& source) {
    std::vector<RecordedComparison> found;
    const auto take = [&](const auto& same) {
        std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(found), same);
        return !found.empty();
    };
    const std::string source_path = file_identity(source);
    if (take([&](const RecordedComparison& candidate) { return candidate.source_path == source_path; })) {
        return found;
    }
    const std::string source_sampled = sampled_of(source).value_or("");
    if (!source_sampled.empty() &&
        take([&](const RecordedComparison& candidate) { return candidate.source_sampled == source_sampled; })) {
        return found;
    }
    if (std::none_of(candidates.begin(), candidates.end(),
                     [](const RecordedComparison& candidate) { return !candidate.source_video.empty(); })) {
        return found;
    }
    const std::string source_video = video_of(source).value_or("");
    if (!source_video.empty()) {
        take([&](const RecordedComparison& candidate) { return candidate.source_video == source_video; });
    }
    return found;
}

// A folder saved for the same test stream in another file: a remux of it
// (raw, MKV or MP4), with the same reference and recipe, read at the same
// frame rate (recipe_directory), of those `recorded` (recorded_comparisons).
// Nothing is read unless a comparison of the same recipe and rate recorded a
// stream; the whole test stream only where its first packets and the
// reference matched one.
std::optional<RecordedComparison> folder_of_same_stream(const RecordedList& recorded_list,
                                                          const std::filesystem::path& source,
                                                          const std::filesystem::path& distorted,
                                                          const ComparisonRecipe& recipe) {
    // A raw stream's recipe has its rate and a remux's has none: the rates
    // are compared as identity.json records them instead.
    const auto without_rate = [](nlohmann::json identity) {
        if (identity.is_object()) {
            identity.erase("test_rate");
        }
        return identity;
    };
    const nlohmann::json identity = without_rate(recipe.identity());
    const std::string rate = rate_text(recipe.test_rate);
    std::vector<RecordedComparison> candidates;
    for (const std::shared_ptr<const RecordedComparison>& recorded : recorded_list) {
        if (!rate.empty() && recorded->distorted_rate == rate && !recorded->distorted_head.empty() &&
            !recorded->distorted_stream.empty() && without_rate(recorded->recipe) == identity) {
            candidates.push_back(*recorded);
        }
    }
    if (candidates.empty()) {
        return std::nullopt;
    }
    const std::string head = stream_of(distorted, head_packets).value_or("");
    std::erase_if(candidates, [&](const RecordedComparison& candidate) {
        return head.empty() || candidate.distorted_head != head;
    });
    if (candidates.empty()) {
        return std::nullopt;  // another stream: told by its first packets
    }
    // Any of them saved with this reference, known by its path, its sampled
    // hash or its stream (each read once, and only where needed).
    const std::string source_path = file_identity(source);
    std::erase_if(candidates, [&](const RecordedComparison& candidate) {
        if (candidate.source_path == source_path) {
            return false;
        }
        const std::string sampled = sampled_of(source).value_or("");
        if (!sampled.empty() && candidate.source_sampled == sampled) {
            return false;
        }
        return candidate.source_video.empty() || video_of(source).value_or("") != candidate.source_video;
    });
    if (candidates.empty()) {
        return std::nullopt;
    }
    // The whole stream: as recorded already, or read where this thread's
    // lookups may (StreamReads), told first.
    std::optional<std::string> whole = stream_of(distorted, 0, false);
    if (!whole) {
        if (!lookup_reads.whole) {
            lookup_reads.whole_wanted = true;
            return std::nullopt;
        }
        if (lookup_reads.cancel && lookup_reads.cancel->cancelled()) {
            return std::nullopt;  // the lookup is over: nothing to read, nor to say
        }
        if (lookup_reads.announce) {
            lookup_reads.announce(distorted);
        }
        const auto started = std::chrono::steady_clock::now();
        whole = stream_of(distorted, 0);
        const std::string stream = whole.value_or("");
        const auto same = std::find_if(candidates.begin(), candidates.end(), [&stream](const RecordedComparison& c) {
            return !stream.empty() && c.distorted_stream == stream;
        });
        app_log::write(app_log::Level::info, "vml.metric_cache",
                       "Read " + platform::path_text(distorted.filename()) +
                           "'s whole stream to tell a remux of a scored test video: " +
                           format_fixed(std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                                            .count(),
                                        1) +
                           " s, " + (stream.empty() ? "not read" : same != candidates.end() ? "a remux" : "another encode"));
        return same == candidates.end() ? std::nullopt : std::optional(*same);
    }
    const std::string stream = whole.value_or("");
    for (const RecordedComparison& candidate : candidates) {
        if (!stream.empty() && candidate.distorted_stream == stream) {
            return candidate;
        }
    }
    return std::nullopt;
}

// The comparison of `recorded_list` saved for the same files under other
// paths.
std::optional<RecordedComparison> folder_of_same_files_in(const RecordedList& recorded_list,
                                                            const std::filesystem::path& source,
                                                            const std::filesystem::path& distorted,
                                                            const ComparisonRecipe& recipe) {
    const nlohmann::json identity = recipe.identity();
    const std::string distorted_path = file_identity(distorted);
    std::optional<std::string> distorted_sampled;
    std::vector<RecordedComparison> candidates;
    // The file first, the recipe only where it is the same: comparing the
    // recipes (JSON) of thousands of comparisons takes longer.
    for (const std::shared_ptr<const RecordedComparison>& recorded : recorded_list) {
        bool same = recorded->distorted_path == distorted_path;
        if (!same && !recorded->distorted_sampled.empty()) {
            if (!distorted_sampled) {
                distorted_sampled = sampled_of(distorted).value_or("");
            }
            same = !distorted_sampled->empty() && *distorted_sampled == recorded->distorted_sampled;
        }
        if (same && recorded->recipe == identity) {
            candidates.push_back(*recorded);
        }
    }
    if (!candidates.empty()) {
        const std::vector<RecordedComparison> same = with_same_reference(candidates, source);
        if (!same.empty()) {
            return same.front();
        }
    }
    // Not this file: perhaps its stream in another (a remux).
    return folder_of_same_stream(recorded_list, source, distorted, recipe);
}

// A folder saved for the same files under other paths (recipe_directory),
// as it was read.
std::optional<std::filesystem::path> folder_of_same_files(const std::filesystem::path& base,
                                                         const std::filesystem::path& source,
                                                         const std::filesystem::path& distorted,
                                                         const ComparisonRecipe& recipe) {
    // One found changed since it was read is read again and all looked at
    // again: a few times at most, should another process change them as fast.
    for (int tries = 0; tries < 4; ++tries) {
        const std::optional<RecordedComparison> found =
            folder_of_same_files_in(*recorded_comparisons(base), source, distorted, recipe);
        if (!found) {
            return std::nullopt;
        }
        if (as_read(*found)) {
            return found->folder;
        }
    }
    return std::nullopt;
}

}  // namespace

void set_video_fingerprinter(VideoFingerprinter fingerprinter) {
    KnownFiles& known = known_files();
    const std::lock_guard lock(known.mutex);
    known.fingerprinter = std::move(fingerprinter);
}

void fingerprint_ahead(const std::filesystem::path& file, const CancelToken& cancel) {
    video_of(file, cancel);
}

void set_stream_fingerprinter(StreamFingerprinter fingerprinter) {
    KnownFiles& known = known_files();
    const std::lock_guard lock(known.mutex);
    known.stream_fingerprinter = std::move(fingerprinter);
}

StreamReads::StreamReads(const CancelToken& cancel, bool whole,
                         std::function<void(const std::filesystem::path&)> announce) {
    lookup_reads.cancel = cancel;
    lookup_reads.whole = whole;
    lookup_reads.announce = std::move(announce);
}

StreamReads::~StreamReads() {
    lookup_reads.cancel.reset();
    lookup_reads.whole = false;
    lookup_reads.announce = nullptr;
}

bool take_whole_stream_wanted() {
    return std::exchange(lookup_reads.whole_wanted, false);
}

namespace {

// The folders of the comparisons saved for the file of identity `path` with
// no test stream recorded.
std::vector<std::filesystem::path> folders_without_stream(const std::filesystem::path& base,
                                                          const std::string& path) {
    // Those saved for it, as they are now (as_read): a few times at most,
    // should another process change them as fast.
    for (int tries = 0; tries < 4; ++tries) {
        std::vector<std::filesystem::path> folders;
        bool changed = false;
        // Held while it is looked through: in a range-for, the temporary
        // shared_ptr would be gone before the loop's body (C++20), and
        // another thread's listing would free the list under it.
        const std::shared_ptr<const RecordedList> listed = recorded_comparisons(base);
        for (const std::shared_ptr<const RecordedComparison>& recorded : *listed) {
            if (recorded->distorted_path == path) {
                changed = !as_read(*recorded) || changed;
                if (recorded->distorted_stream.empty()) {
                    folders.push_back(recorded->folder);
                }
            }
        }
        if (!changed) {
            return folders;
        }
    }
    return {};
}

}  // namespace

void saved_comparison_deleted(const std::filesystem::path& folder) {
    recorded_changed(folder);
}

bool test_stream_wanted(const std::filesystem::path& base, const std::filesystem::path& distorted) {
    return !folders_without_stream(base, file_identity(distorted)).empty();
}

bool record_test_stream(const std::filesystem::path& base, const std::filesystem::path& distorted,
                        const CancelToken& cancel) {
    const std::string path = file_identity(distorted);
    const std::vector<std::filesystem::path> folders = folders_without_stream(base, path);
    if (folders.empty()) {
        return false;  // nothing saved for it, or its stream recorded already: nothing read
    }
    const auto started = std::chrono::steady_clock::now();
    const std::optional<std::string> head = stream_of(distorted, head_packets, true, cancel);
    const std::optional<std::string> stream = head ? stream_of(distorted, 0, true, cancel) : std::nullopt;
    app_log::write(app_log::Level::info, "vml.metric_cache",
                   "Read " + platform::path_text(distorted.filename()) + "'s stream for a remux of it to keep its "
                       "saved scores: " +
                       format_fixed(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                                    1) +
                       " s" + (cancel.cancelled() ? ", stopped" : !stream ? ", not read" : ""));
    if (!head || !stream || file_identity(distorted) != path) {
        return false;  // not read, or the file changed meanwhile
    }
    bool recorded_any = false;
    for (const std::filesystem::path& folder : folders) {
        std::optional<nlohmann::json> identity = read_json(folder / "identity.json");
        if (!identity || !identity->is_object() || text_at(*identity, "distorted", "path") != path) {
            continue;  // saved again for another file meanwhile
        }
        (*identity)["distorted"]["stream_head"] = *head;
        (*identity)["distorted"]["stream"] = *stream;
        (*identity)["distorted"]["stream_format"] = stream_fingerprint_format;
        try {
            write_atomically(folder / "identity.json", python_json_compact(*identity));
            recorded_changed(folder);
            saved_index::changed(folder);
            recorded_any = true;
        } catch (const std::exception&) {
            // a record only: the scores stay found by the file's path and sampled hash
        }
    }
    return recorded_any;
}

void cache_parts::record_identity(const std::filesystem::path& directory, const std::filesystem::path& source,
                                  const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                                  bool only_if_missing) {
    std::error_code error;
    if (only_if_missing && std::filesystem::exists(directory / "identity.json", error)) {
        return;
    }
    // What this folder recorded before (a metric added since): the
    // reference's picture fingerprint is taken from it where it is of this
    // same file, rather than decoded again by each process's first save.
    const std::optional<nlohmann::json> before = read_json(directory / "identity.json");
    const std::string source_identity = file_identity(source);
    nlohmann::json source_side = {{"path", source_identity}, {"sampled", sampled_of(source).value_or("")}};
    const bool recorded_video = before && before->is_object() && text_at(*before, "source", "path") == source_identity &&
                                !text_at(*before, "source", "video").empty();
    if (recorded_video) {
        source_side["video"] = text_at(*before, "source", "video");
    } else if (const std::optional<std::string> video = video_of(source)) {
        source_side["video"] = *video;
    }
    nlohmann::json distorted_side = {{"path", file_identity(distorted)},
                                     {"sampled", sampled_of(distorted).value_or("")}};
    if (const std::string rate = rate_text(recipe.test_rate); !rate.empty()) {
        distorted_side["rate"] = rate;
    }
    // The test stream, for a remux of it to be known by, where it has been
    // read (a lookup that matched it): reading it takes as long as reading
    // the file, which record_test_stream does apart from the saving.
    std::optional<std::string> head = stream_of(distorted, head_packets, false);
    std::optional<std::string> stream = stream_of(distorted, 0, false);
    if (!head || !stream) {
        // Or as recorded before for this same file (a metric added since).
        if (before && before->is_object() && text_at(*before, "distorted", "path") == file_identity(distorted) &&
            !text_at(*before, "distorted", "stream").empty() && stream_of_this_format(*before)) {
            head = text_at(*before, "distorted", "stream_head");
            stream = text_at(*before, "distorted", "stream");
        }
    }
    if (head && stream) {
        distorted_side["stream_head"] = *head;
        distorted_side["stream"] = *stream;
        distorted_side["stream_format"] = stream_fingerprint_format;
    }
    const nlohmann::json identity = {
        {"format", 1}, {"source", std::move(source_side)}, {"distorted", std::move(distorted_side)}};
    try {
        write_atomically(directory / "identity.json", python_json_compact(identity));
        recorded_changed(directory);
    } catch (const std::exception&) {
        return;  // a record only: the scores are saved without it
    }
}

std::filesystem::path recipe_directory(const std::filesystem::path& base, const std::filesystem::path& source,
                                       const std::filesystem::path& distorted, const ComparisonRecipe& recipe) {
    const std::filesystem::path root = base / format_folder;
    const std::string source_identity = file_identity(source);
    const std::string distorted_identity = file_identity(distorted);
    const nlohmann::json identity = recipe.identity();
    const std::filesystem::path directory = root / hash_of(source_identity, distorted_identity, identity);
    std::error_code error;
    if (std::filesystem::exists(directory, error) || !std::filesystem::is_directory(root, error)) {
        return directory;
    }
    std::vector<std::string> algorithms{recipe.scale_algorithm};
    for (const std::string_view algorithm : scale_algorithms) {
        if (algorithm != recipe.scale_algorithm) {
            algorithms.emplace_back(algorithm);
        }
    }
    nlohmann::json legacy_identity = identity;
    if (recipe.resample_test) {
        // A resolution test's scores come from its own scaling algorithm:
        // only the folder of that one is its, under the identity before
        // tests had one.
        algorithms = {recipe.resample_test->scaler};
        legacy_identity["resample_test"].erase("scaler");
    }
    std::vector<std::filesystem::path> legacy;
    for (const std::string& algorithm : algorithms) {
        nlohmann::json old_identity = legacy_identity;
        old_identity["scale_algorithm"] = algorithm;
        const std::filesystem::path path = root / hash_of(source_identity, distorted_identity, old_identity);
        if (std::filesystem::is_directory(path, error)) {
            legacy.push_back(path);
        }
    }
    if (legacy.empty()) {
        // The same files saved under other paths: renamed, moved, copied, or
        // a remuxed reference.
        return folder_of_same_files(base, source, distorted, recipe).value_or(directory);
    }
    std::filesystem::rename(legacy.front(), directory, error);
    if (error) {
        return legacy.front();
    }
    recorded_changed(legacy.front());
    recorded_changed(directory);
    for (std::size_t i = 1; i < legacy.size(); ++i) {
        merge_legacy(legacy[i], directory);
        recorded_changed(legacy[i]);
        saved_index::changed(legacy[i]);
    }
    saved_index::changed(legacy.front());
    saved_index::changed(directory);
    return directory;
}

ColourTags colour_tags_of(const VideoInfo& video) {
    return {video.pix_fmt,     video.color_transfer, video.chroma_location, video.color_space,
            video.color_range, video.height,         video.color_primaries};
}

std::vector<ColourTags> colour_tags(const VideoInfo& source, const VideoInfo& distorted) {
    return {colour_tags_of(source), colour_tags_of(distorted)};
}

bool answers(const MetricRequestSpec& spec, const MetricProvenance& provenance, const std::vector<ColourTags>& infos) {
    return compatible(spec, provenance) && !stale_score(provenance, infos);
}

bool made_as_chosen(const std::string& key, const std::optional<std::string>& choice, const std::string& produced,
                    const std::function<bool(const std::string&)>& gpu_can_score) {
    if (choice == "cpu") {
        return produced == "cpu";
    }
    if (choice == "gpu" && produced == "cpu") {
        return !gpu_can_score(key);
    }
    return true;
}

std::filesystem::path metric_path(const std::filesystem::path& directory, const MetricRequestSpec& spec) {
    return own_metric_path(directory, saved_spec(spec));
}

bool run_takes(const std::filesystem::path& folder, const MetricRequestSpec& spec, const std::filesystem::path& file,
               const MetricProvenance& made, const std::optional<std::string>& choice,
               const std::function<bool(const std::string&)>& gpu_can_score, const std::vector<ColourTags>& infos) {
    return file.filename() == metric_path(folder, spec).filename() &&
           made_as_chosen(spec.key, choice, made.compute_backend, gpu_can_score) && answers(spec, made, infos);
}

std::vector<std::string> reusable_keys(const ComparisonResult& result, const std::vector<MetricRequestSpec>& specs,
                                       const std::map<std::string, std::string>& backends,
                                       const std::function<bool(const std::string&)>& gpu_can_score,
                                       const std::vector<ColourTags>& infos) {
    std::vector<std::string> kept;
#ifndef _WIN32
    // A score loaded without the av1_grain_dav1d mark (a run file's) of a pair
    // that must have it: its AV1 grain may be VideoToolbox's.
    DecodingMarks marks;
    add_grain_marks(marks, result.source_info, result.distorted_info);
    const auto unmarked = [&marks](const std::optional<bool>& mark) {
        return (marks.av1_grain || marks.av1_grain_unknown) && mark == false;
    };
#else
    const auto unmarked = [](const std::optional<bool>&) { return false; };
#endif
    for (const MetricRequestSpec& spec : specs) {
        if (const SequenceMetricResult* sequence = result.metric_results.sequence(spec.key)) {
            // One score for the video (CVVDP).
            if (std::isfinite(sequence->score) && !unmarked(sequence->av1_grain_dav1d) &&
                answers(spec, sequence->provenance, infos)) {
                kept.push_back(spec.key);
            }
            continue;
        }
        const FrameMetricResult* metric = result.metric_results.frame(spec.key);
        if (metric == nullptr || unmarked(metric->av1_grain_dav1d) ||
            std::all_of(metric->values.begin(), metric->values.end(), [](float value) { return std::isnan(value); })) {
            continue;
        }
        std::optional<std::string> choice;
        if (const auto found = backends.find(spec.key); found != backends.end()) {
            choice = found->second;
        }
        if (made_as_chosen(spec.key, choice, metric->provenance.compute_backend, gpu_can_score) &&
            answers(spec, metric->provenance, infos)) {
            kept.push_back(spec.key);
        }
    }
    return kept;
}

MetricResultSet reusable(const ComparisonResult& result, const std::vector<MetricRequestSpec>& specs,
                         const std::map<std::string, std::string>& backends,
                         const std::function<bool(const std::string&)>& gpu_can_score,
                         const std::vector<ColourTags>& infos) {
    MetricResultSet kept;
    for (const std::string& key : reusable_keys(result, specs, backends, gpu_can_score, infos)) {
        kept.copy_from(result.metric_results, key);
    }
    return kept;
}

std::filesystem::path store_result(const std::filesystem::path& base, const std::filesystem::path& source,
                                   const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                                   const ComparisonResult& result, const std::string& label,
                                   const std::vector<MetricRequestSpec>& specs) {
    const std::filesystem::path directory = recipe_directory(base, source, distorted, recipe);
    // Should this end half-way (a crash, a file that cannot be written), the
    // index reads the folder again at its next listing.
    saved_index::changing(directory);
    const std::string context = python_json_compact(context_json(result, label, recipe));
    forget_scores_of_another_picture(directory, nlohmann::json::parse(context));
    write_atomically(directory / "context.json", context);
    recorded_changed(directory);
    // A score saved under a request it does not answer would answer it from
    // then on: loading checks the saved request, not the score. A perceptual
    // score is saved under its own implementation's id (stored_spec), which
    // answers only requests that take that implementation's scores.
    const auto answers_request = [](const MetricRequestSpec& spec, const MetricProvenance& provenance) {
        if (spec.backend_id == "perceptual" || compatible(spec, provenance)) {
            return true;
        }
        app_log::write(app_log::Level::warning, "vml.metric_cache",
                       spec.key + " not saved: its score is " + provenance.implementation_compatibility_id +
                           "'s, the request asks for " + spec.implementation_compatibility_id + "'s");
        return false;
    };
    DecodingMarks marks =
        decoding_marks(result.source_info.codec_name, result.source_info.profile, result.distorted_info.codec_name,
                       result.distorted_info.profile);
    add_grain_marks(marks, result.source_info, result.distorted_info);
    // A kept score that says nothing of its pair's AV1 grain (a run file's,
    // which Windows takes) never replaces a saved one of the current format
    // that says it; any other it replaces as before.
    const auto overwrites_marks = [&](const MetricRequestSpec& spec, const MetricProvenance& provenance,
                                      const std::optional<bool>& mark) {
        if (!(marks.av1_grain || marks.av1_grain_unknown) || mark != false) {
            return false;
        }
        try {
            const auto saved = read_score(metric_path(directory, stored_spec(spec, provenance)));
            return saved && current_format(saved->first) &&
                   saved->first.value(av1_grain_dav1d, nlohmann::json()) == av1_grain_dav1d_version;
        } catch (const std::exception&) {
            return false;
        }
    };
    // A plain metric and its YUV metric's Y share a file (metric_path): it is
    // written once, from the score that has its scores as calculated where
    // one has (saves_exact), and the Y's own file of before goes.
    std::vector<const MetricRequestSpec*> ordered;
    for (const MetricRequestSpec& spec : specs) {
        ordered.push_back(&spec);
    }
    std::stable_partition(ordered.begin(), ordered.end(), [&result](const MetricRequestSpec* spec) {
        const FrameMetricResult* metric = result.metric_results.frame(spec->key);
        return metric != nullptr && !metric->exact.empty();
    });
    std::set<std::filesystem::path> written;
    for (const MetricRequestSpec* spec_of : ordered) {
        const MetricRequestSpec& spec = *spec_of;
        if (const FrameMetricResult* metric = result.metric_results.frame(spec.key)) {
            if (!answers_request(spec, metric->provenance) ||
                overwrites_marks(spec, metric->provenance, metric->av1_grain_dav1d)) {
                continue;
            }
            // A Y never replaces its plain metric's saved score, which another route may have made (a CPU
            // MS-SSIM, 1e-6 from the GPU's on a few frames), except to add the scores as calculated (the CPU's
            // PSNR and SSIM, the same either way).
            const std::filesystem::path path = metric_path(directory, stored_spec(spec, metric->provenance));
            const bool alias = saved_as(spec.key) != spec.key;
            const bool keep_saved = alias && metric->exact.empty() && std::filesystem::exists(path);
            if (!keep_saved && written.insert(path).second) {
                store_metric(directory, *metric, spec, marks);
            }
            if (alias) {
                std::error_code ignored;
                std::filesystem::remove(own_metric_path(directory, spec), ignored);
            }
        } else if (const SequenceMetricResult* sequence = result.metric_results.sequence(spec.key)) {
            if (answers_request(spec, sequence->provenance) &&
                !overwrites_marks(spec, sequence->provenance, sequence->av1_grain_dav1d)) {
                store_metric(directory, *sequence, spec, marks);
            }
        }
    }
    // What the files are, to know them by when their paths change.
    record_identity(directory, source, distorted, recipe, false);
    saved_index::changed(directory);
    return directory;
}

int clear_metrics(const std::filesystem::path& base, const std::filesystem::path& source,
                  const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                  const std::vector<MetricRequestSpec>& specs) {
    const std::filesystem::path directory = recipe_directory(base, source, distorted, recipe);
    saved_index::changing(directory);
    int removed = 0;
    const auto remove = [&removed](const std::filesystem::path& path) {
        std::error_code error;
        if (std::filesystem::remove(path, error)) {
            ++removed;
        }
    };
    for (const MetricRequestSpec& spec : specs) {
        if (is_auto_perceptual(spec)) {
            for (const auto& [path, concrete] : auto_perceptual_candidates(directory, spec)) {
                remove(path);
            }
            continue;
        }
        // A YUV metric's Y goes with the plain metric's file that holds it,
        // and with its own file of before (own_metric_path).
        const MetricRequestSpec saved = saved_spec(spec);
        if (saved.key == "vmaf" && saved.backend_id == "ffmpeg") {
            for (const auto& [path, concrete] : vmaf_equivalents(directory, saved)) {
                remove(path);
            }
        } else if (saved.key == "vmaf_v1" && saved.backend_id == "ffmpeg") {
            for (const auto& [path, concrete] : old_v1_scores(directory, saved)) {
                remove(path);
            }
        }
        remove(metric_path(directory, spec));
        if (saved.key != spec.key) {
            remove(own_metric_path(directory, spec));
        }
    }
    // The folder goes with its last score.
    std::error_code error;
    bool scores_left = false;
    for (std::filesystem::directory_iterator entry(directory, error);
         !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        // Not a save that a crash cut short (write_file_whole's ".name.N.npz" of old).
        scores_left = scores_left || (entry->path().extension() == ".npz" &&
                                      !platform::path_text(entry->path().filename()).starts_with('.'));
    }
    if (!error && !scores_left) {
        std::filesystem::remove_all(directory, error);
        recorded_changed(directory);
    }
    saved_index::changed(directory);
    return removed;
}

std::string lookup_token(const std::filesystem::path& source, const std::filesystem::path& distorted,
                         const AnalysisRequest& request) {
    nlohmann::json metrics = nlohmann::json::array();
    for (const MetricRequestSpec& spec : request.metrics) {
        metrics.push_back(spec.identity());
    }
    return sha256_hex(nlohmann::json{{"recipe", recipe_hash(source, distorted, request.recipe)},
                                     {"test_rate", rate_text(request.recipe.test_rate)},
                                     {"metrics", metrics}}
                          .dump());
}

namespace {

// The bitrate saved beside the scores (bitrate_store.h): saved results too.
std::uintmax_t saved_bitrate_bytes(const std::filesystem::path& base) {
    std::uintmax_t bytes = 0;
    std::error_code error;
    std::filesystem::directory_iterator bitrate(bitrate_store_folder(base), error);
    for (; !error && bitrate != std::filesystem::directory_iterator(); bitrate.increment(error)) {
        std::error_code ignored;
        if (const std::uintmax_t size = bitrate->file_size(ignored); !ignored) {
            bytes += size;
        }
    }
    return bytes;
}

}  // namespace

CacheSummary cache_summary(const std::filesystem::path& base) {
    CacheSummary summary;
    // From the index (saved_index.h), which knows each comparison's files.
    if (const std::optional<std::vector<IndexedComparison>> indexed = saved_index::comparisons(base)) {
        for (const IndexedComparison& comparison : *indexed) {
            ++summary.comparisons;
            summary.bytes += comparison.bytes;
            summary.saved.push_back({base / format_folder / platform::path_from_text(comparison.name),
                                     comparison.saved, comparison.bytes});
        }
        summary.bytes += saved_bitrate_bytes(base);
        return summary;
    }
    // Without one, walked by directory entries, which come with their sizes
    // and times on Windows: asking each file for them takes about ten times
    // as long.
    std::map<std::filesystem::path, CacheSummary::Comparison> comparisons;
    const auto file_now = std::filesystem::file_time_type::clock::now();
    const auto system_now = std::chrono::system_clock::now();
    std::error_code error;
    std::filesystem::recursive_directory_iterator walk(
        base / format_folder, std::filesystem::directory_options::skip_permission_denied, error);
    for (; !error && walk != std::filesystem::recursive_directory_iterator(); walk.increment(error)) {
        std::error_code ignored;
        if (!walk->is_regular_file(ignored)) {
            continue;
        }
        const std::uintmax_t size = walk->file_size(ignored);
        if (!ignored) {
            summary.bytes += size;
        }
        // A comparison's folder holds its context.json (<base>/v2/<hash>),
        // and its saved time is its newest file's.
        if (walk.depth() == 1) {
            CacheSummary::Comparison& comparison = comparisons[walk->path().parent_path()];
            comparison.bytes += ignored ? 0 : size;
            std::error_code untimed;
            const std::filesystem::file_time_type when = walk->last_write_time(untimed);
            if (!untimed) {
                // On the system clock, file_time_type's not being it everywhere
                // (as list_saved_comparisons carries a file's age over).
                comparison.saved = std::max(
                    comparison.saved,
                    system_now - std::chrono::duration_cast<std::chrono::system_clock::duration>(file_now - when));
            }
            if (walk->path().filename() == "context.json") {
                ++summary.comparisons;
                comparison.folder = walk->path().parent_path();  // a comparison, not a folder half-written
            }
        }
    }
    for (auto& [folder, comparison] : comparisons) {
        if (!comparison.folder.empty()) {
            summary.saved.push_back(std::move(comparison));
        }
    }
    std::sort(summary.saved.begin(), summary.saved.end(),
              [](const CacheSummary::Comparison& a, const CacheSummary::Comparison& b) { return a.saved > b.saved; });
    summary.bytes += saved_bitrate_bytes(base);
    return summary;
}

bool reference_where_saved(const std::filesystem::path& folder) {
    const std::optional<nlohmann::json> context = read_json(folder / "context.json");
    if (!context || !context->is_object() || !context->contains("source")) {
        return true;  // not a comparison this app can read: nothing to keep it for
    }
    const std::optional<nlohmann::json> identity = read_json(folder / "identity.json");
    const bool recorded = identity && identity->is_object();
    return reference_still_there(platform::path_from_text(python_str(context->at("source"))),
                                 recorded ? text_at(*identity, "source", "path") : std::string(),
                                 recorded ? text_at(*identity, "source", "sampled") : std::string());
}

bool reference_still_there(const std::filesystem::path& source, const std::string& recorded,
                           const std::string& sampled) {
    std::error_code error;
    if (source.empty() || !std::filesystem::is_regular_file(source, error)) {
        return false;
    }
    if (recorded.empty() || recorded == file_identity(source)) {
        return true;
    }
    // Touched, or copied back, since (another time, the same bytes): known
    // by its content, as the saved scores' lookup knows it (sampled_of,
    // read once for each of the file's times). Else the Videos tab would use
    // these scores while Saved Results called their reference gone.
    return !sampled.empty() && sampled_of(source) == sampled;
}

namespace {

// Deletes the bitrate saved beside the scores (bitrate_store.h), but that of
// the files of the comparisons in `kept` (by their identity.json's sampled
// hashes). The rest is read again from a file still there; a kept
// comparison's graphs have nothing else to draw it from (issue #134).
void clear_saved_bitrate(const std::filesystem::path& base, const std::vector<std::filesystem::path>& kept) {
    const std::filesystem::path folder = bitrate_store_folder(base);
    std::error_code error;
    if (kept.empty()) {
        std::filesystem::remove_all(folder, error);
        return;
    }
    std::set<std::string> wanted;
    for (const std::filesystem::path& comparison : kept) {
        for (const bool reference : {true, false}) {
            if (std::string hash = saved_content_hash(comparison, reference); !hash.empty()) {
                wanted.insert(std::move(hash));
            }
        }
    }
    std::vector<std::filesystem::path> unwanted;
    for (std::filesystem::directory_iterator entry(folder, error);
         !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        if (!wanted.contains(platform::path_text(entry->path().stem()))) {
            unwanted.push_back(entry->path());
        }
    }
    for (const std::filesystem::path& path : unwanted) {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
}

}  // namespace

ClearedResults clear_saved_results(const std::filesystem::path& base, bool keep_without_reference) {
    const std::filesystem::path root = base / format_folder;
    ClearedResults cleared;
    std::error_code error;
    if (!std::filesystem::exists(root, error)) {
        clear_saved_bitrate(base, {});
        return cleared;
    }
    std::vector<std::filesystem::path> kept;
    std::vector<std::filesystem::path> deleted;
    // What is no comparison (a folder half-written, or left by a delete cut
    // short) is deleted too, as it was with everything when nothing is kept:
    // no list shows it, nothing else deletes it.
    std::vector<std::filesystem::path> strays;
    // increment(error), not a range-for: its ++ throws if reading fails.
    for (std::filesystem::directory_iterator entry(root, error);
         !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        std::error_code ignored;
        if (!std::filesystem::is_regular_file(entry->path() / "context.json", ignored)) {
            strays.push_back(entry->path());
            continue;
        }
        if (keep_without_reference && !reference_where_saved(entry->path())) {
            ++cleared.kept;
            kept.push_back(entry->path());
        } else {
            deleted.push_back(entry->path());
        }
    }
    clear_saved_bitrate(base, kept);
    if (cleared.kept == 0) {
        std::filesystem::remove_all(root, error);  // everything, as before
        recorded_forgotten(base);
        cleared.removed = error ? 0 : static_cast<int>(deleted.size());
        return cleared;
    }
    bool all = !error;
    for (const std::filesystem::path& folder : deleted) {
        std::error_code not_removed;
        std::filesystem::remove_all(folder, not_removed);
        all = all && !not_removed;
    }
    for (const std::filesystem::path& stray : strays) {
        std::error_code not_removed;
        std::filesystem::remove_all(stray, not_removed);  // counted as no comparison either way
    }
    recorded_forgotten(base);
    cleared.removed = all ? static_cast<int>(deleted.size()) : 0;
    return cleared;
}

bool saved_file_at(const std::filesystem::path& folder, bool reference, const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        return false;
    }
    const std::optional<nlohmann::json> identity = read_json(folder / "identity.json");
    const std::string recorded =
        identity && identity->is_object() ? text_at(*identity, reference ? "source" : "distorted", "path")
                                          : std::string();
    return recorded.empty() || recorded == file_identity(path);
}

std::string saved_content_hash(const std::filesystem::path& folder, bool reference) {
    const std::optional<nlohmann::json> identity = read_json(folder / "identity.json");
    return identity && identity->is_object() ? text_at(*identity, reference ? "source" : "distorted", "sampled")
                                             : std::string();
}

}  // namespace vml
