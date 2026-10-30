// How a video's film grain is scored (film_grain_scoring.h): the choice and
// the grain nothing here adds as it should, what FFmpeg's decoding is told,
// which decoders may make the pictures, and what the saved scores record.
#include "vml/film_grain_scoring.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <tuple>

#include "grain_decoding.h"
#include "vml/analysis_request.h"
#include "vml/film_grain_detect.h"
#include "vml/hardware_decoding.h"
#include "vml/results_summary.h"
#include "vml/score_identity.h"

namespace vml {
namespace {

VideoInfo grain_video(const std::string& codec, const std::string& pix_fmt, std::optional<unsigned> kinds,
                      std::optional<bool> with = std::nullopt) {
    VideoInfo info;
    info.path = "clip.mkv";
    info.codec_name = codec;
    info.pix_fmt = pix_fmt;
    info.width = 1920;
    info.height = 1080;
    info.film_grain = kinds;
    info.with_film_grain = with;
    return info;
}

constexpr unsigned h274 = film_grain_h274, model_1 = film_grain_h274_model_1, afgs1 = film_grain_afgs1,
                   av1 = film_grain_av1;

// Each kind of grain, codec, depth and choice: how it is scored, whether a
// reason is given for leaving it out, whether only the app's decoder may make
// its pictures, and what a saved score records (nothing where every score
// made before the choice existed scored it so).
TEST(FilmGrainScoring, EachVideoIsScoredWithOrWithoutItsGrainAsItCanBe) {
    const std::string h274_id(score_identity::film_grain_h274), afgs1_id(score_identity::film_grain_afgs1),
        left_out(score_identity::film_grain_left_out);
#ifdef _WIN32
    const std::optional<std::string> av1_id;  // every route added it before
#else
    const std::optional<std::string> av1_id = std::string(score_identity::film_grain_av1);
#endif
    struct Case {
        VideoInfo info;
        GrainScoring scoring;
        bool why;
        bool only_the_app;
        std::optional<std::string> identity;
    };
    const Case cases[] = {
        {grain_video("vvc", "yuv420p10le", std::nullopt), GrainScoring::none, false, false, std::nullopt},
        {grain_video("vvc", "yuv420p10le", 0u), GrainScoring::none, false, false, std::nullopt},
        {grain_video("vvc", "yuv420p10le", h274), GrainScoring::with, false, true, h274_id},
        {grain_video("vvc", "yuv420p10le", model_1, true), GrainScoring::with, false, true, h274_id},
        {grain_video("vvc", "yuv420p10le", h274, false), GrainScoring::without, false, false, std::nullopt},
        {grain_video("vvc", "yuv420p10le", model_1, false), GrainScoring::without, false, false, std::nullopt},
        {grain_video("vvc", "yuv420p", h274, false), GrainScoring::without, false, false, left_out},
        {grain_video("vvc", "yuv420p10le", h274 | afgs1), GrainScoring::without, true, false, left_out},
        {grain_video("vvc", "yuv420p10le", afgs1), GrainScoring::with, false, false, afgs1_id},
        {grain_video("hevc", "yuv420p10le", h274), GrainScoring::without, true, false, std::nullopt},
        {grain_video("hevc", "yuv420p", h274), GrainScoring::without, true, false, left_out},
        {grain_video("h264", "yuv420p", model_1), GrainScoring::without, true, false, std::nullopt},
        {grain_video("hevc", "yuv420p10le", afgs1), GrainScoring::with, false, false, afgs1_id},
        {grain_video("hevc", "yuv420p10le", afgs1, false), GrainScoring::without, false, false, left_out},
        {grain_video("h264", "yuv420p", afgs1), GrainScoring::without, true, false, std::nullopt},
        {grain_video("av1", "yuv420p10le", av1), GrainScoring::with, false, false, av1_id},
        {grain_video("av1", "yuv420p10le", av1, false), GrainScoring::without, false, false, left_out},
    };
    for (const Case& expected : cases) {
        SCOPED_TRACE(expected.info.codec_name + " " + expected.info.pix_fmt + " " +
                     std::to_string(expected.info.film_grain.value_or(99)) + " " +
                     (expected.info.with_film_grain ? (*expected.info.with_film_grain ? "with" : "without") : "-"));
        const FilmGrainScoring grain = effective_film_grain(expected.info);
        EXPECT_EQ(grain.scoring, expected.scoring);
        EXPECT_EQ(!grain.why_without.empty(), expected.why);
        EXPECT_EQ(only_the_app_adds(grain), expected.only_the_app);
        EXPECT_EQ(film_grain_identity(expected.info), expected.identity);
        // A choice to make: grain read, and none forced out.
        EXPECT_EQ(film_grain_choosable(expected.info), grain.scoring != GrainScoring::none && !expected.why);
    }
    // Forced out whatever is chosen, and said so.
    EXPECT_FALSE(effective_film_grain(grain_video("hevc", "yuv420p10le", h274, false)).why_without.empty());
    EXPECT_FALSE(film_grain_choosable(grain_video("hevc", "yuv420p10le", h274, false)));
    EXPECT_TRUE(film_grain_choosable(grain_video("vvc", "yuv420p10le", h274, false)));
    EXPECT_EQ(film_grain_default_choice("with"), true);
    EXPECT_EQ(film_grain_default_choice("without"), false);
    EXPECT_EQ(film_grain_default_choice(""), std::nullopt);  // ask
    EXPECT_EQ(film_grain_default_choice("With"), std::nullopt);
}

// The decoders the scores are made with and Video Compare's take one rule
// (grain_decoding.h): VVdeC for VVC's H.274 added, FFmpeg's decoders told to
// leave the grain out, dav1d's switch; and FFmpeg's pictures in the viewers
// leave out what it would not add as the scores have it.
TEST(FilmGrainScoring, EveryDecoderAddsOrLeavesOutTheGrainAsItIsScored) {
    const auto named = [](const GrainDecoding& decoding) {
        return decoding.decoder != nullptr ? std::string(decoding.decoder) : std::string();
    };
    using Options = std::vector<std::string>;
    const Options left_out{"-export_side_data", "film_grain"};
    struct Case {
        VideoInfo info;
        std::string decoder;
        bool export_grain;
        bool apply_grain;
        Options pictures;  // FFmpeg's, in Frame Compare and the playback fallback
    };
    const Case cases[] = {
        {grain_video("vvc", "yuv420p10le", h274), "libvvdec", false, true, left_out},  // FFmpeg cannot add it
        {grain_video("vvc", "yuv420p10le", model_1), "libvvdec", false, true, left_out},
        {grain_video("vvc", "yuv420p10le", h274, false), "", true, false, left_out},
        {grain_video("vvc", "yuv420p10le", afgs1), "", false, true, {}},  // FFmpeg's vvc adds AFGS1
        {grain_video("hevc", "yuv420p", h274), "", true, false, left_out},  // forced out
        {grain_video("av1", "yuv420p10le", av1), "", false, true, {}},
        {grain_video("av1", "yuv420p10le", av1, false), "", true, false, left_out},
        {grain_video("vvc", "yuv420p10le", 0u), "", false, true, {}},
        {grain_video("vvc", "yuv420p", std::nullopt), "", false, true, {}},  // not known: as decoders add it
    };
    for (const Case& expected : cases) {
        SCOPED_TRACE(expected.info.codec_name + " " + std::to_string(expected.info.film_grain.value_or(99)));
        const GrainDecoding got = video_grain_decoding(expected.info);
        EXPECT_EQ(named(got), expected.decoder);
        EXPECT_EQ(got.export_grain, expected.export_grain);
        EXPECT_EQ(got.apply_grain, expected.apply_grain);
        EXPECT_EQ(picture_grain_options(expected.info), expected.pictures);
    }
}

// VVdeC's own parse delay for the threads, (threads * 24) >> 4 at most 48;
// Video Compare's, half of it and at least 2; and scoring's, one per thread,
// at least 2 and at most 48 (grain_decoding.h).
TEST(FilmGrainScoring, VvdecParsesAheadAsTheRuleForEachUseSays) {
    for (const auto [threads, vvdec, video_compare, scoring] :
         {std::tuple(0u, 0u, 2u, 2u), std::tuple(1u, 1u, 2u, 2u), std::tuple(16u, 24u, 12u, 16u),
          std::tuple(24u, 36u, 18u, 24u), std::tuple(32u, 48u, 24u, 32u), std::tuple(64u, 48u, 24u, 48u)}) {
        EXPECT_EQ(vvdec_default_parse_delay(threads), vvdec) << threads << " threads";
        EXPECT_EQ(video_compare_parse_delay(threads), video_compare) << threads << " threads";
        EXPECT_EQ(scoring_parse_delay(threads), scoring) << threads << " threads";
    }
}

// The results' JSON and CSV record a video's film grain and how it was
// scored; a video without grain has nothing more than before.
TEST(FilmGrainScoring, TheResultsSayHowEachVideosGrainWasScored) {
    nlohmann::ordered_json entry;
    add_media(entry, grain_video("hevc", "yuv420p10le", 0u));
    EXPECT_FALSE(entry.contains("film_grain"));
    add_media(entry, grain_video("vvc", "yuv420p10le", h274, false));
    EXPECT_EQ(entry.at("film_grain"),
              (nlohmann::ordered_json{{"kinds", {"h274"}}, {"scored", "without"}}));
    add_media(entry, grain_video("hevc", "yuv420p10le", h274 | afgs1));
    EXPECT_EQ(entry.at("film_grain").at("scored"), "without");
    EXPECT_EQ(entry.at("film_grain").at("kinds"), (nlohmann::ordered_json{"h274", "afgs1"}));
    EXPECT_TRUE(entry.at("film_grain").contains("why_without"));
}

// FFmpeg's decoding is told to leave the grain out, and refuses to make the
// pictures of H.274's grain added: it adds it at 8 bits only, rounded down.
TEST(FilmGrainScoring, FfmpegLeavesTheGrainOutOrRefusesToAddH274s) {
    using Options = std::vector<std::string>;
    EXPECT_EQ(grain_video("vvc", "yuv420p10le", 0u).decode_options(), Options{});
    EXPECT_EQ(grain_video("vvc", "yuv420p10le", h274, false).decode_options(),
              (Options{"-export_side_data", "film_grain"}));
    EXPECT_EQ(grain_video("hevc", "yuv420p", h274).decode_options(), (Options{"-export_side_data", "film_grain"}));
    EXPECT_EQ(grain_video("hevc", "yuv420p10le", afgs1).decode_options(), Options{});  // FFmpeg 9 adds it
    EXPECT_EQ(grain_video("av1", "yuv420p10le", av1).decode_options(), Options{});
    EXPECT_THROW(grain_video("vvc", "yuv420p10le", h274).decode_options(), FilmGrainRefusal);
    EXPECT_EQ(grain_video("vvc", "yuv420p10le", h274).input_options(), Options{});  // stream copies, ffprobe
}

// No GPU decoder adds H.274's grain or AFGS1; AMD's in the app and Apple's
// cannot leave AV1's out, and Apple's is not known to add it. Video
// Compare's FFmpeg D3D11VA does both.
TEST(FilmGrainScoring, GrainNoGpuDecoderAddsOrLeavesOutIsDecodedInSoftware) {
    DecodingCapabilities pc;
    pc.hwaccels = {"cuda", "d3d11va"};
    pc.vendors = {GpuVendor::nvidia, GpuVendor::amd};
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, grain_video("hevc", "yuv420p10le", afgs1)), std::nullopt);
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, grain_video("hevc", "yuv420p10le", afgs1, false)), "cuda");
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, grain_video("hevc", "yuv420p10le", h274)), "cuda");  // left out
    EXPECT_EQ(pick_decode(pc, GpuVendor::nvidia, grain_video("av1", "yuv420p10le", av1, false)), "cuda");
    EXPECT_EQ(pick_decode(pc, GpuVendor::amd, grain_video("av1", "yuv420p10le", av1)), "d3d11va");
    EXPECT_EQ(pick_decode(pc, GpuVendor::amd, grain_video("av1", "yuv420p10le", av1, false)), std::nullopt);
    EXPECT_EQ(software_decoding_reason(grain_video("av1", "yuv420p10le", av1, false), true),
              "no GPU decoder here for av1 that leaves its film grain out (AMD's and Apple's cannot)");
    EXPECT_FALSE(grain_needs_software_decoding(grain_video("av1", "yuv420p10le", av1, false), "d3d11va"));
    EXPECT_TRUE(grain_needs_software_decoding(grain_video("av1", "yuv420p10le", av1, false), "videotoolbox"));
    EXPECT_TRUE(grain_needs_software_decoding(grain_video("vvc", "yuv420p10le", h274), "d3d11va"));
    DecodingCapabilities mac;
    mac.hwaccels = {"videotoolbox"};
    mac.videotoolbox_codecs = {"av1"};
    EXPECT_EQ(pick_decode(mac, GpuVendor::automatic, grain_video("av1", "yuv420p10le", 0u), MacGpuDecoding::on),
              "videotoolbox");
    EXPECT_EQ(pick_decode(mac, GpuVendor::automatic, grain_video("av1", "yuv420p10le", av1), MacGpuDecoding::on),
              std::nullopt);
}

// A comparison of videos with no grain keeps the identity its saved scores
// were filed under; one with grain records each side's, so that its scores
// with and without the grain are saved apart.
TEST(FilmGrainScoring, SavedScoresAreFiledUnderEachSidesGrain) {
    const VideoInfo plain = grain_video("hevc", "yuv420p10le", 0u);
    const auto recipe = [](const VideoInfo& source, const VideoInfo& distorted) {
        return analysis_request(VmafOptions{}, {"psnr"}, {}, CvvdpSettings{}, source, distorted).recipe.identity();
    };
    const nlohmann::json before = recipe(VideoInfo{}, VideoInfo{});
    EXPECT_FALSE(before.contains("film_grain"));
    EXPECT_EQ(recipe(plain, grain_video("vvc", "yuv420p10le", h274, false)), before);  // as every run scored it
    const nlohmann::json with = recipe(plain, grain_video("vvc", "yuv420p10le", h274));
    EXPECT_EQ(with.at("film_grain"),
              (nlohmann::json{{"source", nullptr}, {"distorted", std::string(score_identity::film_grain_h274)}}));
    EXPECT_EQ(recipe(grain_video("av1", "yuv420p10le", av1, false), plain).at("film_grain"),
              (nlohmann::json{{"source", std::string(score_identity::film_grain_left_out)}, {"distorted", nullptr}}));
    nlohmann::json without_grain = with;
    without_grain.erase("film_grain");
    EXPECT_EQ(without_grain, before);  // nothing else changes
}

}  // namespace
}  // namespace vml
