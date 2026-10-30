// How a video's film grain is scored (film_grain_scoring.h): the choice and
// the grain nothing here adds as it should, what FFmpeg's decoding is told,
// which decoders may make the pictures, and what the saved scores record.
#include "vml/film_grain_scoring.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "vml/analysis_request.h"
#include "vml/film_grain_detect.h"
#include "vml/hardware_decoding.h"
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
    }
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
// cannot leave AV1's out, and Apple's is not known to add it.
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
