// The saved scores, held to the Python app's (reference/metric_cache.json,
// and the comparison folder reference/metric_cache it saved): the names
// requests and recipes are filed under, the files a stored result writes
// (with each score's version, which the Python app left blank but for VMAF's),
// which scores answer a request, what requests load from the Python app's
// files, and the VMAF model a loaded result has.
#include "vml/metric_cache.h"
#include "vml/colour.h"
#include "vml/crop_detect.h"
#include "vml/cvvdp.h"
#include "vml/film_grain_detect.h"

#include <gtest/gtest.h>

#ifdef _WIN32
#include <share.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/build_info.h"
#include "vml/ffmpeg_run.h"
#include "vml/npz.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/result_json.h"
#include "vml/saved_results.h"
#include "vml/bitrate_store.h"
#include "vml/file_fingerprint.h"
#include "vml/score_identity.h"
#include "vml/sha256.h"

namespace vml {
namespace {

VmafOptions options_from(const nlohmann::json& fields) {
    VmafOptions options;
    options.n_subsample = fields.at("n_subsample").get<int>();
    options.model_choice = fields.at("model_choice").get<std::string>();
    options.model_choice_v1 = fields.at("model_choice_v1").get<std::string>();
    options.crop_mode = fields.at("crop_mode") == "none"     ? CropMode::none
                        : fields.at("crop_mode") == "source" ? CropMode::source_only
                                                             : CropMode::automatic;
    options.scale_direction = fields.at("scale_direction") == "distorted_to_source" ? ScaleDirection::distorted_to_source
                                                                                    : ScaleDirection::source_to_distorted;
    options.scale_algorithm = fields.at("scale_algorithm").get<std::string>();
    options.range_end = static_cast<long long>(*test::reference_number(fields.at("duration_limit")));
    if (const nlohmann::json& target = fields.at("resample_test"); !target.is_null()) {
        options.resample_test = ResampleTarget{target.at("width"), target.at("label")};
    }
    options.vship_build = vship_build_from(fields.value("vship_build", std::string("vship-fast")));
    return options;
}

AnalysisRequest request_from(const nlohmann::json& item) {
    std::vector<std::string> keys = item.at("metrics");
    std::map<std::string, std::string> backends = item.at("backends");
    const CvvdpSettings cvvdp =
        item.at("cvvdp").is_null() ? CvvdpSettings{} : CvvdpSettings::from_parameters(item.at("cvvdp"));
    return analysis_request(options_from(item.at("options")), keys, backends, cvvdp, VideoInfo{}, VideoInfo{});
}

std::optional<std::string> text_or_none(const nlohmann::json& data) {
    if (data.is_null()) {
        return std::nullopt;
    }
    return data.get<std::string>();
}

void touch(const std::filesystem::path& path, const std::string& text) { std::ofstream(path, std::ios::binary) << text; }

// The Python app's comparison folder (reference/metric_cache), laid where
// this app files the comparison of `folder`'s s.mp4 and t.mp4 under `base`.
void lay_saved_scores(const test::Folder& folder, const std::filesystem::path& base,
                      const ComparisonRecipe& recipe) {
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const std::filesystem::path directory = recipe_directory(base, folder / "s.mp4", folder / "t.mp4", recipe);
    std::filesystem::create_directories(directory);
    const std::filesystem::path from = std::filesystem::path(VML_REFERENCE_DIR) / "metric_cache";
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(from)) {
        std::filesystem::copy_file(entry.path(), directory / entry.path().filename());
    }
}

// A loaded result holds the scores `expected` lists (as metric_cache.json
// lists them), with their frames and provenance, and the label saved.
void expect_loaded(const std::optional<std::pair<ComparisonResult, std::string>>& loaded,
                   const nlohmann::json& expected, const std::string& label) {
    if (expected.empty()) {
        EXPECT_FALSE(loaded.has_value());
        return;
    }
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->second, label);
    const MetricResultSet& results = loaded->first.metric_results;
    ASSERT_EQ(results.keys().size(), expected.size());
    for (const nlohmann::json& metric : expected) {
        const std::string key = metric.at("key");
        SCOPED_TRACE(key);
        const MetricProvenance wanted = test::provenance_from(metric.at("provenance"));
        const MetricProvenance* provenance = nullptr;
        if (metric.at("kind") == "sequence") {
            const SequenceMetricResult* sequence = results.sequence(key);
            ASSERT_NE(sequence, nullptr);
            EXPECT_EQ(sequence->score, *test::reference_number(metric.at("score")));
            EXPECT_EQ(sequence->values, metric.contains("values") ? test::floats_from(metric.at("values")) : std::vector<float>{});
            provenance = &sequence->provenance;
        } else {
            const FrameMetricResult* frame = results.frame(key);
            ASSERT_NE(frame, nullptr);
            EXPECT_EQ(frame->frame, metric.at("frame").get<std::vector<int>>());
            EXPECT_EQ(frame->time, test::reference_numbers(metric.at("time")));
            EXPECT_EQ(frame->values, test::floats_from(metric.at("values")));
            provenance = &frame->provenance;
        }
        EXPECT_EQ(provenance->implementation, wanted.implementation);
        EXPECT_EQ(provenance->implementation_version, wanted.implementation_version);
        EXPECT_EQ(provenance->compute_backend, wanted.compute_backend);
        EXPECT_EQ(provenance->implementation_compatibility_id, wanted.implementation_compatibility_id);
        EXPECT_EQ(provenance->parameters, wanted.parameters);
    }
}

CvvdpSettings cvvdp_from(const nlohmann::json& item) {
    return item.at("cvvdp").is_null() ? CvvdpSettings{} : CvvdpSettings::from_parameters(item.at("cvvdp"));
}

// Another scaling algorithm files a comparison apart, the default keeping the
// identity saved comparisons have (FilesRequestsAndRecipesAsThePythonApp), but
// not a resolution test, whose algorithm is its own.
TEST(MetricCache, TheScalingAlgorithmFilesAComparisonApart) {
    const auto identity_of = [](const std::string& algorithm, std::optional<ResampleTarget> resample_test) {
        VmafOptions options;
        options.scale_algorithm = algorithm;
        options.resample_test = std::move(resample_test);
        return analysis_request(options, {"psnr"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{}).recipe.identity();
    };
    EXPECT_NE(identity_of("lanczos", std::nullopt), identity_of("bicubic", std::nullopt));
    const ResampleTarget target{1280, "720p"};
    EXPECT_EQ(identity_of("lanczos", target), identity_of("bicubic", target));
}

// Every request's identity and hash, and the recipe's, are the Python app's
// to the byte: a saved score is found by them.
TEST(MetricCache, FilesRequestsAndRecipesAsThePythonApp) {
    const nlohmann::json reference = test::load_reference("metric_cache.json");
    for (const nlohmann::json& item : reference.at("requests")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const AnalysisRequest request = request_from(item);
        const nlohmann::json identity = request.recipe.identity();
        EXPECT_EQ(python_json_compact(identity), item.at("recipe").get<std::string>());
        Sha256 hash;
        const std::string named = python_json_compact({{"source", "S"}, {"distorted", "D"}, {"recipe", identity}});
        hash.update(named.data(), named.size());
        EXPECT_EQ(hash.hexdigest(), item.at("recipe_hash").get<std::string>());
        EXPECT_EQ(nlohmann::json(request.perceptual_backends), item.at("perceptual_backends"));
        ASSERT_EQ(request.metrics.size(), item.at("specs").size());
        for (std::size_t i = 0; i < request.metrics.size(); ++i) {
            EXPECT_EQ(python_json_compact(request.metrics[i].identity()), item.at("specs")[i].at("identity").get<std::string>());
            EXPECT_EQ(metric_identity_hash(request.metrics[i]), item.at("specs")[i].at("hash").get<std::string>());
        }
    }
}

// Black bars in the source only are a recipe of their own ("source"), so its scores are saved apart from Auto's and
// None's, whose identities stay as they were.
TEST(MetricCache, SourceOnlyBlackBarsAreARecipeOfTheirOwn) {
    ComparisonRecipe recipe;
    EXPECT_EQ(recipe.identity().at("crop_mode"), "auto");
    recipe.crop_mode = CropMode::none;
    EXPECT_EQ(recipe.identity().at("crop_mode"), "none");
    recipe.crop_mode = CropMode::source_only;
    EXPECT_EQ(recipe.identity().at("crop_mode"), "source");
}

// Crops entered by hand file a comparison by their lines, apart from every other crop; the crops a row keeps while
// it is not Manual leave its identity as it was.
TEST(MetricCache, ManualCropsAreARecipeOfTheirOwnByTheirLines) {
    const auto identity_of = [](CropMode mode, CropEdges source_crop) {
        VmafOptions options;
        options.crop_mode = mode;
        options.manual_source_crop = source_crop;
        return analysis_request(options, {"psnr"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{}).recipe.identity();
    };
    const nlohmann::json manual = identity_of(CropMode::manual, {22, 22, 0, 0});
    EXPECT_EQ(manual.at("crop_mode"), "manual");
    EXPECT_EQ(manual.at("manual_source_crop"), (nlohmann::json{{"top", 22}, {"bottom", 22}, {"left", 0}, {"right", 0}}));
    EXPECT_NE(manual, identity_of(CropMode::manual, {20, 20, 0, 0}));
    EXPECT_EQ(identity_of(CropMode::automatic, {22, 22, 0, 0}), identity_of(CropMode::automatic, {}));
}

// A range's end is in its identity under the duration limit's name and its
// start only when it has one, so that a duration limit's comparison (the
// range from 0) keeps its saved scores' name.
TEST(MetricCache, ARangesStartIsInItsIdentityOnlyWhenItHasOne) {
    ComparisonRecipe recipe;
    recipe.range_end = 30;
    EXPECT_EQ(recipe.identity().at("duration_limit"), 30.0);
    EXPECT_FALSE(recipe.identity().contains("range_start"));
    recipe.range_start = 10;
    EXPECT_EQ(recipe.identity().at("range_start"), 10.0);
}

// A resolution test's scaling algorithm makes its test video: another one is
// another test, with an identity, a path and a saved target of its own,
// while the default one keeps those of the tests saved before (their
// identities are FilesRequestsAndRecipesAsThePythonApp's).
TEST(MetricCache, AResolutionTestsScalingAlgorithmMakesAnotherTest) {
    ComparisonRecipe bicubic;
    bicubic.resample_test = ResampleTarget{1920, "1080p"};
    ComparisonRecipe lanczos = bicubic;
    lanczos.resample_test->scaler = "lanczos";
    EXPECT_NE(lanczos.identity(), bicubic.identity());
    EXPECT_EQ(synthetic_resample_distorted_path("film.mkv", *bicubic.resample_test),
              std::filesystem::path("film [downscale-1080p-upscale].mkv"));
    EXPECT_EQ(synthetic_resample_distorted_path("film.mkv", *lanczos.resample_test),
              std::filesystem::path("film [downscale-1080p-lanczos-upscale].mkv"));
    EXPECT_EQ(resample_target_to_dict(bicubic.resample_test).dump(), R"({"width":1920,"label":"1080p"})");
    for (const ComparisonRecipe& recipe : {bicubic, lanczos}) {
        EXPECT_EQ(resample_target_from_dict(resample_target_to_dict(recipe.resample_test)), *recipe.resample_test);
    }
    nlohmann::json unknown = resample_target_to_dict(lanczos.resample_test);
    unknown["scaler"] = "../x";  // it goes into a path and FFmpeg's filters
    EXPECT_THROW(resample_target_from_dict(unknown), std::invalid_argument);
}

// A test video added again with the other film grain choice is a row of its
// own, under a made-up path its saved run is known by again.
TEST(MetricCache, AFilmGrainTwinsPathSaysItsChoice) {
    const std::filesystem::path film("encodes/film.mkv");
    const std::filesystem::path without = synthetic_film_grain_variant_path(film, false);
    EXPECT_EQ(without, std::filesystem::path("encodes/film [without film grain].mkv"));
    EXPECT_EQ(film_grain_variant_choice(without, film), false);
    EXPECT_EQ(film_grain_variant_choice(synthetic_film_grain_variant_path(film, true), film), true);
    EXPECT_EQ(film_grain_variant_choice(film, film), std::nullopt);
    EXPECT_EQ(film_grain_variant_choice(without, "encodes/other.mkv"), std::nullopt);
    EXPECT_EQ(film_grain_variant_choice(synthetic_scale_direction_variant_path(film, ScaleDirection::distorted_to_source),
                                        film),
              std::nullopt);
}

// A row scoring CVVDP on a further display is known by a made-up path its
// saved run leads back from to the row it belongs to, the film grain twin's
// included; characters a file name cannot hold become '-'.
TEST(MetricCache, ADisplayCompanionsPathLeadsBackToItsRow) {
    const std::filesystem::path film("encodes/film.mkv");
    const std::filesystem::path on_hdr = synthetic_cvvdp_display_variant_path(film, "HDR TV: 1000 nits / P3?");
    EXPECT_EQ(on_hdr, std::filesystem::path("encodes/film [CVVDP on HDR TV- 1000 nits - P3-].mkv"));
    EXPECT_EQ(cvvdp_display_variant_parent(on_hdr), film);
    const std::filesystem::path twin = synthetic_film_grain_variant_path(film, false);
    EXPECT_EQ(cvvdp_display_variant_parent(synthetic_cvvdp_display_variant_path(twin, "sdr")), twin);
    EXPECT_EQ(cvvdp_display_variant_parent(film), std::nullopt);
    EXPECT_EQ(cvvdp_display_variant_parent(twin), std::nullopt);
    EXPECT_EQ(cvvdp_display_variant_parent("encodes/film [CVVDP on sdr] cut.mkv"), std::nullopt);
}

// A duration limit's scores, saved before ranges under the name of the range
// from 0, answer that range when they reach its end (VMAF with the frame
// after it, which its last frame's motion looked at), cut to it, CVVDP's
// seconds in range pooled again; a ranged run's own (range_frames) answer as
// they are. At 2 fps, the range [0, 1 s) is frames 0 and 1.
TEST(MetricCache, ADurationLimitsScoresAnswerTheRangeFromZeroWhenTheyReachItsEnd) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const AnalysisRequest request = request_from(stored);
    ComparisonResult four = test::result_from(stored.at("result"));  // frames 0 to 3
    for (VideoInfo* info : {&four.source_info, &four.distorted_info}) {
        info->fps = 2.0;
        info->nb_frames = 4;
    }
    SequenceMetricResult cvvdp = *four.metric_results.sequence("cvvdp");
    cvvdp.frame = {0, 2};  // its two seconds at 2 fps
    const auto first_frames = [&](std::size_t count) {
        ComparisonResult result = four;
        result.metric_results = MetricResultSet();
        for (const char* key : {"vmaf", "psnr", "ssimulacra2"}) {
            FrameMetricResult metric = *four.metric_results.frame(key);
            metric.frame.resize(count);
            metric.time.resize(count);
            metric.values.resize(count);
            result.metric_results.add(metric);
        }
        result.metric_results.add(cvvdp);
        return result;
    };
    const auto load = [&](const std::string& cache, long long end, const ComparisonResult& saved) {
        ComparisonRecipe recipe = request.recipe;
        recipe.range_end = end;
        store_result(folder / cache, folder / "s.mp4", folder / "t.mp4", recipe, saved, "t", request.metrics);
        const auto loaded = load_result(folder / cache, folder / "s.mp4", folder / "t.mp4", recipe, request.metrics,
                                        request.perceptual_backends);
        return loaded ? loaded->first.metric_results : MetricResultSet();
    };

    const MetricResultSet whole = load("four", 1, first_frames(4));
    ASSERT_NE(whole.frame("vmaf"), nullptr);
    EXPECT_EQ(whole.frame("vmaf")->frame, (std::vector<int>{0, 1}));
    EXPECT_EQ(whole.frame("psnr")->frame, (std::vector<int>{0, 1}));
    ASSERT_NE(whole.sequence("cvvdp"), nullptr);
    EXPECT_EQ(whole.sequence("cvvdp")->frame, (std::vector<int>{0}));
    EXPECT_FLOAT_EQ(static_cast<float>(whole.sequence("cvvdp")->score), cvvdp.values.front());  // one second's JOD

    const MetricResultSet two = load("two", 1, first_frames(2));
    EXPECT_FALSE(two.has("vmaf"));  // its last frame's motion had no next frame
    EXPECT_TRUE(two.has("psnr"));

    const MetricResultSet three = load("three", 2, first_frames(3));  // the range [0, 2 s) is all four frames
    EXPECT_FALSE(three.has("psnr"));
    EXPECT_TRUE(three.has("cvvdp"));  // its run decoded the four frames within the limit

    ComparisonResult own = first_frames(2);
    FrameMetricResult vmaf = *own.metric_results.frame("vmaf");
    vmaf.provenance.parameters["range_frames"] = "[0,2]";
    own.metric_results = MetricResultSet();
    own.metric_results.add(vmaf);
    EXPECT_EQ(load("own", 1, own).frame("vmaf")->frame, (std::vector<int>{0, 1}));
    // ... and never answer for the whole videos, wherever they were saved.
    EXPECT_FALSE(load("own whole", 0, own).has("vmaf"));

    // A range from a later second: only a ranged run's scores are of it (none
    // was saved before ranges), so one without its mark is not taken.
    ComparisonRecipe later = request.recipe;
    later.range_start = 1;
    later.range_end = 2;
    own.metric_results.add(*first_frames(2).metric_results.frame("psnr"));  // no mark
    store_result(folder / "later", folder / "s.mp4", folder / "t.mp4", later, own, "t", request.metrics);
    const auto loaded_later = load_result(folder / "later", folder / "s.mp4", folder / "t.mp4", later, request.metrics,
                                          request.perceptual_backends);
    ASSERT_TRUE(loaded_later);
    EXPECT_TRUE(loaded_later->first.metric_results.has("vmaf"));
    EXPECT_FALSE(loaded_later->first.metric_results.has("psnr"));
}

// A range takes the whole videos' saved scores, cut to it, where it has none
// of its own: the frames in it, CVVDP's seconds in it pooled again (the last
// second of the whole run only where its run saved how far it reached),
// marked as the range's own, and the range's frame count. Not for the metrics
// a Recalculate forgot, not from a whole run with other black bars, not a
// subsampled score. At 2 fps, the range [1 s, 2 s) is frames 2 and 3.
TEST(MetricCache, ARangeTakesTheWholeVideosScoresCutToIt) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const AnalysisRequest request = request_from(stored);
    ComparisonResult whole = test::result_from(stored.at("result"));  // frames 0 to 3
    for (VideoInfo* info : {&whole.source_info, &whole.distorted_info}) {
        info->fps = 2.0;
        info->nb_frames = 4;
    }
    SequenceMetricResult cvvdp = *whole.metric_results.sequence("cvvdp");
    cvvdp.frame = {0, 2};  // its two seconds
    whole.metric_results.add(cvvdp);
    ComparisonResult reaching = whole;
    cvvdp.provenance.parameters["cvvdp_frames_reached"] = "4";
    reaching.metric_results.add(cvvdp);
    const auto ranged = [&request](long long start, long long end) {
        ComparisonRecipe recipe = request.recipe;
        recipe.range_start = start;
        recipe.range_end = end;
        return recipe;
    };
    const auto load = [&](const std::string& cache, const ComparisonRecipe& recipe,
                          const std::set<std::string>& again = {}) {
        return load_result(folder / cache, folder / "s.mp4", folder / "t.mp4", recipe, request.metrics,
                           request.perceptual_backends, {}, again);
    };
    store_result(folder / "old", folder / "s.mp4", folder / "t.mp4", request.recipe, whole, "t", request.metrics);
    store_result(folder / "new", folder / "s.mp4", folder / "t.mp4", request.recipe, reaching, "t", request.metrics);

    const auto later = load("new", ranged(1, 2));
    ASSERT_TRUE(later);
    const MetricResultSet& cut = later->first.metric_results;
    ASSERT_NE(cut.frame("vmaf"), nullptr);
    EXPECT_EQ(cut.frame("vmaf")->frame, (std::vector<int>{2, 3}));
    EXPECT_EQ(cut.frame("vmaf")->values,
              (std::vector<float>(whole.metric_results.frame("vmaf")->values.begin() + 2,
                                  whole.metric_results.frame("vmaf")->values.end())));
    EXPECT_EQ(cut.frame("psnr")->frame, (std::vector<int>{2, 3}));
    EXPECT_EQ(cut.frame("vmaf")->provenance.parameters.at("range_frames"), "[2,4]");
    ASSERT_NE(cut.sequence("cvvdp"), nullptr);
    EXPECT_EQ(cut.sequence("cvvdp")->frame, (std::vector<int>{2}));
    EXPECT_EQ(cut.sequence("cvvdp")->score, pool_cvvdp_windows({{2, 2, cvvdp.values[1]}}));
    EXPECT_EQ(cut.sequence("cvvdp")->provenance.parameters.at("range_frames"), "[2,4]");
    EXPECT_FALSE(cut.sequence("cvvdp")->provenance.parameters.contains("cvvdp_frames_reached"));
    EXPECT_EQ(later->first.compared_frame_count, 2);
    // A range of milliseconds too: [0.9 s, 2 s) is frames 2 and 3.
    ComparisonRecipe milliseconds = ranged(900, 2000);
    milliseconds.range_in_milliseconds = true;
    const auto in_milliseconds = load("new", milliseconds);
    ASSERT_TRUE(in_milliseconds && in_milliseconds->first.metric_results.frame("vmaf"));
    EXPECT_EQ(in_milliseconds->first.metric_results.frame("vmaf")->frame, (std::vector<int>{2, 3}));

    // The metrics found (none when nothing is).
    const auto keys = [&](const std::string& cache, const ComparisonRecipe& recipe,
                          const std::set<std::string>& again = {}) {
        const auto loaded = load(cache, recipe, again);
        return loaded ? loaded->first.metric_results.keys() : std::vector<std::string>();
    };
    const auto has = [](const std::vector<std::string>& found, const std::string& key) {
        return std::find(found.begin(), found.end(), key) != found.end();
    };
    // Saved before the reach was: the last second's length is not known.
    EXPECT_TRUE(has(keys("old", ranged(1, 2)), "vmaf"));
    EXPECT_FALSE(has(keys("old", ranged(1, 2)), "cvvdp"));
    EXPECT_FALSE(has(keys("old", ranged(1, 0)), "cvvdp"));  // to the end
    const auto to_the_end = load("new", ranged(1, 0));
    ASSERT_TRUE(to_the_end && to_the_end->first.metric_results.frame("psnr"));
    EXPECT_EQ(to_the_end->first.metric_results.frame("psnr")->frame, (std::vector<int>{2, 3}));
    // The first second ends where the second begins.
    const auto first = load("old", ranged(0, 1));
    ASSERT_TRUE(first && first->first.metric_results.frame("vmaf") && first->first.metric_results.sequence("cvvdp"));
    EXPECT_EQ(first->first.metric_results.frame("vmaf")->frame, (std::vector<int>{0, 1}));
    EXPECT_EQ(first->first.metric_results.sequence("cvvdp")->score, pool_cvvdp_windows({{0, 2, cvvdp.values[0]}}));

    // A Recalculate's metrics are not taken from the whole videos.
    EXPECT_FALSE(has(keys("new", ranged(1, 2), {"vmaf"}), "vmaf"));
    EXPECT_TRUE(has(keys("new", ranged(1, 2), {"vmaf"}), "psnr"));

    // The range's own comparison cut other black bars: the whole run's
    // scores are of another comparison; its own score is found.
    ComparisonResult own = whole;
    own.source_crop = CropBox{100, 100, 2, 2};
    own.metric_results = MetricResultSet();
    FrameMetricResult own_psnr = *cut.frame("psnr");  // the range's frames, marked as its own
    own.metric_results.add(own_psnr);
    store_result(folder / "new", folder / "s.mp4", folder / "t.mp4", ranged(1, 2), own, "t", request.metrics);
    EXPECT_EQ(keys("new", ranged(1, 2)), (std::vector<std::string>{"psnr"}));

    // A subsampled whole run (every other frame) is not cut.
    std::vector<MetricRequestSpec> every_other = request.metrics;
    for (MetricRequestSpec& spec : every_other) {
        if (spec.coverage) {
            spec.coverage = FrameCoverage{"sampled", 2};
        }
    }
    ComparisonResult sampled = whole;
    FrameMetricResult psnr = *whole.metric_results.frame("psnr");
    psnr.frame = {0, 2};
    psnr.time = {0.0, 1.0};
    psnr.values.resize(2);
    sampled.metric_results = MetricResultSet();
    sampled.metric_results.add(psnr);
    store_result(folder / "sampled", folder / "s.mp4", folder / "t.mp4", request.recipe, sampled, "t", every_other);
    ASSERT_TRUE(load_result(folder / "sampled", folder / "s.mp4", folder / "t.mp4", request.recipe, every_other,
                            request.perceptual_backends));  // saved, and found for the whole videos
    EXPECT_FALSE(load_result(folder / "sampled", folder / "s.mp4", folder / "t.mp4", ranged(1, 2), every_other,
                             request.perceptual_backends));
}

// Black bars Source only takes Auto's saved scores where Auto compared the
// test video at its whole frame (an encode cropped when it was made), as a
// Source-only run gives them. The comparison saved in metric_cache.json,
// under both recipes.
class SourceOnlyReuse : public ::testing::Test {
protected:
    SourceOnlyReuse() {
        touch(folder / "s.mp4", "s");
        touch(folder / "t.mp4", "t");
        source_only.crop_mode = CropMode::source_only;
        const auto [width, height] = made.distorted_info.frame_size();
        whole_test.distorted_crop = CropBox{width, height, 0, 0};
    }

    std::filesystem::path save(const std::string& cache, const ComparisonRecipe& recipe,
                               const ComparisonResult& result) const {
        return store_result(folder / cache, folder / "s.mp4", folder / "t.mp4", recipe, result, "t", request.metrics);
    }
    std::optional<std::pair<ComparisonResult, std::string>> load(const std::string& cache, const ComparisonRecipe& recipe,
                                                                 const std::set<std::string>& again = {}) const {
        return load_result(folder / cache, folder / "s.mp4", folder / "t.mp4", recipe, request.metrics,
                           request.perceptual_backends, {}, again);
    }
    static std::size_t found(const std::optional<std::pair<ComparisonResult, std::string>>& loaded) {
        return loaded ? loaded->first.metric_results.keys().size() : 0U;
    }
    // `made` with only `score`.
    ComparisonResult only(const FrameMetricResult& score) const {
        ComparisonResult result = made;
        result.metric_results = MetricResultSet();
        result.metric_results.add(score);
        return result;
    }

    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    const AnalysisRequest request = request_from(stored);
    const ComparisonRecipe automatic = request.recipe;  // Auto: the reference's
    ComparisonRecipe source_only = request.recipe;
    const ComparisonResult made = test::result_from(stored.at("result"));  // the source cropped
    ComparisonResult whole_test = made;                                      // with the test video's whole frame
};

// Where Auto kept the test video's whole frame, and with its source box,
// with no test crop; not for the metrics asked for again, not when its own
// comparison cut another source box; not where Auto cropped the test video;
// never the other way round. A resolution test's Auto answers with its crops
// as saved.
TEST_F(SourceOnlyReuse, AutosScoresAnswerWhereAutoKeptTheTestVideosWholeFrame) {
    ASSERT_EQ(automatic.crop_mode, CropMode::automatic);
    save("whole", automatic, whole_test);
    const auto loaded = load("whole", source_only);
    ASSERT_EQ(found(loaded), request.metrics.size());
    EXPECT_FALSE(loaded->first.distorted_crop.has_value());
    EXPECT_EQ(loaded->first.source_crop, made.source_crop);
    const auto again = load("whole", source_only, {"vmaf"});
    EXPECT_EQ(found(again), request.metrics.size() - 1);
    EXPECT_FALSE(again->first.metric_results.has("vmaf"));
    // Its own comparison, of PSNR alone (its own value): Auto's answer the
    // rest only with the same source box, and its own PSNR comes first.
    FrameMetricResult own_psnr = *made.metric_results.frame("psnr");
    own_psnr.values.front() = 1.0F;
    ComparisonResult own = only(own_psnr);
    own.source_crop = CropBox{1920, 804, 0, 138};
    save("whole", source_only, own);
    EXPECT_EQ(found(load("whole", source_only)), 1U);
    own.source_crop = made.source_crop;
    save("whole", source_only, own);
    const auto both = load("whole", source_only);
    ASSERT_EQ(found(both), request.metrics.size());
    EXPECT_EQ(both->first.metric_results.frame("psnr")->values.front(), 1.0F);

    ComparisonResult cropped_test = made;
    cropped_test.distorted_crop = CropBox{whole_test.distorted_crop->w, whole_test.distorted_crop->h - 100, 0, 50};
    save("cropped", automatic, cropped_test);
    EXPECT_FALSE(load("cropped", source_only));

    save("other way", source_only, whole_test);
    EXPECT_FALSE(load("other way", automatic));

    ComparisonResult resolution_test = made;
    resolution_test.distorted_crop = made.source_crop;
    ComparisonRecipe automatic_test = automatic;
    automatic_test.resample_test = ResampleTarget{1280, "720p"};
    ComparisonRecipe source_only_test = source_only;
    source_only_test.resample_test = automatic_test.resample_test;
    save("resolution", automatic_test, resolution_test);
    const auto tested = load("resolution", source_only_test);
    ASSERT_EQ(found(tested), request.metrics.size());
    EXPECT_EQ(tested->first.distorted_crop, made.source_crop);
}

// Where nothing was scaled (the two videos the same size), the scores saved
// with another scaling algorithm or direction answer, as made with the one
// asked for; not for the metrics asked for again, not where its own
// comparison cut other black bars (its own scores come first), and never
// where a video was scaled.
TEST_F(SourceOnlyReuse, AnotherScalersScoresAnswerWhereNothingWasScaled) {
    ComparisonResult same_size = made;
    same_size.source_crop.reset();  // both 1920x1080
    save("same", automatic, same_size);
    ComparisonRecipe lanczos = automatic;
    lanczos.scale_algorithm = "lanczos";
    const auto loaded = load("same", lanczos);
    ASSERT_EQ(found(loaded), request.metrics.size());
    EXPECT_EQ(loaded->first.scale_algorithm, "lanczos");
    EXPECT_FALSE(loaded->first.source_crop.has_value());
    ComparisonRecipe upscaled = automatic;
    upscaled.scale_direction = ScaleDirection::distorted_to_source;
    const auto up = load("same", upscaled);
    ASSERT_EQ(found(up), request.metrics.size());
    EXPECT_EQ(up->first.scale_direction, ScaleDirection::distorted_to_source);
    EXPECT_EQ(found(load("same", lanczos, {"vmaf"})), request.metrics.size() - 1);

    FrameMetricResult own_psnr = *made.metric_results.frame("psnr");
    own_psnr.values.front() = 1.0F;
    ComparisonResult own = only(own_psnr);
    own.source_crop = made.source_crop;
    own.distorted_crop = made.source_crop;  // the same size, other bars
    save("same", lanczos, own);
    EXPECT_EQ(found(load("same", lanczos)), 1U);
    own.source_crop.reset();
    own.distorted_crop.reset();
    save("same", lanczos, own);
    const auto both = load("same", lanczos);
    ASSERT_EQ(found(both), request.metrics.size());
    EXPECT_EQ(both->first.metric_results.frame("psnr")->values.front(), 1.0F);

    save("scaled", automatic, made);  // the source cropped to 1920x800, the test video 1920x1080
    EXPECT_FALSE(load("scaled", lanczos));
}

// Vship's SSIMULACRA2 in Auto's folder comes before the CPU's in its own (as
// within one folder), the CPU's once asked for again; Auto's CVVDP for
// another display is listed, not once CVVDP is asked for again; and a folder
// of its own whose context this app cannot read has its scores left unread.
TEST_F(SourceOnlyReuse, AutosScoresRankAndListAsOneFolders) {
    save("whole", automatic, whole_test);
    FrameMetricResult cpu_score = *made.metric_results.frame("ssimulacra2");
    cpu_score.provenance = MetricProvenance{"ssimulacra2", "libjxl 0.11", "cpu",
                                            score_identity::libjxl_cpu("ssimulacra2"), cpu_score.provenance.parameters};
    cpu_score.provenance.parameters["color_tags"] = nlohmann::json(score_identity::cpu_colour_tags).dump();
    save("whole", source_only, only(cpu_score));
    const auto ranked = [&](const std::set<std::string>& asked_again) {
        const auto loaded = load("whole", source_only, asked_again);
        const FrameMetricResult* score = loaded ? loaded->first.metric_results.frame("ssimulacra2") : nullptr;
        return score ? score->provenance.compute_backend : std::string();
    };
    EXPECT_EQ(ranked({}), "gpu");
    EXPECT_EQ(ranked({"ssimulacra2"}), "cpu");

    CvvdpSettings other_display = cvvdp_from(stored);
    other_display.resize_to_display = !other_display.resize_to_display;
    const AnalysisRequest other_request = analysis_request(options_from(stored.at("options")), {"cvvdp"}, {},
                                                           other_display, VideoInfo{}, VideoInfo{});
    const auto others = [&](const std::set<std::string>& asked_again) {
        const auto listed = other_cvvdp_scores(folder / "whole", folder / "s.mp4", folder / "t.mp4", source_only,
                                               other_request.metrics, {}, asked_again);
        return listed ? listed->scores.size() : 0U;
    };
    EXPECT_EQ(others({}), 1U);
    EXPECT_EQ(others({"cvvdp"}), 0U);

    save("unreadable", automatic, whole_test);
    FrameMetricResult changed = *made.metric_results.frame("psnr");
    changed.values.front() = 1.0F;
    touch(save("unreadable", source_only, only(changed)) / "context.json", "[]");
    const auto from_auto = load("unreadable", source_only);
    ASSERT_EQ(found(from_auto), request.metrics.size());
    EXPECT_NE(from_auto->first.metric_results.frame("psnr")->values.front(), 1.0F);
}

// Storing a result writes the Python app's context.json and score files:
// the same names, the same metadata to the byte but for each score's
// version, which is kept (the Python app saved it blank but for VMAF's) and
// comes back with the score. A resolution test's keeps its target, and comes
// back with it.
TEST(MetricCache, StoresAResultAsThePythonApp) {
    const auto version_of = [](const MetricResultSet& set, const std::string& key) -> std::optional<std::string> {
        if (const FrameMetricResult* frame = set.frame(key)) {
            return frame->provenance.implementation_version;
        }
        if (const SequenceMetricResult* sequence = set.sequence(key)) {
            return sequence->provenance.implementation_version;
        }
        return std::nullopt;
    };
    for (const char* name : {"stored", "stored_resolution_test"}) {
        SCOPED_TRACE(name);
        const nlohmann::json stored = test::load_reference("metric_cache.json").at(name);
        const test::Folder folder;
        touch(folder / "s.mp4", "s");
        touch(folder / "t.mp4", "t");
        const AnalysisRequest request = request_from(stored);
        const ComparisonResult result = test::result_from(stored.at("result"));
        const std::filesystem::path directory = store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4",
                                                             request.recipe, result, stored.at("label"), request.metrics);
        // As the Python app's but for the build that saved it and the black-bar
        // rule, which it did not say
        nlohmann::json context = nlohmann::json::parse(platform::read_file(directory / "context.json").value_or(""));
        EXPECT_EQ(context.value("saved_by", ""), build_stamp());
        EXPECT_EQ(context.value("black_bars_rule", 0), black_bars_rule_version);
        context.erase("saved_by");
        context.erase("black_bars_rule");
        EXPECT_EQ(python_json_compact(context), stored.at("context").get<std::string>());
        std::map<std::string, std::string> files;
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().extension() == ".npz") {
                const auto arrays = read_npz(entry.path());
                ASSERT_TRUE(arrays.has_value()) << entry.path();
                files[platform::path_text(entry.path().filename())] =
                    npy_text_value(arrays->at("metadata")).value_or("");
            }
        }
        EXPECT_EQ(nlohmann::json(files), stored.at("files"));
        const auto loaded = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                                        request.metrics, request.perceptual_backends);
        ASSERT_TRUE(loaded.has_value());
        EXPECT_EQ(loaded->first.resample_target, result.resample_target);
        for (const std::string& key : result.metric_results.keys()) {
            EXPECT_EQ(version_of(loaded->first.metric_results, key), version_of(result.metric_results, key)) << key;
        }
    }
}

// The build that made a score comes back with it, saved again unchanged
// (not as the build that saves it); a score saved before builds said, none.
// The saved results browsed (saved_results.h): every comparison listed,
// newest first, with its label, files and score files by metric; read back
// without a request (a metric's newest file, or the one chosen; an
// unreadable one passed over for the next); and deleted, only a folder of
// its own under the cache's v2 folder.
// A comparison whose reference is no longer where its scores were saved from
// (gone, or another file there now) has the only scores there are: listed as
// such, and kept by Clear saved results unless it is asked to take them too.
// One touched since (the same bytes, another time) is still there, known by
// its sampled hash as the lookup knows it.
TEST(MetricCache, ClearingKeepsTheComparisonsWhoseReferenceIsGone) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    const AnalysisRequest request = request_from(stored);
    const std::filesystem::path cache = folder / "cache";
    std::map<std::string, std::filesystem::path> saved;
    for (const char* name : {"here", "gone", "touched", "replaced"}) {
        const std::filesystem::path reference = folder / (std::string(name) + ".mp4");
        touch(reference, name);
        touch(folder / "t.mp4", "t");
        ComparisonResult result = test::result_from(stored.at("result"));
        result.source = reference;
        result.distorted = folder / "t.mp4";
        saved[name] = store_result(cache, reference, folder / "t.mp4", request.recipe, result, name, request.metrics);
    }
    std::filesystem::remove(folder / "gone.mp4");
    touch(folder / "replaced.mp4", "another file, of another size");
    const std::filesystem::path touched = folder / "touched.mp4";
    std::filesystem::last_write_time(touched, std::filesystem::last_write_time(touched) - std::chrono::hours(1));
    EXPECT_TRUE(reference_where_saved(saved["here"]));
    EXPECT_TRUE(reference_where_saved(saved["touched"]));
    EXPECT_FALSE(reference_where_saved(saved["gone"]));
    EXPECT_FALSE(reference_where_saved(saved["replaced"]));
    // The file at a path is the one saved for (#134): not where another is
    // there now, t.mp4 written again since "here" was saved, say.
    EXPECT_TRUE(saved_file_at(saved["here"], true, folder / "here.mp4"));
    EXPECT_FALSE(saved_file_at(saved["here"], false, folder / "t.mp4"));
    EXPECT_TRUE(saved_file_at(saved["replaced"], false, folder / "t.mp4"));
    EXPECT_FALSE(saved_file_at(saved["gone"], true, folder / "gone.mp4"));
    EXPECT_FALSE(saved_file_at(saved["replaced"], true, folder / "replaced.mp4"));
    for (const SavedComparison& listed : list_saved_comparisons(cache)) {
        EXPECT_EQ(listed.reference_found, listed.label == "here" || listed.label == "touched") << listed.label;
    }

    // Issue #134: the bitrate of the kept ones' files is kept too, the only
    // bitrate there is of the reference gone; the rest is deleted.
    SavedBitrate bitrate;
    bitrate.data = bitrate_data(folder / "t.mp4", {0.0}, {1 / 24.0}, {1000}, {true});
    save_bitrate(cache, saved_content_hash(saved["here"], true), bitrate);
    save_bitrate(cache, saved_content_hash(saved["gone"], true), bitrate);
    save_bitrate(cache, saved_content_hash(saved["gone"], false), bitrate);  // t.mp4, of all four
    save_bitrate(cache, "of no comparison", bitrate);
    // A folder that is no comparison (half-written, or left by a delete cut
    // short): no scores anyone can list or delete, so cleared with the rest,
    // as everything was when nothing was kept.
    const std::filesystem::path stray = cache / "v2" / "no comparison";
    std::filesystem::create_directories(stray);
    touch(stray / "psnr_1.npz", "scores of no comparison");
    const ClearedResults cleared = clear_saved_results(cache, true);
    EXPECT_FALSE(std::filesystem::exists(stray));
    EXPECT_FALSE(load_saved_bitrate(cache, saved_content_hash(saved["here"], true), folder / "here.mp4"));
    EXPECT_TRUE(load_saved_bitrate(cache, saved_content_hash(saved["gone"], true), folder / "gone.mp4"));
    EXPECT_TRUE(load_saved_bitrate(cache, saved_content_hash(saved["gone"], false), folder / "t.mp4"));
    EXPECT_FALSE(load_saved_bitrate(cache, "of no comparison", folder / "t.mp4"));
    EXPECT_EQ(cleared.removed, 2);  // here and touched
    EXPECT_EQ(cleared.kept, 2);
    EXPECT_FALSE(std::filesystem::exists(saved["here"]));
    EXPECT_FALSE(std::filesystem::exists(saved["touched"]));
    EXPECT_TRUE(std::filesystem::exists(saved["gone"] / "context.json"));
    EXPECT_TRUE(std::filesystem::exists(saved["replaced"] / "context.json"));
    EXPECT_EQ(list_saved_comparisons(cache).size(), 2U);

    const ClearedResults everything = clear_saved_results(cache, false);
    EXPECT_EQ(everything.removed, 2);
    EXPECT_EQ(everything.kept, 0);
    EXPECT_TRUE(list_saved_comparisons(cache).empty());
    EXPECT_FALSE(std::filesystem::exists(bitrate_store_folder(cache)));
}

TEST(MetricCache, SavedComparisonsAreListedReadAndDeleted) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    touch(folder / "u.mp4", "u");
    const AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    result.source = folder / "s.mp4";  // as a run gives them: the files' own paths
    result.distorted = folder / "t.mp4";
    const std::filesystem::path cache = folder / "cache";
    const std::filesystem::path first =
        store_result(cache, folder / "s.mp4", folder / "t.mp4", request.recipe, result, "t", request.metrics);
    result.distorted = folder / "u.mp4";
    const std::filesystem::path second =
        store_result(cache, folder / "s.mp4", folder / "u.mp4", request.recipe, result, "u", request.metrics);
    // The first comparison newest: its files a minute later.
    for (const auto& entry : std::filesystem::directory_iterator(first)) {
        std::filesystem::last_write_time(entry.path(), std::filesystem::last_write_time(entry.path()) +
                                                           std::chrono::minutes(1));
    }

    std::vector<SavedComparison> list = list_saved_comparisons(cache);
    ASSERT_EQ(list.size(), 2U);
    EXPECT_EQ(list[0].folder, first);
    EXPECT_EQ(list[0].label, "t");
    EXPECT_EQ(list[1].label, "u");
    EXPECT_EQ(list[0].distorted, folder / "t.mp4");
    EXPECT_EQ(list[0].source, folder / "s.mp4");
    EXPECT_GT(list[0].bytes, 0U);
    const std::vector<std::string> keys = list[0].keys();
    std::vector<std::string> saved_keys = result.metric_results.keys();
    std::sort(saved_keys.begin(), saved_keys.end());
    std::vector<std::string> listed = keys;
    std::sort(listed.begin(), listed.end());
    EXPECT_EQ(listed, saved_keys);
    // In the app's order: VMAF before PSNR.
    EXPECT_LT(std::find(keys.begin(), keys.end(), "vmaf"), std::find(keys.begin(), keys.end(), "psnr"));

    // Read back as saved.
    const auto loaded = load_saved_comparison(list[0]);
    ASSERT_TRUE(loaded);
    EXPECT_EQ(loaded->second, "t");
    EXPECT_EQ(loaded->first.distorted, folder / "t.mp4");
    EXPECT_EQ(loaded->first.metric_results.frame("vmaf")->values, result.metric_results.frame("vmaf")->values);

    // A newer VMAF file that cannot be read: passed over for the readable one,
    // also when it is the one chosen.
    const std::filesystem::path broken = first / "vmaf_ffff.npz";
    touch(broken, "not a score");
    std::filesystem::last_write_time(broken, std::filesystem::last_write_time(broken) + std::chrono::minutes(5));
    list = list_saved_comparisons(cache);
    const auto vmaf_files = std::count_if(list[0].scores.begin(), list[0].scores.end(),
                                          [](const SavedScoreFile& file) { return file.key == "vmaf"; });
    EXPECT_EQ(vmaf_files, 2);
    EXPECT_EQ(std::find_if(list[0].scores.begin(), list[0].scores.end(),
                           [](const SavedScoreFile& file) { return file.key == "vmaf"; })
                  ->file,
              broken);  // the newest first
    const auto chosen = load_saved_comparison(list[0], {{"vmaf", broken}});
    ASSERT_TRUE(chosen);
    EXPECT_EQ(chosen->first.metric_results.frame("vmaf")->values, result.metric_results.frame("vmaf")->values);
    EXPECT_FALSE(read_saved_score(broken));

    // Deleted: only its own folder; another path is refused.
    SavedComparison elsewhere = list[1];
    elsewhere.folder = folder / "s.mp4";  // not a folder under v2
    EXPECT_FALSE(delete_saved_comparison(cache, elsewhere));
    EXPECT_TRUE(std::filesystem::exists(folder / "s.mp4"));
#ifdef _WIN32
    // A comparison one of whose files is open, without leave to delete it
    // (the tab's details reader, a lookup, an antivirus): nothing of it is
    // deleted, and it is still listed, to be deleted again. Deleted file by
    // file, its context.json went first, and the half left was listed by
    // nothing and deletable by nothing.
    {
        const std::filesystem::path held = list[1].scores.front().file;
        std::FILE* open = _wfsopen(held.c_str(), L"rb", _SH_DENYNO);
        ASSERT_NE(open, nullptr);
        EXPECT_FALSE(delete_saved_comparison(cache, list[1]));
        EXPECT_TRUE(std::filesystem::exists(second / "context.json"));
        EXPECT_EQ(list_saved_comparisons(cache).size(), 2U);
        std::fclose(open);
    }
    // A leftover of the same comparison in .deleted, one of its files still
    // open, does not block deleting it (#221's review: its name was taken).
    const std::filesystem::path leftover = cache / "v2" / ".deleted" / second.filename();
    std::filesystem::create_directories(leftover);
    touch(leftover / "vmaf_0.npz", "held");
    std::FILE* still_open = _wfsopen((leftover / "vmaf_0.npz").c_str(), L"rb", _SH_DENYNO);
    ASSERT_NE(still_open, nullptr);
#endif
    EXPECT_TRUE(delete_saved_comparison(cache, list[1]));
    EXPECT_FALSE(std::filesystem::exists(second));
#ifdef _WIN32
    std::fclose(still_open);
#endif
    list = list_saved_comparisons(cache);
    ASSERT_EQ(list.size(), 1U);
    EXPECT_EQ(list[0].label, "t");
}

// #205: the Saved Results tab lists the saved comparisons each time it is
// shown; a listing nobody waited for any more (a newer one, or the window
// closing, which waited for it) still read every folder, minutes on a slow
// share. Asked to stop, it ends at the next folder.
TEST(MetricCache, AListingOfSavedComparisonsStopsWhenAsked) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    const AnalysisRequest request = request_from(stored);
    const std::filesystem::path cache = folder / "cache";
    touch(folder / "s.mp4", "s");
    for (const char* name : {"a.mp4", "b.mp4", "c.mp4"}) {
        touch(folder / name, name);
        ComparisonResult result = test::result_from(stored.at("result"));
        result.source = folder / "s.mp4";
        result.distorted = folder / name;
        store_result(cache, result.source, result.distorted, request.recipe, result, name, request.metrics);
    }
    EXPECT_EQ(list_saved_comparisons(cache).size(), 3U);
    int asked = 0;
    const std::vector<SavedComparison> stopped = list_saved_comparisons(cache, [&asked] { return ++asked > 1; });
    EXPECT_EQ(stopped.size(), 1U);
    EXPECT_EQ(asked, 2);
}

// A file is known by its content where its path changed: the sampled hash
// reads the size and samples of the file (the same bytes anywhere give the
// same hash; a changed byte in a sample, another), and the pictures a video
// fingerprint decodes are mid-frame times a sixth of the length apart.
TEST(MetricCache, FilesAreKnownByTheirContent) {
    const test::Folder folder;
    std::string bytes(3 << 20, 'x');  // past the size read whole
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((i * 7919) % 251);
    }
    touch(folder / "a.mkv", bytes);
    touch(folder / "b.mkv", bytes);
    const std::optional<std::string> a = sampled_file_hash(folder / "a.mkv");
    ASSERT_TRUE(a);
    EXPECT_EQ(sampled_file_hash(folder / "b.mkv"), a);
    bytes[10] ^= 1;  // in the first megabyte
    touch(folder / "b.mkv", bytes);
    EXPECT_NE(sampled_file_hash(folder / "b.mkv"), a);
    touch(folder / "small.mkv", "small");
    EXPECT_TRUE(sampled_file_hash(folder / "small.mkv"));
    EXPECT_FALSE(sampled_file_hash(folder / "missing.mkv"));

    // 8 s at 24 fps: frames 32, 64, 96, 128 and 160, each at its middle; an
    // MP4's 7.999 s and 24.003 fps (its rounded duration's) give the same.
    const std::vector<double> times = fingerprint_times(8.0, 24.0);
    ASSERT_EQ(times.size(), 5U);
    EXPECT_DOUBLE_EQ(times[0], 32.5 / 24.0);
    EXPECT_DOUBLE_EQ(times[4], 160.5 / 24.0);
    EXPECT_EQ(fingerprint_times(7.999, 24.003), times);
    EXPECT_TRUE(fingerprint_times(8.0, 0.0).empty());
}

// The saved scores of files whose paths changed: a renamed or copied test
// video or reference (their sampled hashes), a remuxed reference (its video
// fingerprint, asked only once the test video matched); another reference
// is not taken for it. The result has the files' paths as they are now,
// its videos' details too (Video Compare decodes those, and a run file
// exported from the row keeps them).
TEST(MetricCache, ScoresFollowRenamedAndRemuxedFiles) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mkv", std::string(5000, 's'));
    touch(folder / "t.mkv", std::string(4000, 't'));
    const AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    result.source = folder / "s.mkv";
    result.distorted = folder / "t.mkv";
    result.source_info.path = result.source;
    result.distorted_info.path = result.distorted;
    int fingerprinted = 0;
    set_video_fingerprinter([&fingerprinted](const std::filesystem::path& file) -> std::optional<std::string> {
        ++fingerprinted;
        // The remux holds the reference's stream; the other video another.
        const std::string name = platform::path_text(file.filename());
        return name == "other.mp4" ? std::string("another stream") : std::string("the reference's stream");
    });
    const std::filesystem::path cache = folder / "cache";
    const std::filesystem::path saved =
        store_result(cache, folder / "s.mkv", folder / "t.mkv", request.recipe, result, "t", request.metrics);
    EXPECT_TRUE(std::filesystem::exists(saved / "identity.json"));
    const auto load = [&](const std::string& source, const std::string& distorted) {
        return load_result(cache, folder / source, folder / distorted, request.recipe, request.metrics,
                           request.perceptual_backends);
    };

    // Copied and renamed: the same bytes.
    std::filesystem::copy_file(folder / "s.mkv", folder / "s copy.mkv");
    std::filesystem::copy_file(folder / "t.mkv", folder / "t renamed.mkv");
    const auto copied = load("s copy.mkv", "t renamed.mkv");
    ASSERT_TRUE(copied);
    EXPECT_EQ(copied->first.source, folder / "s copy.mkv");
    EXPECT_EQ(copied->first.distorted, folder / "t renamed.mkv");
    EXPECT_EQ(copied->first.source_info.path, folder / "s copy.mkv");
    EXPECT_EQ(copied->first.distorted_info.path, folder / "t renamed.mkv");
    EXPECT_EQ(recipe_directory(cache, folder / "s copy.mkv", folder / "t renamed.mkv", request.recipe), saved);

    // A remuxed reference: other bytes, the same stream.
    touch(folder / "remux.mp4", std::string(5100, 'm'));
    const int before = fingerprinted;
    ASSERT_TRUE(load("remux.mp4", "t.mkv"));
    EXPECT_GT(fingerprinted, before);

    // Another reference: not this comparison's.
    touch(folder / "other.mp4", std::string(5100, 'o'));
    EXPECT_FALSE(load("other.mp4", "t.mkv"));

    // A new test video: nothing saved for it, nothing decoded to find out.
    touch(folder / "new.mkv", std::string(4100, 'n'));
    const int asked = fingerprinted;
    EXPECT_FALSE(load("remux.mp4", "new.mkv"));
    EXPECT_EQ(fingerprinted, asked);

    // A comparison another process saved meanwhile (the command line, another
    // window): found by a renamed copy too, though this process has looked
    // the saved scores up before.
    touch(folder / "u.mkv", std::string(4200, 'u'));
    result.distorted = folder / "u.mkv";
    const std::filesystem::path elsewhere =
        store_result(folder / "other process", folder / "s.mkv", folder / "u.mkv", request.recipe, result, "u",
                     request.metrics);
    std::filesystem::copy(elsewhere, cache / "v2" / elsewhere.filename());
    std::filesystem::copy_file(folder / "u.mkv", folder / "u renamed.mkv");
    const auto from_another = load("s.mkv", "u renamed.mkv");
    ASSERT_TRUE(from_another);
    EXPECT_EQ(from_another->first.distorted, folder / "u renamed.mkv");

    // Saved by an older version, without identity.json: the lookup leaves
    // the recording, which decodes, to record_saved_files, called after.
    std::filesystem::remove(saved / "identity.json");
    ASSERT_TRUE(load("s.mkv", "t.mkv"));
    EXPECT_FALSE(std::filesystem::exists(saved / "identity.json"));
    record_saved_files(cache, folder / "s.mkv", folder / "t.mkv", request.recipe);
    EXPECT_TRUE(std::filesystem::exists(saved / "identity.json"));

    // Deleted (the Saved Results tab) after a lookup read the records: no
    // longer found by the files' content.
    EXPECT_EQ(recipe_directory(cache, folder / "s copy.mkv", folder / "t renamed.mkv", request.recipe), saved);
    std::filesystem::remove_all(saved);
    EXPECT_NE(recipe_directory(cache, folder / "s copy.mkv", folder / "t renamed.mkv", request.recipe), saved);
    set_video_fingerprinter(nullptr);
}

// #205: every process's first save into a comparison (vml-cli adding a
// metric, the app's first run of the day) decoded the reference again for
// its picture fingerprint (five decodes, which a 2-hour film on a share takes
// a while for), though the folder's identity.json recorded it for this same
// file. It is taken from there.
TEST(MetricCache, ASaveTakesTheReferencesRecordedFingerprint) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mkv", std::string(5000, 's'));
    touch(folder / "t.mkv", std::string(4000, 't'));
    const AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    result.source = folder / "s.mkv";
    result.distorted = folder / "t.mkv";
    const std::filesystem::path cache = folder / "cache";
    set_video_fingerprinter(nullptr);  // none worked out in this process
    const std::filesystem::path saved =
        store_result(cache, folder / "s.mkv", folder / "t.mkv", request.recipe, result, "t", request.metrics);
    // As another process recorded it.
    nlohmann::json identity = nlohmann::json::parse(*platform::read_file(saved / "identity.json"));
    ASSERT_EQ(identity["source"]["path"], file_identity(folder / "s.mkv"));
    identity["source"]["video"] = "recorded";
    std::ofstream(saved / "identity.json") << identity.dump();

    int decoded = 0;
    set_video_fingerprinter([&decoded](const std::filesystem::path&) -> std::optional<std::string> {
        ++decoded;
        return std::string("decoded");
    });
    store_result(cache, folder / "s.mkv", folder / "t.mkv", request.recipe, result, "t", request.metrics);
    EXPECT_EQ(decoded, 0);
    identity = nlohmann::json::parse(*platform::read_file(saved / "identity.json"));
    EXPECT_EQ(identity["source"]["video"], "recorded");

    // Another reference at that path since: worked out again.
    touch(folder / "s.mkv", std::string(6000, 'S'));
    store_result(cache, folder / "s.mkv", folder / "t.mkv", request.recipe, result, "t", request.metrics);
    EXPECT_EQ(decoded, 1);
    set_video_fingerprinter(nullptr);
}

// Issue #136: a lookup whose own folder is missing looked up the times of
// every saved comparison's identity.json and context.json, each time. A
// comparison is read again only where its folder's time changed: a file
// rewritten in place, which leaves it as it was, is not looked at. A
// comparison saved by another process since is found, and one deleted is
// not.
TEST(MetricCache, ComparisonsAreReadAgainOnlyWhereTheirFolderChanged) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mkv", std::string(5000, 's'));
    touch(folder / "t.mkv", std::string(4000, 't'));
    touch(folder / "u.mkv", std::string(4200, 'u'));
    const AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    result.source = folder / "s.mkv";
    result.distorted = folder / "t.mkv";
    const std::filesystem::path cache = folder / "cache";
    const std::filesystem::path saved =
        store_result(cache, result.source, result.distorted, request.recipe, result, "t", request.metrics);
    std::filesystem::copy_file(folder / "t.mkv", folder / "t renamed.mkv");
    std::filesystem::copy_file(folder / "u.mkv", folder / "u renamed.mkv");
    const auto folder_of = [&](const std::string& distorted) {
        return recipe_directory(cache, folder / "s.mkv", folder / distorted, request.recipe);
    };
    ASSERT_EQ(folder_of("t renamed.mkv"), saved);

    // Rewritten in place: not read again, so still found.
    const std::uintmax_t size = std::filesystem::file_size(saved / "identity.json");
    {
        std::fstream file(saved / "identity.json", std::ios::binary | std::ios::in | std::ios::out);
        file << std::string(size, ' ');  // the same size, nothing recorded
    }
    EXPECT_EQ(folder_of("t renamed.mkv"), saved);

    // Saved by another process (the command line, another window): found.
    result.distorted = folder / "u.mkv";
    const std::filesystem::path elsewhere = store_result(folder / "other process", result.source, result.distorted,
                                                         request.recipe, result, "u", request.metrics);
    EXPECT_NE(folder_of("u renamed.mkv"), cache / "v2" / elsewhere.filename());
    std::filesystem::copy(elsewhere, cache / "v2" / elsewhere.filename());
    EXPECT_EQ(folder_of("u renamed.mkv"), cache / "v2" / elsewhere.filename());

    // Deleted: no longer found.
    std::filesystem::remove_all(saved);
    EXPECT_NE(folder_of("t renamed.mkv"), saved);
}

// The saved comparisons are listed once and kept (issue #136: the whole
// folder was listed again at every lookup or save whose own folder is
// missing): a comparison another process saves is found only once the
// folder's time changes; what this process saves is found at once; one
// another process deleted is not picked, though the folder's time stayed.
// The folder's time is set by hand, as NTFS would set it and as a RAM
// drive or FAT would not.
TEST(MetricCache, SavedComparisonsAreListedOnceAndKept) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mkv", std::string(5000, 's'));
    for (const std::string name : {"t", "u", "w"}) {
        touch(folder / (name + ".mkv"), std::string(4000 + name[0], name[0]));
        std::filesystem::copy_file(folder / (name + ".mkv"), folder / (name + " renamed.mkv"));
    }
    const AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    result.source = folder / "s.mkv";
    const std::filesystem::path cache = folder / "cache";
    const auto save = [&](const std::filesystem::path& base, const std::string& name) {
        result.distorted = folder / (name + ".mkv");
        return store_result(base, result.source, result.distorted, request.recipe, result, name, request.metrics);
    };
    const auto folder_of = [&](const std::string& name) {
        return recipe_directory(cache, folder / "s.mkv", folder / (name + " renamed.mkv"), request.recipe);
    };
    const std::filesystem::path v2 = cache / "v2";
    const std::filesystem::path t = save(cache, "t");
    // As another process would (saved here before the lookup, which takes
    // what this process saves).
    const std::filesystem::path elsewhere = save(folder / "other process", "u");
    ASSERT_EQ(folder_of("t"), t);
    const std::filesystem::file_time_type listed = std::filesystem::last_write_time(v2);

    // Saved by another process, the folder's time left: not listed again.
    const std::filesystem::path u = v2 / elsewhere.filename();
    std::filesystem::copy(elsewhere, u);
    std::filesystem::last_write_time(v2, listed);
    EXPECT_NE(folder_of("u"), u);
    // The folder's time changed: listed again, found.
    std::filesystem::last_write_time(v2, listed + std::chrono::seconds(1));
    EXPECT_EQ(folder_of("u"), u);

    // Saved by this process: found without a listing.
    const std::filesystem::path w = save(cache, "w");
    std::filesystem::last_write_time(v2, listed + std::chrono::seconds(1));
    EXPECT_EQ(folder_of("w"), w);

    // Deleted by another process, the folder's time left: not picked.
    std::filesystem::remove_all(t);
    std::filesystem::last_write_time(v2, listed + std::chrono::seconds(1));
    EXPECT_NE(folder_of("t"), t);
    EXPECT_FALSE(std::filesystem::exists(t));
}

// A test video remuxed (raw <-> MKV <-> MP4) keeps its scores by its stream
// fingerprint, recorded after the save, read at the same rate, with the
// same reference: its first packets first, the whole stream only where they
// matched, and only by a lookup allowed to (StreamReads with `whole`, told
// first): any other, and a save, reads none and says one was wanted. An
// encode that changed only later scenes (x265 zones) shares the first
// packets, not the stream; a raw stream read at another rate is another
// comparison; a lookup cancelled reads nothing.
TEST(MetricCache, ARemuxedTestStreamKeepsItsScores) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mkv", std::string(5000, 's'));
    // "<first packets>/<rest>|<container bytes>": the fake fingerprinter's
    // stream, which a remux keeps and its container's bytes do not.
    touch(folder / "t.mkv", "A/a|" + std::string(4000, 'k'));
    AnalysisRequest request = request_from(stored);
    request.recipe.test_rate = 24000.0 / 1001.0;
    ComparisonResult result = test::result_from(stored.at("result"));
    result.source = folder / "s.mkv";
    result.distorted = folder / "t.mkv";
    int heads = 0;
    int wholes = 0;
    set_stream_fingerprinter([&](const std::filesystem::path& file, int packets,
                                 const CancelToken&) -> std::optional<std::string> {
        std::ifstream in(file, std::ios::binary);
        std::string stream;
        std::getline(in, stream, '|');
        ++(packets > 0 ? heads : wholes);
        return packets > 0 ? stream.substr(0, stream.find('/')) : stream;
    });
    const std::filesystem::path cache = folder / "cache";
    const std::filesystem::path saved =
        store_result(cache, folder / "s.mkv", folder / "t.mkv", request.recipe, result, "t", request.metrics);
    const auto identity_of = [&saved] {
        std::ifstream file(saved / "identity.json");
        return nlohmann::json::parse(file);
    };
    // The save reads no stream: recorded after it, once.
    EXPECT_EQ(heads + wholes, 0);
    EXPECT_FALSE(identity_of()["distorted"].contains("stream"));
    EXPECT_TRUE(test_stream_wanted(cache, folder / "t.mkv"));
    const auto saved_time = std::filesystem::last_write_time(saved / "identity.json");
    EXPECT_TRUE(record_test_stream(cache, folder / "t.mkv"));
    EXPECT_EQ(wholes, 1);
    // As when the save and the record fall in one tick of the clock files
    // are stamped from (about 16 ms on Windows by default): the file's time
    // does not show the record.
    std::filesystem::last_write_time(saved / "identity.json", saved_time);
    EXPECT_FALSE(test_stream_wanted(cache, folder / "t.mkv"));
    EXPECT_FALSE(record_test_stream(cache, folder / "t.mkv"));
    EXPECT_EQ(wholes, 1);
    EXPECT_FALSE(test_stream_wanted(cache, folder / "never saved.mkv"));
    const nlohmann::json identity = identity_of();
    EXPECT_EQ(identity["distorted"]["rate"], "23.98");
    EXPECT_EQ(identity["distorted"]["stream_head"], "A");
    EXPECT_EQ(identity["distorted"]["stream"], "A/a");
    ComparisonRecipe raw = request.recipe;
    raw.test_raw = true;
    EXPECT_FALSE(request.recipe.identity().contains("test_rate"));
    EXPECT_DOUBLE_EQ(raw.identity()["test_rate"].get<double>(), 23.976);
    std::vector<std::filesystem::path> announced;
    const auto load = [&](const std::string& source, const std::string& distorted, const ComparisonRecipe& recipe) {
        const StreamReads whole(CancelToken(), true,
                                [&announced](const std::filesystem::path& file) { announced.push_back(file); });
        return load_result(cache, folder / source, folder / distorted, recipe, request.metrics,
                           request.perceptual_backends);
    };

    // The raw stream it was muxed from, given the same rate: its scores. Not
    // by a lookup that may not read whole streams, nor by saving or clearing
    // (recipe_directory without StreamReads): they read no whole stream, and
    // say one was wanted. Not by a lookup cancelled, which reads nothing.
    touch(folder / "t.265", "A/a|" + std::string(3900, 'r'));
    take_whole_stream_wanted();
    const int wholes_before = wholes;
    EXPECT_FALSE(load_result(cache, folder / "s.mkv", folder / "t.265", raw, request.metrics,
                             request.perceptual_backends));
    EXPECT_NE(recipe_directory(cache, folder / "s.mkv", folder / "t.265", raw), saved);
    EXPECT_EQ(wholes, wholes_before);
    EXPECT_TRUE(take_whole_stream_wanted());
    EXPECT_FALSE(take_whole_stream_wanted());
    {
        CancelToken cancelled;
        cancelled.cancel();
        const StreamReads stop(cancelled, true);
        const int before = heads + wholes;
        EXPECT_FALSE(load_result(cache, folder / "s.mkv", folder / "t.265", raw, request.metrics,
                                 request.perceptual_backends));
        EXPECT_EQ(heads + wholes, before);
    }
    const auto found = load("s.mkv", "t.265", raw);
    ASSERT_TRUE(found);
    EXPECT_EQ(found->first.distorted, folder / "t.265");
    EXPECT_EQ(announced, std::vector<std::filesystem::path>{folder / "t.265"});
    EXPECT_EQ(wholes, wholes_before + 1);
    // Read once: a save of it now finds the folder without reading.
    EXPECT_EQ(recipe_directory(cache, folder / "s.mkv", folder / "t.265", raw), saved);
    EXPECT_EQ(wholes, wholes_before + 1);
    // Read at another rate: another comparison, and nothing read to know it.
    ComparisonRecipe at_25 = raw;
    at_25.test_rate = 25.0;
    int read = heads + wholes;
    EXPECT_FALSE(load("s.mkv", "t.265", at_25));
    EXPECT_EQ(heads + wholes, read);

    // Re-encoded in its later scenes only: the same first packets, another
    // stream. Another encode: told apart by its first packets alone.
    touch(folder / "zoned.mkv", "A/b|" + std::string(4000, 'z'));
    EXPECT_FALSE(load("s.mkv", "zoned.mkv", request.recipe));
    touch(folder / "other.mkv", "B/b|" + std::string(4000, 'o'));
    read = wholes;
    EXPECT_FALSE(load("s.mkv", "other.mkv", request.recipe));
    EXPECT_EQ(wholes, read);

    // The same stream against another reference: not this comparison's.
    touch(folder / "other reference.mkv", std::string(5100, 'x'));
    touch(folder / "t.mp4", "A/a|" + std::string(4100, 'p'));
    EXPECT_FALSE(load("other reference.mkv", "t.mp4", request.recipe));
    EXPECT_TRUE(load("s.mkv", "t.mp4", request.recipe));

    // Saved by an older version: no identity.json, so no stream. Found by
    // its path, its files are recorded (record_saved_files) without one,
    // and its stream is wanted until read: then a remux of it is found too.
    touch(folder / "old.mkv", "O/o|" + std::string(4200, 'k'));
    ComparisonResult old_result = result;
    old_result.distorted = folder / "old.mkv";
    const std::filesystem::path old_saved =
        store_result(cache, folder / "s.mkv", folder / "old.mkv", request.recipe, old_result, "old", request.metrics);
    std::filesystem::remove(old_saved / "identity.json");
    EXPECT_FALSE(test_stream_wanted(cache, folder / "old.mkv"));  // nothing recorded of it yet
    ASSERT_TRUE(load("s.mkv", "old.mkv", request.recipe));
    record_saved_files(cache, folder / "s.mkv", folder / "old.mkv", request.recipe);
    EXPECT_TRUE(test_stream_wanted(cache, folder / "old.mkv"));
    touch(folder / "old.mp4", "O/o|" + std::string(4300, 'p'));
    EXPECT_FALSE(load("s.mkv", "old.mp4", request.recipe));  // its remux, before the stream is read
    EXPECT_TRUE(record_test_stream(cache, folder / "old.mkv"));
    EXPECT_FALSE(test_stream_wanted(cache, folder / "old.mkv"));
    EXPECT_TRUE(load("s.mkv", "old.mp4", request.recipe));
    set_stream_fingerprinter(nullptr);
}

// An MP4 cut with an edit list keeps the packets before its first picture
// shown, which its remux to MKV or a raw stream shows: the stream hashed
// leaves them out (packets FFmpeg's MP4 reader times before 0, never a raw
// stream's untimed ones), so the remux is another video. A stream recorded
// before (format 1, which hashed them) is as if none were: read again, and
// never matched meanwhile.
TEST(MetricCache, AStreamRecordedInAnotherFormIsReadAgainAndNotMatched) {
    const std::vector<std::string> command = stream_fingerprint_command("t.mp4", "h264", "ffmpeg", 0);
    const auto bsf = std::find(command.begin(), command.end(), "-bsf:v");
    ASSERT_NE(bsf, command.end());
    EXPECT_EQ(*(bsf + 1), "noise=drop=lt(pts\\,0)*gt(pts\\,nopts),h264_mp4toannexb,filter_units=pass_types=1-5");

    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mkv", std::string(5000, 's'));
    touch(folder / "t.mkv", "A/a|" + std::string(4000, 'k'));
    touch(folder / "t.mp4", "A/a|" + std::string(4100, 'p'));
    AnalysisRequest request = request_from(stored);
    request.recipe.test_rate = 24.0;
    ComparisonResult result = test::result_from(stored.at("result"));
    result.source = folder / "s.mkv";
    result.distorted = folder / "t.mkv";
    set_stream_fingerprinter([](const std::filesystem::path& file, int packets,
                                const CancelToken&) -> std::optional<std::string> {
        std::ifstream in(file, std::ios::binary);
        std::string stream;
        std::getline(in, stream, '|');
        return packets > 0 ? stream.substr(0, stream.find('/')) : stream;
    });
    const std::filesystem::path cache = folder / "cache";
    const std::filesystem::path saved =
        store_result(cache, folder / "s.mkv", folder / "t.mkv", request.recipe, result, "t", request.metrics);
    ASSERT_TRUE(record_test_stream(cache, folder / "t.mkv"));
    nlohmann::json identity = nlohmann::json::parse(std::ifstream(saved / "identity.json"));
    EXPECT_EQ(identity["distorted"]["stream_format"], stream_fingerprint_format);
    const auto load = [&] {
        const StreamReads whole(CancelToken(), true);
        return load_result(cache, folder / "s.mkv", folder / "t.mp4", request.recipe, request.metrics,
                           request.perceptual_backends);
    };
    EXPECT_TRUE(load());

    identity["distorted"].erase("stream_format");  // as recorded before
    {
        std::ofstream(saved / "identity.json") << identity.dump();
    }
    // Read again as a record of an earlier version is: its folder's time
    // changed (a lookup that would take it reads it again where it did: as_read).
    std::filesystem::last_write_time(saved, std::filesystem::last_write_time(saved) + std::chrono::seconds(2));
    EXPECT_FALSE(load());
    EXPECT_TRUE(test_stream_wanted(cache, folder / "t.mkv"));
    EXPECT_TRUE(record_test_stream(cache, folder / "t.mkv"));
    EXPECT_TRUE(load());
    set_stream_fingerprinter(nullptr);
}

// "Recognize test videos remuxed" off is no stream fingerprinter: a remux is
// then not matched, even by a stream read while it was on, and a file looked
// up while it is off is read once it is on again, not left unread until a
// restart.
TEST(MetricCache, TestVideosAreReadOnlyWhileRecognisingRemuxesIsOn) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mkv", std::string(5000, 's'));
    touch(folder / "t.mkv", "A/a|" + std::string(4000, 'k'));
    touch(folder / "t.mp4", "A/a|" + std::string(4100, 'p'));
    AnalysisRequest request = request_from(stored);
    request.recipe.test_rate = 24.0;
    ComparisonResult result = test::result_from(stored.at("result"));
    result.source = folder / "s.mkv";
    result.distorted = folder / "t.mkv";
    const StreamFingerprinter reader = [](const std::filesystem::path& file, int packets,
                                          const CancelToken&) -> std::optional<std::string> {
        std::ifstream in(file, std::ios::binary);
        std::string stream;
        std::getline(in, stream, '|');
        return packets > 0 ? stream.substr(0, stream.find('/')) : stream;
    };
    const std::filesystem::path cache = folder / "cache";
    store_result(cache, folder / "s.mkv", folder / "t.mkv", request.recipe, result, "t", request.metrics);
    const auto load = [&] {
        const StreamReads whole(CancelToken(), true);
        return load_result(cache, folder / "s.mkv", folder / "t.mp4", request.recipe, request.metrics,
                           request.perceptual_backends)
            .has_value();
    };
    set_stream_fingerprinter(nullptr);  // off
    EXPECT_FALSE(record_test_stream(cache, folder / "t.mkv"));
    set_stream_fingerprinter(reader);  // turned on
    EXPECT_TRUE(record_test_stream(cache, folder / "t.mkv"));
    EXPECT_TRUE(load());
    set_stream_fingerprinter(nullptr);  // off again: the remux's stream, read, is not used
    EXPECT_FALSE(load());
}

// A stream re-tagged (h264_metadata from BT.601 to BT.709, or muxed into a
// container that states other tags) is read with another YUV->RGB matrix,
// which SSIMULACRA2, Butteraugli and CVVDP score: another stream. A range
// not stated is read as limited, as a remux to MP4 can drop it.
TEST(MetricCache, AStreamReadWithOtherColourIsAnotherStream) {
    VideoInfo untagged;
    untagged.codec_name = "h264";
    untagged.pix_fmt = "yuv420p";
    VideoInfo bt601 = untagged;
    bt601.color_space = "smpte170m";
    bt601.color_primaries = "smpte170m";
    bt601.color_transfer = "smpte170m";
    VideoInfo bt709 = untagged;
    bt709.color_space = "bt709";
    bt709.color_primaries = "bt709";
    bt709.color_transfer = "bt709";
    const std::string digest(64, 'a');
    EXPECT_NE(stream_fingerprint_of(digest, bt601, 0), stream_fingerprint_of(digest, bt709, 0));
    EXPECT_NE(stream_fingerprint_of(digest, untagged, 0), stream_fingerprint_of(digest, bt709, 0));
    VideoInfo limited = bt709;
    limited.color_range = "tv";
    EXPECT_EQ(stream_fingerprint_of(digest, limited, 0), stream_fingerprint_of(digest, bt709, 0));
    EXPECT_EQ(stream_fingerprint_of(digest, bt709, 0), stream_fingerprint_of(digest, bt709, 0));
}

// A remux that changes only the parameter sets' crop (h264_metadata's
// crop_right) or the pixel aspect ratio has the same coded pictures, but is
// decoded to another size or shape: another stream. A ratio not stated is
// square.
TEST(MetricCache, AStreamReadAtAnotherSizeOrShapeIsAnotherStream) {
    VideoInfo full;
    full.codec_name = "h264";
    full.pix_fmt = "yuv420p";
    full.width = 1920;
    full.height = 1080;
    VideoInfo cropped = full;
    cropped.width = 1904;
    VideoInfo anamorphic = full;
    anamorphic.sar = "4:3";
    VideoInfo unstated = full;
    unstated.sar = "0:1";
    const std::string digest(64, 'a');
    EXPECT_NE(stream_fingerprint_of(digest, cropped, 0), stream_fingerprint_of(digest, full, 0));
    EXPECT_NE(stream_fingerprint_of(digest, anamorphic, 0), stream_fingerprint_of(digest, full, 0));
    EXPECT_EQ(stream_fingerprint_of(digest, unstated, 0), stream_fingerprint_of(digest, full, 0));
}

// A file's bitrate is saved by its content and read back as it was, under
// the path the file has now; frame types the command line did not read are
// told apart from types read and unknown. Settings counts and clears it with
// the saved scores.
TEST(MetricCache, BitrateIsSavedByContent) {
    const test::Folder folder;
    const std::filesystem::path cache = folder / "cache";
    SavedBitrate bitrate;
    bitrate.info.path = folder / "a.mkv";
    bitrate.info.codec_name = "h264";
    bitrate.info.width = 1920;
    bitrate.info.height = 1080;
    bitrate.info.fps = 24.0;
    bitrate.info.duration = 0.125;
    bitrate.data = bitrate_data(bitrate.info.path, {0.0, 1 / 24.0, 2 / 24.0}, {1 / 24.0, 1 / 24.0, 1 / 24.0},
                                {90000, 1200, 3400000000LL}, {true, false, false});
    bitrate.frame_types = std::vector<std::uint8_t>{1, 2, 3};
    save_bitrate(cache, "content", bitrate);
    EXPECT_FALSE(load_saved_bitrate(cache, "other", folder / "a.mkv"));
    EXPECT_FALSE(load_saved_bitrate(cache, "", folder / "a.mkv"));

    const std::optional<SavedBitrate> moved = load_saved_bitrate(cache, "content", folder / "moved.mkv");
    ASSERT_TRUE(moved);
    EXPECT_EQ(moved->info.path, folder / "moved.mkv");
    EXPECT_EQ(moved->info.codec_name, "h264");
    EXPECT_EQ(moved->data.times, bitrate.data.times);
    EXPECT_EQ(moved->data.sizes, bitrate.data.sizes);
    EXPECT_EQ(moved->data.keyframes, bitrate.data.keyframes);
    EXPECT_EQ(moved->frame_types, bitrate.frame_types);

    bitrate.frame_types = std::nullopt;  // the command line's
    save_bitrate(cache, "unread", bitrate);
    bitrate.frame_types = std::vector<std::uint8_t>{};  // read, not known
    save_bitrate(cache, "unknown", bitrate);
    EXPECT_FALSE(load_saved_bitrate(cache, "unread", folder / "a.mkv")->frame_types);
    EXPECT_EQ(load_saved_bitrate(cache, "unknown", folder / "a.mkv")->frame_types, std::vector<std::uint8_t>{});
    // #205: saved as "read, not known" before a reading cut short was told
    // apart from one read whole (no frame_types_whole): read again.
    std::map<std::string, NpyArray> earlier = *read_npz(bitrate_store_folder(cache) / "unknown.npz");
    nlohmann::json metadata = nlohmann::json::parse(*npy_text_value(earlier.at("metadata")));
    metadata.erase("frame_types_whole");
    earlier["metadata"] = npy_text(metadata.dump());
    const std::string bytes = npz_bytes({earlier.begin(), earlier.end()});
    std::ofstream(bitrate_store_folder(cache) / "earlier.npz", std::ios::binary) << bytes;
    const std::optional<SavedBitrate> unsure = load_saved_bitrate(cache, "earlier", folder / "a.mkv");
    ASSERT_TRUE(unsure);
    EXPECT_FALSE(unsure->frame_types);
    EXPECT_EQ(unsure->data.sizes, bitrate.data.sizes);  // its packets still taken

    EXPECT_GT(cache_summary(cache).bytes, 0U);
    clear_saved_results(cache, true);  // as Settings clears: the bitrate too
    EXPECT_FALSE(std::filesystem::exists(bitrate_store_folder(cache)));
    EXPECT_EQ(cache_summary(cache).bytes, 0U);
}

TEST(MetricCache, AScoreKeepsTheBuildThatMadeIt) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    ASSERT_TRUE(result.metric_results.frame("vmaf"));
    FrameMetricResult vmaf = *result.metric_results.frame("vmaf");
    vmaf.provenance.made_by = "2.3 (0123456789ab)";
    result.metric_results.add(vmaf);
    const auto stored_and_loaded = [&](const ComparisonResult& saved) {
        store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe, saved, "t", request.metrics);
        return load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe, request.metrics,
                           request.perceptual_backends);
    };
    const auto loaded = stored_and_loaded(result);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->first.metric_results.frame("vmaf")->provenance.made_by, "2.3 (0123456789ab)");
    const auto again = stored_and_loaded(loaded->first);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(again->first.metric_results.frame("vmaf")->provenance.made_by, "2.3 (0123456789ab)");
    for (const std::string& key : result.metric_results.keys()) {
        if (key != "vmaf" && again->first.metric_results.frame(key)) {
            EXPECT_EQ(again->first.metric_results.frame(key)->provenance.made_by, "") << key;
        }
    }
}

// A pair with an MPEG-2 or MPEG-4 Part 2 video, which the GPUs decode to
// other pictures than FFmpeg's, loads the scores saved since such a video was
// decoded in software, not those saved before (as an HEVC pair's are saved);
// other pairs load every score.
TEST(MetricCache, AnMpegPairsScoresSavedBeforeSoftwareDecodingAreCalculatedAgain) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    // How many scores load when the test video was saved as `saved_as` and
    // the comparison's context then says `now`.
    const auto loaded_after = [&](const std::string& saved_as, const std::string& now) {
        result.distorted_info.codec_name = saved_as;
        const std::filesystem::path directory = store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4",
                                                             request.recipe, result, "t", request.metrics);
        nlohmann::json context = nlohmann::json::parse(*platform::read_file(directory / "context.json"));
        context["distorted_info"]["codec_name"] = now;
        touch(directory / "context.json", context.dump());
        const auto loaded = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                                        request.metrics, request.perceptual_backends);
        return loaded ? loaded->first.metric_results.keys().size() : 0;
    };
    const std::size_t all = loaded_after("hevc", "hevc");
    EXPECT_EQ(all, request.metrics.size());
    EXPECT_EQ(loaded_after("hevc", "mpeg2video"), 0U);
    EXPECT_EQ(loaded_after("mpeg2video", "mpeg2video"), all);
    EXPECT_EQ(loaded_after("hevc", "mpeg4"), 0U);
}

// Scores saved under an older black-bar rule (issue #12's rule 1: a context
// without "black_bars_rule"; rule 2) are read as saved, with no bars found
// again: a rule change applies to new calculations only. Their files go only
// when new scores with other crops are saved there.
TEST(MetricCache, ScoresOfAnOlderBlackBarRuleAreReadAsSaved) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    AnalysisRequest request = request_from(stored);
    request.recipe.crop_mode = CropMode::automatic;
    ComparisonResult result = test::result_from(stored.at("result"));
    const auto [width, height] = result.source_info.frame_size();
    const CropBox letterbox{width, height - 40, 0, 20};
    // Saved with `crop` by `rule` (nothing: rule 1, which wrote none)
    const auto store = [&](const std::optional<CropBox>& crop, std::optional<int> rule,
                           const std::vector<MetricRequestSpec>& metrics) {
        result.source_crop = crop;
        const std::filesystem::path directory =
            store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe, result, "t", metrics);
        nlohmann::json context = nlohmann::json::parse(*platform::read_file(directory / "context.json"));
        if (rule) {
            context["black_bars_rule"] = *rule;
        } else {
            context.erase("black_bars_rule");
        }
        touch(directory / "context.json", context.dump());
    };
    const auto loaded = [&] {
        const auto found = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                                       request.metrics, request.perceptual_backends);
        return found ? found->first.metric_results.keys().size() : 0;
    };
    CvvdpSettings other_display = cvvdp_from(stored);
    other_display.resize_to_display = !other_display.resize_to_display;
    const AnalysisRequest other_request = analysis_request(options_from(stored.at("options")), {"cvvdp"}, {},
                                                           other_display, VideoInfo{}, VideoInfo{});
    const auto other_displays = [&] {
        const auto listed = other_cvvdp_scores(folder / "cache", folder / "s.mp4", folder / "t.mp4",
                                               request.recipe, other_request.metrics);
        return listed ? listed->scores.size() : 0U;
    };
    const std::size_t all = request.metrics.size();
    for (const std::optional<int> rule : {std::optional<int>(), std::optional<int>(2)}) {
        store(letterbox, rule, request.metrics);
        EXPECT_EQ(loaded(), all);
        EXPECT_EQ(other_displays(), 1U);
    }
    // One metric saved again with the same crops: the others' files stay
    store(letterbox, black_bars_rule_version, {request.metrics.front()});
    EXPECT_EQ(loaded(), all);
    // With other crops: the others' files go
    store(std::nullopt, black_bars_rule_version, {request.metrics.front()});
    EXPECT_EQ(loaded(), 1U);
}

// A score of a pair with a video scored with its AV1 film grain, or an AV1
// video whose grain is not known, says in its saved file that it was decoded
// as dav1d decodes it (av1_grain_dav1d): one this run made as its pair says,
// one it kept as it was saved. Outside Windows only scores that say so are
// taken, from the saved scores or kept from a run file (a comparison saved
// before 2.3: none); Windows takes them all, as before. A kept score that
// says nothing never replaces a saved one that says it.
TEST(MetricCache, AnAv1PairsScoresSayTheirGrainWasDav1dsAndOnlyThoseCountOutsideWindows) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const AnalysisRequest request = request_from(stored);
    const ComparisonResult reference = test::result_from(stored.at("result"));
#ifdef _WIN32
    const bool marks_count = false;
#else
    const bool marks_count = true;
#endif
    // The result with its test video `codec`, its grain `grain` (nothing: not
    // known) scored with it unless `without`, and every score marked `mark`.
    const auto made = [&](const std::string& codec, std::optional<unsigned> grain, bool without,
                          std::optional<bool> mark) {
        ComparisonResult result = reference;
        result.distorted_info.codec_name = codec;
        result.distorted_info.film_grain = grain;
        if (without) {
            result.distorted_info.with_film_grain = false;
        }
        MetricResultSet scores;
        for (const std::string& key : result.metric_results.keys()) {
            if (const FrameMetricResult* frame = result.metric_results.frame(key)) {
                FrameMetricResult copy = *frame;
                copy.av1_grain_dav1d = mark;
                scores.add(std::move(copy));
            } else {
                SequenceMetricResult copy = *result.metric_results.sequence(key);
                copy.av1_grain_dav1d = mark;
                scores.add(std::move(copy));
            }
        }
        result.metric_results = std::move(scores);
        return result;
    };
    // Whether each saved file has the mark, and how many scores load.
    const auto saved = [&](const ComparisonResult& result) {
        std::filesystem::remove_all(folder / "cache");
        const std::filesystem::path directory = store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4",
                                                             request.recipe, result, "t", request.metrics);
        std::set<bool> marked;
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().extension() == ".npz") {
                const nlohmann::json metadata =
                    nlohmann::json::parse(*npy_text_value(read_npz(entry.path())->at("metadata")));
                marked.insert(metadata.value("av1_grain_dav1d", 0) == 1);
            }
        }
        const auto loaded = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                                        request.metrics, request.perceptual_backends);
        const std::size_t count = loaded ? loaded->first.metric_results.keys().size() : 0;
        if (loaded) {
            for (const std::string& key : loaded->first.metric_results.keys()) {
                const FrameMetricResult* frame = loaded->first.metric_results.frame(key);
                EXPECT_EQ(frame ? frame->av1_grain_dav1d : loaded->first.metric_results.sequence(key)->av1_grain_dav1d,
                          std::optional(marked.contains(true))) << key;
            }
        }
        EXPECT_EQ(marked.size(), 1u);
        return std::pair(marked.contains(true), count);
    };
    const std::size_t all = request.metrics.size();
    const unsigned av1 = film_grain_av1;

    // Made by this run: marked as the pair says, and taken.
    EXPECT_EQ(saved(made("av1", av1, false, std::nullopt)), std::pair(true, all));
    EXPECT_EQ(saved(made("av1", av1, true, std::nullopt)), std::pair(false, all));  // left out: its own identity
    EXPECT_EQ(saved(made("av1", 0u, false, std::nullopt)), std::pair(false, all));   // no grain
    EXPECT_EQ(saved(made("hevc", std::nullopt, false, std::nullopt)), std::pair(false, all));
    EXPECT_EQ(saved(made("hevc", unsigned(film_grain_afgs1), false, std::nullopt)), std::pair(false, all));
    EXPECT_EQ(saved(made("vvc", unsigned(film_grain_h274), false, std::nullopt)), std::pair(false, all));
    EXPECT_EQ(saved(made("av2", unsigned(film_grain_av2), false, std::nullopt)), std::pair(false, all));
    // Grain not known (a scan that failed): decoded as dav1d decodes it, so
    // marked and taken.
    EXPECT_EQ(saved(made("av1", std::nullopt, false, std::nullopt)), std::pair(true, all));
    // Kept from an earlier result: as it was saved.
    EXPECT_EQ(saved(made("av1", av1, false, false)), std::pair(false, marks_count ? 0 : all));
    EXPECT_EQ(saved(made("av1", std::nullopt, false, true)), std::pair(true, all));
    // A run file's scores (unmarked) stored again over the marked ones: the
    // saved ones keep their mark, and still load everywhere; over unusable
    // ones they are written.
    saved(made("av1", av1, false, std::nullopt));
    const std::filesystem::path again = store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4",
                                                     request.recipe, made("av1", av1, false, false), "t", request.metrics);
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(again)) {
        if (entry.path().extension() == ".npz") {
            const nlohmann::json metadata =
                nlohmann::json::parse(*npy_text_value(read_npz(entry.path())->at("metadata")));
            EXPECT_EQ(metadata.value("av1_grain_dav1d", 0), 1) << entry.path().filename().string();
        }
    }
    const auto kept_marked = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                                         request.metrics, request.perceptual_backends);
    EXPECT_EQ(kept_marked ? kept_marked->first.metric_results.keys().size() : 0, all);
    // A marked file of another format (unusable) is replaced as before.
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(again)) {
        if (entry.path().extension() == ".npz") {
            std::map<std::string, NpyArray> arrays = *read_npz(entry.path());
            nlohmann::json metadata = nlohmann::json::parse(*npy_text_value(arrays.at("metadata")));
            metadata["format_version"] = -1;
            arrays["metadata"] = npy_text(metadata.dump());
            std::ofstream(entry.path(), std::ios::binary | std::ios::trunc)
                << npz_bytes({arrays.begin(), arrays.end()});
        }
    }
    store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                 made("av1", av1, false, false), "t", request.metrics);
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(again)) {
        if (entry.path().extension() == ".npz") {
            const nlohmann::json metadata =
                nlohmann::json::parse(*npy_text_value(read_npz(entry.path())->at("metadata")));
            EXPECT_NE(metadata.value("format_version", -1.0), -1.0) << entry.path().filename().string();
            EXPECT_FALSE(metadata.contains("av1_grain_dav1d")) << entry.path().filename().string();
        }
    }

    // A context saved before 2.3 (no film grain written) asks for the mark
    // outside Windows: the same unmarked scores no longer load there.
    std::filesystem::remove_all(folder / "cache");
    const std::filesystem::path directory =
        store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                     made("hevc", std::nullopt, false, std::nullopt), "t", request.metrics);
    nlohmann::json context = nlohmann::json::parse(*platform::read_file(directory / "context.json"));
    context["distorted_info"]["codec_name"] = "av1";
    touch(directory / "context.json", context.dump());
    const auto old = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe, request.metrics,
                                 request.perceptual_backends);
    EXPECT_EQ(old ? old->first.metric_results.keys().size() : 0, marks_count ? 0 : all);
    // An AV1 video's grain in another shape: not known either. A video with no
    // codec named makes the context unreadable on every system
    // (video_info_from_dict, as the Python app's): nothing loads.
    for (const auto& [change, expected] :
         std::vector<std::pair<std::function<void(nlohmann::json&)>, std::size_t>>{
             {[](nlohmann::json& info) { info["film_grain"] = "av1"; }, marks_count ? 0 : all},
             {[](nlohmann::json& info) { info.erase("codec_name"); }, 0}}) {
        nlohmann::json odd = context;
        change(odd["distorted_info"]);
        touch(directory / "context.json", odd.dump());
        const auto loaded = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                                        request.metrics, request.perceptual_backends);
        EXPECT_EQ(loaded ? loaded->first.metric_results.keys().size() : 0, expected);
    }

    // A run's scores kept for its next run: outside Windows not a run file's
    // (unmarked) of such a pair; this session's (nothing) and marked ones are.
    const auto kept = [&](const ComparisonResult& result) {
        return reusable_keys(result, request.metrics, request.perceptual_backends,
                             [](const std::string&) { return true; }, {})
            .size();
    };
    const std::size_t keepable = kept(made("hevc", std::nullopt, false, false));
    ASSERT_GT(keepable, 0u);
    EXPECT_EQ(kept(made("av1", av1, false, false)), marks_count ? 0 : keepable);
    EXPECT_EQ(kept(made("av1", std::nullopt, false, false)), marks_count ? 0 : keepable);
    EXPECT_EQ(kept(made("av1", av1, false, std::nullopt)), keepable);
    EXPECT_EQ(kept(made("av1", av1, false, true)), keepable);
}

// A pair with H.264 in a lossless 4:4:4 profile, which NVIDIA's GPUs before
// the RTX 50 series decoded to other pictures (9f983c8), or of no known
// profile (a context saved before 2.3) where the video read now has none
// either, loads only the scores saved since; one with other H.264 loads
// every score, also from a context that has no profile.
TEST(MetricCache, ALosslessH264PairsScoresSavedBeforeExactDecodingAreCalculatedAgain) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    result.distorted_info.codec_name = "h264";
    // How many scores load when the test video's profile was `saved_as` and
    // the comparison's context then says `now` (nothing: not recorded), with
    // the video read now as `read_now`.
    const auto loaded_after = [&](const std::string& saved_as, const std::optional<std::string>& now,
                                  const std::string& read_now = "") {
        request.recipe.distorted_profile = read_now;
        result.distorted_info.profile = saved_as;
        const std::filesystem::path directory = store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4",
                                                             request.recipe, result, "t", request.metrics);
        nlohmann::json context = nlohmann::json::parse(*platform::read_file(directory / "context.json"));
        if (now) {
            context["distorted_info"]["profile"] = *now;
        } else {
            context["distorted_info"].erase("profile");
        }
        touch(directory / "context.json", context.dump());
        const auto loaded = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                                        request.metrics, request.perceptual_backends);
        return loaded ? loaded->first.metric_results.keys().size() : 0;
    };
    const std::size_t all = loaded_after("High", "High");
    EXPECT_EQ(all, request.metrics.size());
    EXPECT_EQ(loaded_after("High", "High 4:4:4 Predictive"), 0U);  // saved as by a build without the mark
    EXPECT_EQ(loaded_after("High", std::nullopt), 0U);
    EXPECT_EQ(loaded_after("High", std::nullopt, "High 10"), all);  // the Python app's context
    EXPECT_EQ(loaded_after("High", std::nullopt, "High 4:4:4 Predictive"), 0U);
    EXPECT_EQ(loaded_after("High 4:4:4 Predictive", "High 4:4:4 Predictive"), all);
}

// Which saved scores answer a request: made by a compatible implementation
// (on the GPU, by the request's Vship build), and not known to be wrong;
// and where a perceptual metric's score counts.
TEST(MetricCache, AnswersAsThePythonApp) {
    const nlohmann::json reference = test::load_reference("metric_cache.json");
    for (const nlohmann::json& item : reference.at("answers")) {
        VmafOptions options;
        options.vship_build = vship_build_from(item.value("vship_build", std::string("vship-fast")));
        const AnalysisRequest request =
            analysis_request(options, {"ssimulacra2", "xpsnr", "vmaf"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
        std::vector<ColourTags> infos;
        for (const nlohmann::json& info : item.at("infos")) {
            infos.push_back({info.at("pix_fmt"), info.at("color_transfer"), {}, {}, {}, 0});  // no siting, matrix, range
        }
        EXPECT_EQ(answers(*request.metric(item.at("key")), test::provenance_from(item.at("provenance")), infos),
                  item.at("answers").get<bool>())
            << item.dump();
    }
    for (const nlohmann::json& item : reference.at("made_as_chosen")) {
        const bool can = item.at("gpu_can_score");
        EXPECT_EQ(made_as_chosen(item.at("key"), text_or_none(item.at("choice")), item.at("produced"),
                                 [can](const std::string&) { return can; }),
                  item.at("counts").get<bool>())
            << item.dump();
    }
    const ComparisonResult result = test::result_from(reference.at("reusable").at("result"));
    for (const nlohmann::json& item : reference.at("reusable").at("cases")) {
        const AnalysisRequest asked = request_from(item);
        const bool can = item.at("gpu_can_score");
        const MetricResultSet kept = reusable(result, asked.metrics, item.at("backends"),
                                              [can](const std::string&) { return can; },
                                              colour_tags(result.source_info, result.distorted_info));
        EXPECT_EQ(nlohmann::json(kept.keys()), item.at("kept")) << item.dump();
    }
}

// A GPU score of Vship 5.1 (v2.1's) is kept for a video set to Vship 5.1.2,
// but for a pair a later colour fix reads otherwise: with two guessed
// matrices or its chroma apart, for a score made before those fixes, or with
// a full-range gray video saved before "/4" (its mappings before cannot
// tell scores made before the gray fix). Vship-fast's scores are kept for a video set to
// Vship-fast; neither build's answers the other's. A CPU score saved before
// 8-bit videos were converted to RGB precisely is calculated again for a
// pair with an 8-bit video; one saved before an untagged pair was read with
// one colour matrix, for a pair whose heights gave the videos different
// guesses; the rest are kept, whichever the build.
TEST(MetricCache, OldScoresAreKeptButForAPairALaterColourFixReadsOtherwise) {
    VmafOptions vship51;
    vship51.vship_build = VshipBuild::stock;
    const AnalysisRequest request =
        analysis_request(vship51, {"ssimulacra2", "cvvdp"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
    const MetricRequestSpec& spec = *request.metric("ssimulacra2");
    const AnalysisRequest fast_request =
        analysis_request(VmafOptions{}, {"ssimulacra2", "cvvdp"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
    const MetricRequestSpec& fast_spec = *fast_request.metric("ssimulacra2");
    const std::vector<ColourTags> guessed_apart{{"yuv420p", "", "", "", "", 720}, {"yuv420p", "", "", "", "", 540}};
    const std::vector<ColourTags> guessed_alike{{"yuv420p", "", "", "", "", 1080}, {"yuv420p", "", "", "", "", 720}};
    const std::vector<ColourTags> one_tagged{{"yuv420p", "", "", "bt709", "", 720}, {"yuv420p", "", "", "", "", 540}};
    const std::vector<ColourTags> tagged{{"yuv420p", "", "", "bt709", "", 720}, {"yuv420p", "", "", "bt470bg", "", 540}};
    const std::vector<ColourTags> siting_apart{{"yuv420p", "bt709", "center", "bt709", "", 1080},
                                               {"yuv420p", "bt709", "", "bt709", "", 1080}};
    const std::vector<ColourTags> siting_left{{"yuv420p", "bt709", "left", "bt709", "", 1080},
                                              {"yuv420p", "bt709", "", "bt709", "", 1080}};
    const std::vector<ColourTags> full_gray{{"yuv420p", "bt709", "", "bt709", "tv", 720},
                                            {"gray", "bt709", "", "bt709", "pc", 720}};
    const std::vector<ColourTags> limited_gray{{"yuv420p", "bt709", "", "bt709", "tv", 720},
                                               {"gray", "bt709", "", "bt709", "tv", 720}};
    MetricProvenance gpu{"Vship/ssimulacra2", "Vship 5.1.2", "gpu", "ssimulacra2-vship-gpu-v1",
                         {{"coverage_step", "1"}, {"color_tags", "\"ffvship-5.1.1\""}}};
    for (const auto& pair : {tagged, guessed_alike, siting_left, limited_gray}) {
        EXPECT_TRUE(answers(spec, gpu, pair)) << pair[1].pix_fmt;
    }
    for (const auto& pair : {guessed_apart, one_tagged, siting_apart, full_gray}) {
        EXPECT_FALSE(answers(spec, gpu, pair)) << pair[1].pix_fmt;
    }
    gpu.parameters["color_tags"] = nlohmann::json(score_identity::vship_colour_tags_unlent).dump();
    EXPECT_FALSE(answers(spec, gpu, full_gray));
    gpu.parameters["color_tags"] = nlohmann::json(score_identity::vship_colour_tags).dump();
    EXPECT_TRUE(answers(spec, gpu, guessed_apart));
    EXPECT_TRUE(answers(spec, gpu, siting_apart));
    EXPECT_TRUE(answers(spec, gpu, full_gray));  // as this version saves them: not calculated again each run
    const MetricProvenance fast{"Vship/ssimulacra2", "Vship-fast 5.1.2", "gpu",
                                score_identity::vship_gpu("ssimulacra2"), gpu.parameters};
    EXPECT_TRUE(answers(fast_spec, fast, full_gray));
    EXPECT_FALSE(answers(spec, fast, tagged));
    EXPECT_FALSE(answers(fast_spec, gpu, tagged));
    const MetricProvenance fast_cvvdp{"Vship/cvvdp", "Vship-fast 5.1.2", "gpu", std::string(score_identity::cvvdp), {}};
    const MetricProvenance cvvdp51{"Vship/cvvdp", "Vship 5.1.2", "gpu", std::string(score_identity::vship51_cvvdp), {}};
    EXPECT_TRUE(answers(*fast_request.metric("cvvdp"), fast_cvvdp, tagged));
    EXPECT_FALSE(answers(*fast_request.metric("cvvdp"), cvvdp51, tagged));
    EXPECT_TRUE(answers(*request.metric("cvvdp"), cvvdp51, tagged));
    EXPECT_FALSE(answers(*request.metric("cvvdp"), fast_cvvdp, tagged));
    // The same pairs in 10 bits, where the CPU's RGB did not change.
    const auto ten_bit = [](std::vector<ColourTags> pair) {
        for (ColourTags& info : pair) {
            info.pix_fmt = "yuv420p10le";
        }
        return pair;
    };
    MetricProvenance cpu{"ssimulacra2", "libjxl 0.11", "cpu", "ssimulacra2-libjxl-cpu-v1",
                         {{"coverage_step", "1"}, {"color_tags", "\"vship-5.1.1/2\""}}};
    for (const MetricRequestSpec* asked : {&spec, &fast_spec}) {
        EXPECT_FALSE(answers(*asked, cpu, ten_bit(guessed_apart)));
        EXPECT_FALSE(answers(*asked, cpu, ten_bit(one_tagged)));
        EXPECT_TRUE(answers(*asked, cpu, ten_bit(guessed_alike)));
        EXPECT_TRUE(answers(*asked, cpu, ten_bit(tagged)));
        EXPECT_FALSE(answers(*asked, cpu, tagged));  // 8-bit
    }
    cpu.parameters["color_tags"] = nlohmann::json(score_identity::cpu_colour_tags_8_bit_rgb).dump();
    EXPECT_TRUE(answers(spec, cpu, ten_bit(guessed_apart)));
    EXPECT_FALSE(answers(spec, cpu, guessed_alike));  // 8-bit
    cpu.parameters["color_tags"] = nlohmann::json(score_identity::cpu_colour_tags).dump();
    EXPECT_TRUE(answers(spec, cpu, guessed_apart));
    // FFmpeg's XPSNR above 8 bits, which it may have read wrongly; the app's
    // is kept.
    const AnalysisRequest xpsnr_request =
        analysis_request(VmafOptions{}, {"xpsnr"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
    const MetricRequestSpec& xpsnr_spec = *xpsnr_request.metric("xpsnr");
    MetricProvenance xpsnr = current_ffmpeg_provenance("xpsnr", "9.0.1", {});
    EXPECT_TRUE(answers(xpsnr_spec, xpsnr, tagged));
    EXPECT_FALSE(answers(xpsnr_spec, xpsnr, ten_bit(tagged)));
    xpsnr.implementation = "libvmaf";
    EXPECT_TRUE(answers(xpsnr_spec, xpsnr, ten_bit(tagged)));
}

// Two untagged videos are read with the reference's guess, and one untagged
// video with the other's tag.
TEST(MetricCache, AnUntaggedPairIsReadWithOneMatrix) {
    VideoInfo source;
    source.pix_fmt = "yuv420p";
    source.height = 720;
    VideoInfo distorted = source;
    distorted.height = 540;
    auto [source_read, distorted_read] = with_shared_matrix(source, distorted);
    EXPECT_EQ(source_read.color_space, "bt709");
    EXPECT_EQ(distorted_read.color_space, "bt709");
    EXPECT_TRUE(matrix_shared(source, distorted));
    distorted.color_space = "bt470bg";
    std::tie(source_read, distorted_read) = with_shared_matrix(source, distorted);
    EXPECT_EQ(source_read.color_space, "bt470bg");
    EXPECT_TRUE(matrix_shared(source, distorted));
    source.height = 480;
    distorted.color_space.clear();
    EXPECT_FALSE(matrix_shared(source, distorted));
}

// A video without colour tags is read with the other's, whichever it is; a
// score saved before that is calculated again only where that reads it
// otherwise.
TEST(MetricCache, AnUntaggedVideoIsReadWithTheOthersColours) {
    VideoInfo hlg;
    hlg.pix_fmt = "yuv420p10le";
    hlg.height = 720;
    hlg.color_transfer = "arib-std-b67";
    hlg.color_primaries = "bt2020";
    hlg.color_space = "bt2020nc";
    VideoInfo untagged = hlg;
    untagged.color_transfer.clear();
    untagged.color_primaries.clear();
    untagged.color_space.clear();

    for (const auto& [source_read, distorted_read] :
         {with_shared_colours(hlg, untagged), with_shared_colours(untagged, hlg)}) {
        EXPECT_EQ(source_read.color_transfer, "arib-std-b67");
        EXPECT_EQ(distorted_read.color_transfer, "arib-std-b67");
        EXPECT_EQ(distorted_read.color_primaries, "bt2020");
        EXPECT_EQ(distorted_read.color_space, "bt2020nc");
    }
    EXPECT_TRUE(colours_lent(hlg, untagged));
    EXPECT_FALSE(colours_lent(hlg, hlg));
    EXPECT_FALSE(colours_lent(untagged, untagged));

    const AnalysisRequest request =
        analysis_request(VmafOptions{}, {"ssimulacra2"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
    const MetricRequestSpec& spec = *request.metric("ssimulacra2");
    const MetricProvenance saved{"Vship/ssimulacra2", "Vship-fast 5.1.2", "gpu",
                                 score_identity::vship_gpu("ssimulacra2"),
                                 {{"coverage_step", "1"},
                                  {"color_tags", nlohmann::json(score_identity::vship_colour_tags_unlent).dump()}}};
    EXPECT_FALSE(answers(spec, saved, colour_tags(hlg, untagged)));
    EXPECT_TRUE(answers(spec, saved, colour_tags(hlg, hlg)));
}

// What each request loads from a comparison folder the Python app saved:
// the same scores, frames and provenance, and nothing where it finds none.
TEST(MetricCache, LoadsThePythonAppsSavedScores) {
    const nlohmann::json saved = test::load_reference("metric_cache.json").at("saved");
    const test::Folder folder;
    const std::filesystem::path base = folder / "cache";
    lay_saved_scores(folder, base, request_from(saved.at("loads").at(0)).recipe);
    for (const nlohmann::json& item : saved.at("loads")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const AnalysisRequest request = request_from(item);
        expect_loaded(load_result(base, folder / "s.mp4", folder / "t.mp4", request.recipe, request.metrics,
                                  request.perceptual_backends),
                      item.at("loaded"), saved.at("label"));
    }
}

// What the window finds for a row in the Python app's saved scores: the
// scores its ticked metrics ask for, every other score it can show
// (displayable_metric_specs), and CVVDP's scores for other displays.
TEST(MetricCache, FindsWhatTheWindowShowsAsThePythonApp) {
    const nlohmann::json saved = test::load_reference("metric_cache.json").at("saved");
    const test::Folder folder;
    const std::filesystem::path base = folder / "cache";
    lay_saved_scores(folder, base, request_from(saved.at("window_loads").at(0)).recipe);
    for (const nlohmann::json& item : saved.at("window_loads")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const AnalysisRequest request = request_from(item);
        const std::vector<MetricRequestSpec> shown =
            displayable_metric_specs(options_from(item.at("options")), cvvdp_from(item));
        std::vector<std::string> identities;
        for (const MetricRequestSpec& spec : shown) {
            identities.push_back(python_json_compact(spec.identity()));
        }
        EXPECT_EQ(nlohmann::json(identities), item.at("displayable"));
        expect_loaded(load_result(base, folder / "s.mp4", folder / "t.mp4", request.recipe, request.metrics,
                                  request.perceptual_backends, shown),
                      item.at("loaded"), saved.at("label"));
        const std::optional<OtherCvvdpScores> others =
            other_cvvdp_scores(base, folder / "s.mp4", folder / "t.mp4", request.recipe, request.metrics, shown);
        const nlohmann::json& expected = item.at("other_cvvdp");
        ASSERT_EQ(others.has_value(), !expected.is_null());
        if (others) {
            EXPECT_EQ(others->parameters, expected.at("parameters"));
            ASSERT_EQ(others->scores.size(), expected.at("scores").size());
            for (std::size_t i = 0; i < others->scores.size(); ++i) {
                EXPECT_EQ(nlohmann::json(others->scores[i].first.parameters()),
                          expected.at("scores")[i].at("parameters"));
                EXPECT_EQ(others->scores[i].second, *test::reference_number(expected.at("scores")[i].at("score")));
            }
        }
    }
}

// What Recalculate leaves of the Python app's saved scores: each request's
// scores go, with those that would answer it, and the folder with its last
// score.
TEST(MetricCache, ClearsAsThePythonApp) {
    const nlohmann::json clears = test::load_reference("metric_cache.json").at("saved").at("clears");
    for (const nlohmann::json& item : clears) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const test::Folder folder;
        const std::filesystem::path base = folder / "cache";
        const AnalysisRequest request = request_from(item);
        lay_saved_scores(folder, base, request.recipe);
        EXPECT_EQ(clear_metrics(base, folder / "s.mp4", folder / "t.mp4", request.recipe, request.metrics),
                  item.at("removed").get<int>());
        std::vector<std::string> left;
        const std::filesystem::path directory =
            recipe_directory(base, folder / "s.mp4", folder / "t.mp4", request.recipe);
        std::error_code error;
        for (std::filesystem::directory_iterator entry(directory, error);
             !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
            left.push_back(platform::path_text(entry->path().filename()));
        }
        std::sort(left.begin(), left.end());
        EXPECT_EQ(nlohmann::json(left), item.at("left"));
    }
}

// A VMAF from the saved scores has the model its own file says it was made
// with, and the choice of the request it answers, whatever was saved after
// it: CVVDP alone (no model), or VMAF with another model chosen. Each save
// rewrites context.json with the models of the result being saved.
TEST(MetricCache, ASavedVmafKeepsTheModelItWasMadeWith) {
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const ComparisonResult made = test::result_from(test::load_reference("metric_cache.json").at("stored").at("result"));
    const std::string standard = "version=vmaf_v0.6.1";
    const std::string four_k = "version=vmaf_4k_v0.6.1";
    using Models = std::pair<std::string, std::optional<std::string>>;
    // Saves VMAF made with `model` for this model choice; with no model,
    // CVVDP alone.
    const auto save = [&](const std::string& choice, const std::string& model) {
        VmafOptions options;
        options.model_choice = choice;
        ComparisonResult result = made;
        result.metric_results = MetricResultSet();
        result.model = model;
        if (model.empty()) {
            result.model_choice.reset();
            result.metric_results.add(*made.metric_results.sequence("cvvdp"));
        } else {
            result.model_choice = choice;
            FrameMetricResult vmaf = *made.metric_results.frame("vmaf");
            vmaf.provenance.parameters["model"] = nlohmann::json(model).dump();
            result.metric_results.add(vmaf);
        }
        const AnalysisRequest request = analysis_request(options, {model.empty() ? "cvvdp" : "vmaf"}, {},
                                                         CvvdpSettings{}, VideoInfo{}, VideoInfo{});
        store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe, result, "t", request.metrics);
    };
    const auto load = [&](const std::string& choice) {
        VmafOptions options;
        options.model_choice = choice;
        const AnalysisRequest request =
            analysis_request(options, {"vmaf"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
        const auto loaded = load_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                                        request.metrics, request.perceptual_backends);
        return loaded ? Models(loaded->first.model, loaded->first.model_choice) : Models();
    };

    save(four_k, four_k);
    save(four_k, "");  // another display's CVVDP alone, saved last
    EXPECT_EQ(load(four_k), Models(four_k, four_k));
    save(standard, standard);  // VMAF with another model, saved last
    EXPECT_EQ(load(four_k), Models(four_k, four_k));
    EXPECT_EQ(load(standard), Models(standard, standard));
}

// Arrays come back from an .npz file as they went in, text past ASCII too,
// and a file that is not one reads as none.
TEST(Npz, ReadsWhatItWrites) {
    const test::Folder folder;
    const std::string bytes = npz_bytes({{"metadata", npy_text("{\"a\":\"réf 動画 🎬\"}")},
                                         {"frame", npy_int32({0, 3, -7})},
                                         {"time", npy_float64({0.0, 0.125, 1e300})},
                                         {"values", npy_float32({1.5f, -0.0f, 3.25f})},
                                         {"score", npy_float64_scalar(9.75)}});
    touch(folder / "a.npz", bytes);
    const auto arrays = read_npz(folder / "a.npz");
    ASSERT_TRUE(arrays.has_value());
    EXPECT_EQ(npy_text_value(arrays->at("metadata")), "{\"a\":\"réf 動画 🎬\"}");
    EXPECT_EQ(npy_numbers(arrays->at("frame")), (std::vector<double>{0, 3, -7}));
    EXPECT_EQ(npy_numbers(arrays->at("time")), (std::vector<double>{0.0, 0.125, 1e300}));
    EXPECT_EQ(npy_numbers(arrays->at("values")), (std::vector<double>{1.5, -0.0, 3.25}));
    EXPECT_EQ(npy_numbers(arrays->at("score")), (std::vector<double>{9.75}));
    EXPECT_EQ(arrays->at("score").shape, std::vector<std::size_t>{});
    touch(folder / "b.npz", "not a zip");
    EXPECT_FALSE(read_npz(folder / "b.npz").has_value());
}

// A video set to Vship Vulkan, HIP or CUDA asks for one saved score; set to Vship-fast, for another.
TEST(MetricCache, StockVshipsBuildsShareTheirSavedScores) {
    const auto ids_of = [](VshipBuild build) {
        VmafOptions options;
        options.vship_build = build;
        const VideoInfo video;
        std::vector<std::string> ids;
        for (const MetricRequestSpec& spec :
             analysis_request(options, {"ssimulacra2", "cvvdp"}, {}, CvvdpSettings{}, video, video).metrics) {
            ids.push_back(spec.implementation_compatibility_id);
        }
        return ids;
    };
    EXPECT_EQ(ids_of(VshipBuild::hip), ids_of(VshipBuild::stock));
    EXPECT_EQ(ids_of(VshipBuild::cuda), ids_of(VshipBuild::stock));
    EXPECT_NE(ids_of(VshipBuild::fast), ids_of(VshipBuild::stock));
}

}  // namespace
}  // namespace vml
