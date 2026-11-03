// The saved scores, held to the Python app's (reference/metric_cache.json,
// and the comparison folder reference/metric_cache it saved): the names
// requests and recipes are filed under, the files a stored result writes
// (with each score's version, which the Python app left blank but for VMAF's),
// which scores answer a request, what requests load from the Python app's
// files, and the VMAF model a loaded result has.
#include "vml/metric_cache.h"
#include "vml/colour.h"
#include "vml/cvvdp.h"
#include "vml/film_grain_detect.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/npz.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
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
        EXPECT_EQ(platform::read_file(directory / "context.json"), stored.at("context").get<std::string>());
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
// profile (a context saved before 2.3), loads only the scores saved since;
// one with other H.264 loads every score.
TEST(MetricCache, ALosslessH264PairsScoresSavedBeforeExactDecodingAreCalculatedAgain) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const AnalysisRequest request = request_from(stored);
    ComparisonResult result = test::result_from(stored.at("result"));
    result.distorted_info.codec_name = "h264";
    // How many scores load when the test video's profile was `saved_as` and
    // the comparison's context then says `now` (nothing: not recorded).
    const auto loaded_after = [&](const std::string& saved_as, const std::optional<std::string>& now) {
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
// a full-range gray video. Vship-fast's scores are kept for a video set to
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
    gpu.parameters["color_tags"] = nlohmann::json(score_identity::vship_colour_tags).dump();
    EXPECT_TRUE(answers(spec, gpu, guessed_apart));
    EXPECT_TRUE(answers(spec, gpu, siting_apart));
    EXPECT_FALSE(answers(spec, gpu, full_gray));
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

}  // namespace
}  // namespace vml
