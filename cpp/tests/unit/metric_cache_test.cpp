// The saved scores, held to the Python app's (reference/metric_cache.json,
// and the comparison folder reference/metric_cache it saved): the names
// requests and recipes are filed under, the files a stored result writes,
// which scores answer a request, what requests load from the Python app's
// files, and the VMAF model a loaded result has.
#include "vml/metric_cache.h"
#include "vml/colour.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
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
    options.crop_mode = fields.at("crop_mode") == "none" ? CropMode::none : CropMode::automatic;
    options.scale_direction = fields.at("scale_direction") == "distorted_to_source" ? ScaleDirection::distorted_to_source
                                                                                    : ScaleDirection::source_to_distorted;
    options.scale_algorithm = fields.at("scale_algorithm").get<std::string>();
    options.duration_limit = *test::reference_number(fields.at("duration_limit"));
    if (const nlohmann::json& target = fields.at("resample_test"); !target.is_null()) {
        options.resample_test = ResampleTarget{target.at("width"), target.at("label")};
    }
    return options;
}

AnalysisRequest request_from(const nlohmann::json& item) {
    std::vector<std::string> keys = item.at("metrics");
    std::map<std::string, std::string> backends = item.at("backends");
    const CvvdpSettings cvvdp =
        item.at("cvvdp").is_null() ? CvvdpSettings{} : CvvdpSettings::from_parameters(item.at("cvvdp"));
    return analysis_request(options_from(item.at("options")), keys, backends, cvvdp);
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

// Storing a result writes the Python app's context.json and score files:
// the same names, the same metadata to the byte. A resolution test's keeps
// its target, and comes back with it.
TEST(MetricCache, StoresAResultAsThePythonApp) {
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

// Which saved scores answer a request: made by a compatible implementation,
// and not known to be wrong; and where a perceptual metric's score counts.
TEST(MetricCache, AnswersAsThePythonApp) {
    const nlohmann::json reference = test::load_reference("metric_cache.json");
    const AnalysisRequest request = analysis_request(VmafOptions{}, {"ssimulacra2", "xpsnr", "vmaf"}, {}, CvvdpSettings{});
    for (const nlohmann::json& item : reference.at("answers")) {
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

// A GPU score of Vship 5.1 (v2.1's) is kept, but for a pair a later colour
// fix reads otherwise: with two guessed matrices or its chroma apart, for a
// score made before those fixes, or with a full-range gray video.
// Vship-fast's scores are kept. A CPU score saved before an untagged pair
// was read with one colour matrix is calculated again for a pair whose
// heights gave the videos different guesses, and kept for the rest.
TEST(MetricCache, OldScoresAreKeptButForAPairALaterColourFixReadsOtherwise) {
    const AnalysisRequest request = analysis_request(VmafOptions{}, {"ssimulacra2"}, {}, CvvdpSettings{});
    const MetricRequestSpec& spec = *request.metric("ssimulacra2");
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
    EXPECT_TRUE(answers(spec, fast, full_gray));
    MetricProvenance cpu{"ssimulacra2", "libjxl 0.11", "cpu", "ssimulacra2-libjxl-cpu-v1",
                         {{"coverage_step", "1"}, {"color_tags", "\"vship-5.1.1/2\""}}};
    EXPECT_FALSE(answers(spec, cpu, guessed_apart));
    EXPECT_FALSE(answers(spec, cpu, one_tagged));
    EXPECT_TRUE(answers(spec, cpu, guessed_alike));
    EXPECT_TRUE(answers(spec, cpu, tagged));
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
        const AnalysisRequest request =
            analysis_request(options, {model.empty() ? "cvvdp" : "vmaf"}, {}, CvvdpSettings{});
        store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe, result, "t", request.metrics);
    };
    const auto load = [&](const std::string& choice) {
        VmafOptions options;
        options.model_choice = choice;
        const AnalysisRequest request = analysis_request(options, {"vmaf"}, {}, CvvdpSettings{});
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
