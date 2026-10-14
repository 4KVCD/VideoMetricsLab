// The saved scores, held to the Python app's (reference/metric_cache.json,
// and the comparison folder reference/metric_cache it saved): the names
// requests and recipes are filed under, the files a stored result writes,
// which scores answer a request, what requests load from the Python app's
// files, and the VMAF model a loaded result has.
#include "vml/metric_cache.h"

#include <gtest/gtest.h>

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
    return options;
}

AnalysisRequest request_from(const nlohmann::json& item) {
    std::vector<std::string> keys = item.at("metrics");
    std::map<std::string, std::string> backends = item.at("backends");
    const CvvdpSettings cvvdp =
        item.at("cvvdp").is_null() ? CvvdpSettings{} : CvvdpSettings::from_parameters(item.at("cvvdp"));
    return analysis_request(options_from(item.at("options")), keys, backends, cvvdp);
}

MetricProvenance provenance_from(const nlohmann::json& data) {
    MetricProvenance provenance{data.at("implementation"), data.at("implementation_version"),
                                data.at("compute_backend"), data.at("implementation_compatibility_id"), {}};
    for (auto item = data.at("parameters").begin(); item != data.at("parameters").end(); ++item) {
        provenance.parameters[item.key()] = item.value().dump();
    }
    return provenance;
}

std::vector<float> floats_from(const nlohmann::json& list) {
    std::vector<float> values;
    for (const double value : test::reference_numbers(list)) {
        values.push_back(static_cast<float>(value));
    }
    return values;
}

VideoInfo video_from(const nlohmann::json& data) {
    VideoInfo info;
    info.path = platform::path_from_text(data.at("path").get<std::string>());
    info.width = data.at("width");
    info.height = data.at("height");
    info.fps = *test::reference_number(data.at("fps"));
    info.duration = *test::reference_number(data.at("duration"));
    info.nb_frames = data.at("nb_frames");
    info.codec_name = data.at("codec_name");
    info.sar = data.at("sar");
    info.pix_fmt = data.at("pix_fmt");
    info.bit_rate = data.at("bit_rate");
    info.bit_rate_whole_file = data.at("bit_rate_whole_file");
    info.nominal_fps = *test::reference_number(data.at("nominal_fps"));
    info.average_fps = *test::reference_number(data.at("average_fps"));
    info.color_range = data.at("color_range");
    info.color_space = data.at("color_space");
    info.color_transfer = data.at("color_transfer");
    info.color_primaries = data.at("color_primaries");
    info.chroma_location = data.at("chroma_location");
    return info;
}

std::optional<CropBox> crop_from(const nlohmann::json& data) {
    if (data.is_null()) {
        return std::nullopt;
    }
    return CropBox{data[0], data[1], data[2], data[3]};
}

std::optional<std::string> text_or_none(const nlohmann::json& data) {
    if (data.is_null()) {
        return std::nullopt;
    }
    return data.get<std::string>();
}

// A result as the reference describes it (make_reference.py's result_json).
ComparisonResult result_from(const nlohmann::json& data) {
    ComparisonResult result;
    result.source = platform::path_from_text(data.at("source").get<std::string>());
    result.distorted = platform::path_from_text(data.at("distorted").get<std::string>());
    result.fps = *test::reference_number(data.at("fps"));
    result.model = data.at("model");
    result.source_crop = crop_from(data.at("source_crop"));
    result.distorted_crop = crop_from(data.at("distorted_crop"));
    result.source_info = video_from(data.at("source_info"));
    result.distorted_info = video_from(data.at("distorted_info"));
    result.scale_direction = data.at("scale_direction") == "distorted_to_source" ? ScaleDirection::distorted_to_source
                                                                                 : ScaleDirection::source_to_distorted;
    result.scale_algorithm = data.at("scale_algorithm");
    result.compared_frame_count = data.at("compared_frame_count");
    result.model_choice = text_or_none(data.at("model_choice"));
    result.model_v1 = data.at("model_v1");
    result.model_choice_v1 = text_or_none(data.at("model_choice_v1"));
    for (const nlohmann::json& metric : data.at("metrics")) {
        const std::string key = metric.at("key");
        if (metric.at("kind") == "sequence") {
            SequenceMetricResult sequence{key, *test::reference_number(metric.at("score")),
                                          provenance_from(metric.at("provenance")), {}, {}, {}};
            if (metric.contains("values")) {
                sequence.frame = metric.at("frame").get<std::vector<int>>();
                sequence.time = test::reference_numbers(metric.at("time"));
                sequence.values = floats_from(metric.at("values"));
            }
            result.metric_results.add(sequence);
        } else {
            result.metric_results.add(FrameMetricResult{key, metric.at("frame").get<std::vector<int>>(),
                                                        test::reference_numbers(metric.at("time")),
                                                        floats_from(metric.at("values")),
                                                        provenance_from(metric.at("provenance"))});
        }
    }
    return result;
}

void touch(const std::filesystem::path& path, const std::string& text) { std::ofstream(path, std::ios::binary) << text; }

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
// the same names, the same metadata to the byte.
TEST(MetricCache, StoresAResultAsThePythonApp) {
    const nlohmann::json stored = test::load_reference("metric_cache.json").at("stored");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const AnalysisRequest request = request_from(stored);
    const std::filesystem::path directory =
        store_result(folder / "cache", folder / "s.mp4", folder / "t.mp4", request.recipe,
                     result_from(stored.at("result")), stored.at("label"), request.metrics);
    EXPECT_EQ(platform::read_file(directory / "context.json"), stored.at("context").get<std::string>());
    std::map<std::string, std::string> files;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".npz") {
            const auto arrays = read_npz(entry.path());
            ASSERT_TRUE(arrays.has_value()) << entry.path();
            files[platform::path_text(entry.path().filename())] = npy_text_value(arrays->at("metadata")).value_or("");
        }
    }
    EXPECT_EQ(nlohmann::json(files), stored.at("files"));
}

// Which saved scores answer a request: made by a compatible implementation,
// and not known to be wrong; and where a perceptual metric's score counts.
TEST(MetricCache, AnswersAsThePythonApp) {
    const nlohmann::json reference = test::load_reference("metric_cache.json");
    const AnalysisRequest request = analysis_request(VmafOptions{}, {"ssimulacra2", "xpsnr", "vmaf"}, {}, CvvdpSettings{});
    for (const nlohmann::json& item : reference.at("answers")) {
        std::vector<ColourTags> infos;
        for (const nlohmann::json& info : item.at("infos")) {
            infos.push_back({info.at("pix_fmt"), info.at("color_transfer")});
        }
        EXPECT_EQ(answers(*request.metric(item.at("key")), provenance_from(item.at("provenance")), infos),
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
    const ComparisonResult result = result_from(reference.at("reusable").at("result"));
    for (const nlohmann::json& item : reference.at("reusable").at("cases")) {
        const AnalysisRequest asked = request_from(item);
        const bool can = item.at("gpu_can_score");
        const MetricResultSet kept = reusable(result, asked.metrics, item.at("backends"),
                                              [can](const std::string&) { return can; },
                                              colour_tags(result.source_info, result.distorted_info));
        EXPECT_EQ(nlohmann::json(kept.keys()), item.at("kept")) << item.dump();
    }
}

// What each request loads from a comparison folder the Python app saved:
// the same scores, frames and provenance, and nothing where it finds none.
TEST(MetricCache, LoadsThePythonAppsSavedScores) {
    const nlohmann::json saved = test::load_reference("metric_cache.json").at("saved");
    const test::Folder folder;
    touch(folder / "s.mp4", "s");
    touch(folder / "t.mp4", "t");
    const std::filesystem::path base = folder / "cache";
    bool laid = false;
    for (const nlohmann::json& item : saved.at("loads")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const AnalysisRequest request = request_from(item);
        if (!laid) {
            // The Python app's files, under the name this comparison has here.
            const std::filesystem::path directory = recipe_directory(base, folder / "s.mp4", folder / "t.mp4", request.recipe);
            std::filesystem::create_directories(directory);
            const std::filesystem::path from =
                std::filesystem::path(VML_REFERENCE_DIR) / saved.at("folder").get<std::string>();
            for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(from)) {
                std::filesystem::copy_file(entry.path(), directory / entry.path().filename());
            }
            laid = true;
        }
        const auto loaded = load_result(base, folder / "s.mp4", folder / "t.mp4", request.recipe, request.metrics,
                                        request.perceptual_backends);
        if (item.at("loaded").empty()) {
            EXPECT_FALSE(loaded.has_value());
            continue;
        }
        ASSERT_TRUE(loaded.has_value());
        EXPECT_EQ(loaded->second, saved.at("label").get<std::string>());
        const MetricResultSet& results = loaded->first.metric_results;
        ASSERT_EQ(results.keys().size(), item.at("loaded").size());
        for (const nlohmann::json& expected : item.at("loaded")) {
            const std::string key = expected.at("key");
            SCOPED_TRACE(key);
            const MetricProvenance wanted = provenance_from(expected.at("provenance"));
            const MetricProvenance* provenance = nullptr;
            if (expected.at("kind") == "sequence") {
                const SequenceMetricResult* sequence = results.sequence(key);
                ASSERT_NE(sequence, nullptr);
                EXPECT_EQ(sequence->score, *test::reference_number(expected.at("score")));
                EXPECT_EQ(sequence->values, expected.contains("values") ? floats_from(expected.at("values")) : std::vector<float>{});
                provenance = &sequence->provenance;
            } else {
                const FrameMetricResult* frame = results.frame(key);
                ASSERT_NE(frame, nullptr);
                EXPECT_EQ(frame->frame, expected.at("frame").get<std::vector<int>>());
                EXPECT_EQ(frame->time, test::reference_numbers(expected.at("time")));
                EXPECT_EQ(frame->values, floats_from(expected.at("values")));
                provenance = &frame->provenance;
            }
            EXPECT_EQ(provenance->implementation, wanted.implementation);
            EXPECT_EQ(provenance->implementation_version, wanted.implementation_version);
            EXPECT_EQ(provenance->compute_backend, wanted.compute_backend);
            EXPECT_EQ(provenance->implementation_compatibility_id, wanted.implementation_compatibility_id);
            EXPECT_EQ(provenance->parameters, wanted.parameters);
        }
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
    const ComparisonResult made = result_from(test::load_reference("metric_cache.json").at("stored").at("result"));
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
