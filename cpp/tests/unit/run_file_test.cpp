// Saved runs, held to the Python app's (reference/run_files.json, made by
// its own save_run and load_run): the text a result is saved as, and what
// a file loads as, or why it does not.
#include "vml/run_file.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "reference.h"
#include "vml/build_info.h"

namespace vml {
namespace {

// The Python app's file `text` as this app saves it: with the build that
// saved it last, which the Python app did not say.
std::string with_saved_by(const std::string& text) {
    return text.substr(0, text.size() - 1) + ", \"saved_by\": \"" + build_stamp() + "\"}";
}

// Every kind of score, crops, VMAF v1's model, a resolution test, and no
// label: the file is the Python app's to the byte, but for saved_by.
TEST(RunFile, SavesAResultAsThePythonApp) {
    // Held here: a loop over load_reference(...).at(...) would read the
    // list after the file's JSON is gone.
    const nlohmann::json reference = test::load_reference("run_files.json");
    ASSERT_FALSE(reference.at("saved").empty());
    for (const nlohmann::json& item : reference.at("saved")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        EXPECT_EQ(run_file_text(test::result_from(item.at("result")), item.at("label").get<std::string>()),
                  with_saved_by(item.at("text").get<std::string>()));
    }
}

// A file loads as the result the Python app loads from it, which both save
// alike (the text it saves again); or it is refused with the Python app's
// reason, which the window shows.
TEST(RunFile, LoadsFilesAsThePythonApp) {
    const nlohmann::json reference = test::load_reference("run_files.json");
    ASSERT_FALSE(reference.at("loads").empty());
    for (const nlohmann::json& item : reference.at("loads")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const std::string text = item.at("text");
        if (item.contains("error")) {
            try {
                run_from_text(text);
                ADD_FAILURE() << "loaded";
            } catch (const RunFileError& error) {
                // Python's JSON parser words its errors its own way.
                if (const std::string why = item.at("error"); !why.empty()) {
                    EXPECT_EQ(error.what(), why);
                }
            }
            continue;
        }
        const auto [result, label] = run_from_text(text);
        EXPECT_EQ(label, item.at("label").get<std::string>());
        EXPECT_EQ(run_file_text(result, label), with_saved_by(item.at("resaved").get<std::string>()));
        // A run file records no decoding marks: its scores say none
        // (metric_cache.cpp's av1_grain_dav1d).
        for (const std::string& key : result.metric_results.keys()) {
            const FrameMetricResult* frame = result.metric_results.frame(key);
            EXPECT_EQ(frame ? frame->av1_grain_dav1d : result.metric_results.sequence(key)->av1_grain_dav1d,
                      std::optional(false))
                << key;
        }
    }
}

// A video's H.264 profile is saved where it is known, and loads again:
// without it, Intel's and AMD's GPUs leave the video to software decoding
// (pick_decode). A video without one is saved as before.
TEST(RunFile, KeepsTheVideosProfile) {
    const nlohmann::json reference = test::load_reference("run_files.json");
    ComparisonResult result = test::result_from(reference.at("saved").at(0).at("result"));
    result.source_info.profile = "High";
    result.distorted_info.profile.clear();
    const std::string text = run_file_text(result, "");
    const nlohmann::json saved = nlohmann::json::parse(text);
    EXPECT_EQ(saved.at("source_info").at("profile"), "High");
    EXPECT_FALSE(saved.at("distorted_info").contains("profile"));
    const auto [loaded, label] = run_from_text(text);
    EXPECT_EQ(loaded.source_info.profile, "High");
    EXPECT_EQ(loaded.distorted_info.profile, "");
}

}  // namespace
}  // namespace vml
