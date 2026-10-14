// Saved runs, held to the Python app's (reference/run_files.json, made by
// its own save_run and load_run): the text a result is saved as, and what
// a file loads as, or why it does not.
#include "vml/run_file.h"

#include <gtest/gtest.h>

#include <string>

#include "reference.h"

namespace vml {
namespace {

// Every kind of score, crops, VMAF v1's model, a resolution test, and no
// label: the file is the Python app's to the byte.
TEST(RunFile, SavesAResultAsThePythonApp) {
    // Held here: a loop over load_reference(...).at(...) would read the
    // list after the file's JSON is gone.
    const nlohmann::json reference = test::load_reference("run_files.json");
    ASSERT_FALSE(reference.at("saved").empty());
    for (const nlohmann::json& item : reference.at("saved")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        EXPECT_EQ(run_file_text(test::result_from(item.at("result")), item.at("label").get<std::string>()),
                  item.at("text").get<std::string>());
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
        EXPECT_EQ(run_file_text(result, label), item.at("resaved").get<std::string>());
    }
}

}  // namespace
}  // namespace vml
