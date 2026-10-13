// Work run in a process of its own (isolated.h), through vml-cli, the
// program the children run, with its tasks for the tests.
#include "vml/isolated.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vml/platform/system.h"

namespace vml {
namespace {

// The child's messages come back in order, then its result and the file it
// left; what it throws is thrown here; a child that ends without a result
// crashed, as the Python app words it; and Cancel ends the child.
TEST(Isolated, RunsWorkInAProcessOfItsOwn) {
    set_isolation_program(std::filesystem::path(VML_CLI));
    std::vector<nlohmann::json> messages;
    std::string echoed;
    const nlohmann::json request{{"file", "written"}, {"n", 1.25}};
    const nlohmann::json result = run_isolated(
        "test-echo", "the test", request, [&](const nlohmann::json& message) { messages.push_back(message); },
        CancelToken(), [&](const nlohmann::json&, const std::filesystem::path& folder) {
            std::ifstream file(folder / "echo.txt");
            std::getline(file, echoed);
        });
    EXPECT_EQ(result, request);
    ASSERT_EQ(messages.size(), 2u);
    EXPECT_EQ(messages[0], (nlohmann::json{{"progress", {1, 2, 3.5}}}));
    EXPECT_EQ(messages[1], (nlohmann::json{{"status", "on its way"}}));
    EXPECT_EQ(echoed, "written");

    try {
        run_isolated("test-crash", "the test", nlohmann::json::object(), {}, CancelToken());
        ADD_FAILURE() << "no crash";
    } catch (const IsolatedCrashError& error) {
#ifdef _WIN32
        EXPECT_STREQ(error.what(), "the test crashed (exit code 0xC0000005)");
#else
        EXPECT_STREQ(error.what(), "the test crashed (exit code 5)");  // an exit status keeps its low 8 bits
#endif
    }

    try {
        run_isolated("no-such-task", "the test", nlohmann::json::object(), {}, CancelToken());
        ADD_FAILURE() << "no error";
    } catch (const IsolatedCrashError&) {
        ADD_FAILURE() << "an error the child reports is not a crash";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "there is no task no-such-task");
    }

    // Cancelled once the child says it is waiting: it would wait a minute.
    CancelToken cancel;
    EXPECT_THROW(run_isolated(
                     "test-wait", "the test", nlohmann::json::object(),
                     [&cancel](const nlohmann::json&) { cancel.cancel(); }, cancel),
                 Cancelled);
    set_isolation_program(std::nullopt);
}

// A program the child started can still have a file in the child's folder
// open for a moment after a Cancel has ended them both (FFmpeg, its
// timestamps): the folder goes all the same.
TEST(Isolated, ItsFolderGoesWhenAProgramItStartedHadAFileThereOpen) {
    set_isolation_program(std::filesystem::path(VML_CLI));
    std::filesystem::path folder;
    CancelToken cancel;
    EXPECT_THROW(run_isolated(
                     "test-hold", "the test", nlohmann::json::object(),
                     [&](const nlohmann::json& message) {
                         folder = platform::path_from_text(message.at("folder").get<std::string>());
                         cancel.cancel();
                     },
                     cancel),
                 Cancelled);
    set_isolation_program(std::nullopt);
    ASSERT_FALSE(folder.empty());
    EXPECT_FALSE(std::filesystem::exists(folder));
}

TEST(Isolated, DescribesExitCodesAsThePythonAppDoes) {
    EXPECT_EQ(describe_exit_code(0), "exit code 0");
    EXPECT_EQ(describe_exit_code(3), "exit code 3");
    EXPECT_EQ(describe_exit_code(65535), "exit code 65535");
    EXPECT_EQ(describe_exit_code(static_cast<int>(0xC0000005u)), "exit code 0xC0000005");
    EXPECT_EQ(describe_exit_code(-1), "exit code 0xFFFFFFFF");
}

}  // namespace
}  // namespace vml
