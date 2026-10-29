// Work run in a process of its own (isolated.h), through vml-cli, the
// program the children run, with its tasks for the tests.
#include "vml/isolated.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_videos.h"
#include "vml/app_log.h"
#include "vml/gpu_vmaf.h"
#include "vml/platform/system.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vship.h"

namespace vml {
namespace {

// The child's messages come back in order, then its result and the file it
// left; an error the child reports is thrown here as that error, not as a
// crash.
TEST(Isolated, RunsWorkInAProcessOfItsOwn) {
    set_isolation_program(std::filesystem::path(VML_CLI));
    std::vector<nlohmann::json> messages;
    std::string echoed;
    std::string working_folder;
    const nlohmann::json request{{"file", "written"}, {"n", 1.25}};
    const nlohmann::json result = run_isolated(
        "test-echo", "the test", request, [&](const nlohmann::json& message) { messages.push_back(message); },
        CancelToken(), [&](const nlohmann::json&, const std::filesystem::path& folder) {
            std::ifstream file(folder / "echo.txt");
            std::getline(file, echoed);
            std::getline(file, working_folder);
        });
    EXPECT_EQ(result, request);
    ASSERT_EQ(messages.size(), 2u);
    EXPECT_EQ(messages[0], (nlohmann::json{{"progress", {1, 2, 3.5}}}));
    EXPECT_EQ(messages[1], (nlohmann::json{{"status", "on its way"}}));
    EXPECT_EQ(echoed, "written");
    // This process's working folder, where a relative path names a video.
    EXPECT_EQ(working_folder, platform::path_text(std::filesystem::current_path()));

    try {
        run_isolated("no-such-task", "the test", nlohmann::json::object(), {}, CancelToken());
        ADD_FAILURE() << "no error";
    } catch (const IsolatedCrashError&) {
        ADD_FAILURE() << "an error the child reports is not a crash";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "there is no task no-such-task");
    }
    set_isolation_program(std::nullopt);
}

// Cancel ends the child and what it started. A program the child started
// can still have a file in the child's folder open for a moment after that
// (FFmpeg, its timestamps): the folder goes all the same.
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

// Each GPU check runs under the token it is given (inside a run, the run's):
// cancelled, it throws Cancelled and keeps no answer, so a later caller asks
// again.
TEST(Isolated, AGpuCheckUnderACancelledTokenThrowsAndKeepsNoAnswer) {
    set_isolation_program(std::filesystem::path(VML_CLI));
    CancelToken run;
    run.cancel();
    EXPECT_THROW(gpu_vmaf_backend(run), Cancelled);
    EXPECT_FALSE(gpu_vmaf_probe_ready());
    EXPECT_THROW(detect_vship_device(VshipBuild::fast, run), Cancelled);
    EXPECT_FALSE(vship_probe_ready(VshipBuild::fast));
    EXPECT_FALSE(known_vship_device(VshipBuild::fast).has_value());
    EXPECT_THROW(v1_available(run), Cancelled);
    EXPECT_FALSE(v1_probe_ready());
    set_isolation_program(std::nullopt);
}

// A real crash of the child: the app's log says where it happened, from the
// child's own report, as the Python app logs its threads at the crash.
TEST(Isolated, ACrashIsReportedInTheLogWithWhereItHappened) {
    const test::Folder folder;
    const std::filesystem::path logs = folder / "logs";
    ASSERT_TRUE(app_log::start_logging(logs).has_value());
    set_isolation_program(std::filesystem::path(VML_CLI));
    EXPECT_THROW(run_isolated("test-fault", "the test", nlohmann::json::object(), {}, CancelToken()),
                 IsolatedCrashError);
    set_isolation_program(std::nullopt);
    app_log::stop_logging();
    const std::string log = platform::read_file(logs / "VideoMetricsLab.log").value_or("");
    EXPECT_NE(log.find("the test crashed"), std::string::npos) << log;
    EXPECT_NE(log.find("; its stack at the crash:"), std::string::npos) << log;
#ifdef _WIN32
    EXPECT_NE(log.find("the test crashed (exit code 0xC0000005); its stack at the crash:"), std::string::npos) << log;
    EXPECT_NE(log.find("Fatal exception 0xC0000005 at vml-cli.exe+0x"), std::string::npos) << log;
#else
    EXPECT_NE(log.find("Aborted\nStack:\n"), std::string::npos) << log;
#endif
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
