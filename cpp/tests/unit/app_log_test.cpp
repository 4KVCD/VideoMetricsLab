// The app's log: rotated as the Python app's is, and Copy log's text held to
// the Python app's own (reference/app_log.json, made by its
// log_text_to_share on the same files).
#include "vml/app_log.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include "reference.h"
#include "test_videos.h"

namespace vml {
namespace {

// Sessions with and without a run, a run in an older session or an older
// file, a log cut to the limit (characters past ASCII counted as one each,
// a carriage return inside a line), no log at all.
TEST(AppLog, CopyLogSharesWhatThePythonAppShares) {
    const nlohmann::json reference = test::load_reference("app_log.json");
    ASSERT_FALSE(reference.at("cases").empty());
    for (const nlohmann::json& item : reference.at("cases")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const test::Folder folder;
        const std::filesystem::path logs = folder / "logs";
        std::filesystem::create_directories(logs);
        for (const auto& [name, text] : item.at("files").items()) {
            std::ofstream(logs / name, std::ios::binary) << text.get<std::string>();
        }
        const auto shared = app_log::log_text_to_share(logs, item.at("limit").get<std::size_t>());
        if (item.at("text").is_null()) {
            EXPECT_FALSE(shared.has_value());
            continue;
        }
        ASSERT_TRUE(shared.has_value());
        EXPECT_EQ(shared->first, item.at("text").get<std::string>());
        EXPECT_EQ(shared->second, item.at("shortened").get<bool>());
    }
}

// 30 MB of lines: the log starts again before it would reach 5 MB, the
// older files move up a number each time, and only four are kept, so the
// log takes 25 MB at most. Every line is in the newest files, in order.
TEST(AppLog, StartsAgainAtFiveMegabytesKeepingFourOlderFiles) {
    const test::Folder folder;
    const std::filesystem::path logs = folder / "logs";
    ASSERT_TRUE(app_log::start_logging(logs).has_value());
    const std::string text(10'000, 'x');
    constexpr int lines = 3'000;
    for (int i = 0; i < lines; ++i) {
        app_log::write(app_log::Level::info, "vml.test", std::to_string(i) + " " + text);
    }
    app_log::stop_logging();

    const std::filesystem::path log = logs / "VideoMetricsLab.log";
    EXPECT_FALSE(std::filesystem::exists(logs / "VideoMetricsLab.log.5"));
    // Oldest first, as the lines were written.
    int next = -1;
    for (const std::string suffix : {".4", ".3", ".2", ".1", ""}) {
        std::filesystem::path file = log;
        file += suffix;
        ASSERT_TRUE(std::filesystem::exists(file)) << suffix;
        EXPECT_LT(std::filesystem::file_size(file), 5u * 1024 * 1024) << suffix;
        std::ifstream in(file, std::ios::binary);
        for (std::string line; std::getline(in, line);) {
            // "<date> <time> INFO    [MainThread] vml.test: <number> xxx..."
            const std::size_t at = line.find("vml.test: ");
            ASSERT_NE(at, std::string::npos) << line.substr(0, 80);
            const int number = std::stoi(line.substr(at + 10));
            if (next >= 0) {
                EXPECT_EQ(number, next) << suffix;
            }
            next = number + 1;
        }
    }
    EXPECT_EQ(next, lines);
}

}  // namespace
}  // namespace vml
