// The app's log: rotated as the Python app's is, and Copy log's text held to
// the Python app's own (reference/app_log.json, made by its
// log_text_to_share on the same files).
#include "vml/app_log.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <miniz.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/platform/system.h"

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

// Export log: the log files with anything in them, in one zip to attach to
// a report, as the Python app's export_logs writes it; with no log yet,
// nothing.
TEST(AppLog, TheLogsAreExportedAsOneZip) {
    const test::Folder folder;
    const std::filesystem::path logs = folder / "logs";
    std::filesystem::create_directories(logs);
    std::ofstream(logs / "VideoMetricsLab.log", std::ios::binary) << "today's session\n";
    std::ofstream(logs / "VideoMetricsLab.log.1", std::ios::binary) << "an older session\n";
    std::ofstream(logs / "native-crashes.log", std::ios::binary);  // nothing in it: left out
    std::ofstream(logs / "notes.txt", std::ios::binary) << "not a log\n";
    const std::filesystem::path target = folder / "export.zip";
    EXPECT_EQ(app_log::export_logs(target, logs),
              (std::vector<std::string>{"VideoMetricsLab.log", "VideoMetricsLab.log.1"}));

    const std::string zip = platform::read_file(target).value_or("");
    mz_zip_archive archive{};
    ASSERT_TRUE(mz_zip_reader_init_mem(&archive, zip.data(), zip.size(), 0));
    std::map<std::string, std::string> entries;
    for (mz_uint index = 0; index < mz_zip_reader_get_num_files(&archive); ++index) {
        mz_zip_archive_file_stat stat{};
        ASSERT_TRUE(mz_zip_reader_file_stat(&archive, index, &stat));
        std::size_t size = 0;
        void* bytes = mz_zip_reader_extract_to_heap(&archive, index, &size, 0);
        ASSERT_NE(bytes, nullptr);
        entries[stat.m_filename] = std::string(static_cast<const char*>(bytes), size);
        EXPECT_EQ(stat.m_method, MZ_DEFLATED) << stat.m_filename;
        mz_free(bytes);
    }
    mz_zip_reader_end(&archive);
    EXPECT_EQ(entries, (std::map<std::string, std::string>{{"VideoMetricsLab.log", "today's session\n"},
                                                           {"VideoMetricsLab.log.1", "an older session\n"}}));

    const test::Folder empty;
    EXPECT_TRUE(app_log::export_logs(empty / "none.zip", empty / "logs").empty());
    EXPECT_FALSE(std::filesystem::exists(empty / "none.zip"));
}

// The run's lanes name their threads for the log, and end as soon as the
// run does: threads named, writing a line and ending, many at once, leave
// the heap whole (MinGW's thread-local storage corrupted it when a name was
// a std::string), and each line carries its own thread's name.
TEST(AppLog, ThreadsNamedForTheLogEndCleanly) {
    const test::Folder folder;
    const std::filesystem::path logs = folder / "logs";
    ASSERT_TRUE(app_log::start_logging(logs).has_value());
    for (int round = 0; round < 50; ++round) {
        std::vector<std::thread> threads;
        for (int lane = 0; lane < 8; ++lane) {
            threads.emplace_back([lane] {
                app_log::set_thread_name("vmaf-cpu-" + std::to_string(lane));
                app_log::write(app_log::Level::info, "vml.test", "lane " + std::to_string(lane));
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
    }
    app_log::stop_logging();
    const std::string log = platform::read_file(logs / "VideoMetricsLab.log").value_or("");
    for (int lane = 0; lane < 8; ++lane) {
        EXPECT_NE(log.find("[vmaf-cpu-" + std::to_string(lane) + "] vml.test: lane " + std::to_string(lane)),
                  std::string::npos);
    }
}

}  // namespace
}  // namespace vml
