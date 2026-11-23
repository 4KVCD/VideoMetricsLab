// What runs left in the temporary folder (temp_files.h).
#include "vml/temp_files.h"
#include "temp_files_parts.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "test_videos.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

int a_process_id_not_running() {
    int pid = 4'000'000;
    while (platform::process_running(pid)) {
        pid += 4;
    }
    return pid;
}

// A program ended outright (End task, a crash) cannot remove its temporary
// files, and one CPU SSIMULACRA2 run's pictures alone can take more than a
// gigabyte. The next run removes what processes that no longer run left,
// and what earlier versions left without a process id once nothing has
// changed it for a week: nothing of a running process's, and nothing that is
// not the app's.
TEST(TempFiles, WhatRunsEndedOutrightLeftIsRemovedAndNothingElse) {
    const test::Folder folder;
    const std::filesystem::path temp = folder / "temp";
    std::filesystem::create_directory(temp);
    const std::string gone = std::to_string(a_process_id_not_running());
    const std::string running = std::to_string(platform::process_id());
    const std::filesystem::path pictures = temp / ("videometricslab-perceptual-" + gone + "-k2x9");
    std::filesystem::create_directory(pictures);
    std::ofstream(pictures / "reference-00000001.png") << "png";
    std::filesystem::create_directory(temp / ("vml-isolated-" + gone + "-a1b2"));
    std::filesystem::create_directory(temp / ("vmaf_gpu_run_" + gone + "-c3d4"));
    std::ofstream(temp / ("vml-vship-pts-" + gone + "-e5f6.txt")) << "1 1/1000\n";
    std::ofstream(temp / "vml-vmaf-v1-99999999999-i9j0.json") << "{}";  // an id too large for any process
    std::filesystem::create_directory(temp / ("vml-isolated-" + running + "-g7h8"));  // this process's, running
    const std::filesystem::path old = temp / "vml-vship-pts-oldname1.txt";  // an earlier version's
    std::ofstream(old) << "1 1/1000\n";
    std::filesystem::last_write_time(old, std::filesystem::file_time_type::clock::now() - std::chrono::hours(8 * 24));
    std::filesystem::create_directory(temp / "vmaf_run_recent99");  // an earlier version's, changed today
    std::filesystem::create_directory(temp / ("other-program-" + gone + "-x"));  // not the app's

    EXPECT_EQ(remove_leftovers(temp), 6);
    std::vector<std::string> left;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(temp)) {
        left.push_back(entry.path().filename().string());
    }
    std::sort(left.begin(), left.end());
    EXPECT_EQ(left, (std::vector<std::string>{"other-program-" + gone + "-x", "vmaf_run_recent99",
                                              "vml-isolated-" + running + "-g7h8"}));
}

// #205: a save replaced its files by renaming a whole copy over them, which
// Windows refuses while anything has the file open for reading (std::ifstream
// does not share deletion): a lookup or the Saved Results tab reading
// context.json while a run saved into its folder lost the whole save
// ("[WinError 5] Access is denied"). The rename is tried again once the
// reader is done.
TEST(TempFiles, AFileBeingReadIsReplacedOnceItIsClosed) {
    const test::Folder folder;
    const std::filesystem::path path = folder / "context.json";
    write_file_whole(path, [](std::ostream& out) { out << "old"; });
    const auto text = [&path] {
        std::ifstream file(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(file), {});
    };
    const auto leftovers = [&path] {
        int count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
            count += entry.path().extension() == ".tmp" ? 1 : 0;
        }
        return count;
    };
    std::vector<int> waits;
    auto reading = std::make_unique<std::ifstream>(path, std::ios::binary);
    const auto reader_done = [&](int attempt) {
        waits.push_back(attempt);
        reading.reset();
    };
    EXPECT_NO_THROW(write_file_whole(path, [](std::ostream& out) { out << "new"; }, reader_done));
    EXPECT_EQ(text(), "new");
#ifdef _WIN32
    EXPECT_EQ(waits, std::vector<int>{0});
#endif

    // Still open after every try: the save fails, saying why, and leaves the
    // file as it was and no temporary file.
    waits.clear();
    std::ifstream held(path, std::ios::binary);
    const auto count_waits = [&waits](int attempt) { waits.push_back(attempt); };
#ifdef _WIN32
    EXPECT_THROW(write_file_whole(path, [](std::ostream& out) { out << "newer"; }, count_waits), std::runtime_error);
    EXPECT_EQ(waits.size(), 11U);
    held.close();
    EXPECT_EQ(text(), "new");
#else
    EXPECT_NO_THROW(write_file_whole(path, [](std::ostream& out) { out << "newer"; }, count_waits));
#endif
    EXPECT_EQ(leftovers(), 0);

    // Read-only, it is refused for good: no waiting (#221's review: about 3
    // seconds were waited for nothing). Its file stays as it was.
    held.close();
    waits.clear();
    const std::filesystem::path locked = folder / "read-only.json";
    write_file_whole(locked, [](std::ostream& out) { out << "kept"; });
    std::filesystem::permissions(locked, std::filesystem::perms::owner_write | std::filesystem::perms::group_write |
                                             std::filesystem::perms::others_write,
                                 std::filesystem::perm_options::remove);
#ifdef _WIN32
    EXPECT_THROW(write_file_whole(locked, [](std::ostream& out) { out << "lost"; }, count_waits), std::runtime_error);
    EXPECT_TRUE(waits.empty());
#endif
    std::filesystem::permissions(locked, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
    EXPECT_EQ(leftovers(), 0);
}

}  // namespace
}  // namespace vml
