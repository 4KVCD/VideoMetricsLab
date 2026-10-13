// What runs left in the temporary folder (temp_files.h).
#include "vml/temp_files.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
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

}  // namespace
}  // namespace vml
