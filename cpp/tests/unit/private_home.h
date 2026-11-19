// The test programs run in a home folder of their own, so what their tests
// save or remove in the profile (settings.json, gpu_checks.json, the log,
// saved results) never touches the user's real one: a window test once
// overwrote the user's settings.json, and the perceptual routing tests'
// stand-in GPU failures deleted their gpu_checks.json.
#pragma once

#include <filesystem>
#include <random>
#include <string>
#include <system_error>

#include "vml/platform/system.h"

namespace vml::testing {

// A new folder in TEMP, made the home (USERPROFILE, HOME) from its making to
// its end, for this program and the programs it starts; removed at its end.
// Made at the start of main, before anything reads the home.
class PrivateHome {
public:
    PrivateHome()
        : folder_(std::filesystem::temp_directory_path() / ("vml_tests_home_" + std::to_string(std::random_device()()))) {
        std::filesystem::create_directories(folder_);
        platform::set_environment_variable("USERPROFILE", platform::path_text(folder_));
        platform::set_environment_variable("HOME", platform::path_text(folder_));
    }
    ~PrivateHome() {
        std::error_code ignored;  // a file a child still holds stays behind in TEMP
        std::filesystem::remove_all(folder_, ignored);
    }
    PrivateHome(const PrivateHome&) = delete;
    PrivateHome& operator=(const PrivateHome&) = delete;

private:
    std::filesystem::path folder_;
};

}  // namespace vml::testing
