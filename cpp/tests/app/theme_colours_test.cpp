// Every colour the window asks theme::color for by name is one the theme
// has, light and dark: a name it lacks throws (std::map::at) from inside an
// event handler, which closes the app. ("danger" and "text_muted", which only
// VML Modern's style sheet has, did that to the Videos footer's note.)
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>

#include "theme.h"

namespace vml::app {
namespace {

TEST(ThemeColours, EveryNameTheWindowAsksForIsOneTheThemeHas) {
    static const std::regex asked(R"(theme::color\(\"(\w+)\")");
    std::set<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(VML_SOURCE_DIR) / "app")) {
        if (!entry.is_regular_file() || entry.path().extension() != ".cpp") {
            continue;
        }
        std::ifstream file(entry.path(), std::ios::binary);
        std::stringstream text;
        text << file.rdbuf();
        const std::string code = text.str();
        for (auto match = std::sregex_iterator(code.begin(), code.end(), asked); match != std::sregex_iterator();
             ++match) {
            names.insert((*match)[1].str());
        }
    }
    EXPECT_GT(names.size(), 3u);  // the sources were found
    for (const std::string& name : names) {
        for (const bool dark : {false, true}) {
            // With `dark` given, only the theme's own table answers: no look's
            // colours stand in for a name it lacks.
            EXPECT_NO_THROW(theme::color(name.c_str(), dark)) << name << (dark ? " (dark)" : " (light)");
        }
    }
}

}  // namespace
}  // namespace vml::app
