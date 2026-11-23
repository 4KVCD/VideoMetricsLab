// Limits of the compilers the app is built with that a build here may not
// reach: the maintainer's MSVC refuses a string literal over 16,380 bytes
// (error C2026), which compare's help passed in PR #185 while the build tools
// here took it. Every raw string literal of the sources is held under it, so
// a help text that grows fails here first.
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// MSVC's most bytes in one string literal (C2026); adjacent literals are
// joined after, so a long text is split into several.
constexpr std::size_t msvc_literal_limit = 16380;

// Each raw string literal of `text` longer than `limit` bytes, as
// "<line>: <bytes>".
std::vector<std::string> long_raw_literals(const std::string& text, std::size_t limit) {
    std::vector<std::string> found;
    for (std::size_t at = text.find("R\""); at != std::string::npos; at = text.find("R\"", at + 2)) {
        // R"delimiter( ... )delimiter", the delimiter up to 16 characters
        const std::size_t open = text.find('(', at + 2);
        if (open == std::string::npos || open - (at + 2) > 16 ||
            (at > 0 && (std::isalnum(static_cast<unsigned char>(text[at - 1])) || text[at - 1] == '_'))) {
            continue;
        }
        const std::string delimiter = text.substr(at + 2, open - (at + 2));
        if (delimiter.find_first_of(" \\\")\n") != std::string::npos) {
            continue;
        }
        const std::size_t close = text.find(")" + delimiter + "\"", open + 1);
        if (close == std::string::npos) {
            continue;
        }
        if (close - (open + 1) > limit) {
            const auto line = 1 + std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(at), '\n');
            found.push_back(std::to_string(line) + ": " + std::to_string(close - (open + 1)) + " bytes");
        }
        at = close;
    }
    return found;
}

TEST(SourceLimits, StringLiteralsFitMsvc) {
    // The finder finds: one literal over the limit, one under it.
    const std::string sample = "a = R\"(" + std::string(20, 'x') + ")\";\nb = R\"x(" + std::string(5, 'y') + ")x\";\n";
    EXPECT_EQ(long_raw_literals(sample, 10), std::vector<std::string>{"1: 20 bytes"});

    std::size_t files = 0;
    for (const char* part : {"app", "cli", "core", "platform", "tests"}) {
        const std::filesystem::path folder = std::filesystem::path(VML_SOURCE_DIR) / part;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(folder)) {
            const std::filesystem::path& path = entry.path();
            if (!entry.is_regular_file() || (path.extension() != ".cpp" && path.extension() != ".h" &&
                                              path.extension() != ".mm")) {
                continue;
            }
            std::ifstream file(path, std::ios::binary);
            std::stringstream text;
            text << file.rdbuf();
            ++files;
            for (const std::string& literal : long_raw_literals(text.str(), msvc_literal_limit)) {
                ADD_FAILURE() << path.string() << ":" << literal << ": over MSVC's " << msvc_literal_limit
                              << " (error C2026): end it with )\" and go on in a literal of its own, R\"(";
            }
        }
    }
    EXPECT_GT(files, 100u);  // the sources were found
}

}  // namespace
