// The update check held to the Python app's (reference/update_check.json,
// from its update_check.py): versions compared as numbers, and GitHub's
// answers read into the same release, or refused where it refuses them.
#include "vml/update_check.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string>

#include "reference.h"

namespace vml {
namespace {

TEST(UpdateCheck, ReadsGitHubsAnswerAsThePythonAppDoes) {
    const nlohmann::json reference = test::load_reference("update_check.json");
    EXPECT_EQ(releases_url, reference.at("url").get<std::string>());
    EXPECT_EQ(update_check_timeout_seconds, reference.at("timeout").get<double>());

    const nlohmann::json& versions = reference.at("versions");
    ASSERT_FALSE(versions.empty());
    for (const nlohmann::json& item : versions) {
        const std::string candidate = item.at("candidate");
        const std::string current = item.at("current");
        SCOPED_TRACE("'" + candidate + "' against '" + current + "'");
        EXPECT_EQ(is_newer(candidate, current), item.at("newer").get<bool>());
        const auto tuple = version_tuple(candidate);
        EXPECT_EQ(tuple.has_value(), !item.at("tuple").is_null());
        if (tuple && !item.at("tuple").is_null()) {
            EXPECT_EQ(*tuple, item.at("tuple").get<std::vector<long long>>());
        }
    }

    const nlohmann::json& answers = reference.at("answers");
    ASSERT_FALSE(answers.empty());
    for (const nlohmann::json& item : answers) {
        const std::string body = item.at("body");
        SCOPED_TRACE(body);
        if (item.at("release").is_null()) {
            EXPECT_THROW(release_from_answer(body), UpdateCheckError);
            continue;
        }
        const Release release = release_from_answer(body);
        EXPECT_EQ(release.version, item.at("release").at("version").get<std::string>());
        EXPECT_EQ(release.page_url, item.at("release").at("page_url").get<std::string>());
        EXPECT_EQ(release.notes, item.at("release").at("notes").get<std::string>());
    }
}

}  // namespace
}  // namespace vml
