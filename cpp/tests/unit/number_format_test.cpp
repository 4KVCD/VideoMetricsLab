#include "vml/number_format.h"

#include <gtest/gtest.h>

#include "reference.h"

namespace vml {
namespace {

// The text the CSV and JSON files hold must be Python's, character for
// character, on every system: the shortest digits of a number are the same
// everywhere, so this needs no tolerance.
TEST(NumberFormat, ReprWritesNumbersAsPythonDoes) {
    const nlohmann::json reference = test::load_reference("number_format.json");
    ASSERT_FALSE(reference["repr"].empty());
    for (const nlohmann::json& item : reference["repr"]) {
        const double value = *test::reference_number(item["value"]);
        EXPECT_EQ(python_repr(value), item["text"].get<std::string>()) << "for " << item["value"];
    }
}

TEST(NumberFormat, FixedRoundsAsPythonDoes) {
    const nlohmann::json reference = test::load_reference("number_format.json");
    ASSERT_FALSE(reference["fixed"].empty());
    for (const nlohmann::json& item : reference["fixed"]) {
        const double value = *test::reference_number(item["value"]);
        const int decimals = item["decimals"].get<int>();
        EXPECT_EQ(format_fixed(value, decimals), item["text"].get<std::string>())
            << "for " << item["value"] << " at " << decimals << " decimals";
    }
}

}  // namespace
}  // namespace vml
