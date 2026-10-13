// JSON written as the Python app's json.dumps(value, indent=2) writes it
// (reference/python_json.json).
#include "vml/python_json.h"

#include <gtest/gtest.h>

#include <string>

#include "reference.h"

namespace vml {
namespace {

using Ordered = nlohmann::ordered_json;

// A value from the reference: {"$float": hex} is a number and {"$bytes":
// hex} a string of those bytes; the rest is as it is.
Ordered decoded(const Ordered& value) {
    if (value.is_object() && value.size() == 1 && value.contains("$float")) {
        return *test::reference_number(value["$float"]);
    }
    if (value.is_object() && value.size() == 1 && value.contains("$bytes")) {
        const std::string hex = value["$bytes"];
        std::string bytes;
        for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
            bytes += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
        }
        return bytes;
    }
    if (value.is_object()) {
        Ordered object = Ordered::object();
        for (auto item = value.begin(); item != value.end(); ++item) {
            object[item.key()] = decoded(item.value());
        }
        return object;
    }
    if (value.is_array()) {
        Ordered array = Ordered::array();
        for (const Ordered& item : value) {
            array.push_back(decoded(item));
        }
        return array;
    }
    return value;
}

TEST(PythonJson, WritesAsPythonsJsonDumps) {
    const nlohmann::json reference = test::load_reference("python_json.json");
    ASSERT_FALSE(reference.empty());
    for (const nlohmann::json& expected : reference) {
        const std::string value = expected["value"];
        SCOPED_TRACE(value);
        EXPECT_EQ(python_json_dumps(decoded(Ordered::parse(value))), expected["text"].get<std::string>());
    }
}

}  // namespace
}  // namespace vml
