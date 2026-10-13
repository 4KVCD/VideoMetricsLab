#include "reference.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace vml::test {

nlohmann::json load_reference(std::string_view file_name) {
    const std::string path = std::string(VML_REFERENCE_DIR) + "/" + std::string(file_name);
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    return nlohmann::json::parse(file);
}

std::optional<double> reference_number(const nlohmann::json& text) {
    if (text.is_null()) {
        return std::nullopt;
    }
    // strtod reads float.hex()'s "0x1.7ap+6" as well as "inf" and "nan".
    return std::strtod(text.get<std::string>().c_str(), nullptr);
}

std::vector<double> reference_numbers(const nlohmann::json& list) {
    std::vector<double> numbers;
    for (const nlohmann::json& text : list) {
        numbers.push_back(*reference_number(text));
    }
    return numbers;
}

bool same_number(double expected, double actual) {
    if (std::isnan(expected) || std::isnan(actual)) {
        return std::isnan(expected) && std::isnan(actual);
    }
    if (expected == actual) {
        return true;
    }
#ifdef _WIN32
    return false;
#else
    if (std::isinf(expected) || std::isinf(actual)) {
        return false;
    }
    return std::abs(expected - actual) <= 1e-12 * std::max(std::abs(expected), std::abs(actual));
#endif
}

}  // namespace vml::test
