#include "vml/difference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vml {

namespace {

constexpr double none = std::numeric_limits<double>::quiet_NaN();

// The merge both pairings make: at each of the first's points (`count`), the
// second's point with the same key, if any. Both are in order of their keys.
template <typename FirstKey, typename SecondKey>
std::vector<double> paired_difference(std::size_t count, std::size_t second_count, FirstKey first_key,
                                      SecondKey second_key, std::span<const float> first_values,
                                      std::span<const float> second_values) {
    std::vector<double> differences(count, none);
    std::size_t other = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const long long key = first_key(index);
        while (other < second_count && second_key(other) < key) {
            ++other;
        }
        if (other < second_count && second_key(other) == key) {
            differences[index] = point_difference(first_values[index], second_values[other]);
        }
    }
    return differences;
}

}  // namespace

double point_difference(double a, double b) {
    if (std::isinf(a) && a == b) {
        return 0.0;
    }
    const double difference = a - b;
    return std::isfinite(difference) ? difference : none;
}

long long second_number(double start) { return static_cast<long long>(start); }

std::vector<double> frame_difference(std::span<const int> first_frames, std::span<const float> first_values,
                                     std::span<const int> second_frames, std::span<const float> second_values) {
    const auto first_key = [&](std::size_t i) { return static_cast<long long>(first_frames[i]); };
    const auto second_key = [&](std::size_t i) { return static_cast<long long>(second_frames[i]); };
    return paired_difference(first_values.size(), second_values.size(), first_key, second_key, first_values,
                             second_values);
}

std::vector<double> second_difference(std::span<const double> first_starts, std::span<const float> first_values,
                                      std::span<const double> second_starts, std::span<const float> second_values) {
    const auto first_key = [&](std::size_t i) { return second_number(first_starts[i]); };
    const auto second_key = [&](std::size_t i) { return second_number(second_starts[i]); };
    return paired_difference(first_values.size(), second_values.size(), first_key, second_key, first_values,
                             second_values);
}

std::optional<DifferenceSummary> summarize_difference(std::span<const double> differences) {
    DifferenceSummary summary{0.0, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                              0};
    double sum = 0.0;
    for (const double difference : differences) {
        if (std::isnan(difference)) {
            continue;
        }
        sum += difference;
        summary.minimum = std::min(summary.minimum, difference);
        summary.maximum = std::max(summary.maximum, difference);
        ++summary.paired;
    }
    if (summary.paired == 0) {
        return std::nullopt;
    }
    summary.mean = sum / static_cast<double>(summary.paired);
    return summary;
}

}  // namespace vml
