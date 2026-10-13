#include "vml/time_format.h"

#include <algorithm>
#include <cmath>

namespace vml {

namespace {

// `number` with zeros in front up to `width` digits.
std::string padded(long long number, int width) {
    std::string digits = std::to_string(number);
    if (static_cast<int>(digits.size()) < width) {
        digits.insert(0, static_cast<std::size_t>(width) - digits.size(), '0');
    }
    return digits;
}

}  // namespace

std::string format_hms(double seconds, int decimals) {
    seconds = std::max(0.0, seconds);
    long long whole = 0;
    long long fraction = 0;
    if (decimals > 0) {
        const auto scale = static_cast<long long>(std::pow(10, decimals));
        // Python's round(): halves to the even neighbour.
        const auto units = static_cast<long long>(std::nearbyint(seconds * static_cast<double>(scale)));
        whole = units / scale;
        fraction = units % scale;
    } else {
        whole = static_cast<long long>(seconds);
    }
    std::string text = std::to_string(whole / 3600) + ":" + padded(whole % 3600 / 60, 2) + ":" + padded(whole % 60, 2);
    if (decimals > 0) {
        text += "." + padded(fraction, decimals);
    }
    return text;
}

}  // namespace vml
