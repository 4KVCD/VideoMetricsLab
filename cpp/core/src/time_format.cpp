#include "vml/time_format.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace vml {

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
    const long long hours = whole / 3600;
    const long long minutes = whole % 3600 / 60;
    const long long secs = whole % 60;
    char text[64];
    if (decimals > 0) {
        std::snprintf(text, sizeof text, "%lld:%02lld:%02lld.%0*lld", hours, minutes, secs, decimals, fraction);
    } else {
        std::snprintf(text, sizeof text, "%lld:%02lld:%02lld", hours, minutes, secs);
    }
    return text;
}

}  // namespace vml
