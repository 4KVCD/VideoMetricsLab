// Durations as H:MM:SS. A port of vmaf_app/core/time_format.py.
#pragma once

#include <string>

namespace vml {

// A duration in seconds as H:MM:SS, or H:MM:SS.sss with `decimals`; hours
// are always shown. With decimals the whole value is rounded first, so
// 59.96 s is "0:01:00.0", not "0:00:60.0"; without, the seconds are cut.
std::string format_hms(double seconds, int decimals = 0);

}  // namespace vml
