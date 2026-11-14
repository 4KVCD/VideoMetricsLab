// Durations as H:MM:SS (a port of vmaf_app/core/time_format.py), and
// moments in UTC.
#pragma once

#include <chrono>
#include <string>

namespace vml {

// A duration in seconds as H:MM:SS, or H:MM:SS.sss with `decimals`; hours
// are always shown. With decimals the whole value is rounded first, so
// 59.96 s is "0:01:00.0", not "0:00:60.0"; without, the seconds are cut.
std::string format_hms(double seconds, int decimals = 0);

// A moment in UTC, as ISO 8601 writes it to the second: "2026-11-14T20:28:02Z".
std::string utc_text(std::chrono::system_clock::time_point time);

}  // namespace vml
