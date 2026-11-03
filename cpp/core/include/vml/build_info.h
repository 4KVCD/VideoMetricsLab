// Which build of the app this is: its version and the commit it was built
// from. Written into what the app saves (a score's provenance, a saved
// comparison's context.json, run files, the JSON exports) as history:
// nothing decides by them whether a saved score is used again.
#pragma once

#include <string>

namespace vml {

// The commit the app was built from, as 12 hex digits, with "-dirty" after
// them when built with uncommitted changes, or "unknown" (built where git
// cannot say). Written at every build by core/build_commit.cmake.
const char* build_commit();

// The version and the commit: "2.4 (0123456789ab)".
std::string build_stamp();

}  // namespace vml
