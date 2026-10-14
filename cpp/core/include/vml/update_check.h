// Is there a newer release? Asked once, when the app starts (the Python app's
// core/update_check.py).
//
// The releases are published on GitHub, and GitHub's public API names the
// latest one (drafts and pre-releases left out), so no server of our own is
// needed. No key either: GitHub allows 60 such requests an hour from one
// address. Nothing is sent but the request itself; GitHub sees the address
// it comes from, as it would for any page. The window asks (it has the
// network); this reads the answer.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vml {

inline constexpr std::string_view releases_url = "https://api.github.com/repos/4KVCD/VideoMetricsLab/releases/latest";
// How long the question may take, all of it.
inline constexpr int update_check_timeout_seconds = 8;

// GitHub could not be asked, or its answer could not be read.
class UpdateCheckError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A release on GitHub: its version ("1.3", from the tag "v1.3"), its page
// and its notes.
struct Release {
    std::string version;
    std::string page_url;
    std::string notes;
};

// "v1.2.1" -> {1, 2, 1}; trailing zeros dropped, so 1.3 and 1.3.0 are the
// same version. Nothing for anything that is not a plain version.
std::optional<std::vector<long long>> version_tuple(std::string_view text);

// Whether `candidate` is a newer version than `current`, compared as
// numbers: 1.10 is newer than 1.9.
bool is_newer(std::string_view candidate, std::string_view current);

// The latest release, from the body of GitHub's answer. Throws
// UpdateCheckError when it is not JSON or names no release version.
Release release_from_answer(std::string_view body);

}  // namespace vml
