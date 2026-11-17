// Downloading FFmpeg for Windows when the app cannot find it, as FastFlix
// does: the FFmpeg-Builds project's newest stable Windows x64 GPL build
// (the builds BtbN started), checked against its SHA-256 and unpacked into a
// folder of the app's own. The window does the downloading (Qt's network);
// this is what it decides and unpacks, without Qt, to be tested.
#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "vml/cancel.h"

namespace vml {

// GitHub's answer naming the builds' latest release and its files.
inline constexpr std::string_view ffmpeg_builds_release_url =
    "https://api.github.com/repos/FFmpeg-Builds/FFmpeg-Builds/releases/latest";

// One build to download.
struct FfmpegBuild {
    std::string name;     // the file's name: "ffmpeg-n9.0-latest-win64-gpl-9.0.zip"
    std::string version;  // "9.0"
    std::string url;      // where it is downloaded from
    long long size = 0;   // bytes
    // Its SHA-256 in lowercase hexadecimal, as the release gives it; empty
    // when it does not, and then `checksums_url` (checksums_from_list).
    std::string sha256;
    std::string checksums_url;  // the release's checksums.sha256, if it has one
};

// The newest stable (not master) Windows x64 GPL build, not the shared or
// LGPL ones, of minimum_ffmpeg_version or newer, among a release's files
// (GitHub's JSON answer); nothing when there is none or the answer is not
// one.
std::optional<FfmpegBuild> choose_ffmpeg_build(const std::string& release_json);

// The SHA-256 a checksums file ("<hex>  <file name>" a line, as sha256sum
// writes it) gives the file `name`, in lowercase; nothing when it is not
// listed.
std::optional<std::string> checksum_from_list(const std::string& text, std::string_view name);

// The SHA-256 in GitHub's "digest" of a release file ("sha256:<hex>"), in
// lowercase; nothing for another kind or a malformed one.
std::optional<std::string> sha256_from_digest(std::string_view digest);

class FfmpegUnpackError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Takes ffmpeg.exe and ffprobe.exe out of a build's zip `zip` (its one top
// folder's bin folder) into `folder`, replacing the ones there; nothing else
// in the zip is unpacked. Both or neither: each is unpacked as
// "<name>.part" first. Throws FfmpegUnpackError when the zip cannot be read
// or lacks one, or a file cannot be written; Cancelled when `cancel` is.
void unpack_ffmpeg(const std::filesystem::path& zip, const std::filesystem::path& folder,
                   const CancelToken& cancel);

}  // namespace vml
