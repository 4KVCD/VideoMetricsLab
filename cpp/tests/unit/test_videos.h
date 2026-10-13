// For the tests that run FFmpeg on real clips: FFmpeg and ffprobe where they
// run, a folder of the test's own, clips of FFmpeg's test pattern, and an
// environment variable set for one test.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vml::test {

// FFmpeg and ffprobe as the app finds them, and FFmpeg's version; nothing
// where they do not run or are too old (GitHub's runners).
struct Tools {
    std::string ffmpeg;
    std::string ffprobe;
    std::string version;
};
std::optional<Tools> tools();

// A temporary folder, deleted with everything in it when it goes.
class Folder {
public:
    Folder();
    ~Folder();
    Folder(const Folder&) = delete;
    Folder& operator=(const Folder&) = delete;
    std::filesystem::path operator/(const std::string& name) const { return path_ / name; }

private:
    std::filesystem::path path_;
};

// Runs FFmpeg with `arguments`; the test fails if FFmpeg does.
void run_ffmpeg(const Tools& found, std::vector<std::string> arguments);

// A clip of FFmpeg's test pattern at 24000/1001 frames a second, with
// B-frames, `size` and `seconds` long, encoded by `encoder` ("libx264",
// "libx265") in `pix_fmt`, with `extra` options after the encoder's.
std::filesystem::path make_clip(const Tools& found, const std::filesystem::path& path, const std::string& encoder,
                                const std::string& pix_fmt, double seconds, std::vector<std::string> extra = {},
                                const std::string& size = "640x360");

// An environment variable set for as long as this lives, then put back.
class ScopedVariable {
public:
    ScopedVariable(std::string name, const std::string& value);
    ~ScopedVariable();
    ScopedVariable(const ScopedVariable&) = delete;
    ScopedVariable& operator=(const ScopedVariable&) = delete;

private:
    std::string name_;
    std::string previous_;
};

}  // namespace vml::test
