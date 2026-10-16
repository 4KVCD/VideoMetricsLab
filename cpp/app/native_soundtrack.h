// The soundtrack Video Compare's native playback is timed by (the Python
// app's SingleSoundtrack): the source video's, across switches between test
// videos, played by the app itself (FfmpegSoundtrack: FFmpeg's libraries
// decode it and WASAPI plays it, where the Python app had GStreamer's
// playbin3). Its clock, the samples the sound device has played, times the
// frames.
#pragma once

#include <QString>

#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "vml/soundtrack_clock.h"

namespace vml::app {

// The soundtrack as the playback pool uses it.
class Soundtrack {
public:
    virtual ~Soundtrack() = default;
    // Its clock (ms from the video's first frame), carried on between the
    // clock's steps; nothing while it has none.
    virtual std::optional<double> poll() = 0;
    // The clock's own last reading, to judge lag by.
    virtual std::optional<double> reading_ms() const = 0;
    // Whether it is ready to play from where it was last sought.
    virtual bool ready() const = 0;
    // Why it cannot play, if it cannot.
    virtual std::optional<QString> failed() const = 0;
    // It cannot play: it stops.
    virtual void set_failed(const QString& why) = 0;
    virtual int offset_ms() const = 0;
    // Where the source's first frame is on the file's timeline, so that its
    // clock counts from it; it seeks to `position_ms` again with it.
    virtual void set_offset(int first_ms, int position_ms) = 0;
    virtual void set_playing(bool playing) = 0;
    virtual void seek(int position_ms) = 0;
    // Muted (false), its clock still runs.
    virtual void set_enabled(bool enabled) = 0;
};

class FfmpegSoundtrack : public Soundtrack {
public:
    // The soundtrack of the video at `path`, from `start_ms`; muted unless
    // `enabled`.
    FfmpegSoundtrack(std::filesystem::path path, int start_ms, bool enabled);
    ~FfmpegSoundtrack() override;
    FfmpegSoundtrack(const FfmpegSoundtrack&) = delete;
    FfmpegSoundtrack& operator=(const FfmpegSoundtrack&) = delete;

    std::optional<double> poll() override;
    std::optional<double> reading_ms() const override { return clock_.reading_ms(); }
    bool ready() const override;
    std::optional<QString> failed() const override;
    void set_failed(const QString& why) override;
    int offset_ms() const override { return clock_.offset_ms(); }
    void set_offset(int first_ms, int position_ms) override;
    void set_playing(bool playing) override;
    void seek(int position_ms) override;
    void set_enabled(bool enabled) override;

private:
    // The soundtrack played on the thread (native_soundtrack.cpp).
    class Player;

    void run();
    // Whether the thread is to stop (FFmpeg's reading asks it too).
    bool stopping() const;
    static int interrupted(void* self);

    std::filesystem::path path_;

    // Shared with the thread.
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    bool stop_ = false;
    std::optional<double> wanted_ms_;  // a seek (the file's timeline) to make
    bool want_playing_ = false;
    bool enabled_ = true;
    bool ready_ = false;
    std::optional<QString> failed_;
    // The clock's last reading on the file's timeline (ms), once it has one.
    std::optional<double> clock_ms_;

    // The window's side: the clock's positions, counted from the source's
    // first frame.
    SoundtrackClock clock_;

    std::thread thread_;
};

}  // namespace vml::app
