// Records each test video's stream fingerprint after its scores are saved
// (metric_cache.h's record_test_stream), for a remux of it to keep them. One
// at a time, on a thread of its own: reading a stream takes as long as
// reading its file (minutes for a long one over a network share), which the
// file writes (FileWrites: other saves, exports) and the window's closing
// must not wait for, nor a run share the disk with. A run pauses it (the
// video being read is read again once it resumes); stopping cancels the
// read under way and drops the rest: those videos are recorded the next time
// their scores are saved. Only with Settings' "Recognize test videos
// remuxed" on.
#pragma once

#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#include "vml/cancel.h"

namespace vml::app {

class StreamRecords {
public:
    StreamRecords() = default;
    // Stops, and waits for the thread.
    ~StreamRecords();
    StreamRecords(const StreamRecords&) = delete;
    StreamRecords& operator=(const StreamRecords&) = delete;

    // Told, on the recording thread, of each video as its stream is read,
    // and once none is left to read (or it is paused or stopped).
    void set_reporting(std::function<void(const std::filesystem::path&)> started, std::function<void()> idle);

    // Queues the test video `test`, whose scores were saved under `base`.
    // From any thread; ignored once stopped.
    void add(std::filesystem::path base, std::filesystem::path test);

    // While a run uses the disks: the read under way is cancelled (and made
    // again on resume), and none starts.
    void pause();
    void resume();

    // For the window closing: cancels the read under way, drops the rest,
    // and takes no more.
    void stop();

private:
    void start_locked();
    void record_all();

    std::mutex mutex_;
    std::deque<std::pair<std::filesystem::path, std::filesystem::path>> queue_;
    bool recording_ = false;  // a thread is taking videos off the queue
    bool paused_ = false;
    bool stopped_ = false;
    CancelToken current_;  // the read under way's
    std::function<void(const std::filesystem::path&)> started_;
    std::function<void()> idle_;
    std::thread thread_;
};

}  // namespace vml::app
