// Running a list of videos' metrics (job_runner.h), with a stand-in for
// run_vmaf: a video's tasks run at once and make one result, a task that
// fails leaves the video's other metrics, a cancelled run keeps what had
// finished, and videos share the CPU's two lanes and the GPU's one.
#include "vml/job_runner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vml {
namespace {

// A result with these metrics, each scored on these frames, and this VMAF
// model.
ComparisonResult result_with(const std::vector<std::pair<std::string, std::vector<int>>>& metrics,
                             const std::string& model = "") {
    ComparisonResult result;
    result.model = model;
    for (const auto& [key, frames] : metrics) {
        FrameMetricResult metric{key, frames, {}, {}, {}};
        for (const int frame : frames) {
            metric.time.push_back(frame / 24.0);
            metric.values.push_back(static_cast<float>(frame));
        }
        result.metric_results.add(std::move(metric));
    }
    return result;
}

// A video whose metrics are calculated in these tasks.
VmafJob job_of(const std::vector<MetricTask>& tasks, int subsample = 1) {
    VmafJob job;
    job.options.compute_vmaf = false;
    for (const MetricTask& task : tasks) {
        for (const std::string& key : task.metric_keys) {
            job.metric_keys.push_back(key);
            job.options.set_metric_enabled(key, true);
        }
    }
    job.options.n_subsample = subsample;
    job.tasks = tasks;
    return job;
}

// A video's PSNR and XPSNR on the CPU, and its VMAF on the GPU.
VmafJob cpu_and_gpu(int subsample = 1) {
    return job_of({{"ffmpeg", {"psnr", "xpsnr"}}, {std::string(gpu_vmaf_task), {"vmaf"}}}, subsample);
}

// Lets `count` callers go on once all of them have arrived. Waiting longer
// than a minute means the run never let them meet: they go on, and say so.
class Meeting {
public:
    explicit Meeting(int count) : left_(count) {}

    bool arrive_and_wait() {
        std::unique_lock lock(mutex_);
        if (--left_ == 0) {
            met_.notify_all();
            return true;
        }
        return met_.wait_for(lock, std::chrono::minutes(1), [this] { return left_ <= 0; });
    }

private:
    std::mutex mutex_;
    std::condition_variable met_;
    int left_;
};

// What a run reported, by video.
struct Reported {
    struct Partial {
        ComparisonResult result;
        std::string message;
        std::string stderr_tail;
        MetricFailures failures;
    };
    std::mutex mutex;
    std::vector<std::size_t> started;
    std::map<std::size_t, ComparisonResult> finished;
    std::map<std::size_t, std::string> failed;
    std::map<std::size_t, Partial> partial;
    int updates = 0;
    int cancelled = 0;

    RunEvents events() {
        RunEvents events;
        events.job_started = [this](std::size_t index) {
            const std::lock_guard lock(mutex);
            started.push_back(index);
        };
        events.job_finished = [this](std::size_t index, const ComparisonResult& result) {
            const std::lock_guard lock(mutex);
            finished.emplace(index, result);
        };
        events.job_failed = [this](std::size_t index, const std::string& message, const std::string&) {
            const std::lock_guard lock(mutex);
            failed.emplace(index, message);
        };
        events.job_partially_failed = [this](std::size_t index, const ComparisonResult& result,
                                             const std::string& message, const std::string& stderr_tail,
                                             const MetricFailures& failures) {
            const std::lock_guard lock(mutex);
            partial.emplace(index, Partial{result, message, stderr_tail, failures});
        };
        events.result_updated = [this](std::size_t, const ComparisonResult&) {
            const std::lock_guard lock(mutex);
            ++updates;
        };
        events.cancelled = [this] {
            const std::lock_guard lock(mutex);
            ++cancelled;
        };
        return events;
    }
};

TEST(JobRunner, AVideosTasksRunAtOnceAndMakeOneResult) {
    const std::vector<VmafJob> jobs{cpu_and_gpu(2)};
    Meeting both(2);
    std::mutex mutex;
    std::vector<std::vector<std::string>> asked;  // the metrics each task was run with
    bool met = true;
    Reported reported;
    run_jobs(
        jobs, 1, reported.events(),
        [&](const VmafJob&, const VmafOptions& options, const ProgressCallback&, const StatusCallback&,
            const CancelToken&) {
            const std::vector<std::string> keys = options.requested_metrics();
            const bool together = both.arrive_and_wait();
            const std::lock_guard lock(mutex);
            asked.push_back(keys);
            met = met && together;
            if (options.compute_vmaf) {
                return result_with({{"vmaf", {0, 2, 4}}}, "version=vmaf_v0.6.1");
            }
            // XPSNR as a run of its own scores it: every frame.
            return result_with({{"psnr", {0, 2, 4}}, {"xpsnr", {0, 1, 2, 3, 4, 5}}});
        },
        24, CancelToken());

    EXPECT_TRUE(met);  // the CPU's lane and the GPU's at once: one parallel job limits only the CPU's
    std::sort(asked.begin(), asked.end());
    EXPECT_EQ(asked, (std::vector<std::vector<std::string>>{{"psnr", "xpsnr"}, {"vmaf"}}));
    EXPECT_EQ(reported.started, std::vector<std::size_t>{0});
    EXPECT_EQ(reported.updates, 1);  // the first task's scores, kept while the other ran
    ASSERT_EQ(reported.finished.size(), 1u);
    const ComparisonResult& result = reported.finished.at(0);
    EXPECT_EQ(result.model, "version=vmaf_v0.6.1");  // from the task that calculated VMAF
    for (const std::string key : {"vmaf", "psnr", "xpsnr"}) {
        ASSERT_NE(result.metric_results.frame(key), nullptr) << key;
        // XPSNR on the frames the subsampled VMAF and PSNR have.
        EXPECT_EQ(result.metric_results.frame(key)->frame, (std::vector<int>{0, 2, 4})) << key;
    }
}

TEST(JobRunner, AFailedTaskLeavesTheVideosOtherMetrics) {
    // The first video's VMAF fails on the GPU while its PSNR finishes; both
    // of the second video's tasks fail.
    const std::vector<VmafJob> jobs{cpu_and_gpu(), cpu_and_gpu()};
    Reported reported;
    run_jobs(
        jobs, 1, reported.events(),
        [&](const VmafJob& job, const VmafOptions& options, const ProgressCallback&, const StatusCallback&,
            const CancelToken&) {
            if (options.compute_vmaf) {
                throw VmafRunError("Vulkan ran out of memory.", "the end of its output");
            }
            if (&job == &jobs[1]) {
                throw VmafRunError("FFmpeg could not decode it.");
            }
            return result_with({{"psnr", {0, 1}}, {"xpsnr", {0, 1}}});
        },
        24, CancelToken());

    ASSERT_EQ(reported.partial.size(), 1u);
    const Reported::Partial& first = reported.partial.at(0);
    EXPECT_TRUE(first.result.has_metric("psnr"));
    EXPECT_FALSE(first.result.has_metric("vmaf"));
    EXPECT_EQ(first.message, "VMAF v0.6.1 failed: Vulkan ran out of memory.");
    EXPECT_EQ(first.stderr_tail, "the end of its output");
    EXPECT_EQ(first.failures, (MetricFailures{{"vmaf", "Vulkan ran out of memory."}}));
    // With nothing finished, the video fails, and the run goes on.
    ASSERT_EQ(reported.failed.size(), 1u);
    const std::string& second = reported.failed.at(1);
    EXPECT_TRUE(second == "Vulkan ran out of memory." || second == "FFmpeg could not decode it.") << second;
    EXPECT_TRUE(reported.finished.empty());
    EXPECT_EQ(reported.cancelled, 0);
}

TEST(JobRunner, ACancelledRunKeepsWhatHadFinished) {
    const std::vector<VmafJob> jobs{cpu_and_gpu()};
    CancelToken cancel;
    Reported reported;
    RunEvents events = reported.events();
    // Cancelled once the GPU's VMAF is in, while the CPU's PSNR still runs.
    events.result_updated = [&cancel](std::size_t, const ComparisonResult&) { cancel.cancel(); };
    run_jobs(
        jobs, 1, events,
        [](const VmafJob&, const VmafOptions& options, const ProgressCallback&, const StatusCallback&,
           const CancelToken& token) {
            if (options.compute_vmaf) {
                return result_with({{"vmaf", {0, 1}}});
            }
            for (int waited = 0; !token.cancelled() && waited < 60000; ++waited) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            throw Cancelled();
        },
        24, cancel);

    ASSERT_EQ(reported.finished.size(), 1u);
    EXPECT_TRUE(reported.finished.at(0).has_metric("vmaf"));
    EXPECT_FALSE(reported.finished.at(0).has_metric("psnr"));
    EXPECT_TRUE(reported.failed.empty());
    EXPECT_TRUE(reported.partial.empty());
    EXPECT_EQ(reported.cancelled, 1);
}

TEST(JobRunner, VideosShareTheCpusTwoLanesAndTheGpusOne) {
    // Three videos with metrics on the CPU only, then two with VMAF on the
    // GPU only.
    std::vector<VmafJob> jobs;
    for (int i = 0; i < 3; ++i) {
        jobs.push_back(job_of({{"ffmpeg", {"psnr"}}}));
    }
    for (int i = 0; i < 2; ++i) {
        jobs.push_back(job_of({{std::string(gpu_vmaf_task), {"vmaf"}}}));
    }
    jobs[2].options.n_threads = 5;  // a thread count the user chose is kept
    Meeting first_two(2);
    std::mutex mutex;
    std::map<bool, int> running;  // tasks running at once, on the GPU or not
    std::map<bool, int> most;
    std::vector<std::size_t> cpu_order;
    std::vector<int> threads(jobs.size());
    bool met = true;
    Reported reported;
    run_jobs(
        jobs, 2, reported.events(),
        [&](const VmafJob& job, const VmafOptions& options, const ProgressCallback&, const StatusCallback&,
            const CancelToken&) {
            const bool on_gpu = options.compute_vmaf;
            std::size_t started = 0;
            {
                const std::lock_guard lock(mutex);
                most[on_gpu] = std::max(most[on_gpu], ++running[on_gpu]);
                threads[static_cast<std::size_t>(&job - jobs.data())] = options.n_threads;
                if (!on_gpu) {
                    started = cpu_order.size();
                    cpu_order.push_back(static_cast<std::size_t>(&job - jobs.data()));
                }
            }
            // The first two CPU tasks run at once; the third waits for one.
            const bool together = on_gpu || started >= 2 || first_two.arrive_and_wait();
            const std::lock_guard lock(mutex);
            met = met && together;
            --running[on_gpu];
            return result_with({{on_gpu ? "vmaf" : "psnr", {0}}});
        },
        24, CancelToken());

    EXPECT_TRUE(met);
    EXPECT_EQ(most[false], 2);
    EXPECT_EQ(most[true], 1);
    ASSERT_EQ(cpu_order.size(), 3u);
    EXPECT_EQ(cpu_order.back(), 2u);  // in list order
    // "Auto" threads: an equal share of the 24 cores for each of the CPU's
    // two lanes, for the GPU's videos too (in case the CPU takes over).
    EXPECT_EQ(threads, (std::vector<int>{12, 12, 5, 12, 12}));
    EXPECT_EQ(reported.finished.size(), jobs.size());
}

}  // namespace
}  // namespace vml
