// Running a list of videos' metrics (job_runner.h), with a stand-in for
// run_vmaf: a video's tasks run at once and make one result, a task that
// fails leaves the video's other metrics, a cancelled run keeps what had
// finished, videos share the CPU's two lanes and the GPU's one, and saved
// scores are kept instead of calculated again.
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

// A perceptual task's output with these metrics, each scored on these frames.
PerceptualTaskOutput output_with(const std::vector<std::pair<std::string, std::vector<int>>>& metrics,
                                 long long compared = 0) {
    PerceptualTaskOutput output;
    output.metrics = result_with(metrics).metric_results;
    output.source_crop = CropBox{3840, 1608, 0, 276};
    output.compared_frame_count = compared;
    return output;
}

// Vship's task, beside VMAF on the GPU: each of its passes' scores is kept
// as the pass finishes, its run line shows its passes, and a metric that
// failed while the others finished is reported like a failed task.
// SSIMULACRA2 and Butteraugli chosen for the CPU are a task of their own;
// CVVDP stays on the GPU.
TEST(JobRunner, VshipsPassesAreKeptAsTheyFinish) {
    VmafJob job = job_of({{std::string(gpu_vmaf_task), {"vmaf"}}});
    job.metric_keys = {"vmaf", "ssimulacra2", "butteraugli"};
    job.tasks.push_back({std::string(perceptual_task), {"ssimulacra2", "butteraugli"}});
    const std::vector<VmafJob> jobs{job};
    Reported reported;
    std::mutex mutex;
    std::vector<std::vector<std::vector<std::string>>> passes_shown;
    RunEvents events = reported.events();
    events.task_progress = [&](std::size_t, const std::vector<TaskSnapshot>& tasks) {
        const std::lock_guard lock(mutex);
        passes_shown.push_back(tasks.back().passes);
    };
    int updates_seen_by_second_pass = -1;
    run_jobs(
        jobs, 1, events,
        [](const VmafJob&, const VmafOptions&, const ProgressCallback&, const StatusCallback&, const CancelToken&) {
            return result_with({{"vmaf", {0, 1}}});
        },
        [&](const VmafJob&, const std::vector<std::string>& keys, bool together, int concurrent_cpu_tasks,
            const PerceptualHooks& hooks, const CancelToken&) {
            EXPECT_EQ(keys, (std::vector<std::string>{"ssimulacra2", "butteraugli"}));
            EXPECT_FALSE(together);
            EXPECT_EQ(concurrent_cpu_tasks, 1);  // one parallel job
            hooks.on_pass(1, 2, {"ssimulacra2"});
            hooks.on_pass_done(output_with({{"ssimulacra2", {0, 1}}}, 2));
            {
                const std::lock_guard lock(reported.mutex);
                updates_seen_by_second_pass = reported.updates;
            }
            hooks.on_pass(2, 2, {"butteraugli"});
            PerceptualTaskOutput output = output_with({{"ssimulacra2", {0, 1}}}, 2);
            output.failures = {{"butteraugli", "out of GPU memory"}};
            return output;
        },
        false, 24, CancelToken());

    EXPECT_GE(updates_seen_by_second_pass, 1);  // the first pass's scores, kept before the second started
    // The planned passes from the start, while the task waits for VMAF.
    const std::vector<std::vector<std::string>> one_each{{"ssimulacra2"}, {"butteraugli"}};
    ASSERT_FALSE(passes_shown.empty());
    EXPECT_EQ(passes_shown.front(), one_each);
    EXPECT_EQ(passes_shown.back(), one_each);
    ASSERT_EQ(reported.partial.size(), 1u);
    const Reported::Partial& partial = reported.partial.at(0);
    EXPECT_TRUE(partial.result.has_metric("vmaf"));
    EXPECT_TRUE(partial.result.has_metric("ssimulacra2"));
    EXPECT_FALSE(partial.result.has_metric("butteraugli"));
    EXPECT_EQ(partial.message, "Butteraugli failed: out of GPU memory");
    EXPECT_EQ(partial.failures, (MetricFailures{{"butteraugli", "out of GPU memory"}}));

    VideoInfo source;
    source.path = "reference.mkv";
    source.width = 3840;
    source.height = 2160;
    source.fps = 25.0;
    source.codec_name = "hevc";
    source.pix_fmt = "yuv420p10le";
    VideoInfo test = source;
    test.path = "encode.mkv";
    test.fps = 24.0;
    VmafOptions options;
    options.compute_vmaf = false;
    const std::vector<MetricTask> planned = plan_tasks(source, test, options, {"ssimulacra2", "butteraugli", "cvvdp"},
                                                       {{"ssimulacra2", "cpu"}, {"butteraugli", "gpu"}});
    ASSERT_EQ(planned.size(), 2u);
    EXPECT_EQ(planned[0].backend, perceptual_task);
    EXPECT_EQ(planned[0].metric_keys, (std::vector<std::string>{"butteraugli", "cvvdp"}));
    EXPECT_EQ(planned[1].backend, perceptual_cpu_task);
    EXPECT_EQ(planned[1].metric_keys, std::vector<std::string>{"ssimulacra2"});
}

// A video with perceptual metrics alone has a result made from its job, with
// the perceptual task's black-bar crops and frame count.
TEST(JobRunner, PerceptualMetricsAloneMakeAResult) {
    VmafJob job = job_of({});
    job.source_info.path = "reference.mkv";
    job.source_info.fps = 25.0;
    job.distorted_info.path = "encode.mkv";
    job.result_distorted_path = "named encode.mkv";
    job.metric_keys = {"ssimulacra2"};
    job.tasks = {{std::string(perceptual_task), {"ssimulacra2"}}};
    const std::vector<VmafJob> jobs{job};
    Reported reported;
    run_jobs(
        jobs, 1, reported.events(),
        [](const VmafJob&, const VmafOptions&, const ProgressCallback&, const StatusCallback&,
           const CancelToken&) -> ComparisonResult { throw std::logic_error("no FFmpeg task"); },
        [](const VmafJob&, const std::vector<std::string>&, bool, int, const PerceptualHooks&, const CancelToken&) {
            return output_with({{"ssimulacra2", {0, 3, 6}}}, 9);
        },
        false, 24, CancelToken());

    ASSERT_EQ(reported.finished.size(), 1u);
    const ComparisonResult& result = reported.finished.at(0);
    EXPECT_EQ(result.source, std::filesystem::path("reference.mkv"));
    EXPECT_EQ(result.distorted, std::filesystem::path("named encode.mkv"));
    EXPECT_EQ(result.fps, 25.0);  // the source's, as the Python app has it
    EXPECT_EQ(result.compared_frame_count, 9);
    EXPECT_EQ(result.source_crop, (CropBox{3840, 1608, 0, 276}));
    ASSERT_NE(result.metric_results.frame("ssimulacra2"), nullptr);
    EXPECT_EQ(result.metric_results.frame("ssimulacra2")->frame, (std::vector<int>{0, 3, 6}));
}

// Saved scores, scored on frames 0 and 1.
MetricResultSet saved_scores(const std::vector<std::string>& keys) {
    std::vector<std::pair<std::string, std::vector<int>>> metrics;
    for (const std::string& key : keys) {
        metrics.emplace_back(key, std::vector<int>{0, 1});
    }
    return result_with(metrics).metric_results;
}

// The saved scores a run keeps are left out of its tasks, as the Python app
// plans them: FFmpeg's metrics are calculated together, so a saved one is
// calculated again with the rest unless all of them are saved; SSIMULACRA2,
// Butteraugli and CVVDP are each left out.
TEST(JobRunner, SavedScoresAreNotCalculatedAgain) {
    VideoInfo source;
    source.path = "reference.mkv";
    source.width = 1920;
    source.height = 1080;
    source.pix_fmt = "yuv420p";
    VideoInfo test = source;
    test.path = "encode.mkv";
    VmafOptions options;
    options.vmaf_on_gpu = false;  // VMAF with FFmpeg's other metrics
    const std::vector<std::string> asked{"vmaf", "psnr", "ssimulacra2", "butteraugli"};

    std::vector<MetricTask> planned = plan_tasks(source, test, options, asked, {}, saved_scores({"vmaf", "ssimulacra2"}));
    ASSERT_EQ(planned.size(), 2u);
    EXPECT_EQ(planned[0].backend, "ffmpeg");
    EXPECT_EQ(planned[0].metric_keys, (std::vector<std::string>{"vmaf", "psnr"}));
    EXPECT_EQ(planned[1].backend, perceptual_task);
    EXPECT_EQ(planned[1].metric_keys, std::vector<std::string>{"butteraugli"});

    planned = plan_tasks(source, test, options, asked, {}, saved_scores({"vmaf", "psnr", "butteraugli"}));
    ASSERT_EQ(planned.size(), 1u);
    EXPECT_EQ(planned[0].backend, perceptual_task);
    EXPECT_EQ(planned[0].metric_keys, std::vector<std::string>{"ssimulacra2"});
}

// The saved scores a run keeps fill in its result where it calculates none,
// with the saved VMAF's model; a score it calculates replaces the saved one.
TEST(JobRunner, SavedScoresFillInWhatTheRunDoesNotCalculate) {
    // The first video's VMAF is saved and not calculated; the second's is
    // calculated again with its PSNR (VMAF on the CPU).
    std::vector<VmafJob> jobs{job_of({{"ffmpeg", {"psnr"}}}), job_of({{"ffmpeg", {"vmaf", "psnr"}}})};
    for (VmafJob& job : jobs) {
        job.metric_keys = {"vmaf", "psnr", "ssimulacra2"};
        job.cached_result = result_with({{"vmaf", {0, 1}}, {"ssimulacra2", {0, 1}}}, "version=vmaf_4k_v0.6.1");
        job.cached_metrics = job.cached_result->metric_results;
    }
    Reported reported;
    run_jobs(
        jobs, 1, reported.events(),
        [](const VmafJob&, const VmafOptions& options, const ProgressCallback&, const StatusCallback&,
           const CancelToken&) {
            if (!options.compute_vmaf) {
                return result_with({{"psnr", {0, 1}}});
            }
            ComparisonResult result = result_with({{"psnr", {0, 1}}}, "version=vmaf_v0.6.1");
            result.metric_results.add(FrameMetricResult{"vmaf", {0, 1}, {0.0, 1 / 24.0}, {80.0F, 90.0F}, {}});
            return result;
        },
        24, CancelToken());

    ASSERT_EQ(reported.finished.size(), 2u);
    for (const auto& [index, result] : reported.finished) {
        for (const std::string key : {"vmaf", "psnr", "ssimulacra2"}) {
            EXPECT_TRUE(result.has_metric(key)) << index << " " << key;
        }
    }
    const ComparisonResult& kept = reported.finished.at(0);
    EXPECT_EQ(kept.model, "version=vmaf_4k_v0.6.1");
    EXPECT_EQ(kept.metric_results.frame("vmaf")->values, (std::vector<float>{0.0F, 1.0F}));
    const ComparisonResult& replaced = reported.finished.at(1);
    EXPECT_EQ(replaced.model, "version=vmaf_v0.6.1");
    EXPECT_EQ(replaced.metric_results.frame("vmaf")->values, (std::vector<float>{80.0F, 90.0F}));
}

// Cancelled before it calculated anything, a video is cancelled, whatever of
// it is saved.
TEST(JobRunner, AVideoCancelledBeforeAnyScoreIsCancelledWhateverIsSaved) {
    VmafJob job = job_of({{"ffmpeg", {"psnr"}}});
    job.metric_keys = {"vmaf", "psnr"};
    job.cached_result = result_with({{"vmaf", {0, 1}}}, "version=vmaf_v0.6.1");
    job.cached_metrics = job.cached_result->metric_results;
    const std::vector<VmafJob> jobs{job};
    CancelToken cancel;
    Reported reported;
    run_jobs(
        jobs, 1, reported.events(),
        [&cancel](const VmafJob&, const VmafOptions&, const ProgressCallback&, const StatusCallback&,
                  const CancelToken&) -> ComparisonResult {
            cancel.cancel();
            throw Cancelled();
        },
        24, cancel);

    EXPECT_TRUE(reported.finished.empty());
    EXPECT_TRUE(reported.partial.empty());
    EXPECT_TRUE(reported.failed.empty());
    EXPECT_EQ(reported.cancelled, 1);
}

}  // namespace
}  // namespace vml
