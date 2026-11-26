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
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "vml/film_grain_detect.h"
#include "vml/gpu_cpu_metrics.h"
#include "vml/gpu_decoders.h"
#include "vml/run_record.h"

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
    // Both tasks keep the choice of the GPU: the CPU's holds no VMAF it could
    // move there (HDR VMAF's features would take it).
    bool vmaf_on_gpu_as_planned = true;
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
            vmaf_on_gpu_as_planned = vmaf_on_gpu_as_planned && options.vmaf_on_gpu;
            if (options.compute_vmaf) {
                return result_with({{"vmaf", {0, 2, 4}}}, "version=vmaf_v0.6.1");
            }
            // XPSNR as a run of its own scores it: every frame.
            return result_with({{"psnr", {0, 2, 4}}, {"xpsnr", {0, 1, 2, 3, 4, 5}}});
        },
        24, CancelToken());

    EXPECT_TRUE(met);  // the CPU's lane and the GPU's at once: one parallel job limits only the CPU's
    EXPECT_TRUE(vmaf_on_gpu_as_planned);
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

// PSNR and XPSNR scored beside VMAF on the GPU from one decode (plan_tasks's
// share_decode) are one run of all three, and are shown as the CPU's task
// they were, with the GPU's progress.
TEST(JobRunner, AVvcVideoOnVvdecVulkanShowsAsDecodedOnTheGpu) {
    std::vector<VmafJob> jobs{job_of({{"ffmpeg", {"psnr"}}})};
    jobs[0].source_info.codec_name = "vvc";
    jobs[0].distorted_info.codec_name = "h264";
    Reported reported;
    RunEvents events = reported.events();
    std::vector<std::optional<HardwareDecoding>> shown;  // the plan after each status below
    events.task_progress = [&shown](std::size_t, const std::vector<TaskSnapshot>& tasks) {
        shown.push_back(tasks.at(0).decode);
    };
    std::size_t before_statuses = 0;
    run_jobs(
        jobs, 1, events,
        [&](const VmafJob&, const VmafOptions&, const ProgressCallback&, const StatusCallback& on_status,
            const CancelToken&) {
            before_statuses = shown.size();
            RunStatus planned{"Running", StatusKind::step};
            planned.decoding = HardwareDecoding{};
            on_status(planned);
            on_status({"VVC decoded on the GPU by vvdec-vulkan", StatusKind::vvdec_used});
            on_status({"it refused", StatusKind::vvdec_off});
            return result_with({{"psnr", {0, 1}}});
        },
        24, CancelToken());

    ASSERT_GE(shown.size(), before_statuses + 3);
    EXPECT_EQ(shown.at(before_statuses), HardwareDecoding{});
    EXPECT_EQ(shown.at(before_statuses + 1), (HardwareDecoding{std::string(vvdec_decoder), std::nullopt}));
    EXPECT_EQ(shown.at(before_statuses + 2), HardwareDecoding{});
}

TEST(JobRunner, CpuMetricsBesideTheGpusAreOneRunShownAsTheirOwnTask) {
    MetricTask shared{std::string(gpu_vmaf_task), {"vmaf", "psnr", "xpsnr"}};
    shared.cpu_beside = {"psnr", "xpsnr"};
    const std::vector<VmafJob> jobs{job_of({shared})};
    std::vector<std::vector<std::string>> asked;
    std::vector<TaskSnapshot> running;  // the snapshots while it ran
    Reported reported;
    RunEvents events = reported.events();
    events.task_progress = [&running](std::size_t, const std::vector<TaskSnapshot>& tasks) {
        if (!tasks.empty() && tasks.front().state == "running") {
            running = tasks;
        }
    };
    run_jobs(
        jobs, 1, events,
        [&](const VmafJob&, const VmafOptions& options, const ProgressCallback& on_progress, const StatusCallback&,
            const CancelToken&) {
            asked.push_back(options.requested_metrics());
            on_progress(5, 10, 30.0);
            return result_with({{"psnr", {0, 1}}, {"xpsnr", {0, 1}}, {"vmaf", {0, 1}}}, "version=vmaf_v0.6.1");
        },
        24, CancelToken());

    EXPECT_EQ(asked, (std::vector<std::vector<std::string>>{{"vmaf", "psnr", "xpsnr"}}));
    ASSERT_EQ(running.size(), 2u);
    EXPECT_EQ(running[0].backend, "ffmpeg");
    EXPECT_EQ(running[0].lane, "cpu");
    EXPECT_EQ(running[0].metric_keys, (std::vector<std::string>{"psnr", "xpsnr"}));
    EXPECT_EQ(running[1].backend, gpu_vmaf_task);
    EXPECT_EQ(running[1].lane, "gpu");
    EXPECT_EQ(running[1].metric_keys, std::vector<std::string>{"vmaf"});
    for (const TaskSnapshot& task : running) {
        EXPECT_EQ(task.current, 5);
        EXPECT_EQ(task.total, 10);
    }
    ASSERT_EQ(reported.finished.size(), 1u);
    for (const std::string key : {"vmaf", "psnr", "xpsnr"}) {
        EXPECT_TRUE(reported.finished.at(0).has_metric(key)) << key;
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
    std::vector<RunFacts> facts;  // sent from the last lane, after the other's ended
    events.job_facts = [&facts](std::size_t, const RunFacts& sent) { facts.push_back(sent); };
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
    ASSERT_EQ(facts.size(), 1u);
    const auto outcomes = facts[0].metrics;
    EXPECT_EQ(outcomes, (std::vector<std::pair<std::string, MetricOutcome>>{
                            {"psnr", {metric_cancelled, ""}},
                            {"xpsnr", {metric_cancelled, ""}},
                            {"vmaf", {metric_calculated, ""}}}));
}

// A cancelled run checks no frame pairing, and the video's result, kept and
// saved again with what had finished, keeps the saved verdict: one saved as
// out of sync stays so.
TEST(JobRunner, ACancelledRunKeepsTheSavedFramePairingVerdict) {
    VmafJob job = cpu_and_gpu();
    job.metric_keys.push_back("ssim");
    job.cached_result = result_with({{"ssim", {0, 1}}});
    job.cached_result->frames_short = FramesShort{235, 240, 235, 240, 240, 240};
    job.cached_metrics = job.cached_result->metric_results;
    CancelToken cancel;
    Reported reported;
    RunEvents events = reported.events();
    events.result_updated = [&cancel](std::size_t, const ComparisonResult&) { cancel.cancel(); };
    run_jobs(
        {job}, 1, events,
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
    EXPECT_EQ(reported.finished.at(0).frames_short, job.cached_result->frames_short);
}

// A run's time, the clock of its CancelToken, which its tasks move on.
class FakeClock {
public:
    double now() {
        const std::lock_guard lock(mutex_);
        return now_;
    }
    void advance(double seconds) {
        const std::lock_guard lock(mutex_);
        now_ += seconds;
    }

private:
    std::mutex mutex_;
    double now_ = 100.0;
};

// Each video's facts say how each metric came out (calculated, saved,
// failed with why) and how long each pass took, its paused time apart, with
// its frames and decoders.
TEST(JobRunner, AVideosFactsSayHowItsMetricsCameOutAndHowLongItsPassesTook) {
    VmafJob job = cpu_and_gpu();
    job.metric_keys.emplace_back("ssim");
    job.cached_result = result_with({{"ssim", {0, 1}}});
    job.cached_metrics = job.cached_result->metric_results;
    job.distorted_info.codec_name = "hevc";
    job.distorted_info.pix_fmt = "yuv444p";
    job.distorted_info.width = 1920;
    job.distorted_info.height = 1080;
    FakeClock clock;
    CancelToken cancel([&clock] { return clock.now(); });
    Reported reported;
    RunEvents events = reported.events();
    std::vector<RunFacts> facts;
    events.job_facts = [&facts](std::size_t, const RunFacts& sent) { facts.push_back(sent); };
    const HardwareDecoding decode{"cuda", std::nullopt};
    run_jobs(
        {job}, 1, events,
        [&](const VmafJob&, const VmafOptions& options, const ProgressCallback& progress, const StatusCallback& status,
            const CancelToken&) {
            if (options.compute_vmaf) {
                throw VmafRunError("Vulkan ran out of memory.");
            }
            RunStatus planned{"", StatusKind::decoding_why};
            planned.decoding_why = {"", "its picture size changes partway through"};
            status(planned);
            status({"Calculating", StatusKind::step, "Calculating", decode});
            clock.advance(1.0);
            progress(24, 48, 24.0);
            cancel.pause();
            clock.advance(2.0);
            cancel.resume();
            clock.advance(1.0);
            progress(40, 48, 24.0);  // the last report comes before the end
            ComparisonResult result = result_with({{"psnr", {0, 1}}, {"xpsnr", {0, 1}}});
            result.compared_frame_count = 48;
            return result;
        },
        24, cancel);

    ASSERT_EQ(facts.size(), 1u);
    const RunFacts& video = facts[0];
    EXPECT_EQ(video.seconds, 4.0);
    EXPECT_EQ(video.paused_seconds, 2.0);
    EXPECT_FALSE(video.started_utc.empty());
    EXPECT_EQ(video.settings.metrics, (std::vector<std::string>{"psnr", "xpsnr", "vmaf", "ssim"}));
    EXPECT_EQ(video.metrics, (std::vector<std::pair<std::string, MetricOutcome>>{
                                 {"psnr", {metric_calculated, ""}},
                                 {"xpsnr", {metric_calculated, ""}},
                                 {"vmaf", {metric_failed, "Vulkan ran out of memory."}},
                                 {"ssim", {metric_saved, ""}}}));
    ASSERT_EQ(video.passes.size(), 2u);
    const auto cpu = std::find_if(video.passes.begin(), video.passes.end(),
                                  [](const PassTiming& pass) { return pass.task == "ffmpeg"; });
    ASSERT_NE(cpu, video.passes.end());
    // A second to its first report, of 24 frames, then the other 24 in three seconds less the pause's two: 24 fps,
    // not the 48 frames over that one second. The test video, decoded
    // in software, says why, in the results JSON too: as the run said when it made its plan (it knew of the size
    // change), not as its file's details alone would (4:4:4)
    const std::string why = "its picture size changes partway through";
    EXPECT_EQ(*cpu, (PassTiming{"ffmpeg", {"psnr", "xpsnr"}, "cpu", 4.0, 2.0, 1.0, 48, 24.0, "done", decode,
                                {"", why}}));
    nlohmann::ordered_json entry;
    add_video_record(entry, &video, nullptr, "");
    const auto planned = std::find_if(entry["decode_plan"].begin(), entry["decode_plan"].end(),
                                      [](const auto& plan) { return plan["metrics"].size() == 2; });
    ASSERT_NE(planned, entry["decode_plan"].end());
    EXPECT_EQ((*planned)["decoders"],
              (nlohmann::ordered_json{{"source", "cuda"}, {"test", "software"}, {"test_why", why}}));
    const auto gpu = std::find_if(video.passes.begin(), video.passes.end(),
                                  [](const PassTiming& pass) { return pass.task != "ffmpeg"; });
    EXPECT_EQ(gpu->on, "gpu");
    EXPECT_EQ(gpu->end, "failed");
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

// The parallel jobs can be raised while a run goes on, as the window's check
// box does: a video waiting for the CPU then starts beside the one running.
// The one running keeps every core; the one started after gets its share.
TEST(JobRunner, RaisingTheParallelJobsDuringARunStartsAWaitingVideo) {
    const std::vector<VmafJob> jobs{job_of({{"ffmpeg", {"psnr"}}}), job_of({{"ffmpeg", {"psnr"}}})};
    Meeting both(2);
    std::mutex mutex;
    int started = 0;
    std::vector<int> threads;
    bool met = true;
    Reported reported;
    JobScheduler* running = nullptr;
    JobScheduler scheduler(
        jobs, 1, reported.events(),
        [&](const VmafJob&, const VmafOptions& options, const ProgressCallback&, const StatusCallback&,
            const CancelToken&) {
            bool first = false;
            {
                const std::lock_guard lock(mutex);
                first = started++ == 0;
                threads.push_back(options.n_threads);
            }
            if (first) {
                running->set_parallel_jobs(2);
            }
            // Met only if the second video starts while the first still runs.
            const bool together = both.arrive_and_wait();
            const std::lock_guard lock(mutex);
            met = met && together;
            return result_with({{"psnr", {0}}});
        },
        PerceptualRunner(), false, 24, CancelToken());
    running = &scheduler;
    scheduler.run();

    EXPECT_TRUE(met);
    EXPECT_EQ(scheduler.parallel_jobs(), 2);
    EXPECT_EQ(threads, (std::vector<int>{0, 12}));  // "Auto": every core alone, then half each
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

// A run of two 10 s videos at 24 fps (240 frames each by their length) that
// scored 235 pairs: one of them has fewer frames, or both have gaps
// (frame_pairing.h). Without a frame counter its result says the frames may
// be out of step; with one, counted, whether they are; with a range, it
// says nothing and counts nothing. A metric scored every 2nd frame up to its
// last reaches the end. The counts come from the run's videos, by path.
TEST(JobRunner, AResultShortOfTheVideosLengthsSaysTheirFramesMayBeOutOfStep) {
    const auto job_scoring = [](std::vector<int> frames, long long range_end = 0) {
        VmafJob job = job_of({{"ffmpeg", {"psnr"}}});
        job.source_info.path = "reference.mkv";
        job.distorted_info.path = "encode.mkv";
        for (VideoInfo* video : {&job.source_info, &job.distorted_info}) {
            video->duration = 10.0;
            video->fps = 24.0;
        }
        job.options.range_end = range_end;
        return std::pair{job, frames};
    };
    std::vector<int> short_frames(235);
    std::iota(short_frames.begin(), short_frames.end(), 0);
    std::vector<int> every_second;
    for (int frame = 0; frame < 240; frame += 2) {
        every_second.push_back(frame);
    }
    const auto finished = [](const std::pair<VmafJob, std::vector<int>>& run, const FrameCounter& count) {
        Reported reported;
        RunEvents events = reported.events();
        events.count_frames = count;
        run_jobs(
            {run.first}, 1, events,
            [&run](const VmafJob& job, const VmafOptions&, const ProgressCallback&, const StatusCallback&,
                   const CancelToken&) {
                ComparisonResult result = result_with({{"psnr", run.second}});
                result.source_info = job.source_info;
                result.distorted_info = job.distorted_info;
                return result;
            },
            24, CancelToken());
        return reported.finished.at(0).frames_short;
    };
    int counted = 0;
    const auto counts = [&counted](long long reference, long long encode) -> FrameCounter {
        return [&counted, reference, encode](const VideoInfo& video) -> std::optional<long long> {
            ++counted;
            return video.path == "reference.mkv" ? reference : encode;
        };
    };

    const FramesShort uncounted{235, 240, std::nullopt, std::nullopt, 240, 240};
    EXPECT_EQ(finished(job_scoring(short_frames), nullptr), uncounted);
    EXPECT_EQ(finished(job_scoring(short_frames), counts(235, 240)),
              (FramesShort{235, 240, 235, 240, 240, 240}));  // out of step
    EXPECT_TRUE(finished(job_scoring(short_frames), counts(235, 235))->in_step());
    // A count far from what the length holds (an interlaced video's fields)
    EXPECT_EQ(finished(job_scoring(short_frames), counts(470, 240)), uncounted);
    EXPECT_EQ(counted, 6);
    EXPECT_EQ(finished(job_scoring(short_frames, 5), counts(235, 240)), std::nullopt);
    EXPECT_EQ(finished(job_scoring(every_second), counts(235, 240)), std::nullopt);
    EXPECT_EQ(counted, 6);  // neither counted
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

// A run that keeps all of FFmpeg's metrics from the saved scores builds on
// the saved result, but with its videos' film grain as the run reads and
// scores them: one saved before either was known has none, and the new
// scores would be saved under another identity.
TEST(JobRunner, AResultBuiltOnSavedScoresHasTheVideosFilmGrainAsScored) {
    VmafJob job = job_of({});
    job.source_info.film_grain = 0u;
    job.distorted_info.film_grain = film_grain_h274;
    job.distorted_info.with_film_grain = false;
    job.metric_keys = {"vmaf", "ssimulacra2"};
    job.tasks = {{std::string(perceptual_task), {"ssimulacra2"}}};
    job.cached_result = result_with({{"vmaf", {0, 3, 6}}}, "version=vmaf_v0.6.1");  // no film grain read
    job.cached_result->source_crop = CropBox{3840, 1608, 0, 276};  // the box output_with's task finds
    job.cached_metrics = job.cached_result->metric_results;
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
    EXPECT_TRUE(result.has_metric("vmaf") && result.has_metric("ssimulacra2"));
    EXPECT_EQ(result.source_info.film_grain, 0u);
    EXPECT_EQ(result.distorted_info.film_grain, film_grain_h274);
    EXPECT_EQ(result.distorted_info.with_film_grain, false);
}

// Saved scores of another black-bar box are not mixed with the run's new ones:
// the detection changes between versions, and they would mix different pictures.
TEST(JobRunner, SavedScoresOfOtherBlackBarsAreNotKeptBesideNewOnes) {
    VmafJob job = job_of({});
    job.metric_keys = {"vmaf", "ssimulacra2"};
    job.tasks = {{std::string(perceptual_task), {"ssimulacra2"}}};
    job.cached_result = result_with({{"vmaf", {0, 3, 6}}}, "version=vmaf_v0.6.1");
    job.cached_result->source_crop = CropBox{3840, 1600, 0, 280};
    job.cached_metrics = job.cached_result->metric_results;
    const std::vector<VmafJob> jobs{job};
    Reported reported;
    run_jobs(
        jobs, 1, reported.events(),
        [](const VmafJob&, const VmafOptions&, const ProgressCallback&, const StatusCallback&,
           const CancelToken&) -> ComparisonResult { throw std::logic_error("no FFmpeg task"); },
        [](const VmafJob&, const std::vector<std::string>&, bool, int, const PerceptualHooks&, const CancelToken&) {
            return output_with({{"ssimulacra2", {0, 3, 6}}}, 9);  // its box: 3840x1608
        },
        false, 24, CancelToken());

    ASSERT_EQ(reported.partial.size(), 1u);
    const ComparisonResult& result = reported.partial.at(0).result;
    EXPECT_EQ(reported.partial.at(0).failures.size(), 1u);
    EXPECT_FALSE(result.has_metric("vmaf"));
    EXPECT_TRUE(result.has_metric("ssimulacra2"));
    EXPECT_EQ(result.source_crop, (CropBox{3840, 1608, 0, 276}));
}

// Saved scores, scored on frames 0 and 1.
MetricResultSet saved_scores(const std::vector<std::string>& keys) {
    std::vector<std::pair<std::string, std::vector<int>>> metrics;
    for (const std::string& key : keys) {
        metrics.emplace_back(key, std::vector<int>{0, 1});
    }
    return result_with(metrics).metric_results;
}

// Each saved score a run keeps is left out of its tasks: FFmpeg's metrics
// as SSIMULACRA2, Butteraugli and CVVDP are.
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
    EXPECT_EQ(planned[0].metric_keys, std::vector<std::string>{"psnr"});
    EXPECT_EQ(planned[1].backend, perceptual_task);
    EXPECT_EQ(planned[1].metric_keys, std::vector<std::string>{"butteraugli"});

    planned = plan_tasks(source, test, options, asked, {}, saved_scores({"vmaf", "psnr", "butteraugli"}));
    ASSERT_EQ(planned.size(), 1u);
    EXPECT_EQ(planned[0].backend, perceptual_task);
    EXPECT_EQ(planned[0].metric_keys, std::vector<std::string>{"ssimulacra2"});

    const std::vector<std::string> every{"vmaf", "vmaf_neg", "vmaf_v1", "psnr", "ssim", "xpsnr"};
    planned = plan_tasks(source, test, options, every, {}, saved_scores({"vmaf", "vmaf_v1", "psnr", "ssim"}));
    ASSERT_EQ(planned.size(), 1u);
    EXPECT_EQ(planned[0].metric_keys, (std::vector<std::string>{"vmaf_neg", "xpsnr"}));

    // With VMAF on the GPU, NEG missing beside a VMAF saved from the CPU (one
    // with a model file of its own) is calculated where the two would be, on
    // the CPU, and alone: the saved VMAF is not calculated again.
    options.vmaf_on_gpu = true;
    options.model = "model_path=C\\:/models/own.json";
    MetricResultSet cpu_vmaf = saved_scores({"vmaf"});
    FrameMetricResult made_on_cpu = *cpu_vmaf.frame("vmaf");
    made_on_cpu.provenance.compute_backend = "cpu";
    cpu_vmaf = MetricResultSet();
    cpu_vmaf.add(made_on_cpu);
    planned = plan_tasks(source, test, options, {"vmaf", "vmaf_neg"}, {}, cpu_vmaf);
    ASSERT_EQ(planned.size(), 1u);
    EXPECT_EQ(planned[0].backend, "ffmpeg");
    EXPECT_EQ(planned[0].metric_keys, std::vector<std::string>{"vmaf_neg"});
}

// VMAF YUV is planned as VMAF's U and V, on the CPU in a task of their own
// beside VMAF's (each heavy there: plan_tasks), and made from the three:
// (6 VMAF + U + V) / 8 of each frame, its Y VMAF's score. The U and V asked
// for it alone are not kept, and a failure of theirs is VMAF YUV's.
TEST(JobRunner, VmafYuvIsMadeFromVmafAndItsChromaPlanes) {
    VideoInfo source;
    source.path = "reference.mkv";
    source.width = 1920;
    source.height = 1080;
    source.pix_fmt = "yuv420p";
    VideoInfo test = source;
    test.path = "encode.mkv";
    VmafJob job;
    job.source_info = source;
    job.distorted_info = test;
    job.options.vmaf_on_gpu = false;
    job.metric_keys = {"vmaf", "psnr", "vmaf_yuv", "vmaf_yuv_y"};
    for (const std::string& key : job.metric_keys) {
        job.options.set_metric_enabled(key, true);
    }
    EXPECT_EQ(plan_tasks(source, test, job.options, job.metric_keys, {}, saved_scores({"vmaf_yuv"}))[0].metric_keys,
              (std::vector<std::string>{"vmaf", "psnr"}));
    job.tasks = plan_tasks(source, test, job.options, job.metric_keys);
    ASSERT_EQ(job.tasks.size(), 2u);
    EXPECT_EQ(job.tasks[0].metric_keys, (std::vector<std::string>{"vmaf", "psnr"}));
    EXPECT_EQ(job.tasks[1].metric_keys, (std::vector<std::string>{"vmaf_yuv_u", "vmaf_yuv_v"}));
    EXPECT_TRUE(job.tasks[1].vmaf_on_cpu);

    const auto scores = [](const std::string& key, std::vector<float> values) {
        return FrameMetricResult{key, {0, 1}, {0.0, 1 / 24.0}, std::move(values), {}};
    };
    const std::map<std::string, std::vector<float>> scored{
        {"vmaf", {80, 90}}, {"psnr", {40, 41}}, {"vmaf_yuv_u", {60, 70}}, {"vmaf_yuv_v", {40, 50}}};
    VmafJob second = job;
    second.label = "second";
    std::mutex mutex;
    std::set<std::vector<std::string>> asked;
    Reported reported;
    run_jobs(
        {job, second}, 1, reported.events(),
        [&](const VmafJob& video, const VmafOptions& options, const ProgressCallback&, const StatusCallback&,
            const CancelToken&) {
            const std::vector<std::string> keys = options.requested_metrics();
            {
                const std::lock_guard lock(mutex);
                asked.insert(keys);
            }
            ComparisonResult result;
            for (const std::string& key : keys) {
                if (key == "vmaf_yuv_v" && video.label == "second") {  // its V cannot be scored
                    result.metric_failures[key] = "its chroma is too small";
                } else {
                    result.metric_results.add(scores(key, scored.at(key)));
                }
            }
            return result;
        },
        24, CancelToken());

    EXPECT_EQ(asked, (std::set<std::vector<std::string>>{{"vmaf", "psnr"}, {"vmaf_yuv_u", "vmaf_yuv_v"}}));
    ASSERT_EQ(reported.finished.size(), 1u);
    const MetricResultSet& made = reported.finished.at(0).metric_results;
    ASSERT_NE(made.frame("vmaf_yuv"), nullptr);
    EXPECT_EQ(made.frame("vmaf_yuv")->values, (std::vector<float>{72.5F, 82.5F}));
    EXPECT_EQ(made.frame("vmaf_yuv_y")->values, made.frame("vmaf")->values);
    EXPECT_EQ(made.keys(), (std::vector<std::string>{"vmaf", "psnr", "vmaf_yuv", "vmaf_yuv_y"}));
    ASSERT_EQ(reported.partial.size(), 1u);
    const Reported::Partial& failed = reported.partial.begin()->second;
    EXPECT_FALSE(failed.result.has_metric("vmaf_yuv"));
    EXPECT_FALSE(failed.result.has_metric("vmaf_yuv_u"));
    EXPECT_EQ(failed.failures, (MetricFailures{{"vmaf_yuv", "its chroma is too small"}}));
    EXPECT_EQ(failed.message, "VMAF YUV (6:1:1) failed: its chroma is too small");
}

// A metric that has a GPU route but runs on the CPU (set there, or the GPU
// checks found no GPU for it) is alone in its task: the first keeps the
// CPU-only metrics, each other one has a task of its own, and SSIMULACRA2
// and Butteraugli share their task, scored one at a time in it. Where the GPU
// takes them, the plan is one task, as before.
TEST(JobRunner, AMetricWithAGpuRouteOnTheCpuIsAloneInItsTask) {
    VideoInfo source;
    source.path = "reference.mkv";
    source.width = 1920;
    source.height = 1080;
    source.pix_fmt = "yuv420p";
    VideoInfo test = source;
    test.path = "encode.mkv";
    using Keys = std::vector<std::string>;
    const auto shape = [](const std::vector<MetricTask>& tasks) {
        std::vector<std::tuple<std::string, Keys, bool, bool>> made;
        for (const MetricTask& task : tasks) {
            made.emplace_back(task.backend, task.metric_keys, task.heavy_cpu, task.vmaf_on_cpu);
        }
        return made;
    };
    const std::string ffmpeg = "ffmpeg";
    const std::string vship(perceptual_task);
    const std::string vship_cpu(perceptual_cpu_task);

    // Set to the CPU: VMAF takes CAMBI, CIEDE2000 and VMAF YUV's U and V there with it.
    VmafOptions options;
    options.vmaf_on_gpu = false;
    EXPECT_EQ(shape(plan_tasks(source, test, options,
                               {"vmaf", "psnr", "ssim", "cambi", "ciede2000", "vmaf_yuv", "ssimulacra2", "butteraugli"},
                               {{"ssimulacra2", "cpu"}, {"butteraugli", "cpu"}})),
              (std::vector<std::tuple<std::string, Keys, bool, bool>>{
                  {ffmpeg, {"vmaf", "psnr", "ssim"}, true, true},
                  {ffmpeg, {"cambi"}, true, true},
                  {ffmpeg, {"ciede2000"}, true, true},
                  {ffmpeg, {"vmaf_yuv_u", "vmaf_yuv_v"}, true, true},
                  {vship_cpu, {"ssimulacra2", "butteraugli"}, true, false}}));

    // Set to the GPU on a PC whose checks found no Vulkan engine and no GPU for Vship.
    options.vmaf_on_gpu = true;
    const Keys asked{"psnr", "cambi", "ciede2000", "vmaf_yuv", "ssimulacra2", "butteraugli"};
    EXPECT_EQ(shape(plan_tasks(source, test, options, asked, {}, {}, false, GpuAnswers{EnginePlace{}, false})),
              (std::vector<std::tuple<std::string, Keys, bool, bool>>{
                  {ffmpeg, {"psnr", "cambi"}, true, false},
                  {ffmpeg, {"ciede2000"}, true, false},
                  {ffmpeg, {"vmaf_yuv_u", "vmaf_yuv_v"}, true, false},
                  {vship, {"ssimulacra2", "butteraugli"}, true, false}}));
    // The GPU takes them: one task each, as before.
    EXPECT_EQ(shape(plan_tasks(source, test, options, asked)),
              (std::vector<std::tuple<std::string, Keys, bool, bool>>{
                  {ffmpeg, {"psnr", "cambi", "ciede2000", "vmaf_yuv_u", "vmaf_yuv_v"}, false, false},
                  {vship, {"ssimulacra2", "butteraugli"}, false, false}}));
    // VMAF sent to the CPU by a model file, VMAF on the GPU still set: CAMBI goes with it.
    options.model = "model_path=C\\:/models/own.json";
    EXPECT_EQ(shape(plan_tasks(source, test, options, {"vmaf", "psnr", "cambi"})),
              (std::vector<std::tuple<std::string, Keys, bool, bool>>{{ffmpeg, {"vmaf", "psnr"}, true, true},
                                                                       {ffmpeg, {"cambi"}, true, true}}));
}

// A video's tasks with a metric that has a GPU route on the CPU (heavy_cpu)
// run one at a time, whatever the parallel jobs; a light one runs beside
// them. With two CPU lanes, the second takes the light task rather than the
// second heavy one, so the first heavy task meets the light one.
TEST(JobRunner, AVideosHeavyTasksRunOneAtATime) {
    std::vector<MetricTask> tasks{{"ffmpeg", {"vmaf"}}, {"ffmpeg", {"cambi"}}, {"ffmpeg", {"psnr"}}};
    tasks[0].heavy_cpu = true;
    tasks[1].heavy_cpu = true;
    const VmafJob job = job_of(tasks);
    Meeting heavy_and_light(2);
    std::mutex mutex;
    int heavy_running = 0;
    int most_heavy = 0;
    bool met = true;
    std::vector<int> threads;
    Reported reported;
    run_jobs(
        {job}, 2, reported.events(),
        [&](const VmafJob&, const VmafOptions& options, const ProgressCallback&, const StatusCallback&,
            const CancelToken&) {
            const std::vector<std::string> keys = options.requested_metrics();
            const bool heavy = keys.front() != "psnr";
            {
                const std::lock_guard lock(mutex);
                heavy_running += heavy ? 1 : 0;
                most_heavy = std::max(most_heavy, heavy_running);
                threads.push_back(options.n_threads);
            }
            const bool first = keys.front() == "vmaf";
            const bool together = (!first && heavy) || heavy_and_light.arrive_and_wait();
            const std::lock_guard lock(mutex);
            met = met && together;
            heavy_running -= heavy ? 1 : 0;
            return result_with({{keys.front(), {0}}});
        },
        24, CancelToken());

    EXPECT_TRUE(met);
    EXPECT_EQ(most_heavy, 1);
    EXPECT_EQ(threads, (std::vector<int>{12, 12, 12}));  // two lanes: the light task and one heavy at a time
    ASSERT_EQ(reported.finished.size(), 1u);
    EXPECT_EQ(reported.finished.at(0).metric_results.keys(), (std::vector<std::string>{"vmaf", "cambi", "psnr"}));
}

// The CPU's task says why a failed check of the GPU sent metrics to it, of
// those it calculates only: one note for each reason, with its metrics.
TEST(JobRunner, AFallbackNoteNamesOnlyTheMetricsCalculated) {
    const std::vector<std::pair<std::string, std::string>> failures{
        {"vmaf", "no Vulkan device"}, {"vmaf_neg", "no Vulkan device"}, {"vmaf_v1", "out of memory"}};
    EXPECT_EQ(fallback_notes(failures, {"vmaf_neg", "vmaf_v1", "psnr"}),
              (std::vector<std::string>{
                  "The GPU check failed (no Vulkan device); calculating VMAF NEG on the CPU\xE2\x80\xA6",
                  "The GPU check failed (out of memory); calculating VMAF v1 on the CPU\xE2\x80\xA6"}));
    EXPECT_TRUE(fallback_notes(failures, {"psnr"}).empty());
}

// A metric calculated alone takes the others from the saved scores: VMAF
// with its model and v1 with its model and choice, the run having calculated
// neither (its result names no VMAF model), and PSNR on the frames XPSNR,
// scored on every frame, is cut back to with subsampling.
TEST(JobRunner, AMetricCalculatedAloneJoinsTheSavedScores) {
    VmafJob neg = job_of({{"ffmpeg", {"vmaf_neg"}}});
    neg.metric_keys = {"vmaf", "vmaf_neg", "vmaf_v1"};
    neg.cached_result = result_with({{"vmaf", {0, 1}}, {"vmaf_v1", {0, 1}}}, "version=vmaf_4k_v0.6.1");
    neg.cached_result->model_v1 = "vmaf_v1_4k";
    neg.cached_result->model_choice_v1 = "4k";
    neg.cached_metrics = neg.cached_result->metric_results;
    VmafJob xpsnr = job_of({{"ffmpeg", {"xpsnr"}}}, 2);
    xpsnr.metric_keys = {"psnr", "xpsnr"};
    xpsnr.cached_result = result_with({{"psnr", {0, 2, 4}}});
    xpsnr.cached_metrics = xpsnr.cached_result->metric_results;
    Reported reported;
    run_jobs(
        {neg, xpsnr}, 1, reported.events(),
        [](const VmafJob& job, const VmafOptions&, const ProgressCallback&, const StatusCallback&, const CancelToken&) {
            return job.metric_keys.front() == "vmaf" ? result_with({{"vmaf_neg", {0, 1}}})
                                                     : result_with({{"xpsnr", {0, 1, 2, 3, 4}}});
        },
        24, CancelToken());

    ASSERT_EQ(reported.finished.size(), 2u);
    const ComparisonResult& with_neg = reported.finished.at(0);
    EXPECT_EQ(with_neg.model, "version=vmaf_4k_v0.6.1");
    EXPECT_EQ(with_neg.model_v1, "vmaf_v1_4k");
    EXPECT_EQ(with_neg.model_choice_v1, std::optional<std::string>("4k"));
    for (const std::string key : {"vmaf", "vmaf_neg", "vmaf_v1"}) {
        ASSERT_TRUE(with_neg.metric_results.frame(key)) << key;
        EXPECT_EQ(with_neg.metric_results.frame(key)->values, (std::vector<float>{0.0F, 1.0F})) << key;
    }
    const ComparisonResult& with_xpsnr = reported.finished.at(1);
    ASSERT_TRUE(with_xpsnr.metric_results.frame("xpsnr"));
    EXPECT_EQ(with_xpsnr.metric_results.frame("xpsnr")->frame, (std::vector<int>{0, 2, 4}));
    EXPECT_EQ(with_xpsnr.metric_results.frame("psnr")->frame, (std::vector<int>{0, 2, 4}));
}

// The saved scores a run keeps fill in its result where it calculates none,
// with the saved VMAF's model; a score it calculates replaces the saved one.
TEST(JobRunner, SavedScoresFillInWhatTheRunDoesNotCalculate) {
    // The first video's VMAF is saved and not calculated; the second's is
    // calculated again with its PSNR (VMAF on the CPU), as a run planned
    // otherwise would.
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

// What a run calculates names CIEDE2000, CAMBI, MS-SSIM and VMAF YUV's U and V as the
// GPU's where it may take them (VMAF's place the GPU), not as CPU metrics the
// line then calls GPU metrics; with VMAF on the CPU they are the CPU's. Once
// VMAF's backend is known, U and V only on the Vulkan engine's.
TEST(JobRunner, CpuMetricsTheGpuMayTakeAreNamedSo) {
    const std::vector<std::string> keys{"psnr", "cambi", "ciede2000"};
    EXPECT_EQ(cpu_metrics_text(keys, true), "PSNR; CAMBI and CIEDE2000 (\xCE\x94" "E) on the GPU where it can");
    EXPECT_EQ(cpu_metrics_text({"ciede2000"}, true), "CIEDE2000 (\xCE\x94" "E) on the GPU where it can");
    EXPECT_EQ(cpu_metrics_text(keys, false), "PSNR, CAMBI, CIEDE2000 (\xCE\x94" "E)");
    EXPECT_EQ(cpu_metrics_text({"psnr", "ssim"}, true), "PSNR, SSIM");
    EXPECT_EQ(cpu_metrics_text({"psnr", "ms_ssim"}, true), "PSNR; MS-SSIM on the GPU where it can");
    const std::vector<std::string> chroma{"psnr", "vmaf_yuv_u", "vmaf_yuv_v"};
    EXPECT_EQ(cpu_metrics_text(chroma, true), "PSNR; VMAF YUV: U and VMAF YUV: V on the GPU where it can");
    EXPECT_EQ(cpu_metrics_text(chroma, may_run_on_gpu(chroma, true, false)), "PSNR, VMAF YUV: U, VMAF YUV: V");
    EXPECT_EQ(cpu_metrics_text({"psnr", "vmaf_yuv"}, true), "PSNR; VMAF YUV (6:1:1) on the GPU where it can");
}

}  // namespace
}  // namespace vml
