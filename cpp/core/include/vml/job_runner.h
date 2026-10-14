// Runs the metrics of a list of videos, as the Python app's job queue runs
// them (vmaf_app/core/job_runner.py). Terms used here, as there:
// - a job is one video to compare with its source (VmafJob);
// - a task is the part of a video's metrics that one program calculates in
//   one place: FFmpeg's metrics on the CPU (PSNR, SSIM and XPSNR in the app
//   among them), VMAF on the GPU, Vship's metrics (SSIMULACRA2,
//   Butteraugli, CVVDP) on the GPU in one or more passes, or SSIMULACRA2
//   and Butteraugli set to the CPU, with libjxl's tools;
// - a lane is a thread that runs one task at a time. The CPU's queue has two
//   lanes (one if it holds a single task), and the "parallel jobs" setting
//   says how many of them may work at once. The GPU's queue has one lane.
//
// run_jobs takes the list of videos and runs each video's tasks in the CPU's
// and the GPU's queues, in list order, and reports what happens through
// RunEvents. A JobScheduler does the same, and takes a change of the
// parallel jobs while it runs. A run is cancelled, paused and resumed
// through its CancelToken.
#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/ffmpeg_run.h"
#include "vml/options.h"
#include "vml/results.h"
#include "vml/video.h"
#include "vml/vship_run.h"

namespace vml {

// At most two videos' CPU metrics at once, because a third gains nothing
// (the Python app's MAX_PARALLEL_JOBS, measured there).
inline constexpr int max_parallel_jobs = 2;

// The task of VMAF, NEG and VMAF v1 calculated on the GPU (plan_tasks); the
// other one is FFmpeg's, "ffmpeg".
inline constexpr std::string_view gpu_vmaf_task = "vmaf_gpu";

// The part of a video's metrics that one program calculates in one place.
struct MetricTask {
    // "ffmpeg", gpu_vmaf_task, perceptual_task or perceptual_cpu_task
    std::string backend;
    std::vector<std::string> metric_keys;  // in the registry's order

    // Whether it waits in the GPU's queue rather than the CPU's.
    bool on_gpu() const { return backend == gpu_vmaf_task || backend == perceptual_task; }

    // Whether it calculates SSIMULACRA2, Butteraugli or CVVDP.
    bool perceptual() const { return backend == perceptual_task || backend == perceptual_cpu_task; }
};

// Where SSIMULACRA2 and Butteraugli are each set to run, "gpu" or "cpu"
// ("gpu" where not given).
using PerceptualBackends = std::map<std::string, std::string, std::less<>>;

// The tasks a video's metrics (`metric_keys`) are calculated in (the Python
// app's JobRun._by_place), each in one place: FFmpeg's metrics in one task
// on the CPU, less VMAF, NEG and VMAF v1 where the GPU scores them
// (scores_on_gpu), which make a task of their own on the GPU (in the same
// FFmpeg as PSNR, SSIM and XPSNR they would go at those metrics' pace); and
// SSIMULACRA2, Butteraugli and CVVDP in Vship's task on the GPU, less
// SSIMULACRA2 and Butteraugli set to the CPU (`perceptual_backends`), which
// make a task of their own on the CPU. `options.model` is the run's model=
// value; `options.model_v1` is empty while VMAF v1's Auto is not resolved.
//
// Metrics with a saved score the run keeps (`saved`) are left out as the
// Python app's planner leaves them out (build_execution_plan, _by_place):
// SSIMULACRA2, Butteraugli and CVVDP each on its own; FFmpeg's together, as
// they are calculated together, except that a part of them that runs in a
// place of its own (VMAF on the GPU, or what stays on the CPU) is left out
// when all of that part is saved.
std::vector<MetricTask> plan_tasks(const VideoInfo& source, const VideoInfo& test, const VmafOptions& options,
                                   const std::vector<std::string>& metric_keys,
                                   const PerceptualBackends& perceptual_backends = {},
                                   const MetricResultSet& saved = MetricResultSet());

// One video to compare with its source, with what to calculate.
struct VmafJob {
    VideoInfo source_info;
    VideoInfo distorted_info;
    VmafOptions options;
    // Replaces the result's `distorted` path; none keeps distorted_info's.
    std::optional<std::filesystem::path> result_distorted_path;
    std::vector<std::string> metric_keys;  // the metrics asked for
    std::vector<MetricTask> tasks;         // which calculate them (plan_tasks)
    PerceptualBackends perceptual_backends;
    CvvdpSettings cvvdp;  // the display CVVDP models
    // The saved result, where the run keeps some of its scores, and those
    // scores (the Python app's cached_result and cached_metrics): they fill
    // in what the run does not calculate, and a new score replaces a saved
    // one. With FFmpeg's metrics all saved, the saved result is the base.
    std::optional<ComparisonResult> cached_result;
    MetricResultSet cached_metrics;
};

// The state of one of a video's tasks, for its run line (run_line.h).
struct TaskSnapshot {
    std::string backend;
    std::vector<std::string> metric_keys;
    std::string lane;  // "cpu" or "gpu": the queue it waits in
    // The metrics the CPU is calculating since VMAF on the GPU failed; the
    // progress figures are then theirs.
    std::vector<std::string> cpu_keys;
    std::vector<std::string> done_keys;  // the metrics it has scores for
    // The metrics of each of its passes, in order: Vship's as planned, then
    // as each starts, with any retry after the others; any other task's
    // metrics as one pass.
    std::vector<std::vector<std::string>> passes;
    // The pass under way: its number, how many there are, and the task's
    // progress count when it started, from which the pass's own is counted.
    std::optional<std::array<long long, 3>> phase;
    long long current = 0;
    long long total = 0;
    double fps = 0.0;
    std::string state;        // "done", "failed", "waiting", "running" or "starting"
    std::string waiting_for;  // what a waiting task waits for: "GPU" or "CPU"
    RunStatus step;           // its latest status
    // The last decode plan its statuses named: where it decodes each video,
    // for the window's "Decoder: ...".
    std::optional<HardwareDecoding> decode;
};

// The metrics that failed in a video whose other metrics finished: each
// metric's key with why it failed.
using MetricFailures = std::vector<std::pair<std::string, std::string>>;

// The events a run reports, each on the thread of the lane it happens on.
// The index is the job's in the list. An event left empty is not reported.
struct RunEvents {
    std::function<void(std::size_t index)> job_started;
    // The state of each of the video's tasks (TaskSnapshot), in its tasks'
    // order, whenever one changes.
    std::function<void(std::size_t index, const std::vector<TaskSnapshot>& tasks)> task_progress;
    std::function<void(std::size_t index, const ComparisonResult& result)> job_finished;
    // Why the video failed, and the end of the tool's error output.
    std::function<void(std::size_t index, const std::string& message, const std::string& stderr_tail)> job_failed;
    // Some of the video's tasks failed and the others finished: the finished
    // metrics are the result.
    std::function<void(std::size_t index, const ComparisonResult& result, const std::string& message,
                       const std::string& stderr_tail, const MetricFailures& failures)>
        job_partially_failed;
    // The video's result so far, sent when one of its tasks finishes while
    // another still runs, so that it is kept if the rest is lost.
    std::function<void(std::size_t index, const ComparisonResult& result)> result_updated;
    // The run was cancelled. Sent once, however many videos it stopped.
    std::function<void()> cancelled;
};

// Calculates one task: `options` are the job's, with only the task's
// metrics turned on. run_vmaf in the app; a stand-in in the tests.
using TaskRunner = std::function<ComparisonResult(const VmafJob& job, const VmafOptions& options,
                                                  const ProgressCallback& on_progress,
                                                  const StatusCallback& on_status, const CancelToken& cancel)>;

// Calculates one perceptual task's metrics (`keys`) for a video, reporting
// through `hooks`, the GPU's metrics in one pass when `together`, with
// `concurrent_cpu_tasks` the number of tasks the CPU's queue may run at once
// (which the CPU's perceptual tools share the cores and memory with):
// run_perceptual_task in the app; a stand-in in the tests.
using PerceptualRunner = std::function<PerceptualTaskOutput(const VmafJob& job, const std::vector<std::string>& keys,
                                                            bool together, int concurrent_cpu_tasks,
                                                            const PerceptualHooks& hooks, const CancelToken& cancel)>;

// Runs every job's tasks and returns once all are done or the run is
// cancelled: the CPU's tasks on up to `parallel_jobs` lanes at once
// (max_parallel_jobs at most), the GPU's on a lane of its own, each queue in
// list order, with at most max_parallel_jobs + 1 videos in progress at once.
// "Auto" libvmaf threads become an equal share of `cores` for each CPU lane.
// A failed task leaves the video's other tasks to finish. `runner`
// calculates FFmpeg's and VMAF's tasks, `perceptual` the perceptual ones,
// the GPU's in one pass when `gpu_metrics_together` (Settings > GPU
// metrics).
void run_jobs(const std::vector<VmafJob>& jobs, int parallel_jobs, const RunEvents& events, const TaskRunner& runner,
              const PerceptualRunner& perceptual, bool gpu_metrics_together, unsigned cores,
              const CancelToken& cancel);

// run_jobs for jobs without perceptual tasks.
void run_jobs(const std::vector<VmafJob>& jobs, int parallel_jobs, const RunEvents& events, const TaskRunner& runner,
              unsigned cores, const CancelToken& cancel);

// A run of run_jobs (the Python app's JobScheduler) whose parallel jobs can
// change while it runs: a list of feature-length videos is just when someone
// notices that half of their CPU is idle, and starting the list again to use
// it would waste the work done.
class JobScheduler {
public:
    JobScheduler(std::vector<VmafJob> jobs, int parallel_jobs, RunEvents events, TaskRunner runner,
                 PerceptualRunner perceptual, bool gpu_metrics_together, unsigned cores, CancelToken cancel);
    ~JobScheduler();
    JobScheduler(const JobScheduler&) = delete;
    JobScheduler& operator=(const JobScheduler&) = delete;

    // Runs every job's tasks, as run_jobs does, once.
    void run();

    // How many videos' CPU metrics may run at once (max_parallel_jobs at
    // most), from any thread. Raising it lets a waiting lane start the next
    // video at once; lowering it stops nothing already started, and keeps
    // more from starting until enough have finished. A video started before
    // keeps the libvmaf threads it started with.
    void set_parallel_jobs(int count);
    int parallel_jobs() const;

private:
    struct Parts;
    std::unique_ptr<Parts> parts_;
};

}  // namespace vml
