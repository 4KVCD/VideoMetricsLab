#include "vml/job_runner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include "vml/geometry.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
#include "vml/metrics.h"

namespace vml {

namespace {

// Videos in progress at once: one per CPU lane, plus the one on the GPU. A
// video's CPU and GPU tasks run in separate queues, so one queue can run
// ahead of the other, but not without limit: each video in progress has a
// line of its own, and the list is still worked through in order.
constexpr std::size_t max_videos_in_flight = max_parallel_jobs + 1;

bool contains(const std::vector<std::string>& keys, std::string_view key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

// `result` with only every `step`-th frame (0, step, 2*step...): the frames
// a subsampled request covers, as libvmaf's n_subsample scores them.
FrameMetricResult sampled_every(const FrameMetricResult& result, int step) {
    FrameMetricResult kept{result.key, {}, {}, {}, result.provenance};
    for (std::size_t i = 0; i < result.frame.size(); ++i) {
        if (result.frame[i] % step == 0) {
            kept.frame.push_back(result.frame[i]);
            kept.time.push_back(result.time[i]);
            kept.values.push_back(result.values[i]);
        }
    }
    return kept;
}

// A task's result with each metric on the frames the video's request covers
// (the Python app's as_requested). FFmpeg's metrics cover every
// n_subsample-th frame when libvmaf calculates any of them, and XPSNR alone
// covers every frame. XPSNR in one FFmpeg run with libvmaf's metrics is
// reported for the frames libvmaf's log has. Scored apart from them, beside
// VMAF on the GPU, it comes back for every frame, and kept that way it would
// fall off the shared frames, out of the table and the CSV.
void conform(ComparisonResult& result, const VmafJob& job) {
    const bool libvmaf = std::any_of(job.metric_keys.begin(), job.metric_keys.end(), [](const std::string& key) {
        return key != "xpsnr" && metric_definition(key).ffmpeg_binding;
    });
    const int step = libvmaf ? job.options.n_subsample : 1;
    if (step <= 1) {
        return;
    }
    MetricResultSet conformed;
    for (const std::string& key : result.metric_results.keys()) {
        if (const FrameMetricResult* frame = result.metric_results.frame(key)) {
            conformed.add(sampled_every(*frame, step));
        } else {
            conformed.add(*result.metric_results.sequence(key));
        }
    }
    result.metric_results = std::move(conformed);
}

// Why a task failed, and the end of the tool's error output.
std::pair<std::string, std::string> failure_of(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const VmafRunError& failure) {
        return {failure.what(), failure.stderr_tail()};
    } catch (const std::exception& failure) {
        return {failure.what(), ""};
    } catch (...) {
        return {"an unknown error", ""};
    }
}

bool is_cancel(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const Cancelled&) {
        return true;
    } catch (...) {
        return false;
    }
}

enum Pool { cpu_pool, gpu_pool };

class Scheduler;

// One video: its tasks, which the scheduler's CPU and GPU lanes run, their
// progress, and the result they make together (the Python app's JobRun).
class JobRun {
public:
    JobRun(Scheduler& scheduler, std::size_t index, const VmafJob& job);

    std::size_t index() const { return index_; }
    std::size_t task_count() const { return tasks_.size(); }
    Pool pool_of(std::size_t task) const { return job_.tasks[task].on_gpu() ? gpu_pool : cpu_pool; }

    // Starts the video when a lane takes its first task: decides its share
    // of the cores and sends job_started. False when the run was cancelled
    // first. Only the first call does anything; later calls return its
    // answer.
    bool begin();

    // Runs one task. True when it was the video's last task to end.
    bool run_task(std::size_t task);

    // Reports the video's result once all of its tasks have ended, or why it
    // failed. In a cancelled run, the tasks that had finished are still its
    // result: a video's GPU metrics are often done long before its CPU
    // metrics, and cancelling must not throw them away.
    void finish();

    bool finalized() const;
    bool has_results() const;

    // Changed under the scheduler's lock: whether the video is in progress,
    // and which of its tasks a lane has taken.
    bool started = false;
    std::vector<bool> admitted;

private:
    // How far a task is: frames done, frames expected, frames per second.
    struct Counts {
        long long current = 0;
        long long total = 0;
        double fps = 0.0;
    };

    // One task as it runs.
    struct TaskState {
        std::optional<ComparisonResult> result;  // once it has finished
        std::exception_ptr error;                // once it has failed or was cancelled
        std::optional<Counts> progress;
        // The task's metrics the CPU is calculating since the GPU failed;
        // its progress figures are then theirs.
        std::vector<std::string> cpu_keys;
        std::string waiting_for;  // "GPU" or "CPU" while it waits
        RunStatus step;           // its latest status
    };

    ComparisonResult execute_task(std::size_t task);
    void report_progress(std::size_t task, long long current, long long total, double fps);
    void report_status(std::size_t task, const RunStatus& status);
    // The rest are called with mutex_ held.
    std::vector<TaskSnapshot> snapshots() const;
    void send_snapshots() const;
    std::optional<ComparisonResult> merged() const;
    void send_result_so_far() const;

    Scheduler& scheduler_;
    const std::size_t index_;
    const VmafJob& job_;
    VmafOptions options_;  // the job's, with its share of the cores (set by begin)
    // Guards what follows. The tasks report from their own threads, and the
    // events are sent with it held: a snapshot built first but sent last
    // would leave the line showing an older state, and a result so far sent
    // after the finished result would replace it.
    mutable std::mutex mutex_;
    std::vector<TaskState> tasks_;
    std::vector<std::size_t> failed_;  // the tasks that failed, in the order they did
    std::size_t finished_tasks_ = 0;
    bool begun_ = false;
    bool claimed_ = false;    // begin's answer
    bool finalized_ = false;  // whether its result or failure has been sent
};

// Runs a list of videos' tasks in the CPU's and the GPU's queues (the
// Python app's JobScheduler).
class Scheduler {
public:
    Scheduler(const std::vector<VmafJob>& jobs, int parallel_jobs, const RunEvents& events, const TaskRunner& runner,
              unsigned cores, const CancelToken& cancel)
        : events(events),
          runner(runner),
          cancel(cancel),
          jobs_(jobs),
          parallel_jobs_(std::clamp(parallel_jobs, 1, max_parallel_jobs)),
          cores_(cores) {}

    void run();

    // "Auto" libvmaf threads replaced with a video's share of the CPU, as it
    // starts: a single video runs alone whatever the setting says, so it
    // keeps every core. A thread count the user chose is kept.
    VmafOptions share_cores(VmafOptions options) const {
        const int concurrent = std::min(parallel_jobs_, lane_count_);
        if (options.n_threads <= 0 && concurrent > 1) {
            options.n_threads = static_cast<int>(std::max(1u, cores_ / static_cast<unsigned>(concurrent)));
        }
        return options;
    }

    // Which of a video's tasks lanes have taken.
    std::vector<bool> admitted(const JobRun& run) {
        const std::lock_guard lock(mutex_);
        return run.admitted;
    }

    // Sends `cancelled` once: it means the run stopped, and when several
    // videos are running they all stop together.
    void report_cancelled_once() {
        if (!cancellation_reported_.exchange(true) && events.cancelled) {
            events.cancelled();
        }
    }

    const RunEvents& events;
    const TaskRunner& runner;
    const CancelToken& cancel;

private:
    using Picked = std::pair<JobRun*, std::size_t>;  // a video, and one of its tasks

    std::optional<Picked> next_task(Pool pool);
    std::optional<Picked> take_next(Pool pool);
    void drain_queue(Pool pool);

    const std::vector<VmafJob>& jobs_;
    const int parallel_jobs_;
    const unsigned cores_;
    // How many CPU lanes the run has: at least one, and never more than
    // there are CPU tasks.
    int lane_count_ = 1;
    std::vector<std::unique_ptr<JobRun>> runs_;
    std::atomic<bool> cancellation_reported_{false};
    // Guards what follows; `room_` is signalled whenever a lane may find
    // room for a task.
    std::mutex mutex_;
    std::condition_variable room_;
    std::array<std::deque<Picked>, 2> queues_;
    std::array<int, 2> busy_{0, 0};
    std::set<std::size_t> in_flight_;
    // What ended a lane, to be thrown by run(). The other lanes then start
    // no more tasks: the videos that lane left unfinished would hold their
    // places in progress for good.
    std::exception_ptr lane_error_;
};

// ------------------------------------------------------------------- JobRun

JobRun::JobRun(Scheduler& scheduler, std::size_t index, const VmafJob& job)
    : admitted(job.tasks.size(), false),
      scheduler_(scheduler),
      index_(index),
      job_(job),
      options_(job.options),
      tasks_(job.tasks.size()) {}

bool JobRun::begin() {
    const std::lock_guard lock(mutex_);
    if (begun_) {
        return claimed_;
    }
    begun_ = true;
    claimed_ = !scheduler_.cancel.cancelled();
    if (!claimed_) {
        return false;
    }
    options_ = scheduler_.share_cores(job_.options);
    if (scheduler_.events.job_started) {
        scheduler_.events.job_started(index_);
    }
    if (tasks_.size() > 1) {
        const std::vector<bool> taken = scheduler_.admitted(*this);
        for (std::size_t task = 0; task < tasks_.size(); ++task) {
            if (!taken[task]) {
                tasks_[task].waiting_for = pool_of(task) == gpu_pool ? "GPU" : "CPU";
            }
        }
        send_snapshots();
    }
    return true;
}

bool JobRun::finalized() const {
    const std::lock_guard lock(mutex_);
    return finalized_;
}

bool JobRun::has_results() const {
    const std::lock_guard lock(mutex_);
    return std::any_of(tasks_.begin(), tasks_.end(), [](const TaskState& state) { return state.result.has_value(); });
}

ComparisonResult JobRun::execute_task(std::size_t task) {
    VmafOptions options = options_;
    for (const std::string& key : options_.requested_metrics()) {
        options.set_metric_enabled(key, contains(job_.tasks[task].metric_keys, key));
    }
    ComparisonResult result = scheduler_.runner(
        job_, options,
        [this, task](long long current, long long total, double fps) { report_progress(task, current, total, fps); },
        [this, task](const RunStatus& status) { report_status(task, status); }, scheduler_.cancel);
    conform(result, job_);
    return result;
}

// The scorers report at most a few times a second, so every report is
// passed on (the Python app's, reporting every few frames, are thinned).
void JobRun::report_progress(std::size_t task, long long current, long long total, double fps) {
    const std::lock_guard lock(mutex_);
    tasks_[task].waiting_for.clear();
    tasks_[task].progress = Counts{current, total, fps};
    send_snapshots();
}

void JobRun::report_status(std::size_t task, const RunStatus& status) {
    const std::lock_guard lock(mutex_);
    TaskState& state = tasks_[task];
    state.step = status;
    if (tasks_.size() > 1 && status.kind == StatusKind::gpu_wait) {
        state.waiting_for = "GPU";
    }
    if (pool_of(task) == gpu_pool && status.kind == StatusKind::gpu_vmaf_failed) {
        // FFmpeg's libvmaf takes over on the CPU, from frame 0.
        state.cpu_keys = job_.tasks[task].metric_keys;
        state.progress.reset();
    }
    send_snapshots();
}

bool JobRun::run_task(std::size_t task) {
    {
        const std::lock_guard lock(mutex_);
        tasks_[task].waiting_for.clear();
    }
    std::optional<ComparisonResult> output;
    std::exception_ptr error;
    try {
        output = execute_task(task);
    } catch (...) {
        // The video's other tasks are left to finish, so that one failing
        // does not lose the others' metrics too.
        error = std::current_exception();
    }
    const std::lock_guard lock(mutex_);
    TaskState& state = tasks_[task];
    if (error) {
        state.error = error;
        failed_.push_back(task);
    } else {
        state.result = std::move(output);
        state.waiting_for.clear();
        if (!state.progress) {
            state.progress = Counts{1, 1, 0.0};
        }
    }
    if (tasks_.size() > 1) {
        send_snapshots();  // or the task's part of the line would keep showing its last step
    }
    const bool last = ++finished_tasks_ == tasks_.size();
    if (state.result && !last) {
        // Another task still runs: this one's scores are kept now, not when
        // the whole video is done.
        send_result_so_far();
    }
    return last;
}

void JobRun::finish() {
    const std::lock_guard lock(mutex_);
    finalized_ = true;
    const RunEvents& events = scheduler_.events;
    const bool cancelled = scheduler_.cancel.cancelled();
    std::vector<std::size_t> failed;
    for (const std::size_t task : failed_) {
        if (!cancelled || !is_cancel(tasks_[task].error)) {
            failed.push_back(task);
        }
    }
    const std::optional<ComparisonResult> result = merged();
    // The tasks' own results are not needed once the video's is made: a
    // long list would otherwise hold every video's scores twice.
    for (TaskState& state : tasks_) {
        state.result.reset();
    }
    if (!result) {
        if (cancelled || (!failed.empty() && is_cancel(tasks_[failed.front()].error))) {
            scheduler_.report_cancelled_once();
        } else if (events.job_failed) {
            const auto [message, tail] = failed.empty() ? std::pair<std::string, std::string>(
                                                              "No executable metric task was planned.", "")
                                                        : failure_of(tasks_[failed.front()].error);
            events.job_failed(index_, message, tail);
        }
        return;
    }
    if (failed.empty()) {
        if (events.job_finished) {
            events.job_finished(index_, *result);
        }
        return;
    }
    std::string message;
    std::string stderr_tail;
    MetricFailures failures;
    for (const std::size_t task : failed) {
        const auto [why, tail] = failure_of(tasks_[task].error);
        std::string labels;
        for (const std::string& key : job_.tasks[task].metric_keys) {
            labels += (labels.empty() ? "" : ", ") + std::string(metric_definition(key).label);
            failures.emplace_back(key, why);
        }
        message += (message.empty() ? "" : "\n") + labels + " failed: " + why;
        if (stderr_tail.empty()) {
            stderr_tail = tail;
        }
    }
    if (events.job_partially_failed) {
        events.job_partially_failed(index_, *result, message, stderr_tail, failures);
    }
}

std::vector<TaskSnapshot> JobRun::snapshots() const {
    std::vector<TaskSnapshot> found;
    for (std::size_t task = 0; task < tasks_.size(); ++task) {
        const MetricTask& planned = job_.tasks[task];
        const TaskState& state = tasks_[task];
        TaskSnapshot& snapshot = found.emplace_back();
        snapshot.backend = planned.backend;
        snapshot.metric_keys = planned.metric_keys;
        snapshot.lane = pool_of(task) == gpu_pool ? "gpu" : "cpu";
        snapshot.cpu_keys = state.cpu_keys;
        if (state.result) {
            for (const std::string& key : planned.metric_keys) {
                if (state.result->has_metric(key)) {
                    snapshot.done_keys.push_back(key);
                }
            }
        }
        if (state.progress) {
            snapshot.current = state.progress->current;
            snapshot.total = state.progress->total;
            snapshot.fps = state.progress->fps;
        }
        snapshot.state = state.result                 ? "done"
                         : state.error                ? "failed"
                         : !state.waiting_for.empty() ? "waiting"
                         : state.progress             ? "running"
                                                      : "starting";
        snapshot.waiting_for = state.waiting_for;
        snapshot.step = state.step;
    }
    return found;
}

void JobRun::send_snapshots() const {
    if (scheduler_.events.task_progress) {
        scheduler_.events.task_progress(index_, snapshots());
    }
}

// The video's result from its finished tasks (the Python app's
// JobRun._merged): FFmpeg's, with VMAF, NEG and VMAF v1 from the GPU's task
// beside it. Nothing until a task has finished.
std::optional<ComparisonResult> JobRun::merged() const {
    const ComparisonResult* ffmpeg = nullptr;
    const ComparisonResult* gpu = nullptr;
    for (std::size_t task = 0; task < tasks_.size(); ++task) {
        if (tasks_[task].result) {
            (pool_of(task) == gpu_pool ? gpu : ffmpeg) = &*tasks_[task].result;
        }
    }
    if (ffmpeg == nullptr) {
        return gpu == nullptr ? std::nullopt : std::optional(*gpu);
    }
    ComparisonResult result = *ffmpeg;
    if (gpu != nullptr) {
        for (const std::string& key : gpu->metric_results.keys()) {
            if (const FrameMetricResult* frame = gpu->metric_results.frame(key)) {
                result.metric_results.add(*frame);
            } else {
                result.metric_results.add(*gpu->metric_results.sequence(key));
            }
        }
        // Each VMAF's model comes from the task that calculated it: VMAF with
        // a custom model file is FFmpeg's.
        if (gpu->has_metric("vmaf")) {
            result.model = gpu->model;
        }
        if (gpu->has_metric("vmaf_v1")) {
            result.model_v1 = gpu->model_v1;
            result.model_choice_v1 = gpu->model_choice_v1;
        }
    }
    return result;
}

void JobRun::send_result_so_far() const {
    if (finalized_ || !scheduler_.events.result_updated) {
        return;
    }
    if (const std::optional<ComparisonResult> result = merged()) {
        scheduler_.events.result_updated(index_, *result);
    }
}

// ---------------------------------------------------------------- Scheduler

void Scheduler::run() {
    for (std::size_t index = 0; index < jobs_.size(); ++index) {
        runs_.push_back(std::make_unique<JobRun>(*this, index, jobs_[index]));
    }
    for (const std::unique_ptr<JobRun>& run : runs_) {
        if (run->task_count() == 0) {
            if (run->begin()) {
                run->finish();  // nothing to calculate: says so
            }
            continue;
        }
        for (std::size_t task = 0; task < run->task_count(); ++task) {
            queues_[run->pool_of(task)].emplace_back(run.get(), task);
        }
    }
    // The GPU has one lane. The last lane runs on this thread.
    const int cpu_lanes = std::min(max_parallel_jobs, static_cast<int>(queues_[cpu_pool].size()));
    lane_count_ = std::max(1, cpu_lanes);
    std::vector<Pool> lanes(static_cast<std::size_t>(cpu_lanes), cpu_pool);
    if (!queues_[gpu_pool].empty()) {
        lanes.push_back(gpu_pool);
    }
    std::vector<std::thread> threads;
    for (std::size_t lane = 0; lane + 1 < lanes.size(); ++lane) {
        threads.emplace_back([this, pool = lanes[lane]] { drain_queue(pool); });
    }
    if (!lanes.empty()) {
        drain_queue(lanes.back());
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (cancel.cancelled()) {
        // A video cancelled before all of its tasks started was not finished
        // by any lane, but the tasks it did finish are still its result.
        for (const std::unique_ptr<JobRun>& run : runs_) {
            if (run->started && !run->finalized() && run->has_results()) {
                run->finish();
            }
        }
        report_cancelled_once();
    }
    if (lane_error_) {
        std::rethrow_exception(lane_error_);
    }
}

// The next task this queue may run, in list order, waiting until there is
// room for it; nothing once the queue is empty, the run is cancelled or a
// lane has ended with an error.
std::optional<Scheduler::Picked> Scheduler::next_task(Pool pool) {
    std::unique_lock lock(mutex_);
    while (true) {
        if (cancel.cancelled() || lane_error_ || queues_[pool].empty()) {
            return std::nullopt;
        }
        if (const std::optional<Picked> picked = take_next(pool)) {
            return picked;
        }
        // Whatever makes room signals; the wait's end checks for a Cancel.
        room_.wait_for(lock, std::chrono::milliseconds(100));
    }
}

// Takes the task this queue may start now off it, or nothing when there is
// no room for one. Called with mutex_ held.
std::optional<Scheduler::Picked> Scheduler::take_next(Pool pool) {
    if (busy_[pool] >= (pool == cpu_pool ? parallel_jobs_ : 1)) {
        return std::nullopt;
    }
    // The first task whose video is in progress, or that may start a new
    // one. A later video's task passes an earlier one's only when the
    // earlier video cannot start yet, so the videos in progress can always
    // finish.
    std::deque<Picked>& queue = queues_[pool];
    for (auto entry = queue.begin(); entry != queue.end(); ++entry) {
        JobRun* run = entry->first;
        if (run->started || in_flight_.size() < max_videos_in_flight) {
            const Picked picked = *entry;
            queue.erase(entry);
            ++busy_[pool];
            run->admitted[picked.second] = true;
            if (!run->started) {
                run->started = true;
                in_flight_.insert(run->index());
            }
            return picked;
        }
    }
    return std::nullopt;
}

// One lane: runs this queue's tasks one at a time until the queue is empty
// or the run is cancelled, and finishes each video whose last task it ran.
void Scheduler::drain_queue(Pool pool) {
    while (const std::optional<Picked> picked = next_task(pool)) {
        const auto [run, task] = *picked;
        bool complete = false;
        std::exception_ptr error;
        try {
            complete = run->begin() && run->run_task(task);
            if (complete) {
                run->finish();
            }
        } catch (...) {  // only from an event's receiver: run_task keeps the task's own errors
            error = std::current_exception();
        }
        const std::lock_guard lock(mutex_);
        --busy_[pool];
        if (complete) {
            in_flight_.erase(run->index());
        }
        room_.notify_all();
        if (error) {
            if (!lane_error_) {
                lane_error_ = error;
            }
            return;
        }
    }
}

}  // namespace

std::vector<MetricTask> plan_tasks(const VideoInfo& source, const VideoInfo& test, const VmafOptions& options,
                                   const std::vector<std::string>& metric_keys) {
    const bool vmaf = contains(metric_keys, "vmaf");
    const bool neg = contains(metric_keys, "vmaf_neg");
    GpuModels on_gpu;
    if (vmaf || neg) {
        // The size frames are compared at, where it is known before the run:
        // with black bars kept. Pictures with their black bars cut are
        // always even-sized.
        std::optional<std::pair<int, int>> size;
        if (options.crop_mode == CropMode::none) {
            size = analysis_dimensions(source, test, options);
        }
        on_gpu = scores_on_gpu(vmaf, neg, options.model, options.vmaf_on_gpu,
                               bit_depth(analysis_pix_fmt({source.pix_fmt, test.pix_fmt})), size);
    }
    MetricTask ffmpeg{"ffmpeg", {}};
    MetricTask gpu{std::string(gpu_vmaf_task), {}};
    for (const std::string& key : metric_keys) {
        const bool scored_on_gpu = std::any_of(on_gpu.begin(), on_gpu.end(),
                                               [&key](const auto& model) { return model.first == key; });
        (scored_on_gpu ? gpu : ffmpeg).metric_keys.push_back(key);
    }
    std::vector<MetricTask> tasks;
    for (MetricTask* task : {&ffmpeg, &gpu}) {
        if (!task->metric_keys.empty()) {
            tasks.push_back(std::move(*task));
        }
    }
    return tasks;
}

void run_jobs(const std::vector<VmafJob>& jobs, int parallel_jobs, const RunEvents& events, const TaskRunner& runner,
              unsigned cores, const CancelToken& cancel) {
    Scheduler(jobs, parallel_jobs, events, runner, cores, cancel).run();
}

}  // namespace vml
