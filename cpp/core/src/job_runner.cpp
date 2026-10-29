#include "vml/job_runner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include "vml/app_log.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/result_json.h"
#include "vml/time_format.h"
#include "vml/vship.h"

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
    } catch (const PerceptualRunError& failure) {
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

const app_log::Logger logger("vml.job_runner");

// A video's format, for the log: "3840x2160 23.976 fps vvc yuv420p10le HDR
// PQ, 1:45:36".
std::string media_text(const VideoInfo& info) {
    const std::string hdr = info.color_transfer == "smpte2084"      ? " HDR PQ"
                            : info.color_transfer == "arib-std-b67" ? " HDR HLG"
                                                                    : "";
    return std::to_string(info.width) + "x" + std::to_string(info.height) + " " + format_fixed(info.fps, 3) +
           " fps " + info.codec_name + " " + info.pix_fmt + hdr + ", " + format_hms(info.duration);
}

// The end of the tool's error output, for the log.
std::string stderr_note(const std::string& tail) { return tail.empty() ? "" : "\nLast output:\n" + tail; }

// How long since `started`, for the log: "0:01:05".
std::string took_since(std::chrono::steady_clock::time_point started) {
    return format_hms(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
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

    // The video's name in the log: "Video 2 'label'".
    std::string name() const { return "Video " + std::to_string(index_ + 1) + " '" + job_.label + "'"; }
    // The video's files, what will be calculated and how, for the log.
    std::string describe() const;

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
        // Once it has finished: FFmpeg's or VMAF's result, or a perceptual
        // task's output.
        std::optional<ComparisonResult> result;
        std::optional<PerceptualTaskOutput> perceptual;
        std::exception_ptr error;  // once it has failed or was cancelled
        std::optional<Counts> progress;
        // The task's metrics the CPU is calculating since the GPU failed;
        // its progress figures are then theirs.
        std::vector<std::string> cpu_keys;
        // The metrics of each of its passes (Vship's: planned at the start,
        // then updated as each starts), and the pass under way: its number,
        // how many there are, and the progress count it started at.
        std::optional<std::vector<std::vector<std::string>>> passes;
        std::optional<std::array<long long, 3>> phase;
        std::string waiting_for;  // "GPU" or "CPU" while it waits
        RunStatus step;           // its latest status
        std::optional<HardwareDecoding> decode;  // the last decode plan a status named

        bool finished() const { return result.has_value() || perceptual.has_value(); }
    };

    // A task's name in the log: "GPU metrics (SSIMULACRA2, CVVDP)".
    std::string task_name(std::size_t task) const;
    ComparisonResult execute_task(std::size_t task);
    PerceptualTaskOutput execute_perceptual(std::size_t task);
    void report_progress(std::size_t task, long long current, long long total, double fps);
    void report_status(std::size_t task, const RunStatus& status);
    // Keeps the scores of a GPU pass that finished while its task still runs,
    // and sends the video's result so far.
    void report_pass(const PerceptualTaskOutput& output);
    // Records a GPU pass as it starts: the metrics it scores, and the task's
    // progress count then, from which the pass's own is counted.
    void report_pass_start(std::size_t task, int number, int count, const std::vector<std::string>& keys);
    // From now on the task's progress figures are the CPU's, for `keys`.
    void report_cpu(std::size_t task, const std::vector<std::string>& keys);
    // The rest are called with mutex_ held.
    std::vector<TaskSnapshot> snapshots() const;
    void send_snapshots() const;
    std::optional<PerceptualTaskOutput> perceptual_so_far() const;
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
    // The scores of the GPU passes of Vship's task that have finished while
    // the task still runs.
    std::vector<PerceptualTaskOutput> pass_outputs_;
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
              const PerceptualRunner& perceptual, bool gpu_metrics_together, unsigned cores,
              const CancelToken& cancel)
        : events(events),
          runner(runner),
          perceptual_runner(perceptual),
          gpu_metrics_together(gpu_metrics_together),
          cancel(cancel),
          jobs_(jobs),
          parallel_jobs_(std::clamp(parallel_jobs, 1, max_parallel_jobs)),
          cores_(cores) {}

    void run();

    // "Auto" libvmaf threads replaced with a video's share of the CPU, as it
    // starts: a single video runs alone whatever the setting says, so it
    // keeps every core. A thread count the user chose is kept.
    VmafOptions share_cores(VmafOptions options) const {
        const int concurrent = concurrent_cpu_tasks();
        if (options.n_threads <= 0 && concurrent > 1) {
            options.n_threads = static_cast<int>(std::max(1u, cores_ / static_cast<unsigned>(concurrent)));
        }
        return options;
    }

    // How many tasks the CPU's queue may run at once. The CPU's perceptual
    // tools, which score several frame pairs at a time, divide the cores and
    // memory by it, as share_cores divides libvmaf's threads.
    int concurrent_cpu_tasks() const { return std::max(1, std::min(parallel_jobs_.load(), lane_count_)); }

    // How many videos' CPU metrics may run at once, while the run goes on
    // (JobScheduler::set_parallel_jobs).
    void set_parallel_jobs(int count) {
        {
            const std::lock_guard lock(mutex_);
            parallel_jobs_ = std::clamp(count, 1, max_parallel_jobs);
        }
        room_.notify_all();  // a lane waiting for room may now have it
    }

    int parallel_jobs() const { return parallel_jobs_; }

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
    const PerceptualRunner& perceptual_runner;
    const bool gpu_metrics_together;
    const CancelToken& cancel;

private:
    using Picked = std::pair<JobRun*, std::size_t>;  // a video, and one of its tasks

    std::optional<Picked> next_task(Pool pool);
    std::optional<Picked> take_next(Pool pool);
    void drain_queue(Pool pool);

    const std::vector<VmafJob>& jobs_;
    std::atomic<int> parallel_jobs_;  // changed under mutex_, read anywhere
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
      tasks_(job.tasks.size()) {
    // Vship's passes come from the plan, and are updated as each starts
    // (report_pass_start), with any retry after the others.
    for (std::size_t task = 0; task < tasks_.size(); ++task) {
        if (job.tasks[task].backend == perceptual_task) {
            tasks_[task].passes =
                vship_passes(job.tasks[task].metric_keys, job.options.n_subsample, scheduler.gpu_metrics_together);
        }
    }
}

std::string JobRun::task_name(std::size_t task) const {
    return std::string(pool_of(task) == gpu_pool ? "GPU" : "CPU") + " metrics (" +
           metric_labels(job_.tasks[task].metric_keys) + ")";
}

std::string JobRun::describe() const {
    std::string calculating;
    for (std::size_t task = 0; task < tasks_.size(); ++task) {
        calculating += (calculating.empty() ? "" : "; ") + task_name(task);
    }
    if (calculating.empty()) {
        calculating = "nothing (all saved)";
    }
    std::string kept;
    if (job_.cached_result) {  // saved scores are kept only with their saved result, as merged() keeps them
        kept = metric_labels(job_.cached_metrics.keys());
    }
    const VmafOptions& options = job_.options;
    std::string settings = std::string("black bars ") + (options.crop_mode == CropMode::automatic ? "auto" : "none") +
                           ", duration limit " + format_general(options.duration_limit) + " s, resolution mismatch " +
                           scale_direction_value(options.scale_direction) + " (" + options.scale_algorithm +
                           "), VMAF v0.6.1 model " + options.model_choice + ", VMAF v1 model " +
                           options.model_choice_v1 + ", frame subsample " + std::to_string(options.n_subsample) +
                           ", GPU decode " + (options.gpu_decode ? "on" : "off");
    for (const auto& [key, backend] : job_.perceptual_backends) {
        settings += ", " + std::string(metric_definition(key).label) + " on " + ascii_upper(backend);
    }
    settings += ", Vship build " + std::string(vship_build_label(options.vship_build));
    if (const VshipBuild in_use = vship_build_in_use(options.vship_build); in_use != options.vship_build) {
        settings += " (" + std::string(vship_build_label(in_use)) + " in use)";
    }
    settings += ", CVVDP display " + job_.cvvdp.display.describe();
    std::string text = name() + ":\n  test   " + platform::path_text(job_.distorted_info.path) + " (" +
                       media_text(job_.distorted_info) + ")\n  source " + platform::path_text(job_.source_info.path) +
                       " (" + media_text(job_.source_info) + ")\n  calculating " + calculating +
                       (kept.empty() ? "" : "; saved scores kept: " + kept) + "\n  " + settings;
    if (options.resample_test) {
        text += "\n  resolution test: " + options.resample_test->label;
    }
    return text;
}

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
    logger.info(name() + ": started");
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
    return !pass_outputs_.empty() ||
           std::any_of(tasks_.begin(), tasks_.end(), [](const TaskState& state) { return state.finished(); });
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

PerceptualTaskOutput JobRun::execute_perceptual(std::size_t task) {
    if (!scheduler_.perceptual_runner) {
        throw std::logic_error("no runner for the perceptual metrics");
    }
    PerceptualHooks hooks;
    hooks.on_progress = [this, task](long long current, long long total, double fps) {
        report_progress(task, current, total, fps);
    };
    hooks.on_status = [this, task](const RunStatus& status) { report_status(task, status); };
    hooks.on_pass_done = [this](const PerceptualTaskOutput& output) { report_pass(output); };
    hooks.on_pass = [this, task](int number, int count, const std::vector<std::string>& keys) {
        report_pass_start(task, number, count, keys);
    };
    hooks.on_cpu = [this, task](const std::vector<std::string>& keys) { report_cpu(task, keys); };
    return scheduler_.perceptual_runner(job_, job_.tasks[task].metric_keys, scheduler_.gpu_metrics_together,
                                        scheduler_.concurrent_cpu_tasks(), hooks, scheduler_.cancel);
}

void JobRun::report_pass(const PerceptualTaskOutput& output) {
    const std::lock_guard lock(mutex_);
    pass_outputs_.push_back(output);
    send_result_so_far();
}

void JobRun::report_pass_start(std::size_t task, int number, int count, const std::vector<std::string>& keys) {
    const std::lock_guard lock(mutex_);
    TaskState& state = tasks_[task];
    if (!state.passes) {
        state.passes.emplace();
    }
    std::vector<std::vector<std::string>>& passes = *state.passes;
    if (passes.size() < static_cast<std::size_t>(number)) {
        passes.resize(static_cast<std::size_t>(number));
    }
    passes[static_cast<std::size_t>(number) - 1] = keys;
    passes.resize(std::min(passes.size(), static_cast<std::size_t>(std::max(count, number))));
    state.phase = std::array<long long, 3>{number, count, state.progress ? state.progress->current : 0};
    if (state.progress) {
        // The new pass has no speed yet: the last pass's would be shown as
        // its own, and used for its time left.
        state.progress->fps = 0.0;
    }
    send_snapshots();
}

void JobRun::report_cpu(std::size_t task, const std::vector<std::string>& keys) {
    const std::lock_guard lock(mutex_);
    TaskState& state = tasks_[task];
    state.cpu_keys = keys;
    state.phase.reset();
    state.progress.reset();
    send_snapshots();
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
    logger.info(name() + ": " + status.text);
    const std::lock_guard lock(mutex_);
    TaskState& state = tasks_[task];
    state.step = status;
    if (status.decoding) {
        state.decode = status.decoding;
    }
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
    const std::string half = task_name(task);
    logger.info(name() + ": " + half + " started");
    for (const std::string& note : job_.tasks[task].fallbacks) {
        report_status(task, {note, StatusKind::gpu_vmaf_failed});
    }
    const auto began = std::chrono::steady_clock::now();
    std::optional<ComparisonResult> output;
    std::optional<PerceptualTaskOutput> perceptual;
    std::exception_ptr error;
    try {
        if (job_.tasks[task].perceptual()) {
            perceptual = execute_perceptual(task);
        } else {
            output = execute_task(task);
        }
    } catch (...) {
        // The video's other tasks are left to finish, so that one failing
        // does not lose the others' metrics too.
        error = std::current_exception();
    }
    if (error && is_cancel(error)) {
        logger.info(name() + ": " + half + " cancelled after " + took_since(began));
    } else if (error) {
        const auto [why, tail] = failure_of(error);
        logger.error(name() + ": " + half + " failed after " + took_since(began) + ": " + why + stderr_note(tail));
    } else {
        logger.info(name() + ": " + half + " done in " + took_since(began));
        if (perceptual) {
            for (const auto& [key, why] : perceptual->failures) {
                logger.error(name() + ": " + std::string(metric_definition(key).label) + " failed: " + why);
            }
        }
    }
    const std::lock_guard lock(mutex_);
    TaskState& state = tasks_[task];
    if (error) {
        state.error = error;
        failed_.push_back(task);
    } else {
        state.result = std::move(output);
        state.perceptual = std::move(perceptual);
        state.waiting_for.clear();
        if (!state.progress) {
            state.progress = Counts{1, 1, 0.0};
        }
    }
    if (tasks_.size() > 1) {
        send_snapshots();  // or the task's part of the line would keep showing its last step
    }
    const bool last = ++finished_tasks_ == tasks_.size();
    if (state.finished() && !last) {
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
    // A metric that failed while the rest of its task finished (CVVDP beside
    // SSIMULACRA2 and Butteraugli, say) is reported like a failed task.
    const std::optional<PerceptualTaskOutput> perceptual = perceptual_so_far();
    const PerceptualFailures metric_failures = perceptual ? perceptual->failures : PerceptualFailures{};
    // Cancelled before any task finished, the video is cancelled, whatever
    // scores of it are saved.
    const bool calculated = perceptual.has_value() || std::any_of(tasks_.begin(), tasks_.end(), [](const TaskState& state) {
                                return state.result.has_value();
                            });
    std::optional<ComparisonResult> result;
    if (!cancelled || calculated) {
        result = merged();
    }
    // The tasks' own results are not needed once the video's is made: a
    // long list would otherwise hold every video's scores twice.
    for (TaskState& state : tasks_) {
        state.result.reset();
        state.perceptual.reset();
    }
    pass_outputs_.clear();
    if (!result) {
        if (cancelled || (!failed.empty() && is_cancel(tasks_[failed.front()].error))) {
            logger.info(name() + ": cancelled");
            scheduler_.report_cancelled_once();
        } else {
            const auto [message, tail] = failed.empty() ? std::pair<std::string, std::string>(
                                                              "No executable metric task was planned.", "")
                                                        : failure_of(tasks_[failed.front()].error);
            logger.error(name() + " failed: " + message + stderr_note(tail));
            if (events.job_failed) {
                events.job_failed(index_, message, tail);
            }
        }
        return;
    }
    const std::string metrics = metric_labels(result->metric_results.keys());
    if (failed.empty() && metric_failures.empty()) {
        logger.info(name() + " finished: " + metrics);
        if (events.job_finished) {
            events.job_finished(index_, *result);
        }
        return;
    }
    std::string message;
    std::string stderr_tail;
    PerceptualFailures failures;  // each metric once, its last reason (the Python app's dict)
    for (const std::size_t task : failed) {
        const auto [why, tail] = failure_of(tasks_[task].error);
        for (const std::string& key : job_.tasks[task].metric_keys) {
            set_failure(failures, key, why);
        }
        message += (message.empty() ? "" : "\n") + metric_labels(job_.tasks[task].metric_keys) + " failed: " + why;
        if (stderr_tail.empty()) {
            stderr_tail = tail;
        }
    }
    for (const auto& [key, why] : metric_failures) {
        message += (message.empty() ? "" : "\n") + std::string(metric_definition(key).label) + " failed: " + why;
        set_failure(failures, key, why);
    }
    logger.warning(name() + " finished with failed metrics (kept: " + (metrics.empty() ? "none" : metrics) + "):\n" +
                   message + stderr_note(stderr_tail));
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
        snapshot.passes = state.passes.value_or(std::vector<std::vector<std::string>>{planned.metric_keys});
        snapshot.phase = state.phase;
        // Its metrics that have scores: all that it scored once it is done,
        // or those of its finished passes while it runs.
        const auto scored = [&](const std::string& key) {
            if (state.result) {
                return state.result->has_metric(key);
            }
            if (state.perceptual) {
                return state.perceptual->metrics.has(key);
            }
            return planned.backend == perceptual_task &&
                   std::any_of(pass_outputs_.begin(), pass_outputs_.end(),
                               [&key](const PerceptualTaskOutput& output) { return output.metrics.has(key); });
        };
        for (const std::string& key : planned.metric_keys) {
            if (scored(key)) {
                snapshot.done_keys.push_back(key);
            }
        }
        if (state.progress) {
            snapshot.current = state.progress->current;
            snapshot.total = state.progress->total;
            snapshot.fps = state.progress->fps;
        }
        snapshot.state = state.finished()             ? "done"
                         : state.error                ? "failed"
                         : !state.waiting_for.empty() ? "waiting"
                         : state.progress             ? "running"
                                                      : "starting";
        snapshot.waiting_for = state.waiting_for;
        snapshot.step = state.step;
        snapshot.decode = state.decode;
    }
    return found;
}

void JobRun::send_snapshots() const {
    if (scheduler_.events.task_progress) {
        scheduler_.events.task_progress(index_, snapshots());
    }
}

// The scores of SSIMULACRA2, Butteraugli and CVVDP so far: the output of
// Vship's task (or, until it has one, of its finished passes), with that of
// the CPU's perceptual task. Nothing until one has scores.
std::optional<PerceptualTaskOutput> JobRun::perceptual_so_far() const {
    const PerceptualTaskOutput* gpu_task = nullptr;
    const PerceptualTaskOutput* cpu_task = nullptr;
    for (std::size_t task = 0; task < tasks_.size(); ++task) {
        if (tasks_[task].perceptual) {
            (job_.tasks[task].backend == perceptual_task ? gpu_task : cpu_task) = &*tasks_[task].perceptual;
        }
    }
    std::vector<const PerceptualTaskOutput*> parts;
    if (gpu_task != nullptr) {
        parts.push_back(gpu_task);
    } else {
        for (const PerceptualTaskOutput& output : pass_outputs_) {
            parts.push_back(&output);
        }
    }
    if (cpu_task != nullptr) {
        parts.push_back(cpu_task);
    }
    if (parts.empty()) {
        return std::nullopt;
    }
    if (parts.size() == 1 && (gpu_task != nullptr || cpu_task != nullptr)) {
        return *parts.front();
    }
    PerceptualTaskOutput combined;
    combined.source_crop = parts.front()->source_crop;
    combined.distorted_crop = parts.front()->distorted_crop;
    for (const PerceptualTaskOutput* part : parts) {
        combined.metrics.add_all(part->metrics);
        for (const auto& [key, reason] : part->failures) {
            set_failure(combined.failures, key, reason);
        }
        combined.compared_frame_count = std::max(combined.compared_frame_count, part->compared_frame_count);
    }
    return combined;
}

// The video's result from its finished tasks (the Python app's
// JobRun._merged): FFmpeg's, with VMAF, NEG and VMAF v1 from the GPU's task
// beside it, and the perceptual metrics so far. Nothing until a task, or a
// pass of Vship's, has finished.
std::optional<ComparisonResult> JobRun::merged() const {
    const ComparisonResult* ffmpeg = nullptr;
    const ComparisonResult* gpu = nullptr;
    for (std::size_t task = 0; task < tasks_.size(); ++task) {
        if (tasks_[task].result) {
            (job_.tasks[task].backend == gpu_vmaf_task ? gpu : ffmpeg) = &*tasks_[task].result;
        }
    }
    if (ffmpeg == nullptr) {
        std::swap(ffmpeg, gpu);
    }
    // The saved scores the run keeps, if it keeps any.
    const bool cached = job_.cached_result && !job_.cached_metrics.empty();
    std::optional<ComparisonResult> result;
    if (ffmpeg != nullptr) {
        result = *ffmpeg;
    } else if (cached) {
        // FFmpeg's metrics are all saved: the saved result is the base, with
        // its crops, frame count and the videos' details.
        result = *job_.cached_result;
    }
    if (ffmpeg != nullptr && gpu != nullptr) {
        result->metric_results.add_all(gpu->metric_results);
        // Each VMAF's model comes from the task that calculated it: VMAF with
        // a custom model file is FFmpeg's.
        if (gpu->has_metric("vmaf")) {
            result->model = gpu->model;
        }
        if (gpu->has_metric("vmaf_v1")) {
            result->model_v1 = gpu->model_v1;
            result->model_choice_v1 = gpu->model_choice_v1;
        }
    }
    if (const std::optional<PerceptualTaskOutput> perceptual = perceptual_so_far()) {
        if (!result) {
            // Only perceptual metrics: the result is made from the job, with
            // the perceptual task's crops and frame count.
            result.emplace();
            result->source = job_.source_info.path;
            result->distorted = job_.result_distorted_path.value_or(job_.distorted_info.path);
            result->fps = job_.source_info.fps;
            result->source_crop = perceptual->source_crop;
            result->distorted_crop = perceptual->distorted_crop;
            result->source_info = job_.source_info;
            result->distorted_info = job_.distorted_info;
            result->scale_direction = options_.scale_direction;
            result->scale_algorithm = options_.scale_algorithm;
            result->compared_frame_count = perceptual->compared_frame_count;
        }
        result->metric_results.add_all(perceptual->metrics);
    }
    if (result && cached) {
        // Saved metrics fill in only what this run did not calculate: a new
        // score always replaces the saved one.
        bool carried_vmaf = false;
        bool carried_v1 = false;
        for (const std::string& key : job_.cached_metrics.keys()) {
            if (result->has_metric(key)) {
                continue;
            }
            result->metric_results.copy_from(job_.cached_metrics, key);
            carried_vmaf = carried_vmaf || key == "vmaf";
            carried_v1 = carried_v1 || key == "vmaf_v1";
        }
        // A saved VMAF keeps its model when the run calculated no VMAF: with
        // VMAF on the GPU, a part of FFmpeg's metrics that is all saved is
        // not calculated again, and the saved result is not the base.
        if (carried_vmaf && result->model.empty()) {
            result->model = job_.cached_result->model;
        }
        if (carried_v1 && !result->model_choice_v1) {
            result->model_v1 = job_.cached_result->model_v1;
            result->model_choice_v1 = job_.cached_result->model_choice_v1;
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
    logger.info(std::string(app_log::run_start) + " " + std::to_string(jobs_.size()) +
                " video(s), CPU metrics for up to " + std::to_string(parallel_jobs_.load()) +
                " at once, GPU metrics one video at a time" + (gpu_metrics_together ? ", each video's in one pass" : "") +
                (by_timestamp() ? ", frames paired by timestamp (VML_FRAME_PAIRING)" : ""));
    for (const std::unique_ptr<JobRun>& run : runs_) {
        logger.info(run->describe());
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
        // Named in the log as the Python app names them.
        std::string thread_name = lanes[lane] == gpu_pool ? "vmaf-gpu" : "vmaf-cpu-" + std::to_string(lane);
        threads.emplace_back([this, pool = lanes[lane], thread_name = std::move(thread_name)]() mutable {
            app_log::set_thread_name(std::move(thread_name));
            drain_queue(pool);
        });
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
    logger.info(std::string("Run ended") + (cancel.cancelled() ? " (cancelled)" : ""));
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
    if (busy_[pool] >= (pool == cpu_pool ? parallel_jobs_.load() : 1)) {
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

// What the CPU's task says of the metrics a failed check of the GPU sent it
// (`failures`, each metric's key and why): a note for each reason, with its
// metrics.
std::vector<std::string> fallback_notes(const std::vector<std::pair<std::string, std::string>>& failures) {
    std::vector<std::pair<std::string, std::vector<std::string>>> by_reason;
    for (const auto& [key, why] : failures) {
        const auto same = std::find_if(by_reason.begin(), by_reason.end(),
                                       [&why](const auto& reason) { return reason.first == why; });
        (same != by_reason.end() ? same->second : by_reason.emplace_back(why, std::vector<std::string>{}).second)
            .push_back(key);
    }
    std::vector<std::string> notes;
    for (const auto& [why, keys] : by_reason) {
        notes.push_back("The GPU check failed (" + why + "); calculating " + metric_labels(keys, " and ") +
                        " on the CPU\xE2\x80\xA6");
    }
    return notes;
}

}  // namespace

std::vector<MetricTask> plan_tasks(const VideoInfo& source, const VideoInfo& test, const VmafOptions& options,
                                   const std::vector<std::string>& metric_keys,
                                   const PerceptualBackends& perceptual_backends, const MetricResultSet& saved) {
    // FFmpeg's metrics are calculated together: unless all of them are saved,
    // a saved one is calculated again with the rest. The others are each
    // left out when saved.
    const bool ffmpeg_saved = std::all_of(metric_keys.begin(), metric_keys.end(), [&saved](const std::string& key) {
        return is_perceptual_metric(key) || saved.has(key);
    });
    std::vector<std::string> keys;
    for (const std::string& key : metric_keys) {
        if (is_perceptual_metric(key) ? !saved.has(key) : !ffmpeg_saved) {
            keys.push_back(key);
        }
    }
    const bool vmaf = contains(keys, "vmaf");
    const bool neg = contains(keys, "vmaf_neg");
    const bool v1 = contains(keys, "vmaf_v1");
    GpuModels on_gpu;
    std::vector<std::pair<std::string, std::string>> check_failures;  // (key, why) of scores_on_gpu
    // A resolution test's VMAF is FFmpeg's: its test video is made in
    // FFmpeg's filter graph, from the source.
    if ((vmaf || neg || v1) && !options.resample_test) {
        // The size frames are compared at, where it is known before the run:
        // with black bars kept. Pictures with their black bars cut are
        // always even-sized.
        std::optional<std::pair<int, int>> size;
        if (options.crop_mode == CropMode::none) {
            size = analysis_dimensions(source, test, options);
        }
        on_gpu = scores_on_gpu(vmaf, neg, options.model, options.vmaf_on_gpu,
                               bit_depth(analysis_pix_fmt({source.pix_fmt, test.pix_fmt})), size, v1,
                               options.model_v1, &check_failures);
    }
    MetricTask ffmpeg{"ffmpeg", {}};
    ffmpeg.fallbacks = fallback_notes(check_failures);
    MetricTask gpu{std::string(gpu_vmaf_task), {}};
    // SSIMULACRA2 and Butteraugli chosen for the CPU make a task of their own,
    // in the CPU's queue: in Vship's task they would be calculated after its
    // GPU passes, holding the GPU meanwhile. CVVDP runs on the GPU only.
    MetricTask vship{std::string(perceptual_task), {}};
    MetricTask perceptual_cpu{std::string(perceptual_cpu_task), {}};
    for (const std::string& key : keys) {
        if (is_perceptual_metric(key)) {
            const auto chosen = perceptual_backends.find(key);
            const bool cpu = key != "cvvdp" && chosen != perceptual_backends.end() && chosen->second == "cpu";
            (cpu ? perceptual_cpu : vship).metric_keys.push_back(key);
            continue;
        }
        const bool scored_on_gpu = std::any_of(on_gpu.begin(), on_gpu.end(),
                                               [&key](const auto& model) { return model.first == key; });
        (scored_on_gpu ? gpu : ffmpeg).metric_keys.push_back(key);
    }
    // Once FFmpeg's metrics are split between the GPU and the CPU, a part
    // whose metrics are all saved is not run.
    if (vmaf || neg || v1) {
        for (MetricTask* part : {&ffmpeg, &gpu}) {
            if (std::all_of(part->metric_keys.begin(), part->metric_keys.end(),
                            [&saved](const std::string& key) { return saved.has(key); })) {
                part->metric_keys.clear();
            }
        }
    }
    std::vector<MetricTask> tasks;
    for (MetricTask* task : {&ffmpeg, &gpu, &vship, &perceptual_cpu}) {
        if (!task->metric_keys.empty()) {
            tasks.push_back(std::move(*task));
        }
    }
    return tasks;
}

void run_jobs(const std::vector<VmafJob>& jobs, int parallel_jobs, const RunEvents& events, const TaskRunner& runner,
              const PerceptualRunner& perceptual, bool gpu_metrics_together, unsigned cores,
              const CancelToken& cancel) {
    Scheduler(jobs, parallel_jobs, events, runner, perceptual, gpu_metrics_together, cores, cancel).run();
}

void run_jobs(const std::vector<VmafJob>& jobs, int parallel_jobs, const RunEvents& events, const TaskRunner& runner,
              unsigned cores, const CancelToken& cancel) {
    run_jobs(jobs, parallel_jobs, events, runner, PerceptualRunner(), false, cores, cancel);
}

// ------------------------------------------------------------- JobScheduler

// What the Scheduler refers to, kept here for as long as it runs.
struct JobScheduler::Parts {
    Parts(std::vector<VmafJob> jobs_given, int parallel_jobs, RunEvents events_given, TaskRunner runner_given,
          PerceptualRunner perceptual_given, bool gpu_metrics_together, unsigned cores, CancelToken cancel_given)
        : jobs(std::move(jobs_given)),
          events(std::move(events_given)),
          runner(std::move(runner_given)),
          perceptual(std::move(perceptual_given)),
          cancel(std::move(cancel_given)),
          scheduler(jobs, parallel_jobs, events, runner, perceptual, gpu_metrics_together, cores, cancel) {}

    const std::vector<VmafJob> jobs;
    const RunEvents events;
    const TaskRunner runner;
    const PerceptualRunner perceptual;
    const CancelToken cancel;
    Scheduler scheduler;  // last: it refers to the rest
};

JobScheduler::JobScheduler(std::vector<VmafJob> jobs, int parallel_jobs, RunEvents events, TaskRunner runner,
                           PerceptualRunner perceptual, bool gpu_metrics_together, unsigned cores, CancelToken cancel)
    : parts_(std::make_unique<Parts>(std::move(jobs), parallel_jobs, std::move(events), std::move(runner),
                                     std::move(perceptual), gpu_metrics_together, cores, std::move(cancel))) {}

JobScheduler::~JobScheduler() = default;

void JobScheduler::run() { parts_->scheduler.run(); }

void JobScheduler::set_parallel_jobs(int count) { parts_->scheduler.set_parallel_jobs(count); }

int JobScheduler::parallel_jobs() const { return parts_->scheduler.parallel_jobs(); }

}  // namespace vml
