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
#include "vml/append.h"
#include "vml/chroma_vmaf.h"
#include "vml/crop_detect.h"
#include "vml/ffmpeg_run.h"
#include "vml/film_grain_scoring.h"
#include "vml/geometry.h"
#include "vml/gpu_cpu_metrics.h"
#include "vml/gpu_decoders.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
#include "vml/isolated.h"
#include "vml/libvmaf.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/perceptual.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/result_json.h"
#include "vml/size_changes.h"
#include "vml/time_format.h"
#include "vml/vship.h"

namespace vml {

namespace {

// Whether a run cutting `source_crop` and `distorted_crop` from `source` and
// `distorted` scores the picture `saved` was scored on: the same boxes, or
// none and a box of the whole frame, which every route cuts as none.
bool cuts_as_saved(const ComparisonResult& saved, const VideoInfo& source, const VideoInfo& distorted,
                   const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop) {
    const auto same_box = [](const std::optional<CropBox>& first, const std::optional<CropBox>& second,
                             const VideoInfo& info) {
        const auto [width, height] = info.frame_size();
        const auto cuts_nothing = [&](const std::optional<CropBox>& box) { return !box || box->is_noop(width, height); };
        return first == second || (cuts_nothing(first) && cuts_nothing(second));
    };
    return same_box(saved.source_crop, source_crop, source) && same_box(saved.distorted_crop, distorted_crop, distorted);
}

// Why a saved score is not kept beside a run's new scores (JobRun::saved_scores_fit).
constexpr const char* saved_with_other_black_bars =
    "Its saved score was made with other black bars: it is calculated again on the next run.";

// Videos in progress at once: one per CPU lane, plus the one on the GPU. A
// video's CPU and GPU tasks run in separate queues, so one queue can run
// ahead of the other, but not without limit: each video in progress has a
// line of its own, and the list is still worked through in order.
constexpr std::size_t max_videos_in_flight = max_parallel_jobs + 1;

bool contains(const std::vector<std::string>& keys, std::string_view key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

// Whether `key` is VMAF YUV's U or V calculated only to make VMAF YUV: its
// planes not asked for themselves (`asked`, the video's metrics).
bool only_for_vmaf_yuv(const std::string& key, const std::vector<std::string>& asked) {
    return (key == vmaf_u_key || key == vmaf_v_key) && !contains(asked, key);
}

// A task's `keys` as the video's metrics (`asked`) name them: VMAF YUV in
// place of the U and V plan_tasks put there for it.
std::vector<std::string> as_asked(const std::vector<std::string>& keys, const std::vector<std::string>& asked) {
    std::vector<std::string> named;
    for (const std::string& key : keys) {
        const std::string name = only_for_vmaf_yuv(key, asked) ? "vmaf_yuv" : key;
        if (!contains(named, name)) {
            named.push_back(name);
        }
    }
    return named;
}

// VMAF YUV (MSU's YUV-VMAF 6:1:1), (6 VMAF + U + V) / 8 of each frame, and
// its Y, VMAF's score, where `asked` asks for them and `results` has what
// they are made from but not them (a saved one stands); the U and V
// calculated only for it are left out. Nothing where VMAF and its U and V
// were not scored on the same frames: finish() says why.
void make_vmaf_yuv(MetricResultSet& results, const std::vector<std::string>& asked) {
    const FrameMetricResult* vmaf = results.frame("vmaf");
    const FrameMetricResult* u = results.frame(vmaf_u_key);
    const FrameMetricResult* v = results.frame(vmaf_v_key);
    if (contains(asked, "vmaf_yuv") && !results.has("vmaf_yuv") && vmaf && u && v && u->frame == vmaf->frame &&
        v->frame == vmaf->frame) {
        // The planes' provenance: VMAF's model, scored by the app's libvmaf.
        FrameMetricResult yuv{"vmaf_yuv", vmaf->frame, vmaf->time, {}, u->provenance, u->av1_grain_dav1d};
        yuv.values.reserve(vmaf->values.size());
        for (std::size_t frame = 0; frame < vmaf->values.size(); ++frame) {
            const double sum = 6.0 * vmaf->values[frame] + u->values[frame] + v->values[frame];
            yuv.values.push_back(static_cast<float>(to_6_decimals(sum / 8.0)));
        }
        results.add(std::move(yuv));
    }
    if (contains(asked, "vmaf_yuv_y") && !results.has("vmaf_yuv_y") && vmaf) {
        FrameMetricResult y = *vmaf;
        y.key = "vmaf_yuv_y";
        results.add(std::move(y));
    }
    for (const char* plane : {vmaf_u_key, vmaf_v_key}) {
        if (only_for_vmaf_yuv(plane, asked)) {
            results.remove(plane);
        }
    }
}

// `result` with only every `step`-th frame (0, step, 2*step...): the frames
// a subsampled request covers, as libvmaf's n_subsample scores them.
FrameMetricResult sampled_every(const FrameMetricResult& result, int step) {
    FrameMetricResult kept{result.key, {}, {}, {}, result.provenance, result.av1_grain_dav1d};
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

// A task's metrics but those scored beside them (MetricTask::cpu_beside).
std::vector<std::string> own_keys(const MetricTask& task) {
    std::vector<std::string> keys;
    for (const std::string& key : task.metric_keys) {
        if (!contains(task.cpu_beside, key)) {
            keys.push_back(key);
        }
    }
    return keys;
}

class Scheduler;

// One video: its tasks, which the scheduler's CPU and GPU lanes run, their
// progress, and the result they make together (the Python app's JobRun).
class JobRun {
public:
    JobRun(Scheduler& scheduler, std::size_t index, const VmafJob& job);

    std::size_t index() const { return index_; }
    std::size_t task_count() const { return tasks_.size(); }
    Pool pool_of(std::size_t task) const {
        const MetricTask& planned = job_.tasks[task];
        return planned.on_gpu() || planned.engine_lane ? gpu_pool : cpu_pool;
    }
    bool heavy_cpu(std::size_t task) const { return job_.tasks[task].heavy_cpu; }

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
        // Those of its CPU scorers' metrics the GPU calculates (metrics_on_gpu).
        std::vector<std::string> gpu_keys;
        // The metrics of each of its passes (Vship's: planned at the start,
        // then updated as each starts), and the pass under way: its number,
        // how many there are, and the progress count it started at.
        std::optional<std::vector<std::vector<std::string>>> passes;
        std::optional<std::array<long long, 3>> phase;
        std::string waiting_for;  // "GPU" or "CPU" while it waits
        RunStatus step;           // its latest status
        std::optional<HardwareDecoding> decode;  // the last decode plan a status named
        SoftwareDecodingWhy plan_why;            // why the run's last plan left videos in software (decoding_why)
        bool on_vvdec_vulkan = false;            // its VVC videos are decoded on the GPU (a vvdec_used status)
        // Its passes' times: those that ended, and the one under way (while
        // `open`): its metrics, and the run's clock, paused time and the
        // task's progress count at its start.
        std::vector<PassTiming> timings;
        bool open = false;
        std::vector<std::string> pass_keys;
        double pass_began = 0.0;
        double pass_paused_before = 0.0;
        long long pass_frames_before = 0;
        // The run's clock, paused time and the task's progress count at its first progress report
        std::optional<double> pass_first_frame;
        double pass_paused_at_first_frame = 0.0;
        long long pass_count_at_first_frame = 0;

        bool finished() const { return result.has_value() || perceptual.has_value(); }
    };

    // A task's name in the log: "GPU metrics (SSIMULACRA2, CVVDP)".
    std::string task_name(std::size_t task) const;
    // A task's metrics as the log and the run line show them: those scored in
    // its own place, and those the CPU scores beside them.
    std::pair<std::vector<std::string>, std::vector<std::string>> shown_keys(std::size_t task) const;
    // Whether the task's run may put VMAF's family (CAMBI, CIEDE2000, MS-SSIM, VMAF U and V, HDR VMAF's features) on
    // the GPU: the run's choice, except in a CPU task the planner gave VMAF, NEG or v1 (plan_tasks), or one
    // split from it (vmaf_on_cpu), which stays on the CPU. The run's own placement (models_on_gpu), seeing
    // only this task's metrics, might otherwise move them to the GPU. A CPU task without them keeps the
    // choice: HDR VMAF's features go to the GPU with it (hdr_plan).
    bool vmaf_family_on_gpu(std::size_t task) const;
    // Whether the video's frames may have been paired out of step
    // (frame_pairing.h), as it ends: seen from its result at no cost, and
    // counted (RunEvents::count_frames, which may read whole files) without
    // mutex_ held. A cancelled run checks nothing: the saved result's verdict.
    std::optional<FramesShort> check_frame_pairing();
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
    // Task `task`'s snapshot for its metrics `keys`, as `backend` in `lane`.
    std::optional<HardwareDecoding> shown_decoding(const TaskState& state) const;
    SoftwareDecodingWhy decode_why(const TaskState& state, const std::optional<HardwareDecoding>& shown) const;
    TaskSnapshot snapshot_of(std::size_t task, const std::string& backend, const std::string& lane,
                             const std::vector<std::string>& keys) const;
    void send_snapshots() const;
    std::optional<PerceptualTaskOutput> perceptual_so_far() const;
    std::optional<ComparisonResult> merged() const;
    bool saved_scores_fit() const;
    bool task_calculates(const std::string& key) const;
    void send_result_so_far() const;
    // A pass of `task` begins with `keys`; the one under way ends as `end`.
    void open_pass(std::size_t task, const std::vector<std::string>& keys);
    // `frames`, where given, is how many it compared (a finished task's result's).
    void close_pass(std::size_t task, const std::string& end, std::optional<long long> frames = std::nullopt);
    // Sends the video's facts, once: each metric's outcome from `result`
    // and the metrics that failed (`failed`, each with why).
    void send_facts(const std::optional<ComparisonResult>& result, const PerceptualFailures& failed);

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
    bool facts_sent_ = false;
    // When it started (begin): the run's clock, its paused time then, and in UTC.
    double began_ = 0.0;
    double paused_before_ = 0.0;
    std::string started_utc_;

public:
    // Sends the facts of a video a cancel stopped before it had a result,
    // its metrics cancelled, unless they were sent (Scheduler::run, once the
    // lanes have ended).
    void report_stopped();
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
    // The videos running a heavy_cpu task (MetricTask): each runs one at a time, in either queue.
    std::set<std::size_t> heavy_running_;
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

std::pair<std::vector<std::string>, std::vector<std::string>> JobRun::shown_keys(std::size_t task) const {
    const MetricTask& planned = job_.tasks[task];
    std::vector<std::string> own = as_asked(own_keys(planned), job_.metric_keys);
    std::vector<std::string> beside = as_asked(planned.cpu_beside, job_.metric_keys);
    // HDR VMAF beside the GPU's VMAF scores its features on the GPU too (hdr_plan; only its pictures are converted
    // on the CPU), so it is shown with the GPU's metrics. In the CPU's task it stays there: that task waits for a
    // CPU slot, as the run line says.
    if (pool_of(task) == gpu_pool && job_.options.vmaf_on_gpu) {
        if (const auto hdr = std::find(beside.begin(), beside.end(), "vmaf_hdr"); hdr != beside.end()) {
            own.push_back(*hdr);
            beside.erase(hdr);
        }
    }
    return {std::move(own), std::move(beside)};
}

bool JobRun::vmaf_family_on_gpu(std::size_t task) const {
    const MetricTask& planned = job_.tasks[task];
    const std::vector<std::string>& keys = planned.metric_keys;
    const bool kept_on_cpu = planned.backend != gpu_vmaf_task &&
                             (planned.vmaf_on_cpu || contains(keys, "vmaf") || contains(keys, "vmaf_neg") ||
                              contains(keys, "vmaf_v1"));
    return job_.options.vmaf_on_gpu && !kept_on_cpu;
}

std::string JobRun::task_name(std::size_t task) const {
    const auto [own, beside] = shown_keys(task);
    const bool gpu_offered = vmaf_family_on_gpu(task);
    const bool on_cpu = pool_of(task) == cpu_pool;
    std::string name = std::string(on_cpu ? "CPU" : "GPU") + " metrics (" +
                       (on_cpu ? cpu_metrics_text(own, gpu_offered) : metric_labels(own)) + ")";
    if (!beside.empty()) {
        name += " with CPU metrics (" + cpu_metrics_text(beside, gpu_offered) + ") beside them";
    }
    return name;
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
        // Those the run does not calculate again (a VMAF beside the NEG it
        // keeps on the CPU, say: plan_tasks).
        std::vector<std::string> keys = job_.cached_metrics.keys();
        std::erase_if(keys, [this](const std::string& key) { return task_calculates(key); });
        kept = metric_labels(keys);
    }
    const VmafOptions& options = job_.options;
    // Frames as the window shows them: the first and the last compared.
    const std::string range =
        options.range_in_frames
            ? (options.range_end > 0 ? "frames " + std::to_string(options.range_start) + "-" +
                                           std::to_string(options.range_end - 1)
               : options.range_start > 0 ? "from frame " + std::to_string(options.range_start)
                                         : "whole videos")
        : options.compared_range().in_milliseconds
            ? (options.range_end > 0 ? range_time_text(options.range_start) + "-" + range_time_text(options.range_end)
               : options.range_start > 0 ? "from " + range_time_text(options.range_start)
                                         : "whole videos")
        : options.range_end > 0 ? std::to_string(options.range_start) + "-" + std::to_string(options.range_end) + " s"
        : options.range_start > 0 ? "from " + std::to_string(options.range_start) + " s"
                                  : "whole videos";
    const std::string black_bars = options.crop_mode == CropMode::manual
                                       ? "manual (source " + crop_edges_text(options.manual_source_crop) + ", test " +
                                             crop_edges_text(options.manual_test_crop) + ")"
                                       : crop_mode_text(options.crop_mode);
    std::string settings = "black bars " + black_bars + ", range " + range +
                           ", resolution mismatch " +
                           scale_direction_value(options.scale_direction) + " (" + options.scale_algorithm +
                           "), VMAF v0.6.1 model " + options.model_choice + ", VMAF v1 model " +
                           options.model_choice_v1 +
                           (options.compute_vmaf_hdr ? ", HDR VMAF model " + options.model_choice_hdr : "") +
                           ", frame subsample " + std::to_string(options.n_subsample) + ", GPU decode " +
                           (options.gpu_decode ? "on" : "off");
#ifndef __APPLE__
    // A Mac decodes with its own media engine, whichever maker is asked.
    if (options.gpu_decode && !vendor_label(options.gpu_vendor).empty()) {
        settings += " (" + std::string(vendor_label(options.gpu_vendor)) + ")";
    }
#endif
    for (const auto& [key, backend] : job_.perceptual_backends) {
        settings += ", " + std::string(metric_definition(key).label) + " on " + ascii_upper(backend);
    }
    settings += ", Vship build " + std::string(vship_build_label(options.vship_build));
    if (const VshipBuild in_use = vship_build_in_use(options.vship_build); in_use != options.vship_build) {
        settings += " (" + std::string(vship_build_label(in_use)) + " in use)";
    }
    settings += ", CVVDP display " + job_.cvvdp.display.describe() +
                (job_.cvvdp.resize_to_display ? ", the frames filling it" : "");
    std::string text = name() + ":\n  test   " + platform::path_text(job_.distorted_info.path) + " (" +
                       media_text(job_.distorted_info) + ")\n  source " + platform::path_text(job_.source_info.path) +
                       " (" + media_text(job_.source_info) + ")\n  calculating " + calculating +
                       (kept.empty() ? "" : "; saved scores kept: " + kept) + "\n  " + settings;
    if (options.resample_test) {
        text += "\n  resolution test: " + options.resample_test->label + " (" + options.resample_test->scaler + ")";
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
    began_ = scheduler_.cancel.now();
    paused_before_ = scheduler_.cancel.paused_seconds();
    started_utc_ = utc_text(std::chrono::system_clock::now());
    logger.info(name() + ": started");
    if (scheduler_.events.job_started) {
        scheduler_.events.job_started(index_);
    }
    if (tasks_.size() > 1) {
        const std::vector<bool> taken = scheduler_.admitted(*this);
        for (std::size_t task = 0; task < tasks_.size(); ++task) {
            if (!taken[task]) {
                tasks_[task].waiting_for = pool_of(task) == cpu_pool ? "CPU" : "GPU";
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
    for (const std::string& key : job_.tasks[task].metric_keys) {
        if (key == vmaf_u_key || key == vmaf_v_key) {
            options.set_metric_enabled(key, true);  // VMAF YUV's U and V, which plan_tasks put in its place
        }
    }
    options.vmaf_on_gpu = vmaf_family_on_gpu(task);
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
    if (state.open && state.timings.empty() && number == 1) {
        state.pass_keys = keys;  // the time since the task began is its first pass's
    } else {
        close_pass(task, "done");
        open_pass(task, keys);
    }
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
    if (tasks_[task].open && !tasks_[task].pass_first_frame) {
        tasks_[task].pass_first_frame = scheduler_.cancel.now();
        tasks_[task].pass_paused_at_first_frame = scheduler_.cancel.paused_seconds();
        tasks_[task].pass_count_at_first_frame = current;
    }
    tasks_[task].progress = Counts{current, total, fps};
    send_snapshots();
}

void JobRun::report_status(std::size_t task, const RunStatus& status) {
    if (status.kind == StatusKind::decoding_why) {
        const std::lock_guard lock(mutex_);
        tasks_[task].plan_why = status.decoding_why;  // for the results JSON; the run's log says them itself
        return;
    }
    if (status.kind == StatusKind::metrics_on_gpu) {
        // A CPU scorer started (the scorer logs it): its metrics on the GPU
        // replace what was said of them, as a pass made again on the CPU after
        // the GPU failed says none.
        const auto [metrics, on_gpu] = metrics_on_gpu_from(status.text);
        const std::lock_guard lock(mutex_);
        std::vector<std::string>& gpu_keys = tasks_[task].gpu_keys;
        std::erase_if(gpu_keys, [&metrics](const std::string& key) { return contains(metrics, key); });
        gpu_keys.insert(gpu_keys.end(), on_gpu.begin(), on_gpu.end());
        send_snapshots();
        return;
    }
    logger.info(name() + ": " + status.text);
    if (status.kind == StatusKind::gpu_decoder_failed || status.kind == StatusKind::vvdec_used ||
        status.kind == StatusKind::vvdec_off) {
        // For the run's process (distrust_failed_decoder, take_child_status), logged above: not a step, though the
        // task's decoder line changes (shown_decoding)
        const std::lock_guard lock(mutex_);
        tasks_[task].on_vvdec_vulkan = status.kind == StatusKind::vvdec_used;
        send_snapshots();
        return;
    }
    const std::lock_guard lock(mutex_);
    TaskState& state = tasks_[task];
    state.step = status;
    if (status.decoding) {
        state.decode = status.decoding;
    }
    if (tasks_.size() > 1 && status.kind == StatusKind::gpu_wait) {
        state.waiting_for = "GPU";
    }
    if (pool_of(task) == gpu_pool && !job_.tasks[task].engine_lane && status.kind == StatusKind::gpu_vmaf_failed) {
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
        open_pass(task, job_.tasks[task].metric_keys);
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
    // An FFmpeg or VMAF task's frames are its result's: its last progress report comes before its end
    close_pass(task, !error ? "done" : is_cancel(error) ? "cancelled" : "failed",
               output ? std::optional(output->compared_frame_count) : std::nullopt);
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

std::optional<FramesShort> JobRun::check_frame_pairing() {
    FramesShort short_of;
    {
        const std::lock_guard lock(mutex_);
        if (scheduler_.cancel.cancelled()) {
            // Nothing checked: a result kept with what had finished, and
            // saved again, keeps the saved verdict.
            return job_.cached_result ? job_.cached_result->frames_short : std::nullopt;
        }
        const std::optional<ComparisonResult> result = merged();
        if (!result) {
            return std::nullopt;
        }
        const std::optional<FramesShort> seen =
            frames_short_of_lengths(*result, options_.range_start == 0 && options_.range_end == 0);
        if (!seen) {
            return std::nullopt;
        }
        short_of = *seen;
        // Counted before, for the same pairs: the counts saved with the scores.
        if (job_.cached_result) {
            const std::optional<FramesShort>& saved = job_.cached_result->frames_short;
            if (saved && saved->counted() && saved->scored_pairs == short_of.scored_pairs) {
                short_of.source_frames = saved->source_frames;
                short_of.test_frames = saved->test_frames;
                return short_of;  // with the lengths, which a verdict saved before them lacks
            }
        }
    }
    if (!scheduler_.events.count_frames) {
        return short_of;
    }
    logger.info(name() + ": " + std::to_string(short_of.scored_pairs) + " frame pairs scored where the videos' " +
                "lengths hold " + std::to_string(short_of.length_frames) + "; counting each video's frames");
    try {
        return count_frames_short(short_of, job_.source_info, job_.distorted_info, scheduler_.events.count_frames);
    } catch (const Cancelled&) {
        return short_of;
    }
}

void JobRun::finish() {
    const std::optional<FramesShort> frames_short = check_frame_pairing();
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
    // So is one that failed in an FFmpeg or GPU task (XPSNR above 8 bits that
    // the app cannot score, say: ComparisonResult::metric_failures).
    const std::optional<PerceptualTaskOutput> perceptual = perceptual_so_far();
    PerceptualFailures metric_failures = perceptual ? perceptual->failures : PerceptualFailures{};
    for (const TaskState& state : tasks_) {
        if (state.result) {
            for (const auto& [key, why] : state.result->metric_failures) {
                set_failure(metric_failures, only_for_vmaf_yuv(key, job_.metric_keys) ? "vmaf_yuv" : key, why);
            }
        }
    }
    if (job_.cached_result && !saved_scores_fit()) {
        for (const std::string& key : job_.cached_metrics.keys()) {
            if (!task_calculates(key)) {
                set_failure(metric_failures, key, saved_with_other_black_bars);
            }
        }
    }
    // Cancelled before any task finished, the video is cancelled, whatever
    // scores of it are saved.
    const bool calculated = perceptual.has_value() || std::any_of(tasks_.begin(), tasks_.end(), [](const TaskState& state) {
                                return state.result.has_value();
                            });
    std::optional<ComparisonResult> result;
    if (!cancelled || calculated) {
        result = merged();
    }
    if (result) {
        result->frames_short = frames_short;  // a saved result's own replaced (kept when cancelled)
        if (frames_short && !frames_short->in_step()) {
            logger.warning(name() + ": " + frames_short_text(*frames_short));
        } else if (frames_short && *frames_short->source_frames == *frames_short->test_frames) {
            logger.info(name() + ": the two videos have the same number of frames (" +
                        std::to_string(*frames_short->source_frames) + "): their frames are paired in step");
        } else if (frames_short) {
            logger.info(name() + ": the reference has " + std::to_string(*frames_short->source_frames) +
                        " frames and the test video " + std::to_string(*frames_short->test_frames) +
                        ", of different lengths, and the one with fewer lacks no more of its length: their frames "
                        "are paired in step");
        }
    }
    // VMAF YUV not made, its U and V not failed (finish's failures of them are its own): VMAF failed, or the
    // three were not scored on the same frames (a saved VMAF of another range, say: make_vmaf_yuv).
    if (result && !cancelled && contains(job_.metric_keys, "vmaf_yuv") && !result->has_metric("vmaf_yuv") &&
        std::none_of(metric_failures.begin(), metric_failures.end(),
                     [](const auto& failure) { return failure.first == "vmaf_yuv"; }) &&
        std::none_of(failed.begin(), failed.end(), [this](std::size_t task) {
            return contains(as_asked(job_.tasks[task].metric_keys, job_.metric_keys), "vmaf_yuv");
        })) {
        set_failure(metric_failures, "vmaf_yuv",
                    result->has_metric("vmaf") ? "it needs VMAF and its U and V scored on the same frames"
                                               : "it is made from VMAF's score, which failed");
    }
    PerceptualFailures why_failed = metric_failures;  // each metric once, as the events say it
    for (const std::size_t task : failed) {
        const std::string why = failure_of(tasks_[task].error).first;
        for (const std::string& key : as_asked(job_.tasks[task].metric_keys, job_.metric_keys)) {
            if (std::none_of(metric_failures.begin(), metric_failures.end(),
                             [&key](const auto& failure) { return failure.first == key; })) {
                set_failure(why_failed, key, why);
            }
        }
    }
    send_facts(result, why_failed);
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
        const std::vector<std::string> keys = as_asked(job_.tasks[task].metric_keys, job_.metric_keys);
        for (const std::string& key : keys) {
            set_failure(failures, key, why);
        }
        message += (message.empty() ? "" : "\n") + metric_labels(keys) + " failed: " + why;
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
        const auto [own, beside] = shown_keys(task);
        if (!beside.empty()) {
            // Shown as the CPU's task they were, with the GPU's progress.
            found.push_back(snapshot_of(task, "ffmpeg", "cpu", beside));
        }
        // The engine's pass is shown as the CPU route's task it is (TaskSnapshot::lane).
        const bool gpu_lane = pool_of(task) == gpu_pool && !job_.tasks[task].engine_lane;
        found.push_back(snapshot_of(task, job_.tasks[task].backend, gpu_lane ? "gpu" : "cpu", own));
    }
    return found;
}

TaskSnapshot JobRun::snapshot_of(std::size_t task, const std::string& backend, const std::string& lane,
                                 const std::vector<std::string>& keys) const {
    const MetricTask& planned = job_.tasks[task];
    const TaskState& state = tasks_[task];
    TaskSnapshot snapshot;
    snapshot.backend = backend;
    snapshot.metric_keys = keys;
    snapshot.lane = lane;
    for (const std::string& key : state.cpu_keys) {
        if (contains(keys, key)) {
            snapshot.cpu_keys.push_back(key);
        }
    }
    for (const std::string& key : state.gpu_keys) {
        if (contains(keys, key)) {
            snapshot.gpu_keys.push_back(key);
        }
    }
    snapshot.passes = state.passes.value_or(std::vector<std::vector<std::string>>{keys});
    snapshot.phase = state.phase;
    // Its metrics that have scores: all that it scored once it is done, or
    // those of its finished passes while it runs.
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
    for (const std::string& key : keys) {
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
    snapshot.decode = shown_decoding(state);
    return snapshot;
}

// A task's plan as its decoder line shows it. vvdec-vulkan is not one of FFmpeg's decoders, so a plan has nothing for
// a video it decodes: each VVC video is marked here instead, for the line alone.
std::optional<HardwareDecoding> JobRun::shown_decoding(const TaskState& state) const {
    std::optional<HardwareDecoding> shown = state.decode;
    if (!shown || !state.on_vvdec_vulkan || vvdec_turned_off()) {
        return shown;
    }
    if (ascii_lower(job_.source_info.codec_name) == "vvc") {
        shown->source = std::string(vvdec_decoder);
    }
    if (ascii_lower(job_.distorted_info.codec_name) == "vvc") {
        shown->distorted = std::string(vvdec_decoder);
    }
    return shown;
}

// Why the task's shown plan left each video it did in software: the run's own reasons, given as it made its plan
// from the videos as it scores them (decoding_why_status); else, for a video its plan put on the GPU, which the GPU's
// decoding failing then took off it, or where it gave none (GPU decoding off, a resolution test, whose test video is
// decoded from the source), the reasons found here.
SoftwareDecodingWhy JobRun::decode_why(const TaskState& state, const std::optional<HardwareDecoding>& shown) const {
    if (!shown) {
        return {};
    }
    const VideoInfo& test = options_.resample_test ? job_.source_info : job_.distorted_info;
    SoftwareDecodingWhy why =
        software_decoding_why(shown, job_.source_info, test, options_.gpu_decode, gpu_decoding_distrusted());
    if (!shown->source && !state.plan_why.source.empty()) {
        why.source = state.plan_why.source;
    }
    if (!shown->distorted && !state.plan_why.distorted.empty()) {
        why.distorted = state.plan_why.distorted;
    }
    return why;
}

void JobRun::open_pass(std::size_t task, const std::vector<std::string>& keys) {
    TaskState& state = tasks_[task];
    state.open = true;
    state.pass_keys = keys;
    state.pass_began = scheduler_.cancel.now();
    state.pass_paused_before = scheduler_.cancel.paused_seconds();
    state.pass_frames_before = state.progress ? state.progress->current : 0;
    state.pass_first_frame.reset();
}

void JobRun::close_pass(std::size_t task, const std::string& end, std::optional<long long> frames) {
    TaskState& state = tasks_[task];
    if (!state.open) {
        return;
    }
    state.open = false;
    PassTiming pass;
    pass.task = job_.tasks[task].backend;
    pass.metric_keys = state.pass_keys;
    // Where its scores came from: the CPU once the GPU's VMAF fell back; the engine's pass on the GPU where its
    // scorer said the GPU took them (metrics_on_gpu)
    const bool on_gpu = job_.tasks[task].engine_lane ? !state.gpu_keys.empty()
                                                     : pool_of(task) == gpu_pool && state.cpu_keys.empty();
    pass.on = on_gpu ? "gpu" : "cpu";
    pass.seconds = scheduler_.cancel.now() - state.pass_began;
    pass.paused_seconds = scheduler_.cancel.paused_seconds() - state.pass_paused_before;
    pass.frames = frames.value_or(
        std::max(0LL, (state.progress ? state.progress->current : 0) - state.pass_frames_before));
    if (state.pass_first_frame) {
        pass.setup_seconds = *state.pass_first_frame - state.pass_began;
    } else if (pass.frames == 0) {
        pass.setup_seconds = pass.seconds;  // it scored nothing
    }
    // Its fps: the frames scored after its first report over the time since. The frames that report already
    // counts were scored in time not measured: FFmpeg's first -progress block comes half a second in. A pass too
    // short to time that way (nothing, or under a tenth of a second, after its first report) is timed over its
    // whole length, its start included.
    if (state.pass_first_frame) {
        constexpr double shortest_timed_seconds = 0.1;
        const double scoring = scheduler_.cancel.now() - *state.pass_first_frame -
                               (scheduler_.cancel.paused_seconds() - state.pass_paused_at_first_frame);
        const long long before_first =
            std::clamp(state.pass_count_at_first_frame - state.pass_frames_before, 0LL, pass.frames);
        const long long after_first = pass.frames - before_first;
        const double whole = pass.seconds - pass.paused_seconds;
        if (after_first > 0 && scoring >= shortest_timed_seconds) {
            pass.fps = static_cast<double>(after_first) / scoring;
        } else {
            pass.fps = whole > 0 ? static_cast<double>(pass.frames) / whole : 0.0;
        }
    }
    pass.end = end;
    pass.decode = shown_decoding(state);
    pass.decode_why = decode_why(state, pass.decode);
    state.timings.push_back(std::move(pass));
}

void JobRun::send_facts(const std::optional<ComparisonResult>& result, const PerceptualFailures& failed) {
    if (facts_sent_) {
        return;
    }
    facts_sent_ = true;
    if (!scheduler_.events.job_facts) {
        return;
    }
    const CancelToken& cancel = scheduler_.cancel;
    RunFacts facts;
    facts.settings = VideoSettings{job_.source_info,
                                   job_.distorted_info,
                                   begun_ ? options_ : job_.options,
                                   job_.metric_keys,
                                   {job_.perceptual_backends.begin(), job_.perceptual_backends.end()},
                                   job_.cvvdp};
    if (claimed_) {
        facts.started_utc = started_utc_;
        facts.seconds = cancel.now() - began_;
        facts.paused_seconds = cancel.paused_seconds() - paused_before_;
    }
    for (std::size_t task = 0; task < tasks_.size(); ++task) {
        close_pass(task, "cancelled");  // (none is open once its task has ended)
        append(facts.passes, tasks_[task].timings);
    }
    const bool cancelled = cancel.cancelled();
    for (const std::string& key : job_.metric_keys) {
        MetricOutcome outcome;
        const auto failure = std::find_if(failed.begin(), failed.end(),
                                          [&key](const auto& item) { return item.first == key; });
        if (result && result->has_metric(key)) {
            // A saved score the run kept, or one it calculated
            outcome.status = job_.cached_metrics.has(key) && !task_calculates(key) ? metric_saved
                                                                                  : metric_calculated;
        } else if (failure != failed.end()) {
            outcome = {metric_failed, failure->second};
        } else if (cancelled) {
            outcome.status = metric_cancelled;
        } else {
            outcome = {metric_not_run, "no task was planned to calculate it"};
        }
        facts.metrics.emplace_back(key, std::move(outcome));
    }
    scheduler_.events.job_facts(index_, facts);
}

void JobRun::report_stopped() {
    const std::lock_guard lock(mutex_);
    send_facts(std::nullopt, {});
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
bool JobRun::task_calculates(const std::string& key) const {
    return std::any_of(job_.tasks.begin(), job_.tasks.end(), [&key](const MetricTask& task) {
        return contains(task.metric_keys, key) || contains(task.cpu_beside, key);
    });
}

// Whether the saved scores are of the picture this run scored. The black-bar
// detection can find another box than the one they were made with (v2.4's
// replaced FFmpeg's cropdetect, or a black-bar rule changed since they were
// saved): together they would mix scores of different pictures without a word.
// calculate_saved_scores_of_other_bars has such scores calculated before the
// run where it can tell.
bool JobRun::saved_scores_fit() const {
    const auto fits = [this](const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop) {
        return cuts_as_saved(*job_.cached_result, job_.source_info, job_.distorted_info, source_crop, distorted_crop);
    };
    const std::optional<PerceptualTaskOutput> perceptual = perceptual_so_far();
    return (!perceptual || fits(perceptual->source_crop, perceptual->distorted_crop)) &&
           std::all_of(tasks_.begin(), tasks_.end(), [&fits](const TaskState& state) {
               return !state.result || fits(state.result->source_crop, state.result->distorted_crop);
           });
}

std::optional<ComparisonResult> JobRun::merged() const {
    // The tasks' results in the plan's order: the base is the first (FFmpeg's, with the CPU-only metrics, where
    // it has one: plan_tasks), and the others' scores are added to it.
    std::vector<const ComparisonResult*> parts;
    for (const TaskState& state : tasks_) {
        if (state.result) {
            parts.push_back(&*state.result);
        }
    }
    const ComparisonResult* ffmpeg = parts.empty() ? nullptr : parts.front();
    // The saved scores the run keeps, if it keeps any.
    const bool cached = job_.cached_result && !job_.cached_metrics.empty() && saved_scores_fit();
    std::optional<ComparisonResult> result;
    if (ffmpeg != nullptr) {
        result = *ffmpeg;
    } else if (cached) {
        // FFmpeg's metrics are all saved: the saved result is the base, with
        // its crops, frame count and the videos' details, but their film
        // grain as this run reads and scores them (one saved before either
        // was known has none, and its scores would be saved under another
        // identity).
        result = *job_.cached_result;
        take_film_grain(result->source_info, job_.source_info);
        take_film_grain(result->distorted_info, job_.distorted_info);
    }
    for (std::size_t part = 1; part < parts.size(); ++part) {
        const ComparisonResult& other = *parts[part];
        result->metric_results.add_all(other.metric_results);
        // Each VMAF's model comes from the task that calculated it: VMAF with
        // a custom model file is FFmpeg's.
        if (other.has_metric("vmaf")) {
            result->model = other.model;
        }
        if (other.has_metric("vmaf_v1")) {
            result->model_v1 = other.model_v1;
            result->model_choice_v1 = other.model_choice_v1;
        }
        if (other.has_metric("vmaf_hdr")) {
            result->model_hdr = other.model_hdr;
            result->model_choice_hdr = other.model_choice_hdr;
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
            // As the task scored them: a video whose size changes at the size
            // its crop is in (size_changes.h).
            mark_size_changes(result->source_info, result->distorted_info);
            result->scale_direction = options_.scale_direction;
            result->scale_algorithm = options_.scale_algorithm;
            result->compared_frame_count = perceptual->compared_frame_count;
            if (options_.resample_test) {  // as resample_test_on_gpu's
                result->distorted = synthetic_resample_distorted_path(job_.source_info.path, *options_.resample_test);
                result->resample_target = options_.resample_test;
                result->scale_algorithm = options_.resample_test->scaler;
            }
        }
        result->metric_results.add_all(perceptual->metrics);
    }
    if (result && cached) {
        // Saved metrics fill in only what this run did not calculate: a new
        // score always replaces the saved one.
        bool carried_vmaf = false;
        bool carried_v1 = false;
        bool carried_hdr = false;
        for (const std::string& key : job_.cached_metrics.keys()) {
            if (result->has_metric(key)) {
                continue;
            }
            result->metric_results.copy_from(job_.cached_metrics, key);
            carried_vmaf = carried_vmaf || key == "vmaf";
            carried_v1 = carried_v1 || key == "vmaf_v1";
            carried_hdr = carried_hdr || key == "vmaf_hdr";
        }
        // A saved VMAF keeps its model when the run calculated no VMAF (only
        // NEG, say, or only PSNR, its result the base).
        if (carried_vmaf && result->model.empty()) {
            result->model = job_.cached_result->model;
        }
        if (carried_v1 && !result->model_choice_v1) {
            result->model_v1 = job_.cached_result->model_v1;
            result->model_choice_v1 = job_.cached_result->model_choice_v1;
        }
        if (carried_hdr && !result->model_choice_hdr) {
            result->model_hdr = job_.cached_result->model_hdr;
            result->model_choice_hdr = job_.cached_result->model_choice_hdr;
        }
    }
    if (result) {
        make_vmaf_yuv(result->metric_results, job_.metric_keys);
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
                " at once, GPU metrics one video at a time" +
                (gpu_metrics_together ? ", each video's in one pass" : ""));
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
    if (parallel_jobs_.load() > 1 && queues_[cpu_pool].empty() && !queues_[gpu_pool].empty() && runs_.size() > 1) {
        // Issue #21: PSNR, SSIM and XPSNR in VMAF's GPU pass, say (plan_tasks).
        logger.info("Every task of this run is on the GPU, one video at a time: CPU metrics for several videos at "
                    "once do not apply");
    }
    // The GPU has one lane. The last lane runs on this thread. CPU lanes for
    // the CPU tasks that may run at once: a video's heavy_cpu tasks count as
    // one, which then has every core (share_cores).
    int cpu_tasks_at_once = 0;
    for (const std::unique_ptr<JobRun>& run : runs_) {
        bool heavy = false;
        for (std::size_t task = 0; task < run->task_count(); ++task) {
            if (run->pool_of(task) == cpu_pool) {
                const bool counted = !run->heavy_cpu(task) || !heavy;
                heavy = heavy || run->heavy_cpu(task);
                cpu_tasks_at_once += counted ? 1 : 0;
            }
        }
    }
    const int cpu_lanes = std::min(max_parallel_jobs, cpu_tasks_at_once);
    lane_count_ = std::max(1, cpu_lanes);
    std::vector<Pool> lanes(static_cast<std::size_t>(cpu_lanes), cpu_pool);
    if (!queues_[gpu_pool].empty()) {
        lanes.push_back(gpu_pool);
    }
    std::vector<std::thread> threads;
    for (std::size_t lane = 0; lane + 1 < lanes.size(); ++lane) {
        // Named in the log as the Python app named them.
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
            run->report_stopped();  // its facts, where nothing sent them: a video with no result
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
    // finish. A heavy_cpu task waits while its video runs another (plan_tasks
    // split them so that their CPU time and memory do not add up).
    std::deque<Picked>& queue = queues_[pool];
    for (auto entry = queue.begin(); entry != queue.end(); ++entry) {
        JobRun* run = entry->first;
        const bool heavy = run->heavy_cpu(entry->second);
        if (heavy && heavy_running_.contains(run->index())) {
            continue;
        }
        if (run->started || in_flight_.size() < max_videos_in_flight) {
            const Picked picked = *entry;
            queue.erase(entry);
            ++busy_[pool];
            if (heavy) {
                heavy_running_.insert(run->index());
            }
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
        if (run->heavy_cpu(task)) {
            heavy_running_.erase(run->index());
        }
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

// The size frames are compared at, where it is known before the run: with
// black bars kept, or cut as entered by hand. Pictures with their black bars
// cut are always even-sized.
std::optional<std::pair<int, int>> planned_size(const VideoInfo& source, const VideoInfo& test,
                                                const VmafOptions& options) {
    if (options.crop_mode == CropMode::none) {
        return analysis_dimensions(source, test, options);
    }
    if (options.crop_mode == CropMode::manual && !options.resample_test) {
        try {
            return analysis_dimensions(source, test, options,
                                       box_from_edges(options.manual_source_crop, source, "the source"),
                                       box_from_edges(options.manual_test_crop, test, "the test video"));
        } catch (const CropEntryError&) {
            // Not known: the run says why it cannot cut the crop.
        }
    }
    return std::nullopt;
}

// The metric `key` is part of when it is one that has a GPU route but runs on
// the CPU, or "" for any other. Those are heavy in CPU time and memory there:
// alone at 4K on the Core Ultra 9 285K, VMAF took 5,603 MB on the CPU, CAMBI
// 2,233 MB and CIEDE2000 1,250 MB, the CPU-only metrics 92 to 1,684 MB. VMAF YUV's U
// and V make one metric. `on_vulkan`: the task's CAMBI, CIEDE2000, MS-SSIM, U and V
// and HDR VMAF's features go to the Vulkan engine (VMAF's place for the task
// is the GPU and the engine scores VMAF here: cpu_plan,
// hdr_plan); `chroma_on_gpu`: the engine takes the U and V too.
std::string heavy_cpu_unit(const std::string& key, bool on_vulkan, bool chroma_on_gpu) {
    if (key == "vmaf" || key == "vmaf_neg" || key == "vmaf_v1") {
        return key;  // in a CPU task's keys only: the GPU's task scores them on the GPU
    }
    if (key == vmaf_u_key || key == vmaf_v_key) {
        return on_vulkan && chroma_on_gpu ? "" : "vmaf_yuv";
    }
    // MS-SSIM on the CPU stays with the CPU-only metrics, as before its GPU route: libvmaf's helper threads split
    // its pairs (cpu_threads), and a task of its own would decode the videos again.
    if (key.starts_with("ms_ssim")) {  // MS-SSIM, MS-SSIM YUV and its planes
        return "";
    }
    if (follows_vulkan_engine(key)) {
        return on_vulkan ? "" : key;
    }
    return "";
}

// `keys` (in the registry's order) with the heavy metrics (heavy_cpu_unit)
// after the first taken out: the keys left, whether they hold one, and the
// keys of each other one.
struct HeavySplit {
    std::vector<std::string> kept;
    bool kept_heavy = false;
    std::vector<std::vector<std::string>> apart;
};

HeavySplit split_heavy_units(const std::vector<std::string>& keys, bool on_vulkan, bool chroma_on_gpu) {
    HeavySplit split;
    std::vector<std::string> units;  // in the order they first come
    for (const std::string& key : keys) {
        const std::string unit = heavy_cpu_unit(key, on_vulkan, chroma_on_gpu);
        if (unit.empty()) {
            split.kept.push_back(key);
            continue;
        }
        auto at = std::find(units.begin(), units.end(), unit);
        if (at == units.end()) {
            units.push_back(unit);
            at = units.end() - 1;
            if (units.size() > 1) {
                split.apart.emplace_back();
            }
        }
        if (at == units.begin()) {
            split.kept.push_back(key);
            split.kept_heavy = true;
        } else {
            split.apart[static_cast<std::size_t>(at - units.begin()) - 1].push_back(key);
        }
    }
    return split;
}

}  // namespace

bool plan_asks_vulkan_engine(const VmafOptions& options, const std::vector<std::string>& metric_keys) {
    return options.vmaf_on_gpu &&
           std::any_of(metric_keys.begin(), metric_keys.end(),
                       [](const std::string& key) { return follows_vulkan_engine(key); });
}

GpuAnswers plan_gpu_answers(const VmafOptions& options, const std::vector<std::string>& metric_keys,
                            const PerceptualBackends& perceptual_backends) {
    GpuAnswers answers;
    if (plan_asks_vulkan_engine(options, metric_keys)) {
        answers.engine = engine_place(gpu_probes_token());
    }
    // SSIMULACRA2 and Butteraugli set to the GPU ("gpu" where not given) run on the CPU without one.
    const bool vship_asked = std::any_of(metric_keys.begin(), metric_keys.end(), [&](const std::string& key) {
        const auto chosen = perceptual_backends.find(key);
        return (key == "ssimulacra2" || key == "butteraugli") &&
               (chosen == perceptual_backends.end() || chosen->second != "cpu");
    });
    if (vship_asked) {
        answers.vship = detect_vship_device(options.vship_build, gpu_probes_token()).first.has_value();
    }
    return answers;
}

std::vector<std::string> fallback_notes(const std::vector<std::pair<std::string, std::string>>& failures,
                                        const std::vector<std::string>& calculated) {
    std::vector<std::pair<std::string, std::vector<std::string>>> by_reason;
    for (const auto& [key, why] : failures) {
        if (!contains(calculated, key)) {
            continue;  // a saved score: nothing moved to the CPU
        }
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

std::vector<MetricTask> plan_tasks(const VideoInfo& source, const VideoInfo& test, const VmafOptions& options,
                                   const std::vector<std::string>& metric_keys,
                                   const PerceptualBackends& perceptual_backends, const MetricResultSet& saved,
                                   bool share_decode, const GpuAnswers& gpu_answers) {
    // Each metric with a saved score is left out (the run's result takes the
    // saved one). Where the others are calculated is decided from every
    // metric asked for, saved or not, so that one calculated alone is
    // calculated where it would be with the rest, and its score is the same:
    // VMAF, NEG and v1 are placed together, on the GPU or not, as the run of
    // the whole family would place them, and the CPU's task keeps its place
    // (JobRun::execute_task).
    //
    // VMAF YUV and its Y are made from VMAF and its U and V (JobRun::merged),
    // VMAF asked for with them (with_metric_inputs): the U and V are
    // calculated in their place.
    const bool vmaf_yuv = contains(metric_keys, "vmaf_yuv") && !saved.has("vmaf_yuv");
    std::vector<std::string> keys;
    for (const MetricDefinition& metric : metrics()) {  // the registry's order
        const std::string key(metric.key);
        const bool asked = contains(metric_keys, key) || (vmaf_yuv && (key == vmaf_u_key || key == vmaf_v_key));
        if (asked && !saved.has(key) && key != "vmaf_yuv" && key != "vmaf_yuv_y") {
            keys.push_back(key);
        }
    }
    const bool vmaf = contains(metric_keys, "vmaf");
    const bool neg = contains(metric_keys, "vmaf_neg");
    const bool v1 = contains(metric_keys, "vmaf_v1");
    const bool vmaf_family_left = contains(keys, "vmaf") || contains(keys, "vmaf_neg") || contains(keys, "vmaf_v1");
    const bool chroma_left = contains(keys, vmaf_u_key) || contains(keys, vmaf_v_key);
    const int depth = bit_depth(analysis_pix_fmt({source.pix_fmt, test.pix_fmt}));
    const std::optional<std::pair<int, int>> size =
        vmaf_family_left || chroma_left ? planned_size(source, test, options) : std::nullopt;
    GpuModels on_gpu;
    std::vector<std::pair<std::string, std::string>> check_failures;  // (key, why) of scores_on_gpu
    if (vmaf_family_left) {
        // Planned before the run, whose checks of the GPU have answered (the
        // window waits for them, the command line asks as it starts): none
        // of the run's to cancel.
        on_gpu = scores_on_gpu(vmaf, neg, options.model, options.vmaf_on_gpu, depth, size, gpu_probes_token(), v1,
                               options.model_v1, &check_failures);
    }
    MetricTask ffmpeg{"ffmpeg", {}};
    ffmpeg.fallbacks = fallback_notes(check_failures, keys);
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
    // The CPU's task keeps VMAF, NEG or v1 (a model file, say), and with it the metrics that follow VMAF's place.
    const bool vmaf_on_cpu = contains(ffmpeg.metric_keys, "vmaf") || contains(ffmpeg.metric_keys, "vmaf_neg") ||
                             contains(ffmpeg.metric_keys, "vmaf_v1");
    const bool engine = options.vmaf_on_gpu && (!gpu_answers.engine || gpu_answers.engine->on_engine);
    // CIEDE2000, CAMBI, MS-SSIM and MS-SSIM YUV where the Vulkan engine takes them make a GPU pass of their own
    // (MetricTask::engine_lane), in the GPU's queue after VMAF's: never on the GPU at the same time as VMAF's pass,
    // whose memory a smaller GPU may not have room beside, and the CPU-only metrics do not wait for them. Each pass
    // decodes the videos itself. Those it cannot take at a size
    // known before the run (gpu_metrics_of: MS-SSIM YUV at an odd size) stay with the CPU-only metrics; CAMBI goes,
    // its options aside, and is scored on the CPU in the lane where the GPU cannot take them.
    MetricTask engines{"ffmpeg", {}};
    engines.engine_lane = true;
    if (engine && !vmaf_on_cpu) {
        std::vector<std::string> candidates;
        std::copy_if(ffmpeg.metric_keys.begin(), ffmpeg.metric_keys.end(), std::back_inserter(candidates),
                     [](const std::string& key) { return engine_takes_cpu_metric(key); });
        if (!candidates.empty()) {
            const std::optional<std::pair<int, int>> at = size ? size : planned_size(source, test, options);
            const bool odd = at && (at->first % 2 != 0 || at->second % 2 != 0);
            const std::vector<std::string> taken = gpu_metrics_of(candidates, odd, {});
            for (const std::string& key : candidates) {  // in the registry's order
                if (contains(taken, key)) {
                    engines.metric_keys.push_back(key);
                    std::erase(ffmpeg.metric_keys, key);
                }
            }
        }
    }
    // PSNR, SSIM, XPSNR and HDR VMAF beside VMAF on the GPU, from the GPU's decode, when both parts still run
    // (score_hdr_beside, score_gpu_and_cpu). Not metrics that cannot share its pass: in the GPU's task they would
    // decode the videos again while holding the GPU's queue. HDR VMAF does not change which those are.
    std::vector<std::string> cpu_keys = ffmpeg.metric_keys;
    std::erase(cpu_keys, "vmaf_hdr");
    if (share_decode && !cpu_metrics_need_every_frame(cpu_keys, options.n_subsample) &&
        !gpu.metric_keys.empty() && !ffmpeg.metric_keys.empty() &&
        std::all_of(ffmpeg.metric_keys.begin(), ffmpeg.metric_keys.end(),
                    [](const std::string& key) { return is_cpu_metric(key) || key == "vmaf_hdr"; })) {
        gpu.cpu_beside = std::move(ffmpeg.metric_keys);
        ffmpeg.metric_keys.clear();
        gpu.metric_keys.clear();
        for (const std::string& key : keys) {  // in the registry's order
            if (!is_perceptual_metric(key) && !contains(engines.metric_keys, key)) {
                gpu.metric_keys.push_back(key);
            }
        }
    }
    // At most one metric that has a GPU route but runs on the CPU in a task (heavy_cpu_unit): FFmpeg's task keeps
    // the first with the CPU-only metrics, the GPU's task the first beside VMAF, and each other one is calculated
    // in a task of FFmpeg's own. The CPU's task with VMAF, NEG or v1 keeps the metrics that follow VMAF's place on
    // the CPU (JobRun::execute_task), and so do the tasks split from it. A size not known before the run is taken
    // for one whose U and V the engine takes (the black bars cut leave an even size).
    const bool chroma_on_gpu = !size || chroma_vmaf_fits_gpu(size->first, size->second, depth, options.model);
    std::vector<MetricTask> split_off;
    const auto split = [&](std::vector<std::string>& task_keys, bool on_vulkan, bool vmaf_cpu) {
        HeavySplit parts = split_heavy_units(task_keys, on_vulkan, chroma_on_gpu);
        task_keys = std::move(parts.kept);
        for (std::vector<std::string>& part : parts.apart) {
            MetricTask task{"ffmpeg", std::move(part)};
            task.vmaf_on_cpu = vmaf_cpu;
            task.heavy_cpu = true;
            split_off.push_back(std::move(task));
        }
        return parts.kept_heavy;
    };
    ffmpeg.vmaf_on_cpu = vmaf_on_cpu;
    ffmpeg.heavy_cpu = split(ffmpeg.metric_keys, engine && !vmaf_on_cpu, vmaf_on_cpu);
    const std::size_t from_ffmpeg = split_off.size();
    gpu.heavy_cpu = split(gpu.cpu_beside, engine, false);
    for (std::size_t task = from_ffmpeg; task < split_off.size(); ++task) {
        for (const std::string& key : split_off[task].metric_keys) {
            std::erase(gpu.metric_keys, key);  // no longer beside VMAF
        }
    }
    // SSIMULACRA2 and Butteraugli on the CPU, set there or by Vship finding no GPU: one at a time in their task
    // (MetricRouting::run_on_cpu).
    perceptual_cpu.heavy_cpu = true;
    vship.heavy_cpu = !gpu_answers.vship && (contains(vship.metric_keys, "ssimulacra2") ||
                                             contains(vship.metric_keys, "butteraugli"));
    std::vector<MetricTask> tasks;
    const auto add = [&tasks](MetricTask& task) {
        if (!task.metric_keys.empty()) {
            tasks.push_back(std::move(task));
        }
    };
    add(ffmpeg);
    std::for_each(split_off.begin(), split_off.begin() + static_cast<std::ptrdiff_t>(from_ffmpeg), add);
    add(gpu);
    std::for_each(split_off.begin() + static_cast<std::ptrdiff_t>(from_ffmpeg), split_off.end(), add);
    add(engines);
    add(vship);
    add(perceptual_cpu);
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

CropPair job_crops(const VmafJob& job, const std::string& ffmpeg, const CancelToken& cancel) {
    const VmafOptions& options = job.options;
    if (options.resample_test) {  // both of its sides are the source's, cut by its own box
        const std::optional<CropBox> own = source_own_crop(options.crop_choice(), ffmpeg, job.source_info, cancel,
                                                           std::nullopt, options.scale_algorithm);
        return {own, own};
    }
    return resolve_crops(ffmpeg, job.source_info, job.distorted_info, options.crop_choice(), options.scale_algorithm,
                         cancel, {});
}

void calculate_saved_scores_of_other_bars(std::vector<VmafJob>& jobs, const std::string& ffmpeg,
                                          const CancelToken& cancel) {
    for (VmafJob& job : jobs) {
        if (!job.cached_result || job.cached_metrics.empty()) {
            continue;
        }
        const VmafOptions& options = job.options;
        CropPair now;
        try {
            now = job_crops(job, ffmpeg, cancel);
        } catch (const Cancelled&) {
            return;
        } catch (const std::exception&) {
            continue;  // left as planned: its run finds them again, and says why it cannot
        }
        if (cuts_as_saved(*job.cached_result, job.source_info, job.distorted_info, now.first, now.second)) {
            continue;
        }
        std::vector<MetricTask> tasks;
        try {
            tasks = plan_tasks(job.source_info, job.distorted_info, options, job.metric_keys, job.perceptual_backends,
                               MetricResultSet(), job.share_decode, job.gpu_answers);
        } catch (const std::exception&) {
            continue;  // as planned, its saved scores left out of the result (JobRun::saved_scores_fit)
        }
        logger.info(job.label + ": its saved " + metric_labels(job.cached_metrics.keys()) +
                    " were made with other black bars than this run cuts: calculated again with the rest");
        job.tasks = std::move(tasks);
        job.cached_result.reset();
        job.cached_metrics = MetricResultSet();
    }
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

    std::vector<VmafJob> jobs;  // planned anew only before the run (calculate_saved_scores_of_other_bars)
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

void JobScheduler::calculate_saved_scores_of_other_bars(const std::string& ffmpeg) {
    vml::calculate_saved_scores_of_other_bars(parts_->jobs, ffmpeg, parts_->cancel);
}

void JobScheduler::set_parallel_jobs(int count) { parts_->scheduler.set_parallel_jobs(count); }

int JobScheduler::parallel_jobs() const { return parts_->scheduler.parallel_jobs(); }

}  // namespace vml
