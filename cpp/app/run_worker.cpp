#include "run_worker.h"

#include <algorithm>
#include <exception>
#include <utility>

#include "vml/app_log.h"
#include "vml/ffmpeg_tools.h"
#include "vml/gpu_vmaf.h"
#include "vml/hardware_decoding.h"
#include "vml/perceptual.h"
#include "vml/platform/system.h"
#include "vml/vship_run.h"

namespace vml::app {

RunWorker::RunWorker(std::vector<VmafJob> jobs, int parallel_jobs, bool gpu_metrics_together, Settings settings,
                     QObject* parent)
    : QThread(parent),
      job_count_(jobs.size()),
      gpu_decode_(std::any_of(jobs.begin(), jobs.end(), [](const VmafJob& job) { return job.options.gpu_decode; })),
      settings_(std::move(settings)) {
    qRegisterMetaType<std::vector<vml::TaskSnapshot>>();
    qRegisterMetaType<vml::ComparisonResult>();
    qRegisterMetaType<vml::MetricFailures>();
    RunEvents events;
    events.job_started = [this](std::size_t index) { emit job_started(static_cast<int>(index)); };
    events.task_progress = [this](std::size_t index, const std::vector<TaskSnapshot>& tasks) {
        emit task_progress(static_cast<int>(index), tasks);
    };
    events.result_updated = [this](std::size_t index, const ComparisonResult& result) {
        emit result_updated(static_cast<int>(index), result);
    };
    events.job_finished = [this](std::size_t index, const ComparisonResult& result) {
        ended(index);
        emit job_finished(static_cast<int>(index), result);
    };
    events.job_partially_failed = [this](std::size_t index, const ComparisonResult& result, const std::string& message,
                                         const std::string& stderr_tail, const MetricFailures& failures) {
        ended(index);
        emit job_partially_failed(static_cast<int>(index), result, QString::fromStdString(message),
                                  QString::fromStdString(stderr_tail), failures);
    };
    events.job_failed = [this](std::size_t index, const std::string& message, const std::string& stderr_tail) {
        ended(index);
        emit job_failed(static_cast<int>(index), QString::fromStdString(message), QString::fromStdString(stderr_tail));
    };
    events.cancelled = [this] { emit cancelled(); };
    scheduler_ = std::make_unique<JobScheduler>(
        std::move(jobs), parallel_jobs, std::move(events),
        [this](const VmafJob& job, const VmafOptions& options, const ProgressCallback& on_progress,
               const StatusCallback& on_status, const CancelToken& token) {
            // A resolution test has no test video of its own: it is made from
            // the source during the run.
            if (options.resample_test) {
                return run_resample_test(environment_, job.source_info, options, on_progress, on_status, token);
            }
            return run_vmaf(environment_, job.source_info, job.distorted_info, options, on_progress, on_status, token,
                            job.result_distorted_path);
        },
        [this](const VmafJob& job, const std::vector<std::string>& keys, bool together, int concurrent_cpu_tasks,
               const PerceptualHooks& hooks, const CancelToken& token) {
            const PerceptualRequest request{environment_, job.source_info, job.distorted_info,
                                            job.options,  job.perceptual_backends, job.cvvdp,
                                            concurrent_cpu_tasks};
            return run_perceptual_task(request, keys, together, hooks, token);
        },
        gpu_metrics_together, platform::logical_cores(), cancel_);
}

RunWorker::~RunWorker() = default;

// The run's own lines in the log, as the Python app's JobScheduler writes
// them.
const app_log::Logger run_logger("vml.job_runner");

void RunWorker::cancel() {
    run_logger.info("Run: cancel requested");
    cancel_.cancel();
}

void RunWorker::pause() {
    run_logger.info("Run: paused");
    cancel_.pause();
}

void RunWorker::resume() {
    run_logger.info("Run: resumed");
    cancel_.resume();
}

void RunWorker::set_parallel_jobs(int count) { scheduler_->set_parallel_jobs(count); }

void RunWorker::run() {
    // A background boundary: whatever goes wrong ends the videos the run had
    // not ended, with why, rather than the program.
    try {
        const ToolsStatus tools = check_tools(settings_);
        if (!tools.ok()) {
            std::string why = "FFmpeg cannot be used:";
            for (const std::string& problem : tools.problems()) {
                why += " " + problem;
            }
            fail_the_rest(why);
        } else {
            // A GPU that failed its probe a while ago is asked again: the
            // failure may have passed.
            forget_failed_gpu_vmaf_probe();
            // Which decoders this FFmpeg and these GPUs have is only asked
            // when a video may be decoded on the GPU.
            environment_ = RunEnvironment{
                tools.ffmpeg.path, format_version(tools.ffmpeg.version), tools.ffprobe.path,
                gpu_decode_ ? DecodingCapabilities::detect(tools.ffmpeg.path) : DecodingCapabilities{},
                platform::logical_cores()};
            scheduler_->run();
        }
    } catch (const std::exception& error) {
        fail_the_rest(error.what());
    }
    emit all_finished();
}

void RunWorker::ended(std::size_t index) {
    const std::lock_guard lock(ended_mutex_);
    ended_.insert(index);
}

void RunWorker::fail_the_rest(const std::string& message) {
    for (std::size_t index = 0; index < job_count_; ++index) {
        bool open = false;
        {
            const std::lock_guard lock(ended_mutex_);
            open = ended_.insert(index).second;
        }
        if (open) {
            emit job_failed(static_cast<int>(index), QString::fromStdString(message), QString());
        }
    }
}

}  // namespace vml::app
