// A run of the window's videos (the Python app's vmaf_app/ui/worker.py): the
// job runner's JobScheduler (job_runner.h) on a thread of its own, its
// events sent on as Qt signals. The window's connections to them are queued,
// so they reach it on its own thread, in the order they were sent.
#pragma once

#include <QMetaType>
#include <QString>
#include <QThread>

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "vml/cancel.h"
#include "vml/ffmpeg_run.h"
#include "vml/job_runner.h"
#include "vml/settings.h"

namespace vml::app {

class RunWorker : public QThread {
    Q_OBJECT

public:
    // Runs `jobs`: the CPU's metrics of up to `parallel_jobs` videos at once,
    // each video's GPU metrics in one pass when `gpu_metrics_together`
    // (otherwise one at a time), with the FFmpeg `settings` find, which is
    // checked as the run starts.
    RunWorker(std::vector<VmafJob> jobs, int parallel_jobs, bool gpu_metrics_together, Settings settings,
              QObject* parent = nullptr);
    ~RunWorker() override;

    // From the window's thread, at any time.
    void cancel();
    void pause();
    void resume();
    void set_parallel_jobs(int count);

signals:
    // The run's events (job_runner.h's RunEvents), each with its job's index;
    // then all_finished, the last.
    void job_started(int index);
    void task_progress(int index, const std::vector<vml::TaskSnapshot>& tasks);
    void result_updated(int index, const vml::ComparisonResult& result);
    void job_finished(int index, const vml::ComparisonResult& result);
    void job_partially_failed(int index, const vml::ComparisonResult& result, const QString& message,
                              const QString& stderr_tail, const vml::MetricFailures& failures);
    void job_failed(int index, const QString& message, const QString& stderr_tail);
    void cancelled();
    void all_finished();

protected:
    void run() override;

private:
    // Notes that job `index` has ended, finished or failed.
    void ended(std::size_t index);
    // Fails, with `message`, every job that has not ended.
    void fail_the_rest(const std::string& message);

    const std::size_t job_count_;
    const bool gpu_decode_;  // whether a job may decode a video on the GPU
    const Settings settings_;
    CancelToken cancel_;
    // What the run needs to know about this PC: found as it starts, before
    // any task reads it.
    RunEnvironment environment_;
    std::unique_ptr<JobScheduler> scheduler_;
    std::mutex ended_mutex_;
    std::set<std::size_t> ended_;
};

}  // namespace vml::app

Q_DECLARE_METATYPE(std::vector<vml::TaskSnapshot>)
Q_DECLARE_METATYPE(vml::ComparisonResult)
Q_DECLARE_METATYPE(vml::MetricFailures)
