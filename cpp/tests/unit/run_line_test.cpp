// The line that says how far a video's metrics are (run_line.h), held to
// the Python app's own (run_line.json): from snapshots of the video's tasks,
// with its tooltip, its "Decoder: ..." text, and the steps and metric
// numbers it shows.
#include "vml/run_line.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "reference.h"
#include "vml/ffmpeg_run.h"

namespace vml {
namespace {

using Json = nlohmann::json;

RunStatus status_from(const Json& item) {
    const std::string kind = item["kind"].get<std::string>();
    return {item["text"].get<std::string>(),
            kind == "starting"          ? StatusKind::starting
            : kind == "gpu_pass"        ? StatusKind::gpu_pass
            : kind == "gpu_wait"        ? StatusKind::gpu_wait
            : kind == "gpu_vmaf_failed" ? StatusKind::gpu_vmaf_failed
                                        : StatusKind::step,
            item["brief"].get<std::string>()};
}

TaskSnapshot snapshot_from(const Json& item) {
    TaskSnapshot snapshot;
    snapshot.backend = item["backend"].get<std::string>();
    snapshot.metric_keys = item["metric_keys"].get<std::vector<std::string>>();
    snapshot.lane = item["lane"].get<std::string>();
    snapshot.cpu_keys = item["cpu_keys"].get<std::vector<std::string>>();
    snapshot.done_keys = item["done_keys"].get<std::vector<std::string>>();
    snapshot.passes = item["passes"].get<std::vector<std::vector<std::string>>>();
    if (!item["phase"].is_null()) {
        snapshot.phase = item["phase"].get<std::array<long long, 3>>();
    }
    snapshot.current = item["current"].get<long long>();
    snapshot.total = item["total"].get<long long>();
    snapshot.fps = test::reference_number(item["fps"]).value();
    snapshot.state = item["state"].get<std::string>();
    snapshot.waiting_for = item["waiting_for"].get<std::string>();
    snapshot.step = status_from(item["step"]);
    if (const Json& decode = item["decode"]; !decode.is_null()) {
        const auto hwaccel = [](const Json& value) {
            return value.is_null() ? std::nullopt : std::optional(value.get<std::string>());
        };
        snapshot.decode = HardwareDecoding{hwaccel(decode["source"]), hwaccel(decode["distorted"])};
    }
    return snapshot;
}

TEST(RunLine, IsThePythonAppsLine) {
    const Json reference = test::load_reference("run_line.json");
    ASSERT_FALSE(reference["lines"].empty());
    for (const Json& item : reference["lines"]) {
        std::vector<TaskSnapshot> tasks;
        for (const Json& task : item["tasks"]) {
            tasks.push_back(snapshot_from(task));
        }
        const std::string name = item["name"].get<std::string>();
        const MetricLine line = metric_line(tasks, item["paused"].get<bool>());
        EXPECT_EQ(line.parts, item["line"].get<std::vector<std::string>>()) << name;
        EXPECT_EQ(line.tooltip, item["tooltip"].get<std::string>()) << name;
        EXPECT_EQ(line.plans.empty() ? "" : decoder_text(line.plans, false), item["decoder"].get<std::string>())
            << name;
        EXPECT_EQ(line.plans.empty() ? "" : decoder_text(line.plans, true),
                  item["decoder_source_only"].get<std::string>())
            << name;
    }
    for (const Json& item : reference["steps"]) {
        EXPECT_EQ(step_text(status_from(item["status"])), item["text"].get<std::string>()) << item["status"].dump();
    }
    for (const Json& item : reference["numbers"]) {
        EXPECT_EQ(numbers_text(item["numbers"].get<std::vector<int>>()), item["text"].get<std::string>());
    }
}

// A task in the CPU's queue whose scorer gave CIEDE2000 and CAMBI to the GPU
// (gpu_keys, from a metrics_on_gpu status): the line names them GPU metrics,
// as the results do, with the pass's figures; the rest stay CPU metrics.
TEST(RunLine, MetricsTheGpuTakesFromTheCpusTaskAreGpuMetrics) {
    TaskSnapshot task;
    task.backend = "cpu";
    task.lane = "cpu";
    task.metric_keys = {"psnr", "cambi", "ciede2000"};
    task.passes = {task.metric_keys};
    task.state = "running";
    task.current = 300;
    task.total = 1000;
    task.fps = 30.0;
    EXPECT_EQ(metric_line({task}, false).parts,
              (std::vector<std::string>{"CPU metrics 1–3 of 3: PSNR, CAMBI, CIEDE2000 (ΔE) 30.0% (30.0 fps, "
                                        "0:00:23 remaining)"}));
    task.gpu_keys = {"ciede2000", "cambi"};
    EXPECT_EQ(metric_line({task}, false).parts,
              (std::vector<std::string>{
                  "CPU metrics 1 of 1: PSNR 30.0% (30.0 fps, 0:00:23 remaining)",
                  "GPU metrics 1–2 of 2: CAMBI, CIEDE2000 (ΔE) 30.0% (30.0 fps, 0:00:23 remaining)"}));
    // Done: each where it was calculated.
    task.state = "done";
    task.done_keys = task.metric_keys;
    const std::string tooltip = metric_line({task}, false).tooltip;
    EXPECT_NE(tooltip.find("GPU metrics"), std::string::npos) << tooltip;
    EXPECT_NE(tooltip.find("CPU metrics"), std::string::npos) << tooltip;
}

// The status a CPU scorer sends as it starts, both ways.
TEST(RunLine, MetricsOnGpuStatusTextGoesBothWays) {
    const std::string text = metrics_on_gpu_text({"psnr", "cambi", "ciede2000"}, {"cambi", "ciede2000"});
    EXPECT_EQ(text, "psnr,cambi,ciede2000;cambi,ciede2000");
    const auto [metrics, on_gpu] = metrics_on_gpu_from(text);
    EXPECT_EQ(metrics, (std::vector<std::string>{"psnr", "cambi", "ciede2000"}));
    EXPECT_EQ(on_gpu, (std::vector<std::string>{"cambi", "ciede2000"}));
    const auto [none_metrics, none_on_gpu] = metrics_on_gpu_from(metrics_on_gpu_text({"cambi"}, {}));
    EXPECT_EQ(none_metrics, std::vector<std::string>{"cambi"});
    EXPECT_TRUE(none_on_gpu.empty());
}

}  // namespace
}  // namespace vml
