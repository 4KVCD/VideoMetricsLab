// The line that says how far a video's metrics are (run_line.h), held to
// the Python app's own (run_line.json): from snapshots of the video's tasks,
// with the steps and metric numbers it shows.
#include "vml/run_line.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "reference.h"

namespace vml {
namespace {

using Json = nlohmann::json;

RunStatus status_from(const Json& item) {
    const std::string kind = item["kind"].get<std::string>();
    return {item["text"].get<std::string>(),
            kind == "starting"          ? StatusKind::starting
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
    snapshot.current = item["current"].get<long long>();
    snapshot.total = item["total"].get<long long>();
    snapshot.fps = test::reference_number(item["fps"]).value();
    snapshot.state = item["state"].get<std::string>();
    snapshot.waiting_for = item["waiting_for"].get<std::string>();
    snapshot.step = status_from(item["step"]);
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
        EXPECT_EQ(metric_line(tasks, item["paused"].get<bool>()), item["line"].get<std::vector<std::string>>())
            << item["name"].get<std::string>();
    }
    for (const Json& item : reference["steps"]) {
        EXPECT_EQ(step_text(status_from(item["status"])), item["text"].get<std::string>()) << item["status"].dump();
    }
    for (const Json& item : reference["numbers"]) {
        EXPECT_EQ(numbers_text(item["numbers"].get<std::vector<int>>()), item["text"].get<std::string>());
    }
}

}  // namespace
}  // namespace vml
