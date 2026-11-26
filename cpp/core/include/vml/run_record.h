// What a results file records of how its scores were made (issue #14), so
// that it is a complete record on its own: the run (interface, command line,
// times), the machine that scored it, and for each video its settings as
// resolved (defaults included), its black bars and what they were decided
// from, how each metric asked for came out, what each pass decoded with,
// and how long each pass took. The command line's --json and the window's
// Export results both write them with these builders.
#pragma once

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/crop_detect.h"
#include "vml/cvvdp.h"
#include "vml/hardware_decoding.h"
#include "vml/options.h"
#include "vml/results.h"
#include "vml/video.h"
#include "vml/vship.h"

namespace vml {

// One pass of a video's task as it ran: a task's metrics, or one of Vship's
// passes.
struct PassTiming {
    std::string task;  // the task's backend (MetricTask::backend)
    std::vector<std::string> metric_keys;
    // "gpu" or "cpu": the task's, the CPU when the GPU's VMAF fell back to it, the GPU when it calculated all
    // the pass's metrics
    std::string on;
    double seconds = 0.0;  // start to end, pauses included
    double paused_seconds = 0.0;
    // From its start to its first frame: black-bar detection it waited for,
    // its programs starting, the GPU's turn (all of it where it scored none;
    // unknown where it scored frames but reported none before its end).
    std::optional<double> setup_seconds;
    long long frames = 0;  // the frame pairs (CVVDP: frames) it went through
    double fps = 0.0;      // frames over the time from its first frame to its end, pauses left out; 0: none
    std::string end;       // "done", "failed" or "cancelled"
    // The decode plan the pass's statuses named last: what it was set to
    // decode each video with, after the run's own fallbacks.
    std::optional<HardwareDecoding> decode;
    SoftwareDecodingWhy decode_why;  // why each video `decode` left in software was
    bool operator==(const PassTiming&) const = default;
};

// How a metric asked for came out of a run.
inline constexpr const char* metric_calculated = "calculated";
inline constexpr const char* metric_saved = "saved";  // taken from saved scores
inline constexpr const char* metric_failed = "failed";
inline constexpr const char* metric_cancelled = "cancelled";
inline constexpr const char* metric_not_run = "not run";
// The metric does not apply to the videos (HDR VMAF of a video that is not
// PQ): left out, not a failure.
inline constexpr const char* metric_not_applicable = "not applicable";

// The codes a why comes with when the outcome gives none (MetricOutcome::code):
// stable words for a program to act on, where the why is for a person.
inline constexpr const char* calculation_failed_code = "calculation_failed";  // failed
inline constexpr const char* not_planned_code = "not_planned";                // not run
inline constexpr const char* not_applicable_code = "not_applicable";          // not applicable

struct MetricOutcome {
    std::string status;  // one of the above
    std::string why;     // for failed, not run and not applicable
    // The why's code (docs/RESULTS_JSON.md lists them); empty: the status's
    // own (outcome_code).
    std::string code{};
    bool operator==(const MetricOutcome&) const = default;
};

// The code written beside `outcome`'s why: its own, else its status's;
// empty for an outcome with no why.
std::string outcome_code(const MetricOutcome& outcome);

// What a video's comparison was asked to do (its VmafJob's).
struct VideoSettings {
    VideoInfo source;
    VideoInfo test;
    VmafOptions options;
    std::vector<std::string> metrics;  // asked for, in order
    // Where SSIMULACRA2 and Butteraugli run ("gpu" or "cpu" per key).
    std::map<std::string, std::string, std::less<>> perceptual_backends;
    CvvdpSettings cvvdp;
};

// One video's run, as the job runner saw it (RunEvents::job_facts).
struct RunFacts {
    VideoSettings settings;
    std::string started_utc;  // when the video started (utc_text)
    double seconds = 0.0;     // from its start to its end, pauses included
    double paused_seconds = 0.0;
    std::vector<PassTiming> passes;
    // Each metric asked for, in the order asked.
    std::vector<std::pair<std::string, MetricOutcome>> metrics;
};

// The machine a run scored on.
struct MachineFacts {
    std::string os;   // platform::system_description
    std::string cpu;  // platform::cpu_model_name
    // Every GPU the Vulkan loader lists, and the one Vship's metrics run on
    // ("" when no probe has answered).
    std::vector<VulkanGpu> gpus;
    std::string vship_gpu;
    std::string ffmpeg_version;  // "9.0.1", or "unknown"
    std::string ffmpeg_path;
};

// This machine, as far as it is known now without waiting: the GPUs from
// the last Vship probe that answered (none before one has).
MachineFacts this_machine(const std::string& ffmpeg_path, const std::string& ffmpeg_version);

// The run a document's scores come from.
struct RunRecord {
    std::string user_interface;  // "cli" or "gui" ('interface' is a macro in Windows' headers)
    // The command line as given, for the command line's run.
    std::optional<std::vector<std::string>> command_line;
    // When it started and finished (utc_text), and how long it took, pauses
    // included and apart. The window's export may hold videos of several of
    // its runs (`runs`): the first's start, the last's end, and the runs'
    // times added up.
    std::string started_utc;
    std::string finished_utc;
    std::optional<double> wall_seconds;
    std::optional<double> paused_seconds;
    int runs = 1;
    MachineFacts machine;
};

// The document's "run", "machine" and "summary_rules" (how each metric's
// score and statistics are worked out from its frames).
void add_run_record(nlohmann::ordered_json& document, const RunRecord& record);

// A video's record, added to its entry, from its run's `facts` and its
// `result` (either may be missing):
// - "settings": every setting that changes its scores or how to read them,
//   defaults included: the metrics, the models as resolved, CVVDP's display
//   (named `cvvdp_display`, "" for a display that is no preset), each
//   metric's frame coverage, the range asked and the frames compared, the
//   frame pairing, the scaling each side had, decoding, where the metrics ran;
// - "black_bars": the mode (and the crops entered by hand), and for each video
//   the box detected, what it was decided from, whether the store already
//   had it when the video started, and the box compared (common_picture may
//   have moved it), from kept_crop_answer;
// - "metric_status", "decode_plan" and "timings".
// Each is null where the facts it comes from are missing (a saved run
// loaded); the last two also for a video that never started (its saved
// scores answered it, or the run was cancelled first). With `result`, also,
// facts or none: "whole_file" (scores_of_a_range) and "resolution_test"
// (its target, or null), which BD-rate's import reads.
void add_video_record(nlohmann::ordered_json& entry, const RunFacts* facts, const ComparisonResult* result,
                      const std::string& cvvdp_display);

}  // namespace vml
