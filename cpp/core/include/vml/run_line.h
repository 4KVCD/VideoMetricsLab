// The line that says how far a video's metrics are, for example
//
//     CPU metrics 1–4 of 4: VMAF v1, PSNR, SSIM, XPSNR 23.3% (9.8 fps, 0:00:22 remaining)
//
// built from snapshots of the video's tasks (job_runner.h): which of its
// metrics are being calculated, where (on the CPU or the GPU) and how far
// along they are; with its tooltip and its "Decoder: ..." text. A port of
// vmaf_app/ui/run_line.py. Its words are English, as the command line shows
// them, or the window's language through a Wording.
#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "vml/ffmpeg_run.h"
#include "vml/hardware_decoding.h"
#include "vml/job_runner.h"

namespace vml {

// A value for a text's {field}: text, a whole number, or a number with a
// fraction; and a text's values, by their fields' names.
using WordingValue = std::variant<std::string, long long, double>;
using WordingValues = std::vector<std::pair<std::string, WordingValue>>;

// How the line is put into words. `text` gives an English text, with its
// {fields} as the Python app's tr() takes it, in the reader's language with
// the fields filled from `values`; `message` gives one of the engine's
// messages in the reader's language (the window's tr_message).
struct Wording {
    std::function<std::string(std::string_view english, const WordingValues& values)> text;
    std::function<std::string(std::string_view message)> message;
};

// The English, as the command line shows it: each text as it is, its fields
// filled in as Python's str.format fills them ("{name}", "{name:.1f}").
const Wording& english();

// The step a task is on, from its status, shown before any progress
// figures arrive: for example "GPU decode failed, retrying". Nothing for a
// status that only says FFmpeg is starting. The decode plan is left out.
std::string step_text(const RunStatus& status, const Wording& wording = english());

// Metric numbers as the line shows them: "3", "1–2" or "1, 3".
std::string numbers_text(std::vector<int> numbers);

// A video's line, its tooltip, and where its tasks decode.
struct MetricLine {
    // One part per place, the CPU's and then the GPU's: the metrics under
    // way, each numbered among that place's metrics, with their progress,
    // for example "GPU metrics 1–2 of 2: VMAF v0.6.1, VMAF NEG 50.0% (55.0
    // fps, 0:00:10 remaining)", or what they wait for.
    std::vector<std::string> parts;
    // Every metric on a line of its own, with its progress.
    std::string tooltip;
    // Each decoding task's plan, by the task's name on the line ("CPU
    // metrics"), in the tasks' order. Where two tasks of one place decode
    // differently, each name gives its metrics' numbers ("CPU metrics 1–4").
    std::vector<std::pair<std::string, HardwareDecoding>> plans;
};

// A video's line (the Python app's metric_line). `paused` puts "paused" in
// place of the speed and the time left.
MetricLine metric_line(const std::vector<TaskSnapshot>& tasks, bool paused, const Wording& wording = english());

// Where each video is decoded, as the line shows it: "Decoder: Source: GPU,
// test video: CPU", each task named where they differ ("test video: CPU (CPU
// metrics) / GPU (GPU metrics)"). `source_only` for a resolution test, which
// decodes only the source.
std::string decoder_text(const std::vector<std::pair<std::string, HardwareDecoding>>& plans, bool source_only,
                         const Wording& wording = english());

}  // namespace vml
