// The line that says how far a video's metrics are, for example
//
//     CPU metrics 1–4 of 4: VMAF v1, PSNR, SSIM, XPSNR 23.3% (9.8 fps, 0:00:22 remaining)
//
// built from snapshots of the video's tasks (job_runner.h): which of its
// metrics are being calculated, where (on the CPU or the GPU) and how far
// along they are. A port of vmaf_app/ui/run_line.py's line; its tooltip and
// "Decoder: ..." text come with the window, and Vship's passes with Vship.
#pragma once

#include <string>
#include <vector>

#include "vml/ffmpeg_run.h"
#include "vml/job_runner.h"

namespace vml {

// The step a task is on, from its status, shown before any progress
// figures arrive: for example "GPU decode failed, retrying". Nothing for a
// status that only says FFmpeg is starting. The decode plan is left out.
std::string step_text(const RunStatus& status);

// Metric numbers as the line shows them: "3", "1–2" or "1, 3".
std::string numbers_text(std::vector<int> numbers);

// A video's line, one part per place, the CPU's and then the GPU's: the
// metrics under way, each numbered among that place's metrics, with their
// progress, for example "GPU metrics 1–2 of 2: VMAF v0.6.1, VMAF NEG 50.0%
// (55.0 fps, 0:00:10 remaining)", or what they wait for. `paused` puts
// "paused" in place of the speed and the time left.
std::vector<std::string> metric_line(const std::vector<TaskSnapshot>& tasks, bool paused);

}  // namespace vml
