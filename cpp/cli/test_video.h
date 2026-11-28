// One test video of a command-line run: what the window keeps in a row.
#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gates.h"
#include "vml/analysis_request.h"
#include "vml/options.h"
#include "vml/results.h"
#include "vml/run_record.h"
#include "vml/video.h"

namespace vml::cli {

// A frame to make stills of: where, and why (the metrics whose worst dips
// it is the worst frame of, or the weak segment it is the worst frame of).
struct StillFrame {
    int frame = 0;
    double time = 0.0;
    std::string from;                  // "worst" or "segment"
    std::vector<std::string> metrics;  // the metrics that chose it, in the app's order
    std::optional<double> score;       // the first of them's score there
    std::optional<int> segment;        // for "segment": its index in weak_segments' list
};

// A frame's stills --stills wrote (stills.h): the reference's, the test
// video's, and the two side by side.
struct VideoStill {
    StillFrame frame;
    std::filesystem::path reference;
    std::filesystem::path test;
    std::filesystem::path side_by_side;
};

struct TestVideo {
    std::filesystem::path path;
    std::optional<VideoInfo> info;
    // The reference as this video is compared with: the run's, with this
    // video's colour tags where only this video has them (colours_taken).
    VideoInfo source;
    // Which of the two took the other's colour tags, having none of its own
    // ("the reference" or "the test video"), or nothing.
    std::string colours_taken;
    VmafOptions options;
    std::vector<std::string> metrics;  // the metric keys it is compared on
    // What it asks for, as its saved scores are looked up and saved under.
    AnalysisRequest request;
    // What a run of it saves: the request's metrics with their planes, which
    // are calculated with them asked for or not (with_planes).
    std::vector<MetricRequestSpec> saved_metrics;
    // Whether its scores are read from and saved to the saved scores: not
    // those of a video read with --colour or with the other video's colour
    // tags, nor of a raw stream, as the saved scores know a video by its
    // file, not by the colours or frame rate it was read with.
    bool save_scores = true;
    // What the reader should know: metrics left out and why. Each note's
    // code (codes.h) is in note_codes, in step: add them with note().
    std::vector<std::string> notes;
    std::vector<std::string> note_codes;
    std::string error;  // why it was not compared at all
    std::string error_code{};  // error's (codes.h)
    std::optional<ComparisonResult> result;
    // When each metric's score was saved, by its key: of the scores saved
    // show read (the score summary' "saved"); empty for a run's.
    std::map<std::string, std::chrono::system_clock::time_point> saved_times;
    // The metrics that failed and why, in the order they failed.
    std::vector<std::pair<std::string, std::string>> failures;
    // The codes of the failures the command line found itself, by metric;
    // a failure of the run has none here (calculation_failed_code).
    std::map<std::string, std::string, std::less<>> failure_codes;
    // The metrics asked for that do not apply to these videos, left out of
    // `metrics`: each with its outcome (metric_not_applicable).
    std::vector<std::pair<std::string, MetricOutcome>> not_applicable;
    bool from_cache = false;  // nothing had to be calculated
    std::optional<std::filesystem::path> csv_file;  // where --csv wrote its scores
    std::optional<std::filesystem::path> frames_json_file;  // where --frames-json wrote them
    std::optional<std::filesystem::path> run_file;  // where --run-dir wrote its result
    std::vector<std::filesystem::path> segment_files;  // where --segments-dir wrote its segments
    std::vector<VideoStill> stills;  // what --stills wrote
    // What its files are named after (--name's template, filled in), when not
    // its file's name.
    std::string file_name;
    std::string display;      // the preset CVVDP models, when CVVDP is asked for
    // For a file compared twice (--scale-to both, --film-grain both), as the
    // window's "Test both directions" and film grain twin rows: the scale
    // direction it was given in place of --scale-to's, and what its name adds
    // to the file's for the second of the two (" [upscale-distorted-to-source]",
    // " [without film grain]"), as the window names its row. Its film grain
    // choice is its info's with_film_grain.
    std::optional<ScaleDirection> scale_direction;
    std::string variant;
    // For a resolution test (--resolution-test): its target, and the file it
    // decodes, the reference, which its scores are saved under as the
    // window's are; `path` is then a made-up stand-in that names it
    // (synthetic_resample_distorted_path).
    std::optional<ResampleTarget> resample_test;
    std::optional<std::filesystem::path> media_path;
    // CVVDP on each further --cvvdp-display: a TestVideo of its own for each,
    // with CVVDP alone, run as a comparison of its own.
    std::vector<TestVideo> other_displays;
    bool for_display_only = false;  // one of another TestVideo's other_displays
    // Its quality gate's checks (--require, --fail-on-segments): none without one.
    std::vector<GateCheck> checks;
    // How its run went (RunEvents::job_facts), or how its saved scores answered
    // it all: none for a video that was not compared.
    std::optional<RunFacts> facts;

    // Adds a note and its code, unless the same note is there already.
    // Whether it was added.
    bool note(const char* code, const std::string& text) {
        if (std::find(notes.begin(), notes.end(), text) != notes.end()) {
            return false;
        }
        notes.push_back(text);
        note_codes.emplace_back(code);
        return true;
    }
    // The code of `metric`'s failure: the command line's, else the run's.
    std::string failure_code(const std::string& metric) const {
        const auto found = failure_codes.find(metric);
        return found != failure_codes.end() ? found->second : std::string(calculation_failed_code);
    }

    // The file its saved scores are found and saved under: media_path, else `path`.
    const std::filesystem::path& saved_path() const { return media_path ? *media_path : path; }
    // Its name as the reports give it: its file's, with `variant`.
    std::string name() const;
    // Its name as progress lines give it: with the display, for one of
    // another video's other displays.
    std::string label() const;
    // What its files are named after: file_name, else its file's name
    // without the extension, then `variant`.
    std::string file_stem() const;
};

}  // namespace vml::cli
