// One test video of a command-line run: what the window keeps in a row.
#pragma once

#include <algorithm>
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

    // Its name as the reports give it.
    std::string name() const;
    // Its name as progress lines give it: with the display, for one of
    // another video's other displays.
    std::string label() const;
    // What its files are named after: file_name, else its file's name
    // without the extension.
    std::string file_stem() const;
};

}  // namespace vml::cli
