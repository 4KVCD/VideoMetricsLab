// The command line's arguments, read as the Python app's argparse parser
// reads them: the same commands and options, unambiguous abbreviations of
// long options, --option=value, and exit code 2 for a wrong command.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gates.h"
#include "vml/video.h"
#include "vml/weak_frames.h"

namespace vml::cli {

inline constexpr const char* program = "VideoMetricsLab-cli";
// The exit codes, as the user guide's command line section lists them. When
// more than one applies: cancelled, then timed out, then an input not read,
// then a failure, then a quality gate.
inline constexpr int exit_ok = 0;
inline constexpr int exit_failed = 1;  // a video or a metric failed, or a file could not be written
inline constexpr int exit_usage = 2;   // a wrong command: nothing was read or run
// Every score asked for is there, but a quality gate (--require,
// --fail-on-segments: gates.h) is not met.
inline constexpr int exit_gate_failed = 3;
// An input could not be read: the reference, a test video, a pattern that
// finds no file, the --batch list, bdrate's files; the others are still done.
inline constexpr int exit_input = 4;
// --timeout or --video-timeout stopped the run, or a video's (timeout's own code).
inline constexpr int exit_timeout = 124;
inline constexpr int exit_cancelled = 130;

// Which settings a command takes and where its saved scores are: options of
// compare, bitrate, devices, saved and report (--cache-dir of compare,
// bitrate and saved; --no-save and --no-cache of compare and bitrate).
struct ProfileArguments {
    // --settings: the settings file read in place of the profile's, never written.
    std::optional<std::string> settings_file;
    bool no_settings = false;  // --no-settings: every setting at its default (Settings::defaults)
    std::optional<std::string> cache_dir;  // --cache-dir: the saved scores' folder in place of the settings'
    bool no_save = false;   // --no-save: saved scores read, nothing saved
    bool no_cache = false;  // --no-cache: saved scores neither read nor saved
};

// The compare command's arguments.
struct CompareArguments {
    ProfileArguments profile;
    std::vector<std::string> command_line;  // every word given, for the results' record
    std::optional<std::string> reference;  // none with --batch
    std::vector<std::string> tests;  // files, or patterns such as encodes\*.mkv; none with --batch
    // --batch: a CSV file listing the references and test videos (batch_file.h), in place of
    // REFERENCE and TEST, so that one process compares them all.
    std::optional<std::string> batch;
    // --scores-csv: a CSV file of each test video's bitrate and scores, as bdrate reads them.
    std::optional<std::string> scores_csv;
    std::optional<std::vector<std::string>> metrics;  // keys, in the registry's order
    std::string model = "auto";
    std::string model_v1 = "auto";   // a word of v1_model_choices (model_select.h)
    std::string model_hdr = "auto";  // a word of hdr_model_choices
    std::string black_bars = "auto";  // "auto", "source" (the source's only) or "none"
    bool black_bars_given = false;    // --black-bars was given (refused beside a crop entered)
    // Crops entered by hand (--source-crop, --test-crop): either one makes the run's black bars
    // CropMode::manual.
    std::optional<CropEdges> source_crop;
    std::optional<CropEdges> test_crop;
    std::string scale_to = "test";  // "test", "reference" or "both" (each direction, a video of its own)
    std::string scaler = "bicubic";
    std::optional<long long> duration_ms;  // milliseconds from the start of the videos
    std::optional<std::pair<long long, long long>> range_ms;  // milliseconds; an end of 0: to the end
    // The first frame and the one past the last (--frames); an end of 0: to the end.
    std::optional<std::pair<long long, long long>> frames;
    std::optional<std::string> colour;
    // "with" or "without" (VideoInfo::with_film_grain), or "both" (each, a video of its own)
    std::optional<std::string> film_grain;
    std::optional<double> fps;            // a raw YUV file's frame rate, a raw stream's in place of what it stores
    std::optional<std::string> size;      // a raw YUV file's picture size, "1920x1080"
    std::optional<std::string> pix_fmt;   // a raw YUV file's pixel format, as FFmpeg names it
    std::vector<std::string> cvvdp_displays;
    // --resolution-test: each width and scaling algorithm, in the order given, each once.
    std::vector<std::pair<int, std::string>> resolution_tests;
    bool cvvdp_fill = false;  // CvvdpSettings::resize_to_display
    int subsample = 1;
    std::optional<int> threads;
    std::optional<std::string> vmaf_on;
    std::optional<std::string> vship;       // "vship-fast" or "vship" (vship_build_value)
    std::optional<std::string> vship_pass;  // "one" or "each" (Settings::gpu_metrics_together)
    bool cpu = false;
    bool no_gpu_decode = false;
    std::optional<std::string> gpu_decode;  // "auto", or the maker: "nvidia", "intel" or "amd"
    std::optional<std::string> vvc_decoder;  // "gpu" or "cpu" (Settings::vvc_decode_on_gpu)
    std::optional<std::string> test_remuxes;  // "on" or "off" (Settings::recognise_test_remuxes)
    // --gpu: the GPU to score on, by name (gpu_choice.h); "auto" for a discrete one first (Settings::gpu)
    std::optional<std::string> gpu;
    std::optional<int> parallel;
    std::optional<std::string> frame_counts;  // "check" or "skip" (Settings::check_frame_counts)
    bool recalculate = false;
    bool difference = false;  // the first test video minus the second, too
    std::optional<std::string> csv;
    // --frames-json: the folder for each video's per-frame scores as JSON (<video>.frames.json).
    std::optional<std::string> frames_json;
    // --score-summary: the folder for the score summary (score_summary.h),
    // a JSON and a CSV file a reference; with the columns and CVVDP's
    // displays the settings give (filled in by the command, not parsed).
    std::optional<std::string> score_summary;
    std::vector<std::string> score_summary_columns;
    std::string cvvdp_presets_json;
    std::optional<std::string> json_path;  // "-" prints the JSON instead of the tables
    std::optional<int> worst;  // --worst N: each metric's N worst dips, after the tables
    // --segments: where scores fall past these (weak_frames.h), with the
    // options that shape the segments; unset ones take SegmentRules' defaults.
    std::vector<WeakCondition> segments;
    // --segments saved: the conditions and options saved in the window's
    // Settings > Weak segments (weak_segment_*), taken once the settings are
    // read (take_saved_segments); until then `segments` holds the conditions
    // given after it, which change the profile's.
    bool segments_saved = false;
    // --segments profile:NAME: that profile's rather than the one in use
    // (segments_saved set too).
    std::optional<std::string> segments_profile;
    // --segments preset:KEY: that weak segment preset's (segment_presets.h;
    // segments_saved set too).
    std::optional<std::string> segments_preset;
    bool segments_all = false;
    bool segments_any = false;  // --segments-any: any condition, whatever the profile or preset says
    std::optional<int> segment_min_frames;
    std::optional<int> segment_merge_frames;
    std::optional<std::string> segments_dir;  // the folder for each video's segment files
    // --segments-report: one file of every test video's segments (JSON, or CSV for a .csv name).
    std::optional<std::string> segments_report;
    // --require: the bars each test video's scores must reach (gates.h), from every --require given.
    std::vector<GateRule> require;
    // --fail-on-segments: a test video with a weak segment (by --segments) fails the quality gate.
    bool fail_on_segments = false;
    // --stills: the folder for PNGs of the worst frames (stills.h); --stills-from
    // "worst" (each metric's worst dips) or "segments" (each weak segment's
    // worst frame), --stills-count how many dips of each metric, or segments.
    std::optional<std::string> stills;
    std::optional<std::string> stills_from;
    std::optional<int> stills_count;
    // --run-dir: the folder for each video's saved run (<name>.metrics.json), as Save analysis results writes it.
    std::optional<std::string> run_dir;
    // --log-file: the file the app's log is written to as well, as the window's log (file_names.h's fields in it).
    std::optional<std::string> log_file;
    // --name: the template each test video's files are named after, in place of its file's name (file_names.h).
    std::optional<std::string> name;
    // --notify: the run's notification as it ends, as the app's Settings > Notifications say (notify.h).
    bool notify = false;
    bool quiet = false;
    bool verbose = false;
    // --progress: "text" (lines for a person) or "jsonl" (Progress::event: a JSON object a line, for a program).
    std::string progress = "text";
    // --timeout: whole seconds the command may take, from its start; past them it is cancelled (exit_timeout).
    std::optional<long long> timeout;
    // --video-timeout: whole seconds each video's run may take, from its start; past them it alone is stopped.
    std::optional<long long> video_timeout;
    // --cancel-on-stdin-eof: the end of standard input cancels the run, as Ctrl+C does.
    bool cancel_on_stdin_eof = false;
};

// The bitrate command's arguments.
struct BitrateArguments {
    ProfileArguments profile;
    std::vector<std::string> videos;  // files, or patterns such as encodes\*.mkv
    std::optional<double> fps;        // a raw stream's frame rate, in place of what it stores
    std::optional<std::string> csv;   // the folder for each video's CSV files
    bool no_adjust_start = false;     // the CSV files' times from the file's start (only with csv)
    bool difference = false;          // the first video's bitrate minus the second's (two videos only)
    std::optional<std::string> json_path;  // "-" prints the JSON instead of the table
    bool quiet = false;
};

// The bdrate command's arguments.
struct BdrateArguments {
    std::string anchor;  // the anchor's CSV file: a row per encode
    std::string test;    // the test encoder's
    std::optional<std::vector<std::string>> metrics;  // keys, in the registry's order
    std::optional<std::string> group_by;   // a column of the anchor's file
    std::optional<std::string> json_path;  // "-" prints the JSON instead of the table
    bool quiet = false;
};

// The devices command's arguments.
struct DevicesArguments {
    std::optional<std::string> json_path;  // "-" prints the JSON instead of the lines
    ProfileArguments profile;  // its settings only
};

// The saved command's arguments (list, show or delete the saved results),
// and the report command's (run files): what to do, its items (IDs, test
// videos' paths, or run files), and for show and report what to write, in
// compare's own fields (--json, --csv, --frames-json, --run-dir, --segments
// and its options, --worst, -q), which compare's reports read.
struct SavedArguments {
    std::string action;  // "list", "show" or "delete"; "report" for the report command
    std::vector<std::string> items;
    std::optional<std::string> filter;  // list --filter: part of a label, test video's or reference's path
    CompareArguments output;
    ProfileArguments profile;  // --settings, --no-settings; saved's --cache-dir
};

// The target command's arguments: where encodes reach a score.
struct TargetArguments {
    std::vector<std::string> scores;  // files, or patterns: CSV, results JSON or run files
    std::string metric;               // a key
    double value = 0;                 // the score to reach
    std::string by = "kbps";          // or a column of the CSV files', in lower case (crf)
    std::string stat = "score";       // a metric summary's statistic (loop.h's summary_stats)
    std::optional<std::string> sequence;
    std::optional<std::string> json_path;  // "-" prints the JSON instead of the table
    bool quiet = false;
};

enum class Command { none, compare, devices, bitrate, bdrate, saved, report, target, diff };

struct Arguments {
    Command command = Command::none;
    // Help or the version was asked for: print this and exit 0.
    std::optional<std::string> print_and_exit;
    CompareArguments compare;
    DevicesArguments devices;
    SavedArguments saved;  // the saved command's, the report command's, or diff's (OLD and NEW its items)
    TargetArguments target;
    BitrateArguments bitrate;
    BdrateArguments bdrate;
};

// A wrong command; the message is argparse's (usage, then the error).
class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The arguments, without the program's name. Throws UsageError.
Arguments parse_arguments(const std::vector<std::string>& arguments);

// The top-level help, printed when no command is given.
std::string top_help();

// The app's name and version, as --version prints it.
const std::string& version_line();

}  // namespace vml::cli
