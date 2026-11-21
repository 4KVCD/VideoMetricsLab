// The saved scores: scores kept on disk so that a run can reuse those
// calculated before instead of calculating them again. A port of the Python
// app's metric_cache.py (and the parts of result_cache.py the command line
// uses), on the same files, so that either app finds what the other saved.
//
// Each comparison has a folder of its own, <base>/v2/<hash>, named by a hash
// of the two files' identities (file_identity) and the comparison recipe. It
// holds context.json, which describes the comparison, and one .npz file per
// metric score, named by the metric and a hash of its request.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "vml/analysis_request.h"
#include "vml/cancel.h"
#include "vml/results.h"

namespace vml {

// The format of the saved scores this app reads and writes.
inline constexpr int metric_cache_format_version = 2;

// The black-bar rule a comparison's detected bars were cut by, as its
// context.json says it ("black_bars_rule"; 1 where it says none). 2: odd
// bars rounded up to even, as HandBrake rounds them. 3: an encode cropped
// when it was made is compared at its own size, not read as the source
// scaled (common_picture, issue #18). A record only: saved scores of an
// older rule are read as saved, never checked or calculated again for it (a
// new rule applies to new calculations; Recalculate makes them anew).
inline constexpr int black_bars_rule_version = 3;

// The folder the scores are saved in unless Settings name another:
// .videometricslab/results_cache in the user's home.
std::filesystem::path default_cache_dir();

// How the saved scores recognise a reference whose path changed: given a
// file, its video_fingerprint (file_fingerprint.h), made with the tools of
// the app or the command line. None set (the default): files are recognised
// by their path and by their sampled_file_hash only.
using VideoFingerprinter = std::function<std::optional<std::string>(const std::filesystem::path&)>;
void set_video_fingerprinter(VideoFingerprinter fingerprinter);

// How the saved scores recognise a test video remuxed (raw, MKV, MP4):
// given a file, a number of packets (0: all) and what stops the read, its
// stream_fingerprint (file_fingerprint.h). None set (the default, and
// Settings' "Recognize test videos remuxed" off): test videos are
// recognised by their path and sampled_file_hash only, and nothing reads
// their streams.
using StreamFingerprinter =
    std::function<std::optional<std::string>(const std::filesystem::path&, int, const CancelToken&)>;
void set_stream_fingerprinter(StreamFingerprinter fingerprinter);

// How the lookups on this thread (load_result, other_cvvdp_scores) read a
// test video's stream to know a remux of a saved one by. A whole stream takes
// as long as its file takes to read: minutes for a film over a network.
// Without one of these, as when saving or clearing scores, a lookup reads
// no whole stream: only a stream's first packets, and where a remux could
// be told only by the whole stream, it finds nothing and says so
// (take_whole_stream_wanted). With one, `cancel` stops its reads, and with
// `whole` it may read whole streams, `announce` told of each file first.
class StreamReads {
public:
    explicit StreamReads(const CancelToken& cancel, bool whole = false,
                         std::function<void(const std::filesystem::path&)> announce = {});
    ~StreamReads();
    StreamReads(const StreamReads&) = delete;
    StreamReads& operator=(const StreamReads&) = delete;
};

// Whether a lookup on this thread, since the last call, found a saved
// comparison whose test stream began as its test video's does, with the same
// reference and settings, but could not read the whole stream to tell
// whether it is that video remuxed or another encode (StreamReads without
// `whole`). Cleared by the call.
bool take_whole_stream_wanted();

// Reads the test video `distorted`'s stream fingerprint and records it in
// the identity.json of each comparison under `base` saved for that file
// without one, for a remux of it to keep their scores. store_result does
// not read it (it takes as long as reading the file: minutes for a long one
// over a network share): this is called after it, apart from the saving.
// Nothing is read where nothing needs it. Whether one was recorded.
bool record_test_stream(const std::filesystem::path& base, const std::filesystem::path& distorted,
                        const CancelToken& cancel = {});
// Whether record_test_stream would read `distorted`: a comparison saved for
// it has no test stream recorded, as one saved by a version before streams
// were (its files recorded since, record_saved_files) has none until its
// test video is calculated again. Reads nothing of the file.
bool test_stream_wanted(const std::filesystem::path& base, const std::filesystem::path& distorted);

// The hash that names a comparison's folder: of both files' identities and
// the recipe's.
std::string recipe_hash(const std::filesystem::path& source, const std::filesystem::path& distorted,
                        const ComparisonRecipe& recipe);

// The hash of a metric's request, which names its score's file.
std::string metric_identity_hash(const MetricRequestSpec& spec);

// The comparison's folder under `base`. Folders saved under the older
// identity, one per scaling algorithm, are merged into it the first time it
// is asked for; if the first of those cannot be renamed (it is open
// elsewhere), it is used where it is. With none, a folder saved for the
// same files elsewhere (identity.json): the test video by its path or its
// sampled_file_hash, the reference by those or, where the test video
// matched, by its video_fingerprint, so that a renamed, moved or copied
// file, and a remuxed reference, keep their scores. Failing that, a test
// stream saved in another file (a remux) with the same recipe, reference
// and frame rate: by its stream_fingerprint's first packets, then the
// reference, then the whole stream. Its scores are then read and saved
// there.
std::filesystem::path recipe_directory(const std::filesystem::path& base, const std::filesystem::path& source,
                                       const std::filesystem::path& distorted, const ComparisonRecipe& recipe);

// What the rules against known-wrong saved SSIMULACRA2, Butteraugli and
// CVVDP scores read of a video: its pixel format, transfer tag, chroma
// siting, colour matrix, range, height and primaries (video_infos).
struct ColourTags {
    std::string pix_fmt;
    std::string color_transfer;
    std::string chroma_location;
    std::string color_space;
    std::string color_range;
    int height = 0;
    std::string color_primaries{};  // last and defaulted: a list that leaves it out knows it is not read
};
ColourTags colour_tags_of(const VideoInfo& video);
std::vector<ColourTags> colour_tags(const VideoInfo& source, const VideoInfo& distorted);

// Whether a score made with `provenance` answers `spec`: made by a
// compatible implementation, and not known to be wrong for these videos.
bool answers(const MetricRequestSpec& spec, const MetricProvenance& provenance, const std::vector<ColourTags>& infos);

// Whether a score of `key` made on `produced` ("gpu" or "cpu") counts for
// the metric set to `choice` ("gpu" or "cpu"; nothing for a metric with no
// such choice, where any counts). SSIMULACRA2 and Butteraugli give
// different scores on the GPU and on the CPU, so set to CPU, only a CPU
// score counts, and set to GPU, a CPU score counts only where the GPU
// cannot calculate the metric (`gpu_can_score`).
bool made_as_chosen(const std::string& key, const std::optional<std::string>& choice, const std::string& produced,
                    const std::function<bool(const std::string&)>& gpu_can_score);

// The scores of `result` that a run asking for `specs` keeps instead of
// calculating them again: those that exist, were made where their metric
// is set to be calculated (`backends`), and answer the request.
MetricResultSet reusable(const ComparisonResult& result, const std::vector<MetricRequestSpec>& specs,
                         const std::map<std::string, std::string>& backends,
                         const std::function<bool(const std::string&)>& gpu_can_score,
                         const std::vector<ColourTags>& infos);

// The keys of the scores reusable() keeps, in the order of `specs`, without
// copying the scores: for the window, which asks for each row whenever the
// row is drawn.
std::vector<std::string> reusable_keys(const ComparisonResult& result, const std::vector<MetricRequestSpec>& specs,
                                       const std::map<std::string, std::string>& backends,
                                       const std::function<bool(const std::string&)>& gpu_can_score,
                                       const std::vector<ColourTags>& infos);

// Saves a run's result in `base`: the comparison's context.json and the
// score of each request in `specs` the result has. Returns the comparison's
// folder. Throws std::filesystem::filesystem_error (or std::runtime_error)
// when a file cannot be written.
std::filesystem::path store_result(const std::filesystem::path& base, const std::filesystem::path& source,
                                   const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                                   const ComparisonResult& result, const std::string& label,
                                   const std::vector<MetricRequestSpec>& specs);

// The saved scores of `specs` for a comparison, with the comparison's
// details and its label; nothing when none are saved. `backends` says where
// SSIMULACRA2 and Butteraugli are set to be calculated: set to CPU, a Vship
// GPU score never answers. The saved scores of `supplemental` come too, for
// the metrics `specs` do not ask for: a metric asked for has its own saved
// score or none, or a score made another way (XPSNR of every frame for one
// of every n-th) would stand for it and never be calculated as asked.
// With black bars set to Source only, Auto's saved scores of the comparison
// answer where they are Source only's (Auto compared the test video at its
// whole frame), and the comparison's scores saved with another scaling
// algorithm or direction answer where it scaled neither video; except for
// the metric keys in `not_from_other_recipes`: those the user asked to
// calculate again, whose scores under this recipe are not saved yet.
std::optional<std::pair<ComparisonResult, std::string>> load_result(
    const std::filesystem::path& base, const std::filesystem::path& source, const std::filesystem::path& distorted,
    const ComparisonRecipe& recipe, const std::vector<MetricRequestSpec>& specs,
    const std::map<std::string, std::string>& backends, const std::vector<MetricRequestSpec>& supplemental = {},
    const std::set<std::string>& not_from_other_recipes = {});

// A comparison saved before its files were recorded (identity.json, which
// an older version did not write): records them now, so that its scores are
// found when the files are renamed, moved or remuxed. It reads a few MB of
// each file and decodes five pictures of the reference (seconds for a 4K
// film, once per reference in a process), so it is called after the saved
// scores load_result found are shown, never before. Nothing to do for a
// comparison not saved, or already recorded.
void record_saved_files(const std::filesystem::path& base, const std::filesystem::path& source,
                        const std::filesystem::path& distorted, const ComparisonRecipe& recipe);

// CVVDP's scores saved for a comparison with other displays than the one
// its request in `specs` or `supplemental` models, each with the CVVDP
// parameters it was made with (the Python app's other_cvvdp_scores), and
// that request's parameters; nothing when neither asks for CVVDP. Never an
// answer for the request: a CVVDP score holds for its own display only.
// With black bars set to Source only, Auto's come too where load_result
// would take them, and so do another scaling algorithm's or direction's
// (not with "cvvdp" in `not_from_other_recipes`).
struct OtherCvvdpScores {
    nlohmann::json parameters;                             // the request's
    std::vector<std::pair<CvvdpSettings, double>> scores;  // made for other displays
};
std::optional<OtherCvvdpScores> other_cvvdp_scores(const std::filesystem::path& base,
                                                   const std::filesystem::path& source,
                                                   const std::filesystem::path& distorted,
                                                   const ComparisonRecipe& recipe,
                                                   const std::vector<MetricRequestSpec>& specs,
                                                   const std::vector<MetricRequestSpec>& supplemental = {},
                                                   const std::set<std::string>& not_from_other_recipes = {});

// Deletes a comparison's saved scores of `specs`, and those that would
// answer the same requests (another VMAF score made with the same model, a
// VMAF v1 score saved under VMAF's key, either implementation's SSIMULACRA2
// or Butteraugli), so that none comes back after a Recalculate; the folder
// goes with its last score. Returns how many files were deleted.
int clear_metrics(const std::filesystem::path& base, const std::filesystem::path& source,
                  const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                  const std::vector<MetricRequestSpec>& specs);

// A token for a row's request to the saved scores: of the two files'
// identities, the recipe, the test video's frame rate (by which a remux of
// it is known: a lookup made before the video was read, without it, is made
// again) and each metric's request. The window refuses a saved result read
// under a token that is no longer the row's: a file or the row's settings
// changed while it was being read.
std::string lookup_token(const std::filesystem::path& source, const std::filesystem::path& distorted,
                         const AnalysisRequest& request);

// A saved score as its file holds it, whatever request it answers: its
// metric (the result's key), samples and provenance; and whether the Videos
// tab's lookup refuses it for its comparison (the context.json beside it)
// whatever it asks, and scores the videos again: one made from other
// pictures than the pair is decoded to now (its decoding marks: exact
// decoding, AV1 grain), or known to be wrong (FFmpeg's XPSNR above 8 bits,
// Vship 5.1's old colour mappings, a stale CPU score). Nothing for a file
// that cannot be read or is of another format. For browsing the saved
// results (saved_results.h), which need no request to read them by.
struct SavedScore {
    std::variant<FrameMetricResult, SequenceMetricResult> metric;
    bool outdated = false;
};
std::optional<SavedScore> read_saved_score(const std::filesystem::path& file);

// The comparison a saved folder's context.json describes: its result,
// without scores, and its label; nothing where there is no context.json, or
// one this app cannot read.
std::optional<std::pair<ComparisonResult, std::string>> saved_comparison(const std::filesystem::path& folder);

// The sampled hash (file_fingerprint.h) a saved folder's identity.json
// records for its reference (`reference`) or test video: the key of their
// saved bitrate (bitrate_store.h). Empty where none is recorded.
std::string saved_content_hash(const std::filesystem::path& folder, bool reference);

// Whether the file at `path` is the reference (`reference`) or the test
// video of the comparison saved in `folder` as its identity.json recorded it
// (its path, size and time); true where it recorded none (saved before it
// was recorded), false where no file is there.
bool saved_file_at(const std::filesystem::path& folder, bool reference, const std::filesystem::path& path);

// How many comparisons are saved under `base`, and the bytes their files
// and the saved bitrate (bitrate_store.h) take, for Settings.
struct CacheSummary {
    int comparisons = 0;
    std::uintmax_t bytes = 0;
};
CacheSummary cache_summary(const std::filesystem::path& base);

// Whether the reference of the comparison saved in `folder` is still where
// its scores were saved from: a file at its path, and the same file, as its
// identity.json recorded it (path, size and time; where the comparison has
// one). Without it, nothing saved there can be calculated again: the scores
// saved are the only ones there are.
bool reference_where_saved(const std::filesystem::path& folder);

// What clear_saved_results did: how many comparisons it deleted (0 when not
// all it meant to could be, as the Python app counted them), and how many it
// kept.
struct ClearedResults {
    int removed = 0;
    int kept = 0;
};

// Deletes the comparisons saved under `base` (its v2 folder) and the saved
// bitrate (its bitrate folder; nothing else there): every comparison, or
// with `keep_without_reference` every one whose reference is still where it
// was (reference_where_saved), keeping those whose scores could not be
// calculated again, and the bitrate of their files.
ClearedResults clear_saved_results(const std::filesystem::path& base, bool keep_without_reference);

}  // namespace vml
