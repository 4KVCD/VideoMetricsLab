// The codes the command line's JSON gives beside each error, failure, note
// and reason a metric was not calculated: stable words a program can act on,
// where the text beside them is for a person and may change. A code is never
// renamed or reused for something else; new ones may be added.
// docs/RESULTS_JSON.md lists them all. Metric outcomes' codes of the run
// itself are core's (run_record.h).
#pragma once

#include "vml/run_record.h"

namespace vml::cli::codes {

// Why a video, a reference or the --batch list was not compared (`error_code`).
inline constexpr const char* no_match = "no_match";            // a pattern found no file
inline constexpr const char* unreadable = "unreadable";        // FFmpeg could not read the video
inline constexpr const char* no_metric = "no_metric";          // none of the metrics asked for can be calculated for it
inline constexpr const char* not_comparable = "not_comparable";  // the pair cannot be compared (lengths differ...)
inline constexpr const char* run_failed = "run_failed";        // its run failed as a whole
inline constexpr const char* timed_out = "timed_out";          // --video-timeout stopped it (also a metric's)
inline constexpr const char* reference_unreadable = "reference_unreadable";
inline constexpr const char* no_test_video = "no_test_video";  // a --batch reference's rows find only it
inline constexpr const char* batch_unreadable = "batch_unreadable";  // the --batch list could not be read
inline constexpr const char* settings_unreadable = "settings_unreadable";  // the --settings file could not be read
inline constexpr const char* no_packets = "no_packets";        // bitrate: a raw YUV file has no encoded packets
// bdrate's files: one that cannot be read, or no metric or sequence they share.
inline constexpr const char* rates_unreadable = "rates_unreadable";
inline constexpr const char* no_common_metric = "no_common_metric";
inline constexpr const char* no_common_sequence = "no_common_sequence";

// Why a metric was not calculated, beyond core's (`failed_metric_codes`, `metric_status`).
inline constexpr const char* needs_every_frame = "needs_every_frame";  // CVVDP with --subsample above 1
inline constexpr const char* no_gpu = "no_gpu";                        // a GPU-only metric, and no GPU for it
inline constexpr const char* hdr_vmaf_colours = "hdr_vmaf_colours";    // HDR VMAF of a pair not PQ (HDR10)

// What a note says (`note_codes`).
inline constexpr const char* gpu_fallback = "gpu_fallback";            // a GPU metric fell back
inline constexpr const char* metric_left_out = "metric_left_out";      // a default metric was left out
inline constexpr const char* not_applicable = not_applicable_code;    // a metric does not apply to the videos
inline constexpr const char* colours_taken = "colours_taken";          // a video took the other's colour tags
inline constexpr const char* colours_guessed = "colours_guessed";      // a video's colours were guessed
inline constexpr const char* film_grain_unknown = "film_grain_unknown";
inline constexpr const char* film_grain_left_out = "film_grain_left_out";
inline constexpr const char* sdr_display = "sdr_display";              // HDR video, CVVDP on an SDR display
inline constexpr const char* near_miss_crop = "near_miss_crop";        // crops a few lines apart, scaled
inline constexpr const char* frame_counts = "frame_counts";            // fewer frames compared than expected

}  // namespace vml::cli::codes
