# The results file (JSON and CSV)

The command line's `--json` and the window's exports with **Results** and
**JSON** ticked in Settings > Storage > Exports include write the same
document. With **CSV** ticked, the same content is written with a
row for each video, each column named by the key's path in the JSON
(`settings.scaling.compared_at`, `cvvdp_displays.1.score`). A list of plain
values is one cell, its JSON.

This page describes format 2 (VideoMetricsLab 2.5). A later format only adds
keys, so a reader should skip the keys it does not know. A key that applies is
always written, defaults included: a missing value is `null`, never a key left
out. Times are UTC in ISO 8601 (`2026-11-14T21:42:25Z`), durations are in
seconds, sizes are in bytes and bitrates are in bits per second. JSON has no
infinity: an endless number (identical frames' PSNR) is the string `"inf"`
(`"-inf"`), in this file and every other the app writes.

## The format of each file

Every JSON file the app writes for a program to read begins with `app`
(`"VideoMetricsLab"`), `version`, `commit` (the build's) and `format`,
which says what the file is and which version of it. A later version of a
format only adds keys; one that changed a key's meaning would be a new
version. Each has a JSON Schema in [`docs/schemas`](schemas), which the
tests hold the app's output to.

| `format` | The file | Written by | Schema |
| --- | --- | --- | --- |
| `2` | The results (this page) | `compare --json`, the window's results exports | `results.schema.json` |
| `"segments-1"` | A video's weak segments, `<video>.<metrics>.segments.json` | `compare --segments-dir`, the window's exports | `segments.schema.json` |
| `"segments-report-1"` | Every video's weak segments in one file | `compare --segments-report`, the window's exports | `segments-report.schema.json` |
| `"frames-1"` | A video's per-frame scores ([below](#per-frame-scores-as-json)) | `compare --frames-json`, the window's exports | `frames.schema.json` |
| `"worst-frames-1"` | A metric's worst dips ([below](#worst-frames-as-json-and-csv)) | the Worst Frames tab's **Worst frames as JSON...** | `worst-frames.schema.json` |
| `"worst-frames-report-1"` | Every metric's worst dips of a video, `<video> - worst frames.json` ([below](#worst-frames-as-json-and-csv)) | the window's exports | `worst-frames-report.schema.json` |
| `"bitrate-1"` | Each video's bitrate | `bitrate --json` | `bitrate.schema.json` |
| `"bdrate-1"` | BD-rates | `bdrate --json` | `bdrate.schema.json` |
| `"devices-1"` | Where each metric would be calculated | `devices --json` | `devices.schema.json` |
| `"saved-list-1"` | The saved comparisons | `saved list --json` | `saved-list.schema.json` |
| `"target-1"` | The rate or setting at which encodes reach a score | `target --json` | `target.schema.json` |
| `"diff-1"` | How a run's scores and weak segments differ from an earlier one's | `diff --json` | `diff.schema.json` |

The results' format is a number, `2`, as it was before the others had one;
the others are `<kind>-<version>`. A run file (`.metrics.json`) is the app's
own, which it reads back: its `format_version` is not part of this
convention.

## The document

| Key | Meaning |
| --- | --- |
| `app`, `version`, `commit` | The program that wrote the file, its version and its build's commit (`-dirty`: built with uncommitted changes). |
| `url` | The program's home, `https://github.com/4KVCD/VideoMetricsLab`. |
| `format` | `2`. |
| `cancelled` | Command line only: whether the run was cancelled. A cancelled run's videos hold the scores it finished. |
| `timed_out` | Command line with `--timeout` or `--video-timeout` only: whether either stopped the run or a video. |
| `run` | The run the scores come from (below). |
| `machine` | The machine that scored them (below). |
| `summary_rules` | How each score and statistic is worked out from the frames (below). |
| `reference` | The reference video: `path` and its file details (see [File details](#file-details)). `null` when the window has no reference. |
| `videos` | One entry for each test video (below). |
| `difference` | Command line with `--difference` only: the first video minus the second. |
| `log_file` | Command line with `--log-file` only: the log's file, its fields filled in. |
| `profile` | Command line only: the settings and saved scores the run used (below). |
| `read_from` | `saved show` and `report` only: `"saved results"` or `"run files"`, where the scores were read without their videos. |
| `error`, `error_code` | Command line only, when nothing was compared: why, and its [code](#codes). The reference could not be read (`reference_unreadable`): `reference` has only its `path` and `videos` is empty; or the `--batch` list could not be read (`batch_unreadable`): no `reference`, an empty `batch`, and no `run` or `machine`. |

With `--batch`, the command line's `--json` is `{"batch": [...]}`: this document
for each reference in turn, or `{"reference": path, "error": why, "error_code":
code}` for a reference that could not be read (`reference_unreadable`) or
whose rows find only it (`no_test_video`).

### `profile`

| Key | Meaning |
| --- | --- |
| `settings` | Where the settings came from: `profile` (the app's, as the window last saved them), `file` (`--settings`) or `defaults` (`--no-settings`). |
| `settings_file` | The file read: the profile's `settings.json` or `--settings`' file. `null` for the defaults, or a profile that has no settings file yet (the defaults then). |
| `app_data_folder` | The folder of the app's own files: `.videometricslab` in the home folder, or the one `VML_HOME` names. |
| `options_from_settings` | The options left out of the command that the settings filled in, as they are written (`--metrics`, `--vship`, `--gpu`...), whether or not this run used them. `--segments` and `--notify` are listed where they take the settings' profile or services. |
| `saved_scores` | `folder` (the saved scores' folder; `null` with `--no-cache`), `read` (whether saved scores were used: not with `--recalculate`, `--no-cache`, or Settings' **Reuse a saved result** off) and `save` (whether new ones were saved: not with `--no-save` or `--no-cache`). |

The `bitrate` command's JSON has a `profile` too, its `saved_scores` the
saved bitrate's.

### `run`

| Key | Meaning |
| --- | --- |
| `interface` | `"cli"` (the command line) or `"gui"` (the window). |
| `command_line` | The command line's arguments as given, as a list; `null` from the window. |
| `started_utc`, `finished_utc` | When the run started and ended. |
| `wall_seconds` | Start to end, pauses included. |
| `paused_seconds` | Time spent paused (the command line cannot pause: `0`). |
| `runs` | The window's export can hold videos from several runs. This counts them; the start is the first run's, the end the last's, and the times are added up. `0` when no exported video comes from a run in this window (saved scores, saved runs). While a run is still going, its end and times are `null`. |

### `machine`

| Key | Meaning |
| --- | --- |
| `os` | The operating system and its version (`Windows-11-10.0.26300-SP0`). |
| `cpu` | The CPU's model name. |
| `gpus` | Every GPU Vulkan lists: `name`, `vendor`, `driver_version` (as the maker numbers its drivers: `616.92` for NVIDIA, `101.8991` for Intel on Windows), `driver_version_number` (Vulkan's raw number), `driver_name`, `driver_info`. Empty until the app's GPU check has answered. |
| `vship_gpu` | The GPU SSIMULACRA2, Butteraugli and CVVDP run on (`null` when none answered). |
| `ffmpeg` | `version` and `path` of the FFmpeg used. |

### `summary_rules`

Text describing the statistics: `median`, `stdev` (population),
`worst_percent` (the `worst_*_percent` values are percentiles, numpy's
linear method, not means of the worst frames), `percent_of_frames`,
`frames_with_no_score` (frames without a score are left out; a frame
identical to the reference scores infinity and counts). Under `metrics`, for
each metric: how its `score` comes from its frames (the mean; XPSNR's
average in decibels; CVVDP's whole-video score), whether its worst scores are
the `low` or `high` ones, and its `bands`.

## Each video

| Key | Meaning |
| --- | --- |
| `path`, `reference` | The test video and the reference it was compared with. |
| `label` | Window only: the name the window gives it. |
| `status` | `ok`, `partial` (some metrics failed: `failed_metrics` says why), and on the command line also `failed` (with `error`) or `cancelled`. |
| `error`, `error_code` | Command line only, for a video not compared: why, and its [code](#codes). |
| `notes`, `note_codes` | Command line only: what the reader should know about how the video was scored (a metric left out, colours guessed, a GPU metric that fell back...), and each note's [code](#codes), in step. |
| `gate` | Command line with `--require` or `--fail-on-segments` only: the video's quality gate. `passed` (whether no check failed) and `checks`, one for each `--require` rule in order, then `no weak segments`: `rule` (as written back, `vmaf.worst_1_percent>=90`), `metric` and `statistic` (not for the weak segments' check), `value` (the video's figure, or its number of weak segments; `null` where it has none), `passed` (`true`, `false`, or `null` for a metric that does not apply to the video, which does not fail the gate), and `why` where it failed with no value or does not apply. |
| `failed_metrics`, `failed_metric_codes` | Command line only: each metric that failed, why, and its [code](#codes). |
| `from_saved_scores` | Command line only: whether the scores came from saved scores. |
| `stills` | Command line only, with `--stills`: each frame it made stills of, in frame order: `frame`, `time`, `from` (`"worst"` or `"segment"`), the `metrics` that chose it, the first one's `score` there, the weak `segment`'s index (in `weak_segments`' list) for `"segment"`, and the paths of its `reference`, `test` and `side_by_side` PNGs. |
| `csv`, `frames_json`, `run_file`, `segment_files` | Command line only: where `--csv`, `--frames-json`, `--run-dir` and `--segments-dir` wrote this video's files (the `.<metrics>.segments.json`, then the `.csv`), for a script to find them under the names they were given (`--name`, or a number after a name taken). A further CVVDP display's entry in `cvvdp_displays` has its own `csv` and `run_file`. |
| `width` ... `encoder_settings` | The test video's details (see [File details](#file-details)). |
| `film_grain` | Where the stream carries film grain: its `kinds` and whether it was `scored` `with` or `without` it (and `why_without`). |
| `compared_frames` | How many frame pairs were compared. |
| `frame_sync` | Whether the video's frames are in sync with the reference's (see the user guide on frames being compared in order). `status`: `no_mismatch_seen` (the number of frames compared matches the videos' length); `may_be_out_of_sync` (at least 2 fewer were compared, and at least one file doesn't store its frame count, so the frames weren't counted: turn on Settings > Storage > Read entire files to check frame counts when needed, or use `--frame-counts check`); `out_of_sync` (the two have different numbers of frames, so the scores from where they differ are wrong); `in_sync` (fewer were compared, but both have the same number of frames); or `not_checked` (only part of the videos, a resolution test, a length that isn't known, or no per-frame metric). `note`: what it means, in plain words, and for `may_be_out_of_sync` how to find out. `compared_pairs`, `length_frames` (the frames the shorter video's length should have), and `reference_frames` and `test_frames` (`null` when not counted). |
| `vmaf_model` | The VMAF model the result records. |
| `reference_crop`, `test_crop` | The picture compared in each video (`x`, `y`, `width`, `height`), or `null` for the whole frame. |
| `metrics` | Each metric's score and statistics (below). |
| `whole_file` | Whether the scores are of the whole videos (`false` for a range), as the scores record it, so known for a saved run loaded into the window too. Only scores of the whole file go with its `bit_rate`. |
| `resolution_test` | For a resolution test, what the reference was scaled to and back (`label`, `width`, `scaler`); else `null`. Its `bit_rate` is the reference's. |
| `variant_of` | Window only: for a row that scores another row's file another way (a film grain twin, a "Test both" companion, CVVDP on a further display listed apart), that file; else `null`. |
| `cvvdp_display`, `cvvdp_displays` | The display CVVDP's score is for, and every display CVVDP was scored on: `display`, `score`, `display_parameters`, `cvvdp` (its full summary), and for a further display its own `timings`. |
| `settings` | Every setting that changes the scores or how to read them (below). |
| `weak_segments` | Command line with `--segments` only: the video's weak segments by the conditions given, as a [segments file](#the-format-of-each-file) has them but without its head: `conditions` (as written back, `vmaf<90`), `left_out` (conditions on a metric the video was not scored with, where any), `combine` (`any` or `all`), `min_frames`, `merge_frames`, `ranges` (the frames counted, `first_frame` and `last_frame` each; empty: all), and `segments`, each with `first_frame`, `last_frame`, `start_time`, `end_time` (seconds), `frames`, for each condition's metric its `min`, `max` and `mean` there, `weak_by` (the conditions that hold on any of its frames, the Worst Frames tab's Weak by) and `condition_frames` (how many of its frames each holds on); then the video's `segment_count`, `weak_frames`, `weak_seconds`, `weak_percent` (of the frames compared) and `condition_counts` (each condition's `condition`, the frames it `holds` on and how many are `in_segments`). |
| `black_bars` | How the black bars were found (below). |
| `metric_status` | For each metric asked for: `status` is `calculated`, `saved` (taken from saved scores), `failed` or `cancelled`, `not run`, or `not applicable` (command line: the metric does not apply to these videos, such as HDR VMAF of a video that is not PQ; it is left out, and not a failure); `why` and its [code](#codes) for the failed, not run and not applicable ones. |
| `decode_plan` | For each pass, its `metrics` and the `decoders` it used for the `source` and the `test` video, after any fallback: FFmpeg's hardware decoder (`cuda`, `d3d11va`, `vulkan`, `videotoolbox`), `vvdec` (VVC decoded on the GPU by vvdec-vulkan) or `software`; for a video decoded in `software`, `source_why` or `test_why` says why (GPU decoding is off, a format or codec GPU decoders do not decode exactly, VVdeC decoding VVC on the CPU, the GPU's decoding having failed in the run, or no GPU decoder here for its codec). |
| `timings` | `started_utc`, `seconds` (pauses included), `paused_seconds`, and `passes` (below). |

`settings`, `black_bars` and `metric_status` are `null` for a saved run
loaded into the window, which does not record them. `decode_plan` and
`timings` are also `null` for a video that did not run: its saved scores
answered it, or a Cancel stopped it before it started.

### `metrics`

Each metric's key (`vmaf`, `vmaf_neg`, `vmaf_v1`, `vmaf_hdr`, `psnr`, `ssim`, `xpsnr`,
`ssimulacra2`, `butteraugli`, `cvvdp`) holds:

| Key | Meaning |
| --- | --- |
| `score` | The video's score (`summary_rules` says how). `"inf"` for identical videos' PSNR. |
| `frames` | Frames with a score (CVVDP: frames compared). |
| `median`, `stdev`, `min`, `max` | Of the frames' scores. |
| `worst_is` | `low` or `high`. |
| `worst_10_percent` ... `worst_0.1_percent` | The score that share of the frames is worse than. |
| `identical_frames` | Frames identical to the reference (where any). |
| `percent_of_frames` | The share of frames in each band, in %. |
| `worst_frames` | The worst frame of each of the 10 worst dips, worst first: `frame` (numbered from the video's first frame, as the encoder numbers them), `time` (seconds) and `score`, three lists in step. Frames within 1 s of a worse one are the same dip. CVVDP's are under `per_second`, a frame per second. |
| `per_second` | CVVDP only: the same statistics of its per-second scores. |
| `computed_on`, `implementation`, `version` | Where and with what the metric was calculated (`cpu` or `gpu`; `libvmaf/vulkan`, `Vship/ssimulacra2`...). |

### `settings`

| Key | Meaning |
| --- | --- |
| `metrics` | The metrics asked for, in order. |
| `vmaf_model`, `vmaf_v1_model` | `choice` (`__auto__` picks by the size compared at), the `model` used (the install's path of its file, or a version libvmaf has built in), and its `id`, the same on every install: a bundled model's or libvmaf's own version (`vmaf_v0.6.1`, `vmaf_v1.0.16_3d0h`), `null` for a model file of the user's. `null` where that VMAF was not asked for (VMAF NEG uses `vmaf_model`). |
| `vmaf_hdr_model` | The same for HDR VMAF; present only where HDR VMAF was asked for. |
| `custom_model_path` | A VMAF model file given by its path, or `null`. |
| `cvvdp` | `display` (the preset's name, `""` for a display that is no preset) and its `parameters`; `null` without CVVDP. |
| `frame_subsample` | `asked` (every n-th frame) and `every_nth_frame` per metric: libvmaf's metrics follow it, XPSNR and CVVDP score every frame. |
| `range` | The range asked for: `whole`, or `start` and `end` in `unit` (`seconds` or `frames`). With CVVDP and a range of frames, `cvvdp_seconds` (`start`, `end`): the whole seconds CVVDP scored, those nearest the frames (an `end` of 0: to the videos' end). |
| `compared_frames` | The frames compared, counted from the first frame: `first`, `last`, `count`. |
| `frame_pairing` | `frame number`: frame N of one video is compared with frame N of the other, each counted from its own first frame; nothing is dropped or repeated to line them up. |
| `scaling` | `direction` (`source_to_distorted` scales the reference to the test video's size, `distorted_to_source` the other way), `algorithm`, `compared_at` (the size compared), and for `source` and `test` the scaling each had (`from`, `to`) or `null` when left at its size. A resolution test gives `resolution_test` (`label`, `width`, `scaler`) instead. |
| `decoding` | `gpu_decode` (whether videos may be decoded on the GPU), `gpu_maker` (`automatic` or the maker chosen), `vvc` (`gpu` or `cpu`: VVdeC decodes VVC either way). |
| `calculated_on` | Where the metrics were set to run: `vmaf`, `perceptual` (per metric), `vship` (the Vship chosen) and `vship_in_use`, `libvmaf_threads` (`0`: automatic). Each metric's `computed_on` says where it actually ran. |

### `black_bars`

`mode` is `auto` (both videos), `source` (the reference's alone), `none` or
`manual` (with `manual_source` and `manual_test`, the edges entered). Then
`source` and `test`:

| Key | Meaning |
| --- | --- |
| `compared` | The picture compared (`null`: the whole frame). |
| `detected` | The box detection found. |
| `adjusted` | Whether the box compared differs from the one found (both videos are cut to their common picture). |
| `found_utc`, `found_before_run` | When detection ran, and whether that was before this video started (a box kept from an earlier run). |
| `detection_seconds` | How long detection took. |
| `previews_decoded`, `previews_judged`, `too_few_judged` | How many sample frames were decoded and how many were bright enough to judge; too few judged means the whole frame. |
| `median_bars` | Each edge's median bar, in pixels, over the judged frames. |
| `shape_changes` | Whether the bars change during the video (the loosest box is then used). |
| `bars_cut_before_rounding` | The edges cut before rounding to the chroma grid. |

Only `compared` is written for a video whose bars are not detected (mode
`none` or `manual`, and the test video with `source`). The other keys are
`null` where they are not known: every one but `compared` when no box was
kept (the video changes size, detection failed, or saved scores answered
the video and its box was found by a version before 2.5, whose boxes 2.5
does not reuse as it rounds bars differently). `median_bars`,
`shape_changes` and `bars_cut_before_rounding` are `null` when too few
frames were judged.

### `passes`

Each pass of a video's tasks: `task` (`vmaf_gpu`, `ffmpeg`, `perceptual`,
`cvvdp`...), its `metrics`, `together` (several metrics in one pass), `on`
(`gpu` or `cpu`: where the task ran; a GPU VMAF pass can calculate PSNR, SSIM
and XPSNR on the CPU beside it, and each metric's `computed_on` says so),
`seconds` (pauses included), `paused_seconds`, `setup_seconds` (from its start
to its first frame: waiting for black-bar detection, starting its programs,
waiting for the GPU), `frames`, `fps` (frames per second from its first frame
to its end, pauses left out) and `end` (`done`, `failed` or `cancelled`).
`setup_seconds` and `fps` are `null` when the pass ended before it reported
any progress, as a short video's can.

## File details

For the reference and each test video:

| Key | Meaning |
| --- | --- |
| `width`, `height`, `fps`, `codec`, `pixel_format`, `duration` | As FFmpeg reads the video stream. |
| `file_size` | The file's size. |
| `container` | FFmpeg's name for the container (`matroska,webm`). |
| `bit_rate`, `bit_rate_of` | The bitrate, and where it comes from: the `video` stream's own; an `estimate` where the file does not record the video's (the file's rate minus its audio, subtitles, attachments such as fonts, and the container's own bytes); or the whole `file`'s, where the other streams' rates are not known either. |
| `stream_bytes` | The video stream's size, where the container records it. |
| `profile`, `level` | The stream's profile and level (`40`: level 4.0). |
| `bit_depth`, `chroma_subsampling` | From the pixel format (`4:2:0`). |
| `colour` | `range`, `matrix`, `transfer`, `primaries`, `chroma_location`, as the app reads the video: its tags, or those given or taken from the other video where it has none. |
| `writing_library`, `encoder`, `encoder_settings` | The container's and the stream's encoder tags (`Lavf63.1.101`, `Lavc63.1.101 libx264`) and the encoder settings tag where the file has one. |

## Codes

A code is a word that says what an error, failure, note or reason is, for a
program to act on, where the text beside it is for a person and may change
with any version. A code keeps its meaning: it is never renamed or used for
something else, and new ones may be added, so a reader should take a code it
does not know as the general case (an error, a failure, a note).

| Code | Where | Meaning |
| --- | --- | --- |
| `no_match` | `error_code` | A pattern found no file. |
| `unreadable` | `error_code` | FFmpeg could not read the video (no such file, or not a video). |
| `no_metric` | `error_code` | None of the metrics asked for can be calculated for the video. |
| `not_comparable` | `error_code` | The two videos cannot be compared as asked (their lengths differ, say). |
| `run_failed` | `error_code` | The video's run failed as a whole. |
| `timed_out` | `error_code`, `failed_metric_codes`, `metric_status` | `--video-timeout` stopped the video's run: what it had not finished. |
| `reference_unreadable` | `error_code` | The reference could not be read. |
| `no_test_video` | `error_code` | `--batch`: the reference's rows find only it. |
| `batch_unreadable` | `error_code` | The `--batch` list could not be read. |
| `calculation_failed` | `failed_metric_codes`, `metric_status` | The metric failed while it was calculated. |
| `needs_every_frame` | `failed_metric_codes` | CVVDP asked for with `--subsample` above 1. |
| `no_gpu` | `failed_metric_codes` | A metric calculated on the GPU only, and no GPU for it. |
| `not_planned` | `metric_status` | Not run: no task was planned to calculate it. |
| `no_saved_score` | `metric_status` | Window only, not run: a video shown from saved scores that has none for the metric. |
| `hdr_vmaf_colours` | `metric_status` | Not applicable: HDR VMAF scores PQ (HDR10) video, with colours libvmaf converts, only. |
| `gpu_fallback` | `note_codes` | A GPU metric fell back, to the CPU or to stock Vship. |
| `metric_left_out` | `note_codes` | A metric ticked by default in the app cannot be calculated, and was left out. |
| `not_applicable` | `note_codes`, `metric_status` | A metric does not apply to the videos (its `metric_status` says why, with a more precise code such as `hdr_vmaf_colours` where there is one). |
| `colours_taken` | `note_codes` | A video with no colour tags took the other video's. |
| `colours_guessed` | `note_codes` | A video's colours were guessed (no tags, and none to take). |
| `film_grain_unknown` | `note_codes` | Whether a video has film grain could not be found out. |
| `film_grain_left_out` | `note_codes` | A video's film grain was left out of its score. |
| `sdr_display` | `note_codes` | HDR video, CVVDP modelled the app's default SDR display. |
| `near_miss_crop` | `note_codes` | The two videos' pictures were a few lines apart, and one was scaled to the other. |
| `frame_counts` | `note_codes` | Fewer frames were compared than the videos' length should have (`frame_sync` says more). |

The `bitrate` command's videos have an `error_code` too: `no_match`,
`unreadable`, or `no_packets` (a raw YUV file, which has no encoded packets).
`bdrate`'s JSON, when its files give no BD-rate, is its `app`, `version`,
`commit`, `format`, `anchor` and `test` with `error` and `error_code`:
`rates_unreadable` (a file cannot be read, or is not one it reads),
`no_common_metric` or `no_common_sequence`. `target`'s JSON, in the same
way, is its head with `error` and `error_code`: `rates_unreadable`,
`no_metric_scores` (no file has scores of the metric) or `no_sequence` (no
encode is of `--sequence`); a sequence with no answer has a `why_code`:
`too_few_encodes`, `tied_scores` or `bad_number`. `diff`'s, when a run
cannot be read: `run_unreadable`.

## Per-frame scores as JSON

**Export per-frame scores** on the Saved Results tab, and exports with
**Per-frame scores** and **JSON** ticked in Settings > Storage > Exports
include, write each video's per-frame scores as JSON: the same table as
their CSV, for scripts that read JSON. So does the command line's
`--frames-json`.

| Key | Meaning |
| --- | --- |
| `app`, `version`, `commit`, `format` | As in [every file](#the-format-of-each-file): `format` is `"frames-1"`. |
| `reference`, `test` | The videos' paths. |
| `fps` | The comparison's frame rate, the one `time_s` counts in. |
| `columns` | The CSV's header: `frame`, `time_s`, then a column per frame metric (`vmaf`, `vmaf_neg`, `psnr`, `ssim`, `xpsnr` always first, then the others), and `cvvdp_second_jod`, `cvvdp_video_jod` where CVVDP was scored. |
| `rows` | One list a frame, in the columns' order, each on a line of its own: a blank cell of the CSV is `null`, an endless PSNR `"inf"` (before VideoMetricsLab 2.7, Python's `Infinity`, which is not JSON). `time_s` is rounded to the microsecond, as in the CSV. |

In Python, `pandas.DataFrame(doc["rows"], columns=doc["columns"])` gives the
CSV's table.

## Score summary as JSON and CSV

A leaner file than the results: a comparison's scores alone, a row a
metric, with the statistics of its frames and how each was made, as Saved
Results' details panel shows them. The window saves it (the details panel's
save icon, the export menus' **Score summary**, Settings > Storage > Exports
include); the command line writes it with `--score-summary FOLDER` (with
`compare`, `report` and `saved show`), one file of each format a reference,
`<reference>.score-summary.json` and `.csv`.
The columns are those ticked in Settings > Saved Results (the window's
settings, or the command line's `--settings`).

| Key | Meaning |
| --- | --- |
| `app`, `url`, `version`, `commit`, `format` | As in [every file](#the-format-of-each-file), with the program's home; `format` is `"score-summary-1"`. |
| `columns` | The columns each score has, in order: `median`, `stdev`, `min`, `max`, `worst_10`, `worst_5`, `worst_1`, `worst_0_1` (the score the worst 10% ... 0.1% of the frames are worse than), `bands` (each of the metric's score bands, `band` as Metric Graphs heads it and `share` of the frames in it, in percent), `model` (VMAF's model, CVVDP's display), `made` (what calculated it, and its version), `device` (`cpu` or `gpu`), `saved` (when the score was saved, in UTC; `null` where not known). |
| `comparisons` | Each comparison: `label` (its test video, as the window names it), `test` and `reference` (their paths), and `scores`: a row a metric, `metric` (its key), `label` (its name) and `score` (the whole video's), then the columns. |

The CSV has a row a score, headed `Comparison`, `Metric`, `Score` and the
columns' names; numbers at their metric's precision, the bands as `> 90:
98.3%; ...`.

## Worst frames as JSON and CSV

**Worst frames as JSON...** and **Worst frames as CSV...** on the Worst
Frames tab (a right-click on its Worst frames card's save icon or on the
list; the save icon itself writes both at one click) write the dips listed: the metric, the number of dips and the range
shown there.

| Key | Meaning |
| --- | --- |
| `app`, `version`, `commit`, `format` | As in [every file](#the-format-of-each-file): `format` is `"worst-frames-1"`. |
| `metric` | The metric the dips are of (`vmaf`). |
| `dips` | How many dips were asked for (10, 25 or 50); fewer are listed where the video has fewer. |
| `ranges` | The frames looked in, `first_frame` and `last_frame` each (Metric Graphs' highlighted range); empty: the whole video. |
| `worst_frames` | One object per dip, worst first: `rank` (1 is the worst), `frame`, `time` (seconds, where the frame starts) and `scores`, every scored metric's score at that frame by its key, in the app's order of metrics; `null` where a metric has none there. |

The CSV has a row per dip, in the same order: `rank`, `frame`, `time`, then
a column per scored metric (blank where it has no score).

In an export of what Settings > Storage > Exports include ticks (**Worst
frames**), each video's `<video> - worst frames.json` holds every scored
metric's worst frames, as many dips as the Worst Frames tab lists, over the
whole video: the head (`format` is `"worst-frames-report-1"`), `dips`, then
`metrics`, one object a metric (in the app's order
of metrics) with its `metric` and `worst_frames` as above. Its CSV is the
rows above, each led by a `metric` column, one metric after another.
