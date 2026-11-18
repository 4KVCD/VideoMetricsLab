# The results file (JSON and CSV)

The command line's `--json` and the window's **Export results as JSON** write
the same document. **Export results as CSV** writes the same content with a
row for each video, each column named by the key's path in the JSON
(`settings.scaling.compared_at`, `cvvdp_displays.1.score`). A list of plain
values is one cell, its JSON.

This page describes format 2 (VideoMetricsLab 2.5). A later format only adds
keys, so a reader should skip the keys it does not know. A key that applies is
always written, defaults included: a missing value is `null`, never a key left
out. Times are UTC in ISO 8601 (`2026-11-14T21:42:25Z`), durations are in
seconds, sizes are in bytes and bitrates are in bits per second.

## The document

| Key | Meaning |
| --- | --- |
| `app`, `version`, `commit` | The program that wrote the file, its version and its build's commit (`-dirty`: built with uncommitted changes). |
| `format` | `2`. |
| `cancelled` | Command line only: whether the run was cancelled. A cancelled run's videos hold the scores it finished. |
| `run` | The run the scores come from (below). |
| `machine` | The machine that scored them (below). |
| `summary_rules` | How each score and statistic is worked out from the frames (below). |
| `reference` | The reference video: `path` and its file details (see [File details](#file-details)). `null` when the window has no reference. |
| `videos` | One entry for each test video (below). |
| `difference` | Command line with `--difference` only: the first video minus the second. |
| `log_file` | Command line with `--log-file` only: the log's file, its fields filled in. |

With `--batch`, the command line's `--json` is `{"batch": [...]}`: this document
for each reference in turn, or `{"reference": path, "error": why}` for a
reference that could not be read.

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
| `from_saved_scores` | Command line only: whether the scores came from saved scores. |
| `csv`, `run_file`, `segment_files` | Command line only: where `--csv`, `--run-dir` and `--segments-dir` wrote this video's files (the `.segments.json`, then the `.segments.csv`), for a script to find them under the names they were given (`--name`, or a number after a name taken). A further CVVDP display's entry in `cvvdp_displays` has its own `csv` and `run_file`. |
| `width` ... `encoder_settings` | The test video's details (see [File details](#file-details)). |
| `film_grain` | Where the stream carries film grain: its `kinds` and whether it was `scored` `with` or `without` it (and `why_without`). |
| `compared_frames` | How many frame pairs were compared. |
| `frame_pairing` | Whether the video's frames line up with the reference's, frame for frame (see the user guide's "Frames are paired by number"). `status`: `no_mismatch_seen` (as many frame pairs were compared as the videos' length holds); `may_be_out_of_step` (at least 2 fewer, and the frames could not be counted: a file states no count, and Settings > Storage > Read whole files to count frames when they may not line up, or `--frame-counts check`, was off); `out_of_step` (counted: the two have different numbers of frames, and the scores from where they part are wrong); `in_step` (counted, the same: the frames line up); or `not_checked` (a part of the videos, a resolution test, a length not known, or no per-frame metric). `note`: what it means in plain words, and for `may_be_out_of_step` how to tell. `compared_pairs`, `length_frames` (the frames the shorter video's length holds), and `reference_frames` and `test_frames` (`null` when not counted). |
| `vmaf_model` | The VMAF model the result records. |
| `reference_crop`, `test_crop` | The picture compared in each video (`x`, `y`, `width`, `height`), or `null` for the whole frame. |
| `metrics` | Each metric's score and statistics (below). |
| `cvvdp_display`, `cvvdp_displays` | The display CVVDP's score is for, and every display CVVDP was scored on: `display`, `score`, `display_parameters`, `cvvdp` (its full summary), and for a further display its own `timings`. |
| `settings` | Every setting that changes the scores or how to read them (below). |
| `black_bars` | How the black bars were found (below). |
| `metric_status` | For each metric asked for: `status` is `calculated`, `saved` (taken from saved scores), `failed` or `cancelled`, or `not run`; `why` for the failed and not run ones. |
| `decode_plan` | For each pass, its `metrics` and the `decoders` it used for the `source` and the `test` video, after any fallback: FFmpeg's hardware decoder (`cuda`, `d3d11va`, `vulkan`, `videotoolbox`), `vvdec` (VVC decoded on the GPU by vvdec-vulkan) or `software`; for a video decoded in `software`, `source_why` or `test_why` says why (GPU decoding is off, a format or codec GPU decoders do not decode exactly, VVdeC decoding VVC on the CPU, the GPU's decoding having failed in the run, or no GPU decoder here for its codec). |
| `timings` | `started_utc`, `seconds` (pauses included), `paused_seconds`, and `passes` (below). |

`settings`, `black_bars` and `metric_status` are `null` for a saved run
loaded into the window, which does not record them. `decode_plan` and
`timings` are also `null` for a video that did not run: its saved scores
answered it, or a Cancel stopped it before it started.

### `metrics`

Each metric's key (`vmaf`, `vmaf_neg`, `vmaf_v1`, `psnr`, `ssim`, `xpsnr`,
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
| `vmaf_model`, `vmaf_v1_model` | `choice` (`__auto__` picks by the size compared at) and the `model` used. `null` where that VMAF was not asked for (VMAF NEG uses `vmaf_model`). |
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
