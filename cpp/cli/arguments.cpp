#include "arguments.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <regex>
#include <set>

#include "file_names.h"
#include "loop.h"
#include "vml/append.h"
#include "vml/crop_detect.h"
#include "vml/ffprobe.h"
#include "vml/gpu_decoders.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/python_text.h"

namespace vml::cli {

namespace {

// ------------------------------------------------------------------ the texts
//
// As Python's argparse prints them where its output is not a terminal (80
// columns wide). The reference test holds them to the Python app's.

const char* const top_usage = "usage: VideoMetricsLab-cli [-h] [--version] COMMAND ...\n";

const char* const compare_usage = R"(usage: VideoMetricsLab-cli compare [-h] [-m LIST] [--model MODEL]
                                   [--model-v1 MODEL] [--model-hdr MODEL]
                                   [--black-bars {auto,source,none}]
                                   [--source-crop T:B:L:R] [--test-crop T:B:L:R]
                                   [--scale-to {test,reference,both}]
                                   [--scaler {bicubic,bilinear,lanczos,spline}]
                                   [--duration SECONDS | --range START-END |
                                   --frames FIRST-LAST]
                                   [--colour {pq-bt2020,hlg-bt2020,bt709}]
                                   [--film-grain {with,without,both}]
                                   [--fps RATE]
                                   [--size WIDTHxHEIGHT] [--pix-fmt FORMAT]
                                   [--cvvdp-display PRESET] [--cvvdp-fill]
                                   [--subsample N] [--threads N]
                                   [--vmaf-on {gpu,cpu}]
                                   [--vship {vship-fast,vship,vship-hip,vship-cuda}]
                                   [--vship-pass {one,each}] [--cpu]
                                   [--no-gpu-decode |
                                   --gpu-decode {auto,nvidia,intel,amd}]
                                   [--vvc-decoder {gpu,cpu}] [--gpu NAME]
                                   [--parallel N] [--frame-counts {check,skip}]
                                   [--recalculate] [--difference]
                                   [--test-remuxes {on,off}]
                                   [--csv FOLDER] [--json FILE]
                                   [--frames-json FOLDER] [--batch CSV]
                                   [--score-summary FOLDER]
                                   [--scores-csv FILE] [--worst N]
                                   [--segments CONDITIONS]
                                   [--segments-all | --segments-any]
                                   [--segment-min-frames N]
                                   [--segment-merge-frames N]
                                   [--segments-dir FOLDER]
                                   [--segments-report FILE] [--require RULES]
                                   [--fail-on-segments] [--stills FOLDER]
                                   [--stills-from {worst,segments}]
                                   [--stills-count N] [--run-dir FOLDER]
                                   [--log-file FILE] [--name TEMPLATE]
                                   [--notify] [-q] [-v] [--progress {text,jsonl}]
                                   [--timeout SECONDS] [--video-timeout SECONDS]
                                   [--cancel-on-stdin-eof]
                                   [--settings FILE | --no-settings]
                                   [--cache-dir FOLDER] [--no-save | --no-cache]
                                   [REFERENCE] [TEST ...]
)";

const char* const devices_usage = R"(usage: VideoMetricsLab-cli devices [-h] [--json FILE]
                                   [--settings FILE | --no-settings]
)";

const char* const bitrate_usage = R"(usage: VideoMetricsLab-cli bitrate [-h] [--fps RATE] [--csv FOLDER]
                                   [--no-adjust-start] [--difference]
                                   [--json FILE] [-q]
                                   [--settings FILE | --no-settings]
                                   [--cache-dir FOLDER] [--no-save | --no-cache]
                                   VIDEO [VIDEO ...]
)";

const char* const saved_usage = R"(usage: VideoMetricsLab-cli saved [-h] [--filter TEXT] [--json FILE]
                                 [--csv FOLDER] [--frames-json FOLDER]
                                 [--score-summary FOLDER]
                                 [--run-dir FOLDER] [--worst N]
                                 [--segments CONDITIONS]
                                 [--segments-all | --segments-any]
                                 [--segment-min-frames N]
                                 [--segment-merge-frames N]
                                 [--segments-dir FOLDER]
                                 [--segments-report FILE] [--stills FOLDER]
                                 [--stills-from {worst,segments}]
                                 [--stills-count N] [-q]
                                 [--settings FILE | --no-settings]
                                 [--cache-dir FOLDER]
                                 {list,show,delete} [ITEM ...]
)";

const char* const report_usage = R"(usage: VideoMetricsLab-cli report [-h] [--json FILE] [--csv FOLDER]
                                  [--frames-json FOLDER] [--run-dir FOLDER]
                                  [--score-summary FOLDER]
                                  [--worst N] [--segments CONDITIONS]
                                  [--segments-all | --segments-any]
                                  [--segment-min-frames N]
                                  [--segment-merge-frames N]
                                  [--segments-dir FOLDER]
                                  [--segments-report FILE] [--stills FOLDER]
                                  [--stills-from {worst,segments}]
                                  [--stills-count N] [-q]
                                  [--settings FILE | --no-settings]
                                  RUNFILE [RUNFILE ...]
)";

const char* const target_usage = R"(usage: VideoMetricsLab-cli target [-h] --scores FILE --metric KEY --value N
                                  [--by COLUMN] [--stat STAT]
                                  [--sequence NAME] [--json FILE] [-q]
)";

const char* const diff_usage = R"(usage: VideoMetricsLab-cli diff [-h] [--segments CONDITIONS]
                                [--segments-all | --segments-any]
                                [--segment-min-frames N]
                                [--segment-merge-frames N] [--json FILE] [-q]
                                [--settings FILE | --no-settings]
                                [--cache-dir FOLDER]
                                OLD NEW
)";

const char* const bdrate_usage = R"(usage: VideoMetricsLab-cli bdrate [-h] --anchor FILE --test FILE [-m LIST]
                                  [--group-by COLUMN] [--json FILE] [-q]
)";

const char* const examples = R"(examples:
  VideoMetricsLab-cli compare reference.mkv encode.mkv
      the app's default metrics for one encode
  VideoMetricsLab-cli compare reference.mkv encodes\*.mkv -m vmaf,ssimulacra2
      two metrics for every .mkv in a folder, with a summary side by side
  VideoMetricsLab-cli compare reference.mkv a.mkv b.mkv -m all --csv results
      every metric, and each video's per-frame scores in results\a.csv, b.csv
  VideoMetricsLab-cli compare reference.mkv a.mkv b.mkv --difference --csv results
      a.mkv minus b.mkv frame by frame as well, in results\a_minus_b.csv
  VideoMetricsLab-cli compare reference.mkv encode.mkv --duration 30 --cpu
      the first 30 seconds only, everything calculated on the CPU
  VideoMetricsLab-cli compare reference.mkv encode.mkv -m vmaf --json - -q
      only JSON on the output, for a script to read
  VideoMetricsLab-cli compare reference.y4m encode.y4m -m cvvdp --colour pq-bt2020 --duration 2
      HDR10 video in files with no colour tags, the first 2 seconds
  VideoMetricsLab-cli compare reference.mkv encode.mkv --range 1:00:00-1:02:00
      two minutes, an hour into the videos
  VideoMetricsLab-cli compare reference.y4m encode.266
      a raw VVC stream from an encoder, at the frame rate it stores
  VideoMetricsLab-cli compare reference.yuv encode.266 --size 1920x1080 --pix-fmt yuv420p10le --fps 24000/1001
      a raw YUV reference, 1080p 10-bit 4:2:0, and a raw VVC stream
  VideoMetricsLab-cli compare reference.mkv encode.mkv -m cvvdp --cvvdp-display "30-inch 4K HDR monitor, dim room"
      CVVDP on a display other than the app's default (repeat it for several)
  VideoMetricsLab-cli compare --batch clips.csv -m vmaf --scores-csv x264.csv
      every clip's encodes in one run, their rates and scores for bdrate
  VideoMetricsLab-cli bitrate encodes\*.mkv
      each encode's length, frames and bitrate, as the Bitrate Viewer tab
  VideoMetricsLab-cli bitrate encode.266 --csv bitrate
      a raw VVC stream's bitrate, and its graphs' figures as CSV files
  VideoMetricsLab-cli bitrate a.mkv b.mkv --difference --csv bitrate
      a.mkv's bitrate minus b.mkv's as well, in bitrate\a_minus_b_bitrate_*.csv
  VideoMetricsLab-cli bdrate --anchor anchor.csv --test test.csv
      each sequence's BD-rate for each metric the files have, and their mean
  VideoMetricsLab-cli bdrate --anchor anchor.csv --test test.csv --json - -q
      only JSON on the output, for a script to read
  VideoMetricsLab-cli devices
      what would calculate each metric on this PC
  VideoMetricsLab-cli saved list --filter encode
      the saved comparisons of videos named encode..., with their IDs
  VideoMetricsLab-cli saved show 3fa9c1 --segments "vmaf<90" --json -
      a saved comparison's scores and weak segments, its videos gone or not
  VideoMetricsLab-cli report runs\*.metrics.json --worst 5
      the scores of run files, and each metric's 5 worst dips
  VideoMetricsLab-cli target --scores x264.csv --metric vmaf --value 95
      the rate at which each sequence's encodes reach VMAF 95
  VideoMetricsLab-cli diff old.metrics.json new.metrics.json --segments "vmaf<90"
      each score's change, and which weak segments were fixed or are new
)";

const char* const top_sections = R"(
positional arguments:
  COMMAND
    compare   calculate metrics for test videos against a reference
    devices   show FFmpeg and which GPU each metric would use
    bitrate   show videos' bitrate, as the Bitrate Viewer tab
    bdrate    calculate BD-rates from two encoders' rates and scores
    saved     list, show or delete the saved comparisons, without the videos
    report    report the scores of run files (.metrics.json)
    target    find the rate or CRF at which encodes reach a score
    diff      how two runs' scores and weak segments differ

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit

)";

const char* const compare_sections = R"(
Calculates metrics for each TEST video against REFERENCE. Options left out take the app's saved settings (or --settings FILE's, or with --no-settings the defaults); scores already saved by the app or an earlier run are not calculated again (--no-cache: they are).

positional arguments:
  REFERENCE             the reference video (not with --batch)
  TEST                  a test video, or a pattern such as encodes\*.mkv (the
                        reference is left out of what a pattern finds); not
                        with --batch

options:
  -h, --help            show this help message and exit
  -m, --metrics LIST    comma-separated: vmaf, vmaf_neg, vmaf_v1, vmaf_hdr,
                        psnr, ssim, xpsnr, ssimulacra2, butteraugli, cvvdp,
                        psnr_yuv, psnr_yuv_mse, ssim_yuv, ms_ssim,
                        ms_ssim_yuv, psnr_hvs, cambi, ciede2000, vmaf_yuv; a
                        plane of psnr_yuv, ssim_yuv, ms_ssim_yuv, psnr_hvs or
                        vmaf_yuv with _y, _u or _v (psnr_yuv_u); or all (every
                        one but vmaf_hdr, which scores PQ (HDR10) videos only,
                        and the planes) (default: the app's)
  --model MODEL         VMAF v0.6.1's model: auto (4K model for a 4K
                        comparison), standard, 4k, or a model .json file
                        (default: auto)
  --model-v1 MODEL      VMAF v1's model: auto (the one for the size frames are
                        compared at), or a bundled one: vmaf_v1_3d0h or
                        vmaf_v1_5d0h (1080p, 3 or 5 picture heights away),
                        vmaf_v1_1d5h_2160 or vmaf_v1_3d0h_2160 (4K, 1.5 or 3),
                        or their high frame rate models vmaf_v1_hfr_3d0h,
                        vmaf_v1_hfr_5d0h, vmaf_v1_hfr_1d5h_2160,
                        vmaf_v1_hfr_3d0h_2160 (default: auto)
  --model-hdr MODEL     HDR VMAF's model: auto (the one for the size frames
                        are compared at), or a bundled one: hdr_vmaf_3d0h or
                        hdr_vmaf_5d0h (1080p, 3 or 5 picture heights away),
                        hdr_vmaf_1d5h_2160 or hdr_vmaf_3d0h_2160 (4K, 1.5 or
                        3), or their high frame rate models hdr_vmaf_hfr_3d0h,
                        hdr_vmaf_hfr_5d0h, hdr_vmaf_hfr_1d5h_2160,
                        hdr_vmaf_hfr_3d0h_2160 (default: auto)
  --black-bars {auto,source,none}
                        auto: detect and cut black bars; source: detect and
                        cut the source's black bars only, the test video
                        compared at its full frame (an encode cropped when it
                        was made); none: compare the whole pictures (default:
                        auto)
  --source-crop T:B:L:R
                        cut these numbers of lines from the reference's top,
                        bottom, left and right (even numbers, HandBrake's
                        order) in place of finding its black bars; with it
                        or --test-crop, no bars are detected and a video
                        given no crop is compared at its full frame (less
                        an odd-sized frame's odd line)
  --test-crop T:B:L:R   the same for the test videos
  --scale-to {test,reference,both}
                        which video's size the other is scaled to when they
                        differ (default: test); both: a test video of
                        another size is compared both ways, the second
                        named "[upscale-distorted-to-source]"
  --scaler {bicubic,bilinear,lanczos,spline}
                        scaling algorithm (default: bicubic)
  --duration SECONDS    compare only the first SECONDS of each video, in
                        seconds or H:MM:SS, to the millisecond (90, 1.5,
                        1:30.25)
  --range START-END     compare only that part of each video, from START to
                        END, each as --duration reads it (1:00-1:30.5);
                        without END, to the videos' end
  --frames FIRST-LAST   compare only frames FIRST to LAST of each video, both
                        included, counted from 0; without LAST, to the
                        videos' end. CVVDP scores whole seconds: its part of
                        frames, or of a time not in whole seconds, is rounded
                        to the nearest seconds
  --colour, --color {pq-bt2020,hlg-bt2020,bt709}
                        the colours of a video with no colour tags (y4m, say):
                        pq-bt2020 (HDR10), hlg-bt2020 or bt709 (SDR). A tagged
                        video keeps its own. Without it such a video takes the
                        other video's tags, if that one has them, and is
                        otherwise taken as SDR, which changes CVVDP,
                        SSIMULACRA2 and Butteraugli. Scores with colours given
                        or taken are neither read from nor saved to the saved
                        scores
  --film-grain {with,without,both}
                        the film grain a video's stream asks the player to
                        add: with, as players show the video, or without, the
                        coded pictures only (default: the app's; with where it
                        asks); both: a test video with grain is compared
                        with it and without it, the second named "[with
                        film grain]" or "[without film grain]". Grain no
                        decoder here adds as it should (H.274's outside
                        VVC, AFGS1 in H.264) is always left out
  --fps RATE            the frame rate of a raw video stream (a .264, .265,
                        .266 or .obu file, with no container), such as
                        24000/1001, 23.976 or 60, in place of the one it
                        stores: required for one that stores none, and for a
                        raw YUV file (.yuv), which records none. These scores
                        are neither read from nor saved to the saved scores
  --size WIDTHxHEIGHT   the picture size of a raw YUV file (.yuv), which
                        records none, such as 1920x1080: required for one,
                        with --pix-fmt and --fps
  --pix-fmt FORMAT      the pixel format of a raw YUV file (.yuv), as FFmpeg
                        names it, such as yuv420p or yuv420p10le: required for
                        one
  --cvvdp-display PRESET
                        the display CVVDP models, by preset name (default: the
                        app's). Repeat it for a CVVDP score on each display;
                        each extra one decodes the videos again. Presets:
                        '30-inch 4K monitor, office', '24-inch 1080p monitor,
                        office', '30-inch 4K HDR monitor, dim room', '30-inch
                        4K HDR monitor, dark room', '65-inch 4K HDR TV, 1000
                        nits, living room', '65-inch 4K HDR TV, 2000 nits,
                        living room', '65-inch 4K HDR TV, 4000 nits, living
                        room', '6-inch phone', or one saved in the app
  --cvvdp-fill          scale the compared frames up to fill CVVDP's display,
                        aspect kept (default: off, as the official tool: a
                        1080p video covers a quarter of a 4K display). These
                        scores are saved apart
  --subsample N         score every N-th frame (default: 1, every frame; CVVDP
                        needs every frame)
  --threads N           libvmaf threads (default: automatic)
  --vmaf-on {gpu,cpu}   where VMAF v0.6.1, VMAF NEG and VMAF v1 is calculated
                        (default: the app's)
  --vship {vship-fast,vship,vship-hip,vship-cuda}
                        the Vship build that calculates SSIMULACRA2,
                        Butteraugli and CVVDP on the GPU: vship-fast (the
                        app's own; the window's GPU), or stock Vship: vship
                        (the window's Vship Vulkan), or if Vulkan has problems
                        vship-hip (AMD) or vship-cuda (NVIDIA). Stock Vship's
                        three share their scores, which differ slightly from
                        vship-fast's and are saved apart (default: the app's)
  --vship-pass {one,each}
                        one: SSIMULACRA2, Butteraugli and CVVDP in one pass
                        per video, which decodes it once (faster only on a
                        very powerful GPU, and needs much GPU memory); each: a
                        pass for each. The scores are the same (default: the
                        app's)
  --cpu                 calculate on the CPU every metric that can be (CVVDP
                        is GPU only): VMAF as with --vmaf-on cpu, unless
                        --vmaf-on says otherwise, and SSIMULACRA2 and
                        Butteraugli rather than on the GPU where one can
)"
// (Two literals: MSVC takes at most 16,380 bytes in one, error C2026. The
// test SourceLimits.StringLiteralsFitMsvc holds every literal under it.)
R"(  --no-gpu-decode       decode the videos on the CPU
  --gpu-decode {auto,nvidia,intel,amd}
                        decode the videos on the GPU: auto, the first GPU
                        whose decoder takes each video (NVIDIA, Intel, then
                        AMD), or that maker's, which this PC must have; a Mac
                        uses its own (default: the app's)
  --vvc-decoder {gpu,cpu}
                        decode a VVC video on the GPU (vvdec-vulkan,
                        experimental), else on the CPU (VVdeC); the scores are
                        the same, and --no-gpu-decode decodes it on the CPU too
                        (default: the app's)
  --gpu NAME            the GPU that calculates VMAF, VMAF v1, SSIMULACRA2,
                        Butteraugli and CVVDP, by its name or a part of it
                        ("radeon"), on a PC with more than one; its maker's
                        decoder is tried first. auto: a discrete GPU first
                        (default: the app's)
  --parallel N          videos calculated at once on the CPU, 1-2 (default:
                        the app's)
  --frame-counts {check,skip}
                        when a video may be out of sync with the reference
                        (fewer frames were compared than its length should
                        have), the frame counts stored in the files are
                        checked. check: also read a file that doesn't store
                        one from start to finish to count its frames, which
                        can add minutes per video; skip: don't (default: the
                        app's)
  --recalculate         calculate every metric again instead of using saved
                        scores
  --test-remuxes {on,off}
                        on: read each test video calculated once more after
                        the run, so that a remux of it (raw, MKV, MP4) keeps
                        its saved scores, and check a video whose stream
                        begins as a saved one's by reading its stream; off:
                        neither (default: the app's, off unless changed)
  --difference          with two test videos: the first minus the second as
                        well, frame by frame (CVVDP second by second); with
                        --csv also in FOLDER as <first>_minus_<second>.csv
  --csv FOLDER          write each video's per-frame scores to FOLDER as
                        <video>.csv (further --cvvdp-display scores as
                        <video>_<display>.csv); nothing is overwritten: a
                        name taken gets a number, <video>_2.csv, which is
                        said, and is in the --json
  --json FILE           also write the results as JSON to FILE; - prints the
                        JSON instead of the tables
  --frames-json FOLDER  write each video's per-frame scores to FOLDER as JSON,
                        <video>.frames.json, as --csv's but JSON (further
                        --cvvdp-display scores as <video>_<display>.frames.json)
  --score-summary FOLDER
                        write the videos' score summary to FOLDER: each
                        metric's score with the statistics and how it was
                        made that Settings > Saved Results picks, as
                        <reference>.score-summary.json and .csv
  --batch CSV           compare the videos a CSV file lists, in place of
                        REFERENCE and TEST, in one run: a row per test video,
                        with the columns reference and test (a file or a
                        pattern), and if wanted sequence and any others
                        (kept for --scores-csv). Paths are relative to the
                        CSV file's folder. The tables come per reference;
                        --json writes {"batch": [the results of each
                        reference]}
  --scores-csv FILE     write each test video's bitrate (kbps) and scores to
                        FILE, a row per video, as bdrate's --anchor and
                        --test read them
  --worst N             after the tables, each metric's N worst dips: the
                        worst frame of each (a frame within 1 s of a worse
                        one is part of its dip), with every metric's score
                        there
  --segments CONDITIONS
                        list the stretches where scores fall past a
                        threshold, such as "vmaf<90,ssimulacra2<70" (quoted:
                        < and > are the shell's; Butteraugli's worst are
                        high: "butteraugli>2"); a frame is weak where any of
                        them holds. Frames are numbered from the video's
                        first, as its encoder numbers them; a subsample's
                        frames between two weak ones are weak, and CVVDP is
                        judged second by second. "saved" takes the
                        conditions of the weak segment profile in use in the
                        app's Settings > Weak segments, "profile:NAME" those
                        of profile NAME, "preset:KEY" those of a preset
                        (animation, cgi, digital, grainy-film, hdr,
                        preview); those on metrics not compared are left
                        out, and its options apply where none are given
                        here. Conditions after it change it:
                        "preset:grainy-film,vmaf<80" replaces the preset's
                        VMAF condition, and a metric it has none for is
                        added
  --segments-all        a frame is weak only where every condition holds
  --segments-any        a frame is weak where any condition holds, also where
                        the profile or preset says every one must (the
                        default otherwise)
  --segment-min-frames N
                        leave out segments shorter than N frames (default 1)
  --segment-merge-frames N
                        join segments at most N frames apart (default 0)
  --segments-dir FOLDER
                        write each video's segments to FOLDER as
                        <video>.<metrics>.segments.json and .csv, <metrics>
                        the conditions' metrics joined by + (such as
                        <video>.vmaf+ssimulacra2.segments.json)
  --segments-report FILE
                        write every test video's segments to FILE as one
                        report: JSON, or CSV where FILE ends in .csv (a row
                        per segment, led by its video and reference). The
                        fields of --json's path are replaced in it
  --require RULES       a quality gate: the bars each test video's scores
                        must reach, such as "vmaf>=95,vmaf.worst_1_percent>=90"
                        (quoted: < and > are the shell's); a statistic after
                        the metric: score (the default), median, min, max,
                        worst_10_percent, worst_5_percent, worst_1_percent or
                        worst_0.1_percent (CVVDP's but its score: of its
                        seconds). Repeat it to add rules. A video below a bar
                        makes the exit code 3; a metric that does not apply
                        to a video (HDR VMAF of SDR) is left out of its gate
  --fail-on-segments    a quality gate: a test video with a weak segment, by
                        --segments, makes the exit code 3
  --stills FOLDER       write PNGs of the worst frames to FOLDER, as Video
                        Compare's Export frame as PNG makes them (HDR as SDR):
                        <video>.frame<N>.reference.png, .test.png and
                        .side-by-side.png; the JSON lists them
  --stills-from {worst,segments}
                        worst: each metric's worst dips (default); segments:
                        the worst frame of each weak segment (--segments)
  --stills-count N      how many dips of each metric, or segments, the
                        weakest first (default 5)
  --run-dir FOLDER      write each video's result to FOLDER as
                        <video>.metrics.json, the file the app's Save
                        analysis results writes and Metric Graphs' Add saved
                        run opens (further --cvvdp-display results as
                        <video>_<display>.metrics.json)
  --log-file FILE       write the app's log to FILE as well, as the app writes
                        its own: the run in full, whatever -v says (added to
                        if FILE is there; {version}, {date} and {time} are
                        replaced in its path, and with one test video {test}
                        and {reference})
  --name TEMPLATE       name each test video's files (--csv, --segments-dir,
                        --run-dir) after TEMPLATE in place of <video>, with
                        {test} and {reference} (the files' names without
                        extension), {version} (the app's), {date} and {time}
                        (the run's start, YYYYMMDD and HHMMSS), and
                        {sequence} (--batch's); {{ and }} are braces. The same
                        fields are replaced in --json's and --scores-csv's
                        paths
  --notify              send a notification as the run ends, as the app's
                        Settings > Notifications say: to the services on
                        there, with its message, for the outcomes and run
                        lengths it is on for. Whether it was sent is said on
                        the progress lines; the exit code is the run's
  -q, --quiet           no progress, only the results
  -v, --verbose         the app's log as well
  --progress {text,jsonl}
                        text: progress lines for a person; jsonl: a JSON
                        object a line on stderr for a program to read (events
                        run_started, video_started, progress, note, message,
                        problem, video_done, run_done), with -q too
                        (default: text)
  --timeout SECONDS     cancel the run after SECONDS (whole seconds or
                        H:MM:SS) from the command's start, as Ctrl+C does:
                        what finished is given, and the exit code is 124
  --video-timeout SECONDS
                        stop a video whose run takes more than SECONDS from
                        its start; the others go on. It fails with the code
                        timed_out, and the exit code is 124
  --cancel-on-stdin-eof
                        cancel the run, as Ctrl+C does, when standard input
                        ends: for a program that starts this one and cannot
                        send Ctrl+C, closing its end of the pipe
  --settings FILE       read the settings from FILE (a settings.json, such as
                        a copy of the app's) in place of the app's; options
                        left out take its settings, and it is never written
  --no-settings         every setting at its default, as a new install has
                        them, whatever the app's settings say
  --cache-dir FOLDER    read and save the saved scores in FOLDER, in place of
                        the folder the settings name
  --no-save             read the saved scores, but save nothing: no scores,
                        black bars or remux records are written to them
  --no-cache            neither read nor save the saved scores: every metric
                        is calculated, and nothing is kept

)";

const char* const devices_sections = R"(
Shows FFmpeg's version and what would calculate each metric on this PC.

options:
  -h, --help       show this help message and exit
  --json FILE      also write what it finds as JSON to FILE; - prints the JSON
                   instead of the lines
  --settings FILE  read the settings (FFmpeg's folder, the GPU chosen) from
                   FILE in place of the app's
  --no-settings    every setting at its default, whatever the app's say
)";

const char* const bitrate_sections = R"(
Reads each VIDEO's video stream with ffprobe, as the app's Bitrate Viewer tab does, and prints its length, frames and bitrate: the average, and the lowest and highest one-second bitrate (Mb/s).

positional arguments:
  VIDEO                 a video file, or a pattern such as encodes\*.mkv

options:
  -h, --help            show this help message and exit
  --fps RATE            a raw stream's frame rate (.266, .265, .264, .obu),
                        such as 24000/1001, in place of the one it stores:
                        required for one that stores none
  --csv FOLDER          also write each video's frames, seconds and groups of
                        pictures to FOLDER as <video>_bitrate_frames.csv,
                        _seconds.csv and _gops.csv
  --no-adjust-start     with --csv: times from the file's start, not from the
                        stream's first frame
  --difference          with two videos: the first one's bitrate minus the
                        second's as well, from each stream's start: the
                        averages and the seconds both have (paired by number);
                        with --csv also in FOLDER as
                        <first>_minus_<second>_bitrate_frames.csv (at the
                        first's frames), _seconds.csv and _gops.csv
  --json FILE           also write the results as JSON to FILE; - prints the
                        JSON instead of the table
  -q, --quiet           no progress, only the results
  --settings FILE       read the settings (FFmpeg's folder, the saved
                        bitrate's folder) from FILE in place of the app's
  --no-settings         every setting at its default, whatever the app's say
  --cache-dir FOLDER    read and save the saved bitrate in FOLDER, in place of
                        the folder the settings name
  --no-save             read the saved bitrate, but save none
  --no-cache            neither read nor save the saved bitrate

)";

const char* const saved_sections = R"(
The comparisons the app has saved (Settings > Saved Results > Folder, the
Saved Results tab), without their videos: their files may be gone.

positional arguments:
  {list,show,delete}    list: every saved comparison, newest first, with its
                        ID; show: the scores of those ITEM names, as compare
                        gives them; delete: those ITEM names (asks nothing)
  ITEM                  a saved comparison's ID, or the first hex digits of
                        it (at least 6; 12 to delete), or for show a test
                        video's path: every comparison of that video

options:
  -h, --help            show this help message and exit
  --filter TEXT         list: only those whose label, test video's or
                        reference's path has TEXT in it (as the Saved
                        Results tab's Filter)
  --json FILE           also write the results as JSON to FILE (compare's,
                        format 2); - prints the JSON instead of the tables
  --csv FOLDER          write each video's per-frame scores to FOLDER as
                        <video>.csv
  --frames-json FOLDER  write them as JSON, <video>.frames.json
  --score-summary FOLDER
                        write their score summary,
                        <reference>.score-summary.json and .csv (the
                        columns Settings > Saved Results picks)
  --run-dir FOLDER      write each video's result as <video>.metrics.json
  --worst N             after the tables, each metric's N worst dips
  --segments CONDITIONS
                        the weak segments where scores fall past these, such
                        as "vmaf<90,ssimulacra2<70"; or saved, profile:NAME or
                        preset:KEY, as compare takes them
  --segments-all        a frame is weak only where every condition holds
  --segments-any        a frame is weak where any condition holds
  --segment-min-frames N
                        leave out segments shorter than N frames (default 1)
  --segment-merge-frames N
                        join segments at most N frames apart (default 0)
  --segments-dir FOLDER
                        write each video's segments to FOLDER, as compare's
  --segments-report FILE
                        write every video's segments to FILE as one report
  --stills FOLDER       write PNGs of the worst frames to FOLDER, as compare's
                        (the videos must still be there)
  --stills-from {worst,segments}
                        worst: each metric's worst dips (default); segments:
                        the worst frame of each weak segment
  --stills-count N      how many dips of each metric, or segments (default 5)
  -q, --quiet           only the results
  --settings FILE       read the settings (the saved results' folder, the
                        weak segment profiles) from FILE in place of the app's
  --no-settings         every setting at its default, whatever the app's say
  --cache-dir FOLDER    the saved results in FOLDER, in place of the
                        settings' folder (compare --cache-dir's)
)";

const char* const report_sections = R"(
Reports the scores of run files (.metrics.json: compare --run-dir, the
window's Export > Run file) as compare reports them, without their videos.

positional arguments:
  RUNFILE               a run file

options:
  -h, --help            show this help message and exit
  --json FILE           also write the results as JSON to FILE (compare's,
                        format 2); - prints the JSON instead of the tables
  --csv FOLDER          write each video's per-frame scores to FOLDER as
                        <video>.csv
  --frames-json FOLDER  write them as JSON, <video>.frames.json
  --score-summary FOLDER
                        write their score summary,
                        <reference>.score-summary.json and .csv (the
                        columns Settings > Saved Results picks)
  --run-dir FOLDER      write each video's result as <video>.metrics.json
  --worst N             after the tables, each metric's N worst dips
  --segments CONDITIONS
                        the weak segments where scores fall past these, such
                        as "vmaf<90,ssimulacra2<70"; or saved, profile:NAME or
                        preset:KEY, as compare takes them
  --segments-all        a frame is weak only where every condition holds
  --segments-any        a frame is weak where any condition holds
  --segment-min-frames N
                        leave out segments shorter than N frames (default 1)
  --segment-merge-frames N
                        join segments at most N frames apart (default 0)
  --segments-dir FOLDER
                        write each video's segments to FOLDER, as compare's
  --segments-report FILE
                        write every video's segments to FILE as one report
  --stills FOLDER       write PNGs of the worst frames to FOLDER, as compare's
                        (the videos must still be there)
  --stills-from {worst,segments}
                        worst: each metric's worst dips (default); segments:
                        the worst frame of each weak segment
  --stills-count N      how many dips of each metric, or segments (default 5)
  -q, --quiet           only the results
)";

const char* const target_sections = R"(
Where each sequence's encodes reach a score: the rate (or another column,
such as crf) at which the curve bdrate integrates, the log of the rate
against the score interpolated with PCHIP, reaches --value. Beyond the
scores the encodes have, the straight line through the two nearest goes on
to it, marked as extrapolated. Exit code 1 when a sequence has no answer.

options:
  -h, --help       show this help message and exit
  --scores FILE    the encodes: a CSV file with a row per encode (compare
                   --scores-csv: the columns sequence, kbps and the
                   metrics' scores), compare's --json, or run files
                   (.metrics.json); repeat it, or give a pattern, for more
  --metric KEY     the metric: vmaf, vmaf_neg, vmaf_v1, vmaf_hdr, psnr,
                   ssim, psnr_yuv, psnr_yuv_mse, ssim_yuv, xpsnr, ms_ssim,
                   ms_ssim_yuv, psnr_hvs, cambi, ciede2000, vmaf_yuv,
                   ssimulacra2, butteraugli, cvvdp
  --value N        the score to reach
  --by COLUMN      kbps (default), or a CSV file's column of numbers, such
                   as crf from compare --batch's extra columns
  --stat STAT      the statistic to reach, from JSON files only: score (the
                   default, the one the app shows), median, min, max,
                   worst_10_percent, worst_5_percent, worst_1_percent,
                   worst_0.1_percent
  --sequence NAME  only this sequence (its reference's name without the
                   extension)
  --json FILE      also write the answers as JSON to FILE; - prints the
                   JSON instead of the table
  -q, --quiet      no warnings, only the results

)";

const char* const diff_sections = R"(
How a run's scores differ from an earlier one's, for an encoding loop: each
statistic of each metric both have (old, new and the change), and with
--segments, which of the old run's weak segments are fixed, which remain
(overlapping a new one) and which new ones appeared.

positional arguments:
  OLD                   the earlier run: a run file (.metrics.json), or a
                        saved comparison's ID (at least 6 hex digits, saved
                        list gives them)
  NEW                   the later run, in either form

options:
  -h, --help            show this help message and exit
  --segments CONDITIONS
                        the weak segments where scores fall past these, such
                        as "vmaf<90,ssimulacra2<70"; or saved, profile:NAME or
                        preset:KEY, as compare takes them
  --segments-all        a frame is weak only where every condition holds
  --segments-any        a frame is weak where any condition holds
  --segment-min-frames N
                        leave out segments shorter than N frames (default 1)
  --segment-merge-frames N
                        join segments at most N frames apart (default 0)
  --json FILE           also write the differences as JSON to FILE; - prints
                        the JSON instead of the table
  -q, --quiet           no warnings, only the results
  --settings FILE       read the settings (the saved results' folder, the
                        weak segment profiles) from FILE in place of the app's
  --no-settings         every setting at its default, whatever the app's say
  --cache-dir FOLDER    a saved comparison's ID is looked for in FOLDER, in
                        place of the settings' folder (compare --cache-dir's)

)";

const char* const bdrate_sections = R"(
Calculates each sequence's BD-rate (Bjontegaard delta rate) for each metric: how much more (+) or less (-) bitrate the test encoder needs than the anchor for the same score, in %, over the scores both reach; then their mean and median. Each encoder's rate curve is the log of its rate against the score, interpolated with PCHIP; give each sequence at least 4 encodes.

options:
  -h, --help          show this help message and exit
  --anchor FILE       the anchor's encodes: a CSV file with a row per encode
                      and the columns sequence, kbps, and the scores of any
                      metrics by key (vmaf, psnr, ...; empty where not
                      scored), other columns only read for --group-by; or a
                      JSON file, compare's --json (an encode per video) or a
                      run file (.metrics.json)
  --test FILE         the test encoder's encodes, in either form; sequences
                      are paired by name
  -m, --metrics LIST  comma-separated: vmaf, vmaf_neg, vmaf_v1, vmaf_hdr,
                      psnr, ssim, xpsnr, ssimulacra2, butteraugli, cvvdp,
                      psnr_yuv, psnr_yuv_mse, ssim_yuv, ms_ssim, ms_ssim_yuv,
                      psnr_hvs, cambi, ciede2000, vmaf_yuv; a plane of
                      psnr_yuv, ssim_yuv, ms_ssim_yuv, psnr_hvs or vmaf_yuv
                      with _y, _u or _v (psnr_yuv_u); or all but vmaf_hdr and
                      the planes (default: those the files have columns for)
  --group-by COLUMN   a column of the anchor's file, such as class: a mean and
                      median for each of its values as well
  --json FILE         also write the results as JSON to FILE; - prints the
                      JSON instead of the table
  -q, --quiet         no warnings, only the results

)";

const std::string version_text = std::string("VideoMetricsLab ") + VML_VERSION;

// ------------------------------------------------------------ argparse's way
//
// A port of what Python's argparse does with the parsers in
// vmaf_app/cli.py: the same reading of option strings (abbreviations,
// --option=value, -mVALUE, -qv), of arguments ("--", negative numbers),
// the same order of reading and of errors, and the same messages.

// A wrong command line: argparse's ArgumentError. The message is what
// follows "error: ".
struct ArgumentError {
    std::string message;
};

// Help or the version, to be printed (the program then exits with 0).
struct Printed {
    std::string text;
};

// How many strings an action takes (argparse's nargs).
enum class Nargs {
    none,         // a flag
    one,          // a value
    one_or_more,  // a positional's list
    command,      // a command and the strings after it, for its own parser
};

struct Action {
    std::vector<std::string> option_strings;  // empty for a positional
    std::string metavar;                      // a positional's name
    Nargs nargs;
    bool required = false;
    // Converts the action's strings, checks them, and keeps the value.
    // Throws ArgumentError.
    std::function<void(const std::vector<std::string>&)> take;

    // Its name in messages: "-m/--metrics", or "REFERENCE".
    std::string name() const {
        if (option_strings.empty()) {
            return metavar;
        }
        std::string joined;
        for (const std::string& option : option_strings) {
            joined += (joined.empty() ? "" : "/") + option;
        }
        return joined;
    }
};

struct Parser {
    std::vector<Action> actions;  // in the order they were added
    // Pairs of options that cannot be given together, by their first
    // option strings.
    std::vector<std::pair<std::string, std::string>> exclusive;

    const Action* option(const std::string& name) const {
        for (const Action& action : actions) {
            if (!action.option_strings.empty() && action.option_strings.front() == name) {
                return &action;
            }
        }
        return nullptr;
    }
};

// One reading of an option string (argparse's option tuple): its action
// (nullptr: no such option), the option string it was read as, and an
// argument given in the same string.
struct OptionTuple {
    const Action* action;
    std::string option_string;
    std::optional<std::string> separator;  // "=", "" or none
    std::optional<std::string> explicit_argument;
};

using OptionStrings = std::vector<std::pair<std::string, const Action*>>;  // in the order they were added

const Action* exact_option(const OptionStrings& options, const std::string& text) {
    for (const auto& [option, action] : options) {
        if (option == text) {
            return action;
        }
    }
    return nullptr;
}

// argparse's _get_option_tuples: the options a string can be an
// abbreviation of, or a short option with its argument attached.
std::vector<OptionTuple> option_tuples(const OptionStrings& options, const std::string& text) {
    std::vector<OptionTuple> found;
    const std::size_t equals = text.find('=');
    const std::string prefix = text.substr(0, equals);
    const std::optional<std::string> separator =
        equals == std::string::npos ? std::nullopt : std::optional<std::string>("=");
    const std::optional<std::string> explicit_argument =
        equals == std::string::npos ? std::nullopt : std::optional(text.substr(equals + 1));
    if (text[1] == '-') {
        // A long option is only split at the "=".
        for (const auto& [option, action] : options) {
            if (option.starts_with(prefix)) {
                found.push_back({action, option, separator, explicit_argument});
            }
        }
        return found;
    }
    // A short option can have its argument attached: -mvmaf.
    const std::string short_prefix = text.substr(0, 2);
    for (const auto& [option, action] : options) {
        if (option == short_prefix) {
            found.push_back({action, option, "", text.substr(2)});
        } else if (option.starts_with(prefix)) {
            found.push_back({action, option, separator, explicit_argument});
        }
    }
    return found;
}

// argparse's _parse_optional: how a string is read, as one or more options,
// or nothing for an argument.
std::optional<std::vector<OptionTuple>> parse_optional(const OptionStrings& options, const std::string& text) {
    if (text.empty() || text[0] != '-') {
        return std::nullopt;
    }
    if (const Action* action = exact_option(options, text)) {
        return std::vector<OptionTuple>{{action, text, std::nullopt, std::nullopt}};
    }
    if (text.size() == 1) {
        return std::nullopt;  // "-" is an argument
    }
    if (const std::size_t equals = text.find('='); equals != std::string::npos) {
        if (const Action* action = exact_option(options, text.substr(0, equals))) {
            return std::vector<OptionTuple>{{action, text.substr(0, equals), "=", text.substr(equals + 1)}};
        }
    }
    if (std::vector<OptionTuple> found = option_tuples(options, text); !found.empty()) {
        return found;
    }
    // A negative number is an argument (no option here looks like one).
    if (text.size() >= 2 && (std::isdigit(static_cast<unsigned char>(text[1])) ||
                             (text[1] == '.' && text.size() >= 3 && std::isdigit(static_cast<unsigned char>(text[2]))))) {
        return std::nullopt;
    }
    if (text.find(' ') != std::string::npos) {
        return std::nullopt;
    }
    return std::vector<OptionTuple>{{nullptr, text, std::nullopt, std::nullopt}};
}

// How many of the pattern's strings, from `at`, each positional takes, as
// argparse's regular expressions match them: -*A-* for one, -*A[A-]* for one
// or more, -*A[-AO]* for a command. Nothing if they cannot all match.
std::optional<std::vector<std::size_t>> match_positionals(const std::vector<const Action*>& positionals,
                                                          std::size_t first, const std::string& pattern,
                                                          std::size_t at) {
    if (first == positionals.size()) {
        return std::vector<std::size_t>{};
    }
    std::size_t end = at;
    while (end < pattern.size() && pattern[end] == '-') {
        ++end;
    }
    if (end >= pattern.size() || pattern[end] != 'A') {
        return std::nullopt;
    }
    ++end;
    const Nargs nargs = positionals[first]->nargs;
    const auto in_tail = [nargs](char c) {
        return c == '-' || (nargs != Nargs::one && c == 'A') || (nargs == Nargs::command && c == 'O');
    };
    std::size_t longest = end;
    while (longest < pattern.size() && in_tail(pattern[longest])) {
        ++longest;
    }
    // Greedy: the longest tail the rest can follow.
    for (std::size_t stop = longest + 1; stop-- > end;) {
        if (auto rest = match_positionals(positionals, first + 1, pattern, stop)) {
            rest->insert(rest->begin(), stop - at);
            return rest;
        }
    }
    return std::nullopt;
}

// Reads command-line strings with a Parser, as argparse's parse_known_args
// reads them: its _parse_known_args, whose consume_optional and
// consume_positionals are the methods of the same names.
class ArgumentReader {
public:
    ArgumentReader(const Parser& parser, const std::vector<std::string>& strings);

    // The strings that are none of the parser's. Throws ArgumentError, and
    // Printed (from a help or version action).
    std::vector<std::string> read();

private:
    void take_action(const Action& action, const std::vector<std::string>& values);
    std::size_t consume_optional(std::size_t start);
    std::size_t consume_positionals(std::size_t start);
    void check_required() const;

    const Parser& parser_;
    const std::vector<std::string>& strings_;
    OptionStrings options_;
    std::vector<const Action*> positionals_;  // those not taken yet
    // Each string as an option (O), an argument (A), or the "--" (-) after
    // which every string is an argument; and how each option is read.
    std::string pattern_;
    std::map<std::size_t, std::vector<OptionTuple>> option_at_;
    std::set<const Action*> seen_;
    std::vector<std::string> extras_;
};

ArgumentReader::ArgumentReader(const Parser& parser, const std::vector<std::string>& strings)
    : parser_(parser), strings_(strings) {
    for (const Action& action : parser.actions) {
        for (const std::string& option : action.option_strings) {
            options_.emplace_back(option, &action);
        }
        if (action.option_strings.empty()) {
            positionals_.push_back(&action);
        }
    }
    for (std::size_t i = 0; i < strings.size(); ++i) {
        if (strings[i] == "--") {
            pattern_ += '-';
            pattern_.append(strings.size() - i - 1, 'A');
            break;
        }
        if (std::optional<std::vector<OptionTuple>> tuples = parse_optional(options_, strings[i])) {
            option_at_[i] = std::move(*tuples);
            pattern_ += 'O';
        } else {
            pattern_ += 'A';
        }
    }
}

std::vector<std::string> ArgumentReader::read() {
    // Positionals and options in turn, to the last option.
    std::size_t start = 0;
    if (!option_at_.empty()) {
        const std::size_t last_option = option_at_.rbegin()->first;
        while (start <= last_option) {
            std::size_t next_option = start;
            while (next_option <= last_option && !option_at_.contains(next_option)) {
                ++next_option;
            }
            if (start != next_option) {
                const std::size_t end = consume_positionals(start);
                const bool moved = end > start;
                start = end;
                if (moved) {
                    continue;
                }
            }
            if (!option_at_.contains(start)) {
                extras_.insert(extras_.end(), strings_.begin() + start, strings_.begin() + next_option);
                start = next_option;
            }
            start = consume_optional(start);
        }
    }
    const std::size_t stop = consume_positionals(start);
    extras_.insert(extras_.end(), strings_.begin() + stop, strings_.end());
    check_required();
    return std::move(extras_);
}

// Takes `values` for `action`; not with an option already given that
// excludes it.
void ArgumentReader::take_action(const Action& action, const std::vector<std::string>& values) {
    seen_.insert(&action);
    action.take(values);
    for (const auto& [a, b] : parser_.exclusive) {
        const Action* other = &action == parser_.option(a)   ? parser_.option(b)
                              : &action == parser_.option(b) ? parser_.option(a)
                                                             : nullptr;
        if (other != nullptr && seen_.contains(other)) {
            throw ArgumentError{"argument " + action.name() + ": not allowed with argument " + other->name()};
        }
    }
}

// Takes the option at `start`, its value, and the flags run together with
// it (-qv); returns where the next string to read is.
std::size_t ArgumentReader::consume_optional(std::size_t start) {
    const std::vector<OptionTuple>& tuples = option_at_.at(start);
    if (tuples.size() > 1) {
        std::string matches;
        for (const OptionTuple& tuple : tuples) {
            matches += (matches.empty() ? "" : ", ") + tuple.option_string;
        }
        throw ArgumentError{"ambiguous option: " + strings_[start] + " could match " + matches};
    }
    OptionTuple tuple = tuples.front();
    std::vector<std::pair<const Action*, std::vector<std::string>>> taken;
    std::size_t stop = start + 1;
    while (true) {
        if (tuple.action == nullptr) {
            extras_.push_back(strings_[start]);
            return start + 1;
        }
        const Action& action = *tuple.action;
        if (!tuple.explicit_argument) {
            // The option's value is the next string, if it is an argument.
            std::vector<std::string> values;
            if (action.nargs == Nargs::one) {
                if (start + 1 >= pattern_.size() || pattern_[start + 1] != 'A') {
                    throw ArgumentError{"argument " + action.name() + ": expected one argument"};
                }
                values.push_back(strings_[start + 1]);
            }
            stop = start + 1 + values.size();
            taken.emplace_back(&action, std::move(values));
            break;
        }
        const std::string argument = *tuple.explicit_argument;
        if (action.nargs == Nargs::one) {
            taken.emplace_back(&action, std::vector<std::string>{argument});
            break;
        }
        // A flag with more after it: -qv is -q -v.
        if (tuple.option_string[1] == '-' || argument.empty() || (tuple.separator && !tuple.separator->empty()) ||
            argument[0] == '-') {
            throw ArgumentError{"argument " + action.name() + ": ignored explicit argument " + python_string_repr(argument)};
        }
        taken.emplace_back(&action, std::vector<std::string>{});
        const std::string next = std::string("-") + argument[0];
        if (const Action* other = exact_option(options_, next)) {
            const std::string rest = argument.substr(1);
            tuple = {other, next, std::nullopt, std::nullopt};
            if (!rest.empty()) {
                tuple.separator = rest[0] == '=' ? "=" : "";
                tuple.explicit_argument = rest[0] == '=' ? rest.substr(1) : rest;
            }
        } else {
            extras_.push_back("-" + argument);
            break;
        }
    }
    for (const auto& [action, values] : taken) {
        take_action(*action, values);
    }
    return stop;
}

// Takes as many of the positionals left as can match the strings from
// `start`, from the first; returns where the next string to read is.
std::size_t ArgumentReader::consume_positionals(std::size_t start) {
    std::vector<std::size_t> counts;
    for (std::size_t matched = positionals_.size(); matched > 0; --matched) {
        const std::vector<const Action*> some(positionals_.begin(), positionals_.begin() + matched);
        if (std::optional<std::vector<std::size_t>> found = match_positionals(some, 0, pattern_, start)) {
            counts = std::move(*found);
            break;
        }
    }
    for (std::size_t i = 0; i < counts.size(); ++i) {
        std::vector<std::string> values(strings_.begin() + start, strings_.begin() + start + counts[i]);
        // The "--" is not a value.
        const std::size_t dash = pattern_.find('-', start);
        if (positionals_[i]->nargs == Nargs::command ? pattern_[start] == '-' : dash < start + counts[i]) {
            values.erase(std::find(values.begin(), values.end(), "--"));
        }
        start += counts[i];
        take_action(*positionals_[i], values);
    }
    positionals_.erase(positionals_.begin(), positionals_.begin() + counts.size());
    return start;
}

// Throws for the required arguments not given, naming them all.
void ArgumentReader::check_required() const {
    std::string missing;
    for (const Action& action : parser_.actions) {
        if (action.required && !seen_.contains(&action)) {
            missing += (missing.empty() ? "" : ", ") + action.name();
        }
    }
    if (!missing.empty()) {
        throw ArgumentError{"the following arguments are required: " + missing};
    }
}

// Reads `strings` with `parser`, as argparse's parse_known_args does, and
// returns the strings that are none of its. Throws ArgumentError, and
// Printed (from a help or version action).
std::vector<std::string> parse_known(const Parser& parser, const std::vector<std::string>& strings) {
    return ArgumentReader(parser, strings).read();
}

// ---------------------------------------------------------------- the values

// Every metric's key, in the registry's order: "vmaf, vmaf_neg, ...".
std::string metric_keys_text() {
    std::string keys;
    for (const MetricDefinition& metric : metrics()) {
        keys += (keys.empty() ? "" : ", ") + std::string(metric.key);
    }
    return keys;
}

// The metric keys named in a --metrics value, in the registry's order. "all"
// is every metric but HDR VMAF, which refuses all but PQ videos (with it, a
// run of SDR videos asked for "all" would fail), and the planes, which are
// named one by one.
std::vector<std::string> metric_list(const std::string& text) {
    std::vector<std::string> keys;
    if (ascii_lower(python_strip(text)) == "all") {
        for (const MetricDefinition& metric : metrics()) {
            if (metric.key != "vmaf_hdr" && !metric.is_plane()) {
                keys.emplace_back(metric.key);
            }
        }
        return keys;
    }
    std::vector<std::string> named;
    std::string unknown;
    for (std::size_t start = 0; start <= text.size();) {
        std::size_t end = text.find(',', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string part = ascii_lower(python_strip(text.substr(start, end - start)));
        if (!part.empty() && std::find(named.begin(), named.end(), part) == named.end()) {
            named.push_back(part);
            if (find_metric(part) == nullptr) {
                unknown += (unknown.empty() ? "" : ", ") + part;
            }
        }
        start = end + 1;
    }
    if (!unknown.empty() || named.empty()) {
        throw ArgumentError{"unknown metric " + (unknown.empty() ? text : unknown) +
                            "; choose from: all, " + metric_keys_text()};
    }
    for (const MetricDefinition& metric : metrics()) {
        if (std::find(named.begin(), named.end(), metric.key) != named.end()) {
            keys.emplace_back(metric.key);
        }
    }
    return keys;
}

// A whole number above 0, read as Python's int() reads it (the app's
// _positive types).
long long positive_int(const std::string& text) {
    const std::optional<long long> value = python_int(text);
    if (!value) {
        throw ArgumentError{"invalid parse value: " + python_string_repr(text)};
    }
    if (*value <= 0) {
        throw ArgumentError{"must be above 0"};
    }
    return *value;
}

// The conditions of a --segments value ("vmaf<90,ssimulacra2<70"), each
// naming a metric.
std::vector<WeakCondition> weak_conditions(const std::string& text) {
    std::vector<WeakCondition> conditions;
    for (std::size_t start = 0; start <= text.size();) {
        std::size_t end = text.find(',', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string part = ascii_lower(python_strip(text.substr(start, end - start)));
        start = end + 1;
        if (part.empty()) {
            continue;
        }
        std::optional<WeakCondition> condition = parse_weak_condition(part);
        if (!condition) {
            throw ArgumentError{"invalid condition " + python_string_repr(part) +
                                ": write METRIC<VALUE or METRIC>VALUE, such as vmaf<90"};
        }
        condition->key = std::string(python_strip(condition->key));
        if (find_metric(condition->key) == nullptr) {
            throw ArgumentError{"unknown metric " + condition->key + "; choose from: " + metric_keys_text()};
        }
        conditions.push_back(std::move(*condition));
    }
    if (conditions.empty()) {
        throw ArgumentError{"no condition given: write METRIC<VALUE or METRIC>VALUE, such as vmaf<90"};
    }
    return conditions;
}

// --segments' value: conditions; or "saved", "profile:NAME" or "preset:KEY",
// with conditions after it, if any, that change it (take_saved_segments).
void read_segments(const std::string& value, CompareArguments& parsed) {
    std::optional<std::string> source;
    std::string conditions;
    for (std::size_t start = 0; start <= value.size();) {
        std::size_t end = value.find(',', start);
        if (end == std::string::npos) {
            end = value.size();
        }
        const std::string part(python_strip(value.substr(start, end - start)));
        start = end + 1;
        const std::string lower = ascii_lower(part);
        if (lower == "saved" || lower.starts_with("profile:") || lower.starts_with("preset:")) {
            if (source) {
                throw ArgumentError{"give one of saved, profile:NAME and preset:KEY, not " + *source + " and " + part};
            }
            source = part;
        } else {
            conditions += (conditions.empty() ? "" : ",") + part;
        }
    }
    const std::string lower = source ? ascii_lower(*source) : std::string();
    parsed.segments_saved = source.has_value();
    parsed.segments_profile = lower.starts_with("profile:")
                                  ? std::optional(std::string(python_strip(source->substr(8))))
                                  : std::nullopt;
    parsed.segments_preset = lower.starts_with("preset:")
                                 ? std::optional(std::string(python_strip(source->substr(7))))
                                 : std::nullopt;
    // Conditions alone must be some; after a profile, none is fine.
    parsed.segments = source && conditions.empty() ? std::vector<WeakCondition>() : weak_conditions(conditions);
}

// A whole number from 0 up, read as Python's int() reads it.
int int_from_zero(const std::string& text) {
    const std::optional<long long> value = python_int(text);
    if (!value) {
        throw ArgumentError{"invalid int value: " + python_string_repr(text)};
    }
    if (*value < 0 || *value > std::numeric_limits<int>::max()) {
        throw ArgumentError{"must be 0 or more"};
    }
    return static_cast<int>(*value);
}

// A whole number from 1 to `most`, the range the app's own field takes
// (options_panel.cpp): --threads and --subsample.
int int_up_to(const std::string& text, int most) {
    const long long value = positive_int(text);
    if (value > most) {
        throw ArgumentError{"must be from 1 to " + std::to_string(most)};
    }
    return static_cast<int>(value);
}

// A time in milliseconds: "90", "1:30" or "1:01:30" (the minutes and
// seconds after the first part in two digits, under 60); with `fraction`,
// its seconds may have up to 3 decimals ("1.5", "1:30.25", as the window's
// range is to the millisecond). Nothing for other text.
std::optional<long long> time_ms(std::string_view text, bool fraction) {
    static const std::regex time(R"re(([0-9]{1,9})(?::([0-9]{2}))?(?::([0-9]{2}))?(?:\.([0-9]{1,3}))?)re");
    const std::string stripped(python_strip(text));
    std::smatch parts;
    if (!std::regex_match(stripped, parts, time) || (parts[4].matched && !fraction)) {
        return std::nullopt;
    }
    long long seconds = 0;  // no overflow: 9 digits of hours
    for (std::size_t part = 1; part <= 3 && parts[part].matched; ++part) {
        const long long value = std::stoll(parts[part].str());
        if (part > 1 && value >= 60) {
            return std::nullopt;
        }
        seconds = seconds * 60 + value;
    }
    std::string decimals = parts[4].str();  // "5" is 500 ms
    decimals.resize(3, '0');
    return seconds * 1000 + std::stoll(decimals);
}

// --timeout and --video-timeout: whole seconds, above 0.
long long duration_seconds(const std::string& text) {
    const std::optional<long long> ms = time_ms(text, false);
    if (!ms) {
        throw ArgumentError{"invalid time: " + python_string_repr(text) + " (whole seconds, or H:MM:SS)"};
    }
    if (*ms <= 0) {
        throw ArgumentError{"must be above 0"};
    }
    return *ms / 1000;
}

// --duration: the first milliseconds of the videos, above 0.
long long duration_ms(const std::string& text) {
    const std::optional<long long> ms = time_ms(text, true);
    if (!ms) {
        throw ArgumentError{"invalid time: " + python_string_repr(text) + " (seconds or H:MM:SS, to the millisecond)"};
    }
    if (*ms <= 0) {
        throw ArgumentError{"must be above 0"};
    }
    return *ms;
}

// --range: START-END in milliseconds, each as --duration reads it; no
// END: to the videos' end (0). An END written, 0 included, is after START.
std::pair<long long, long long> time_range_ms(const std::string& text) {
    const std::size_t dash = text.find('-');
    const std::optional<long long> start =
        dash == std::string::npos ? std::nullopt : time_ms(text.substr(0, dash), true);
    const std::string end_text = dash == std::string::npos ? std::string() : text.substr(dash + 1);
    const bool to_the_end = python_strip(end_text).empty();
    const std::optional<long long> end = to_the_end ? std::optional(0LL) : time_ms(end_text, true);
    if (!start || !end) {
        throw ArgumentError{"invalid time range: " + python_string_repr(text) +
                            " (START-END, each seconds or H:MM:SS, to the millisecond; no END: to the end)"};
    }
    if (!to_the_end && *end <= *start) {
        throw ArgumentError{"its end must be after its start"};
    }
    return {*start, *end};
}

// --frames: FIRST-LAST, whole frame numbers counted from 0, LAST included;
// no LAST: to the videos' end. As a range: the first frame and the one past
// the last (0: to the end).
std::pair<long long, long long> frame_span(const std::string& text) {
    static const std::regex form(R"(\s*(\d{1,15})\s*-\s*(\d{1,15})?\s*)");
    std::smatch parts;
    if (!std::regex_match(text, parts, form)) {
        throw ArgumentError{"invalid frame range: " + python_string_repr(text) +
                            " (FIRST-LAST, frame numbers from 0; no LAST: to the end)"};
    }
    const long long first = std::stoll(parts[1].str());
    if (!parts[2].matched) {
        return {first, 0};
    }
    const long long last = std::stoll(parts[2].str());
    if (last < first) {
        throw ArgumentError{"its last frame must not be before its first"};
    }
    return {first, last + 1};
}

// A crop entered by hand (--source-crop, --test-crop): top:bottom:left:right,
// even numbers of lines (box_from_edges says why an odd one is refused).
CropEdges crop_edges(const std::string& text) {
    const std::optional<CropEdges> edges = parse_crop_edges(text);
    if (!edges) {
        throw ArgumentError{"invalid crop: " + python_string_repr(text) +
                            " (TOP:BOTTOM:LEFT:RIGHT, lines cut from each edge, as 22:22:0:0)"};
    }
    for (const int lines : {edges->top, edges->bottom, edges->left, edges->right}) {
        if (lines % 2 != 0) {
            throw ArgumentError{"invalid crop: " + python_string_repr(text) +
                                " (each edge an even number of lines: 4:2:0 video keeps its colour in pairs)"};
        }
    }
    return *edges;
}

// A frame rate (--fps): a fraction of whole numbers below 2^53, as many as a
// double holds exactly ("24000/1001"), or a number ("23.976"), above 0.
double frame_rate(const std::string& text) {
    const std::optional<double> value = typed_frame_rate(text);
    if (!value) {
        throw ArgumentError{"invalid parse value: " + python_string_repr(text)};
    }
    if (!(*value > 0) || !std::isfinite(*value)) {
        throw ArgumentError{"must be above 0"};
    }
    if (*value < lowest_typed_frame_rate || *value > highest_typed_frame_rate) {
        throw ArgumentError{"must be between 0.001 and 10000"};
    }
    return *value;
}

// A picture size (--size): a width and a height of 1 to 5 digits, above 0,
// given back as FFmpeg's -video_size reads it ("1920x1080").
std::string picture_size(const std::string& text) {
    static const std::regex size("([0-9]{1,5})x([0-9]{1,5})");
    const std::string stripped(python_strip(text));
    std::smatch parts;
    if (!std::regex_match(stripped, parts, size)) {
        throw ArgumentError{"invalid parse value: " + python_string_repr(text)};
    }
    const long long width = *python_int(parts[1].str());
    const long long height = *python_int(parts[2].str());
    if (width <= 0 || height <= 0) {
        throw ArgumentError{"must be above 0"};
    }
    return std::to_string(width) + "x" + std::to_string(height);
}

// A pixel format's name (--pix-fmt), as FFmpeg writes them: lower-case
// letters, digits and underscores ("yuv420p10le").
std::string pixel_format(const std::string& text) {
    static const std::regex name("[0-9a-z_]+");
    if (!std::regex_match(text, name)) {
        throw ArgumentError{"invalid parse value: " + python_string_repr(text)};
    }
    return text;
}

// A model option's words (--model-v1, --model-hdr): auto, then the bundled
// models' ids.
std::vector<std::string> model_words(const std::vector<ModelChoice>& choices) {
    std::vector<std::string> words;
    for (const ModelChoice& model : choices) {
        words.push_back(model.word);
    }
    return words;
}

std::string choice(const std::string& value, const std::vector<std::string>& choices) {
    if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
        std::string listed;
        for (const std::string& c : choices) {
            listed += (listed.empty() ? "" : ", ") + c;
        }
        throw ArgumentError{"invalid choice: " + python_string_repr(value) + " (choose from " + listed + ")"};
    }
    return value;
}

// An action that converts its one value with `read`, its message prefixed
// with the action's name, as argparse reports a type's error.
Action option(std::vector<std::string> names, std::function<void(const std::string&)> read) {
    Action action{std::move(names), "", Nargs::one, false, nullptr};
    const std::string name = action.name();
    action.take = [name, read = std::move(read)](const std::vector<std::string>& values) {
        try {
            read(values.front());
        } catch (const ArgumentError& error) {
            throw ArgumentError{"argument " + name + ": " + error.message};
        }
    };
    return action;
}

Action flag(std::vector<std::string> names, bool& target) {
    return {std::move(names), "", Nargs::none, false, [&target](const std::vector<std::string>&) { target = true; }};
}

Action help(const std::string& text) {
    return {{"-h", "--help"}, "", Nargs::none, false, [text](const std::vector<std::string>&) { throw Printed{text}; }};
}

// The options that choose the settings (--settings, --no-settings) and, with
// `cache`, the saved scores (--cache-dir, --no-save, --no-cache), added to
// `parser`, keeping what they read in `profile`.
void add_profile_options(Parser& parser, ProfileArguments& profile, bool cache) {
    parser.actions.push_back(option({"--settings"}, [&profile](const std::string& value) {
        profile.settings_file = value;
    }));
    parser.actions.push_back(flag({"--no-settings"}, profile.no_settings));
    parser.exclusive.emplace_back("--settings", "--no-settings");
    if (cache) {
        parser.actions.push_back(option({"--cache-dir"}, [&profile](const std::string& value) {
            profile.cache_dir = value;
        }));
        parser.actions.push_back(flag({"--no-save"}, profile.no_save));
        parser.actions.push_back(flag({"--no-cache"}, profile.no_cache));
        parser.exclusive.emplace_back("--no-save", "--no-cache");
        parser.exclusive.emplace_back("--cache-dir", "--no-cache");
    }
}

// The compare command's parser, keeping what it reads in `parsed`.
Parser compare_parser(CompareArguments& parsed) {
    const std::string help_text = std::string(compare_usage) + compare_sections + examples;
    Parser parser;
    parser.actions = {
        help(help_text),
        // Required unless --batch gives them: checked once read (batch_misuse).
        {{}, "REFERENCE", Nargs::one, false,
         [&parsed](const std::vector<std::string>& values) { parsed.reference = values.front(); }},
        {{}, "TEST", Nargs::one_or_more, false,
         [&parsed](const std::vector<std::string>& values) { parsed.tests = values; }},
        option({"-m", "--metrics"}, [&parsed](const std::string& value) { parsed.metrics = metric_list(value); }),
        option({"--model"}, [&parsed](const std::string& value) { parsed.model = value; }),
        option({"--model-v1"},
               [&parsed](const std::string& value) { parsed.model_v1 = choice(value, model_words(v1_model_choices())); }),
        option({"--model-hdr"},
               [&parsed](const std::string& value) {
                   parsed.model_hdr = choice(value, model_words(hdr_model_choices()));
               }),
        option({"--black-bars"},
               [&parsed](const std::string& value) {
                   parsed.black_bars = choice(value, {"auto", "source", "none"});
                   parsed.black_bars_given = true;
               }),
        option({"--source-crop"}, [&parsed](const std::string& value) { parsed.source_crop = crop_edges(value); }),
        option({"--test-crop"}, [&parsed](const std::string& value) { parsed.test_crop = crop_edges(value); }),
        option({"--scale-to"},
               [&parsed](const std::string& value) { parsed.scale_to = choice(value, {"test", "reference", "both"}); }),
        option({"--scaler"},
               [&parsed](const std::string& value) {
                   parsed.scaler = choice(value, {"bicubic", "bilinear", "lanczos", "spline"});
               }),
        option({"--duration"}, [&parsed](const std::string& value) { parsed.duration_ms = duration_ms(value); }),
        option({"--range"}, [&parsed](const std::string& value) { parsed.range_ms = time_range_ms(value); }),
        option({"--frames"}, [&parsed](const std::string& value) { parsed.frames = frame_span(value); }),
        option({"--colour", "--color"},
               [&parsed](const std::string& value) {
                   parsed.colour = choice(value, {"pq-bt2020", "hlg-bt2020", "bt709"});
               }),
        option({"--film-grain"},
               [&parsed](const std::string& value) { parsed.film_grain = choice(value, {"with", "without", "both"}); }),
        option({"--fps"}, [&parsed](const std::string& value) { parsed.fps = frame_rate(value); }),
        option({"--size"}, [&parsed](const std::string& value) { parsed.size = picture_size(value); }),
        option({"--pix-fmt"}, [&parsed](const std::string& value) { parsed.pix_fmt = pixel_format(value); }),
        option({"--cvvdp-display"}, [&parsed](const std::string& value) { parsed.cvvdp_displays.push_back(value); }),
        flag({"--cvvdp-fill"}, parsed.cvvdp_fill),
        option({"--subsample"},
               [&parsed](const std::string& value) { parsed.subsample = int_up_to(value, 60); }),
        option({"--threads"}, [&parsed](const std::string& value) { parsed.threads = int_up_to(value, 128); }),
        option({"--vmaf-on"}, [&parsed](const std::string& value) { parsed.vmaf_on = choice(value, {"gpu", "cpu"}); }),
        option({"--vship"},
               [&parsed](const std::string& value) {
                   parsed.vship = choice(value, {"vship-fast", "vship", "vship-hip", "vship-cuda"});
               }),
        option({"--vship-pass"},
               [&parsed](const std::string& value) { parsed.vship_pass = choice(value, {"one", "each"}); }),
        flag({"--cpu"}, parsed.cpu),
        flag({"--no-gpu-decode"}, parsed.no_gpu_decode),
        option({"--gpu-decode"},
               [&parsed](const std::string& value) {
                   std::vector<std::string> words = gpu_maker_words();
                   words.insert(words.begin(), "auto");
                   parsed.gpu_decode = choice(value, words);
               }),
        option({"--vvc-decoder"},
               [&parsed](const std::string& value) { parsed.vvc_decoder = choice(value, {"gpu", "cpu"}); }),
        option({"--gpu"}, [&parsed](const std::string& value) { parsed.gpu = value; }),
        option({"--parallel"},
               [&parsed](const std::string& value) {
                   const std::optional<long long> number = python_int(value);
                   if (!number) {
                       throw ArgumentError{"invalid int value: " + python_string_repr(value)};
                   }
                   parsed.parallel = static_cast<int>(std::stoi(choice(std::to_string(*number), {"1", "2"})));
               }),
        option({"--frame-counts"},
               [&parsed](const std::string& value) { parsed.frame_counts = choice(value, {"check", "skip"}); }),
        flag({"--recalculate"}, parsed.recalculate),
        option({"--test-remuxes"},
               [&parsed](const std::string& value) { parsed.test_remuxes = choice(value, {"on", "off"}); }),
        flag({"--difference"}, parsed.difference),
        option({"--csv"}, [&parsed](const std::string& value) { parsed.csv = value; }),
        option({"--json"}, [&parsed](const std::string& value) { parsed.json_path = value; }),
        option({"--frames-json"}, [&parsed](const std::string& value) { parsed.frames_json = value; }),
        option({"--score-summary"}, [&parsed](const std::string& value) { parsed.score_summary = value; }),
        option({"--batch"}, [&parsed](const std::string& value) { parsed.batch = value; }),
        option({"--scores-csv"}, [&parsed](const std::string& value) { parsed.scores_csv = value; }),
        option({"--worst"}, [&parsed](const std::string& value) { parsed.worst = int_up_to(value, 1000); }),
        option({"--segments"}, [&parsed](const std::string& value) { read_segments(value, parsed); }),
        flag({"--segments-all"}, parsed.segments_all),
        flag({"--segments-any"}, parsed.segments_any),
        option({"--segment-min-frames"},
               [&parsed](const std::string& value) {
                   parsed.segment_min_frames = int_up_to(value, std::numeric_limits<int>::max());
               }),
        option({"--segment-merge-frames"},
               [&parsed](const std::string& value) { parsed.segment_merge_frames = int_from_zero(value); }),
        option({"--segments-dir"}, [&parsed](const std::string& value) { parsed.segments_dir = value; }),
        option({"--segments-report"}, [&parsed](const std::string& value) { parsed.segments_report = value; }),
        option({"--require"},
               [&parsed](const std::string& value) {
                   try {
                       for (GateRule& rule : read_gate_rules(value)) {
                           parsed.require.push_back(std::move(rule));
                       }
                   } catch (const std::invalid_argument& error) {
                       throw ArgumentError{error.what()};
                   }
               }),
        flag({"--fail-on-segments"}, parsed.fail_on_segments),
        option({"--stills"}, [&parsed](const std::string& value) { parsed.stills = value; }),
        option({"--stills-from"},
               [&parsed](const std::string& value) { parsed.stills_from = choice(value, {"worst", "segments"}); }),
        option({"--stills-count"}, [&parsed](const std::string& value) { parsed.stills_count = int_up_to(value, 100); }),
        option({"--run-dir"}, [&parsed](const std::string& value) { parsed.run_dir = value; }),
        option({"--log-file"}, [&parsed](const std::string& value) { parsed.log_file = value; }),
        option({"--name"}, [&parsed](const std::string& value) { parsed.name = value; }),
        flag({"--notify"}, parsed.notify),
        flag({"-q", "--quiet"}, parsed.quiet),
        flag({"-v", "--verbose"}, parsed.verbose),
        option({"--progress"}, [&parsed](const std::string& value) { parsed.progress = choice(value, {"text", "jsonl"}); }),
        option({"--timeout"}, [&parsed](const std::string& value) { parsed.timeout = duration_seconds(value); }),
        option({"--video-timeout"},
               [&parsed](const std::string& value) { parsed.video_timeout = duration_seconds(value); }),
        flag({"--cancel-on-stdin-eof"}, parsed.cancel_on_stdin_eof),
    };
    parser.exclusive = {{"--duration", "--range"},
                        {"--duration", "--frames"},
                        {"--range", "--frames"},
                        {"--no-gpu-decode", "--gpu-decode"},
                        {"--segments-all", "--segments-any"}};
    add_profile_options(parser, parsed.profile, true);
    return parser;
}

// The bitrate command's parser, keeping what it reads in `parsed`.
Parser bitrate_parser(BitrateArguments& parsed) {
    Parser parser;
    parser.actions = {
        help(std::string(bitrate_usage) + bitrate_sections + examples),
        {{}, "VIDEO", Nargs::one_or_more, true,
         [&parsed](const std::vector<std::string>& values) { parsed.videos = values; }},
        option({"--fps"}, [&parsed](const std::string& value) { parsed.fps = frame_rate(value); }),
        option({"--csv"}, [&parsed](const std::string& value) { parsed.csv = value; }),
        flag({"--no-adjust-start"}, parsed.no_adjust_start),
        flag({"--difference"}, parsed.difference),
        option({"--json"}, [&parsed](const std::string& value) { parsed.json_path = value; }),
        flag({"-q", "--quiet"}, parsed.quiet),
    };
    add_profile_options(parser, parsed.profile, true);
    return parser;
}

// The output options of saved show and report, in `parsed`'s own fields.
std::vector<Action> saved_output_options(CompareArguments& parsed) {
    return {
        option({"--json"}, [&parsed](const std::string& value) { parsed.json_path = value; }),
        option({"--csv"}, [&parsed](const std::string& value) { parsed.csv = value; }),
        option({"--frames-json"}, [&parsed](const std::string& value) { parsed.frames_json = value; }),
        option({"--score-summary"}, [&parsed](const std::string& value) { parsed.score_summary = value; }),
        option({"--run-dir"}, [&parsed](const std::string& value) { parsed.run_dir = value; }),
        option({"--worst"}, [&parsed](const std::string& value) { parsed.worst = int_up_to(value, 1000); }),
        option({"--segments"}, [&parsed](const std::string& value) { read_segments(value, parsed); }),
        flag({"--segments-all"}, parsed.segments_all),
        flag({"--segments-any"}, parsed.segments_any),
        option({"--segment-min-frames"},
               [&parsed](const std::string& value) {
                   parsed.segment_min_frames = int_up_to(value, std::numeric_limits<int>::max());
               }),
        option({"--segment-merge-frames"},
               [&parsed](const std::string& value) { parsed.segment_merge_frames = int_from_zero(value); }),
        option({"--segments-dir"}, [&parsed](const std::string& value) { parsed.segments_dir = value; }),
        option({"--segments-report"}, [&parsed](const std::string& value) { parsed.segments_report = value; }),
        option({"--stills"}, [&parsed](const std::string& value) { parsed.stills = value; }),
        option({"--stills-from"},
               [&parsed](const std::string& value) { parsed.stills_from = choice(value, {"worst", "segments"}); }),
        option({"--stills-count"}, [&parsed](const std::string& value) { parsed.stills_count = int_up_to(value, 100); }),
        flag({"-q", "--quiet"}, parsed.quiet),
    };
}

// The saved command's parser, keeping what it reads in `parsed`.
Parser saved_parser(SavedArguments& parsed) {
    Parser parser;
    parser.actions = {
        help(std::string(saved_usage) + saved_sections + examples),
        {{}, "{list,show,delete}", Nargs::one, true,
         [&parsed](const std::vector<std::string>& values) {
             parsed.action = choice(values.front(), {"list", "show", "delete"});
         }},
        {{}, "ITEM", Nargs::one_or_more, false,
         [&parsed](const std::vector<std::string>& values) { parsed.items = values; }},
        option({"--filter"}, [&parsed](const std::string& value) { parsed.filter = value; }),
    };
    for (Action& action : saved_output_options(parsed.output)) {
        parser.actions.push_back(std::move(action));
    }
    parser.exclusive = {{"--segments-all", "--segments-any"}};
    add_profile_options(parser, parsed.profile, false);
    parser.actions.push_back(option({"--cache-dir"}, [&parsed](const std::string& value) {
        parsed.profile.cache_dir = value;
    }));
    return parser;
}

// The report command's parser.
Parser report_parser(SavedArguments& parsed) {
    Parser parser;
    parser.actions = {
        help(std::string(report_usage) + report_sections + examples),
        {{}, "RUNFILE", Nargs::one_or_more, true,
         [&parsed](const std::vector<std::string>& values) { parsed.items = values; }},
    };
    for (Action& action : saved_output_options(parsed.output)) {
        parser.actions.push_back(std::move(action));
    }
    parser.exclusive = {{"--segments-all", "--segments-any"}};
    add_profile_options(parser, parsed.profile, false);
    return parser;
}

// Why the saved command's options do not fit its action, or nothing.
std::optional<std::string> saved_misuse(const SavedArguments& parsed) {
    const CompareArguments& output = parsed.output;
    if (parsed.action == "list" && !parsed.items.empty()) {
        return "list takes no ITEM (--filter narrows it)";
    }
    if (parsed.action != "list" && parsed.items.empty()) {
        return parsed.action + " needs at least one ITEM: a saved comparison's ID (saved list gives them)";
    }
    if (parsed.action != "list" && parsed.filter) {
        return "argument --filter: only with list";
    }
    const bool writes = output.csv || output.frames_json || output.score_summary || output.run_dir || output.worst ||
                        !output.segments.empty() || output.segments_saved || output.segments_report ||
                        output.segments_dir || output.stills;
    if (parsed.action != "show" && writes) {
        return "the scores' options (--csv, --segments...) are show's";
    }
    if (parsed.action == "delete" && output.json_path) {
        return "argument --json: only with list and show";
    }
    return std::nullopt;
}

// A number as Python's float() reads it, and finite.
double finite_float(const std::string& text) {
    const std::optional<double> value = python_float(text);
    if (!value) {
        throw ArgumentError{"invalid float value: " + python_string_repr(text)};
    }
    if (!std::isfinite(*value)) {
        throw ArgumentError{"must be a finite number"};
    }
    return *value;
}

// The target command's parser, keeping what it reads in `parsed`.
Parser target_parser(TargetArguments& parsed) {
    Action scores = option({"--scores"}, [&parsed](const std::string& value) { parsed.scores.push_back(value); });
    Action metric = option({"--metric"}, [&parsed](const std::string& value) {
        const std::string key = ascii_lower(python_strip(value));
        if (find_metric(key) == nullptr) {
            std::string keys;
            for (const MetricDefinition& each : metrics()) {
                keys += (keys.empty() ? "" : ", ") + std::string(each.key);
            }
            throw ArgumentError{"no metric has the key " + python_string_repr(value) + "; choose from: " + keys};
        }
        parsed.metric = key;
    });
    Action target = option({"--value"}, [&parsed](const std::string& value) { parsed.value = finite_float(value); });
    scores.required = metric.required = target.required = true;
    Parser parser;
    parser.actions = {
        help(std::string(target_usage) + target_sections + examples),
        std::move(scores),
        std::move(metric),
        std::move(target),
        option({"--by"}, [&parsed](const std::string& value) { parsed.by = ascii_lower(python_strip(value)); }),
        option({"--stat"}, [&parsed](const std::string& value) { parsed.stat = choice(value, summary_stats()); }),
        option({"--sequence"}, [&parsed](const std::string& value) { parsed.sequence = value; }),
        option({"--json"}, [&parsed](const std::string& value) { parsed.json_path = value; }),
        flag({"-q", "--quiet"}, parsed.quiet),
    };
    return parser;
}

// The diff command's parser: OLD and NEW are `parsed`'s items, in order.
Parser diff_parser(SavedArguments& parsed) {
    CompareArguments& output = parsed.output;
    Parser parser;
    parser.actions = {
        help(std::string(diff_usage) + diff_sections + examples),
        {{}, "OLD", Nargs::one, true,
         [&parsed](const std::vector<std::string>& values) { parsed.items.push_back(values.front()); }},
        {{}, "NEW", Nargs::one, true,
         [&parsed](const std::vector<std::string>& values) { parsed.items.push_back(values.front()); }},
        option({"--segments"}, [&output](const std::string& value) { read_segments(value, output); }),
        flag({"--segments-all"}, output.segments_all),
        flag({"--segments-any"}, output.segments_any),
        option({"--segment-min-frames"},
               [&output](const std::string& value) {
                   output.segment_min_frames = int_up_to(value, std::numeric_limits<int>::max());
               }),
        option({"--segment-merge-frames"},
               [&output](const std::string& value) { output.segment_merge_frames = int_from_zero(value); }),
        option({"--json"}, [&output](const std::string& value) { output.json_path = value; }),
        flag({"-q", "--quiet"}, output.quiet),
    };
    parser.exclusive = {{"--segments-all", "--segments-any"}};
    // A saved comparison's ID is looked for where saved finds it: the
    // settings' folder, or --cache-dir's.
    add_profile_options(parser, parsed.profile, false);
    parser.actions.push_back(option({"--cache-dir"}, [&parsed](const std::string& value) {
        parsed.profile.cache_dir = value;
    }));
    return parser;
}

// The bdrate command's parser, keeping what it reads in `parsed`.
Parser bdrate_parser(BdrateArguments& parsed) {
    Action anchor = option({"--anchor"}, [&parsed](const std::string& value) { parsed.anchor = value; });
    Action test = option({"--test"}, [&parsed](const std::string& value) { parsed.test = value; });
    anchor.required = test.required = true;
    Parser parser;
    parser.actions = {
        help(std::string(bdrate_usage) + bdrate_sections + examples),
        std::move(anchor),
        std::move(test),
        option({"-m", "--metrics"}, [&parsed](const std::string& value) { parsed.metrics = metric_list(value); }),
        option({"--group-by"}, [&parsed](const std::string& value) { parsed.group_by = value; }),
        option({"--json"}, [&parsed](const std::string& value) { parsed.json_path = value; }),
        flag({"-q", "--quiet"}, parsed.quiet),
    };
    return parser;
}

// Reads a command's strings with its parser; its errors are its own
// (argparse's error() of the command's parser).
std::vector<std::string> parse_command(const Parser& parser, const std::string& usage, const std::string& prog,
                                       const std::vector<std::string>& strings) {
    try {
        return parse_known(parser, strings);
    } catch (const ArgumentError& error) {
        throw UsageError(usage + prog + ": error: " + error.message);
    }
}

}  // namespace

std::string top_help() {
    return std::string(top_usage) + "\n" + version_text + ": video quality metrics from the command line.\n" +
           top_sections + examples + "\n" + program + " compare --help lists every option.\n";
}

const std::string& version_line() { return version_text; }

namespace {

// An option given that only --segments uses, when --segments is not.
std::optional<std::string> segment_option_alone(const CompareArguments& arguments) {
    if (!arguments.segments.empty() || arguments.segments_saved) {
        return std::nullopt;
    }
    if (arguments.segments_all) {
        return "--segments-all";
    }
    if (arguments.segments_any) {
        return "--segments-any";
    }
    if (arguments.segment_min_frames) {
        return "--segment-min-frames";
    }
    if (arguments.segment_merge_frames) {
        return "--segment-merge-frames";
    }
    if (arguments.segments_dir) {
        return "--segments-dir";
    }
    if (arguments.segments_report) {
        return "--segments-report";
    }
    if (arguments.fail_on_segments) {
        return "--fail-on-segments";
    }
    return std::nullopt;
}

// Why the stills' options do not fit, or nothing: one given without
// --stills, or the segments' stills without --segments.
std::optional<std::string> stills_misuse(const CompareArguments& arguments) {
    if (!arguments.stills && (arguments.stills_from || arguments.stills_count)) {
        return std::string("argument ") + (arguments.stills_from ? "--stills-from" : "--stills-count") +
               ": only with --stills, whose stills it picks";
    }
    if (arguments.stills_from == "segments" && arguments.segments.empty() && !arguments.segments_saved) {
        return "argument --stills-from: segments only with --segments, whose segments it takes";
    }
    return std::nullopt;
}

// What is wrong with the videos given: REFERENCE and TEST, or --batch's list
// in their place (as argparse words a missing argument, or a conflict).
std::optional<std::string> batch_misuse(const CompareArguments& arguments) {
    if (!arguments.batch) {
        const std::string missing = !arguments.reference       ? "REFERENCE, TEST"
                                    : arguments.tests.empty() ? "TEST"
                                                              : "";
        return missing.empty() ? std::nullopt
                               : std::optional("the following arguments are required: " + missing);
    }
    if (arguments.reference) {
        return "argument --batch: not allowed with REFERENCE and TEST: its list gives the videos";
    }
    if (arguments.difference) {
        return "argument --difference: not allowed with argument --batch";
    }
    return std::nullopt;
}

}  // namespace

Arguments parse_arguments(const std::vector<std::string>& arguments) {
    Arguments parsed;
    std::vector<std::string> command_extras;
    Parser top;
    top.actions = {
        help(top_help()),
        {{"--version"}, "", Nargs::none, false,
         [](const std::vector<std::string>&) { throw Printed{version_text + "\n"}; }},
        {{}, "COMMAND", Nargs::command, false,
         [&](const std::vector<std::string>& values) {
             std::string command;
             try {
                 command = choice(values.front(),
                                  {"compare", "devices", "bitrate", "bdrate", "saved", "report", "target", "diff"});
             } catch (const ArgumentError& error) {
                 throw ArgumentError{"argument COMMAND: " + error.message};
             }
             const std::vector<std::string> rest(values.begin() + 1, values.end());
             if (command == "compare") {
                 parsed.command = Command::compare;
                 parsed.compare.command_line = arguments;
                 const Parser parser = compare_parser(parsed.compare);
                 const std::string prog = std::string(program) + " compare";
                 command_extras = parse_command(parser, compare_usage, prog, rest);
                 if (const std::optional<std::string> misuse = batch_misuse(parsed.compare)) {
                     throw UsageError(std::string(compare_usage) + prog + ": error: " + *misuse);
                 }
                 if (const std::optional<std::string> option = segment_option_alone(parsed.compare)) {
                     throw UsageError(std::string(compare_usage) + prog + ": error: argument " + *option +
                                      (*option == "--fail-on-segments"
                                           ? ": only with --segments, whose conditions find the weak segments"
                                           : ": only with --segments, whose segments it shapes"));
                 }
                 if (const std::optional<std::string> misuse = stills_misuse(parsed.compare)) {
                     throw UsageError(std::string(compare_usage) + prog + ": error: " + *misuse);
                 }
                 if (parsed.compare.name) {
                     if (const std::optional<std::string> wrong =
                             name_template_error(*parsed.compare.name, parsed.compare.batch.has_value())) {
                         throw UsageError(std::string(compare_usage) + prog + ": error: argument --name: " + *wrong);
                     }
                 }
             } else if (command == "bitrate") {
                 parsed.command = Command::bitrate;
                 const Parser parser = bitrate_parser(parsed.bitrate);
                 const std::string prog = std::string(program) + " bitrate";
                 command_extras = parse_command(parser, bitrate_usage, prog, rest);
                 if (parsed.bitrate.no_adjust_start && !parsed.bitrate.csv) {
                     throw UsageError(std::string(bitrate_usage) + prog +
                                      ": error: argument --no-adjust-start: only with --csv, whose times it sets");
                 }
             } else if (command == "saved" || command == "report") {
                 const bool saved = command == "saved";
                 parsed.command = saved ? Command::saved : Command::report;
                 parsed.saved.action = saved ? "" : "report";
                 const Parser parser = saved ? saved_parser(parsed.saved) : report_parser(parsed.saved);
                 const char* usage = saved ? saved_usage : report_usage;
                 const std::string prog = std::string(program) + " " + command;
                 command_extras = parse_command(parser, usage, prog, rest);
                 if (const std::optional<std::string> misuse = saved ? saved_misuse(parsed.saved) : std::nullopt) {
                     throw UsageError(std::string(usage) + prog + ": error: " + *misuse);
                 }
                 if (const std::optional<std::string> option = segment_option_alone(parsed.saved.output)) {
                     throw UsageError(std::string(usage) + prog + ": error: argument " + *option +
                                      ": only with --segments, whose segments it shapes");
                 }
                 if (const std::optional<std::string> misuse = stills_misuse(parsed.saved.output)) {
                     throw UsageError(std::string(usage) + prog + ": error: " + *misuse);
                 }
             } else if (command == "target") {
                 parsed.command = Command::target;
                 const Parser parser = target_parser(parsed.target);
                 command_extras = parse_command(parser, target_usage, std::string(program) + " target", rest);
             } else if (command == "diff") {
                 parsed.command = Command::diff;
                 parsed.saved.action = "diff";
                 const Parser parser = diff_parser(parsed.saved);
                 const std::string prog = std::string(program) + " diff";
                 command_extras = parse_command(parser, diff_usage, prog, rest);
                 if (const std::optional<std::string> option = segment_option_alone(parsed.saved.output)) {
                     throw UsageError(std::string(diff_usage) + prog + ": error: argument " + *option +
                                      ": only with --segments, whose segments it shapes");
                 }
             } else if (command == "bdrate") {
                 parsed.command = Command::bdrate;
                 const Parser parser = bdrate_parser(parsed.bdrate);
                 command_extras = parse_command(parser, bdrate_usage, std::string(program) + " bdrate", rest);
             } else {
                 parsed.command = Command::devices;
                 Parser parser{
                     {help(std::string(devices_usage) + devices_sections),
                      option({"--json"}, [&parsed](const std::string& value) { parsed.devices.json_path = value; })},
                     {}};
                 add_profile_options(parser, parsed.devices.profile, false);
                 command_extras = parse_command(parser, devices_usage, std::string(program) + " devices", rest);
             }
         }},
    };
    try {
        std::vector<std::string> extras = parse_known(top, arguments);
        append(extras, command_extras);
        if (!extras.empty()) {
            std::string listed;
            for (const std::string& extra : extras) {
                listed += (listed.empty() ? "" : " ") + extra;
            }
            throw ArgumentError{"unrecognized arguments: " + listed};
        }
    } catch (const ArgumentError& error) {
        throw UsageError(std::string(top_usage) + program + ": error: " + error.message);
    } catch (const Printed& printed) {
        parsed.print_and_exit = printed.text;
    }
    return parsed;
}

}  // namespace vml::cli
