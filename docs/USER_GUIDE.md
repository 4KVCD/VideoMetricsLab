# User guide

## Calculate metrics

1. In **Videos**, select a reference and add encoded/distorted files: drop
   them on the test list (a file dropped on the reference box replaces the
   reference), paste a reference path, or use Browse. Drag rows to reorder
   them; click a column heading to sort by it, and again to reverse.
2. Select the metric checkboxes in each row. Cell edits apply to selected rows;
   header checkboxes apply across rows. While there are test videos, one is
   always selected, so the Options panel always shows settings: the first one
   added is selected, a click where there is no row changes nothing, and
   removing the selected rows selects the next. VMAF, PSNR, SSIM and XPSNR are enabled
   by default; VMAF NEG, SSIMULACRA2 and Butteraugli can be selected separately.
3. Review crop, scale, model, range and subsampling settings before starting.
   **Range** compares only a part of the videos, from one time to another, to
   the millisecond (hh:mm:ss.zzz; an end of 00:00:00.000 compares to the
   videos' end). It compares the frames whose times are from the first time up
   to, not including, the second (frame *n* is at *n* / frame rate), each time
   to the nearest millisecond as Metric Graphs shows it, so a time copied from
   the graph is that frame's. CVVDP
   scores whole seconds: a range that isn't whole seconds is rounded to the
   nearest seconds for it. Switching between **Time** and **Frames** keeps
   the range: it is shown as the same frames in the other unit (at the
   reference's frame rate), and switched back with no change, as it was
   typed. A range's
   scores are those of the same frames in a run over the whole videos. An end
   within a video's last, partial second compares to its end: 00:00:00 to
   00:00:13 for a 300-frame encode of 12.5 seconds. A range late in a video
   starts decoding near it where the file proves its frames can be counted from
   their timestamps. Otherwise the frames before it are decoded, so it takes a
   while to start, and its progress may stay at 0% until then.

The model selector includes Netflix VMAF v0 models and the bundled VMAF v1
models (1080p, phone, 4K, and high-frame-rate variants). VMAF v1 requires an
FFmpeg build linked against a libvmaf version that supports the v1 feature set;
older FFmpeg packages will report a clear model/feature error and should use a
v0 model or be upgraded. For SDR, VMAF v1 is best measured at 10-bit precision.

**HDR VMAF** scores PQ (HDR10) videos with Netflix's HDR VMAF models (1080p,
phone, 4K, and high-frame-rate variants), chosen in **HDR VMAF model**. It is
shown but unticked at first, as an SDR video fails it: tick its column to
score it (its tick only matters for HDR10 videos). Both videos are converted to the
model's ICtCp form and its features calculated on the GPU when VMAF runs on the
GPU (else on the CPU), with the same scores. The GPU's conversion is checked
against the CPU's on the first frame; where the GPU cannot convert the
pictures exactly as the CPU does (8-bit videos, a Mac), they are converted on
the CPU and only the features go to the GPU. It uses the same
decode of each video as VMAF on the GPU and PSNR, SSIM and XPSNR in the app
(where they cannot share a decode, it decodes the videos once more). A video that is not PQ is refused, with
the reason in its log; its other metrics are still scored. It is not
calculated in resolution tests.
4. Choose **Calculate metrics**. Optional two-job parallelism can help on large
   CPUs, but increases resource use and is not always faster.
5. Inspect per-frame curves in **Metric Graphs**. Click a metric's mean column
   to show its detailed statistics. The save icon at the statistics card's
   top right saves the table as shown (the frame range's when one is
   highlighted) at one click, and the one at the right end of the metrics'
   row the graph as PNG (see [Saving and exporting](#saving-and-exporting)).
   Right-click either for **Statistics as JSON...** or **as CSV...**, or
   **Graph as PNG...**, with the Save dialog; right-click the statistics for
   a row's video's export too.
   With two videos shown, **Difference** draws the first minus the second,
   frame by frame, in a band under the graph with its own scale. With it on,
   the graph's save icon saves it with the PNG, and the right-click menus
   offer **Difference as CSV...**:
   every metric's difference, as `<first>_minus_<second>.csv` (the
   videos' labels), in the per-frame scores' columns: a cell is blank where
   either video lacks the frame or the metric.
   To look at one part of the videos (a scene you re-encoded, say), enter its
   first and last frames in **Highlight range from frame** and **to frame**,
   its first and last times in **Highlight range from time** and **to time**
   (h:mm:ss.sss, or seconds: the frames from the first time to the second,
   both included; the frame boxes then show them, and the time boxes the
   frames' own times), or hold Shift and drag across the graph: the frames
   between them are shaded on
   every metric's graph, and the table and the exported PNG give the
   statistics of those frames only. CVVDP has a score for whole videos only, so its column shows
   — then; its graph's statistics are of the whole seconds the frames fall in.
   **Clear range** goes back to the whole videos. The CSV export always holds
   every frame. **Go to time** moves to the frame shown at a time, as **Go to
   frame** does to a frame.
6. Find a video's weak spots in **Worst Frames**. On the left, every video
   with its number of weak segments and its worst score on the metric shown,
   so the weakest encode stands out; click one to work on it. Under the
   videos, the conditions that make a frame weak (thresholds you set, any or
   every condition, the shortest segment, the gap to join across), with each
   condition's line under them: how many frames it holds on and how many of
   them are in the segments, with why any are not (a run shorter than the
   shortest segment, or, with every condition, a frame where another doesn't
   hold). On the right, pick the metric in the bar at the top. The graph
   shows the video's scores with its weak segments shaded and its worst
   frames dotted. Below it, side by side where the window has room and one
   above the other where it hasn't:
   - **Worst frames**: the worst frame of each of the metric's 10, 25 or 50
     worst dips (a frame within 1 s of a worse one is part of its dip), with
     the conditions' metrics' scores there; every metric's is in the row's
     tooltip.
   - **Weak segments**: the stretches where the conditions hold; with more
     than one condition and **any condition holds**, the **Weak by** column
     names those behind each segment.

   Click a row to mark it on the graph; double-click it to see it in Video
   Compare (a segment from its first frame). Point at the
   graph and **Details**, under the conditions, tells the frame there: its
   time, its place among the worst frames, the weak segment it is in and the
   conditions that hold on it, and every metric's score (a worst frame's dot
   a few pixels away is taken, so a dip's worst frame is easy to land on).
   With the pointer elsewhere, it tells the row selected. Only frames
   in Metric Graphs' highlighted range count, if there is one.
   The conditions start from the weak segment profile in use (see [Weak
   segment profiles](#weak-segment-profiles)): its conditions on the metrics
   the video has (else one on the metric shown). **Profile** picks another
   to start from, and **From preset** loads one of the built-in presets for
   a kind of video (animation, grainy film, HDR...; see the table under
   [Weak segment profiles](#weak-segment-profiles)) in place of the
   conditions shown, so you can try another on the fly; a preset with no
   condition on the video's metrics (HDR without CVVDP scores) is greyed
   out. What you change or load here holds until **Save to profile** keeps
   it in the profile picked, or **Save as new profile** in a new one; a
   profile's condition on a metric the video doesn't have is kept for videos
   that do (a preset's, once you've loaded one).
   Each card's save icon saves what it shows at one click (the metric, the
   conditions, the dips and the range); a right-click on the icon, or on its
   list or graph, offers each with the Save dialog:
   - Worst frames: **Worst frames as CSV...** or **as JSON...**, each frame
     with its rank, time and every metric's score
     (`<video>.vmaf.worst.csv`; the format is in
     [RESULTS_JSON.md](RESULTS_JSON.md#worst-frames-as-json-and-csv)).
   - The graph: **Graph as PNG...**, in Settings' Exported graphs colours.
   - Weak segments: **Weak segments as JSON...** or **as CSV...**, named
     after the metrics they cover (`<video>.vmaf.segments.json`,
     `<video>.vmaf+ssimulacra2.segments.csv`), or every scored test video's
     in one file (**All videos' weak segments as JSON or CSV...**; see
     [Exporting weak segments](#exporting-weak-segments)).

**Highlight best/worst results**, under the video table, colours each metric's
best score green and its worst red among the test videos that have one; a
selected row keeps them, with the selection drawn along their edges.

**Also score CVVDP on...**, under the Options panel's display presets, scores
CVVDP on further displays in the same run: pick a display, and each selected
video gets a row of its own that calculates CVVDP alone on that display (the
other metrics are not calculated again). Repeat it for more displays.
**Export results** lists every display's CVVDP score in the video's entry:
`cvvdp_displays` in the JSON, columns `cvvdp_displays.0.score`,
`cvvdp_displays.1.score`... in the CSV.

**Settings** has five sub-tabs: **General** (the ffmpeg folder, the window,
and how many test videos Video Compare decodes), **Metrics** (the defaults for
newly added videos, and the GPU metrics), **Weak segments**, **Storage** and
**Notifications** (see [Notifications](#notifications)).

**Settings > General > Window** chooses whether the app reopens the videos it
had open when it was last closed (off by default).

**Video names** there sets how test videos' names are shown in Metric Graphs,
Video Compare and the Worst Frames tab's videos list, where long names that differ only near their ends (an
encode's CRF, its encoder) would otherwise be cut where they differ:

- **What tells them apart** (the default) leaves out the start and end that
  the videos shown share, at a word's edge, keeping the word before the first
  difference: "…CRF 20", "…CRF 22". One video, or videos with nothing shared
  worth leaving out, show their whole names.
- **First and last characters** shows each name's first and last so many
  characters, with "…" between.
- **Whole names** shows them as they are.

Video Compare's **Test video** list always shows whole names, and the title
over the video shows the whole name where it fits, the shortened one only
where it doesn't. Tooltips, exports and file names always keep the whole name.

**Settings > Appearance** chooses how the window looks; nothing else changes
with it. **Interface** is a card each, with a picture of it, VML Modern first:

- **VML Modern** (the default): boxes as cards, each tab's main action
  (**Calculate metrics**, **Open in Metric Graphs**, **Play**, **Calculate
  bitrate**) in the accent colour, actions that delete in red, and ticked
  boxes and the selection in the accent.
- **VML Classic**: the window as it was before VML Modern, in the system's
  own controls.

VML Modern shows at once; VML Classic, chosen while VML Modern is shown, when
the app is next started. **Accent**, usable with VML Modern, is the system's
(**Same as Windows**), one of six colours, a colour picked with **Custom...**,
or one typed as `#RRGGBB` in the box beside it; text on it is dark or light,
whichever reads. VML Classic keeps the system's accent. **Theme** is the
system's (**Same as Windows**), **Light** or **Dark**.

VML Modern draws Light and Dark each in a look of its own, picked under
**Light look** and **Dark look**; with **Same as Windows**, the window
changes look as Windows changes between light and dark:

- **Light**: VML Modern's own light colours.
- **Dim**: Light, a step darker throughout (the cards, the window behind them,
  the buttons, the edges and the graphs' background), for less glare; the
  text as Light's.
- **Paper**: Dim, warmed a little, like unbleached paper.
- **Desert**: warm cream with dark text and edges, after Windows' Desert
  contrast theme, with its burnt-orange accent.
- **Dark**: VML Modern's own dark colours.
- **Mocha**: dark chocolate browns with cream text and an amber accent (after
  the Solar Dusk palette).
- **Joker**: plum purples with lavender-white text, a vivid green accent and
  purple links (after the Dracula palette).
- **Nitrate**: sepia film noir: green-black shadows with cream text and a gold
  accent, like an old film print.
- **Phosphor**: sepia film noir at the most contrast: near-black with umber
  edges, bright cream text and a gold accent.

While a look with an accent of its own (Desert, Mocha, Joker, Nitrate,
Phosphor) is the one the theme
shows, the accent's first choice reads **Theme default** and gives that
accent. The looks are VML Modern's alone: VML Classic shows Light or Dark.

In VML Modern on Windows, the window's title bar is the app's own, in the
theme's colours, with Windows' minimize, maximize and close (hovering
maximize shows Windows 11's snap layouts, and Alt+Space or a click on the
icon opens the window menu). Three small buttons sit in it:

- **Progress**, always there: **Ready** before a run, then how many videos are
  done and about how far the run has got, with a thin bar (**Paused** while
  paused), then how it ended: **Done**, **Done · 1 failed** in the error color,
  or **Cancelled**. That stays until the next run, or until the ticked videos
  or their metrics change, when it is **Ready** again. Its tooltip says more:
  the ticked videos before a run; during it the status line, then each video
  in progress with its lines from the Videos tab; after it, how the run ended
  and each failed video's reason. A click opens the Videos tab.
- **Theme**: a menu of the themes (Same as Windows, Light, Dark) and, below
  them, **Light look** and **Dark look** menus of the looks, as in Settings >
  Appearance.
- **Update to ...**, when a newer version is available: a click shows what's
  new and the download. VML Classic asks with a dialog when the app starts
  instead.

On macOS and Linux, and on Windows in high-contrast mode or when VML Modern
was chosen since the app started, the title bar stays the system's and the
three buttons sit at the right end of the tab bar. VML Classic has the
**Progress** button alone, at the right end of the tab bar.

While metrics are calculated, Settings stays open, but the ffmpeg folder, the
saved results' folder and **Clear saved results** are locked until the run
ends, as a note at the top of the tab says, above the sub-tabs: the run uses
them. Everything else can be changed; the GPU metrics' choices apply from the
next run.

**Settings > Storage > Exported graphs** chooses the colours of the PNGs that
Metric Graphs and the Bitrate Viewer export: Light (the default), Dark, or the
same as the window, and, if you like, a **Background** and a **Text** colour of
your own in place of the theme's (**Reset color** in their dialog goes back to
it). The lines keep their own colours.

SSIMULACRA2, Butteraugli and CVVDP are calculated on the GPU with the app's own
Vship ([Vship-fast](https://github.com/4KVCD/Vship-fast)), a faster Vship whose
metrics compute as libjxl's and the reference CVVDP's do,
with Vulkan on any GPU (NVIDIA's, AMD's and Intel's). If no GPU can be used,
the input format is unsupported, or GPU scoring fails, SSIMULACRA2 and
Butteraugli are calculated on the CPU, with the scores of libjxl's official
tools; CVVDP runs on
the GPU only. The GPU and the CPU convert the video to RGB differently, so
their scores differ: on a PC whose GPU can calculate them, a score calculated
on the CPU is marked "(CPU)" and the next run calculates it again on the GPU.
These metrics are opt-in and do not add processing to runs where they are
unchecked.

Each video's Performance > Vship compute chooses the Vship: GPU, the app's own,
the fastest and most accurate; or stock Vship (upstream's release): Vship Vulkan,
or for a GPU whose Vulkan driver has problems Vship HIP (AMD, Windows) or Vship
CUDA (NVIDIA, Windows). Settings > Metrics > GPU metrics sets the one newly
added videos start with, and the command line's default. Stock Vship's three share their saved
scores. GPU's differ slightly and are saved apart: switching a video between
GPU and stock Vship shows the other's saved scores, or calculates them on the
next run. HIP needs an AMD GPU and CUDA an NVIDIA one: on any other GPU, or on
a run where they fail, the metrics go to the CPU as below.
Where the app's own Vship fails on the GPU, stock Vship calculates its metrics
instead, and such a score is marked "[Vship]" and calculated again with the
app's own on the next run ("[GPU]" marks the reverse). On a PC whose GPU the
app's own Vship cannot use at all but stock Vship can, a GPU video's scores
are stock Vship's, kept and reused as its own. A video set to stock Vship
never uses the app's own Vship: where stock Vship fails, SSIMULACRA2 and
Butteraugli are calculated on the CPU, and CVVDP is not.

On a PC with more than one GPU (Windows), Settings > Metrics > GPU metrics >
GPU chooses the one that calculates VMAF, VMAF v1, SSIMULACRA2, Butteraugli and CVVDP; with
GPU decoding automatic, its maker's decoder is tried first. AMD's and NVIDIA's
decoders decode on the chosen GPU when it is theirs. Automatic, the
default, uses a discrete GPU first. Two GPUs of one name are told apart by
their place, as Windows lists them (`#1`, `#2`). A metric the chosen GPU cannot calculate
is calculated on the CPU, not on another GPU, so a GPU kept free (one that is
encoding, say) stays free. The choice takes effect when the app is next
started; the command line's `--gpu` chooses for one run.

Settings > Metrics > GPU metrics > VVC decoder chooses how a VVC video is
decoded for every metric (Windows). CPU, the default, decodes it with VVdeC; GPU, which is
experimental, decodes it on the GPU, which frees the CPU but can be slower than
a fast CPU. The
scores are the same, film grain too: the GPU adds H.274's grain as VVdeC
does. The CPU decodes it anyway for a scaled video or one with AFGS1 grain
added, with CUDA VMAF, with GPU decoding off, and for black-bar
detection and Video Compare; where the GPU's decoder cannot start or fails,
the rest of the run decodes VVC on the CPU.

VMAF v0.6.1 and VMAF NEG are calculated on the GPU too, with a Vulkan version
of libvmaf's calculation on any GPU (NVIDIA's, AMD's and Intel's). An NVIDIA
GPU (GeForce GTX 16 and RTX 20 series or newer) whose Vulkan cannot calculate
them uses a bundled libvmaf built with CUDA instead. The two give the same
scores to the last decimal. The choice is each
video's Performance > VMAF compute, GPU by default and greyed out
at CPU without a GPU. A small integrated GPU can be slower at it than a fast
CPU: choose CPU there. The last choice is what newly added videos start with.
The GPU's scores agree with FFmpeg's libvmaf on the CPU to within a thousandth
of a point on every frame, so a saved VMAF score is reused whichever is
chosen. PSNR, SSIM and XPSNR have no GPU code: the app
calculates them on the CPU with the bundled libvmaf-fast, from the pictures
decoded for VMAF where it can, so the videos are not decoded twice. With VMAF
on the CPU, FFmpeg calculates them in the same run, except XPSNR of videos of more than 8 bits: FFmpeg's `xpsnr`
filter reads such pictures wrongly at some widths, so the app calculates it,
beside FFmpeg's run.

VMAF v1 follows the same choice, with any GPU that has a Vulkan driver. Its
four features, detail loss (ADM), motion, banding (CAMBI) and colour (SpEED),
are calculated on the GPU with Vulkan -- on an NVIDIA GPU too -- and give libvmaf's CPU values to the last bit, and libvmaf's
own model turns them into the score, so VMAF v1 with the GPU is the score
FFmpeg's libvmaf gives. It is about 17 times the CPU's speed on 4K video with
an RTX 5090 that also decodes both videos, 2.5 times with a VVC test video
(decoded by the CPU, which then sets the pace), and 1.4 to 1.7 times on an
Intel integrated GPU. A video too small for SpEED
(under about 160 pixels a side, more with the models that scale it down) is
calculated on the CPU. A video's VMAF on the GPU waits for the
GPU like its other GPU metrics, one video at a time, and goes before them. If the GPU calculation fails, or crashes, that
video's VMAF is calculated on the CPU instead, as it is when the GPU's driver
fails the self-test the app runs on it when it starts. Without a GPU, FFmpeg's
libvmaf calculates everything.

The files must describe corresponding frames. The app rejects several timing
and geometry mismatches; this is not automatic content alignment. A source with
a different cut, opening sequence or frame offset must be aligned first.

Frames are compared in order: frame 1,000 of the test video with frame 1,000
of the reference. If the test video has a few more or fewer frames than the
reference, even though both are the same length, the two drift out of sync
from that point: each frame is compared with the wrong reference frame, and
every score after that point is wrong, even though the videos play in sync.
A common cause is a Blu-ray remux with small gaps in its timing where clips
were joined: an encoder set to a constant frame rate fills each gap with a
duplicated frame. Encoding with a variable frame rate that matches the source
avoids it.

When fewer frames were compared than videos of that length should have, the
test video may be out of sync. It isn't always: two videos with the same gaps
have the same frames. So the app then checks the frame counts the two files
store. MP4 files store one, and so do MKV files made by mkvmerge; files made
by FFmpeg or HandBrake don't. If the counts differ, a **⚠** appears before
the test video's name, and its tooltip gives both counts. If they match, the
video is in sync and nothing is shown.

If a file doesn't store its frame count, the **⚠** says the video *may* be
out of sync, and its tooltip says how to find out: turn on **Settings >
Storage > Read entire files to check frame counts when needed**, then
recalculate the video (right-click it: **Recalculate selected metrics**).
With that setting on, such a file is read from start to finish to count its
frames. This happens only for a video that looks out of sync, after it has
been scored, but for a film on a network drive it can add several minutes per
video. It's off by default; on the command line, `--frame-counts check` does
the same for one run.

Every video in a results JSON says whether its frames are in sync with the
reference's (`frame_sync`, with a plain-language note: see
[RESULTS_JSON.md](RESULTS_JSON.md)).

### Notifications

**Settings > Notifications** sends a message when a run ends, to the
services you choose, so a run of hours needn't be watched. The list on the
left names them; a tick sends to that one, and choosing it shows its
settings and where its keys come from.

- **When a run ends:** finished, finished with failures (a video or some of
  its metrics failed) and cancelled, each on or off, and only for runs of at
  least so many minutes (a quick check then sends nothing).
- **Pushover:** your user key, and the token of an application you create on
  pushover.net for this app; the messages' priority.
- **Discord:** a channel's webhook URL (the channel's settings, Integrations >
  Webhooks). The message's colour says how the run ended, and when the
  message lists the test videos, each is a field of its own (its name, then
  its line), up to Discord's 25. **Mention** names whom it notifies:
  `<@&role ID>`, `<@user ID>` (Discord's Developer Mode shows IDs: right-click,
  Copy ID), `@here` or `@everyone`, either when a run has failures or is
  cancelled or for every run; a name in the message itself mentions no one.
  **Send without a push or a sound** keeps the messages that mention no one
  quiet.
- **Gotify:** your server's address and an application's token (its Apps
  tab), with a priority from 0 to 10.
- **ntfy:** the server (`https://ntfy.sh`, or your own), the topic, and an
  access token where the topic needs one; a priority from lowest to highest.
  On a public server anyone who knows a topic's name can read it, so the
  topic is kept like a key: choose one that can't be guessed.
- **Pushbullet:** an access token (pushbullet.com, Settings > Account), and a
  device's iden to send to that one alone (empty: every device).
- **Slack:** an incoming webhook's URL (`https://hooks.slack.com/...`, from a
  Slack app's Incoming Webhooks). The message is sent as plain text, so
  nothing in it is read as Slack's markup.
- **Telegram:** a bot's token (from @BotFather) and the chat's ID (send the
  bot a message, then open `api.telegram.org/bot<token>/getUpdates`), or a
  public channel's `@name`; it can send without a sound.
- **Webhook:** any address that takes a POST (Home Assistant, n8n, a script),
  and a header to send with it (`Authorization: Bearer ...`), if it needs one.
  It gets the run as JSON:

  ```json
  {"event": "run_ended", "app": "VideoMetricsLab 2.7",
   "outcome": "finished_with_failures", "outcome_text": "Finished with failures",
   "title": "...", "message": "...", "finished_at": "2026-11-21T12:00:00Z",
   "duration_seconds": 3725, "reference": "reference.mkv",
   "videos": [{"name": "encode.mkv", "result": "Scored", "scored": true,
               "frames": 1440,
               "metrics": {"vmaf": {"mean": 95.21, "min": 80.1, "max": 99.0,
                                    "median": 95.2, "worst_10": 90.1,
                                    "worst_5": 88.9, "worst_1": 88.2,
                                    "worst_0_1": 84.0}}}]}
  ```

  `outcome` is `finished`, `finished_with_failures` or `cancelled`; `title`
  and `message` are the message as the templates below make it; a statistic
  a metric has no value for is `null`.
- Gotify, ntfy and the webhook take an `http://` address too, for a server
  on your own network: there the keys and the message are sent unencrypted.
  Every other service, and these with an `https://` address, use HTTPS.
- **Send test** sends the message shown in the preview, there and then, and
  says what the service answered.
- **Message:** a title, a message, and a line for each test video, each made
  of text and fields in braces, filled in when it is sent: `{outcome}`,
  `{duration}`, `{reference}`, `{finished}`, `{app}`, `{video_count}`,
  `{scored_count}`, `{failed_count}` and `{videos}` (each video's line) in the
  title and message; in a video's line `{name}`, `{result}`, `{frames}`,
  `{scores}` (each metric's mean), `{scores.min}` (each metric's minimum; also
  `max`, `median`, `worst_10`, `worst_5`, `worst_1`, `worst_0_1`), and a
  metric's own: `{vmaf}`, `{vmaf.min}`, `{ssimulacra2.worst_1}`. **Insert
  field** lists them all, and the preview shows the message as it will look.
  **Start from** sets all three to a preset: **Status only** (how the run
  ended and how long it took: no file name or score leaves the PC),
  **Standard** (the default: also the reference, and each test video's name
  and mean scores) or **Detailed** (also minimums and the worst 1 %).
- **Keys:** the services' keys, tokens and webhook URLs (and ntfy's topic)
  let anyone who has them send messages as you. **Encrypted for this Windows user**, the default, stores
  them so that only you, on this PC, can read them back (Windows' DPAPI): a
  backup or a copy of your profile doesn't carry them, and on another PC
  they're entered again. **As plain text in settings.json** stores them as
  they are, which anyone who can read the file can use. (On macOS and Linux
  they are stored as plain text for now.)

A message that can't be sent stops nothing: the status bar and the log say
why, with the service's answer. A service that asks to wait a moment (Discord
when sent to too often) gets the message again once, after that moment. The
keys are never written to the log.

The command line sends the same message with `--notify` (see [Command
line](#command-line)), as these settings say: to the services on, with the
message built here, for the outcomes and run lengths ticked. On macOS and
Linux it sends it with the system's `curl`.

### AV2 videos

AV2, AOM's codec after AV1 (not yet final), is decoded by splot, an AV2
decoder, on the CPU: FFmpeg knows no AV2 and no GPU decodes it, so the
app reads AV2 files itself. It reads them as AOM's AV2 encoder (AVM's
`avmenc`) writes them: WebM (its default), IVF (`--ivf`) and raw OBU streams
(`--obu`, `.obu`; these store no frame rate that can be read, so the app asks for it and the command line needs `--fps`).
Their film grain can be scored with it or without it, as AV1's. splot's
pictures are AVM's own decoder's, checked at 8 and 10 bits, with and without
film grain. A WebM or IVF file's frame rate is found from its timestamps.

## Interpreting scores

VMAF, VMAF NEG, PSNR, SSIM, XPSNR, SSIMULACRA2, Butteraugli and the metrics
below are different measurements, not interchangeable quality percentages.
Higher values generally indicate closer agreement for the same comparison
recipe, except Butteraugli, CAMBI and CIEDE2000, where lower is better. Change the crop, scale or model and you change the
question being measured. Use visual inspection alongside scores.

The graph's non-VMAF threshold bands are heuristic defaults from synthetic
calibration, not universal equivalents of VMAF thresholds. SSIM is displayed
with more decimal places than the dB metrics. XPSNR's sequence aggregate uses
its distortion-based convention, not an arithmetic average of dB values.
Identical XPSNR frames score infinity and contribute zero distortion while
remaining in the aggregate's frame count. An em dash indicates unavailable data.

**Add/remove metrics**, above the Videos table (or a right-click on the
metric columns' headings), lists the 17 metrics in two sections, each with a
few words on what it measures, and on hover what its scores mean: its unit,
which way is better, and its broad ranges where they have a source (the
app's own bands, or the metric's authors' scale). *General purpose* (VMAF
v0.6.1, VMAF NEG, VMAF v1, HDR VMAF, PSNR, SSIM, XPSNR, SSIMULACRA2,
Butteraugli and CVVDP) is what a new install shows, HDR VMAF unticked.
*Specialized* (PSNR YUV, PSNR YUV avg. MSE, SSIM YUV, MS-SSIM, MS-SSIM YUV,
PSNR-HVS-M, CAMBI, CIEDE2000 and VMAF YUV) is folded away
until one of them is shown. A tick there shows or hides a metric's column,
nothing more; the tick in a column says whether that row calculates it, and a
run calculates what is ticked and shown. The Specialized metrics start
ticked but hidden, so showing one is enough to calculate it (settings already
saved keep the ticks they have). Settings > Metrics' **Default
metrics**, what new videos start with ticked, lists the same 19 in the same
two sections, side by side.

PSNR, SSIM and XPSNR are of the luma (Y) plane alone: PSNR is what an
encoder reports as PSNR-Y, not a weighted average of Y, U and V, and its
score is the mean of the frames' PSNR. XPSNR takes the reference as the
original, which FFmpeg's `xpsnr` filter wants as its first input whatever
its inputs are called, so it differs from FFmpeg commands that give the test
video first (`[dist][ref]xpsnr`).

PSNR YUV (6:1:1) and SSIM YUV (6:1:1), among the Specialized metrics, weigh the planes as
JVET's common test conditions weigh PSNR (their YUV BD-rates) and MSU's
codec comparisons weigh YUV-SSIM, for comparing with those: each frame's (6 x Y + Cb + Cr) / 8 of its three planes' PSNR or
SSIM, and the score is the mean of the frames'. VVenC's and VTM's own
YUV-PSNR line is a different average (one PSNR of the planes' errors
weighted by their sample counts), so it does not match these. Only the app calculates them, with
its libvmaf-fast, as FFmpeg's metrics have no such average; with VMAF on
the CPU, it calculates them beside FFmpeg's run. Each plane's
PSNR stops at libvmaf's ceiling of 6 x bits + 12 dB (60 dB at 8 bits, 72 at
10), as the PSNR column's does, where FFmpeg's `psnr` filter gives identical
planes infinity. Videos in other chroma formats are compared as 4:2:0, as
for every metric here, so a 4:4:4 video's chroma scores are those of its
4:2:0 form. Turning either on makes the decoders hand over whole pictures
rather than their luma alone, which costs some speed and memory.

PSNR YUV avg. MSE (6:1:1) is the YUV-PSNR MSU's codec comparisons report
(their "YUV-PSNR avg. MSE 6:1:1"): each frame's PSNR of its planes' squared
errors weighted (6 x Y + Cb + Cr) / 8, and the score the PSNR of the frames'
mean error, as MSU's VQMT averages a video's PSNR. PSNR YUV averages the dB
instead, of the planes and of the frames, so this one is a little lower,
more so where a few frames or one plane are much worse. It is calculated from
the same planes' PSNR as PSNR YUV, in the same pass: beside PSNR YUV it costs
nothing more, alone it costs what PSNR YUV does. A plane at libvmaf's ceiling
counts as the ceiling (MSU gives an identical frame 100 dB), so near-lossless
frames score a little lower than MSU's.

VMAF YUV (6:1:1) is the YUV-VMAF MSU's codec comparisons report: the VMAF
column's model applied to each of the three planes as if it were a luma
plane, and each frame's (6 x Y + U + V) / 8 of the three scores. Its Y is
the VMAF score itself, so showing VMAF YUV calculates VMAF too (its column
is shown with it). Only the app calculates the U and V, with its
libvmaf-fast, from the same pictures as the run's other metrics: on the GPU
where VMAF runs on the app's Vulkan engine, else on the CPU (a custom model,
videos above 10 bits and chroma planes of an odd size also go to the CPU).
Each chroma plane has a
quarter of the luma's pixels, so the two add about half of VMAF's own work.
It needs pictures of at least 64x64 (chroma planes of 32x32); a smaller one
fails VMAF YUV alone, with the reason.

MS-SSIM, MS-SSIM YUV (6:1:1), PSNR-HVS-M, CAMBI and CIEDE2000 are the other
metrics codec comparisons such as MSU's and Xiph's report. Only the app
calculates them, with its libvmaf-fast, beside VMAF on the GPU or the CPU
or alone, on every GPU maker's route:

- MS-SSIM is SSIM of the luma at five scales (Wang et al. 2003); MS-SSIM YUV
  weighs its three planes' scores 6:1:1, as SSIM YUV does.
- PSNR-HVS-M (Ponomarenko et al. 2007, as Xiph's Daala tools calculate it)
  is the PSNR of the error of 8x8 DCT blocks weighted by the eye's contrast
  sensitivity and masking, of 0.8 Y + 0.1 Cb + 0.1 Cr. An identical (or fully
  masked) frame scores infinity, and the video's score is the mean of the
  frames' error, -10 log10(mean(10^(-dB / 10))), as Daala's tool averages it.
- CAMBI is Netflix's banding score of the test video alone, without the
  reference: 0 is no banding; about 5 starts to annoy, and 24 is the most
  Netflix has seen (unwatchable). A test video of fewer bits than
  the comparison (an 8-bit encode of a 10-bit source) is scored as the 8-bit
  video it is, a smaller one at its own size, and a PQ (HDR10) one on the PQ
  curve.
- CIEDE2000 is the mean colour difference (ΔE00) of the frame's pixels: 0 is
  identical, about 1 just noticeable. libvmaf gives it as 45 - 20 log10(ΔE00);
  the app shows ΔE00 itself.

A comparison one of them cannot score fails that metric alone, with the
reason, and the others are calculated as usual: MS-SSIM needs pictures of at
least 176x176 (MS-SSIM YUV 352x352), CAMBI at least 216 pixels on one side
and at most 3840x2160 (width plus height at most 6092), PSNR-HVS-M 16x16 and
at most 12 bits, and CIEDE2000 8, 10, 12 or 16 bits. MS-SSIM YUV, PSNR-HVS-M
and CIEDE2000 read the chroma, so turning one on makes the decoders hand over
whole pictures, as the 6:1:1 metrics do.

**Y, U and V of the YUV metrics**, in Add/remove metrics' Specialized section, adds
a column for each plane (Y, U, V) beside PSNR YUV, SSIM YUV, MS-SSIM YUV and
PSNR-HVS-M, for the ones shown. The planes are calculated in their metric's
pass at no extra cost, whenever that metric is ticked; they have no tick box
of their own. Videos scored before the switch was turned on show empty plane
columns until they are calculated again. The planes are saved, exported and
graphed as metrics of their own (`psnr_yuv_y`, `psnr_yuv_u`, `psnr_yuv_v`,
and so on), and the command line takes them by these names with `-m`; its
`all` leaves them out. With the switch on, its default metrics include the
planes of the YUV metrics it calculates.

Subsampling reduces libvmaf-backed scored frames. XPSNR-only currently keeps
every frame; do not interpret different coverage as identical measurements.

## Compare without calculating

Load a reference and encodes, then open **Video Compare**. Metric scores are
optional. Use still mode for frame inspection or playback mode for motion.

- Hold **S** to show the source; release it to return to the selected encode.
- Left/right switches encodes; Space toggles playback.
- Use the frame/timestamp controls to select a position.
- **Zoom** shows the frame fitted to the window, or at a size you choose or type in:
  100% is the video's own pixels. Drag a zoomed frame to move it.
- Opened from the Worst Frames tab's **Worst frames** list, **Previous worst** and
  **Next worst** step through that list's frames while its video is shown.
- **Loop around frame N** plays the seconds around the frame you went to
  (from 1 s before to 1 s after it unless changed) again and again; Play
  starts them from the beginning. The row names that frame and, after it,
  the frames that repeat; while Loop is ticked the timeline shades them.
  Each side is one value, in seconds to the millisecond or in frames of the
  video shown: typing either one sets the other. The seconds are kept, so a
  video of another frame rate loops as long, in its own frames.
- The frame pair's save icon, at the top card's right end, saves the frame of the reference and of the test video as
  two PNG files in Settings' Export folder (Documents unless set); a
  right-click, **Frame pair as PNG...**, in a folder you pick. They are named
  after the video, the frame, its time, the Worst frames score when it is
  one, and the side. They are cropped and scaled as compared, HDR mapped to
  SDR.
- A video whose stream asks the player to add film grain (VVC's H.274
  grain, such as VVenC's film grain analysis writes, AV1's, AFGS1) is shown
  as it is scored: with its grain, or without it if it is scored without
  (Film grain in the Options panel). VVC's H.274 grain is added by VVdeC,
  at any bit depth.
- Review HDR preview and source-resolution options for your comparison.

This is an A/B switching view, not two permanently side-by-side players.
Crop geometry is most reliable when supplied by a completed analysis. Read
the preview status when an automatic crop has not yet been determined.
Keyboard shortcuts depend on focus; editing a text field may consume a key.

## Bitrate without calculating

Use **Bitrate Viewer** to add and analyze files independently. Metric runs also
add inputs for bitrate analysis. The three views show packet/frame size,
one-second bitrate and GOP-based bitrate. Only the first video stream is
counted: audio, subtitles and container overhead are excluded. Overall video
bitrate is video packet bytes × 8 divided by the measured video duration.
With two videos shown, **Difference** draws the first minus the second in a
band under the graph with its own scale: frame by frame in the frame view, and
at each moment in the second and GOP views (where two encodes' GOPs differ,
each moment's GOP bitrate is subtracted).
Under the graph, **Go to frame** and **Go to time**, and the range from
frame or from time, work as on Metric Graphs: the table then gives the
range's figures. The save icon at the graph card's top right saves the graph
as PNG and each drawn file's bitrate data, as the command line's `bitrate
--csv` writes it (`<video>_bitrate_frames.csv`, `_seconds.csv`, `_gops.csv`)
and as JSON (`<video>_bitrate.json`: the same three tables), in the formats
Settings > Storage > **Quick saves** ticks; a right-click on it or on the
graph offers **Graph as PNG...**, **Bitrate data as JSON...** and **as
CSV...**. A time is the first video shown's, from its
start when **Adjust stream start time to zero** is ticked.
The command line's `bitrate` command (below) reads the same numbers.
Each file's reading is saved with the saved results and is not read again:
the file adds at once the next time, renamed, moved or copied too (it is
known by its content, as saved scores are). A remux is read again.

## BD-rate

**BD-rate** shows how much bitrate a test encoder saves (−, in green) or
costs (+, in red) against an anchor for the same score, for each sequence and
each of the nine metrics, with the mean and median over the sequences. Fill
the **Anchor** and **Test** boxes with **Add videos…** (videos scored in this
session: each one's sequence is its reference's file name, its rate the
video's own bitrate) or **Import CSV/JSON…** (a file in the command line's
`bdrate` format, below); give each a name. Sequences are paired by name. A
cell is n/a where there is no BD-rate; its tooltip says why, and
**Warnings** lists what else needs a look. **Group by** adds a mean and
median for each value of a column of the anchor's CSV files, such as
`class`. Click a sequence to draw its rate curves for the metric chosen:
bitrate on a log scale, a dot per encode, and the scores both curves cover
shaded, the range the BD-rate averages over. **Copy as markdown** copies the
table as the command line prints it. The table card's save icon saves every
number in full, named after the two encoders, as JSON (the command line's
`bdrate --json`) and CSV; the chart card's, the chart as PNG. A right-click
offers **Table as JSON...**, **Table as CSV...** or **Chart as PNG...**.
The numbers are the command line's `bdrate`.

## Command line

`VideoMetricsLab-cli.exe`, beside the app, calculates metrics without the
window (from a source build: `cpp\build\windows\cli\Release\vml-cli.exe`; on a
Mac it is inside the app, `VideoMetricsLab.app/Contents/MacOS/VideoMetricsLab-cli`).
The examples below are Windows' PowerShell; on a Mac they are the same with
`/` in paths, and a script reads the JSON with `jq`:

```powershell
.\VideoMetricsLab-cli.exe compare reference.mkv encode1.mkv encode2.mkv
```

It runs what **Calculate metrics** runs and prints each video's scores, and
for several videos a summary side by side. The app's saved settings are its
defaults (FFmpeg's folder, the cache, the metrics a new video starts
with); options override them for one run and nothing is written to the
settings. Scores go to the same cache, so a comparison made on the command
line opens in the app already calculated, and scores the app has saved are
not calculated again.

More examples:

```powershell
# Two metrics for every .mkv in a folder (the reference is left out of what a pattern finds)
.\VideoMetricsLab-cli.exe compare reference.mkv encodes\*.mkv -m vmaf,ssimulacra2

# Every metric, with each video's per-frame scores as results\<video>.csv
.\VideoMetricsLab-cli.exe compare reference.mkv a.mkv b.mkv -m all --csv results

# The first 30 seconds only, everything calculated on the CPU
.\VideoMetricsLab-cli.exe compare reference.mkv encode.mkv --duration 30 --cpu

# HDR10 in files with no colour tags (y4m), the first 2 seconds (CVVDP models an HDR display for HDR video)
.\VideoMetricsLab-cli.exe compare reference.y4m encode.y4m -m cvvdp --colour pq-bt2020 --duration 2

# Two minutes, an hour into the videos
.\VideoMetricsLab-cli.exe compare reference.mkv encode.mkv --range 1:00:00-1:02:00

# CVVDP on two displays of your choice: a 30-inch HDR monitor and a 65-inch HDR TV
.\VideoMetricsLab-cli.exe compare reference.mkv encode.mkv -m cvvdp --cvvdp-display "30-inch 4K HDR monitor, dim room" --cvvdp-display "65-inch 4K HDR TV, 1000 nits, living room"

# A raw VVC stream straight from an encoder (no container), at the frame rate it stores
.\VideoMetricsLab-cli.exe compare reference.y4m encode.266

# A raw YUV reference (pictures only: 1080p, 10-bit 4:2:0) and a raw VVC stream
.\VideoMetricsLab-cli.exe compare reference.yuv encode.266 --size 1920x1080 --pix-fmt yuv420p10le --fps 24000/1001

# A 1080p encode against a 4K reference, compared at 4K
.\VideoMetricsLab-cli.exe compare reference_4k.mkv encode_1080p.mkv --scale-to reference

# For a pipeline: each encode's result, per-frame scores, JSON and log under its own names,
# such as runs\encode-VML-2.6.metrics.json and logs\encode-20261117-180912.log
.\VideoMetricsLab-cli.exe compare reference.mkv encode.mkv --name "{test}-VML-{version}" --run-dir runs --csv scores --json "{test}-VML-{version}-Score-Results.json" --log-file "logs\{test}-{date}-{time}.log"

# In a script: read the score from JSON
$run = .\VideoMetricsLab-cli.exe compare reference.mkv encode.mkv -m vmaf --json - -q | ConvertFrom-Json
$run.videos[0].metrics.vmaf.score

# Each encode's length, frames and bitrate, as the Bitrate Viewer tab
.\VideoMetricsLab-cli.exe bitrate encodes\*.mkv

# A raw VVC stream's bitrate, and its graphs' figures as CSV files in the folder bitrate
.\VideoMetricsLab-cli.exe bitrate encode.266 --csv bitrate

# What would calculate each metric on this PC
.\VideoMetricsLab-cli.exe devices

# The same as JSON, for a script (devices --json FILE writes it to FILE)
.\VideoMetricsLab-cli.exe devices --json -
```

| Option | Meaning |
| --- | --- |
| `-m`, `--metrics LIST` | Comma-separated: `vmaf`, `vmaf_neg`, `vmaf_v1`, `vmaf_hdr`, `psnr`, `ssim`, `xpsnr`, `ssimulacra2`, `butteraugli`, `cvvdp`, or `all` (every one but `vmaf_hdr`, which scores PQ (HDR10) videos only). Left out: the metrics a video added to the app starts with. |
| `--model auto\|standard\|4k\|FILE` | VMAF v0.6.1's model. `auto` uses the 4K model for a 4K comparison. |
| `--model-v1 MODEL` | VMAF v1's model, as the Options panel's **VMAF v1 model**: `auto` (the one for the size frames are compared at), or a bundled one by its name: `vmaf_v1_3d0h`, `vmaf_v1_5d0h` (1080p), `vmaf_v1_1d5h_2160`, `vmaf_v1_3d0h_2160` (4K), or their high frame rate models `vmaf_v1_hfr_3d0h`, `vmaf_v1_hfr_5d0h`, `vmaf_v1_hfr_1d5h_2160`, `vmaf_v1_hfr_3d0h_2160`. |
| `--model-hdr MODEL` | HDR VMAF's model, as the Options panel's **HDR VMAF model**: `auto` (the one for the size frames are compared at), or a bundled one by its name: `hdr_vmaf_3d0h`, `hdr_vmaf_5d0h` (1080p), `hdr_vmaf_1d5h_2160`, `hdr_vmaf_3d0h_2160` (4K), or their high frame rate models `hdr_vmaf_hfr_3d0h`, `hdr_vmaf_hfr_5d0h`, `hdr_vmaf_hfr_1d5h_2160`, `hdr_vmaf_hfr_3d0h_2160`. |
| `--black-bars auto\|source\|none` | Detect and cut black bars; `source`: cut the source's only and compare the test video's full frame (an encode cropped when it was made), as the Options panel's **Source only (test video already cropped)**, which takes the scores saved by `auto` where `auto` compared the test video's full frame too; or compare whole pictures. |
| `--scale-to test\|reference` | Which video's size the other is scaled to when they differ. |
| `--scaler bicubic\|bilinear\|lanczos\|spline` | Scaling algorithm. |
| `--duration SECONDS` | Compare only the first SECONDS of each video (whole seconds, or H:MM:SS). |
| `--range START-END` | Compare only that part of each video, from START to END (whole seconds, or H:MM:SS: `1:00-1:30`); without END, to the videos' end. |
| `--colour pq-bt2020\|hlg-bt2020\|bt709` | The colours of a video with no colour tags (y4m, say); a tagged video keeps its own. Left out, such a video takes the other video's tags when that one has them (a y4m encode of a tagged HDR10 source is read as HDR10), and is otherwise taken as SDR, which changes CVVDP, SSIMULACRA2 and Butteraugli. Scores with colours given or taken are not saved. |
| `--film-grain with\|without` | Score with the film grain a video's stream asks the player to add, as players show it, or without it, the coded pictures only. Left out: as Settings' **Film grain** default says; with, where it asks. |
| `--fps RATE` | The frame rate of a raw video stream or raw YUV file (below), such as `24000/1001`, `23.976` or `60`, in place of a rate the stream stores. Required for a raw YUV file and for a raw stream that stores none, when the reference or a test video is one. A raw test stream's scores are saved under this rate; a raw reference's are not saved. |
| `--size WIDTHxHEIGHT`, `--pix-fmt FORMAT` | The picture size and pixel format (as FFmpeg names it: `yuv420p`, `yuv420p10le`...) of a raw YUV file (below). Required with `--fps` for one. |
| `--cvvdp-display PRESET` | The display CVVDP models, by preset name. Repeat it for a CVVDP score on each display; the JSON then lists each display's name, score, display parameters and full CVVDP statistics in the video's `cvvdp_displays`. Left out: the app's default display, Automatic unless Settings name a preset: the HDR display (`30-inch 4K HDR monitor, dim room`, the official `standard_hdr_pq`) for an HDR reference (PQ or HLG), the SDR one (`30-inch 4K monitor, office`, `standard_4k`) otherwise. |
| `--cvvdp-fill` | Scale the compared frames up to fill CVVDP's display, aspect kept, as the Options panel's **Scale the video to fill the display**. Left out: off, as the official tool (a 1080p video covers a quarter of a 4K display). These scores are saved apart. |
| `--subsample N` | Score every N-th frame. CVVDP needs every frame. |
| `--cpu` | Calculate on the CPU every metric that can be (CVVDP is GPU only): VMAF as with `--vmaf-on cpu`, unless `--vmaf-on` says otherwise, and SSIMULACRA2 and Butteraugli rather than on the GPU. |
| `--vmaf-on` `gpu\|cpu` | Where VMAF v0.6.1, VMAF NEG and VMAF v1 are calculated. Left out: as the app's Options panel was last set. |
| `--vship` `vship-fast\|vship\|vship-hip\|vship-cuda` | The Vship that calculates SSIMULACRA2, Butteraugli and CVVDP on the GPU: `vship-fast`, the app's own (the window's GPU), or stock Vship: `vship` (the window's Vship Vulkan), or if Vulkan has problems `vship-hip` (AMD) or `vship-cuda` (NVIDIA). Stock Vship's three share their scores, which differ slightly from `vship-fast`'s and are saved apart. Left out: as Settings > Metrics > GPU metrics sets it. |
| `--vship-pass one\|each` | SSIMULACRA2, Butteraugli and CVVDP in one pass per video, which decodes it once (faster only on a very powerful GPU, and needs much GPU memory), or a pass for each. The scores are the same. Left out: as Settings > Metrics > GPU metrics sets it. |
| `--no-gpu-decode` | Decode the videos on the CPU, VVC too. |
| `--gpu-decode auto\|nvidia\|intel\|amd` | Decode the videos on the GPU, as the Options panel's **GPU decode**: `auto`, the first GPU whose decoder takes each video (NVIDIA, Intel, then AMD), or that maker's, which the PC must have. A Mac uses its own. Left out: on or off as the app's Options panel was last set, the maker automatic. |
| `--gpu NAME` | The GPU that calculates VMAF, VMAF v1, SSIMULACRA2, Butteraugli and CVVDP, on a PC with more than one: its name as Windows gives it, or a part of it (`radeon`, `"rtx 5090"`), with `#2` for the second of two GPUs of one name; its maker's decoder is tried first. `auto`: a discrete GPU first. A name no GPU has is refused, with the GPUs there are, and so is a part that more than one GPU's name has, with those GPUs. Left out: as Settings > Metrics > GPU metrics sets it. |
| `--vvc-decoder gpu\|cpu` | Decode a VVC video on the GPU (vvdec-vulkan, experimental) or on the CPU (VVdeC). The scores are the same. Left out: as Settings > Metrics > GPU metrics sets it (the CPU unless changed). |
| `--threads N` | libvmaf threads. Left out: automatic. |
| `--parallel 1\|2` | Videos calculated at once on the CPU. |
| `--frame-counts check\|skip` | When a video may be out of sync with the reference, the frame counts stored in the files are always checked. check: also read a file that doesn't store one from start to finish to count its frames, which can add minutes per video; skip: don't. Default: Settings > Storage. |
| `--recalculate` | Ignore saved scores. |
| `--test-remuxes on\|off` | On: read each test video calculated once more after the run, so that a remux of it keeps its saved scores, and read a video whose stream begins as a saved one's to tell whether it is that video (see [Saved results and caches](#saved-results-and-caches)). Off: neither. Left out: as **Settings > Storage > Recognize test videos remuxed after scoring** (off unless changed). |
| `--difference` | With exactly two test videos: the first minus the second as well, frame by frame (CVVDP second by second), paired as **Difference** in Metric Graphs pairs them. The tables end with a difference table (Score: the two scores subtracted; Mean, Min, Max: of the frames' differences; Paired: the frames, or seconds, both have), the JSON gets a `difference` section, and with `--csv` it is written as `<first>_minus_<second>.csv` (the files' names). |
| `--csv FOLDER` | Write each video's per-frame scores as `<video>.csv`, as the window's per-frame scores export writes them. |
| `--json FILE` | Also write the results as JSON. `--json -` prints the JSON instead of the tables, for a script to read. |
| `--frames-json FOLDER` | Write each video's per-frame scores as JSON, `<video>.frames.json`, as the window's per-frame JSON export writes them (see [The results file](RESULTS_JSON.md#per-frame-scores-as-json)). The same names and numbers after a name taken as `--csv`. |
| `--batch CSV` | Compare the videos a CSV file lists, in place of REFERENCE and TEST, all in one run (the start-up and the GPU checks once, not once per reference). A row per test video, with the columns `reference` and `test` (a file or a pattern) and, if wanted, `sequence` and any others, which `--scores-csv` keeps; paths are relative to the CSV file's folder. The tables come per reference, and `--json` writes `{"batch": [...]}`, each reference's results as without `--batch`. A reference that cannot be read is said and the others are still compared (exit code 4). Not with `--difference`. |
| `--scores-csv FILE` | Write each test video's bitrate (`kbps`) and scores to FILE, a row per video, with its `sequence` (the `--batch` row's, else the reference's file name), `test`, `reference` and the `--batch` row's other columns: the file `bdrate --anchor` and `--test` read. A video whose bitrate is not known, or with only a part compared (`--range`, `--duration`), is left out, said. A CSV file for programs: no byte order mark (before 2.7 it had one), its lines ended CR LF. |
| `--worst N` | After the tables, each metric's N worst dips: the worst frame of each (a frame within 1 s of a worse one is part of its dip), with every metric's score there. |
| `--segments CONDITIONS` | List the stretches where scores fall past a threshold, such as `"vmaf<90,ssimulacra2<70"` (quoted: `<` and `>` are the shell's; Butteraugli's worst are high: `"butteraugli>2"`). A frame is weak where any of them holds. Frames are numbered from the video's first, as its encoder numbers them; CVVDP is judged second by second. `--segments saved` takes the conditions of the weak segment profile in use in the app's Settings > Weak segments, `--segments profile:NAME` those of profile NAME (any case; `Default` too), and `--segments preset:KEY` those of a preset (see [Weak segment profiles](#weak-segment-profiles)); conditions after it change it: `"preset:grainy-film,vmaf<80"` replaces the preset's VMAF condition (all of those given on a metric replace its conditions on it: `"preset:grainy-film,vmaf<85,vmaf>20"` keeps both), and a metric it has none for is added; those on metrics not compared are left out, and said; the profile's any/every choice, shortest segment and gap apply where the command gives none. A profile with no condition on the metrics compared takes one on the metric shown, as the Worst Frames tab starts: below the metric's first band of worse frames (`vmaf<85`), on the metric Metric Graphs shows where it is compared, else the first compared; a note says so. A preset none of whose conditions is on a metric compared is refused. |
| `--segments-all`, `--segments-any` | A frame is weak only where every `--segments` condition holds; or where any one does, also when the profile or preset says every one must (any is the default otherwise). One or the other. |
| `--segment-min-frames N`, `--segment-merge-frames N` | Leave out segments shorter than N frames (default 1); join segments at most N frames apart (default 0). |
| `--segments-dir FOLDER` | Write each video's segments to FOLDER as `<video>.<metrics>.segments.json` and `.csv`, `<metrics>` the conditions' metrics joined by `+` (`<video>.vmaf+ssimulacra2.segments.json`), as the app names them. |
| `--segments-report FILE` | Write every test video's segments (with `--batch`, every reference's) to FILE as one report, as the app's exports of all the videos with **Weak segments** ticked: JSON, or CSV when FILE ends in `.csv`. The fields of `--json`'s path are filled in in its path. |
| `--require RULES` | A quality gate: the bars each test video's scores must reach, comma-separated, such as `"vmaf>=95,vmaf.worst_1_percent>=90,butteraugli<=1.5"` (quoted: `<` and `>` are the shell's). Each is METRIC, a statistic after a dot if wanted, then `>=`, `>`, `<=` or `<` and a number. The statistics are those of the JSON's `metrics`: `score` (the default), `median`, `min`, `max`, `worst_10_percent`, `worst_5_percent`, `worst_1_percent` and `worst_0.1_percent`; for CVVDP, all but its score are of its per-second scores (`cvvdp.min` is its worst second). Repeat it to add rules. A video that misses a bar, or whose metric was not scored, fails its gate and makes the exit code 3; a metric that does not apply to a video (HDR VMAF of SDR video) is left out of its gate. Each video's checks follow the tables, and are in the JSON's `gate`. A rule's metric must be one of those compared. |
| `--fail-on-segments` | A quality gate: a test video with a weak segment, by `--segments`' conditions, fails its gate (exit code 3). With `--require`, both must pass. |
| `--stills FOLDER` | Write PNGs of the worst frames to FOLDER, as Video Compare's frame pair save makes them (cropped and scaled as the metrics compared them, HDR mapped to SDR): `<video>.frame<N>.reference.png`, `.test.png`, and `.side-by-side.png` with the reference on the left. The JSON lists each video's `stills`. |
| `--stills-from {worst,segments}` | `worst` (default): each metric's worst dips, as `--worst` finds them, a frame two metrics chose once. `segments`: the worst frame of each weak segment `--segments` finds, by its first condition's metric, the weakest segments first. |
| `--stills-count N` | How many dips of each metric, or how many segments (default 5, at most 100). |
| `--run-dir FOLDER` | Write each video's result to FOLDER as `<video>.metrics.json`, the file the Videos tab's **Run file (.metrics.json)...** writes: **Import run file...**, on the Videos tab or Metric Graphs, opens it on any PC, without its saved scores. |
| `--log-file FILE` | Also write the app's log to FILE, as the window writes its own: the whole run, whatever `-v` says, added to the end when FILE is there. |
| `--name TEMPLATE` | Name each test video's files (`--csv`, `--segments-dir`, `--run-dir`) after TEMPLATE in place of `<video>`. Its fields: `{test}` and `{reference}` (the files' names without extension), `{version}` (the app's: `2.6`), `{date}` and `{time}` (the run's start: `20261117`, `180912`) and with `--batch` `{sequence}` (the row's, else the reference's name); `{{` and `}}` are braces. In the names, characters other than letters, digits, `-`, `_` and `.` become `_`, as in `<video>`. The same fields, but `{sequence}`, are filled in in `--json`'s, `--scores-csv`'s, `--segments-report`'s and `--log-file`'s paths, `{test}` and `{reference}` only when one video is compared. A name already taken gets a number, which the JSON's `csv`, `run_file` and `segment_files` give. |
| `--notify` | Send a notification as the run ends, as **Settings > Notifications** in the app say: to the services on there, with its message, for the outcomes and run lengths it is on for (see [Notifications](#notifications)). Whether it was sent, or why not, is said on the progress lines (a problem with `-q` too); the exit code is the run's either way. On macOS and Linux it needs `curl`, which sends it. |
| `-q`, `--quiet` | No progress, only the results. |
| `-v`, `--verbose` | The app's log as well. |
| `--progress text\|jsonl` | `text`: progress lines for a person (the default). `jsonl`: a JSON object a line on the error stream, for a program to read, `-q` or not (below). |
| `--timeout SECONDS` | Cancel the run after SECONDS (whole seconds, or H:MM:SS) from the command's start, as Ctrl+C does: what finished is given and saved, the JSON says `"cancelled": true, "timed_out": true`, and the exit code is 124. Counting the frames of videos answered by their saved scores (`--frame-counts check`) is part of the run. Reached while the videos are read or the results written, it ends the command at once (exit code 124, no results), as Ctrl+C does then; once the results are given (a notification sent, test streams recorded), or once Ctrl+C has cancelled the run, it does nothing. |
| `--video-timeout SECONDS` | Stop a video whose run takes more than SECONDS from its start (a further `--cvvdp-display`'s run counts as its own); the other videos go on. What it had not finished fails with the code `timed_out`, and the exit code is 124. |
| `--cancel-on-stdin-eof` | Cancel the run, as Ctrl+C does, when the standard input ends. For a program that starts this one and cannot send it Ctrl+C: it keeps the input open, and closes it to cancel. Not with an input that is empty from the start (`< NUL`), which would cancel at once. Once the results are given, its end cancels nothing: the command ends as the run did. |
| `--settings FILE` | Read the settings from FILE (a `settings.json`, such as a copy of the app's, kept with a script) in place of the app's. Every option left out takes its value from it, as it would from the app's. It is never written; a FILE that is not there or is not JSON is an input not read (exit code 4). |
| `--no-settings` | Take every setting at its default, as a new install has them, whatever the app's settings say: the same command gives the same run on any PC. |
| `--cache-dir FOLDER` | Read and save the saved scores (and the black bars found) in FOLDER, in place of the folder Settings > Storage names. |
| `--no-save` | Read the saved scores, but save nothing to them: no scores, black bars or records of remuxes or moved files. For a trial whose scores should not be kept. |
| `--no-cache` | Neither read nor save the saved scores: every metric is calculated, and nothing is kept. Not with `--no-save` or `--cache-dir`. |

`compare --help` lists the same with examples. The last column of each
video's table says where a score was calculated: a metric asked for on the
GPU is calculated on the CPU where no GPU can. When a GPU metric falls back,
to the CPU or from the app's own Vship to stock Vship, because it failed on the
GPU during the run or because the GPU's check crashed, failed or scored
wrongly, a note says so and why (in the JSON too, with the video's other
notes). A Vship pass whose FFmpeg fails to give it the videos' frames is
first made once more with the app's own Vship, with no note: only a second failure
falls back. The app lists the fallbacks under **GPU fallbacks** in the
tooltip of the video's name.

Options left out take the app's settings, as the window last saved them,
so a script that runs where someone also uses the app can get other
metrics, another GPU or another Vship from one day to the next. To hold a
run to settings of its own, give `--settings FILE` (a copy of
`settings.json` kept with the script) or `--no-settings` (the defaults);
`devices` takes them too. The JSON's `profile` says where the run's
settings came from and which options left out they filled in (see [The
results file](RESULTS_JSON.md)). These change the settings only: the saved
scores, the GPU checks and the logs stay in the app's folder,
`.videometricslab` in your home folder, unless `--cache-dir` or
`--no-cache` says otherwise. To move all of it, set the environment
variable `VML_HOME` to a folder: the app's own files (settings, saved
results, logs, GPU checks) are kept there in place of `.videometricslab`, a
profile of its own. The window takes `VML_HOME` too, so a window started
with it set shows that profile's settings and saved results.

Progress is written to the error stream and results to the output stream, so
`--json -` can be piped. Piped or redirected, both are UTF-8 whatever the
system's code page, so a script reads them as UTF-8 (in Python,
`subprocess.run(..., encoding="utf-8")`); a console is sent every character
as it is, whatever its code page.

A note that is the same for several videos (no colour tags, say) is said
once, with the videos it is of. With `--progress jsonl`, every line on the
error stream is a JSON object: its `event`, then `seconds` since the command
started, then what the event says:

| `event` | What it says |
| --- | --- |
| `note` | A note on how videos are scored (as their JSON's `notes`): `code`, `text`, and the `videos` it is of. Notes found while the videos are read come before `run_started`. |
| `run_started` | The run starts: how many `jobs` (a job is a video, or a further `--cvvdp-display` of one). |
| `video_started` | A job starts: its `job` (from 0) and its video's `path` (and `cvvdp_display` for a further display). |
| `progress` | A job's figures, at most once a second: `job`, `path` and its `tasks`, each with its `metrics`, `state` (`starting`, `waiting`, `running`, `done`, `failed`), the frames `done` of `total` and `fps`. |
| `video_done` | A job ended: `job`, `path`, `status` (`done`, `partial`, `failed`, `cancelled` or `timed_out`) and its line as `text`. |
| `message`, `problem` | Any other progress line, as `text`; a `problem` is one `-q` would still show, and every error the command says is one. |
| `log` | A line of the app's log (its failures, and everything with `-v`): its `level` (`ERROR`, `WARNING`, `INFO`), the `logger` it is from, and its `text`. |
| `run_done` | Last, once the command got as far as reading its videos: its `exit_code`, and whether it was `cancelled` and `timed_out`. |

A reader should skip events and keys it does not know: later versions may
add some.

The exit code says how the run went, so that a script can act on it without
reading the text:

| Code | Meaning |
| --- | --- |
| 0 | Every video got every metric asked for that applies to it. A metric that does not apply to the videos, such as HDR VMAF of a video that is not PQ (HDR10), is left out with a note and its status `not applicable`, as is a metric ticked by default in the app that cannot be calculated (CVVDP with `--subsample`, say). |
| 1 | A video failed or lost a metric (a metric asked for by name that cannot be calculated, such as `-m cvvdp` with `--subsample`, counts), or a file asked for could not be written. |
| 2 | A wrong command: nothing was read or run. Among them a raw stream that stores no frame rate without `--fps`, a raw YUV file without `--size`, `--pix-fmt` and `--fps`, `--difference` without two test videos, patterns that find only the reference, and FFmpeg that cannot be used. |
| 3 | A quality gate failed: every score asked for is there, but a test video misses a `--require` bar or has a weak segment with `--fail-on-segments`. |
| 4 | An input could not be read: the reference, a test video, a pattern that finds no file, or the `--batch` list. The other videos are still compared, and the JSON says which could not be read. |
| 124 | Timed out: `--timeout` cancelled the run, or `--video-timeout` stopped a video. |
| 130 | Cancelled with Ctrl+C (or Ctrl+Break, the console window closed, or the end of the standard input with `--cancel-on-stdin-eof`). |

When more than one applies, 130 comes first, then 124, then 4, then 1, then 3. With `--json`,
the results are written for every code but 2, also when nothing could be
compared: a reference that cannot be read gives a document with its `error`
and `error_code` and no videos, and a `--batch` list that cannot be read
one with an empty `batch`. Each error, failure, note and reason in the JSON
has a code beside it, a word that stays the same from version to version
where the text may change; [The results file](RESULTS_JSON.md#codes) lists
them.
Ctrl+C stops the run as **Cancel** does; scores that had finished are saved.
Closing the console window does the same on Windows' classic console. A
program that starts this one can send it Ctrl+Break (on Windows, start it in
a new process group: in Python, `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP`
and `send_signal(signal.CTRL_BREAK_EVENT)`), or use `--cancel-on-stdin-eof`.
Ending it outright (End task, `TerminateProcess`) ends the FFmpeg and other
programs it started too, but gives no results.

Unlike the app, the command line does not ask before a long SSIMULACRA2 or
Butteraugli run on the CPU. Text is English whatever the app's language.

### Saved results and run files

The scores the app has saved (the Saved Results tab's, by the window or the
command line) can be listed, read and deleted without their videos, which may
be gone:

```powershell
# Every saved comparison, newest first, with its ID; --filter narrows them by file name
.\VideoMetricsLab-cli.exe saved list --filter encode

# One comparison's scores, by the first 6 or more hex digits of its ID, as compare gives them
.\VideoMetricsLab-cli.exe saved show 3fa9c1 --segments "vmaf<90" --worst 5 --json -

# Every comparison of a test video, by its path
.\VideoMetricsLab-cli.exe saved show encodes\encode.mkv --frames-json scores

# Delete one, by at least 12 hex digits of its ID (it asks nothing)
.\VideoMetricsLab-cli.exe saved delete 3fa9c1d24b07

# Run files (.metrics.json: compare --run-dir, the window's Run file (.metrics.json)...), reported the same way
.\VideoMetricsLab-cli.exe report runs\*.metrics.json --segments saved
```

`saved` and `report` take the settings as compare does: the app's, `--settings FILE`'s or, with `--no-settings`, the defaults. They give the saved results' folder and the weak segment profiles. `saved --cache-dir FOLDER` reads and deletes the saved results in FOLDER, as `compare --cache-dir FOLDER` saved them.

`saved show` and `report` take compare's outputs: `--json`, `--csv`,
`--frames-json`, `--run-dir`, `--worst`, `--segments` (with `saved`,
`profile:NAME` and `preset:KEY`) and its options, `--segments-dir`,
`--segments-report` and `--stills` (whose videos must still be there). The JSON is compare's results file, its `read_from`
`"saved results"` or `"run files"`; a video's `settings`, `black_bars`,
`metric_status`, `decode_plan` and `timings` are `null`, as for a saved run
the window opens, which does not record them. With several references, it
is `{"batch": [...]}`, as `compare --batch` writes it. `saved list --json`
gives each comparison's `id`, `test`, `reference`, `reference_found`,
`saved_utc`, `bytes` and `metrics` (format `saved-list-1`). The exit code is
4 for an item that names no saved comparison (or a run file that cannot be
read), 2 for an ID too short, or one that begins more than one comparison's
ID to delete.

### Raw video streams

An encoder's output with no container (`.264`, `.265`, `.hevc`, `.266`,
`.obu`, `.m2v`, `.m4v`, `.mjpeg`...) has no timestamps, and may store no frame
rate. The app reads the rate from the headers of H.264, HEVC, VVC, AV1,
MPEG-1, MPEG-2, MPEG-4 Part 2, Dirac and AVS2 streams that store one. For any
other, Media info says the frame rate is unknown and the app asks for it, as a
number or a fraction such as 24000/1001, when the video is added and again at
**Calculate metrics**; a rate in the headers is only what the encoder was told,
so a stream can also store a wrong one, which the app then refuses as a mismatch
with the reference. The Bitrate Viewer asks the same. A raw test stream's
scores are kept in the saved scores under the rate it was read at (a raw
reference's are not), and Video Compare decodes it from its start to reach a frame,
which takes a while far into a long stream. The command line reads the same
rate from the same headers and needs `--fps` for any other raw stream; `--fps`
also replaces a stored rate, for every raw stream of the command. Its frames are counted when the video is read (the file is read
once, not decoded) and timed at that rate, and FFmpeg decodes it. A raw stream
scores exactly as the same stream in a
container: checked for H.264, HEVC, VVC, AV1 (OBU), EVC, MPEG-1, MPEG-2,
MPEG-4 Part 2, H.263, H.261, MJPEG and Dirac. FFmpeg also reads raw AVS2,
AVS3, CAVS, VC-1 and APV streams; these are not checked (FFmpeg's AVS2
decoder gave the one tried fewer frames than it holds, in a container too,
so it was refused as cut short).

A container can say what a raw stream may not: its colours and where its
chroma samples sit, which an encoder may leave out and which change
SSIMULACRA2, Butteraugli and CVVDP (`--colour` gives the colours; a note
after the results says when they were guessed), and the shape of its pixels
(H.263 and H.261 record their own). Black bars are found from bytes partway
through an H.264, HEVC, VVC, MPEG-1, MPEG-2, MJPEG or Dirac stream. Any other
stream, or one that cannot be decoded from there (its only keyframe its
first, or an H.264 or VVC stream with its parameter sets only at its start),
is read from the start up to the last part sampled, which takes a while for
a long stream (`--black-bars none` skips it).

A raw YUV file (`.yuv`) is pictures and nothing else: `--size`, `--pix-fmt`
and `--fps` say what they are, for every `.yuv` file of the command. It is
read as FFmpeg's `rawvideo`, and scores exactly as the same pictures in a
container. It has no colour tags: it takes the other video's when that one
has them, and `--colour` gives them otherwise.

### Bitrate

`bitrate VIDEO...` reads each video's video stream with ffprobe, as the
**Bitrate Viewer** tab does, without decoding it, and prints a table of its
length, frames, average bitrate and lowest and highest one-second bitrate.
It needs only ffprobe. A raw stream that stores no frame rate needs `--fps`, as for `compare`; a raw
YUV file has no bitrate to read.

| Option | Meaning |
| --- | --- |
| `--fps RATE` | A raw stream's frame rate, such as `24000/1001`, in place of a rate it stores. |
| `--csv FOLDER` | Also write the tab's three graphs' figures as `<video>_bitrate_frames.csv` (each frame's time, size and whether it is a keyframe), `_bitrate_seconds.csv` (each second's kb/s) and `_bitrate_gops.csv` (each group of pictures' frames, bytes and kb/s). Times count from the stream's first frame, as the tab's **Adjust stream start time to zero** does. |
| `--no-adjust-start` | With `--csv`: times from the file's start. |
| `--difference` | With exactly two videos: the first's bitrate minus the second's as well, as the tab's **Difference** pairs them, from each stream's start. The table ends with the averages' difference (in Mb/s and %) and the mean, lowest and highest of the seconds both have (paired by number), the JSON gets a `difference` section, and with `--csv` it is written as `<first>_minus_<second>_bitrate_frames.csv` (each of the first's frames' size in bytes minus the second's frame of that number), `_bitrate_seconds.csv` and `_bitrate_gops.csv` (kb/s over time); a cell is empty where only one video has the frame or the time. |
| `--json FILE` | Also write the results as JSON; `--json -` prints the JSON instead of the table. |
| `-q`, `--quiet` | No progress, only the results. |
| `--settings FILE`, `--no-settings` | The settings (FFmpeg's folder, the saved results' folder), as `compare` takes them. |
| `--cache-dir FOLDER`, `--no-save`, `--no-cache` | Where the saved bitrate is, and whether it is read and saved, as `compare`'s saved scores. |

The exit code is 0 when every video was read, 1 when a file could not be
written, 2 for a wrong command (a raw stream that stores no frame rate,
without `--fps`) or when ffprobe cannot be used, 4 when a video could not be
read (a pattern that finds no file, a raw YUV file, which has no packets, or
a file FFmpeg cannot read: the others are still given, and the JSON's
`error_code` says which), and 130 after Ctrl+C.

### BD-rate

`bdrate --anchor A.csv --test B.csv` calculates the BD-rate (Bjontegaard
delta rate) of a test encoder against an anchor for each sequence and
metric: how much more (+) or less (-) bitrate, in %, the test needs for the
same score, averaged over the scores both reach. Each encoder's rate curve
is the natural log of its rate against the score, interpolated with PCHIP
(a monotone cubic) and integrated exactly; Butteraugli, where lower is
better, has its scores negated first. Then the mean and median over the
sequences. It calculates no scores itself: it reads them from the two
files.

Each file is a CSV file, or a JSON file (below). A CSV file has a header
row and a row per encode. The
`sequence` and `kbps` columns are needed; a column named by a metric's key
(`vmaf`, `vmaf_neg`, `vmaf_v1`, `vmaf_hdr`, `psnr`, `ssim`, `xpsnr`, `ssimulacra2`,
`butteraugli`, `cvvdp`) holds its scores, empty where not scored. Other
columns, such as `qp` or `class`, are only read for `--group-by`.

```
sequence,qp,kbps,vmaf,psnr,ssim,cvvdp
crowd_run_1920x1080_50,22,41250.5,95.12,38.41,0.9871,9.62
crowd_run_1920x1080_50,27,18820.0,90.33,35.97,0.9763,9.31
```

A JSON file is either results as `compare --json` (a `--batch`'s too) and
Export results write them, an encode for each video, or a run file
(`.metrics.json`), the encode of its video. Each encode's sequence is its
reference's file name without its extension, its rate the test video's
bitrate, and its scores the video's. A video scored over a range, a
resolution test, failed or cancelled, or of no known bitrate is left out, as
is a row that scores another row's file another way (a film grain twin, a
"Test both" companion, CVVDP on a further display): that file's own entry is
its encode.

Sequences are paired by name; one in a single file is left out with a
warning. A cell is n/a, with a warning on stderr, where there is no
BD-rate: a metric not scored at every encode of the sequence, fewer than 2
encodes, two encodes with the same score, or score ranges that do not
overlap. Fewer than 4 encodes give a warning as well. The means and
medians are over the sequences that have a BD-rate.

| Option | Meaning |
| --- | --- |
| `--anchor FILE` | The anchor's encodes, CSV or JSON. |
| `--test FILE` | The test encoder's encodes, CSV or JSON. |
| `-m`, `--metrics LIST` | The metrics to show, or `all`. Default: those the files have columns for. |
| `--group-by COLUMN` | A mean and median for each value of this column of the anchor's file as well. |
| `--json FILE` | Also write the results as JSON, every number in full; `--json -` prints the JSON instead of the table. |
| `-q`, `--quiet` | No warnings, only the results. |

The table is in markdown: a column per metric and rows for the
mean, the median, each group's mean and median, and each sequence. The exit
code is 0 when both files were read, n/a cells or not; 1 when the JSON file
could not be written; 2 for a wrong command (a `--group-by` column the
anchor's file does not have, among them); 4 when a file cannot be read or
gives no BD-rate (no `sequence` or `kbps` column, a cell that is not a
number, no metric's scores, no sequence in both files), which `--json` then
writes as `error` and `error_code` (`rates_unreadable`, `no_common_metric`
or `no_common_sequence`).

`compare --scores-csv` writes these files. For many clips, list each
encoder's encodes in a `--batch` file and score them in one run:

```
reference,test,class
clips\crowd_run.mkv,x264\crowd_run_crf*.mkv,natural
clips\sintel.mkv,x264\sintel_crf*.mkv,animation
```

```powershell
.\VideoMetricsLab-cli.exe compare --batch x264.csv -m vmaf,psnr --scores-csv x264_scores.csv
.\VideoMetricsLab-cli.exe compare --batch x265.csv -m vmaf,psnr --scores-csv x265_scores.csv
.\VideoMetricsLab-cli.exe bdrate --anchor x264_scores.csv --test x265_scores.csv --group-by class
```

The rate is each test video's bitrate as the Videos table shows it: where the
file does not record the video's own, an estimate (the file's bitrate less
its other streams and the container's bytes, marked ≈).

### Encoding loops: target and diff

Two commands help a script that encodes, scores and encodes again.
`target` says at what rate (or CRF) each sequence's encodes reach a score;
`diff` says how a new run's scores and weak segments differ from an earlier
run's.

```powershell
# The rate, then the CRF, at which each sequence's encodes reach VMAF 95
.\VideoMetricsLab-cli.exe target --scores x264_scores.csv --metric vmaf --value 95
.\VideoMetricsLab-cli.exe target --scores x264_scores.csv --metric vmaf --value 95 --by crf --json -

# The rate at which the worst 1% of frames reach VMAF 90, from run files
.\VideoMetricsLab-cli.exe target --scores runs\*.metrics.json --metric vmaf --value 90 --stat worst_1_percent

# How the new encode's scores, and its weak segments, differ from the old one's
.\VideoMetricsLab-cli.exe diff runs\old.metrics.json runs\new.metrics.json --segments "vmaf<90"
```

`target` reads the files `bdrate` reads (`--scores`, repeated or a pattern
for more): CSV files such as `compare --scores-csv` writes, `compare --json`
results, or run files. For each sequence it reads the curve `bdrate`
integrates, the log of the rate against the score interpolated with PCHIP,
at `--value`: the rate there, and the two encodes either side. With `--by
COLUMN`, a CSV file's column of numbers takes the rate's place, as it is
rather than its log: `crf` from a `--batch` file's extra column, say.
Beyond the scores the encodes have, the straight line through the two
nearest goes on to the value, and the answer is marked as extrapolated: an
encode there, and a check of its score, is the way to be sure. `--stat`
reads another statistic of each encode than its score (`median`, `min`,
`max`, `worst_10_percent`, `worst_5_percent`, `worst_1_percent`,
`worst_0.1_percent`), from JSON files only; `--sequence NAME` keeps one
sequence. A sequence with fewer than 2 encodes with the score, or two with
the same score, has no answer. The JSON (format `target-1`) gives each
sequence's `setting`, `extrapolated`, `bracket` (the two encodes' `setting`,
`kbps` and `score`), and where there is no answer `why` and `why_code`
(`too_few_encodes`, `tied_scores` or `bad_number`). The exit code is 0 when
every sequence has an answer, extrapolated or not; 1 when one has none (or
the JSON file could not be written); 2 for a `--by` column no file has; 4
when a file cannot be read, no file has scores of the metric, or no encode
is of `--sequence` (`error_code` `rates_unreadable`, `no_metric_scores` or
`no_sequence`).

`diff OLD NEW` takes two runs, each a run file or a saved comparison's ID
(at least 6 of its hex digits, as `saved list` gives them), looked for in
the settings' saved results folder (`--settings` and `--no-settings` choose
the settings, `--cache-dir FOLDER` another folder, as `saved` takes them). It shows each
metric both runs have, old, new and the change, for the score and the worst
1%; its JSON (format `diff-1`) has every statistic, and the metrics only
one run has. With `--segments` (and its options, as compare takes them), it
finds both runs' weak segments and matches them by their frames: an old
segment that no new one overlaps is fixed, overlapping ones remain
(chained: an old segment overlapping two new ones joins them), and a new one
that overlaps no old one appeared. The JSON's `segments` has both runs'
segments, as a segments file has them, and `fixed`, `remaining` and
`appeared` as indexes into them. Frames are matched by number, so the runs
should be of the same reference (a warning says when their references
differ). The exit code is 0; 1 when the JSON file could not be written; 2
for a wrong command; 4 when a run cannot be read (`error_code`
`run_unreadable`).

## Saved results and caches

Save portable results as `.metrics.json` (a right-click on the Videos card's
save icon, **Run file (.metrics.json)...**) or export CSV. For the scores
alone, the Videos card's save icon saves the selected videos' results (or
every scored video's) at one click; or tick **Results**, and **JSON** or
**CSV**, in the tab bar's **Export** panel (see [Saving and
exporting](#saving-and-exporting)): either gives the same file,
every video's score and statistics for each metric (those of every metric,
not only the one Metric Graphs shows), in the shape the command line's
`--json` writes. Both also record how the scores were made: the run, the
machine (system, CPU, GPUs and their drivers, FFmpeg), and for each video
its file, every setting (defaults too), its black bars and how they were
found, how each metric came out, the decoders used and how long each pass
took. [The results file](RESULTS_JSON.md) describes every key.

### The Saved Results tab

**Saved Results** lists every comparison whose scores the app has saved (its
results cache, Settings > Storage > Saved results), newest first, with its
test video, reference, when it was saved, its metrics and its size; **Filter**
narrows it by file name. Select one or more to use them without adding the
videos again:

- **Open in Metric Graphs** (or double-click, or the first entry of the
  right-click menu) adds them as lines of their own, named "*video*
  (saved)".
- The save icon at the list's top right saves the selected comparisons'
  results at one click. Right-click the selected comparisons for **Export
  this video's...** (one selected) or **Export the selected videos'...**
  (several), what the tab bar's Export panel exports (see [Saving and
  exporting](#saving-and-exporting)); below it, and on a right-click on the
  save icon, each kind on its own, as the Videos tab's offers them:
  - **Results as JSON...** or **as CSV...**, as Export results saves them;
  - **Per-frame scores as JSON...** or **as CSV...**: a file per
    comparison to a folder (the JSON is the CSV's table: see [Per-frame
    scores as JSON](RESULTS_JSON.md#per-frame-scores-as-json));
  - **Graphs as ZIP...**: the graphs the Videos tab's Graphs as ZIP saves, for
    comparisons of one reference. The bitrate graphs of a video no longer on
    disk, or replaced since by another file of the same name (a new
    encode), come from its saved bitrate, where the Bitrate Viewer or the
    command line read it before;
  - **Weak segments as JSON or CSV...**: the weak segments report, by the
    rules the Export panel's **Weak segments by** picks: the profile in use,
    any profile or a preset. The archive's weak segments use the same rules.
    Weak segments are not saved with the scores; they are found from them as
    they are exported;
  - **Run file (.metrics.json)...**: each comparison as a run file, which
    **Import run file...** reads back, as the Videos tab's.
- **Delete...** removes their saved scores, after asking: the videos are
  scored again the next time they are compared. While metrics are
  calculated, it is locked for the comparisons the run is calculating (its
  reference with each of its test videos, also where one was saved under a
  file's old name or folder), whose scores the run reads and saves; others
  can still be deleted, those of the same reference too.

Drag the line between two column headings to make a column wider or
narrower (double-click it to fit the column to its text); the last column
shown fills what is left. Until a column is resized, the columns fit the
tab as the window grows and shrinks. Right-click a heading for the list of
columns: untick one to hide it, tick it to show it again (the test video's
is always shown). **Reset column widths**, under the list, fits them to the
tab again. The widths and the hidden columns are kept for the next time the
app opens.

A comparison whose reference video is no longer where its scores were saved
from (deleted, moved, unreachable, or another file there now) shows its
reference as "*name* (not found)", in italics: its saved scores are the only
ones there are, and could not be calculated again without it. **Delete...**
says so when it would delete any such, and **Settings > Storage > Clear saved
results** keeps them, with their videos' saved bitrate, saying how many it
kept; delete them here.

**Saved scores** shows the selected comparison's scores: each metric's score
for the whole video, as Metric Graphs' table shows it, and when and how it was
made (GPU or CPU, the program, the VMAF model by its version, the CVVDP
display). In VML Modern each metric is a tile, five to a row, its score in
large type; in VML Classic a line each, two metrics a row.
Where a metric has more than one (CVVDP for two displays; SSIMULACRA2 on the
GPU and the CPU), a list picks one: the newest is used unless another is
picked (in VML Modern the tile's score follows the pick). A score
the Videos tab would not use, and would calculate again (one made by an
older version that read these videos wrongly, say), is marked "out of date"
and is not used: not opened, exported or offered to pick.

### Saving from the Videos tab

The save icon at the test videos card's top right saves the selected test
videos' results (or every scored video's, with none selected) at one click,
in the formats Settings > Storage > **Quick saves** ticks. A right-click on
it acts on the same videos, each kind with the Save dialog:

- **Results as JSON...** / **as CSV...** and **Per-frame scores as
  JSON...** / **as CSV...**, as Saved Results' save icon offers them;
- **Graphs as ZIP...** (below);
- **Weak segments as JSON or CSV...**, by the profile in use;
- **Run file (.metrics.json)...**: each video's whole run, which **Import
  run file...** reads back, on this PC or another. One is saved where
  you say; several into a folder, each named after its video.

The table's right-click menu has the same entries under its own export
(**Export this video's...** or **Export the selected videos'...**).

Metric Graphs has its own tab: the Videos tab no longer has a button to open
it. A video removed from the graph with its ✕ stays away when you come back
to the tab; **Show removed videos** there brings back those of the Videos
tab.

### Exporting graphs

**Graphs as ZIP...** (a right-click on the Videos card's save icon) saves the graphs of the
test videos scored against the reference (the selected ones, or all) as one
`.zip` file of PNG images, drawn as Metric Graphs' and the Bitrate Viewer's **Graph as PNG...** draw
them:

- each test video's graph of every metric it has: `<video> - <metric>.png`;
- with two or more test videos, every metric with all of them together:
  `All test videos - <metric>.png`;
- the bitrate (second based) of the reference with all of the test videos,
  `All test videos - Bitrate.png`, and of the reference with each one,
  `<video> - Bitrate.png`.

The files are named after the test videos; where two share a name, each gets
its folder's name after it, in brackets. What the tabs show (the lines ticked,
the page, the zoom) is left as it is: the graphs are drawn apart from them.
The Bitrate Viewer reads each video's bitrate after its run; if it is still
reading, or a video's bitrate was never read (its scores came from saved
results, say), the indicator says "Saving..." (its tooltip: the bitrate
graphs are still being built), and the file is saved as soon as they are. The files are read where they are
now, even when their saved scores were made before they were moved, renamed
or copied.

### Saving and exporting

There are two ways to save, and one place that says how it went:

- **What a card shows, at one click:** each table's, list's and graph's
  card has a save icon in its top-right corner. It saves what the card
  shows into the export folder (Settings > Storage > **Export folder**,
  Documents while blank), in the formats **Quick saves** ticks (JSON
  and/or CSV; graphs are always PNG), never over a file already there.
  Its tooltip says what it will save, or, grayed, why it can't. A
  right-click offers the same, one at a time, with the Save dialog.
- **Everything for these videos:** **Export**, at the tab bar's right end,
  opens a panel: which videos (the open tab's selection, on Videos, Metric
  Graphs' statistics or Saved Results, or all scored videos), what to
  include, the weak segment rules, and the folder (**Change...**). Its
  ticks are Settings' **Exports include**, below: a change in either is
  the other's. **On this tab** is the open tab's one-click save.
- **How it went:** beside Export, "Saved 2 files · Open folder" once
  written (it fades), or "Not saved: ... · Details" where it failed. No
  window opens to say an export worked.

**Settings > Storage > Exports include** chooses what every export holds,
by the Export panel, the right-click menus or after each run. Tick:

- **Results**: scores and statistics, as the
  results file the command line's `--json` writes: a video's alone
  (`<video> - results.json`, `.csv`), or every video's in one file;
- **Per-frame scores**: every frame's score of each metric, a file for each
  video (`<video> - frames.json`, `.csv`). Not ticked until you tick it: a
  long video's are big;
- **Graphs**: each video's graph of every metric and its bitrate with the
  reference's (`<video> - <metric>.png`), with several videos all of them
  together too, as **Graphs as ZIP...** draws them, once the bitrate is
  read;
- **Worst frames**: each video's worst frames of every metric, as many
  dips as the Worst Frames tab lists (10, 25 or 50), each with every
  metric's score there (`<video> - worst frames.json`, `.csv`; see
  [RESULTS_JSON.md](RESULTS_JSON.md#worst-frames-as-json-and-csv));
- **Weak segments**: by the profile in use: a video's
  (`<video>.<metrics>.segments.json`, `.csv`), or every video's in one
  report. Nothing while the profile has no condition.

**In** chooses the formats results, per-frame scores, worst frames and weak
segments are written in:
**JSON**, **CSV** or both. With neither, only graphs are exported.

**As** saves them as **A .zip file** (`<video> - VideoMetricsLab.zip`, or
named after the reference for every video), or as **Loose files** in a
folder you choose. With several videos, each video's own files (its graphs,
per-frame scores and worst frames) are in a folder of its name, and the
files of all of them together (the results, the weak segments report, the
"All test videos" graphs) beside those folders.

The right-click menus of the Videos table, of Metric Graphs' statistics and
of the Saved Results tab offer the same export, with a Save dialog, named
after what is ticked: **Export this video's results, graphs and weak
segments...** for the video clicked (once its metrics are all scored, or
failed: a metric that failed leaves the others' scores to export), and
**Export the selected videos'...** for the selected rows: on Saved Results,
and in the Videos table where you right-click one of several rows
selected. Every scored video's is the Export panel's. Untick
what you never want, CSV say, and it's left out of every export. A line of a run file imported with **Import run file...** has no
export of its own (it has no row in the Videos table), but is in all the
videos' results.

**Settings > Storage > After each run > Export each video** exports each
video as Exports include say once its run ends: **Save to** puts it next to
the test video, or in the export folder (next to the video while none is
set). No file is replaced: where a name is taken, the new files are
numbered. Settings from before these
choices carry over: results and graphs in a .zip, and weak segments as loose
files where they were exported alone.

### Weak segment profiles

What counts as a weak segment depends on the video: animation, grainy film
and HDR call for different metrics and thresholds. **Settings > Weak
segments** keeps them as named profiles. **Profile** picks the one in use,
which the rules below it edit; **New...** starts a profile from the one
shown, **Rename...** and **Delete** work on the one in use (**Default**, the
rules kept before there were profiles, has neither). A profile's name can't
contain a comma, which separates `--segments`' parts on the command line. The profile in use is
where the Worst Frames tab's conditions start, and what the weak segments
report, the exports after each run and the command line's `--segments saved`
follow; `--segments profile:NAME` takes another.

**From preset...** loads a starting point for a kind of video into the
profile in use, replacing its rules, to adjust as needed; it adds no
profile. To keep the rules shown as well, make a profile with **New...**
first, then load the preset into it. The Worst Frames tab has **From
preset** too, which loads a preset into its conditions only, to save from
there if you keep it. (The command line takes
a preset as `--segments preset:KEY`.) Any condition makes a frame weak:

| Preset (`KEY`) | Conditions | Shortest segment | Join gap | For |
| --- | --- | --- | --- | --- |
| Animation (`animation`) | VMAF < 93, SSIMULACRA2 < 75 | 3 | 6 | Flat colour and sharp lines, which encode well and score high: banding and damaged lines. |
| CGI (`cgi`) | VMAF < 92, SSIMULACRA2 < 70 | 3 | 6 | Clean rendered detail and smooth gradients. |
| Grain-free digital (`digital`) | VMAF < 90, SSIMULACRA2 < 65 | 3 | 6 | Clean live action: the usual thresholds of a high-quality encode. |
| Grainy film (`grainy-film`) | VMAF < 85, SSIMULACRA2 < 55 | 6 | 12 | Grain lowers the scores of frames whose look is kept, so lower thresholds, and grain alone is not weak. |
| HDR (`hdr`) | CVVDP < 9 JOD | 1 | 24 | CVVDP models HDR displays; VMAF's models are for SDR video. CVVDP must be scored. |
| Low-bitrate preview (`preview`) | VMAF < 70, SSIMULACRA2 < 40 | 12 | 24 | Only severe breakdowns, for encodes not meant to look like the source. |

The thresholds are starting points on each metric's scale: VMAF's 0-100
(90 and above is a good encode), SSIMULACRA2's (70 and above high quality,
50 medium, 30 low), CVVDP's JOD (10: no visible difference).

### Exporting weak segments

The weak segments exports follow the profile in use (or, from the Worst
frames tab, the conditions shown there), and are named after the metrics
their conditions cover. A condition on a metric a video was not scored with
is left out for that video, as in the tab and on the command line, also
when every condition must hold; the JSON lists it under `left_out`.

- With **Weak segments** ticked in **Settings > Storage > Exports
  include**, the Export panel's all scored videos saves every test video
  scored against the reference, by the rules its **Weak segments by**
  picks (the profile in use until another is), in one report, as JSON and
  CSV.
  The JSON gives the conditions and options, then each video's `name`,
  `path`, `reference`, its segments in short and `segments`; the CSV is a row
  per segment, led by its video and reference. With strict conditions it can be far smaller than the
  results file; with loose ones, on a film, far larger.
- **Settings > Storage > After each run** saves each video's segments as
  `<video>.<metrics>.segments.json` and `.csv` once its run ends, with the
  rest that Exports include (see [Saving and
  exporting](#saving-and-exporting)). Nothing is saved while no
  condition is set. A profile with no condition on the video's metrics
  takes one on the metric shown, as the Worst Frames tab starts, and its
  own are listed as left out.

Every weak segments JSON tells, as the Worst Frames tab shows them:

- for each segment, besides its frames, times and each condition's metric's
  `min`, `max` and `mean` there: `weak_by`, the conditions that hold on any
  of its frames (the tab's **Weak by** column), and `condition_frames`, how
  many of its frames each holds on;
- for each video: `segment_count`, `weak_frames` (the segments' frames added
  up), `weak_seconds` (their lengths added up), `weak_percent` (of the
  frames compared, or of those in the highlighted range), and
  `condition_counts`, for each condition the frames it `holds` on and how
  many of them are `in_segments`: fewer where a run was shorter than the
  shortest segment, or, with every condition, where another did not hold.

The command line's `--json` gives the same in each video's `weak_segments`,
and its tables say each condition's frames and, with several conditions any
of which makes a frame weak, a **Weak by** column.

The command line does the same with `--segments saved`, `--segments-dir`
and `--segments-report` (see [Command line](#command-line)).

Cached results are reused when files and relevant settings match. A
compatible subset can load as partially calculated; request calculation to
fill missing metrics.

Saved scores follow the files, not their names or folders. A file renamed,
moved or copied is recognised by its content: its size and samples of its
bytes from the start, the end and between (a few megabytes read). A
reference remuxed into another container (MKV, MP4, with any other tracks)
is recognised by its video stream: its codec, picture size, frame rate,
colour and length, and five pictures decoded from it. The reference's
stream is only looked at when the test video has saved scores, so a new
encode costs nothing to look up. A remux that changes the stream's colour
tags, or a re-encode, is another video and is scored again. Scores found
this way belong to the files where they are now: Video Compare plays those,
and a run file exported from the row names them.

With **Settings > Storage > Recognize test videos remuxed after scoring**
(off by default; the command line's `--test-remuxes on`), a test video
remuxed (its raw stream, `.264`, `.265`, `.266` or `.obu`, muxed into MKV or
MP4 with its audio, an MKV remuxed to MP4, a title tag edited) keeps its scores
too, with the same reference and settings, read at the same frame rate. It
is recognised by its coded pictures, byte for byte: any re-encode, even of
one scene, is another video, and so is an MP4 cut with an edit list that hides
its first pictures, against a remux that shows them. That takes reading each test video once more
after its scores are saved, as long as reading the file takes (about 45 s
for a 4 GB film on a network share):
- the app reads it in the background, saying so on the status line, and
  never during a run (a run pauses it; closing the app stops it, and it is
  read at the next save);
- the command line reads it after writing its results, saying so.

Looking a new encode up reads only its first packets. A video whose first
packets and reference match a saved comparison's (a remux, or an encode
redone in later scenes only) is read whole to tell which: in the app after
the other videos' saved scores are shown, on the status line, and given up
when a run starts (the video is calculated); on the command line before
the run, said first. A test video whose colour tags were changed is read
with another colour and is scored again, as a reference is. A test video
recorded by an older version is read once more, as after a save, before a
remux of it is recognised. Off, nothing reads test streams, and test videos are
recognised by their files only.

Settings and the default cache are under `~/.videometricslab/`. Back up that
folder before maintenance; changing checkout should not require clearing it.
A file replaced by another causes a cache miss. Use explicit saved-result
loading when you need to inspect a portable result. Right-click recalculation
forgets matching cached results, so do not use it just to refresh the display.

Result files may contain absolute media paths, and exported results the
machine's CPU and GPUs. Review them before sharing.
See [troubleshooting](TROUBLESHOOTING.md) for missing results or playback errors.
