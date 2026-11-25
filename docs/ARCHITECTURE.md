# Architecture

## Boundaries

The app is C++ (`cpp`), in four parts, each a CMake target:

| Part | What it holds |
| --- | --- |
| `vml_platform` (`cpp/platform`) | What differs between systems: starting programs and their pipes, loading libraries, the console, the PATH, the GPUs. Its headers are the same on every system. |
| `vml_core` (`cpp/core`) | The engine: reading videos, planning and running the metrics, the saved scores and saved runs, statistics. |
| `vml_cli` (`cpp/cli`) | The command line, `vml-cli` (`VideoMetricsLab-cli.exe` in the package). |
| `vml_app` (`cpp/app`) | The window, `VideoMetricsLab` (Qt 6), with Video Compare's native playback, which only the window uses. |

The window and the command line use the engine; the engine uses neither,
and links no Qt and no operating-system headers, so the same code runs under
both and on every system. CMake holds that boundary: `vml_core` links only
`vml_platform` and its vendored libraries (`cpp/third_party`).

The native libraries in `native` (C++ with a C API, built by the
`scripts/build_*.ps1` into `resources/native`) are loaded when first needed:
the GPU frame decoders and the software decoder (`native/gpu_frames.h`) and
Video Compare's Direct3D 11 presenter (`d3d11_tonemap.dll`). The bundled
libvmaf, Vship and FFmpeg's libraries are loaded the same way, from
`resources/tools`.

| Concern | Main files (`cpp/core/src` unless named) |
| --- | --- |
| Media metadata and geometry | `ffprobe.cpp`, `video.cpp`, `geometry.cpp`, `crop_detect.cpp`, `model_select.cpp` |
| Metric requests and planning | `analysis_request.cpp`, `options.cpp`, `job_runner.cpp` (`plan_tasks`, `run_jobs`, `JobScheduler`) |
| FFmpeg's runs | `ffmpeg_run.cpp`, `ffmpeg_tools.cpp`, `hardware_decoding.cpp` |
| What each GPU maker's decoder can do | `gpu_decoders.cpp`: one row per decoder, which features ask rather than compare a maker's name |
| A range late in a video | `range_seek.cpp`, `plan_range_seek` in `gpu_frames.cpp`: seeks only where the file proves its frames' numbers (`VML_RANGE_SEEK=0` turns it off) |
| Film grain | `film_grain_detect.cpp` (what a stream describes), `film_grain_scoring.cpp` (scored with or without it), `grain_feed.cpp` (a video's pictures decoded by the app and fed to FFmpeg); `native/grain/grain_decoding.h` (which decoder adds it) |
| Metrics calculated in the app | `libvmaf.cpp` (the bundled libvmaf), `libvmaf_cpu.cpp` (PSNR, SSIM, XPSNR), `gpu_vmaf.cpp`, `vmaf_cuda.cpp`, `vmaf_vulkan.cpp`, `vmaf_v1_gpu.cpp` (VMAF on the GPU), `gpu_frames.cpp` (decoding), `isolated.cpp` (the process of its own they run in) |
| SSIMULACRA2, Butteraugli, CVVDP | `vship.cpp`, `vship_run.cpp` (on the GPU), `perceptual_cpu.cpp`, `cpu_pair_scorer.cpp`, `ssimulacra2.cpp`, `butteraugli.cpp` (on the CPU, the app's own engines), `perceptual.cpp`, `cvvdp.cpp` |
| Metric registry | `metrics.cpp` |
| Results and persistence | `results.cpp`, `metric_cache.cpp`, `run_file.cpp`, `result_json.cpp`, `export_csv.cpp`, `settings.cpp` |
| Statistics and graphs | `stats.cpp`; `app/graph_panel.cpp`, `app/chart.cpp`, `app/export_image.cpp` (Graph as PNG) |
| The window's runs | `app/run_worker.cpp`, `app/main_window_run.cpp`, `run_line.cpp` |
| Still comparison | `frame_extract.cpp`, `app/frame_compare_panel.cpp` |
| Native playback | `native_playback.cpp`, `soundtrack_clock.cpp`; `app/native_pool.cpp`, `native_decoder.cpp`, `native_device.cpp`, `native_presentation.cpp`, `native_soundtrack.cpp`, `native_dav1d.cpp` |
| FFmpeg playback fallback | `video_playback.cpp`, `app/playback_worker.cpp` |
| Bitrate analysis | `bitrate.cpp`, `app/bitrate_panel.cpp` |
| Processes and cancelling | `cpp/platform` (`process.h`), `cancel.cpp` |

## Metric lifecycle

1. The window reads the reference and the test videos off its thread
   (`app/probe_worker.cpp`, `ffprobe.cpp`) and checks they can be compared.
2. A run takes each row's options as they are into a `VmafJob`
   (`job_runner.h`): the two videos, the options, the metrics asked for, and
   the tasks that calculate them (`plan_tasks`). A task is the part of a
   video's metrics one program calculates in one place: FFmpeg's metrics on
   the CPU (PSNR, SSIM and XPSNR in the app where it decodes the videos
   itself), VMAF, NEG and v1 on the GPU, Vship's metrics on the GPU, and
   SSIMULACRA2 and Butteraugli set to the CPU, by the app's engines. Where
   the app decodes a video on the CPU, or both on a GPU whose pictures reach
   them as in their own run (`pass_feeds_cpu_metrics`), PSNR, SSIM and XPSNR
   join the GPU's VMAF task (`MetricTask::cpu_beside`) and are scored from
   its frames.
3. Saved scores are looked up by the request's identity (`metric_cache.h`):
   the comparison recipe and each metric's request, not where or how it is
   calculated. A saved score fills in its metric, and a task whose metrics
   are all saved is left out.
4. The CPU's queue runs two tasks at once at most (the parallel jobs
   setting), the GPU's one. GPU work runs in a process of its own
   (`isolated.h`): a crash in libvmaf, Vship or the GPU's driver ends that
   process, not the run, and FFmpeg's filters then calculate VMAF and the
   others on the CPU, and the app's engines SSIMULACRA2 and Butteraugli.
5. Each task's scores come back as a `MetricResultSet` (`results.h`): each
   metric's frame or sequence result, with its provenance.
6. The window shows each result as it arrives, saves its scores off its
   thread (`app/file_writes.cpp`), and draws the graphs.

A run works on its own copy of each row's options: a row edited while it
runs changes the next run, not this one. A finished result is attached only
to the row, source and settings it was calculated for.

### Adding a metric

**The registry entry is the only source** (`metrics.cpp`): its key, labels,
column header, unit, direction and bands, and in `guides` the few words
beside its name and its tooltip text (`metric_guide`). Add/remove metrics,
Settings > Metrics' Default metrics (`MetricList`) and the column header's
tooltip (`metric_tip`) read them, and the user guide says the same; nothing
is written twice.

**Its section and first state come from one list.** A new metric is
Specialized unless the maintainer agrees it is general purpose. Specialized
means in `metrics_hidden_at_first`, which makes it start ticked but hidden,
in new settings and in saved ones (`hide_new_metrics`, by `listed_metrics`),
so showing it is enough to calculate it. A general-purpose metric starts
shown, ticked by its `default_compute_<key>`: unticked where most videos
would fail it, as HDR VMAF is (PQ videos only).

**Ranges only with a source:** the app's own calibrated bands (its
thresholds) or the metric's authors' published scale, with the source named
in `ranges_source` and in a comment beside the numbers. Without one, the
tooltip says what it measures, its unit and which way is better, and gives
no ranges. The line on grainy film against clean content, the lists'
footer, applies to every metric.

**A plane-weighted (YUV) metric** also gets its Y, U and V entries
(`plane_of`), shown with *Y, U and V of the YUV metrics*: no tick of their
own, outside the count and Settings' default metrics, their headers'
tooltips from their metric's text.

**The command line by default:** its `-m` key and its place in `all` (or
why it is left out, as HDR VMAF is), `--help`'s text, and its JSON key in
`docs/RESULTS_JSON.md`, in the same change.

**Everywhere else it shows,** in the same change: exports, Metric Graphs
and Worst Frames, Weak segments' conditions (its thresholds), BD-rate, and
its text in all 20 catalogs (`scripts/i18n_catalog.py check`).

`Metrics.EveryMetricFollowsTheRulesForAddingOne` fails on a metric without
its few words or tooltip text, outside the sections or in one that doesn't
match `metrics_hidden_at_first`, with ranges and no `ranges_source` (or a
source and no ranges), or with bands and no ranges.

## Persistence contracts

The C++ app reads and writes the Python app's files, so a user's saved
scores, saved runs and settings carry over from earlier versions, and either
app opens the other's. (SSIMULACRA2, Butteraugli and CVVDP scores saved on
the GPU by earlier versions, with Vship 5.1, are kept for videos set to stock
Vship (the window's "Vship Vulkan", "Vship HIP" and "Vship CUDA", which share their scores), as its own; videos set to the
app's own Vship ("GPU"), whose scores differ a little, calculate theirs. Those of a pair a later colour fix reads
otherwise are calculated again: score_identity.h.)

Saved runs are `.metrics.json` files (format v2, `run_file.h`): each metric
serialized on its own, with its kind, its frame or time axis where it has
one, its values and its provenance, so that metrics sampled differently
round-trip without being flattened into one table. A missing frame value is
JSON null.

The saved scores are kept per metric:

```text
<cache>/v2/<sha256 recipe hash>/context.json
<cache>/v2/<sha256 recipe hash>/<metric>_<sha256 request hash>.npz
```

The recipe hash takes both files' identities (full path, size and
modification time: `file_identity`) and the comparison recipe; the request
hash the metric's key, its scientific parameters, its frame coverage and its
implementation compatibility ID. Where and how a metric is calculated (GPU
or CPU, the GPU's maker, threads) is left out, so a score calculated one way
stands for the other wherever the implementations declare themselves the
same. A damaged file is a miss for its metric only.

Each score's provenance says which build of the app calculated it
(`made_by`, "2.4 (0123456789ab)": the version and the commit,
`vml/build_info.h`), kept as it was when the score is saved again; a
`context.json`, a saved run and the JSON exports say which build saved them
(`saved_by`, `commit`). They are history only: no lookup reads them, so an
older app reads these files as before.

User data lives under `~/.videometricslab/` (settings, the saved scores by
default, the log), whatever folder the app runs from.

## Playback and threading

Video Compare plays a pair natively where it can: one decoder per video
(the source, and the selected test video with its neighbours), each test
frame paired with the source frame of the same number before the one
presenter draws either, in time with the source's soundtrack
(`app/native_pool.h`). Where native playback cannot play a pair, the frames
come from `ffmpeg.exe`. HDR shows as HDR on a display in HDR mode, and is
tone mapped to SDR on any other, or when the user chooses HDR to SDR.

Work that takes time (reading videos, runs, file writes, playback's decoding)
runs off the window's thread and reports with queued signals, so the window
sees events on its own thread, in the order they were sent. File writes run
on one thread of their own, in the order they were asked for, and closing
the window ends the run and waits for them.

## The window's interfaces

The window has two interfaces, chosen in Settings > Window > Interface
(`app/theme.h`):

- **VML Classic:** the system's style (windows11 on Windows, Fusion on
  macOS) with its own palette, and no style sheet of the app's own.
- **VML Modern:** Fusion, with a palette and one application style sheet
  made from Modern's colours and the accent.

Both follow light and dark. A panel is laid out once, for both. What
differs is drawn by the style sheet, from a role a widget is given with
`theme::set_role`:

| Role | For | In Modern |
| --- | --- | --- |
| `primary` | the one action a tab is for (Calculate metrics, Play) | the accent |
| `danger` | an action that removes something | the danger colour |
| `quiet` | a lesser action, read as text | no frame |
| `card` | a box of controls that belong together | a card, as group boxes are |
| `tile` | a figure in a card, such as a saved score | a raised, rounded tile |
| `toolbar` | a strip of controls above or below a view | a raised strip |
| `separator` | a line between a toolbar's groups (`theme::separator()`) | a line of the border colour |
| `segmented` | a tab bar of a few choices, and its tab widget (Settings' sections, Metric Graphs' metrics) | a row of pills, the current one in the accent |
| `notice` | a label set apart: what a run locks, say | a tinted strip |
| `unboxed` | a group box only Classic frames (`theme::classic_box`) | no box, no title: its cards are the boxes |
| `footer` | a strip of buttons along a card's bottom | a strip of the card's alternate colour |
| `title`, `hint` | a card's title and the line beside it | semibold; muted |
| `choice` | a frame picked as a whole, with a `checked` property (Settings' interface cards) | a raised card, the one picked tinted and edged in the accent |
| `segments` | a row of checkable buttons, one of a few (Settings' Theme) | a sunken strip, the one picked raised |

Classic draws every role alike, as the system draws the widget.

VML Modern on Windows draws the window's title bar itself (`app/title_bar.h`,
`TitleBar`, the window's menu widget). Qt's `Qt::ExpandedClientAreaHint`,
asked for with no title or buttons of Qt's own, gives the app the window's
area up to its top edge while the system's frame stays: its shadow, rounded
corners and resize edges. `TitleBar::attach` gives back the window styles
Windows' own behaviours look for (double-click to maximize, snapping, the
taskbar's minimize), and the window's `nativeEvent` hands its messages to
`TitleBar::native_event`. That answers `WM_NCHITTEST` for every point of the
window: the edges resize (unless maximized), the maximize button is
`HTMAXBUTTON` (so Windows 11 shows its snap layouts over it, and the button's
hover comes as non-client messages), the bar's other buttons are client, the
rest of the bar is caption, and the window's contents are client. It must
answer everywhere: left to Qt, `QWindowsWindow::handleNonClientHitTest`
makes a mouse press or release of its own for such a window whenever the
buttons' state has changed (Qt 6.12, QTBUG-152532), so every click came twice, a button's
menu closed as it opened and double clicks went unseen. It also shows the system menu for
Alt+Space, the icon and a right-click, with Minimize and Maximize enabled
(Qt's own menu for Alt+Space would have them off). The bar is made when the
window is built, so it follows the interface the app started in, and not in
high-contrast mode, where the system's colours belong. Its three buttons
(`TitleButtons`: a run's progress, the theme, an update) are VML Modern's on
every system: in the bar, or at the tab bar's right end where there is none,
before the export indicator and the Export button, which is the tab bar's
last in both interfaces. VML Classic has the progress button alone, there
too. It is
always there; what it shows (Ready, the run's progress, Done or Cancelled
until the ticks change) is `RunBadge`'s (`app/run_badge.h`).

When you add to the window:

- **Give a new control its role where one fits.** Most controls need none.
  A tab has at most one `primary`.
- **A role is one of the table's.** A new one needs its rule in
  `modern_style_sheet` first. `vml_appearance_tests` fails on a role the
  style sheet doesn't style: misspelt, or added without its rule.
- **A hint's slant is `{hint_style}`** in a `theme::style` template: italic
  in Classic, upright in Modern. A widget whose own sheet only Classic
  should draw uses `theme::style_classic`.
- **Classic stays as it is.** Layout that is Modern's alone is built with
  `theme::container` (a card, a toolbar or a footer whose margins are
  Modern's only), `theme::modern_only` and `theme::classic_only` (a widget
  one interface shows), and `theme::classic_box` (a group box Classic frames
  and titles, Modern leaves unboxed). Each follows the interface applied.
- **Colour nothing in a panel by hand.** A colour comes from the palette, from
  `theme::color`, or from a role. Anything else shows in one interface and not
  the other.
- **Look at the change in both interfaces**, light and dark.
  `vml_appearance_tests` saves every tab in each when `VML_UI_SNAPSHOTS` names
  a folder (`CONTRIBUTING.md`).

### Buttons that open a panel

A button that opens a menu or a panel behaves as a Qt button's menu does,
whichever it opens. A menu set with `QPushButton::setMenu` or
`QToolButton::setMenu` does all of this already (the title bar's theme
button). A panel is a `PopupPanel` (`app/popup_panel.h`; Add/remove metrics'
`MetricsPanel`, the tab bar's `ExportPanel`), which does it given its button:

- **It opens on the press**, not on the click's release a whole press later,
  which reads as lag (130-140 ms against a menu's 28, measured).
  `PopupPanel::open_on_press` connects the button's `pressed`, which a mouse
  press, Space and a `click()` (UI Automation's Invoke) all emit, each
  opening the panel once. Not `clicked`.
- **The button is drawn pressed while its panel is open**, and up once it
  closes, however it opened (a press, Space, `click()`, a right-click
  elsewhere) and however it closed. `PopupPanel::set_anchor` gives the panel
  its button, which it holds down from its show to its hide. Drawn up at the
  press instead, the button flashed pressed for as long as the panel took to
  build.
- **It closes on a second press on its button, a press outside it, or Esc.**
  The second press closes it without reaching the button, which would open it
  again at once: the panel asks Qt not to replay it (`Qt::WA_NoMouseReplay`,
  `set_anchor` again), and since Qt replays it anyway once any popup has been
  closed with Esc, later and with the press's own time, the button's filter
  from `open_on_press` drops a press of that time. A press anywhere else
  closes it and still lands (a tab clicked is switched to).

A new button that opens a panel takes all three: a `PopupPanel` given its
button with `set_anchor`, opened with `open_on_press`. Not a `QDialog`, not a
popup of its own, not `clicked`. `MetricsPanelInWindow.APressOpensThePanelAtOnce`
checks each such button; add a new one to it.

## Asking and telling the user

A window that stops the user is the last way to tell or ask them anything,
not the first. Each rule here is one a reviewer can check.

- **What happened is told where it happened.** A result, a note or a
  failure goes beside what the user did: a card's line (the Videos footer's
  note), a row's ⚠ and tooltip, the tab bar's indicator (exports,
  `ExportIndicator`). No window opens to say something worked.
- **A line in the tab bar's indicator** (`ExportIndicator`: an export's
  outcome, `show_note`'s line of what a click did) goes through that one
  widget, never a label of its own beside it. It fades by its opacity and
  then hides keeping its room (`retainSizeWhenHidden`), so the tab bar's
  corner is never laid out again as it goes: laid out again, the buttons
  beside it were drawn over for a frame, a flash where the line had just
  faded. A failure stays until clicked; a note never replaces one.
- **An action that can't run yet is disabled**, its tooltip saying why, rather
  than enabled and then refusing in a window.
- **A destructive action that can be redone is clicked twice**
  (`ClickTwice`, `app/click_twice.h`): the first click arms the button in the
  danger colour and puts a line beside it saying what the second click does;
  the second, while armed, does it (Remove all, Start over). Saved scores
  are kept by such actions, so they can be had again.
- **A modal only where the action can't go on without an answer:** a
  decision partway through an action (Calculate the rest?), closing the
  window during a run, a form whose values the action needs (the frame rate
  of a raw stream), and what can't be undone at all (Clear saved results,
  which deletes every saved score: a `danger` message, its headline in the
  danger colour, Cancel Enter's answer). And the system's own file dialogs.
- **Every modal goes through `Dialogs`** (`app/dialogs.h`), which the
  window's tests answer, and is built on `AppDialog` (`app/app_dialog.h`): in
  VML Modern on Windows the app's own title bar (`TitleBar`, as the main
  window's), elsewhere the system's; a message's icon drawn by the app in
  the theme's colours (`message_icon`), never the system's. Not
  `QMessageBox`, `QInputDialog`, `QProgressDialog`, `QColorDialog` or a bare
  `QDialog`: `show_message`, `ask_line`, `ProgressDialog`, `choose_color` and
  `AppDialog` take their places. `QFileDialog` is called in `dialogs.cpp`
  alone. `DialogRules.EveryDialogIsTheAppsOwn` fails on any other.

## Removing and deleting

What a control takes away decides how it looks and what it asks. "Remove"
takes something out of the window; "Delete" takes files off the disk. A
control's words, colour and question follow from which of the three it is:

- **Takes a row out of a view** (a line from Metric Graphs, a video from
  the Videos table): nothing is lost; its scores are saved and it can be
  added again. Its control is the circled ✕ that starts the row
  (`RemoveMarkDelegate`, `app/widgets.h`), drawn in the text's colour,
  heavier under the pointer, never in the danger colour; one click removes
  it, with no question. Remove selected, under the Videos table, is the same
  for the rows selected. Beside a video's ✕, its send mark (an arrow out of
  a box) puts its line back in Metric Graphs; a click that has nothing to
  do says why in the tab bar's indicator (`ExportIndicator::show_note`),
  never in a window.
- **Clears the session** (Remove all, Start over): everything in the window
  at once, still nothing on disk. The `danger` role, clicked twice
  (`ClickTwice`, "Asking and telling the user").
- **Deletes files** (saved results): what is deleted must be scored again,
  and some can't be (their reference video is gone). The `danger` role on
  every such button, and a `danger` message before it runs, Cancel Enter's
  answer.

A control locked during a run says so in its tooltip and is drawn disabled,
the row's ✕ with Remove selected.

## Exports

The window offers two kinds of export, through two doors (issue #224), and
says how each went in one place. A new export is one of the two.

**A quick save saves what a card shows.** Every card that shows something
exportable (a table or list, a graph, the statistics, BD-rate's table and
chart, the Bitrate Viewer's graph, Video Compare's frame pair) has a save
icon (`SaveButton`, quick_save.h) in its top-right corner.
- One click saves what the card shows (its metric, conditions, dips, range
  and videos) into the export folder (Settings' Export folder, Documents
  where it is blank), in the formats Settings > Storage > *Quick saves*
  names: JSON and/or CSV for data, PNG for a graph. Its files are named as
  the Save dialog suggests them (`suggested_file_name`).
- Its tooltip says what it will save, in which formats, and where. Where it
  can't save, it is grayed and its tooltip says why: before the click, not
  in a window after it.
- A right-click on it, or on the card's view, offers Save as...: the same
  saves one at a time, each with the Save dialog (`Dialogs::save_file`).
  Items read "{What} as {FORMAT}...": "Graph as PNG...", "Worst frames as
  CSV...", JSON before CSV; "..." (three ASCII dots) ends every item that
  opens a dialog.
- It ignores Settings' *Exports include* and zip choice; it uses the export
  folder and, for a graph, Settings' *Exported graphs* colors.

**The archive exports the saved recipe.** "Everything for these encodes":
the tab bar's Export button, at its right end in both interfaces, opens a
panel (`ExportPanel`, export_panel.h).
- Which videos: the open tab's selection where it lists videos (Videos,
  Metric Graphs' statistics, Saved Results), else every scored video, with
  every scored video the other choice.
- What: Settings > Storage > *Exports include* itself, not a copy (a change
  in the panel is saved there): the items (results, per-frame scores,
  graphs, worst frames, weak segments), the formats (JSON, CSV or both, for
  data; graphs are always PNG), a .zip or loose files; the weak segment
  rules picked in the panel (the profile in use until another is); the
  Worst Frames tab's number of dips; the whole videos.
- Where: the export folder, with Change...; no dialog.
- The panel also links to the open tab's own quick save, for the one-off.
- A right-click on a list of videos (the Videos table, Metric Graphs'
  statistics, Saved Results) keeps "Export {this video's | the selected
  videos'} {contents}..." (`export_action_text`), the same archive with a
  dialog, for those who select first. It is the only archive item outside
  the panel. After each run, when Settings asks for it, each video's own.
- One video's files are at the top. With several, each video's own files
  (`"<video> - ..."`) are in a folder of its name (`in_video_folders`), and
  the files of all of them together (results, the weak segments report,
  "All test videos - ..." graphs) at the top.

**Nothing is replaced.** Where a name is taken, every file of the export is
numbered alike (`unique_output_names`). The user's own choice in a save
dialog is the exception: they were asked.

**Confirmation is the indicator** (`ExportIndicator`, export_indicator.h)
beside the Export button: "Saving..." while an export is written, "Saved 2
files · Open folder" once it is (it fades), "Not saved: {why} · Details"
where it failed (kept until clicked). No export opens a window to say it
worked, or a status line to say where; a refusal is said on the grayed
control before the click. The after-run export says only where it fails.

### Adding an export

**A new tab or card that shows data** gets, in the same change:
- a `SaveButton` in the card's top-right corner, its `why_not` and
  `describe` saying what it saves or why it can't (grayed), its `save`
  going through `QuickSaves::save`: what it reads of the card is taken at
  the click, and its files are made and written on the file writer, so the
  window answers meanwhile;
- its Save as... items on a right-click, "{What} as {FORMAT}...", JSON
  before CSV;
- the Export panel's "On this tab" link, where it is the tab's main card
  (`MainWindow::tab_save`).

A table, list or chart that is not data to save (a list of files to read,
one to pick from) is marked `not_exported`, with why. The appearance tests
(`EveryCardThatShowsDataHasASaveIcon`) fail on any other without a save
icon on its card.

**A new kind of data** decides, and says in its change:
- whether it is an archive item: then a key in `export_item_keys`, which
  makes it a tick in Settings' *Exports include* and in the Export panel
  (`export_item_title`), and is written after each run too (`export_files`);
- where its files go in the archive: a video's own (`"<video> - ..."`, in
  its folder with several videos, `in_video_folders`) or all the videos'
  together (`shared_stems`);
- its file names: `suggested_file_name`, numbered alike with the rest of
  its export where taken (`unique_output_names`).

**Settings say what is written**, and nothing else does: *Exports include*
(`Settings::export_items`, `export_formats`, `export_zip`) for the archive,
*Quick saves* (`Settings::quick_save_formats`) for the save icons, each a
settings.json key. A new file format is:
- a key in `export_format_keys`, in the order everything lists it (JSON,
  CSV, then the new one), with its tooltip in *Exports include*'s row
  (`SettingsPanel::build_export_choices`); the *Quick saves* row, the
  Export panel and `quick_formats` take it from there;
- ticked by default only for new settings: in settings already saved it
  starts off, until the user ticks it;
- written by every quick save and archive item whose data it can hold, and
  refused (grayed, with why) by those it can't;
- written by the command line too (below).

Graphs stay PNG: a graph format is added the same way, as its own setting.

**The command line by default.** A new export, kind of data or file format
comes with its `vml-cli` support in the same change: the same file written
by the same option where one exists (`--json`, `--csv`, `--frames-json`,
`--run-dir`...), else a new one, with its test, and the command line's own
docs. It is left out only where the data needs the window (what is on
screen, such as Video Compare's frame pair as shown), or where the
maintainer has agreed to leave it out; the change says which and why.

**One writer for each kind of file**, in core, shared by the window and the
command line (as `bd_rate_json`), so that both write the same file. A new
file kind or key is documented in `docs/RESULTS_JSON.md`, a new control in
`docs/USER_GUIDE.md`, and its text is in all 20 catalogs
(`scripts/i18n_catalog.py check`).

Not exports, and so not in these menus: *Copy as markdown* (the clipboard),
*Export log* (Settings, for diagnosis) and *Import run file...*.

## Performance rules

- Keep frame scores in arrays (32-bit floats, as libvmaf and FFmpeg give
  them), not per-frame objects.
- Graphs reduce each line to one min/max pair per pixel column.
- Bound decode queues; do not decode every encode of a series ahead.
- Decide hardware decoding per video, with software as the fallback.
- Measure CPU, memory, GPU and I/O separately before naming a bottleneck.
- A speedup must not change any score.
