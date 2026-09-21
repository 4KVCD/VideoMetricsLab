# What the build ships of GStreamer

The GStreamer wheels are a complete media framework: 446 binaries, 302 MB,
with WebRTC, cloud transcription, a QUIC stack, three AV1 encoders and a CD
burner. This app plays two videos through D3D11 and one soundtrack. It
reaches 95 of those binaries, so the Windows build ships those and leaves the
rest behind: **302 MB → 53 MB**, and a 451 MB folder became 217 MB (86 MB
zipped).

Nothing in the development installation is modified. The pruning happens
when PyInstaller reads the spec.

## How the file list is decided

`scripts/gstreamer_bundle.py` has one hand-maintained list and computes
everything else from it:

1. **`KEEP_PLUGINS`** — the plugin DLLs, each with a note saying what the app
   uses it for: the elements the pipelines create by name (`filesrc`,
   `decodebin3`, `videocrop`, `d3d11upload`, `d3d11convert`, `d3d11videosink`,
   `capssetter`, `appsink`, `appsrc`, `playbin3`), the parsers and demuxers
   `decodebin3` needs for the files people compare, `libav` and `dav1d` for what no GPU
   decodes (VVC, 10-bit H.264, ProRes), and the soundtrack path down to
   `wasapi2sink`.
2. **Their dependencies**, read from the DLL import tables (normal and
   delay-loaded) with `pefile` and followed transitively across all the
   wheels. This is why an upgrade of the wheels needs no edit here: a new
   `avcodec-62.dll` is found because `gstlibav.dll` imports it. When a DLL
   name exists in two wheels, both copies are kept — which one Windows loads
   depends on the search order at run time.
3. **What imports cannot reveal**: `gst-plugin-scanner.exe` (spawned, so it
   loads plugins out of process), the two `gspawn-win64-helper*.exe` GLib
   spawns it with, `gstd3d11-1.0-0.dll` (loaded by name through ctypes for
   the HDR tone-map), the `gi` bindings and their extension modules, and every
   typelib (3 MB; choosing among them would save nothing worth a missed one).
4. **Each wheel's `__init__.py`**, because `gstreamer_libs.gstreamer_env()`
   imports the other packages by name to assemble `PATH` and the plugin path.

Left out: 228 of the 268 plugins, `libstdc++-6.dll` (25 MB that nothing in the wheels
imports), the x264/x265/SVT-AV1 encoders, librsvg, OpenSSL, the text stack
(harfbuzz, cairo, pango, freetype, fontconfig), `vvdec.dll` (VVC is decoded by
FFmpeg's own decoder inside `avcodec`), and the GPL plugin wheels entirely —
the bundled FFmpeg is LGPL, so the bundle contains no GPL code.

Inspect the decision without building:

```powershell
.venv\Scripts\python.exe scripts\gstreamer_bundle.py          # summary
.venv\Scripts\python.exe scripts\gstreamer_bundle.py --json   # every file, kept and omitted
```

## How the build proves it still works

The app falls back to FFmpeg playback on any GStreamer error, so a missing
plugin would look like a slow build rather than a broken one. Two checks make
it fail the build instead.

**`scripts/verify_gstreamer_bundle.py`** runs after PyInstaller. It inspects
the packaged runtime and the development installation in child interpreters
started with `-S` (no site directory, so the development wheels and their
`.pth` bootstrap are out of reach), with a fresh registry in a temporary
folder and the plugin path restricted to the runtime under test. It fails if:

- any plugin was loaded from outside the runtime under test;
- an allow-listed plugin did not load (a dependency is missing) or a plugin
  not on the list did (the policy leaked);
- an element in `REQUIRED_ELEMENTS` (`vmaf_app/core/gstreamer_playback.py`)
  is missing;
- an audio or video codec the development installation can decode, the
  bundle cannot — compared by the caps decoders accept, minus the losses
  named and justified in `ALLOWED_LOSSES` (Bluetooth and VoIP audio, DVD
  private-stream spellings of AC-3 and DTS, karaoke graphics).

With `-VerifyMedia` (or `--media` when run by hand) it also decodes each
given file through the app's own branch topology — `filesrc → decodebin3 →
videocrop → d3d11upload → d3d11convert` for video, `audioconvert →
audioresample → volume` for audio — and names the decoder that did it, so a
run shows `d3d11h265dec` for HEVC and `avdec_h266` for VVC:

```powershell
./scripts/build_release.ps1 -VerifyMedia "D:\clips\hevc-hdr.mkv", "D:\clips\vvc.mkv", "D:\clips\av1.mkv"
```

**`VideoMetricsLab.exe --self-test`** runs inside the packaged
process itself and checks the same `REQUIRED_ELEMENTS` list, so a bundle
that passes the verifier but fails to load in the frozen environment (a
missing hidden import, say) is still caught. The build script waits for it
and fails on a missing report or a `FAIL` line.

`tests/test_gstreamer_bundle.py` covers the policy without a build: the
dependency walk on made-up graphs, and the real wheels when installed —
every allow-listed plugin exists, everything a shipped binary imports is
shipped, the dead weight is absent, the total stays under 70 MiB, and every
element the playback code creates by name is in `REQUIRED_ELEMENTS`.

## Changing what is shipped

- **A new element in the playback code**: add it to `REQUIRED_ELEMENTS`
  (the test suite insists) and, if its plugin is not already kept, to
  `KEEP_PLUGINS` with a note. Run the build; the verifier and self-test
  confirm it loads.
- **A codec someone needs**: find its plugin in the development
  installation (`gst-inspect-1.0` from the `gstreamer_cli` wheel, or the
  registry through `gi`), add it to `KEEP_PLUGINS`, rebuild.
- **Upgrading the wheels**: nothing to edit unless a plugin was renamed, in
  which case `collect()` refuses with the name. Review the verifier's list
  of waived codec losses in case a plugin the app relies on moved.
