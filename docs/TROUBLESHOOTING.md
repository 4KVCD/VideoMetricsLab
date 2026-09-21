# Troubleshooting

## FFmpeg not found or filters missing

Use the app's tool-location picker or place a compatible full build on PATH.
Check the exact executable the app is using:

```powershell
ffmpeg -version
ffprobe -version
ffmpeg -filters | Select-String 'libvmaf|xpsnr'
```

FFmpeg 9+ is the tested baseline. A version number alone does not guarantee
that a build contains all required filters. Restart after changing PATH.

## Results do not reappear

Check the configured cache folder, original file paths and calculation settings.
Moving/replacing media can change cache identity. Different metric selections
may load partial results. Do not delete the cache to diagnose a cache miss.
Back it up and try loading an explicitly saved `.vmafrun.json` instead.

## Slow or incorrect playback

Read the playback status to identify native versus fallback presentation.
Try one encode first, then add neighbors. Record codec, dimensions, bit depth,
HDR metadata and whether each input is hardware decoded. Software decoding
for one stream does not imply that every stream must use software.

Check Windows HDR mode and the selected preview mode for washed-out highlights
or colors. Do not use a screenshot alone as proof of HDR output correctness.
Use a short shareable clip to investigate synchronization or frame mismatches.

## Crash or hang

Preserve the exact error and action sequence. Include Windows, Python, PySide6,
FFmpeg and GStreamer versions and the commit/release. For a source launch:

```powershell
.venv\Scripts\python.exe -X faulthandler -m vmaf_app.main
```

Windows Event Viewer → Windows Logs → Application may identify the faulting
module. Redact personal paths before reporting. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
for the currently unresolved GC-related crash.

## Packaged build diagnostics

Run `VideoMetricsLab.exe --self-test`; `--quiet` omits the dialog.
The report is saved to `~/.videometricslab/self-test.txt`. Self-test checks
availability, not actual HDR appearance, frame lock or performance.

Use the repository's bug template for non-security reports. Never upload an
entire cache or private movie as the default diagnostic attachment.
