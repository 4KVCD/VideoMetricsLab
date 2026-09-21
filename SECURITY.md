# Security policy

## Scope and support

This is an actively developed desktop application. There is no long-term
support branch or guaranteed response time. Reports should identify the exact
commit or release and whether the issue reproduces on the current code.

Media parsing uses native libraries and external tools. Keep Python, Qt,
GStreamer, FFmpeg and GPU drivers maintained. Do not treat the app as a sandbox
for hostile files, and do not run it as administrator for ordinary use.

## Private reporting

Use the repository's **Security → Report a vulnerability** option if enabled.
If it is unavailable, use a private contact method explicitly listed by a
maintainer on their GitHub profile. There is no dedicated security inbox yet.
Do not put exploit details, private videos or secrets in public issues.

Include affected versions, impact, reproduction steps, and a minimal safe
sample if you can share one. Coordinate public disclosure with maintainers.

## Privacy of diagnostics

Saved results and logs may contain absolute paths, filenames, hardware details
and source metadata. Inspect and redact attachments before uploading them.
Do not share your entire cache or settings directory by default.

## Before opening this repository publicly

Maintainers should enable GitHub private vulnerability reporting and secret
scanning where available, and provide a monitored private contact route.
