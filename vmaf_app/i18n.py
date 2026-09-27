"""The window's text in the user's language.

The app opens in Windows' display language when it has a translation for it
-- Settings > Window > Language can choose another -- and in English
otherwise. Only what the window shows is translated. The log, saved results,
exported files and the self-test stay in English: a log is read by whoever is
fixing a problem, and the files by other programs. So do metric names (VMAF,
SSIMULACRA2, CVVDP...) and what FFmpeg itself says.

In the code, tr("Calculate metrics"): the English text is the key into
translations/<code>.json, and is what shows when there is no translation.
Placeholders are str.format fields, tr("{done} done", done=3); a literal
brace in such a text is doubled. ntr() takes the form for a number by the
language's plural rules. N_() marks English text kept as data (a label in a
table of definitions) that is translated with tr() where it is shown.

The core modules speak English: their statuses and errors go to the log as
they are. tr_message() translates such a message as the window shows it, when
it has one of the shapes in MESSAGE_TEMPLATES; anything else -- FFmpeg's own
words, a path -- stays as it came.
"""
from __future__ import annotations

import contextlib
import functools
import json
import re
import threading
from collections.abc import Iterator
from pathlib import Path

#: Every language the window can be shown in: code -> its name in itself.
#: The order is the Settings list's.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh_CN": "简体中文",
    "zh_TW": "繁體中文",
    "es": "Español",
    "pt_BR": "Português (Brasil)",
    "de": "Deutsch",
    "fr": "Français",
    "ja": "日本語",
    "ru": "Русский",
    "ko": "한국어",
    "it": "Italiano",
    "pl": "Polski",
    "tr": "Türkçe",
    "ar": "العربية",
    "id": "Bahasa Indonesia",
    "vi": "Tiếng Việt",
    "uk": "Українська",
    "th": "ไทย",
    "cs": "Čeština",
    "hu": "Magyar",
    "nl": "Nederlands",
}
#: Written right to left: the whole window is mirrored.
RIGHT_TO_LEFT = frozenset({"ar"})
TRANSLATIONS_DIR = Path(__file__).resolve().parent / "translations"

_strings: dict[str, str] = {}
_plurals: dict[str, list[str]] = {}
_language = "en"
_local = threading.local()


def N_(text: str) -> str:  # noqa: N802 -- gettext's name for it
    """Marks `text` for translation where it is shown; returns it as it is."""
    return text


def language_for(ui_language: str) -> str:
    """The translation for a language name as Windows gives it -- "de-DE",
    "zh-Hant-TW", "pt-BR" -- or "en" when there is none.

    Chinese in Taiwan, Hong Kong or Macau, or written in Traditional
    characters, gets Traditional Chinese; any other Chinese, Simplified.
    Every Portuguese gets the Brazilian translation, every Spanish the one
    Spanish."""
    parts = [part for part in re.split(r"[-_]", ui_language.strip()) if part]
    if not parts:
        return "en"
    base = parts[0].lower()
    if base == "zh":
        rest = {part.lower() for part in parts[1:]}
        return "zh_TW" if rest & {"hant", "tw", "hk", "mo"} else "zh_CN"
    if base == "pt":
        return "pt_BR"
    return base if base in LANGUAGES else "en"


def windows_language() -> str:
    """The translation for Windows' display language, or "en"."""
    from PySide6.QtCore import QLocale

    languages = QLocale.system().uiLanguages()
    return language_for(languages[0]) if languages else "en"


def set_language(code: str) -> str:
    """Shows the window's text in `code` from now on; English for a code
    with no translation, or one whose file cannot be read. The language
    set."""
    global _language
    strings: dict[str, str] = {}
    plurals: dict[str, list[str]] = {}
    if code != "en" and code in LANGUAGES:
        try:
            data = json.loads((TRANSLATIONS_DIR / f"{code}.json").read_text(encoding="utf-8"))
            strings = {key: value for key, value in data.get("strings", {}).items() if value}
            plurals = {key: value for key, value in data.get("plurals", {}).items() if value}
        except (OSError, ValueError, AttributeError):
            code = "en"
    else:
        code = "en"
    _strings.clear()
    _strings.update(strings)
    _plurals.clear()
    _plurals.update(plurals)
    _language = code
    _translate_line.cache_clear()
    return code


def language() -> str:
    return _language


def _active() -> bool:
    return _language != "en" and not getattr(_local, "english", False)


@contextlib.contextmanager
def in_english() -> Iterator[None]:
    """tr() and the rest give English on this thread while inside: for
    text the window shows that also goes to the log."""
    before = getattr(_local, "english", False)
    _local.english = True
    try:
        yield
    finally:
        _local.english = before


def tr(text: str, /, **values: object) -> str:
    """`text` in the window's language, its {placeholders} filled in."""
    if _active():
        text = _strings.get(text, text)
    return text.format(**values) if values else text


def plural_index(code: str, count: int) -> int:
    """Which of a language's forms a number takes (CLDR's rules for
    whole numbers). The catalogs list the forms in this order."""
    n = abs(count)
    if code in {"zh_CN", "zh_TW", "ja", "ko", "th", "vi", "id"}:
        return 0
    if code == "fr":
        return 0 if n in (0, 1) else 1
    if code in {"ru", "uk"}:
        if n % 10 == 1 and n % 100 != 11:
            return 0
        return 1 if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else 2
    if code == "pl":
        if n == 1:
            return 0
        return 1 if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else 2
    if code == "cs":
        return 0 if n == 1 else 1 if 2 <= n <= 4 else 2
    if code == "ar":
        if n <= 2:
            return n  # zero, one, two
        return 3 if 3 <= n % 100 <= 10 else 4 if 11 <= n % 100 <= 99 else 5
    return 0 if n == 1 else 1


#: How many forms each language's catalog gives a counted text.
PLURAL_FORMS = {code: 1 if code in {"zh_CN", "zh_TW", "ja", "ko", "th", "vi", "id"} else
                3 if code in {"ru", "uk", "pl", "cs"} else 6 if code == "ar" else 2
                for code in LANGUAGES}


def ntr(singular: str, plural: str, count: int, /, **values: object) -> str:
    """The text for `count` of something, {count} filled in: `singular`
    for one in English, `plural` otherwise, and in another language the
    form its rules give."""
    forms = _plurals.get(singular) if _active() else None
    text = (forms[min(plural_index(_language, count), len(forms) - 1)] if forms
            else singular if count == 1 else plural)
    return text.format(count=count, **values)


# ---------------------------------------------------------------- core messages
#: The shapes of the statuses and errors the core modules send that the
#: window shows, as templates: "{name}" matches any text, which is itself
#: translated when it is a message too. Keep in step with the messages'
#: wording in vmaf_app.core; one that no longer matches is shown in English.
MESSAGE_TEMPLATES: tuple[str, ...] = (
    # --- a video's steps on the run line ("distorted" shown as "test video")
    "Detecting black bars in source and test video…",
    "Detecting black bars in source…",
    "Running ffmpeg (GPU decode: {plan})…",
    "GPU decode failed, retrying (GPU decode: {plan})…",
    "GPU decode failed, retrying…",
    "Vship GPU ({device}): calculating {metrics} (GPU decode: {plan})…",
    "Vship GPU ({device}): calculating {metrics}…",
    "GPU decode failed for the source, decoding it in software (GPU decode: {plan})…",
    "GPU decode failed for the test video, decoding it in software (GPU decode: {plan})…",
    "GPU decode failed for the source, decoding it in software…",
    "GPU decode failed for the test video, decoding it in software…",
    "Calculating SSIMULACRA2/Butteraugli on the CPU as frames are extracted (GPU decode: off)…",
    "Calculating SSIMULACRA2/Butteraugli on the CPU as frames are extracted…",
    "Calculating selected perceptual metric(s) on CPU…",
    "{metrics} failed in the shared GPU pass; calculating it in a pass of its own…",
    "{metrics} failed in the shared GPU pass; calculating each in a pass of its own…",
    "{metrics} failed on the GPU; calculating it on the CPU…",
    "Vship GPU unavailable ({reason}); using CPU reference metrics…",
    "Vship GPU compute failed ({error}); using CPU reference metrics…",
    "Waiting for the GPU: another video's Vship pass is running…",
    # --- why a video or a metric failed
    "{metrics} failed: {error}",
    "Cancelled by user",
    "GPU scoring failed ({error}). It was not retried on the CPU, which would take hours to days for a video "
    "over 10 minutes. Choose CPU for {metrics} to calculate it on the CPU anyway.",
    "GPU scoring failed ({error}); the CPU retry failed too: {reason}",
    "calculated over {cpu_frames} frames on the CPU but {gpu_frames} on the GPU, so it was not kept",
    "{metric} needs a supported NVIDIA or AMD GPU and could not use it: {reason}",
    "{metric} could not be calculated: {reason}",
    "Vship GPU acceleration is only bundled for Windows.",
    "No supported NVIDIA CUDA or AMD HIP GPU was detected.",
    "Vship GPU calculation failed: {error}",
    "Vship produced no scores.",
    "FFmpeg produced no frame pairs for Vship.",
    "FFmpeg failed while decoding the {side} for Vship: {error}",
    "FFmpeg failed while decoding the {side} for Vship.",
    "Could not start FFmpeg for Vship: {error}",
    "FFmpeg ended partway through a {side} frame.",
    "Could not allocate Vship pinned frame memory: {error}",
    "Could not select the Vship GPU: {error}",
    "Could not initialize Vship {metric}: {error}",
    "Vship {metric} failed: {error}",
    "Vship {metric} returned a non-finite value.",
    "This Vship ({version}) has no CVVDP.",
    "CVVDP needs the video's frame rate, and none was found (it was read as {fps} fps).",
    "Vship cannot open the CVVDP display file at {path}.",
    "Vship does not support resolution round-trip tests yet.",
    "CVVDP scores every frame; it cannot be subsampled.",
    "Perceptual metrics in one task must use the same frame coverage.",
    "Vship does not support the {depth}-bit RGB format {format}.",
    "Vship does not support the decoded pixel format {format}.",
    "Vship does not support {sampling} {depth}-bit YUV video.",
    "Vship does not support the {matrix} color matrix.",
    "Vship's current GPU metric API does not support ICtCp input.",
    "Vship does not support the {transfer} transfer function.",
    "Vship does not support the {primaries} color primaries.",
    "Vship does not support the {range} range tag.",
    "Vship does not support {siting} chroma siting.",
    "Variable-frame-rate video is not supported safely yet.",
    "Variable-frame-rate video is not supported safely yet. Convert both videos to the same constant frame rate "
    "before comparing them.",
    "Frame rates do not match ({source} vs {test} fps).",
    "Durations do not match ({source} vs {test} seconds). Set a duration limit within both files if comparing "
    "only their common opening segment.",
    "The duration limit extends beyond the end of one of the videos.",
    "The two videos are different shapes after cropping: source {source} versus distorted {test}. Scoring them "
    "would stretch one to fit the other and the result would be meaningless. If one is letterboxed, set "
    "black-bar handling to 'Auto-detect' so the bars are removed before comparison.",
    "FFmpeg could not prepare lossless perceptual-metric frames.",
    "FFmpeg produced no frame pairs for perceptual metrics.",
    "{metric} did not produce a numeric score: {value}",
    "Could not parse {metric} score.",
    "Could not run {metric}: {error}",
    "{metric} did not finish a frame within {seconds} s.",
    "{metric} failed for a frame.",
    "Perceptual CPU metrics do not support resolution round-trip tests yet.",
    "{tool} is not installed. Install the reference {tool2} executable or set {variable}_PATH.",
    "ffmpeg exited with code {code}",
    "ffmpeg finished but no metric log was produced.",
    "No results for requested metrics: {metrics}",
    "Select at least one metric to calculate.",
    "Crop detection cancelled",
    "Crop detection timed out",
    "Could not run crop detection: {error}",
    "Crop detection failed for {path}: {detail}",
    "Could not auto-detect black bars in {path}: {detail}. Choose 'None (use full frame)' for this video to "
    "continue without cropping.",
    # --- reading files, settings, tools
    "ffprobe was not found. Make sure ffmpeg is installed and on PATH, or set a custom ffmpeg folder in Settings.",
    "Could not start ffprobe for {path}: {error}",
    "ffprobe timed out while reading {path}",
    "Probe of {path} was cancelled",
    "ffprobe failed for {path}:",
    "Could not parse ffprobe output for {path}",
    "No video stream found in {path}",
    "Video file is missing: {path}",
    "Could not start ffprobe: {error}",
    "Bitrate analysis was cancelled.",
    "ffprobe returned no packets for the first video stream.",
    "This video has no usable frame rate.",
    "Frame extraction was cancelled.",
    "No frame was returned at frame {frame}; it may be beyond the end of the video.",
    "Every display value must be a finite number.",
    "The display resolution must be at least 16 x 16.",
    "The display's screen size must be greater than zero.",
    "The display's viewing distance must be greater than zero.",
    "The display's peak brightness must be greater than zero.",
    "The display's contrast must be greater than zero.",
    "The display's exposure must be greater than zero.",
    "Ambient light cannot be negative and reflectivity must be between 0 and 1.",
    "A preset needs a name.",
    '"{name}" is a built-in preset; choose another name.',
    "No custom model file selected.",
    "Built-in VMAF model is missing: {path}",
    "Unsupported analysis result format version: {version}",
    "Could not save settings: {error}",
    "{tool} could not be run ({error}).",
    "not found",
    "exited with code {code}",
    "ffmpeg {version} is too old -- version {minimum} or newer is required.",
    # --- Video Compare's playback, a " · " part at a time
    "Playing",
    "Paused",
    "Playback ended",
    "Opening with GStreamer/D3D11…",
    "Opening with FFmpeg tone-map fallback… ({reason})",
    "Opening with FFmpeg tone-map fallback…",
    "GStreamer could not start native playback: {error}",
    "GStreamer playback failed; trying FFmpeg fallback: {error}",
    "Could not decode comparison with ffmpeg: {error}",
    "Video {number} could not play: {error}",
    "Selected video failed: {error}",
    "Buffering selected comparison; adjacent videos are preparing in the background",
    "Preparing source/current/adjacent videos",
    "GPU tone mapping and RGB",
    "native fallback: {reason}",
    "{count}/{limit} streams",
    "SDR preview (native playback unavailable)",
    "Video playback is unavailable for synthetic resolution tests; their processed frames are displayed "
    "automatically.",
    "Could not create the GStreamer pipeline.",
    "GStreamer could not open the comparison.",
    "GStreamer could not seek to that frame.",
    "Could not open the {side} video file for decoding.",
    "D3D11 device unavailable",
    "Native frame lock currently requires matching frame rates",
    "Audio preroll timed out",
    "Video frame has no usable presentation timestamp",
    "Buffering locked pair",
    "Frame-locked GPU pair",
    "source soundtrack",
    "audio unavailable",
    "Vulkan GPU decode + GPU tone mapping/RGB",
    "hardware decode + GPU tone mapping/RGB (host transfer)",
    "software decode + GPU tone mapping/RGB",
    "CPU fallback (GPU processing unavailable)",
    "retry: {error}",
    "Decoder returned a truncated frame.",
    "Decoder produced no frames.",
    "ffmpeg returned no video frame.",
)


def _bare(text: str) -> str:
    """Without a closing "...", "…" or full stop: the run line shows some
    messages trimmed of theirs."""
    return text.rstrip(".…。 ")


def _pattern(template: str) -> re.Pattern[str]:
    parts = re.split(r"\{(\w+)\}", _bare(template))
    regex = "".join(
        re.escape(part).replace(re.escape("…"), r"(?:…|\.\.\.)") if i % 2 == 0 else f"(?P<{part}>.+?)"
        for i, part in enumerate(parts)
    )
    return re.compile(regex, re.DOTALL)


@functools.cache
def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Most specific first -- the most fixed text -- so "Vship {metric}
    failed: {error}" is tried before "{metrics} failed: {error}"."""
    ordered = sorted(MESSAGE_TEMPLATES, key=lambda t: -len(re.sub(r"\{\w+\}", "", t)))
    return tuple((template, _pattern(template)) for template in ordered)


def tr_message(text: str) -> str:
    """A message from the core modules in the window's language, line by
    line, where it has a known shape (MESSAGE_TEMPLATES); otherwise as it
    came."""
    if not _active() or not text:
        return text
    return "\n".join(_translate_line(line) for line in text.split("\n"))


@functools.lru_cache(maxsize=1024)
def _translate_line(line: str) -> str:
    if not line.strip():
        return line
    if line in _strings:
        return _strings[line]
    # Matched without its closing "..." or full stop, which is put back in
    # the translation's own form -- or left off, as the line had it.
    bare = _bare(line)
    ending = line[len(bare):]
    for template, pattern in _patterns():
        match = pattern.fullmatch(bare)
        if match is None or template not in _strings:
            continue
        values = {name: _translate_line(value) for name, value in match.groupdict().items()}
        try:
            translated = _strings[template].format(**values)
        except (KeyError, IndexError, ValueError):
            return line
        return translated if ending.strip() else _bare(translated) + ending
    return line
