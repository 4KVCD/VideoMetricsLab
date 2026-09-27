import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.i18n_catalog import keys, problems
from vmaf_app import i18n


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """A made-up language's catalog: `write` it, and the window is in it."""
    def write(strings=None, plurals=None, code="de"):
        (tmp_path / f"{code}.json").write_text(
            json.dumps({"strings": strings or {}, "plurals": plurals or {}}), encoding="utf-8")
        return i18n.set_language(code)

    monkeypatch.setattr(i18n, "TRANSLATIONS_DIR", tmp_path)
    yield write
    i18n.set_language("en")


@pytest.mark.parametrize(("windows", "expected"), [
    ("de-DE", "de"), ("de", "de"), ("fr-CA", "fr"), ("pt-BR", "pt_BR"), ("pt-PT", "pt_BR"), ("es-MX", "es"),
    ("zh-Hans-CN", "zh_CN"), ("zh-CN", "zh_CN"), ("zh-SG", "zh_CN"), ("zh-Hant-TW", "zh_TW"), ("zh-TW", "zh_TW"),
    ("zh-HK", "zh_TW"), ("zh-MO", "zh_TW"), ("ar-SA", "ar"), ("en-GB", "en"), ("sv-SE", "en"), ("", "en"),
])
def test_windows_languages_map_to_the_translations(windows, expected):
    assert i18n.language_for(windows) == expected


def test_english_is_the_text_itself_with_its_placeholders():
    assert i18n.tr("Settings saved.") == "Settings saved."
    assert i18n.tr("{count} done", count=3) == "3 done"
    assert i18n.ntr("{count} video failed", "{count} videos failed", 1) == "1 video failed"
    assert i18n.ntr("{count} video failed", "{count} videos failed", 2) == "2 videos failed"
    assert i18n.tr_message("Cancelled by user") == "Cancelled by user"


def test_a_translation_is_used_and_english_where_it_has_none(catalog):
    assert catalog({"Settings saved.": "Einstellungen gespeichert.", "{count} done": "{count} fertig"}) == "de"
    assert i18n.tr("Settings saved.") == "Einstellungen gespeichert."
    assert i18n.tr("{count} done", count=3) == "3 fertig"
    assert i18n.tr("Not in the catalog") == "Not in the catalog"


def test_the_log_stays_in_english(catalog, caplog):
    catalog({"Done.": "Fertig."})
    with i18n.in_english():
        assert i18n.tr("Done.") == "Done."
    assert i18n.tr("Done.") == "Fertig."


@pytest.mark.parametrize(("code", "counts"), [
    ("de", {1: 0, 2: 1, 5: 1, 21: 1}),
    ("fr", {0: 0, 1: 0, 2: 1}),
    ("ja", {1: 0, 2: 0, 100: 0}),
    ("ru", {1: 0, 2: 1, 4: 1, 5: 2, 11: 2, 12: 2, 21: 0, 22: 1, 25: 2, 111: 2}),
    ("pl", {1: 0, 2: 1, 5: 2, 12: 2, 21: 2, 22: 1}),
    ("cs", {1: 0, 2: 1, 4: 1, 5: 2}),
    ("ar", {0: 0, 1: 1, 2: 2, 3: 3, 10: 3, 11: 4, 99: 4, 100: 5, 103: 3}),
])
def test_plural_rules(code, counts):
    assert {count: i18n.plural_index(code, count) for count in counts} == counts
    assert max(counts.values()) < i18n.PLURAL_FORMS[code]


def test_a_counted_text_takes_the_languages_form(catalog):
    catalog(plurals={"{count} video failed": ["{count} видео не удалось", "{count} видео — сбой", "{count} видео, сбой"]},
            code="ru")
    assert [i18n.ntr("{count} video failed", "{count} videos failed", n) for n in (1, 3, 5)] == [
        "1 видео не удалось", "3 видео — сбой", "5 видео, сбой"]


def test_a_core_message_is_translated_by_its_shape_line_by_line(catalog):
    catalog({
        "{metrics} failed on the GPU; calculating it on the CPU…": "{metrics} ist auf der GPU gescheitert; CPU…",
        "Vship GPU unavailable ({reason}); using CPU reference metrics…": "Vship-GPU fehlt ({reason}); CPU…",
        "No supported NVIDIA CUDA or AMD HIP GPU was detected.": "Keine GPU gefunden.",
        "Detecting black bars in source and test video…": "Schwarze Balken werden gesucht…",
    })
    assert i18n.tr_message("SSIMULACRA2 failed on the GPU; calculating it on the CPU…") == \
        "SSIMULACRA2 ist auf der GPU gescheitert; CPU…"
    # The reason inside is a message of its own, and "..." is "…".
    assert i18n.tr_message(
        "Vship GPU unavailable (No supported NVIDIA CUDA or AMD HIP GPU was detected.); using CPU reference metrics...") \
        == "Vship-GPU fehlt (Keine GPU gefunden.); CPU…"
    # The run line trims the ellipsis: the translation is trimmed to match.
    assert i18n.tr_message("Detecting black bars in source and test video") == "Schwarze Balken werden gesucht"
    assert i18n.tr_message("Unknown words\nNo supported NVIDIA CUDA or AMD HIP GPU was detected.") == \
        "Unknown words\nKeine GPU gefunden."


def test_every_catalog_is_complete_and_keeps_the_placeholders():
    """Each shipped language has every text the window shows, with the same
    {placeholders} (a missing one would raise when shown) and as many
    counted forms as its plural rules use."""
    expected = keys()
    for path in sorted(i18n.TRANSLATIONS_DIR.glob("*.json")):
        found = problems(path.stem, json.loads(path.read_text(encoding="utf-8")), expected)
        assert not found, f"{path.name}: " + "\n".join(found[:20])


def test_every_language_in_settings_has_a_catalog():
    """Settings > Window > Language offers exactly the languages that ship a
    catalog: one without would show English under its own name."""
    shipped = {path.stem for path in i18n.TRANSLATIONS_DIR.glob("*.json")}
    assert shipped == set(i18n.LANGUAGES) - {"en"}


_MEASURE_WINDOW = """
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vmaf_app.core.settings import Settings

settings = Path(tempfile.mkdtemp()) / "settings.json"
Settings.path = staticmethod(lambda: settings)
app = QApplication([])
from vmaf_app import i18n
from vmaf_app.ui.main_window import MainWindow

MainWindow._check_ffmpeg = lambda self, prompt=False: True  # its dialog would wait for an answer
for code in i18n.LANGUAGES:
    i18n.set_language(code)
    win = MainWindow()
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    print(code, win.minimumSizeHint().width())
    win.close()
"""


@pytest.mark.skipif(sys.platform != "win32", reason="measured with Windows' own fonts")
def test_the_window_fits_its_default_1280_pixels_in_every_language():
    """A new window opens 1280 pixels wide. Text that is longer in some
    language must still fit, or the window opens wider than that and runs
    off smaller screens (German first needed 1,590). Measured in a process
    of its own on the Windows platform: the suite's offscreen platform has
    other fonts, about twice as wide."""
    result = subprocess.run(
        [sys.executable, "-c", _MEASURE_WINDOW], capture_output=True, text=True, timeout=120,
        cwd=Path(__file__).resolve().parents[1], env={**os.environ, "QT_QPA_PLATFORM": "windows"},
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    widths = {code: int(width) for code, width in (line.split() for line in result.stdout.splitlines())}
    assert set(widths) == set(i18n.LANGUAGES)
    too_wide = {code: width for code, width in widths.items() if width > 1280}
    assert not too_wide, f"minimum width over 1280 px: {too_wide}"


def test_the_catalog_check_finds_missing_stale_and_broken_entries():
    expected = ({"{count} done", "Done."}, {"{count} video failed": "{count} videos failed"})
    found = problems("de", {"strings": {"{count} done": "fertig", "Old text": "Alt"},
                            "plurals": {"{count} video failed": ["{count} Video"]}}, expected)
    assert "missing: 'Done.'" in found and "stale: 'Old text'" in found
    assert any(line.startswith("placeholders differ: '{count} done'") for line in found)
    assert "needs 2 forms: '{count} video failed'" in found


def test_the_language_is_logged_and_applied_at_startup(qapp, monkeypatch, caplog):
    from vmaf_app import main as entry

    monkeypatch.setattr(i18n, "windows_language", lambda: "sv")
    caplog.set_level(logging.INFO, logger="vmaf_app.main")
    try:
        assert entry.apply_language(qapp, "") == "en"
        assert "Language: en (as Windows; Windows: sv)" in caplog.text
    finally:
        i18n.set_language("en")
