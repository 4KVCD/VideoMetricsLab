import json

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


def test_windows_languages_map_to_the_translations(subtests):
    def check(windows, expected):
        assert i18n.language_for(windows) == expected

    for windows, expected in [
        ("de-DE", "de"), ("de", "de"), ("fr-CA", "fr"), ("pt-BR", "pt_BR"), ("pt-PT", "pt_BR"), ("es-MX", "es"),
        ("zh-Hans-CN", "zh_CN"), ("zh-CN", "zh_CN"), ("zh-SG", "zh_CN"), ("zh-Hant-TW", "zh_TW"), ("zh-TW", "zh_TW"),
        ("zh-HK", "zh_TW"), ("zh-MO", "zh_TW"), ("ar-SA", "ar"), ("en-GB", "en"), ("sv-SE", "en"), ("", "en"),
    ]:
        with subtests.test(windows=windows, expected=expected):
            check(windows, expected)


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


def test_plural_rules(subtests):
    def check(code, counts):
        assert {count: i18n.plural_index(code, count) for count in counts} == counts
        assert max(counts.values()) < i18n.PLURAL_FORMS[code]

    for code, counts in [
        ("de", {1: 0, 2: 1, 5: 1, 21: 1}),
        ("fr", {0: 0, 1: 0, 2: 1}),
        ("ja", {1: 0, 2: 0, 100: 0}),
        ("ru", {1: 0, 2: 1, 4: 1, 5: 2, 11: 2, 12: 2, 21: 0, 22: 1, 25: 2, 111: 2}),
        ("pl", {1: 0, 2: 1, 5: 2, 12: 2, 21: 2, 22: 1}),
        ("cs", {1: 0, 2: 1, 4: 1, 5: 2}),
        ("ar", {0: 0, 1: 1, 2: 2, 3: 3, 10: 3, 11: 4, 99: 4, 100: 5, 103: 3}),
    ]:
        with subtests.test(code=code, counts=counts):
            check(code, counts)


def test_every_catalog_is_complete_and_keeps_the_placeholders():
    """Each shipped language has every text the window shows, with the same
    {placeholders} (a missing one would raise when shown) and as many
    counted forms as its plural rules use."""
    expected = keys()
    for path in sorted(i18n.TRANSLATIONS_DIR.glob("*.json")):
        found = problems(path.stem, json.loads(path.read_text(encoding="utf-8")), expected)
        assert not found, f"{path.name}: " + "\n".join(found[:20])
