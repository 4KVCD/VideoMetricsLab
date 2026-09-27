"""The window's translatable text, and what each language's catalog lacks.

    python scripts/i18n_catalog.py keys              every key, as JSON
    python scripts/i18n_catalog.py check [CODE ...]  missing and stale keys

The keys are the English text in tr("..."), N_("...") and ntr("...", "...")
calls with literal text, anywhere in vmaf_app, and vmaf_app.i18n's
MESSAGE_TEMPLATES. A catalog, vmaf_app/translations/<code>.json, holds
{"strings": {English: translation}, "plurals": {English singular: [forms]}}
with as many forms as vmaf_app.i18n.PLURAL_FORMS gives its language.
tests/test_i18n.py runs the same check on every catalog.
"""
from __future__ import annotations

import ast
import json
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmaf_app.i18n import LANGUAGES, MESSAGE_TEMPLATES, PLURAL_FORMS, TRANSLATIONS_DIR


def keys(package: Path = ROOT / "vmaf_app") -> tuple[set[str], dict[str, str]]:
    """(the strings, {singular: plural} for the counted ones)."""
    strings: set[str] = set(MESSAGE_TEMPLATES)
    plurals: dict[str, str] = {}
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            literal = [isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in node.args[:2]]
            if node.func.id in {"tr", "N_"} and literal[:1] == [True]:
                strings.add(node.args[0].value)
            elif node.func.id == "ntr" and literal == [True, True]:
                plurals[node.args[0].value] = node.args[1].value
    return strings, plurals


def fields(text: str) -> list[tuple[str, str]]:
    """The {placeholders} of a text, with their format specs, in a stable order."""
    return sorted((name, spec) for _literal, name, spec, _conversion in string.Formatter().parse(text)
                  if name is not None)


def problems(code: str, catalog: dict, expected: tuple[set[str], dict[str, str]]) -> list[str]:
    strings, plurals = expected
    found: list[str] = []
    given = catalog.get("strings", {})
    given_plurals = catalog.get("plurals", {})
    found += [f"missing: {key!r}" for key in sorted(strings - set(given))]
    found += [f"stale: {key!r}" for key in sorted(set(given) - strings)]
    found += [f"missing plural: {key!r}" for key in sorted(set(plurals) - set(given_plurals))]
    found += [f"stale plural: {key!r}" for key in sorted(set(given_plurals) - set(plurals))]
    for key, value in given.items():
        if key in strings and (not isinstance(value, str) or not value.strip()):
            found.append(f"empty: {key!r}")
        elif key in strings and fields(value) != fields(key):
            found.append(f"placeholders differ: {key!r} -> {value!r}")
    for key, forms in given_plurals.items():
        if key not in plurals:
            continue
        if not isinstance(forms, list) or len(forms) != PLURAL_FORMS[code]:
            found.append(f"needs {PLURAL_FORMS[code]} forms: {key!r}")
            continue
        for form in forms:
            if fields(form) != fields(plurals[key]):
                found.append(f"placeholders differ: {key!r} -> {form!r}")
    return found


def main(argv: list[str]) -> int:
    expected = keys()
    if argv[:1] == ["keys"]:
        strings, plurals = expected
        print(json.dumps({"strings": sorted(strings), "plurals": dict(sorted(plurals.items()))},
                         ensure_ascii=False, indent=1))
        return 0
    codes = argv[1:] or [code for code in LANGUAGES if code != "en"]
    status = 0
    for code in codes:
        path = TRANSLATIONS_DIR / f"{code}.json"
        if not path.is_file():
            print(f"{code}: no catalog")
            status = 1
            continue
        found = problems(code, json.loads(path.read_text(encoding="utf-8")), expected)
        print(f"{code}: {'OK' if not found else f'{len(found)} problems'}")
        for line in found:
            print("   ", line)
        status |= bool(found)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
