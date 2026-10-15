"""The window's translatable text, and what each language's catalog lacks.

    python scripts/i18n_catalog.py keys              every key, as JSON
    python scripts/i18n_catalog.py check [CODE ...]  missing and stale keys

The keys are the English text of the C++ window's tr("..."), N_("...") and
ntr("...", "...") calls with literal text (cpp/app), the engine's texts
the window shows (N_("...") in cpp/core, and the run line's say("...")),
and the engine's message shapes it translates (the templates in
cpp/app/i18n.cpp). A catalog,
resources/translations/<code>.json, holds {"strings": {English:
translation}, "plurals": {English singular: [forms]}} with as many forms
as the language's plural rules have (plural_forms, as cpp/app/i18n.cpp's
plural_index gives them). Run `check` after changing the window's text.
"""
from __future__ import annotations

import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The window's code, and the engine's, whose texts the window shows.
SOURCES = (ROOT / "cpp" / "app", ROOT / "cpp" / "core")
I18N = ROOT / "cpp" / "app" / "i18n.cpp"
TRANSLATIONS_DIR = ROOT / "resources" / "translations"


# ------------------------------------------------------------- C++ sources

def tokens(source: str):
    """The C++ source's tokens that matter here, comments left out: string
    literals as ("string", their text), identifiers as ("name", it), and
    other characters as ("punct", it)."""
    at, end = 0, len(source)
    while at < end:
        c = source[at]
        if source.startswith("//", at):
            at = source.find("\n", at)
            at = end if at < 0 else at
        elif source.startswith("/*", at):
            at = source.find("*/", at + 2)
            at = end if at < 0 else at + 2
        elif c.isspace():
            at += 1
        elif c == "'" and at > 0 and (source[at - 1].isalnum() or source[at - 1] == "_"):
            at += 1  # a digit separator, as in 10'000
        elif c == "'":  # a character literal, skipped
            at += 1
            while at < end and source[at] != "'":
                at += 2 if source[at] == "\\" else 1
            at += 1
        elif c.isalpha() or c == "_":
            raw = re.match(r'(?:u8|u|U|L)?R"([^(\s]*)\(', source[at:])
            prefixed = re.match(r'(?:u8|u|U|L)"', source[at:])
            if raw:  # R"delimiter( ... )delimiter"
                start = at + raw.end()
                close = source.index(")" + raw.group(1) + '"', start)
                yield "string", source[start:close]
                at = close + len(raw.group(1)) + 2
            elif prefixed:
                at += prefixed.end() - 1  # the literal itself is read next
            else:
                word = re.match(r"[A-Za-z_]\w*", source[at:]).group()
                yield "name", word
                at += len(word)
        elif c == '"':
            text, at = _literal(source, at + 1)
            yield "string", text
        else:
            yield "punct", c
            at += 1


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "a": "\a", "b": "\b",
            "f": "\f", "v": "\v", "?": "?"}


def _literal(source: str, at: int) -> tuple[str, int]:
    """The text of the string literal whose first character is at `at`, and
    where it ends. Its escapes are decoded as the compiler stores a narrow
    literal: \\x and octal escapes are bytes (the sources spell UTF-8 so,
    "\\xE2\\x80\\xA6" for an ellipsis), the rest characters in UTF-8; the
    whole is then read as UTF-8."""
    out = bytearray()
    while source[at] != '"':
        c = source[at]
        if c != "\\":
            out += c.encode("utf-8")
            at += 1
            continue
        kind = source[at + 1]
        if kind in "uU":
            digits = 4 if kind == "u" else 8
            out += chr(int(source[at + 2:at + 2 + digits], 16)).encode("utf-8")
            at += 2 + digits
        elif kind == "x":
            hex_digits = re.match(r"[0-9A-Fa-f]+", source[at + 2:]).group()
            out.append(int(hex_digits, 16))
            at += 2 + len(hex_digits)
        elif kind in "01234567":
            octal = re.match(r"[0-7]{1,3}", source[at + 1:]).group()
            out.append(int(octal, 8))
            at += 1 + len(octal)
        else:
            out += _ESCAPES[kind].encode("utf-8")
            at += 2
    # A literal of bytes that are not text (a file signature, say) is never
    # a key: its stray bytes are only replaced.
    return out.decode("utf-8", errors="replace"), at + 1


def _strings(found: list, at: int) -> tuple[str | None, int]:
    """The text of the adjacent string literals from token `at`, joined as
    the compiler joins them, and the token after them; None if there is no
    literal there."""
    parts = []
    while at < len(found) and found[at][0] == "string":
        parts.append(found[at][1])
        at += 1
    return ("".join(parts) if parts else None), at


def _translated(found: list, index: int) -> bool:
    """Whether the name at token `index` is a call that translates its text
    (or marks it to be translated where it is shown): the window's tr, ntr
    and N_ by their bare names or as app::tr (not another class's, as
    QObject::tr, nor a member's); the engine's N_; and the run line's
    Wording (cpp/core/src/run_line.cpp: say and wording.text), which the
    window gives its translation."""
    value = found[index][1]
    before = found[index - 1] if index >= 1 else None
    if value == "text":
        return index >= 2 and found[index - 2:index] == [("name", "wording"), ("punct", ".")]
    if value not in {"tr", "N_", "ntr", "say"} or before in {("punct", "."), ("punct", ">")}:
        return False
    if before == ("punct", ":") and index >= 2 and found[index - 2] == ("punct", ":"):  # qualified
        return index >= 3 and found[index - 3] == ("name", "app")
    return True


def keys(sources: tuple[Path, ...] = SOURCES) -> tuple[set[str], dict[str, str]]:
    """(the strings, {singular: plural} for the counted ones)."""
    strings: set[str] = set(templates())
    plurals: dict[str, str] = {}
    files = [path for folder in sources for pattern in ("*.cpp", "*.h") for path in folder.rglob(pattern)]
    for path in sorted(files):
        found = list(tokens(path.read_text(encoding="utf-8")))
        for index, (kind, value) in enumerate(found):
            if kind != "name" or not _translated(found, index):
                continue
            if index + 1 >= len(found) or found[index + 1] != ("punct", "("):
                continue
            first, after = _strings(found, index + 2)
            if first is None:
                continue  # text that is not a literal, or the declaration
            if value == "ntr":
                if after < len(found) and found[after] == ("punct", ","):
                    second, _ = _strings(found, after + 1)
                    if second is not None:
                        plurals[first] = second
            else:
                strings.add(first)
    return strings, plurals


def templates() -> list[str]:
    """The engine's message shapes the window translates: the templates array
    in cpp/app/i18n.cpp."""
    found = list(tokens(I18N.read_text(encoding="utf-8")))
    for index in range(len(found) - 2):
        if found[index] == ("name", "templates") and found[index + 1:index + 3] == [("punct", "["),
                                                                                   ("punct", "]")]:
            shapes, at = [], index + 3
            while found[at] != ("punct", "}"):
                text, after = _strings(found, at)
                if text is None:
                    at += 1
                else:
                    shapes.append(text)
                    at = after
            return shapes
    raise RuntimeError("cpp/app/i18n.cpp has no templates array")


def languages() -> list[str]:
    """The window's languages' codes, as cpp/app/i18n.cpp lists them."""
    text = I18N.read_text(encoding="utf-8")
    block = text[text.index("all_languages = {"):]
    block = block[:block.index("};")]
    return re.findall(r'\{"([A-Za-z_]+)",', block)


def plural_forms(code: str) -> int:
    """How many forms a counted text has in `code`, as cpp/app/i18n.cpp's
    plural_index gives them."""
    if code in {"zh_CN", "zh_TW", "ja", "ko", "th", "vi", "id"}:
        return 1
    if code in {"ru", "uk", "pl", "cs"}:
        return 3
    return 6 if code == "ar" else 2


# ---------------------------------------------------------------- catalogs

def fields(text: str) -> list[tuple[str, str]]:
    """The {placeholders} of a text, with their format specs, in a stable order."""
    return sorted((name, spec) for _literal, name, spec, _conversion in string.Formatter().parse(text)
                  if name is not None)


def problems(code: str, catalog: dict, expected: tuple[set[str], dict[str, str]]) -> list[str]:
    """What is wrong with the `code` catalog: texts missing, stale, empty, or
    with other placeholders."""
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
        if not isinstance(forms, list) or len(forms) != plural_forms(code):
            found.append(f"needs {plural_forms(code)} forms: {key!r}")
            continue
        for form in forms:
            if fields(form) != fields(plurals[key]):
                found.append(f"placeholders differ: {key!r} -> {form!r}")
    return found


def main(argv: list[str]) -> int:
    """Prints the keys, or checks every catalog; returns the exit code."""
    sys.stdout.reconfigure(encoding="utf-8")  # the texts, whatever the console's code page
    expected = keys()
    if argv[:1] == ["keys"]:
        strings, plurals = expected
        print(json.dumps({"strings": sorted(strings), "plurals": dict(sorted(plurals.items()))},
                         ensure_ascii=False, indent=1))
        return 0
    codes = argv[1:] or [code for code in languages() if code != "en"]
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
