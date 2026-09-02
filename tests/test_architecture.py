"""Guards the layering the README describes, so it stays true as the code
grows rather than becoming an aspirational comment."""
from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "vmaf_app" / "core"

# ffmpeg_locate and result_cache use Qt purely as a platform abstraction --
# QSettings for the remembered ffmpeg location, QStandardPaths for the
# per-user cache directory. That is not a UI dependency, and reimplementing
# per-platform config paths by hand would be worse. Anything else in core
# importing Qt is a layering break.
_QT_FOR_PLATFORM_PATHS = {"ffmpeg_locate.py", "result_cache.py"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_does_not_import_qt_widgets():
    offenders = {}
    for path in CORE.glob("*.py"):
        qt = {m for m in _imported_modules(path) if m.startswith("PySide6")}
        widgets = {m for m in qt if "QtWidgets" in m or "QtGui" in m}
        if widgets:
            offenders[path.name] = sorted(widgets)
        elif qt and path.name not in _QT_FOR_PLATFORM_PATHS:
            offenders[path.name] = sorted(qt)
    assert not offenders, (
        f"core must stay headless-runnable; Qt imported by: {offenders}"
    )


def test_core_does_not_import_the_ui_layer():
    offenders = {
        path.name: sorted(m for m in _imported_modules(path) if m.startswith("vmaf_app.ui"))
        for path in CORE.glob("*.py")
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"core must not depend on ui: {offenders}"
