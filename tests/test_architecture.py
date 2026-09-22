"""Guards the layering the README describes, so it stays true as the code
grows rather than becoming an aspirational comment."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parent.parent / "vmaf_app" / "core"

# This uses Qt purely as a platform abstraction for the remembered ffmpeg
# location. That is not a UI dependency. Anything else in core importing Qt
# is a layering break.
_QT_FOR_PLATFORM_PATHS = {"ffmpeg_locate.py"}


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


def test_metric_execution_layers_are_explicitly_headless():
    """Future backends must plug into these modules without importing Qt."""
    names = {"metrics.py", "metric_results.py", "comparison_recipe.py", "execution.py", "metric_cache.py"}
    offenders = {}
    for path in CORE.glob("*.py"):
        if path.name not in names:
            continue
        imports = _imported_modules(path)
        bad = sorted(name for name in imports if name.startswith(("PySide6", "vmaf_app.ui")))
        if bad:
            offenders[path.name] = bad
    assert not offenders


def test_no_core_module_spawns_a_visible_console_window():
    """Every subprocess must go through vmaf_app.core.proc.

    The app runs under pythonw.exe, which has no console, so a child process
    started without CREATE_NO_WINDOW gets its own console window that flashes
    up and vanishes. With one per added video it looks like a malfunction.
    Calling subprocess directly is how that comes back.
    """
    offenders: dict[str, list[str]] = {}
    for path in CORE.glob("*.py"):
        if path.name == "proc.py":
            continue  # the one place that is allowed to call it
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            value = node.func.value
            if isinstance(value, ast.Name) and value.id == "subprocess" \
                    and node.func.attr in {"run", "Popen", "call", "check_output"}:
                bad.append(f"subprocess.{node.func.attr} at line {node.lineno}")
        if bad:
            offenders[path.name] = bad
    assert not offenders, (
        f"these bypass vmaf_app.core.proc and will flash a console window: {offenders}"
    )


def test_the_hidden_flag_is_only_applied_on_windows():
    import os

    from vmaf_app.core.proc import hidden_kwargs

    kwargs = hidden_kwargs()
    if os.name == "nt":
        assert kwargs.get("creationflags") == 0x0800_0000
    else:
        assert kwargs == {}, "the flag does not exist off Windows"


def test_building_a_window_does_not_repoint_the_cache_at_the_user(tmp_path):
    """The suite's isolation must survive MainWindow's constructor.

    It did not: the constructor re-applies whatever its loaded Settings say
    the cache directory is, and a blank setting means "the platform folder"
    -- so every test that built a window was reading and writing the user's
    real results cache, however carefully the fixture had redirected it.
    """
    from PySide6.QtWidgets import QApplication

    from vmaf_app.core import result_cache
    from vmaf_app.ui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    before = result_cache.cache_dir()
    assert tmp_path in before.parents or before == tmp_path / "results_cache"

    MainWindow()

    assert result_cache.cache_dir() == before, (
        "constructing the window moved the cache out of the test's temp folder"
    )


def test_the_isolation_guard_refuses_the_users_real_cache_directory():
    """The fixture's backstop has to actually fire, or it is decoration."""
    from vmaf_app.core import result_cache

    real = result_cache.default_cache_dir()
    with pytest.raises(AssertionError, match="real folder"):
        result_cache.set_cache_dir_override(real)


def test_the_documented_smoke_command_is_runnable():
    """The smoke script is the only end-to-end check there is, so it going
    stale is expensive: it was both unimportable from the repository root
    (a script's own directory goes first on sys.path, not the caller's) and
    written against a two-argument progress callback the runner stopped
    calling when fps was added. Neither failure showed up until someone ran
    it, which is exactly when it is least welcome.

    This imports it and inspects the callback, without launching ffmpeg.
    """
    import inspect
    import runpy

    module = runpy.run_path(
        str(Path(__file__).resolve().parent / "smoke_run.py"),
        run_name="not_main",  # importing must not start a run
    )

    assert module["parse_args"]([]).ten_bit is False
    assert module["parse_args"](["--10bit"]).ten_bit is True

    source = inspect.getsource(module["main"])
    assert "def on_progress(current, total, fps)" in source, (
        "the progress callback no longer matches ProgressCallback"
    )
    assert "FrameComparison.from_result(result)" in source, (
        "the frame-preview smoke test no longer matches extract_frame_png"
    )
