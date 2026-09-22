"""Capture the empty application UI used by README.md.

Run from the repository root: python -m scripts.capture_readme
Settings and caches are isolated in a temporary directory; user state is not
loaded or changed.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from vmaf_app import APP_NAME
from vmaf_app.core import result_cache
from vmaf_app.core.settings import Settings
from vmaf_app.ui.main_window import MainWindow


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "docs" / "screenshot-videos.png"
    with tempfile.TemporaryDirectory(prefix="video-metrics-screenshot-") as temporary:
        folder = Path(temporary)
        Settings.path = staticmethod(lambda: folder / "settings.json")
        Settings.path().write_text(
            json.dumps(
                {
                    "cache_dir": str(folder / "cache"),
                    "remember_window_size": False,
                    "parallel_jobs": 1,
                }
            ),
            encoding="utf-8",
        )
        result_cache.set_cache_dir_override(folder / "cache")

        app = QApplication.instance() or QApplication([])
        app.setApplicationName(APP_NAME)
        font_path = Path("C:/Windows/Fonts/segoeui.ttf")
        if font_path.exists():
            QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Segoe UI", 10))

        window = MainWindow()
        try:
            window.resize(1500, 900)
            window.show()
            QTest.qWait(400)
            if not window.grab().save(str(destination)):
                raise RuntimeError(f"Could not save {destination}")
            print(f"Saved empty-state screenshot to {destination}")
        finally:
            window.close()
            QTest.qWait(200)
            result_cache.set_cache_dir_override(None)


if __name__ == "__main__":
    main()
