from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from vmaf_app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Video Metrics Calculator")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
