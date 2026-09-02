"""Forces Qt to render offscreen for the whole test suite.

Without this, every test that constructs a QMainWindow/QApplication (most of
this suite) creates a REAL, visible window on the developer's desktop --
and since tests never call .close() on them, they accumulate indefinitely
across test runs instead of disappearing when the pytest process exits. This
must be set before PySide6 is imported anywhere, which is why it lives here:
conftest.py is loaded by pytest before any test module in this directory.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
