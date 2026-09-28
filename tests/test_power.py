import sys

import pytest

from vmaf_app.core import power


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power requests")
def test_the_request_is_held_and_released_with_the_windows_flags(monkeypatch):
    import ctypes

    calls = []
    monkeypatch.setattr(ctypes.windll.kernel32, "SetThreadExecutionState",
                        lambda flags: calls.append(flags) or 0x80000000)
    assert power.keep_system_awake(True) is True
    assert power.keep_system_awake(False) is True
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, then ES_CONTINUOUS alone (released).
    assert calls == [0x80000001, 0x80000000]


def test_nothing_is_held_off_windows(monkeypatch):
    monkeypatch.setattr(power.sys, "platform", "linux")
    assert power.keep_system_awake(True) is False
