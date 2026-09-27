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


def _setting(guid: str, value: int):
    """A POWERBROADCAST_SETTING as Windows sends it: GUID, length, data."""
    import ctypes
    import uuid

    buffer = ctypes.create_string_buffer(uuid.UUID(guid).bytes_le + (4).to_bytes(4, "little")
                                         + value.to_bytes(4, "little"))
    return buffer, ctypes.addressof(buffer)


@pytest.mark.parametrize(("guid", "value", "said"), [
    ("6fe69556-704a-47a0-8f24-c28d936fda47", 0, "Screen turned off"),
    ("6fe69556-704a-47a0-8f24-c28d936fda47", 1, "Screen turned on"),
    ("6fe69556-704a-47a0-8f24-c28d936fda47", 2, "Screen dimmed"),
    ("5d3e9a59-e9d5-4b00-a6bd-ff34ff516548", 1, "On battery"),
])
def test_screen_and_power_source_notices_are_read(guid, value, said):
    buffer, address = _setting(guid, value)
    assert power.describe_power_notice(0x8013, address) == said
    del buffer


def test_sleep_and_wake_notices_are_read():
    assert power.describe_power_notice(0x0004, 0) == "Windows is going to sleep"
    assert power.describe_power_notice(0x0012, 0) == "Windows woke from sleep"
    assert power.describe_power_notice(0x000A, 0) is None  # a battery-level notice: not logged

