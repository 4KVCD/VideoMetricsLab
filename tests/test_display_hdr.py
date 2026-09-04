from vmaf_app.core import display_hdr
from vmaf_app.core.display_hdr import DisplayHdrInfo


def test_non_windows_display_query_reports_unknown_instead_of_guessing(monkeypatch):
    monkeypatch.setattr(display_hdr.os, "name", "posix")

    assert display_hdr.query_display_hdr() == DisplayHdrInfo()


def test_a_driver_query_failure_does_not_break_the_frame_viewer(monkeypatch):
    monkeypatch.setattr(display_hdr.os, "name", "nt")
    monkeypatch.setattr(
        display_hdr,
        "_query_windows_display_hdr",
        lambda _handle: (_ for _ in ()).throw(OSError("driver unavailable")),
    )

    assert display_hdr.query_display_hdr(1234) == DisplayHdrInfo()
