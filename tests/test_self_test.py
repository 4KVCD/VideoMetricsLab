"""The packaged build's self-test (vmaf_app.main.self_test)."""
from vmaf_app.main import SelfTestReport


def test_a_failed_check_fails_the_self_test():
    report = SelfTestReport("title")
    report.ok("ffmpeg 9.0 and ffprobe")
    report.warn("no D3D11 GPU decoders registered")
    assert not report.failed
    report.fail("ffmpeg.exe could not be run (not found).")
    assert report.failed
    assert report.text.splitlines() == [
        "title", "  OK    ffmpeg 9.0 and ffprobe", "  WARN  no D3D11 GPU decoders registered",
        "  FAIL  ffmpeg.exe could not be run (not found).",
    ]


def test_the_word_fail_in_a_passing_line_does_not_fail_it():
    """Failure was read back from the text: "FAIL" anywhere in it -- a
    tool's path, say -- failed a good run."""
    report = SelfTestReport("title")
    report.ok(r"ssimulacra2 (C:\FAILSAFE\tools\ssimulacra2.exe)")
    assert not report.failed
