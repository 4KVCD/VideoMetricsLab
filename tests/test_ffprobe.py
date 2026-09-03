import subprocess
from pathlib import Path

import pytest

from vmaf_app.core import ffprobe
from vmaf_app.core.ffprobe import ProbeError


def test_ffprobe_timeout_becomes_a_readable_probe_error(monkeypatch):
    monkeypatch.setattr(ffprobe, "ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(
        ffprobe.proc_util, "run",
        lambda *a, **kw: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("ffprobe", 60)
        ),
    )

    with pytest.raises(ProbeError, match="timed out"):
        ffprobe.probe_video(Path("slow.mp4"))
