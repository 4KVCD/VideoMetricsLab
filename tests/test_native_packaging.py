"""Every frame decoder scripts/build_gpu_frames.ps1 builds is packaged and
checked by the release build: mf_frames.dll (AMD's GPU through Windows' own
decoders) was built but neither packaged nor checked, so a released app
quietly decoded AMD's videos the slower way."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _built() -> set[str]:
    script = (ROOT / "scripts" / "build_gpu_frames.ps1").read_text(encoding="utf-8")
    return set(re.findall(r"Name = '(\w+)'", script))


def test_every_frame_decoder_built_is_packaged():
    spec = (ROOT / "VideoMetricsLab.spec").read_text(encoding="utf-8")
    packaged = set(re.findall(r'\("(\w+_frames)", \(', spec))
    assert _built() and _built() <= packaged, _built() - packaged


def test_every_frame_decoder_built_is_checked_in_the_release():
    script = (ROOT / "scripts" / "build_release.ps1").read_text(encoding="utf-8")
    checked = set(re.findall(r"'(\w+_frames)\.dll'", script))
    assert _built() <= checked, _built() - checked
