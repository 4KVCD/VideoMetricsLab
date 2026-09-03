from pathlib import Path

import pytest

from vmaf_app.core import result_cache
from vmaf_app.core.models import FrameScore, VideoInfo, VmafOptions, VmafRunResult

OPTIONS = VmafOptions()


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(result_cache, "_cache_dir", lambda: tmp_path)


def _make_file(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _fake_result(source: Path, distorted: Path) -> VmafRunResult:
    info = VideoInfo(path=distorted, width=1920, height=1080, fps=30.0, duration=5.0, nb_frames=10, codec_name="h264")
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=90.0) for i in range(10)]
    return VmafRunResult(
        source=source, distorted=distorted, frames=frames, fps=30.0, model="version=vmaf_v0.6.1",
        source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
    )


def test_store_then_load_cached_round_trips(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)

    assert result_cache.load_cached(source, distorted, OPTIONS) is None  # nothing cached yet

    result = _fake_result(source, distorted)
    result_cache.store(source, distorted, result, label="my-run", options=OPTIONS)

    loaded = result_cache.load_cached(source, distorted, OPTIONS)
    assert loaded is not None
    loaded_result, label = loaded
    assert label == "my-run"
    assert len(loaded_result.frames) == len(result.frames)


def test_cache_miss_when_distorted_file_size_differs(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted_v1 = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(source, distorted_v1, _fake_result(source, distorted_v1), label="v1", options=OPTIONS)

    # Same name, different size (e.g. the file was re-encoded) -- must miss.
    distorted_v2 = _make_file(tmp_path / "distorted.mp4", 999)
    assert result_cache.load_cached(source, distorted_v2, OPTIONS) is None


def test_cache_miss_when_filename_differs_even_with_same_size(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    a = _make_file(tmp_path / "a.mp4", 500)
    b = _make_file(tmp_path / "b.mp4", 500)
    result_cache.store(source, a, _fake_result(source, a), label="a", options=OPTIONS)

    assert result_cache.load_cached(source, b, OPTIONS) is None


def test_cache_miss_for_same_filename_and_size_in_a_different_directory(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    a = _make_file(tmp_path / "encode-a" / "movie.mp4", 500)
    b = _make_file(tmp_path / "encode-b" / "movie.mp4", 500)
    result_cache.store(source, a, _fake_result(source, a), label="a", options=OPTIONS)

    assert result_cache.load_cached(source, b, OPTIONS) is None


def test_cache_miss_when_a_same_size_file_is_replaced_in_place(tmp_path):
    import os

    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "movie.mp4", 500)
    result_cache.store(source, distorted, _fake_result(source, distorted), label="old", options=OPTIONS)
    old_mtime = distorted.stat().st_mtime_ns
    distorted.write_bytes(b"y" * 500)
    os.utime(distorted, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))

    assert result_cache.load_cached(source, distorted, OPTIONS) is None


def test_clear_removes_the_cached_entry(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(source, distorted, _fake_result(source, distorted), label="x", options=OPTIONS)
    assert result_cache.load_cached(source, distorted, OPTIONS) is not None

    result_cache.clear(source, distorted, OPTIONS)
    assert result_cache.load_cached(source, distorted, OPTIONS) is None


def test_clear_on_nonexistent_entry_does_not_raise(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.clear(source, distorted, OPTIONS)  # never stored -- must not raise


def test_cache_miss_when_calculation_options_change(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    original = VmafOptions(n_subsample=1, extra_features=["name=psnr"])
    changed = VmafOptions(n_subsample=5, extra_features=["name=psnr"])
    result_cache.store(
        source, distorted, _fake_result(source, distorted), label="original", options=original
    )

    assert result_cache.load_cached(source, distorted, changed) is None
