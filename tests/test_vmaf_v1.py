"""VMAF v1 as a column of its own beside VMAF v0.6.1: runner, request,
saved results and saved scores from when it was a VMAF model choice."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from vmaf_app.core import result_cache
from vmaf_app.core.builtin_models import builtin_model_path
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.model_select import V1_DEFAULT_MODEL, V1_UHD_MODEL, resolve_v1_model, v1_model_for_resolution
from vmaf_app.core.models import ComparisonResult, FrameScores, VideoInfo, VmafOptions
from vmaf_app.core.run_io import load_run, save_run
from vmaf_app.core.vmaf_runner import _build_libvmaf_opts, _parse_log

OLD_V1 = "__builtin:vmaf_v1_1d5h_2160"


def test_auto_picks_the_v1_model_for_the_size_compared_at():
    assert v1_model_for_resolution(3840, 1608) == V1_UHD_MODEL
    assert v1_model_for_resolution(1920, 1080) == V1_DEFAULT_MODEL
    hfr = VmafOptions(model_choice_v1="__builtin:vmaf_v1_hfr_3d0h")
    assert resolve_v1_model(hfr, 3840, 2160) == f"path={builtin_model_path('vmaf_v1_hfr_3d0h')}"
    with pytest.raises(ValueError):
        resolve_v1_model(VmafOptions(model_choice_v1="version=vmaf_v0.6.1"), 1920, 1080)


def test_each_model_in_the_pass_is_named_and_the_v1_file_is_referenced_by_name(tmp_path):
    v1 = resolve_v1_model(VmafOptions(), 1920, 1080)
    options = VmafOptions(compute_vmaf=True, compute_vmaf_neg=True, compute_vmaf_v1=True, model_v1=v1)
    model = next(opt for opt in _build_libvmaf_opts(options, tmp_path / "log.json", "version=vmaf_v0.6.1")
                 if opt.startswith("model="))
    assert model == (r"model=version=vmaf_v0.6.1\\:name=vmaf|path=vmaf_v1.0.16_3d0h.json\\:name=vmaf_v1"
                     r"|version=vmaf_v0.6.1neg\\:name=vmaf_neg")
    alone = replace(options, compute_vmaf=False, compute_vmaf_neg=False)
    assert r"model=path=vmaf_v1.0.16_3d0h.json\\:name=vmaf_v1" in _build_libvmaf_opts(alone, tmp_path / "log.json", "")


def test_the_log_is_read_into_its_own_column(tmp_path):
    log = tmp_path / "log.json"
    log.write_text(json.dumps({"frames": [
        {"frameNum": 0, "metrics": {"vmaf": 90.0, "vmaf_v1": 88.5}},
        {"frameNum": 1, "metrics": {"vmaf": 91.0, "vmaf_v1": 89.5}},
    ]}), encoding="utf-8")
    frames = _parse_log(log, 24.0)
    assert list(frames.values("vmaf")) == [90.0, 91.0]
    assert list(frames.values("vmaf_v1")) == [88.5, 89.5]


def test_the_v1_request_is_keyed_by_its_own_model_choice():
    def spec(choice):
        request = analysis_request_from_vmaf_options(
            VmafOptions(compute_vmaf_v1=True, model_choice_v1=choice), ("vmaf_v1",))
        return dict(request.metrics[0].parameters)

    assert spec("__auto__") == {"model": "", "model_choice": "__auto__"}
    assert spec(OLD_V1) == {"model": OLD_V1, "model_choice": OLD_V1}


def _result(tmp_path, model_choice, key="vmaf", size=(3840, 1608)):
    info = VideoInfo(path=tmp_path / "test.mkv", width=size[0], height=size[1], fps=24.0, duration=1.0,
                     nb_frames=2, codec_name="vvc")
    source = replace(info, path=tmp_path / "source.mkv")
    provenance = MetricProvenance("ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "ffmpeg-libvmaf-v1",
                                  {"model": f"path={builtin_model_path('vmaf_v1_1d5h_2160')}"})
    metrics = MetricResultSet([FrameMetricResult(key, [0, 1], [0.0, 1 / 24], [95.5, 96.5], provenance)])
    return ComparisonResult(source=source.path, distorted=info.path, frames=FrameScores.empty(), fps=24.0,
                            model=f"path={builtin_model_path('vmaf_v1_1d5h_2160')}", source_crop=None,
                            distorted_crop=None, source_info=source, distorted_info=info,
                            model_choice=model_choice, metric_results=metrics)


def test_a_result_file_from_when_v1_was_a_vmaf_model_opens_with_its_scores_in_the_v1_column(tmp_path):
    path = tmp_path / "old.metrics.json"
    save_run(_result(tmp_path, OLD_V1), path)
    loaded, _label = load_run(path)
    assert not loaded.frames.has("vmaf"), "v1 scores must never pass for VMAF v0.6.1"
    assert list(loaded.frames.values("vmaf_v1")) == [95.5, 96.5]
    assert loaded.model_choice_v1 == OLD_V1 and loaded.model_choice == "__auto__"


def test_a_result_file_keeps_both_models(tmp_path):
    result = _result(tmp_path, "version=vmaf_4k_v0.6.1")
    result.model_v1, result.model_choice_v1 = "path=x.json", "__builtin:vmaf_v1_3d0h"
    path = tmp_path / "new.metrics.json"
    save_run(result, path)
    loaded, _label = load_run(path)
    assert loaded.frames.has("vmaf") and loaded.model_choice == "version=vmaf_4k_v0.6.1"
    assert (loaded.model_v1, loaded.model_choice_v1) == ("path=x.json", "__builtin:vmaf_v1_3d0h")


def _cache_with_an_old_v1_score(tmp_path, size=(3840, 1608)):
    """A score saved before VMAF v1 had its own column: key "vmaf", with a
    bundled v1 model as the VMAF model choice."""
    source, distorted = tmp_path / "source.mkv", tmp_path / "test.mkv"
    source.write_bytes(b"s" * 100)
    distorted.write_bytes(b"d" * 50)
    old = analysis_request_from_vmaf_options(VmafOptions(model_choice=OLD_V1), ("vmaf",))
    result_cache.store(source, distorted, _result(tmp_path, OLD_V1, size=size), "old", old)
    return source, distorted


def _found(source, distorted, **options):
    keys = ("vmaf_v1",) if options.get("compute_vmaf_v1") else ("vmaf",)
    request = analysis_request_from_vmaf_options(VmafOptions(**options), keys)
    loaded = result_cache.load_cached(source, distorted, request)
    return None if loaded is None else {key: list(loaded[0].frames.values(key)) for key in loaded[0].frames.metric_keys}


def test_a_saved_v1_score_from_the_old_key_shows_in_the_v1_column(tmp_path):
    source, distorted = _cache_with_an_old_v1_score(tmp_path)
    assert _found(source, distorted, compute_vmaf=False, compute_vmaf_v1=True, model_choice_v1=OLD_V1) \
        == {"vmaf_v1": [95.5, 96.5]}
    # Auto: the 4K v1 model for this 4K comparison.
    assert _found(source, distorted, compute_vmaf=False, compute_vmaf_v1=True) == {"vmaf_v1": [95.5, 96.5]}
    # Another v1 model's column must not take it, nor VMAF v0.6.1.
    assert _found(source, distorted, compute_vmaf=False, compute_vmaf_v1=True,
                  model_choice_v1="__builtin:vmaf_v1_3d0h") is None
    assert _found(source, distorted) is None


def test_auto_does_not_take_an_old_v1_score_of_the_other_size(tmp_path):
    source, distorted = _cache_with_an_old_v1_score(tmp_path, size=(1920, 1080))
    assert _found(source, distorted, compute_vmaf=False, compute_vmaf_v1=True) is None


def test_recalculating_vmaf_v1_clears_its_old_key_score_too(tmp_path):
    source, distorted = _cache_with_an_old_v1_score(tmp_path)
    request = analysis_request_from_vmaf_options(
        VmafOptions(compute_vmaf=False, compute_vmaf_v1=True, model_choice_v1=OLD_V1), ("vmaf_v1",))
    result_cache.clear(source, distorted, request)
    assert _found(source, distorted, compute_vmaf=False, compute_vmaf_v1=True, model_choice_v1=OLD_V1) is None


def test_an_old_v1_score_is_only_ever_read_as_vmaf_v1(tmp_path):
    """The generic loader must not hand the old "vmaf"-keyed v1 score to
    the VMAF v0.6.1 column through the equivalence rules either."""
    source, distorted = _cache_with_an_old_v1_score(tmp_path)
    for choice in ("__auto__", "version=vmaf_v0.6.1", "version=vmaf_4k_v0.6.1"):
        assert _found(source, distorted, model_choice=choice) is None, choice


def test_the_videos_tab_has_a_vmaf_v1_column_and_model_list(qapp_v1, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from vmaf_app.ui import main_window as main_window_module
    from vmaf_app.ui.main_window import COL_VMAF_V1, MainWindow

    win = MainWindow()
    row = win._add_table_row(tmp_path / "a.mkv")
    win.distorted_table.selectRow(row)
    win._on_table_selection_changed()
    win.model_v1_combo.setCurrentIndex(win.model_v1_combo.findText("VMAF v1 HFR (1080p / 3H)"))
    assert win._rows[row].options.model_choice_v1 == "__builtin:vmaf_v1_hfr_3d0h"
    assert win.model_combo.findText("VMAF v1 (1080p / 3H)") == -1, "v1 models belong to their own list"

    path = tmp_path / "old.metrics.json"
    save_run(_result(tmp_path, OLD_V1), path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), "")))
    monkeypatch.setattr(main_window_module.MainWindow, "_vship_available", staticmethod(lambda: True))
    win._on_load_saved_run()
    loaded = next(rd for rd in win._rows if rd.path == tmp_path / "test.mkv")
    index = win._rows.index(loaded)
    assert win.distorted_table.item(index, COL_VMAF_V1).text() == "96.00"
    assert loaded.options.model_choice_v1 == OLD_V1 and loaded.options.model_choice == "__auto__"
    win.close()


@pytest.fixture
def qapp_v1():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
